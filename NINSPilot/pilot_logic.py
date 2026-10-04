"""Source-associated Slicer workflow. Numeric helpers live in pilot_core."""
import csv
import importlib.util
import json
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import scipy
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pilot_core as core


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


class PilotLogic:
    intake_fields = ("patient_mapping", "orientation", "native_series", "coverage", "noncontrast", "hu_units")
    roles = tuple(f"{role}_{side}" for side in ("R", "L") for role in ("SPF", "RIM", "CONTACT", "AIR", "ROUTE"))

    def __init__(self):
        self.state = None
        self.report = None
        self.contact_reports = {}
        self.clock_phase = None
        self.clock_start = None

    def _tag(self, node, role):
        node.SetAttribute("NINSPilot.session", self.state["session_id"])
        node.SetAttribute("NINSPilot.role", role)
        node.SetNodeReferenceID("NINSPilot.volume", self.state["nodes"]["source"])
        self.state["nodes"][role] = node.GetID()
        return node

    def node(self, role):
        if not self.state:
            raise ValueError("Start or resume a session")
        n = slicer.mrmlScene.GetNodeByID(self.state["nodes"].get(role, ""))
        if n is None:
            raise ValueError("Missing session node: " + role)
        if n.GetParentTransformNode():
            raise ValueError("Transformed inputs require a separate reviewed working copy: " + role)
        if role != "source" and (n.GetAttribute("NINSPilot.session") != self.state["session_id"] or n.GetNodeReferenceID("NINSPilot.volume") != self.state["nodes"]["source"]):
            raise ValueError("Session or source association changed: " + role)
        return n

    @staticmethod
    def source_record(volume):
        if not volume or not volume.IsA("vtkMRMLScalarVolumeNode") or not volume.GetImageData() or volume.GetParentTransformNode():
            raise ValueError("Select an untransformed scalar CT volume")
        a = slicer.util.arrayFromVolume(volume)
        if a.ndim != 3 or min(a.shape) <= 1 or not np.isfinite(volume.GetSpacing()).all() or min(volume.GetSpacing()) <= 0:
            raise ValueError("Expected a 3D source with finite positive spacing")
        m = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(m)
        matrix = [[m.GetElement(r, c) for c in range(4)] for r in range(4)]
        if not np.isfinite(matrix).all() or abs(np.linalg.det(np.array(matrix)[:3, :3])) < 1e-10:
            raise ValueError("Invalid source geometry")
        return dict(voxel_sha256=core.array_digest(a), dimensions_ijk=list(a.shape[::-1]),
                    spacing_mm=list(volume.GetSpacing()), ijk_to_ras=matrix)

    def check_source(self):
        if self.source_record(self.node("source")) != self.state["source"]:
            raise ValueError("Source voxels or geometry changed; start a new session")

    def start(self, volume, case_id, reviewer_id, output_root, cohort="setup", synthetic=False):
        core.coded_id(case_id)
        core.coded_id(reviewer_id)
        source = self.source_record(volume)
        if cohort not in ("setup", "challenge", "synthetic"):
            raise ValueError("Unknown cohort")
        if max(source["spacing_mm"]) >= 1.0-1e-6 and cohort in ("setup", "evaluation"):
            raise ValueError("Primary pilot requires native submillimeter sampling; use challenge cohort otherwise")
        root = Path(output_root).expanduser().resolve()
        repository = Path(__file__).resolve().parents[1]
        if root == repository or repository in root.parents:
            raise ValueError("Choose a secure output directory outside the code repository")
        if synthetic != (cohort == "synthetic"):
            raise ValueError("Synthetic fixtures must use the synthetic cohort")
        if synthetic and volume.GetAttribute("NINSPilot.synthetic") != "true":
            raise ValueError("Selected volume is not a generated synthetic fixture")
        if not synthetic and volume.GetAttribute("NINSPilot.synthetic") == "true":
            raise ValueError("A synthetic fixture cannot be enrolled as a patient")
        directory = root / (case_id + "_" + uuid.uuid4().hex[:10])
        directory.mkdir(parents=True, exist_ok=False)
        self.pause_clock()
        self.state = dict(version=core.VERSION, session_id=directory.name, directory=str(directory),
                          case_id=case_id, reviewer_id=reviewer_id, cohort=cohort, synthetic=synthetic,
                          created_utc=utc(), source=source, nodes={"source": volume.GetID()},
                          sides={s: dict(status="assessable", reason="") for s in ("R", "L")},
                          intake={k: False for k in self.intake_fields}, timing_seconds={},
                          settings=dict(air_max_hu=-300., bone_min_hu=300., patch_radius_mm=8.,
                                        electrode_width_mm=2., electrode_length_mm=3., geometric_tolerance_mm=.25,
                                        contact_mode="unspecified"), reference=None, accepted=None, backend=None)
        self.report = None
        self.contact_reports = {}
        for role in self.roles:
            cls = "vtkMRMLMarkupsClosedCurveNode" if role.startswith("RIM") else "vtkMRMLMarkupsCurveNode" if role.startswith("ROUTE") else "vtkMRMLMarkupsFiducialNode"
            n = self._tag(slicer.mrmlScene.AddNewNodeByClass(cls, "NINS_" + case_id + "_" + role), role)
            n.CreateDefaultDisplayNodes()
            n.GetDisplayNode().SetSelectedColor(.15, .85, .75)
            n.GetDisplayNode().SetPointLabelsVisibility(False)
            n.GetDisplayNode().SetPropertiesLabelVisibility(False)
            n.GetDisplayNode().SetGlyphScale(1.5)
            if "Curve" in cls:
                n.SetCurveTypeToLinear()
        roi = self._tag(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "NINS_nasal_ROI"), "roi")
        bounds = [0.] * 6
        volume.GetRASBounds(bounds)
        roi.SetCenter(*[(bounds[2*i]+bounds[2*i+1])/2 for i in range(3)])
        roi.SetSize(*[bounds[2*i+1]-bounds[2*i] for i in range(3)])
        roi.CreateDefaultDisplayNodes()
        roi.GetDisplayNode().SetFillOpacity(.03)
        self.show_session()
        self.save_progress()
        return self.state

    def show_session(self):
        for cls in ("vtkMRMLMarkupsNode", "vtkMRMLModelNode", "vtkMRMLSegmentationNode"):
            for node in slicer.util.getNodesByClass(cls):
                if node.GetAttribute("NINSPilot.session") and node.GetDisplayNode():
                    node.GetDisplayNode().SetVisibility(node.GetAttribute("NINSPilot.session") == self.state["session_id"])
        if "broad_model" in self.state["nodes"]:
            self.node("broad_model").GetDisplayNode().SetVisibility(False)
        slicer.util.setSliceViewerLayers(background=self.node("source"), foreground=None, label=None)

    def set_review(self, intake, sides, settings=None):
        core.side_decisions(sides)
        self.state["sides"] = sides
        self.state["intake"] = {k: intake.get(k) is True for k in self.intake_fields}
        if settings:
            for k in ("air_max_hu", "bone_min_hu", "patch_radius_mm", "electrode_width_mm", "electrode_length_mm", "geometric_tolerance_mm"):
                v = float(settings[k])
                if not np.isfinite(v) or (k not in ("air_max_hu", "bone_min_hu") and v <= 0):
                    raise ValueError("Invalid parameter: " + k)
            if settings["contact_mode"] not in ("unspecified", "mucosal_contact", "transforaminal_concept"):
                raise ValueError("Unknown contact concept")
            if self.state["backend"] and any(settings[k] != self.state["backend"]["settings"][k] for k in ("air_max_hu", "bone_min_hu")):
                raise ValueError("Thresholds describe the saved prediction; start a new session to compare thresholds")
            self.state["settings"].update(settings)

    def points(self, role):
        n = self.node(role)
        result = []
        inv = np.linalg.inv(self.state["source"]["ijk_to_ras"])
        dims = np.array(self.state["source"]["dimensions_ijk"])
        for i in range(n.GetNumberOfControlPoints()):
            if n.GetNthControlPointPositionStatus(i) != n.PositionDefined:
                raise ValueError("Finish or remove undefined point: " + role)
            p = [0.] * 3
            n.GetNthControlPointPositionWorld(i, p)
            ijk = (inv @ np.r_[p, 1])[:3]
            if not np.isfinite(p).all() or np.any(ijk < 0) or np.any(ijk > dims-1):
                raise ValueError("Landmark outside source volume: " + role)
            result.append(p)
        return result

    def _intake_gate(self):
        self.check_source()
        missing = [k for k, v in self.state["intake"].items() if v is not True]
        if missing:
            raise ValueError("Complete source checks: " + ", ".join(missing))

    def freeze_reference(self):
        self._intake_gate()
        if self.state["reference"]:
            raise ValueError("Reference is already preserved; start another session for an independent rater")
        sides = core.side_decisions(self.state["sides"])
        if not sides:
            raise ValueError("No assessable sides; save the refusal record")
        reference = dict(recorded_utc=utc(), reviewer_id=self.state["reviewer_id"], sides=self.state["sides"], points={})
        for side in sides:
            if len(self.points("SPF_"+side)) != 1:
                raise ValueError("Place one SPF reference on " + side)
            core.rim_geometry(self.points("RIM_"+side))
            for role in ("SPF_", "RIM_"):
                reference["points"][role+side] = self.points(role+side)
        self.state["reference"] = reference
        write_json(Path(self.state["directory"])/"reference_before_segmentation.json", reference)
        self.pause_clock()
        self.save_progress()

    def roi_record(self):
        r = self.node("roi")
        m = r.GetObjectToWorldMatrix()
        return dict(matrix=[[m.GetElement(i, j) for j in range(4)] for i in range(4)], size=list(r.GetSize()))

    def _roi_mask(self):
        roi = self.roi_record()
        half = np.array(roi["size"])/2
        if not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError("ROI has invalid size")
        m = np.linalg.inv(roi["matrix"]) @ np.array(self.state["source"]["ijk_to_ras"])
        shape = tuple(self.state["source"]["dimensions_ijk"][::-1])
        mask = np.zeros(shape, bool)
        j, i = np.indices(shape[1:])
        for k in range(shape[0]):
            inside = np.ones(shape[1:], bool)
            for axis in range(3):
                inside &= abs(m[axis, 0]*i + m[axis, 1]*j + m[axis, 2]*k + m[axis, 3]) <= half[axis]
            mask[k] = inside
        if not mask.any():
            raise ValueError("ROI does not intersect the source")
        return mask

    def segmentation_array(self, segment_id):
        return slicer.util.arrayFromSegmentBinaryLabelmap(self.node("segmentation"), segment_id, self.node("source")).astype(np.uint8)

    def generate(self, backend="threshold"):
        self._intake_gate()
        if not self.state["reference"]:
            raise ValueError("Preserve independent SPF/rim annotations before segmentation")
        reference_sides = core.side_decisions(self.state["reference"]["sides"])
        if not set(core.side_decisions(self.state["sides"])).issubset(reference_sides):
            raise ValueError("A newly assessable side needs a separate pre-segmentation reference session")
        if self.state["backend"]:
            raise ValueError("Start a separate session for another method; the original proposal is preserved")
        if backend not in ("threshold", "manual", "totalsegmentator"):
            raise ValueError("Unknown backend")
        source = self.node("source")
        a = slicer.util.arrayFromVolume(source)
        settings = self.state["settings"]
        mask = self._roi_mask()
        bounds = [[0, n-1] for n in a.shape[::-1]]
        air, bone = core.proposal_masks(a, bounds, settings["air_max_hu"], settings["bone_min_hu"])
        metadata = dict(method=backend, generated_utc=utc(), settings=dict(settings), roi=self.roi_record())
        if backend == "manual":
            air.fill(0)
            bone.fill(0)
        elif backend == "totalsegmentator":
            # The extension must already be installed. Inference is local;
            # its backend may download model weights on first use.
            import importlib.metadata
            try:
                from TotalSegmentator import TotalSegmentatorLogic
                metadata["model_package_version"] = importlib.metadata.version("TotalSegmentator")
                import torch  # noqa: F401
            except ImportError as exc:
                raise ValueError("Install and initialize the Slicer TotalSegmentator extension first") from exc
            ts = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "NINS_broad_model_proposal")
            ts.CreateDefaultDisplayNodes()
            self._tag(ts, "broad_model")
            ts_logic = TotalSegmentatorLogic()
            ts_logic.useStandardSegmentNames = False
            ts_logic.process(source, ts, quality="normal", task="head_glands_cavities", interactive=False)
            regions = []
            for name in ("nasal_cavity_right", "nasal_cavity_left"):
                sid = ts.GetSegmentation().GetSegmentIdBySegmentName(name)
                if not sid:
                    raise ValueError("Model output lacks " + name)
                regions.append(slicer.util.arrayFromSegmentBinaryLabelmap(ts, sid, source).astype(bool))
            air &= np.logical_or(*regions)
            metadata.update(task="head_glands_cavities", quality="normal", interpretation="Nasal model mask restricts HU air proposal; not an SPF/SPG model")
            ts.GetDisplayNode().SetVisibility(False)
        air &= mask
        bone &= mask
        if backend != "manual" and not air.any():
            raise ValueError("No candidate air in ROI; review thresholds and source")
        seg = self._tag(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "NINS_editable_candidates"), "segmentation")
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(source)
        seg.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
        for sid, name, color, voxels in (("air", "Air candidate - review nasal extent", (.2,.7,.9), air), ("bone", "Bone candidate", (.95,.85,.65), bone)):
            seg.GetSegmentation().AddEmptySegment(sid, name, color)
            slicer.util.updateSegmentBinaryLabelmapFromArray(voxels, seg, sid, source)
        self.state["backend"] = metadata
        self.node("roi").GetDisplayNode().SetVisibility(False)
        self.state["prediction_hashes"] = {sid: core.array_digest(x) for sid, x in (("air",air),("bone",bone))}
        prediction = Path(self.state["directory"])/"prediction.seg.nrrd"
        if prediction.exists() or not slicer.util.saveNode(seg, str(prediction)):
            raise RuntimeError("Could not preserve original proposal")
        if "broad_model" in self.state["nodes"]:
            slicer.util.saveNode(self.node("broad_model"), str(Path(self.state["directory"])/"broad_model.seg.nrrd"))
        write_json(Path(self.state["directory"])/"prediction_manifest.json", metadata)
        self.save_progress()
        self.start_clock("correction")
        return seg

    def _surface(self):
        seg = self.node("segmentation")
        if not seg.GetSegmentation().GetSegment("air"):
            raise ValueError("Air segment was removed")
        seg.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
        seg.RemoveClosedSurfaceRepresentation()
        seg.CreateClosedSurfaceRepresentation()
        seg.GetDisplayNode().SetVisibility3D(True)
        seg.GetDisplayNode().SetOpacity3D(.25)
        mesh = vtk.vtkPolyData()
        seg.GetClosedSurfaceRepresentation("air", mesh)
        if mesh.GetNumberOfCells() == 0:
            raise ValueError("Reviewed air mask is empty")
        return mesh

    @staticmethod
    def _mesh_digest(mesh):
        return core.digest(dict(points=core.array_digest(vtk_to_numpy(mesh.GetPoints().GetData())),
                                cells=core.array_digest(vtk_to_numpy(mesh.GetPolys().GetData()))))

    def build_patch(self, side):
        """Open local air-boundary patch, with ROI cut planes excluded by a guard."""
        self.check_source()
        if not self.state["backend"] or self.roi_record() != self.state["backend"]["roi"]:
            raise ValueError("ROI changed after generation; start a new segmentation session")
        if side not in core.side_decisions(self.state["sides"]):
            raise ValueError("Side is unassessable")
        p = self.points("CONTACT_"+side)
        if len(p) != 1:
            raise ValueError("Place one contact point")
        radius = self.state["settings"]["patch_radius_mm"]
        ijk = (np.linalg.inv(self.state["source"]["ijk_to_ras"]) @ np.r_[p[0], 1])[:3]
        edge_mm = np.minimum(ijk, np.array(self.state["source"]["dimensions_ijk"])-1-ijk)*self.state["source"]["spacing_mm"]
        if min(edge_mm) <= radius + max(self.state["source"]["spacing_mm"]):
            raise ValueError("Patch could include the source volume boundary; reduce its radius or use adequate source coverage")
        roi = self.roi_record()
        local = (np.linalg.inv(roi["matrix"]) @ np.r_[p[0], 1])[:3]
        if np.any(np.array(roi["size"])/2-abs(local) <= radius + max(self.state["source"]["spacing_mm"])):
            raise ValueError("Patch could include an artificial ROI boundary; enlarge ROI and start a new segmentation session")
        sphere = vtk.vtkSphere()
        sphere.SetCenter(p[0])
        sphere.SetRadius(radius)
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(self._surface())
        clip.SetClipFunction(sphere)
        clip.InsideOutOn()
        clip.Update()
        connected = vtk.vtkPolyDataConnectivityFilter()
        connected.SetInputConnection(clip.GetOutputPort())
        connected.SetExtractionModeToClosestPointRegion()
        connected.SetClosestPoint(p[0])
        connected.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(connected.GetOutputPort())
        clean.Update()
        if clean.GetOutput().GetNumberOfCells() == 0:
            raise ValueError("No surface near the contact point")
        mesh = vtk.vtkPolyData()
        mesh.DeepCopy(clean.GetOutput())
        role = "PATCH_"+side
        if role in self.state["nodes"]:
            n = self.node(role)
        else:
            n = self._tag(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "NINS_contact_patch_"+side), role)
            n.CreateDefaultDisplayNodes()
        n.SetAndObservePolyData(mesh)
        n.GetDisplayNode().SetColor(.9,.4,.15)
        n.GetDisplayNode().SetBackfaceCulling(False)
        n.SetAttribute("NINSPilot.air_hash", core.array_digest(self.segmentation_array("air")))
        n.SetAttribute("NINSPilot.patch_input", core.digest(dict(contact=p, radius=radius, roi=roi)))
        self.state["accepted"] = None
        return n

    def snapshot(self):
        if not self.state["backend"] or self.roi_record() != self.state["backend"]["roi"]:
            raise ValueError("ROI changed after generation; start a new segmentation session")
        if not set(core.side_decisions(self.state["sides"])).issubset(core.side_decisions(self.state["reference"]["sides"])):
            raise ValueError("No independent reference was preserved for the newly assessable side")
        snapshot = dict(points={r: self.points(r) for r in self.roles}, sides=self.state["sides"],
                        intake=self.state["intake"], settings=self.state["settings"], roi=self.roi_record(),
                        masks={s: core.array_digest(self.segmentation_array(s)) for s in ("air", "bone")}, patches={})
        for side in core.side_decisions(self.state["sides"]):
            role = "PATCH_"+side
            if role in self.state["nodes"]:
                n = self.node(role)
                expected = core.digest(dict(contact=self.points("CONTACT_"+side), radius=self.state["settings"]["patch_radius_mm"], roi=self.roi_record()))
                if n.GetAttribute("NINSPilot.air_hash") != snapshot["masks"]["air"] or n.GetAttribute("NINSPilot.patch_input") != expected:
                    raise ValueError("Air mask, contact point or patch settings changed; rebuild patch " + side)
                snapshot["patches"][side] = self._mesh_digest(n.GetPolyData())
        return snapshot

    def accept(self):
        self._intake_gate()
        sides = core.side_decisions(self.state["sides"])
        if not sides or not self.state["backend"]:
            raise ValueError("Generate and review a segmentation on at least one side")
        if not self.segmentation_array("air").any():
            raise ValueError("The air segment is empty; finish manual segmentation before acceptance")
        for side in sides:
            if len(self.points("SPF_"+side)) != 1:
                raise ValueError("Place one SPF point on " + side)
            core.rim_geometry(self.points("RIM_"+side))
        snap = self.snapshot()
        self.state["accepted"] = dict(digest=core.digest(snap), reviewer_id=self.state["reviewer_id"], accepted_utc=utc())
        self.pause_clock()
        self.save_progress()

    @staticmethod
    def _distance(mesh, point):
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        closest, cell, sub, d2 = [0.]*3, vtk.reference(0), vtk.reference(0), vtk.reference(0.)
        locator.FindClosestPoint(point, closest, cell, sub, d2)
        return float(d2)**.5, closest

    def _route(self, side, mesh):
        points = np.asarray(self.points("ROUTE_"+side))
        if len(points) < 2:
            return dict(status="not_measured", reason="Place an entry-to-contact route with at least two points")
        step = min(self.state["source"]["spacing_mm"])/2
        samples = []
        length = 0.
        for a, b in zip(points[:-1], points[1:]):
            dist = float(np.linalg.norm(b-a))
            length += dist
            samples.extend(np.linspace(a, b, max(2, int(np.ceil(dist/step))+1)))
        inv = np.linalg.inv(self.state["source"]["ijk_to_ras"])
        voxels = self.segmentation_array("air")
        radii = []
        for p in samples:
            ijk = np.rint((inv @ np.r_[p, 1])[:3]).astype(int)
            if np.any(ijk < 0) or np.any(ijk >= self.state["source"]["dimensions_ijk"]) or not voxels[tuple(ijk[::-1])]:
                return dict(status="outside_reviewed_lumen", length_mm=length, minimum_sampled_radius_mm=0.)
            radii.append(self._distance(mesh, p)[0])
        return dict(status="measured", length_mm=length, minimum_sampled_radius_mm=min(radii), sampling_step_mm=step,
                    definition="Nearest reviewed air-boundary distance along sampled polyline; not device insertion clearance")

    def _contact_module(self):
        path = Path(__file__).resolve().parents[1]/"scripts"/"(C) NINS_Contact_Geometry_0912.py"
        spec = importlib.util.spec_from_file_location("nins_contact", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def measure(self):
        self.check_source()
        snap = self.snapshot()
        accepted = self.state["accepted"]
        if not accepted or accepted["digest"] != core.digest(snap):
            raise ValueError("Review and accept the current inputs before measuring")
        mesh = self._surface()
        contact = self._contact_module()
        self.contact_reports = {}
        report = dict(version=core.VERSION, created_utc=utc(), case_id=self.state["case_id"],
                      cohort=self.state["cohort"], synthetic=self.state["synthetic"], source=self.state["source"],
                      backend=self.state["backend"], settings=self.state["settings"], acceptance=accepted,
                      timing_seconds=self.elapsed(), sides={}, coordinate_system="RAS", length_units="mm",
                      software=dict(slicer=slicer.app.applicationVersion, numpy=np.__version__, scipy=scipy.__version__, vtk=vtk.vtkVersion.GetVTKVersion()),
                      source_code_sha256={p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                      not_assessed=["SPG segmentation", "electrical recruitment", "tissue pressure", "device deformation", "whole-device insertion", "clinical validity"])
        for side in ("R", "L"):
            if side not in core.side_decisions(self.state["sides"]):
                report["sides"][side] = self.state["sides"][side]
                continue
            result = dict(status="measured", aperture=core.rim_geometry(self.points("RIM_"+side)), route=self._route(side, mesh))
            role = "PATCH_"+side
            result["contact"] = dict(status="not_measured", reason="Review a local patch and place contact/air points")
            if role in self.state["nodes"]:
                patch = self.node(role)
                target = self.points("SPF_"+side)[0]
                d, closest = self._distance(patch.GetPolyData(), target)
                result["surface_to_spf"] = dict(distance_mm=d, closest_surface_point_ras_mm=closest,
                                                definition="Shortest 3D distance from SPF reference to reviewed local exposed air-tissue patch")
                if len(self.points("CONTACT_"+side)) == 1 and len(self.points("AIR_"+side)) == 1:
                    s = self.state["settings"]
                    try:
                        cr = contact.run_slicer(patch, self.node("CONTACT_"+side), self.node("AIR_"+side), self.node("SPF_"+side),
                                                footprints_mm=[(s["electrode_width_mm"], s["electrode_length_mm"])],
                                                tolerance_mm=s["geometric_tolerance_mm"])
                        self.contact_reports[side] = cr
                        result["contact"] = contact.metrics_only(cr)
                    except ValueError as exc:
                        result["contact"] = dict(status="refused", reason=str(exc))
            report["sides"][side] = result
        self.report = report
        return report

    def start_clock(self, phase):
        if phase not in ("reference", "correction"):
            raise ValueError("Unknown timed task")
        self.pause_clock()
        self.clock_phase, self.clock_start = phase, time.monotonic()

    def elapsed(self):
        times = dict(self.state["timing_seconds"]) if self.state else {}
        if self.clock_phase:
            times[self.clock_phase] = times.get(self.clock_phase, 0.) + time.monotonic()-self.clock_start
        return times

    def pause_clock(self):
        if self.state:
            self.state["timing_seconds"] = self.elapsed()
        self.clock_phase = self.clock_start = None

    def save_progress(self):
        if not self.state:
            raise ValueError("No session")
        self.check_source()
        folder = Path(self.state["directory"])/("checkpoint_"+uuid.uuid4().hex[:8])
        folder.mkdir()
        files = {}
        for role in self.state["nodes"]:
            if role == "source":
                continue
            node = self.node(role)
            suffix = ".seg.nrrd" if node.IsA("vtkMRMLSegmentationNode") else ".vtp" if node.IsA("vtkMRMLModelNode") else ".mrk.json"
            path = folder/(role+suffix)
            if not slicer.util.saveNode(node, str(path)):
                raise RuntimeError("Could not save " + role)
            files[role] = path.name
        patch_provenance = {role: {key: self.node(role).GetAttribute(key) for key in ("NINSPilot.air_hash", "NINSPilot.patch_input")} for role in files if role.startswith("PATCH_")}
        record = dict(self.state, timing_seconds=self.elapsed(), files=files, patch_provenance=patch_provenance)
        # MRML IDs are runtime references, never used to reconnect a saved session.
        record.pop("nodes")
        write_json(folder/"session.json", record)
        return str(folder/"session.json")

    def resume(self, session_file, volume):
        path = Path(session_file).resolve()
        state = json.loads(path.read_text())
        if state.get("version") != core.VERSION or state.get("source") != self.source_record(volume):
            raise ValueError("Session version or source fingerprint does not match")
        self.pause_clock()
        self.state = dict(state, nodes={"source": volume.GetID()})
        self.report = None
        for role, name in state["files"].items():
            if role not in self.roles + ("roi", "segmentation", "broad_model", "PATCH_R", "PATCH_L") or Path(name).name != name:
                raise ValueError("Invalid checkpoint file entry")
            file = str(path.parent/name)
            if name.endswith(".seg.nrrd"):
                n = slicer.util.loadSegmentation(file)
            elif name.endswith(".vtp"):
                n = slicer.util.loadModel(file)
            else:
                n = slicer.util.loadMarkups(file)
            self._tag(n, role)
            if role.startswith("PATCH_"):
                for key, value in state.get("patch_provenance", {}).get(role, {}).items():
                    n.SetAttribute(key, value)
        # A resumed review must be accepted again; loading geometry is not acceptance.
        self.state["accepted"] = None
        self.contact_reports = {}
        self.show_session()
        return self.state

    def export(self):
        self.check_source()
        if not self.report or not self.state["accepted"] or self.report["acceptance"]["digest"] != core.digest(self.snapshot()):
            raise ValueError("Measurements are missing or inputs changed; review and recalculate")
        self.pause_clock()
        folder = Path(self.state["directory"])/("run_"+uuid.uuid4().hex[:10])
        folder.mkdir()
        self.report["timing_seconds"] = self.elapsed()
        self.report["checkpoint"] = self.save_progress()
        write_json(folder/"report.json", self.report)
        with (folder/"measurements.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "side", "status", "measurement", "value", "unit"])
            for side, rec in self.report["sides"].items():
                if rec["status"] == "unassessable":
                    writer.writerow([self.state["case_id"], side, rec["status"], "reason", rec["reason"], ""])
                    continue
                for key, value in rec["aperture"].items():
                    if isinstance(value, (int, float)):
                        writer.writerow([self.state["case_id"], side, "measured", key, value, "mm2" if key.endswith("mm2") else "mm"])
                if "surface_to_spf" in rec:
                    writer.writerow([self.state["case_id"], side, "measured", "local_surface_to_spf_3d_mm", rec["surface_to_spf"]["distance_mm"], "mm"])
        for side, cr in self.contact_reports.items():
            self._contact_module().export_report(cr, folder/side)
        # Capture the actual reviewed views, keeping failed captures explicit.
        captures = []
        lm = slicer.app.layoutManager()
        if lm:
            old_layout = lm.layout
            lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            old_views, visibility = [], []
            try:
                for cls in ("vtkMRMLMarkupsNode", "vtkMRMLModelNode", "vtkMRMLSegmentationNode"):
                    for n in slicer.util.getNodesByClass(cls):
                        d = n.GetDisplayNode()
                        if d and n.GetAttribute("NINSPilot.session") != self.state["session_id"]:
                            visibility.append((d, d.GetVisibility()))
                            d.SetVisibility(False)
                for view, orientation in (("Red", "Axial"), ("Yellow", "Sagittal"), ("Green", "Coronal")):
                    widget = lm.sliceWidget(view)
                    sn, composite = widget.mrmlSliceNode(), widget.mrmlSliceCompositeNode()
                    matrix = vtk.vtkMatrix4x4()
                    matrix.DeepCopy(sn.GetSliceToRAS())
                    old_views.append((sn, composite, matrix, composite.GetBackgroundVolumeID(), composite.GetForegroundVolumeID(), composite.GetLabelVolumeID()))
                    composite.SetBackgroundVolumeID(self.node("source").GetID())
                    composite.SetForegroundVolumeID(None)
                    composite.SetLabelVolumeID(None)
                    sn.SetOrientation(orientation)
                for side in core.side_decisions(self.state["sides"]):
                    p = self.points("SPF_"+side)[0]
                    slicer.modules.markups.logic().JumpSlicesToLocation(*p, True)
                    slicer.app.processEvents()
                    for view in ("Red", "Yellow", "Green"):
                        widget = lm.sliceWidget(view)
                        target = folder/(side+"_"+view+".png")
                        ok = bool(widget and widget.sliceView().grab().save(str(target)))
                        captures.append(dict(side=side, view=view, saved=ok, file=target.name))
            finally:
                for sn, composite, matrix, background, foreground, label in old_views:
                    composite.SetBackgroundVolumeID(background)
                    composite.SetForegroundVolumeID(foreground)
                    composite.SetLabelVolumeID(label)
                    sn.GetSliceToRAS().DeepCopy(matrix)
                    sn.UpdateMatrices()
                for display, visible in visibility:
                    display.SetVisibility(visible)
                lm.setLayout(old_layout)
        write_json(folder/"screenshots.json", captures)
        return str(folder)

    @staticmethod
    def demo(kind="straight"):
        a = core.synthetic_case(kind)
        v = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "SYNTHETIC_NINS_"+kind)
        v.SetSpacing(.5,.5,.5)
        v.SetOrigin(-20,-20,-16)
        slicer.util.updateVolumeFromArray(v, a)
        v.CreateDefaultDisplayNodes()
        v.GetDisplayNode().AutoWindowLevelOff()
        v.GetDisplayNode().SetWindowLevel(1800, 200)
        v.SetAttribute("NINSPilot.synthetic", "true")
        slicer.util.setSliceViewerLayers(background=v)
        slicer.util.resetSliceViews()
        return v
