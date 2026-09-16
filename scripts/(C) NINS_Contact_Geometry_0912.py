"""Research-only contact geometry explorer. Units: mm, world RAS.

Import without scene side effects. See the companion Run Contact Geometry note.
NumPy core; VTK/Slicer imported only by adapters. No electrical or pressure model.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

VERSION = "0.1.0"


def unit(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-9:
        raise ValueError("Expected a finite nonzero 3-vector")
    return x / np.linalg.norm(x)


def ellipse_mesh(width, length, rings=8, sectors=64):
    """Triangulated disk, including its boundary; polygon area converges to pi*a*b."""
    if not (np.isfinite([width, length]).all() and width > 0 and length > 0):
        raise ValueError("Footprint dimensions must be positive and finite")
    if rings < 2 or sectors < 12:
        raise ValueError("Use at least 2 rings and 12 sectors")
    theta = np.arange(sectors) * 2 * np.pi / sectors
    xy = [[0., 0.]]
    for r in range(1, rings + 1):
        xy.extend(np.column_stack((width / 2 * r / rings * np.cos(theta),
                                   length / 2 * r / rings * np.sin(theta))))
    faces = [[0, 1 + j, 1 + (j + 1) % sectors] for j in range(sectors)]
    for r in range(1, rings):
        a, b = 1 + (r - 1) * sectors, 1 + r * sectors
        for j in range(sectors):
            k = (j + 1) % sectors
            faces.extend([[a + j, b + j, b + k], [a + j, b + k, a + k]])
    return np.asarray(xy), np.asarray(faces, dtype=int)


def triangle_areas(points, faces):
    q = points[faces]
    return np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1) / 2


def vertex_weights(points, faces):
    weights = np.zeros(len(points))
    areas = triangle_areas(points, faces)
    for j in range(3):
        np.add.at(weights, faces[:, j], areas / 3)
    return weights


def evaluate(surface, center, normal, axis, target, width, length, angle_deg,
             tolerance_mm, ray_half_length_mm=8., rings=8, sectors=64):
    """surface(origins, normal, half_length) -> Nx3 hits, NaN for absent/ambiguous.

    Flat seating is parallel to the supplied local tangent frame, translated to
    the highest sampled tissue point. Conforming is an ideal geometric surface,
    NOT a claim that an electrode will deform or maintain actual tissue contact.
    Gaps are measured along the frame normal, not shortest Euclidean distance.
    """
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("Provide a nonnegative geometric apposition tolerance")
    if not np.isfinite(ray_half_length_mm) or ray_half_length_mm <= 0:
        raise ValueError("Ray half length must be positive")
    center, target = np.asarray(center, float), np.asarray(target, float)
    if center.shape != (3,) or target.shape != (3,) or not np.isfinite([center, target]).all():
        raise ValueError("Center and target must be finite RAS 3-vectors")
    n = unit(normal)
    u = unit(np.asarray(axis) - np.dot(axis, n) * n)
    v = np.cross(n, u)
    if not np.isfinite(angle_deg):
        raise ValueError("Angle must be finite")
    a = np.deg2rad(angle_deg)
    u, v = np.cos(a) * u + np.sin(a) * v, -np.sin(a) * u + np.cos(a) * v
    xy, faces = ellipse_mesh(width, length, rings, sectors)
    origins = center + xy[:, 0, None] * u + xy[:, 1, None] * v
    points = np.asarray(surface(origins, n, ray_half_length_mm), float)
    if points.shape != origins.shape:
        raise ValueError("Surface sampler returned an incorrect shape")
    valid = np.isfinite(points).all(axis=1)
    weights = vertex_weights(origins, faces)
    base = dict(width_mm=float(width), length_mm=float(length), angle_deg=float(angle_deg),
                center_ras=center.tolist(), normal_ras=n.tolist(), axis_ras=u.tolist(),
                sampled_coverage_fraction=float(weights[valid].sum() / weights.sum()),
                nominal_projected_area_mm2=float(np.pi * width * length / 4),
                sampled_projected_area_mm2=float(weights.sum()))
    if not valid.all():
        return dict(base, status="rejected", reason="missing_or_ambiguous_surface",
                    missing_sample_count=int((~valid).sum()))
    heights = (points - origins) @ n
    transverse = points - origins - heights[:, None] * n
    if np.max(np.linalg.norm(transverse, axis=1)) > 1e-5 or np.max(abs(heights)) > ray_half_length_mm + 1e-5:
        raise ValueError("Surface hits must lie on their search segments")
    flat = origins + heights.max() * n
    gaps = heights.max() - heights
    q = points[faces]
    raw_normals = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    norms = np.linalg.norm(raw_normals, axis=1)
    if np.any(norms < 1e-12):
        return dict(base, status="rejected", reason="degenerate_surface")
    slopes = np.degrees(np.arccos(np.clip((raw_normals @ n) / norms, -1, 1)))
    # A quadratic fit describes surface shape only; residual exposes poor fit.
    design = np.column_stack((np.ones(len(xy)), xy, .5 * xy[:, 0] ** 2,
                              xy[:, 0] * xy[:, 1], .5 * xy[:, 1] ** 2))
    coeff = np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                            heights * np.sqrt(weights), rcond=None)[0]
    residual = heights - design @ coeff
    first = np.eye(2) + np.outer(coeff[1:3], coeff[1:3])
    second = np.array([[coeff[3], coeff[4]], [coeff[4], coeff[5]]]) / np.sqrt(1 + sum(coeff[1:3] ** 2))
    curvature = np.sort(np.linalg.eigvals(np.linalg.solve(first, second)).real)
    base.update(status="evaluated", geometric_tolerance_mm=float(tolerance_mm),
                flat_gap_max_mm=float(gaps.max()),
                flat_gap_mean_mm=float(np.average(gaps, weights=weights)),
                flat_apposition_projected_area_mm2=float(weights[gaps <= tolerance_mm + 1e-10].sum()),
                flat_apposition_fraction=float(weights[gaps <= tolerance_mm + 1e-10].sum() / weights.sum()),
                conforming_surface_area_mm2=float(triangle_areas(points, faces).sum()),
                surface_relief_mm=float(np.ptp(heights)),
                maximum_facet_tilt_deg=float(slopes.max()),
                quadratic_principal_curvatures_per_mm=curvature.tolist(),
                quadratic_fit_rms_mm=float(np.sqrt(np.average(residual ** 2, weights=weights))),
                flat_target_distance_mean_mm=float(np.average(np.linalg.norm(flat - target, axis=1), weights=weights)),
                conforming_target_distance_mean_mm=float(np.average(np.linalg.norm(points - target, axis=1), weights=weights)),
                # mesh payload retained in memory, exported separately from metrics
                _surface_points=points, _flat_points=flat, _faces=faces,
                _flat_gaps=gaps)
    return base


def pareto_ids(candidates, mode):
    """No clinical score. Identify nondominated candidates within each geometry."""
    if mode not in ("flat", "conforming"):
        raise ValueError("mode must be flat or conforming")
    good = [c for c in candidates if c["status"] == "evaluated"]
    if mode == "flat":
        values = [[-c["flat_apposition_projected_area_mm2"], c["flat_gap_max_mm"],
                   c["flat_target_distance_mean_mm"]] for c in good]
    else:
        values = [[-c["conforming_surface_area_mm2"], c["surface_relief_mm"],
                   c["conforming_target_distance_mean_mm"]] for c in good]
    front = []
    for i, val in enumerate(values):
        x = np.asarray(val)
        if not any(np.all(np.asarray(y) <= x + 1e-9) and np.any(np.asarray(y) < x - 1e-9)
                   for j, y in enumerate(values) if i != j):
            front.append(good[i]["candidate_id"])
    return front


def search(surface, center, normal, axis, target, footprints_mm, angles_deg,
           offsets_mm, tolerance_mm, ray_half_length_mm=8., rings=8, sectors=64):
    """Bounded grid search, not a global optimum. Offsets are in initial tangent axes."""
    n = unit(normal)
    u = unit(np.asarray(axis) - np.dot(axis, n) * n)
    v = np.cross(n, u)
    if not footprints_mm or not angles_deg or not offsets_mm:
        raise ValueError("Supply nonempty footprint, rotation and offset lists")
    candidates = []
    for du, dv in offsets_mm:
        if not np.isfinite([du, dv]).all():
            raise ValueError("Offsets must be finite")
        c = np.asarray(center) + du * u + dv * v
        for width, length in footprints_mm:
            for angle in angles_deg:
                row = evaluate(surface, c, n, u, target, width, length, angle,
                               tolerance_mm, ray_half_length_mm, rings, sectors)
                row.update(candidate_id=f"C{len(candidates) + 1:04d}",
                           offset_u_mm=float(du), offset_v_mm=float(dv))
                candidates.append(row)
    return {"version": VERSION, "coordinate_system": "world RAS", "units": "mm",
            "status": "geometry_only_requires_review", "target_semantics": "SPF reference point, not SPG",
            "not_assessed": ["electrical recruitment", "perfusion", "pressure", "deformation",
                             "retention", "insertion collision", "device thickness clearance"],
            "parameters": {"footprints_mm": footprints_mm, "angles_deg": angles_deg,
                           "offsets_mm": offsets_mm, "tolerance_mm": tolerance_mm,
                           "ray_half_length_mm": ray_half_length_mm, "rings": rings,
                           "sectors": sectors, "center_ras": np.asarray(center).tolist(),
                           "normal_ras": n.tolist(), "axis_ras": u.tolist(),
                           "target_ras": np.asarray(target).tolist()},
            "flat_pareto_ids": pareto_ids(candidates, "flat"),
            "conforming_pareto_ids": pareto_ids(candidates, "conforming"),
            "candidates": candidates}


class VTKSurface:
    """Intersect an OPEN reviewed mucosal patch. Reject multiple surface layers.

    Uses cell candidates plus triangle intersections so open and folded surfaces
    are handled without the closed-surface assumptions of vtkOBBTree all-hits.
    """
    def __init__(self, polydata):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(polydata)
        tri.PassLinesOff()
        tri.PassVertsOff()
        tri.Update()
        self.mesh = vtk.vtkPolyData()
        self.mesh.DeepCopy(tri.GetOutput())
        if self.mesh.GetNumberOfCells() == 0:
            raise ValueError("Mucosal patch has no triangles")
        self.vertices = vtk_to_numpy(self.mesh.GetPoints().GetData()).astype(float)
        self.faces = np.array([[self.mesh.GetCell(i).GetPointId(j) for j in range(3)]
                               for i in range(self.mesh.GetNumberOfCells())])
        if not np.isfinite(self.vertices).all():
            raise ValueError("Surface contains nonfinite vertices")
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()
        self.sha256 = hashlib.sha256(self.vertices.tobytes() + self.faces.tobytes()).hexdigest()

    def closest_frame(self, seed, air_point):
        import vtk
        point, cell, sub, dist = [0., 0., 0.], vtk.reference(0), vtk.reference(0), vtk.reference(0.)
        self.locator.FindClosestPoint(seed, point, cell, sub, dist)
        tri = self.vertices[self.faces[int(cell)]]
        n = unit(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        direction = np.asarray(air_point) - point
        if abs(np.dot(n, direction)) < .1:
            raise ValueError("Air reference must be clearly off the patch tangent plane")
        if np.dot(n, direction) < 0:
            n = -n
        # Project superior RAS direction, falling back to anterior for near-horizontal patches.
        axis = np.array([0., 0., 1.]) if abs(n[2]) < .9 else np.array([0., 1., 0.])
        return np.asarray(point), n, unit(axis - np.dot(axis, n) * n), math.sqrt(float(dist))

    def __call__(self, origins, normal, half_length):
        import vtk
        result = np.full_like(origins, np.nan, dtype=float)
        ids = vtk.vtkIdList()
        for i, origin in enumerate(origins):
            p0, p1 = origin - half_length * normal, origin + half_length * normal
            ids.Reset()
            self.locator.FindCellsAlongLine(p0, p1, 1e-7, ids)
            if not ids.GetNumberOfIds():
                continue
            triangles = self.vertices[self.faces[[ids.GetId(j) for j in range(ids.GetNumberOfIds())]]]
            e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            h = np.cross(np.broadcast_to(normal, e2.shape), e2)
            det = np.einsum("ij,ij->i", e1, h)
            usable = abs(det) > 1e-12
            inv = np.divide(1., det, out=np.zeros_like(det), where=usable)
            s = origin - triangles[:, 0]
            u = inv * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, e1)
            v = inv * (q @ normal)
            t = inv * np.einsum("ij,ij->i", e2, q)
            good = usable & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (abs(t) <= half_length)
            hits = np.sort(t[good])
            # Merge duplicated intersections at shared triangle edges only.
            if len(hits) and np.ptp(hits) <= 1e-5:
                result[i] = origin + np.mean(hits) * normal
        return result


def metrics_only(report):
    return dict(report, candidates=[{k: v for k, v in c.items() if not k.startswith("_")}
                                    for c in report["candidates"]])


def export_report(report, directory):
    """Unique run directory, never overwrite prior runs. OBJ coordinates remain RAS mm."""
    out = Path(directory) / ("contact_" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    clean = metrics_only(report)
    (out / "report.json").write_text(json.dumps(clean, indent=2, allow_nan=False))
    scalar_keys = sorted({k for c in clean["candidates"] for k, v in c.items()
                          if isinstance(v, (str, int, float, bool))})
    with (out / "candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean["candidates"])
    for c in report["candidates"]:
        if c["status"] != "evaluated":
            continue
        for mode, key in [("flat", "_flat_points"), ("conforming", "_surface_points")]:
            with (out / f"{c['candidate_id']}_{mode}_RAS_mm.obj").open("w") as f:
                f.write("# Contact face only; not a solid or fabrication-ready device. RAS mm.\n")
                for p in c[key]:
                    f.write("v %.9g %.9g %.9g\n" % tuple(p))
                for tri in c["_faces"] + 1:
                    f.write("f %d %d %d\n" % tuple(tri))
    return str(out)


def run_slicer(patch_model, contact_seed, air_reference, spf_target, *,
               footprints_mm, tolerance_mm, angles_deg=(0., 45., 90.),
               offsets_mm=((0., 0.),), ray_half_length_mm=8., rings=8, sectors=64,
               output_dir=None):
    """Inputs: exact model/one-point markup names or nodes. Read-only unless export requested.

    Patch model must be an isolated open face, not an entire closed airway.
    Transformed inputs are refused rather than mixing local and world frames.
    """
    import slicer
    def resolve(value, cls):
        if isinstance(value, str):
            matches = [n for n in slicer.util.getNodesByClass(cls) if n.GetName() == value]
            if len(matches) != 1:
                raise ValueError(f"Expected one exact {cls} named {value}; found {len(matches)}")
            value = matches[0]
        if not value.IsA(cls) or value.GetParentTransformNode():
            raise ValueError("Wrong input type or parent transform; use an untransformed copy")
        return value
    def point(node):
        if node.GetNumberOfControlPoints() != 1 or node.GetNumberOfDefinedControlPoints() != 1:
            raise ValueError("Each landmark must contain exactly one defined point")
        p = [0., 0., 0.]
        node.GetNthControlPointPositionWorld(0, p)
        return p
    model = resolve(patch_model, "vtkMRMLModelNode")
    landmarks = [resolve(n, "vtkMRMLMarkupsFiducialNode") for n in
                 (contact_seed, air_reference, spf_target)]
    seed, air, target = [point(n) for n in landmarks]
    surface = VTKSurface(model.GetPolyData())
    center, normal, axis, snap = surface.closest_frame(seed, air)
    if snap > 2.:
        raise ValueError("Contact seed is more than 2 mm from patch; confirm placement")
    report = search(surface, center, normal, axis, target, footprints_mm, list(angles_deg),
                    list(offsets_mm), tolerance_mm, ray_half_length_mm, rings, sectors)
    report["provenance"] = {"surface_sha256": surface.sha256,
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "slicer_version": slicer.app.applicationVersion,
                            "numpy_version": np.__version__,
                            "source_node_ids": [n.GetID() for n in [model] + landmarks],
                            "contact_seed_snap_mm": snap, "air_reference_ras": air,
                            "frame_method": "seed triangle normal; oriented toward air reference"}
    if output_dir:
        report["export_directory"] = export_report(report, output_dir)
    print(f"Contact geometry: {sum(c['status'] == 'evaluated' for c in report['candidates'])} / "
          f"{len(report['candidates'])} candidates evaluated.")
    print("Flat tradeoffs:", report["flat_pareto_ids"])
    print("Conforming tradeoffs:", report["conforming_pareto_ids"])
    return report


def show_candidate(report, candidate_id, mode="flat"):
    """Add a new preview face only. Never replace or remove scene nodes."""
    import vtk
    import slicer
    from vtk.util.numpy_support import numpy_to_vtk
    if mode not in ("flat", "conforming"):
        raise ValueError("mode must be flat or conforming")
    c = next(c for c in report["candidates"] if c["candidate_id"] == candidate_id)
    if c["status"] != "evaluated":
        raise ValueError("Rejected candidate has no preview")
    p = vtk.vtkPoints()
    p.SetData(numpy_to_vtk(c["_flat_points" if mode == "flat" else "_surface_points"], deep=True))
    cells = vtk.vtkCellArray()
    for face in c["_faces"]:
        cells.InsertNextCell(3)
        for i in face:
            cells.InsertCellPoint(int(i))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(p)
    mesh.SetPolys(cells)
    if mode == "flat":
        gaps = numpy_to_vtk(c["_flat_gaps"], deep=True)
        gaps.SetName("Normal gap to mucosa (mm)")
        mesh.GetPointData().SetScalars(gaps)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"CONTACT_{candidate_id}_{mode}")
    node.SetAttribute("NINSContact.source", VERSION)
    node.SetAttribute("NINSContact.purpose", "Research contact face; no thickness or mechanics")
    node.SetAndObservePolyData(mesh)
    node.CreateDefaultDisplayNodes()
    display = node.GetDisplayNode()
    display.SetColor(.1, .8, .8) if mode == "conforming" else display.SetColor(1., .65, .1)
    display.SetOpacity(.85)
    display.SetBackfaceCulling(False)
    if mode == "flat":
        display.SetActiveScalarName("Normal gap to mucosa (mm)")
        display.SetScalarVisibility(True)
    return node
