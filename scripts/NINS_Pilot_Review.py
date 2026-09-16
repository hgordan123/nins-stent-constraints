"""Create empty, source-bound review markups and export pilot observations.

No automated anatomical claims or changes to existing volumes/landmarks.
Use runpy.run_path in the Slicer Python console. Each call creates a new session.
"""
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.1.0"
POINT_ROLES = ("SPF_R", "SPF_L", "CONTACT_R", "CONTACT_L", "ENTRY_R", "ENTRY_L")
CURVE_ROLES = ("SPF_RIM_R", "SPF_RIM_L")
REVIEW_FIELDS = ("patient_mapping_verified", "orientation_verified", "coverage_verified",
                 "intensity_units_verified", "native_series_verified", "noncontrast_verified",
                 "SPF_R_reviewed", "SPF_L_reviewed")


def _token(value):
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", value):
        raise ValueError("Use an anonymous alphanumeric ID with underscores")
    return value


def _write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def volume_record(volume_id):
    import slicer
    import vtk
    node = slicer.mrmlScene.GetNodeByID(volume_id)
    if node is None or not node.IsA("vtkMRMLScalarVolumeNode") or not node.GetImageData():
        raise ValueError("Choose a loaded scalar volume by node ID")
    if node.GetParentTransformNode():
        raise ValueError("Transformed volume requires a separately reviewed working copy")
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return dict(volume_id=volume_id, dimensions_ijk=list(node.GetImageData().GetDimensions()),
                spacing_mm=list(node.GetSpacing()), scalar_type=node.GetImageData().GetScalarTypeAsString(),
                scalar_range=list(node.GetImageData().GetScalarRange()),
                ijk_to_ras=[[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


def _array_hash(volume_id):
    import slicer
    import numpy as np
    array = np.ascontiguousarray(slicer.util.arrayFromVolume(slicer.mrmlScene.GetNodeByID(volume_id)))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def readiness(volume, review, points):
    """Conservative intake gate, not clinical validation or automatic enrollment."""
    reasons = [field for field in REVIEW_FIELDS if review.get(field) is not True]
    if max(volume["spacing_mm"]) >= 1.0 - 1e-6:
        reasons.append("not_submillimeter_primary_sampling")
    if min(volume["dimensions_ijk"]) <= 1:
        reasons.append("not_a_3d_volume")
    for role in ("SPF_R", "SPF_L"):
        if len(points.get(role, [])) != 1:
            reasons.append(role + "_needs_one_defined_point")
    return {"status": "ready_for_assisted_run" if not reasons else "review_incomplete",
            "unresolved": reasons}


def prepare(volume_id, source_id, output_dir):
    import slicer
    source_id = _token(source_id)
    volume = volume_record(volume_id)
    session_id = source_id + "_" + uuid.uuid4().hex[:8]
    directory = Path(output_dir) / session_id
    directory.mkdir(parents=True, exist_ok=False)
    session = dict(version=VERSION, session_id=session_id, source_id=source_id,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   started_monotonic=time.monotonic(), volume=volume,
                   volume_array_sha256=_array_hash(volume_id),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   slicer_version=slicer.app.applicationVersion,
                   source_status="unverified_source_not_enrolled_patient",
                   nodes={}, directory=str(directory))
    for role in POINT_ROLES + CURVE_ROLES:
        cls = "vtkMRMLMarkupsFiducialNode" if role in POINT_ROLES else "vtkMRMLMarkupsClosedCurveNode"
        n = slicer.mrmlScene.AddNewNodeByClass(cls, "PILOT_" + session_id + "_" + role)
        n.SetAttribute("NINSPilot.session", session_id)
        n.SetAttribute("NINSPilot.role", role)
        n.SetNodeReferenceID("NINSPilot.volume", volume_id)
        n.CreateDefaultDisplayNodes()
        n.GetDisplayNode().SetSelectedColor(0.2, 0.9, 0.8)
        if role in CURVE_ROLES:
            n.SetCurveTypeToLinear()
        session["nodes"][role] = n.GetID()
    _write_new(directory / "session.json", session)
    return session


def capture(session, reviewer_id, review=None, task_minutes=None):
    """Export an honest incomplete record if review or landmarks are missing.

reviewer_id must be a code. task_minutes is active human time, not wall time.
This export contains numeric geometry, IDs and flags only; no volume or DICOM.
"""
    import slicer
    import vtk
    import math
    _token(reviewer_id)
    if task_minutes is not None and (not math.isfinite(task_minutes) or task_minutes < 0):
        raise ValueError("Active task time must be finite and nonnegative")
    volume = volume_record(session["volume"]["volume_id"])
    if volume != session["volume"] or _array_hash(volume["volume_id"]) != session["volume_array_sha256"]:
        raise ValueError("Volume geometry or voxels changed since review setup")
    r2i = vtk.vtkMatrix4x4()
    slicer.mrmlScene.GetNodeByID(volume["volume_id"]).GetRASToIJKMatrix(r2i)
    points = {}
    for role, nid in session["nodes"].items():
        n = slicer.mrmlScene.GetNodeByID(nid)
        if n is None or n.GetAttribute("NINSPilot.session") != session["session_id"]:
            raise ValueError("Missing or mismatched review node: " + role)
        if n.GetParentTransformNode() or n.GetNodeReferenceID("NINSPilot.volume") != volume["volume_id"]:
            raise ValueError("Review node transform/source changed: " + role)
        points[role] = []
        for i in range(n.GetNumberOfControlPoints()):
            if n.GetNthControlPointPositionStatus(i) != n.PositionDefined:
                continue
            p = [0.0, 0.0, 0.0]
            n.GetNthControlPointPositionWorld(i, p)
            if not all(math.isfinite(x) for x in p):
                raise ValueError("Nonfinite control point")
            ijk = r2i.MultiplyPoint(p + [1.0])
            if any(ijk[a] < 0 or ijk[a] > volume["dimensions_ijk"][a]-1 for a in range(3)):
                raise ValueError("Control point outside source volume: " + role)
            points[role].append(p)
    review = {key: (review or {}).get(key) is True for key in REVIEW_FIELDS}
    result = dict(session_id=session["session_id"], source_id=session["source_id"],
                  capture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_array_sha256=session["volume_array_sha256"],
                  reviewer_id=reviewer_id, exported_utc=datetime.now(timezone.utc).isoformat(),
                  review=review, points_ras_mm=points, volume=volume,
                  active_task_minutes=task_minutes,
                  wall_elapsed_minutes=(time.monotonic()-session["started_monotonic"])/60,
                  **readiness(volume, review, points))
    result["not_assessed"] = ["SPG localization", "stimulation", "contact pressure",
                              "true Feret diameter", "stent insertion", "clinical validity"]
    path = Path(session["directory"]) / ("review_" + uuid.uuid4().hex[:10] + ".json")
    _write_new(path, result)
    return str(path), result
