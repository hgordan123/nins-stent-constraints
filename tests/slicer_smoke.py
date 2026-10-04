"""Run with Slicer --ignore-slicerrc --python-script tests/slicer_smoke.py.

Uses only generated volumes in its own Slicer process. Never run in a patient scene.
"""
from pathlib import Path
import json
import sys
import tempfile
import traceback

import numpy as np
import slicer
import vtk

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"NINSPilot"))
from pilot_logic import PilotLogic


def point(node, p):
    node.AddControlPoint(vtk.vtkVector3d(*map(float, p)))


def run():
    root = Path(tempfile.mkdtemp(prefix="nins-slicer-smoke-"))
    slicer.util.mainWindow().resize(1450, 1050)
    log = []
    def ok(label, condition):
        assert condition, label
        log.append(label)
        print("PASS:", label, flush=True)
    def refuses(label, fn):
        try:
            fn()
        except ValueError:
            ok(label, True)
        else:
            raise AssertionError(label)
    for kind in ("straight", "narrow", "artifact"):
        logic = PilotLogic()
        volume = logic.demo(kind)
        logic.start(volume, "DEMO_"+kind, "TEST", root, cohort="synthetic", synthetic=True)
        settings = dict(logic.state["settings"])
        settings["patch_radius_mm"] = 3.5
        sides = dict(R=dict(status="assessable", reason=""), L=dict(status="unassessable", reason="Synthetic one-side check"))
        logic.set_review(dict.fromkeys(logic.intake_fields, True), sides, settings)
        point(logic.node("SPF_R"), [7,0,6])
        for t in np.linspace(0, 2*np.pi, 32, endpoint=False):
            point(logic.node("RIM_R"), [7+2*np.cos(t), 2*np.sin(t),6])
        radius = 8 if kind == "straight" else 5
        point(logic.node("CONTACT_R"), [7,0,(radius-.5)*.5])
        point(logic.node("AIR_R"), [7,0,0])
        for y in (-10,-5,0):
            point(logic.node("ROUTE_R"), [7,y,0])
        refuses(kind+" generation needs reference", lambda: logic.generate())
        logic.freeze_reference()
        logic.generate()
        refuses(kind+" duplicate prediction refused", lambda: logic.generate())
        logic.build_patch("R")
        refuses(kind+" measurement needs acceptance", logic.measure)
        logic.accept()
        report = logic.measure()
        ok(kind+" one-side measurement", report["sides"]["L"]["status"] == "unassessable")
        ok(kind+" aperture dimension", abs(report["sides"]["R"]["aperture"]["minimum_feret_mm"]-4) < .03)
        ok(kind+" 3D distance available", report["sides"]["R"]["surface_to_spf"]["distance_mm"] > 0)
        if kind != "artifact":
            ok(kind+" contact candidates evaluated", any(c["status"] == "evaluated" for c in report["sides"]["R"]["contact"]["candidates"]))
        export = Path(logic.export())
        ok(kind+" export report", (export/"report.json").exists())
        captures = json.loads((export/"screenshots.json").read_text())
        ok(kind+" slice evidence", len(captures) == 3 and all(c["saved"] for c in captures))
        ok(kind+" independent initial mask preserved", (Path(logic.state["directory"])/"prediction.seg.nrrd").exists())
        checkpoint = logic.save_progress()
        roi = logic.node("roi")
        original_size = list(roi.GetSize())
        roi.SetSize(original_size[0]-1, original_size[1], original_size[2])
        refuses(kind+" ROI change cannot relabel old prediction", logic.accept)
        roi.SetSize(*original_size)
        restored = PilotLogic()
        restored.resume(checkpoint, volume)
        ok(kind+" resume needs review", restored.state["accepted"] is None)
        restored.accept()
        restored.measure()
        mask = restored.segmentation_array("air")
        mask[32,40,54] = 1-mask[32,40,54]
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, restored.node("segmentation"), "air", volume)
        refuses(kind+" edited mask invalidates patch", restored.accept)
        dirty = restored.save_progress()
        dirty_restore = PilotLogic()
        dirty_restore.resume(dirty, volume)
        refuses(kind+" stale patch remains stale after resume", dirty_restore.accept)
        mask[32,40,54] = 1-mask[32,40,54]
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, restored.node("segmentation"), "air", volume)
        restored.accept()
        restored.measure()
        restored.node("SPF_R").SetNthControlPointPosition(0, 6.9, 0, 6)
        refuses(kind+" landmark edit invalidates export", restored.export)
        array = slicer.util.arrayFromVolume(volume)
        array[0,0,0] += 1
        slicer.util.arrayFromVolumeModified(volume)
        refuses(kind+" changed source refused", restored.accept)
    # Load the actual module, create its complete UI, and run its demo callback.
    import qt
    factory = slicer.app.moduleManager().factoryManager()
    factory.registerModule(qt.QFileInfo(str(REPO/"NINSPilot"/"NINSPilot.py")))
    factory.loadModules(["NINSPilot"])
    slicer.util.selectModule("NINSPilot")
    widget = slicer.modules.ninspilot.widgetRepresentation().self()
    widget.output.currentPath = str(root)
    widget.status.text = widget.demo()
    widget.role.setCurrentIndex(widget.role.findText("SPF_R"))
    widget.place()
    ok("panel landmark placement selects session node", slicer.app.applicationLogic().GetSelectionNode().GetActivePlaceNodeID() == widget.logic.node("SPF_R").GetID())
    slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
    widget.reviewed.checked = True
    widget.accept()
    widget.measure()
    widget.export()
    ok("panel synthetic flow", widget.logic.report["synthetic"] is True)
    slicer.app.processEvents()
    slicer.util.mainWindow().grab().save(str(root/"panel.png"))
    write = dict(status="passed", checks=log, count=len(log), output=str(root), slicer=slicer.app.applicationVersion)
    (root/"validation.json").write_text(json.dumps(write, indent=2))
    print("NINS_VALIDATION", json.dumps(write), flush=True)


try:
    run()
except Exception:
    traceback.print_exc()
    import sys
    slicer.util.exit(1)
else:
    # Slicer's module factory may remove sys from the shared console namespace.
    # Its command-line script wrapper still references it while returning.
    import sys
    slicer.util.exit(0)
