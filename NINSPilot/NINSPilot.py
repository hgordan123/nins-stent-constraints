"""Guided research workflow, loaded through Slicer's Additional module paths."""
import json
from pathlib import Path
import sys

import ctk
import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget

MODULE_DIR = str(Path(__file__).resolve().parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
from pilot_logic import PilotLogic


class NINSPilot(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = _("NINS Pilot")
        parent.categories = [_("Research")]
        parent.dependencies = ["Segmentations", "Markups", "SegmentEditor"]
        parent.contributors = ["NINS research team"]
        parent.helpText = _("Assisted CT segmentation, reviewed SPF morphometry and contact geometry. Research prototype. Start with the synthetic demonstration and read docs/PILOT_PANEL.md in the repository.")
        parent.acknowledgementText = ""


class NINSPilotWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = PilotLogic()
        self.busy = False
        self.sections = []
        self.status = qt.QLabel(_("Start a session or load the synthetic demonstration."))
        self.status.wordWrap = True
        self.layout.addWidget(self.status)
        f = self.section(_("1. Source and session"))
        self.volume = self.selector(["vtkMRMLScalarVolumeNode"])
        f.addRow(_("Source CT"), self.volume)
        self.case = qt.QLineEdit()
        self.case.placeholderText = _("Coded case ID, for example P001")
        self.reviewer = qt.QLineEdit()
        self.reviewer.placeholderText = _("Coded reviewer ID")
        self.output = ctk.ctkPathLineEdit()
        self.output.filters = ctk.ctkPathLineEdit.Dirs
        self.output.currentPath = str(Path.home()/"NINS-pilot-data")
        self.cohort = qt.QComboBox()
        for title, data in [(_("Setup"), "setup"), (_("Challenge"), "challenge"), (_("Synthetic only"), "synthetic")]:
            self.cohort.addItem(title, data)
        f.addRow(_("Case"), self.case)
        f.addRow(_("Reviewer"), self.reviewer)
        f.addRow(_("Cohort"), self.cohort)
        f.addRow(_("Secure study directory"), self.output)
        self.button(f, _("Start session"), self.start)
        self.button(f, _("Resume checkpoint with selected CT"), self.resume)
        self.button(f, _("Load synthetic demonstration"), self.demo)
        self.intake = {}
        for key, title in zip(self.logic.intake_fields, [_("Patient/source mapping verified"), _("Patient orientation verified"), _("Native series and sampling verified"), _("Target coverage reviewed"), _("Noncontrast series confirmed"), _("CT intensity units verified as HU")]):
            c = qt.QCheckBox(title)
            self.intake[key] = c
            f.addRow(c)
        self.side_checks, self.side_reasons = {}, {}
        for side in ("R", "L"):
            check = qt.QCheckBox(_("Assessable"))
            check.checked = True
            reason = qt.QLineEdit()
            reason.placeholderText = _("Required if unassessable")
            self.side_checks[side], self.side_reasons[side] = check, reason
            f.addRow(_("Side ")+side, check)
            f.addRow(_("Reason ")+side, reason)
        self.concept = qt.QComboBox()
        for title, data in [(_("Not specified"), "unspecified"), (_("Mucosal contact concept"), "mucosal_contact"), (_("Transforaminal concept"), "transforaminal_concept")]:
            self.concept.addItem(title, data)
        f.addRow(_("Electrode concept"), self.concept)

        f = self.section(_("2. Independent reference annotations"))
        text = qt.QLabel(_("Before displaying candidate segmentation, place SPF points and ordered rim points on each assessable side. Keep a second rater's annotations in a separate session."))
        text.wordWrap = True
        f.addRow(text)
        self.role = qt.QComboBox()
        self.role.addItems(list(self.logic.roles))
        f.addRow(_("Annotation"), self.role)
        self.button(f, _("Place selected annotation"), self.place)
        self.button(f, _("Preserve reference before segmentation"), self.freeze)
        self.phase = qt.QComboBox()
        self.phase.addItem(_("Reference annotation"), "reference")
        self.phase.addItem(_("Segmentation correction"), "correction")
        f.addRow(_("Active work timer"), self.phase)
        self.button(f, _("Start / resume timer"), self.startTimer)
        self.button(f, _("Pause timer"), self.pauseTimer)
        self.timer_label = qt.QLabel()
        f.addRow(self.timer_label)

        f = self.section(_("3. Candidate segmentation and correction"))
        text = qt.QLabel(_("Resize the visible ROI to include the nasal airway and target region. Thresholding creates air/bone proposals; inspect sinus leakage, missing walls and artificial ROI edges. Use a separate session to compare methods."))
        text.wordWrap = True
        f.addRow(text)
        self.backend = qt.QComboBox()
        for title, data in [(_("HU threshold proposals"), "threshold"), (_("Manual baseline: empty masks"), "manual"), (_("TotalSegmentator nasal initialization"), "totalsegmentator")]:
            self.backend.addItem(title, data)
        f.addRow(_("Method"), self.backend)
        self.params = {}
        for key, title, value, low, high in [
            ("air_max_hu", _("Air upper threshold (HU)"), -300., -1023., 1000.),
            ("bone_min_hu", _("Bone lower threshold (HU)"), 300., -500., 4000.),
            ("patch_radius_mm", _("Local patch radius (mm)"), 8., 1., 30.),
            ("electrode_width_mm", _("Candidate face width (mm)"), 2., .1, 20.),
            ("electrode_length_mm", _("Candidate face length (mm)"), 3., .1, 30.),
            ("geometric_tolerance_mm", _("Exploratory gap tolerance (mm)"), .25, .01, 5.)]:
            spin = qt.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.decimals = 3
            spin.value = value
            self.params[key] = spin
            f.addRow(title, spin)
        self.button(f, _("Generate and preserve initial masks"), self.generate)
        self.editor = slicer.qMRMLSegmentEditorWidget()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.editor.setMRMLSegmentEditorNode(self.editor_node)
        self.editor.setEffectNameOrder(["Paint", "Erase", "Draw", "Scissors", "Islands", "Grow from seeds", "Smoothing"])
        self.editor.unorderedEffectsVisible = False
        f.addRow(self.editor)

        f = self.section(_("4. Review geometry and calculate"))
        text = qt.QLabel(_("Place CONTACT on the exposed surface and AIR inside adjacent air. ROUTE is an optional entry-to-contact polyline. Build each local patch after correcting the air mask, then inspect its anatomy. Patch apposition tolerance is an exploratory setting, not an accepted design limit."))
        text.wordWrap = True
        f.addRow(text)
        self.button(f, _("Build right contact patch"), lambda: self.patch("R"))
        self.button(f, _("Build left contact patch"), lambda: self.patch("L"))
        self.reviewed = qt.QCheckBox(_("I reviewed the masks, rims and any contact patches in the source CT"))
        f.addRow(self.reviewed)
        self.button(f, _("Accept current geometry"), self.accept)
        self.button(f, _("Calculate measurements"), self.measure)
        self.results = qt.QPlainTextEdit()
        self.results.readOnly = True
        self.results.minimumHeight = 220
        f.addRow(self.results)

        f = self.section(_("5. Save and export"))
        self.button(f, _("Save progress / refusal record"), self.save)
        self.button(f, _("Export measurements and slice evidence"), self.export)
        self.layout.addStretch(1)
        self.showStep(0)
        self.timer = qt.QTimer()
        self.timer.connect("timeout()", self.updateClock)
        self.timer.start(1000)

    def section(self, title):
        box = ctk.ctkCollapsibleButton()
        box.text = title
        self.layout.addWidget(box)
        self.sections.append(box)
        return qt.QFormLayout(box)

    def showStep(self, index):
        for i, box in enumerate(self.sections):
            box.collapsed = i != index

    def selector(self, types):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = types
        selector.noneEnabled = True
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def button(self, form, title, action):
        button = qt.QPushButton(title)
        button.connect("clicked()", lambda: self.perform(action))
        form.addRow(button)
        return button

    def perform(self, action):
        if self.busy:
            return
        self.busy = True
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.status.text = _("Working...")
            slicer.app.processEvents()
            message = action()
            self.status.text = str(message or _("Done. Review the result before continuing."))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.status.text = str(exc)
            slicer.util.errorDisplay(str(exc))
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.busy = False

    def sync(self):
        sides = {s: dict(status="assessable" if self.side_checks[s].checked else "unassessable", reason=self.side_reasons[s].text) for s in ("R", "L")}
        settings = {k: w.value for k, w in self.params.items()}
        settings["contact_mode"] = self.concept.currentData
        self.logic.set_review({k: w.checked for k, w in self.intake.items()}, sides, settings)

    def restoreFields(self):
        state = self.logic.state
        self.case.text, self.reviewer.text = state["case_id"], state["reviewer_id"]
        self.cohort.setCurrentIndex(self.cohort.findData(state["cohort"]))
        for k, v in state["intake"].items():
            self.intake[k].checked = v
        for side, rec in state["sides"].items():
            self.side_checks[side].checked = rec["status"] == "assessable"
            self.side_reasons[side].text = rec["reason"]
        for k, w in self.params.items():
            w.value = state["settings"][k]
        self.concept.setCurrentIndex(self.concept.findData(state["settings"]["contact_mode"]))
        self.reviewed.checked = False
        self.results.clear()
        self.editor.setSegmentationNode(self.logic.node("segmentation") if "segmentation" in state["nodes"] else None)
        if "segmentation" in state["nodes"]:
            self.editor.setSourceVolumeNode(self.logic.node("source"))

    def start(self):
        self.logic.start(self.volume.currentNode(), self.case.text, self.reviewer.text, self.output.currentPath,
                         cohort=self.cohort.currentData, synthetic=self.cohort.currentData == "synthetic")
        self.restoreFields()
        self.logic.start_clock("reference")
        self.showStep(0)
        return _("Session started. Verify source checks and place reference annotations.")

    def resume(self):
        path = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), _("Choose checkpoint session.json"), self.output.currentPath, "JSON (*.json)")
        if not path:
            return _("Resume canceled")
        self.logic.resume(path, self.volume.currentNode())
        self.restoreFields()
        return _("Progress loaded. Review and accept again before measurement.")

    def place(self):
        n = self.logic.node(self.role.currentText)
        slicer.modules.markups.logic().SetActiveList(n)
        slicer.modules.markups.logic().StartPlaceMode(1 if self.role.currentText.startswith(("RIM", "ROUTE")) else 0)
        return _("Click in the CT slices to place points. Press Escape when finished.")

    def freeze(self):
        self.sync()
        self.logic.freeze_reference()
        self.showStep(2)
        return _("Reference preserved. Candidate segmentation can now be generated.")

    def generate(self):
        self.sync()
        self.editor.setSegmentationNode(self.logic.generate(self.backend.currentData))
        self.editor.setSourceVolumeNode(self.logic.node("source"))
        self.reviewed.checked = False
        self.showStep(2)
        return _("Original masks saved. Correct the editable candidates; correction timer is running.")

    def patch(self, side):
        self.sync()
        self.logic.build_patch(side)
        self.reviewed.checked = False
        self.showStep(3)
        return _("Patch created. Inspect its location and surface before accepting.")

    def accept(self):
        self.sync()
        if not self.reviewed.checked:
            raise ValueError(_("Confirm anatomical review before accepting"))
        self.logic.accept()
        self.reviewed.checked = False
        return _("Current geometry accepted. Later changes require new acceptance.")

    def measure(self):
        self.sync()
        report = self.logic.measure()
        lines = [_("Research geometry; not a fabrication specification.")]
        for side, rec in report["sides"].items():
            lines.append("\n"+side+": "+rec["status"])
            if rec["status"] == "unassessable":
                lines.append(rec["reason"])
                continue
            for key in ("projected_area_mm2", "minimum_feret_mm", "maximum_feret_mm", "plane_rms_mm", "plane_max_residual_mm"):
                lines.append(key+": %.3f" % rec["aperture"][key])
            if "surface_to_spf" in rec:
                lines.append("Local surface-to-SPF 3D distance: %.3f mm" % rec["surface_to_spf"]["distance_mm"])
            lines.append("Route: "+rec["route"]["status"])
            contact = rec["contact"]
            candidates = contact.get("candidates", [])
            lines.append("Contact candidates evaluated: %d/%d" % (sum(c["status"] == "evaluated" for c in candidates), len(candidates)))
            if contact.get("reason"):
                lines.append(contact["reason"])
        self.results.setPlainText("\n".join(lines))
        self.showStep(3)
        return _("Measurements calculated. Export retains missing and refused results.")

    def save(self):
        self.sync()
        self.logic.pause_clock()
        return _("Checkpoint saved: ")+self.logic.save_progress()

    def export(self):
        self.sync()
        return _("Export saved: ")+self.logic.export()

    def startTimer(self):
        if not self.logic.state:
            raise ValueError(_("Start a session first"))
        self.logic.start_clock(self.phase.currentData)
        return _("Active work timer started. Pause it when leaving the task.")

    def pauseTimer(self):
        self.logic.pause_clock()
        return _("Timer paused")

    def updateClock(self):
        times = self.logic.elapsed()
        self.timer_label.text = _("Reference: %.1f min | Correction: %.1f min | %s") % (times.get("reference", 0)/60, times.get("correction", 0)/60, self.logic.clock_phase or _("paused"))

    def demo(self):
        v = self.logic.demo()
        self.volume.setCurrentNode(v)
        self.case.text, self.reviewer.text = "DEMO", "DEMO_REVIEWER"
        self.cohort.setCurrentIndex(self.cohort.findData("synthetic"))
        self.start()
        self.params["patch_radius_mm"].value = 3.5
        for check in self.intake.values():
            check.checked = True
        import numpy as np
        for side, x in (("R", 7.), ("L", -7.)):
            self.logic.node("SPF_"+side).AddControlPoint(vtk_vector(x, 0, 6))
            for t in np.linspace(0, 2*np.pi, 32, endpoint=False):
                self.logic.node("RIM_"+side).AddControlPoint(vtk_vector(x+2*np.cos(t), 2*np.sin(t), 6))
            self.logic.node("CONTACT_"+side).AddControlPoint(vtk_vector(x, 0, 3.7))
            self.logic.node("AIR_"+side).AddControlPoint(vtk_vector(x, 0, 0))
            for y in (-12., -6., 0.):
                self.logic.node("ROUTE_"+side).AddControlPoint(vtk_vector(x, y, 0))
        self.sync()
        self.logic.freeze_reference()
        self.editor.setSegmentationNode(self.logic.generate("threshold"))
        for side in ("R", "L"):
            self.logic.build_patch(side)
        self.editor.setSourceVolumeNode(v)
        slicer.util.resetThreeDViews()
        self.showStep(3)
        return _("Synthetic fixture ready. Review the masks and patches, then accept and calculate. This is not a pilot patient.")

    def cleanup(self):
        self.timer.stop()
        self.logic.pause_clock()
        self.editor.setMRMLSegmentEditorNode(None)
        if self.editor_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.editor_node)


def vtk_vector(x, y, z):
    import vtk
    return vtk.vtkVector3d(float(x), float(y), float(z))
