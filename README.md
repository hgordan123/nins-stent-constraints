# NINS SPG morphometry and stent geometry pilot

Research scripts for reviewing sinus CT anatomy, calculating morphometric measurements and exploring electrode contact geometry in 3D Slicer. The pilot combines human anatomical review with automated geometry and candidate segmentation.

The first milestone is a 12-patient feasibility pilot: three setup cases and nine evaluation cases after the method is fixed. The latest project records confirm no enrolled or clinician-reviewed pilot cases. The software is not a validated device-design or stimulation model.

## Start here

- [Team synopsis and working workflow](docs/TEAM_WORKFLOW.md)
- [Shareable Word document](docs/NINS%20SPG%20Pilot%20Team%20Workflow.docx)
- [Pilot setup and annotation guide](docs/PILOT_QUICKSTART.md)
- [Existing autosegmentation and proposed integration](docs/AUTOMATION_ROADMAP.md)
- [Stent measurement engine guide](RUNNING.md)
- [Contact geometry guide](docs/CONTACT_GEOMETRY.md)

## What is implemented

| Component | File | Current behavior |
|---|---|---|
| Image intake | `scripts/NINS_Pilot_Intake.py` | Inventories NRRD exports in ZIP archives, verifies payloads and stages a temporary copy |
| Guided review | `scripts/NINS_Pilot_Review.py` | Creates eight empty, source-associated annotation nodes and exports review state |
| Stent geometry | `nins_stent_constraints.py` | Existing v0.3.0 engine: seeded SPF refinement, coronal airway profiles and descriptive morphometry |
| Contact geometry | `scripts/(C) NINS_Contact_Geometry_0912.py` | Evaluates candidate electrode faces against a supplied, reviewed surface patch |
| Legacy autosegmentation QC | `scripts/(C) Autoseg_Measurements_0729.py` | Threshold tissue masks, surface/curve extraction and seeded bone-label proposals; anatomical review required |
| Manual measurement | `scripts/Landmark_Measurements_0510.py` | Computes the original study measurements from manually supplied lines and curves |
| Agreement analysis | `scripts/(C) Autoseg_Agreement_0729.py` | Compares compatible manual/automatic measurement tables |

The pilot review helper does not invoke the segmentation or measurement engines. The proposed unified Slicer panel and TotalSegmentator integration are **not implemented**. In the legacy QC script, `run()` defaults to the `engine` stage; segmentation stages must be requested explicitly.

## Measurement interpretation

Some unchanged legacy code uses labels that overstate what the geometry establishes. Apply these definitions when reviewing output:

| Legacy output | Interpretation for the pilot |
|---|---|
| `conduction_gap_mm`, `within_budget` | Coronal surface-to-SPF distance and comparison with a historical constant. The 5 mm constant is not a validated stimulation threshold. |
| `min_feret_aperture` | Shortest sampled chord through the seed in the named plane. It is not true aperture-plane Feret width. |
| `compressible` | Bony-minus-mucosal geometric difference. It does not establish safe tissue compression. |
| `stent diameter` | Local geometric diameter estimate. It does not establish whole-device insertion clearance. |
| Estimated SPG point | A derived reference from supplied anatomy, not a directly observed ganglion segmentation. |

The publication preserves the existing stent algorithm and its output keys for compatibility. Read the current definitions above before using historical examples or source comments.

## Requirements and checks

Run scene operations inside 3D Slicer with its bundled Python, NumPy, SciPy and VTK. The intake helper uses the Python standard library and can run outside Slicer. Contact numerical tests require NumPy; VTK adapter tests run only when VTK is available.

From the repository directory:

```bash
python3 scripts/tests/test_pilot_intake.py
PythonSlicer scripts/tests/test_contact_geometry.py
PythonSlicer test_calibration.py
```

Use the `PythonSlicer` executable in your Slicer installation. On macOS, its usual path is `/Applications/Slicer.app/Contents/bin/PythonSlicer`. The calibration suite requires VTK; ordinary system Python may not include it.

These checks exercise intake, review readiness rules and synthetic geometry/calibration. They do not validate clinical anatomy, stimulation, tissue pressure or insertion. The full legacy QC pipeline needs a separate Slicer validation run on reviewed cases. The original manual-arm repetitions and dataset remain separate from this pilot.

## Data handling

Copy the [blank templates](templates/) into your secure study workspace before entering case information. Keep original imaging, patient linkage, landmarks, segmentations and per-case outputs outside this repository. No patient scans, patient mappings or case-derived meshes are included in this publication.

The Word document is a dated team snapshot. The Markdown guides use repository-relative links and are the working instructions for this repository.
