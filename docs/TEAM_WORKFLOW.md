# NINS SPG Pilot Team Workflow

Team synopsis and working workflow

September 13, 2026 | Version 1.0

Project lead: Dr. Alfred-Marc Iloreta

## Purpose and current status

We are building a repeatable imaging workflow to describe nasal anatomy relevant to an intranasal stent intended to stimulate the sphenopalatine ganglion (SPG) from the sphenopalatine foramen (SPF) region. Our immediate aim is to establish which measurements are reproducible, where the automation fails, and how much expert review each case requires.

The first milestone is a **12-patient feasibility pilot**, with three setup cases followed by nine evaluation cases after the method is fixed. We will use 3D Slicer for image review, annotation and measurement, building on the existing scripts. A future trained model can automate more of the workflow once we have consistent reference annotations.

The distinction between the SPF and the SPG matters throughout the study. The SPF is a bony anatomical reference; it contains neurovascular structures and is covered by mucosa. An estimated distance to the SPF is not a measured distance to the ganglion or evidence of effective electrical stimulation. [1]

| Pilot item | Status on September 12 |
|---|---|
| Patient slots | 12 prepared; no enrolled patients confirmed in the pilot records |
| Imaging candidates | 8 integrity-checked volumes: 6 submillimeter and 2 at 1 mm |
| Patient identity and overlap | Pending confirmation; volume count is not patient count |
| First source | SRC_A02_E09, approximately 0.508 x 0.508 x 0.500 mm; staged in Slicer for review |
| Intake software | Seven tests passed; review-session creation and incomplete export exercised in Slicer |
| Clinical review | SPF annotations, source eligibility and measurement acceptance remain pending |

### What we need from the first team review

Confirm the first eligible scan and its patient linkage, identify the bilateral SPF and intended contact region, assign the primary and second raters, and agree on the definitions we will use for the three setup cases. The electrode configuration remains an open design decision: contact on intact mucosa or extension through the SPF.

The pilot will produce an anatomical measurement dataset, reviewed examples, a failure log and an estimate of operator effort. Mechanical fit, tissue contact pressure and electrical recruitment will require subsequent validation.



## Pilot design and case workflow

### Select and group patients

Use native submillimeter sinus CT for the primary pilot. Confirm adult eligibility, original series provenance, intensity units, contrast status, orientation and coverage before quantitative use. Keep 1 mm scans, contrast studies and uncalibrated cone-beam CT in a separately reported challenge group unless the protocol is revised.

P001-P003 are setup cases. Previously used atlas or debugging cases belong in this group. P004-P012 are reserved for evaluation after definitions, software and tolerances are fixed. Keep all series, sides and repeated annotations from one patient in the same group. An evaluation patient used for tuning becomes a setup case and needs a replacement.

Seek a range of anatomy, including straightforward cases, septal deviation, concha bullosa, mucosal disease, dental artifact and prior surgery. These are selection targets; the current source candidates have not yet been assigned these features. Twelve patients is a practical starting target, not a statistical sample-size justification.

### Run each case in this order

| Step | Responsible role | Record required before moving on |
|---|---|---|
| 1 Intake | Coordinator and clinician | Unique coded patient, selected series, provenance, geometry and eligibility decision |
| 2 Independent reference | Primary and second raters | Separate SPF annotations before either rater sees algorithm results; retain both and an adjudicated reference |
| 3 Local surface review | Rater with clinician review | Exposed air-tissue interface, bony boundaries, contact region and insertion-entry landmark |
| 4 Assisted measurement | Imaging analyst | Versioned input and configuration; original algorithm output saved before correction |
| 5 Correction and QC | Rater and clinician | Accepted or rejected anatomy, changed outputs, failure reason and active correction time |
| 6 Export | Imaging analyst and coordinator | Reviewed landmarks and surfaces, measurement rows, slice evidence and review record |

Repeat blinded reference placement on three preselected pilot patients, including a difficult case, in three separate sessions with earlier annotations hidden. This pilot exercise does not replace the original manual study's three-repetition requirement.

Use a working scene containing the selected source and its associated annotations. Preserve the original manual dataset and prior runs. An unassessable side or a refused measurement stays visible in the record and in the reported denominator.



## Measurements and interpretation

Every reported value needs a defined anatomical reference, measurement domain, unit, method and review status. Compare automatic and manual values only when they measure the same quantity.

| Design question | Measurement | Current capability |
|---|---|---|
| Where is the SPF reference | Bilateral point and ordered nasal-side aperture rim | Empty review nodes and coordinate export are implemented; anatomical placement requires review |
| What is the aperture geometry | Aperture plane, area and minimum and maximum Feret widths | True aperture-plane and Feret analysis still needs implementation |
| How far is the exposed surface from the SPF | Surface-to-reference distance with slice or 3D domain specified | Existing script provides a coronal-section estimate |
| What local dimensions can accommodate the device | Width, height, area and inscribed diameter along a defined route | Existing coronal-station profile is exploratory |
| How does an electrode face meet the mucosa | Reviewed patch orientation, curvature, apposition and gap | Contact optimizer exists; accepted anatomical cases remain pending |
| Can the device be inserted and remain seated | Swept-volume collision, deformation, contact pressure and retention | Subsequent engineering work |
| Where is the neural target relative to the electrode | Observed or estimated SPG location with uncertainty | These CT exports do not establish an observed ganglion location |

### Interpret the existing scripts consistently

The output currently called **conduction gap** is a surface-to-SPF distance. Record it as such. The existing 5 mm comparison does not establish a stimulation threshold.

The function currently called **minimum Feret aperture** searches for the shortest sampled chord through a seed. Record it as a seeded minimum chord in the named plane. True Feret width is the separation of parallel supporting lines around a contour.

A bony-minus-mucosal dimension is a geometric difference. It does not establish safely compressible tissue. A local inscribed diameter also does not demonstrate whole-device insertability. The ideal conforming electrode face is an upper-bound geometry, not a validated deformation model.

### Keep measurement provenance

The measurement table records patient, side, run, rater, repetition, metric, value, unit, method, reference, plane or station, review status, failure reason and configuration fingerprint. The review record links these outputs to the source geometry and software version. Missing or rejected values remain labeled rather than being silently replaced.



## Team responsibilities and decisions

Dr. Iloreta leads the project and adjudicates anatomy. The team will fill the remaining roles.

| Role | Responsibility | Assigned person |
|---|---|---|
| Clinical lead | Accept anatomy, contact configuration and definitions | Dr. Iloreta |
| Study coordinator | Maintain eligibility, linkage and complete case records | To assign |
| Primary rater | Annotate references, review surfaces and log effort | To assign |
| Second clinician rater | Place independent references and assess disagreements | To assign |
| Imaging and software analyst | Version scripts, retain raw results and maintain exports | To assign |
| Device engineer | Translate geometry into fit requirements and tests | To assign |
| Analysis lead | Report agreement, failures and operator effort | To assign |

### Working action tracker

Update owners and resolutions at each review. Date and version every method change.

| Action or decision | Proposed owner | When needed | Status or resolution |
|---|---|---|---|
| Confirm source eligibility and patient mapping | Coordinator and clinical lead | Before first case | Open |
| Choose mucosal contact or extension through SPF | Clinical lead and engineer | Before contact design | Open |
| Assign raters and repeatability cases | Clinical lead | Before annotation | Open |
| Define SPF rim, patch, entry and planes | Clinical lead and raters | During P001-P003 | Open |
| Set metric-specific tolerances and freeze the method | Clinical, engineering and analysis leads | Before P004-P012 | Open |
| Select atlas refinement or model training | Imaging and clinical leads | After pilot results | Open |

Assign calendar dates once case supply and reviewer time are confirmed.



## Case review worksheet

Copy per case and session. Use coded identifiers; retain the identity link in the institutional record.

**Patient code:** ____________________  **Source ID:** ____________________

**Rater code:** ______________________  **Review date:** __________________

**Run or session ID:** __________________  **Record location:** __________________

**Group:** [ ] Setup  [ ] Evaluation  [ ] Challenge  [ ] Undecided

### Intake

Record each item as confirmed, unresolved or unsuitable; verify acquisition and laterality from source records.

| Check | Finding or unresolved issue |
|---|---|
| Unique patient mapping and adult eligibility | |
| Selected original series and reconstruction | |
| Voxel spacing and anatomical coverage | |
| Intensity units and contrast status | |
| Patient orientation and right-left correspondence | |
| Anatomy, disease, prior surgery and image artifact | |

### Side review

| Observation | Right | Left |
|---|---|---|
| SPF reference and rim assessable | | |
| References saved and disagreements resolved | | |
| Contact surface and entry landmark reviewed | | |
| Algorithm anatomy accepted or correction needed | | |
| Unassessable measurement or failure reason | | |

**Active reference time:** __________ min  **Active correction time:** __________ min

**Case disposition:** [ ] Continue  [ ] Needs correction  [ ] Challenge  [ ] Unsuitable

**Reviewer decision and next action:** ______________________________________

______________________________________________________________________



## Analysis and the next milestone

### What we will report

Report unique patients and assessable sides, the proportion of attempted outputs accepted without correction, failure reasons, active annotation and correction time, and measurement errors against the matched reference. Keep both sides clustered within patients and preserve all failed or missing outputs in the accounting.

Evaluate local boundary error in millimeters for surfaces; whole-volume overlap alone can conceal errors around the small target region. Report independent localization error only when the algorithm was not given that target point. For a manually seeded method, report refinement displacement and repeatability separately.

Use paired measurement differences and exploratory agreement summaries, including Bland-Altman analysis where appropriate. The small pilot will not establish population coverage, clinical effectiveness or model generalization. Measurement-specific acceptance tolerances remain to be agreed before evaluation cases begin.

### Milestone completion

The pilot is complete when 12 unique patients have documented intake decisions, evaluation cases have reviewed outcomes or explicit refusals, repeatability observations are collected, and a report describes measurement error, failures and operator effort. The report should support a specific decision about which automation step to develop next.

If assisted measurements are reliable but localization remains the bottleneck, use reviewed reference cases to compare atlas-based localization with a dedicated learned localizer. If surface interpretation or measurement definitions remain unstable, resolve those before increasing model complexity. A custom nnU-Net model and a Slicer-connected annotation workflow are possible later tools, not implemented pilot capabilities. [4, 5]

### Working records and implementation

Use the [pilot console guide](PILOT_QUICKSTART.md) with the [blank templates](../templates/). Maintain the patient roster, source tracking, side review, measurement dictionary, measurements and method-freeze record in your secure study workspace. The original study folder remains NINS / SPG Anatomy Study.

The intake helper inventories and verifies image exports. The review helper creates empty source-associated annotations and exports review state. Segmentation and the existing measurement or contact-geometry engines require additional reviewed inputs and a separate run. Original manual-arm data retain their existing definitions.

### References

1. Crespi J et al. Measurement and implications of the distance between the sphenopalatine ganglion and nasal mucosa. Journal of Headache and Pain. 2018;19:14. [Anatomical imaging study](https://pubmed.ncbi.nlm.nih.gov/29442191/).
2. 3D Slicer. [Markups script repository](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/markups.html).
3. 3D Slicer. [DICOM script repository](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/dicom.html).
4. MIC-DKFZ. [nnU-Net project and documentation](https://github.com/MIC-DKFZ/nnUNet).
5. Project MONAI. [MONAI Label](https://project-monai.github.io/label.html).

