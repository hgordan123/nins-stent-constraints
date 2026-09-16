# Start a pilot review in Slicer

The helper creates an annotation session for a selected scan and saves review observations. It does not segment anatomy or enroll patients automatically.

## Prepare the study workspace

Copy the files in `templates/` into a secure working directory outside the repository. Confirm which original series belongs to each unique patient before assigning P001-P012. Use P001-P003 for setup and P004-P012 for evaluation after the method is fixed. Keep all series and sides from one patient in the same group.

Load the intended CT in Slicer. Confirm original acquisition/reconstruction, adult eligibility, patient linkage, orientation, coverage, intensity units and contrast status. Prefer native submillimeter sinus CT for the primary pilot. Keep uncalibrated CBCT and 1 mm scans in the challenge group unless the protocol is revised.

## Optional archive inventory

For existing NRRD exports inside ZIP archives, the intake helper can check their geometry and raw payloads outside Slicer:

```bash
python3 scripts/NINS_Pilot_Intake.py /path/to/source-exports.zip --verify --output /secure/pilot/source-inventory.json
```

Choose a new output filename. Source IDs identify archive members, not patients. `stage(archive, entry_index, output, expected_sha256)` copies a supported raw, scalar 3D NRRD into a new temporary file with spatial/scalar header fields. It does not de-identify facial imaging or establish source provenance. Do not load an output after a failed verification.

## Create empty annotation objects

In the Slicer Python console:

```python
from pathlib import Path
import runpy

repo = Path('/absolute/path/to/nins-stent-constraints')
pilot = runpy.run_path(str(repo / 'scripts/NINS_Pilot_Review.py'))
print([(n.GetID(), n.GetName())
       for n in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')])

session = pilot['prepare'](
    volume_id='REPLACE_WITH_SELECTED_VOLUME_NODE_ID',
    source_id='SOURCE01',
    output_dir='/absolute/path/to/secure-pilot-data/reviews')
print(session['nodes'])
```

Select the exact volume ID from the printed list. The helper creates unique nodes associated with that source; their IDs are in `session['nodes']`.

| Annotation role | Operator input |
|---|---|
| SPF_R and SPF_L | One reviewed point inside each assessable SPF |
| CONTACT_R and CONTACT_L | One point on the intended exposed mucosal contact region |
| ENTRY_R and ENTRY_L | One point inside each ipsilateral lumen at the selected piriform entry |
| SPF_RIM_R and SPF_RIM_L | Ordered nasal-side aperture rim points; leave unseen portions unresolved |

The rim curves use linear interpolation. No aperture area or true Feret measurement is computed by this helper. Preserve separate independent references before showing algorithm results to the raters.

## Capture review state

```python
path, record = pilot['capture'](
    session,
    reviewer_id='RATER01',
    review={
        'patient_mapping_verified': False,
        'orientation_verified': False,
        'coverage_verified': False,
        'intensity_units_verified': False,
        'native_series_verified': False,
        'noncontrast_verified': False,
        'SPF_R_reviewed': False,
        'SPF_L_reviewed': False,
    },
    task_minutes=None)
print(record['status'], record['unresolved'], path)
```

Change each flag only after the named reviewer confirms it. `task_minutes` is measured active human work, not elapsed computer time. Every capture creates a new JSON observation with source geometry, numeric world-RAS coordinates, review flags and software/source fingerprints.

`ready_for_assisted_run` means the helper's bilateral SPF/intake checks passed on a submillimeter 3D source. It does not mean the contact patch, rim, segmentation or device is validated. An unassessable side stays in the records; the helper's bilateral readiness gate remains incomplete.

The helper refuses capture if the source image or geometry changed, annotations were reassigned/transformed, or a defined point lies outside the source. It does not alter original scans or the manual study dataset.

## Continue to measurement

Use the [stent engine guide](../RUNNING.md) and [contact geometry guide](CONTACT_GEOMETRY.md) after the required anatomy is reviewed. The helpers do not automatically translate node names, run the engines or append their outputs to the pilot CSVs. Preserve raw predictions and record accepted/rejected outputs and correction time separately.

Read the [automation roadmap](AUTOMATION_ROADMAP.md) for the planned integration.
