
# Run Contact Geometry Optimizer

This tool explores electrode contact faces against a reviewed mucosal surface. It is a geometry and fit analysis, not a stimulation, pressure, deformation, or clinical efficacy model.

## Required scene inputs

Create an isolated open model patch for the mucosal surface over the intended contact region. Do not pass a closed airway model or a whole-head model. Add three one-point fiducials:

- `CONTACT_SEED`: inside the intended contact patch
- `AIR_REFERENCE`: on the airway side of the patch, used to orient the surface normal
- `SPF_TARGET`: the reviewed SPF reference point

All inputs must be in world RAS with no parent transform. The contact seed must be within 2 mm of the patch.

## Run in the Slicer Python console

```python
from pathlib import Path
import runpy
repo = Path('/absolute/path/to/nins-stent-constraints')
contact = runpy.run_path(str(repo / 'scripts/(C) NINS_Contact_Geometry_0912.py'))
run_slicer = contact['run_slicer']
show_candidate = contact['show_candidate']
report = run_slicer(
    "MucosaPatch",
    "CONTACT_SEED",
    "AIR_REFERENCE",
    "SPF_TARGET",
    footprints_mm=[(2.0, 4.0), (3.0, 5.0), (4.0, 6.0), (5.0, 7.0)],
    angles_deg=(0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
    offsets_mm=((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
    tolerance_mm=0.25,  # Example geometry comparison tolerance, not a validated design limit
    output_dir="/absolute/path/to/contact-runs")
```

The output is a bounded grid of candidates. `flat_pareto_ids` trades off flat electrode apposition, maximum normal gap, and distance to the SPF reference. `conforming_pareto_ids` reports the analogous tradeoff for an ideal conforming face. A Pareto candidate is not automatically a selected design.

To preview a reviewed candidate in Slicer:

```python
show_candidate(report, report["flat_pareto_ids"][0], mode="flat")
```

## How to interpret the result

The useful geometry outputs are contact fraction within the tolerance, maximum and mean normal gap, patch area, local surface relief and tilt, curvature, and target distance. Flat faces are intentionally evaluated separately from ideal conforming faces; conforming contact is an upper-bound geometry and does not claim the device will deform that way.

The report explicitly does not estimate electrical recruitment, cerebral perfusion, contact pressure, tissue deformation, retention, insertion collision, or device thickness clearance. Those need separate engineering or physiological models. Keep the SPF target semantics explicit: it is a reference point for placement, not a CT-visible SPG centroid.

## QC before using a candidate

Confirm the patch is the intended mucosal surface, the SPF target is clinician-reviewed, the normal points into the airway, and the candidate does not cross bone, septum, turbinate, or a second surface layer. Exported OBJ files are contact faces in RAS millimetres, not solid or fabrication-ready devices.

## Validation status

The numerical core has analytical tests for area, curvature, missing surfaces, rigid-transform invariance, Pareto filtering, and export. The VTK adapter tests require VTK. Clinical validity and real-subject performance remain unvalidated until reviewed patches and seeds are run.
