# Run the stent measurement engine

The existing `nins_stent_constraints.py` engine computes exploratory geometry from a selected CT and clinician-placed seeds. This publication preserves its v0.3.0 algorithm. The pilot adds intake and review around that engine; it does not establish stimulation effectiveness or a fabrication specification.

## Prepare the source and seeds

Verify the selected source, patient orientation, acquisition and intensity units. Use a working scene containing the intended source and its associated seeds. A point being inside the volume bounds does not prove it belongs to that scan. The engine's histogram-based calibration is an algorithmic estimate and cannot establish HU calibration for arbitrary CBCT.

| Point node name | Operator placement | Purpose |
|---|---|---|
| `SPF_R`, `SPF_L` | Reviewed point inside each assessable SPF | Local SPF geometry and surface-distance estimates |
| `piriform_R`, `piriform_L` | Inside the nasal airway at the selected anterior entry | Explicit anterior corridor reference |
| `choana_R`, `choana_L` | Inside the airway at the posterior choana | Explicit posterior corridor reference |
| `vidian_R`, `vidian_L` | Reviewed point in the vidian canal | Additional anatomical distance |
| `FR_R`, `FR_L` | Reviewed point in foramen rotundum | Additional anatomical distance |

The engine supports a single assessable side. Without explicit corridor endpoints, it attempts a geometric derivation that must be reviewed. Anatomical labels are supplied by the operator. Keep the entry and choana seeds in the nasal airway proper rather than an adjacent sinus.

The new pilot helper uses unique session-prefixed nodes. It does **not** automatically translate those into the legacy names above. Prepare correctly associated legacy seeds on a separate working scene. Retain independent reference annotations before exposing raters to algorithm results.

## Run in Slicer

```python
from pathlib import Path
import runpy

repo = Path('/absolute/path/to/nins-stent-constraints')
engine = runpy.run_path(str(repo / 'nins_stent_constraints.py'))
report = engine['run'](
    volume_node_name='EXACT_SELECTED_VOLUME_NAME',
    write_json_dir='/absolute/path/to/secure-pilot-data/new-run')
engine['print_report'](report)
```

Use a unique volume name and a new output directory for each run. Inspect all reported failures and flags. Save the original output before correction.

## Interpret the output

- `CONDUCTION GAP` is a coronal surface-to-SPF distance. The historical 5 mm budget does not establish stimulation.
- The reported aperture is a seeded minimum chord in its stated plane, not a true Feret width.
- The narrowest station describes a local section, not whole-device insertability or an accepted stent size.
- `compressible` is a geometric difference, not a tested mechanical property.
- Descriptive airway volumes depend on the algorithm's defined corridor and side boundaries. They are not named sinus volumes.

Review weak midline fits, refused corridors, abrupt profile changes, large SPF seed shifts, out-of-volume seeds and missing local lumens. Record failures rather than substituting a plausible number.

The method uses geometric priors, including a nasal half-width limit and bounded endpoint searches. Its bony contour is a star-shaped approximation. Disease, prior surgery, thin walls and artifact can violate those assumptions. Validation against reviewed pilot cases remains pending.

Use the [team workflow](docs/TEAM_WORKFLOW.md) to record accepted measurements, correction time, failures and the method version.
