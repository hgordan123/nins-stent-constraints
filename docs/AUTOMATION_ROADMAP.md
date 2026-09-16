# Assisted segmentation workflow

Status on September 13, 2026: existing segmentation components are published here; a unified Slicer panel and pretrained-model integration are proposed work.

## Segmentation already in the code

`scripts/(C) Autoseg_Measurements_0729.py` contains:

1. `stage_tissue`: intensity-based air, enclosed-air, body, bone, dense-bone and candidate metal masks, plus island cleanup and threshold-sensitivity checks.
2. `stage_curves`: surface reconstruction and curve extraction in specified planes, using supplied anatomical references. Ambiguous results can be refused.
3. `stage_bones`: Slicer Grow from seeds proposals for palatine, pterygoid, sphenoid and other bone. The outputs remain marked for confirmation.

These components belong to the older manual-versus-automatic QC arm. They require its source and landmark setup. `run()` defaults to `stages=('engine',)` and does not automatically execute all segmentation stages. The pilot helper does not invoke this pipeline.

The enclosed-air mask removes slice-border-connected components. That heuristic can remove true airway communicating with outside air; it is not a validated complete nasal airway label. Likewise, a bone threshold cannot independently identify the SPF or separate named contiguous bones reliably.

## Proposed interface

| User action | Automation to connect | Review required |
|---|---|---|
| Generate candidate segmentation | Select source and nasal region; create airway and bone proposals at the source geometry | Source units, region coverage, leakage and missing anatomy |
| Review target region | Show candidate masks with SPF and contact annotations; enable painting and seeded correction | SPF location and rim, exposed contact surface and relevant bony boundaries |
| Calculate and export | Pass accepted geometry into supported measurement/contact engines; retain prediction and corrected result | Matching measurement definitions, output acceptance and failure reasons |

The intended benefit is less tracing and a repeatable review sequence. Measure that benefit with active annotation/correction time and local boundary error in the setup cases. Do not assume a percentage time saving before measuring it.

## Pretrained model option

TotalSegmentator's `head_glands_cavities` task lists right and left nasal cavities, nasopharynx and hard palate among its outputs. A Slicer extension exposes specialized segmentation tasks. That makes it a candidate source of initial anatomical masks without training a model from scratch.

Use a broad model mask to guide local processing, then assess and refine the target boundary against the original CT. The documented labels do not include the SPF or SPG. Model performance and label meaning must be checked on this cohort before using its masks for morphometry. Keep SPF and rim confirmation manual in the initial pilot.

Sources: [TotalSegmentator tasks](https://github.com/wasserth/TotalSegmentator#subtasks), [Slicer extension](https://github.com/lassoan/SlicerTotalSegmentator), [Slicer Grow from seeds](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html#grow-from-seeds).

## Implementation order

First expose the existing tissue proposal and correction steps through one source-associated interface. Compare its correction burden with a TotalSegmentator-initialized workflow on the three setup cases. Freeze the selected method and measurement definitions before the nine evaluation cases. Develop a dedicated SPF localizer only after the reviewed annotations and pilot failure analysis support that next step.

This roadmap does not claim that the integration or a dedicated learned localizer has been built.
