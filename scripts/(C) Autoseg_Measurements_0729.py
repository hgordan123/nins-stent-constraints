import contextlib
import hashlib
import io
import json
import os
import platform
import sys

import numpy as np
import slicer

# (C) Autoseg_Measurements_0729.py
# Claude-authored. Autosegmentation QC arm for the SPG Anatomy Study.
#
# This is an INDEPENDENT SECOND ARM for quality control. The manual arm
# (Landmark_Measurements_0510.py, driven by the measurer with Dr. Iloreta)
# remains the primary dataset and is never altered by this script.
#
# Workflow:
#   Stage 0  'engine'  Load the manual arm's math in a sandbox, hash-pinned.
#   Stage 1  'tissue'  Air / body / bone / metal tissue classes.
#   Stage 2  'frame'   SPG centroid, midline, side, snapped plane offsets.
#   Stage 3  'curves'  Cut contours at prescribed slices into curve nodes.
#   Stage 4  'bones'   Seeded separation of palatine / pterygoid / sphenoid.
#   Stage 5  'qc'      Failure checks and the screenshot capture plan.
#
# Special mode:
#   replay_manual=True  Computes the measurements from the MANUAL node names
#                       instead of the auto ones. This is the V0b parity test:
#                       it must reproduce all 20 printed values of the manual
#                       arm to 2 decimal places. Run it before trusting any
#                       segmentation output.
#
# Non-interactive by design (no input()) so it runs through the Slicer MCP
# execute_python_code tool as well as the Python console.
#
# Usage (Slicer Python console or MCP):
#   exec(open('/abs/path/to/(C) Autoseg_Measurements_0729.py').read())
#   report = run(stages=('engine',))
#
# NOTE: window/level (W350/L50 per protocol) is a DISPLAY setting only.
# slicer.util.arrayFromVolume returns raw stored HU and is unaffected by it.

SCRIPT_VERSION = "0729.1"

# ---- PARAMETERS -------------------------------------------------------------
# Every tunable lives here as an uppercase module-level constant. collect_params
# scans globals() for uppercase names, so a constant physically cannot be added
# without appearing in the emitted provenance record.

# The manual measurement engine this arm must agree with, arithmetically.
ENGINE_FILENAME = "Landmark_Measurements_0510.py"
ENGINE_SHA256_PINNED = (
    "9fe43850229ca99bef5f613803a817bd3b7379fde87201266d14540108be9eb0"
)

# Tissue classes. Bounds are INCLUSIVE on both ends.
#   Lower bound matters: a strict > -1024 silently drops voxels clamped at
#   exactly -1024 (which is what most scanners emit for pure air), punching
#   holes through the air mask. An unbounded lower end admits out-of-FOV
#   reconstruction padding (-2000, -3024) as "air".
#   Upper bound barely matters: across the whole -400..-300 span the extracted
#   contour moves roughly 0.04-0.10 mm, an order of magnitude below this
#   study's resolution. Swept anyway, as a partial-volume detector.
AIR_HU_MIN = -1024
AIR_HU_MAX = -350
AIR_HU_MAX_SWEEP = (-400, -350, -300)

# Unlike air, this one genuinely matters: 300 vs 400 can delete a thin plate
# and change the connected-component topology of the bone mask.
BONE_HU_MIN = 300
BONE_HU_MAX = 3071
BONE_HU_DENSE = 700
BONE_HU_MIN_SWEEP = (250, 300, 400)
BONE_MEDIAN_MM = 0.0

# Metal must be distinguished from dense normal bone, which the first real
# scan showed this threshold was failing to do. Petrous and skull base cortical
# bone reaches about 2000 to 2250 HU and never saturates; metal saturates the
# CT ceiling at 3071 and blooms. Raised from 2000, which was flagging normal
# petrous bone as metal on a head CT and would have done so on essentially
# every subject. METAL_SATURATION_HU plus the saturated fraction is the
# confirmatory signal, since threshold alone cannot separate the two.
METAL_HU_MIN = 2500
METAL_SATURATION_HU = 3070
METAL_MIN_SATURATED_FRACTION = 0.02
METAL_MIN_VOXELS = 20
BODY_HU_MIN = -400
ORBIT_FAT_HU = (-140, -20)

AIR_ISLAND_MIN_VOXELS = 20
BONE_ISLAND_MIN_VOXELS = 100
OUT_OF_FOV_FRAC_FLAG = 0.001
SENSITIVITY_FLAG_MM = 0.3

# Curve geometry. Linear, not spline: splines overshoot on residual
# marching-cubes stair-step, and overshoot on a morphometry contour is an
# invented measurement.
CURVE_TYPE = "linear"
CURVE_CONTROL_SPACING_MM = 1.5
CURVE_MIN_CONTROL_POINTS = 4
CURVE_MAX_CONTROL_POINTS = 60
CURVE_POINTS_PER_SEGMENT = 10
PLANE_EPS_MM = 0.02

# Loop selection and trimming.
LOOP_MIN_PERIMETER_MM = 4.0
LOOP_ANCHOR_MAX_DIST_MM = 3.0
LOOP_MARGIN_MIN = 1.5
TRIM_NORMAL_MIN = 0.2
CHEEK_ANGULAR_HALFWIDTH_DEG = 45.0
HULL_TOL_MM = 2.0
SEPTUM_BAND_MM = 4.0
MT_HOLE_MIN_AREA_MM2 = 4.0

# Surface smoothing. Applied to polydata only, never the labelmap: labelmap
# smoothing above ~0.1 severs thin turbinate walls.
SURFACE_SMOOTH_PASSBAND = 0.1
SURFACE_SMOOTH_ITERATIONS = 20

# QC thresholds.
STABILITY_DZ_MM = 1.0
STABILITY_MAX_MM = 1.5
SPG_RESIDUAL_FLAG_MM = 2.0
MIDLINE_DISAGREE_MM = 3.0
LEAK_SUP_MM = 35.0
COHORT_MAD_FLAG = 2.5
QC_IMAGE_SIZE = (1200, 1200)

# Node namespace. Manual node names never start with this.
AUTO_PREFIX = "AUTO_"
AUTO_FOLDER_NAME = "AUTO ARM"
RERUN_MODE = "replace"  # replace | version | abort

# Bone seed geometry, in mm, expressed in the SPG frame.
PTERYGOID_SEED_R_MM = 3.0
PTERYGOID_EXTEND_MM = 10.0
SPHENOID_SHELL_MM = 1.5
SPHENOID_SUP_MM = 8.0

VALID_STAGES = ("engine", "tissue", "frame", "curves", "bones", "qc")

# ---- 0510 RESULT KEYS -------------------------------------------------------
# Verbatim key strings from Landmark_Measurements_0510.py. These are the join
# keys against the study spreadsheets, so they are reproduced character for
# character, commas and all. CURVE_DEPENDENT marks the 9 keys this QC arm can
# actually speak to; the other 11 derive from landmarks and lines alone and are
# reported as "not assessable by this arm", never as agreement.

K_SPF_DIAM_AX = "SPF DIAMETER - AXIAL (mm)"
K_SPF_DIAM_COR = "SPF DIAMETER - CORONAL (mm)"
K_ANS_SPF_SAG = "ANS to SPF - SAGITTAL ONLY (mm)"
K_ANS_SPF_SUM = "ANS to SPF (SUMMED) (mm)"
K_SPG_CHEEK_AX = "SPG to CHEEK - AXIAL (mm)"
K_SPG_CHEEK_COR = "SPG to CHEEK - CORONAL (mm)"
K_SPF_FLOOR_COR = "SPF to NASAL FLOOR - CORONAL (mm)"
K_SPG_CAVITY_AX = "SPG to NASAL CAVITY - AXIAL (mm)"
K_SEPTUM_MT_AX = "NASAL SEPTUM to MIDDLE MEDIAL TURB - AXIAL (mm)"
K_SPF_MT_COR = "SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL (mm)"
K_IT_FLOOR = "INFERIOR TURB, SUPERIOR to NASAL FLOOR (mm)"
K_MT_FLOOR_COR = "MIDDLE TURB, INF SURFACE to NASAL FLOOR - CORONAL (mm)"
K_FR_SPF_SI = "INF FORAMEN ROTUNDUM to SPF - CORONAL, Sup/Inf (mm)"
K_FR_SPF_SUM = "INF FORAMEN ROTUNDUM to SPF - CORONAL, Summed (mm)"
K_OF_SPF_SI = "ORBITAL FLOOR @ PPF to SPF - CORONAL, Sup/Inf (mm)"
K_OF_SPF_SUM = "ORBITAL FLOOR @ PPF to SPF - CORONAL, Summed (mm)"
K_IOF_SPF_SI = "IOF MEDIAL to SPF - CORONAL, Sup/Inf (mm)"
K_IOF_SPF_SUM = "IOF MEDIAL to SPF - CORONAL, Summed (mm)"
K_LAT_WALL_COR = "LATERAL NASAL WALL THICKNESS @ SPF - CORONAL (mm)"
K_HARD_PALATE_SAG = "MIDLINE HARD PALATE LENGTH - SAGITTAL (mm)"

RESULT_KEY_ORDER = (
    K_SPF_DIAM_AX, K_SPF_DIAM_COR, K_ANS_SPF_SAG, K_ANS_SPF_SUM,
    K_SPG_CHEEK_AX, K_SPG_CHEEK_COR, K_SPF_FLOOR_COR, K_SPG_CAVITY_AX,
    K_SEPTUM_MT_AX, K_SPF_MT_COR, K_IT_FLOOR, K_MT_FLOOR_COR,
    K_FR_SPF_SI, K_FR_SPF_SUM, K_OF_SPF_SI, K_OF_SPF_SUM,
    K_IOF_SPF_SI, K_IOF_SPF_SUM, K_LAT_WALL_COR, K_HARD_PALATE_SAG,
)

CURVE_DEPENDENT_KEYS = frozenset({
    K_SPG_CHEEK_AX, K_SPG_CHEEK_COR, K_SPG_CAVITY_AX, K_SEPTUM_MT_AX,
    K_SPF_MT_COR, K_IT_FLOOR, K_MT_FLOOR_COR, K_OF_SPF_SI, K_OF_SPF_SUM,
})

# Stable snake_case slug per verbatim key, for CSV columns and filenames.
RESULT_KEY_SLUGS = {
    K_SPF_DIAM_AX: "spf_diam_ax",
    K_SPF_DIAM_COR: "spf_diam_cor",
    K_ANS_SPF_SAG: "ans_to_spf_sag",
    K_ANS_SPF_SUM: "ans_to_spf_summed",
    K_SPG_CHEEK_AX: "spg_to_cheek_ax",
    K_SPG_CHEEK_COR: "spg_to_cheek_cor",
    K_SPF_FLOOR_COR: "spf_to_nasal_floor_cor",
    K_SPG_CAVITY_AX: "spg_to_nasal_cavity_ax",
    K_SEPTUM_MT_AX: "septum_to_mt_ax",
    K_SPF_MT_COR: "spf_inf_to_mt_cor",
    K_IT_FLOOR: "it_sup_to_nasal_floor",
    K_MT_FLOOR_COR: "mt_inf_to_nasal_floor_cor",
    K_FR_SPF_SI: "fr_to_spf_si",
    K_FR_SPF_SUM: "fr_to_spf_summed",
    K_OF_SPF_SI: "orbital_floor_to_spf_si",
    K_OF_SPF_SUM: "orbital_floor_to_spf_summed",
    K_IOF_SPF_SI: "iof_to_spf_si",
    K_IOF_SPF_SUM: "iof_to_spf_summed",
    K_LAT_WALL_COR: "lat_nasal_wall_cor",
    K_HARD_PALATE_SAG: "hard_palate_length_sag",
}

# Node roles the measurements consume. Values are node NAMES, supplied by the
# caller. In replay_manual mode these are the manual names; otherwise they
# default to the AUTO_ names this script generates.
NODE_ROLES = (
    "vidian_ax", "ppf_ax", "gp_sag",          # the 3 SPG centroid lines
    "landmark_list",                          # fiducial point list
    "spf_diam_ax", "spf_diam_cor",            # lines
    "lat_nasal_wall_cor", "hard_palate_sag",  # lines
    "cheek_ax", "cheek_cor",                  # curves
    "nasal_cavity_ax", "septum_ax", "mt_medial_ax",
    "mt_medial_cor", "it_superior_cor", "mt_inferior_cor",
    "orbital_floor_cor",
)

# Landmark roles, resolved to 0-based indices into the fiducial point list.
LANDMARK_ROLES = ("ANS", "SPF_inf", "nasal_floor", "FR_inf", "IOF_medial")

ENGINE_CALLABLES = (
    "get_line_points", "get_curve", "in_plane_distance",
    "find_closest_point", "closest_distance_between_curves",
    "best_intersection_point",
)

SKIPPED = "Skipped"


############ PROVENANCE ############################


def collect_params(overrides=None):
    """Snapshot every uppercase module constant, with optional overrides.

    Scans globals() rather than a hand-maintained list, so a new constant
    cannot be added without appearing in the emitted record. Overrides are
    validated against the known set (a typo'd name is an error, never a silent
    no-op) and applied to a copy, so module globals are never mutated and two
    sweep runs in one session cannot contaminate each other.
    """
    params = {
        k: v for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
        and isinstance(v, (int, float, str, bool, tuple, list, dict))
    }
    if overrides:
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(
                f"unknown parameter override(s): {sorted(unknown)}"
            )
        params = dict(params, **overrides)
    return params


def config_hash(params):
    """Short stable hash of a parameter set."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def environment_record():
    """Versions that could change a numerical result.

    Records the Rosetta detail deliberately: this Slicer is an x86_64 build on
    arm64, and numerical libraries can differ between the two.
    """
    try:
        import vtk
        vtk_version = vtk.vtkVersion.GetVTKVersion()
    except Exception:
        vtk_version = "unavailable"
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = "unavailable"
    try:
        import SimpleITK
        sitk_version = SimpleITK.__version__
    except Exception:
        sitk_version = "unavailable"
    return {
        "slicer_version": slicer.app.applicationVersion,
        "slicer_revision": slicer.app.revision,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy_version,
        "vtk": vtk_version,
        "SimpleITK": sitk_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def volume_record(volume_node):
    """Everything needed to prove which voxels a result came from."""
    if volume_node is None:
        return None
    import vtk
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    bounds = [0.0] * 6
    volume_node.GetRASBounds(bounds)
    record = {
        "name": volume_node.GetName(),
        "id": volume_node.GetID(),
        "spacing": list(volume_node.GetSpacing()),
        "dimensions": list(volume_node.GetImageData().GetDimensions())
        if volume_node.GetImageData() else None,
        "origin": list(volume_node.GetOrigin()),
        "ijk_to_ras": [
            [ijk_to_ras.GetElement(r, c) for c in range(4)] for r in range(4)
        ],
        "ras_bounds": bounds,
        "dicom_instance_uids": volume_node.GetAttribute("DICOM.instanceUIDs"),
    }
    array = slicer.util.arrayFromVolume(volume_node)
    if array is not None:
        record["hu_min"] = float(array.min())
        record["hu_max"] = float(array.max())
    # A resampled derivative shifts HU slightly and non-uniformly between
    # subjects, moving every threshold-derived contour by a subject-dependent
    # amount. Detectable via a spacing mismatch against the parent volume.
    record["spacing_is_isotropic"] = bool(
        len(set(round(s, 4) for s in volume_node.GetSpacing())) == 1
    )
    return record


############ SANDBOXED ENGINE LOADER ############################


class EngineTripwire(Exception):
    """Raised when the manual engine reaches its first interactive prompt.

    Deliberately a RAISING stub rather than one returning ''. If a future edit
    to the manual engine changes its control flow, a returning stub would let
    execution walk on into node creation; this bails loudly instead.
    """


def engine_path():
    """Absolute path to the manual measurement engine, alongside this file."""
    here = os.path.dirname(os.path.abspath(_this_file()))
    return os.path.join(here, ENGINE_FILENAME)


def _this_file():
    """Path of this script, whether exec'd or imported.

    exec(open(path).read()) leaves __file__ unset, so fall back to the
    documented project location.
    """
    if "__file__" in globals():
        return globals()["__file__"]
    return os.path.join(
        os.path.expanduser("~"),
        "Dropbox/My Mac (Alfreds-MacBook-Pro-2.local)/Documents/Obsidian Vault",
        "03 Projects/NINS/SPG Anatomy Study/06 System",
        "(C) Autoseg_Measurements_0729.py",
    )


def load_engine(path=None, allow_hash_mismatch=False):
    """Load the manual engine's math into a private namespace. Scene-safe.

    Why this works: in Landmark_Measurements_0510.py lines 15-192 are pure
    function definitions, line 198 sets results = {}, the first module-level
    interactive call is line 236, and the first module-level node creation is
    line 267. Every input() at lines 107-179 sits inside a function body and so
    does not fire at load time. A load that trips the tripwire at line 236
    therefore leaves the scene completely untouched while binding every math
    function.

    Returns (namespace, record). Raises on any integrity failure.
    """
    path = path or engine_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"manual engine not found at {path}")

    actual = _sha256_file(path)
    if actual != ENGINE_SHA256_PINNED:
        message = (
            f"manual engine hash mismatch\n"
            f"  path:     {path}\n"
            f"  expected: {ENGINE_SHA256_PINNED}\n"
            f"  actual:   {actual}\n"
            f"{ENGINE_FILENAME} has changed. Re-run the V0b parity test, "
            f"confirm all 20 values still reproduce to 2 dp, then update "
            f"ENGINE_SHA256_PINNED deliberately. Subjects processed under "
            f"different hashes belong to different instruments and cannot be "
            f"pooled."
        )
        if not allow_hash_mismatch:
            raise RuntimeError(message)
        print(f"  WARNING: {message}")

    with open(path, "r") as fh:
        source = fh.read()

    fired = {"tripwire": False}

    def _tripwire(*_args, **_kwargs):
        fired["tripwire"] = True
        raise EngineTripwire(
            "manual engine reached an interactive prompt, as expected"
        )

    namespace = {"input": _tripwire, "__name__": "_spg_manual_engine"}

    before = _node_id_set()
    banner = io.StringIO()
    try:
        with contextlib.redirect_stdout(banner):
            exec(compile(source, path, "exec"), namespace)
    except EngineTripwire:
        pass
    except SystemExit:
        pass
    after = _node_id_set()

    # Post-load assertions. Each catches an edit that changes behaviour
    # without anyone thinking to check.
    if not fired["tripwire"]:
        raise RuntimeError(
            "manual engine completed without reaching a prompt. Its control "
            "flow has changed and the sandbox guarantee no longer holds."
        )
    if namespace.get("results") != {}:
        raise RuntimeError(
            f"manual engine populated results during load: "
            f"{namespace.get('results')!r}. Expected an empty dict."
        )
    if before != after:
        added = sorted(after - before)
        raise RuntimeError(
            f"manual engine mutated the scene during load, adding {added}. "
            f"The sandbox guarantee is broken; do not proceed."
        )
    missing = [name for name in ENGINE_CALLABLES
               if not callable(namespace.get(name))]
    if missing:
        raise RuntimeError(f"manual engine did not bind: {missing}")

    record = {
        "path": path,
        "sha256": actual,
        "sha256_matches_pin": actual == ENGINE_SHA256_PINNED,
        "tripwire_fired": True,
        "results_empty": True,
        "scene_unchanged": True,
        "callables_bound": list(ENGINE_CALLABLES),
        "banner_suppressed_chars": len(banner.getvalue()),
    }
    return namespace, record


############ LITERATURE PRIORS ############################
# Published skull base foramen morphometry, used for ADVISORY FLAGGING ONLY.
#
# These never reject a measurement, never constrain the manual arm, and never
# alter a recorded value. The reason is that this study exists to characterise
# SPG-region anatomy for device design: if published windows were used to
# reject outliers, the study would partly be measuring its own priors and could
# mask exactly the variation the NINS device has to accommodate.
#
# A subject outside a band is a subject to LOOK AT, not one to exclude.

PRIORS_FILENAME = "(C) literature_priors.json"


def load_priors(path=None):
    """Load the literature priors table. Absence is not an error."""
    path = path or os.path.join(
        os.path.dirname(os.path.abspath(_this_file())), PRIORS_FILENAME)
    if not os.path.exists(path):
        return {"priors": [], "sources": {}, "loaded": False, "path": path}
    with open(path) as handle:
        data = json.load(handle)
    data["loaded"] = True
    data["path"] = path
    data["sha256"] = _sha256_file(path)
    data["by_id"] = {p["id"]: p for p in data.get("priors", [])}
    return data


def prior_flag(priors, prior_id, value_mm):
    """Compare one value to a published band. Returns None when in band.

    Deliberately returns an advisory dict, never a rejection: the caller emits
    it as a 'warn' at most.
    """
    entry = (priors.get("by_id") or {}).get(prior_id)
    if entry is None or value_mm is None:
        return None
    band = entry.get("flag_band_mm")
    if not band:
        return None
    low, high = band
    if low <= value_mm <= high:
        return None
    citations = [priors["sources"][s]["citation"]
                 for s in entry.get("sources", [])
                 if s in priors.get("sources", {})]
    return {
        "prior_id": prior_id,
        "value_mm": float(value_mm),
        "band_mm": band,
        "published_mean_mm": entry.get("mean_mm"),
        "published_sd_mm": entry.get("sd_mm"),
        "confidence": entry.get("confidence"),
        "definition_note": entry.get("definition_note"),
        "sources": citations,
        "advisory": "outside the published band; inspect, do not exclude",
    }


def check_spg_against_spf(spg, landmarks, priors, check):
    """Sanity-check the computed SPG centroid against the SPF landmark.

    This catches something the least-squares residual cannot. Three badly
    placed lines can intersect crisply (tiny residual) at completely the wrong
    point. Published cadaveric work converges on the SPG sitting about 4 to
    6.3 mm from the SPF, lateral and slightly posterior or inferior, so a
    centroid 20 mm away means the lines were misplaced rather than that the
    anatomy is unusual.
    """
    spf = (landmarks or {}).get("SPF_inf")
    if spg is None or spf is None:
        return None
    distance = float(np.linalg.norm(np.asarray(spg) - np.asarray(spf)))
    record = {"spg_to_spf_mm": distance}
    flag = prior_flag(priors, "spg_to_spf_distance", distance)
    if flag:
        record["flag"] = flag
        check("spg_far_from_spf", "warn",
              f"computed SPG centroid sits {distance:.1f} mm from the SPF "
              f"landmark, outside the published {flag['band_mm']} mm band "
              f"(cadaveric studies converge near "
              f"{flag['published_mean_mm']} mm). The three input lines may be "
              f"misplaced; a small least-squares residual does not rule this "
              f"out, because three wrong lines can still intersect cleanly. "
              f"Advisory only, do not exclude the subject on this basis.")
    return record


def check_measurements_against_priors(results, priors, check):
    """Advisory comparison of the 20 measurements to published bands."""
    flags = {}
    for entry in priors.get("priors", []):
        for key in entry.get("maps_to_measurement_keys", []) or []:
            raw = (results or {}).get(key)
            if not raw or raw == SKIPPED:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            flag = prior_flag(priors, entry["id"], value)
            if flag:
                flags[key] = flag
                check(f"prior_outlier_{RESULT_KEY_SLUGS.get(key, key)}", "warn",
                      f"{key} = {value:.2f} mm falls outside the published "
                      f"band {flag['band_mm']} mm "
                      f"(confidence: {flag['confidence']}). "
                      f"{flag['definition_note'][:150]} "
                      f"Advisory only.")
    return flags


############ SCENE AND NAMESPACE GUARDS ############################


def _node_id_set():
    nodes = slicer.mrmlScene.GetNodes()
    return {nodes.GetItemAsObject(i).GetID() for i in range(nodes.GetNumberOfItems())}


def _all_nodes():
    nodes = slicer.mrmlScene.GetNodes()
    return [nodes.GetItemAsObject(i) for i in range(nodes.GetNumberOfItems())]


DATA_NODE_TOKENS = ("Markups", "ScalarVolume", "LabelMapVolume",
                    "Segmentation", "Model", "LinearTransform",
                    "BSplineTransform", "GridTransform")


def _is_data_node(node):
    """True for nodes that carry study data, as opposed to view or display
    infrastructure. Used by the duplicate-name guard, which must not trip on
    Slicer's own repeated view node names."""
    if node is None:
        return False
    class_name = node.GetClassName() or ""
    if any(token in class_name
           for token in ("Display", "Storage", "SubjectHierarchy",
                         "SegmentEditor", "SliceComposite")):
        return False
    return any(token in class_name for token in DATA_NODE_TOKENS)


def _preservable_node_ids():
    """IDs of the data-bearing, non-auto nodes that must never be lost.

    Display, storage and subject-hierarchy nodes are excluded because they are
    derived satellites: removing an auto markups node cascades to its own
    display and storage nodes, and counting those as losses would make the
    preservation check fire on every legitimate re-run. What actually matters
    is that no manual volume, markup, segmentation or transform disappears.
    """
    ids = set()
    for node in _all_nodes():
        if is_auto_node(node):
            continue
        class_name = node.GetClassName() or ""
        if any(token in class_name
               for token in ("Display", "Storage", "SubjectHierarchy",
                             "SegmentEditor")):
            continue
        ids.add(node.GetID())
    return ids


def is_auto_node(node):
    """True only for nodes this script created, by ATTRIBUTE not name.

    Deletion and mutation are scoped by attribute deliberately: a name-prefix
    check would happily delete a human node that happens to be named AUTO_*.
    """
    return node is not None and node.GetAttribute("SPGAutoSeg.source") == "auto"


def assert_namespace_clean():
    """The AUTO_ namespace is reserved. A human node in it is an error.

    Also refuses to run when two nodes share a name anywhere in the scene,
    because ambiguous getNode results would corrupt both arms. That check
    incidentally protects the manual arm and is worth having regardless.
    """
    problems = []
    seen = {}
    for node in _all_nodes():
        name = node.GetName()
        if not name:
            continue
        # Only DATA nodes participate in the duplicate check. Every real
        # Slicer scene legitimately has repeated infrastructure names: the
        # slice node and slice composite node are both called "Red", and
        # display and storage nodes repeat freely. Checking those would make
        # the guard refuse to run on any genuine scene, which is exactly what
        # it did the first time it met one.
        # Auto nodes are excluded: they are this script's own disposable
        # output and clear_previous_auto_nodes removes them moments later, so
        # counting them here would let a previous run block the next one.
        if _is_data_node(node) and not is_auto_node(node):
            seen.setdefault(name, []).append(node.GetID())
        if name.startswith(AUTO_PREFIX) and not is_auto_node(node):
            problems.append(
                f"node {name!r} ({node.GetID()}) occupies the reserved "
                f"{AUTO_PREFIX} namespace but carries no SPGAutoSeg.source "
                f"attribute"
            )
    duplicates = {n: ids for n, ids in seen.items() if len(ids) > 1}
    if duplicates:
        problems.append(f"duplicate node names in scene: {duplicates}")
    if problems:
        raise RuntimeError(
            "namespace check failed, refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return {"duplicates": {}, "reserved_namespace_clean": True}


def get_node_strict(name, role=""):
    """Exact-name node lookup.

    Never slicer.util.getNode: it matches by pattern and will happily return
    AUTO_MT_medial_ax_v2 when asked for AUTO_MT_medial_ax.
    """
    if not name:
        return None
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    if node is None:
        return None
    if node.GetName() != name:
        raise RuntimeError(
            f"node lookup for {role or name!r} returned {node.GetName()!r}, "
            f"expected exactly {name!r}"
        )
    return node


def assert_no_transform(node, role=""):
    """The manual engine mixes World and node-local coordinate accessors.

    get_line_points (lines 19-20) and get_curve (line 27) use the World
    variants; landmark loading (lines 413-449) uses GetNthControlPointPosition,
    which is node-local. In a transform-free scene these agree. A parent
    transform on either arm breaks the comparison silently, so it is asserted
    rather than hoped for.
    """
    if node is None:
        return
    if hasattr(node, "GetParentTransformNode") and node.GetParentTransformNode():
        raise RuntimeError(
            f"node {node.GetName()!r} ({role}) has a parent transform. The "
            f"manual engine mixes World and node-local accessors, so a "
            f"transform makes the two arms silently incomparable. Harden the "
            f"transform before running."
        )


def tag_auto_node(node, run_id, cfg_hash, requires_confirmation=False,
                   extra=None):
    """Stamp provenance onto a created node and lock it."""
    node.SetAttribute("SPGAutoSeg.source", "auto")
    node.SetAttribute("SPGAutoSeg.runId", run_id)
    node.SetAttribute("SPGAutoSeg.configHash", cfg_hash)
    node.SetAttribute("SPGAutoSeg.version", SCRIPT_VERSION)
    node.SetAttribute("SPGAutoSeg.qcStatus", "pending")
    node.SetAttribute(
        "SPGAutoSeg.requiresConfirmation",
        "true" if requires_confirmation else "false",
    )
    for key, value in (extra or {}).items():
        node.SetAttribute(f"SPGAutoSeg.{key}", str(value))
    if hasattr(node, "SetLocked"):
        node.SetLocked(True)
    return node


def adopt_engine_side_effects(before_ids, run_id, cfg_hash, tag):
    """Capture and rename nodes the manual engine created as a side effect.

    find_closest_point (0510 lines 53-56) and four blocks in STEP 4 drop
    visualisation fiducials into the scene. Several use FIXED name strings
    ('Inferior Turbinate - most superior point' and friends) which WILL collide
    with a manual run's leftovers. Snapshot, diff, rename into the AUTO_
    namespace, tag, lock.
    """
    adopted = []
    for node in _all_nodes():
        if node.GetID() in before_ids:
            continue
        original = node.GetName() or node.GetID()
        if not original.startswith(AUTO_PREFIX):
            node.SetName(f"{AUTO_PREFIX}{tag}_{original}")
        tag_auto_node(node, run_id, cfg_hash,
                      extra={"derivedFrom": "engine_side_effect",
                             "originalName": original})
        adopted.append({"id": node.GetID(), "name": node.GetName(),
                        "original_name": original})
    return adopted


def clear_previous_auto_nodes(mode):
    """Apply RERUN_MODE. Scoped by attribute, never by name."""
    existing = [n for n in _all_nodes() if is_auto_node(n)]
    if not existing:
        return {"mode": mode, "removed": [], "preexisting": 0}
    if mode == "abort":
        raise RuntimeError(
            f"{len(existing)} prior auto node(s) present and RERUN_MODE is "
            f"'abort'. Refusing to run so an accidental regeneration cannot "
            f"become a data-integrity event."
        )
    if mode == "version":
        return {"mode": mode, "removed": [], "preexisting": len(existing)}
    removed = []
    for node in existing:
        if not is_auto_node(node):  # belt and braces before any removal
            continue
        removed.append({"id": node.GetID(), "name": node.GetName()})
        slicer.mrmlScene.RemoveNode(node)
    return {"mode": mode, "removed": removed, "preexisting": len(existing)}


############ MEASUREMENTS ############################


def resolve_nodes(node_names, strict_roles=()):
    """Look up every supplied node name and assert it is transform-free."""
    resolved, missing = {}, []
    for role, name in (node_names or {}).items():
        if not name:
            continue
        node = get_node_strict(name, role)
        if node is None:
            missing.append({"role": role, "name": name})
            continue
        assert_no_transform(node, role)
        resolved[role] = node
    for role in strict_roles:
        if role not in resolved:
            raise RuntimeError(f"required node role {role!r} could not be resolved")
    return resolved, missing


def compute_spg_centroid(engine, nodes, node_names):
    """SPG centroid as the least-squares intersection of the 3 manual lines.

    Cannot be auto-derived: it comes from hand-placed lines, and 4 of the 9
    curve-dependent measurements anchor on it. That is why this arm validates
    curve TRACING, not measurement independence.
    """
    needed = ("vidian_ax", "ppf_ax", "gp_sag")
    have = [r for r in needed if r in nodes]
    if len(have) != 3:
        return None, {
            "status": "unavailable",
            "missing": [r for r in needed if r not in nodes],
        }
    pa, pb = [], []
    endpoints = {}
    for role in needed:
        p0, p1 = engine["get_line_points"](nodes[role])
        pa.append(p0)
        pb.append(p1)
        endpoints[role] = {"name": node_names.get(role),
                           "p0": [float(v) for v in p0],
                           "p1": [float(v) for v in p1]}
    spg, residuals = engine["best_intersection_point"](np.array(pa), np.array(pb))
    residual_scalar = (
        float(np.sqrt(np.sum(residuals)))
        if getattr(residuals, "size", 0) else None
    )
    record = {
        "status": "ok",
        "spg_ras": [float(v) for v in spg],
        "residual_raw": (residuals.tolist()
                         if hasattr(residuals, "tolist") else residuals),
        "residual_mm": residual_scalar,
        "lines": endpoints,
    }
    # Three lines that do not nearly intersect mean inconsistent line
    # placement, and every downstream plane offset inherits that error.
    if residual_scalar is not None and residual_scalar > SPG_RESIDUAL_FLAG_MM:
        record["flag"] = (
            f"SPG line intersection residual {residual_scalar:.2f} mm exceeds "
            f"{SPG_RESIDUAL_FLAG_MM} mm"
        )
    return np.array(spg), record


def load_landmarks(nodes, landmark_indices):
    """Read the named landmarks, in either of the two layouts measurers use.

    Layout A, one node per landmark: the scene holds separate single-point
    fiducial nodes ('ANS R', 'SPF_inf_border R', ...). This is what the
    measurer actually produces in practice, so it is tried first.

    Layout B, one indexed point list: a single fiducial node with the
    landmarks at known indices, which is what the manual engine's STEP 3
    prompts for.

    Either way, positions come from GetNthControlPointPosition (node-local) to
    match the manual engine exactly. assert_no_transform has already
    guaranteed local equals World.
    """
    # Layout A: a node supplied per landmark role.
    per_role = {role: nodes[role] for role in LANDMARK_ROLES if role in nodes}
    if per_role:
        coords, record = {}, {"status": "ok", "layout": "one_node_per_landmark",
                              "resolved": {}}
        for role, node in per_role.items():
            if node.GetNumberOfControlPoints() < 1:
                continue
            coords[role] = np.array(node.GetNthControlPointPosition(0))
            record["resolved"][role] = {
                "node": node.GetName(),
                "position": [float(v) for v in coords[role]],
            }
        missing = [r for r in LANDMARK_ROLES if r not in coords]
        if missing:
            record["missing"] = missing
        return coords, record

    # Layout B: an indexed point list.
    node = nodes.get("landmark_list")
    if node is None:
        return {}, {"status": "unavailable",
                    "reason": "supply either one node per landmark role, or a "
                              "landmark_list node plus landmark_indices"}
    count = node.GetNumberOfControlPoints()
    coords, record = {}, {"status": "ok", "layout": "indexed_point_list",
                          "n_control_points": count,
                          "resolved": {}, "labels": []}
    for i in range(count):
        label = node.GetNthControlPointLabel(i)
        desc = node.GetNthControlPointDescription(i)
        record["labels"].append({"index": i, "label": label, "description": desc})
    for role in LANDMARK_ROLES:
        index = (landmark_indices or {}).get(role)
        if index is None:
            continue
        if not (0 <= int(index) < count):
            raise RuntimeError(
                f"landmark index {index} for {role!r} is out of range "
                f"(0-{count - 1})"
            )
        coords[role] = np.array(node.GetNthControlPointPosition(int(index)))
        record["resolved"][role] = {
            "index": int(index),
            "label": node.GetNthControlPointLabel(int(index)),
            "description": node.GetNthControlPointDescription(int(index)),
            "position": [float(v) for v in coords[role]],
        }
    return coords, record


def _fmt(value):
    """Match the manual engine's f'{x:.2f}' output exactly.

    Both arms round to 2 dp so the comparison introduces no rounding
    asymmetry. Consumers must treat SKIPPED as missing, never as 0.
    """
    return f"{value:.2f}"


def compute_measurements(engine, nodes, node_names, spg, landmarks,
                         run_id, cfg_hash, tag="replay"):
    """Compute all 20 measurements using the manual engine's own functions.

    Arithmetic is the manual engine's, unmodified: any auto-vs-manual
    difference is therefore attributable to the curves alone. The manual arm's
    quirks are reproduced deliberately rather than corrected, including
    find_closest_point's 3D argmin over the whole curve with a 2D in-plane
    report, because those are properties of the instrument being compared
    against, not bugs.
    """
    results = {key: SKIPPED for key in RESULT_KEY_ORDER}
    detail, side_effects = {}, []
    before_ids = _node_id_set()

    def line_length(role, key):
        node = nodes.get(role)
        if node is None:
            return
        p0, p1 = engine["get_line_points"](node)
        value = float(np.linalg.norm(p1 - p0))
        results[key] = _fmt(value)
        detail[key] = {"role": role, "node": node_names.get(role),
                       "value_mm": value}

    ans = landmarks.get("ANS")
    spf_inf = landmarks.get("SPF_inf")
    nasal_floor = landmarks.get("nasal_floor")
    fr = landmarks.get("FR_inf")
    iof = landmarks.get("IOF_medial")

    # --- lines ---
    line_length("spf_diam_ax", K_SPF_DIAM_AX)
    line_length("spf_diam_cor", K_SPF_DIAM_COR)
    line_length("lat_nasal_wall_cor", K_LAT_WALL_COR)
    line_length("hard_palate_sag", K_HARD_PALATE_SAG)

    # --- landmark pairs ---
    if ans is not None and spf_inf is not None:
        value = float(engine["in_plane_distance"](ans, spf_inf, axis=0))
        results[K_ANS_SPF_SAG] = _fmt(value)
        detail[K_ANS_SPF_SAG] = {"value_mm": value}
        value = float(np.linalg.norm(np.array(spf_inf) - np.array(ans)))
        results[K_ANS_SPF_SUM] = _fmt(value)
        detail[K_ANS_SPF_SUM] = {"value_mm": value}

    if spf_inf is not None and nasal_floor is not None:
        value = float(abs(spf_inf[2] - nasal_floor[2]))
        results[K_SPF_FLOOR_COR] = _fmt(value)
        detail[K_SPF_FLOOR_COR] = {"value_mm": value}

    if spf_inf is not None and fr is not None:
        si = float(abs(spf_inf[2] - fr[2]))
        summed = float(np.linalg.norm(np.array(spf_inf) - np.array(fr)))
        results[K_FR_SPF_SI] = _fmt(si)
        results[K_FR_SPF_SUM] = _fmt(summed)
        detail[K_FR_SPF_SI] = {"value_mm": si}
        detail[K_FR_SPF_SUM] = {"value_mm": summed}

    if spf_inf is not None and iof is not None:
        si = float(abs(iof[2] - spf_inf[2]))
        summed = float(np.linalg.norm(np.array(iof) - np.array(spf_inf)))
        results[K_IOF_SPF_SI] = _fmt(si)
        results[K_IOF_SPF_SUM] = _fmt(summed)
        detail[K_IOF_SPF_SI] = {"value_mm": si}
        detail[K_IOF_SPF_SUM] = {"value_mm": summed}

    # --- curve-dependent: the 9 this arm can actually assess ---
    if spg is not None and "cheek_ax" in nodes:
        _, _, dist = engine["find_closest_point"](
            spg, node_names["cheek_ax"], axis_xyz=2)
        results[K_SPG_CHEEK_AX] = _fmt(float(dist))
        detail[K_SPG_CHEEK_AX] = {"node": node_names["cheek_ax"],
                                  "value_mm": float(dist)}

    if spg is not None and "cheek_cor" in nodes:
        _, _, dist = engine["find_closest_point"](
            spg, node_names["cheek_cor"], axis_xyz=1)
        results[K_SPG_CHEEK_COR] = _fmt(float(dist))
        detail[K_SPG_CHEEK_COR] = {"node": node_names["cheek_cor"],
                                   "value_mm": float(dist)}

    if spg is not None and "nasal_cavity_ax" in nodes:
        _, _, dist = engine["find_closest_point"](
            spg, node_names["nasal_cavity_ax"], axis_xyz=2)
        results[K_SPG_CAVITY_AX] = _fmt(float(dist))
        detail[K_SPG_CAVITY_AX] = {"node": node_names["nasal_cavity_ax"],
                                   "value_mm": float(dist)}

    if "septum_ax" in nodes and "mt_medial_ax" in nodes:
        pt_septum, pt_mt, min_dist = engine["closest_distance_between_curves"](
            node_names["septum_ax"], node_names["mt_medial_ax"])
        results[K_SEPTUM_MT_AX] = _fmt(float(min_dist))
        detail[K_SEPTUM_MT_AX] = {
            "septum_node": node_names["septum_ax"],
            "mt_node": node_names["mt_medial_ax"],
            "closest_on_septum": [float(v) for v in pt_septum],
            "closest_on_mt": [float(v) for v in pt_mt],
            "value_mm": float(min_dist),
        }

    if spf_inf is not None and "mt_medial_cor" in nodes:
        points = engine["get_curve"](node_names["mt_medial_cor"])
        matched = points[np.argmin(np.abs(points[:, 2] - spf_inf[2]))]
        value = float(abs(spf_inf[0] - matched[0]))
        results[K_SPF_MT_COR] = _fmt(value)
        detail[K_SPF_MT_COR] = {"node": node_names["mt_medial_cor"],
                                "matched_point": [float(v) for v in matched],
                                "value_mm": value}

    if nasal_floor is not None and "it_superior_cor" in nodes:
        points = engine["get_curve"](node_names["it_superior_cor"])
        most_superior = points[np.argmax(points[:, 2])]
        value = float(abs(most_superior[2] - nasal_floor[2]))
        results[K_IT_FLOOR] = _fmt(value)
        detail[K_IT_FLOOR] = {"node": node_names["it_superior_cor"],
                              "extreme_point": [float(v) for v in most_superior],
                              "value_mm": value}

    if nasal_floor is not None and "mt_inferior_cor" in nodes:
        points = engine["get_curve"](node_names["mt_inferior_cor"])
        most_inferior = points[np.argmin(points[:, 2])]
        value = float(abs(most_inferior[2] - nasal_floor[2]))
        results[K_MT_FLOOR_COR] = _fmt(value)
        detail[K_MT_FLOOR_COR] = {"node": node_names["mt_inferior_cor"],
                                  "extreme_point": [float(v) for v in most_inferior],
                                  "value_mm": value}

    if spf_inf is not None and "orbital_floor_cor" in nodes:
        points = engine["get_curve"](node_names["orbital_floor_cor"])
        most_inferior = points[np.argmin(points[:, 2])]
        si = float(abs(most_inferior[2] - spf_inf[2]))
        summed = float(np.linalg.norm(most_inferior - np.array(spf_inf)))
        results[K_OF_SPF_SI] = _fmt(si)
        results[K_OF_SPF_SUM] = _fmt(summed)
        detail[K_OF_SPF_SI] = {"node": node_names["orbital_floor_cor"],
                               "extreme_point": [float(v) for v in most_inferior],
                               "value_mm": si}
        detail[K_OF_SPF_SUM] = {"value_mm": summed}

    side_effects = adopt_engine_side_effects(before_ids, run_id, cfg_hash, tag)
    return results, detail, side_effects


############ 1D EDGE REFINEMENT ############################
# The one automation pattern that survived testing on real data.
#
# Seven approaches to automated anatomical IDENTIFICATION were tried and all
# failed: the region is not separable by density or connectivity. What does
# work is REFINEMENT along a direction the operator supplies. One dimension,
# roughly 700 HU of contrast at a bone edge, and an unambiguous crossing.
#
# The division of labour is: the human identifies the structure and gives the
# direction; the algorithm places the edge. That attacks the dominant variance
# source directly, because "exactly where does this bony crest end" is the
# judgement that drifts between repeats, not "which crest is it".

EDGE_HU = 300.0             # soft tissue to cortical bone crossing
EDGE_STEP_MM = 0.1          # profile sampling, well below voxel size
EDGE_SEARCH_MM = 3.0        # how far either side of a seed to look
EDGE_CHORD_MARGIN_MM = 3.0  # overshoot when profiling a chord
RIM_PROMINENCE_HU = 120.0   # a rim must rise this far above the lumen baseline
RIM_SEARCH_MM = 8.0         # how far outward to look for a bounding rim
LUMEN_HU_MAX = 200.0        # above this a sample window is not lumen
FERET_N_ANGLES = 90         # 2 degree steps over a half turn
APERTURE_N_PROFILES = 5     # parallel profiles per chord refinement
APERTURE_SPREAD_MM = 1.2    # half-height of the profile fan


def sample_along_line(array, ras_to_ijk, p0, p1, step_mm=EDGE_STEP_MM):
    """Trilinearly interpolated HU profile between two RAS points.

    Trilinear rather than nearest neighbour: the whole point is sub-voxel
    edge placement, which nearest neighbour quantises away.
    """
    import scipy.ndimage as ndi
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    if length <= 0:
        return np.zeros(1), np.array([np.nan]), p0[None, :]
    n = max(2, int(round(length / float(step_mm))) + 1)
    t = np.linspace(0.0, length, n)
    d = (p1 - p0) / length
    pts = p0[None, :] + t[:, None] * d[None, :]
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    ijk = homo @ np.asarray(ras_to_ijk).T
    coords = np.vstack([ijk[:, 2], ijk[:, 1], ijk[:, 0]])   # (k, j, i)
    hu = ndi.map_coordinates(array.astype(np.float32), coords,
                             order=1, mode="nearest")
    return t, hu, pts


def find_crossings(t, hu, level):
    """Sub-voxel positions where a profile crosses `level`, by linear interp."""
    hu = np.asarray(hu, float)
    sign = np.sign(hu - float(level))
    sign[sign == 0] = 1.0
    idx = np.where(np.diff(sign) != 0)[0]
    out = []
    for i in idx:
        h0, h1 = hu[i], hu[i + 1]
        if h1 == h0:
            continue
        frac = (float(level) - h0) / (h1 - h0)
        out.append({"t": float(t[i] + frac * (t[i + 1] - t[i])),
                    "rising": bool(h1 > h0)})
    return out


def refine_point_to_edge(array, ras_to_ijk, seed, direction,
                         level=EDGE_HU, search_mm=EDGE_SEARCH_MM,
                         step_mm=EDGE_STEP_MM, prefer=None):
    """Snap a seed onto a bone edge along a supplied direction.

    `prefer` disambiguates which edge is wanted:
      "rising"  soft tissue into bone
      "falling" bone out into soft tissue
      None      nearest crossing of either kind

    The preference matters. A seed placed inside a 2.3 mm crest, which is what
    happens in practice on the crista ethmoidalis, can be nearer the far margin
    than the near one, so "nearest" alone is ambiguous exactly where the
    operator most needs help. For a bounded gap use refine_chord instead, which
    finds both margins together and is unambiguous by construction.

    Returns the refined RAS point plus how far it moved, so the shift is
    auditable rather than silent.
    """
    d = np.asarray(direction, float)
    norm = np.linalg.norm(d)
    if norm == 0:
        return None
    d = d / norm
    seed = np.asarray(seed, float)
    p0, p1 = seed - d * search_mm, seed + d * search_mm
    t, hu, _ = sample_along_line(array, ras_to_ijk, p0, p1, step_mm)
    crossings = find_crossings(t, hu, level)
    n_all = len(crossings)
    if prefer == "rising":
        crossings = [c for c in crossings if c["rising"]]
    elif prefer == "falling":
        crossings = [c for c in crossings if not c["rising"]]
    if not crossings:
        return None
    best = min(crossings, key=lambda c: abs(c["t"] - search_mm))
    refined = p0 + d * best["t"]
    return {"refined_ras": refined,
            "shift_mm": float(np.linalg.norm(refined - seed)),
            "rising_into_bone": best["rising"],
            "prefer": prefer,
            "n_crossings_in_window": n_all,
            "ambiguous": bool(prefer is None and n_all > 1),
            "hu_at_seed": float(np.interp(search_mm, t, hu))}


def refine_chord(array, ras_to_ijk, p_ant, p_post, level=EDGE_HU,
                 margin_mm=EDGE_CHORD_MARGIN_MM, step_mm=EDGE_STEP_MM,
                 polarity="gap"):
    """Refine both ends of a chord onto the true bony margins between them.

    polarity="gap"  the span between two bones, e.g. the SPF aperture
    polarity="slab" the span OF a bone, e.g. lateral nasal wall thickness

    This is the SPF case. The operator marks anterior and posterior borders,
    but a bony crest such as the crista ethmoidalis is 2 to 3 mm thick, so a
    seed placed 'on' it can sit anywhere within it. The gap between the two
    bony margins is found as the longest sub-threshold run overlapping the
    operator's chord, and its ends are the refined borders.
    """
    p_ant = np.asarray(p_ant, float); p_post = np.asarray(p_post, float)
    raw = float(np.linalg.norm(p_post - p_ant))
    if raw <= 0:
        return None
    d = (p_post - p_ant) / raw
    a = p_ant - d * margin_mm
    b = p_post + d * margin_mm
    t, hu, _ = sample_along_line(array, ras_to_ijk, a, b, step_mm)

    below = (hu < level) if polarity == "gap" else (hu >= level)
    if not below.any():
        return {"status": "no_gap", "raw_chord_mm": raw, "polarity": polarity,
                "note": ("profile never drops below the bone threshold"
                         if polarity == "gap" else
                         "profile never rises above the bone threshold")}
    # contiguous runs of sub-threshold voxels
    edges = np.diff(below.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])
    if below[0]: starts = [0] + starts
    if below[-1]: ends = ends + [len(below) - 1]
    runs = [(s, e) for s, e in zip(starts, ends) if e > s]
    if not runs:
        return {"status": "no_gap", "raw_chord_mm": raw, "polarity": polarity}

    # keep the run overlapping the operator's chord, longest wins on a tie
    lo_t, hi_t = margin_mm, margin_mm + raw
    def overlap(run):
        s, e = t[run[0]], t[run[1]]
        return max(0.0, min(e, hi_t) - max(s, lo_t))
    runs.sort(key=lambda r: (overlap(r), t[r[1]] - t[r[0]]), reverse=True)
    s, e = runs[0]
    if overlap((s, e)) <= 0:
        return {"status": "no_overlapping_gap", "raw_chord_mm": raw,
                "polarity": polarity}

    # sub-voxel ends of that run.
    #
    # A run reaching the edge of the sampled window is NOT a bony margin, it is
    # the window running out. On real anatomy this is the normal case at one
    # end: the SPF has bone anteriorly at the crista ethmoidalis but opens
    # posteriorly into the nasal cavity, so no posterior margin exists along
    # the chord. Reporting the window edge as a margin silently inflated the
    # aperture and MULTIPLIED jitter variance by 2.5x on the first real scan.
    # Each end is therefore refined only if it is genuinely bounded.
    def sub_edge(i, j):
        h0, h1 = hu[i], hu[j]
        if h1 == h0:
            return float(t[i])
        frac = (level - h0) / (h1 - h0)
        return float(t[i] + frac * (t[j] - t[i]))

    ant_bounded = bool(s > 0)          # plain bool, not numpy, so the
    post_bounded = bool(e + 1 < len(t))  # result stays JSON serialisable
    t_start = sub_edge(s - 1, s) if ant_bounded else float(t[s])
    t_end = sub_edge(e, e + 1) if post_bounded else float(t[e])

    ref_ant = a + d * t_start if ant_bounded else np.asarray(p_ant, float)
    ref_post = a + d * t_end if post_bounded else np.asarray(p_post, float)
    if not (ant_bounded and post_bounded):
        unbounded = ([] if ant_bounded else ["anterior"]) + \
                    ([] if post_bounded else ["posterior"])
        return {"status": "partially_bounded",
                "polarity": polarity,
                "raw_chord_mm": raw,
                "refined_chord_mm": None,
                "delta_mm": None,
                "unbounded_ends": unbounded,
                "refined_ant_ras": ref_ant,
                "refined_post_ras": ref_post,
                "ant_shift_mm": float(np.linalg.norm(ref_ant - np.asarray(p_ant, float))),
                "post_shift_mm": float(np.linalg.norm(ref_post - np.asarray(p_post, float))),
                "ant_bounded": ant_bounded, "post_bounded": post_bounded,
                "note": ("no bone found beyond the " + " and ".join(unbounded) +
                         " end within the search margin, so that end is left "
                         "at the operator's placement and no refined span is "
                         "reported")}
    return {"status": "ok",
            "ant_bounded": True, "post_bounded": True,
            "polarity": polarity,
            "raw_chord_mm": raw,
            "refined_chord_mm": float(t_end - t_start),
            "delta_mm": float((t_end - t_start) - raw),
            "refined_ant_ras": ref_ant,
            "refined_post_ras": ref_post,
            "ant_shift_mm": float(np.linalg.norm(ref_ant - p_ant)),
            "post_shift_mm": float(np.linalg.norm(ref_post - p_post)),
            "n_runs_considered": len(runs),
            "min_hu_in_gap": float(hu[s:e + 1].min()),
            "max_hu_in_gap": float(hu[s:e + 1].max())}


def cast_ray_to_bone(array, ras_to_ijk, origin, direction, level=EDGE_HU,
                     max_mm=40.0, step_mm=EDGE_STEP_MM, skip_mm=0.0):
    """First bone crossing from a point along a direction.

    Replaces hand-placing the far point of a 'distance to surface'
    measurement. The direction is fixed by protocol and the crossing is
    objective, so the far end never gets placed by eye at all.
    """
    d = np.asarray(direction, float)
    norm = np.linalg.norm(d)
    if norm == 0:
        return None
    d = d / norm
    origin = np.asarray(origin, float)
    p0 = origin + d * float(skip_mm)
    p1 = origin + d * float(max_mm)
    t, hu, _ = sample_along_line(array, ras_to_ijk, p0, p1, step_mm)
    rising = [c for c in find_crossings(t, hu, level) if c["rising"]]
    if not rising:
        return None
    first = min(rising, key=lambda c: c["t"])
    hit = p0 + d * first["t"]
    return {"hit_ras": hit,
            "distance_mm": float(first["t"] + skip_mm),
            "n_crossings": len(rising)}


def bilateral_delta(value_r, value_l, tolerance_mm=None):
    """Left versus right on one repeat: a free error signal at zero cost.

    A paired structure differing wildly between sides is wrong NOW, without
    waiting for a second repeat or for analysis.
    """
    if value_r is None or value_l is None:
        return {"status": "incomplete"}
    diff = float(value_r) - float(value_l)
    mean = 0.5 * (float(value_r) + float(value_l))
    out = {"right": float(value_r), "left": float(value_l),
           "difference_mm": diff, "mean_mm": mean,
           "percent_of_mean": (100.0 * abs(diff) / mean) if mean else None}
    if tolerance_mm is not None:
        out["within_tolerance"] = bool(abs(diff) <= float(tolerance_mm))
    return out


# Which measurements are refinable, and how. Anything not listed here is left
# entirely alone.
REFINABLE = {
    "spf_diam_ax":        {"key": None, "mode": "aperture",
                           "why": "SPF aperture: a true foramen bounded by bony rims, "
                                  "one of which is thin enough that a fixed threshold "
                                  "misses it, so rim prominence over several profiles"},
    "spf_diam_cor":       {"key": None, "mode": "aperture",
                           "why": "same, in the coronal plane"},
    "lat_nasal_wall_cor": {"key": None, "mode": "chord", "polarity": "slab",
                           "why": "a bone thickness, so the span OF bone rather than between bones"},
}


def refine_measurements(nodes, node_names, volume_node, landmarks,
                        params=None, engine=None):
    """Compute edge-refined versions of the refinable measurements.

    Reports raw, refined and delta side by side. **Nothing here writes back
    into the results dict.** The manual value stays the record; the refined
    value is a parallel quantity, and the delta between them is the evidence
    of what refinement is worth on real anatomy.
    """
    import vtk
    if volume_node is None:
        return {"status": "no_volume"}
    array = slicer.util.arrayFromVolume(volume_node)
    mat = vtk.vtkMatrix4x4(); volume_node.GetIJKToRASMatrix(mat)
    M = np.array([[mat.GetElement(r, c) for c in range(4)] for r in range(4)])
    ras_to_ijk = np.linalg.inv(M)
    level = (params or {}).get("EDGE_HU", EDGE_HU)

    out = {"status": "ok", "level_hu": level, "items": {}}
    for role, spec in REFINABLE.items():
        node = nodes.get(role)
        if node is None:
            out["items"][role] = {"status": "node_absent"}
            continue
        p0 = [0.0, 0.0, 0.0]; p1 = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        if spec["mode"] == "aperture":
            res = refine_aperture(array, ras_to_ijk, p0, p1)
        else:
            res = refine_chord(array, ras_to_ijk, p0, p1, level=level,
                               polarity=spec["polarity"])
        if not res or res.get("status") != "ok":
            out["items"][role] = {"status": (res or {}).get("status", "failed"),
                                  "raw_mm": float(np.linalg.norm(
                                      np.array(p1) - np.array(p0))),
                                  "mode": spec["mode"]}
            continue
        item = {"status": "ok", "node": node_names.get(role),
                "mode": spec["mode"], "why": spec["why"],
                "method": res.get("method", "chord"),
                "raw_mm": round(res["raw_chord_mm"], 3),
                "refined_mm": round(res["refined_chord_mm"], 3),
                "delta_mm": round(res["delta_mm"], 3)}
        if "profile_spans_mm" in res:      # multi-profile quality metrics
            item.update({"n_profiles_ok": res["n_profiles_ok"],
                         "n_profiles": res["n_profiles"],
                         "profile_sd_mm": round(res["profile_sd_mm"], 3),
                         "profile_iqr_mm": round(res["profile_iqr_mm"], 3)})
        else:
            item["end_shifts_mm"] = [round(res.get("ant_shift_mm", 0), 3),
                                     round(res.get("post_shift_mm", 0), 3)]

        # Advisory companion: the direction-free minimum aperture through the
        # operator's midpoint. Reported ALONGSIDE, never substituted, because it
        # is a different measurand and swapping it in is the PI's decision.
        if spec["mode"] == "aperture":
            mid = (np.asarray(p0, float) + np.asarray(p1, float)) / 2.0
            plane = "coronal" if role.endswith("_cor") else "axial"
            mf = min_feret_aperture(array, ras_to_ijk, mid, plane=plane)
            entry = {"status": mf.get("status", "failed"), "plane": plane,
                     "note": ("different measurand from the protocol chord, "
                              "advisory only, adopting it is an amendment")}
            if mf.get("status") == "ok":
                entry.update({"aperture_mm": round(mf["aperture_mm"], 3),
                              "angle_deg": round(mf["angle_deg"], 1),
                              "vs_chord_mm": round(mf["aperture_mm"]
                                                   - res["refined_chord_mm"], 3)})
            item["min_feret_advisory"] = entry
        out["items"][role] = item

    # Distance-to-surface by ray cast, which removes a hand-placed far point.
    spf = landmarks.get("SPF_inf") if landmarks else None
    if spf is not None:
        ray = cast_ray_to_bone(array, ras_to_ijk, spf, [0.0, 0.0, -1.0],
                               level=level, max_mm=60.0, skip_mm=1.0)
        nf = landmarks.get("nasal_floor")
        entry = {"status": "ok" if ray else "no_crossing",
                 "why": ("vertical drop from the SPF to the first bone below, "
                         "so the nasal floor is never hand placed")}
        if ray:
            entry["refined_mm"] = round(ray["distance_mm"], 3)
            entry["hit_ras"] = [round(float(x), 2) for x in ray["hit_ras"]]
        if nf is not None:
            entry["raw_mm"] = round(float(abs(spf[2] - nf[2])), 3)
            if ray:
                entry["delta_mm"] = round(ray["distance_mm"]
                                          - abs(spf[2] - nf[2]), 3)
        out["items"]["spf_to_nasal_floor_raycast"] = entry
    return out


def refine_chord_by_rims(array, ras_to_ijk, p_ant, p_post,
                         prominence=RIM_PROMINENCE_HU,
                         search_mm=RIM_SEARCH_MM, step_mm=EDGE_STEP_MM):
    """Find a foramen's margins by LOCAL RIM PROMINENCE, not an absolute level.

    A fixed threshold is the wrong tool for a thin bony rim. The sphenopalatine
    foramen is a true foramen, bounded circumferentially, but its posterior rim
    at the sphenoidal process of the palatine bone is thin enough that partial
    volume caps it at about 227 HU on a 0.5 mm scan. A 300 HU threshold looks
    straight through it and wrongly reports the foramen as open posteriorly.

    Instead: locate the lumen around the chord midpoint, search outward each
    way for the first local maximum rising `prominence` above the lumen
    baseline, and take the margin as the half-maximum crossing on the lumen
    side of that peak. Adaptive, local, and indifferent to how dense the rim is.
    """
    p_ant = np.asarray(p_ant, float); p_post = np.asarray(p_post, float)
    raw = float(np.linalg.norm(p_post - p_ant))
    if raw <= 0:
        return None
    d = (p_post - p_ant) / raw
    a = p_ant - d * search_mm
    b = p_post + d * search_mm
    t, hu, _ = sample_along_line(array, ras_to_ijk, a, b, step_mm)
    t = t - search_mm                      # 0 at the anterior seed

    # Locate the lumen rather than assuming the seed midpoint is inside it: a
    # seed displaced into bone gives a bone-valued "baseline" and finds no rim.
    # Take the NEAREST window to the midpoint that is actually lumen, not the
    # globally darkest one. Measured on the real scan, "globally darkest" was
    # the worst of the three rules tried (SD 0.521 mm vs 0.457 mm), because the
    # SPF opens into the nasal cavity and the darkest point along an overshot
    # chord is often that cavity rather than the foramen.
    half = max(3, int(round(0.5 / step_mm)))
    span = (t >= 0.0) & (t <= raw)
    if span.sum() < 2 * half + 1:
        span = np.ones_like(t, dtype=bool)
    idxs = np.where(span)[0]
    meds = np.array([float(np.median(hu[max(0, i - half):min(len(hu), i + half + 1)]))
                     for i in idxs])
    centre = int(np.argmin(np.abs(t[idxs] - raw / 2.0)))
    order = sorted(range(len(idxs)), key=lambda w: abs(w - centre))
    pick = next((w for w in order if meds[w] < LUMEN_HU_MAX), int(np.argmin(meds)))
    mid_i = int(idxs[pick])
    baseline = float(meds[pick])

    def find_rim(direction):
        i = mid_i
        best = None
        while 0 < i < len(hu) - 1:
            i += direction
            if not (0 <= i < len(hu)):
                break
            if hu[i] >= baseline + prominence:
                # walk to the local peak
                j = i
                while 0 < j < len(hu) - 1 and hu[j + direction] > hu[j]:
                    j += direction
                best = j
                break
        if best is None:
            return None
        peak = float(hu[best])
        level = baseline + 0.5 * (peak - baseline)     # local half maximum
        k = best
        while k != mid_i and hu[k] > level:
            k -= direction
        k2 = k + direction
        if hu[k2] == hu[k]:
            return {"t": float(t[k]), "peak_hu": peak, "level": level}
        frac = (level - hu[k]) / (hu[k2] - hu[k])
        return {"t": float(t[k] + frac * (t[k2] - t[k])),
                "peak_hu": peak, "level": level, "peak_t": float(t[best])}

    ant_rim = find_rim(-1)
    post_rim = find_rim(+1)
    if ant_rim is None or post_rim is None:
        missing = ([] if ant_rim else ["anterior"]) + ([] if post_rim else ["posterior"])
        return {"status": "rim_not_found", "raw_chord_mm": raw,
                "missing_rims": missing, "baseline_hu": baseline}
    span = float(post_rim["t"] - ant_rim["t"])
    return {"status": "ok",
            "method": "rim_prominence",
            "raw_chord_mm": raw,
            "refined_chord_mm": span,
            "delta_mm": span - raw,
            "baseline_hu": baseline,
            "anterior_rim": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in ant_rim.items()},
            "posterior_rim": {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in post_rim.items()},
            "refined_ant_ras": a + d * (ant_rim["t"] + search_mm),
            "refined_post_ras": a + d * (post_rim["t"] + search_mm)}


def min_feret_aperture(array, ras_to_ijk, seed, plane="axial",
                       n_angles=FERET_N_ANGLES, prominence=RIM_PROMINENCE_HU,
                       search_mm=RIM_SEARCH_MM, step_mm=EDGE_STEP_MM):
    """The narrowest rim-to-rim aperture through `seed`, over all in-plane angles.

    This takes ONE seed point and no direction. That is the whole point. A
    hand-drawn chord carries two error sources, where the operator put the ends
    AND what angle they drew at, and an oblique chord across an irregular
    aperture is always wider than the true one. Searching every angle for the
    minimum removes the second source completely: the measurand stops being
    "whatever chord the operator drew" and becomes a well defined property of
    the anatomy, so two measurers who click anywhere inside the same foramen
    get the same number.

    It also answers the question the device actually poses. A stent has to pass
    the narrowest constriction, not an average or an arbitrary chord.

    Measured on the real scan this reads 3.63 mm where the operator's chord read
    6.35 mm, against a published SPF width of 3.79 +/- 0.35 mm.

    Note this is a different measurand from the protocol's current chord, so
    adopting it is a protocol amendment for the PI to decide, not a refinement
    that can be applied silently.
    """
    seed = np.asarray(seed, float)
    baseline = _lumen_baseline_at(array, ras_to_ijk, seed, step_mm)
    if baseline is None or baseline >= LUMEN_HU_MAX:
        return {"status": "seed_not_in_lumen", "baseline_hu": baseline}

    best = None
    for ang in np.linspace(0.0, np.pi, int(n_angles), endpoint=False):
        if plane == "axial":                      # R-A plane, constant S
            u = np.array([np.cos(ang), np.sin(ang), 0.0])
        elif plane == "coronal":                  # R-S plane, constant A
            u = np.array([np.cos(ang), 0.0, np.sin(ang)])
        else:                                     # sagittal, A-S plane
            u = np.array([0.0, np.cos(ang), np.sin(ang)])
        d_pos = _rim_distance(array, ras_to_ijk, seed, u, baseline,
                              prominence, search_mm, step_mm)
        d_neg = _rim_distance(array, ras_to_ijk, seed, -u, baseline,
                              prominence, search_mm, step_mm)
        if d_pos is None or d_neg is None:
            continue
        total = d_pos + d_neg
        if best is None or total < best["aperture_mm"]:
            best = {"aperture_mm": total, "angle_deg": float(np.degrees(ang)),
                    "axis_ras": u.tolist(),
                    "p1_ras": (seed + u * d_pos).tolist(),
                    "p2_ras": (seed - u * d_neg).tolist()}
    if best is None:
        return {"status": "no_bounded_direction", "baseline_hu": baseline}
    best.update({"status": "ok", "method": "min_feret",
                 "plane": plane, "baseline_hu": baseline,
                 "n_angles": int(n_angles)})
    return best


def _lumen_baseline_at(array, ras_to_ijk, seed, step_mm=EDGE_STEP_MM, radius_mm=0.6):
    """Median HU in a small neighbourhood of `seed`, as the lumen reference."""
    vals = []
    for axis in np.eye(3):
        for sgn in (+1.0, -1.0):
            _, hu, _ = sample_along_line(array, ras_to_ijk, seed,
                                         seed + axis * sgn * radius_mm, step_mm)
            if len(hu):
                vals.extend(hu.tolist())
    return float(np.median(vals)) if vals else None


def _rim_distance(array, ras_to_ijk, origin, direction, baseline,
                  prominence=RIM_PROMINENCE_HU, search_mm=RIM_SEARCH_MM,
                  step_mm=EDGE_STEP_MM):
    """Distance from `origin` to the half-maximum of the first prominent rim."""
    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    t, hu, _ = sample_along_line(array, ras_to_ijk, origin,
                                 origin + direction * search_mm, step_mm)
    peak = None
    for i in range(1, len(hu)):
        if hu[i] - baseline >= prominence and (peak is None or hu[i] > hu[peak]):
            peak = i
        elif peak is not None and hu[i] < baseline + 0.3 * prominence:
            break
    if peak is None:
        return None
    level = baseline + 0.5 * (hu[peak] - baseline)
    for i in range(peak, 0, -1):
        if hu[i - 1] < level <= hu[i]:
            frac = (level - hu[i - 1]) / (hu[i] - hu[i - 1])
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return None


def refine_aperture(array, ras_to_ijk, p_ant, p_post, offset_dir=None,
                    n_profiles=APERTURE_N_PROFILES,
                    spread_mm=APERTURE_SPREAD_MM, **kw):
    """Rim refinement averaged over several parallel profiles.

    A single 1D profile is at the mercy of trabecular texture and noise at
    whichever height it happens to cross. Sampling several parallel chords
    through the foramen and taking the MEDIAN span is the standard remedy: it
    is robust to one profile clipping a spicule or catching a gap, and it costs
    only a few extra samples.

    offset_dir defaults to the superior-inferior component orthogonal to the
    chord, which for an axial SPF chord walks up and down the foramen's height.
    """
    p_ant = np.asarray(p_ant, float); p_post = np.asarray(p_post, float)
    chord = p_post - p_ant
    L = float(np.linalg.norm(chord))
    if L <= 0:
        return None
    ax = chord / L
    if offset_dir is None:
        offset_dir = np.array([0.0, 0.0, 1.0])
        offset_dir = offset_dir - (offset_dir @ ax) * ax
        if np.linalg.norm(offset_dir) < 1e-6:
            offset_dir = np.array([1.0, 0.0, 0.0])
            offset_dir = offset_dir - (offset_dir @ ax) * ax
    offset_dir = np.asarray(offset_dir, float)
    offset_dir /= np.linalg.norm(offset_dir)

    offsets = np.linspace(-spread_mm, spread_mm, int(n_profiles))
    spans, details = [], []
    for off in offsets:
        shift = offset_dir * float(off)
        r = refine_chord_by_rims(array, ras_to_ijk, p_ant + shift,
                                 p_post + shift, **kw)
        ok = bool(r and r.get("status") == "ok")
        details.append({"offset_mm": round(float(off), 2),
                        "status": (r or {}).get("status", "failed"),
                        "span_mm": (round(r["refined_chord_mm"], 3) if ok else None)})
        if ok:
            spans.append(r["refined_chord_mm"])
    if not spans:
        return {"status": "no_profile_succeeded", "raw_chord_mm": L,
                "profiles": details}
    spans = np.asarray(spans)
    return {"status": "ok",
            "method": "rim_prominence_multiprofile",
            "raw_chord_mm": L,
            "refined_chord_mm": float(np.median(spans)),
            "delta_mm": float(np.median(spans) - L),
            "n_profiles_ok": int(len(spans)),
            "n_profiles": int(n_profiles),
            "spread_mm": float(spread_mm),
            "profile_spans_mm": [round(float(x), 3) for x in spans],
            "profile_sd_mm": float(np.std(spans)),
            "profile_iqr_mm": float(np.percentile(spans, 75)
                                    - np.percentile(spans, 25)),
            "profiles": details}


############ SEGMENT EDITOR PLUMBING ############################


@contextlib.contextmanager
def segment_editor_session(segmentation_node, volume_node):
    """Segment Editor widget, guaranteed to clean up.

    Leaked segment editor widgets are a known Slicer crash source, hence the
    contextmanager. Also shims the 5.10 master -> source rename so the same
    code works either side of it.
    """
    widget = slicer.qMRMLSegmentEditorWidget()
    widget.setMRMLScene(slicer.mrmlScene)
    editor_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentEditorNode")
    try:
        widget.setMRMLSegmentEditorNode(editor_node)
        widget.setSegmentationNode(segmentation_node)
        setter = (getattr(widget, "setSourceVolumeNode", None)
                  or getattr(widget, "setMasterVolumeNode"))
        setter(volume_node)
        yield widget, editor_node
    finally:
        widget.setMRMLSegmentEditorNode(None)
        widget.setSegmentationNode(None)
        slicer.mrmlScene.RemoveNode(editor_node)
        del widget


def _set_intensity_mask(editor_node, enabled, lo=None, hi=None):
    """Intensity mask, across the 5.10 rename."""
    for flag_name, range_name in (
        ("SetSourceVolumeIntensityMask", "SetSourceVolumeIntensityMaskRange"),
        ("SetMasterVolumeIntensityMask", "SetMasterVolumeIntensityMaskRange"),
    ):
        flag = getattr(editor_node, flag_name, None)
        if flag is None:
            continue
        flag(enabled)
        if enabled and lo is not None:
            getattr(editor_node, range_name)(lo, hi)
        return True
    return False


def apply_effect(widget, effect_name, parameters=None):
    """Select an effect, set its string parameters, apply."""
    widget.setActiveEffectByName(effect_name)
    effect = widget.activeEffect()
    if effect is None:
        raise RuntimeError(f"Segment Editor effect {effect_name!r} unavailable")
    for key, value in (parameters or {}).items():
        effect.setParameter(key, str(value))
    effect.self().onApply()
    widget.setActiveEffectByName("")
    return True


def ensure_segmentation(name, volume_node, run_id, cfg_hash):
    """Create the auto segmentation node with surface conversion pinned off.

    Smoothing factor and Decimation factor are pinned to 0 deliberately: the
    only smoothing applied to a cut surface is the controlled windowed-sinc
    pass in smooth_surface, because labelmap smoothing above ~0.1 severs thin
    turbinate walls.
    """
    node = get_node_strict(name)
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", name)
        tag_auto_node(node, run_id, cfg_hash)
    node.CreateDefaultDisplayNodes()
    node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
    segmentation = node.GetSegmentation()
    segmentation.SetConversionParameter("Smoothing factor", "0.0")
    segmentation.SetConversionParameter("Decimation factor", "0.0")
    return node


def add_segment_from_mask(segmentation_node, volume_node, name, mask):
    """Import a boolean numpy mask (K,J,I order) as a named segment."""
    # CreateAndAddLabelVolume, not a bare AddNewNodeByClass: a
    # vtkMRMLScalarVolumeNode is not a labelmap, and the label volume must
    # inherit the source volume's geometry or the segment lands misaligned.
    volumes_logic = slicer.modules.volumes.logic()
    labelmap = volumes_logic.CreateAndAddLabelVolume(
        slicer.mrmlScene, volume_node, f"__tmp_{name}")
    try:
        array = slicer.util.arrayFromVolume(labelmap)
        # arrayFromVolume returns a (K,J,I) VIEW, so write in place and then
        # notify, rather than rebinding the name.
        array[:] = mask.astype(array.dtype)
        slicer.util.arrayFromVolumeModified(labelmap)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmap, segmentation_node)
        segmentation = segmentation_node.GetSegmentation()
        segment_id = segmentation.GetNthSegmentID(
            segmentation.GetNumberOfSegments() - 1)
        segmentation.GetSegment(segment_id).SetName(name)
        return segment_id
    finally:
        slicer.mrmlScene.RemoveNode(labelmap)


############ STAGE 1: TISSUE CLASSES ############################


def stage_tissue(volume_node, params, run_id, cfg_hash, check):
    """Build the air / enclosed-air / body / bone / metal classes.

    Everything downstream cuts contours out of these, so this stage owns the
    HU decisions and the diagnostics that say whether they were safe.
    """
    import scipy.ndimage as ndi

    array = slicer.util.arrayFromVolume(volume_node)  # (K, J, I)
    spacing = volume_node.GetSpacing()
    voxel_count = int(array.size)
    record = {"voxel_count": voxel_count, "spacing": list(spacing)}

    # Out-of-FOV reconstruction padding masquerading as air. This is why the
    # air threshold is bounded below rather than an open `< -300`.
    below = int(np.count_nonzero(array < params["AIR_HU_MIN"]))
    record["voxels_below_air_min"] = below
    record["out_of_fov_fraction"] = below / max(voxel_count, 1)
    if record["out_of_fov_fraction"] > params["OUT_OF_FOV_FRAC_FLAG"]:
        check("crop_outside_fov", "warn",
              f"{record['out_of_fov_fraction']:.4%} of voxels sit below "
              f"{params['AIR_HU_MIN']} HU, so the crop extends past the "
              f"reconstruction circle. Padding can connect to external air "
              f"and defeat the enclosed-air border drop.")

    # Inclusive on both ends: a strict > would drop voxels clamped at exactly
    # AIR_HU_MIN, which is what most scanners emit for pure air.
    air = (array >= params["AIR_HU_MIN"]) & (array <= params["AIR_HU_MAX"])
    bone = (array >= params["BONE_HU_MIN"]) & (array <= params["BONE_HU_MAX"])
    dense_bone = array >= params["BONE_HU_DENSE"]
    metal = array >= params["METAL_HU_MIN"]

    record["air_voxels"] = int(air.sum())
    record["bone_voxels"] = int(bone.sum())
    record["dense_bone_voxels"] = int(dense_bone.sum())
    record["metal_voxels"] = int(metal.sum())

    # Body: largest non-air component, then per-slice hole fill. The skin
    # surface of this is the right source for cheek contours, far more robust
    # than thresholding soft tissue directly.
    body_raw = array >= params["BODY_HU_MIN"]
    labels, n_labels = ndi.label(body_raw)
    if n_labels:
        sizes = ndi.sum(body_raw, labels, range(1, n_labels + 1))
        body = labels == (int(np.argmax(sizes)) + 1)
        for k in range(body.shape[0]):
            body[k] = ndi.binary_fill_holes(body[k])
    else:
        body = np.zeros_like(body_raw)
    record["body_voxels"] = int(body.sum())

    # Enclosed air: per-slice, drop components touching the slice border, so
    # room and table air never enters the mask.
    enclosed = np.zeros_like(air)
    for k in range(air.shape[0]):
        sl = air[k]
        if not sl.any():
            continue
        lab, n = ndi.label(sl)
        if not n:
            continue
        border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
        border.discard(0)
        keep = np.isin(lab, [i for i in range(1, n + 1) if i not in border])
        enclosed[k] = keep
    record["enclosed_air_voxels"] = int(enclosed.sum())

    # Metal: streak artefact contaminates the whole axial slice through the
    # metal, not just its neighbourhood, so the affected z-range is what
    # matters downstream.
    saturated = int((array >= params["METAL_SATURATION_HU"]).sum())
    record["metal_saturated_voxels"] = saturated
    record["metal_saturated_fraction"] = (
        saturated / max(record["metal_voxels"], 1))
    record["metal_confirmed"] = bool(
        record["metal_voxels"] >= params["METAL_MIN_VOXELS"]
        and record["metal_saturated_fraction"]
        >= params["METAL_MIN_SATURATED_FRACTION"])
    if record["metal_confirmed"]:
        z_indices = np.where(metal.any(axis=(1, 2)))[0]
        record["metal_slice_range_k"] = [int(z_indices.min()),
                                         int(z_indices.max())]
        check("metal_present", "warn",
              f"{record['metal_voxels']} voxels above "
              f"{params['METAL_HU_MIN']} HU across slices "
              f"{record['metal_slice_range_k']}. Every curve on a plane "
              f"intersecting that range is forced to requires_confirmation.")
        bone = bone & ~metal
    else:
        record["metal_slice_range_k"] = None

    # Bone threshold sweep. Unlike air, this one genuinely matters: a large
    # jump in component count between the sweep values means this subject's
    # thin walls are marginal and every downstream separation is fragile.
    sweep = {}
    for value in params["BONE_HU_MIN_SWEEP"]:
        mask = (array >= value) & (array <= params["BONE_HU_MAX"])
        _, n_components = ndi.label(mask)
        sweep[str(value)] = {
            "voxels": int(mask.sum()),
            "components": int(n_components),
            "volume_cm3": float(mask.sum() * np.prod(spacing) / 1000.0),
        }
    record["bone_threshold_sweep"] = sweep
    counts = [v["components"] for v in sweep.values()]
    if counts and max(counts) > 3 * max(min(counts), 1):
        check("bone_threshold_fragile", "warn",
              f"bone component count varies {min(counts)} to {max(counts)} "
              f"across BONE_HU_MIN_SWEEP, so thin walls in this subject are "
              f"marginal")

    # Air threshold sweep, retained as the partial-volume detector. Contour
    # displacement per curve is computed in stage_curves.
    air_sweep = {}
    for value in params["AIR_HU_MAX_SWEEP"]:
        mask = (array >= params["AIR_HU_MIN"]) & (array <= value)
        air_sweep[str(value)] = {"voxels": int(mask.sum())}
    record["air_threshold_sweep"] = air_sweep

    segmentation_node = ensure_segmentation(
        f"{AUTO_PREFIX}SEG", volume_node, run_id, cfg_hash)
    segment_ids = {}
    for name, mask in (
        ("AUTO_air_all", air),
        ("AUTO_air_enclosed", enclosed),
        ("AUTO_body", body),
        ("AUTO_bone", bone),
        ("AUTO_bone_dense", dense_bone & bone),
        ("AUTO_metal", metal),
    ):
        if not mask.any():
            continue
        segment_ids[name] = add_segment_from_mask(
            segmentation_node, volume_node, name, mask)

    # Island cleanup on the two classes that feed contours.
    with segment_editor_session(segmentation_node, volume_node) as (widget, _):
        for name, min_size in (
            ("AUTO_air_enclosed", params["AIR_ISLAND_MIN_VOXELS"]),
            ("AUTO_bone", params["BONE_ISLAND_MIN_VOXELS"]),
        ):
            if name not in segment_ids:
                continue
            widget.setCurrentSegmentID(segment_ids[name])
            apply_effect(widget, "Islands", {
                "Operation": "REMOVE_SMALL_ISLANDS",
                "MinimumSize": int(min_size),
            })

    record["segments"] = segment_ids
    record["segmentation_node"] = segmentation_node.GetName()
    return record, segmentation_node


############ STAGE 2: FRAME ############################


def stage_frame(volume_node, segmentation_node, spg, landmarks, params, check):
    """Anatomical frame: midline, side sign, and the four cut plane offsets.

    Plane offsets snap to the nearest voxel-center plane because that is where
    Slicer's slice views land, and therefore where the measurer drew.
    """
    import scipy.ndimage as ndi

    record = {}
    if spg is None:
        return {"status": "unavailable",
                "reason": "SPG centroid required, supply the 3 line nodes"}

    origin = volume_node.GetOrigin()
    spacing = volume_node.GetSpacing()

    def snap(value, axis):
        """Snap to the voxel-center lattice, then nudge off it.

        PLANE_EPS_MM matters: marching-cubes vertices sit on the voxel-center
        lattice, so a cut exactly through one hits many vertices exactly and
        produces degenerate coincident geometry the stripper cannot join.
        """
        o, s = origin[axis], spacing[axis]
        snapped = o + round((value - o) / s) * s
        return snapped + params["PLANE_EPS_MM"]

    # Midline, two independent estimates. Disagreement usually means a
    # deviated septum, which breaks the symmetry priors used for bone seeds.
    array = slicer.util.arrayFromVolume(volume_node)
    body = array >= params["BODY_HU_MIN"]
    centroid_x = None
    if body.any():
        com = ndi.center_of_mass(body)
        centroid_ras = _ijk_to_ras(volume_node, (com[2], com[1], com[0]))
        centroid_x = float(centroid_ras[0])
    record["midline_from_body_centroid"] = centroid_x
    x_mid = centroid_x if centroid_x is not None else float(spg[0])
    record["x_mid"] = x_mid

    side_sign = 1.0 if float(spg[0]) >= x_mid else -1.0
    record["side_sign"] = side_sign
    record["side_inferred"] = "R" if side_sign > 0 else "L"

    planes = {
        "axial_spg": {"orientation": "axial", "axis": 2,
                      "offset": snap(float(spg[2]), 2)},
    }
    spf_inf = landmarks.get("SPF_inf")
    if spf_inf is not None:
        planes["coronal_spf_inf"] = {"orientation": "coronal", "axis": 1,
                                     "offset": snap(float(spf_inf[1]), 1)}
    planes["coronal_ppf"] = {"orientation": "coronal", "axis": 1,
                             "offset": snap(float(spg[1]), 1)}
    planes["sagittal_side"] = {"orientation": "sagittal", "axis": 0,
                               "offset": snap(float(spg[0]), 0)}
    record["planes"] = planes
    record["status"] = "ok"
    return record


def _ijk_to_ras(volume_node, ijk):
    import vtk
    matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix)
    homogeneous = [ijk[0], ijk[1], ijk[2], 1.0]
    out = [0.0, 0.0, 0.0, 1.0]
    matrix.MultiplyPoint(homogeneous, out)
    return out[:3]


############ STAGE 3: CURVE ENGINE ############################

PLANE_BASIS = {
    "axial": (0, 1, 2),     # u=R, v=A, constant=S
    "coronal": (0, 2, 1),   # u=R, v=S, constant=A
    "sagittal": (1, 2, 0),  # u=A, v=S, constant=R
}


def segment_surface(segmentation_node, segment_id):
    """World-space closed surface polydata for one segment."""
    import vtk
    segmentation_node.CreateClosedSurfaceRepresentation()
    poly = vtk.vtkPolyData()
    # In Slicer 5.10 this writes into an output argument rather than
    # returning; older signatures returned the polydata. Support both.
    getter = getattr(segmentation_node, "GetClosedSurfaceRepresentation", None)
    if getter is not None:
        try:
            getter(segment_id, poly)
        except TypeError:
            got = getter(segment_id)
            if got is not None:
                poly.DeepCopy(got)
    if poly.GetNumberOfPoints() == 0:
        name = slicer.vtkSegmentationConverter\
            .GetSegmentationClosedSurfaceRepresentationName()
        got = segmentation_node.GetSegmentation()\
            .GetSegment(segment_id).GetRepresentation(name)
        if got is None:
            raise RuntimeError(
                f"no closed surface representation for segment {segment_id}")
        poly.DeepCopy(got)
    parent = segmentation_node.GetParentTransformNode()
    if parent is not None:
        transform = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            parent, None, transform)
        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetTransform(transform)
        filt.SetInputData(poly)
        filt.Update()
        poly = filt.GetOutput()
    return poly


def smooth_surface(poly, passband, iterations):
    """Windowed-sinc, topology preserving. Feature and boundary smoothing off."""
    import vtk
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(int(iterations))
    smoother.SetPassBand(float(passband))
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def cut_loops(poly, orientation, offset, params):
    """Cut a surface with an axis-aligned plane, return ordered 3D loops.

    vtkCleanPolyData between cutter and stripper is mandatory, not cosmetic:
    the cutter emits thousands of independent 2-point line cells with
    duplicated coincident points, and without point merging the stripper
    cannot join them, yielding one "loop" per triangle edge.
    """
    import vtk

    axis = {"axial": 2, "coronal": 1, "sagittal": 0}[orientation]
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    origin = [0.0, 0.0, 0.0]
    origin[axis] = float(offset)

    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(poly)
    cutter.SetValue(0, 0.0)
    cutter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(cutter.GetOutputPort())
    cleaner.PointMergingOn()
    cleaner.SetTolerance(0.0)
    cleaner.Update()
    cleaned = cleaner.GetOutput()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cleaner.GetOutputPort())
    stripper.SetMaximumLength(100000)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    stripped = stripper.GetOutput()

    loops = []
    points = stripped.GetPoints()
    lines = stripped.GetLines()
    if points is not None and lines is not None:
        id_list = vtk.vtkIdList()
        lines.InitTraversal()
        while lines.GetNextCell(id_list):
            n = id_list.GetNumberOfIds()
            if n < 3:
                continue
            coords = np.array([points.GetPoint(id_list.GetId(i))
                               for i in range(n)])
            closed = bool(id_list.GetId(0) == id_list.GetId(n - 1))
            if closed and len(coords) > 1:
                coords = coords[:-1]
            loops.append({"points": coords, "closed": closed})

    # Independent count via connectivity. A mismatch means the stripper failed
    # to join and the loop set is untrustworthy, so it is reported rather than
    # silently used.
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(cleaned)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    n_regions = connectivity.GetNumberOfExtractedRegions()

    loops = [orient_loop(loop, orientation) for loop in loops]
    # Compare like with like: the connectivity count is unfiltered, so it must
    # be checked against the RAW stripper count, before the small-perimeter
    # cull. Comparing it to the filtered list made this alarm fire on every
    # curve of every real scan, which is worse than useless.
    n_raw = len(loops)
    loops = [loop for loop in loops
             if loop["perimeter_mm"] >= params["LOOP_MIN_PERIMETER_MM"]]
    return loops, {"n_loops_raw": n_raw,
                   "n_loops_kept": len(loops),
                   "n_regions_connectivity": int(n_regions),
                   "counts_agree": n_raw == int(n_regions)}


def orient_loop(loop, orientation):
    """Make the point sequence a pure function of the geometry.

    Force counter-clockwise in a fixed 2D basis by shoelace sign, then rotate
    the start index to the lexicographically minimal vertex. Without this the
    emitted sequence depends on VTK's internal ordering and the run is not
    reproducible.
    """
    u_ax, v_ax, c_ax = PLANE_BASIS[orientation]
    points = loop["points"]
    u, v = points[:, u_ax], points[:, v_ax]
    area = 0.5 * float(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v))
    if area < 0:
        points = points[::-1]
        u, v = points[:, u_ax], points[:, v_ax]
        area = -area
    start = int(np.lexsort((v, u))[0])
    points = np.roll(points, -start, axis=0)
    diffs = np.diff(np.vstack([points, points[:1]]), axis=0)
    loop.update({
        "points": points,
        "signed_area_mm2": area,
        "perimeter_mm": float(np.sum(np.linalg.norm(diffs, axis=1))),
        "centroid": points.mean(axis=0).tolist(),
        "orientation": orientation,
        "u_axis": u_ax, "v_axis": v_ax, "const_axis": c_ax,
    })
    return loop


def loop_normals_2d(loop):
    """Outward 2D normals for a CCW loop.

    Sign convention consequence worth knowing: a loop cut from the BONE
    segment encloses bone, so its outward normal points into the airway.
    "Medial surface of the middle turbinate" is therefore exactly "the arc
    whose outward normal points medially".
    """
    points = loop["points"]
    u_ax, v_ax = loop["u_axis"], loop["v_axis"]
    p2 = np.column_stack([points[:, u_ax], points[:, v_ax]])
    tangents = np.roll(p2, -1, axis=0) - np.roll(p2, 1, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    tangents = tangents / lengths
    return np.column_stack([tangents[:, 1], -tangents[:, 0]])


def longest_contiguous_run(mask, seed_index, closed=True):
    """Keep only the run of True values containing the seed.

    This is the step that stops a predicate like "normal faces medially" from
    returning two disjoint arcs on opposite nasal walls, which is the most
    common way an automated contour goes wrong while still looking plausible.
    """
    n = len(mask)
    if n == 0 or not mask.any():
        return np.zeros(n, dtype=bool)
    if not mask[seed_index % n]:
        candidates = np.where(mask)[0]
        seed_index = int(candidates[
            np.argmin(np.abs(candidates - (seed_index % n)))])
    keep = np.zeros(n, dtype=bool)
    keep[seed_index % n] = True
    limit = n if closed else n
    i = seed_index
    for _ in range(limit):
        j = (i + 1) % n
        if not mask[j] or keep[j]:
            break
        keep[j] = True
        i = j
    i = seed_index
    for _ in range(limit):
        j = (i - 1) % n
        if not mask[j] or keep[j]:
            break
        keep[j] = True
        i = j
    return keep


def resample_polyline(points, spacing_mm, min_points, max_points):
    """Arc-length resample to roughly spacing_mm between control points."""
    if len(points) < 2:
        return points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(arc[-1])
    if total <= 0:
        return points[:1]
    n = int(np.clip(round(total / float(spacing_mm)) + 1,
                    int(min_points), int(max_points)))
    targets = np.linspace(0.0, total, n)
    return np.column_stack([
        np.interp(targets, arc, points[:, axis]) for axis in range(3)
    ])


def create_curve_node(name, points, run_id, cfg_hash, params,
                      requires_confirmation=False, extra=None):
    """Emit a locked, tagged, linear markups curve.

    Linear rather than spline: splines overshoot on residual marching-cubes
    stair-step, and overshoot on a morphometry contour is an invented
    measurement. Density is then restored via points-per-segment so the
    minimum-distance searches in the manual engine do not quantize.
    """
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
    slicer.util.updateMarkupsControlPointsFromArray(node, np.asarray(points))
    node.SetCurveTypeToLinear()
    node.SetNumberOfPointsPerInterpolatingSegment(
        int(params["CURVE_POINTS_PER_SEGMENT"]))
    display = node.GetDisplayNode()
    if display is not None:
        display.SetSelectedColor(0.0, 1.0, 1.0)   # cyan marks auto output
        display.SetPropertiesLabelVisibility(False)
        display.SetGlyphScale(1.0)
    tag_auto_node(node, run_id, cfg_hash, requires_confirmation, extra)
    return node


############ CURVE SPECIFICATIONS ############################
# One entry per curve the manual engine actually consumes. Note this targets
# the SCRIPT's set of 9, not Appendix A's list of 10: only these feed a
# measurement. The discrepancy between the two lists is logged as an open
# question rather than silently reconciled here.
#
# source        which tissue segment to cut
# plane         key into the stage_frame planes dict
# score         how to pick among candidate loops, never "largest" blindly
# trim          None, or (mode, argument)
# role          the node role the measurement layer looks the result up by
# confirm       True when this output is structurally less reliable

CURVE_SPECS = (
    # Cheek: the facial skin surface, which is the one loop that ENCLOSES the
    # SPG. Stated that way it is unambiguous and needs no distance heuristic.
    # Scoring it by centroid proximity was what produced a "cheek" curve 6.7 mm
    # from a ganglion that sits tens of millimetres deep to the skin.
    {"key": "cheek_ax", "role": "cheek_ax", "source": "AUTO_body",
     "plane": "axial_spg", "score": "largest_enclosing", "anchor": "spg",
     "perimeter_mm": (200.0, 4000.0),
     "trim": ("angular_outer", "lateral"), "confirm": False},
    {"key": "cheek_cor", "role": "cheek_cor", "source": "AUTO_body",
     "plane": "coronal_spf_inf", "score": "largest_enclosing", "anchor": "spg",
     "perimeter_mm": (200.0, 4000.0),
     "trim": ("angular_outer", "lateral"), "confirm": False},
    # Ipsilateral nasal airway at the SPG axial level. Gated on a real area
    # window and on a majority of points being ipsilateral, so a small sinus
    # cell or the contralateral cavity cannot win on proximity.
    {"key": "nasal_cavity_ax", "role": "nasal_cavity_ax",
     "source": "AUTO_air_enclosed", "plane": "axial_spg",
     "score": "nearest_anchor", "anchor": "spg",
     "area_mm2": (40.0, 2500.0), "min_side_fraction": 0.5,
     "trim": ("hemispace", None), "confirm": False},
    # The septum is the medial wall of that same airway loop, so the same
    # selection with a different trim. Side gate relaxed because a midline
    # structure legitimately straddles.
    {"key": "septum_ax", "role": "septum_ax", "source": "AUTO_air_enclosed",
     "plane": "axial_spg", "score": "nearest_anchor", "anchor": "spg",
     "area_mm2": (40.0, 2500.0), "min_side_fraction": 0.35,
     "trim": ("septum_band", None), "confirm": False},
    # Turbinates are small bone islands. Anchored on the SPF landmark rather
    # than the SPG, since the SPF sits on the lateral nasal wall beside them.
    {"key": "mt_medial_ax", "role": "mt_medial_ax", "source": "AUTO_bone",
     "plane": "axial_spg", "score": "nearest_anchor", "anchor": "SPF_inf",
     "perimeter_mm": (8.0, 200.0), "min_side_fraction": 0.5,
     "trim": ("normal", "medial"), "confirm": False},
    # Coronal turbinates remain the least reliable of the set: coronally they
    # are peninsulas off the lateral wall rather than holes in the air region,
    # so there is no topological handle and selection rests on position priors.
    {"key": "mt_medial_cor", "role": "mt_medial_cor", "source": "AUTO_bone",
     "plane": "coronal_spf_inf", "score": "nearest_anchor", "anchor": "SPF_inf",
     "perimeter_mm": (5.0, 250.0), "min_side_fraction": 0.5,
     "trim": ("normal", "medial"), "confirm": True},
    {"key": "it_superior_cor", "role": "it_superior_cor", "source": "AUTO_bone",
     "plane": "coronal_spf_inf", "score": "nearest_anchor",
     "anchor": "nasal_floor", "perimeter_mm": (5.0, 250.0),
     "min_side_fraction": 0.5,
     "trim": ("normal", "superior"), "confirm": True},
    {"key": "mt_inferior_cor", "role": "mt_inferior_cor", "source": "AUTO_bone",
     "plane": "coronal_spf_inf", "score": "nearest_anchor", "anchor": "SPF_inf",
     "perimeter_mm": (5.0, 250.0), "min_side_fraction": 0.5,
     "trim": ("normal", "inferior"), "confirm": True},
    # Orbital floor anchored on the inferior orbital fissure landmark, which
    # sits on the orbit itself, rather than on the SPG below it.
    {"key": "orbital_floor_cor", "role": "orbital_floor_cor",
     "source": "AUTO_bone", "plane": "coronal_ppf", "score": "superior_nearest",
     "anchor": "IOF_medial", "perimeter_mm": (5.0, 900.0),
     "min_side_fraction": 0.4,
     "trim": ("normal", "inferior"), "confirm": True},
)

TRIM_DIRECTIONS = {
    "superior": (0.0, 1.0),
    "inferior": (0.0, -1.0),
}


def _direction_2d(loop, name, side_sign):
    """Unit direction in the loop's own 2D basis."""
    if name in TRIM_DIRECTIONS:
        return np.array(TRIM_DIRECTIONS[name], dtype=float)
    if name == "medial":
        # Medial means toward the midline, which is -side_sign along u.
        return np.array([-side_sign, 0.0])
    if name == "lateral":
        return np.array([side_sign, 0.0])
    raise ValueError(f"unknown trim direction {name!r}")


def loop_min_distance(loop, point):
    """Distance from a point to the NEAREST POINT on the loop.

    Not the centroid. Centroid distance is meaningless for an enclosing
    contour: the facial skin outline's centroid sits mid-head, a few
    centimetres from the SPG, while the contour itself is tens of
    millimetres away. Scoring on centroids made a 1507 mm skin contour look
    both closer and contralateral, which is how the cheek curve ended up
    6.7 mm from the SPG on a real head.
    """
    return float(np.linalg.norm(loop["points"] - np.asarray(point),
                                axis=1).min())


def loop_side_fraction(loop, x_mid, side_sign):
    """Fraction of the loop's points on the ipsilateral side.

    A fraction, not a centroid test, so a contour spanning the midline is not
    mislabelled by where its centre of mass happens to fall.
    """
    return float(((loop["points"][:, 0] - x_mid) * side_sign > 0).mean())


def loop_encloses(loop, point):
    """Even-odd point-in-polygon test in the loop's own 2D basis."""
    u_ax, v_ax = loop["u_axis"], loop["v_axis"]
    pu, pv = float(point[u_ax]), float(point[v_ax])
    xs, ys = loop["points"][:, u_ax], loop["points"][:, v_ax]
    inside = False
    n = len(xs)
    for i in range(n):
        j = (i - 1) % n
        if ((ys[i] > pv) != (ys[j] > pv)) and (
                pu < (xs[j] - xs[i]) * (pv - ys[i]) / (ys[j] - ys[i] + 1e-12)
                + xs[i]):
            inside = not inside
    return inside


def select_loop(loops, spec, spg, frame, params, landmarks=None):
    """Pick one loop, or refuse. Never 'take the largest' unconditionally.

    Returns (loop, record). loop is None when the pick is ambiguous, in which
    case the caller emits candidates instead of a curve: a missing curve is
    safe and visible, a wrong curve in a morphometry dataset is not.

    Anchors come only from things the arm already legitimately reuses, namely
    the SPG centroid and the manual point landmarks. The manual CURVES are
    never consulted, because scoring against them would make the agreement
    analysis circular.
    """
    if not loops:
        return None, {"status": "no_loops"}

    x_mid, sign = frame["x_mid"], frame["side_sign"]
    landmarks = landmarks or {}
    anchor_name = spec.get("anchor", "spg")
    anchor = spg if anchor_name == "spg" else landmarks.get(anchor_name)
    if anchor is None:
        anchor = spg
    anchor = np.asarray(anchor)

    p_win = spec.get("perimeter_mm")
    a_win = spec.get("area_mm2")
    min_side = spec.get("min_side_fraction", 0.0)

    scores, diag = [], []
    for loop in loops:
        d = loop_min_distance(loop, anchor)
        side_frac = loop_side_fraction(loop, x_mid, sign)
        perim, area = loop["perimeter_mm"], abs(loop["signed_area_mm2"])
        info = {"perim": round(perim, 1), "area": round(area, 1),
                "d_anchor": round(d, 2), "side_frac": round(side_frac, 2)}

        if spec["score"] == "largest_enclosing":
            # The outer body surface: the one loop that actually contains the
            # anchor. Unambiguous, and immune to the centroid problem.
            score = perim if loop_encloses(loop, anchor) else 0.0
            info["encloses"] = bool(score > 0)
        elif spec["score"] == "nearest_anchor":
            score = 1.0 / max(d, 0.1)
        elif spec["score"] == "superior_nearest":
            score = (1.0 / max(d, 0.1)
                     if float(loop["points"][:, 2].max()) > spg[2] else 0.0)
        else:
            raise ValueError(f"unknown score mode {spec['score']!r}")

        # Geometric plausibility, applied as hard gates rather than soft
        # penalties so an implausible loop cannot win on proximity alone.
        if p_win and not (p_win[0] <= perim <= p_win[1]):
            score = 0.0
            info["reject"] = "perimeter"
        if a_win and not (a_win[0] <= area <= a_win[1]):
            score = 0.0
            info["reject"] = "area"
        if side_frac < min_side:
            score = 0.0
            info["reject"] = "side"

        info["score"] = round(score, 5)
        scores.append(score)
        diag.append(info)

    order = list(np.argsort(scores)[::-1])
    best, best_score = int(order[0]), float(scores[order[0]])
    second_score = float(scores[order[1]]) if len(order) > 1 else 0.0
    margin = (best_score / second_score) if second_score > 0 else float("inf")

    record = {
        "status": "ok",
        "n_candidates": len(loops),
        "n_viable": int(sum(1 for s in scores if s > 0)),
        "chosen_index": best,
        "anchor": anchor_name,
        "best_score": best_score,
        "second_score": second_score,
        "margin": margin if margin != float("inf") else None,
        "candidates": [dict(diag[i], index=i) for i in order[:5]],
    }
    if best_score <= 0:
        record["status"] = "no_viable_candidate"
        return None, record
    if margin < params["LOOP_MARGIN_MIN"]:
        record["status"] = "ambiguous"
        record["reason"] = (
            f"best score beats second by {margin:.2f}x, below the required "
            f"{params['LOOP_MARGIN_MIN']}x margin"
        )
        return None, record
    return loops[best], record


def _on_convex_hull(points_2d, tol_mm):
    """Boolean mask of points lying on (or within tol of) the convex hull.

    Isolates an outer surface from a contour that also wraps into internal
    cavities, without reference to any anatomical anchor.
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return np.ones(len(points_2d), dtype=bool)
    if len(points_2d) < 4:
        return np.ones(len(points_2d), dtype=bool)
    try:
        hull = ConvexHull(points_2d)
    except Exception:
        return np.ones(len(points_2d), dtype=bool)
    verts = points_2d[hull.vertices]
    # Distance from every point to the hull boundary polygon.
    best = np.full(len(points_2d), np.inf)
    for i in range(len(verts)):
        a, b = verts[i], verts[(i + 1) % len(verts)]
        ab = b - a
        denom = float(ab @ ab)
        if denom == 0:
            d = np.linalg.norm(points_2d - a, axis=1)
        else:
            t = np.clip((points_2d - a) @ ab / denom, 0.0, 1.0)
            d = np.linalg.norm(points_2d - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best <= float(tol_mm)


def trim_loop(loop, spec, spg, frame, params):
    """Reduce a full contour to the anatomically relevant arc.

    Always via longest_contiguous_run, so a direction predicate cannot return
    two disjoint arcs on opposite walls of the nasal cavity.
    """
    mode, argument = spec["trim"] if spec["trim"] else (None, None)
    points = loop["points"]
    if mode is None:
        return points, {"mode": None, "kept": len(points), "of": len(points)}

    u_ax, v_ax = loop["u_axis"], loop["v_axis"]
    p2 = np.column_stack([points[:, u_ax], points[:, v_ax]])
    spg2 = np.array([spg[u_ax], spg[v_ax]])
    side_sign = frame["side_sign"]
    seed = int(np.argmin(np.linalg.norm(p2 - spg2, axis=1)))

    if mode == "normal":
        direction = _direction_2d(loop, argument, side_sign)
        normals = loop_normals_2d(loop)
        mask = (normals @ direction) > params["TRIM_NORMAL_MIN"]
    elif mode in ("angular", "angular_outer"):
        direction = _direction_2d(loop, argument, side_sign)
        vectors = p2 - spg2
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cosines = (vectors / norms) @ direction
        mask = cosines > np.cos(
            np.deg2rad(params["CHEEK_ANGULAR_HALFWIDTH_DEG"]))
        if mode == "angular_outer":
            # Restrict to the OUTER surface, via the convex hull.
            #
            # At the SPG axial level the nasal airway is open posteriorly
            # through the choanae, so it is not an enclosed hole and the body
            # outline dives into it. The resulting loop therefore contains both
            # facial skin and internal airway wall, and an angular window alone
            # will happily keep the internal part, which is how the cheek curve
            # came out 1.08 mm from the SPG.
            #
            # The hull is used deliberately instead of a distance-from-SPG
            # cutoff: the measurement this curve feeds IS the distance from the
            # SPG, so trimming by that distance would make the answer a
            # function of the trim threshold rather than of anatomy.
            mask = mask & _on_convex_hull(p2, params["HULL_TOL_MM"])
    elif mode == "hemispace":
        mask = (points[:, 0] - frame["x_mid"]) * side_sign > 0
    elif mode == "septum_band":
        mask = np.abs(points[:, 0] - frame["x_mid"]) < params["SEPTUM_BAND_MM"]
    else:
        raise ValueError(f"unknown trim mode {mode!r}")

    keep = longest_contiguous_run(mask, seed, closed=loop.get("closed", True))
    trimmed = points[keep]
    return trimmed, {"mode": mode, "argument": argument,
                     "kept": int(keep.sum()), "of": len(points),
                     "seed_index": seed}


def stage_curves(segmentation_node, segments, spg, frame, landmarks,
                 params, run_id, cfg_hash, side, check):
    """Cut every specified curve, or refuse and emit candidates."""
    if frame.get("status") != "ok":
        return {"status": "unavailable", "reason": "frame stage did not run"}

    record = {"status": "ok", "curves": {}, "refused": {}, "node_names": {}}
    surfaces = {}

    for spec in CURVE_SPECS:
        key = spec["key"]
        plane = frame["planes"].get(spec["plane"])
        if plane is None:
            record["refused"][key] = {"status": "no_plane",
                                      "plane": spec["plane"]}
            continue
        segment_id = segments.get(spec["source"])
        if segment_id is None:
            record["refused"][key] = {"status": "no_source_segment",
                                      "source": spec["source"]}
            continue

        if spec["source"] not in surfaces:
            raw = segment_surface(segmentation_node, segment_id)
            surfaces[spec["source"]] = smooth_surface(
                raw, params["SURFACE_SMOOTH_PASSBAND"],
                params["SURFACE_SMOOTH_ITERATIONS"])
        poly = surfaces[spec["source"]]

        loops, cut_info = cut_loops(
            poly, plane["orientation"], plane["offset"], params)
        if not cut_info["counts_agree"]:
            check(f"stripper_mismatch_{key}", "warn",
                  f"stripper found {cut_info['n_loops_stripper']} loops but "
                  f"connectivity found {cut_info['n_regions_connectivity']} "
                  f"regions; the loop set for {key} is untrustworthy")

        loop, selection = select_loop(loops, spec, spg, frame, params, landmarks)
        if loop is None:
            record["refused"][key] = dict(selection, cut=cut_info)
            check(f"curve_refused_{key}", "warn",
                  f"{key} not emitted: {selection.get('reason', selection['status'])}. "
                  f"Candidates left in the scene for inspection.")
            for i, candidate in enumerate(loops[:4]):
                name = f"{AUTO_PREFIX}cand_{key}_{side}_{i}"
                node = create_curve_node(
                    name, candidate["points"], run_id, cfg_hash, params,
                    requires_confirmation=True,
                    extra={"qcStatus": "candidate", "curveKey": key})
                node.SetAttribute("SPGAutoSeg.qcStatus", "candidate")
            continue

        trimmed, trim_info = trim_loop(loop, spec, spg, frame, params)
        if len(trimmed) < 2:
            record["refused"][key] = {"status": "trim_empty", "trim": trim_info}
            check(f"curve_trim_empty_{key}", "warn",
                  f"{key} trimmed to {len(trimmed)} points, not emitted")
            continue

        resampled = resample_polyline(
            trimmed, params["CURVE_CONTROL_SPACING_MM"],
            params["CURVE_MIN_CONTROL_POINTS"],
            params["CURVE_MAX_CONTROL_POINTS"])

        confirm = spec["confirm"]
        metal_range = None
        name = f"{AUTO_PREFIX}{key}_{side}"
        create_curve_node(name, resampled, run_id, cfg_hash, params,
                          requires_confirmation=confirm,
                          extra={"curveKey": key,
                                 "planeOffsetRAS": plane["offset"],
                                 "sourceSegment": spec["source"],
                                 "trimMode": str(trim_info.get("mode"))})
        record["curves"][key] = {
            "node": name, "selection": selection, "trim": trim_info,
            "cut": cut_info, "n_control_points": len(resampled),
            "plane_offset_ras": plane["offset"],
            "requires_confirmation": confirm,
        }
        record["node_names"][spec["role"]] = name

    # The manual engine takes argmin of z on the operator's orbital floor curve
    # (line 679), which reads as the INFERIOR cortex, while anchoring on
    # orbital fat would give the SUPERIOR cortex of the maxillary roof. They
    # differ by the bone thickness, 0.5 to 1.5 mm. Flagged rather than guessed.
    if "orbital_floor_cor" in record["curves"]:
        check("protocol_ambiguity_orbital_floor", "warn",
              "orbital floor surface definition is ambiguous: the manual "
              "engine's argmin-of-z reads as the inferior cortex, an "
              "orbital-fat anchor would give the superior cortex, and they "
              "differ by 0.5 to 1.5 mm. Needs a decision from Dr. Iloreta "
              "before either version enters the dataset.")

    record["n_emitted"] = len(record["curves"])
    record["n_refused"] = len(record["refused"])
    return record


############ STAGE 4: BONE SEPARATION ############################
# Honest statement of the limit, before any code.
#
# The palatine bone, the pterygoid plates and the sphenoid are CONTINUOUS BONE
# at this resolution. The sphenopalatine suture and the palatine-sphenoid
# articulation are not resolvable as an HU gap in adult CT, and the palatine's
# orbital and sphenoidal processes interdigitate with the sphenoid body.
# Connected-component labelling on the bone class returns ONE component
# containing all three plus the maxilla and vomer.
#
# So components cannot solve this and are not presented as if they could. What
# follows is a geodesic watershed on the bone class, seeded from landmark
# anchored priors. It is deterministic and reproducible. It is NOT anatomy: the
# boundaries it draws are a function of seed placement. This stage therefore
# never returns status "ok", and no measurement may depend on where it says one
# bone ends and the next begins.


def _dist_to_segment(points, a, b):
    """Perpendicular distance from each point to the segment ab."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq == 0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip((points - a) @ ab / length_sq, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _ras_grid(volume_node, shape):
    """RAS coordinate of every voxel centre, as (K,J,I,3)."""
    import vtk
    matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix)
    m = np.array([[matrix.GetElement(r, c) for c in range(4)]
                  for r in range(4)])
    k, j, i = np.indices(shape)
    ijk1 = np.stack([i, j, k, np.ones_like(i)], axis=-1).astype(float)
    return ijk1 @ m.T[:, :3]


def stage_bones(volume_node, spg, frame, nodes, node_names, params,
                run_id, cfg_hash, side, check):
    """Seeded separation of palatine, pterygoid and sphenoid. Never 'ok'."""
    import scipy.ndimage as ndi

    if frame.get("status") != "ok" or spg is None:
        return {"status": "unavailable",
                "reason": "needs the frame stage and an SPG centroid"}

    array = slicer.util.arrayFromVolume(volume_node)
    spacing = volume_node.GetSpacing()
    voxel_cm3 = float(np.prod(spacing)) / 1000.0
    bone = (array >= params["BONE_HU_MIN"]) & (array <= params["BONE_HU_MAX"])
    dense = (array >= params["BONE_HU_DENSE"]) & bone
    ras = _ras_grid(volume_node, array.shape)
    flat_ras = ras.reshape(-1, 3)

    x_mid = frame["x_mid"]
    sign = frame["side_sign"]
    record = {"status": "requires_confirmation", "seeds": {}, "structures": {}}

    # Components are used only for the two things they are legitimately good
    # for: reporting whether the bone mass is one piece (it will be), and
    # stripping unattached islands.
    _, n_components = ndi.label(bone)
    record["bone_components"] = int(n_components)
    record["bone_is_single_mass"] = bool(n_components <= 2)

    seeds = {}

    # Pterygoid: anchored on the operator's vidian canal line, which already
    # runs through the pterygoid base. Anchored on real user input, so the most
    # reliable of the three.
    vidian = nodes.get("vidian_ax")
    if vidian is not None:
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        vidian.GetNthControlPointPositionWorld(0, p0)
        vidian.GetNthControlPointPositionWorld(1, p1)
        direction = np.array(p1) - np.array(p0)
        norm = np.linalg.norm(direction)
        if norm > 0:
            extended = np.array(p1) + (direction / norm) * params[
                "PTERYGOID_EXTEND_MM"]
            distances = _dist_to_segment(flat_ras, p0, extended).reshape(
                array.shape)
            seeds["pterygoid"] = dense & (
                distances <= params["PTERYGOID_SEED_R_MM"])
            record["seeds"]["pterygoid"] = {"mode": "vidian_line_anchored",
                                            "reliability": "highest"}

    # Sphenoid: anchored on the sphenoid sinus air component, the posterior
    # most enclosed-air component straddling the midline. Far better than a
    # superior-posterior box, when it is identifiable.
    air = (array >= params["AIR_HU_MIN"]) & (array <= params["AIR_HU_MAX"])
    labels, n_air = ndi.label(air)
    sphenoid_air = None
    if n_air:
        best_y = None
        for index in range(1, n_air + 1):
            mask = labels == index
            if int(mask.sum()) < 200:
                continue
            xs = ras[..., 0][mask]
            if not (xs.min() < x_mid < xs.max()):
                continue          # must straddle the midline
            centre_y = float(ras[..., 1][mask].mean())
            if best_y is None or centre_y < best_y:   # posterior = lower A
                best_y, sphenoid_air = centre_y, mask
    if sphenoid_air is not None:
        shell = ndi.binary_dilation(
            sphenoid_air,
            iterations=max(1, int(round(params["SPHENOID_SHELL_MM"]
                                        / min(spacing)))))
        seeds["sphenoid"] = dense & shell & ~sphenoid_air
        record["seeds"]["sphenoid"] = {"mode": "sphenoid_sinus_anchored",
                                       "reliability": "good"}
    else:
        box = (ras[..., 2] > spg[2] + params["SPHENOID_SUP_MM"]) & (
            ras[..., 1] < spg[1] + 6.0)
        seeds["sphenoid"] = dense & box
        record["seeds"]["sphenoid"] = {"mode": "fallback_box",
                                       "reliability": "low"}
        check("sphenoid_seed_fallback", "warn",
              "no sphenoid sinus air component identified (poorly "
              "pneumatised or opacified); fell back to a geometric box, so "
              "the sphenoid label is materially less trustworthy")

    # Palatine: no air landmark exists, so this is a box in the SPG frame. The
    # WEAKEST of the three, and labelled as such.
    lo_x, hi_x = sorted([x_mid + sign * 2.0, float(spg[0]) - sign * 1.0])
    palatine_box = (
        (ras[..., 0] > lo_x) & (ras[..., 0] < hi_x)
        & (ras[..., 1] > spg[1] - 4.0) & (ras[..., 1] < spg[1] + 12.0)
        & (ras[..., 2] > spg[2] - 20.0) & (ras[..., 2] < spg[2] + 4.0))
    seeds["palatine"] = dense & palatine_box
    record["seeds"]["palatine"] = {"mode": "spg_frame_box",
                                   "reliability": "weakest"}

    # Background bone. NON-OPTIONAL. Without it, grow-from-seeds is forced to
    # assign every bone voxel to one of the three targets and the palatine seed
    # swallows the entire maxilla. This is the single most likely way to get a
    # result that looks like a segmentation and is nonsense.
    far = (
        (ras[..., 1] > spg[1] + 20.0)                       # maxilla, anterior
        | (np.abs(ras[..., 0] - x_mid) > 32.0)              # zygoma, lateral
        | (ras[..., 2] > spg[2] + 25.0)                     # ethmoid, superior
        | (ras[..., 2] < spg[2] - 28.0))                    # palate, inferior
    seeds["other"] = dense & far
    record["seeds"]["other"] = {"mode": "background", "reliability": "n/a"}

    for name, mask in seeds.items():
        record["seeds"][name]["voxels"] = int(mask.sum())
    empty = [n for n, m in seeds.items() if not m.any()]
    if empty:
        check("bone_seeds_empty", "warn",
              f"no dense-bone voxels found for seed(s) {empty}; those "
              f"structures cannot be grown")
        seeds = {n: m for n, m in seeds.items() if m.any()}
    if "other" not in seeds or len(seeds) < 2:
        return dict(record, status="failed",
                    reason="need a background seed plus at least one target")

    bones_node = ensure_segmentation(
        f"{AUTO_PREFIX}SEG_BONES", volume_node, run_id, cfg_hash)
    segment_ids = {}
    for name, mask in seeds.items():
        segment_ids[name] = add_segment_from_mask(
            bones_node, volume_node, f"{AUTO_PREFIX}bone_{name}_seed", mask)

    with segment_editor_session(bones_node, volume_node) as (widget, editor):
        _set_intensity_mask(editor, True, params["BONE_HU_MIN"],
                            params["BONE_HU_MAX"])
        editor.SetOverwriteMode(
            slicer.vtkMRMLSegmentEditorNode.OverwriteAllSegments)
        widget.setCurrentSegmentID(segment_ids["other"])
        widget.setActiveEffectByName("Grow from seeds")
        effect = widget.activeEffect()
        if effect is None:
            record["status"] = "failed"
            record["reason"] = "Grow from seeds effect unavailable"
            return record
        effect.self().onPreview()
        slicer.app.processEvents()      # the preview must finish before apply
        effect.self().onApply()
        widget.setActiveEffectByName("")
        _set_intensity_mask(editor, False)

        for name, segment_id in segment_ids.items():
            if name == "other":
                continue
            widget.setCurrentSegmentID(segment_id)
            apply_effect(widget, "Islands",
                         {"Operation": "KEEP_LARGEST_ISLAND"})

    # Report geometry per structure, in the SPG frame.
    #
    # Read each segment in the REFERENCE VOLUME's geometry. Exporting to a
    # labelmap node instead would crop the result to the segmentation's own
    # bounding box, which no longer aligns with the volume grid and silently
    # breaks every RAS lookup below.
    for name, segment_id in segment_ids.items():
        mask = None
        try:
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(
                bones_node, segment_id, volume_node)
        except TypeError:
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(
                bones_node, segment_id)
        except Exception as exc:
            record["structures"][name] = {"error": str(exc)}
            continue
        if mask is None:
            record["structures"][name] = {"voxels": 0}
            continue
        mask = np.asarray(mask).astype(bool)
        if mask.shape != array.shape:
            record["structures"][name] = {
                "voxels": int(mask.sum()),
                "note": f"segment geometry {mask.shape} does not match the "
                        f"volume grid {array.shape}; positional statistics "
                        f"skipped rather than computed against the wrong grid",
            }
            continue
        if not mask.any():
            record["structures"][name] = {"voxels": 0}
            continue
        coords = ras[mask]
        centroid = coords.mean(axis=0)
        record["structures"][name] = {
            "voxels": int(mask.sum()),
            "volume_cm3": float(mask.sum() * voxel_cm3),
            "centroid_ras": [float(v) for v in centroid],
            "centroid_in_spg_frame_mm": [
                float((centroid[0] - spg[0]) * sign),
                float(centroid[1] - spg[1]),
                float(centroid[2] - spg[2])],
            "bbox_ras": [float(coords[:, a].min()) for a in range(3)]
                        + [float(coords[:, a].max()) for a in range(3)],
            "requires_confirmation": True,
        }

    # Two structural checks are safe to hard-code because they are logical,
    # not empirical. Volume plausibility is deliberately NOT hard-coded against
    # remembered literature values; it is flagged against the cohort's own
    # running distribution once n >= 5, in the agreement analysis.
    structures = record["structures"]

    def centroid(name):
        entry = structures.get(name) or {}
        return entry.get("centroid_ras")

    palatine, pterygoid, sphenoid = (centroid("palatine"),
                                     centroid("pterygoid"),
                                     centroid("sphenoid"))
    if palatine and pterygoid and pterygoid[1] >= palatine[1]:
        check("bone_topology_pterygoid", "error",
              f"pterygoid centroid (A={pterygoid[1]:.1f}) is not posterior to "
              f"palatine (A={palatine[1]:.1f}); the seeds are wrong")
    if palatine and sphenoid and sphenoid[2] <= palatine[2]:
        check("bone_topology_sphenoid", "error",
              f"sphenoid centroid (S={sphenoid[2]:.1f}) is not superior to "
              f"palatine (S={palatine[2]:.1f}); the seeds are wrong")

    for name, segment_id in segment_ids.items():
        segment = bones_node.GetSegmentation().GetSegment(segment_id)
        if segment is not None:
            segment.SetName(f"{AUTO_PREFIX}bone_{name}_{side}")
    tag_auto_node(bones_node, run_id, cfg_hash, requires_confirmation=True)

    check("bone_separation_requires_confirmation", "warn",
          "bone separation boundaries are watershed boundaries in the bone "
          "mask, not sutures. Reproducible, but set by seed placement. Usable "
          "for figures and contours; no measurement may depend on where one "
          "bone is said to end.")
    record["segmentation_node"] = bones_node.GetName()
    return record


############ STAGE 5: QC ############################


def stage_qc(volume_node, segmentation_node, segments, spg, frame, curves,
             params, check, landmarks=None):
    """Remaining failure detectors, plus the screenshot capture plan.

    Screenshots are emitted as a plan rather than taken here: capture is an MCP
    tool, not an in-Slicer call. The agent executes the plan.
    """
    import scipy.ndimage as ndi

    record = {"status": "ok", "checks": [], "capture_plan": []}

    # Airway leak detection. The good test is the bridge test: erode by one
    # voxel and re-label. If a single component becomes several, the pieces
    # were joined by a one-voxel bridge, which is either a leak through a
    # dehiscent wall or a real ostium. Either way the operator should look.
    if volume_node is not None:
        array = slicer.util.arrayFromVolume(volume_node)
        air = (array >= params["AIR_HU_MIN"]) & (array <= params["AIR_HU_MAX"])
        labels, n = ndi.label(air)
        if n:
            sizes = ndi.sum(air, labels, range(1, n + 1))
            largest = labels == (int(np.argmax(sizes)) + 1)
            eroded = ndi.binary_erosion(largest, iterations=1)
            sub_labels, n_sub = ndi.label(eroded)
            if n_sub > 1:
                sub_sizes = ndi.sum(eroded, sub_labels, range(1, n_sub + 1))
                total = float(sub_sizes.sum())
                significant = [s for s in sub_sizes if s / max(total, 1) > 0.05]
                if len(significant) > 1:
                    record["checks"].append({"id": "airway_bridge",
                                             "pieces": len(significant)})
                    check("airway_bridge", "warn",
                          f"eroding the airway by one voxel splits it into "
                          f"{len(significant)} significant pieces, so they "
                          f"were joined by a one-voxel bridge: either a leak "
                          f"through a thin wall or a real ostium")

    emitted = (curves or {}).get("curves", {})

    # Slice-neighbourhood stability. The best automatic wrong-loop detector:
    # recut the same structure 1 mm either side and see whether the arc moves.
    if emitted and segmentation_node is not None:
        stability = {}
        for key, info in emitted.items():
            spec = next((s for s in CURVE_SPECS if s["key"] == key), None)
            segment_id = segments.get(spec["source"]) if spec else None
            if spec is None or segment_id is None:
                continue
            plane = frame["planes"].get(spec["plane"])
            try:
                poly = smooth_surface(
                    segment_surface(segmentation_node, segment_id),
                    params["SURFACE_SMOOTH_PASSBAND"],
                    params["SURFACE_SMOOTH_ITERATIONS"])
                primary, _ = cut_loops(poly, plane["orientation"],
                                       plane["offset"], params)
                chosen, _ = select_loop(primary, spec, spg, frame, params, landmarks)
                if chosen is None:
                    continue
                drifts = []
                for delta in (-params["STABILITY_DZ_MM"],
                              params["STABILITY_DZ_MM"]):
                    neighbour, _ = cut_loops(
                        poly, plane["orientation"], plane["offset"] + delta,
                        params)
                    other, _ = select_loop(neighbour, spec, spg, frame, params, landmarks)
                    if other is None:
                        continue
                    deltas = (chosen["points"][:, None, :]
                              - other["points"][None, :, :])
                    drifts.append(float(np.linalg.norm(deltas, axis=2)
                                        .min(axis=1).mean()))
                if drifts:
                    worst = max(drifts)
                    stability[key] = worst
                    if worst > params["STABILITY_MAX_MM"]:
                        check(f"curve_unstable_{key}", "warn",
                              f"{key} moves {worst:.2f} mm when the cut plane "
                              f"shifts by {params['STABILITY_DZ_MM']} mm, above "
                              f"the {params['STABILITY_MAX_MM']} mm limit; "
                              f"usually a wrong loop or a bifurcating structure")
            except Exception as exc:      # never let QC break the run
                record["checks"].append({"id": f"stability_error_{key}",
                                         "error": str(exc)})
        record["slice_stability_mm"] = stability

    # Cross-curve consistency. HARD ERRORS: a violation means two curves cannot
    # both be right, so neither should be trusted.
    def centroid_of(key):
        info = emitted.get(key)
        if not info:
            return None
        node = get_node_strict(info["node"])
        if node is None:
            return None
        points = np.array([node.GetNthControlPointPosition(i)
                           for i in range(node.GetNumberOfControlPoints())])
        return points.mean(axis=0) if len(points) else None

    septum, mt_ax, cheek_ax = (centroid_of("septum_ax"),
                               centroid_of("mt_medial_ax"),
                               centroid_of("cheek_ax"))
    sign = frame.get("side_sign", 1.0)
    if septum is not None and mt_ax is not None:
        if (mt_ax[0] - septum[0]) * sign <= 0:
            check("crosscurve_mt_vs_septum", "error",
                  "the middle turbinate curve is not lateral to the septum "
                  "curve; both cannot be right")
    if cheek_ax is not None:
        for key in ("mt_medial_ax", "nasal_cavity_ax", "septum_ax"):
            other = centroid_of(key)
            if other is not None and (cheek_ax[0] - other[0]) * sign <= 0:
                check("crosscurve_cheek_lateral", "error",
                      f"the cheek curve is not lateral to {key}; both cannot "
                      f"be right")
                break

    # Capture plan. The decisive shot for the bone split is the 2D overlay, not
    # the 3D render: from outside, the 3D view shows a plausible bone surface
    # in three colours, while the 2D overlay shows the colour boundary running
    # straight across continuous uninterrupted cortical bone, which is the fact
    # the operator actually needs in order to judge the labelling honestly.
    planes = frame.get("planes", {})

    def shot(name, view, orientation=None, offset=None, note=""):
        record["capture_plan"].append({
            "name": name, "view_type": view,
            "slice_orientation": orientation, "slice_offset": offset,
            "image_size": list(params["QC_IMAGE_SIZE"]), "note": note})

    shot("tissue_overview", "application", note="air and bone at 50% fill")
    if "axial_spg" in planes:
        shot("axial_curves", "slice", "axial", planes["axial_spg"]["offset"],
             "nasal cavity, septum, middle turbinate medial")
        shot("bone_split_axial", "slice", "axial",
             planes["axial_spg"]["offset"],
             "DECISIVE: colour boundary across continuous cortical bone")
    if "coronal_spf_inf" in planes:
        shot("coronal_turbinates", "slice", "coronal",
             planes["coronal_spf_inf"]["offset"],
             "least reliable curves, confirm each")
        shot("bone_split_coronal", "slice", "coronal",
             planes["coronal_spf_inf"]["offset"], "DECISIVE, as above")
    if "coronal_ppf" in planes:
        shot("orbital_floor", "slice", "coronal",
             planes["coronal_ppf"]["offset"],
             "both surface-definition variants, different colours")
    shot("bone_split_3d", "3d", note="context only, NOT the decisive view")

    # Anything refused gets its own shot so the ambiguity is visible.
    for key in (curves or {}).get("refused", {}):
        spec = next((s for s in CURVE_SPECS if s["key"] == key), None)
        plane = planes.get(spec["plane"]) if spec else None
        if plane:
            shot(f"candidates_{key}", "slice", plane["orientation"],
                 plane["offset"], f"AUTO_cand_{key}_* nodes, resolve or reject")

    record["n_capture_shots"] = len(record["capture_plan"])
    return record


############ ORCHESTRATOR ############################


def pick_volume(volume_node_name=None):
    """Resolve the working volume. Prefers an explicit name, then the
    cropped/active volume, then the scalar volume with the most slices."""
    if volume_node_name:
        node = get_node_strict(volume_node_name, "volume")
        if node is None:
            raise RuntimeError(f"volume node {volume_node_name!r} not found")
        return node
    selection = slicer.app.applicationLogic().GetSelectionNode()
    active_id = selection.GetActiveVolumeID() if selection else None
    if active_id:
        node = slicer.mrmlScene.GetNodeByID(active_id)
        if node is not None:
            return node
    candidates = [
        n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if "DERIVED" not in (n.GetName() or "").upper()
        and n.GetImageData() is not None
    ]
    if not candidates:
        raise RuntimeError("no suitable scalar volume node found in the scene")
    return max(candidates,
               key=lambda n: n.GetImageData().GetDimensions()[2])


def run(volume_node_name=None,
        node_names=None,
        landmark_indices=None,
        subject_id=None,
        side="R",
        stages=("engine",),
        param_overrides=None,
        rerun_mode=None,
        replay_manual=False,
        write_json_dir=None,
        allow_hash_mismatch=False):
    """Run the requested stages and return a single merged report dict.

    Stages are separately invocable so no one execute_python_code call runs
    long enough to hit the Slicer Web Server timeout. Nodes persist in the
    scene between calls, so stages compose across invocations.

    replay_manual=True is the V0b parity test: pass the MANUAL node names and
    landmark indices, and every one of the 20 values must match the manual
    engine's printed output to 2 decimal places.
    """
    bad_stages = [s for s in stages if s not in VALID_STAGES]
    if bad_stages:
        raise ValueError(
            f"unknown stage(s) {bad_stages}; valid stages are {VALID_STAGES}"
        )

    params = collect_params(param_overrides)
    cfg_hash = config_hash(params)
    # A sweep must never clobber the baseline run's nodes.
    mode = rerun_mode or ("version" if param_overrides else params["RERUN_MODE"])
    run_id = f"{subject_id or 'nosubject'}_{side}_{cfg_hash}"

    report = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": (_sha256_file(_this_file())
                          if os.path.exists(_this_file()) else None),
        "run_id": run_id,
        "config_hash": cfg_hash,
        "params": params,
        "param_overrides": param_overrides or {},
        "rerun_mode": mode,
        "subject_id": subject_id,
        "side": side,
        "stages_requested": list(stages),
        "replay_manual": bool(replay_manual),
        "arm": "manual_replay" if replay_manual else "auto",
        "environment": environment_record(),
        "checks": [],
        "status": "ok",
    }

    def check(check_id, severity, detail_text):
        report["checks"].append(
            {"id": check_id, "severity": severity, "detail": detail_text})
        if severity == "error":
            report["status"] = "failed"
        elif severity == "warn" and report["status"] == "ok":
            report["status"] = "ok_with_flags"

    priors = load_priors()
    report["literature_priors"] = {
        "loaded": priors.get("loaded", False),
        "path": priors.get("path"),
        "sha256": priors.get("sha256"),
        "n_priors": len(priors.get("priors", [])),
        "usage": "advisory flagging only, never rejection",
    }

    report["namespace"] = assert_namespace_clean()
    manual_ids_before = _preservable_node_ids()

    volume = None
    try:
        volume = pick_volume(volume_node_name)
    except RuntimeError as exc:
        check("volume_missing", "warn", str(exc))
    report["volume"] = volume_record(volume)

    if "engine" in stages or replay_manual:
        engine, engine_record = load_engine(
            allow_hash_mismatch=allow_hash_mismatch)
        report["engine"] = engine_record
        if not engine_record["sha256_matches_pin"]:
            check("engine_hash_mismatch", "error",
                  "manual engine hash does not match the pin")
    else:
        engine = None

    # Replay mode clears too. The manual engine drops visualisation fiducials
    # as a side effect of find_closest_point, so successive replays would
    # otherwise accumulate identically named copies. Clearing is scoped by
    # attribute, so a manual node can never be caught by it.
    if mode != "version":
        report["cleared"] = clear_previous_auto_nodes(mode)

    # Inputs the auto arm cannot derive and therefore reuses from the manual
    # scene: the 3 SPG lines, the landmark point list, and the 4 line
    # measurements. This is exactly why the arm validates curve TRACING rather
    # than measurement independence.
    nodes, spg, landmarks = {}, None, {}
    if node_names:
        nodes, missing = resolve_nodes(node_names)
        report["nodes_resolved"] = sorted(nodes)
        report["nodes_missing"] = missing
        for entry in missing:
            check("node_missing", "warn",
                  f"node {entry['name']!r} for role {entry['role']!r} not found")

        if engine is not None:
            spg, spg_record = compute_spg_centroid(engine, nodes, node_names)
            report["spg"] = spg_record
            if spg_record.get("flag"):
                check("spg_residual", "warn", spg_record["flag"])
        landmarks, landmark_record = load_landmarks(nodes, landmark_indices)
        report["landmarks"] = landmark_record

        # Advisory only. Catches misplaced input lines, which a small
        # least-squares residual does not rule out.
        spg_check = check_spg_against_spf(spg, landmarks, priors, check)
        if spg_check:
            report["spg_vs_spf_prior"] = spg_check

    completed = []
    segmentation_node, segments = None, {}
    auto_curve_names = {}

    # Segmentation stages are skipped entirely in replay mode: the parity test
    # must exercise the measurement path only.
    if not replay_manual:
        if "tissue" in stages:
            if volume is None:
                check("tissue_no_volume", "error",
                      "tissue stage needs a volume node")
            else:
                tissue, segmentation_node = stage_tissue(
                    volume, params, run_id, cfg_hash, check)
                report["tissue"] = tissue
                segments = tissue["segments"]
                completed.append("tissue")

        if "frame" in stages:
            if segmentation_node is None:
                segmentation_node = get_node_strict(f"{AUTO_PREFIX}SEG")
            frame = stage_frame(volume, segmentation_node, spg, landmarks,
                                params, check)
            report["frame"] = frame
            if frame.get("status") == "ok":
                completed.append("frame")
                if frame["side_inferred"] != side:
                    # A left/right swap is silent and catastrophic, so it is an
                    # error rather than a note.
                    check("side_mismatch", "error",
                          f"side argument is {side!r} but the SPG centroid sits "
                          f"on the {frame['side_inferred']!r} of the midline "
                          f"estimate at x={frame['x_mid']:.2f}")

        if "curves" in stages:
            frame = report.get("frame", {})
            if not segments:
                segments = report.get("tissue", {}).get("segments", {})
            if segmentation_node is None:
                segmentation_node = get_node_strict(f"{AUTO_PREFIX}SEG")
            if frame.get("status") == "ok" and segments and spg is not None:
                curves = stage_curves(
                    segmentation_node, segments, spg, frame, landmarks,
                    params, run_id, cfg_hash, side, check)
                report["curves"] = curves
                auto_curve_names = curves.get("node_names", {})
                completed.append("curves")
            else:
                check("curves_prerequisites", "warn",
                      "curves stage needs the tissue and frame stages plus an "
                      "SPG centroid; skipped")

        if "bones" in stages:
            frame = report.get("frame", {})
            if frame.get("status") == "ok" and spg is not None:
                bones = stage_bones(volume, spg, frame, nodes, node_names,
                                    params, run_id, cfg_hash, side, check)
                report["bones"] = bones
                if bones.get("status") in ("requires_confirmation", "failed"):
                    completed.append("bones")
            else:
                check("bones_prerequisites", "warn",
                      "bones stage needs the frame stage and an SPG centroid; "
                      "skipped")

        if "qc" in stages:
            frame = report.get("frame", {})
            if not segments:
                segments = report.get("tissue", {}).get("segments", {})
            if segmentation_node is None:
                segmentation_node = get_node_strict(f"{AUTO_PREFIX}SEG")
            if frame.get("status") == "ok":
                report["qc"] = stage_qc(
                    volume, segmentation_node, segments, spg, frame,
                    report.get("curves"), params, check, landmarks)
                completed.append("qc")
            else:
                check("qc_prerequisites", "warn",
                      "qc stage needs the frame stage; skipped")

    report["stages_completed"] = completed

    # Measurements. In replay mode the supplied MANUAL names are used verbatim,
    # which is what makes this the parity test. Otherwise the curve roles are
    # overridden with the auto nodes just generated, while lines and landmarks
    # stay manual.
    if engine is not None and node_names:
        measurement_names = dict(node_names)
        if not replay_manual and auto_curve_names:
            measurement_names.update(auto_curve_names)
        report["measurement_node_names"] = measurement_names

        # Record, per curve role, whether an AUTO curve was actually used or
        # the run silently fell back to the manual node because the auto curve
        # was refused. Without this, a refused curve produces a 0.00 mm
        # difference that reads as perfect agreement when it is really the
        # manual curve being compared against itself.
        if not replay_manual:
            curve_roles = {s["role"] for s in CURVE_SPECS}
            report["curve_source"] = {
                role: ("auto" if role in auto_curve_names else "manual_fallback")
                for role in sorted(curve_roles)
            }
            fell_back = [r for r, v in report["curve_source"].items()
                         if v == "manual_fallback"]
            if fell_back:
                check("curve_manual_fallback", "warn",
                      f"{len(fell_back)} curve role(s) fell back to the manual "
                      f"node because no auto curve was emitted: {fell_back}. "
                      f"Any measurement depending on these is NOT an auto "
                      f"versus manual comparison and must be excluded from the "
                      f"agreement analysis.")

        measurement_nodes, _ = resolve_nodes(measurement_names)
        results, detail, side_effects = compute_measurements(
            engine, measurement_nodes, measurement_names, spg, landmarks,
            run_id, cfg_hash, tag="replay" if replay_manual else side)
        report["results"] = results
        report["results_detail"] = detail
        report["engine_side_effect_nodes"] = side_effects
        report["results_slugs"] = {
            RESULT_KEY_SLUGS[k]: v for k, v in results.items()}
        computed = [k for k, v in results.items() if v != SKIPPED]
        report["n_computed"] = len(computed)
        report["n_skipped"] = len(results) - len(computed)
        report["curve_dependent_computed"] = sorted(
            set(computed) & CURVE_DEPENDENT_KEYS)
        report["not_assessable_by_this_arm"] = sorted(
            set(RESULT_KEY_ORDER) - CURVE_DEPENDENT_KEYS)
        report["prior_outliers"] = check_measurements_against_priors(
            results, priors, check)

        # Edge refinement, reported alongside. Never writes into results.
        if volume is not None:
            report["edge_refinement"] = refine_measurements(
                measurement_nodes, measurement_names, volume, landmarks,
                params, engine)
            for role, item in (report["edge_refinement"].get("items") or {}).items():
                if item.get("status") == "ok" and item.get("delta_mm") is not None:
                    if abs(item["delta_mm"]) > 1.0:
                        check(f"refinement_shift_{role}", "warn",
                              f"{role}: raw {item.get('raw_mm')} mm vs refined "
                              f"{item.get('refined_mm')} mm, a "
                              f"{item['delta_mm']:+.2f} mm difference. The manual "
                              f"value is unchanged; this is reported for "
                              f"comparison only.")

    # The primary dataset must come out of this untouched: every data-bearing
    # non-auto node present at the start must still be here.
    scene_ids_after = _node_id_set()
    lost = sorted(manual_ids_before - scene_ids_after)
    report["manual_nodes_preserved"] = not lost
    report["manual_nodes_tracked"] = len(manual_ids_before)
    report["auto_nodes_cleared_count"] = len(
        report.get("cleared", {}).get("removed", []))
    if lost:
        check("manual_nodes_lost", "error",
              f"manual node(s) present before the run are now missing: {lost}")

    if write_json_dir:
        os.makedirs(write_json_dir, exist_ok=True)
        path = os.path.join(write_json_dir, f"{run_id}_autoarm_report.json")
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        report["written_to"] = path

    return report


def print_report(report):
    """Console summary. The full record is the returned dict."""
    print("=" * 68)
    print(f" Autoseg QC Arm  v{report['script_version']}  [{report['arm']}]")
    print("=" * 68)
    print(f"  run id:      {report['run_id']}")
    print(f"  config hash: {report['config_hash']}")
    print(f"  status:      {report['status']}")
    if "engine" in report:
        eng = report["engine"]
        print(f"  engine hash: {eng['sha256'][:16]}... "
              f"(pin {'OK' if eng['sha256_matches_pin'] else 'MISMATCH'})")
        print(f"  sandbox:     tripwire fired, results empty, scene unchanged")
    print(f"  manual nodes preserved: {report.get('manual_nodes_preserved')}")
    if "spg" in report and report["spg"].get("spg_ras"):
        spg = report["spg"]
        residual = spg.get("residual_mm")
        print(f"  SPG centroid: "
              f"{[round(v, 3) for v in spg['spg_ras']]}"
              + (f"  residual {residual:.3f} mm" if residual is not None else ""))
    if "results" in report:
        print()
        print(f"  {report['n_computed']}/20 computed, "
              f"{report['n_skipped']} skipped")
        print("  " + "-" * 64)
        for key in RESULT_KEY_ORDER:
            value = report["results"][key]
            marker = "C" if key in CURVE_DEPENDENT_KEYS else " "
            print(f"  {marker} {key}: {value}")
        print("  " + "-" * 64)
        print("  C = curve-dependent, assessable by this QC arm (9 of 20).")
        print("      The other 11 derive from landmarks and lines alone and")
        print("      are NOT assessable by this arm.")
    if report.get("spg_vs_spf_prior"):
        distance = report["spg_vs_spf_prior"]["spg_to_spf_mm"]
        status = "outside" if report["spg_vs_spf_prior"].get("flag") else "within"
        print(f"  SPG to SPF landmark: {distance:.2f} mm "
              f"({status} the published band)")
    if report.get("prior_outliers"):
        print()
        print("  literature advisories (inspect, do NOT exclude):")
        for key, flag in report["prior_outliers"].items():
            print(f"    {key} = {flag['value_mm']:.2f} mm, "
                  f"published band {flag['band_mm']} mm "
                  f"[confidence {flag['confidence']}]")
    if report["checks"]:
        print()
        print("  checks:")
        for entry in report["checks"]:
            print(f"    [{entry['severity']}] {entry['id']}: {entry['detail']}")
    print("=" * 68)


print(f"(C) Autoseg_Measurements_0729.py v{SCRIPT_VERSION} loaded. "
      f"Nothing has run yet.")
print("  Parity test:  report = run(replay_manual=True, node_names={...}, "
      "landmark_indices={...}); print_report(report)")
print(f"  Node roles:   {', '.join(NODE_ROLES)}")
print(f"  Landmarks:    {', '.join(LANDMARK_ROLES)}")
