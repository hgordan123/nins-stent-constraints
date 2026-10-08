"""NINS stent design constraints from a minimal seed set.

Run inside 3D Slicer (Python Interactor or the slicer MCP).

WHAT THIS IS
------------
A focused tool for one question: does a transnasal stent surface get within
5 mm of the sphenopalatine foramen, and what are the stent's dimensions?

It is NOT the repeatability study. It shares no data with the manual arm and
writes nothing into it. It exists to turn a handful of seed clicks into the
device design constraint table.

WHAT THE OPERATOR PLACES
------------------------
Point fiducials, named exactly. Anywhere INSIDE the named structure: these are
seeds, not landmarks, and none of them has to sit on an edge.

  REQUIRED for the headline number
    SPF_R, SPF_L          one point inside the sphenopalatine foramen

  REQUIRED for length and the lumen profile
    piriform_R, piriform_L    inside the nasal airway at the piriform aperture
    choana_R,   choana_L      inside the airway at the posterior choana

  OPTIONAL, safety clearances
    vidian_R, vidian_L        inside the vidian canal
    FR_R,     FR_L            inside foramen rotundum

Only one side need be present. Missing optional seeds are reported as absent,
never guessed.

WHAT IS DERIVED, AND WHAT IS NOT
--------------------------------
Derived with no operator input, and verified on a real scan:
  - the midsagittal plane, by reflective symmetry of the bone mask.
    Reproducible to SD 0.13 mm across 4 bone thresholds and 2 downsample
    factors, with a sharp optimum (Dice falls 0.037 at 1 mm off-plane).
  - the SPF centre and both apertures, by direction-free minimum feret.
  - the lumen cross section at any plane.

NOT derived, established by testing rather than argument:
  - anatomical identification. Seven approaches have now failed, most recently
    deriving midline bony landmarks by argmax on the symmetry plane: it put ANS
    at the chin and PNS on the sphenoid floor, and read the hard palate at
    66.5 mm against a published 45 to 55 mm. "Most anterior midline bony point
    OF THE MAXILLA" needs to know what the maxilla is. That is recognition.
    Geometry automates. Recognition does not. Hence the seeds above.

HOW TO RUN
----------
    exec(open("(C) NINS_Stent_Constraints_0811.py").read())
    report = run()                       # or run(volume_node_name="...")
    print_report(report)

Add write_json_dir="/path" to persist the full record.
"""

import json
import os
import time
import datetime
import hashlib

import numpy as np
import vtk
import slicer
from scipy import ndimage

SCRIPT_VERSION = "0.3.0"

# ---- tunables. Every uppercase global is captured into the provenance record.
AIR_HU_MAX = -350.0          # airway lumen, mucosal surface
BONE_HU_MIN = 300.0          # cortical bone
RIM_PROMINENCE_HU = 120.0    # a rim must rise this far above the lumen baseline
RIM_SEARCH_MM = 8.0          # how far outward to look for a bounding rim
LUMEN_HU_MAX = 200.0         # above this a sample point is not lumen
FERET_N_ANGLES = 90          # 2 degree steps over a half turn
PROFILE_STEP_MM = 1.0        # lumen sampling interval along the corridor
BONY_N_RAYS = 180            # rays for the bony corridor contour
SYMMETRY_DOWNSAMPLE = 4
SYMMETRY_BONE_HU = 300.0
CONDUCTION_BUDGET_MM = 5.0   # the design budget this study tests
# A PRIOR, not a measurement: the nasal airway proper lies within this distance
# of the midline, while the maxillary sinus lies lateral to it. Needed because
# the sinus is genuinely continuous with the cavity through the maxillary
# ostium, and it has a far larger inscribed circle, so an unconstrained
# centreline tracker migrates into it and reports the sinus as the corridor.
NASAL_HALF_WIDTH_MM = 16.0
AXIS_STEP_SEARCH_MM = 3.0    # how far the centreline may move per 1 mm station
# Bounds for deriving the corridor endpoints, relative to the SPF centre.
CORRIDOR_POST_SEARCH_MM = 30.0   # posterior septal edge lies within this of the SPF
CORRIDOR_ANT_SEARCH_MM = 45.0    # anterior maxillary edge likewise

# ---- self-calibration. Absolute HU thresholds hold only for a calibrated
# MDCT on the kernel they were tuned on, and cone beam CT has no reliable HU
# calibration at all. These derive the working thresholds from the volume's own
# air and soft-tissue histogram modes instead, as fractions of their separation.
CALIB_AIR_FRACTION = 0.55       # air ceiling, from the air mode toward soft
CALIB_BONE_FRACTION = 0.30      # bone floor, from the soft mode upward
CALIB_LUMEN_FRACTION = 0.16     # "not lumen" ceiling
# 3 sigma, not 4. At 4 sigma on the reference scan this demanded 139 HU, which
# stepped straight past a thin coronal SPF rim whose prominence sits between
# 120 and 139, and the aperture jumped from 4.47 to 6.17 mm. Thin rims are the
# known failure mode here: the posterior SPF rim partial-volumes down to about
# 227 HU, which is why an absolute threshold was abandoned in the first place.
CALIB_RIM_NOISE_SIGMA = 3.0     # a rim must clear this many noise SDs
APERTURE_SENSITIVITY_TOL_MM = 0.5   # flag if +/-25% prominence moves it by more
CALIB_RIM_MIN_FRACTION = 0.10   # and at least this fraction of the separation
CALIB_MIN_SEPARATION_HU = 500.0 # below this the histogram is not usable
# Nothing real is less dense than air. Many reconstructions pad outside the
# field of view with -2000 or -3024, and that padding is a huge discrete
# population: on the second subject tested it became the "air" mode and shifted
# every class down one, giving a bone floor of -367 HU that called 32% of the
# volume bone, while still reporting status ok.
# Padding is a SPIKE at the extreme low value with a gap above it. Detecting it
# that way is calibration independent; a fixed floor is not, and would strip
# legitimate air from an offset or cone beam scan.
CALIB_PAD_MIN_FRACTION = 0.005   # a spike must hold at least this much
CALIB_PAD_MIN_GAP_HU = 300.0     # and sit this far below the next population
# A head CT always contains bone, and never mostly bone. Both bounds matter:
# the ceiling catches the padding inversion (which calls ~90% of the head bone)
# and the floor catches the opposite inversion, where stripping a delta-shaped
# air population leaves soft tissue reading as air and bone as soft tissue, so
# the derived bone floor sits above everything and finds no bone at all.
CALIB_MIN_BONE_FRACTION = 0.02
CALIB_MAX_BONE_FRACTION = 0.55

# ---- competence gate. The failure mode on an out-of-scope scan is a plausible
# number, not an error, so the tool has to know what it cannot do.
GATE_VOXEL_WARN_MM = 0.6        # SPF aperture is about 3.6 mm across
GATE_VOXEL_REFUSE_MM = 1.0      # below about 3.6 samples across, feret is noise
GATE_METAL_HU = 3000.0          # saturated voxels
GATE_METAL_REFUSE_FRACTION = 0.005  # coarse proxy: streak reaches far beyond the metal
GATE_MIN_BONE_VOXELS = 5000

SEED_ROLES = ("SPF", "piriform", "choana", "vidian", "FR")
REQUIRED_SEEDS = ("SPF",)
CORRIDOR_SEEDS = ("piriform", "choana")


# --------------------------------------------------------------- provenance

def collect_params():
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")
            and isinstance(v, (int, float, str, bool, tuple, list))}


def config_hash(params):
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ------------------------------------------------------------ volume access

def _volume_arrays(volume_node):
    array = slicer.util.arrayFromVolume(volume_node).astype(np.float32)
    mat = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(mat)
    M = np.array([[mat.GetElement(r, c) for c in range(4)] for r in range(4)])
    return array, M, np.linalg.inv(M)


def _sample(array, ras_to_ijk, points):
    """Trilinear sample at RAS points (N,3)."""
    pts = np.atleast_2d(np.asarray(points, float))
    ijk = (np.c_[pts, np.ones(len(pts))] @ ras_to_ijk.T)[:, :3]
    k, j, i = ijk[:, 2], ijk[:, 1], ijk[:, 0]
    k0 = np.floor(k).astype(int); j0 = np.floor(j).astype(int)
    i0 = np.floor(i).astype(int)
    out = np.zeros(len(pts))
    K, J, I = array.shape
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                w = (np.abs(1 - dk - (k - k0)) * np.abs(1 - dj - (j - j0))
                     * np.abs(1 - di - (i - i0)))
                out += w * array[np.clip(k0 + dk, 0, K - 1),
                                 np.clip(j0 + dj, 0, J - 1),
                                 np.clip(i0 + di, 0, I - 1)]
    return out


def _profile(array, ras_to_ijk, origin, direction, length_mm, step_mm=0.1):
    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    t = np.arange(0.0, length_mm + 1e-9, step_mm)
    pts = np.asarray(origin, float)[None, :] + t[:, None] * direction[None, :]
    return t, _sample(array, ras_to_ijk, pts)


def _th(calib, key, default):
    """Calibrated threshold if available, otherwise the module default."""
    v = (calib or {}).get(key)
    return default if v is None else v


def calibrate_intensities(array, sample_stride=3):
    """Derive intensity thresholds from the volume's own histogram.

    Every head CT has a large air population and a large soft-tissue
    population. Locating those two modes fixes the intensity scale for this
    scan, so the thresholds track the scanner, kernel and calibration instead
    of being pinned to the one machine they were tuned on. On the reference
    scan this recovers the hand-tuned values without being told them: air max
    -430.8 against a hand-set -350, bone min 344.3 against 300, rim prominence
    109.7 against 120.

    Returns status "not_bimodal" when the two modes are not separated enough to
    trust, which is the signature of an uncalibrated or truncated volume.
    """
    full = array[::sample_stride, ::sample_stride, ::sample_stride].ravel()

    def _fit(a):
        """Locate the air and soft tissue modes in one candidate sample."""
        lo, hi = float(np.percentile(a, 0.1)), float(np.percentile(a, 99.9))
        if hi - lo < CALIB_MIN_SEPARATION_HU:
            return {"status": "not_bimodal",
                    "reason": f"intensity range is only {hi - lo:.0f} HU wide"}
        hist, edges = np.histogram(a, bins=256, range=(lo, hi))
        centres = (edges[:-1] + edges[1:]) / 2.0
        smooth = np.convolve(hist, np.ones(5) / 5.0, mode="same")
        # Pad the ends so a mode sitting in the first or last bin is still a
        # local maximum. Without this, a population at the extreme of the
        # histogram range is simply invisible to the peak search.
        floor = 0.02 * smooth.max()
        padded = np.concatenate(([0.0], smooth, [0.0]))
        peaks = [i - 1 for i in range(1, len(padded) - 1)
                 if padded[i] >= padded[i - 1] and padded[i] > padded[i + 1]
                 and padded[i] > floor]
        if len(peaks) < 2:
            return {"status": "not_bimodal",
                    "reason": f"only {len(peaks)} intensity mode(s) found; a head "
                              f"CT should show distinct air and soft tissue "
                              f"populations"}
        # Air is the LOWEST major mode: nothing in a head is less dense than
        # air. "The two largest peaks" is the wrong rule and inverted the whole
        # scale whenever bone was plentiful.
        air_i = min(peaks, key=lambda i: centres[i])
        above = [i for i in peaks
                 if centres[i] - centres[air_i] >= CALIB_MIN_SEPARATION_HU]
        if not above:
            return {"status": "not_bimodal",
                    "reason": f"no intensity mode more than "
                              f"{CALIB_MIN_SEPARATION_HU:.0f} HU above the air mode"}
        soft_i = max(above, key=lambda i: smooth[i])
        air_mode, soft_mode = float(centres[air_i]), float(centres[soft_i])
        sep = soft_mode - air_mode
        bone_min = soft_mode + CALIB_BONE_FRACTION * sep
        head = a[a > air_mode + CALIB_AIR_FRACTION * sep]
        bone_fraction = float((head > bone_min).mean()) if head.size else 1.0
        if not (CALIB_MIN_BONE_FRACTION <= bone_fraction
                <= CALIB_MAX_BONE_FRACTION):
            return {"status": "not_bimodal", "air_mode_hu": air_mode,
                    "soft_mode_hu": soft_mode, "separation_hu": float(sep),
                    "bone_fraction": bone_fraction,
                    "reason": f"the derived bone floor of {bone_min:.0f} HU would "
                              f"call {bone_fraction * 100:.0f}% of the head bone, "
                              f"which is not credible for a head CT; the "
                              f"intensity modes have probably been misidentified"}
        sigma, half = 0.0, 0.12 * sep
        for _ in range(3):
            band = a[(a > soft_mode - half) & (a < soft_mode + half)]
            if band.size < 100:
                break
            sigma = float(np.percentile(band, 75)
                          - np.percentile(band, 25)) / 1.349
            half = max(0.06 * sep, 3.0 * sigma)
        return {"status": "ok", "bone_fraction": bone_fraction,
                "air_mode_hu": air_mode, "soft_mode_hu": soft_mode,
                "separation_hu": float(sep), "noise_sd_hu": sigma,
                "AIR_HU_MAX": air_mode + CALIB_AIR_FRACTION * sep,
                "BONE_HU_MIN": bone_min,
                "LUMEN_HU_MAX": soft_mode + CALIB_LUMEN_FRACTION * sep,
                "RIM_PROMINENCE_HU": max(CALIB_RIM_NOISE_SIGMA * sigma,
                                         CALIB_RIM_MIN_FRACTION * sep)}

    # Out-of-FOV padding is a discrete spike at the low extreme with a gap above
    # it. On the second real subject tested, 20% of the volume was -3024 padding;
    # it became the "air" mode and shifted every class down one, giving a bone
    # floor of -367 HU while still reporting status ok. Both interpretations are
    # tried and the plausible one wins, because in a noiseless phantom true air
    # is also a spike with a gap, so the spike alone cannot decide it.
    candidates = [(full, 0.0)]
    vmin = float(full.min())
    spike = full <= vmin + 1.0
    if spike.mean() >= CALIB_PAD_MIN_FRACTION:
        rest = full[~spike]
        if rest.size and float(rest.min()) - vmin >= CALIB_PAD_MIN_GAP_HU:
            candidates.insert(0, (rest, float(spike.mean())))

    first_result = None
    for sample, pad in candidates:
        res = _fit(sample)
        if first_result is None:
            first_result = res
        if res.get("status") == "ok":
            res["padding_fraction"] = pad
            return res
    # No candidate was usable: report the preferred candidate's failure so the
    # reason reflects the padding-stripped interpretation when one was tried.
    return first_result


def assess_competence(volume_node, array, calib):
    """Decide whether this scan is inside the tool's competence.

    Returns status refused, marginal or ok, with the reasons. A refusal is a
    success: the alternative is a plausible-looking number from a scan the
    method cannot support.
    """
    spacing = list(volume_node.GetSpacing())
    reasons, warnings, checks = [], [], {}

    worst = max(spacing)
    checks["voxel_mm"] = [round(s, 3) for s in spacing]
    checks["samples_across_spf"] = round(3.63 / worst, 2)
    if worst > GATE_VOXEL_REFUSE_MM:
        reasons.append(
            f"voxel size {worst:.2f} mm gives only {3.63 / worst:.1f} samples "
            f"across a 3.6 mm SPF aperture; sub-voxel aperture measurement is "
            f"not supportable above {GATE_VOXEL_REFUSE_MM} mm")
    elif worst > GATE_VOXEL_WARN_MM:
        warnings.append(
            f"voxel size {worst:.2f} mm is coarser than the {GATE_VOXEL_WARN_MM} mm "
            f"this method was characterised at; apertures will be less precise")

    if calib.get("status") != "ok":
        reasons.append(f"intensity calibration failed: {calib.get('reason')}. "
                       f"Thresholds cannot be derived for this volume.")
        checks["calibration"] = calib.get("status")
    else:
        checks["calibration"] = "ok"
        checks["separation_hu"] = round(calib["separation_hu"], 1)
        checks["noise_sd_hu"] = round(calib["noise_sd_hu"], 1)
        if calib["noise_sd_hu"] > 0.08 * calib["separation_hu"]:
            warnings.append(
                f"soft tissue noise {calib['noise_sd_hu']:.0f} HU is high "
                f"relative to tissue contrast; edges will be less reliable")

        bone = array > calib["BONE_HU_MIN"]
        n_bone = int(bone.sum())
        checks["bone_voxels"] = n_bone
        if n_bone < GATE_MIN_BONE_VOXELS:
            reasons.append(f"only {n_bone} bone voxels; too little bone for the "
                           f"symmetry search to find a midsagittal plane")
        else:
            # symmetry needs bone on both sides of the volume's own centre
            xs = bone.sum(axis=(0, 1))
            half = len(xs) // 2
            l, r = int(xs[:half].sum()), int(xs[half:].sum())
            bal = min(l, r) / max(max(l, r), 1)
            checks["bone_lateral_balance"] = round(bal, 3)
            if bal < 0.25:
                warnings.append(
                    f"bone is markedly one-sided in this volume (balance "
                    f"{bal:.2f}); the symmetry midline may be unreliable, "
                    f"usually a sign the field of view is cropped off centre")

    metal = float((array >= GATE_METAL_HU).mean())
    checks["metal_saturated_fraction"] = round(metal, 5)
    if metal > GATE_METAL_REFUSE_FRACTION:
        reasons.append(f"{metal * 100:.1f}% of voxels are saturated at or above "
                       f"{GATE_METAL_HU:.0f} HU; streak artefact will corrupt "
                       f"edge positions")
    elif metal > GATE_METAL_REFUSE_FRACTION / 4:
        warnings.append(f"{metal * 100:.2f}% saturated voxels; check for streak "
                        f"artefact near the measurement planes")

    status = "refused" if reasons else ("marginal" if warnings else "ok")
    return {"status": status, "refusals": reasons, "warnings": warnings,
            "checks": checks}


# ------------------------------------------------- midsagittal by symmetry

def find_midsagittal(array, ijk_to_ras, downsample=SYMMETRY_DOWNSAMPLE,
                     bone_hu=SYMMETRY_BONE_HU):
    """Midsagittal plane as the reflection that best maps the bone mask onto itself.

    Bone is used rather than the whole image because soft tissue, the table and
    the head holder are not symmetric. Scored by Dice on a soft occupancy mask,
    which is bounded and gives the search a gradient to follow.

    Verified on a real scan: SD 0.13 mm across 4 thresholds x 2 downsample
    factors, and a sharp optimum. Returns None if the optimum is flat, which
    means the scan is too asymmetric or too cropped to trust.
    """
    d = (array > bone_hu).astype(np.float32)[::downsample, ::downsample,
                                             ::downsample]
    if d.max() <= 0:
        return None
    d = d / d.max()
    K, J, I = d.shape
    jc, kc = J / 2.0, K / 2.0
    kk, jj, ii = np.meshgrid(np.arange(K), np.arange(J), np.arange(I),
                             indexing="ij")

    def score(x0, yaw, roll):
        xp = x0 + yaw * (jj - jc) + roll * (kk - kc)
        mi = 2.0 * xp - ii
        valid = (mi >= 0) & (mi <= I - 1)
        if not valid.any():
            return -1.0
        mic = np.clip(mi, 0, I - 1)
        f0 = np.floor(mic).astype(int)
        f1 = np.minimum(f0 + 1, I - 1)
        frac = mic - f0
        mirror = d[kk, jj, f0] * (1 - frac) + d[kk, jj, f1] * frac
        a, b = d[valid], mirror[valid]
        if a.sum() < 1 or b.sum() < 1:
            return -1.0
        return float(2 * (a * b).sum() / (a.sum() + b.sum()))

    best = None
    for x0 in np.arange(I * 0.35, I * 0.65, 1.0):
        s = score(x0, 0.0, 0.0)
        if best is None or s > best[0]:
            best = (s, x0, 0.0, 0.0)
    for _ in range(3):
        s0, x0, yaw, roll = best
        for dx in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
            for dy in (-0.06, -0.02, 0.0, 0.02, 0.06):
                for dr in (-0.06, -0.02, 0.0, 0.02, 0.06):
                    s = score(x0 + dx, yaw + dy, roll + dr)
                    if s > best[0]:
                        best = (s, x0 + dx, yaw + dy, roll + dr)
    s, x0, yaw, roll = best

    # Sharpness. A flat optimum means the estimate is arbitrary, so refuse it.
    spacing_ds = abs(ijk_to_ras[0, 0]) * downsample or 1.0
    drop_1mm = s - max(score(x0 - 1.0 / spacing_ds, yaw, roll),
                       score(x0 + 1.0 / spacing_ds, yaw, roll))
    point = ijk_to_ras @ np.array([x0 * downsample, J * downsample / 2.0,
                                   K * downsample / 2.0, 1.0])
    return {"dice": s,
            "x_ras_mm": float(point[0]),
            "yaw_deg": float(np.degrees(np.arctan(yaw))),
            "roll_deg": float(np.degrees(np.arctan(roll))),
            "dice_drop_at_1mm": float(drop_1mm),
            "sharp_enough": bool(drop_1mm > 0.01),
            "downsample": int(downsample), "bone_hu": float(bone_hu)}


# ------------------------------------------------- direction-free aperture

def _lumen_baseline(array, ras_to_ijk, seed, radius_mm=0.6):
    vals = []
    for axis in np.eye(3):
        for sgn in (1.0, -1.0):
            _, hu = _profile(array, ras_to_ijk, seed, axis * sgn, radius_mm)
            vals.extend(hu.tolist())
    return float(np.median(vals)) if vals else None


def _rim_distance(array, ras_to_ijk, origin, direction, baseline,
                  prominence=RIM_PROMINENCE_HU, search_mm=RIM_SEARCH_MM):
    t, hu = _profile(array, ras_to_ijk, origin, direction, search_mm)
    # The FIRST rim that clears prominence, not the highest one along the ray.
    # Taking the highest walked straight past a weak inner margin to whatever
    # denser bone lay behind it, over-reporting the aperture: on a phantom with
    # a 165 HU inner rim at 2 mm and a 900 HU outer rim at 3 mm it returned
    # 6.41 mm instead of 4 mm, at every prominence setting. A foramen is
    # bounded by the first margin outward, whatever sits behind it.
    peak = None
    for i in range(1, len(hu)):
        if hu[i] - baseline >= prominence:
            j = i
            while j + 1 < len(hu) and hu[j + 1] > hu[j]:
                j += 1
            peak = j
            break
    if peak is None:
        return None
    level = baseline + 0.5 * (hu[peak] - baseline)
    for i in range(peak, 0, -1):
        if hu[i - 1] < level <= hu[i]:
            frac = (level - hu[i - 1]) / (hu[i] - hu[i - 1])
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return None


def min_feret_aperture(array, ras_to_ijk, seed, plane="axial",
                       n_angles=FERET_N_ANGLES, calib=None):
    """Narrowest rim-to-rim aperture through `seed`, over all in-plane angles.

    Takes ONE seed and no direction, so the operator's angle cannot influence
    the result. Rims are found by LOCAL prominence above the lumen baseline,
    not an absolute threshold: the SPF's posterior rim is thin enough that
    partial volume caps it near 227 HU, and a 300 HU threshold looks straight
    through it and wrongly reports the foramen as open.

    On the reference scan this read 3.63 mm where a hand-drawn chord read
    6.35 mm, against a published SPF width of 3.79 +/- 0.35 mm.
    """
    seed = np.asarray(seed, float)
    baseline = _lumen_baseline(array, ras_to_ijk, seed)
    if baseline is None or baseline >= _th(calib, "LUMEN_HU_MAX", LUMEN_HU_MAX):
        return {"status": "seed_not_in_lumen", "baseline_hu": baseline}
    prom = _th(calib, "RIM_PROMINENCE_HU", RIM_PROMINENCE_HU)
    best = None
    for ang in np.linspace(0.0, np.pi, int(n_angles), endpoint=False):
        if plane == "axial":
            u = np.array([np.cos(ang), np.sin(ang), 0.0])
        elif plane == "coronal":
            u = np.array([np.cos(ang), 0.0, np.sin(ang)])
        else:
            u = np.array([0.0, np.cos(ang), np.sin(ang)])
        dp = _rim_distance(array, ras_to_ijk, seed, u, baseline, prom)
        dn = _rim_distance(array, ras_to_ijk, seed, -u, baseline, prom)
        if dp is None or dn is None:
            continue
        total = dp + dn
        if best is None or total < best["aperture_mm"]:
            best = {"aperture_mm": total,
                    "angle_deg": float(np.degrees(ang)),
                    "p1_ras": (seed + u * dp).tolist(),
                    "p2_ras": (seed - u * dn).tolist()}
    if best is None:
        return {"status": "no_bounded_direction", "baseline_hu": baseline}
    best.update({"status": "ok", "plane": plane, "baseline_hu": baseline,
                 "prominence_hu": prom})
    return best


def aperture_with_stability(array, ras_to_ijk, seed, plane="axial", calib=None,
                            tol_mm=APERTURE_SENSITIVITY_TOL_MM):
    """Minimum feret plus a check that it does not hinge on the threshold.

    Re-measures at +/-25% rim prominence. If the answer moves by more than
    tol_mm the aperture is sitting on a detection cliff, which happens when a
    rim is thin enough to be marginal against noise. On the reference scan the
    coronal SPF aperture reads 4.47 mm at 120 HU and 6.17 mm at 139 HU, so the
    number alone would look perfectly reasonable and be threshold-determined.
    Reported, not silently resolved: the tool cannot know which rim is real.
    """
    base = min_feret_aperture(array, ras_to_ijk, seed, plane=plane, calib=calib)
    if base.get("status") != "ok":
        return base
    prom = base["prominence_hu"]
    alts = []
    for scale in (0.75, 1.25):
        c = dict(calib or {})
        c["RIM_PROMINENCE_HU"] = prom * scale
        r = min_feret_aperture(array, ras_to_ijk, seed, plane=plane, calib=c)
        if r.get("status") == "ok":
            alts.append(r["aperture_mm"])
    if alts:
        spread = max(max(alts), base["aperture_mm"]) - min(min(alts),
                                                           base["aperture_mm"])
        base["threshold_spread_mm"] = float(spread)
        base["threshold_sensitive"] = bool(spread > tol_mm)
        base["aperture_at_prominence"] = {
            f"{prom * 0.75:.0f}": round(alts[0], 3),
            f"{prom:.0f}": round(base["aperture_mm"], 3),
            f"{prom * 1.25:.0f}": (round(alts[1], 3) if len(alts) > 1 else None)}
    return base


def spf_centre(array, ras_to_ijk, seed, calib=None):
    """SPF centre and both apertures from one seed."""
    ax = aperture_with_stability(array, ras_to_ijk, seed, plane="axial", calib=calib)
    co = aperture_with_stability(array, ras_to_ijk, seed, plane="coronal", calib=calib)
    if ax.get("status") != "ok" or co.get("status") != "ok":
        return {"status": "aperture_failed", "axial": ax, "coronal": co}
    centre = (np.array(ax["p1_ras"]) + np.array(ax["p2_ras"])
              + np.array(co["p1_ras"]) + np.array(co["p2_ras"])) / 4.0
    return {"status": "ok", "centre_ras": centre.tolist(),
            "aperture_axial_mm": ax["aperture_mm"],
            "aperture_coronal_mm": co["aperture_mm"],
            "aperture_threshold_sensitive": bool(ax.get("threshold_sensitive")
                                                 or co.get("threshold_sensitive")),
            "shift_from_seed_mm": float(np.linalg.norm(centre - np.asarray(seed, float))),
            "axial": ax, "coronal": co}


# ------------------------------------------------------- lumen cross section

def lumen_at_coronal(array, ijk_to_ras, ras_to_ijk, seed_ras, spacing,
                     target_ras=None, midline_x=None, side_sign=None,
                     prev_mask=None, prev_axis_ki=None,
                     axis_search_mm=AXIS_STEP_SEARCH_MM, calib=None):
    """Mucosal and bony cross section on the coronal plane through `seed_ras`.

    Mucosal: the enclosed airway component containing the seed. Components
    touching the slice border are dropped, which is what stops the region
    flooding out of the nostril and reporting the whole head as lumen.

    Bony: rays cast from the lumen centroid to the first bone. This is a
    star-shaped approximation about the centroid, stated plainly because it
    under-reports a re-entrant recess. It is used rather than a flood fill
    because a flood fill of everything below 300 HU leaks into the orbit and
    the soft tissues of the face.
    """
    ijk = (ras_to_ijk @ np.array([*seed_ras, 1.0]))[:3]
    j = int(round(ijk[1]))
    K, J, I = array.shape
    if not (0 <= j < J):
        return {"status": "plane_outside_volume"}
    sl = array[:, j, :]                       # [k, i] = [S, R]
    dz, dx = spacing[2], spacing[0]

    air = sl < _th(calib, "AIR_HU_MAX", AIR_HU_MAX)

    # Restrict to the ipsilateral side of the midline. Without this the two
    # nasal cavities merge, inferiorly and through the choana, into one
    # component: on the reference scan that reported 2000 mm2 and a 25 mm
    # inscribed diameter for a nasal airway, and made the profile oscillate
    # between 310 and 838 mm2 from slice to slice. This is the reason the
    # symmetry plane is computed at all.
    if midline_x is not None and side_sign is not None:
        cols = np.arange(sl.shape[1])
        xs = (ijk_to_ras @ np.stack([cols, np.full_like(cols, j),
                                     np.zeros_like(cols),
                                     np.ones_like(cols)]).astype(float))[0]
        ipsilateral = (((xs - midline_x) * side_sign) > 0) & (
            np.abs(xs - midline_x) <= NASAL_HALF_WIDTH_MM)
        air = air & ipsilateral[None, :]

    lab, n = ndimage.label(air)
    if n == 0:
        return {"status": "no_air_on_plane"}
    border = set(np.unique(np.concatenate(
        [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    border.discard(0)
    seed_ki = (int(round(ijk[2])), int(round(ijk[0])))
    seed_ki = (min(max(seed_ki[0], 0), K - 1), min(max(seed_ki[1], 0), I - 1))
    l = lab[seed_ki]

    # Continuity along the corridor. Choosing each station's component
    # independently lets the selection jump into the maxillary sinus, which is
    # enclosed, ipsilateral, and far larger than the airway, so it looks like a
    # valid pick and silently reports a 25 mm "stent diameter". Preferring the
    # component that overlaps the previous station keeps the profile on one
    # continuous structure.
    if prev_mask is not None:
        overlaps = {}
        for cand in range(1, n + 1):
            if cand in border:
                continue
            ov = int((prev_mask & (lab == cand)).sum())
            if ov > 0:
                overlaps[cand] = ov
        if overlaps:
            l = max(overlaps, key=overlaps.get)

    if l == 0 or l in border:
        best, l = None, 0
        for cand in range(1, n + 1):
            if cand in border:
                continue
            c = np.argwhere(lab == cand)
            if len(c) < 20:
                continue
            d = np.hypot((c[:, 0] - ijk[2]) * dz, (c[:, 1] - ijk[0]) * dx).min()
            if best is None or d < best:
                best, l = d, cand
    if l == 0:
        return {"status": "no_enclosed_airway_on_plane"}

    comp = lab == l
    dt = ndimage.distance_transform_edt(comp, sampling=(dz, dx))
    edge = comp & ~ndimage.binary_erosion(comp)
    ei = np.argwhere(edge)

    # The stent dimension is the largest circle that fits at this station,
    # centred on a TRACKED centreline rather than on a straight line between
    # the two seeds. Two failures on the reference scan forced this:
    #   - a global maximum over the component reported 25 mm, because the
    #     maxillary sinus is genuinely continuous with the nasal cavity through
    #     the maxillary ostium, so the component legitimately contains it;
    #   - a straight seed-to-seed axis reported 1.0 mm almost everywhere,
    #     because the nasal corridor curves and the straight line runs along
    #     the lumen edge.
    # Tracking the local distance-transform maximum near the previous station's
    # centre follows the corridor without leaving it.
    if prev_axis_ki is not None:
        r_k = max(1, int(round(axis_search_mm / dz)))
        r_i = max(1, int(round(axis_search_mm / dx)))
        k0, i0 = prev_axis_ki
        win = np.zeros_like(dt, dtype=bool)
        win[max(0, k0 - r_k):k0 + r_k + 1, max(0, i0 - r_i):i0 + r_i + 1] = True
        search = dt * (win & comp)
    else:
        # First station: stay near the operator's seed, so the corridor starts
        # where they pointed rather than in the largest air space in frame.
        r_k = max(1, int(round(12.0 / dz)))
        r_i = max(1, int(round(12.0 / dx)))
        k0, i0 = seed_ki
        win = np.zeros_like(dt, dtype=bool)
        win[max(0, k0 - r_k):k0 + r_k + 1, max(0, i0 - r_i):i0 + r_i + 1] = True
        search = dt * (win & comp)
    if search.max() <= 0:
        return {"status": "no_lumen_near_axis"}
    ax_ki = np.unravel_index(int(np.argmax(search)), search.shape)
    out = {"status": "ok",
           "component_area_mm2": float(comp.sum() * dz * dx),
           "mucosal_inscribed_diameter_mm": float(2 * dt[ax_ki]),
           "component_max_diameter_mm": float(2 * dt.max()),
           "axis_ki": (int(ax_ki[0]), int(ax_ki[1]))}

    # bony contour by ray casting from the inscribed-circle centre (the axis
    # point found above), which sits inside the lumen by construction
    cen_ras = (ijk_to_ras @ np.array([ax_ki[1], j, ax_ki[0], 1.0]))[:3]
    rays = []
    for ang in np.linspace(0, 2 * np.pi, BONY_N_RAYS, endpoint=False):
        u = np.array([np.cos(ang), 0.0, np.sin(ang)])
        t, hu = _profile(array, ras_to_ijk, cen_ras, u, 40.0)
        hit = np.where(hu >= _th(calib, "BONE_HU_MIN", BONE_HU_MIN))[0]
        rays.append(float(t[hit[0]]) if len(hit) else np.nan)
    rays = np.array(rays)
    if np.isfinite(rays).sum() >= BONY_N_RAYS * 0.8:
        r = np.nan_to_num(rays, nan=np.nanmedian(rays))
        ang = np.linspace(0, 2 * np.pi, BONY_N_RAYS, endpoint=False)
        out["bony_area_mm2"] = float(0.5 * np.sum(r ** 2) * (2 * np.pi / BONY_N_RAYS))
        out["bony_max_inscribed_diameter_mm"] = float(2 * r.min())
        out["bony_contour_method"] = "star-shaped rays from lumen centroid"
    else:
        out["bony_status"] = "too_many_rays_unbounded"

    if target_ras is not None and len(ei):
        d = np.hypot((ei[:, 0] - (ras_to_ijk @ np.array([*target_ras, 1.0]))[2]) * dz,
                     (ei[:, 1] - (ras_to_ijk @ np.array([*target_ras, 1.0]))[0]) * dx)
        out["surface_to_target_mm"] = float(d.min())
    out["centroid_ras"] = cen_ras.tolist()
    out["_mask"] = comp
    return out


def corridor_profile(array, ijk_to_ras, ras_to_ijk, p_ant, p_post, spacing,
                     step_mm=PROFILE_STEP_MM, midline_x=None, side_sign=None,
                     calib=None):
    """Lumen cross section every step_mm from the piriform seed to the choana seed.

    Sampling the corridor continuously rather than at named anatomical stations
    is deliberate. A stent is a continuous device, so taper and binding point
    are what matter, and it sidesteps the basal lamella ambiguity entirely
    (two CT studies disagree by 4.3 mm on where that landmark is).
    """
    p_ant = np.asarray(p_ant, float); p_post = np.asarray(p_post, float)
    L = float(np.linalg.norm(p_post - p_ant))
    if L <= 0:
        return {"status": "degenerate_corridor"}
    u = (p_post - p_ant) / L
    stations = []
    prev_mask = None
    prev_axis = None
    for t in np.arange(0.0, L + 1e-9, step_mm):
        p = p_ant + u * t
        r = lumen_at_coronal(array, ijk_to_ras, ras_to_ijk, p, spacing,
                             midline_x=midline_x, side_sign=side_sign,
                             prev_mask=prev_mask, prev_axis_ki=prev_axis,
                             calib=calib)
        if r.get("status") == "ok":
            prev_mask = r.pop("_mask", None)
            prev_axis = r.get("axis_ki")
            stations.append({
                "t_mm": round(float(t), 2),
                "component_area_mm2": round(r["component_area_mm2"], 1),
                "mucosal_diam_mm": round(r["mucosal_inscribed_diameter_mm"], 3),
                "bony_diam_mm": (round(r["bony_max_inscribed_diameter_mm"], 3)
                                 if "bony_max_inscribed_diameter_mm" in r else None)})
    if not stations:
        return {"status": "no_valid_stations", "corridor_length_mm": L}

    # Continuity. A real airway changes smoothly along its axis, so a station
    # whose area leaps away from its neighbours means component selection
    # jumped to a different structure. Flag rather than silently average: a
    # jumpy profile is the signature of a merged or wrong lumen.
    areas = np.array([s["component_area_mm2"] for s in stations])
    jumps = 0
    for i in range(1, len(areas)):
        lo, hi = sorted((areas[i - 1], areas[i]))
        if lo > 0 and hi / lo > 2.0:
            stations[i]["discontinuity"] = True
            jumps += 1
    clean = [s for s in stations if not s.get("discontinuity")]
    md = [s["mucosal_diam_mm"] for s in clean] or [s["mucosal_diam_mm"] for s in stations]
    bd = [s["bony_diam_mm"] for s in clean if s["bony_diam_mm"] is not None]
    binding = min(clean or stations, key=lambda s: s["mucosal_diam_mm"])
    return {"status": "ok",
            "corridor_length_mm": L,
            "a_range": [float(p_ant[1]), float(p_post[1])],
            "n_stations": len(stations),
            "n_discontinuities": jumps,
            "profile_is_smooth": bool(jumps <= max(1, len(stations) // 20)),
            "mucosal_min_diameter_mm": float(min(md)),
            "mucosal_max_diameter_mm": float(max(md)),
            "binding_point_t_mm": binding["t_mm"],
            "binding_point_diameter_mm": binding["mucosal_diam_mm"],
            "bony_min_diameter_mm": float(min(bd)) if bd else None,
            "compressible_margin_mm": (round(float(min(bd) - min(md)), 3)
                                       if bd else None),
            "stations": stations}


def derive_corridor_endpoints(array, ijk_to_ras, ras_to_ijk, spf_centre,
                              midline_x, yaw_deg, side_sign, calib=None):
    """Corridor endpoints from the two structures that sit in the midline.

    Per Dr. Iloreta: the posterior edge of the septum and the anterior edge of
    the maxilla lie in the midline almost always, so both are findable on the
    symmetry plane, which is the one frame this tool derives reliably.

    Searching all midline bone does NOT work: an unbounded argmax put the
    anterior point at the chin and the posterior point behind the sphenoid,
    because the sphenoid sinus also puts air in the midline and the clivus is
    also midline bone. Bounding the search to a height band around the SPF and
    to a limited distance either side of it selects the vomer and the maxilla.

    On the reference scan this gave a corridor of 55.5 mm against a published
    nasal cavity length of about 45 to 60 mm, verified by eye. n=1, unvalidated,
    which is why the operator's seeds win whenever they are present.
    """
    sa, ss = float(spf_centre[1]), float(spf_centre[2])
    step = 0.5
    ty = np.tan(np.radians(yaw_deg))

    def midline_bone(a_mm, s_lo, s_hi):
        """Bone anywhere in a thin midline slab at this A, over an S band.

        Sampled over a +/- 1 mm slab rather than a single line: the vomer is
        thin and a one-voxel-wide probe drops in and out of it, which truncated
        the derived corridor from 55.5 mm to 33.5 mm on the reference scan.
        """
        svals = np.arange(s_lo, s_hi, step)
        for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):
            xs = midline_x + ty * (a_mm - sa) + dx
            pts = np.stack([np.full_like(svals, xs),
                            np.full_like(svals, a_mm), svals], axis=-1)
            if (_sample(array, ras_to_ijk, pts)
                    > _th(calib, "BONE_HU_MIN", BONE_HU_MIN)).any():
                return True
        return False

    post = None
    for a in np.arange(sa, sa - CORRIDOR_POST_SEARCH_MM, -step):
        if midline_bone(a, ss - 16.0, ss + 4.0):
            post = a
    ant = None
    for a in np.arange(sa, sa + CORRIDOR_ANT_SEARCH_MM, step):
        if midline_bone(a, ss - 20.0, ss - 4.0):
            ant = a
    if post is None or ant is None:
        return None
    # Place the endpoints inside the airway on the correct side, halfway
    # between the midline and the SPF, and let the centreline tracker do the
    # rest from there.
    x = midline_x + side_sign * abs(float(spf_centre[0]) - midline_x) * 0.5
    return {"p_ant": np.array([x, float(ant), ss]),
            "p_post": np.array([x, float(post), ss]),
            "derived_length_mm": float(ant - post),
            "source": "derived from midline septal and maxillary edges"}


def morphometrics(array, ijk_to_ras, ras_to_ijk, spacing, midline_x, yaw_deg,
                  corridor_by_side, calib=None):
    """Population morphometry, reported SEPARATELY from the device constraints.

    These are descriptors of the anatomy, not dimensions a stent is designed
    against, and they must not be read as design inputs. Everything here is
    geometry on the symmetry frame plus calibrated tissue classes, which is the
    class of measurement that tested clean.

    Deliberately NOT included: named sinus volumes. Telling the maxillary sinus
    from the ethmoid air cells is anatomical recognition, which has failed eight
    times in this project. Total enclosed air per side is reported instead,
    which is geometrically defined and means exactly what it says.
    """
    if midline_x is None:
        return {"status": "no_midline",
                "why": "every quantity here is defined against the midsagittal "
                       "plane, so without a trustworthy one none of it is defined"}
    air_max = _th(calib, "AIR_HU_MAX", AIR_HU_MAX)
    dz, dy, dx = spacing[2], spacing[1], spacing[0]
    voxel_mm3 = dx * dy * dz
    K, J, I = array.shape

    # RAS x of every column, so "which side" is a real question about geometry
    cols = np.arange(I)
    xs = (ijk_to_ras @ np.stack([cols, np.zeros_like(cols), np.zeros_like(cols),
                                 np.ones_like(cols)]).astype(float))[0]
    dist = xs - midline_x

    out = {"status": "ok", "voxel_mm3": round(voxel_mm3, 5), "sides": {},
           "caveat": ("descriptive morphometry, not device design inputs; "
                      "no anatomical structure is named here because naming "
                      "one requires recognition, which does not automate")}

    air = array < air_max
    rows = np.arange(J)
    ys = (ijk_to_ras @ np.stack([np.zeros_like(rows), rows, np.zeros_like(rows),
                                 np.ones_like(rows)]).astype(float))[1]

    # ONE anterior-posterior extent for both sides. Measuring each side over
    # whatever corridor happened to be available made the volumes
    # incomparable: with a corridor on the right only, the left was integrated
    # over the entire volume and the asymmetry ratio read 9.16x on a subject
    # whose septum is nearly straight.
    ranges = [c["a_range"] for c in (corridor_by_side or {}).values()
              if c and c.get("status") == "ok" and c.get("a_range")]
    if ranges:
        a_lo = max(min(r) for r in ranges)          # intersection, so both
        a_hi = min(max(r) for r in ranges)          # sides see the same span
        band = (ys >= a_lo) & (ys <= a_hi)
        extent = [round(float(a_lo), 1), round(float(a_hi), 1)]
    else:
        band = np.ones(J, dtype=bool)
        extent = None
    out["shared_a_extent_mm"] = extent
    out["extent_note"] = ("both sides integrated over the same span, otherwise "
                          "the asymmetry ratio is not a ratio of like things")

    for side, sign in (("R", 1.0), ("L", -1.0)):
        if extent is None:
            # Without an anterior-posterior extent this integrates every air
            # voxel near the midline, including the air around the head: it
            # read 253 cm3 per side on a scan with no corridor, which is not a
            # nasal airway by any reading. Refused rather than reported.
            out["sides"][side] = {
                "nasal_airway_volume_mm3": None,
                "status": "needs_corridor",
                "why": ("airway volume is only defined between the corridor "
                        "endpoints; without them this would integrate the air "
                        "around the head. Place piriform and choana seeds, or "
                        "let them be derived from an SPF seed.")}
            continue
        ipsi = (dist * sign > 0) & (np.abs(dist) <= NASAL_HALF_WIDTH_MM)
        mask = air & ipsi[None, None, :] & band[None, :, None]
        vol = float(mask.sum()) * voxel_mm3
        out["sides"][side] = {
            "nasal_airway_volume_mm3": round(vol, 1),
            "corridor_a_extent_mm": extent,
            "definition": (f"enclosed air within {NASAL_HALF_WIDTH_MM:.0f} mm of "
                           f"the midline on this side, between the corridor "
                           f"endpoints")}

    # Septal deviation. Measured per ROW, requiring a genuine non-air gap that
    # spans the midline: that gap is the septum, whatever it is made of.
    # Collapsing over height instead just asked "is there air near the midline",
    # which answered 0.15 mm (a third of a voxel) on two different subjects.
    near = np.abs(dist) <= NASAL_HALF_WIDTH_MM
    near_cols = np.where(near)[0]
    devs = []
    if len(near_cols) > 4:
        d_near = dist[near_cols]
        order = np.argsort(d_near)
        cols_sorted = near_cols[order]
        d_sorted = d_near[order]
        mid_k = int(np.searchsorted(d_sorted, 0.0))
        j_step = max(1, J // 80)
        k_step = max(1, K // 80)
        for j in range(0, J, j_step):
            if not band[j]:
                continue
            for k in range(0, K, k_step):
                row = air[k, j, cols_sorted]
                if not row.any():
                    continue
                left = np.where(row[:mid_k])[0]
                right = np.where(row[mid_k:])[0]
                if not (len(left) and len(right)):
                    continue
                l_edge = d_sorted[left[-1]]          # innermost air on the left
                r_edge = d_sorted[mid_k + right[0]]  # innermost air on the right
                gap = r_edge - l_edge
                if gap < 0.5 or gap > 12.0:          # not a septum
                    continue
                if row[left[-1] + 1:mid_k + right[0]].any():
                    continue                          # the gap is not solid
                devs.append(float((l_edge + r_edge) / 2.0))
    if devs:
        devs = np.asarray(devs)
        out["septal_deviation"] = {
            "max_abs_mm": round(float(np.abs(devs).max()), 2),
            "median_signed_mm": round(float(np.median(devs)), 2),
            "p95_abs_mm": round(float(np.percentile(np.abs(devs), 95)), 2),
            "toward": ("right" if np.median(devs) > 0 else "left"),
            "n_rows": int(len(devs)),
            "definition": ("signed offset from the symmetry plane of the centre "
                           "of the solid tissue gap separating the two airways; "
                           "positive is right")}
    else:
        out["septal_deviation"] = {"status": "no_septum_rows_found"}

    r = out["sides"].get("R", {}).get("nasal_airway_volume_mm3")
    l = out["sides"].get("L", {}).get("nasal_airway_volume_mm3")
    if not (r and l):
        out["airway_asymmetry"] = {"status": "needs_corridor"}
    if r and l:
        out["airway_asymmetry"] = {
            "volume_ratio_larger_over_smaller": round(max(r, l) / max(min(r, l), 1e-9), 3),
            "difference_mm3": round(abs(r - l), 1),
            "larger_side": "R" if r > l else "L"}
    return out


# ------------------------------------------------------------------- driver

def _seeds_in_scene():
    """Resolve seed fiducials by name. Never guesses a missing seed."""
    found = {}
    for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
        name = node.GetName()
        for role in SEED_ROLES:
            for side in ("R", "L"):
                if name == f"{role}_{side}":
                    if node.GetNumberOfControlPoints() < 1:
                        continue
                    p = [0.0, 0.0, 0.0]
                    node.GetNthControlPointPositionWorld(0, p)
                    found[(role, side)] = np.array(p)
    return found


def pick_volume(volume_node_name=None):
    if volume_node_name:
        return slicer.mrmlScene.GetFirstNodeByName(volume_node_name)
    vols = [v for v in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if v.GetImageData() is not None]
    if not vols:
        return None
    return max(vols, key=lambda v: np.prod(v.GetImageData().GetDimensions()))


def run(volume_node_name=None, write_json_dir=None, skip_symmetry=False):
    t0 = time.time()
    params = collect_params()
    report = {"script_version": SCRIPT_VERSION,
              "run_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
              "config_hash": config_hash(params),
              "params": params,
              "conduction_budget_mm": CONDUCTION_BUDGET_MM,
              "checks": [], "sides": {}}

    def check(cid, severity, detail):
        report["checks"].append({"id": cid, "severity": severity, "detail": detail})

    volume = pick_volume(volume_node_name)
    if volume is None:
        check("no_volume", "error", "no scalar volume in the scene")
        report["status"] = "failed"
        return report
    report["volume"] = {"name": volume.GetName(),
                        "spacing": list(volume.GetSpacing()),
                        "dimensions": list(volume.GetImageData().GetDimensions())}
    array, M, R2I = _volume_arrays(volume)
    spacing = volume.GetSpacing()

    # Calibrate to this volume, then decide whether the scan is inside the
    # tool's competence. A refusal here is the point of the gate: on an
    # out-of-scope scan the failure mode is a plausible number, not an error.
    calib = calibrate_intensities(array)
    report["calibration"] = calib
    gate = assess_competence(volume, array, calib)
    report["competence"] = gate
    for w in gate["warnings"]:
        check("competence_marginal", "warn", w)
    if gate["status"] == "refused":
        for r in gate["refusals"]:
            check("competence_refused", "error", r)
        report["status"] = "refused"
        report["elapsed_s"] = round(time.time() - t0, 1)
        return report

    if not skip_symmetry:
        sym = find_midsagittal(array, M,
                               bone_hu=_th(calib, "BONE_HU_MIN", SYMMETRY_BONE_HU))
        report["midsagittal"] = sym
        if sym is None:
            check("symmetry_failed", "warn", "no bone found for symmetry search")
        elif not sym["sharp_enough"]:
            check("symmetry_flat", "warn",
                  f"the symmetry optimum is flat (Dice drops only "
                  f"{sym['dice_drop_at_1mm']:.4f} at 1 mm off-plane), so the "
                  f"midline estimate is not trustworthy on this scan")

    sym = report.get("midsagittal")
    midline_x = (sym.get("x_ras_mm")
                 if sym and sym.get("sharp_enough") else None)

    seeds = _seeds_in_scene()
    # Seeds are scene-global, so a seed left over from a previously loaded
    # volume will silently be used on this one. Reject any that fall outside
    # this volume's bounds.
    K0, J0, I0 = array.shape
    outside = []
    for key, pt in list(seeds.items()):
        ijk = (R2I @ np.array([*pt, 1.0]))[:3]
        if not (0 <= ijk[0] < I0 and 0 <= ijk[1] < J0 and 0 <= ijk[2] < K0):
            outside.append(f"{key[0]}_{key[1]}")
            del seeds[key]
    if outside:
        check("seeds_outside_volume", "error",
              f"seed(s) {outside} lie outside {volume.GetName()} and were "
              f"ignored. They are almost certainly left over from a different "
              f"scan: markup nodes belong to the scene, not to a volume.")
    report["seeds_found"] = sorted(f"{r}_{s}" for (r, s) in seeds)
    def _add_morphometrics(corridors):
        try:
            report["morphometrics"] = morphometrics(
                array, M, R2I, spacing, midline_x,
                (sym or {}).get("yaw_deg", 0.0), corridors, calib=calib)
        except Exception as exc:
            report["morphometrics"] = {"status": "error", "detail": str(exc)}
            check("morphometrics_error", "warn", f"morphometry failed: {exc}")

    if not seeds:
        check("no_seeds", "error",
              "no seed fiducials found. Expected names like SPF_R, piriform_R, "
              "choana_R. See the module docstring. Descriptive morphometry is "
              "still reported below: it needs only the midsagittal plane.")
        _add_morphometrics({})
        report["status"] = "failed"
        report["elapsed_s"] = round(time.time() - t0, 1)
        return report

    for side in ("R", "L"):
        if not any(s == side for (_, s) in seeds):
            continue
        rec = {}
        spf_seed = seeds.get(("SPF", side))
        if spf_seed is None:
            check(f"no_spf_{side}", "error",
                  f"side {side} has seeds but no SPF_{side}; the headline "
                  f"conduction gap cannot be computed")
        else:
            sc = spf_centre(array, R2I, spf_seed, calib=calib)
            rec["spf"] = sc
            if sc["status"] != "ok":
                check(f"spf_aperture_{side}", "error",
                      f"SPF_{side}: {sc['status']}. Re-place the seed well "
                      f"inside the foramen lumen.")
            else:
                for pl in ("axial", "coronal"):
                    a = sc[pl]
                    if a.get("threshold_sensitive"):
                        check(f"aperture_threshold_sensitive_{pl}_{side}", "warn",
                              f"side {side}: the {pl} SPF aperture moves "
                              f"{a['threshold_spread_mm']:.2f} mm when the rim "
                              f"prominence is varied by +/-25% "
                              f"({a['aperture_at_prominence']}). A rim here is "
                              f"marginal against image noise, so this aperture "
                              f"is threshold-determined. Confirm it by eye.")
                if sc["shift_from_seed_mm"] > 3.0:
                    check(f"spf_centre_moved_{side}", "warn",
                          f"SPF_{side} centre sits {sc['shift_from_seed_mm']:.2f} mm "
                          f"from the seed; check the seed is in the foramen and "
                          f"not an adjacent air space")
                lum = lumen_at_coronal(array, M, R2I, sc["centre_ras"], spacing,
                                       target_ras=sc["centre_ras"],
                                       midline_x=midline_x,
                                       side_sign=(1.0 if side == "R" else -1.0),
                                       calib=calib)
                lum.pop("_mask", None)
                rec["spf_station"] = lum
                if lum.get("status") == "ok" and "surface_to_target_mm" in lum:
                    gap = lum["surface_to_target_mm"]
                    rec["conduction_gap_mm"] = gap
                    rec["within_budget"] = bool(gap <= CONDUCTION_BUDGET_MM)
                    if gap > CONDUCTION_BUDGET_MM:
                        check(f"budget_exceeded_{side}", "warn",
                              f"side {side}: mucosal surface to SPF is "
                              f"{gap:.2f} mm, over the "
                              f"{CONDUCTION_BUDGET_MM:.1f} mm budget")
                else:
                    check(f"spf_station_{side}", "warn",
                          f"side {side}: {lum.get('status')}")

        pir, cho = seeds.get(("piriform", side)), seeds.get(("choana", side))
        corridor_source = "seeded"
        if (pir is None or cho is None) and midline_x is not None \
                and rec.get("spf", {}).get("status") == "ok":
            der = derive_corridor_endpoints(
                array, M, R2I, rec["spf"]["centre_ras"], midline_x,
                (sym or {}).get("yaw_deg", 0.0),
                1.0 if side == "R" else -1.0, calib=calib)
            if der is not None:
                pir = pir if pir is not None else der["p_ant"]
                cho = cho if cho is not None else der["p_post"]
                corridor_source = "derived"
                rec["corridor_endpoints"] = der["source"]
                check(f"corridor_derived_{side}", "warn",
                      f"side {side}: corridor endpoints DERIVED from the midline "
                      f"septal and maxillary edges, not seeded. Length "
                      f"{der['derived_length_mm']:.1f} mm. This derivation is "
                      f"validated on one scan by eye only; place piriform_{side} "
                      f"and choana_{side} to override it.")
        if pir is not None and cho is not None:
            if midline_x is None:
                check(f"corridor_no_midline_{side}", "error",
                      f"side {side}: no trustworthy midsagittal plane, so the "
                      f"two nasal cavities cannot be separated. Without the "
                      f"midline the lumen component merges across the septum "
                      f"and reports roughly double the true area. Corridor "
                      f"profile refused rather than reported wrong.")
            else:
                rec["corridor_source"] = corridor_source
                rec["corridor"] = corridor_profile(
                    array, M, R2I, pir, cho, spacing,
                    midline_x=midline_x,
                    side_sign=(1.0 if side == "R" else -1.0), calib=calib)
                cor = rec["corridor"]
                if cor.get("status") != "ok":
                    check(f"corridor_{side}", "warn",
                          f"side {side}: {cor.get('status')}")
                elif not cor.get("profile_is_smooth"):
                    check(f"corridor_jumpy_{side}", "warn",
                          f"side {side}: {cor['n_discontinuities']} of "
                          f"{cor['n_stations']} stations jump by more than 2x "
                          f"in area against their neighbour. The lumen "
                          f"selection is switching structures; inspect the "
                          f"profile before using these dimensions.")
        else:
            missing = [n for n, s in (("piriform", pir), ("choana", cho)) if s is None]
            check(f"corridor_seeds_{side}", "warn",
                  f"side {side}: no corridor profile, missing {missing}")

        for role in ("vidian", "FR"):
            seed = seeds.get((role, side))
            if seed is None:
                rec[f"{role}_clearance_mm"] = None
                continue
            if rec.get("spf", {}).get("status") == "ok":
                d = float(np.linalg.norm(np.array(rec["spf"]["centre_ras"]) - seed))
                rec[f"{role}_clearance_mm"] = round(d, 3)
        report["sides"][side] = rec

    # Morphometry last, and in its own block, so it can never be mistaken for
    # a device constraint.
    _add_morphometrics({sd: rec.get("corridor")
                        for sd, rec in report["sides"].items()})

    report["status"] = "ok" if not any(
        c["severity"] == "error" for c in report["checks"]) else "failed"
    report["elapsed_s"] = round(time.time() - t0, 1)

    if write_json_dir:
        os.makedirs(write_json_dir, exist_ok=True)
        path = os.path.join(
            write_json_dir,
            f"stent_constraints_{report['run_datetime'].replace(':', '')}.json")
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        report["written_to"] = path
    return report


def print_report(report):
    B = report.get("conduction_budget_mm", CONDUCTION_BUDGET_MM)
    print("=" * 72)
    print(f" NINS stent design constraints  v{report['script_version']}"
          f"   [{report.get('status')}]")
    print("=" * 72)
    print(f"  volume:  {report.get('volume', {}).get('name')}")
    print(f"  config:  {report.get('config_hash')}   "
          f"{report.get('elapsed_s')} s")
    cal = report.get("calibration", {})
    if cal.get("status") == "ok":
        print(f"  calib:   air {cal['air_mode_hu']:.0f} / soft "
              f"{cal['soft_mode_hu']:.0f} HU, noise {cal['noise_sd_hu']:.0f}"
              f"  ->  air<{cal['AIR_HU_MAX']:.0f}  bone>{cal['BONE_HU_MIN']:.0f}"
              f"  rim>{cal['RIM_PROMINENCE_HU']:.0f}")
    elif cal:
        print(f"  calib:   FAILED ({cal.get('reason')})")
    gate = report.get("competence", {})
    if gate:
        vx = gate["checks"].get("voxel_mm")
        print(f"  scan:    {gate['status'].upper()}   voxel {vx}"
              f"   {gate['checks'].get('samples_across_spf')} samples across the SPF")
        for r in gate.get("refusals", []):
            print(f"           REFUSED: {r}")
        for w in gate.get("warnings", []):
            print(f"           marginal: {w}")
    print(f"  seeds:   {', '.join(report.get('seeds_found', [])) or 'none'}")
    sym = report.get("midsagittal")
    if sym:
        print(f"  midline: x = {sym['x_ras_mm']:.2f} mm, yaw {sym['yaw_deg']:+.2f} deg, "
              f"Dice {sym['dice']:.3f} "
              f"({'sharp' if sym['sharp_enough'] else 'FLAT, do not trust'})")
    for side, rec in report.get("sides", {}).items():
        print()
        print(f"  --- side {side} " + "-" * 52)
        spf = rec.get("spf", {})
        if spf.get("status") == "ok":
            warn = "  [THRESHOLD-SENSITIVE]" if spf.get(
                "aperture_threshold_sensitive") else ""
            print(f"    SPF aperture      axial {spf['aperture_axial_mm']:.2f} mm"
                  f"   coronal {spf['aperture_coronal_mm']:.2f} mm{warn}")
        gap = rec.get("conduction_gap_mm")
        if gap is not None:
            verdict = "WITHIN" if rec.get("within_budget") else "OVER"
            print(f"    CONDUCTION GAP    {gap:.2f} mm   "
                  f"(budget {B:.1f} mm -> {verdict})")
        st = rec.get("spf_station", {})
        if st.get("status") == "ok":
            print(f"    at SPF station    stent dia {st['mucosal_inscribed_diameter_mm']:.2f} mm"
                  f"   (airway component {st['component_area_mm2']:.0f} mm2)")
        cor = rec.get("corridor", {})
        if cor.get("status") == "ok":
            flag = ("" if cor.get("profile_is_smooth")
                    else f"  [JUMPY: {cor['n_discontinuities']} discontinuities]")
            src = rec.get("corridor_source", "seeded")
            print(f"    corridor length   {cor['corridor_length_mm']:.2f} mm"
                  f"   ({cor['n_stations']} stations, {src}){flag}")
            print(f"    stent diameter    min {cor['mucosal_min_diameter_mm']:.2f}"
                  f"  max {cor['mucosal_max_diameter_mm']:.2f} mm")
            print(f"    binding point     {cor['binding_point_diameter_mm']:.2f} mm"
                  f" at {cor['binding_point_t_mm']:.1f} mm from the piriform seed")
            if cor.get("compressible_margin_mm") is not None:
                print(f"    compressible      {cor['compressible_margin_mm']:.2f} mm"
                      f" (bony minus mucosal)")
        for role in ("vidian", "FR"):
            d = rec.get(f"{role}_clearance_mm")
            if d is not None:
                print(f"    {role:<8} clearance {d:.2f} mm from the SPF centre")
    mm = report.get("morphometrics") or {}
    if mm.get("status") == "ok":
        print()
        print("  --- morphometry (descriptive, NOT device constraints) " + "-" * 14)
        for sd, rec in mm.get("sides", {}).items():
            v = rec.get("nasal_airway_volume_mm3")
            print(f"    side {sd} airway volume  "
                  + (f"{v:.0f} mm3" if v is not None
                     else f"not available ({rec.get('status')})"))
        sep = mm.get("septal_deviation")
        if sep:
            print(f"    septal deviation      {sep['max_abs_mm']:.2f} mm max, "
                  f"median {sep['median_signed_mm']:+.2f} mm toward the {sep['toward']}")
        asym = mm.get("airway_asymmetry")
        if asym and asym.get("volume_ratio_larger_over_smaller"):
            print(f"    airway asymmetry      {asym['volume_ratio_larger_over_smaller']:.2f}x, "
                  f"{asym['larger_side']} larger")
    elif mm:
        print(f"\n  morphometry: {mm.get('status')}")
    if report.get("checks"):
        print()
        print("  checks:")
        for c in report["checks"]:
            print(f"    [{c['severity']}] {c['id']}: {c['detail']}")
    print("=" * 72)


print(f"(C) NINS_Stent_Constraints_0811.py v{SCRIPT_VERSION} loaded. "
      f"Nothing has run yet.")
print(f"  Seeds expected: " + ", ".join(f"{r}_R/{r}_L" for r in SEED_ROLES))
print("  Thresholds self-calibrate per scan; scans outside competence are refused.")
print(f"  Run:  report = run(); print_report(report)")
