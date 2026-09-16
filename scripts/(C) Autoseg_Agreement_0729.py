"""(C) Autoseg_Agreement_0729.py

Claude-authored. Auto versus manual agreement analysis for the SPG Anatomy
Study autosegmentation QC arm.

Runs OUTSIDE 3D Slicer, on any interpreter with numpy. scipy is used for exact
distribution quantiles when present and a verified numpy-only fallback is used
when it is not (neither /usr/bin/python3 nor PythonSlicer has pingouin,
statsmodels or pandas, and they disagree on scipy, so nothing beyond numpy is
assumed). matplotlib is optional and only affects figures.

    /Applications/Slicer.app/Contents/bin/PythonSlicer \
      "(C) Autoseg_Agreement_0729.py" --input measurements.csv --out-dir ./out

WHAT THIS DOES NOT DO, deliberately:

  * No pooling across measurements. A single Bland-Altman spanning a 4 mm
    diameter and a 45 mm length is dominated by scale and means nothing.
  * No overall ICC, for the same reason.
  * No Pearson r, no R squared, no Spearman. Correlation measures association,
    not agreement: auto = manual + 5 mm gives r = 1.00, and a 5 mm offset would
    propagate straight into a device dimension. Correlation is also driven by
    the spread of the sample, so identical error looks excellent in a
    heterogeneous cohort and poor in a homogeneous one.
  * No paired t-test as evidence of agreement. A non-significant p is absence
    of evidence, not evidence of equivalence, and it is rewarded by noisy data
    and small n. Intervals against a pre-registered margin instead.
  * Left and right are never pooled as independent observations. Two sides of
    one skull are correlated, and pooling 40 sides inflates precision. This is
    a common error in bilateral morphometry.

Input is the long-format CSV both arms emit:
    subject_id, side, arm, rep, measurement_key,
    measurement_label_verbatim, value_mm, status

  arm    : 'manual' or 'auto'
  rep    : 1..3 for manual, 0 for auto (the auto pipeline is deterministic
           and has no reps; see the robustness note in the report)
  status : 'ok', 'skipped', or a failure code. 'Skipped' is MISSING, never 0.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "0729.1"

# The 9 measurements this arm can actually assess. The other 11 derive from
# landmarks and lines alone, so the auto arm reproduces them identically by
# construction. That is not agreement and is reported separately.
CURVE_DEPENDENT_KEYS = (
    "SPG to CHEEK - AXIAL (mm)",
    "SPG to CHEEK - CORONAL (mm)",
    "SPG to NASAL CAVITY - AXIAL (mm)",
    "NASAL SEPTUM to MIDDLE MEDIAL TURB - AXIAL (mm)",
    "SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL (mm)",
    "INFERIOR TURB, SUPERIOR to NASAL FLOOR (mm)",
    "MIDDLE TURB, INF SURFACE to NASAL FLOOR - CORONAL (mm)",
    "ORBITAL FLOOR @ PPF to SPF - CORONAL, Sup/Inf (mm)",
    "ORBITAL FLOOR @ PPF to SPF - CORONAL, Summed (mm)",
)

# Pre-registered acceptance thresholds. Fix these in the protocol addendum
# BEFORE the cohort runs. Fixing them in advance is the difference between a
# QC arm and a post-hoc narrative.
VOXEL_MM = 0.5
BIAS_MAX_MM = 0.5              # one voxel
ICC_LOWER_CI_MIN = 0.75
CURVE_MEAN_DIST_MAX_MM = 0.5   # one voxel
CURVE_HD95_MAX_MM = 1.5        # three voxels
COMPLETION_MIN = 0.90
ALPHA = 0.05

# Pre-declared failure taxonomy. A method with excellent agreement on the 60%
# of cases where it works is not a validated method, so the denominator and
# the reasons are reported alongside every result.
FAILURE_TAXONOMY = (
    "deviated_septum", "concha_bullosa", "dental_amalgam_streak",
    "mucosal_thickening", "thin_palatine_bone", "ambiguous_loop",
    "other",
)


############ DISTRIBUTIONS ############################
# scipy when available, otherwise a numpy-only implementation. The fallback is
# differentially tested against scipy in (C) Autoseg Tests/test_agreement.py,
# so it is verified rather than merely hoped to be right.

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _scipy_stats = None
    _HAVE_SCIPY = False


def _betacf(a, b, x, itmax=300, eps=3.0e-14):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1.0e-30:
        d = 1.0e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _bisect_ppf(cdf, p, lo, hi, tol=1e-12, itmax=300):
    """Invert a monotone CDF by bisection, expanding the bracket as needed."""
    while cdf(hi) < p and hi < 1e12:
        hi *= 2.0
    while cdf(lo) > p and lo > -1e12:
        lo *= 2.0 if lo < 0 else 0.5
        if lo == 0:
            break
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def t_ppf(p, df):
    """Inverse Student t CDF."""
    if _HAVE_SCIPY:
        return float(_scipy_stats.t.ppf(p, df))

    def cdf(t):
        x = df / (df + t * t)
        ib = _betainc(0.5 * df, 0.5, x)
        return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib

    return _bisect_ppf(cdf, p, -100.0, 100.0)


def f_ppf(p, dfn, dfd):
    """Inverse F CDF."""
    if _HAVE_SCIPY:
        return float(_scipy_stats.f.ppf(p, dfn, dfd))

    def cdf(f):
        if f <= 0:
            return 0.0
        x = dfn * f / (dfn * f + dfd)
        return _betainc(0.5 * dfn, 0.5 * dfd, x)

    return _bisect_ppf(cdf, p, 1e-8, 100.0)


############ STATISTICS ############################


def within_subject_sd(reps_by_subject):
    """Manual repeatability from the 3 reps: s_w and the repeatability coefficient.

    s_w is the pooled within-subject SD (the square root of the one-way ANOVA
    within-subject mean square). RC = 2.77 * s_w is the value below which the
    difference between two manual measurements on the same subject lies 95% of
    the time.

    This also answers the study's standing 'point repeatability' open question
    as a by-product, and supplies the honest acceptance criterion for the QC
    arm: is the auto-versus-manual difference smaller than the manual arm's own
    rep-to-rep noise?
    """
    ss, df = 0.0, 0
    n_subjects = 0
    for reps in reps_by_subject:
        values = np.asarray([v for v in reps if v is not None], dtype=float)
        if len(values) < 2:
            continue
        ss += float(np.sum((values - values.mean()) ** 2))
        df += len(values) - 1
        n_subjects += 1
    if df == 0:
        return {"s_w": None, "rc": None, "n_subjects": n_subjects, "df": 0}
    s_w = math.sqrt(ss / df)
    return {"s_w": s_w, "rc": 2.77 * s_w, "n_subjects": n_subjects, "df": df}


def icc_a1(matrix, alpha=ALPHA):
    """ICC(A,1): two-way mixed effects, absolute agreement, single measurement.

    Two-way because both subject and method are modelled. Mixed because there
    are exactly these two fixed methods, not a sample from a rater population.
    ABSOLUTE AGREEMENT rather than consistency, because consistency (ICC(C,1),
    Shrout and Fleiss ICC(3,1)) removes systematic method bias from the
    numerator, and a systematic offset in millimetres is precisely the failure
    that matters when the output feeds device dimensions.

    matrix is (n subjects, k methods). Returns the estimate with its CI.
    Confidence limits follow McGraw and Wong (1996).
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return {"icc": None, "ci": (None, None), "n": int(data.shape[0]),
                "reason": "need at least 2 subjects and 2 methods"}
    if np.isnan(data).any():
        return {"icc": None, "ci": (None, None), "n": int(data.shape[0]),
                "reason": "missing values, listwise deletion required first"}

    n, k = data.shape
    grand = data.mean()
    row_means, col_means = data.mean(axis=1), data.mean(axis=0)

    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((data - grand) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0:
        return {"icc": None, "ci": (None, None), "n": n,
                "reason": "zero variance"}
    icc = (ms_rows - ms_error) / denominator

    lower = upper = None
    try:
        if 0 < icc < 1:
            a = (k * icc) / (n * (1 - icc))
            b = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
            v_num = (a * ms_cols + b * ms_error) ** 2
            v_den = ((a * ms_cols) ** 2 / (k - 1)
                     + (b * ms_error) ** 2 / ((n - 1) * (k - 1)))
            v = v_num / v_den if v_den > 0 else (n - 1) * (k - 1)
            f_lower = f_ppf(1 - alpha / 2, n - 1, v)
            f_upper = f_ppf(1 - alpha / 2, v, n - 1)
            lower = (n * (ms_rows - f_lower * ms_error)
                     / (f_lower * (k * ms_cols + (k * n - k - n) * ms_error)
                        + n * ms_rows))
            upper = (n * (f_upper * ms_rows - ms_error)
                     / (k * ms_cols + (k * n - k - n) * ms_error
                        + n * f_upper * ms_rows))
            lower, upper = max(-1.0, lower), min(1.0, upper)
    except (ValueError, ZeroDivisionError, OverflowError):
        lower = upper = None

    return {"icc": icc, "ci": (lower, upper), "n": n, "k": k,
            "ms_rows": ms_rows, "ms_cols": ms_cols, "ms_error": ms_error,
            "form": "ICC(A,1) two-way mixed, absolute agreement, single measure"}


def bland_altman(auto_values, manual_means, s_w=None, alpha=ALPHA):
    """Bland-Altman limits of agreement, with the CI on the limits themselves.

    At n = 20 the limits are genuinely imprecise, roughly plus or minus 0.8 SD
    each, so reporting bare point limits would overstate the result. Bland and
    Altman suggest around 100 observations for tight limits; the study has 20,
    so the honest move is to show the interval.

    When s_w is supplied, also reports the limits against a SINGLE manual
    measurement rather than the 3-rep mean, since that is how a clinician would
    actually use it: Var(d_single) = Var(d_observed) + (2/3) * s_w^2.
    """
    auto = np.asarray(auto_values, dtype=float)
    manual = np.asarray(manual_means, dtype=float)
    diff = auto - manual
    mean = (auto + manual) / 2.0
    n = len(diff)
    if n < 2:
        return {"n": n, "reason": "need at least 2 paired observations"}

    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    t_crit = t_ppf(1 - alpha / 2, n - 1)

    se_bias = sd / math.sqrt(n)
    loa_low, loa_high = bias - 1.96 * sd, bias + 1.96 * sd
    se_loa = math.sqrt(3.0 * sd * sd / n)

    result = {
        "n": n,
        "bias": bias,
        "bias_ci": (bias - t_crit * se_bias, bias + t_crit * se_bias),
        "sd_diff": sd,
        "loa_lower": loa_low,
        "loa_upper": loa_high,
        "loa_lower_ci": (loa_low - t_crit * se_loa, loa_low + t_crit * se_loa),
        "loa_upper_ci": (loa_high - t_crit * se_loa, loa_high + t_crit * se_loa),
        "loa_width": loa_high - loa_low,
        "rmsd": float(np.sqrt(np.mean(diff ** 2))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "bias_in_voxels": bias / VOXEL_MM,
    }

    if s_w:
        var_single = sd * sd + (2.0 / 3.0) * s_w * s_w
        sd_single = math.sqrt(var_single)
        result["sd_diff_vs_single_manual"] = sd_single
        result["loa_vs_single_manual"] = (bias - 1.96 * sd_single,
                                          bias + 1.96 * sd_single)

    # Proportional bias: does the difference scale with magnitude? If so the
    # limits should be computed on the log scale for that measurement instead.
    if n >= 3 and float(np.std(mean)) > 0:
        slope, intercept = np.polyfit(mean, diff, 1)
        residuals = diff - (slope * mean + intercept)
        dof = n - 2
        if dof > 0:
            se_slope = math.sqrt(
                float(np.sum(residuals ** 2)) / dof
                / float(np.sum((mean - mean.mean()) ** 2)))
            t_slope = t_ppf(1 - alpha / 2, dof)
            result["proportional_bias_slope"] = float(slope)
            result["proportional_bias_slope_ci"] = (
                float(slope - t_slope * se_slope),
                float(slope + t_slope * se_slope))
            result["proportional_bias_present"] = bool(
                (slope - t_slope * se_slope) * (slope + t_slope * se_slope) > 0)
    return result


def curve_distances(points_a, points_b):
    """Geometric agreement between two curves: mean nearest neighbour and HD95.

    Required, not optional. The derived scalar can be right for the wrong
    reason: two visibly different curves can share a nearest point to the SPG
    centroid and produce identical numbers. The curve distance is what says
    whether the algorithm traced the same anatomy. Without it the arm is a
    coincidence detector.
    """
    a = np.asarray(points_a, dtype=float)
    b = np.asarray(points_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return {"mean_nn": None, "hd95": None, "hd_max": None}
    deltas = a[:, None, :] - b[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    a_to_b = distances.min(axis=1)
    b_to_a = distances.min(axis=0)
    both = np.concatenate([a_to_b, b_to_a])
    return {
        "mean_nn": float(both.mean()),
        "hd95": float(np.percentile(both, 95)),
        "hd_max": float(both.max()),
        "n_points_auto": int(len(a)),
        "n_points_manual": int(len(b)),
    }


############ INPUT ############################


def read_long_csv(path):
    """Read the long-format measurement CSV. 'Skipped' is missing, never 0."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            status = (row.get("status") or "").strip().lower()
            raw = (row.get("value_mm") or "").strip()
            value = None
            if status in ("ok", "") and raw and raw.lower() != "skipped":
                try:
                    value = float(raw)
                except ValueError:
                    value = None
            rows.append({
                "subject_id": (row.get("subject_id") or "").strip(),
                "side": (row.get("side") or "").strip().upper(),
                "arm": (row.get("arm") or "").strip().lower(),
                "rep": int(row["rep"]) if (row.get("rep") or "").strip().isdigit() else 0,
                "key": (row.get("measurement_label_verbatim")
                        or row.get("measurement_key") or "").strip(),
                "value": value,
                "status": status or "ok",
                "failure_reason": (row.get("failure_reason") or "").strip(),
            })
    return rows


def organise(rows):
    """Index as [key][side][subject] -> {'manual': [r1,r2,r3], 'auto': v}."""
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"manual": {}, "auto": None})))
    for row in rows:
        if not row["key"] or not row["subject_id"]:
            continue
        cell = table[row["key"]][row["side"]][row["subject_id"]]
        if row["arm"] == "manual":
            cell["manual"][row["rep"]] = row["value"]
        elif row["arm"] == "auto":
            cell["auto"] = row["value"]
    return table


############ ANALYSIS ############################


def analyse_measurement(subjects, alpha=ALPHA):
    """Full agreement analysis for one measurement on one side."""
    manual_reps, manual_means, auto_values, used, dropped = [], [], [], [], []

    for subject_id in sorted(subjects):
        cell = subjects[subject_id]
        reps = [cell["manual"].get(r) for r in (1, 2, 3)]
        present = [v for v in reps if v is not None]
        if present:
            manual_reps.append(reps)
        if not present or cell["auto"] is None:
            dropped.append({
                "subject_id": subject_id,
                "reason": ("no manual value" if not present
                           else "no auto value"),
            })
            continue
        manual_means.append(float(np.mean(present)))
        auto_values.append(float(cell["auto"]))
        used.append(subject_id)

    repeatability = within_subject_sd(manual_reps)
    result = {
        "n_paired": len(used),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "subjects_used": used,
        "manual_repeatability": repeatability,
        "completion_rate": (len(used) / (len(used) + len(dropped))
                            if (used or dropped) else 0.0),
    }
    if len(used) < 2:
        result["reason"] = "fewer than 2 paired observations"
        return result

    result["manual_mean"] = float(np.mean(manual_means))
    result["auto_mean"] = float(np.mean(auto_values))
    result["bland_altman"] = bland_altman(
        auto_values, manual_means, repeatability.get("s_w"), alpha)
    result["icc"] = icc_a1(
        np.column_stack([manual_means, auto_values]), alpha)

    # The criterion that actually decides this: is the auto-versus-manual
    # difference smaller than the manual arm's own rep-to-rep noise? Far less
    # n-hungry than pinning down the limits at n = 20, and more persuasive.
    ba, rc = result["bland_altman"], repeatability.get("rc")
    verdict = {}
    verdict["bias_within_one_voxel"] = abs(ba["bias"]) <= BIAS_MAX_MM
    verdict["loa_within_manual_rc"] = (
        bool(ba["loa_width"] <= rc) if rc else None)
    icc_lower = result["icc"].get("ci", (None, None))[0]
    verdict["icc_lower_ci_ok"] = (
        bool(icc_lower >= ICC_LOWER_CI_MIN) if icc_lower is not None else None)
    verdict["completion_ok"] = result["completion_rate"] >= COMPLETION_MIN
    decided = [v for v in verdict.values() if v is not None]
    verdict["passes_all_decidable"] = bool(decided) and all(decided)
    result["verdict"] = verdict
    return result


############ REPORTING ############################


def _fmt(value, places=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    return f"{value:.{places}f}"


def _ci(pair, places=3):
    if not pair or pair[0] is None or pair[1] is None:
        return "n/a"
    return f"[{pair[0]:.{places}f}, {pair[1]:.{places}f}]"


def render_markdown(analysis, meta):
    """Per measurement and side. Never pooled."""
    out = []
    out.append("# Auto versus manual agreement")
    out.append("")
    out.append(f"Generated by `(C) Autoseg_Agreement_0729.py` v{SCRIPT_VERSION}. "
               f"Input `{os.path.basename(meta['input'])}`.")
    out.append("")
    out.append("Manual is the primary dataset. The autosegmentation arm is "
               "validation only and no auto value enters the recorded dataset "
               "until this analysis passes.")
    out.append("")
    out.append("## Scope")
    out.append("")
    out.append("This arm assesses **9 of the 20** measurements. The remaining "
               "11 derive from landmarks and lines alone, so the auto arm "
               "reproduces them identically by construction. That is not "
               "agreement and they are excluded rather than reported as "
               "perfect.")
    out.append("")
    out.append("It is also a **curve tracing** arm, not an independent "
               "measurement arm: the SPG centroid and all landmarks are reused "
               "from the manual scene, and 4 of the 9 anchor on that centroid.")
    out.append("")
    out.append("Left and right are reported separately. Two sides of one skull "
               "are correlated, so pooling them as independent observations "
               "would inflate precision.")
    out.append("")

    for key in CURVE_DEPENDENT_KEYS:
        by_side = analysis.get(key)
        if not by_side:
            continue
        out.append(f"## {key}")
        out.append("")
        header = ("| Side | n | Manual mean | s_w | RC | Auto mean | Bias "
                  "| Bias 95% CI | LoA | LoA width | ICC(A,1) | ICC 95% CI "
                  "| Completion | Verdict |")
        out.append(header)
        out.append("|" + "---|" * 14)
        for side in ("R", "L"):
            res = by_side.get(side)
            if not res:
                continue
            if res.get("n_paired", 0) < 2:
                out.append(f"| {side} | {res.get('n_paired', 0)} | "
                           + "n/a | " * 11 + "insufficient data |")
                continue
            ba, icc = res["bland_altman"], res["icc"]
            rep = res["manual_repeatability"]
            verdict = ("PASS" if res["verdict"]["passes_all_decidable"]
                       else "REVIEW")
            out.append(
                f"| {side} | {res['n_paired']} | {_fmt(res['manual_mean'],2)} "
                f"| {_fmt(rep.get('s_w'))} | {_fmt(rep.get('rc'))} "
                f"| {_fmt(res['auto_mean'],2)} | {_fmt(ba['bias'])} "
                f"| {_ci(ba['bias_ci'])} "
                f"| [{_fmt(ba['loa_lower'],2)}, {_fmt(ba['loa_upper'],2)}] "
                f"| {_fmt(ba['loa_width'],2)} | {_fmt(icc.get('icc'))} "
                f"| {_ci(icc.get('ci'))} "
                f"| {res['completion_rate']:.0%} | {verdict} |")
        out.append("")

        for side in ("R", "L"):
            res = by_side.get(side)
            if not res or res.get("n_paired", 0) < 2:
                continue
            ba = res["bland_altman"]
            notes = []
            notes.append(
                f"limits of agreement are themselves imprecise at n = "
                f"{ba['n']}: lower limit CI {_ci(ba['loa_lower_ci'],2)}, "
                f"upper limit CI {_ci(ba['loa_upper_ci'],2)}")
            if "loa_vs_single_manual" in ba:
                notes.append(
                    f"against a SINGLE manual measurement rather than the "
                    f"3-rep mean, limits widen to "
                    f"{_ci(ba['loa_vs_single_manual'],2)}")
            if ba.get("proportional_bias_present"):
                notes.append(
                    f"**proportional bias detected** (slope "
                    f"{_fmt(ba['proportional_bias_slope'])}, CI "
                    f"{_ci(ba['proportional_bias_slope_ci'])}); recompute this "
                    f"measurement on the log scale and report ratio limits")
            if res["verdict"].get("loa_within_manual_rc") is False:
                notes.append(
                    "limits are WIDER than the manual arm's own repeatability "
                    "coefficient, so the auto curve is not behaving like "
                    "another manual pass")
            if res["n_dropped"]:
                notes.append(f"{res['n_dropped']} subject(s) dropped: "
                             + ", ".join(f"{d['subject_id']} ({d['reason']})"
                                         for d in res["dropped"]))
            out.append(f"**{side}:** " + "; ".join(notes) + ".")
            out.append("")

    out.append("## Not assessable by this arm")
    out.append("")
    out.append("These 11 measurements come from landmarks and lines only. The "
               "auto arm consumes the same manual inputs, so any apparent "
               "agreement is arithmetic identity, not validation.")
    out.append("")
    for key in meta.get("not_assessable", []):
        out.append(f"- {key}")
    out.append("")
    out.append("## Interpretation notes")
    out.append("")
    out.append("- **ICC is never the primary read.** It is a variance ratio, "
               "so for a measurement with a narrow anatomical range across a "
               "healthy cohort it looks poor even at sub-millimetre error, and "
               "flattering for a wide-range measurement at the same error. "
               "Bias, limits of agreement and RMSD decide; ICC supports.")
    out.append("- **The honest criterion** is whether the auto-versus-manual "
               "difference is smaller than the manual arm's own rep-to-rep "
               "noise (RC column). If it is, the auto curve is behaving like "
               "another manual pass.")
    out.append("- **A discrepancy under one voxel "
               f"({VOXEL_MM} mm) is not meaningfully resolvable** and should "
               "be reported as such rather than as agreement.")
    out.append("- **Auto-arm robustness replaces repeatability.** Re-running "
               "is deterministic, so reporting that as perfect repeatability "
               "would be dishonest. Its real analogue is sensitivity to "
               "upstream choices: re-crop the ROI, shift the slice by plus or "
               "minus one, change the window and level, and report the spread.")
    out.append("- **Auto curves are denser and linear** where manual curves "
               "are sparse and spline, which makes closest-point searches find "
               "systematically shorter minima. Small but systematic, so it "
               "belongs in the bias term rather than the limits.")
    return "\n".join(out)


def write_summary_csv(analysis, path):
    fields = ["measurement", "side", "n_paired", "manual_mean", "s_w", "rc",
              "auto_mean", "bias", "bias_ci_low", "bias_ci_high", "sd_diff",
              "loa_lower", "loa_upper", "loa_width", "loa_vs_single_low",
              "loa_vs_single_high", "rmsd", "icc_a1", "icc_ci_low",
              "icc_ci_high", "completion_rate", "proportional_bias",
              "verdict"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in CURVE_DEPENDENT_KEYS:
            for side in ("R", "L"):
                res = (analysis.get(key) or {}).get(side)
                if not res or res.get("n_paired", 0) < 2:
                    continue
                ba, icc = res["bland_altman"], res["icc"]
                rep = res["manual_repeatability"]
                single = ba.get("loa_vs_single_manual", (None, None))
                writer.writerow({
                    "measurement": key, "side": side,
                    "n_paired": res["n_paired"],
                    "manual_mean": res["manual_mean"], "s_w": rep.get("s_w"),
                    "rc": rep.get("rc"), "auto_mean": res["auto_mean"],
                    "bias": ba["bias"], "bias_ci_low": ba["bias_ci"][0],
                    "bias_ci_high": ba["bias_ci"][1], "sd_diff": ba["sd_diff"],
                    "loa_lower": ba["loa_lower"], "loa_upper": ba["loa_upper"],
                    "loa_width": ba["loa_width"],
                    "loa_vs_single_low": single[0],
                    "loa_vs_single_high": single[1], "rmsd": ba["rmsd"],
                    "icc_a1": icc.get("icc"),
                    "icc_ci_low": icc.get("ci", (None, None))[0],
                    "icc_ci_high": icc.get("ci", (None, None))[1],
                    "completion_rate": res["completion_rate"],
                    "proportional_bias": ba.get("proportional_bias_present"),
                    "verdict": ("PASS" if res["verdict"]["passes_all_decidable"]
                                else "REVIEW"),
                })


def write_figures(analysis, out_dir):
    """Small-multiple Bland-Altman panels, one per measurement and side.

    Optional: matplotlib is absent from PythonSlicer, so this is skipped with a
    note rather than being allowed to fail the run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"written": 0, "skipped": "matplotlib not available"}

    written = 0
    for key in CURVE_DEPENDENT_KEYS:
        by_side = analysis.get(key) or {}
        panels = [(s, r) for s, r in sorted(by_side.items())
                  if r.get("n_paired", 0) >= 2]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5),
                                 squeeze=False)
        for ax, (side, res) in zip(axes[0], panels):
            ba = res["bland_altman"]
            pairs = res.get("_pairs")
            if pairs:
                a, m = np.array(pairs["auto"]), np.array(pairs["manual"])
                ax.scatter((a + m) / 2, a - m, s=28, alpha=0.8,
                           edgecolor="k", linewidth=0.4)
            for level, style, label in (
                (ba["bias"], "-", f"bias {ba['bias']:.2f}"),
                (ba["loa_upper"], "--", f"+1.96 SD {ba['loa_upper']:.2f}"),
                (ba["loa_lower"], "--", f"-1.96 SD {ba['loa_lower']:.2f}"),
            ):
                ax.axhline(level, linestyle=style, linewidth=1.2, label=label)
            ax.axhspan(ba["loa_lower_ci"][0], ba["loa_lower_ci"][1],
                       alpha=0.12, color="grey")
            ax.axhspan(ba["loa_upper_ci"][0], ba["loa_upper_ci"][1],
                       alpha=0.12, color="grey")
            ax.set_title(f"{side}  (n={ba['n']})", fontsize=10)
            ax.set_xlabel("mean of auto and manual (mm)")
            ax.set_ylabel("auto minus manual (mm)")
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(key, fontsize=11)
        fig.tight_layout()
        slug = "".join(c if c.isalnum() else "_" for c in key)[:60]
        fig.savefig(os.path.join(out_dir, f"ba_{slug}.png"), dpi=150)
        plt.close(fig)
        written += 1
    return {"written": written}


############ MAIN ############################


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto versus manual agreement analysis for the SPG "
                    "Anatomy Study QC arm.")
    parser.add_argument("--input", required=True,
                        help="long-format measurement CSV")
    parser.add_argument("--out-dir", default=".", help="output directory")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_long_csv(args.input)
    table = organise(rows)

    all_keys = sorted({r["key"] for r in rows if r["key"]})
    not_assessable = [k for k in all_keys if k not in CURVE_DEPENDENT_KEYS]

    analysis = {}
    for key in CURVE_DEPENDENT_KEYS:
        if key not in table:
            continue
        analysis[key] = {
            side: analyse_measurement(subjects, args.alpha)
            for side, subjects in table[key].items()
        }
        for side, subjects in table[key].items():
            pairs = {"auto": [], "manual": []}
            for subject_id in analysis[key][side].get("subjects_used", []):
                cell = subjects[subject_id]
                present = [v for v in
                           (cell["manual"].get(r) for r in (1, 2, 3))
                           if v is not None]
                pairs["auto"].append(cell["auto"])
                pairs["manual"].append(float(np.mean(present)))
            analysis[key][side]["_pairs"] = pairs

    meta = {"input": args.input, "not_assessable": not_assessable,
            "script_version": SCRIPT_VERSION, "scipy": _HAVE_SCIPY,
            "alpha": args.alpha, "n_rows": len(rows)}

    report_path = os.path.join(args.out_dir, "agreement_report.md")
    with open(report_path, "w") as handle:
        handle.write(render_markdown(analysis, meta))
    summary_path = os.path.join(args.out_dir, "agreement_summary.csv")
    write_summary_csv(analysis, summary_path)
    json_path = os.path.join(args.out_dir, "agreement_full.json")
    with open(json_path, "w") as handle:
        json.dump({"meta": meta, "analysis": analysis}, handle,
                  indent=2, default=str)

    figures = ({"written": 0, "skipped": "disabled"} if args.no_figures
               else write_figures(analysis, args.out_dir))

    print(f"rows read:        {len(rows)}")
    print(f"measurements:     {len(analysis)} of 9 assessable present")
    print(f"not assessable:   {len(not_assessable)}")
    print(f"exact quantiles:  {'scipy' if _HAVE_SCIPY else 'numpy fallback'}")
    print(f"report:           {report_path}")
    print(f"summary:          {summary_path}")
    print(f"full json:        {json_path}")
    print(f"figures:          {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
