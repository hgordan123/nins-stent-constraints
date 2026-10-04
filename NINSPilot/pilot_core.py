"""Numerical primitives for the assisted pilot. No Slicer or Qt dependencies."""
import hashlib
import json
import re

import numpy as np
from scipy.spatial import ConvexHull, distance

VERSION = "0.4.0"


def coded_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,64}", value):
        raise ValueError("Use a coded ID containing letters, numbers and underscores")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def side_decisions(sides):
    """An unassessable side needs a reason; it never becomes a successful side."""
    if set(sides) != {"R", "L"}:
        raise ValueError("Record both side decisions")
    for side, record in sides.items():
        if record.get("status") not in ("assessable", "unassessable"):
            raise ValueError("Choose assessable or unassessable for " + side)
        if record["status"] == "unassessable" and not record.get("reason", "").strip():
            raise ValueError("Record an unassessable reason for " + side)
    return [s for s in ("R", "L") if sides[s]["status"] == "assessable"]


def proposal_masks(array, bounds_ijk, air_max=-300., bone_min=300.):
    """HU threshold proposals, restricted to an explicit IJK ROI. No smoothing.

    Air includes all in-range voxels in the ROI, including sinus/external air.
    The operator must correct its anatomical extent; it is not a nasal label.
    """
    a = np.asarray(array)
    bounds = np.asarray(bounds_ijk)
    if a.ndim != 3 or bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("Expected a 3D volume and three finite IJK bounds")
    if not np.equal(bounds, np.floor(bounds)).all():
        raise ValueError("ROI bounds must be integer voxel indices")
    bounds = bounds.astype(int)
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] >= a.shape[::-1]) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("ROI must lie within the source grid")
    if not np.isfinite([air_max, bone_min]).all() or not -1024 < air_max < bone_min:
        raise ValueError("Use ordered finite HU thresholds above the air floor")
    slices = tuple(slice(lo, hi + 1) for lo, hi in bounds[::-1])
    crop = a[slices]
    air, bone = np.zeros(a.shape, np.uint8), np.zeros(a.shape, np.uint8)
    air[slices] = np.isfinite(crop) & (crop >= -1024) & (crop <= air_max)
    bone[slices] = np.isfinite(crop) & (crop >= bone_min)
    return air, bone


def _simple_polygon(xy):
    def orient(a, b, c):
        x, y = b - a, c - a
        return x[0] * y[1] - x[1] * y[0]
    def on_segment(a, b, p):
        return abs(orient(a, b, p)) < 1e-8 and np.all(p >= np.minimum(a, b)-1e-8) and np.all(p <= np.maximum(a, b)+1e-8)
    n = len(xy)
    for i in range(n):
        a, b = xy[i], xy[(i+1) % n]
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            c, d = xy[j], xy[(j+1) % n]
            o = [orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)]
            if (o[0]*o[1] < 0 and o[2]*o[3] < 0) or any((on_segment(a,b,c), on_segment(a,b,d), on_segment(c,d,a), on_segment(c,d,b))):
                return False
    return True


def rim_geometry(points):
    """Feret widths of convex hull in a least-squares aperture plane.

    Area is the ordered projected polygon, not its convex hull. The plane's
    residuals are reported, not silently interpreted as anatomical precision.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Provide at least three finite, ordered rim points")
    if np.allclose(p[0], p[-1], atol=1e-8):
        p = p[:-1]
    if len(p) < 3 or len(np.unique(p, axis=0)) != len(p):
        raise ValueError("Rim has duplicate or insufficient vertices")
    center = p.mean(axis=0)
    _, singular, basis = np.linalg.svd(p-center, full_matrices=False)
    if singular[1] < 1e-6:
        raise ValueError("Rim is collinear")
    xy = (p-center) @ basis[:2].T
    if not _simple_polygon(xy):
        raise ValueError("Projected rim self-intersects; review point order")
    hull = xy[ConvexHull(xy).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    widths = np.ptp(hull @ normals.T, axis=0)
    residual = (p-center) @ basis[2]
    area = abs(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1))) / 2
    return dict(projected_area_mm2=float(area), minimum_feret_mm=float(widths.min()),
                maximum_feret_mm=float(distance.pdist(hull).max()),
                plane_rms_mm=float(np.sqrt(np.mean(residual**2))),
                plane_max_residual_mm=float(abs(residual).max()),
                plane_origin_ras_mm=center.tolist(), plane_normal_ras=basis[2].tolist(),
                definition="Ordered rim projected to least-squares plane; convex-hull supporting-line Feret widths")


def synthetic_case(kind="straight"):
    """Small numerical fixtures, not simulated patients or anatomical validation."""
    if kind not in ("straight", "narrow", "artifact"):
        raise ValueError("Unknown synthetic fixture")
    k, j, i = np.indices((64, 80, 80))
    a = np.full(k.shape, 40, np.int16)
    radius = 8 if kind == "straight" else 5
    air = ((i-54)**2 + (k-32)**2 < radius**2) | ((i-26)**2 + (k-32)**2 < 8**2)
    a[air] = -1000
    shell = ((i-54)**2 + (k-32)**2 < (radius+2)**2) | ((i-26)**2 + (k-32)**2 < 10**2)
    a[shell & ~air] = 900
    if kind == "artifact":
        a[26:38, 25:40, 45:63] = 3500
    return a
