"""Capsule geometry shared by every comparison method: the exact residual
forms the leap builders enforce (separating-face box certificate, smooth
capsule-capsule clearance) plus segment/segment and segment/box distances
for the independent judges.

References:
  include/leap/problem/obstacles.hpp      (separating-face certificate)
  include/leap/problem/self_collision.hpp (smooth pair residual)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- box faces
def separating_face_certificate(A, B, center, half, dsafe, skip_bottom=True):
    """leap's CapsuleCuboidConstraint residual: g = dsafe - m <= 0 with
    m = max_faces min(off(A), off(B)); the -z face is skipped when the box
    stands on the ground. R = I (all leap boxes are axis-aligned).

    Returns (g, dgdA, dgdB): gradient wrt the two world endpoints (each 3,).
    """
    yA = np.asarray(A) - center
    yB = np.asarray(B) - center
    best = -np.inf
    bk, bs, bend = 0, 1.0, 0
    for k in range(3):
        for s in (1.0, -1.0):
            if skip_bottom and k == 2 and s < 0:
                continue
            oa = s * yA[k] - half[k]
            ob = s * yB[k] - half[k]
            if oa <= ob:
                m, end = oa, 0
            else:
                m, end = ob, 1
            if m > best:
                best, bk, bs, bend = m, k, s, end
    g = dsafe - best
    dgdA = np.zeros(3)
    dgdB = np.zeros(3)
    grad = np.zeros(3)
    grad[bk] = -bs
    if bend == 0:
        dgdA = grad
    else:
        dgdB = grad
    return g, dgdA, dgdB


# ------------------------------------------------- segment-segment distance
def seg_seg_closest(A1, B1, A2, B2):
    """Closest points between segments [A1,B1] and [A2,B2] (Ericson 5.1.9).
    Returns (t, s, D) with D the SQUARED distance, points A1+t*d1, A2+s*d2."""
    d1 = B1 - A1
    d2 = B2 - A2
    r = A1 - A2
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    eps = 1e-14
    if a <= eps and e <= eps:
        t = s = 0.0
    elif a <= eps:
        t = 0.0
        s = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            s = 0.0
            t = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            t = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > eps else 0.0
            s = (b * t + f) / e
            if s < 0.0:
                s = 0.0
                t = np.clip(-c / a, 0.0, 1.0)
            elif s > 1.0:
                s = 1.0
                t = np.clip((b - c) / a, 0.0, 1.0)
    p1 = A1 + t * d1
    p2 = A2 + s * d2
    diff = p1 - p2
    return t, s, float(diff @ diff)


def selfcol_residual(A1, B1, A2, B2, dsafe):
    """leap's CapsuleSelfCollisionConstraint row: g = (dsafe^2 - D)/(2 dsafe).
    Returns (g, w, t, s) where w = -(p1-p2)/dsafe is dD-chain weight so that
    dg/dq = w . (dp1/dq - dp2/dq) with the closest points held (envelope)."""
    t, s, D = seg_seg_closest(A1, B1, A2, B2)
    g = (dsafe * dsafe - D) / (2.0 * dsafe)
    p1 = A1 + t * (B1 - A1)
    p2 = A2 + s * (B2 - A2)
    w = -(p1 - p2) / dsafe
    return g, w, t, s


# ------------------------------------------------------------ exact judges
def point_box_distance(p, center, half):
    d = np.maximum(np.abs(p - center) - half, 0.0)
    inside = np.all(np.abs(p - center) <= half)
    if inside:
        return -float(np.min(half - np.abs(p - center)))
    return float(np.linalg.norm(d))


def seg_box_distance(A, B, center, half, n=64):
    """Conservative sampled min distance from segment to axis-aligned box
    (matches pyleap.collision's dense audit approach)."""
    ts = np.linspace(0.0, 1.0, n)
    pts = A[None, :] + ts[:, None] * (B - A)[None, :]
    d = np.abs(pts - center) - half
    outside = np.maximum(d, 0.0)
    dist_out = np.linalg.norm(outside, axis=1)
    inside_depth = np.max(d, axis=1)  # negative when strictly inside
    dist = np.where(inside_depth < 0.0, inside_depth, dist_out)
    return float(np.min(dist))
