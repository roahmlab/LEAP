"""Barycentric (CGL) and Bezier evaluation helpers.

Each mode's Q/V/A/U columns are a Chebyshev polynomial sampled at
Chebyshev-Gauss-Lobatto nodes (ascending in time); the exact interpolant is
the barycentric formula. Contact forces under the Bezier parameterization are
Bernstein polynomials of their control points.
"""

from __future__ import annotations

import numpy as np


def cgl_bary_weights(taus: np.ndarray) -> np.ndarray:
    """Closed-form barycentric weights for a CGL grid (alternating +-1, ends halved)."""
    w = np.ones(len(taus))
    w[1::2] = -1.0
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def generic_bary_weights(taus: np.ndarray) -> np.ndarray:
    """O(n^2) product-form weights for an arbitrary node set (non-CGL fallback)."""
    n = len(taus)
    w = np.ones(n)
    for j in range(n):
        d = taus[j] - np.delete(taus, j)
        w[j] = 1.0 / np.prod(d)
    return w


def is_cgl(taus: np.ndarray, tol: float = 1e-9) -> bool:
    n = len(taus)
    if n < 2:
        return False
    k = np.arange(n)
    ref = -np.cos(np.pi * k / (n - 1))  # ascending CGL on [-1, 1]
    return bool(np.max(np.abs(np.asarray(taus) - ref)) < tol)


def bary_weights(taus: np.ndarray) -> np.ndarray:
    """Weights for `taus`: closed-form when the grid is CGL, generic otherwise."""
    taus = np.asarray(taus, float)
    return cgl_bary_weights(taus) if is_cgl(taus) else generic_bary_weights(taus)


def bary_row(taus: np.ndarray, w: np.ndarray, tau: float) -> np.ndarray:
    """Row b with f(tau) = b @ f(taus); exact-hit guarded."""
    d = tau - taus
    hit = np.isclose(d, 0.0, atol=1e-14)
    if hit.any():
        b = np.zeros_like(taus)
        b[np.argmax(hit)] = 1.0
        return b
    b = w / d
    return b / b.sum()


def bezier_eval(ctrl: np.ndarray, s: float) -> np.ndarray:
    """De Casteljau evaluation of a Bezier curve at s in [0, 1].

    ctrl is (dim, nc): column j = control point j. Returns the (dim,) value.
    """
    pts = np.array(ctrl, dtype=float, copy=True)
    nc = pts.shape[1]
    for r in range(1, nc):
        pts[:, : nc - r] = (1.0 - s) * pts[:, : nc - r] + s * pts[:, 1 : nc - r + 1]
    return pts[:, 0]
