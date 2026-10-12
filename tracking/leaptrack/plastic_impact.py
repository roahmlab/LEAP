"""The optimizer's plastic impact map, evaluated on a MuJoCo model.

`plastic_impact_map` is the rollout's touchdown diagnostic.

Reference implementation of what the trajectory optimizer applies at every
touchdown (`RobotModel::impactVelocity`: pinocchio `impulseDynamics` with
restitution coefficient 0 and a proximal damping mu = 1e-9 on the Delassus
matrix, one iteration):

    Lambda = (J M^{-1} J^T + eps I)^{-1} J v-          impulse (sticking)
    v+     = v- - M^{-1} J^T Lambda                    = P v-

All rows of every impacting contact are constrained (3 per point contact, 6 per
flat foot): the impact is perfectly inelastic AND sticking.
"""
from __future__ import annotations

import numpy as np


def plastic_impact_map(M: np.ndarray, J: np.ndarray, v_minus: np.ndarray,
                       eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Return (v_plus, impulse) of the sticking plastic impact with contact
    Jacobian `J` (k x nv) and mass matrix `M` (nv x nv)."""
    if J.shape[0] == 0:
        return v_minus.copy(), np.zeros(0)
    MinvJT = np.linalg.solve(M, J.T)
    D = J @ MinvJT + eps * np.eye(J.shape[0])
    imp = np.linalg.solve(D, J @ v_minus)
    return v_minus - MinvJT @ imp, imp
