"""Physical actuator reconstruction for reduced solution parameterizations.

An archive produced with inverse-dynamics actuator elimination stores the
reconstructed torque at collocation nodes for compatibility. Interpolating
those node values is not the represented continuous-time input, however. The
true input is

    u(t) = [M(q)a + h(q, v) + motor_loss(v) - J(q)'lambda]actuated.

This module is deliberately small and archive-driven so every Python consumer
(controller reference, exporter, and independent validator) can evaluate that
same map without mirroring solver recipe choices.
"""

from __future__ import annotations

import numpy as np

from pyleap.solution import SolutionData, lam_slices


class InverseDynamicsActuation:
    """Reconstruct delivered actuator torque from one solution archive."""

    def __init__(self, sol: SolutionData):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(sol.urdf)
        self.data = self.model.createData()
        self.act = np.asarray(sol.actuated_idx, int)
        self.contacts = {
            str(c["name"]): int(c.get("dim", 3)) for c in sol.contacts
        }
        self.frame_id = {
            name: self.model.getFrameId(name) for name in self.contacts
        }
        self.lwa = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        motor = (sol.meta.get("robot") or {}).get("motor_parameters") or {}
        nu = len(self.act)

        def vector(name: str) -> np.ndarray:
            raw = np.asarray(motor.get(name, np.zeros(nu)), float).reshape(-1)
            if raw.size == 1:
                raw = np.full(nu, float(raw[0]))
            if raw.size != nu:
                raise ValueError(
                    f"archive motor.{name}: expected {nu} entries, got {raw.size}"
                )
            return raw

        self.coulomb = vector("coulomb")
        self.viscous = vector("viscous")
        self.armature = vector("armature")
        self.bias = vector("bias")
        self.velocity_smoothing = float(motor.get("velocity_smoothing", 0.05))
        self.model.armature[self.act] = self.armature

    def eval(self, q, v, a, lam, active_contacts) -> np.ndarray:
        """Return actuator-order delivered torque at one trajectory sample."""
        pin, model, data = self.pin, self.model, self.data
        q = np.asarray(q, float)
        v = np.asarray(v, float)
        a = np.asarray(a, float)
        lam = np.asarray(lam, float)
        active = list(active_contacts)

        tau = pin.rnea(model, data, q, v, a).copy()
        va = v[self.act]
        eps = self.velocity_smoothing
        smooth_sign = va / np.sqrt(va * va + eps * eps)
        tau[self.act] += (
            self.coulomb * smooth_sign + self.viscous * va + self.bias
        )

        if active:
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
            slices = lam_slices([(name, self.contacts[name]) for name in active])
            for name in active:
                dim = self.contacts[name]
                J = pin.getFrameJacobian(
                    model, data, self.frame_id[name], self.lwa
                )[:dim, :]
                tau -= J.T @ lam[slices[name]]
        return tau[self.act]

