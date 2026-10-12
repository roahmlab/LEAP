"""Continuous-time reference over a loaded solution.

ReferenceTrajectory (eval / eval_full / mode_at / mode_boundaries /
active_contacts / q0 and the joint_names/nq/nu/actuated_idx/duration/modes
attributes), built on pyleap.solution.load_solution, so it reads the .h5 archive and the legacy
CSV stem alike.

lam_interp:
  "nodes"  (default) barycentric interpolation of the NODE forces
           (spectral interpolant).
  "bezier" de Casteljau on the raw Bezier control points (archive only):
           the curve the optimizer's force rows are stated on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyleap.interp import bary_row, bary_weights, bezier_eval
from pyleap.solution import ModeData, SolutionData, load_solution


class _ModeView:
    """One mode + its barycentric weights (kept ReferenceTrajectory-internal)."""

    def __init__(self, m: ModeData):
        self.data = m
        self.Q, self.V, self.A, self.Lam, self.U = m.Q, m.V, m.A, m.Lam, m.U
        self.taus = m.taus
        self.w = bary_weights(m.taus)
        self.T = m.T
        self.t0 = m.t0
        self.active_contacts = list(m.active_contacts)
        self.lam_ctrl_physical = m.lam_ctrl_physical


class ReferenceTrajectory:
    def __init__(
        self,
        sol: SolutionData | str | Path,
        *,
        lam_interp: str = "nodes",
        reconstruct_actuation: bool | None = None,
    ):
        if not isinstance(sol, SolutionData):
            sol = load_solution(sol)
        if lam_interp not in ("nodes", "bezier"):
            raise ValueError(f"lam_interp must be 'nodes' or 'bezier', got {lam_interp}")
        self.solution = sol
        self.meta = sol.meta
        self.joint_names: list[str] = sol.joint_names
        self.nq = sol.nq
        self.nu = sol.nu
        self.actuated_idx = sol.actuated_idx
        self.modes = [_ModeView(m) for m in sol.modes]
        self.duration = sol.duration
        if reconstruct_actuation is None:
            reconstruct_actuation = sol.derived_actuation
        if reconstruct_actuation and not sol.derived_actuation:
            raise ValueError(
                "reconstruct_actuation=True requires an inverse-dynamics archive"
            )
        if reconstruct_actuation:
            from pyleap.actuation import InverseDynamicsActuation

            self._actuation = InverseDynamicsActuation(sol)
        else:
            self._actuation = None
        if lam_interp == "bezier" and any(m.lam_ctrl_physical is None
                                          for m in self.modes
                                          if m.Lam.shape[0] > 0):
            raise ValueError("lam_interp='bezier' needs Bezier control points, which "
                             "only the .h5 archive carries (not the legacy CSVs)")
        self.lam_interp = lam_interp

    # --- schedule helpers ---
    def mode_at(self, t: float) -> int:
        for s, m in enumerate(self.modes):
            if t < m.t0 + m.T or s == len(self.modes) - 1:
                return s
        return len(self.modes) - 1

    @property
    def mode_boundaries(self) -> list[float]:
        return [m.t0 + m.T for m in self.modes[:-1]]

    def active_contacts(self, t: float) -> list[str]:
        return self.modes[self.mode_at(t)].active_contacts

    def contact_dims_at(self, t: float) -> tuple[int, ...]:
        """Per-active-contact lam width at time t (3 = point, 6 = wrench)."""
        m = self.modes[self.mode_at(t)].data
        return m.contact_dims or (3,) * len(m.active_contacts)

    def reset_kind_after(self, s: int) -> str | None:
        """Reset map kind between mode s and s+1 (archive only; else None)."""
        return self.solution.modes[s].reset_after

    # --- evaluation ---
    def _tau(self, m: _ModeView, t: float) -> float:
        tau = 2.0 * (t - m.t0) / m.T - 1.0
        return float(np.clip(tau, -1.0, 1.0))

    def eval(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return (q_ref, qd_ref, u_ref, mode_index) at global time t."""
        s = self.mode_at(t)
        m = self.modes[s]
        tau = self._tau(m, t)
        b = bary_row(m.taus, m.w, tau)
        q, v = m.Q @ b, m.V @ b
        if self._actuation is None:
            u = m.U @ b
        else:
            a = m.A @ b
            if self.lam_interp == "bezier" and m.Lam.shape[0] > 0:
                lam = bezier_eval(m.lam_ctrl_physical, 0.5 * (tau + 1.0))
            else:
                lam = m.Lam @ b
            u = self._actuation.eval(q, v, a, lam, m.active_contacts)
        return q, v, u, s

    def eval_full(self, t: float):
        """Return (q_ref, qd_ref, a_ref, lam_ref, u_ref, mode_index) at time t.

        lam_ref rows follow the mode's active-contact order (the same order
        active_contacts(t) returns).
        """
        s = self.mode_at(t)
        m = self.modes[s]
        tau = self._tau(m, t)
        b = bary_row(m.taus, m.w, tau)
        if self.lam_interp == "bezier" and m.Lam.shape[0] > 0:
            lam = bezier_eval(m.lam_ctrl_physical, 0.5 * (tau + 1.0))
        else:
            lam = m.Lam @ b
        q, v, a = m.Q @ b, m.V @ b, m.A @ b
        u = (
            m.U @ b
            if self._actuation is None
            else self._actuation.eval(q, v, a, lam, m.active_contacts)
        )
        return q, v, a, lam, u, s

    def lam_map(self, t: float) -> dict[str, np.ndarray]:
        """Per-contact planned force/wrench at time t: contact name -> the
        contact's rows of lam (length 3 or 6; force-first). Keys follow the
        mode's active-contact order; swing feet are absent."""
        m = self.modes[self.mode_at(t)].data
        lam = self.eval_full(t)[3]
        out: dict[str, np.ndarray] = {}
        for name, sl in m.lam_slices.items():
            if sl.stop <= lam.size:
                out[name] = np.asarray(lam[sl], float).copy()
            else:  # defensive: meta dims disagreed with nlam (loader warned)
                out[name] = np.zeros(sl.stop - sl.start)
        return out

    def q0(self) -> tuple[np.ndarray, np.ndarray]:
        """Initial (q, v) at t=0 (start of mode 0)."""
        q, v, _, _ = self.eval(0.0)
        return q, v
