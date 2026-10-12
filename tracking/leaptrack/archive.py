"""LEAP solution archive (.h5) reader and its continuous-time evaluator.

Contents: barycentric (CGL) + Bezier evaluation, `lam_slices`, `ModeData`, `SolutionData`,
`load_solution` (the .h5 archive only; `SolutionData.internode` is always None),
`InverseDynamicsActuation` (archives solved with torque elimination,
`parameterization.derived_actuation`) and `ReferenceTrajectory`.

The archive layout is documented in `new_src/docs/solution_format.md`.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_SUPPORTED = (1,)


# ---------------------------------------------------------------- interp.py
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


# -------------------------------------------------------------- solution.py
def lam_slices(active_contacts) -> dict[str, slice]:
    """Per-contact row slices into a mode's stacked `lam` vector.

    `active_contacts` entries may be contact dicts (`{"name": ..., "dim": ...}`,
    the archive/meta form; `dim` defaults to 3), `(name, dim)` pairs, or bare
    names (dim 3). Rows are force-first per contact: a dim-6 flat-foot wrench is
    `[fx, fy, fz, tx, ty, tz]`, so `lam[sl][:3]` is always the force.
    """
    out: dict[str, slice] = {}
    off = 0
    for c in active_contacts:
        if isinstance(c, dict):
            name, dim = c["name"], int(c.get("dim", 3))
        elif isinstance(c, (tuple, list)):
            name, dim = c[0], int(c[1])
        else:
            name, dim = c, 3
        out[str(name)] = slice(off, off + dim)
        off += dim
    return out


@dataclass(frozen=True)
class ModeData:
    T: float
    t0: float                      # global start time of the mode
    taus: np.ndarray               # (Nn,) collocation nodes in [-1, 1], ascending
    Q: np.ndarray                  # (nq, Nn)
    V: np.ndarray                  # (nv, Nn)
    A: np.ndarray                  # (na, Nn)
    Lam: np.ndarray                # (nlam, Nn) contact forces AT the nodes
    U: np.ndarray                  # (nu, Nn)
    active_contacts: tuple[str, ...]
    dims: dict
    Lam_ctrl: np.ndarray | None = None    # (nlam, Nc) raw Bezier ctrl pts (archive only)
    Lam_scale: np.ndarray | None = None   # (Nc,) preconditioner sigma (archive only)
    cons: np.ndarray | None = None        # (Nn, 7) per-node max violations (archive only)
    reset_after: str | None = None        # "continuity" | "plastic_impact" | "none"
    contact_dims: tuple[int, ...] = ()    # per active contact (3 = point, 6 = wrench)

    @property
    def lam_slices(self) -> dict[str, slice]:
        """Contact name -> rows of Lam (force-first; see module fn lam_slices)."""
        dims = self.contact_dims or (3,) * len(self.active_contacts)
        return lam_slices(list(zip(self.active_contacts, dims)))

    @property
    def lam_ctrl_physical(self) -> np.ndarray | None:
        """Physical Bezier control polygon (nlam, Nc); force curve = its
        Bernstein combination at s = (tau+1)/2."""
        if self.Lam_ctrl is None:
            return None
        s = self.Lam_scale if self.Lam_scale is not None else np.ones(self.Lam_ctrl.shape[1])
        return self.Lam_ctrl * s[None, :]


@dataclass(frozen=True)
class SolutionData:
    meta: dict                     # the raw parsed metadata (format-specific superset)
    modes: tuple[ModeData, ...]
    duration: float
    source: str                    # "archive"
    path: str
    extras: dict = field(default_factory=dict)  # archive-only: duals, y, rows/*
    internode: object = None       # always None here (the /internode trace reader is not ported)

    # --- format-agnostic accessors ---
    @property
    def _model(self) -> dict:
        return self.meta.get("model", self.meta)

    @property
    def joint_names(self) -> list[str]:
        return list(self._model["joint_names"])

    @property
    def nq(self) -> int:
        return int(self._model["nq"])

    @property
    def nv(self) -> int:
        return int(self._model["nv"])

    @property
    def actuated_idx(self) -> np.ndarray:
        return np.asarray(self._model["actuated_idx"], int)

    @property
    def nu(self) -> int:
        return len(self._model["actuated_idx"])

    @property
    def contacts(self) -> list[dict]:
        return list(self._model["contacts"])

    @property
    def urdf(self) -> str:
        if "urdf" in self.meta:
            return self.meta["urdf"]
        return self.meta["robot"]["urdf"]

    @property
    def metrics(self) -> dict | None:
        return self.meta.get("metrics")

    @property
    def derived_actuation(self) -> bool:
        """Whether continuous-time torque is the archived inverse-dynamics map.

        The override fallback reads experimental schema-1 archives written
        before the explicit parameterization metadata field was added.
        """
        param = self.meta.get("parameterization") or {}
        if bool(param.get("derived_actuation", False)):
            return True
        return any(
            str(x).lower() in ("solver.elimu=true", "solver.elimu=1")
            for x in (self.meta.get("overrides") or ())
        )


def _contact_dims(active_contacts, nlam: int, where: str) -> tuple[int, ...]:
    """Per-contact lam widths from the meta contact entries (default 3). A
    mismatch against nlam is suspicious but must not kill loading."""
    dims = tuple(int(c.get("dim", 3)) if isinstance(c, dict) else 3
                 for c in active_contacts)
    if dims and sum(dims) != nlam:
        warnings.warn(f"{where}: contact dims {dims} sum to {sum(dims)} "
                      f"but nlam={nlam}; per-contact lam slices may be wrong")
    return dims


def _load_archive(path: Path) -> SolutionData:
    import h5py

    with h5py.File(path, "r") as f:
        raw = f["/meta_json"][()]
        meta = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        schema = int(meta.get("schema_version", -1))
        if schema not in SCHEMA_SUPPORTED:
            raise ValueError(
                f"{path}: unsupported archive schema_version {schema} "
                f"(supported: {SCHEMA_SUPPORTED}); update leaptrack/archive.py -- "
                f"the single place that knows the format")
        modes: list[ModeData] = []
        for seg in meta["segments"]:
            g = f[f"/segments/{seg['index']:02d}"]
            kw = dict(
                T=float(seg["T"]),
                t0=float(seg["t0"]),
                taus=np.asarray(g["taus"][()], float),
                Q=g["Q"][()], V=g["V"][()], A=g["A"][()],
                Lam=g["lam_nodes"][()], U=g["U"][()],
                active_contacts=tuple(c["name"] for c in seg["active_contacts"]),
                dims=dict(seg["dims"]),
                reset_after=seg.get("reset_after"),
                contact_dims=_contact_dims(seg["active_contacts"],
                                           int(seg["dims"]["nlam"]),
                                           f"segment {seg['index']}"),
            )
            if "lam_ctrl" in g:
                kw["Lam_ctrl"] = g["lam_ctrl"][()]
                kw["Lam_scale"] = g["lam_scale"][()]
            if "cons" in g:
                kw["cons"] = g["cons"][()]
            modes.append(ModeData(**kw))
        extras = {}
        for k in ("duals/gamma", "solution/y", "rows/g", "rows/term", "rows/ineq",
                  "rows/mu", "contacts/active", "resets/max_viol"):
            if k in f:
                extras[k] = f[k][()]
    duration = sum(m.T for m in modes)
    return SolutionData(meta=meta, modes=tuple(modes), duration=duration,
                        source="archive", path=str(path), extras=extras,
                        internode=None)


def load_solution(path: str | Path) -> SolutionData:
    """Load a solution from a .h5 archive path."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"solution archive not found: {p}")
    if p.suffix not in (".h5", ".hdf5"):
        raise FileNotFoundError(f"no solution at '{path}': expected a .h5 archive "
                                "(the legacy CSV stem loader is not part of this package)")
    return _load_archive(p)


# ------------------------------------------------------------- actuation.py
class InverseDynamicsActuation:
    """Reconstruct delivered actuator torque from one solution archive.

        u(t) = [M(q)a + h(q, v) + motor_loss(v) - J(q)'lambda]actuated.
    """

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
            if raw.size == 0:
                # new_src's leap_solve echoes the robot yaml's motor vectors
                # UN-expanded (empty under `motor_model: none`), where the old
                # solver wrote nu zeros: an absent coefficient is zero.
                raw = np.zeros(nu)
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


# ------------------------------------------------------------- reference.py
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
    """Continuous-time reference over a loaded solution.

    lam_interp:
      "nodes"  barycentric interpolation of the NODE forces (spectral interpolant;
               can overshoot the friction cone between nodes).
      "bezier" de Casteljau on the raw Bezier control points: cone-feasible between
               nodes by convexity, matching what the optimizer actually enforced.
    """

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
            self._actuation = InverseDynamicsActuation(sol)
        else:
            self._actuation = None
        if lam_interp == "bezier" and any(m.lam_ctrl_physical is None
                                          for m in self.modes
                                          if m.Lam.shape[0] > 0):
            raise ValueError("lam_interp='bezier' needs Bezier control points, which "
                             "only the .h5 archive carries")
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
        """Reset map kind between mode s and s+1."""
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
