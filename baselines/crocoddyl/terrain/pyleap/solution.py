"""Solution loading -- THE only module that knows the on-disk formats.

`load_solution(path_or_stem)` accepts either:
  * a `.h5` solution archive written by leap_solve / io::writeSolutionArchive
    (normative spec: docs/solution_format.md), or
  * a legacy multi-file stem: `{stem}_meta.json` + `{stem}_mode{s}.csv`
    (rows [Q;V;A;Lam;U], cols = CGL nodes ascending in time).

Both normalize into the same `SolutionData`, so every consumer (simulation,
viz, diagnostics) is format-agnostic. Update SCHEMA_SUPPORTED here -- and only
here -- when the archive schema evolves.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyleap.internode import InterNodeTraces, traces_from_h5

SCHEMA_SUPPORTED = (1,)


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
    source: str                    # "archive" | "legacy_csv"
    path: str
    extras: dict = field(default_factory=dict)  # archive-only: duals, y, rows/*
    # archive-only, and only when the solve stored one (output.internode_trace > 0):
    # the dense off-node constraint residual trace (pyleap.internode)
    internode: InterNodeTraces | None = None

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


def _mode_from_matrix(M: np.ndarray, ms: dict, t0: float) -> dict:
    d = ms["dims"]
    splits = np.cumsum([d["nq"], d["nv"], d["na"], d["nlam"]])
    Q, V, A, Lam, U = np.vsplit(M, splits)
    return dict(
        T=float(ms["T"]),
        t0=t0,
        taus=np.asarray(ms["taus"], float),
        Q=Q, V=V, A=A, Lam=Lam, U=U,
        active_contacts=tuple(c["name"] for c in ms["active_contacts"]),
        dims=dict(d),
        contact_dims=_contact_dims(ms["active_contacts"], int(d["nlam"]),
                                   "legacy meta"),
    )


def _load_legacy(stem: str) -> SolutionData:
    meta_path = Path(f"{stem}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    modes: list[ModeData] = []
    t0 = 0.0
    for s, ms in enumerate(meta["modes"]):
        M = np.loadtxt(f"{stem}_mode{s}.csv", delimiter=",", ndmin=2)
        modes.append(ModeData(**_mode_from_matrix(M, ms, t0)))
        t0 += float(ms["T"])
    return SolutionData(meta=meta, modes=tuple(modes), duration=t0,
                        source="legacy_csv", path=str(meta_path))


def _load_archive(path: Path) -> SolutionData:
    import h5py  # deferred: legacy loading must work without h5py

    with h5py.File(path, "r") as f:
        raw = f["/meta_json"][()]
        meta = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        schema = int(meta.get("schema_version", -1))
        if schema not in SCHEMA_SUPPORTED:
            raise ValueError(
                f"{path}: unsupported archive schema_version {schema} "
                f"(supported: {SCHEMA_SUPPORTED}); update pyleap/solution.py -- "
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
        internode = traces_from_h5(f, meta)
    duration = sum(m.T for m in modes)
    return SolutionData(meta=meta, modes=tuple(modes), duration=duration,
                        source="archive", path=str(path), extras=extras,
                        internode=internode)


def load_solution(path_or_stem: str | Path) -> SolutionData:
    """Load a solution from a .h5 archive path or a legacy CSV stem."""
    p = Path(path_or_stem)
    if p.suffix == ".h5" or p.is_file() and p.suffix in (".h5", ".hdf5"):
        if not p.is_file():
            raise FileNotFoundError(f"solution archive not found: {p}")
        return _load_archive(p)
    stem = str(path_or_stem)
    if Path(f"{stem}_meta.json").is_file():
        return _load_legacy(stem)
    raise FileNotFoundError(
        f"no solution at '{path_or_stem}': expected a .h5 archive or a legacy "
        f"stem with {stem}_meta.json + {stem}_mode*.csv")
