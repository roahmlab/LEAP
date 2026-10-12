"""Dense off-node constraint traces (`/internode` in the .h5 archive).

Spectral collocation pins each constraint only AT the CGL nodes; between them
the polynomial interpolant carries a residual. `leap_solve` with
`output.internode_trace: N` evaluates EVERY throughout constraint of every
segment on the interpolants at N interior taus plus the nodes (C++
`leap::io::interNodeViolations`, the optimizer's own `con->value()`), and stores
the signed worst row per (term, sample). This module turns that into
time-indexed traces; the plotting front end is
`simulation.diagnostics.plot_constraints`.

Sign convention (mirrors the C++ probe):
  inequality (g <= 0 feasible): worst = max_r g(r) -- negative is slack margin,
                                positive is the violation
  equality:                     worst = max_r |g(r)| -- violation as stored

so `violation` and (for inequalities) `margin` both come from one array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TermTrace:
    """One constraint term on one segment, sampled along the interpolant."""

    name: str
    group: str
    ineq: bool
    segment: int
    t: np.ndarray         # (nSamp,) global time, ascending
    worst: np.ndarray     # (nSamp,) signed worst row (see module docstring)
    at_node: np.ndarray   # (nSamp,) bool: the sample IS a collocation node
    # Where the optimizer actually CONSTRAINED this term on this segment, as
    # attachment loci: any of "first", "interior", "last", "allnodes", "interp".
    # A term is evaluated everywhere but only enforced there.
    enforced_at: tuple[str, ...] = ()
    # (nSamp,) bool: this sample is an enforcement point (a node the term is
    # attached at). Off-node samples are never enforcement points -- that is the
    # whole point of the trace.
    enforced: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.enforced is None:  # unknown loci -> treat every node as enforced
            object.__setattr__(self, "enforced", self.at_node.copy())

    @property
    def violation(self) -> np.ndarray:
        """Violation metric >= 0: max(0, worst) for inequalities, worst for equalities."""
        return np.maximum(self.worst, 0.0) if self.ineq else self.worst

    @property
    def margin(self) -> np.ndarray | None:
        """Signed slack of an inequality (positive = feasible with room); None for eq."""
        return -self.worst if self.ineq else None

    def between(self) -> np.ndarray:
        """The violation at the off-node samples only."""
        return self.violation[~self.at_node]

    def at_nodes(self) -> np.ndarray:
        """The violation at the ENFORCED collocation nodes (what the solver's gate
        sees). Nodes the term is not attached at are excluded -- see unenforced()."""
        return self.violation[self.at_node & self.enforced]

    def unenforced(self) -> np.ndarray:
        """The violation at node samples where this term is deliberately NOT
        enforced (e.g. a contact acceleration at the liftoff node). Free by
        construction."""
        return self.violation[self.at_node & ~self.enforced]


@dataclass(frozen=True)
class InterNodeTraces:
    samples: int                    # off-node samples per segment (the yaml knob)
    traces: tuple[TermTrace, ...]   # one per (segment, term), segment-major

    @property
    def term_names(self) -> list[str]:
        """Distinct term names, in first-appearance order."""
        seen: dict[str, None] = {}
        for tr in self.traces:
            seen.setdefault(tr.name, None)
        return list(seen)

    @property
    def group_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for tr in self.traces:
            seen.setdefault(tr.group, None)
        return list(seen)

    def by_term(self, name: str) -> list[TermTrace]:
        """All per-segment traces of one term, ordered by time."""
        return sorted((tr for tr in self.traces if tr.name == name),
                      key=lambda tr: tr.t[0])

    def by_group(self, group: str) -> list[TermTrace]:
        return sorted((tr for tr in self.traces if tr.group == group),
                      key=lambda tr: tr.t[0])

    def worst_of(self, name: str) -> tuple[float, float]:
        """(at-node max, between-node max) violation of one term over all segments."""
        parts = self.by_term(name)
        at = max((float(p.at_nodes().max(initial=0.0)) for p in parts), default=0.0)
        bet = max((float(p.between().max(initial=0.0)) for p in parts), default=0.0)
        return at, bet


def traces_from_h5(f, meta: dict) -> InterNodeTraces | None:
    """Build InterNodeTraces from an open h5py file + its parsed meta_json.

    Returns None when the solve did not store a trace (`output.internode_trace: 0`).
    Called by pyleap.solution._load_archive -- the archive layout stays there and
    here, nowhere else.
    """
    spec = meta.get("internode")
    if not spec:
        return None
    segs = meta["segments"]
    out: list[TermTrace] = []
    for s, seg_spec in enumerate(spec["segments"]):
        terms = seg_spec.get("terms", [])
        if not terms:
            continue
        g = f[f"/internode/{s:02d}"]
        taus = np.asarray(g["taus"][()], float)
        at_node = np.asarray(g["at_node"][()], bool)
        worst = np.asarray(g["worst"][()], float)      # (nTerms, nSamp)
        t0, T = float(segs[s]["t0"]), float(segs[s]["T"])
        t = t0 + 0.5 * T * (taus + 1.0)
        order = np.argsort(t, kind="stable")           # dense block then nodes -> sort
        at_sorted = at_node[order]
        node_pos = np.flatnonzero(at_sorted)           # sample index of node k
        for c, term in enumerate(terms):
            where = tuple(term.get("enforced_at", ()))
            enf = np.zeros(at_sorted.size, bool)
            if "allnodes" in where:
                enf[node_pos] = True
            if "interior" in where and node_pos.size > 2:
                enf[node_pos[1:-1]] = True
            if "first" in where and node_pos.size:
                enf[node_pos[0]] = True
            if "last" in where and node_pos.size:
                enf[node_pos[-1]] = True
            if not where:  # pre-`enforced_at` archive: assume every node counts
                enf[node_pos] = True
            out.append(TermTrace(name=term["name"], group=term["group"],
                                 ineq=bool(term["ineq"]), segment=s,
                                 t=t[order], worst=worst[c][order],
                                 at_node=at_sorted, enforced_at=where, enforced=enf))
    return InterNodeTraces(samples=int(spec.get("samples", 0)), traces=tuple(out))
