# Solution archive format (`<stem>.h5`)

One HDF5 file per solve, written by `leap_solve` (and by every program that calls
`writeSolutionArchive`, `io/include/leap/io/solution_archive.hpp`), schema version 1.
Readers: `python/leap_solve.py` (`load_archive`), `python/report_solution.py`,
`tracking/leaptrack/archive.py`, and plain `h5py`.

The archive holds everything needed to render the trajectory, to analyze constraint
satisfaction offline, to track the reference with a controller, and to restart: the node
values of every field, the raw Bezier force control points with their preconditioner
scales, the final duals, the exact decision vector, the raw residual vector with its
row-to-term map, and the verbatim problem and robot yamls.

## Root attributes

| attribute | type | value |
|---|---|---|
| `schema_version` | i64 | `1` |
| `generator` | str | `leap_solve`, `hopper_jump`, ... |

## Datasets

`S` = number of elements (Chebyshev segments) of the final mesh. All floats are IEEE
f8; 2-D datasets read in C order with shape `(rows, cols)`. Columns of the trajectory
matrices are the CGL collocation nodes **ascending in time**.

| path | dtype / shape | contents |
|---|---|---|
| `/meta_json` | vlen str | JSON metadata blob (schema below) |
| `/segments/{s:02d}/Q` | f8 (nq, Nn) | joint positions at the nodes |
| `/segments/{s:02d}/V` | f8 (nv, Nn) | velocities |
| `/segments/{s:02d}/A` | f8 (na, Nn) | accelerations |
| `/segments/{s:02d}/U` | f8 (nu, Nn) | motor torque `u` (under torque elimination: the inverse-dynamics torque at the nodes) |
| `/segments/{s:02d}/lam_nodes` | f8 (nlam, Nn) | contact forces AT the nodes (the Bezier curve sampled there); shape (0, Nn) on an airborne element |
| `/segments/{s:02d}/lam_ctrl` | f8 (nlam, Nc) | RAW decision Bezier control points `C` (stance elements only) |
| `/segments/{s:02d}/lam_scale` | f8 (Nc,) | preconditioner `sigma`; physical control polygon = `lam_ctrl @ diag(lam_scale)`; the force curve is its Bernstein combination at `s = (tau + 1) / 2`, cone-feasible BETWEEN the nodes by convexity |
| `/segments/{s:02d}/taus` | f8 (Nn,) | collocation nodes in [-1, 1] |
| `/segments/{s:02d}/cons` | f8 (Nn, 7) | per-node max violation, columns `[dynamics, foot, boundary, ineq, other, kin1, kin2]` |
| `/contacts/active` | i8 (S, nContacts) | contact schedule (1 = active on that element), contact order = `model.contacts` |
| `/resets/max_viol` | f8 (S-1,) | per-junction max abs reset residual |
| `/duals/gamma` | f8 (md,) | final multipliers |
| `/solution/y` | f8 (np+md,) | the exact decision vector (primal, then duals) |
| `/rows/g` | f8 (md,) | raw constraint residuals at the zero-dual judging point |
| `/rows/term` | i32 (md,) | row -> term id (index into `meta.terms`) |
| `/rows/ineq` | i8 (md,) | 1 where the row is an inequality (violation = `max(0, g)`), 0 equality (`abs(g)`) |
| `/rows/mu` | f8 (md,) | final per-row penalty weights |

The `/internode` group (the dense off-node trace of `output.internode_trace`) is not
written; the key is accepted and ignored. The between-node question is answered by the solver's fine-grid judge (the
`[finegrid]` lines of the log) and, independently of the solver, by
`python/report_solution.py`.

## `meta_json` schema

```jsonc
{
  "schema_version": 1,
  "generator": "leap_solve",
  "problem": { "name": ..., "type": "contact_sequence | jump_fence_v2 | terrain_walk | custom",
               "yaml": "<verbatim problem yaml>" },
  "robot":   { "urdf": ..., "motor_model": "none | friction_inertia", "motor_parameters": {...},
               "saturation": {...}, "lag": {...}, "yaml": "<verbatim robot yaml>" },
  "solver":  { "backend": ..., "fingerprint": "<every recipe field>", "config_hash": ... },
  "overrides": ["solver.split=8", ...],             // the applied command-line overrides
  "model": { "nq", "nv", "nu", "joint_names": [...], "contacts": [{"name", "dim"}, ...], "actuated_idx": [...] },
  "collision": {                                    // only when the robot declares collision_capsules
    "capsules": [{"name", "frame", "a", "b", "radius"}, ...],   // resolved local endpoints
    "margin": 0.02,
    "obstacles": [{"name", "center", "radius"}, ...],          // spheres
    "boxes": [{"name", "center", "half", "rot"}, ...],         // cuboids: half extents, axes = columns of rot
    "enforced_capsules": [...]
  },
  "terms": ["e0/dyn", "e0/pos_stance_foot", ...],   // term id -> name (for /rows/term; "e<k>/" = element)
  "segments": [ { "index", "parent_mode",           // parent_mode = the element's phase (contact mode)
                  "label", "t0", "T",               // global start time, duration (s)
                  "Nn", "degree", "Nlam",
                  "dims": {"nq", "nv", "na", "nlam", "nu"},
                  "active_contacts": [{"name", "dim"}, ...],
                  "reset_after": "continuity" | "plastic_impact" | "none" }, ... ],
  "row_layout": { "reset_row0": [...], "np": ..., "md": ... },
  "metrics": { "gate",                 // zero-dual max|c| at the nodes (the node gate)
               "internode_dyn_max",    // the fine-grid dynamics worst
               "gnorm_final", "cost",
               "group_violations": {"dyn", "foot", "kin", "bc", "reset", "ineq"},
               "flow_b": {...} },      // the last flow's report
  "run": { "timestamp_utc", "wall_s", "omp_threads", "warnings": [] }
}
```

## Conventions

- **Time**: node `k` of segment `s` is at `t = t0_s + T_s (tau_k + 1) / 2`. Every field
  is a Chebyshev polynomial sampled at the CGL nodes; evaluate it anywhere with the
  barycentric interpolant (`report_solution.py: interp_weights`,
  `tracking/leaptrack/archive.py`).
- **Contact forces between the nodes**: the spectral interpolant of `lam_nodes` can
  overshoot the friction cone; the Bezier curve from `lam_ctrl` and `lam_scale` is what
  the optimizer constrained.
- **Reset maps**: at a `plastic_impact` junction the velocity jumps (`v+` is the
  sticking-impact image of `v-`); `continuity` junctions are smooth. A controller that
  integrates an error across a junction should reset it there.
- **Torque**: `U` is `u` in `M a + C v + g = B u + J' lam`, before any motor loss.
- **Feasibility offline**: `/rows/g`, `/rows/term`, `/rows/ineq` and `meta.terms` give
  every row's residual at the nodes; `report_solution.py` recomputes the physics on a
  fine grid from the URDF alone.

## Reading it

```python
import sys; sys.path.insert(0, "new_src/python")
from leap_solve import load_archive
sol = load_archive("results/sol_fence_go2.h5")     # every numeric dataset + sol["meta"]
Q0 = sol["segments/00/Q"]                            # (nq, Nn) of element 0
meta = sol["meta"]
print(meta["metrics"]["gate"], [s["reset_after"] for s in meta["segments"]])
```

```python
import h5py, json, numpy as np
f = h5py.File("results/sol_fence_go2.h5")
meta = json.loads(f["/meta_json"][()].decode())
seg = meta["segments"][3]
t_nodes = seg["t0"] + seg["T"] * (np.asarray(f["/segments/03/taus"]) + 1) / 2
C = f["/segments/00/lam_ctrl"][()] * f["/segments/00/lam_scale"][()][None, :]   # physical control polygon
g, term, ineq = f["/rows/g"][()], f["/rows/term"][()], f["/rows/ineq"][()]
viol = np.where(ineq > 0, np.maximum(g, 0.0), np.abs(g))
worst = {meta["terms"][t]: viol[term == t].max() for t in np.unique(term)}
```
