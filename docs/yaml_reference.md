# Problem yaml reference

One file poses one problem. Every key can be overridden on the command line as
`section.key=value` (sequences as comma-separated values, e.g. `solver.lsqnqs=8,8,8`).
Unknown keys are rejected (a typo never silently becomes a default), except in the
`solver:` block of the fence and terrain types, where unknown keys are ignored.

```yaml
schema: 1
name: hopper2d_jump_fence          # the archive stem defaults to sol_<name>
robot: ../robots/hopper2d.yaml      # relative to this file
problem: {type: contact_sequence | jump_fence_v2 | terrain_walk, ...}
obstacles: {...}                    # fence family only
terrain: {...}                      # terrain family only
solver: {...}                       # transcription knobs (degree, control points, split, defect weight)
objective: {...}                    # optional: the composed objective (replaces the builder's cost knobs)
unified: {...}                      # the method (defaults: solver/include/leap/solver/recipe.hpp)
output: {stem: ..., archive: true, finegrid_dt: 0.01, finegrid_dt2: 0.001}
```

## `problem:` for `type: contact_sequence` (tasks/contact_sequence.hpp)

The generic problem type: a prescribed contact sequence. You prescribe the
contact modes (which contacts carry force, for how long), where each contact stands, the
rest poses at the two ends and the contact model; the builder attaches the benchmarks' rows,
splits every mode into graded elements, builds the cold seed and runs the staged
pipeline through `StaticFamily` (adaptive h-refinement included; no obstacle
continuation pre-stage). Any robot the robot yaml describes, any number of contacts, any
sequence: `problems/examples/hopper2d_jump.yaml` (a jump in place),
`go2_jump.yaml` (a forward jump of a quadruped), `go2_step.yaml` (one step with a
three-foot support). `docs/TUTORIAL.md` section 1 walks through one.

| key | default | meaning |
|---|---|---|
| `modes` | required | the sequence: `[{contacts: [names], T: s, feet: [{contact, pos}], shift: [dx, dy, dz], label}]`. `contacts` are the contact frames that carry force in the mode (empty = flight); `T` its duration; `feet` places contacts ACTIVE in the mode (default: where they were); `shift` is added to the previous placement of every contact that LANDS in this mode (a forward jump: `shift: [0.3, 0, 0]` on the landing mode) |
| `pose_start` | required | `[{joint, angle}]`: the joint angles at t = 0; the base is derived (x = y = 0, the lowest contact on z = 0, or on `start`) |
| `pose_end` | [] | the joint angles at t = T; the base follows the final placements of the last mode's contacts, which the pose must stand on up to a base translation (else an error says so). Empty: the end is at rest in a free pose |
| `start` | [] | `[{contact, pos}]`: explicit placements during the first mode (must be the foot positions of `pose_start` up to a base translation) |
| `rest_start`, `rest_end` | true, true | pin the whole configuration (`pose_start` / `pose_end`) and zero velocity at that end; `rest_end` with no `pose_end` pins the velocity only |
| `mu`, `friceps` | 0.8, 1e-2 | friction coefficient, cone apex smoothing |
| `lammin` | 5 | per-contact normal-force floor in stance (N) on the control points |
| `landvz` | 0.5 | touchdown speed cap per landing foot (m/s, <= 0 off) |
| `copx`, `copy`, `gammaz` | 0.08, 0.03, 0.05 | flat-foot (dim 6) CoP box and yaw-torque bound |
| `impactcone` | false | friction-cone admissibility of the plastic impact impulse at every touchdown |
| `landramp` | false | a landed contact's wrench is pinned to zero at the first control point after the touchdown, the floor from the second |
| `hop` | 0 | a pure flight (no contact): minimum liftoff base speed `sqrt(2 g hop)` at the node before it (<= 0 off); the validation checks the flight is long enough |
| `footclear` | 0.03 | the feet that land in the next mode stay above this height at the quarter points of the mode (`ineq_swing_z`, a point condition the fine-grid judge and the relocation leave alone; <= 0 off) |
| `ulim`, `vlim`, `qbox` | 1, 1, 1 | URDF limit boxes scaled (<= 0 turns the box off) |
| `ubern` | false | torque box on the Bernstein coefficients of each element (certified between the nodes) |
| `attmax` | 0.5 | base roll / pitch / yaw box (rad; <= 0 off) |
| `baseheight`, `groundclearance`, `mineffort` | true, true, true | base z >= 0; monitor frames above the floor (a stance foot's own monitor exempt); the effort cost |
| `planarlock` | [] | joints pinned to zero in q, v and a at every node (sagittal robots described by a 3-D URDF; the dynamics keeps the in-plane rows) |
| `selfcolmargin`, `selfcolinterp`, `selfcolpairs` | 0, 0, [] | capsule self-collision rows (`[{a, b, margin}]`) |
| `noslipnq` | 3 | off-node no-slip points per stance element (movable) |
| `seedapex` | -1 | mid-swing / mid-flight lift of the seed (m; < 0 = `hop`, else `footclear`) |

`solver:` for this type: `deg` (4), `nc` (4), `split` (4, elements per mode, a stance
mode graded by `hgrade` (0.6) toward its contact changes, a flight uniform), `lsqnq` (6,
least-squares dynamics-defect points per element), `rho` (required: the defect weight
and the unit of every penalty, 3e6 for a 12 kg hopper, 3e8 for a humanoid), `precond`
(true). `obstacles:` as for the fence (static rows at the nodes and at `interp` points;
there is no continuation, so the seed should not cross them). `objective:` as below.

Resets: a plastic impact of the new contact set wherever a contact becomes active,
continuity otherwise. Rows per mode, with their names: the header of
`tasks/include/leap/tasks/contact_sequence.hpp`.

## `problem:` for `type: jump_fence_v2` (tasks/fence_jump.hpp)

| key | default | meaning |
|---|---|---|
| `tstance`, `tair`, `tstanceland` | 0.5, 0.45, 0 (= tstance) | push-off, flight and landing durations (s) |
| `hop` | 0.18 | hop height -> minimum liftoff vertical speed `sqrt(2 g hop)` |
| `step` | [0.6, 0] | base displacement over the flight (m) |
| `mu`, `friceps` | 0.85, 1e-2 | friction coefficient, cone apex smoothing |
| `lammin`, `lammax` | 5, 0 (off) | per-contact normal-force floor / cap (N) on the control points |
| `landvz` | 0.5 | touchdown speed cap per foot (m/s, <= 0 off) |
| `copx`, `copy`, `gammaz` | 0.08, 0.03, 0.05 | flat-foot (dim 6) CoP box and yaw-torque bound |
| `footclear` | 0.05 | mid-flight monitor clearance at the flight quarter points |
| `mineffort` | true | the effort objective |
| `wforce`, `wvel`, `wacc`, `wbaseatt`, `wbaseomega` | 0 | extra cost weights (dimensionless, robot-scaled) |
| `ulim`, `vlim`, `qbox` | 1, 1, 1 | URDF limit boxes scaled (<= 0 turns the box off) |
| `ubern` | false | torque box on the Bernstein coefficients of each element (certified between nodes) |
| `attmax`, `attvmax`, `liftoffatt`, `liftoffomega` | 0.5, 0, 0, 0 | base attitude / rate boxes |
| `poses` | required | `[{joint, stand, squat}]`: the two rest poses by joint name; base height from FK |
| `planarlock` | [] | joints pinned to zero at every node (sagittal robots) |
| `strictrest`, `restbox` | true, 0 | full pose + zero velocity pins at both ends (or a box of half-width restbox) |
| `endpointaccel` | false | zero acceleration at both ends |
| `baseheight`, `groundclearance` | true, true | base z >= 0 rows; monitor frames above the floor |
| `ownership` | true | the touchdown event rows (landing foot pin, touchdown velocity rows) are owned once, by the flight's last node; false repeats them on the landing element's first node. This is the mode-boundary rule only: inside a mode the sub-element split always uses the unique-owner rule of `problem/mesh_refinement.hpp` |
| `liftofffull`, `pinliftoff` | true, true | pin the whole wrench (or fz only) to zero at liftoff |
| `landramp` | false | the terrain's landing ramp: the whole wrench is pinned to zero at the first control point of the first landing element and the floor `lammin` starts at the second. Off in the fence benchmark problems: there the floor holds from the first landing control point on and the impact is governed by `impactcone` and `landvz` |
| `contactposall`, `stanceacc` | false, false | full foot pins at every stance node; stance acceleration rows |
| `landfootpin`, `landjointpin`, `holdlanding` | true, false, false | touchdown placement pin, joint posture pin, landing joints held |
| `zerolandxy`, `landvxy` | false, 0 | tangential touchdown velocity pinned / bounded |
| `impactcone` | false | friction-cone admissibility of the plastic impact impulse |
| `actsymmetry` | false | left/right actuated joint symmetry rows |
| `noslipnq`, `qboxnq`, `vboxnq`, `dynnq` | 3, 0, 0, 0 | off-node points per element for no-slip, joint box, speed box, hard dynamics rows |
| `swingpathnq`, `swingprogressnq` | 0, 0 | flight foot path / progress rows at global flight fractions |
| `liftoffxmax`, `landingxslack`, `posepathslack` | -1 (off) | optional boxes on the liftoff x, the landing side, the joint path |
| `selfcolmargin`, `selfcolinterp`, `selfcolpairs` | 0, 0, [] | capsule self-collision rows (`[{a, b, margin}]`) |
| `seedmode`, `seedapex`, `linearseed`, `staticseed` | 1, -1, false, false | the cold seed (0 static hold, 1 piecewise with a flight bump, 3 linear chord) |
| `implicitu`, `usamples` | false, 0 | inverse-dynamics torque elimination and its off-node torque-box points |
| `lsqmodemask`, `lsqgauss` | 7, false | which modes carry the LS dynamics defect (bits: stance 1, flight 4); Gauss points |

`obstacles:` `margin` (0.02), `interp` (off-node points per element), `rounded` (exact
rounded capsule-box distance outside), `smooth` (log-sum-exp temperature, 0 = exact),
`capsules` / `capsule_margins` (which capsules, per-capsule margins), `spheres:
[{name, center, radius}]`, `boxes: [{name, center, size, rpy}]`.

## `problem:` for `type: terrain_walk` (tasks/terrain_walk.hpp)

| key | default | meaning |
|---|---|---|
| `tstart` | 0.4 | initial support duration (starts at rest) |
| `start` | required | `[{contact, pos}]`: initial placement of EVERY contact |
| `steps` | required | `[{tswing, tsupport, feet: [{contact, pos, clearz}]}]` |
| `nominal` | [] | `[{joint, angle}]`: the IK reference pose (not a constraint) |
| `mu`, `friceps`, `lammin`, `landvz`, `copx`, `copy`, `gammaz` | as the fence | contact model |
| `stancepin` | true | spot-pin every stance foot at every node |
| `zeroland`, `zerolandxy`, `landramp`, `landcont` | true, false, false, false | touchdown velocity pinned (all / tangential), landed force ramps from zero (whole wrench pinned at the first control point of the first post-touchdown element, floor from the second), velocity-free touchdowns as continuity junctions |
| `junctionownership` | plain | how the sub-element split re-attaches a mode's node rows. `plain`: an `AllNodes` row lands on both sides of every interior junction, so a row reading only q, v is stated twice there, each copy with its own multiplier and penalty term. `unique` = the generic rule of `problem/mesh_refinement.hpp` (such a row once, on the left child's last node; what the fence and `StaticFamily` use). The original mode-to-mode junctions are as authored either way |
| `approachfrac`, `liftfrac` | 0, 0 | level vertical approach / lift windows as fractions of the swing |
| `monitorz` | 0 | floor height for the monitor frames |
| `ulim`, `ulimjoint`, `vlim`, `qbox`, `attmax`, `attbox`, `basevbox` | 1, [], 1, 1, 0.5, [], [] | hardware and attitude boxes (`ulimjoint: [{joint, scale}]`) |
| `posture`, `basezfrac` | [], 0 | named joint boxes (`{joint, lo, hi}` or `{joint, center, band}`), base-height floor |
| `wvel` | 0 | joint-motion cost weight |
| `noslipnq`, `noslipacc` | 3, false | off-node no-slip points per support phase; acceleration rows there too |
| `vboxnq`, `attnq`, `qboxnq`, `uboxnq`, `uhull` | 0 | movable off-node box points; Bernstein-hull torque box degree |
| `planarlock`, `selfcolmargin`, `selfcolinterp`, `selfcolpairs` | as the fence | |
| `clearzscale`, `lsqmodemask` | 1, 7 | continuation probe; which modes carry the LS defect (support 1, partial 2, flight 4) |

`terrain:` `margin` (0.03), `topslack` (0.02), `topinset` (0 = derived), `interp` (2),
`windowpad` (0 = every pillar on every mode), `capsules` (empty = all), `pillars:
[{name, center: [x, y], size: [lx, ly], top}]`.

## `solver:` (transcription, both families)

| key | default | meaning |
|---|---|---|
| `deg`, `nc` | 8, 7 | polynomial degree per element, force control points |
| `split` | 2 | elements per mode (fence: `splits: [k0, k1, k2]`, `degs`, `lsqnqs` per mode) |
| `hgrade`, `hgradeends` | 1, false | fence: geometric grading of the stance elements toward the contact change |
| `lsqnq`, `rho` (or `rhoabs`, or `wdyn` x `mumax`) | 4, 0 | LS dynamics-defect points per element and weight |
| `precond` | true | force control points scaled by `sigma_j` |
| `elimu`, `elimuq` | false, 0 | terrain: inverse-dynamics torque elimination and its off-node torque points |
| `dynnq` | 0 | hard off-node dynamics rows |

## `objective:` (tasks/include/leap/tasks/objective.hpp)

Optional. When present it replaces the task builder's own cost knobs (`mineffort`,
`wvel`, `wacc`, `wforce`, `wbaseatt`, `wbaseomega`); absent, the builder's costs stand
(the benchmark problems). What the solver does with it is `unified.objective` (below).

```yaml
objective:
  quadrature: clenshaw_curtis   # f = sum_e (T_e/2) sum_k w_k l(z_k), the quadrature of int l dt (default); `sum` = plain node sum
  scale: 1.0                    # the solver minimizes scale x f (same minimizer)
  terms:
    - {type: effort, weight: 1.0}                             # 1/2 w ||u||^2;  weights: [per actuated joint] instead of weight
    - {type: joint_velocity, weight: 0.5, joints: [..]}       # 1/2 w sum_j v_j^2 (default joints: the actuated ones)
    - {type: joint_acceleration, weight: 0.1, joints: [..]}   # 1/2 w sum_j a_j^2
    - {type: posture, weight: 1.0, joints: [..], target: [..]} # 1/2 w sum_j (q_j - target_j)^2 (default target: 0)
    - {type: contact_force, weight: 1e-4}                     # 1/2 w ||lam||^2 on stance elements
  # any term: phases: [k, ...] restricts it to those parent contact modes (default: every element)
```

Weights are raw (the builders' robot scaling, e.g. `(effort limit / velocity limit)^2`
for `wvel`, is the user's business here).

## `unified:` (the method; solver/include/leap/solver/recipe.hpp)

Penalties are relative to `rho` (`mucrel` 0.1, `mumaxrel`
1, `a0murel` 1/3, `premurel` 0.1). The main knobs: `tol` (1e-3), `finedt` (0.01), `gn`
(true), `sharejunction` (true), `activemargin` (2e-2), the step controller (`h0`, `hmin`,
`hmax`, `hgrow`, `hshrink`, `ratiolow`, `ratiohigh`, `solveretry`, `blowup`), the
windows (`dsseg`, `window`, `stall`, `hcollapse`, `tolflow`), the pre-stage (`pre`,
`preladder`, `pregate`, `presmax`, `presteps`), phase A0 (`a0*`), phase C (`mugrow`,
`dualstep`, `csmax`, `csteps`), the level stop (`finefrac`, `finefraccap`, `fineplateau`),
relocation (`relocate`, `rounds`, `reloc*`) and refinement (`levels`, `theta`,
`maxsplits`, `splitlo`, `splithi`, `abortfail`). `threads` bounds the evaluator's OpenMP
team (0 = `OMP_NUM_THREADS`).

Further options:

| key | default | meaning |
|---|---|---|
| `objective` | none | `none`: feasibility mode (phase C on the restoration twin, the objective reported only). `posed`: phase C descends the posed objective + the augmented penalty |
| `objstop` | stationary | objective mode's end on a judged mesh: `stationary` = first-order certificate (below) + objective plateau; `feasibility` = the criterion alone |
| `kkttol` | 1e-3 | certificate: relative reduced gradient of `L_A` and relative complementarity (`alm/first_order.hpp`) |
| `objtol`, `objwindows` | 1e-3, 2 | objective plateau: relative change <= `objtol` over `objwindows` boundaries (also what counts as objective progress against the max-violation plateau exit) |
| `gn`, `gna0`, `exactpd` | true, true, false | Gauss-Newton metric; with `gn=false` the exact-Newton metric in phase C (`gna0=false`: everywhere), stepped by LDLT or, with `exactpd`, by Cholesky with `h` shrunk until `I + h H` is positive definite (then raise `solveretry`, lower `hmin`) |
| `backend` | block_schur | `block_schur` (alias `cholmod_sn`), `sparse_cholmod`, `sparse_llt`, `sparse_ldlt`, `dense` (`linalg/linear_step.hpp`) |
| `cholmod` | true | CHOLMOD for the sparse factorizations when compiled in |
| `dualreset` | false | after each multiplier update of phase C, restart the Nesterov momentum, re-anchor the look-ahead at the new iterate and raise `h` to at least the opening step. Off: the step history carries across the update |
| `junctionmean` | true | at the entry of every flow run, project the iterate onto the junction quotient by the group MEAN (the orthogonal projection in the full coordinates). `false` snaps every member of a junction group to the tail node's value instead |
| `mucarry` | false | carry the per-term penalties (and their stall history) across the relocation rounds of a level. Off: every phase C call, round or level, re-seeds every term at `mucrel x rho` and clears the stall history. A level start always re-seeds |

`unified.objective=effort` and `objscale` are rejected
with a pointer to the `objective:` block; `statrate`, `stoprate`, `objdenom` and
`objstallrel` are accepted at their defaults only.

## `output:`

`stem` (archive path without extension), `archive` (true), `finegrid_dt` (judge grid, s),
`finegrid_dt2` (a second, reported-only grid, 0 = off). `internode_trace` and
`full_internode` are accepted and ignored (the dense trace is not written).

`snapshots` (a directory; empty = off): the budget-study candidates (Fig. 3)
(`solver/include/leap/solver/snapshots.hpp`). Every accepted step of every stage is
watched; whenever the wall clock crosses a budget of the log grid `snapshot_perdecade`
(24) budgets per decade from `snapshot_tmin` (0.0421696 s) to `snapshot_tmax`
(14398.79 s), the incumbent (the iterate with the smallest posed zero-dual node gate so
far) is written as an ordinary solution archive if it changed since the last write; it is
also written before every mesh change and the final iterate is always written last.
`snapshot_seed` (true) writes the cold seed as snapshot 0. The directory gets
`snap_NNNN.h5`, `snapshots.csv` (index, file, wall_s, step, stage, level, gate, why),
`steps.csv` (every accepted step's wall and gate) and `summary.txt`. The writer does not
change the flow; the cost is one values-only evaluation of the posed rows per step
during the obstacle rungs and a smooth-certificate phase A0, and an archive write at
each grid crossing.
`tracking/scripts/budget_study.py` plans, rolls out and plots them.
