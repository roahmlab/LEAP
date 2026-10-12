# Writing your own multi-mode trajectory optimization problem

There are two ways to pose a problem. The first needs no C++: a yaml of problem type
`contact_sequence` names the contact modes, the durations, the foot placements, the rest
poses and the contact model, and `leap_solve` does the rest (section 1). The second is
the formulation API in C++, for rows and seeds the yaml type does not express
(sections 2 to 10): it builds, solves, visualizes, verifies and reports the same problem
as the yaml, the planar hopper jumping in place, in `examples/hopper_jump.cpp` (about 350
lines). Both end in the same solution archive and the same report.

Contents

1. A problem as a yaml: the contact-sequence type
2. What a problem is made of (the C++ API)
3. Importing your robot
4. Posing the problem: elements, rows, costs, resets
5. The mesh
6. The cold seed
7. Solving
8. Judging, reporting, visualizing, verifying
9. Tuning tips
10. The hopper jump in C++, end to end

---

## 1. A problem as a yaml: the contact-sequence type

A prescribed contact sequence with its switching times is a yaml. The planar hopper jumping in place, `problems/examples/hopper2d_jump.yaml`:

```yaml
schema: 1
name: hopper2d_jump
robot: ../../robots/hopper2d.yaml

problem:
  type: contact_sequence
  modes:                       # the contact sequence: which contacts carry force, for how long
    - {label: push_off, contacts: [foot], T: 0.8}
    - {label: flight,   contacts: [],     T: 0.391}
    - {label: landing,  contacts: [foot], T: 0.8}     # lands where it took off
  pose_start:                  # rest pose at t = 0: joint angles (the base follows from the feet)
    - {joint: hip,  angle: -0.4982}
    - {joint: knee, angle: 0.9}
  pose_end:                    # rest pose at t = T
    - {joint: hip,  angle: -0.2202}
    - {joint: knee, angle: 0.4}
  hop: 0.17                    # the flight must leave the ground with vz >= sqrt(2 g hop)
  footclear: 0.03              # the foot clears the floor mid-flight
  mu: 0.85                     # friction
  lammin: 5.0                  # normal-force floor in stance (N)
  landvz: 0.5                  # touchdown speed cap (m/s)
  attmax: 0.5                  # |pitch| <= 0.5 rad
  ubern: true                  # torque box certified between the nodes
  planarlock: [base_py, base_rx, base_rz]   # a sagittal robot in a 3-D URDF

solver:                        # the transcription
  deg: 4                       # polynomial degree per element
  nc: 4                        # force control points
  split: 6                     # elements per mode (graded 0.6 toward the contact changes)
  hgrade: 0.6
  lsqnq: 6                     # least-squares dynamics-defect points per element
  rho: 3.0e6                   # the defect weight and the unit of every penalty
```

```bash
OMP_NUM_THREADS=12 build/leap_solve problems/examples/hopper2d_jump.yaml output.stem=results/hopper2d_jump
python python/report_solution.py results/hopper2d_jump.h5 --out results/hopper2d_jump_report --mu 0.85
```

What the builder does with it (every row by name: the header of
`tasks/include/leap/tasks/contact_sequence.hpp`): one element chain per mode (split into
`split` graded elements), the dynamics at every node, the stance feet pinned where they
stand and not slipping (at the nodes and at `noslipnq` points between them), the
friction cone and the normal-force floor on the Bezier control points of the force,
the force tapered to zero where a contact lifts off, the landing placement, the
touchdown speed cap and (with `impactcone`) the admissibility of the plastic impulse at
every touchdown, the hardware boxes from the URDF, the base attitude box, the ground
keep-out of the monitor frames, the swing feet's clearance at the quarter points of the
swing, the rest pins at both ends, the effort cost and the dynamics defect; a plastic
impact reset wherever a contact becomes active, continuity otherwise. The seed is an
inverse-kinematics pose at every mode boundary, a quadratic Bezier through a lifted
mid-swing pose for the swing feet (and the base in a flight), forces from statics and
torques from inverse dynamics. Then the pipeline runs: phase A0, phase C per
mesh level, the judges, the adaptive bisection.

The log ends with the summary line (`met=1` when the node gate and the fine grid are
below `unified.tol`), the archive is `results/hopper2d_jump.h5`, and the report
re-verifies it with pinocchio (section 8):

```
[unified] robot=hopper2d type=contact_sequence pipeline=unified met=... gate=... fine=... levels=... ... total=... solvewall=... ...
```

Other sequences are other `modes` lists. A quadruped's forward jump
(`problems/examples/go2_jump.yaml`): the four feet push off, a flight, the four feet
land 0.3 m ahead:

```yaml
  modes:
    - {label: push_off, contacts: [FL_foot, FR_foot, RL_foot, RR_foot], T: 0.5}
    - {label: flight,   contacts: [],                                   T: 0.4}
    - {label: landing,  contacts: [FL_foot, FR_foot, RL_foot, RR_foot], T: 0.5, shift: [0.3, 0.0, 0.0]}
```

One step on three feet (`problems/examples/go2_step.yaml`), the end pose left free (no
`pose_end`: the robot ends at rest with every foot on its placement):

```yaml
  modes:
    - {label: stand,   contacts: [FL_foot, FR_foot, RL_foot, RR_foot], T: 0.4}
    - {label: swing,   contacts: [FR_foot, RL_foot, RR_foot],          T: 0.4}
    - {label: support, contacts: [FL_foot, FR_foot, RL_foot, RR_foot], T: 0.4, shift: [0.15, 0.0, 0.0]}
```

`shift` moves every contact that lands in that mode relative to where it stood; `feet:
[{contact, pos}]` places a contact absolutely. Every key, with its default, is in
`docs/yaml_reference.md`. Things the type does not do, and the C++ API does: rows at
one specific node (a liftoff attitude, a mid-flight posture), a different degree per
mode, a swing trajectory shape, obstacle continuation, torque elimination, or your own
constraint class. The two benchmark builders (`tasks/fence_jump.cpp`,
`tasks/terrain_walk.cpp`) show what a task-specific builder with those devices looks
like.

Some advice that applies to every yaml problem:

* **Durations and the hop.** A flight of `T` seconds carries at most a hop of
  `(g T / 2)^2 / (2 g)`; the validation refuses a `hop` the flight cannot carry. Stances
  of 0.4 to 0.8 s are typical.
* **`rho`.** The least-squares dynamics defect weight is also the unit of every penalty
  of the method (phase C seeds every term at `0.1 rho` and grows it to `rho`): 3e6 for
  a 12 kg hopper, 3e8 for a 50 to 90 kg humanoid.
* **Poses.** The rest poses must stand on the placements: `pose_end` is checked against
  the final placements up to a base translation. Omit `pose_end` to leave the end pose
  free.

---

## 2. What a problem is made of (the C++ API)

A trajectory is a chain of **elements**. Each element is one Chebyshev segment of
polynomial degree `degree` (`degree + 1` collocation nodes) over a duration `T`, with a
fixed **contact set** (the frames that carry force on it). On an element the decision
variables are the node values of `q`, `v`, `a` and `u`, plus the Bezier **control
points** of each contact force (`Nc` of them, so the force is a polynomial of degree
`Nc - 1` whose feasibility is certified between the nodes by convexity). Consecutive
elements are joined by a **reset map**: `ContinuityReset` (same `q`, same `v`) or
`PlasticImpactReset` (same `q`, `v+` the post-impact velocity of the contacts that land).

Elements that belong to the same contact mode share a `phase` index. A problem has a
`model`, the elements, the resets, and that is all (`problem/trajectory_problem.hpp`).

Constraints and costs are **terms** attached to elements:

| call | what it does |
|---|---|
| `el.addNode(con, Where::FirstNode / LastNode / Interior / AllNodes)` | the rows of `con` at those collocation nodes |
| `el.addNodeAt(con, k)` | at node `k` only |
| `el.addOffNode(con, taus)` | the rows at points `taus` in (-1, 1) *between* the nodes, on the interpolated state (movable by the relocation pass) |
| `el.addForceControlPoints(con, {j, ...})` | a force-only row on control points `j` of the Bezier polygon (friction cone, normal-force floor, liftoff pin) |
| `el.addFieldFunctionals(con, Field::U, rows)` | the rows on linear functionals `sum_k W_k field_k` of one nodal field (e.g. the Bernstein coefficients of the torque polynomial) |
| `el.addCost(cost)` | a node cost summed over the element's nodes |
| `el.setDefectCost(dyn, taus, rho)` | the least-squares dynamics defect `(rho/2) sum_m ||g_dyn(tau_m)||^2` between the nodes |

A `Constraint` (`problem/constraint.hpp`) is written against the physical per-node
vector `z = [q; v; a; lam; u]` and a `ModelEvalCache` the model fills for that node. It
declares its name (the **term**; every term gets its own penalty and shows up by name in
the diagnostics), its rows, its sense (`g = 0` or `g <= 0`), which model quantities it
reads (`modelNeeds()`), and supplies `value` and `jacobian`. You almost never write one:
the catalogue (`problem/constraints.hpp`, `obstacles.hpp`, `self_collision.hpp`) covers
dynamics (full, floating-base rows, selected rows), frame position / velocity /
acceleration / orientation and their bounds, ground keep-out of the monitor frames,
pins and boxes on any field, the friction cone, wrench boxes, the impact cone, motor
envelopes, capsule obstacles and self-collision. Costs (`costs.hpp`): control effort,
joint velocity, joint acceleration, joint posture, contact force.

Two conventions matter:

* **Names drive everything.** Rows with the same name form one term (one penalty, one
  line in the diagnostics). The diagnostics group names by prefix: `dyn*` dynamics,
  `pos_* / vel_* / acc_* / pin_foot* / pin_sole* / pin_land*` foot, `pin*` boundary,
  `ineq* / box*` inequality. The fine-grid judge skips `pin*` rows (node-only by
  nature) and the relocation never moves them. Keep to these prefixes.
* **Junction ownership.** At the junction between two elements of the same mode, a row
  that reads only `q` and `v` should be stated once (the quotient identifies the two
  nodes' `q`, `v`). `Where::Interior` + `Where::LastNode` on the left element and
  `Where::Interior` on the right does that; `splitElement` applies the same rule
  automatically (`mapWhereUniqueOwner`). Rows reading `a`, `lam` or `u` live on both
  nodes because those fields are one-sided at a junction.

## 3. Importing your robot

The solver reads a **URDF** through pinocchio and a small **robot yaml** that says what
the URDF cannot (`model/robot_description.hpp`; `robots/hopper2d.yaml` is a complete
example):

```yaml
schema: 1
name: hopper2d
urdf: ../assets/hopper2d.urdf        # relative to this file
base: {type: floating_explicit_xyz_euler}
gravity: [0.0, 0.0, -9.81]
joint_order: [base_px, base_py, base_pz, base_rx, base_ry, base_rz, hip, knee]  # checked
actuated_joints: [hip, knee]          # the columns of B, in this order
contacts:
  - {frame: foot, dim: 3}             # dim 3 point force, dim 6 flat-foot wrench [f; tau]
monitor_frames: [foot, knee_point]    # ground keep-out points (position only)
collision_capsules:                   # the capsule model for obstacle / self-collision rows
  - {name: shank, frame: knee, a: [0, 0, -0.02], b: [0, 0, -0.28], radius: 0.040}
actuation:
  motor_model: none                   # or friction_inertia with coulomb/viscous/armature/bias
  effort_limits: from_urdf            # or explicit: [..] in actuated order
  velocity_limits: from_urdf
```

Requirements on the URDF: the floating base is **explicit**, six 1-DOF joints named
`base_px, base_py, base_pz, base_rx, base_ry, base_rz` (prismatic x, y, z then revolute
roll, pitch, yaw) so that `q` is a vector space with `nq == nv`
(`assets/` has six). Every contact / monitor / capsule frame must exist in the URDF
(`pinocchio::Model::existFrame`). Joint limits come from the URDF unless `explicit`.

Loading it:

```cpp
const RobotDescription rd = loadRobotDescription(root + "/robots/hopper2d.yaml");
std::shared_ptr<RobotModel> model = std::make_shared<RobotModel>(rd);
const ContactSet stance = model->allContacts();        // every contact of the table
const ContactSet air;                                  // no contact
const int impactSet = model->ensureImpactSet(stance);  // register impact sets BEFORE makeData()
```

`RobotModel` (`model/robot_model.hpp`) is the one call site into pinocchio: `evalNode`
computes, for a node, exactly the quantities the attached terms declared (RNEA and its
derivatives, frame kinematics, capsule endpoints, impact impulses). Forward kinematics
for seeds and rest poses:

```cpp
ModelDataPtr data = m.makeData();
ModelEvalCache cache;
m.initCache(cache);
m.evalNode(*data, q, zero, zero, kFootPos, cache);      // cache.footPos[c] = contact position
```

Rest poses are given as joint angles; the base is derived, never guessed (section 10
shows why this matters).

## 4. Posing the problem

Build the constraint objects once and share them between elements, so that one name is
one term:

```cpp
auto dyn     = std::make_shared<DynamicsConstraint>(m, stance);                     // "dyn": M a + C v + g - J' lam - B u = 0
auto velFoot = std::make_shared<FrameVelocityConstraint>(0, Vector3d::Zero(), "vel_foot");
auto posFoot = std::make_shared<FramePositionConstraint>(0, footTarget, "pos_foot");
auto boxU    = std::make_shared<BoxConstraint>(Field::U, uIdx, -ulim, ulim, "box_u");
auto fric    = std::make_shared<FrictionConeConstraint>(stance, mu, 1e-2, "ineq_fric");
auto effort  = std::make_shared<ControlEffortCost>();
```

Then an element per mode:

```cpp
Element pushOff;
pushOff.label = "push_off";  pushOff.phase = 0;  pushOff.T = 0.8;
pushOff.contacts = stance;   pushOff.degree = 4;  pushOff.forceControlPoints = 4;
pushOff.addNode(dyn, Where::AllNodes);
pushOff.addNode(pinQRest, Where::FirstNode);     // start at rest in the squat
pushOff.addNode(pinVRest, Where::FirstNode);
pushOff.addNode(velFoot, Where::Interior);       // no slip (the first node is at rest anyway)
pushOff.addNode(velFoot, Where::LastNode);
pushOff.addNode(posFoot, Where::Interior);       // the foot stays where it stands
pushOff.addNode(posFoot, Where::LastNode);
pushOff.addNode(vzUp, Where::LastNode);          // liftoff speed for the hop: base vz >= sqrt(2 g hop)
pushOff.addOffNode(velFoot, uniformInteriorTaus(3));  // no slip between the nodes too
pushOff.setDefectCost(dyn, uniformInteriorTaus(6), rho);
pushOff.addCost(effort);
```

Rows that hold everywhere (hardware boxes, planar lock, ground keep-out) go on every
element with `Where::AllNodes`. Rows about an event (liftoff speed, touchdown speed,
landing placement) go on the node of that event. The touchdown of the hopper:

```cpp
flight.addNode(posFoot, Where::LastNode);                               // land where it took off
flight.addNode(std::make_shared<FrameVelocityBoundConstraint>(0, 2, -1.0, -0.5, "ineq_land_vz", 3), Where::LastNode);
```

The chain:

```cpp
TrajectoryProblem problem;
problem.model = model;
problem.elements = {pushOff, flight, landing};
problem.addReset(std::make_shared<ContinuityReset>(nq, nv));                 // liftoff: smooth
problem.addReset(std::make_shared<PlasticImpactReset>(impactSet, nq, nv));   // touchdown: impact map
```

Costs: `ControlEffortCost` is `1/2 ||u||^2` per node (or `1/2 sum_j w_j u_j^2` with
per-joint weights); the node costs are summed with the element's Clenshaw-Curtis weights
times `T/2` (`ProblemFamily::objectiveNodeWeights`), so the objective is the quadrature
of `int 1/2 ||u||^2 dt`. Other costs weight joint velocities, accelerations, a posture or
the contact forces; weights are plain numbers in SI units, so scale them by the robot's
limits (the task builders use `(effort limit / velocity limit)^2`). A yaml-posed problem
composes the same catalogue in its `objective:` block (`docs/yaml_reference.md`).

What the solver does with the costs is the recipe's choice (section 7): in the default
feasibility mode (`recipe.objective = "none"`) phase C runs on the restoration twin, the
costs stripped, and the objective is only reported; with `recipe.objective = "posed"`
phase C descends the objective plus the augmented penalty and, on a judged mesh, ends on
a first-order certificate (`kkttol`, `objtol`, `objwindows`) or, with
`recipe.objstop = "feasibility"`, on the criterion alone.

The dynamics defect cost `setDefectCost(dyn, taus, rho)` is what keeps the equations of
motion satisfied **between** the nodes (the collocated rows only hold at the nodes).
Its weight `rho` is the unit of every penalty of the method (section 7), so choose it
once: `3e6` for a 12 kg hopper, `3e8` for the humanoids of the benchmark.

## 5. The mesh

Elements can be bisected at any physical-time fraction, before or after the seed:

```cpp
problem = splitElement(problem, e, 0.5);   // element e becomes two; node terms mapped with junction ownership,
                                           // off-node / control-point rows copied, costs and defect kept
```

Grade the mesh toward the contact events: the force transient of a liftoff lives in a
few tens of milliseconds, and a uniform mesh wastes nodes on the quiet part of the
stance. The example does

```cpp
splitGraded(problem, 2, 6, 0.6, false);  // landing: 6 elements, finest at its start (touchdown)
splitGraded(problem, 1, 4, 1.0, true);   // flight: 4 uniform
splitGraded(problem, 0, 6, 0.6, true);   // push-off: 6 elements, finest at its end (liftoff)
```

(`splitGraded` is a ten-line helper in the example calling `splitElement`.) Rows that
depend on which element owns an event are attached **after** the split, on that
element:

```cpp
const int lastPushOff = problem.elementsOfPhase(0).back();
for (int e = 0; e < nElements; ++e) {
  Element& el = problem.elements[e];
  if (el.contacts.active.empty()) continue;
  el.addForceControlPoints(fric, range(0, e == lastPushOff ? Nc - 1 : Nc));
  if (e != lastPushOff) { el.addForceControlPoints(lamFloor, range(0, Nc)); continue; }
  for (int j = 0; j + 1 < Nc; ++j) el.addForceControlPoints(taperedFloor(j), {j});   // fz >= lamMin (1 - j/(Nc-1))
  el.addForceControlPoints(lamLiftoffPin, {Nc - 1});                                // force zero at liftoff
}
```

Two details worth copying: the pinned liftoff control point is left out of the cone
(a zero force sits exactly on the smoothed cone's boundary and would never satisfy
`g <= 0`), and the mid-flight clearance rows are placed only on the flight elements
inside the middle half of the flight (next to liftoff and touchdown the foot is on the
floor by construction, so a clearance row there is infeasible).

Then `problem.validate()` and `CompiledProblem::compile(problem)`: the layout (decision
vector), the row space, the interpolation weights, everything the solver needs. The
solver's own adaptive refinement bisects elements of its choice later; the generic
`StaticFamily` handles that for any problem.

## 6. The cold seed

The seed is a full decision vector `y0` (primal, then zero duals) of the compiled
problem. `ElementViewMutable` maps one element's fields; the pattern is: node
configurations from a simple guess, velocities and accelerations by spectral
differentiation (so the kinematic rows start exactly satisfied), forces from statics,
torques from inverse dynamics:

```cpp
ElementViewMutable X(y0.data(), el);
for (int k = 0; k < el.Nn; ++k) X.Q.col(k) = qSquat;              // or a blend / a hop (see the example)
const Eigen::MatrixXd& D = prob.grid(e).D();
const double a = prob.timeScale(e);                               // 2 / T
X.V = a * X.Q * D.transpose();
X.A = a * X.V * D.transpose();
if (el.Nc > 0) {                                                  // the force polygon: weight on fz
  Eigen::MatrixXd Cphys = Eigen::MatrixXd::Zero(el.dims.nlam, el.Nc);
  Cphys.row(2).setConstant(weight);
  X.C = Cphys * el.sigma.cwiseInverse().asDiagonal();             // decision = physical / sigma
}
const Eigen::MatrixXd lamNodes = prob.nodeForces(y0, e);
for (int k = 0; k < el.Nn; ++k)
  X.U.col(k) = m.inverseDynamicsTorque(*data, X.Q.col(k), X.V.col(k), X.A.col(k), lamNodes.col(k), el.contacts);
```

The seed does not have to be feasible, but it has to be in the right basin: hold the poses you know,
put the base where the forward kinematics says it is, give the flight its hop.

## 7. Solving

```cpp
StaticFamily family("hopper_jump", problem, y0, rho);   // any TrajectoryProblem + seed
SolverRecipe recipe;                                    // the default recipe
recipe.applyPenaltyScale(rho);                          // penalties in units of rho
recipe.levels = 6;                                      // adaptive refinement levels
PipelineResult res = runPipeline(family, recipe, "[hopper_jump]");
```

`runPipeline` (`solver/pipeline.hpp`) runs the staged method: phase A0 (pure
feasibility at zero multipliers), then per mesh level phase C (frozen-dual
Levenberg-Marquardt windows of the augmented Lagrangian with damped multiplier updates
and per-term penalty growth), the relocation of movable off-node points, the judges, and
Dörfler marking + bisection + exact prolongation to the next level until the node gate
and the fine grid are below `recipe.tol` (1e-3). For a problem with obstacle rows the
pre-stage adds the obstacle-family continuation (the fence and terrain families
implement it; `StaticFamily` has no obstacle family).

What the log tells you, line by line:

```
seed: zero-dual max|c| ...                  the worst row of the seed
phase A0 done in ... s: max|c| ...          A0 stops when feasible (1e-3), stalled, or out of budget
[hp] level 0 done: ... steps, node gate ..., fine grid ... (<term> elem <e> t=...)
[hp]   element indicators: 0:... 1:... ... 6:...*     * = marked for bisection
[hp] level 0 -> 1: ... -> ... elements, prolongation error ...
...
[hp] criterion met: max(node gate, fine grid) = ... <= 1.0e-03
```

The two numbers that matter are the **node gate** (the largest violation of any row at
zero multipliers) and the **fine grid** (the largest violation of any throughout row,
re-evaluated every `finedt` = 0.01 s between the nodes).

The recipe is the `SolverRecipe` struct (`solver/recipe.hpp`), or the `unified:` block of a
yaml for `leap_solve`; `docs/yaml_reference.md` lists every knob.

## 8. Judging, reporting, visualizing, verifying

After the solve, judge the final iterate on its own problem and write the archive:

```cpp
Level& L = res.level;
Evaluation ev;
const double gate = L.evaluator->zeroDualViolation(res.y, L.penalties->muRow(), ev);
printViolationsByGroup(*L.restoration, ev);
const FineGridReport fr = fineGridViolations(*L.restoration, res.y, 0.01);
printFineGrid(fr, 12);
ArchiveRunInfo info;  info.generator = "hopper_jump";  info.urdfPath = rd.urdfPath;  info.robotYamlText = rd.sourceText;
writeSolutionArchive(*L.restoration, ev, res.y, L.penalties->muRow(), info, "results/hopper_jump.h5");
```

The archive (`docs/solution_format.md`) holds, per element, `Q, V,
A, U` at the nodes, the node forces and the raw force control polygon, the per-node
violation table, plus the duals, the exact decision vector, the raw residual vector with
its row-to-term map, and a JSON blob with the model, the collision geometry, the element
table and the metrics.

`python/report_solution.py` turns it into figures, an animation and a report, and
re-verifies the solution **independently of the solver** with pinocchio's Python
bindings (it only reads the archive and the URDF):

```bash
python python/report_solution.py results/hopper_jump.h5 --out results/hopper_jump_report --mu 0.85
```

It writes `states.png`, `velocities.png`, `torques.png`, `forces.png`, `violations.png`,
`animation.gif` (a stick figure with the capsules, x-z projection) and `report.md` with
the solve summary and this table (one row per check, with the worst value, the threshold
and pass / fail):

| check | threshold |
|---|---|
| equations of motion on the fine grid, max residual (N, N m) | 1e-3 |
| v = dq/dt and a = dv/dt at the nodes | 1e-3 |
| stance contact height on the fine grid (m) | 1e-3 |
| stance contact speed on the fine grid (m/s) | 1e-3 |
| flight: monitor frames below the floor (m) | 1e-3 |
| friction cone on the control points (N) | 1e-3 |
| normal force floor on the control points (N) | 1e-3 |
| torque / joint speed / joint range beyond the URDF limits | 1e-3 |
| configuration and velocity continuity across junctions | 1e-3 |
| plastic impact map v+ = Pi(q) v- at touchdown (m/s) | 1e-3 |
| rest at both ends | 1e-3 |

Every row is recomputed from scratch: `pin.rnea` for the equations of motion with the
Bezier force curve and the torque polynomial interpolated as the solver does, frame
Jacobians for the contact terms, the mass matrix for the impact map.

## 9. Tuning tips

* **Check the formulation before the solver.** Look at the `violations by group` line:
  it names the worst term of each group. The archive's `/rows/g` + `/rows/term` give
  every row's residual if the group line is not enough. Derive the base position of a
  rest pose from the foot by forward kinematics rather than pinning both independently.
* **Attach rows only where they can hold.** A smoothed friction cone with `eps > 0` is
  violated by a zero force; a clearance row at a point where the foot is pinned to the
  floor is infeasible.
* **Mesh.** Grade the stance elements toward liftoff and touchdown (0.6 per element
  in the benchmark). Degree 4 with `Nc = 4` is the benchmark's fence choice; degree 8
  with `Nc = 7` the terrain one (fewer, longer elements). The adaptive refinement bisects
  a coarse mesh, but every level costs a phase C.
* **Between-node feasibility.** Collocated rows hold at the nodes only. Use
  `addOffNode` for no-slip, clearance and box rows (the relocation pass moves those
  points onto the between-node maxima), `addForceControlPoints` for anything on the
  force, `addFieldFunctionals` with the Bernstein coefficient rows for a torque box
  that must hold everywhere, and the defect cost for the dynamics.
* **Penalties.** Everything scales with `rho` (`applyPenaltyScale`): phase C seeds every
  term at `0.1 rho` and grows it by 3x on a stall up to `rho`; A0 uses `rho / 3`.
  `rho` itself sets how strongly the between-node dynamics is enforced against the
  objective. The benchmark uses 3e6 (hopper, N^2 units) and 3e8 (humanoids).
* **Step control.** `h0` (1e-4) is the first pseudo-time step; the ratio controller
  doubles it on good steps up to `hmax` (5). `blowup` (10x the best window) rolls back
  a diverging window.
* **Windows and stops.** A window is 20 pseudo-time units; the multiplier update
  happens at the window end. `stall` (12 windows without 1 % progress) and `csteps`
  (4000 accepted steps per level) bound a level. The level stop hands over to
  refinement as soon as the node gate is below half the fine-grid value.
* **Seeds.** Hold known poses, blend between them with zero end slopes, give flights
  their apex, derive forces and torques from statics / inverse dynamics. The seed need
  not satisfy the dynamics, but its contact geometry (the foot on the floor, the base
  at the right height) must be right.
* **Threads.** `OMP_NUM_THREADS` sets the evaluator's team (one pinocchio data per
  thread); 12 physical cores pinned with `OMP_PLACES=cores` is the benchmark setting.
  More threads than elements do not help the factorization.
* **Exact Newton** (`recipe.gn = false`) is available; for cold starts use Gauss-Newton.

## 10. The hopper jump in C++, end to end

`examples/hopper_jump.cpp`, in the order of this guide:

1. **Robot** (section 3): `robots/hopper2d.yaml`, 8 DOF (6 base + hip + knee), one
   point foot, monitors `foot` and `knee_point`, three capsules; 120 N m, 20 rad/s
   limits from the URDF; weight 118.7 N.
2. **Poses**: `restPose(m, hip, knee, foot)` puts the base so that the foot is at
   `foot = (0, 0, 0)`: squat (-0.498, 0.90) at the start, stand (-0.220, 0.40) at the end.
   The hop is 0.17 m, so the liftoff speed must exceed `sqrt(2 g 0.17) = 1.83 m/s`
   and the flight of 0.391 s is long enough (`2 vz / g = 0.372 s`).
3. **Modes** (section 4): push-off 0.8 s, flight 0.391 s, landing 0.8 s; rows as listed
   in the code block of section 3 plus, on every element, `box_q`, `box_v`, the
   Bernstein-certified `box_u`, `box_att` (pitch within 0.5 rad), `pin_planar`
   (y, roll, yaw at zero; the dynamics keeps the in-plane rows,
   `SelectedRowsDynamicsConstraint`), the effort cost and the defect cost with
   `rho = 3e6` on 6 points.
4. **Mesh** (section 5): 6 + 4 + 6 graded elements, then the force rows per final element.
5. **Seed** (section 6): the squat held through the push-off, a smoothstep blend to the
   stand with a `4 s (1 - s) hop` bump of the base through the flight, the stand through
   the landing; forces and torques from statics and inverse dynamics.
6. **Solve** (section 7): `runPipeline` on a `StaticFamily` with the default recipe.
7. **Report** (section 8): the program prints the criterion, the accepted steps and
   levels, the control effort, the base apex, the peak torque over its limit and the
   peak normal force, and `report_solution.py` writes the figures, the animation
   (`animation.gif`, the stick figure with the capsules) and
   `results/hopper_jump_report/report.md`.

To make it your problem: change the robot yaml, the poses, the mode list and the rows;
keep the seed recipe, the names, the ownership rule and the force rows on the control
points. For a yaml-driven task with its own obstacle family and h-refinement operators,
implement `ProblemFamily` as `tasks/fence_jump.cpp` and `tasks/terrain_walk.cpp` do.
