# LEAP (new_src): design

The LEAP solver (the gradient-flow augmented-Lagrangian trajectory optimizer for
contact-constrained robots): a problem is posed through one interface and solved by one
driver.

## 1. Layout

One static library `leap`, one driver, tests. Each module has `include/leap/<module>/`
and `src/`:

| module     | contents |
|------------|----------|
| `spectral` | Chebyshev–Gauss–Lobatto grid and differentiation, barycentric interpolation, Bernstein/Bezier basis and de Casteljau subdivision, Clenshaw–Curtis and Gauss–Legendre quadrature |
| `model`    | `RobotDescription` (robot yaml), `RobotModel` (pinocchio over a URDF; dynamics residual, frame kinematics, impact map, capsules; every derivative the constraints consume) |
| `problem`  | the formulation API: `Constraint`, `Cost`, `ResetMap` base classes and their catalogue; `Phase` / `TrajectoryProblem` (modes, meshes, attachments); the decision-vector `Layout`; `CompiledProblem` (rows, terms, interpolation and defect points resolved) |
| `alm`      | the PHR augmented Lagrangian: penalties per term, the evaluator (residuals, merit, gradient, Gauss–Newton or exact-Newton metric), the junction quotient, the first-order residual |
| `linalg`   | the condensed linear step: the per-element assembly (`ElementMetricAssembly`) and its backends: block-Schur over elements (dense per element + CHOLMOD / Eigen on the separators), whole-matrix CHOLMOD / Eigen LLT / Eigen LDLT, dense |
| `solver`   | one frozen-dual flow (linearly implicit LM step, merit acceptance, ratio step control, Nesterov look-ahead, windows, multiplier updates), the judges (node gate, fine grid), relocation, h-refinement, the staged pipeline, the recipe |
| `tasks`    | the problem builders that turn a problem yaml into a `TrajectoryProblem` + seed: `contact_sequence` (the generic type: a prescribed contact sequence with placements and rest poses, run through `StaticFamily`), `fence_jump` and `terrain_walk` (the benchmarks, with their obstacle families); the yaml `objective:` block (a cost composed from the catalogue) |
| `io`       | yaml reader, HDF5 solution archive, text reports |
| `apps`     | `leap_solve <problem.yaml> [key=value ...]` |
| `tests`    | finite-difference oracles for every derivative, unit tests of the pieces |

Style: plain classes with virtual interfaces where the user extends (constraints, costs,
resets, task builders); no templates beyond Eigen, no factories, no lambdas; dispatch on
strings happens once, in the yaml loaders.

## 2. The formulation interface

A trajectory optimization problem is a chain of **elements** (one Chebyshev segment
each: a duration, an active contact set, a degree, `Nc` force control points, whether
the torque is eliminated), consecutive elements joined by a **reset map** (continuity or
plastic impact). Elements of one contact mode share a `phase` index. This is the
`TrajectoryProblem`:

```cpp
Element stance;
stance.contacts = model->allContacts();  stance.degree = 8;  stance.T = 0.4;
stance.forceControlPoints = 7;           stance.deriveTorque = false;
stance.addNode(constraint, Where::AllNodes);          // collocated at the nodes
stance.addNode(constraint, Where::LastNode);
stance.addOffNode(constraint, taus);                  // between the nodes (movable)
stance.addForceControlPoints(constraint, points);     // on the Bezier force polygon
stance.addFieldFunctionals(constraint, Field::U, W);  // on linear functionals of a nodal field
stance.addCost(cost);
stance.setDefectCost(dynamics, taus, rho);            // least-squares dynamics defect
problem.elements.push_back(stance);
problem.addReset(std::make_shared<ContinuityReset>(nq, nv));
```

A `Constraint` is written against the per-node PHYSICAL vector `z = [q; v; a; lam; u]`
(sizes from `NodeDims`) and a `ModelEvalCache`; it declares its name (the term, which
carries its own penalty), its rows, its sense (`g = 0` or `g <= 0`), the model
quantities it needs, and supplies `value`, `jacobian` and the contracted Hessian
(`model/model_hessians.hpp` serves every model-dependent one in closed form; linear
rows add nothing). The catalogue (`problem/constraints.hpp`): dynamics (full, floating-base rows
only under torque elimination, planar-selected rows), frame position / velocity /
acceleration / orientation and their bounds, ground keep-out of the monitor frames,
linear pins and boxes on any field, motor envelope, friction cone, wrench (CoP / yaw)
boxes, impact cone, capsule-vs-sphere, capsule-vs-cuboid (exact, rounded, smooth
certificates), capsule self-collision.

How the physical vector maps onto decision variables is the transcription's business
(`problem/layout.hpp`): `q, v, a` (and `u` unless eliminated) are nodal values on the
CGL grid of the element; the contact force of a stance element is a Bezier curve with
`Nc` control points (scaled by `sigma`), so the node force is `lam_k = sum_j E(k,j) sigma_j C_j`;
under torque elimination `u` is the actuated rows of the inverse dynamics at the node.
The lift of a node Jacobian to decision columns is one small dense product per node.

The task builders (`tasks/`) pose each contact mode once and split it into elements
(`split` per mode, graded widths, later the h-refinement bisections); junctions inside a
mode are continuity resets, which the junction quotient turns into shared coordinates.
`tasks/contact_sequence.cpp` is the generic builder (any sequence of contact sets, the
benchmarks' rows, an inverse-kinematics seed) and the template for a new task-specific one;
`StaticFamily` (`tasks/static_family.hpp`) runs any `TrajectoryProblem` plus seed through
the pipeline with the generic h-refinement of `problem/mesh_refinement.hpp`.

## 3. The method

The NLP is
`min f(x) + (rho/2) sum w_m ||r_m(x)||^2  s.t. g_E = 0, g_I <= 0`; the PHR augmented
Lagrangian `L_A(x, gamma, mu)`; every stage is a frozen-dual flow `dx/ds = -M^-1 grad L_A`
integrated by the linearly implicit step `(I + h H) dx = -h grad L_A` with the
Gauss–Newton metric on an anticipatory active set, strict merit acceptance with
backtracking, a ratio step controller, Nesterov look-ahead with monotone restart; windows
end with the damped multiplier update `gamma += tau mu c_eff` and the per-term MuStall
penalty step; the staged pipeline is obstacle rungs (if the seed crosses an obstacle),
phase A0 (pure feasibility at zero duals), phase C per mesh level with relocation rounds
of the movable off-node points, the node-gate / fine-grid judge, Dörfler marking and
exact prolongation.

Five defaults have a knob beside them (`docs/yaml_reference.md`): the penalty reseed
at every phase C call (`unified.mucarry`), no step reset after a multiplier update
(`unified.dualreset`), the mean projection of a junction-discontinuous entry iterate
(`unified.junctionmean`), the terrain split's double statement of q-only rows at interior
junctions (`problem.junctionownership`), and no landing force ramp on the fence
(`problem.landramp`). At an h-refinement bisection, `splitElement` states the dynamics
row on both sides of the new junction.

Two further options: the exact-Newton metric (`gn=false`) adds
every term's contracted Hessian `sum_i w_i d2 g_i / dz2` lifted to decision coordinates,
including the reset maps' and the curvature of the inverse-dynamics lift under torque
elimination; it is used in phase C (the pure-feasibility LM stages keep Gauss–Newton).
The objective
mode (`objective=posed`) runs phase C on the posed problem (its costs weighted by the
objective's quadrature) instead of the restoration twin, with the first-order residual
of `alm/first_order.hpp` and an objective plateau as the stationary stop on a judged
mesh (`solver/pipeline.cpp`, `ObjectiveStopCheck`).

## 4. Verification

Every derivative is checked against finite differences at three levels: the model
(`test_model_fd`: dynamics residual, frame kinematics, impacts, capsules), the constraint
catalogue (`test_constraints_fd`), and the assembled augmented Lagrangian
(`test_alm_fd` on a hand-built problem with every attachment kind; `test_problem_fd` on
the posed benchmark problems: the flow right-hand side against the finite-difference
merit gradient along random directions and against the finite-difference row
Jacobians, the Gauss-Newton metric against those Jacobians, every linear backend
against a dense solve). The exact-Newton metric's contracted Hessians are checked kernel
by kernel against finite differences on four robots and against CppAD tapes of the same
quantities, and in situ on three posed problems with every multiplier live
(`test_hessian_fd`).

## 5. Performance

The per-step cost is the evaluation (OpenMP over nodes, points and defects), the metric
assembly (OpenMP over elements: node blocks scattered into the element's dense local
matrix, off-node / defect / reset rows stacked and applied as one rank-k update at
factorization time) and the block-Schur factorization (dense Cholesky per element in
parallel, CHOLMOD on the junction separators).

