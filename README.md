# LEAP: gradient-flow augmented-Lagrangian trajectory optimization for contact-constrained robots

C++ Implementation of the algorithm introduced in the manuscript *LEAP: Gradient Flow Augmented Lagrangian Trajectory
Optimization for Contact Constrained Robots*.
The solver tackles multi-contact trajectory optimization.
The main features:
- Pseudospectral collocation
- Solve by integrating a primal-dual gradient flow of a Powell-Hestenes-Rockafellar augmented Lagrangian.
- Benchmarks problems with tracking in mujoco.


> **AI Assistance Notice:** All content below this notice was written with the assistance of a Large Language Model (LLM).

## Code organization
```
new_src/
  spectral/   CGL grid, differentiation, barycentric interpolation, Bernstein/Bezier, quadrature
  model/      RobotDescription (robots/*.yaml) + RobotModel (pinocchio over the URDF)
  problem/    the formulation API: Constraint / Cost / ResetMap catalogue, TrajectoryProblem
              (elements + attachments), Layout, CompiledProblem (row space), mesh refinement
  alm/        the PHR augmented Lagrangian: per-term penalties, the evaluator (residuals, merit,
              gradient, Gauss-Newton or exact-Newton metric), the junction quotient
  linalg/     the linearly implicit step and its factorization backends (block-Schur + CHOLMOD, ...)
  solver/     the flow driver, the judges (node gate, fine grid), relocation, h-refinement,
              the staged pipeline, the recipe (the `unified:` yaml block)
  tasks/      problem builders: contact_sequence (generic), fence_jump and terrain_walk
              (the benchmarks), the yaml `objective:` block, StaticFamily
  io/         yaml reader (vendored fkYAML), the HDF5 solution archive
  apps/       leap_solve
  examples/   hopper_jump.cpp: a problem written against the C++ API
  python/     leap_solve.py (run + parse + read an archive), report_solution.py (plots,
              animation, pinocchio check of a solution)
  tracking/   the MuJoCo tracking study (Table III, Fig. 2, Fig. 3)
  baselines/  Crocoddyl and Aligator: code and instructions to run them on the benchmark
  tests/      finite-difference and structural tests (ctest)
  problems/   the ten benchmark problem yamls; examples/ (contact_sequence problems,
              a posed-objective example)
  robots/, assets/   robot yamls and URDFs (hopper2d, hopper2d_foot, go2, h1, g1_29dof, talos)
  scripts/    run_benchmark.sh, reproduce_paper.sh, make_results_table.py
  docs/       REPRODUCE.md, TUTORIAL.md, yaml_reference.md, solution_format.md, design.md
```

## Build

Dependencies: a C++20 compiler, CMake >= 3.22, Eigen 3.4+, pinocchio 3.x with its CppAD
headers, HDF5 (C), and optionally OpenMP, SuiteSparse CHOLMOD and LAPACK. `environment.yml`
is a conda environment with all of them:

```bash
conda env create -f environment.yml && conda activate leap_build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/c++
cmake --build build -j
ctest --test-dir build --output-on-failure        # 7 tests
```

CHOLMOD and LAPACK are optional; the configure log reports whether they were found.

## Run

```bash
# the hopper jumping in place
OMP_NUM_THREADS=4 build/leap_solve problems/examples/hopper2d_jump.yaml output.stem=results/hopper2d_jump
python python/report_solution.py results/hopper2d_jump.h5 --out results/hopper2d_jump_report --mu 0.85

# one benchmark instance: the Go2 fence jump
OMP_NUM_THREADS=12 build/leap_solve problems/go2_jump_fence.yaml output.stem=results/sol_fence_go2
```

`leap_solve <problem.yaml> [dotted.key=value ...]`: every yaml key can be overridden on
the command line.

* The yaml describes the task: `robot:`, `problem:`, the transcription settings of
  `solver:`, and optional `obstacles:` / `terrain:` / `objective:` blocks.
* The `unified:` block sets the method (`solver/include/leap/solver/recipe.hpp`); a yaml
  without it runs the default method.
* `docs/yaml_reference.md` documents every key.

The driver prints the solve log and writes `<output.stem>.h5`, the solution archive
(`docs/solution_format.md`). The summary line of the log:

```
[unified] robot=<robot> type=<type> pipeline=unified met=<0|1> gate=<...> fine=<...> ... total=<steps> solvewall=<seconds> ...
```

* `met=1`: the node gate (`gate`, largest constraint violation at the collocation nodes)
  and the fine-grid violation (`fine`, largest violation on a 0.01 s grid) are below
  `unified.tol` = 1e-3.
* `total`: the number of accepted steps.
* `solvewall`: the solve time in seconds.

`python/report_solution.py` turns an archive into figures, an animation and a report, and
checks the solution with pinocchio's Python bindings: equations of motion on a fine grid,
contact kinematics, friction cone, hardware limits, junction conditions, rest. From
Python:

```python
import sys; sys.path.insert(0, "new_src/python")
from leap_solve import solve, load_archive
r = solve("new_src/problems/examples/go2_jump.yaml", {"output.stem": "sol_go2_jump"}, threads=12)
print(r.met, r.gate, r.fine, r.steps, r.wall)
sol = load_archive(r.archive)      # numpy arrays per dataset + sol["meta"]
```

## Posing a problem

**From a yaml** (`problem.type: contact_sequence`, `docs/TUTORIAL.md` section 1): list
the modes, the poses and the contact model:

```yaml
problem:
  type: contact_sequence
  modes:
    - {label: push_off, contacts: [FL_foot, FR_foot, RL_foot, RR_foot], T: 0.5}
    - {label: flight,   contacts: [],                                   T: 0.4}
    - {label: landing,  contacts: [FL_foot, FR_foot, RL_foot, RR_foot], T: 0.5, shift: [0.3, 0, 0]}
  pose_start: [{joint: FL_thigh_joint, angle: 1.0}, {joint: FL_calf_joint, angle: -2.0}, ...]
  pose_end:   [{joint: FL_thigh_joint, angle: 0.78}, {joint: FL_calf_joint, angle: -1.57}, ...]
  hop: 0.15
  mu: 0.85
solver: {deg: 4, nc: 4, split: 6, hgrade: 0.6, lsqnq: 6, rho: 3.0e6}
```

`problems/examples/` has three:

* `hopper2d_jump.yaml`: the hopper's jump in place;
* `go2_jump.yaml`: the Go2's forward jump;
* `go2_step.yaml`: one Go2 step on three feet.

**In C++** (`docs/TUTORIAL.md` sections 2 to 10): a `TrajectoryProblem` is a chain of
`Element`s joined by `ResetMap`s. Each element is one Chebyshev segment, with a contact
set, a degree, a duration and `Nc` force control points, and carries its attachments:

```cpp
Element stance;
stance.contacts = model->allContacts();
stance.degree = 4;  stance.T = 0.8;  stance.forceControlPoints = 4;
stance.addNode(dyn, Where::AllNodes);                               // collocated at the nodes
stance.addNode(pinPose, Where::FirstNode);
stance.addOffNode(noSlipVel, uniformInteriorTaus(3));               // between the nodes (movable)
stance.addForceControlPoints(frictionCone, {0, 1, 2, 3});           // on the Bezier polygon
stance.addFieldFunctionals(boxU, Field::U, bernsteinRows);          // on linear functionals of u
stance.addCost(std::make_shared<ControlEffortCost>());
stance.setDefectCost(dyn, uniformInteriorTaus(6), rho);             // LS dynamics defect
problem.elements.push_back(stance);
problem.addReset(std::make_shared<PlasticImpactReset>(impactSet, nq, nv));
StaticFamily family("my_problem", problem, seed, rho);
PipelineResult res = runPipeline(family, SolverRecipe(), "[my_problem]");
```

A `Constraint` (`problem/include/leap/problem/constraint.hpp`) is written against the
per-node vector `z = [q; v; a; lam; u]`. It:

* names its term (one penalty per term);
* declares its rows and their sense, and the model quantities it needs;
* supplies the value and the Jacobian.

The catalogue (`constraints.hpp`, `obstacles.hpp`, `self_collision.hpp`, `costs.hpp`)
covers:

* dynamics and frame kinematics;
* boxes and pins on any field;
* the friction cone, wrench boxes and the impact cone;
* capsule obstacles and self-collision;
* the quadratic costs.

## Reproducing the paper

`docs/REPRODUCE.md` has the full instructions. In short:

```bash
SNAPSHOTS=1 CORE_BASE=0 scripts/reproduce_paper.sh results/repro 4 repro   # Table II: the ten instances, 4 lanes of 12 cores
cd tracking                                                                 # environment: tracking/environment.yml
for t in fence terrain; do $PY scripts/track.py --task $t --all --archive-dir ../results/repro/$t --archive-tag repro --out ../results/repro/tracked/$t/leap --cores 0-4; done
$PY scripts/table3.py ../results/repro/tracked                              # Table III
```

`docs/REPRODUCE.md` then covers the budget study (Fig. 3), the time-lapse (Fig. 2), and the
Crocoddyl and Aligator runs (`baselines/README.md`).

## Solver options

All documented in `docs/yaml_reference.md`:

* `unified.objective=posed`: phase C descends the problem's posed objective (the yaml's
  `objective:` block, or the costs a C++ problem attaches) plus the augmented penalty,
  and stops on a first-order certificate. Example:
  `problems/examples/hopper2d_jump_fence_objective.yaml`.
* `unified.gn=false`: the exact-Newton metric in phase C.
* `unified.backend=block_schur | sparse_cholmod | sparse_llt | sparse_ldlt | dense`: the
  factorization of the step.

## Tests

```bash
ctest --test-dir build --output-on-failure      # or: build/tests/test_<name> .
```

| test | checks |
|---|---|
| `test_spectral` | CGL grid, differentiation, barycentric weights, Bernstein basis, de Casteljau, quadrature |
| `test_model_fd` | every model derivative (dynamics residual, frame kinematics, impacts, capsules) vs finite differences |
| `test_constraints_fd` | every constraint Jacobian vs finite differences on go2 and h1 |
| `test_alm_fd` | the flow right-hand side vs the FD merit gradient, the Gauss-Newton metric vs FD Jacobians, every linear backend vs a dense solve, on a hand-built problem with every attachment kind |
| `test_problem_fd` | the same on the fence and terrain benchmark problems |
| `test_hessian_fd` | every analytic contracted Hessian vs finite differences and CppAD, the exact-Newton metric of three problems |
| `test_contact_sequence` | the generic problem type: the three example yamls build and seed, error messages, the flow right-hand side vs FD, the hopper jump solves to the criterion |
