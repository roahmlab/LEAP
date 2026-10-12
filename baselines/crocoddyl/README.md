# Crocoddyl baseline

Crocoddyl (stock `SolverFDDP`, version 3.2.1, OpenMP build) on the ten benchmark instances:

| task | robots |
|---|---|
| `terrain` (uneven terrain walk) | `hopper2d_foot`, `go2`, `h1`, `g1_29dof`, `talos` |
| `fence` (fence jump) | `hopper2d`, `go2`, `h1`, `g1_29dof`, `talos` |

All commands below run from this folder (`new_src/baselines/crocoddyl/`).

## 1. Environment setup

Requirements: Linux x86-64 with AVX2, conda or mamba, git, cmake (3.22 or newer), GNU make,
network access to conda-forge and github.com.

```bash
./setup.sh --create-env
```

This

1. creates the conda environment `env/` from `conda-linux-64.lock` (Python 3.12.14,
   crocoddyl 3.2.1, aligator 0.19.1, pinocchio 4.1.0, casadi 3.7.2, coal 3.0.4, numpy 2.5.2,
   h5py 3.16.0, gcc/g++ 16.2.0; the direct dependencies are also listed in `environment.yml`);
2. unpacks the seed archives (`*.h5.xz` to `*.h5`);
3. runs `fence/comparisons/crocoddyl_unified/setup_stock.sh` (skipped if the library exists):
   clones Crocoddyl v3.2.1 (commit `174c6082f0aef4dca80fa767f47fe025d6c4fef1`), applies
   `fence/comparisons/crocoddyl_unified/eigen5-packaging.patch` and builds `libcrocoddyl.so`
   (Release, `BUILD_WITH_MULTITHREADS=ON BUILD_WITH_NTHREADS=12`, the environment's gcc/g++)
   into `build/crocoddyl_unified_release_mt/`;
4. builds the native model modules of `fence/` and `terrain/`;
5. imports every module the solvers load.

To use an existing environment or an existing stock build instead:

```bash
CROCODDYL_ENV_PREFIX=/path/to/env CROCODDYL_BUILD_ROOT=/path/to/build ./setup.sh [--skip-stock]
```

| variable | meaning | default |
|---|---|---|
| `CROCODDYL_ENV_PREFIX` | conda environment | `env/` |
| `CROCODDYL_BUILD_ROOT` | directory holding `crocoddyl_unified_release_mt/libcrocoddyl.so` | `build/` |
| `CROCODDYL_STAGE_ROOT` | terrain: directory for the iteration archive while a solve runs | `run.sh`: parent of the run's output folder; `solve.py` called directly: `stage/` |

Export the same variables when calling `run.sh` and `run_all.sh`. The solver process loads
`libcrocoddyl.so` from `CROCODDYL_BUILD_ROOT` (`LD_PRELOAD`) and the environment's
`lib/` (`LD_LIBRARY_PATH`); see `*/comparisons/crocoddyl_unified/run_python.sh`.

## 2. Running

One instance:

```bash
./run.sh <fence|terrain> <robot> <out_dir> [--cpus LIST] [solve.py options]
./run.sh fence hopper2d results/fence/hopper2d --cpus 0-11
```

All ten, one after the other:

```bash
./run_all.sh results --cpus 0-11
```

* `--cpus`: 1 to 12 distinct physical cores (default: the first 12 physical cores of the
  current CPU affinity). The solver uses one OpenMP thread per listed core.
* Defaults passed to `solve.py`: `--solver-seconds 14400 --wall-seconds 86400`. Other
  `solve.py` options are appended, e.g. `--solver-seconds 600`; `--prepare-only` builds the
  problem and audits the initial guess without starting the solver.
* `solve.py` settings: `--iterations 10000` per call, `--rounds 40` calls, `--reg-init 1e-3`,
  `--th-stop 1e-12`, `--dt 0.01`.
* The solve stops when the physical audit of an iterate passes
  (`max(node, fine grid) <= 1e-3`), when FDDP converges or stops, or when a budget is
  reached. Fence: every iterate is audited. Terrain: an iterate is audited when its cost is
  at most `0.5 x posed rows x 1e-6` and at least `max(2 s, 9 x duration of the previous
  audit)` of solver time after the previous audit, and every `max(60 s, 10 x duration of
  the previous audit)` of solver time. Exit status of `run.sh`: 0 = audit passed,
  2 = stopped without passing, 3 = terrain post-processing failed.
* The first run of each robot compiles its CasADi residual library into
  `fence/comparisons/crocoddyl_impact_cone/generated/` or
  `terrain/comparisons/aligator_unified_terrain/generated/`.

The commands `run.sh` executes (first line of `<out_dir>.log` and of `<out_dir>.postprocess.log`):

```bash
# fence
fence/comparisons/crocoddyl_impact_cone/run_python.sh fence/comparisons/crocoddyl_impact_cone/solve.py \
    --robot <robot> --output <out_dir> --cpus <cpus> --solver-seconds 14400 --wall-seconds 86400
# terrain
terrain/comparisons/crocoddyl_uneven_terrain_along_the_way/run_python.sh terrain/comparisons/crocoddyl_uneven_terrain_along_the_way/solve.py \
    --robot <robot> --output <out_dir> --cpus <cpus> --solver-seconds 14400.0 --wall-seconds 86400.0 --stage-root <dir>
taskset -c <cpus> terrain/comparisons/crocoddyl_uneven_terrain_along_the_way/run_python.sh \
    terrain/comparisons/crocoddyl_uneven_terrain_along_the_way/postprocess.py <out_dir> --workers <number of cpus>
```

## 3. Inputs and outputs

Inputs of an instance (read by `solve.py`):

| task | robot | problem | initial guess |
|---|---|---|---|
| fence | `<robot>` | `fence/showcase_v2/unified/impact_cone/problems/<robot>_jump_fence.yaml` | `fence/showcase_v2/unified/impact_cone/initial_guess/archives/seed_fence_<robot>.h5` |
| terrain | hopper2d_foot | `terrain/showcase_v2/unified/flatfoot_20260913/problems/hopper2d_foot_terrain_walk.yaml` | `terrain/showcase_v2/unified/initial_guess/archives/seed_terrain_hopper2d_foot.h5` |
| terrain | go2 | `terrain/showcase_v2/terrain/problems/go2_terrain_walk.yaml` | `terrain/showcase_v2/unified/initial_guess/archives/seed_terrain_go2.h5` |
| terrain | h1 | `terrain/showcase_v2/unified/trackable_20260913/problems/h1_ap15_ank07_att02_torso05.yaml` | `terrain/showcase_v2/unified/initial_guess/archives/seed_terrain_h1_ap15_ank07_att02_torso05.h5` |
| terrain | g1_29dof | `terrain/comparisons_leap_benchmark/terrain/problems/g1_29dof_terrain_walk.yaml` | `terrain/comparisons_leap_benchmark/terrain/initial_guess/archives/seed_terrain_g1_29dof.h5` |
| terrain | talos | `terrain/showcase_v2/terrain/problems/talos_terrain_walk.yaml` | `terrain/showcase_v2/unified/initial_guess/archives/seed_terrain_talos.h5` |

The robot model is the `robot:` file of the problem yaml (`fence/robots/`, `terrain/robots/`)
and the URDF it names (`fence/assets/`, `terrain/assets/`).

Outputs, in `<out_dir>/`:

| file | content |
|---|---|
| `protocol.json` | settings, command, library versions, sha256 of inputs and sources, mesh, CPUs |
| `inputs/`, `executed_source/` | copies of the inputs (problem yaml, robot yaml, URDF, seed, generated library) and of the sources |
| `initial.npz` | initial `xs`, `us`, `mesh_json` |
| `iterations.h5` | every recorded iterate (`xs`, `us`) and the per-iteration records (`records_json`) |
| `iterations.jsonl` | the same records, one JSON object per line |
| `status.json` | latest progress |
| `calls.json` | per FDDP call: convergence flag, iterations, solver seconds, cost, stopping value |
| `solution.npz`, `result.json` | final iterate; final audit, `passed`, `stop_reason`, `solver_s` (solver time), `wall_s` |
| `failure.json` | traceback, if the process failed |
| terrain only: `audits.jsonl` | offline audit of every stored iterate (`postprocess.py`) |
| terrain only: `progress.jsonl`, `progress.csv` | one row per record with its audit |
| terrain only: `decoded.h5`, `summary.json`, `progress.png` | q, v, a, torque, contact wrench of every stored iterate; first pass, best and final iterate; violation against solver time |

`stop_reason` is one of `physical_gate`, `solver_budget`, `wall_budget`,
`native_converged_infeasible`, `native_stopped_infeasible`, `iteration_budget`. Logs:
`<out_dir>.log`, terrain also `<out_dir>.postprocess.log`. `run_all.sh` also writes
`<out_root>/run_all.tsv`. During a terrain solve, `iterations.h5` and `iterations.jsonl` are
written to `<stage root>/<robot>_<timestamp>_<pid>/` and moved into `<out_dir>` when the
solve ends.

Final iterate and compute-budget candidates of a run (fence: needs `iterations.h5`; terrain:
needs `progress.csv`, i.e. after post-processing):

```bash
$CROCODDYL_ENV_PREFIX/bin/python tools/derive_candidates.py <fence|terrain> <robot> <out_dir> [--out <cand_dir>]
```

It prints `final_iterate` (candidate id of the last valid iterate: fence = record index in
`iterations.h5`, terrain = row index of `progress.csv`) and `final_iterate_solver_s` (its
recorded solver time). `--out` writes `<cand_dir>/plan.json`: the iterate retained at each
budget (0, then 24 budgets per decade from 0.042 s to 14399 s; the valid iterate with the
smallest node violation recorded up to the budget, the last valid iterate from the run's end
on). Any Python with numpy and h5py runs it.

## 4. Tracking a run

Environment and interpreter of the tracker: `new_src/tracking/README.md` (section
Environment). The controller configuration is selected by `LEAPTRACK_PROFILE`, as described
there.

```bash
cd ../../tracking
export PY=<tracker python> MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=<absolute path of out_dir>
# the final iterate
$PY ../baselines/crocoddyl/tools/derive_candidates.py fence hopper2d $RUN
taskset -c 0 $PY scripts/track.py --task fence --robot hopper2d --method crocoddyl \
    --run-dir $RUN --candidate <final_iterate> --out tracked/fence/crocoddyl/hopper2d
# any recorded iterate: --candidate <id>; the run's own selection (lowest recorded violation): --candidate best
```

`track.py` writes `rollout.npz`, `metrics.json`, `trajectory.npz`, `evaluation.json`,
`evaluation.npz`, `run.log` and the problem yaml into `--out`.

Every candidate of the compute-budget grid:

```bash
$PY ../baselines/crocoddyl/tools/derive_candidates.py fence hopper2d $RUN --out <cand_dir>
$PY scripts/budget_study.py plan --baselines <cand_dir>/plan.json --methods crocoddyl --out <study>
$PY scripts/budget_study.py run --out <study> --methods crocoddyl --cores 0-11
$PY scripts/budget_study.py collect --out <study>
```
