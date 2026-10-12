# Aligator baseline

Solves the ten problems with Aligator's `SolverFDDP` (aligator 0.19.1) from the LEAP cold-seed
archives and writes one run folder per problem, in the layout `new_src/tracking` reads:

| task | robots |
|---|---|
| `terrain` | `hopper2d_foot` `go2` `h1` `g1_29dof` `talos` |
| `fence` | `hopper2d` `go2` `h1` `g1_29dof` `talos` |

Layout: `src/` holds the code and the inputs (paths inside it are relative to `src/`); `build.sh`,
`run.sh`, `run_all.sh`, `final_iterate.py` and `tools/derive_candidates.py` are the entry points.

## 1. Environment and build

Requirements: Linux x86-64 with AVX2, conda (or mamba), 12 physical cores per instance, `cc` on
`PATH`, `taskset`, `gzip`.

```bash
cd new_src/baselines/aligator
conda create -p <prefix> --file conda-linux-64.txt          # exact package set (linux-64)
#   or: conda env create -p <prefix> -f environment.yml     # the main packages pinned to the same builds
export ALIGATOR_PYTHON=<prefix>/bin/python
./build.sh
```

`build.sh` (run once):

1. restores the ten seed archives `src/**/seed_*.h5` from their `.h5.gz` copies;
2. compiles seven Python extensions in place (`python setup_*.py build_ext --inplace`, logs in
   `build_logs/`):

| module | sources (under `src/comparisons/`) |
|---|---|
| `aligator_bench._generated_sparse` | `aligator_bench/setup_sparse.py`, `generated_sparse.cpp` |
| `aligator_bench._hybrid_model` | `aligator_bench/setup_hybrid.py`, `hybrid_model.cpp`, `inverse_model.hpp` |
| `_public_models` | `aligator_unified_fence/scripts/setup_public_models.py`, `public_models.cpp` |
| `_public_run` | `aligator_unified_fence/scripts/setup_public_run.py`, `public_run.cpp` |
| `_trace_run` (fence) | `aligator_unified_fence/fence_benchmark_along_the_way/setup_trace.py`, `trace_run.cpp` |
| `_trace_run` (terrain) | `aligator_unified_terrain/along_the_way/setup_trace.py`, `trace_run.cpp` |
| `_terrain_models` | `aligator_unified_terrain/scripts/setup_terrain_models.py`, `terrain_models.cpp` |

Compilers: `CC` / `CXX`, default `<prefix>/bin/cc` and `<prefix>/bin/c++` (conda-forge GCC 16.2.0).

At run time, the first solve of each robot and control hold generates C code with CasADi and compiles
it with `cc -O1` (the `cc` on `PATH`) into `src/comparisons/aligator_unified_fence/generated/` or
`src/comparisons/aligator_unified_terrain/generated/`; later runs reuse it.

## 2. Running

One instance:

```bash
./run.sh <fence|terrain> <robot> <out_dir> [--cpus LIST] [--python PATH]
# e.g.
./run.sh fence hopper2d /data/aligator/fence/hopper2d --cpus 36-47
```

All ten, one after the other (`<root>/<task>/<robot>/`, log `<root>/<task>/<robot>.log`; order:
terrain hopper2d_foot, go2, h1, g1_29dof, talos, then fence hopper2d, go2, h1, g1_29dof, talos):

```bash
./run_all.sh <root> [--cpus LIST] [--python PATH]
```

`--cpus`: 12 logical CPUs on 12 distinct physical cores (default: the first 12 physical cores of the
current CPU affinity). `--python`: default `$ALIGATOR_PYTHON`, else `python` on `PATH`. `<out_dir>` must
not exist. Each solve uses 12 OpenMP threads pinned to these CPUs; BLAS runs single-threaded.

### Terrain instances

`run.sh` runs (`T=src/comparisons/aligator_unified_terrain/terrain_benchmark_along_the_way`):

```bash
python $T/run.py --robot <name> --cpus <cpus> --out <out_dir>
```

with `<name>` = `hopper2d_foot`, `go2`, `h1`, `g1_29dof_repaired` (robot `g1_29dof`) or `talos`.
`run.py` runs a sequence of phases, each one call of
`src/comparisons/aligator_unified_terrain/scripts/solve_public.py --robot <key> --out <out_dir>/phase_NN
--threads 12 --core-offset 0 --store-iterations ...` under `taskset -c <cpus>` (`<key>` = `<name>`,
except `h1_ap15_ank07_att02_torso05` for `h1`). Within a phase, the calls stop at the first returned
trajectory whose maximum constraint violation is <= 1e-3 at the nodes and on the 0.01 s grid. Phases
run until one ends that way (go2 and talos: not before the end of the listed phases) or until the
cumulative solve time reaches 14400 s; then `run.py` runs `$T/postprocess.py <out_dir> --workers 12`.
The phases:

* `go2`, `talos` (options read from `src/comparisons/aligator_unified_terrain/reports/selected.json`
  and the `result.json` files under `src/comparisons/aligator_unified_terrain/runs/`), common options
  `--obstacle-scale 1.0 --rounds 1 --call-seconds 86400 --reg-init 0.001 --tolerance 1e-10
  --geometry-weight 1.0 --clearance-weight 1.0`, each phase after the first with
  `--resume <out_dir>/phase_<previous>/solution.npz`:

  | robot | phase | `--dt` | `--hold` | `--iterations` | `--cost-threshold` | `--mu-init` |
  |---|---|---|---|--:|---|---|
  | go2 | 00 | 0.01 | linear | 190 | 1e-07 | 0.1 |
  | talos | 00 | 0.01 | linear | 31 | 1e-07 | 0.1 |
  | talos | 01 | 0.01 | linear | 28 | 1e-08 | 0.1 |
  | talos | 02 | 0.01 | linear | 28 | 9.999999999999999e-10 | 0.1 |
  | talos | 03 | 0.01 | linear | 28 | 9.999999999999999e-11 | 0.1 |
  | talos | 04 | 0.01 | quadratic | 14 | 1e-07 | - |
  | talos | 05 | 0.01 | quadratic | 16 | 1e-08 | - |

  If the last of these does not meet the 1e-3 test, further phases repeat its options with
  `--iterations 10000`, without `--cost-threshold`, resuming from the previous phase, with
  `--rounds` / `--call-seconds` set from the remaining budget (at most 4 calls of 300 s).
* `hopper2d_foot`, `h1`, `g1_29dof_repaired`, each phase with `--rounds min(4, ceil(r/300))
  --call-seconds min(300, r)` (`r` = remaining budget in seconds):

  | phase | options |
  |---|---|
  | 00 | `--dt 0.01 --hold linear` |
  | 01 | `--dt 0.01 --hold quadratic --resume <out_dir>/phase_00/solution.npz` |
  | 02 | `--dt 0.005 --hold quadratic --resume <out_dir>/phase_01/solution.npz` |
  | 03, 04, ... | `--dt 0.005 --hold quadratic --resume <out_dir>/phase_<previous>/solution.npz` |

### Fence instances

`run.sh` runs these steps (`F=src/comparisons/aligator_unified_fence/fence_benchmark_along_the_way`,
every command under `taskset -c <cpus>`):

| step | command | when |
|---|---|---|
| phase 00 | `python $F/run.py --robot <robot> --out <out_dir> --core-offset 0 --budget 1200 --call-seconds 300 --rounds 4 --iterations 10000 --dt 0.005 --no-postprocess` | always |
| audit | `python $F/postprocess.py <out_dir> --workers 12` | the trajectory returned by phase 00 is not finite (`run.py` then exits with status 1) |
| phase 01 | `python $F/extend_run.py <out_dir> --budget 14400 --max-calls N1 --no-postprocess` | always (if phase 00 passed, this only sets the budget to 14400 s) |
| phase 02 | `python $F/extend_run.py <out_dir> --budget 14400 --geometry-weight 100 --max-calls N2 --no-postprocess` | not passed and solve time < 14400 s |
| phase 03 | `python $F/extend_run.py <out_dir> --budget 14400 --geometry-weight 100 --dual-step 1 --continue-duals [--max-calls 12] --no-postprocess` | not passed and solve time < 14400 s; `--max-calls 12` for talos only |
| audit | `python $F/postprocess.py <out_dir> --workers 12` | always |

`N1` = 5 for h1, 4 otherwise; `N2` = 10 for h1, 8 otherwise. "Passed": the trajectory returned by the
last call has a maximum constraint violation <= 1e-3 at the nodes and on the 0.01 s grid
(`summary.json` field `returned_audit.passed`); every phase stops after such a call, after its call
limit, or when the cumulative solve time reaches its budget.

## 3. Inputs

All under `src/`:

| input | fence | terrain |
|---|---|---|
| problem | `showcase_v2/unified/impact_cone/problems/<robot>_jump_fence.yaml` | `showcase_v2/unified/flatfoot_20260913/problems/hopper2d_foot_terrain_walk.yaml`, `showcase_v2/terrain/problems/go2_terrain_walk.yaml`, `showcase_v2/unified/trackable_20260913/problems/h1_ap15_ank07_att02_torso05.yaml`, `comparisons_leap_benchmark/terrain/problems/g1_29dof_terrain_walk.yaml`, `showcase_v2/terrain/problems/talos_terrain_walk.yaml` |
| seed archive | `showcase_v2/unified/impact_cone/initial_guess/archives/seed_fence_<robot>.h5` | `showcase_v2/unified/initial_guess/archives/seed_terrain_{hopper2d_foot,go2,h1_ap15_ank07_att02_torso05,talos}.h5`, `comparisons_leap_benchmark/terrain/initial_guess/archives/seed_terrain_g1_29dof.h5` |
| robot model | `robots/<robot>.yaml`, `assets/<robot>.urdf` | same |
| phase options | - | `comparisons/aligator_unified_terrain/reports/selected.json`, `comparisons/aligator_unified_terrain/runs/{go2_cold,talos_cold,talos_quadratic}/result.json` |

## 4. Outputs

Fence run folder `<out_dir>/`:

* `plan.json` (command, settings, budget), `summary.json` (phases, cumulative solve time
  `solve_compute_s`, returned-trajectory audit; after the audit: `last_within_budget`,
  `best_observed`, `first_pass_time_s`, ...), `reference_inputs/` (copies of the problem yaml, robot
  yaml, URDF and seed archive);
* one folder per phase, `00_current_impact_cone/`, `01_budget_extension/`, ...: `config.json`,
  `result.json` (per call: clocks, native iterations, audit), `initial.npz`, `input_XX.npz` and
  `call_XX.npz` (input and returned trajectory of call XX), `solution.npz`, `trace_XX.bin` (every
  iterate of call XX: float64, `result.json` `calls[XX].rows` x `width`), `model_source/`;
  extension phases also `parent_candidate.npz`, `duals_before_XX.json`, `duals_after_XX.json`,
  `extension_runner_source.py` (phase 03: `parent_duals.json`);
* written by the audit: `iterations.csv` (one row per stored iterate: `phase`, `call`, `sample`,
  `time_s` = cumulative solve time, `node_max`, `fine_max`, `within_budget`, ...),
  `best_observed_solution.npz`, `deadline_solution.npz`, `last_finite_solution.npz`, and per phase
  `constraint_catalog.json`, `diagnostics/call_XX_YYYYYY.h5`.

Terrain run folder `<out_dir>/`:

* `experiment.json` (protocol, every phase command, exit code and solve time; cumulative `solve_s`),
  `phase_NN.log`, `reference_inputs/`, `source_snapshot/`;
* `phase_NN/`: `config.json`, `result.json`, `initial.npz`, `imported_seed_fields.npz`, `call_JJ.npz`,
  `history_JJ.npz`, `solution.npz`, `trace_JJ.primal.bin`, `trace_JJ.metrics.bin`, `executed_source/`;
* written by the audit: `iterations.csv` (same columns as above), `summary.json` (`solve_s`, `final`,
  `best_observed`, `first_pass_1e3`, ...), `best_observed_solution.npz`, `final_solution.npz`,
  `final_solution.h5`, `first_pass_1e3_solution.npz`, `first_pass_2e3_solution.npz` (when reached),
  `REPORT.md`, `*.png`, and per phase `constraint_catalog.json`, `diagnostics/`, `trajectory.h5`.

Final iterate and compute-budget candidates of a run folder:

```bash
# candidate id phase:call:sample of the last iterate within the 14400 s budget
# (last row of iterations.csv with within_budget = True)
$ALIGATOR_PYTHON final_iterate.py <out_dir>
$ALIGATOR_PYTHON final_iterate.py <out_dir> -v    # with its time_s, node_max, fine_max
# final iterate (JSON); with --out, <dir>/plan.json: the candidate retained at budget 0 and at 24
# budgets per decade from 0.042 s to 14399 s (the valid iterate with the smallest node_max recorded
# up to the budget; from the run's end on, its last valid iterate)
$ALIGATOR_PYTHON tools/derive_candidates.py <fence|terrain> <robot> <out_dir> --out <dir>
```

`<robot>` as in `run.sh`.

## 5. Tracking a run with `new_src/tracking/scripts/track.py`

With the tracking environment of `new_src/tracking/README.md` (section "Environment"):

```bash
cd new_src/tracking
PY=<python of the tracking environment>
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=<out_dir>                 # a fence or terrain run folder written by run.sh
ID=$($ALIGATOR_PYTHON ../baselines/aligator/final_iterate.py $RUN)
# the last iterate within the budget
taskset -c <cpu> $PY scripts/track.py --task fence --robot go2 --method aligator --run-dir $RUN \
    --candidate $ID --out <track_dir>
# the run's best observed candidate (best_observed_solution.npz)
taskset -c <cpu> $PY scripts/track.py --task fence --robot go2 --method aligator --run-dir $RUN \
    --candidate best --out <track_dir_best>
```

`--task` / `--robot` as in `run.sh` (terrain g1: `--robot g1_29dof`). `--candidate` also accepts any
`phase:call:sample` of `iterations.csv`. The controller configuration is chosen with
`LEAPTRACK_PROFILE` (see `new_src/tracking/README.md`). `<track_dir>` receives `rollout.npz`,
`metrics.json`, `trajectory.npz`, `evaluation.json`, `evaluation.npz`, `run.log` and the problem yaml.

Budget study (one rollout per candidate of a `tools/derive_candidates.py --out <dir>` plan):

```bash
$PY scripts/budget_study.py plan --baselines <dir>/plan.json --methods aligator --out <study>
$PY scripts/budget_study.py run --out <study> --cores <cpus> --methods aligator    # also writes <study>/results.json
```
