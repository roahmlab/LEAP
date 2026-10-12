# Reproducing the paper's results

| result | section |
|---|---|
| Table II, LEAP's solve times | 2 |
| Table III, tracking ratios of every method's solution | 3 |
| Fig. 3, the compute-budget study | 4 |
| Fig. 2, the tracking time-lapse | 5 |
| Crocoddyl and Aligator runs | `baselines/README.md` |

All commands run from `new_src/` unless a `cd` says otherwise.

## 1. Setup

### 1.1 Build the solver

```bash
conda env create -f environment.yml && conda activate leap_build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/c++
cmake --build build -j
ctest --test-dir build --output-on-failure        # 7 tests
```

CHOLMOD and LAPACK are optional; the configure log reports whether they were found.

### 1.2 Create the tracker environment

```bash
conda env create -f tracking/environment.yml       # environment "leap_track"
conda activate leap_track
export PY=$(which python) MUJOCO_GL=egl
```

### 1.3 Cores

Every solve runs on 12 physical cores, given as a `taskset` core list. Rollouts run on one
core each. Use idle physical cores, and never put two jobs on SMT siblings of the same
core.

## 2. Table II: solve the ten instances

The problem files are `problems/<robot>_jump_fence.yaml` and
`problems/<robot>_terrain_walk.yaml`.

All ten instances, in parallel lanes of 12 cores each:

```bash
SNAPSHOTS=1 CORE_BASE=0 scripts/reproduce_paper.sh results/repro 4 repro
```

* **Lanes.** Lane `i` uses cores `CORE_BASE + 12i` to `CORE_BASE + 12i + 11`.
* **`SNAPSHOTS=1`** also writes the candidates the budget study needs (section 4) to
  `results/repro/<family>/snapshots_<robot>/`.
* **Outputs.** Per instance, the solver log `results/repro/<family>/<family>_<robot>_repro.log`
  and the solution archive `results/repro/<family>/sol_<family>_<robot>_repro.h5`.

One instance at a time (`scripts/run_benchmark.sh <fence|terrain> <robot> <cores> <tag> [key=value ...]`):

```bash
scripts/run_benchmark.sh fence   hopper2d      0-11 repro
scripts/run_benchmark.sh fence   go2           0-11 repro
scripts/run_benchmark.sh fence   h1            0-11 repro
scripts/run_benchmark.sh fence   g1_29dof      0-11 repro
scripts/run_benchmark.sh fence   talos         0-11 repro
scripts/run_benchmark.sh terrain hopper2d_foot 0-11 repro
scripts/run_benchmark.sh terrain go2           0-11 repro
scripts/run_benchmark.sh terrain g1_29dof      0-11 repro
scripts/run_benchmark.sh terrain talos         0-11 repro
scripts/run_benchmark.sh terrain h1            0-11 repro
```

These write to `results/<family>/`, or to the directory in `LEAP_RESULTS_DIR`. To also
write the budget-study candidates, append
`output.snapshots=results/<family>/snapshots_<robot>`.

**Reading the results.** The solve time of Table II is `solvewall=` (seconds) in the
log's line that starts with `[unified]`. That line also has:

* `met=1` when the instance meets the benchmark criterion;
* `gate`, the node violation, and `fine`, the fine-grid violation;
* `total`, the number of accepted steps.

The table of all ten:

```bash
python3 scripts/make_results_table.py results/repro --tag repro
```

## 3. Table III: track the solutions

Each solution is tracked once in MuJoCo by the whole-body controller of
`tracking/leaptrack/`. The protocol is the default of `tracking/scripts/track.py`.

### 3.1 LEAP

```bash
cd tracking
for t in fence terrain; do
  $PY scripts/track.py --task $t --all --archive-dir ../results/repro/$t --archive-tag repro \
      --out ../results/repro/tracked/$t/leap --cores 0-4
done
```

`--cores` gives one core per robot. Each case
directory `tracked/<task>/leap/<robot>/` holds `rollout.npz`, `trajectory.npz`,
`evaluation.json`, `evaluation.npz`, `metrics.json` and `run.log`.

### 3.2 Crocoddyl and Aligator

From runs made with `baselines/` (`baselines/README.md`), one case per run folder.
`<id>` is the run's final iterate, which the method's README shows how to print:

```bash
$PY scripts/track.py --task fence --robot h1 --method crocoddyl --run-dir <run folder> --candidate <id> \
    --out ../results/repro/tracked/fence/crocoddyl/h1
$PY scripts/track.py --task fence --robot h1 --method aligator --run-dir <run folder> --candidate <id> \
    --out ../results/repro/tracked/fence/aligator/h1
```

### 3.3 The table

```bash
$PY scripts/table3.py ../results/repro/tracked
```

This prints the Table III value of every case found below the root, as `max_i e_i /
e_bar_i`. A value below 1 means the goal is reached. Fence Hopper is judged at the end of
its plan and every other case at the end of the hold.

It also prints the number of cases per method that reach the goal. `--csv <file>` writes
every rollout as a CSV row.

## 4. Fig. 3: the compute-budget study

This needs the snapshots of section 2 (`SNAPSHOTS=1`) and, for the Crocoddyl and
Aligator series, the candidate files of their runs (`baselines/README.md` section 4:
`<cand>/<method>_<task>_<robot>/plan.json`).

```bash
cd tracking
$PY scripts/budget_study.py plan --snapshots ../results/repro --out ../results/repro/budget --baselines <cand>/*/plan.json
$PY scripts/budget_study.py run --out ../results/repro/budget --cores 0-35 --python $PY
$PY scripts/budget_study.py collect --out ../results/repro/budget
$PY scripts/budget_study.py plot --out ../results/repro/budget
```

* **`plan`** selects the candidate each budget retains. Without `--baselines` the plan
  holds LEAP only.
* **`run`** rolls out every distinct candidate, one per core.
* **`collect`** writes `results.csv` and `results.json`.
* **`plot`** writes the figure,
  `budget/figure/rollout_violation_under_time_budget_combined_normalized_terminal_log.{pdf,png,svg,csv,json}`.
  The `.json` lists, per series, the earliest budget from which the retained solution
  stays trackable to the end of the run.

## 5. Fig. 2: the tracking time-lapse

This needs the LEAP rollouts of section 3.1, and `pdfinfo` (poppler-utils) on `PATH`.

```bash
cd tracking
$PY scripts/fetch_figure2_assets.py
$PY scripts/figure2.py --fence ../results/repro/tracked/fence/leap --terrain ../results/repro/tracked/terrain/leap \
    --out ../results/repro/fig2
```

* **`fetch_figure2_assets.py`** downloads the robot meshes into `tracking/figure2/assets/`
  and checks them against `figure2/assets/SHA256SUMS` (118 files, 71 MB, once).
* **`figure2.py`** renders the panels and composes the figures; `--reuse` composes from
  the saved renders. Fig. 2 is
  `results/repro/fig2/combined/leap_tracking_timelapse_combined.{png,pdf}`.
