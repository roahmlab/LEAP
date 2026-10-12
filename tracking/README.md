# MuJoCo tracking study

The commands below do four things:

* track a solution archive written by `new_src/build/leap_solve`, or a Crocoddyl or
  Aligator run, in MuJoCo with a whole-body inverse-dynamics QP controller;
* score the final state against the problem's terminal goal (Table III);
* run the compute-budget study (Fig. 3);
* render the tracking time-lapse (Fig. 2).

Run them from `new_src/tracking/`.

## Layout

```
leaptrack/            the package
  archive.py          read and interpolate a solution archive
  reference.py        reference trajectories, contact schedule, touchdown clock
  spec_fence.py, spec_terrain.py, plan.py, ik.py, robot.py, geometry.py
                      problem yaml -> scene geometry, gait plan, terminal goal
  scene.py            the MuJoCo model (robot, floor, fence or stepping stones)
  plastic_impact.py   the impact map at contact-make events
  wbc.py              the whole-body QP controller (WholeBodyQP, ContactPolicy, WbcGains)
  controllers.py      gains and options per task and robot; the controller configurations
  determinism.py      ProxQP build and thread settings
  rollout.py          the closed loop
  recorded.py         Crocoddyl and Aligator run folders as references
  evaluate.py         constraint families and final state of a rollout
  score.py            the Table III value
  report.py           verdicts, plots, REPORT.md
  budget.py           the budget study
  figure.py           Fig. 3
figure2/              the Fig. 2 renderer
  assets/             SHA256SUMS and sources (ASSETS.md) of the robot meshes, the font
scripts/
  track.py            track one case, a task, or k repeats of a case
  table3.py           Table III from tracked folders
  budget_study.py     plan | run | collect | plot of the budget study
  figure2.py          Fig. 2
  fetch_figure2_assets.py   download the robot meshes of Fig. 2
  report.py           REPORT.md of a tracked folder
  compare_rollouts.py compare two case folders
environment.yml       conda environment
requirements.txt      pip requirements (same versions)
```

## Environment

```bash
conda env create -f environment.yml && conda activate leap_track
export PY=$(which python) MUJOCO_GL=egl
```

Import `leaptrack` before `proxsuite` in your own scripts: the package selects ProxQP's
plain build and sets one thread per math library.

## Track LEAP solutions

```bash
# one archive -> <out>/{rollout.npz, trajectory.npz, evaluation.json, evaluation.npz, metrics.json, run.log, <problem>.yaml}
taskset -c 0 $PY scripts/track.py --task fence --robot go2 --archive ../results/fence/sol_fence_go2.h5 --out tracked/fence/leap/go2
# every robot of a task, archives named sol_<task>_<robot>[_<tag>].h5, one core per robot
$PY scripts/track.py --task terrain --all --archive-dir ../results/terrain --archive-tag repro --out tracked/terrain/leap --cores 0-4
# k repeats of one case -> <out>/rep<k>/
$PY scripts/track.py --task fence --robot h1 --archive ../results/fence/sol_fence_h1.h5 --out tracked/fence/leap/h1 --repeat 3 --cores 0-2
```

The problem yaml is read from the archive; `--problem-yaml` overrides it.

## Track Crocoddyl and Aligator runs

```bash
# a run folder written by ../baselines/; <id> = its final iterate (see ../baselines/<method>/README.md)
taskset -c 0 $PY scripts/track.py --task fence --robot h1 --method aligator --run-dir <run folder> --candidate <id> --out tracked/fence/aligator/h1
# any other candidate of the run (Crocoddyl: record index; Aligator: phase:call:sample), or its own selection
taskset -c 0 $PY scripts/track.py --task fence --robot h1 --method aligator --run-dir <run folder> --candidate 3:32:152 --out <dir>
taskset -c 0 $PY scripts/track.py --task fence --robot h1 --method aligator --run-dir <run folder> --candidate best --out <dir>
```

`--run-dir` and `--candidate` are both required with `--method`; one call tracks one run.
`$PY -m leaptrack.recorded show --task <task> --method <method> --robot <robot> --run-dir <run folder> --candidate <id>`
prints what the reference built from a candidate looks like without rolling it out.

## Table III

```bash
$PY scripts/table3.py tracked                    # every case below tracked/
$PY scripts/table3.py tracked --csv t3.csv       # also every rollout as a CSV row
```

The value printed per case is `max_i e_i / e_bar_i` of the final state. The six errors and
their thresholds are:

| error | threshold |
|---|--:|
| base position | 0.05 m |
| base orientation | 0.10 rad |
| maximum actuated-joint error | 0.20 rad |
| maximum foot error | 0.04 m |
| base speed | 0.10 m/s |
| maximum joint speed | 0.10 rad/s |

A value below 1 means the goal is reached. Fence Hopper is judged at the end of its plan;
every other case at the end of the 1.5 s hold.

## Budget study (Fig. 3)

```bash
# 1. solve with snapshots, from new_src/ (or SNAPSHOTS=1 scripts/reproduce_paper.sh for all ten)
scripts/run_benchmark.sh fence go2 0-11 snap output.snapshots=results/fence/snapshots_go2
# 2. plan (reads <root>/<task>/snapshots_<robot>/); --baselines <plan.json> ... adds the Crocoddyl and
#    Aligator series of runs made with ../baselines/ (their tools/derive_candidates.py --out)
cd tracking; $PY scripts/budget_study.py plan --snapshots ../results --out budget --baselines <cand>/*/plan.json
# 3. one rollout per distinct candidate, one per core
$PY scripts/budget_study.py run --out budget --cores 0-35 --python $PY
# 4. table and figure
$PY scripts/budget_study.py collect --out budget
$PY scripts/budget_study.py plot --out budget
```

Each step writes into the `--out` folder:

* `budget/<task>/<method>/<robot>/<candidate>/` holds each candidate's rollout and
  `summary.json`.
* `results.csv` and `results.json` hold the table.
* The figure is
  `budget/figure/rollout_violation_under_time_budget_combined_normalized_terminal_log.{pdf,png,svg,csv,json}`;
  the `.json` lists the earliest trackable budget per series.

Options:

* `run --methods leap` runs one method only.
* `run --track-args ...`, placed last on the line, passes flags to every rollout.

## Time-lapse (Fig. 2)

This needs `pdfinfo` (poppler-utils) on `PATH`.

```bash
$PY scripts/fetch_figure2_assets.py                                   # robot meshes -> figure2/assets/ (once)
$PY scripts/track.py --task fence   --all --archive-dir <fence archives>   --archive-tag <tag> --out tracked/fence/leap   --cores 0-4
$PY scripts/track.py --task terrain --all --archive-dir <terrain archives> --archive-tag <tag> --out tracked/terrain/leap --cores 0-4
$PY scripts/figure2.py --fence tracked/fence/leap --terrain tracked/terrain/leap --out fig2
```

**`fetch_figure2_assets.py`**

* Downloads the 118 mesh files (71 MB) listed in `figure2/assets/SHA256SUMS` and checks
  each against it. Files already present and verified are skipped. The sources and
  licenses are in `figure2/assets/ASSETS.md`.
* `--check` verifies only. `--dest DIR` downloads into `DIR`; then pass `--assets DIR`
  to `figure2.py`.

**Inputs of `figure2.py`**

* `--fence`, `--terrain`: the `--out` folders of `track.py --task <task> --all`.
* Solution archives: by default, the `archive` path in each case's `metrics.json`.
  Otherwise use `--fence-archive-dir DIR [--fence-archive-tag TAG]` and
  `--terrain-archive-dir DIR [--terrain-archive-tag TAG]`, for
  `DIR/sol_<task>_<robot>[_<TAG>].h5`.
* `--assets DIR`: the mesh folder (default `figure2/assets/`).

**Outputs (under `--out`)**

* `combined/leap_tracking_timelapse_combined.{png,pdf}`: Fig. 2.
* `fence/leap_fence_tracking_timelapse_transposed.{png,pdf}` and
  `terrain/leap_terrain_tracking_timelapse_transposed.{png,pdf}`.
* `src_fence/{isometric,side}_dense_gradient/` and `src_terrain/{isometric,side}/`: the
  per-view renders. Each holds a PNG, a PDF and a JSON file per robot, the view's
  combined figure, `manifest.json` and `validation.json`.

**Options of `figure2.py`**

* `--reuse` composes from the renders under `--out` and renders nothing.
* `--force-render` re-renders every view.
* `--figures fence terrain combined` selects the figures to make (default: all three).

The unit tests run with `$PY -m unittest figure2.test_sampling`.

## Options

* **Controller configuration.** The environment variable `LEAPTRACK_PROFILE` selects it:
  * `impact_aware` (default). For fence `h1` and fence `talos` it adds the impact
    projection window (`impact_window` 0.03 s, `impact_projection`) and the CoM task
    (`w_com` 20); for fence `h1`, also the swing gains of `g1`.
  * `published`: the per-task and per-robot tables of `controllers.py` without these
    additions.

  `wp4` is accepted as another name for `impact_aware`. Every rollout records the
  configuration in `gains.config_profile`.
* **Rollout options.** `track.py` accepts `--hold`, `--settle`, `--clock`,
  `--early-jump`, `--friction`, and every `WbcGains` field as `--<field>`; see
  `track.py --help`.
* **Repeats.** Compare repeated rollouts with `$PY scripts/compare_rollouts.py --reps <dir>`,
  or two case folders with `$PY scripts/compare_rollouts.py <case_a> <case_b>`.
* **Reports.** `$PY scripts/report.py --task <task> --root <tracked task folder>` writes
  `REPORT.md` and plots.
