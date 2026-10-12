# Crocoddyl and Aligator

Each method has its own folder with its environment, code and run scripts:

| method | folder |
|---|---|
| Crocoddyl 3.2.1 | `crocoddyl/README.md` |
| Aligator 0.19.1 | `aligator/README.md` |

Both write one run folder per instance, `<root>/<task>/<robot>/`, for task `terrain`
(`hopper2d_foot`, `go2`, `h1`, `g1_29dof`, `talos`) and task `fence` (`hopper2d`, `go2`,
`h1`, `g1_29dof`, `talos`).

## 1. Run the ten instances

```bash
cd crocoddyl
./setup.sh --create-env
./run_all.sh <root>/crocoddyl --cpus 0-11

cd ../aligator
conda create -p <prefix> --file conda-linux-64.txt
export ALIGATOR_PYTHON=<prefix>/bin/python
./build.sh
./run_all.sh <root>/aligator --cpus 0-11
```

## 2. Table II: solve times

The solve time of a run is the recorded solve time of its final iterate.

```bash
# Crocoddyl: prints final_iterate and final_iterate_solver_s
$CROCODDYL_ENV_PREFIX/bin/python crocoddyl/tools/derive_candidates.py <task> <robot> <root>/crocoddyl/<task>/<robot>
# Aligator: prints the candidate and its time_s
$ALIGATOR_PYTHON aligator/final_iterate.py <root>/aligator/<task>/<robot> -v
```

## 3. Table III: tracking

From `new_src/tracking`, with the tracker's environment (`new_src/docs/REPRODUCE.md`
section 1.2), for each run. `<id>` is the final iterate printed in section 2:

```bash
$PY scripts/track.py --task <task> --robot <robot> --method crocoddyl \
    --run-dir <root>/crocoddyl/<task>/<robot> --candidate <id> --out <tracked>/<task>/crocoddyl/<robot>
$PY scripts/track.py --task <task> --robot <robot> --method aligator \
    --run-dir <root>/aligator/<task>/<robot> --candidate <id> --out <tracked>/<task>/aligator/<robot>
$PY scripts/table3.py <tracked>
```

## 4. Fig. 3: compute-budget candidates

The candidates of each run, as a `plan.json` per run:

```bash
$CROCODDYL_ENV_PREFIX/bin/python crocoddyl/tools/derive_candidates.py <task> <robot> <root>/crocoddyl/<task>/<robot> \
    --out <cand>/crocoddyl_<task>_<robot>
$ALIGATOR_PYTHON aligator/tools/derive_candidates.py <task> <robot> <root>/aligator/<task>/<robot> \
    --out <cand>/aligator_<task>_<robot>
```

Then, from `new_src/tracking`, the budget study with LEAP's snapshots
(`new_src/docs/REPRODUCE.md` section 4) and these plans:

```bash
$PY scripts/budget_study.py plan --snapshots <LEAP results root> --baselines <cand>/*/plan.json --out <study>
$PY scripts/budget_study.py run --out <study> --cores 0-35 --python $PY
$PY scripts/budget_study.py collect --out <study>
$PY scripts/budget_study.py plot --out <study>
```
