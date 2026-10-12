"""The budget study (Fig. 3): plan, run, collect, plot.

    PY=<the tracker's python>      # see ../README.md
    # 1. solve with snapshots (one directory per instance, written by the solver):
    #      scripts/run_benchmark.sh fence go2 0-11 snap output.snapshots=results/fence/snapshots_go2
    # 2. plan the candidates every budget retains, from the snapshot indices
    $PY scripts/budget_study.py plan --snapshots <root-with-fence/snapshots_<robot>,terrain/snapshots_<robot>> --out <study> [--baselines <plan.json> ...]
    # 3. one MuJoCo rollout per distinct candidate (subprocesses pinned to --cores)
    $PY scripts/budget_study.py run  --out <study> --cores 0-35 [--only terrain:go2 fence:hopper2d]
    # 4. the result table and the figure
    $PY scripts/budget_study.py collect --out <study>
    $PY scripts/budget_study.py plot --out <study> [--baselines <results.json of another study>]

`--snapshots <root>` expects `<root>/<task>/snapshots_<robot>/snapshots.csv` for every
instance it finds (missing instances are skipped); `--snapshot-dir task:robot:<dir>` adds
one explicitly. The grid is 24 budgets per decade from 0.042 s to 14 399 s plus budget 0,
unless --per-decade/--t-min/--t-max say otherwise.

`plan --baselines <plan.json> ...` adds the Crocoddyl and Aligator series of the given
plan files (those `new_src/baselines/<method>/tools/derive_candidates.py --out` writes;
`--methods`, `--tasks`, `--robots` narrow them): the recorded candidate every budget
retains. `run` rolls them out with `scripts/track.py --method <m>
--candidate <id>`, with the same `--track-args` as the LEAP candidates; `run --methods
crocoddyl,aligator` runs only those. `collect` tabulates every method, and `plot` draws a
baseline series from this study's rollouts when it rolled any of its candidates out, from
`plot --baselines <results.json>` otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from leaptrack import budget  # noqa: E402

TASKS = {"terrain": ["hopper2d_foot", "go2", "h1", "g1_29dof", "talos"], "fence": ["hopper2d", "go2", "h1", "g1_29dof", "talos"]}


def parse_cores(spec: str | None):
    if not spec:
        return None
    cores = []
    for part in spec.split(","):
        a, b = (part.split("-") + [part])[:2]
        cores += list(range(int(a), int(b) + 1))
    return cores


def cmd_plan(args):
    grid = budget.budget_grid(args.per_decade, args.t_min, args.t_max)
    series = []
    dirs = []
    if args.snapshots:
        for task, robots in TASKS.items():
            for robot in robots:
                d = Path(args.snapshots) / task / f"snapshots_{robot}"
                if (d / "snapshots.csv").exists():
                    dirs.append((task, robot, d))
    for spec in args.snapshot_dir or []:
        task, robot, d = spec.split(":", 2)
        dirs.append((task, robot, Path(d)))
    if not dirs and args.baselines is None:
        raise SystemExit("no snapshot directories found")
    for task, robot, d in dirs:
        s = budget.plan_series(task, robot, d, grid)
        series.append(s)
        print(f"[plan] {task:8s} {robot:14s} {s['n_snapshots']:4d} snapshots, {len(s['candidates']):3d} distinct candidate(s) over "
              f"{len(grid)} budgets, run ended at {s['last_time_s']:.1f} s")
    if args.baselines is not None:
        split = lambda x: [v for v in x.split(",") if v] if x else None  # noqa: E731
        for path in args.baselines:
            for s in budget.recorded_series(path, methods=split(args.methods) or ("crocoddyl", "aligator"),
                                            tasks=split(args.tasks), robots=split(args.robots)):
                series.append(s)
                print(f"[plan] {s['task']:8s} {s['robot']:14s} {s['method']:10s} {s['samples']:5d} recorded samples, "
                      f"{len(s['candidates']):3d} distinct candidate(s) ({s['baseline_plan']}), run ended at {s['last_time_s']:.1f} s")
    plan = budget.make_plan(series, grid, args.per_decade, args.t_min, args.t_max)
    if args.baselines is not None:
        plan["baselines_plan"] = sorted({s["baseline_plan"] for s in series if budget.is_recorded(s)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(f"[plan] {sum(len(s['candidates']) for s in series)} rollouts planned -> {out / 'plan.json'}")


def cmd_run(args):
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    cores = parse_cores(args.cores)
    only = {tuple(x.split(":")) for x in args.only} if args.only else None
    python = Path(args.python) if args.python else budget.default_python()
    methods = [m for m in args.methods.split(",") if m] if args.methods else None
    budget.run_series(plan, out, python, jobs=args.jobs if not cores else len(cores), cores=cores, extra=args.track_args, only=only,
                      redo=args.redo, methods=methods)
    budget.collect(plan, out)


def cmd_collect(args):
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    res = budget.collect(plan, out)
    n = {}
    for r in res["rows"]:
        n[r["status"]] = n.get(r["status"], 0) + 1
    print(f"[collect] {len(res['rows'])} candidates: {n} -> {out / 'results.json'}")


def cmd_plot(args):
    from leaptrack import figure
    out = Path(args.out)
    pdf, summary = figure.make_figure(out / "results.json", out / "figure", baselines_json=args.baselines, hopper_joints=args.hopper_joints)
    for s in summary:
        print(f"[plot] {s['task']:8s} {s['robot']:14s} {s['method']:10s} evaluated {s['evaluated']:3d}/{s['budgets']:3d} "
              f"earliest trackable budget {s['earliest_trackable_budget_s']} s  final {s['final_rollout_metric']}")
    print(f"[plot] figure: {pdf}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--snapshots", type=Path, default=None, help="root holding <task>/snapshots_<robot>/")
    p.add_argument("--snapshot-dir", action="append", help="task:robot:<dir>")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-decade", type=int, default=budget.PER_DECADE)
    p.add_argument("--t-min", type=float, default=budget.T_MIN)
    p.add_argument("--t-max", type=float, default=budget.T_MAX)
    p.add_argument("--baselines", nargs="+", default=None, metavar="PLAN_JSON",
                   help="add the Crocoddyl / Aligator series of these plan.json files "
                        "(written by new_src/baselines/<method>/tools/derive_candidates.py --out)")
    p.add_argument("--methods", default=None, help="with --baselines: comma list (default crocoddyl,aligator)")
    p.add_argument("--tasks", default=None, help="with --baselines: comma list of tasks")
    p.add_argument("--robots", default=None, help="with --baselines: comma list of robots")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cores", default=None, help="taskset list, one rollout per core (e.g. 64-99)")
    p.add_argument("--jobs", type=int, default=8, help="parallel rollouts when --cores is not given")
    p.add_argument("--only", nargs="*", help="task:robot ...")
    p.add_argument("--methods", default=None, help="comma list of methods to run (default: every series of the plan)")
    p.add_argument("--python", default=None, help="the simulation interpreter (default: $LEAP_SIM_PYTHON or this one)")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--track-args", nargs=argparse.REMAINDER, default=None, help="extra flags passed to scripts/track.py")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("collect")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("plot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baselines", type=Path, default=None, help="results.json of another study (Crocoddyl and Aligator rows)")
    p.add_argument("--hopper-joints", type=int, default=2)
    p.set_defaults(fn=cmd_plot)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
