#!/usr/bin/env python
"""Print Table III (`max_i e_i / e_bar_i` per task, robot and method) from tracked
folders. Standard library only.

    python table3.py <root> [<root> ...] [--task fence|terrain] [--csv out.csv]

Every `evaluation.json` below a root is one rollout: `<root>/<task>/<method>/<robot>[/rep<k>]/`,
`<root>/<task>/<robot>[/rep<k>]/`, `<root>/<robot>[/rep<k>]/` (with --task or a `fence` /
`terrain` path component), or a single case directory. Several roots are shown side by
side. A rollout's method is its `evaluation.json["method"]`; with Crocoddyl / Aligator
rollouts present, every row carries a method column and the goal counts per method are
printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from leaptrack.score import JUDGE_AT_T, terminal_ratio  # noqa: E402

ORDER = [("terrain", r) for r in ("hopper2d_foot", "go2", "h1", "g1_29dof", "talos")] + \
        [("fence", r) for r in ("hopper2d", "go2", "h1", "g1_29dof", "talos")]
METHODS = ("leap", "crocoddyl", "aligator")
PROFILE_ALIASES = {"wp4": "impact_aware"}      # controllers.PROFILE_ALIASES (stdlib only here)


def profile_of(evaluation: dict) -> str:
    p = (evaluation.get("gains") or {}).get("config_profile") or "published"
    return PROFILE_ALIASES.get(p, p)


def method_of(evaluation: dict) -> str:
    m = evaluation.get("method", "leap") or "leap"
    return "leap" if m in ("leap", "leap_unified") else m


def task_of(path: Path, robot: str, forced):
    if forced:
        return forced
    for part in reversed(path.parts):
        if part in ("fence", "terrain"):
            return part
    return "terrain" if robot == "hopper2d_foot" else None


def collect(root: Path, forced_task):
    """{(task, robot, method): [(ratio, tag, worst, fell, wall, path, profile)]} under root."""
    out = {}
    root = Path(root)
    evs = sorted(root.rglob("evaluation.json")) if root.is_dir() else []
    for ev in evs:
        d = ev.parent
        robot = d.parent.name if d.name.startswith("rep") and d.name[3:].isdigit() else d.name
        if robot.startswith("rep"):
            continue
        task = task_of(d, robot, forced_task)
        if task is None:
            print(f"{ev}: cannot tell the task, pass --task", file=sys.stderr)
            continue
        e = json.loads(ev.read_text())
        robot = e.get("robot", robot)
        tag, r, worst, _ = terminal_ratio(e, task, robot)
        wall = None
        mt = d / "metrics.json"
        if mt.exists():
            wall = json.loads(mt.read_text()).get("wall_s")
        out.setdefault((task, robot, method_of(e)), []).append(
            (r, tag, worst, bool(e.get("outcome", {}).get("fell")), wall, str(d), profile_of(e)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--task", choices=("fence", "terrain"))
    ap.add_argument("--csv", default=None, help="also write every rollout as a CSV row")
    ap.add_argument("--labels", default=None, help="comma list of column labels for the roots")
    a = ap.parse_args(argv)
    labels = a.labels.split(",") if a.labels else [r.name or str(r) for r in a.roots]
    per_root = [collect(r, a.task) for r in a.roots]
    methods = sorted({k[2] for pr in per_root for k in pr}, key=lambda m: METHODS.index(m) if m in METHODS else 9)
    with_method = any(m != "leap" for m in methods)     # baselines present: method column
    mcol = lambda m: f" {m:10s}" if with_method else ""  # noqa: E731
    cases = sorted({k for pr in per_root for k in pr},
                   key=lambda k: (ORDER.index(k[:2]) if k[:2] in ORDER else 99,
                                  METHODS.index(k[2]) if k[2] in METHODS else 9))

    def fmt_vals(vals):
        return ", ".join(f"{v[0]:.4g}" + ("*" if v[3] else "") for v in vals)

    head = f"{'task':8s} {'robot':14s}" + mcol("method") + f" {'judged':6s}"
    for lab in labels:
        head += f"  {lab:>28s}"
    print(head)
    print("-" * len(head))
    for task, robot, method in cases:
        line = (f"{task:8s} {robot:14s}" + mcol(method)
                + f" {('at_T' if (task, robot) in JUDGE_AT_T else 'end'):6s}")
        for pr in per_root:
            vals = pr.get((task, robot, method), [])
            line += f"  {fmt_vals(vals) if vals else '—':>28s}"
        print(line)
    print("(* = fell)")
    if with_method:
        print("\ngoal counts (cases where every rollout reaches the goal / cases tracked)")
        for lab, pr in zip(labels, per_root):
            parts = []
            for m in methods:
                ks = [k for k in pr if k[2] == m]
                n_goal = sum(all(v[0] < 1 for v in pr[k]) for k in ks)
                parts.append(f"{m} {n_goal}/{len(ks)}")
            print(f"  {lab:12s} " + ", ".join(parts))
    print("\nper rollout: limiting metric and wall time")
    for lab, pr in zip(labels, per_root):
        for (task, robot, method) in cases:
            for r, tag, worst, fell, wall, path, prof in pr.get((task, robot, method), []):
                print(f"  {lab:12s} {task:8s} {robot:14s}" + mcol(method) + f" ratio_{tag:4s} {r:9.4g}  limiting {worst:16s} "
                      f"fell={fell!s:5s} wall={'' if wall is None else f'{wall:.0f}s':>6s}  {prof:12s} {path}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["root", "task", "robot", "tag", "ratio", "limiting", "fell", "wall_s", "path"]
                       + (["method"] if with_method else []) + ["profile"])
            for lab, pr in zip(labels, per_root):
                for (task, robot, method), vals in pr.items():
                    for r, tag, worst, fell, wall, path, prof in vals:
                        w.writerow([lab, task, robot, tag, r, worst, fell, wall, path]
                                   + ([method] if with_method else []) + [prof])
    return 0


if __name__ == "__main__":
    sys.exit(main())
