#!/usr/bin/env python
"""Verdicts + REPORT.md for a tracked folder (one `<robot>/` per case, each with
`rollout.npz`, `metrics.json`, `trajectory.npz`, `evaluation.json`; repeats in
`<robot>/rep<k>/`).

    $PY report.py --task fence --root tracked/fence                 # -> tracked/fence/{REPORT.md, verdicts.json, plan_diagnostics.json, plots/}
    $PY report.py --task terrain --root tracked/terrain --no-plots --robots go2,h1
    $PY report.py --task fence --case tracked/fence/go2             # verdict.json of ONE case, printed

The logic is `leaptrack.report`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from leaptrack.controllers import TASKS  # noqa: E402
from leaptrack.report import verdict, write_report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--root", default=None, help="results root with one <robot>/ per case")
    ap.add_argument("--case", default=None, help="a single case directory: print its verdict only")
    ap.add_argument("--robots", default=None, help="comma list (default: every robot with results under --root)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-diagnostics", action="store_true", help="skip the plan diagnostics (section 2)")
    ap.add_argument("--title", default=None, help="report title suffix")
    args = ap.parse_args(argv)
    if args.case:
        v = verdict(Path(args.case))
        print(json.dumps({k: v[k] for k in ("task", "robot", "trackable", "bounded", "final_state", "strict_bound",
                                             "reason", "final", "time_shift", "wall_s")}, indent=1))
        return 0
    if not args.root:
        ap.error("--root or --case")
    robots = [r for r in args.robots.split(",") if r] if args.robots else None
    write_report(args.task, Path(args.root), robots, plots=not args.no_plots, title=args.title,
                 diagnostics=not args.no_diagnostics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
