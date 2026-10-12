#!/usr/bin/env python3
"""Markdown table of the ten benchmark solves (Table II) from their logs.

    python scripts/make_results_table.py <results_root> [--tag repro]

<results_root>/<family>/<family>_<robot>_<tag>.log are the logs scripts/run_benchmark.sh
(or scripts/reproduce_paper.sh) wrote. Per instance: the threads, whether the criterion
is met (met=1, node gate and fine grid <= 1e-3), the accepted steps, the node gate, the
fine-grid violation, the elements per refinement level and the solve wall in seconds
(the Table II value). Only the logs are read.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOL = 1e-3
ORDER = [("fence", "hopper2d"), ("fence", "go2"), ("fence", "h1"), ("fence", "g1_29dof"), ("fence", "talos"),
         ("terrain", "hopper2d_foot"), ("terrain", "go2"), ("terrain", "h1"), ("terrain", "g1_29dof"), ("terrain", "talos")]


def summary(log: Path):
    """The fields of the log's `[unified]` summary line, or None."""
    if not log.exists():
        return None
    for line in open(log, errors="replace"):
        if line.startswith("[unified]"):
            return dict(re.findall(r"(\w+)=(\S+)", line))
    return None


def header(log: Path):
    """The first line of a run_benchmark.sh log: threads, start time, load."""
    if not log.exists():
        return {}
    first = open(log, errors="replace").readline()
    return dict(re.findall(r"(\w+)=(\S+)", first))


def fmt(x: str | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}e}"
    except ValueError:
        return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", type=Path)
    ap.add_argument("--tag", default="repro")
    a = ap.parse_args()
    print("| instance | threads | criterion | steps | node gate | fine grid | levels: elements | solve wall s |")
    print("|---|--:|---|--:|---|---|---|--:|")
    missing = 0
    for fam, robot in ORDER:
        log = a.results_root / fam / f"{fam}_{robot}_{a.tag}.log"
        s = summary(log)
        h = header(log)
        if s is None:
            missing += 1
            state = "not run" if not log.exists() else "no [unified] line (running or failed)"
            print(f"| {fam} {robot} | {h.get('threads', '-')} | {state} | - | - | - | - | - |")
            continue
        met = s.get("met") == "1" and float(s["gate"]) <= TOL and float(s["fine"]) <= TOL
        print(f"| {fam} {robot} | {h.get('threads', '-')} | {'met' if met else 'FAIL'} | {s['total']} | {fmt(s['gate'])} | "
              f"{fmt(s['fine'])} | {s['elements']} | {float(s['solvewall']):.1f} |")
    if missing:
        print(f"\n{missing} instance(s) without a result", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
