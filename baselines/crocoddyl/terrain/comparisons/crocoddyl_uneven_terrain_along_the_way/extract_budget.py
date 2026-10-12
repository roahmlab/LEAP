#!/usr/bin/env python3
"""Pick the candidate available within a time budget (after postprocess.py has audited them).

    run_python.sh extract_budget.py <robot dir> --seconds 600 [--clock solver|wall] [--best] [--output cand.npz]
--clock solver uses solver_s (native work only), wall uses available_wall_s (end-to-end incl.
recording). Without --best the LAST candidate within the budget is returned, with --best
the smallest audited violation within it."""
import argparse, json
from pathlib import Path
import h5py, numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('directory', type=Path); ap.add_argument('--seconds', type=float, required=True)
    ap.add_argument('--clock', choices=['solver', 'wall'], default='solver'); ap.add_argument('--best', action='store_true'); ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (args.directory / 'progress.jsonl').read_text().splitlines() if l.strip()]
    key = {'solver': 'solver_s', 'wall': 'available_wall_s'}[args.clock]
    eligible = [r for r in rows if r[key] is not None and r[key] <= args.seconds and r['maximum_violation'] is not None]
    if not eligible: raise SystemExit('No audited candidate within the budget')
    chosen = min(eligible, key=lambda r: r['maximum_violation']) if args.best else eligible[-1]
    if args.output:
        if args.output.exists(): ap.error('Output exists')
        with h5py.File(args.directory / 'iterations.h5', 'r') as f:
            np.savez_compressed(args.output, xs=f['xs'][chosen['stored_index']], us=f['us'][chosen['stored_index']], mesh_json=f.attrs['mesh_json'], metrics_json=json.dumps(chosen))
    print(json.dumps({k: v for k, v in chosen.items() if k not in ('node', 'fine', 'worst_times')}, indent=2))


if __name__ == '__main__': main()
