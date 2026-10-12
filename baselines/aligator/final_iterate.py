#!/usr/bin/env python3
"""Print the candidate id `phase:call:sample` of the last iterate recorded within the solve
budget of an Aligator run folder: the last row of <run>/iterations.csv with within_budget = True.

    python final_iterate.py <run folder>        -> candidate id
    python final_iterate.py <run folder> -v     -> id, time_s, node_max, fine_max
"""
import csv
import sys
from pathlib import Path

if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != '-v'):
    sys.exit(__doc__)
with (Path(sys.argv[1]) / 'iterations.csv').open(newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['within_budget'].lower() == 'true']
if not rows:
    sys.exit(f'{sys.argv[1]}: no iterate within budget')
r = rows[-1]
cid = f"{r['phase']}:{r['call']}:{r['sample']}"
if len(sys.argv) == 3:
    print(f"candidate {cid}  time_s {r['time_s']}  node_max {r['node_max']}  fine_max {r['fine_max']}")
else:
    print(cid)
