#!/usr/bin/env python3
"""Sequential five-robot suite with one consistent physical-core allocation."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROBOTS = ('hopper2d', 'go2', 'h1', 'g1_29dof', 'talos')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cpus', required=True)
    ap.add_argument('--wall-seconds', type=float, default=86400.)
    ap.add_argument('--solver-seconds', type=float, default=14400.)
    ap.add_argument('--robots', choices=ROBOTS, nargs='+', default=ROBOTS)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    if len(set(args.robots)) != len(args.robots):
        ap.error('Each robot may be listed only once')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for robot in args.robots:
        command = [sys.executable, str(HERE/'solve.py'), '--robot', robot,
                   '--output', str(args.output/robot), '--cpus', args.cpus,
                   '--wall-seconds', str(args.wall_seconds), '--solver-seconds', str(args.solver_seconds)]
        if args.prepare_only:
            command.append('--prepare-only')
        with (args.output/f'{robot}.log').open('w') as log:
            returned = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(robot=robot, command=command, exit_code=returned.returncode,
                            prepare_only=args.prepare_only))
        (args.output/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
        if returned.returncode not in (0, 2):
            return returned.returncode
    return 0 if all(r['exit_code'] == 0 for r in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
