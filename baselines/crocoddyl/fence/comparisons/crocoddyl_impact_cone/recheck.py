#!/usr/bin/env python3
"""Audit an archived initial, final or iteration trajectory; never optimize."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import numpy as np
from physics import inputs, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument('--initial', action='store_true')
    selection.add_argument('--index', type=int)
    ap.add_argument('--dt', type=float, default=.01)
    args = ap.parse_args()
    if not 0 < args.dt <= .01:
        ap.error('Audit dt must be positive and no greater than 0.01 seconds')
    protocol = json.loads((args.directory/'protocol.json').read_text())
    info, *_ = inputs(protocol['robot'], protocol['settings']['dt'])
    if args.index is None:
        with np.load(args.directory/('initial.npz' if args.initial else 'solution.npz')) as z:
            mesh = json.loads(str(z['mesh_json']))
            xs, us = list(z['xs']), list(z['us'])
    else:
        import h5py
        with h5py.File(args.directory/'iterations.h5', 'r', swmr=True) as f:
            if not 0 <= args.index < f.attrs['committed_iterations']:
                raise ValueError('Iteration not committed')
            mesh = json.loads(f.attrs['mesh_json'])
            xs, us = list(f['xs'][args.index]), list(f['us'][args.index])
    result = audit(SimpleNamespace(info=info, mesh=mesh), xs, us, args.dt)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
