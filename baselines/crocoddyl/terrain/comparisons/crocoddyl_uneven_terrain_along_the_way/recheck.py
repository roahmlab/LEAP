#!/usr/bin/env python3
"""Independent re-audit of an archived candidate; never optimizes.

    run_python.sh recheck.py <robot dir> [--initial | --stored N | --solution] [--dt 0.01]
Exit 0 = passes the 1e-3 criterion, 2 = infeasible."""
from pathlib import Path
import argparse, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    sel = ap.add_mutually_exclusive_group(); sel.add_argument('--initial', action='store_true'); sel.add_argument('--stored', type=int); sel.add_argument('--solution', action='store_true')
    ap.add_argument('--dt', type=float, default=.01)
    args = ap.parse_args()
    import tw_residuals, tw_verify
    protocol = json.loads((args.directory / 'protocol.json').read_text())
    info, mesh, *_ = tw_residuals.cold(protocol['robot'], protocol['settings']['dt'], protocol['hold'])
    if args.stored is not None:
        import h5py
        with h5py.File(args.directory / 'iterations.h5', 'r', swmr=True) as f:
            if not 0 <= args.stored < f.attrs['committed_iterations']: raise ValueError('Candidate not committed')
            xs, us = list(f['xs'][args.stored]), list(f['us'][args.stored])
    else:
        with np.load(args.directory / ('initial.npz' if args.initial else 'solution.npz')) as z: xs, us = list(z['xs']), list(z['us'])
    result = tw_verify.audit(info, mesh, xs, us, args.dt)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 2


if __name__ == '__main__': raise SystemExit(main())
