"""Read the supplied cold archive, preserving polynomial and event ownership."""
from pathlib import Path
import hashlib
import sys
from math import comb

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'comparisons')]
from pyleap.solution import load_solution
from pyleap.interp import bary_weights, bary_row

ROBOTS = ('hopper2d', 'go2', 'h1', 'g1_29dof', 'talos')


class Seed:
    def __init__(self, robot):
        if robot not in ROBOTS:
            raise ValueError(robot)
        self.path = ROOT / f'showcase_v2/unified/impact_cone/initial_guess/archives/seed_fence_{robot}.h5'
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.sol = load_solution(self.path)
        if self.sol.meta.get('generator') != 'leap_unified_initial_guess':
            raise ValueError('Input is not an exported cold initial guess')
        with h5py.File(self.path) as archive:
            if np.any(archive['duals/gamma'][:]):
                raise ValueError('Expected zero seed multipliers')
        self.starts = np.array([m.t0 for m in self.sol.modes])
        self.weights = [bary_weights(m.taus) for m in self.sol.modes]

    def element(self, t, side='right'):
        # Tolerance deals only with roundoff at archived element boundaries.
        idx = np.flatnonzero(abs(self.starts-t) < 1e-11)
        if len(idx):
            return max(0, int(idx[0]) - (side == 'left'))
        return int(np.clip(np.searchsorted(self.starts, t)-1, 0, len(self.starts)-1))

    def sample(self, t, side='right', element=None):
        i = self.element(t, side) if element is None else element
        m = self.sol.modes[i]
        s = float(np.clip((t-m.t0)/m.T, 0., 1.))
        b = bary_row(m.taus, self.weights[i], 2*s-1)
        value = {name: getattr(m, attr) @ b
                 for name, attr in [('q', 'Q'), ('v', 'V'), ('a', 'A'), ('u', 'U')]}
        value['lambda'] = np.zeros(sum(c['dim'] for c in self.sol.contacts))
        ctrl = m.lam_ctrl_physical
        if ctrl is not None:
            n = ctrl.shape[1]-1
            force = ctrl @ np.array([comb(n, j)*s**j*(1-s)**(n-j) for j in range(n+1)])
        else:
            force = m.Lam @ b
        offset = 0
        for c in self.sol.contacts:
            if c['name'] in m.lam_slices:
                value['lambda'][offset:offset+c['dim']] = force[m.lam_slices[c['name']]]
            offset += c['dim']
        return value

    def check_nodes(self):
        error = 0.
        for i, m in enumerate(self.sol.modes):
            for j, tau in enumerate(m.taus):
                v = self.sample(m.t0 + .5*(tau+1)*m.T, element=i)
                for name, attr in [('q', 'Q'), ('v', 'V'), ('a', 'A'), ('u', 'U')]:
                    error = max(error, float(np.max(abs(v[name]-getattr(m, attr)[:, j]))))
                offset=0
                for contact in self.sol.contacts:
                    if contact['name'] in m.lam_slices:
                        expected=m.Lam[m.lam_slices[contact['name']],j]
                        error=max(error,float(np.max(abs(v['lambda'][offset:offset+contact['dim']]-expected))))
                    offset+=contact['dim']
        if error > 1e-8:
            raise AssertionError(f'Seed interpolation error {error}')
        return error


if __name__ == '__main__':
    import json
    for robot in ROBOTS:
        seed = Seed(robot)
        print(json.dumps(dict(robot=robot, sha256=seed.sha256,
                              elements=len(seed.sol.modes), node_import_error=seed.check_nodes())))
