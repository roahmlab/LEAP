"""Terrain benchmark inputs: the benchmark yaml, the archived LEAP cold seed and the
event-aware shooting mesh, with both sides of every mode event preserved.

The spec is loaded with `windows=True`, so a swing with approachfrac / liftfrac is the C++ mode chain
[lift,] flight [, approach] (terrain_walk.cpp:873-975) with hover rows in the window
modes, the touchdown rows on the mode that lands, and -- with zeroland + landcont -- a
continuity junction instead of the plastic impact at the touchdown. Nothing here
depends on an optimizer.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys
from math import comb

import h5py
import numpy as np
import pinocchio as pin
import coal

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'comparisons')]
from pyleap.solution import load_solution
from pyleap.interp import bary_weights, bary_row
from common.spec_terrain import load_terrain_spec
from common.plan import make_rcms

# The terrain benchmark's robots (h1 on the problem of new_src/problems/h1_terrain_walk.yaml).
ROBOTS = ('hopper2d_foot', 'go2', 'g1_29dof', 'talos', 'h1')
# Other robots whose benchmark yaml and cold seed exist in the tree (not part of this study).
EXTRA_ROBOTS = ('hopper2d', 'apollo')
ALL_ROBOTS = ROBOTS + EXTRA_ROBOTS
_PROBLEM_YAML = {
    'hopper2d_foot': 'showcase_v2/unified/flatfoot_20260913/problems/hopper2d_foot_terrain_walk.yaml',
    'h1': 'showcase_v2/unified/trackable_20260913/problems/h1_ap15_ank07_att02_torso05.yaml',
    # the g1 problem of new_src/problems/g1_29dof_terrain_walk.yaml
    'g1_29dof': 'comparisons_leap_benchmark/terrain/problems/g1_29dof_terrain_walk.yaml',
}
_SEED_ARCHIVE = {
    'h1': 'seed_terrain_h1_ap15_ank07_att02_torso05.h5',   # exported from the adopted yaml (the LEAP budget run's seed)
    # ROOT-relative path
    'g1_29dof': 'comparisons_leap_benchmark/terrain/initial_guess/archives/seed_terrain_g1_29dof.h5',
}


def problem_yaml(robot):
    """The benchmark yaml the LEAP unified solve of this robot used."""
    return ROOT / _PROBLEM_YAML.get(robot, f'showcase_v2/terrain/problems/{robot}_terrain_walk.yaml')


def seed_path(robot):
    """The exported LEAP cold initial guess (docs/solution_format.md archive)."""
    name = _SEED_ARCHIVE.get(robot, f'seed_terrain_{robot}.h5')
    return ROOT / name if '/' in name else ROOT / 'showcase_v2/unified/initial_guess/archives' / name


class Seed:
    """Read the supplied cold archive, preserving polynomial and event ownership."""

    def __init__(self, robot):
        if robot not in ALL_ROBOTS:
            raise ValueError(robot)
        self.path = seed_path(robot)
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
        idx = np.flatnonzero(abs(self.starts - t) < 1e-11)
        if len(idx):
            return max(0, int(idx[0]) - (side == 'left'))
        return int(np.clip(np.searchsorted(self.starts, t) - 1, 0, len(self.starts) - 1))

    def sample(self, t, side='right', element=None):
        i = self.element(t, side) if element is None else element
        m = self.sol.modes[i]
        s = float(np.clip((t - m.t0) / m.T, 0., 1.))
        b = bary_row(m.taus, self.weights[i], 2 * s - 1)
        value = {name: getattr(m, attr) @ b
                 for name, attr in [('q', 'Q'), ('v', 'V'), ('a', 'A'), ('u', 'U')]}
        value['lambda'] = np.zeros(sum(c['dim'] for c in self.sol.contacts))
        ctrl = m.lam_ctrl_physical
        if ctrl is not None:
            n = ctrl.shape[1] - 1
            force = ctrl @ np.array([comb(n, j) * s ** j * (1 - s) ** (n - j) for j in range(n + 1)])
        else:
            force = m.Lam @ b
        offset = 0
        for c in self.sol.contacts:
            if c['name'] in m.lam_slices:
                value['lambda'][offset:offset + c['dim']] = force[m.lam_slices[c['name']]]
            offset += c['dim']
        return value

    def check_nodes(self):
        error = 0.
        for i, m in enumerate(self.sol.modes):
            for j, tau in enumerate(m.taus):
                v = self.sample(m.t0 + .5 * (tau + 1) * m.T, element=i)
                for name, attr in [('q', 'Q'), ('v', 'V'), ('a', 'A'), ('u', 'U')]:
                    error = max(error, float(np.max(abs(v[name] - getattr(m, attr)[:, j]))))
                offset = 0
                for contact in self.sol.contacts:
                    if contact['name'] in m.lam_slices:
                        expected = m.Lam[m.lam_slices[contact['name']], j]
                        error = max(error, float(np.max(abs(v['lambda'][offset:offset + contact['dim']] - expected))))
                    offset += contact['dim']
        if error > 1e-8:
            raise AssertionError(f'Seed interpolation error {error}')
        return error


def inputs(robot, dt):
    """Spec, seed, mesh and the imported shooting states of one robot.

    The mesh has one zero-duration 'initial' allocation stage (free initial q), flow
    stages whose edges are the union of the requested dt grid, the global 0.01 s
    ticks, every archived seed element boundary and the mid-swing clearance times,
    and one zero-duration event stage between modes: 'junction' (continuity) after
    a support mode, after a lift window and after a flight that an approach window
    follows, 'impact' (all-contact plastic reset) after the mode that lands the swing
    foot -- unless the yaml states the zero-velocity touchdown as a continuity junction
    (zeroland + landcont), in which case that event is a 'junction' too and the
    touchdown rows still sit on it (see event_after)."""
    seed = Seed(robot)
    path = problem_yaml(robot)
    spec = load_terrain_spec(path, windows=True); r = spec.robot
    if r.nq != r.nv: raise ValueError('Explicit-base robots only (nq == nv)')
    if seed.sol.nq != r.nq or seed.sol.nv != r.nv or not np.array_equal(seed.sol.actuated_idx, r.act_idx):
        raise ValueError('Archived seed model dimensions/actuation differ')
    if seed.sol.joint_names != list(r.model.names)[1:]: raise ValueError('Archived seed joint order differs')
    if [(c['name'], c['dim']) for c in seed.sol.contacts] != [(c.name, c.dim) for c in r.contacts]:
        raise ValueError('Archived seed contact order differs')
    S = np.zeros((r.nv, r.nu)); S[r.act_idx, np.arange(r.nu)] = 1.
    starts = np.r_[0., np.cumsum([m.T for m in spec.modes])]
    assert abs(starts[-1] - seed.sol.duration) < 1e-9
    cms = [pin.StdVec_RigidConstraintModel(make_rcms(r, [c.name for c in r.contacts], m.targets, kp=0.))
           if m.kind == 'support' else None for m in spec.modes]
    info = SimpleNamespace(seed=seed, spec=spec, r=r, S=S, cms=cms, path=path, starts=starts, robot=robot)
    info.elements = []; info.clear_times = {}
    for mode, md in enumerate(spec.modes):
        elements = [i for i, m in enumerate(seed.sol.modes) if starts[mode] - 1e-9 <= m.t0 < starts[mode + 1] - 1e-9]
        info.elements.append(elements)
        info.clear_times[mode] = [seed.sol.modes[i].t0 + f * seed.sol.modes[i].T for i in elements for f in [.25, .5, .75]] if md.kind == 'swing' and md.clearz >= 0 else []
        for i in elements:
            assert set(seed.sol.modes[i].lam_slices) == set(md.active), (robot, mode, i)
    mesh = [dict(kind='initial', mode=0, t=0., h=0.)]; xs = [np.zeros(2 * r.nv)]; imported = [seed.sample(0.)]
    for mode, md in enumerate(spec.modes):
        a, b = starts[mode:mode + 2]
        edges = np.unique(np.round(np.r_[a, b, np.arange(0., starts[-1], dt), seed.starts,
                                         np.arange(0., starts[-1], .01), info.clear_times[mode]], 13))
        edges = edges[(edges >= a - 1e-11) & (edges <= b + 1e-11)]
        for t, end in zip(edges[:-1], edges[1:]):
            value = seed.sample(t)
            mesh.append(dict(kind='flow', mode=mode, t=float(t), h=float(end - t)))
            imported.append(value); xs.append(np.r_[value['q'], value['v']])
        if mode < len(spec.modes) - 1:
            value = seed.sample(b, side='left')
            mesh.append(dict(kind=event_after(spec, mode), mode=mode, t=float(b), h=0.))
            imported.append(value); xs.append(np.r_[value['q'], value['v']])
    value = seed.sample(starts[-1]); imported.append(value); xs.append(np.r_[value['q'], value['v']])
    return info, mesh, xs, [], imported


def event_after(spec, mode):
    """Kind of the zero-duration event stage that ends mode `mode` (terrain_walk.cpp:940-975):
    a continuity junction after a support mode, a lift window or a flight followed by
    an approach window; after the mode that lands the swing foot the all-contact plastic
    impact, or a continuity junction when the yaml says zeroland + landcont."""
    md = spec.modes[mode]
    if md.kind != 'swing' or not md.landing: return 'junction'
    return 'junction' if (spec.zeroland and spec.landcont) else 'impact'


def lands_here(spec, mode, kind):
    """Whether the event stage of kind `kind` that ends `mode` carries the touchdown rows."""
    md = spec.modes[mode]
    return kind in ('impact', 'junction') and md.kind == 'swing' and bool(md.landing) and md.swing_foot is not None


def ramp(info, mode, contact, t):
    """Force-floor factor: the floor ramps to zero inside the archived element next to a
    pinned event (a landing foot's first element under landRamp, a lifting foot's last)."""
    md = info.spec.modes[mode]; els = info.elements[mode]; a, b = info.starts[mode:mode + 2]
    first, last = info.seed.sol.modes[els[0]], info.seed.sol.modes[els[-1]]
    factor = 1.
    if contact in md.land_feet: factor = min(factor, max(0., (t - a) / first.T))
    if contact in md.lift_feet: factor = min(factor, max(0., (b - t) / last.T))
    return factor


def skip_floor(info, mode, contact, t):
    """Where the judge does not grade the raw normal-force floor (the ramp elements)."""
    md = info.spec.modes[mode]; els = info.elements[mode]
    first, last = info.seed.sol.modes[els[0]], info.seed.sol.modes[els[-1]]
    return (contact in md.land_feet and t <= first.t0 + first.T + 1e-11) or (contact in md.lift_feet and t >= last.t0 - 1e-11)


def geometry(info):
    """Self-collision pairs as a pinocchio/coal capsule geometry model (the terrain
    keep-out rows are generated symbolically, see tw_residuals.symbolic)."""
    r = info.r; geom = pin.GeometryModel(); d = r.data(); q = info.seed.sample(0.)['q']
    pin.framesForwardKinematics(r.model, d, q)
    for name, cap in r.capsules.items():
        fid = r.frame_id(cap.frame); frame = r.model.frames[fid]; a = cap.a_local
        b = d.oMf[fid].actInv(d.oMf[r.frame_id(cap.b_frame)].act(cap.b_local))
        check = r.data(); pin.framesForwardKinematics(r.model, check, q + .03 * np.sin(np.arange(r.nv) + 1))
        assert np.max(abs(check.oMf[fid].actInv(check.oMf[r.frame_id(cap.b_frame)].act(cap.b_local)) - b)) < 1e-9
        axis = b - a; rot = pin.Quaternion.FromTwoVectors(np.array([0., 0., 1.]), axis).matrix()
        geom.addGeometryObject(pin.GeometryObject(name, frame.parentJoint, fid, frame.placement * pin.SE3(rot, .5 * (a + b)), coal.Capsule(cap.radius, float(np.linalg.norm(axis)))))
    ids = {g.name: i for i, g in enumerate(geom.geometryObjects)}; rows = []
    for a, b, safe in info.spec.selfcol_pairs:
        geom.addCollisionPair(pin.CollisionPair(ids[a], ids[b])); rows.append((f'self_{a}_{b}', safe - r.capsules[a].radius - r.capsules[b].radius))
    return geom, rows


def reset(info, x, mode):
    """LEAP's plastic impact: pinocchio impulseDynamics on the next support mode's
    contacts, one damped proximal iteration, world-aligned contact frames."""
    r = info.r; d = r.data(); cms = info.cms[mode + 1]
    cds = pin.StdVec_RigidConstraintData([cm.createData() for cm in cms])
    pin.initConstraintDynamics(r.model, d, cms, cds)
    pin.impulseDynamics(r.model, d, x[:r.nv], x[r.nv:], cms, cds, 0., pin.ProximalSettings(1e-12, 1e-9, 1))
    return np.r_[x[:r.nv], d.dq_after.copy()]


if __name__ == '__main__':
    import json
    for robot in ROBOTS:
        info, mesh, xs, _, _ = inputs(robot, .01)
        kinds = {}
        for step in mesh: kinds[step['kind']] = kinds.get(step['kind'], 0) + 1
        print(json.dumps(dict(robot=robot, yaml=str(info.path.relative_to(ROOT)), seed_sha256=info.seed.sha256[:16],
                              elements=len(info.seed.sol.modes), modes=len(info.spec.modes), stages=kinds,
                              node_import_error=info.seed.check_nodes(), zeroland=info.spec.zeroland,
                              landramp=info.spec.landramp, zerolandxy=info.spec.zerolandxy, landcont=info.spec.landcont,
                              approachfrac=info.spec.approachfrac, liftfrac=info.spec.liftfrac,
                              windows=[m.window for m in info.spec.modes if m.window])))
