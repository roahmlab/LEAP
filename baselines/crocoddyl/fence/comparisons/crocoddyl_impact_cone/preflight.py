#!/usr/bin/env python3
"""Model and archive validation only: an optimizer call is an error here."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import crocoddyl


def forbidden(*args, **kwargs):
    raise AssertionError('Optimization is forbidden during preflight')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot', choices=['hopper2d', 'go2', 'h1', 'g1_29dof', 'talos'], required=True)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    # Neither the Python optimizer constructor nor the native solve wrapper may run.
    crocoddyl.SolverFDDP = forbidden
    import _impact_native
    _impact_native.run = forbidden
    from physics import build, audit, physical, ROOT, Geometry
    from verify import impact_impulse
    from geometry_reference import Physics
    b = build(args.robot)
    r, nv = b.info.r, b.info.r.nv
    rng = np.random.default_rng(17)
    import_errors = {'state': 0., 'control': 0.}
    for k, step in enumerate(b.mesh):
        t, h = step['t'], step['h']
        sample = b.info.seed.sample(t, side='right' if step['kind'] == 'flow' else 'left')
        import_errors['state'] = max(import_errors['state'], float(np.max(abs(b.xs[k]-np.r_[sample['q'], sample['v']]))))
        if step['kind'] == 'flow':
            element = b.info.seed.element(t)
            ss = [b.info.seed.sample(t+c*h, element=element) for c in [0., .5, 1.]]
            control = np.r_[np.concatenate([s['a'] for s in ss]), ss[0]['u'], ss[-1]['u'], ss[0]['lambda'], ss[-1]['lambda']]
        else:
            control = np.zeros_like(b.us[k])
        import_errors['control'] = max(import_errors['control'], float(np.max(abs(b.us[k]-control))))
    assert max(import_errors.values()) == 0.
    # Cross-check against the archived LEAP fields, not a regenerated heuristic.
    assert b.info.seed.check_nodes() < 1e-8
    times = np.array([s['t'] for s in b.mesh] + [b.info.seed.sol.duration])
    assert max(min(abs(times-t)) for t in np.arange(0, times[-1], .01)) < 1e-10
    assert max(min(abs(times-t)) for t in b.info.seed.starts) < 1e-10

    flow = next(i for i, s in enumerate(b.mesh) if s['kind'] == 'flow')
    air = next(i for i, s in enumerate(b.mesh) if s['kind'] == 'flow' and s['mode'] == 1)
    impact = next(i for i, s in enumerate(b.mesh) if s['kind'] == 'impact')
    derivatives = []
    for k in [flow, air, impact]:
        m = b.problem.runningModels[k]
        x = b.xs[k] + 1e-4*rng.normal(size=2*nv)
        u = b.us[k] + 1e-4*rng.normal(size=len(b.us[k]))
        dx, du = rng.normal(size=x.size), rng.normal(size=u.size)
        epsilon = 1e-7 if k == impact else 1e-6
        d, plus, minus = m.createData(), m.createData(), m.createData()
        m.calc(d, x, u); m.calcDiff(d, x, u)
        m.calc(plus, x+epsilon*dx, u+epsilon*du)
        m.calc(minus, x-epsilon*dx, u-epsilon*du)
        fd = (plus.cost-minus.cost)/(2*epsilon)
        analytic = d.Lx@dx+d.Lu@du
        ce = float(abs(fd-analytic)/max(1., abs(fd), abs(analytic)))
        de = float(np.max(abs((plus.xnext-minus.xnext)/(2*epsilon)-d.Fx@dx-d.Fu@du)))
        assert ce < 3e-4, (k, ce, fd, analytic)
        assert de < 2e-4, (k, de)
        assert np.max(abs(d.Lxx-d.Lxx.T)) < 1e-8
        if k != impact:
            q, v, *_ = b.verification.dense(r, x, u, b.mesh[k]['h'], 1.)
            np.testing.assert_allclose(d.xnext, np.r_[q, v], atol=1e-12, rtol=1e-12)
        else:
            np.testing.assert_allclose(d.xnext, Physics(b.info).reset(x), atol=1e-7, rtol=1e-7)
            assert np.max(abs(d.xnext-x)) > 1e-8
        derivatives.append(dict(stage=k, kind=b.mesh[k]['kind'], cost_relative=ce, dynamics_absolute=de))

    # A sliding impact activates the cone even though the reset itself is valid.
    x = b.xs[impact].copy()
    x[:nv] += 1e-3*rng.normal(size=nv)
    x[nv:] = rng.normal(size=nv)
    x[nv] += 3.; x[nv+2] -= .2
    u = np.zeros_like(b.us[impact])
    impulse = impact_impulse(b.info, x)
    expected, offset = [], 0
    for contact in r.contacts:
        f = impulse[offset:offset+3]; offset += contact.dim
        expected.extend([np.sqrt(f[0]**2+f[1]**2+b.info.spec.friceps**2)-b.info.spec.friceps-b.info.spec.mu*f[2], -f[2]])
    d = b.cone.createData(); b.cone.evaluate(x, u, d); b.cone.computeJacobians(x, u, d)
    np.testing.assert_allclose(d.value, expected, atol=1e-7, rtol=1e-7)
    assert max(expected) > 1e-3, 'Sliding test must require an inadmissible impulse'
    dx = rng.normal(size=x.size); eps = 1e-7
    plus, minus = b.cone.createData(), b.cone.createData()
    b.cone.evaluate(x+eps*dx, u, plus); b.cone.evaluate(x-eps*dx, u, minus)
    fd = (plus.value-minus.value)/(2*eps)
    error = float(np.max(abs(fd-d.Jx@dx))/max(1., np.max(abs(fd))))
    assert error < 3e-4, ('impulse derivatives', error)
    changed = [v.copy() for v in b.xs]; changed[impact] = x
    report_changed = audit(b, changed, b.us)
    assert report_changed['node']['impact_cone'] > 1e-3 or report_changed['node']['impact_normal'] > 1e-3
    assert not report_changed['passed']

    # A point capsule near a box corner: rounded clearance.
    point = np.array([.58, .58, .5])
    caps = np.array([[0, 0, .1, *point, *point]])
    boxes = np.array([[0., 0., .5, .5, .5, .5]])
    geom = Geometry(len(u), r.model, caps, boxes, np.empty((0, 3)), 0.)
    gd = geom.createData(); geom.evaluate(x, u, gd)
    np.testing.assert_allclose(gd.value, [.1-np.sqrt(2)*.08], atol=1e-12)
    assert gd.value[0] < 0

    sym = physical.symbolic(b.info)
    last = next(s for s in reversed(b.mesh) if s['kind'] == 'flow' and s['mode'] == 0)
    p = physical.bounds(b.info, last, sym)
    for j in range(sum(c.dim for c in r.contacts)):
        row = sym.row[f'force1_{j}']
        assert p[row] == 0 and p[sym.nr+row] == 0
    row = sym.row['force0_0']
    assert p[row] < 0 and p[sym.nr+row] > 0
    report = audit(b, b.xs, b.us)
    assert not report['passed']
    assert 'impact_cone' in report['node'] and 'impact_normal' in report['node']
    assert 'fence_face_certificate' not in report['node']
    # Exercise native threaded evaluation, but no FDDP construction or iteration.
    b.problem.nthreads = 1
    c1 = b.problem.calc(b.xs, b.us); b.problem.calcDiff(b.xs, b.us)
    b.problem.nthreads = 12
    c12 = b.problem.calc(b.xs, b.us); b.problem.calcDiff(b.xs, b.us)
    np.testing.assert_allclose(c1, c12, rtol=1e-12, atol=1e-10)
    result = dict(robot=args.robot, validation_passed=True, optimization_started=False,
                  input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [b.info.path,b.info.r.yaml_path,b.info.r.urdf_path,b.info.seed.path]},
                  native_sha256=hashlib.sha256(Path(_impact_native.__file__).read_bytes()).hexdigest(),
                  stages=len(b.mesh), seed_import=import_errors, derivatives=derivatives,
                  impact_derivative_relative=error, seed_node_max=report['node_max'],
                  seed_fine_max=report['fine_max'], cost_serial=float(c1), cost_parallel=float(c12))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
