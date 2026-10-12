"""Impact-cone inputs and event-aware mesh."""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from seed import Seed, ROOT
from spec import load_fence_spec
from common.plan import make_rcms

def inputs(robot, dt):
    seed = Seed(robot)
    path = ROOT / f'showcase_v2/unified/impact_cone/problems/{robot}_jump_fence.yaml'
    spec = load_fence_spec(path)
    r = spec.robot
    if seed.sol.nq != r.nq or seed.sol.nv != r.nv:
        raise ValueError('Seed and problem robot dimensions differ')
    if list(seed.sol.joint_names) != list(r.model.names)[1:]:
        raise ValueError('Seed and problem joint ordering differ')
    if [(c['name'], c['dim']) for c in seed.sol.contacts] != [(c.name, c.dim) for c in r.contacts]:
        raise ValueError('Seed and problem contact ordering differ')
    np.testing.assert_allclose(seed.sol.duration, spec.t_stance+spec.t_air+spec.t_land, atol=1e-12, rtol=0)
    first, last = seed.sample(0.), seed.sample(seed.sol.duration)
    np.testing.assert_allclose(np.r_[first['q'], first['v']], np.r_[spec.q_hold0, np.zeros(r.nv)], atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.r_[last['q'], last['v']], np.r_[spec.q_hold2, np.zeros(r.nv)], atol=1e-10, rtol=0)
    S = np.zeros((r.nv, r.nu)); S[r.act_idx, np.arange(r.nu)] = 1.
    cms = [pin.StdVec_RigidConstraintModel(make_rcms(r, [c.name for c in r.contacts], target, kp=0.))
           for target in [spec.feet0, spec.feet2]]
    info = SimpleNamespace(seed=seed, spec=spec, r=r, S=S, cms=cms, path=path)
    info.prox = pin.ProximalSettings(1e-12, 1e-9, 20)
    info.taper_start = seed.sol.modes[7].t0
    bounds = [0., spec.t_stance, spec.t_stance+spec.t_air, seed.sol.duration]
    mesh, xs, us, imported = [], [], [], []
    for mode in range(3):
        a, b = bounds[mode:mode+2]
        edges = np.unique(np.round(np.r_[a, b, np.arange(a, b, dt),
                    np.arange(0., seed.sol.duration, .01), seed.starts], 13))
        edges = edges[(edges >= a-1e-11) & (edges <= b+1e-11)]
        for t, end in zip(edges[:-1], edges[1:]):
            value = seed.sample(t)
            mesh.append(dict(kind='flow', mode=mode, t=float(t), h=float(end-t)))
            imported.append(value); xs.append(np.r_[value['q'], value['v']]); us.append(value['u'])
        if mode < 2:
            value = seed.sample(b, side='left')
            mesh.append(dict(kind='liftoff' if mode == 0 else 'impact', mode=mode, t=b, h=0.))
            imported.append(value); xs.append(np.r_[value['q'], value['v']])
            us.append(value['u'] if mode == 0 else np.zeros(0))
    value = seed.sample(bounds[-1]); imported.append(value)
    xs.append(np.r_[value['q'], value['v']])
    return info, mesh, xs, us, imported
