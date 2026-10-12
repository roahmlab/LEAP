"""Built-in Aligator shooting models; plastic reset updated by orchestration.

No StageFunction, dynamics, cost, or solver subclass is implemented here.
"""
from types import SimpleNamespace
import json
import numpy as np
import aligator as al
import pinocchio as pin
import coal

from seed import Seed, ROOT
from common.spec_fence import load_fence_spec
from common.plan import make_rcms, _monitors


def configure_data(data, settings):
    """Set the native public force-data settings omitted by its constructor."""
    from _native_settings import set_force_settings
    count=0
    if isinstance(data,al.ContactForceData):
        set_force_settings(data,settings)
        count+=1
    if hasattr(data,'sub_cost_data'):
        for entry in data.sub_cost_data: count+=configure_data(entry.data(),settings)
    for name in ('cost_data','residual_data','sub_data'):
        if hasattr(data,name): count+=configure_data(getattr(data,name),settings)
    for name in ('stage_data','constraint_data'):
        if hasattr(data,name):
            for child in getattr(data,name): count+=configure_data(child,settings)
    return count


def impact(info, x):
    r = info.r
    cms = info.cms[1]
    cds = pin.StdVec_RigidConstraintData([c.createData() for c in cms])
    d = r.data()
    pin.initConstraintDynamics(r.model, d, cms, cds)
    pin.impulseDynamics(r.model, d, x[:r.nv], x[r.nv:], cms, cds,
                       0., pin.ProximalSettings(1e-12, 1e-9, 1))
    return np.r_[x[:r.nv], d.dq_after.copy()]


def affine_impact(info, x):
    y = impact(info, x)
    n = len(x)
    jac = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n); dx[j] = 1e-6
        jac[:, j] = (impact(info, x+dx)-impact(info, x-dx))/(2e-6)
    return al.dynamics.LinearDiscreteDynamics(jac, np.zeros((n, 0)), y-jac@x)


def geometry(info, obstacle_scale):
    r = info.r
    geom = pin.GeometryModel()
    d = r.data()
    pin.framesForwardKinematics(r.model, d, info.spec.q_hold0)
    for name, cap in r.capsules.items():
        fid = r.frame_id(cap.frame)
        frame = r.model.frames[fid]
        a = cap.a_local
        b = d.oMf[fid].actInv(d.oMf[r.frame_id(cap.b_frame)].act(cap.b_local))
        # A capsule spanning named frames is native-compatible only if its
        # second endpoint is rigidly attached to the first frame's parent.
        for scale in (.03, -.07):
            check = r.data()
            q = info.spec.q_hold0 + scale*np.sin(np.arange(r.nv)+1)
            pin.framesForwardKinematics(r.model, check, q)
            other = check.oMf[fid].actInv(check.oMf[r.frame_id(cap.b_frame)].act(cap.b_local))
            if np.max(abs(other-b)) > 1e-9:
                raise ValueError(f'Capsule {name} is not rigid; native collision cannot represent it')
        axis = b-a
        rot = pin.Quaternion.FromTwoVectors(np.array([0., 0., 1.]), axis).matrix()
        placement = frame.placement*pin.SE3(rot, .5*(a+b))
        geom.addGeometryObject(pin.GeometryObject(name, frame.parentJoint, fid, placement,
                              coal.Capsule(cap.radius, float(np.linalg.norm(axis)))))
    ids = {g.name: i for i, g in enumerate(geom.geometryObjects)}
    rows = []
    for j, (center, half) in enumerate(info.spec.obstacle_boxes):
        center, half = center.copy(), half.copy()
        center[2] *= obstacle_scale; half[2] *= obstacle_scale
        bid = geom.addGeometryObject(pin.GeometryObject(f'fence{j}', 0, 0,
                                    pin.SE3(np.eye(3), center), coal.Box(*(2*half))))
        for name, cid in ids.items():
            geom.addCollisionPair(pin.CollisionPair(cid, bid))
            rows.append((f'fence_{name}_{j}', info.spec.obstacle_margin))
    for a, b, safe in info.spec.selfcol_pairs:
        geom.addCollisionPair(pin.CollisionPair(ids[a], ids[b]))
        rows.append((f'self_{a}_{b}', safe-r.capsules[a].radius-r.capsules[b].radius))
    return geom, rows


def inputs(robot, dt):
    seed = Seed(robot)
    path = ROOT / f'showcase_v2/unified/impact_cone/problems/{robot}_jump_fence.yaml'
    spec = load_fence_spec(path, allow_impact_cone=True)
    if not spec.impact_cone:
        raise ValueError('The impact-cone benchmark requires problem.impactcone=true')
    r = spec.robot
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


def build(info, mesh, xs, *, obstacle_scale=1., restoration=False, barrier=1e-3):
    if info.spec.impact_cone:
        raise ValueError('The legacy RK2 model has no impulse cone; use solve_public.py')
    r, s = info.r, info.spec
    nv, nx = r.nv, 2*r.nv
    space = al.manifolds.MultibodyPhaseSpace(r.model)
    geom, collisions = geometry(info, obstacle_scale)
    odes = [al.dynamics.MultibodyConstraintFwdDynamics(space, info.S, cm, info.prox) for cm in info.cms]
    odes.insert(1, al.dynamics.MultibodyFreeFwdDynamics(space, info.S))
    forces = [[al.ContactForceResidual(nx, r.model, info.S, cm, info.prox, np.zeros(c.dim), c.name)
               for c in r.contacts] for cm in info.cms]
    stages, checks = [], []
    for k, step in enumerate(mesh):
        mode, kind = step['mode'], step['kind']
        nu = 0 if kind == 'impact' else r.nu
        if kind == 'flow':
            dyn = al.dynamics.IntegratorRK2(odes[mode], step['h'])
        elif kind == 'impact':
            dyn = affine_impact(info, xs[k])
        else:
            dyn = al.dynamics.LinearDiscreteDynamics(np.eye(nx), np.zeros((nx, nu)), np.zeros(nx))
        cost = al.CostStack(space, nu)
        rows = []
        def add(fn, lo, hi, name, scale=1.):
            lo = np.broadcast_to(lo, (fn.nr,)).copy()
            hi = np.broadcast_to(hi, (fn.nr,)).copy()
            rows.append((fn, lo, hi, name, scale))
            if not restoration:
                return
            eq = np.isfinite(lo) & (lo == hi)
            for idx, side in [(np.flatnonzero(eq), 0),
                              (np.flatnonzero(np.isfinite(lo) & ~eq), 1),
                              (np.flatnonzero(np.isfinite(hi) & ~eq), -1)]:
                if not len(idx): continue
                sub = al.LinearFunctionComposition(fn, np.eye(fn.nr)[idx])
                sign = 1 if side >= 0 else -1
                target = lo[idx] if side >= 0 else hi[idx]
                f = al.LinearFunctionComposition(sub, sign*scale*np.eye(len(idx)), -sign*scale*target)
                if side == 0:
                    c = al.QuadraticResidualCost(space, f, np.eye(len(idx)))
                else:
                    c = al.RelaxedLogBarrierCost(space, f, barrier**2, barrier)
                cost.addCost(c, 1.)
        state = al.StateErrorResidual(space, nu, np.zeros(nx))
        act = np.array(r.act_idx)
        mid, half = .5*(r.q_lower+r.q_upper), .5*(r.q_upper-r.q_lower)
        add(state[act.tolist()], mid-s.qbox*half, mid+s.qbox*half, 'box_q')
        add(state[(act+nv).tolist()], -s.vlim*r.velocity_limit, s.vlim*r.velocity_limit, 'box_v')
        add(state[2:3], 0., np.inf, 'base_z')
        idx = [i for i in [3, 4, 5] if i not in s.planarlock_idx]
        if idx and s.attmax > 0: add(state[idx], -s.attmax, s.attmax, 'box_att')
        if s.planarlock_idx: add(state[s.planarlock_idx], 0., 0., 'planar')
        if nu: add(al.ControlErrorResidual(nx, np.zeros(nu)), -.995*s.ulim*r.effort_limit,
                   .995*s.ulim*r.effort_limit, 'box_u', .01)
        for j, (name, margin) in enumerate(collisions):
            add(al.FrameCollisionResidual(nx, nu, r.model, geom, j), margin, np.inf, name)
        contact_names = [c.name for c in r.contacts]
        for fid, name in _monitors(r):
            if mode != 1 and name in contact_names: continue
            add(al.FrameTranslationResidual(nx, nu, r.model, np.zeros(3), fid)[2:3],
                max(0., s.footclear) if mode == 1 else 0., np.inf, 'clear')
        target = s.feet0 if mode == 0 else s.feet2
        for j, c in enumerate(r.contacts):
            if mode != 1 or kind == 'impact':
                if c.dim == 6:
                    fn = al.FramePlacementResidual(nx, nu, r.model, pin.SE3(np.eye(3), target[c.name]), c.frame_id)
                else:
                    fn = al.FrameTranslationResidual(nx, nu, r.model, target[c.name], c.frame_id)
                add(fn, 0., 0., 'foot_placement')
            if mode != 1:
                vel = al.FrameVelocityResidual(nx, nu, r.model, pin.Motion.Zero(), c.frame_id, pin.LOCAL)
                add(vel[:c.dim], 0., 0., 'foot_velocity')
                force = forces[0 if mode == 0 else 1][j]
                if kind == 'liftoff':
                    add(force, 0., 0., 'liftoff_force', .01)
                else:
                    floor = s.lammin
                    if mode == 0:
                        floor *= np.clip((s.t_stance-step['t'])/(s.t_stance-info.taper_start), 0., 1.)
                    add(force[2:3], floor, np.inf, 'force_floor', .01)
                    # Inscribed polygon: a conservative approximation to the
                    # circular cone, represented only by native linear maps.
                    # Native MultibodyFrictionConeResidual assumes 3D strides
                    # and has a 0/0 derivative at zero tangential force.
                    theta=2*np.pi*np.arange(64)/64
                    facets=np.zeros((64,c.dim))
                    facets[:,0],facets[:,1]=np.cos(theta),np.sin(theta)
                    facets[:,2]=-s.mu*np.cos(np.pi/64)
                    add(al.LinearFunctionComposition(force,facets),-np.inf,0.,'cone_polygon',.01)
                    if c.dim == 6:
                        A = np.zeros((6, 6))
                        for a, (moment, bound) in enumerate([(4, s.copx), (3, s.copy), (5, s.gammaz)]):
                            A[2*a, moment] = 1.; A[2*a+1, moment] = -1.
                            A[2*a:2*a+2, 2] = -bound
                        add(al.LinearFunctionComposition(force, A), -np.inf, 0., 'cop_yaw', .01)
            elif kind == 'impact' and s.landvz > 0:
                vel = al.FrameVelocityResidual(nx, nu, r.model, pin.Motion.Zero(), c.frame_id, pin.WORLD)
                p = target[c.name]
                transport = np.array([[0., 0., 1., p[1], -p[0], 0.]])
                add(al.LinearFunctionComposition(vel, transport), -s.landvz, np.inf, 'landing_speed')
        if kind == 'liftoff': add(state[nv+2:nv+3], s.vz_liftoff, np.inf, 'liftoff_speed')
        stage = al.StageModel(cost, dyn)
        if not restoration:
            for fn, lo, hi, name, scale in rows:
                eq = np.isfinite(lo) & (lo == hi)
                for idx, equality in [(np.flatnonzero(eq), True), (np.flatnonzero(~eq), False)]:
                    if not len(idx): continue
                    f = al.LinearFunctionComposition(fn, scale*np.eye(fn.nr)[idx],
                                                     -scale*lo[idx] if equality else np.zeros(len(idx)))
                    cs = al.constraints.EqualityConstraintSet() if equality else al.constraints.BoxConstraint(scale*lo[idx], scale*hi[idx])
                    stage.addConstraint(f, cs)
        stages.append(stage); checks.append(rows)
    goal = np.r_[s.q_hold2, np.zeros(nv)]
    termfn = al.StateErrorResidual(space, 0, goal)
    terminal = al.QuadraticResidualCost(space, termfn, np.eye(nx)) if restoration else al.ConstantCost(space, 0, 0.)
    problem = al.TrajOptProblem(np.r_[s.q_hold0, np.zeros(nv)], stages, terminal)
    if not restoration: problem.addTerminalConstraint(termfn, al.constraints.EqualityConstraintSet())
    return SimpleNamespace(problem=problem, checks=checks, info=info, mesh=mesh,
                           odes=odes, forces=forces, space=space, goal=goal,
                           restoration=restoration, obstacle_scale=obstacle_scale)
