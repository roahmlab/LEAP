"""Independent numeric audit of the physical fence, without Aligator models.

The dense state curve is the native RK2 partial-step extension. Acceleration
and force are recovered from physical forward dynamics. Its two kinematic
defects and force-unit EoM defect are all checked, as separate physical rows.
"""
import numpy as np
import pinocchio as pin
from common.geometry import seg_seg_closest
from common.plan import _monitors


def axis_box_distance(A, B, center, half):
    a, u = A-center, B-A
    cuts = [0., 1.]
    for k in range(3):
        if abs(u[k]) > 1e-15:
            for sign in (1., -1.):
                s = (sign*half[k]-a[k])/u[k]
                if 0 < s < 1: cuts.append(s)
    cuts = sorted(set(cuts))
    candidates = list(cuts)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        p = a+.5*(lo+hi)*u
        active = abs(p) > half
        den = u[active] @ u[active]
        s = -(u[active] @ (a-np.sign(p)*half)[active])/den if den > 0 else .5*(lo+hi)
        candidates.append(np.clip(s, lo, hi))
    return min(np.linalg.norm(np.maximum(abs(a+s*u)-half, 0.)) for s in candidates)


class Physics:
    def __init__(self, info):
        self.info, self.r = info, info.r
        self.d = self.r.data()
        self.fd = [self.r.data() for _ in range(3)]
        self.cd = [pin.StdVec_RigidConstraintData([c.createData() for c in cm]) for cm in info.cms]
        for mode, contact in [(0, 0), (2, 1)]:
            pin.initConstraintDynamics(self.r.model, self.fd[mode], info.cms[contact], self.cd[contact])

    def ode(self, x, u, mode):
        nv, d = self.r.nv, self.fd[mode]
        if mode == 1:
            a = pin.aba(self.r.model, d, x[:nv], x[nv:], self.info.S@u).copy()
            force = np.zeros(sum(c.dim for c in self.r.contacts))
        else:
            c = 0 if mode == 0 else 1
            a = pin.constraintDynamics(self.r.model, d, x[:nv], x[nv:], self.info.S@u,
                self.info.cms[c], self.cd[c], self.info.prox).copy()
            force = d.lambda_c.copy()
        return np.r_[x[nv:], a], force

    def rk2(self, x, u, h, mode):
        f, _ = self.ode(x, u, mode)
        g, _ = self.ode(x+h*f/2, u, mode)
        return x+h*g

    def dense(self, x, u, h, elapsed, mode):
        state = self.rk2(x,u,elapsed,mode)
        rhs, force = self.ode(state,u,mode)
        if abs(elapsed)<1e-14:
            derivative = self.ode(x,u,mode)[0]
        else:
            eps=min(1e-6,h/10)
            derivative=(self.rk2(x,u,elapsed-2*eps,mode)-8*self.rk2(x,u,elapsed-eps,mode)
                       +8*self.rk2(x,u,elapsed+eps,mode)-self.rk2(x,u,elapsed+2*eps,mode))/(12*eps)
        return state, rhs[self.r.nv:], force, derivative-rhs

    def reset(self, x):
        nv, r = self.r.nv, self.r
        d = r.data(); cms = self.info.cms[1]
        cds = pin.StdVec_RigidConstraintData([cm.createData() for cm in cms])
        pin.initConstraintDynamics(r.model, d, cms, cds)
        pin.impulseDynamics(r.model, d, x[:nv], x[nv:], cms, cds,
                           0., pin.ProximalSettings(1e-12, 1e-9, 1))
        return np.r_[x[:nv], d.dq_after.copy()]


def hermite(x0, x1, h, s):
    n = len(x0)//2
    q0, v0, q1, v1 = x0[:n], x0[n:], x1[:n], x1[n:]
    c0, c1 = q0, h*v0
    c2 = 3*(q1-q0)-h*(2*v0+v1)
    c3 = 2*(q0-q1)+h*(v0+v1)
    return c0+s*(c1+s*(c2+s*c3)), (c1+s*(2*c2+3*s*c3))/h, (2*c2+6*s*c3)/h**2


def audit(info, mesh, xs, us, fine_dt=.01):
    if not all(np.isfinite(x).all() for x in xs+us):
        return dict(passed=False, finite=False)
    r, spec, phys = info.r, info.spec, Physics(info)
    nv = r.nv
    node, fine, shooting = {}, {}, {}
    worst = {}
    def bump(table, name, value, t):
        v = max(0., float(np.max(np.atleast_1d(value))))
        if not np.isfinite(v): raise ValueError(f'Nonfinite {name}')
        if v > table.get(name, -1):
            table[name] = v
            group='node' if table is node else 'fine' if table is fine else 'shooting'
            worst[f'{group}:{name}'] = float(t)
    def state_rows(table, q, v, u, force, mode, t, kind='flow', a=None):
        d = phys.d
        pin.computeJointJacobians(r.model, d, q); pin.updateFramePlacements(r.model, d)
        mid, half = .5*(r.q_lower+r.q_upper), .5*(r.q_upper-r.q_lower)
        qa, va = q[r.act_idx], v[r.act_idx]
        bump(table, 'box_q', np.r_[mid-spec.qbox*half-qa, qa-mid-spec.qbox*half], t)
        bump(table, 'box_v', abs(va)-spec.vlim*r.velocity_limit, t)
        if len(u): bump(table, 'box_u', abs(u)-.995*spec.ulim*r.effort_limit, t)
        bump(table, 'base_z', -q[2], t)
        idx = [i for i in [3, 4, 5] if i not in spec.planarlock_idx]
        if idx and spec.attmax > 0: bump(table, 'box_att', abs(q[idx])-spec.attmax, t)
        if spec.planarlock_idx: bump(table, 'planar', abs(q[spec.planarlock_idx]), t)
        caps = {}
        for name, c in r.capsules.items():
            A = d.oMf[r.frame_id(c.frame)].act(c.a_local)
            B = d.oMf[r.frame_id(c.b_frame)].act(c.b_local)
            caps[name] = (A, B)
            for center, halfbox in spec.obstacle_boxes:
                bump(table, 'fence', c.radius+spec.obstacle_margin-axis_box_distance(A,B,center,halfbox), t)
        for ca, cb, safe in spec.selfcol_pairs:
            _, _, dist2 = seg_seg_closest(*caps[ca], *caps[cb])
            bump(table, 'self_collision', safe-np.sqrt(max(0.,dist2)), t)
        contacts = [c.name for c in r.contacts]
        for fid, name in _monitors(r):
            if mode != 1 and name in contacts: continue
            bump(table, 'clear', (max(0.,spec.footclear) if mode == 1 else 0.)-d.oMf[fid].translation[2], t)
        off, generalized = 0, np.zeros(nv)
        for c in r.contacts:
            p = d.oMf[c.frame_id].translation
            J = pin.getFrameJacobian(r.model, d, c.frame_id, pin.LOCAL_WORLD_ALIGNED)[:c.dim].copy()
            if mode != 1 or kind == 'impact':
                target = spec.feet0[c.name] if mode == 0 else spec.feet2[c.name]
                bump(table, 'foot_position', abs(p-target), t)
                if c.dim == 6: bump(table, 'foot_orientation', abs(pin.log3(d.oMf[c.frame_id].rotation)), t)
            if mode != 1: bump(table, 'foot_velocity', abs(J@v), t)
            if kind == 'impact' and spec.landvz > 0:
                bump(table, 'landing_speed', -spec.landvz-(J@v)[2], t)
            f = force[off:off+c.dim]; off += c.dim
            generalized += J.T@f
            if mode != 1:
                bump(table, 'normal', -f[2], t)
                bump(table, 'cone', np.sqrt(f[0]**2+f[1]**2+spec.friceps**2)-spec.friceps-spec.mu*f[2], t)
                # As in the reference, floor deficits in the pinned ramp are
                # reported separately and excluded from the scored maximum.
                if mode == 2 or t < info.taper_start-1e-11:
                    bump(table, 'force_floor', spec.lammin-f[2], t)
                if c.dim == 6:
                    bump(table, 'cop', [abs(f[4])-spec.copx*f[2], abs(f[3])-spec.copy*f[2]], t)
                    bump(table, 'yaw', abs(f[5])-spec.gammaz*f[2], t)
            else: bump(table, 'inactive_force', abs(f), t)
        if kind == 'liftoff':
            bump(table, 'liftoff_force', abs(force), t)
            bump(table, 'liftoff_speed', spec.vz_liftoff-v[2], t)
        if a is not None:
            defect = pin.rnea(r.model, d, q, v, a)-info.S@u-generalized
            bump(table, 'dynamics_force_units', abs(defect), t)
    for k, step in enumerate(mesh):
        mode, t, h, kind = step['mode'], step['t'], step['h'], step['kind']
        x, u = xs[k], us[k]
        if kind == 'impact':
            force = np.zeros(sum(c.dim for c in r.contacts))
            bump(node, 'impact_reset', abs(xs[k+1]-phys.reset(x)), t)
            state_rows(node, x[:nv], x[nv:], u, force, mode, t, kind)
            continue
        _, force = phys.ode(x, u, mode)
        state_rows(node, x[:nv], x[nv:], u, force, mode, t, kind)
        if kind == 'liftoff':
            bump(node, 'liftoff_continuity', abs(xs[k+1]-x), t)
            continue
        predicted = phys.rk2(x,u,h,mode)
        bump(shooting, 'rk2_gap', abs(predicted-xs[k+1]), t)
        for ss in [0., 1.]:
            state,a,f,kin = phys.dense(x,u,h,ss*h,mode)
            state_rows(node,state[:nv],state[nv:],u,f,mode,t+ss*h,a=a)
            bump(node,'kinematics_q',abs(kin[:nv]),t+ss*h)
            bump(node,'kinematics_v',abs(kin[nv:]),t+ss*h)
        ticks = np.arange(int(np.ceil((t-1e-12)/fine_dt)), int(np.floor((t+h+1e-12)/fine_dt))+1)*fine_dt
        for tt in ticks:
            ss = float(np.clip((tt-t)/h,0.,1.))
            state,a,f,kin = phys.dense(x,u,h,ss*h,mode)
            state_rows(fine,state[:nv],state[nv:],u,f,mode,tt,a=a)
            bump(fine,'kinematics_q',abs(kin[:nv]),tt)
            bump(fine,'kinematics_v',abs(kin[nv:]),tt)
    bump(node,'initial_state',abs(xs[0]-np.r_[spec.q_hold0,np.zeros(nv)]),0.)
    bump(node,'terminal_state',abs(xs[-1]-np.r_[spec.q_hold2,np.zeros(nv)]),info.seed.sol.duration)
    node_max, fine_max = max(node.values(),default=0.), max(fine.values(),default=0.)
    return dict(passed=max(node_max,fine_max,max(shooting.values(),default=0.))<=1e-3,
                finite=True,node_max=node_max,fine_max=fine_max,node=node,fine=fine,
                shooting=shooting,fine_dt=fine_dt,
                worst_times=worst,
                torque_integral=sum(s['h']*float(u@u) for s,u in zip(mesh,us)),
                dense_curve='RK2 partial-step q/v; independent physical acceleration/reaction; both kinematic derivatives audited')
