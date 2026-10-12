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
