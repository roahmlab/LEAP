"""The terrain restoration rows on the lifted polynomial transcription.

Adapted (2026-09-14) from comparisons/aligator_unified_terrain/scripts/public_problem.py:
the SAME symbolic row set, row order and bound schedule (so the CasADi-generated
residual libraries are shared through their expression digest), minus the ALIGATOR
problem builder. Nothing here depends on an optimizer.

Control layout of a flow stage (hold='quadratic', the default):
    u = [a0, am, a1 (3 nv) | tauB0, tauB1, tauB2 (3 nt) | FB0, FB1, FB2 (3 nf)]
acceleration samples at s = 0, 1/2, 1 (quadratic a(s), cubic v, quartic q by exact
integration); torque and all-contact wrench as quadratic Bernstein curves.
"""
from types import SimpleNamespace
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from tw_inputs import ROOT, inputs, ramp, lands_here
from common.plan import _monitors
from aligator_bench.solve_fence_lifted import generate

# Shared generated-library cache (expression digest checked; identical expressions
# reuse the compiled model.so of the aligator terrain comparison).
GENERATED = ROOT / 'comparisons/aligator_unified_terrain/generated'


def dimensions(r):
    nv, nt, nf = r.nv, r.nu, sum(c.dim for c in r.contacts)
    count = 3 if getattr(r, 'hold', 'linear') == 'quadratic' else 2
    return nv, nt, nf, 3 * nv + count * (nt + nf)


def controls_from_samples(r, points):
    """Flow-stage control from the seed sampled at s = 0, 1/2, 1: the three
    acceleration samples, and torque / wrench Bernstein coefficients that
    interpolate the three samples exactly (quadratic) or the endpoints (linear)."""
    count = 3 if getattr(r, 'hold', 'linear') == 'quadratic' else 2

    def coeff(index):
        a, m, b = [p[index] for p in points]
        return np.array([a, 2 * m - .5 * (a + b), b]) if count == 3 else np.array([a, b])
    return np.r_[np.concatenate([p[2] for p in points]), coeff(3).ravel(), coeff(4).ravel()]


def cold(robot, dt, hold='quadratic'):
    """The benchmark inputs plus the cold control samples imported from the LEAP seed."""
    info, mesh, xs, _, fields = inputs(robot, dt)
    info.r.hold = hold; info.r.native_acceleration = True
    nv, nt, nf, nu = dimensions(info.r)
    us = []
    for step in mesh:
        if step['kind'] != 'flow':
            control = np.zeros(nu)
            if step['kind'] == 'initial': control[:nv] = xs[1][:nv]
            us.append(control); continue
        t, h = step['t'], step['h']
        element = info.seed.element(t)
        samples = [info.seed.sample(t + c * h, element=element) for c in [0., .5, 1.]]
        us.append(controls_from_samples(info.r, [(v['q'], v['v'], v['a'], v['u'], v['lambda']) for v in samples]))
    return info, mesh, xs, us, fields


def symbolic(info):
    """CasADi raw rows (state, inputs, contact kinematics, monitors, cones, wrench
    boxes, keep-out certificates) and the generated bounded-residual library."""
    r, s = info.r, info.spec
    nv, nt, nf, nu = dimensions(r)
    nx = 2 * nv
    x, u = ca.SX.sym('x', nx), ca.SX.sym('u', nu)
    model = cpin.Model(r.model); data = model.createData()
    cpin.computeJointJacobians(model, data, x[:nv]); cpin.updateFramePlacements(model, data)
    rows, names = [], []

    def add(expr, name):
        expr = ca.vec(expr)
        for j in range(expr.numel()): rows.append(expr[j]); names.append(f'{name}_{j}')
    count = (nu - 3 * nv) // (nt + nf)
    add(x, 'x')
    for ep in range(count): add(u[3 * nv + ep * nt:3 * nv + (ep + 1) * nt], f'tau{ep}')
    for ep in range(count): add(u[3 * nv + count * nt + ep * nf:3 * nv + count * nt + (ep + 1) * nf], f'force{ep}')
    for c in r.contacts:
        pose = data.oMf[c.frame_id]
        add(pose.translation, f'pos_{c.name}')
        if c.dim == 6: add((pose.rotation - ca.SX.eye(3)) / np.sqrt(2.), f'rot_{c.name}')
        J = cpin.getFrameJacobian(model, data, c.frame_id, pin.LOCAL_WORLD_ALIGNED)[:c.dim, :]
        add(J @ x[nv:], f'vel_{c.name}')
        for j in range(3):
            expression = ca.SX.zeros(c.dim) if getattr(r, 'native_acceleration', False) else J @ u[j * nv:(j + 1) * nv] + ca.jtimes(J @ x[nv:], x[:nv], x[nv:])
            add(expression, f'acc{j}_{c.name}')
    for fid, name in _monitors(r): add(data.oMf[fid].translation[2], f'mon_{name}')
    for endpoint in range(count):
        offset = 3 * nv + count * nt + endpoint * nf
        for c in r.contacts:
            f = u[offset:offset + c.dim]; offset += c.dim
            add(ca.sqrt(f[0] ** 2 + f[1] ** 2 + s.friceps ** 2) - s.friceps - s.mu * f[2], f'cone{endpoint}_{c.name}')
            if c.dim == 6:
                add(ca.vertcat(f[4] - s.copx * f[2], -f[4] - s.copx * f[2],
                               f[3] - s.copy * f[2], -f[3] - s.copy * f[2],
                               f[5] - s.gammaz * f[2], -f[5] - s.gammaz * f[2]), f'wrench{endpoint}_{c.name}')
    for name in s.keepout_capsules:
        cap = r.capsules[name]
        A = data.oMf[r.frame_id(cap.frame)].act(ca.SX(cap.a_local))
        B = data.oMf[r.frame_id(cap.b_frame)].act(ca.SX(cap.b_local))
        for j, (center, half) in enumerate(s.keepout_boxes):
            faces = [ca.fmin(sign * (A[k] - center[k]) - half[k], sign * (B[k] - center[k]) - half[k])
                     for k, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1)]]
            best = faces[0]
            for value in faces[1:]: best = ca.fmax(best, value)
            add(cap.radius + s.obstacle_margin - best, f'keepout_{name}_{j}')
    raw = ca.vertcat(*rows); nr = raw.numel()
    p = ca.SX.sym('p', 3 * nr)
    residual = p[2 * nr:] * (ca.fmax(p[:nr] - raw, 0) - ca.fmax(raw - p[nr:2 * nr], 0))
    obj = SimpleNamespace(x=x, u=u, p=p, cost_y_expr=residual, cost_y_expr_e=residual)
    directory = GENERATED / (f'{r.name}_collocation_physical_v2' + ('_quadratic' if count == 3 else '') + ('_native_acceleration' if getattr(r, 'native_acceleration', False) else ''))
    library, funcs, digest = generate(obj, directory, [("constraints", raw)], optimized=True)
    return SimpleNamespace(library=library, funcs=funcs, digest=digest, names=names,
                           nr=nr, row={name: j for j, name in enumerate(names)}, nu=nu)


def bounds(info, step, sym, fraction=0.):
    """Bounds p = [lo; hi; w] of every raw row at one collocation point of one stage
    (unposed rows keep w = 0 and infinite bounds). The physical problem statement."""
    r, s = info.r, info.spec; nv, nt, nf, nu = dimensions(r)
    lo, hi, w = np.full(sym.nr, -1e20), np.full(sym.nr, 1e20), np.zeros(sym.nr)

    def vec(prefix, l, h, weight=1.):
        l, h = np.atleast_1d(l), np.atleast_1d(h)
        for j in range(max(len(l), len(h))):
            i = sym.row[f'{prefix}_{j}']; lo[i], hi[i], w[i] = l[j % len(l)], h[j % len(h)], weight

    def row(name, l, h, weight=1.):
        i = sym.row[name]; lo[i], hi[i], w[i] = l, h, weight
    mode, kind = step['mode'], step['kind']; md = s.modes[mode]; t = step['t'] + fraction * step['h']
    mid, half = .5 * (r.q_lower + r.q_upper), .5 * (r.q_upper - r.q_lower)
    # joint position / speed boxes (URDF limits scaled by qbox / vlim), base z >= 0,
    # attitude box (attmax, or the per-axis attbox), named posture boxes, the base
    # linear-velocity box (basevbox), planar locks, rest at t = 0 and t = T
    for j, k in enumerate(r.act_idx):
        row(f'x_{k}', mid[j] - s.qbox * half[j], mid[j] + s.qbox * half[j])
        row(f'x_{nv + k}', -s.vlim * r.velocity_limit[j], s.vlim * r.velocity_limit[j])
    row('x_2', 0., 1e20)
    for k, lim in s.att_limits.items(): row(f'x_{k}', -lim, lim)
    for _, k, l, h in s.posture: row(f'x_{k}', max(l, -1e20), min(h, 1e20))
    for k, lim in s.basev_limits.items(): row(f'x_{nv + k}', -lim, lim)
    for k in s.planarlock_idx: row(f'x_{k}', 0., 0.)
    if abs(t) < 1e-11 or abs(t - info.starts[-1]) < 1e-11:
        for k in range(nv): row(f'x_{nv + k}', 0., 0.)
    # input coefficients: torque box (ulim x URDF, times the per-joint ulimjoint
    # scales); stance normal floor (ramped), cone, CoP / yaw boxes; inactive or
    # event-pinned wrench coefficients zero
    count = (nu - 3 * nv) // (nt + nf)
    for ep in range(count):
        if kind == 'flow': vec(f'tau{ep}', -s.effort_box, s.effort_box)
        offset = 0; te = step['t'] + ep / (count - 1) * step['h']
        for c in r.contacts:
            if c.name in md.active and kind == 'flow':
                row(f'force{ep}_{offset + 2}', s.lammin * ramp(info, mode, c.name, te), 1e20)
                vec(f'cone{ep}_{c.name}', -1e20, 0.)
                if c.dim == 6: vec(f'wrench{ep}_{c.name}', np.full(6, -1e20), np.zeros(6))
                pinned = (c.name in md.land_feet and abs(te - info.starts[mode]) < 1e-10) or (c.name in md.lift_feet and abs(te - info.starts[mode + 1]) < 1e-10)
                if pinned:
                    for j in range(c.dim): row(f'force{ep}_{offset + j}', 0., 0.)
            else:
                for j in range(c.dim): row(f'force{ep}_{offset + j}', 0., 0.)
            offset += c.dim
    # stance feet: placement, level sole (dim 6), zero velocity, zero classical
    # acceleration (the acceleration rows are the native ContactAcceleration
    # residual; the generated acc rows are identically zero); hover rows of the swing
    # foot in a lift / approach window (xy pinned over hover_target, level sole,
    # z >= the target's height: terrain_walk.cpp hoverRows); touchdown rows on the
    # swing foot at the event that ends the landing mode (impact, or the landcont junction)
    landing = lands_here(s, mode, kind)
    for c in r.contacts:
        if c.name in md.active:
            vec(f'pos_{c.name}', md.targets[c.name], md.targets[c.name])
            if c.dim == 6: vec(f'rot_{c.name}', np.zeros(9), np.zeros(9))
            vec(f'vel_{c.name}', np.zeros(c.dim), np.zeros(c.dim))
            if kind == 'flow': vec(f'acc{int(round(2 * fraction))}_{c.name}', np.zeros(c.dim), np.zeros(c.dim))
        if kind == 'flow' and md.hover_target is not None and c.name == md.swing_foot:
            row(f'pos_{c.name}_0', md.hover_target[0], md.hover_target[0]); row(f'pos_{c.name}_1', md.hover_target[1], md.hover_target[1])
            row(f'pos_{c.name}_2', md.hover_target[2], 1e20)
            if c.dim == 6: vec(f'rot_{c.name}', np.zeros(9), np.zeros(9))
        if landing and c.name == md.swing_foot:
            vec(f'pos_{c.name}', md.swing_to, md.swing_to)
            if c.dim == 6: vec(f'rot_{c.name}', np.zeros(9), np.zeros(9))
            if s.zeroland: vec(f'vel_{c.name}', np.zeros(c.dim), np.zeros(c.dim))
            else:
                if s.zerolandxy:
                    row(f'vel_{c.name}_0', 0., 0.); row(f'vel_{c.name}_1', 0., 0.)
                if s.landvz > 0: row(f'vel_{c.name}_2', -s.landvz, 1e20)
    # mid-swing clearance at the archived elements' 1/4, 1/2, 3/4 times; monitor
    # frames above monitorz; terrain keep-out certificate <= 0
    if any(abs(t - tt) < 1e-10 for tt in info.clear_times[mode]): row(f'pos_{md.swing_foot}_2', md.clearz, 1e20)
    for _, name in _monitors(r): vec(f'mon_{name}', s.monitorz, 1e20)
    for name in sym.names:
        if name.startswith('keepout_'): row(name, -1e20, 0., 1.)
    return np.r_[lo, hi, w]


def posed_rows(p, nr):
    """Number of raw rows with a positive weight in one bound vector."""
    return int(np.count_nonzero(p[2 * nr:] > 0))
