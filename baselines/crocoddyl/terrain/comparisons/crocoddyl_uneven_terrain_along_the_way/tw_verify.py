"""Independent NumPy/Pinocchio physical audit of a terrain shooting trajectory, plus
the exact dense reconstruction of its polynomial curves and the decoding of the
native control vector into q, v, a, torque and wrench coefficients.

Adapted (2026-09-14) from comparisons/aligator_unified_terrain/scripts/verify_public.py
(the shared judge of the aligator/crocoddyl terrain comparisons). It uses no generated
residual, no ALIGATOR and no Crocoddyl object: every row is re-evaluated from the
robot model with pinocchio and the closed-form geometry certificates of common/.
"""
import numpy as np
import pinocchio as pin
from common.geometry import seg_seg_closest, separating_face_certificate
from common.plan import _monitors
from tw_inputs import reset, skip_floor, lands_here

# v2 (2026-09-14): + the trackability-study rows (per-axis attitude box, named posture
# boxes, base linear-velocity box, per-joint torque box, hover rows of the swing windows,
# touchdown rows on a landcont junction).
AUDIT_VERSION = 'terrain-physical-v2'


def weights(s):
    """Quartic q / cubic v / quadratic a weights of the three acceleration samples."""
    a = np.array([2 * s * s - 3 * s + 1, 4 * s - 4 * s * s, 2 * s * s - s])
    v = np.array([2 * s ** 3 / 3 - 1.5 * s * s + s, 2 * s * s - 4 * s ** 3 / 3, 2 * s ** 3 / 3 - .5 * s * s])
    q = np.array([s ** 4 / 6 - .5 * s ** 3 + .5 * s * s, 2 * s ** 3 / 3 - s ** 4 / 3, s ** 4 / 6 - s ** 3 / 6])
    return q, v, a


def held_coefficients(r, u):
    nv, nt, nf = r.nv, r.nu, sum(c.dim for c in r.contacts)
    count = (len(u) - 3 * nv) // (nt + nf)
    if count not in (2, 3) or len(u) != 3 * nv + count * (nt + nf): raise ValueError('Unknown control hold')
    return u[3 * nv:3 * nv + count * nt].reshape(count, nt), u[3 * nv + count * nt:].reshape(count, nf)


def dense(r, x, u, h, s):
    """q, v, a, torque, wrench of a flow stage at fraction s in [0, 1] of its duration h."""
    nv = r.nv; rates = u[:3 * nv].reshape(3, nv); qw, vw, aw = weights(s)
    q = x[:nv] + h * s * x[nv:] + h * h * (qw @ rates)
    v = x[nv:] + h * (vw @ rates); a = aw @ rates
    tau, force = held_coefficients(r, u)
    hold = np.array([(1 - s) ** 2, 2 * s * (1 - s), s * s]) if len(tau) == 3 else np.array([1 - s, s])
    return q, v, a, hold @ tau, hold @ force


def decode(r, xs, us):
    """Named arrays of one candidate: q, v (K+1, nv); a (K, 3, nv) samples at s = 0,
    1/2, 1; u (K, count, nt) torque Bernstein coefficients; lam (K, count, nf) wrench
    Bernstein coefficients (all contacts, robot yaml order, LOCAL_WORLD_ALIGNED
    [fx fy fz (tx ty tz)]). Event stages carry dummy (zero) controls."""
    xs = np.asarray(xs); us = np.asarray(us); nv, nt = r.nv, r.nu; nf = sum(c.dim for c in r.contacts)
    count = (us.shape[1] - 3 * nv) // (nt + nf)
    return dict(q=xs[:, :nv], v=xs[:, nv:], a=us[:, :3 * nv].reshape(len(us), 3, nv),
                u=us[:, 3 * nv:3 * nv + count * nt].reshape(len(us), count, nt),
                lam=us[:, 3 * nv + count * nt:].reshape(len(us), count, nf))


def inverse(info, q, v, a, tau, force):
    r = info.r; d = r.data()
    pin.computeJointJacobians(r.model, d, q); pin.updateFramePlacements(r.model, d)
    generalized = info.S @ tau; offset = 0
    for c in r.contacts:
        J = pin.getFrameJacobian(r.model, d, c.frame_id, pin.LOCAL_WORLD_ALIGNED)[:c.dim]
        generalized += J.T @ force[offset:offset + c.dim]; offset += c.dim
    return pin.rnea(r.model, d, q, v, a) - generalized


def audit(info, mesh, xs, us, fine_dt=.01, row_sink=None):
    """Node gate (every collocation point, event and shooting gap) and fine grid
    (every throughout row at the global ticks t = k * fine_dt, both owning sides of a
    boundary). passed <=> max(node_max, fine_max) <= 1e-3, raw mixed units."""
    r, s = info.r, info.spec; nv = r.nv
    if not all(np.isfinite(x).all() for x in xs + us): return dict(passed=False, finite=False)
    tables = {'node': {}, 'fine': {}}; locations = {}; data = r.data(); monitors = _monitors(r)

    def bump(group, name, value, t):
        raw = np.asarray(value)
        if not np.isfinite(raw).all(): raise ValueError(f'Nonfinite physical {name}')
        if row_sink is not None: row_sink(group, name, np.maximum(raw.ravel(), 0.), t)
        val = max(0., float(np.max(raw, initial=0.)))
        if val > tables[group].get(name, -1.): tables[group][name] = val; locations[f'{group}:{name}'] = float(t)

    def rows(group, q, v, a, tau, force, mode, t, kind='flow'):
        md = s.modes[mode]
        pin.computeJointJacobians(r.model, data, q); pin.updateFramePlacements(r.model, data)
        if a is not None: pin.forwardKinematics(r.model, data, q, v, a)
        mid, half = .5 * (r.q_lower + r.q_upper), .5 * (r.q_upper - r.q_lower)
        bump(group, 'box_q', np.r_[mid - s.qbox * half - q[r.act_idx], q[r.act_idx] - mid - s.qbox * half], t)
        bump(group, 'box_v', abs(v[r.act_idx]) - s.vlim * r.velocity_limit, t)
        bump(group, 'box_u', abs(tau) - s.effort_box, t)
        bump(group, 'base_z', -q[2], t)
        if s.att_limits: bump(group, 'box_att', [abs(q[k]) - lim for k, lim in s.att_limits.items()], t)
        if s.posture: bump(group, 'box_posture', [max(l - q[k], q[k] - h) for _, k, l, h in s.posture], t)
        if s.basev_limits: bump(group, 'box_v_base', [abs(v[k]) - lim for k, lim in s.basev_limits.items()], t)
        if s.planarlock_idx: bump(group, 'planar', abs(q[s.planarlock_idx]), t)
        caps = {}
        for name, c in r.capsules.items():
            A = data.oMf[r.frame_id(c.frame)].act(c.a_local); B = data.oMf[r.frame_id(c.b_frame)].act(c.b_local); caps[name] = (A, B)
            if name in s.keepout_capsules:
                for center, halfbox in s.keepout_boxes:
                    bump(group, 'terrain_keepout', separating_face_certificate(A, B, center, halfbox, c.radius + s.obstacle_margin)[0], t)
        for ca_, cb, safe in s.selfcol_pairs:
            _, _, dist2 = seg_seg_closest(*caps[ca_], *caps[cb]); bump(group, 'self_collision', (safe * safe - dist2) / (2 * safe), t)
        for fid, name in monitors: bump(group, 'ground_clearance', s.monitorz - data.oMf[fid].translation[2], t)
        off = 0; generalized = info.S @ tau; landing = lands_here(s, mode, kind)
        for c in r.contacts:
            pose = data.oMf[c.frame_id]; J = pin.getFrameJacobian(r.model, data, c.frame_id, pin.LOCAL_WORLD_ALIGNED)[:c.dim].copy()
            if c.name in md.active:
                bump(group, 'foot_position', abs(pose.translation - md.targets[c.name]), t)
                if c.dim == 6: bump(group, 'foot_orientation', abs(pin.log3(pose.rotation)), t)
                bump(group, 'foot_velocity', abs(J @ v), t)
                if a is not None:
                    acc = pin.getFrameClassicalAcceleration(r.model, data, c.frame_id, pin.LOCAL_WORLD_ALIGNED).vector[:c.dim]
                    bump(group, 'foot_acceleration', abs(acc), t)
            if kind == 'flow' and md.hover_target is not None and c.name == md.swing_foot:
                bump(group, 'hover_xy', abs(pose.translation[:2] - md.hover_target[:2]), t)
                if c.dim == 6: bump(group, 'hover_orientation', abs(pin.log3(pose.rotation)), t)
                bump(group, 'hover_z', md.hover_target[2] - pose.translation[2], t)
            if landing and c.name == md.swing_foot:
                bump(group, 'touchdown_position', abs(pose.translation - md.swing_to), t)
                if c.dim == 6: bump(group, 'touchdown_orientation', abs(pin.log3(pose.rotation)), t)
                if s.zeroland: bump(group, 'touchdown_velocity', abs(J @ v), t)
                else:
                    if s.zerolandxy: bump(group, 'touchdown_xy_velocity', abs((J @ v)[:2]), t)
                    if s.landvz > 0: bump(group, 'touchdown_speed', -s.landvz - (J @ v)[2], t)
            f = force[off:off + c.dim]; off += c.dim; generalized += J.T @ f
            if kind != 'flow': continue
            if c.name in md.active:
                bump(group, 'normal', -f[2], t)
                if not skip_floor(info, mode, c.name, t): bump(group, 'force_floor', s.lammin - f[2], t)
                bump(group, 'cone', np.sqrt(f[0] ** 2 + f[1] ** 2 + s.friceps ** 2) - s.friceps - s.mu * f[2], t)
                if c.dim == 6:
                    bump(group, 'cop', [abs(f[4]) - s.copx * f[2], abs(f[3]) - s.copy * f[2]], t)
                    bump(group, 'yaw', abs(f[5]) - s.gammaz * f[2], t)
                if (c.name in md.land_feet and abs(t - info.starts[mode]) < 1e-10) or (c.name in md.lift_feet and abs(t - info.starts[mode + 1]) < 1e-10): bump(group, 'force_pin', abs(f), t)
            else: bump(group, 'inactive_force', abs(f), t)
        if group == 'node' and any(abs(t - tt) < 1e-10 for tt in info.clear_times[mode]):
            fid = r.frame_id(md.swing_foot); bump(group, 'swing_clearance', md.clearz - data.oMf[fid].translation[2], t)
        if a is not None: bump(group, 'dynamics', abs(pin.rnea(r.model, data, q, v, a) - generalized), t)
    effort = 0.
    for k, step in enumerate(mesh):
        x, u = xs[k], us[k]; t, h, mode, kind = step['t'], step['h'], step['mode'], step['kind']
        if kind == 'initial':
            bump('node', 'initial_allocation', abs(xs[k + 1] - np.r_[u[:nv], np.zeros(nv)]), 0.)
            bump('node', 'initial_dummy', np.r_[abs(x), abs(u[nv:])], 0.); continue
        if kind != 'flow':
            expected = reset(info, x, mode) if kind == 'impact' else x
            bump('node', 'impact_reset' if kind == 'impact' else 'continuity', abs(xs[k + 1] - expected), t)
            bump('node', 'event_dummy_controls', abs(u), t)
            rows('node', x[:nv], x[nv:], None, np.zeros(r.nu), np.zeros(sum(c.dim for c in r.contacts)), mode, t, kind); continue
        for ss in [0., .5, 1.]:
            q, v, a, tau, f = dense(r, x, u, h, ss); rows('node', q, v, a, tau, f, mode, t + ss * h)
        bump('node', 'continuity', abs(np.r_[q, v] - xs[k + 1]), t + h)
        for tt in np.arange(int(np.ceil((t - 1e-12) / fine_dt)), int(np.floor((t + h + 1e-12) / fine_dt)) + 1) * fine_dt:
            q, v, a, tau, f = dense(r, x, u, h, float(np.clip((tt - t) / h, 0., 1.))); rows('fine', q, v, a, tau, f, mode, float(tt))
        tau_ctrl, force_ctrl = held_coefficients(r, u)
        if len(tau_ctrl) == 3:
            # the middle Bernstein coefficients obey the same boxes (convex hull certificate)
            qm, vm, *_ = dense(r, x, u, h, .5)
            rows('node', qm, vm, None, tau_ctrl[1], force_ctrl[1], mode, t + .5 * h)
            for ss, ww in [(.5 - np.sqrt(15) / 10, 5 / 18), (.5, 4 / 9), (.5 + np.sqrt(15) / 10, 5 / 18)]:
                torque = dense(r, x, u, h, ss)[3]; effort += h * ww * (torque @ torque)
        else:
            ta, tb = tau_ctrl; effort += h * (ta @ ta + ta @ tb + tb @ tb) / 3
    bump('node', 'initial_rest', abs(xs[1][nv:]), 0.)
    bump('node', 'terminal_rest', abs(xs[-1][nv:]), info.starts[-1])
    # Audit the independent terminal state even if the last shooting gap is nonzero.
    rows('node', xs[-1][:nv], xs[-1][nv:], None, np.zeros(r.nu), np.zeros(sum(c.dim for c in r.contacts)), len(s.modes) - 1, info.starts[-1], 'terminal')
    maxima = {group: max(table.values(), default=0.) for group, table in tables.items()}
    return dict(passed=max(maxima.values()) <= 1e-3, finite=True, audit_version=AUDIT_VERSION, node=tables['node'], fine=tables['fine'],
                node_max=maxima['node'], fine_max=maxima['fine'], fine_dt=fine_dt, worst_times=locations, torque_integral=float(effort),
                kinematics='dq/dt=v and dv/dt=a exactly by polynomial integration')


# Diagnostic grouping of the audit families (mirrors the LEAP report's families).
FAMILY_GROUP = {
    'dynamics': 'dynamics', 'continuity': 'kinematics', 'impact_reset': 'resets', 'initial_allocation': 'boundary',
    'initial_dummy': 'boundary', 'event_dummy_controls': 'boundary', 'initial_rest': 'boundary', 'terminal_rest': 'boundary',
    'foot_position': 'stance foot', 'foot_orientation': 'stance foot', 'foot_velocity': 'stance foot', 'foot_acceleration': 'stance foot',
    'touchdown_position': 'events', 'touchdown_orientation': 'events', 'touchdown_velocity': 'events', 'touchdown_xy_velocity': 'events',
    'touchdown_speed': 'events', 'swing_clearance': 'events', 'force_pin': 'events',
    'hover_xy': 'events', 'hover_orientation': 'events', 'hover_z': 'events',
    'box_q': 'joint box', 'box_v': 'speed box', 'box_v_base': 'speed box', 'box_u': 'torque box', 'box_att': 'attitude box', 'planar': 'attitude box',
    'box_posture': 'posture box',
    'normal': 'cone / floor / CoP / yaw', 'force_floor': 'cone / floor / CoP / yaw', 'cone': 'cone / floor / CoP / yaw',
    'cop': 'cone / floor / CoP / yaw', 'yaw': 'cone / floor / CoP / yaw', 'inactive_force': 'cone / floor / CoP / yaw',
    'base_z': 'ground keep-out', 'ground_clearance': 'ground keep-out', 'terrain_keepout': 'obstacle keep-out', 'self_collision': 'self-collision',
}
