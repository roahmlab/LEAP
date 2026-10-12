"""Replicate the leap terrain/fence builders' seed IK exactly.

Reference: the C++ builders' `ikEvent` / `nominalPose` (tasks/src/terrain_walk.cpp).
Damped, step-limited Gauss-Newton least squares on the whole q, min-norm
solve, actuated joints clamped into their URDF boxes each iteration.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from .robot import Robot


def nominal_pose(robot: Robot, nominal_angles: dict) -> np.ndarray:
    """q = 0, apply nominal joint angles, then lift the base so the lowest
    contact touches z = 0. Returns q_nominal (terrain_walk.cpp:251-266)."""
    m = robot.model
    q = np.zeros(m.nq)
    for jname, ang in nominal_angles.items():
        jid = m.getJointId(jname)
        q[m.joints[jid].idx_q] = ang
    d = robot.data()
    pin.framesForwardKinematics(m, d, q)
    zmin = min(d.oMf[c.frame_id].translation[2] for c in robot.contacts)
    q[2] = -zmin
    return q


def frame_point_jacobian(robot: Robot, d, q, frame_id):
    """3 x nv world-aligned translation Jacobian of a frame origin."""
    J = pin.computeFrameJacobian(robot.model, d, q, frame_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J[:3, :]


def ik_event(robot: Robot, targets: dict, q_nominal: np.ndarray,
             base_z: float, lock_idx=(), sole_level_contacts=(),
             w_pos=1.0, w_rot=1.0, w_att=0.02, w_z=0.02, w_lock=1.0,
             w_reg=0.01, iters=150) -> np.ndarray:
    """Solve one gait-event IK. `targets` maps contact name -> world position
    (3,). For dim-6 contacts in `sole_level_contacts`, adds log3(R) rows
    (level sole). Mirrors terrain_walk.cpp:140-246."""
    m = robot.model
    d = robot.data()
    contacts = [c for c in robot.contacts if c.name in targets]
    q = q_nominal.copy()
    xy = np.mean([np.asarray(targets[c.name])[:2] for c in contacts], axis=0)
    q[0], q[1] = xy
    q[2] = base_z
    for i in lock_idx:          # terrain_walk.cpp:174 zeroes the locked coords
        q[i] = 0.0
    act = robot.act_idx
    for _ in range(iters):
        pin.framesForwardKinematics(m, d, q)
        rows, resid = [], []
        for c in contacts:
            J = frame_point_jacobian(robot, d, q, c.frame_id)
            r = d.oMf[c.frame_id].translation - np.asarray(targets[c.name])
            rows.append(w_pos * J)
            resid.append(w_pos * r)
            if c.name in sole_level_contacts:
                Jf = pin.computeFrameJacobian(m, d, q, c.frame_id,
                                              pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                R = d.oMf[c.frame_id].rotation
                rows.append(w_rot * (pin.Jlog3(R) @ R.T @ Jf[3:, :]))
                resid.append(w_rot * pin.log3(R))
        for i in (3, 4, 5):
            e = np.zeros(m.nv); e[i] = 1.0
            rows.append(w_att * e[None, :])
            resid.append([w_att * q[i]])
        e = np.zeros(m.nv); e[2] = 1.0
        rows.append(w_z * e[None, :])
        resid.append([w_z * (q[2] - base_z)])
        for i in lock_idx:
            e = np.zeros(m.nv); e[i] = 1.0
            rows.append(w_lock * e[None, :])
            resid.append([w_lock * q[i]])
        for i in act:
            e = np.zeros(m.nv); e[i] = 1.0
            rows.append(w_reg * e[None, :])
            resid.append([w_reg * (q[i] - q_nominal[i])])
        A = np.vstack(rows)
        r = np.concatenate([np.atleast_1d(x) for x in resid])
        dq = np.linalg.lstsq(A, -r, rcond=None)[0]
        step = min(1.0, 0.3 / max(np.abs(dq).max(), 1e-30))
        q = q + step * dq
        q[act] = np.clip(q[act], robot.q_lower, robot.q_upper)
        if np.abs(dq).max() < 1e-12:
            break
    return q


def ik_residual(robot: Robot, q, targets: dict) -> float:
    """Worst |FK - target| component (inf norm -- the C++ echo's posErr
    convention, so the value is comparable against the builder banner)."""
    d = robot.data()
    pin.framesForwardKinematics(robot.model, d, q)
    worst = 0.0
    for c in robot.contacts:
        if c.name in targets:
            worst = max(worst, float(np.max(np.abs(
                d.oMf[c.frame_id].translation - np.asarray(targets[c.name])))))
    return worst
