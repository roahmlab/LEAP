"""Whole-body inverse-dynamics QP controller for the floating-base benchmark
robots (self-contained port of the design in `../../simulation/controllers.py`,
solved with ProxQP).

Every control period the controller solves one small dense convex QP in

    x = [ a (nv) ; f (nf) ; s (nf) ]

generalized acceleration, one force (3) or wrench (6, `[fx fy fz tx ty tz]`,
world axes, torque about the contact frame) per SUPPORT contact, and a
penalized contact-acceleration slack. With M, h = C v + g and the stacked
LOCAL_WORLD_ALIGNED contact Jacobian J_c from pinocchio (the optimizer's
model, same URDF and coordinate order):

    minimize   sum_i w_i (a_i - a_des,i)^2                       base + joint tracking
             + w_swing  sum_swing |J_f a + Jdot_f v - a_des,f|^2 swing-foot Cartesian tracking
             + w_force  |f - f_ref|^2 / (m g)^2                  planned contact forces
             + w_tau    |(tau(x) - u_ref) / tau_max|^2           planned torques
             + w_contact |s|^2
    s.t.       M_base a + h_base - J_c,base^T f = 0              unactuated dynamics
               a_lock = 0                                        planar-lock DOFs (reaction unknown)
               J_c a + Jdot_c v = a_c,des + s                    stance feet hold their anchor
               f in the friction pyramid, f_z >= 0, CoP/yaw boxes (dim 6)
               |tau(x)| <= tau_max,   tau(x) = M_act a + h_act - J_c,act^T f

with a_des = a_ref + Kp (q_ref - q) + Kd (v_ref - v) on every DOF (floating
base included: the base is steered through the contact forces) and the torque
`tau(x)` of the solution sent to the simulator. At zero tracking error and a
consistent reference, (a_ref, lam_ref, u_ref) is the minimizer, so the
controller reduces to feed-forward of the optimizer's own torques.

The contact set is decided by `ContactPolicy` from the plan's schedule AND the
simulator's measured contacts: a scheduled foot is load-bearing only while it
physically touches (with a short debounce), a foot touching late in ITS OWN
swing becomes support early (a genuine early landing), and lingering contacts
right after a scheduled liftoff are not support. Everything gait-specific
comes through `RefSample.foot_phase` (per foot: established stance / swing
with progress / just landed), so the same controller runs the fence jump (all
feet swing together) and the stepping-stone walk (one foot at a time). Support feet hold the position (and,
for flat feet, orientation) at which they were anchored (Baumgarte drift
correction), never dragged toward a planned foothold they missed: a planted
foot cannot move without slipping.

No motor model: the robot yamls declare `motor_model: none`, and so does the
optimizer, so the QP torque (clipped to the URDF box) is applied directly.

`contact_geometry` lives in `leaptrack.scene`, `RefSample`/FOOT_* in `leaptrack.reference`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

from .determinism import proxsuite  # the plain ProxQP build (leaptrack.determinism)
import numpy as np
import pinocchio as pin

from .scene import contact_projection, wrench_map
from .reference import FOOT_LANDED, FOOT_SWING, RefSample


@dataclass
class WbcGains:
    # acceleration-space PD (shared across robots: mass-independent)
    kp_joint: float = 300.0
    kd_joint: float = 35.0
    kp_base_pos: float = 150.0
    kd_base_pos: float = 25.0
    kp_base_rot: float = 200.0
    kd_base_rot: float = 30.0
    # task weights
    w_joint: float = 5.0
    w_base_pos: float = 20.0
    w_base_rot: float = 30.0
    w_swing: float = 30.0           # swing-foot task with support (a late foot seeking the ground)
    w_swing_flight: float = 500.0   # airborne: foot placement dominates joint tracking
    w_contact: float = 1.0e4        # near-hard: a planted foot cannot accelerate
    flight_base_scale: float = 0.0  # base POSITION tracking weight scale while airborne (ballistic: 0)
    flight_base_rot_scale: float = 0.0  # base ATTITUDE weight scale airborne (limb swing can steer it)
    flight_joint_scale: float = 0.02 # joint tracking weight scale while airborne (foot placement wins)
    swing_joint_scale: float = 0.02  # with support: joint weight scale on a SWING leg's chain (its world-frame foot task wins)
    stance_force_floor: bool = True  # support feet the plan loads above lammin keep fz >= lammin (the problem's own floor)
    foothold_relative: bool = True  # shift the base position reference by the landing foothold error
    w_tau: float = 0.05
    w_force: float = 0.01
    # Optional balance tasks (off by default).
    # CoM acceleration is a task on the complete robot, including swing limbs.
    w_com: float = 0.0
    kp_com: float = 150.0
    kd_com: float = 30.0
    w_angular_momentum: float = 0.0
    kd_angular_momentum: float = 10.0
    level_stance: bool = False
    physical_stance: bool = False  # level an early contact at the measured surface, even before the planned touchdown
    force_floor_tolerance: float = 0.0
    kp_motor: float = 0.0         # torque-limit-normalized actuator impedance
    kd_motor: float = 0.0
    torque_filter: float = 0.0   # first-order command filter time constant [s]
    inner_velocity_gain: float = 0.0  # optional servo at the physics step [1/s]
    joint_limit_rate: float = 0.0  # opt-in exponential joint-limit braking [1/s]
    joint_limit_margin: float = 0.02  # inward position margin [rad]
    joint_limit_slack_weight: float = 0.0  # positive: soften braking if torque authority is exhausted
    capture_gain: float = 0.0   # swing placement feedback on divergent CoM error
    capture_max: float = 0.03   # maximum foothold correction [m]
    # contact drift correction and swing tracking
    kp_contact: float = 200.0
    kd_contact: float = 30.0
    kp_swing: float = 600.0
    kd_swing: float = 50.0
    # Late in the flight the swing-foot reference is lowered by touchdown_depth
    # (smoothly, from touchdown_blend of the flight on) and a scheduled foot
    # that has not touched yet keeps the full bias.
    touchdown_depth: float = 0.01
    touchdown_blend: float = 0.6
    stone_touchdown_depth: float = 0.002  # separate target bias from the collision-mask threshold
    consistent_swing: bool = False  # differentiate the biased/clamped swing-height target
    coupled_wrench: bool = False  # realize each flat-foot wrench with unilateral corner forces
    measured_patch: bool = False  # use measured top contacts and allow rocking until the sole lands
    point_contact: bool = False  # account for the physical sphere offset at point feet
    # cone: friction pyramid sides; mu/cop/gammaz default to the problem's
    friction_sides: int = 8
    mu: float | None = None
    fz_min: float = 0.0
    ridge: float = 1e-8
    # Impact window. For `impact_window` seconds after each measured touchdown
    # (sub-step contact events, `ContactState.touchdown`) the derivative gains of the
    # landing leg's chain (and of the floating base, with `impact_base`) are multiplied
    # by `impact_kd_scale`. `impact_projection` instead removes from the velocity error
    # the part a contact impulse at the landing feet can change,
    # e_v - M^-1 J' (J M^-1 J')^-1 J e_v, on every DOF.
    impact_window: float = 0.0       # s after a measured touchdown (0: off)
    impact_kd_scale: float = 0.0     # Kd factor on the affected DOFs inside the window
    impact_base: bool = True         # the floating base's DOFs are affected too
    impact_projection: bool = False  # impact-invariant projection instead of the Kd factor
    # Sole clearance: the swing-foot position target of a flat foot is raised by the
    # depth of its planned sole's lowest corner (the simulated sole = yaml foot_box +
    # SOLE_MARGIN) below the landing surface; zero once the planned sole is level.
    sole_clearance: bool = False
    # Depth bias after the planned touchdown only: `landed_depth` >= 0 replaces
    # `touchdown_depth` after the planned touchdown (FOOT_LANDED, or a scheduled foot
    # the plan loads) and leaves the swing before it untouched; negative = follow
    # `touchdown_depth`.
    landed_depth: float = -1.0
    # Re-arm window of the touchdown clock (reference.TouchdownClock `rearm`, s; 0 =
    # off): a touchdown whose feet all lose contact within it sends the reference back
    # to the pre-impact approach until the feet touch again.
    touchdown_rearm: float = 0.0
    # The configuration that produced these gains (`controllers.gains_for`):
    # "impact_aware" or "published". A label, recorded with the gains in every
    # metrics/evaluation file.
    config_profile: str = "published"
    # contact policy
    contact_grace: float = 0.0       # s: keep a support foot after contact loss (0: measured contact decides)
    early_contact_phase: float = 0.8 # flight progress after which a touch is a landing
    # QP
    eps_abs: float = 1e-7
    max_iter: int = 400

    def to_dict(self):
        return asdict(self)


@dataclass
class ContactState:
    support: list                  # load-bearing contacts (canonical order)
    swing: list                    # the others
    anchors: dict                  # support contact -> (p_anchor(3), R_anchor(3x3))
    early: set = field(default_factory=set)
    t: float = 0.0                 # controller time of this state
    touchdown: dict = field(default_factory=dict)  # contact -> time of its latest measured touchdown


class ContactPolicy:
    def __init__(self, contact_names, gains: WbcGains):
        self.names = list(contact_names)
        self.g = gains
        self.reset()

    def reset(self):
        self.last_touch = {f: -np.inf for f in self.names}
        self.latched = set()
        self.anchors = {}
        self._init = False
        self.touchdown = {}
        self._measured_prev = set()

    def update(self, t: float, measured: set, ref: RefSample, foot_pose: dict,
               touchdowns: dict | None = None) -> ContactState:
        """`foot_pose`: contact -> (p, R) at the current state (for new anchors).
        `touchdowns`: contact -> time of its latest touchdown seen by the simulator
        at the physics step (a contact that started, possibly between two control
        ticks); a foot that touches at this tick and did not at the previous one
        counts as a touchdown too."""
        sched = set(ref.scheduled)
        if not self._init:
            for f in sched:            # the scheduled initial stance counts as touching from the start
                self.last_touch[f] = t
            self._init = True
            self._measured_prev = set(measured)
        for f in self.names:
            if f in measured and f not in self._measured_prev:
                self.touchdown[f] = t
            if touchdowns and f in touchdowns and touchdowns[f] > self.touchdown.get(f, -np.inf):
                self.touchdown[f] = float(touchdowns[f])
        self._measured_prev = set(measured)
        for f in measured:
            self.last_touch[f] = t
        support = set()
        for f in sched:
            if f in measured or t - self.last_touch[f] <= self.g.contact_grace:
                support.add(f)
        # early landing: a foot touching during the late part of ITS swing (or
        # after it has been put down by the plan) is a genuine early landing
        for f in measured - sched:
            st, s = ref.foot_phase.get(f, (FOOT_LANDED, 1.0))
            if (st == FOOT_SWING and s >= self.g.early_contact_phase) or st == FOOT_LANDED:
                self.latched.add(f)
        # a latched foot stays support while it keeps touching; drop it once
        # the schedule says stance (it is then handled as a scheduled foot)
        self.latched = {f for f in self.latched - sched
                        if t - self.last_touch[f] <= self.g.contact_grace}
        support |= self.latched
        # anchors: keep for continuing support feet, create for new ones
        for f in list(self.anchors):
            if f not in support:
                del self.anchors[f]
        for f in support:
            if f not in self.anchors:
                p, R = foot_pose[f]
                self.anchors[f] = (p.copy(), R.copy())
        sup = [f for f in self.names if f in support]
        return ContactState(support=sup, swing=[f for f in self.names if f not in support],
                            anchors=dict(self.anchors), early=set(self.latched),
                            t=float(t), touchdown=dict(self.touchdown))


class WholeBodyQP:
    def __init__(self, robot, contact_dims: dict, *, gains: WbcGains | None = None,
                 mu: float = 0.85, cop: dict | None = None, gammaz: float = 0.05,
                 lock_idx=(), tau_max=None, lammin: float = 5.0):
        """`robot`: common.robot.Robot (pinocchio model). `cop`: contact name ->
        (copx, copy) half-extents for dim-6 feet. `lock_idx`: pinocchio DOF
        indices held at zero by the simulator (planar lock). `lammin`: the
        problem's stance normal-force floor (a scheduled foot the plan loads
        above it must be touching)."""
        self.g = gains or WbcGains()
        self.robot = robot
        self.model = robot.model
        self.data = self.model.createData()
        self.rdata = self.model.createData()
        self.nv = int(self.model.nv)
        self.act = np.asarray(robot.act_idx, int)
        self.nu = len(self.act)
        act_set = set(self.act.tolist())
        self.base = np.array([i for i in range(self.nv) if i not in act_set], int)
        assert len(self.base) == 6, self.base
        self.lock = np.asarray(sorted(set(int(i) for i in lock_idx)), int)
        self.dyn_base = np.array([i for i in self.base if i not in set(self.lock.tolist())], int)
        self.contact_names = list(contact_dims)
        self.dims = {f: int(d) for f, d in contact_dims.items()}
        self.fids = {f: self.model.getFrameId(f) for f in self.contact_names}
        self.mu = float(self.g.mu if self.g.mu is not None else mu)
        self.cop = dict(cop or {})
        self.gammaz = float(gammaz)
        self.lammin = float(lammin)
        self.slab_tol = 0.002        # = rollout.STONE_SIDE_TOL: below this the stone top does not exist for the foot
        # callable(p_ref (3,)) -> top z of the stone under that xy, or None for
        # the ground plane; set by the rollout for terrain scenes
        self.surface_z_fn = None
        self.tau_max = (np.asarray(robot.effort_limit, float) if tau_max is None
                        else np.broadcast_to(np.asarray(tau_max, float), (self.nu,)).copy())
        self.tau_max = np.where(np.isfinite(self.tau_max) & (self.tau_max > 0), self.tau_max, 1e6)
        self.total_mass = robot.total_mass()
        self.force_scale = max(self.total_mass * 9.81, 1.0)
        g = self.g
        self.kp = np.zeros(self.nv)
        self.kd = np.zeros(self.nv)
        self.kp[self.act] = g.kp_joint
        self.kd[self.act] = g.kd_joint
        self.kp[self.base[:3]] = g.kp_base_pos
        self.kd[self.base[:3]] = g.kd_base_pos
        self.kp[self.base[3:]] = g.kp_base_rot
        self.kd[self.base[3:]] = g.kd_base_rot
        self.wa = np.full(self.nv, g.w_joint)
        self.wa[self.base[:3]] = g.w_base_pos
        self.wa[self.base[3:]] = g.w_base_rot
        self.wa[self.lock] = 0.0
        # actuated joints on the kinematic chain of each contact frame (base
        # excluded): the joints a swing foot's world-frame task must own
        self.chain = {}
        for f, fid in self.fids.items():
            pj = self.model.frames[fid].parentJoint
            js = [int(j) for j in self.model.supports[pj]]
            vidx = [self.model.joints[j].idx_v for j in js if j > 0]
            self.chain[f] = np.array([i for i in vidx if i in act_set], int)
        self._qps = {}
        self.n_fallback = 0
        self.n_cold_retries = 0
        self.n_solves = 0
        self.n_impact_ticks = 0          # control ticks inside an impact window
        self.last_info = {}
        self.control_dt = 1e-3
        self._previous_tau = None
        self.com_acc_override = None
        self.contact_points = {}
        # contact -> (k, 3) sole corners in the contact frame (the simulated sole;
        # set by the rollout from RobotSim.foot_corners, read by `sole_clearance`)
        self.sole_corners = {}

    # ------------------------------------------------------------ kinematics
    def frame_state(self, q, v):
        """Current M, h, and per-contact (J, p, R, vel, Jdot v) in LWA."""
        m, d = self.model, self.data
        pin.crba(m, d, q)
        M = np.triu(d.M) + np.triu(d.M, 1).T
        h = pin.nonLinearEffects(m, d, q, v).copy()
        pin.forwardKinematics(m, d, q, v, np.zeros(self.nv))
        pin.updateFramePlacements(m, d)
        pin.computeJointJacobians(m, d, q)
        out = {}
        for f, fid in self.fids.items():
            k = self.dims[f]
            J = pin.getFrameJacobian(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:k].copy()
            vel = np.asarray(pin.getFrameVelocity(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector)[:k].copy()
            jdv = np.asarray(pin.getFrameClassicalAcceleration(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector)[:k].copy()
            out[f] = dict(J=J, p=d.oMf[fid].translation.copy(), R=d.oMf[fid].rotation.copy(), vel=vel, jdv=jdv)
        return M, h, out

    def ref_frames(self, q, v, a):
        m, d = self.model, self.rdata
        pin.forwardKinematics(m, d, q, v, a)
        pin.updateFramePlacements(m, d)
        out = {}
        for f, fid in self.fids.items():
            k = self.dims[f]
            out[f] = dict(
                p=d.oMf[fid].translation.copy(), R=d.oMf[fid].rotation.copy(),
                vel=np.asarray(pin.getFrameVelocity(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector)[:k].copy(),
                acc=np.asarray(pin.getFrameClassicalAcceleration(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector)[:k].copy())
        return out

    def foot_poses(self, q):
        m, d = self.model, self.rdata
        pin.framesForwardKinematics(m, d, q)
        return {f: (d.oMf[fid].translation.copy(), d.oMf[fid].rotation.copy()) for f, fid in self.fids.items()}

    @staticmethod
    def _pose_err(k, p, R, p_ref, R_ref):
        e = [p - p_ref]
        if k == 6:
            e.append(R_ref @ pin.log3(R_ref.T @ R))
        return np.concatenate(e)

    @staticmethod
    def swing_height(position, velocity, acceleration, depth, depth_rate=0., depth_acc=0.,
                     surface=None, stone_depth=0.):
        """Height and its derivatives on the active branch of the landing target.

        A target clamped to a fixed stone top has zero vertical velocity and
        acceleration.
        """
        position -= depth
        velocity -= depth_rate
        acceleration -= depth_acc
        if surface is not None and position < surface - stone_depth:
            return surface - stone_depth, 0., 0.
        return position, velocity, acceleration

    # ------------------------------------------------------------ QP
    def compute(self, q, v, ref: RefSample, cs: ContactState):
        g = self.g
        nv = self.nv
        M, h, cur = self.frame_state(q, v)
        self.last_mass = M
        if g.point_contact:
            for f in cs.support:
                if self.dims[f] != 3 or not len(self.contact_points.get(f, [])):
                    continue
                fid = self.fids[f]
                offset = np.mean(self.contact_points[f], axis=0)
                G = wrench_map([offset]).T
                J6 = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED).copy()
                vel6 = pin.getFrameVelocity(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED).vector.copy()
                drift6 = pin.getFrameClassicalAcceleration(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED).vector.copy()
                cur[f]['J'] = G @ J6
                cur[f]['vel'] = G @ vel6
                cur[f]['jdv'] = G @ drift6 + np.cross(vel6[3:], np.cross(vel6[3:], offset))
        rf = self.ref_frames(ref.q, ref.v, ref.a)
        support, swing = cs.support, cs.swing
        nf = sum(self.dims[f] for f in support)
        ng = self.nu if g.joint_limit_rate > 0. and g.joint_limit_slack_weight > 0. else 0
        patches = {}
        for f in support:
            if self.dims[f] == 6 and (g.coupled_wrench or g.measured_patch):
                if g.measured_patch and len(self.contact_points.get(f, [])):
                    patches[f] = np.asarray(self.contact_points[f])
                else:
                    cx, cy = self.cop.get(f, (.05, .03))
                    patches[f] = np.array([[sx*cx, sy*cy, 0.] for sx in (-1., 1.)
                                           for sy in (-1., 1.)]) @ cur[f]['R'].T
        npatch = sum(3*len(p) for p in patches.values())
        n = nv + 2 * nf + ng + npatch
        of, os_ = nv, nv + nf
        og = nv + 2 * nf
        op = og + ng
        patch_slices = {}
        for f, points in patches.items():
            patch_slices[f] = slice(op, op + 3*len(points))
            op += 3*len(points)

        # tracking target. Base POSITION reference: foothold-relative when the
        # support feet were anchored away from their planned footholds (a
        # planted foot cannot be moved to the plan, so the plan moves to it).
        q_ref = ref.q.copy()
        if g.foothold_relative and support:
            offs = []
            for f in support:
                tgt = ref.targets.get(f)
                if tgt is not None:
                    offs.append(cs.anchors[f][0][:2] - np.asarray(tgt)[:2])
            if offs:
                q_ref[self.base[:2]] += np.mean(offs, axis=0)
        qe = q_ref - q
        qe[self.base[3:]] = (qe[self.base[3:]] + np.pi) % (2 * np.pi) - np.pi
        ev = ref.v - v
        kd = self.kd
        landing = []
        if g.impact_window > 0.0:
            landing = [f for f in self.contact_names
                       if 0.0 <= cs.t - cs.touchdown.get(f, -np.inf) < g.impact_window]
        if landing:
            self.n_impact_ticks += 1
            if g.impact_projection:
                Jl = np.vstack([cur[f]["J"] for f in landing])
                MiJt = np.linalg.solve(M, Jl.T)
                ev = ev - MiJt @ np.linalg.solve(Jl @ MiJt, Jl @ ev)
            else:
                idx = set()
                for f in landing:
                    idx.update(int(i) for i in self.chain[f])
                if g.impact_base:
                    idx.update(int(i) for i in self.base)
                kd = self.kd.copy()
                kd[sorted(idx)] *= g.impact_kd_scale
        a_des = ref.a + self.kp * qe + kd * ev
        wa = self.wa.copy()
        if not support:                      # airborne: the base is unactuated
            wa[self.base[:3]] *= g.flight_base_scale
            wa[self.base[3:]] *= g.flight_base_rot_scale
            wa[self.act] *= g.flight_joint_scale
        else:
            # a swing leg's joints yield to its world-frame foot task (a walk's
            # single/triple support; in the fence this only touches the few ms
            # in which one foot has landed and the other has not)
            for f in swing:
                wa[self.chain[f]] *= g.swing_joint_scale
            if g.physical_stance:
                for f in cs.early:
                    wa[self.chain[f]] *= g.swing_joint_scale

        rows, rhs = [], []   # weighted least squares rows

        def add(A, b, w):
            sw = np.sqrt(w)
            rows.append(A * sw)
            rhs.append(b * sw)

        A = np.zeros((nv, n))
        A[:, :nv] = np.eye(nv)
        for i in range(nv):
            if wa[i] > 0:
                add(A[i:i + 1], a_des[i:i + 1], wa[i])
        if support and g.w_com > 0.0:
            m, d, rd = self.model, self.data, self.rdata
            Jcom = pin.jacobianCenterOfMass(m, d, q).copy()
            pin.centerOfMass(m, d, q, v, np.zeros(nv))
            pcom, vcom, drift = d.com[0].copy(), d.vcom[0].copy(), d.acom[0].copy()
            pin.centerOfMass(m, rd, q_ref, ref.v, ref.a)
            desired = (rd.acom[0] + g.kp_com * (rd.com[0] - pcom)
                       + g.kd_com * (rd.vcom[0] - vcom) - drift)
            if self.com_acc_override is not None:
                desired = self.com_acc_override - drift
            Acom = np.zeros((3, n))
            Acom[:, :nv] = Jcom
            add(Acom, desired, g.w_com)
        if support and g.w_angular_momentum > 0.:
            m,d,rd = self.model,self.data,self.rdata
            pin.computeCentroidalMapTimeVariation(m,d,q,v)
            Ag,dAg = d.Ag.copy(),d.dAg.copy()
            momentum = Ag @ v
            pin.computeCentroidalMomentumTimeVariation(m,rd,ref.q,ref.v,ref.a)
            target = (rd.dhg.vector[3:] + g.kd_angular_momentum *
                      (rd.hg.vector[3:]-momentum[3:]) - (dAg @ v)[3:])
            Am = np.zeros((3,n)); Am[:,:nv] = Ag[3:]
            # Normalize by mass so a gain has the same meaning across robots.
            add(Am/self.total_mass,target/self.total_mass,g.w_angular_momentum)
        # Swing feet: WORLD-frame Cartesian tracking of the planned foot path.
        # Airborne the base is uncontrollable and ballistic, so the ONLY way to
        # land on time and on the spot is to place the feet relative to the
        # world despite the base error; the task therefore dominates joint
        # tracking in flight (w_swing_flight) and is a secondary task with
        # support (a late foot is driven to the ground).
        w_sw = g.w_swing_flight if not support else g.w_swing
        capture = np.zeros(2)
        if swing and g.capture_gain > 0.:
            m,d,rd = self.model,self.data,self.rdata
            pin.centerOfMass(m,d,q,v)
            com,cv = d.com[0].copy(),d.vcom[0].copy()
            pin.centerOfMass(m,rd,ref.q,ref.v)
            foot_z = np.mean([r['p'][2] for r in rf.values()])
            omega = np.sqrt(9.81/max(.1,float(rd.com[0][2]-foot_z)))
            capture = g.capture_gain * ((com-rd.com[0])[:2]+(cv-rd.vcom[0])[:2]/omega)
            capture = np.clip(capture,-g.capture_max,g.capture_max)

        def touchdown_depth(f):
            """Per-foot bias: blended in over the last part of the foot's own
            swing, held at the full value once the plan has put the foot down
            (a scheduled foot that is not touching yet), and ALSO for an
            established stance foot that has lost contact while the plan
            still loads it (planned normal force above `lammin`). (At the
            fence liftoff the planned force is pinned to zero, so nothing
            changes there.)"""
            landed = g.landed_depth if g.landed_depth >= 0.0 else g.touchdown_depth
            if g.touchdown_depth <= 0.0 and landed <= 0.0:
                return 0.0
            st, s = ref.foot_phase.get(f, (FOOT_LANDED, 1.0))
            if st == FOOT_LANDED:
                return landed
            if st == FOOT_SWING and s > g.touchdown_blend:
                if g.touchdown_depth <= 0.0:
                    return 0.0
                x = (s - g.touchdown_blend) / max(1.0 - g.touchdown_blend, 1e-9)
                return g.touchdown_depth * (3 * x * x - 2 * x ** 3)
            lam_f = ref.lam.get(f)
            loaded = (lam_f is not None and
                      (lam_f[2] >= max(1e-6,self.lammin-g.force_floor_tolerance)
                       if g.force_floor_tolerance > 0. else lam_f[2] > self.lammin))
            if f in ref.scheduled and loaded:
                return landed
            return 0.0
        for f in swing:
            k = self.dims[f]
            c, r = cur[f], rf[f]
            p_r = r["p"].copy()
            v_r, a_r = r["vel"].copy(), r["acc"].copy()
            if g.capture_gain > 0.:
                _, progress = ref.foot_phase.get(f,(FOOT_LANDED,1.))
                blend = np.clip((progress-.25)/.5,0.,1.)
                p_r[:2] += (3*blend**2-2*blend**3)*capture
            zs = self.surface_z_fn(p_r) if self.surface_z_fn is not None else None
            if g.sole_clearance and k == 6 and len(self.sole_corners.get(f, ())):
                low = p_r[2] + float(np.min((np.asarray(self.sole_corners[f]) @ r["R"].T)[:, 2]))
                surf = 0.0 if zs is None else float(zs)
                if low < surf:
                    p_r[2] += surf - low
            if g.consistent_swing:
                depth = touchdown_depth(f)
                depth_rate = depth_acc = 0.
                state, progress = ref.foot_phase.get(f, (FOOT_LANDED, 1.))
                duration = getattr(ref, 'phase_duration', 0.)
                if state == FOOT_SWING and g.touchdown_blend < progress < 1. and duration > 0.:
                    interval = duration * (1. - g.touchdown_blend)
                    x = (progress - g.touchdown_blend) / (1. - g.touchdown_blend)
                    depth_rate = g.touchdown_depth * (6*x - 6*x*x) / interval
                    depth_acc = g.touchdown_depth * (6 - 12*x) / interval**2
                p_r[2], v_r[2], a_r[2] = self.swing_height(
                    p_r[2], v_r[2], a_r[2], depth, depth_rate, depth_acc,
                    surface=zs, stone_depth=g.stone_touchdown_depth)
            elif zs is None:
                p_r[2] -= touchdown_depth(f)          # ground plane: the full bias (an early plastic impact)
            else:
                # The foot lands on a STONE that exists for it only as a top face
                # (rollout.update_stone_masks): the biased target is clamped to
                # `slab_tol` below the top, so the bias engages the surface instead
                # of sinking the foot into the masked slab, and a reference that
                # approaches from inside the slab (README §9) is lifted to the
                # surface instead of followed into it.
                p_r[2] = max(p_r[2] - touchdown_depth(f), zs - g.stone_touchdown_depth)
            err = self._pose_err(k, c["p"], c["R"], p_r, r["R"])
            target = a_r - g.kp_swing * err - g.kd_swing * (c["vel"] - v_r) - c["jdv"]
            Af = np.zeros((k, n))
            Af[:, :nv] = c["J"]
            add(Af, target, w_sw)
        # forces and torques
        Jst = np.vstack([cur[f]["J"] for f in support]) if support else np.zeros((0, nv))
        f_ref = np.concatenate([ref.lam.get(f, np.zeros(self.dims[f])) for f in support]) if support else np.zeros(0)
        if nf:
            Af = np.zeros((nf, n))
            Af[:, of:of + nf] = np.eye(nf)
            add(Af, f_ref, g.w_force / self.force_scale**2)
            As = np.zeros((nf, n))
            As[:, os_:os_+nf] = np.eye(nf)
            add(As, np.zeros(nf), g.w_contact)
        if ng:
            Ag = np.zeros((ng, n)); Ag[:, og:og+ng] = np.eye(ng)
            add(Ag, np.zeros(ng), g.joint_limit_slack_weight)
        T = np.zeros((self.nu, n))
        T[:, :nv] = M[self.act]
        if nf:
            T[:, of:of + nf] = -Jst[:, self.act].T
        c_tau = h[self.act]
        add(T / self.tau_max[:, None], (ref.u - c_tau) / self.tau_max, g.w_tau)

        Aall = np.vstack(rows)
        ball = np.concatenate(rhs)
        H = Aall.T @ Aall + g.ridge * np.eye(n)
        gv = -Aall.T @ ball

        # equalities
        eq_rows, eq_rhs = [], []
        Ab = np.zeros((len(self.dyn_base), n))
        Ab[:, :nv] = M[self.dyn_base]
        if nf:
            Ab[:, of:of + nf] = -Jst[:, self.dyn_base].T
        eq_rows.append(Ab)
        eq_rhs.append(-h[self.dyn_base])
        if self.lock.size:
            Al = np.zeros((len(self.lock), n))
            Al[np.arange(len(self.lock)), self.lock] = 1.0
            eq_rows.append(Al)
            eq_rhs.append(np.zeros(len(self.lock)))
        if nf:
            Ac = np.zeros((nf, n))
            Ac[:, :nv] = Jst
            Ac[:, os_:os_+nf] = -np.eye(nf)
            tgt = []
            offset = 0
            for f in support:
                k = self.dims[f]
                c = cur[f]
                p_a, R_a = cs.anchors[f]
                if k == 6 and g.level_stance:
                    # level_stance: the anchor is level, not the pose at first touch
                    p_a = p_a.copy()
                    if g.physical_stance:
                        R_a = np.eye(3)
                        top = self.surface_z_fn(p_a) if self.surface_z_fn is not None else None
                        p_a[2] = 0. if top is None else top
                    else:
                        R_a = rf[f]["R"]
                        p_a[2] = rf[f]["p"][2]
                err = self._pose_err(k, c["p"], c["R"], p_a, R_a)
                desired = -g.kp_contact * err - g.kd_contact * c["vel"] - c["jdv"]
                if g.measured_patch and f in patches:
                    P, drift = contact_projection(patches[f], c['vel'][3:])
                    Ac[offset:offset+k, :nv] = P @ c['J']
                    desired = P @ desired - drift
                tgt.append(desired)
                offset += k
            eq_rows.append(Ac)
            eq_rhs.append(np.concatenate(tgt))
        offset = of
        for f in support:
            if f in patches:
                Ap = np.zeros((6, n))
                Ap[:, offset:offset+6] = np.eye(6)
                Ap[:, patch_slices[f]] = -wrench_map(patches[f])
                eq_rows.append(Ap)
                eq_rhs.append(np.zeros(6))
            offset += self.dims[f]
        Aeq = np.vstack(eq_rows)
        beq = np.concatenate(eq_rhs)
        sc = np.maximum(np.linalg.norm(Aeq, axis=1), 1.0)
        Aeq, beq = Aeq / sc[:, None], beq / sc

        # inequalities  C x <= u
        C_rows, u_rows = [], []
        ang = 2 * np.pi * np.arange(g.friction_sides) / g.friction_sides
        j = of
        for f in support:
            k = self.dims[f]
            for th in ang:
                r = np.zeros(n)
                r[j] = np.cos(th)
                r[j + 1] = np.sin(th)
                r[j + 2] = -self.mu
                C_rows.append(r)
                u_rows.append(0.0)
            r = np.zeros(n)
            r[j + 2] = -1.0
            C_rows.append(r)
            fz_min = g.fz_min
            lam_f = ref.lam.get(f)
            if (g.stance_force_floor and lam_f is not None
                    and lam_f[2] >= max(1e-6, self.lammin - g.force_floor_tolerance)):
                fz_min = max(fz_min, self.lammin)   # the plan loads this foot: keep it loaded
            u_rows.append(-fz_min)
            if k == 6 and not g.measured_patch:
                cx, cy = self.cop.get(f, (0.05, 0.03))
                for mo, bound in ((4, cx), (3, cy), (5, self.gammaz)):
                    for sg in (-1.0, 1.0):
                        r = np.zeros(n)
                        r[j + mo] = sg
                        r[j + 2] = -bound
                        C_rows.append(r)
                        u_rows.append(0.0)
            j += k
        for f, points in patches.items():
            for j in range(patch_slices[f].start, patch_slices[f].stop, 3):
                for th in ang:
                    row = np.zeros(n)
                    # Inscribed polygon: every admitted corner force is inside
                    # the circular Coulomb cone, including between its facets.
                    row[j:j+3] = [np.cos(th), np.sin(th), -self.mu*np.cos(np.pi/g.friction_sides)]
                    C_rows.append(row); u_rows.append(0.)
                row = np.zeros(n); row[j+2] = -1.
                C_rows.append(row); u_rows.append(0.)
        Tn = T / self.tau_max[:, None]
        cn = c_tau / self.tau_max
        if g.joint_limit_rate > 0.:
            # Exponential position barrier: b'' + 2*w*b' + w^2*b >= 0
            # for each distance to a joint stop. No extra torque is allowed.
            rate = g.joint_limit_rate
            lo = self.model.lowerPositionLimit[self.act]
            hi = self.model.upperPositionLimit[self.act]
            margin = np.minimum(g.joint_limit_margin, .1 * (hi - lo))
            for j, i in enumerate(self.act):
                if np.isfinite(hi[j]) and hi[j] < 1e10:
                    row = np.zeros(n); row[i] = 1.
                    if ng: row[og+j] = -1.
                    C_rows.append(row)
                    u_rows.append(rate**2 * (hi[j] - margin[j] - q[i]) - 2*rate*v[i])
                if np.isfinite(lo[j]) and lo[j] > -1e10:
                    row = np.zeros(n); row[i] = -1.
                    if ng: row[og+j] = -1.
                    C_rows.append(row)
                    u_rows.append(rate**2 * (q[i] - lo[j] - margin[j]) + 2*rate*v[i])
                if ng:
                    row = np.zeros(n); row[og+j] = -1.
                    C_rows.append(row); u_rows.append(0.)
        C_rows.extend(list(Tn))
        u_rows.extend(list(1.0 - cn))
        C_rows.extend(list(-Tn))
        u_rows.extend(list(1.0 + cn))
        Cin = np.vstack(C_rows)
        uin = np.asarray(u_rows)
        lin = np.full(len(uin), -1e20)

        key = tuple(support)
        qp = self._qps.get(key)
        if qp is None or qp.model.dim != n:
            qp = proxsuite.proxqp.dense.QP(n, Aeq.shape[0], Cin.shape[0])
            qp.settings.eps_abs = g.eps_abs
            qp.settings.eps_rel = 0.0
            qp.settings.max_iter = g.max_iter
            qp.settings.eps_primal_inf = 1e-14
            qp.settings.eps_dual_inf = 1e-14
            qp.settings.verbose = False
            qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            qp.init(H, gv, Aeq, beq, Cin, lin, uin)
            self._qps[key] = qp
        else:
            qp.update(H, gv, Aeq, beq, Cin, lin, uin)
        qp.solve()
        self.n_solves += 1
        self._last_qp = (H, gv, Aeq, beq, Cin, lin, uin)
        x = np.asarray(qp.results.x)
        ok = (qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
              and np.all(np.isfinite(x)))
        if not ok:
            # retry cold: a fresh QP object without the warm start
            qp = self._qps[key] = proxsuite.proxqp.dense.QP(n, Aeq.shape[0], Cin.shape[0])
            qp.settings.eps_abs = g.eps_abs
            qp.settings.eps_rel = 0.0
            qp.settings.max_iter = g.max_iter
            qp.settings.eps_primal_inf = 1e-14
            qp.settings.eps_dual_inf = 1e-14
            qp.settings.verbose = False
            qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            qp.init(H, gv, Aeq, beq, Cin, lin, uin)
            qp.solve()
            self.n_cold_retries += 1
            x = np.asarray(qp.results.x)
            ok = (qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
                  and np.all(np.isfinite(x)))
        if not ok:
            self.n_fallback += 1
            tau = ref.u + np.minimum(self.kp[self.act], 120.0) * qe[self.act] \
                + np.minimum(self.kd[self.act], 8.0) * (ref.v - v)[self.act]
            tau = np.clip(tau, -self.tau_max, self.tau_max)
            self.last_info = dict(ok=False, status=str(qp.results.info.status))
            return tau, self.last_info
        tau = T @ x + c_tau
        tau += self.tau_max * (g.kp_motor * qe[self.act]
                              + g.kd_motor * (ref.v - v)[self.act])
        tau_c = np.clip(tau, -self.tau_max, self.tau_max)
        if g.torque_filter > 0 and self._previous_tau is not None:
            alpha = self.control_dt / (self.control_dt + g.torque_filter)
            tau_c = self._previous_tau + alpha * (tau_c - self._previous_tau)
        self._previous_tau = tau_c.copy()
        forces = {}
        j = 0
        for f in support:
            k = self.dims[f]
            forces[f] = x[of + j:of + j + k].copy()
            j += k
        self.last_info = dict(ok=True, iters=int(qp.results.info.iter), a=x[:nv].copy(),
                              forces=forces, slack=float(np.linalg.norm(x[os_:os_+nf])) if nf else 0.0,
                              joint_limit_slack=float(np.max(x[og:og+ng])) if ng else 0.,
                              tau_ratio=float(np.max(np.abs(tau) / self.tau_max)),
                              clipped=bool(np.any(np.abs(tau) > self.tau_max + 1e-9)))
        return tau_c, self.last_info
