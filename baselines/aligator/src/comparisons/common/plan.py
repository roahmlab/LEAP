"""Solver-neutral transcription plan for the fence / terrain benchmarks:
the knot grid, per-knot contact sets and constraint markers, the leap seed
evaluated at the knots, and pinocchio RigidConstraintModels per stage (used
both by forward-dynamics transcriptions and by the independent judges).

This mirrors aligator_bench/builders.py's schedule exactly (same dt logic,
same boundary-stage constraint placement) but depends only on pinocchio, so
the crocoddyl / acados ports and the evaluators share it.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pinocchio as pin

from . import spec_fence, spec_terrain
from .robot import Robot


@dataclasses.dataclass
class Knot:
    dt: float
    t: float
    active: list                  # stance contact names
    targets: dict                 # contact -> world target (active feet)
    force_pins: tuple = ()        # feet whose lam == 0 at this knot
    floor_skip: tuple = ()        # feet whose lammin floor row is dropped
    # leap's stancePin / contactPosAll: EVERY stance foot is pinned on its
    # commanded placement (+ a level sole for dim 6) at EVERY node it is in
    # contact -- not only where it touches down. In a forward-dynamics port
    # the contact model only stabilizes this asymptotically (Baumgarte), so
    # the rows have to be stated explicitly or the problem is a weaker one.
    stance_pins: dict = None      # contact -> target (position + sole pin)
    touchdown_pins: dict = None   # contact -> target (placement at touchdown)
    touchdown_vel: str = None     # "zero" | "xy" | "vz": which rows to impose
    land_vz: float = 0.0          # |foot vz| cap used by "xy" / "vz"
    start_pins: dict = None       # contact -> target (position pin only)
    # Feet that carry a no-slip row J(q) v == 0 at this knot WITHOUT being in
    # the stage's contact set: the knot AT a liftoff instant belongs to the
    # next (flight/swing) mode, but leap's stance mode still states its rows
    # at that shared boundary node (vel_* at AllNodes incl. LastNode).
    noslip: tuple = ()
    vz_lb: float = None           # v_base_z >= vz_lb at this knot
    clear: tuple = None           # (foot, z_lb) swing clearance at this knot
    flight_monitors: bool = False # fence: feet included in ground keep-out
    monitor_lb: float = 0.0       # monitor-frame height floor at this knot


@dataclasses.dataclass
class Plan:
    bench: str
    spec: object
    robot: Robot
    knots: list
    ts: np.ndarray                # knot times, len N+1
    dts: list
    stage_contacts: list
    stage_rcm: list               # pinocchio RCMs per stage (shared objects)
    xs_init: list
    us_init: list
    x0: np.ndarray
    x_goal: np.ndarray = None     # fence strictrest terminal state
    terminal_foot_pins: dict = None   # terrain: contact -> target
    terminal_v_zero: bool = False
    q0_free: bool = False         # terrain: leap does NOT pin q(0)
    fd_forces: bool = False       # singular contact force split
    monitors: list = None         # [(frame_id, name)]
    nsteps: int = 0
    # What the port imposes at a touchdown instant vs what leap states there.
    # They differ whenever leap allows a plastic impact (fence: always;
    # terrain: hopper2d `zerolandxy`, biped5 `landvz` only), because no
    # shooting port here can host a state jump. Recorded in every result file
    # so the deviation is machine-visible, not only prose.
    touchdown_mode: str = "zero"        # "zero" | "xy" | "vz"
    leap_touchdown_mode: str = "zero"

    @property
    def impulse_free(self) -> bool:
        return self.touchdown_mode == "zero"

    @property
    def touchdown_deviates(self) -> bool:
        return self.touchdown_mode != self.leap_touchdown_mode


def make_rcms(robot: Robot, active, targets, kp=100.0):
    """pinocchio RigidConstraintModels at the target footholds (identity sole
    rotation for dim-6), Baumgarte (kp, 2 sqrt(kp))."""
    rcms = []
    for c in robot.contacts:
        if c.name not in active:
            continue
        fr = robot.model.frames[c.frame_id]
        ctype = (pin.ContactType.CONTACT_6D if c.dim == 6
                 else pin.ContactType.CONTACT_3D)
        ref = pin.SE3(np.eye(3), np.asarray(targets[c.name], dtype=float))
        cm = pin.RigidConstraintModel(ctype, robot.model, fr.parentJoint,
                                      fr.placement, 0, ref,
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        cm.name = c.name
        bp = cm.m_baumgarte_parameters
        bp.Kp = kp
        bp.Kd = 2.0 * np.sqrt(kp)
        cm.setBaumgarteCorrectorParameters(bp)
        rcms.append(cm)
    return rcms


def contacts_singular(robot: Robot, q) -> bool:
    d = robot.data()
    pin.computeJointJacobians(robot.model, d, q)
    pin.updateFramePlacements(robot.model, d)
    Js = [pin.getFrameJacobian(robot.model, d, c.frame_id,
                               pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
          [:(6 if c.dim == 6 else 3)] for c in robot.contacts]
    sv = np.linalg.svd(np.vstack(Js), compute_uv=False)
    # RELATIVE threshold.
    return bool(sv.min() < 1e-2 * sv.max())


def _nk(T, dt_target, dt_round="round"):
    """Knots per phase. "round" (default) rounds
    T/dt, so a phase's step can exceed dt_target by up to half a knot;
    "ceil" guarantees every step <= dt_target (used when the knot grid must
    itself be the judge's dt=1e-2 grid, so no interior samples exist)."""
    if dt_round == "ceil":
        return max(3, int(np.ceil(T / dt_target - 1e-9)))
    return max(3, int(round(T / dt_target)))


def _monitors(robot: Robot):
    import yaml
    names = yaml.safe_load(open(robot.yaml_path)).get("monitor_frames", [])
    return [(robot.frame_id(n), n) for n in names]


def fence_plan(spec: spec_fence.FenceSpec, dt_target=0.01, kp=400.0,
               impulse_free=True, dt_round="round") -> Plan:
    r = spec.robot
    all_names = [c.name for c in r.contacts]
    phases = [("push", spec.t_stance), ("air", spec.t_air),
              ("land", spec.t_land)]
    nks = [_nk(T, dt_target, dt_round) for _, T in phases]
    rcms0 = make_rcms(r, all_names, spec.feet0, kp)
    rcms2 = make_rcms(r, all_names, spec.feet2, kp)

    knots, dts, stage_contacts, stage_rcm = [], [], [], []
    times = [0.0]
    for pi, (pname, T) in enumerate(phases):
        nk = nks[pi]
        dt = T / nk
        for k in range(nk):
            kn = Knot(dt=dt, t=times[-1], active=[], targets={})
            if pname == "push":
                kn.active = list(all_names)
                kn.targets = dict(spec.feet0)
                rcms = rcms0
                # contactPosAll. Knot 0 carries the full-state pin already
                # (strictRest), so its placement rows would be 0 == 0.
                if k > 0:
                    kn.stance_pins = dict(spec.feet0)
                if k == nk - 1:
                    kn.force_pins = tuple(all_names)
                    kn.floor_skip = tuple(all_names)
            elif pname == "air":
                rcms = []
                kn.flight_monitors = True
                # ineq_clear_air: in flight every monitor frame stays above
                # footClear (fence_jump.hpp; 0 in every shipped problem).
                kn.monitor_lb = max(0.0, spec.footclear)
                if k == 0:
                    kn.vz_lb = spec.vz_liftoff
                    # This knot IS leap's mode-0 LastNode (t = T0): the
                    # stance placement and no-slip rows still hold there
                    # (fence_jump_v2.cpp attaches stanceFeet0 + vel_* at
                    # AllNodes of the push mode, boundary included).
                    kn.stance_pins = dict(spec.feet0)
                    kn.noslip = tuple(all_names)
            else:
                kn.active = list(all_names)
                kn.targets = dict(spec.feet2)
                rcms = rcms2
                if k == 0:
                    kn.touchdown_pins = dict(spec.feet2)
                    # leap: ineq_land_vz (foot vz >= -landVzMax) at the flight's
                    # last node + a PlasticImpactReset. No shooting port here
                    # can host a state jump, so the touchdown is impulse-free
                    # (a strictly STRONGER condition); "vz" keeps only leap's
                    # own cap and lets the contact model absorb the rest.
                    kn.touchdown_vel = "zero" if impulse_free else "vz"
                    kn.land_vz = spec.landvz
                else:
                    kn.stance_pins = dict(spec.feet2)
            knots.append(kn)
            dts.append(dt)
            stage_contacts.append(list(kn.active))
            stage_rcm.append(rcms)
            times.append(times[-1] + dt)

    ts = np.array(times)
    xs_init, us_init = [], []
    for k, t in enumerate(ts):
        q, v, a = spec_fence.seed_state(spec, t)
        xs_init.append(np.concatenate([q, v]))
        if k < len(knots):
            # leap's seed zeroes the LAST force control point of the push-off
            # (fence_jump_v2.cpp finalizeForce on the seed), and the ports pin
            # lam == 0 at their liftoff knot -- so the seeded force there is
            # zero and u is the full inverse-dynamics torque, not the
            # force-compensated one.
            in_stance = (len(stage_contacts[k]) > 0
                         and not knots[k].force_pins)
            _, u = spec_fence.seed_force_torque(spec, q, v, a, in_stance)
            us_init.append(u)
    x0 = np.concatenate([spec.q_hold0, np.zeros(r.nv)])
    x_goal = np.concatenate([spec.q_hold2, np.zeros(r.nv)])
    return Plan(bench="fence", spec=spec, robot=r, knots=knots, ts=ts,
                dts=dts, stage_contacts=stage_contacts, stage_rcm=stage_rcm,
                xs_init=xs_init, us_init=us_init, x0=x0, x_goal=x_goal,
                fd_forces=contacts_singular(r, spec.q_hold0),
                monitors=_monitors(r), nsteps=len(knots),
                touchdown_mode="zero" if impulse_free else "vz",
                leap_touchdown_mode="vz")


def terrain_plan(spec: spec_terrain.TerrainSpec, dt_target=0.02,
                 kp=100.0, impulse_free=True, dt_round="round") -> Plan:
    r = spec.robot
    knots, dts, stage_contacts, stage_rcm = [], [], [], []
    times = [0.0]
    # Which touchdown rows leap actually states (terrain_walk.cpp:770-793).
    leap_td = "zero" if spec.zeroland else ("xy" if spec.zerolandxy else "vz")
    td_mode = "zero" if impulse_free else leap_td
    for mi, md in enumerate(spec.modes):
        nk = _nk(md.T, dt_target, dt_round)
        dt = md.T / nk
        rcms = make_rcms(r, md.active, md.targets, kp)
        # leap states ineq_swing_z at interp taus {-0.5, 0, +0.5} of EVERY
        # sub-segment, and the h-refinement gives each mode `split`
        # sub-segments -- so 3*split points per swing, at the fractions
        # (i + (tau+1)/2)/split, placed on the nearest interior knot. The
        # row exists whenever clearz >= 0 (the C++ gate; -1 = off).
        clear_ks = set()
        if md.kind == "swing" and md.clearz >= 0.0:
            split = getattr(spec, "split", 1)
            fracs = [(i + (tau + 1.0) / 2.0) / split
                     for i in range(split) for tau in (-0.5, 0.0, 0.5)]
            clear_ks = {min(nk - 1, max(1, int(np.floor(f * nk + 0.5))))
                        for f in fracs}
        for k in range(nk):
            kn = Knot(dt=dt, t=times[-1], active=list(md.active),
                      targets=dict(md.targets), monitor_lb=spec.monitorz)
            # stancePin: every foot in contact, every node.
            kn.stance_pins = dict(md.targets)
            if md.kind == "swing" and k == 0 and mi > 0:
                # leap's support mode still owns this shared boundary node
                # (its AllNodes rows include the last node = this instant):
                # the lifting foot keeps its placement pin and no-slip row.
                prev = spec.modes[mi - 1]
                if md.swing_foot in prev.targets:
                    kn.stance_pins[md.swing_foot] = prev.targets[md.swing_foot]
                    kn.noslip = (md.swing_foot,)
            if md.kind == "support":
                pins, skip = [], []
                if k == 0 and md.touchdown_feet:
                    kn.touchdown_pins = {c: md.targets[c]
                                         for c in md.touchdown_feet}
                    kn.touchdown_vel = td_mode
                    kn.land_vz = spec.landvz
                    # the touchdown pin already states these feet's placement
                    # rows -- a second identical stance-pin row would make
                    # the equality block rank-deficient
                    for c in md.touchdown_feet:
                        kn.stance_pins.pop(c, None)
                if k == 0 and md.land_feet:   # landRamp only
                    pins += list(md.land_feet)
                    skip += list(md.land_feet)
                if k == nk - 1 and md.lift_feet:
                    pins += list(md.lift_feet)
                    skip += list(md.lift_feet)
                kn.force_pins = tuple(pins)
                kn.floor_skip = tuple(skip)
            else:
                if k in clear_ks:
                    kn.clear = (md.swing_foot, md.clearz)
            if mi == 0 and k == 0:
                kn.start_pins = dict(md.targets)
                kn.stance_pins = None      # the start pins already say this
            knots.append(kn)
            dts.append(dt)
            stage_contacts.append(list(md.active))
            stage_rcm.append(rcms)
            times.append(times[-1] + dt)

    ts = np.array(times)
    n_modes = len(spec.modes)
    mode_starts = np.concatenate([[0.0], np.cumsum([m.T for m in spec.modes])])
    xs_init, us_init = [], []
    for k, t in enumerate(ts):
        # knot times are accumulated sums, mode_starts a cumsum: at an exact
        # mode boundary they can differ by roundoff, so nudge by 1e-9 to
        # resolve the boundary knot to the mode that starts there.
        mi = min(np.searchsorted(mode_starts, t + 1e-9, side="right") - 1,
                 n_modes - 1)
        tloc = max(t - mode_starts[mi], 0.0)
        q, v, a = spec_terrain.seed_state(spec, mi, tloc)
        xs_init.append(np.concatenate([q, v]))
        if k < len(knots):
            _, u = spec_terrain.seed_force_torque(spec, mi, q, v, a)
            us_init.append(u)
    x0 = np.concatenate([spec.q_event[0], np.zeros(r.nv)])
    return Plan(bench="terrain", spec=spec, robot=r, knots=knots, ts=ts,
                dts=dts, stage_contacts=stage_contacts, stage_rcm=stage_rcm,
                xs_init=xs_init, us_init=us_init, x0=x0,
                terminal_foot_pins=dict(spec.stance_after[-1]),
                terminal_v_zero=True, q0_free=True,
                fd_forces=contacts_singular(r, spec.q_event[0]),
                monitors=_monitors(r), nsteps=len(knots),
                touchdown_mode=td_mode, leap_touchdown_mode=leap_td)
