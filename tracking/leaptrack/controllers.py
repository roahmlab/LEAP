"""Controller and rollout configuration of the tracking study, per TASK and per
ROBOT -- never per method or per solution.

ONE controller architecture: the whole-body inverse-dynamics QP
`leaptrack.wbc.WholeBodyQP` (ProxQP; a_des = a_ref + Kp e + Kd e_dot on every
DOF, planned contact forces / torques as regularizers, unactuated dynamics +
anchored support feet as equalities, friction pyramid / CoP box / torque box
as inequalities) with the measured-contact `ContactPolicy` and a
touchdown-aware reference clock (`leaptrack.reference.TouchdownClock`). The
fence and the terrain task share the architecture; what differs between the
tasks is only which optional terms of that QP are switched on (`TASK_GAINS`),
and per robot only numbers (`ROBOT_GAINS[task][robot]`).

Every number here is recorded in `metrics.json` / `evaluation.json` of each
rollout and listed in the generated REPORT.md.
"""
from __future__ import annotations

from .wbc import WbcGains

TASKS = ("fence", "terrain")
# The robots of each suite = the instances LEAP solves AND tracks. The terrain
# hopper is the flat-foot variant hopper2d_foot.
TASK_ROBOTS = {
    "fence": ("hopper2d", "go2", "h1", "g1_29dof", "talos"),
    "terrain": ("go2", "h1", "g1_29dof", "talos", "hopper2d_foot"),
}
# Every robot a CLI may name (the union, plus the terrain point-foot hopper2d).
ALL_ROBOTS = ("hopper2d", "go2", "h1", "g1_29dof", "talos", "hopper2d_foot")
ROBOTS = ("hopper2d", "go2", "h1", "g1_29dof", "talos")     # the benchmark's own robot set

# The task-level switches (the same for every robot of the task). Everything
# not listed is `wbc.WbcGains()`'s default.
TASK_GAINS = {
    # Rest-to-rest leap. All feet swing together and the base is ballistic in
    # flight (base tracking off, world-frame foot placement dominates).
    #   * `coupled_wrench`: each flat-foot wrench is realized by four unilateral
    #     corner forces, each inside the friction cone, instead of a wrench in a
    #     CoP box. A sole cannot pull.
    #     (Point feet are unaffected: the switch only touches dim-6 contacts.)
    #   * `touchdown_depth` 0: the leap plans land WITH speed (a plastic impact,
    #     |vz| <= landvz), so the swing-foot reference is NOT lowered before
    #     touchdown. A late touchdown is handled by the clock.
    #   * `contact_grace` 0.02 s and `w_force` 0.2: the push-off unloads a
    #     diagonal pair of feet to the problem's normal-force floor (5 N on go2,
    #     2 % of body weight).
    #     The grace keeps a scheduled foot in the support set for 20 ms after it
    #     stops touching, and the heavier force-tracking weight keeps the QP's
    #     load split on the plan's instead of re-solving it around the missing
    #     foot.
    "fence": dict(coupled_wrench=True, touchdown_depth=0.0, contact_grace=0.02, w_force=0.2),
    # Stepping-stone walk: one foot swings at a time, the touchdown velocity is
    # zero (`zeroland`) for every robot but the point-foot hopper2d.
    #   * `coupled_wrench` again.
    #   * The plan skims the target stone and may enter footprints BELOW the
    #     top face (the benchmark keeps only the collision capsules clear of
    #     the stones, with a 1 cm slack, so the foot frame may pass through its
    #     own stone). The simulator models stones as top faces for the feet and
    #     stops doing so once a foot is more than 2 mm below the top.
    #     The swing target is therefore biased 1 cm below the PLAN and
    #     clamped 1 cm below the landing surface, with the surface itself
    #     reported by `rollout.run` as soon as the foot is within 5 cm of a
    #     footprint and RAISED to the top face whenever the foot is already
    #     below it.
    "terrain": dict(coupled_wrench=True, touchdown_depth=0.01, stone_touchdown_depth=0.01),
}

# Per-robot numbers (allowed by the task statement; the architecture is the
# task's). Everything not named here is `wbc.WbcGains()`'s default.
_SWING_STIFF = dict(kp_swing=1500.0, kd_swing=80.0)   # Cartesian swing-foot PD
_BASE_2X = dict(kp_base_pos=300.0, kd_base_pos=35.0, kp_base_rot=400.0, kd_base_rot=40.0)
ROBOT_GAINS = {
    "fence": {
        # g1: a stiff Cartesian swing PD and a flight joint weight of 0.3 instead
        # of 0.02.
        "g1_29dof": dict(_SWING_STIFF, flight_joint_scale=0.3),
        # talos: the stiff swing PD plus the stiffer base PD.
        "talos": dict(_SWING_STIFF, **_BASE_2X),
        # h1 keeps NO contact grace: its push-off never unloads a foot to the
        # normal-force floor (it stands on two large soles, not four points).
        "h1": dict(contact_grace=0.0),
        # hopper2d: capture-point foot placement in flight (the foothold is moved
        # by the centre-of-mass position and velocity error, at most 5 cm from
        # the world target).
        "hopper2d": dict(capture_gain=1.0, capture_max=0.05),
    },
    "terrain": {
        # h1: the stiff Cartesian swing PD.
        "h1": dict(_SWING_STIFF),
    },
}

# Cases judged AT THE END OF THE PLAN (t = T) instead of at the end of the hold,
# with their own final-state tolerances (everything not named keeps
# report.CRITERIA). hopper2d's fence rest state is an upright stand on a point
# foot (an inverted triple pendulum with two actuators), so it is judged at T:
# the run has to reach the planned final state with (A) the error inside the
# strict transient bound over the whole plan and (B) the final-state test at T,
# with a speed tolerance of 0.10 m/s and a foot tolerance of 0.04 m (the capture
# placement above moves the foothold up to 5 cm on purpose). This is a property
# of the ROBOT's rest state, so it applies to any method's hopper2d fence solution.
JUDGE_AT_T = {
    "fence": {"hopper2d": dict(final_speed=0.10, final_foot=0.04)},
}

# Simulation / rollout options per task (not controller gains).
TASK_OPTIONS = {
    # fence: the feet arrive with speed, so a foot touching late in the flight
    # IS the landing -> the clock jumps to the touchdown state (early_jump).
    "fence": dict(control_dt=1e-3, hold=1.5, settle=0.05, time_warp=True, early_jump=True,
                  persistent_stones=False),
    # terrain: the plans skim the target stone with zero touchdown velocity, so
    # an early touch is not a landing event worth a clock jump: the foot is
    # anchored where it touched and the clock keeps running.
    "terrain": dict(control_dt=1e-3, hold=1.5, settle=0.05, time_warp=True, early_jump=False,
                    persistent_stones=True),
}

# Per-robot rollout options (same idea as ROBOT_GAINS).
ROBOT_OPTIONS = {
    "fence": {},
    # h1's terrain plan lands with the base still descending, so a foot touching
    # early IS the landing: the clock jumps to the touchdown state instead of
    # waiting.
    "terrain": {"h1": dict(early_jump=True),
                # hopper2d_foot: a ballistic hop lands with the base descending, so
                # a foot touching early IS the landing: the clock jumps to the
                # touchdown state instead of waiting.
                "hopper2d_foot": dict(early_jump=True)},
}


# ------------------------------------------------------------------ profiles
# Two configurations of the controller. Select with the environment variable
# LEAPTRACK_PROFILE (inherited by every subprocess the scripts start) or the
# `profile` argument of `gains_for`; every rollout records the one used
# (`gains.config_profile`).
#   "impact_aware" (default): the tables above plus
#     * fence, every robot: `impact_projection` as the mode of the impact window (for
#       `impact_window` seconds after a measured touchdown the velocity error is
#       projected onto the motions a contact impulse at the landing feet cannot
#       change); it acts only where `impact_window` > 0;
#     * fence h1: `impact_window` 0.03 s, `w_com` 20, the swing PD `_SWING_STIFF` and
#       `flight_joint_scale` 0.3;
#     * fence talos: `impact_window` 0.03 s, `w_com` 20.
#     "wp4" is accepted as another name.
#   "published": the tables above alone.
PROFILE_ENV = "LEAPTRACK_PROFILE"
PROFILES = ("impact_aware", "published")
DEFAULT_PROFILE = "impact_aware"
PROFILE_ALIASES = {"wp4": "impact_aware"}
IMPACT_AWARE_TASK_GAINS = {
    "fence": dict(impact_projection=True),
}
IMPACT_AWARE_ROBOT_GAINS = {
    "fence": {
        "h1": dict(_SWING_STIFF, flight_joint_scale=0.3, w_com=20.0, impact_window=0.03),
        "talos": dict(w_com=20.0, impact_window=0.03),
    },
}


def active_profile(profile: str | None = None) -> str:
    import os
    p = (profile or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE).strip().lower()
    p = PROFILE_ALIASES.get(p, p)
    if p not in PROFILES:
        raise ValueError(f"unknown controller profile {p!r} (one of {PROFILES}; env {PROFILE_ENV})")
    return p


def gains_for(task: str, robot: str, overrides: dict | None = None, profile: str | None = None) -> WbcGains:
    prof = active_profile(profile)
    kw = dict(TASK_GAINS[task])
    kw.update(ROBOT_GAINS.get(task, {}).get(robot, {}))
    if prof == "impact_aware":
        kw.update(IMPACT_AWARE_TASK_GAINS.get(task, {}))
        kw.update(IMPACT_AWARE_ROBOT_GAINS.get(task, {}).get(robot, {}))
    kw["config_profile"] = prof
    kw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return WbcGains(**kw)


def options_for(task: str, overrides: dict | None = None, robot: str | None = None) -> dict:
    kw = dict(TASK_OPTIONS[task])
    if robot is not None:
        kw.update(ROBOT_OPTIONS.get(task, {}).get(robot, {}))
    kw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return kw
