"""Parse a fence problem yaml into a solver-agnostic spec.

Mirrors the C++ fence builder's problem statement (poses from angles, targets).
Only the PROBLEM statement is kept - solver pacing keys are ignored.

Every `problem:` key is CLASSIFIED (see _MODELLED / _TRANSCRIPTION /
_MUST_EQUAL below) and an unclassified key, or a key the port cannot honour
set away from the value the port assumes, is a hard error. Silently
defaulting a knob the C++ builder defaults differently is exactly how the
two problem statements drift apart.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from common.robot import Robot, load_robot

GRAVITY = 9.81  # fence_jump_v2.cpp:34 hardcodes kGravity

# Keys whose VALUE this port reads and enforces.
_MODELLED = {
    "impactcone", "type", "tstance", "tair", "tstanceland", "hop", "step", "mu", "friceps",
    "lammin", "landvz", "copx", "copy", "gammaz", "footclear", "ulim", "vlim",
    "qbox", "attmax", "planarlock", "poses", "selfcolmargin", "selfcolpairs",
}
# Keys that describe leap's SPECTRAL TRANSCRIPTION only: a shooting port has
# no analogue (its knot grid replaces them). Ignored on purpose.
_TRANSCRIPTION = {
    "noslipnq", "selfcolinterp", "lsqmodemask",
}
# Keys the port cannot express and therefore ASSUMES. Any other value is a
# different problem, so refuse rather than silently mis-model it. Values are
# what the ports were written against.
_MUST_EQUAL = {
    "stanceacc": False,      # no additional stance-acceleration rows
    "mineffort": True,       # builder default; restoration explicitly disables effort
    "pinliftoff": True,      # liftoff force pin
    "holdlanding": False,
    "landingxslack": -1.0,
    "posepathslack": -1.0,
    "strictrest": True,      # x0 / terminal full-state pins
    "ownership": True,       # row-ownership: no effect on the feasible set
    "liftofffull": True,     # full-wrench liftoff pin
    "contactposall": True,   # stance foot placement pins at every stance knot
    "landfootpin": True,     # touchdown placement pins
    "ubern": True,           # Bernstein-certified torque box -> 0.995 * ulim
    "restbox": 0.0,
    "seedmode": 1,           # the piecewise seed replicated in seed_state()
    "seedapex": -1.0,        # seed flight apex override (seed_state uses hop)
    "staticseed": False,
}
# What the C++ BUILDER uses when the key is absent (fence_jump.hpp /
# fence_jump_v2.hpp). Where this differs from _MUST_EQUAL, an ABSENT key
# already means a different problem than the port assumes, so the absent
# case must be checked too -- contactposall and ubern default to false in
# C++ but every shipped yaml (and the port) sets/assumes true.
_CPP_DEFAULT = dict(_MUST_EQUAL, contactposall=False, ubern=False)


def _classify_problem_keys(p: dict, path) -> None:
    for k in set(p) | set(_MUST_EQUAL):
        if k in _MODELLED or k in _TRANSCRIPTION:
            continue
        if k in _MUST_EQUAL:
            v = p.get(k, _CPP_DEFAULT[k])
            want = _MUST_EQUAL[k]
            if isinstance(want, bool):
                ok = bool(v) == want
            else:
                ok = float(v) == float(want)
            if not ok:
                how = f"= {p[k]!r}" if k in p else \
                    f"is absent (C++ default {_CPP_DEFAULT[k]!r})"
                raise ValueError(
                    f"{path}: problem.{k} {how}, but the comparison ports "
                    f"assume {want!r} and cannot express the alternative. "
                    "Extend the ports (or set the key) before comparing.")
            continue
        raise ValueError(
            f"{path}: unknown problem key '{k}'. Classify it in "
            "comparisons/common/spec_fence.py (_MODELLED / _TRANSCRIPTION / "
            "_MUST_EQUAL) so it cannot silently diverge from the C++ builder.")


# obstacles: subkeys the port models (or safely ignores); anything else --
# `rounded` (a different residual form), `spheres` (a whole family),
# `capsules`/`capsule_margins` (a restricted/reweighted enforcement set) --
# is refused rather than silently dropped. `interp` is leap's off-node
# enforcement-point list: transcription, no knot-grid analogue.
_OBS_KEYS = {"margin", "boxes", "interp"}


def _check_obstacles(obs: dict, path) -> None:
    bad = set(obs) - _OBS_KEYS
    if bad:
        raise ValueError(
            f"{path}: obstacles keys {sorted(bad)} are not ported "
            "(comparisons/common/spec_fence.py _OBS_KEYS). rounded/spheres/"
            "capsules/capsule_margins change the stated constraint set; "
            "extend the ports before comparing.")


@dataclasses.dataclass
class FenceSpec:
    name: str
    robot: Robot
    # phases
    t_stance: float
    t_air: float
    t_land: float
    # physics / constraint params
    hop: float
    step: np.ndarray          # (2,) landing displacement
    mu: float
    friceps: float
    lammin: float
    landvz: float
    ulim: float
    vlim: float
    qbox: float
    attmax: float
    footclear: float          # flight-only monitor-frame height floor
    copx: float
    copy: float
    gammaz: float
    planarlock_idx: list      # q indices pinned to 0 at every node
    selfcol_margin: float
    selfcol_pairs: list       # [(name_a, name_b, dsafe)]
    obstacle_margin: float
    obstacle_boxes: list      # [(center(3,), half(3,))]
    # poses and derived targets
    q_squat: np.ndarray       # base at origin
    q_stand: np.ndarray
    q_hold0: np.ndarray       # squat @ (0,0)
    q_hold2: np.ndarray       # stand @ step
    feet0: dict               # contact name -> world pos in push stance
    feet2: dict               # contact name -> world pos in landing stance
    vz_liftoff: float         # sqrt(2 g hop)


def _pose_from_angles(robot: Robot, poses: list, key: str) -> np.ndarray:
    m = robot.model
    q = np.zeros(m.nq)
    for p in poses:
        jid = m.getJointId(p["joint"])
        q[m.joints[jid].idx_q] = p[key]
    d = robot.data()
    pin.framesForwardKinematics(m, d, q)
    zs = [d.oMf[c.frame_id].translation[2] for c in robot.contacts]
    assert max(zs) - min(zs) < 1e-5, f"{key} pose feet not level: {zs}"
    q[2] = -min(zs)
    return q


def _foot_targets(robot: Robot, q: np.ndarray) -> dict:
    d = robot.data()
    pin.framesForwardKinematics(robot.model, d, q)
    return {c.name: np.array(d.oMf[c.frame_id].translation)
            for c in robot.contacts}


def load_fence_spec(yaml_path) -> FenceSpec:
    yaml_path = Path(yaml_path).resolve()
    doc = yaml.safe_load(open(yaml_path))
    robot = load_robot((yaml_path.parent / doc["robot"]).resolve())
    p = doc["problem"]
    if p.get("impactcone") is not True:
        raise ValueError("This benchmark requires problem.impactcone: true")
    assert p["type"] == "jump_fence_v2", p["type"]
    _classify_problem_keys(p, yaml_path)

    t_stance = float(p["tstance"])
    t_land = float(p.get("tstanceland", 0.0)) or t_stance
    q_squat = _pose_from_angles(robot, p["poses"], "squat")
    q_stand = _pose_from_angles(robot, p["poses"], "stand")
    step = np.asarray(p["step"], dtype=float)
    q_hold0 = q_squat.copy()
    q_hold2 = q_stand.copy()
    q_hold2[0], q_hold2[1] = step

    m = robot.model
    lock_idx = [m.joints[m.getJointId(n)].idx_q for n in p.get("planarlock", [])]

    pairs = []
    margin = float(p.get("selfcolmargin", 0.0))
    for pr in p.get("selfcolpairs", []):
        ra = robot.capsules[pr["a"]].radius
        rb = robot.capsules[pr["b"]].radius
        # a negative per-pair margin is the C++ "use the default" sentinel
        mval = float(pr.get("margin", -1.0))
        pairs.append((pr["a"], pr["b"],
                      ra + rb + (margin if mval < 0.0 else mval)))

    obs = doc.get("obstacles", {})
    _check_obstacles(obs, yaml_path)
    boxes = [(np.asarray(b["center"], dtype=float),
              0.5 * np.asarray(b["size"], dtype=float))
             for b in obs.get("boxes", [])]
    for c, h in boxes:
        # geometry.separating_face_certificate hardcodes the -z face drop;
        # leap drops it only when the bottom is unreachable (bottom - dsafe
        # <= 0, obstacles.hpp bottomFaceUnreachable). Every shipped fence
        # box sits on the ground; refuse a floating one rather than model
        # it with the wrong face set.
        if c[2] - h[2] > 0.0:
            raise ValueError(f"{yaml_path}: floating obstacle box (bottom at "
                             f"z={c[2] - h[2]:.3f} > 0) -- the ported "
                             "certificate always drops the -z face.")
    if float(p.get("footclear", 0.05)) > 0.0:
        # leap states ineq_clear_air at footClear only at three OFF-NODE
        # interpolation points per flight element (nodal floor stays 0);
        # a knot grid cannot state that row where leap does.
        raise ValueError(f"{yaml_path}: problem.footclear > 0 is not ported "
                         "(leap enforces it off-node only); set 0.0.")

    return FenceSpec(
        name=doc["name"], robot=robot,
        t_stance=t_stance, t_air=float(p["tair"]), t_land=t_land,
        hop=float(p["hop"]), step=step,
        # defaults below are FenceJumpSpec's (include/leap/examples/fence_jump.hpp),
        # not zero: an absent key means the C++ builder used ITS default.
        mu=float(p.get("mu", 0.85)), friceps=float(p.get("friceps", 1e-2)),
        lammin=float(p.get("lammin", 5.0)),
        landvz=float(p.get("landvz", 0.5)),
        ulim=float(p.get("ulim", 1.0)), vlim=float(p.get("vlim", 1.0)),
        qbox=float(p.get("qbox", 1.0)), attmax=float(p.get("attmax", 0.5)),
        footclear=float(p.get("footclear", 0.05)),
        copx=float(p.get("copx", 0.08)), copy=float(p.get("copy", 0.03)),
        gammaz=float(p.get("gammaz", 0.05)),
        planarlock_idx=lock_idx,
        selfcol_margin=margin, selfcol_pairs=pairs,
        # 0.02 is ObstacleScene's default (obstacle_scene.hpp), NOT zero
        obstacle_margin=float(obs.get("margin", 0.02)), obstacle_boxes=boxes,
        q_squat=q_squat, q_stand=q_stand, q_hold0=q_hold0, q_hold2=q_hold2,
        feet0=_foot_targets(robot, q_hold0),
        feet2=_foot_targets(robot, q_hold2),
        vz_liftoff=float(np.sqrt(2.0 * GRAVITY * float(p["hop"]))))
