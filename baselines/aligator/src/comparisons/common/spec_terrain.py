"""Parse a terrain problem yaml into a solver-agnostic spec,
including the mode schedule (support_0, swing_k/support_k alternation), the
stance tables, the inset keep-out boxes and the seed poses.

Mirrors the C++ terrain builder's problem statement (schedule, IK, swing arcs,
keep-out boxes).

Every `problem:` key is CLASSIFIED (see _MODELLED / _TRANSCRIPTION /
_MUST_EQUAL) and an unclassified key is a hard error, so a knob the C++
builder honours cannot be silently dropped here.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from .ik import ik_event, ik_residual, nominal_pose
from .robot import Robot, load_robot, _joint_v_index

# Keys whose VALUE this port reads and enforces.
_MODELLED = {
    "type", "tstart", "mu", "friceps", "lammin", "landvz", "copx", "copy",
    "gammaz", "monitorz", "ulim", "vlim", "qbox", "attmax", "stancepin",
    "zeroland", "zerolandxy", "landramp", "wvel", "planarlock",
    "selfcolmargin", "selfcolpairs", "nominal", "start", "steps",
    # --- trackability / posture study keys (2026-09-13/14), parsed into the
    # spec below; the ALIGATOR terrain driver (comparisons/aligator_unified_terrain)
    # enforces every one of them, the other ports read what they need ---
    # landcont: with zeroland, the touchdown is a continuity junction instead
    # of a plastic-impact reset (terrain_walk.cpp:962-966).
    "landcont",
    # basevbox: base linear-velocity box |v_base_i| <= basevbox[i] (rows
    # box_v_base, terrain_walk.cpp:1039-1064; an entry <= 0 = unbounded).
    "basevbox",
    # approachfrac / liftfrac: the last / first fraction of every swing is its
    # own mode in which the swing foot hovers level over its placement (xy
    # pinned, sole level, z >= the stone; terrain_walk.cpp:858-931). Exposed as
    # extra TerrainMode entries when load_terrain_spec(..., windows=True); the
    # default keeps the 2N+1 support/swing schedule (the windows have the same
    # contact set, so a knot-grid plan that ignores them poses a RELAXATION).
    "approachfrac", "liftfrac",
    # ulimjoint: per-joint torque scale on top of ulim (terrain_walk.cpp:1067-1073).
    "ulimjoint",
    # attbox / posture: per-axis attitude box replacing attmax, named joint
    # boxes (terrain_walk.cpp:1000-1024, 1125-1150).
    "attbox", "posture",
}
# leap's SPECTRAL TRANSCRIPTION only; a knot grid replaces them.
_TRANSCRIPTION = {
    "noslipnq", "noslipacc", "vboxnq", "selfcolinterp", "lsqmodemask",
    # basezfrac: the base-height floor of the posture study (leap row
    # ineq_base_z with a nominal-height fraction). NOT modelled by the ports;
    # tolerated so the tracking loader can read plans solved with it.
    "basezfrac",
}
# Keys the port cannot express and therefore ASSUMES (leap defaults from
# include/leap/examples/terrain_walk.hpp).
_MUST_EQUAL = {
    "stancepin": True,   # stance foot placement pins at every stance node
}


def _classify_problem_keys(p: dict, path) -> None:
    for k, v in p.items():
        if k in _MUST_EQUAL:
            want = _MUST_EQUAL[k]
            ok = bool(v) == want if isinstance(want, bool) else float(v) == float(want)
            if not ok:
                raise ValueError(
                    f"{path}: problem.{k} = {v!r}, but the comparison ports "
                    f"assume {want!r} and cannot express the alternative.")
            continue
        if k in _MODELLED or k in _TRANSCRIPTION:
            continue
        raise ValueError(
            f"{path}: unknown problem key '{k}'. Classify it in "
            "comparisons/common/spec_terrain.py (_MODELLED / _TRANSCRIPTION / "
            "_MUST_EQUAL) so it cannot silently diverge from the C++ builder.")


@dataclasses.dataclass
class TerrainMode:
    kind: str                 # "support" | "swing"
    T: float
    active: list              # stance contact names
    targets: dict             # contact name -> world pos (all active feet)
    swing_foot: str = None
    swing_from: np.ndarray = None
    swing_to: np.ndarray = None
    clearz: float = 0.0
    # Feet that touched down at this support mode's FIRST instant. They always
    # get the placement pin + the touchdown-velocity rows; whether their force
    # additionally has to START at zero is `land_feet` (leap's landRamp).
    touchdown_feet: tuple = ()
    land_feet: tuple = ()     # feet whose force starts at 0 (landRamp only)
    lift_feet: tuple = ()     # feet whose force ends at 0 (support modes)
    # Swing-phase windows (approachfrac / liftfrac, windows=True only):
    # window = None (a whole swing or its flight part), "lift" (the foot rises
    # vertically off swing_from: xy pinned there, sole level, z >= its height)
    # or "approach" (the mirror image over swing_to). `landing` marks the mode
    # whose last instant carries the touchdown rows (the swing / approach).
    window: str = None
    landing: bool = False
    hover_target: np.ndarray = None


@dataclasses.dataclass
class TerrainSpec:
    name: str
    robot: Robot
    modes: list               # list[TerrainMode], length 2N+1
    stance_after: list        # N+1 dicts contact->pos (gait event tables)
    # params
    mu: float
    friceps: float
    lammin: float
    wvel: float
    wvel_weight: float        # wvel * (U0/V0)^2
    attmax: float
    ulim: float
    vlim: float
    qbox: float
    landvz: float
    zeroland: bool
    zerolandxy: bool
    landramp: bool
    landcont: bool            # zeroland + landcont: touchdown = continuity junction
    attbox: list              # [] = attmax on roll/pitch/yaw; else per-axis limits (<= 0 = unbounded)
    att_limits: dict          # base q index (3, 4, 5) -> limit actually enforced (planar lock removed)
    posture: list             # [(joint name, v index, lo, hi)] named joint boxes (box_posture)
    ulim_joint: list          # [(joint name, scale)] per-joint torque scales on top of ulim
    effort_box: np.ndarray    # per actuated joint: ulim * scale_j * URDF effort limit (the box_u bound)
    basevbox: list            # [] or 3 numbers; enforced per axis when > 0 and not planar-locked
    basev_limits: dict        # base v index (0, 1, 2) -> limit actually enforced
    approachfrac: float
    liftfrac: float
    monitorz: float
    copx: float
    copy: float
    gammaz: float
    planarlock_idx: list
    selfcol_margin: float
    selfcol_pairs: list       # [(a, b, dsafe)]
    # terrain
    keepout_boxes: list       # [(center(3,), half(3,))] INSET boxes
    keepout_capsules: list    # capsule names enforced
    obstacle_margin: float
    topslack: float
    pillars: list             # true pillars [(cx,cy,lx,ly,top)] for the judge
    # transcription info the ports need for row PLACEMENT (leap enforces
    # ineq_swing_z at 3 interp points per SUB-segment, and the h-refinement
    # splits each mode into `split` sub-segments)
    split: int
    # seed
    q_nominal: np.ndarray
    z_nominal: float
    q_event: list             # N+1 IK poses
    q_arc: list               # N mid-swing IK poses
    ik_worst: float


def load_terrain_spec(yaml_path, ik_iters=150, windows=False) -> TerrainSpec:
    """`windows=True` splits every swing into [lift,] flight [, approach] modes
    when the yaml sets liftfrac / approachfrac (the C++ mode plan); the default
    keeps one swing mode per step (2N+1 modes) for the knot-grid ports and the
    tracking loader."""
    yaml_path = Path(yaml_path).resolve()
    doc = yaml.safe_load(open(yaml_path))
    robot = load_robot((yaml_path.parent / doc["robot"]).resolve())
    p = doc["problem"]
    assert p["type"] == "terrain_walk", p["type"]
    _classify_problem_keys(p, yaml_path)
    m = robot.model

    nominal = {e["joint"]: float(e["angle"]) for e in p.get("nominal", [])}
    q_nom = nominal_pose(robot, nominal)
    z_nom = q_nom[2]

    start = {e["contact"]: np.asarray(e["pos"], dtype=float) for e in p["start"]}
    steps = p["steps"]
    n_steps = len(steps)

    # stanceAfter[k]: placements after step k
    stance_after = [dict(start)]
    for st in steps:
        cur = dict(stance_after[-1])
        for f in st["feet"]:
            cur[f["contact"]] = np.asarray(f["pos"], dtype=float)
        stance_after.append(cur)

    lock_names = p.get("planarlock", [])
    lock_idx = [m.joints[m.getJointId(n)].idx_q for n in lock_names]
    dim6 = {c.name for c in robot.contacts if c.dim == 6}

    # ------------------------------------------------------------- IK seeds
    def event_ik(targets):
        base_z = float(np.mean([t[2] for t in targets.values()])) + z_nom
        return ik_event(robot, targets, q_nom, base_z, lock_idx=lock_idx,
                        sole_level_contacts=dim6, iters=ik_iters)

    q_event = [event_ik(stance_after[k]) for k in range(n_steps + 1)]
    ik_worst = max(ik_residual(robot, q_event[k], stance_after[k])
                   for k in range(n_steps + 1))

    # ------------------------------------------------------------ modes
    land_ramp = bool(p.get("landramp", False))
    approach_frac = float(p.get("approachfrac", 0.0))
    lift_frac = float(p.get("liftfrac", 0.0))
    if not (0.0 <= approach_frac <= 0.5 and 0.0 <= lift_frac <= 0.5 and approach_frac + lift_frac < 1.0):
        raise ValueError(f"{yaml_path}: approachfrac / liftfrac out of range")
    all_names = [c.name for c in robot.contacts]
    modes = [TerrainMode("support", float(p.get("tstart", 0.4)),
                         list(all_names), dict(stance_after[0]))]
    q_arc = []
    for k, st in enumerate(steps):
        swing = [f["contact"] for f in st["feet"]]
        assert len(swing) == 1, "one swing foot per step expected"
        foot = swing[0]
        frm = stance_after[k][foot]
        to = stance_after[k + 1][foot]
        # -1 = off is the C++ default (FootPlacement::clearZ); the row is
        # stated when clearz >= 0 (terrain_walk.cpp:796)
        clearz = float(st["feet"][0].get("clearz", -1.0))
        stance_names = [n for n in all_names if n != foot]
        stance_targets = {n: stance_after[k][n] for n in stance_names}
        tsw = float(st["tswing"])
        if windows and (approach_frac > 0.0 or lift_frac > 0.0):
            # terrain_walk.cpp:873-931: lift (liftFrac*Tswing, hover over the
            # stone left), flight ((1-approach-lift)*Tswing, clearz points, the
            # touchdown rows only without an approach), approach
            # (approachFrac*Tswing, hover over the target + touchdown rows).
            if lift_frac > 0.0:
                modes.append(TerrainMode("swing", tsw * lift_frac, stance_names,
                                         stance_targets, swing_foot=foot,
                                         swing_from=frm, swing_to=to, clearz=-1.0,
                                         window="lift", hover_target=frm.copy()))
            modes.append(TerrainMode("swing", tsw * (1.0 - approach_frac - lift_frac),
                                     stance_names, stance_targets, swing_foot=foot,
                                     swing_from=frm, swing_to=to, clearz=clearz,
                                     landing=approach_frac <= 0.0))
            if approach_frac > 0.0:
                modes.append(TerrainMode("swing", tsw * approach_frac, stance_names,
                                         stance_targets, swing_foot=foot,
                                         swing_from=frm, swing_to=to, clearz=-1.0,
                                         window="approach", landing=True,
                                         hover_target=to.copy()))
        else:
            modes.append(TerrainMode("swing", tsw, stance_names,
                                     stance_targets, swing_foot=foot,
                                     swing_from=frm, swing_to=to, clearz=clearz,
                                     landing=True))
        # touchdown_feet: always (the placement pin + touchdown velocity rows
        # live at the swing's last instant in leap). land_feet: only under
        # landRamp -- without it leap does NOT pin the landed foot's first
        # force control point (hopper2d / biped5 carry landramp: false).
        modes.append(TerrainMode("support", float(st["tsupport"]),
                                 list(all_names), dict(stance_after[k + 1]),
                                 touchdown_feet=(foot,),
                                 land_feet=(foot,) if land_ramp else ()))
        # mid-swing arc pose (terrain_walk.cpp:1105-1132)
        apex = 0.5 * (frm + to)
        apex = apex.copy()
        apex[2] = max(clearz, max(frm[2], to[2]) + 0.03) + 0.02
        tg = dict(stance_targets)
        tg[foot] = apex
        base_z = (float(np.mean([stance_after[k][n][2] for n in stance_names]))
                  if stance_names else 0.0) + z_nom
        # the C++ arc IK levels EVERY dim-6 sole, the flying one included
        # (terrain_walk.cpp:1130 passes the full contact list): that is what
        # keeps the swing sole out of the keep-out boxes.
        q_arc.append(ik_event(robot, tg, q_nom, base_z, lock_idx=lock_idx,
                              sole_level_contacts=dim6, iters=ik_iters))
    # lift_feet on support modes: support k (0-based mode 2k) lifts the swing
    # foot of step k+1
    support_modes = [md for md in modes if md.kind == "support"]
    assert len(support_modes) == n_steps + 1
    for k in range(n_steps):
        support_modes[k].lift_feet = (steps[k]["feet"][0]["contact"],)

    # ------------------------------------------------------------ selfcol
    pairs = []
    margin = float(p.get("selfcolmargin", 0.0))
    for pr in p.get("selfcolpairs", []):
        ra = robot.capsules[pr["a"]].radius
        rb = robot.capsules[pr["b"]].radius
        # a negative per-pair margin is the C++ "use the default" sentinel
        mval = float(pr.get("margin", -1.0))
        pairs.append((pr["a"], pr["b"],
                      ra + rb + (margin if mval < 0.0 else mval)))

    # ------------------------------------------------------------ terrain
    ter = doc.get("terrain", {})
    for k in ter:
        if k not in ("margin", "topslack", "topinset", "interp", "windowpad",
                     "capsules", "pillars"):
            raise ValueError(f"{yaml_path}: unknown terrain key '{k}'")
    ter_margin = float(ter.get("margin", 0.03))       # terrain_walk.hpp default
    ter_slack = float(ter.get("topslack", 0.02))
    # windowPad 0 = every pillar on every mode, which is what the port does.
    if float(ter.get("windowpad", 0.0)) > 0.0:
        raise ValueError(f"{yaml_path}: terrain.windowpad > 0 is not modelled "
                         "(the port attaches every pillar to every knot)")
    keepout_caps = list(ter.get("capsules", [])) or list(robot.capsules.keys())
    # topinset 0 => the C++ derives max_c(radius_c) + margin + topslack over the
    # ENFORCED capsules (terrain_walk.cpp:963-967).
    inset = float(ter.get("topinset", 0.0))
    if inset <= 0.0 and ter.get("pillars"):
        dsafe_max = max(robot.capsules[c].radius + ter_margin
                        for c in keepout_caps)
        inset = dsafe_max + ter_slack
    boxes, pillars = [], []
    for pl in ter.get("pillars", []):
        cx, cy = pl["center"]
        lx, ly = pl["size"]
        top = float(pl["top"])
        pillars.append((cx, cy, lx, ly, top))
        top_i = top - inset
        if top_i > 1e-9:
            boxes.append((np.array([cx, cy, top_i / 2.0]),
                          np.array([lx / 2.0, ly / 2.0, top_i / 2.0])))

    # ------------------------------------------------------------ boxes
    ulim = float(p.get("ulim", 1.0))
    ulim_joint = [(e["joint"], float(e.get("scale", 1.0))) for e in p.get("ulimjoint", [])]
    effort_box = ulim * np.abs(np.asarray(robot.effort_limit, dtype=float))
    for name, scale in ulim_joint:
        if name not in robot.actuated_joints:
            raise ValueError(f"{yaml_path}: ulimjoint '{name}' is not an actuated joint")
        effort_box[list(robot.actuated_joints).index(name)] *= scale
    posture = []
    for e in p.get("posture", []):
        lo, hi = -np.inf, np.inf
        if "band" in e:
            c, w = float(e.get("center", 0.0)), float(e["band"])
            if w < 0.0:
                raise ValueError(f"{yaml_path}: posture band must be >= 0")
            lo, hi = c - w, c + w
        if "lo" in e:
            lo = float(e["lo"])
        if "hi" in e:
            hi = float(e["hi"])
        if not lo <= hi:
            raise ValueError(f"{yaml_path}: posture '{e['joint']}' wants lo <= hi")
        posture.append((e["joint"], _joint_v_index(m, e["joint"]), lo, hi))
    attmax = float(p.get("attmax", 0.5))
    attbox = [float(v) for v in p.get("attbox", [])]
    if attbox and len(attbox) != 3:
        raise ValueError(f"{yaml_path}: attbox wants 3 numbers [roll, pitch, yaw]")
    att_limits = {}
    for i in (3, 4, 5):
        lim = attbox[i - 3] if attbox else attmax
        if lim > 0.0 and i not in lock_idx:
            att_limits[i] = lim
    basevbox = [float(v) for v in p.get("basevbox", [])]
    basev_limits = {}
    for i in range(min(3, len(basevbox))):
        if basevbox[i] > 0.0 and i not in lock_idx:
            basev_limits[i] = basevbox[i]

    # ------------------------------------------------------------ wvel
    U0 = float(np.mean(np.abs(robot.effort_limit)))
    V0 = max(float(np.mean(np.abs(robot.velocity_limit))), 1e-6)
    wvel = float(p.get("wvel", 0.0))

    # Defaults below are TerrainWalkSpec's (terrain_walk.hpp), not zero: an
    # absent key means the C++ builder used ITS default.
    return TerrainSpec(
        name=doc["name"], robot=robot, modes=modes, stance_after=stance_after,
        mu=float(p.get("mu", 0.8)), friceps=float(p.get("friceps", 1e-2)),
        lammin=float(p.get("lammin", 5.0)), wvel=wvel,
        wvel_weight=wvel * (U0 / V0) ** 2,
        attmax=attmax, ulim=ulim,
        vlim=float(p.get("vlim", 1.0)), qbox=float(p.get("qbox", 1.0)),
        landvz=float(p.get("landvz", 0.5)),
        zeroland=bool(p.get("zeroland", True)),
        zerolandxy=bool(p.get("zerolandxy", False)),
        landramp=land_ramp, landcont=bool(p.get("landcont", False)),
        attbox=attbox, att_limits=att_limits, posture=posture,
        ulim_joint=ulim_joint, effort_box=effort_box,
        basevbox=basevbox, basev_limits=basev_limits,
        approachfrac=approach_frac, liftfrac=lift_frac,
        monitorz=float(p.get("monitorz", 0.0)),
        copx=float(p.get("copx", 0.08)), copy=float(p.get("copy", 0.03)),
        gammaz=float(p.get("gammaz", 0.05)),
        planarlock_idx=lock_idx, selfcol_margin=margin, selfcol_pairs=pairs,
        keepout_boxes=boxes, keepout_capsules=keepout_caps,
        obstacle_margin=ter_margin, topslack=ter_slack, pillars=pillars,
        split=int(doc.get("solver", {}).get("split", 1)),
        q_nominal=q_nom, z_nominal=z_nom, q_event=q_event, q_arc=q_arc,
        ik_worst=ik_worst)


# ---------------------------------------------------------------- seed eval
def seed_state(spec: TerrainSpec, mode_idx: int, tloc: float):
    """Seed (q, v, a) inside mode `mode_idx` at local time tloc
    (terrain_walk.cpp:1193-1259): supports hold the event pose; swings run a
    quadratic Bezier through the arc pose. v, a analytic."""
    md = spec.modes[mode_idx]
    nq = spec.robot.nq
    if md.window is not None or any(m.window is not None for m in spec.modes):
        raise NotImplementedError("seed_state models the 2N+1 schedule only (load with windows=False)")
    if md.kind == "support":
        k = mode_idx // 2
        return spec.q_event[k].copy(), np.zeros(nq), np.zeros(nq)
    k = (mode_idx - 1) // 2
    qa, qb = spec.q_event[k], spec.q_event[k + 1]
    ctrl = 2.0 * spec.q_arc[k] - 0.5 * (qa + qb)
    T = md.T
    s = tloc / T
    q = (1 - s) ** 2 * qa + 2 * s * (1 - s) * ctrl + s * s * qb
    dq_ds = 2 * (1 - s) * (ctrl - qa) + 2 * s * (qb - ctrl)
    d2q_ds2 = 2 * (qb - 2 * ctrl + qa)
    return q, dq_ds / T, d2q_ds2 / (T * T)


def seed_force_torque(spec: TerrainSpec, mode_idx: int, q, v, a):
    """Per-node (lam, u) seed: min-norm floating-base LS for lam
    (terrain_walk.cpp:1275-1301), u from the actuated rows."""
    r = spec.robot
    m, d = r.model, r.data()
    md = spec.modes[mode_idx]
    tau = pin.rnea(m, d, q, v, a)
    pin.computeJointJacobians(m, d, q)
    pin.updateFramePlacements(m, d)
    unact = [i for i in range(r.nv) if i not in set(r.act_idx)]
    Js = {}
    for c in r.contacts:
        if c.name in md.active:
            J = pin.getFrameJacobian(m, d, c.frame_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Js[c.name] = J[:(6 if c.dim == 6 else 3), :]
    lam = {}
    if Js:
        K = np.hstack([Js[n][:, unact].T for n in Js])  # 6 x sum(dims)
        rhs = tau[unact]
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        off = 0
        for n in Js:
            dim = Js[n].shape[0]
            lam[n] = sol[off:off + dim]
            off += dim
        for n in Js:
            tau = tau - Js[n].T @ lam[n]
    u = tau[r.act_idx]
    return lam, u
