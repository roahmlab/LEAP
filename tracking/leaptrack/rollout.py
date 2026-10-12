"""Track one LEAP solution in MuJoCo with the task's whole-body QP controller
(plastic-impact contact configuration) and reduce the log to a
`TrackedTrajectory`.

Contents: `build_scene`, `run` (the rollout loop with the approach-ring stone surface
and the touchdown clock), `evaluate`, `STONE_SIDE_TOL`, `update_stone_masks`, `_json`,
`summarize`, `print_summary`, `TrackedTrajectory`, `from_rollout`, `describe`, and
`track_archive`, the one-call API (rollout + evaluation + run.log). The suite runner is
`scripts/track.py`.

Loop (control period 1 ms = 10 MuJoCo steps of 0.1 ms):

    q, v      <- MuJoCo state (optimizer layout; joint order resolved by name)
    measured  <- feet whose contact geom touches the floor / a stone / the fence
    ref       <- the solution sampled at the reference clock (reference.TouchdownClock)
    support   <- ContactPolicy(schedule of the plan, measured contacts)
    tau       <- WholeBodyQP(q, v, ref, support)        (leaptrack.wbc)
    hold tau for the sub-steps; project the planar-lock DOFs to zero

The MuJoCo scene is the robot of the optimizer (same URDF, pinocchio's
composite inertias, one contact geom per optimizer contact, no joint damping /
armature / motor model), the floor, the problem's fence box (which the
robot's collision capsules physically hit) or its stepping stones, with the
plastic-impact contact configuration of `scene.SETTINGS` and the PROBLEM's
friction coefficient (`problem.mu`) on every colliding geom. After the
reference ends the terminal pose is held `hold` seconds.

Outputs per case (`out_dir/`): `rollout.npz` (raw log), `metrics.json` (the
rollout's own summary), `trajectory.npz` (the `TrackedTrajectory`),
`evaluation.json/.npz` (tracking error, fine-grid constraint violation,
final-state error), `run.log`, and the problem yaml the reference was built
from (`<robot>_<suffix>.yaml`).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import mujoco
import numpy as np

from .controllers import gains_for, options_for
from .plastic_impact import plastic_impact_map
from .reference import FOOT_LANDED, FOOT_SWING, METHOD, load_leap_reference, make_clock
from .scene import SETTINGS, SOLE_MARGIN, SceneBox, build, describe as describe_settings, robot_yaml
from .wbc import ContactPolicy, WholeBodyQP
from . import determinism

SCHEMA_VERSION = 1
CLOCKS = ("touchdown", "timewarp", "none")

# ================================================================ sim_trajectory
# continuous signals (linearly interpolated by resample); everything else is
# a flag/index held constant over the tick
CONTINUOUS = ("q", "v", "a", "lam", "u", "f_constraint", "t_ref",
              "q_ref", "v_ref", "a_ref", "lam_ref", "u_ref",
              "foot_pos", "foot_pos_ref")
FLAGS = ("scheduled", "support", "measured", "phase", "frozen", "capsule_hits")


@dataclass
class TrackedTrajectory:
    """Uniformly sampled time series of the SIMULATED robot (q, v, a, lam, u,
    f_constraint) together with the REFERENCE the controller was following at
    each instant (sampled at the reference clock `t_ref`), contact
    bookkeeping, foot positions, the plan phase and a JSON metadata block.
    N rows = control ticks; the base is q[:3] position and q[3:6]
    roll-pitch-yaw; `meta["act_idx"]` maps u to v-indices; contact `k`
    occupies `lam[:, lam_off[k]:lam_off[k+1]]`."""
    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    lam: np.ndarray
    u: np.ndarray
    f_constraint: np.ndarray     # N x nv, the simulator's TOTAL constraint generalized force (contacts + joint stops + ...)
    t_ref: np.ndarray
    q_ref: np.ndarray
    v_ref: np.ndarray
    a_ref: np.ndarray
    lam_ref: np.ndarray
    u_ref: np.ndarray
    scheduled: np.ndarray        # N x ncontacts (0/1) stance per the plan at t_ref
    support: np.ndarray          # N x ncontacts (0/1) load-bearing per the controller
    measured: np.ndarray         # N x ncontacts (0/1) touching per the simulator
    foot_pos: np.ndarray         # N x 3 ncontacts, simulated contact-frame positions
    foot_pos_ref: np.ndarray     # N x 3 ncontacts, reference contact-frame positions
    phase: np.ndarray            # N, plan phase index at t_ref (meta["phase_names"])
    frozen: np.ndarray           # N (0/1) reference clock frozen by the time warp
    capsule_hits: np.ndarray     # N, number of physical capsule-obstacle contacts
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- helpers
    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def contacts(self) -> list:
        return list(self.meta["contacts"])

    @property
    def contact_dims(self) -> list:
        return [int(d) for d in self.meta["contact_dims"]]

    @property
    def lam_off(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.contact_dims)]).astype(int)

    @property
    def act_idx(self) -> np.ndarray:
        return np.asarray(self.meta["act_idx"], int)

    @property
    def duration(self) -> float:
        """Reference duration T (the hold follows)."""
        return float(self.meta["duration"])

    def lam_of(self, name: str, ref: bool = False) -> np.ndarray:
        k = self.contacts.index(name)
        off = self.lam_off
        src = self.lam_ref if ref else self.lam
        return src[:, off[k]:off[k + 1]]

    def foot_of(self, name: str, ref: bool = False) -> np.ndarray:
        k = self.contacts.index(name)
        src = self.foot_pos_ref if ref else self.foot_pos
        return src[:, 3 * k:3 * k + 3]

    def lam_dict(self, i: int, ref: bool = False) -> dict:
        off = self.lam_off
        src = self.lam_ref if ref else self.lam
        return {c: src[i, off[k]:off[k + 1]] for k, c in enumerate(self.contacts)}

    # ------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f.name: np.asarray(getattr(self, f.name)) for f in fields(self) if f.name != "meta"}
        meta = dict(self.meta, schema_version=SCHEMA_VERSION)
        np.savez_compressed(path, meta=np.array(json.dumps(meta, default=_json)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrackedTrajectory":
        d = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        kw = {f.name: d[f.name] for f in fields(cls) if f.name != "meta"}
        return cls(meta=meta, **kw)

    # ------------------------------------------------------------- resample
    def resample(self, dt: float, t_end: float | None = None) -> "TrackedTrajectory":
        """Uniform grid 0, dt, 2dt, ... <= t_end (default: the last logged
        instant). Continuous signals are linearly interpolated, flags/phase
        held from the previous tick; `meta["resampled_dt"]` records it."""
        t_end = float(self.t[-1]) if t_end is None else min(float(t_end), float(self.t[-1]))
        tg = np.arange(0.0, t_end + 0.5 * dt, dt)
        tg = tg[tg <= self.t[-1] + 1e-12]
        idx_prev = np.clip(np.searchsorted(self.t, tg, side="right") - 1, 0, self.n - 1)
        kw = {"t": tg}
        for name in CONTINUOUS:
            arr = np.asarray(getattr(self, name), float)
            if arr.ndim == 1:
                kw[name] = np.interp(tg, self.t, arr)
            else:
                kw[name] = np.column_stack([np.interp(tg, self.t, arr[:, j]) for j in range(arr.shape[1])])
        for name in FLAGS:
            kw[name] = np.asarray(getattr(self, name))[idx_prev]
        return TrackedTrajectory(meta=dict(self.meta, resampled_dt=float(dt)), **kw)

    def window(self, t0: float, t1: float) -> np.ndarray:
        """Boolean mask t0 <= t <= t1."""
        return (self.t >= t0 - 1e-12) & (self.t <= t1 + 1e-12)

    def ref_window(self) -> np.ndarray:
        """Ticks where the controller was still following the reference
        (reference clock <= T); the complement is the hold."""
        return self.t_ref <= self.duration + 1e-9


def from_rollout(out: dict, metrics: dict, ref, *, benchmark: str, method: str,
                 extra_meta: dict | None = None) -> TrackedTrajectory:
    """Adapter for `run`'s log dict (`out`) + `metrics.json` dict + the
    reference object."""
    names = list(out["contacts"])
    dims = [int(d) for d in out["contact_dims"]]
    robot = ref.robot
    mnames = list(robot.model.names)[1:]
    meta = dict(
        benchmark=benchmark, method=method, robot=ref.robot_name,
        archive=str(ref.npz_path), archive_dir=str(Path(ref.npz_path).parent),
        contacts=names, contact_dims=dims,
        joint_names=mnames, actuated_joints=list(robot.actuated_joints),
        act_idx=[int(i) for i in robot.act_idx], nq=int(robot.nq), nv=int(robot.nv), nu=int(robot.nu),
        planarlock=list(ref.planarlock), lock_idx=[mnames.index(j) for j in ref.planarlock],
        effort_limit=[float(x) for x in robot.effort_limit],
        velocity_limit=[float(x) for x in robot.velocity_limit],
        total_mass=float(robot.total_mass()),
        duration=float(out["duration"]), hold=float(out["hold"]),
        control_dt=float(out["control_dt"]), sim_dt=float(out["sim_dt"]),
        time_shift=float(out["time_shift"]), unstable=bool(out["unstable"]),
        phase_names=list(getattr(ref, "phase_names", ("push", "flight", "land", "hold"))),
        phase_times=[float(x) for x in ref.phase_times],
        success=bool(metrics.get("success", False)), fell=bool(metrics.get("fell", False)),
        controller=metrics.get("controller", "wbc.WholeBodyQP"), gains=metrics.get("gains", {}),
        controller_options=metrics.get("controller_options", {}),
        controller_build_s=metrics.get("controller_build_s", 0.),
        settle_static=metrics.get("settle_static", False),
        force_smoothing=metrics.get("force_smoothing", 0.),
        continuous_coast=metrics.get("continuous_coast", False),
        interpolation=metrics.get("interpolation", 'original'),
        full_terminal_hold=metrics.get("full_terminal_hold", False),
        actual_terminal_hold=metrics.get("actual_terminal_hold", 0.),
        reference_completed=metrics.get("reference_completed", False),
        mujoco=metrics.get("mujoco", ""), mujoco_settings=metrics.get("mujoco_settings", {}),
        determinism=metrics.get("determinism", {}),
        rollout_metrics_keys=sorted(metrics.keys()),
    )
    meta.update(extra_meta or {})
    n = len(out["t"])
    return TrackedTrajectory(
        t=np.asarray(out["t"], float), q=np.asarray(out["q"], float), v=np.asarray(out["v"], float),
        a=np.asarray(out["a"], float), lam=np.asarray(out["lam"], float), u=np.asarray(out["tau"], float),
        f_constraint=np.asarray(out["f_constraint"], float),
        t_ref=np.asarray(out["t_ref"], float), q_ref=np.asarray(out["q_ref"], float),
        v_ref=np.asarray(out["v_ref"], float), a_ref=np.asarray(out["a_ref"], float),
        lam_ref=np.asarray(out["lam_ref"], float), u_ref=np.asarray(out["u_ref"], float),
        scheduled=np.asarray(out["scheduled"], float).reshape(n, len(names)),
        support=np.asarray(out["support"], float).reshape(n, len(names)),
        measured=np.asarray(out["measured"], float).reshape(n, len(names)),
        foot_pos=np.asarray(out["foot_pos"], float), foot_pos_ref=np.asarray(out["foot_pos_ref"], float),
        phase=np.asarray(out["phase"], int), frozen=np.asarray(out["frozen"], float),
        capsule_hits=np.asarray(out["capsule_hits"], int), meta=meta)


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def describe(tr: TrackedTrajectory) -> str:
    m = tr.meta
    return (f"{m['benchmark']}/{m['method']}/{m['robot']}: {tr.n} ticks @ {m['control_dt']*1e3:.1f} ms, "
            f"T={m['duration']:.3f}s + hold {m['hold']:.2f}s, nq={m['nq']} nu={m['nu']} "
            f"contacts={dict(zip(m['contacts'], m['contact_dims']))} planarlock={m['planarlock']} "
            f"success={m['success']} time_shift={m['time_shift']*1e3:+.0f}ms")


# ================================================================ rollout.py
# Stones exist for a foot only where the plan expects contact (the benchmark
# keeps only the collision capsules clear of the stones, with a `topslack`
# allowance). Each control tick a foot may collide with a
# stone iff (a) the foot's lowest point is not more than STONE_SIDE_TOL below
# the stone top (a foot never meets a side face) and (b) the foot is not in
# the first part of its swing (the plan puts it down only at the end: swing
# progress >= the controller's early-contact phase, or stance / just landed).
STONE_SIDE_TOL = 0.002


def update_stone_masks(rs, d, stones: list, foot_phase: dict | None = None, early_phase: float = 0.8,
                       *, persistent: bool = False) -> dict:
    """`stones`: [(name, top)] of the scene boxes that are stones; `foot_phase`:
    the reference sample's per-foot (state, progress). Returns
    {foot: [allowed stone names]} after setting the geom masks."""
    if not stones:
        return {}
    bottoms = rs.foot_bottom_z(d)
    # Keep an established top contact through compression or sole tilt. The
    # entry rule still excludes side contacts; liftoff releases the latch.
    latched = getattr(rs, '_stance_stones', {}) if persistent else {}
    if persistent:
        g2stone = {rs.box_geoms[nm]: nm for nm, _ in stones}
        g2foot = {gid: f for f, gid in rs.foot_geom.items()}
        for c in d.contact:
            foot = g2foot.get(c.geom1) or g2foot.get(c.geom2)
            stone = g2stone.get(c.geom1) or g2stone.get(c.geom2)
            if foot is not None and stone is not None and c.dist <= 0. and abs(c.frame[2]) > .99:
                latched.setdefault(foot, set()).add(stone)
    out = {}
    for f, zb in bottoms.items():
        st, prog = (foot_phase or {}).get(f, (FOOT_LANDED, 1.0))
        may_land = not (st == FOOT_SWING and prog < early_phase)
        allowed = [nm for nm, top in stones if top <= zb + STONE_SIDE_TOL] if may_land else []
        if persistent:
            if may_land:
                allowed = sorted(set(allowed) | latched.get(f, set()))
            else:
                latched.pop(f, None)
        rs.set_foot_box_mask(f, allowed)
        out[f] = allowed
    if persistent:
        rs._stance_stones = latched
    return out


def summarize(out, ref, rs, wbc, impacts, wall, capsule_collision):
    t = out["t"]
    T = float(out["duration"])
    q, qr = out["q"], out["q_ref"]
    v = out["v"]
    act = ref.robot.act_idx
    in_ref = out["t_ref"] <= T + 1e-9       # the (time-warped) reference window
    e = q - qr
    e[:, 3:6] = (e[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    rms = lambda x: float(np.sqrt(np.mean(x**2))) if x.size else float("nan")
    names = list(out["contacts"])
    nfe = len(names)
    fp = out["foot_pos"].reshape(len(t), nfe, 3)
    fpr = out["foot_pos_ref"].reshape(len(t), nfe, 3)
    fdist = np.linalg.norm(fp - fpr, axis=2)
    sup = out["support"].astype(bool)
    iT = int(np.argmin(np.abs(out["t_ref"] - T)))
    # physical clearance of the simulated and reference q (fence: dense
    # capsule-to-box gap; terrain: top slack minus capsule intrusion), min over time
    robot = ref.robot
    boxes = ref.obstacle_boxes
    # sample period of the log (control_dt when the log holds a single tick)
    sample_dt = float(t[1] - t[0]) if len(t) > 1 else float(out['control_dt'])
    sel = np.arange(0, len(t), max(1, int(round(0.005 / sample_dt))))
    cs_sim = ref.clearance_series(q[sel])
    cs_ref = ref.clearance_series(qr[sel])
    gap_sim = float(np.min(cs_sim)) if cs_sim is not None else float("nan")
    gap_ref = float(np.min(cs_ref)) if cs_ref is not None else float("nan")
    has_clearance = cs_sim is not None
    # z-limit and attitude for the fall test: relative to the reference
    fell = bool(np.any(q[:, 2] < 0.5 * qr[:, 2].min()) or np.abs(e[:, 3:6]).max() > 0.6 or out["unstable"])
    fall_detail = dict(min_z=float(q[:, 2].min()), min_z_ref=float(qr[:, 2].min()),
                       max_att_err=float(np.abs(e[:, 3:6]).max()), unstable=bool(out["unstable"]))
    # touchdowns: for every touchdown event of the plan and each of its feet,
    # the first physical touch after the foot's liftoff, vs the instant the
    # plan would have landed it given the reference clock at liftoff
    # (fence: one event, all feet; terrain: one per step). Keyed "<foot>" for
    # a single event, "<foot>@<k>" otherwise.
    td = {}
    t_ref = out["t_ref"]
    events = list(ref.touchdown_events)
    for ei, (t_td, feet) in enumerate(events):
        idx_td = np.where(t_ref >= t_td - 1e-9)[0]
        if not idx_td.size:
            continue
        i_td = int(idx_td[0])
        for f in feet:
            k = names.index(f)
            key = f if len(events) == 1 else f"{f}@{ei}"
            sched_f = out["scheduled"][:, k] > 0.5
            unsched = np.where(~sched_f[:i_td + 1])[0]          # unscheduled ticks up to this touchdown
            if not unsched.size:
                continue
            # the liftoff = start of the LAST contiguous unscheduled run before i_td
            runs = np.split(unsched, np.where(np.diff(unsched) > 1)[0] + 1)
            i_lo = int(runs[-1][0])
            shift_lo = float(t[i_lo] - t_ref[i_lo])               # reference clock offset at liftoff
            idx = np.where((out["measured"][:, k] > 0) & (t > t[i_lo] + 0.02))[0]
            if not idx.size:
                continue
            j = int(idx[0])
            tgt = np.asarray(ref.targets_at(t_td + 1e-6).get(f, [np.nan] * 3), float)
            td[key] = dict(t=float(t[j]), dt_vs_plan=float(t[j] - (t_td + shift_lo)),
                           xy_err=float(np.linalg.norm(fp[j, k, :2] - tgt[:2])),
                           pos=fp[j, k].tolist(), event=int(ei), foot=f)
    # stance drift: support-foot displacement while support (per foot, max)
    drift = {}
    for k, f in enumerate(names):
        idx = np.where(sup[:, k])[0]
        if idx.size:
            segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            drift[f] = float(max(np.linalg.norm(fp[s_, k, :2] - fp[s_[0], k, :2], axis=1).max() for s_ in segs))
    tau = out["tau"]
    tau_max = wbc.tau_max
    pnames = list(ref.phase_names)
    imp_summary = [dict(t=i["t"], new=i["new"], phase=pnames[i["phase"]] if i["phase"] < len(pnames) else str(i["phase"]),
                        foot_speed_minus=i["foot_speed_minus"], foot_speed_after_rel=i["foot_speed_after_rel"],
                        dv_pred_norm=i["dv_pred_norm"], dv_sim_norm=i["dv_sim_norm"],
                        dv_mismatch_rel=i["dv_mismatch_rel"]) for i in impacts]
    final_err = np.abs(e[-1])
    settled_speed = float(np.linalg.norm(v[-1][:3]))
    hits = int(np.sum(out["capsule_hits"] > 0))
    # terminal footholds (terrain: the feet must end on their stones)
    tft = ref.terminal_foot_targets
    final_foot_err = {f: float(np.linalg.norm(fp[-1, names.index(f)] - np.asarray(tft[f], float)))
                      for f in names if f in tft}
    feet_ok = (ref.terminal_foot_tol is None or not final_foot_err
               or max(final_foot_err.values()) <= ref.terminal_foot_tol)
    completion_ok = (not out.get("full_terminal_hold", False) or
                     (out.get("reference_completed", False) and out["actual_terminal_hold"] >= float(out["hold"]) - 1e-9))
    success = (completion_ok and not fell and hits == 0 and (not has_clearance or gap_sim >= 0.0)
               and float(np.linalg.norm(e[-1, :3])) < 0.05 and settled_speed < 0.05
               and np.abs(e[-1, 3:6]).max() < 0.1 and feet_ok)
    return dict(
        benchmark=ref.benchmark, method=ref.method, robot=ref.robot_name, duration=T, hold=float(out["hold"]),
        control_dt=float(out["control_dt"]), sim_dt=float(out["sim_dt"]), wall_s=float(wall),
        unstable=bool(out["unstable"]), simulated_s=float(t[-1]),
        reference_completed=bool(out.get("reference_completed", out["t_ref"][-1] >= T)),
        actual_terminal_hold=float(out.get("actual_terminal_hold", 0.)),
        full_terminal_hold=bool(out.get("full_terminal_hold", False)),
        base_xyz_rmse=rms(e[in_ref][:, :3]), base_rpy_rmse=rms(e[in_ref][:, 3:6]),
        joint_rmse=rms(e[in_ref][:, act]),
        base_xyz_max=float(np.abs(e[in_ref][:, :3]).max()), base_rpy_max=float(np.abs(e[in_ref][:, 3:6]).max()),
        joint_max=float(np.abs(e[in_ref][:, act]).max()),
        foot_rmse=rms(fdist[in_ref]), foot_max=float(fdist[in_ref].max()),
        foot_swing_rmse=rms(fdist[in_ref][~sup[in_ref]]),
        at_T=dict(base_xyz_err=float(np.linalg.norm(e[iT, :3])), base_rpy_err=float(np.abs(e[iT, 3:6]).max()),
                  joint_err_max=float(np.abs(e[iT, act]).max())),
        final=dict(base_xyz_err=float(np.linalg.norm(e[-1, :3])), base_rpy_err=float(np.abs(e[-1, 3:6]).max()),
                   joint_err_max=float(final_err[act].max()), base_speed=settled_speed,
                   speed_norm=float(np.linalg.norm(v[-1]))),
        fence_gap_sim=float(gap_sim), fence_gap_ref=float(gap_ref), clearance_label=ref.clearance_label,
        final_foot_err=final_foot_err, terminal_foot_tol=ref.terminal_foot_tol,
        capsule_fence_hit_ticks=hits, capsule_fence_max_depth=float(out["capsule_depth"].max()),
        capsule_collision_enabled=bool(capsule_collision),
        touchdowns=td, stance_drift=drift,
        tau_ratio_max=float(np.max(np.abs(tau) / tau_max)),
        tau_clipped_frac=float(np.mean(np.any(np.abs(tau) >= tau_max - 1e-9, axis=1))),
        tau_rmse_vs_ref=rms(tau[in_ref] - out["u_ref"][in_ref]),
        qp_fallbacks=int(wbc.n_fallback), qp_cold_retries=int(wbc.n_cold_retries), qp_solves=int(wbc.n_solves),
        impact_window_ticks=int(getattr(wbc, "n_impact_ticks", 0)),
        qp_slack_max=float(np.nanmax(out["qp_slack"])) if np.any(np.isfinite(out["qp_slack"])) else 0.0,
        impacts=imp_summary, fell=fell, fall_detail=fall_detail, success=bool(success))


def print_summary(mt):
    fin = mt["final"]
    print(f"[{mt['method']}/{mt['robot']}] {'SUCCESS' if mt['success'] else 'FAIL'}  "
          f"sim {mt['simulated_s']:.2f}s (ref {mt['duration']:.2f}s + hold {mt['hold']:.1f}s) in {mt['wall_s']:.0f}s wall; "
          f"reference time shift {mt.get('time_shift', 0.0)*1e3:+.0f} ms ({mt.get('frozen_ticks', 0)} frozen ticks)")
    print(f"  tracking RMSE over the reference: base xyz {mt['base_xyz_rmse']*1e3:.1f} mm, "
          f"base rpy {mt['base_rpy_rmse']:.4f} rad, joints {mt['joint_rmse']:.4f} rad, feet {mt['foot_rmse']*1e3:.1f} mm")
    print(f"  final (after hold): base err {fin['base_xyz_err']*1e3:.1f} mm / {fin['base_rpy_err']:.4f} rad, "
          f"joint err max {fin['joint_err_max']:.4f} rad, base speed {fin['base_speed']:.2e} m/s")
    print(f"  {mt.get('clearance_label', 'clearance')} sim/ref {mt['fence_gap_sim']*1e3:+.1f}/{mt['fence_gap_ref']*1e3:+.1f} mm, capsule-obstacle hit ticks {mt['capsule_fence_hit_ticks']}"
          f" (max depth {mt['capsule_fence_max_depth']*1e3:.1f} mm), fell {mt['fell']} {mt['fall_detail']}")
    print(f"  torque: max |tau|/tau_max {mt['tau_ratio_max']:.3f}, clipped ticks {mt['tau_clipped_frac']:.1%}, "
          f"QP fallbacks {mt['qp_fallbacks']}/{mt['qp_solves']}, slack max {mt['qp_slack_max']:.2f}")
    for f, tdd in mt["touchdowns"].items():
        print(f"  touchdown {f}: t={tdd['t']:.3f} ({tdd['dt_vs_plan']*1e3:+.0f} ms vs plan), xy err {tdd['xy_err']*1e3:.1f} mm")
    if mt.get("final_foot_err"):
        print("  final foothold err: " + ", ".join(f"{f} {e*1e3:.1f} mm" for f, e in mt["final_foot_err"].items()))
    for im in mt["impacts"]:
        print(f"  impact t={im['t']:.3f} {im['phase']} new={im['new']}: foot speed before {im['foot_speed_minus']:.3f} m/s, "
              f"after/before {im['foot_speed_after_rel']:.3f}; |dv| pred {im['dv_pred_norm']:.3f} sim {im['dv_sim_norm']:.3f} "
              f"mismatch/|v-| {im['dv_mismatch_rel']:.3f}")


# =================================================================== track.py
def build_scene(ref, *, capsule_collision: bool = True, friction: float | None = None):
    """The MuJoCo model for a reference: robot + floor + the problem's boxes,
    plastic-impact settings, friction = the problem's mu (or `friction`, a
    DIAGNOSTIC override)."""
    boxes = [SceneBox(sb.name, np.asarray(sb.center, float), np.asarray(sb.half, float),
                      collide_capsules=bool(sb.collide_capsules)) for sb in ref.scene_boxes()]
    any_capsule_box = any(b.collide_capsules for b in boxes)
    settings = SETTINGS.with_(friction=float(ref.spec.mu if friction is None else friction))
    return build(ref.robot_name, scene_boxes=boxes, planarlock=tuple(ref.planarlock),
                 capsule_collision=capsule_collision and any_capsule_box, settings=settings)


def run(task: str, robot: str, archive=None, *, problem_yaml=None, out_dir: Path | None = None,
        gains=None, control_dt: float = 1e-3, clock: str = "touchdown",
        hold: float = 1.5, settle: float = 0.05, time_warp: bool = True, early_jump: bool = True,
        persistent_stones: bool = False, capsule_collision: bool = True, duration: float | None = None,
        verbose: bool = True, friction: float | None = None, approach_ring: float = 0.05,
        reference=None):
    """One tracking rollout. Returns (log dict, metrics dict, reference, RobotSim).

    `reference`: an already-built reference object (what `load_leap_reference`
    returns) to track instead of loading `archive`."""
    ref = reference if reference is not None else load_leap_reference(
        task, robot, archive, problem_yaml=problem_yaml, work_dir=out_dir)
    spec = ref.spec
    rs = build_scene(ref, capsule_collision=capsule_collision, friction=friction)
    m, d = rs.model, mujoco.MjData(rs.model)
    gains = gains or gains_for(task, robot)
    # QP CoP box = the physical sole (yaml foot_box + SOLE_MARGIN) minus half
    # the margin: the planner's box lies strictly inside the physical sole
    # (the usual planner/plant relation), the controller keeps some authority.
    fb = {c["frame"]: c.get("foot_box") for c in robot_yaml(robot)["contacts"]}
    cop = {c.name: (float(fb[c.name]["x"]) + 0.5 * SOLE_MARGIN, float(fb[c.name]["y"]) + 0.5 * SOLE_MARGIN)
           for c in ref.robot.contacts if c.dim == 6}
    wbc = WholeBodyQP(ref.robot, ref.contact_dims, gains=gains, mu=spec.mu, cop=cop,
                      gammaz=spec.gammaz, lock_idx=rs.lock_idx, lammin=spec.lammin)
    policy = ContactPolicy(ref.contact_names, gains)
    if gains.sole_clearance:
        wbc.sole_corners = {f: np.asarray(c) for f, c in getattr(rs, "foot_corners", {}).items()}
    if not time_warp:
        clock = "none"
    clk = make_clock(clock, ref, gains, early_jump)
    feedback = wbc

    stone_boxes = [sb for sb in ref.scene_boxes() if not sb.collide_capsules]
    stones = [(sb.name, float(np.asarray(sb.center)[2] + np.asarray(sb.half)[2])) for sb in stone_boxes]
    foot_state = {}          # per control tick: contact -> (position, lowest point z)
    if stone_boxes:
        # Landing surface under (or within `approach_ring` of) the planned foot
        # position `p`, which the QP turns into the swing target
        # `max(plan_z - touchdown_depth, surface_z - stone_touchdown_depth)`.
        #
        # While the foot is ABOVE the top face, report the top face; once it is
        # more than `lift_tol` below it, report the top face RAISED by the clamp,
        # i.e. command the foot exactly onto the surface.
        lift_tol = 0.002

        def surface_z(p):
            best = None
            for sb in stone_boxes:
                c, hh = np.asarray(sb.center), np.asarray(sb.half)
                dx, dy = abs(p[0] - c[0]) - hh[0], abs(p[1] - c[1]) - hh[1]
                if dx <= approach_ring and dy <= approach_ring:
                    top = float(c[2] + hh[2])
                    best = top if best is None else max(best, top)
            if best is None:
                return None
            near = None
            for f, (pf, zb) in foot_state.items():
                d2 = float((pf[0] - p[0]) ** 2 + (pf[1] - p[1]) ** 2)
                if near is None or d2 < near[0]:
                    near = (d2, zb)
            if near is not None and near[1] < best - lift_tol:
                return best + gains.stone_touchdown_depth     # target = the surface itself
            return best
        wbc.surface_z_fn = surface_z
    dt = float(m.opt.timestep)
    n_sub = max(1, int(round(control_dt / dt)))
    control_dt = n_sub * dt
    wbc.control_dt = control_dt
    T = ref.duration if duration is None else min(duration, ref.duration)
    T_end = T + (hold if duration is None else 0.0)
    n_ctrl = int(round(T_end / control_dt))

    # initial state = the solution's t = 0 state; settle on the frozen t = 0
    # reference (the soles start at z = +0)
    q0, v0 = ref.q0()
    rs.set_state(d, q0, v0)
    mujoco.mj_forward(m, d)
    s0 = ref.eval(0.0)
    s0 = replace(s0, v=np.zeros_like(s0.v), a=np.zeros_like(s0.a))
    for _ in range(int(round(settle / control_dt))):
        mujoco.mj_forward(m, d)
        update_stone_masks(rs, d, stones, s0.foot_phase, gains.early_contact_phase, persistent=persistent_stones)
        q, v = rs.get_state(d)
        foot_state.update({f: (p_, rs.foot_bottom_z(d)[f]) for f, p_ in rs.foot_positions(d).items()})
        cs = policy.update(0.0, rs.feet_in_contact(d), s0, wbc.foot_poses(q))
        if gains.measured_patch or gains.point_contact:
            # the measured contact points, read by WholeBodyQP for these two switches
            wbc.contact_points = rs.top_contact_points(d)
        tau, info = feedback.compute(q, v, s0, cs)
        rs.apply_torque(d, tau)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
            rs.project_locks(d)
    policy.reset()
    mujoco.mj_forward(m, d)

    names = ref.contact_names
    dims = ref.contact_dims
    keys = ("t", "t_ref", "frozen", "q", "v", "a", "lam", "q_ref", "v_ref", "a_ref", "lam_ref", "u_ref",
            "tau", "f_constraint", "scheduled", "support", "measured", "foot_pos", "foot_pos_ref", "foot_wrench",
            "qp_force", "qp_ok", "qp_slack", "qp_acc", "foot_bottom_z", "contact_count", "stone_mask",
            "tau_ratio", "phase", "capsule_hits", "capsule_depth")
    log = {k: [] for k in keys}
    nlam = sum(dims.values())

    def stack_lam(per_contact: dict) -> np.ndarray:
        row = np.zeros(nlam)
        off = 0
        for f in names:
            w = per_contact.get(f)
            if w is not None:
                row[off:off + dims[f]] = np.asarray(w)[:dims[f]]
            off += dims[f]
        return row

    d_diag = mujoco.MjData(m)
    impacts = []
    touch_prev = rs.feet_in_contact(d)
    touch_time = {}          # contact -> time of its latest touchdown at the physics step
    v_prev_step = rs.get_state(d)[1]
    step_i = 0
    unstable = False
    terminal_started = None
    t_wall = time.perf_counter()
    for i in range(n_ctrl + 1):
        t = i * control_dt
        mujoco.mj_forward(m, d)
        q, v = rs.get_state(d)
        measured = rs.feet_in_contact(d)
        s = clk.sample(t, measured) if clk is not None else ref.eval(t)
        if s.t >= T - 1e-9 and terminal_started is None:
            terminal_started = t
        update_stone_masks(rs, d, stones, s.foot_phase, gains.early_contact_phase, persistent=persistent_stones)
        foot_state.update({f: (p_, rs.foot_bottom_z(d)[f]) for f, p_ in rs.foot_positions(d).items()})
        cs = policy.update(t, measured, s, wbc.foot_poses(q), touchdowns=touch_time)
        if gains.measured_patch or gains.point_contact:
            wbc.contact_points = rs.top_contact_points(d)
        tau, info = feedback.compute(q, v, s, cs)
        # --- log ---
        fpos = rs.foot_positions(d)
        fref = wbc.foot_poses(s.q)
        wr = rs.contact_wrenches(d)
        hits = rs.capsule_obstacle_contacts(d)
        log["t"].append(t)
        log["t_ref"].append(s.t)
        log["frozen"].append(1.0 if (clk is not None and getattr(clk, "frozen", False)) else 0.0)
        log["q"].append(q)
        log["v"].append(v)
        log["q_ref"].append(s.q.copy())
        log["v_ref"].append(s.v.copy())
        log["a_ref"].append(s.a.copy())
        log["lam_ref"].append(stack_lam(s.lam))
        log["u_ref"].append(s.u.copy())
        log["tau"].append(tau.copy())
        # simulated acceleration and contact wrenches UNDER THE TORQUE JUST
        # COMPUTED, evaluated on a COPY of the simulator state so that the
        # simulation itself is untouched.
        mujoco.mj_copyData(d_diag, m, d)
        rs.apply_torque(d_diag, tau)
        mujoco.mj_forward(m, d_diag)
        log["a"].append(d_diag.qacc[rs.dofadr].copy())
        log["lam"].append(stack_lam(rs.contact_wrenches(d_diag)))
        log["f_constraint"].append(d_diag.qfrc_constraint[rs.dofadr].copy())
        log["scheduled"].append([1.0 if f in s.scheduled else 0.0 for f in names])
        log["support"].append([1.0 if f in cs.support else 0.0 for f in names])
        log["measured"].append([1.0 if f in measured else 0.0 for f in names])
        log["foot_pos"].append(np.concatenate([fpos[f] for f in names]))
        log["foot_pos_ref"].append(np.concatenate([fref[f][0] for f in names]))
        log["foot_wrench"].append(np.concatenate([wr[f] for f in names]))
        qf = np.zeros(nlam)
        off = 0
        for f in names:
            if info.get("ok") and f in info["forces"]:
                qf[off:off + dims[f]] = info["forces"][f]
            off += dims[f]
        log["qp_force"].append(qf)
        log["qp_ok"].append(1.0 if info.get("ok") else 0.0)
        log["qp_slack"].append(float(info.get("slack", np.nan)))
        log["qp_acc"].append(np.asarray(info.get("a", np.full(ref.nv, np.nan))).copy())
        bottoms = rs.foot_bottom_z(d)
        log["foot_bottom_z"].append([bottoms[f] for f in names])
        log["contact_count"].append([sum(c.dist <= 0. and rs.foot_geom[f] in (c.geom1, c.geom2)
                                         for c in d.contact) for f in names])
        log["stone_mask"].append([int(m.geom_contype[rs.foot_geom[f]]) for f in names])
        log["tau_ratio"].append(float(info.get("tau_ratio", np.nan)))
        log["phase"].append(s.phase)
        log["capsule_hits"].append(len(hits))
        log["capsule_depth"].append(max([h[2] for h in hits], default=0.0))
        if not np.all(np.isfinite(q)) or np.abs(q).max() > 50 or np.linalg.norm(v) > 100:
            unstable = True
            break
        if i == n_ctrl:
            break
        # --- advance ---
        rs.apply_torque(d, tau)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
            rs.project_locks(d)
            step_i += 1
            touch = rs.feet_in_contact(d)
            new = touch - touch_prev
            for f in new:
                touch_time[f] = float(step_i * dt)
            if new:
                Mfull = np.zeros((m.nv, m.nv))
                mujoco.mj_fullM(m, d, Mfull)
                rows = []
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                for f in sorted(touch):
                    mujoco.mj_jacSite(m, d, jp, jr, rs.foot_site[f])
                    rows.append(jp.copy())
                    if dims[f] == 6:
                        rows.append(jr.copy())
                J = np.vstack(rows)
                vm = v_prev_step
                vpl, _ = plastic_impact_map(Mfull, J, vm.copy())
                impacts.append(dict(t=float(step_i * dt), new=sorted(new), touching=sorted(touch),
                                    v_minus=vm.copy(), v_plus_pred=vpl.copy(),
                                    foot_speed_minus=float(np.linalg.norm(J @ vm)),
                                    J=J, phase=int(s.phase), step=step_i))
            touch_prev = touch
            v_prev_step = d.qvel.copy()
    wall = time.perf_counter() - t_wall

    out = {k: np.asarray(val) for k, val in log.items()}
    out["contacts"] = np.array(names)
    out["contact_dims"] = np.array([dims[f] for f in names])
    out["duration"] = T
    out["hold"] = T_end - T
    out["actual_terminal_hold"] = max(0., float(out["t"][-1]) - terminal_started) if terminal_started is not None else 0.
    out["reference_completed"] = terminal_started is not None
    out["full_terminal_hold"] = False
    out["control_dt"] = control_dt
    out["sim_dt"] = dt
    out["dofadr"] = rs.dofadr
    out["unstable"] = unstable
    out["time_shift"] = getattr(clk, "shift", 0.0) if clk is not None else 0.0
    tlog = out["t"]
    for ev_ in impacts:
        j = int(np.argmin(np.abs(tlog - (ev_["t"] + 0.002))))
        va = out["v"][j]
        dv_pred = ev_["v_plus_pred"] - ev_["v_minus"]
        dv_sim = va - ev_["v_minus"]
        vm = np.linalg.norm(ev_["v_minus"]) + 1e-9
        ev_["dv_pred_norm"] = float(np.linalg.norm(dv_pred))
        ev_["dv_sim_norm"] = float(np.linalg.norm(dv_sim))
        ev_["dv_mismatch_rel"] = float(np.linalg.norm(dv_pred - dv_sim) / vm)
        ev_["foot_speed_after"] = float(np.linalg.norm(ev_["J"] @ va))
        ev_["foot_speed_after_rel"] = float(ev_["foot_speed_after"] / (ev_["foot_speed_minus"] + 1e-9))
        del ev_["J"]
    metrics = summarize(out, ref, rs, wbc, impacts, wall, capsule_collision)
    metrics.update(method=METHOD, archive_dir=str(Path(ref.npz_path).parent), archive=str(ref.npz_path),
                   archive_gate=float(ref.settings.get("gate", float("nan"))), time_shift=float(out["time_shift"]),
                   frozen_ticks=int(np.sum(out["frozen"])), gains=gains.to_dict(),
                   controller="wbc.WholeBodyQP", controller_options={}, task=task, settle=settle,
                   clock=(clock if clk is not None else "none"), time_warp=bool(clk is not None),
                   early_jump=bool(early_jump), clock_waits=list(getattr(clk, "waits", [])) if clk is not None else [],
                   clock_rearms=list(getattr(clk, "rearms", [])) if clk is not None else [],
                   approach_ring=float(approach_ring), persistent_stones=bool(persistent_stones),
                   planned_impacts=[float(x) for x in getattr(ref, "impact_times", [])],
                   mujoco=describe_settings(rs.settings), mujoco_settings=rs.settings.to_dict(),
                   friction=float(rs.settings.friction), problem_mu=float(spec.mu),
                   problem_yaml=str(ref.settings.get("problem_yaml", "")),
                   problem_yaml_source=str(ref.settings.get("problem_yaml_source", "")),
                   determinism=determinism.describe())
    if verbose:
        print_summary(metrics)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "rollout.npz", **out)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_json))
        traj = from_rollout(out, metrics, ref, benchmark=task, method=METHOD,
                            extra_meta=dict(problem_yaml=metrics["problem_yaml"]))
        traj.save(out_dir / "trajectory.npz")
        if verbose:
            print(f"wrote {out_dir / 'rollout.npz'}  {out_dir / 'metrics.json'}  {out_dir / 'trajectory.npz'}")
    return out, metrics, ref, rs


def evaluate(out_dir: Path, task: str, verbose: bool = False, ref=None):
    from .evaluate import evaluate_dir
    return evaluate_dir(Path(out_dir), task=task, verbose=verbose, ref=ref)


# ================================================================== the API
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def track_archive(task, robot, archive, out_dir, problem_yaml=None, hold=1.5, settle=0.05,
                  clock="touchdown", early_jump=None, gains=None, verbose=False, *,
                  control_dt=None, duration=None, friction=None, capsule_collision=True,
                  approach_ring=0.05, evaluate_after=True):
    """Track one LEAP archive for (task, robot) with the paper's protocol and
    write `out_dir/{rollout.npz, metrics.json, trajectory.npz, evaluation.json,
    evaluation.npz, run.log}` (+ the problem yaml the reference was built from).
    Returns the metrics dict (with `evaluation` = the evaluation summary).

    `early_jump=None` takes the task / robot default of `controllers.TASK_OPTIONS`
    / `ROBOT_OPTIONS` (fence: True; terrain: False except h1 and hopper2d_foot);
    True / False force it. `gains`: a `wbc.WbcGains` (default
    `controllers.gains_for(task, robot)`). `duration` simulates only this long
    (no hold, no evaluation)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = options_for(task, dict(hold=hold, settle=settle, control_dt=control_dt, early_jump=early_jump),
                       robot=robot)
    gains = gains or gains_for(task, robot)
    header = dict(task=task, robot=robot, archive=str(archive), problem_yaml=(str(problem_yaml) if problem_yaml else None),
                  clock=clock, friction=friction, duration=duration, capsule_collision=capsule_collision,
                  approach_ring=approach_ring, options=opts, gains=gains.to_dict(),
                  python=sys.executable, mujoco=mujoco.__version__)
    with (out_dir / "run.log").open("w") as lg:
        lg.write("leaptrack.track_archive " + json.dumps(header, default=_json) + "\n")
        lg.flush()
        streams = (lg, sys.stdout) if verbose else (lg,)
        with contextlib.redirect_stdout(_Tee(*streams)):
            t0 = time.perf_counter()
            out, metrics, ref, rs = run(task, robot, archive, problem_yaml=problem_yaml, out_dir=out_dir,
                                        gains=gains, clock=clock, duration=duration, friction=friction,
                                        capsule_collision=capsule_collision, approach_ring=approach_ring,
                                        verbose=True, **opts)
            if evaluate_after and duration is None:
                summary = evaluate(out_dir, task, verbose=True, ref=ref)
                metrics["evaluation"] = summary
            print(f"total wall {time.perf_counter() - t0:.0f} s")
    return metrics
