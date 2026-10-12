"""Verdicts and REPORT.md of a tracked folder: is each solution trackable?

Contents: `CRITERIA`, `verdict`, `repeats_of`, `onset`, `plan_diagnostics`, `plot_case`
and the REPORT.md writer `write_report`. The archive and problem yaml of a case are read
from its `metrics.json`. Repeats are looked up both as
`<case>/rep<k>/` (this package's `--repeat`) and `<case>/repeats/rep<k>/` (the
old layout).

Trackable = (A) the tracking error stays BOUNDED over the whole simulated
horizon (the plan and the terminal hold) and (B) the FINAL STATE of the plan is
reached at the end of the hold. The thresholds are the same for every robot
and both tasks:

  (A) bounded:  the error does not diverge -- the robot never falls (base height
                never below half of the plan's minimum, attitude error never
                above 0.6 rad: the rollout's own `fell` flag), the simulation
                never blows up, and the error comes back (which (B) certifies).
                A stricter transient bound (base <= 0.10 m, attitude <= 0.30 rad
                at every instant) is reported as its own column.
  (B) final:    at the end of the hold, ||base position error|| <= 0.05 m,
                attitude error <= 0.10 rad, max actuated-joint error <= 0.20
                rad, base speed <= 0.05 m/s, every foot within 0.03 m of its
                planned terminal placement, no physical capsule-obstacle
                contact during the whole run (fence) and non-negative
                clearance.
  Exception:    cases listed in `controllers.JUDGE_AT_T` are judged at the end
                of the plan with the tolerances given there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .controllers import ALL_ROBOTS, JUDGE_AT_T, ROBOT_GAINS, ROBOT_OPTIONS, TASK_GAINS, TASK_OPTIONS, TASK_ROBOTS
from .reference import load_leap_reference
from .rollout import TrackedTrajectory

CRITERIA = dict(
    bound_base_pos=0.10,     # m, whole horizon
    bound_att=0.30,          # rad, whole horizon
    final_base_pos=0.05,     # m, end of hold
    final_att=0.10,          # rad
    final_joint=0.20,        # rad, max over actuated joints
    final_speed=0.05,        # m/s, base
    final_foot=0.03,         # m, terminal foot placement
)


def wrap(e):
    return (e + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ verdict
def verdict(rdir: Path) -> dict:
    rdir = Path(rdir)
    tr = TrackedTrajectory.load(rdir / "trajectory.npz")
    mt = json.loads((rdir / "metrics.json").read_text())
    ev = json.loads((rdir / "evaluation.json").read_text()) if (rdir / "evaluation.json").exists() else {}
    act = tr.act_idx
    e = tr.q - tr.q_ref
    e[:, 3:6] = wrap(e[:, 3:6])
    ebase = np.linalg.norm(e[:, :3], axis=1)
    eatt = np.abs(e[:, 3:6]).max(axis=1)
    ejoint = np.abs(e[:, act]).max(axis=1)
    in_ref = tr.ref_window()
    C = CRITERIA
    fell = bool(mt["fell"]) or bool(mt.get("unstable", False))
    task_name = mt.get("task", mt.get("benchmark"))
    at_T_tol = JUDGE_AT_T.get(task_name, {}).get(mt["robot"])
    judged_at_T = at_T_tol is not None
    tr_t = tr.t
    if judged_at_T:
        # The final state is the state at the end of the plan (last tick of the
        # reference window), (A) is the strict bound over the plan, and the fall
        # criterion is the plan's window only (controllers.JUDGE_AT_T says why).
        C = dict(CRITERIA, **at_T_tol)
        kT = int(np.flatnonzero(in_ref)[-1])
        nc = len(tr.contacts)
        fp, fr = tr.foot_pos[kT].reshape(nc, 3), tr.foot_pos_ref[kT].reshape(nc, 3)
        feet_T = {c: float(np.linalg.norm(fp[i] - fr[i])) for i, c in enumerate(tr.contacts)}
        mt = dict(mt, final=dict(base_xyz_err=float(ebase[kT]), base_rpy_err=float(eatt[kT]),
                                 joint_err_max=float(ejoint[kT]), base_speed=float(np.linalg.norm(tr.v[kT, :3]))),
                  final_foot_err=feet_T, capsule_fence_hit_ticks=int(tr.capsule_hits[in_ref].sum()),
                  fell=bool(mt.get("unstable", False) or ebase[in_ref].max() > C["bound_base_pos"]
                            or eatt[in_ref].max() > C["bound_att"]))
        fell = bool(mt["fell"])
        ebase, eatt, ejoint = ebase[in_ref], eatt[in_ref], ejoint[in_ref]
        tr_t = tr.t[in_ref]
        in_ref = np.ones(len(ebase), dtype=bool)
    bounded_pos = float(ebase.max()) <= C["bound_base_pos"]
    bounded_att = float(eatt.max()) <= C["bound_att"]
    strict_bound = bounded_pos and bounded_att
    bounded = not fell
    fin = mt["final"]
    feet = mt.get("final_foot_err", {}) or {}
    foot_ok = (max(feet.values()) <= C["final_foot"]) if feet else True
    hits = int(mt.get("capsule_fence_hit_ticks", 0))
    gap = mt.get("fence_gap_sim", float("nan"))
    clear_ok = (hits == 0) and (not np.isfinite(gap) or gap >= 0.0)
    final_ok = (fin["base_xyz_err"] <= C["final_base_pos"] and fin["base_rpy_err"] <= C["final_att"]
                and fin["joint_err_max"] <= C["final_joint"] and fin["base_speed"] <= C["final_speed"]
                and foot_ok and clear_ok)
    reasons = []
    if judged_at_T:
        reasons.append("judged at T (controllers.JUDGE_AT_T)")
    if fell:
        reasons.append("fell / diverged" if not judged_at_T else "left the strict bound before T")
    if not fell and not final_ok:
        if fin["base_xyz_err"] > C["final_base_pos"]:
            reasons.append(f"final base error {fin['base_xyz_err']*1e3:.0f} mm")
        if fin["base_rpy_err"] > C["final_att"]:
            reasons.append(f"final attitude error {fin['base_rpy_err']:.2f} rad")
        if fin["joint_err_max"] > C["final_joint"]:
            reasons.append(f"final joint error {fin['joint_err_max']:.2f} rad")
        if fin["base_speed"] > C["final_speed"]:
            reasons.append(f"not at rest ({fin['base_speed']:.2f} m/s)")
        if not foot_ok:
            reasons.append(f"feet {max(feet.values())*1e3:.0f} mm off their terminal placement")
        if not clear_ok:
            reasons.append("obstacle contact" if hits else "negative clearance")
    # first instant the error left the bound (diagnostic)
    t_out = None
    bad = (ebase > C["bound_base_pos"]) | (eatt > C["bound_att"])
    if bad.any():
        t_out = float(tr_t[int(np.argmax(bad))])
    out = dict(
        task=task_name, method=mt.get("method", tr.meta.get("method")), robot=mt["robot"],
        trackable=bool(bounded and final_ok),
        bounded=bool(bounded), final_state=bool(final_ok), fell=fell, reason="; ".join(reasons),
        strict_bound=bool(strict_bound), judged_at_T=judged_at_T,
        max_base_err=float(ebase.max()), max_att_err=float(eatt.max()), max_joint_err=float(ejoint.max()),
        max_base_err_ref_window=float(ebase[in_ref].max()), t_error_left_bound=t_out,
        base_rmse=float(mt["base_xyz_rmse"]), att_rmse=float(mt["base_rpy_rmse"]), joint_rmse=float(mt["joint_rmse"]),
        foot_rmse=float(mt["foot_rmse"]),
        final=dict(base_pos=float(fin["base_xyz_err"]), att=float(fin["base_rpy_err"]), joint_max=float(fin["joint_err_max"]),
                   base_speed=float(fin["base_speed"]), foot_max=(max(feet.values()) if feet else float("nan"))),
        at_T=dict(mt["at_T"]),
        capsule_hits=hits, clearance_sim=gap, clearance_ref=mt.get("fence_gap_ref"),
        time_shift=float(mt.get("time_shift", 0.0)), clock_waits=mt.get("clock_waits", []),
        touchdowns={k: dict(dt=v["dt_vs_plan"], xy_err=v["xy_err"]) for k, v in mt.get("touchdowns", {}).items()},
        tau_clipped_frac=float(mt["tau_clipped_frac"]), qp_fallbacks=int(mt["qp_fallbacks"]), qp_solves=int(mt["qp_solves"]),
        wall_s=float(mt["wall_s"]), duration=float(mt["duration"]), hold=float(mt["hold"]),
        friction=mt.get("friction"), problem_mu=mt.get("problem_mu"), archive_gate=mt.get("archive_gate"),
        archive=mt.get("archive"), problem_yaml=mt.get("problem_yaml"),
        gains=mt.get("gains"), early_jump=mt.get("early_jump"), clock=mt.get("clock"), settle=mt.get("settle"),
        violations_fine=(ev.get("violations_fine") or {}), violations_fine_reference=(ev.get("violations_fine_reference") or {}),
        criteria=dict(C),
    )
    (rdir / "verdict.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return out


def repeat_dirs(rdir: Path) -> list:
    """`<case>/rep<k>/` (this package) or `<case>/repeats/rep<k>/` (the old layout)."""
    rdir = Path(rdir)
    reps = sorted(p for p in rdir.glob("rep[0-9]*") if (p / "trajectory.npz").exists())
    if (rdir / "repeats").is_dir():
        reps += sorted(p for p in (rdir / "repeats").glob("rep*") if (p / "trajectory.npz").exists())
    return reps


def repeats_of(rdir: Path):
    """(trackable, n) over the repeat runs of a case (their verdict.json is
    written on the way), else None."""
    reps = repeat_dirs(rdir)
    if not reps:
        return None
    vs = [verdict(p) if (p / "metrics.json").exists() else None for p in reps]
    vs = [v for v in vs if v is not None]
    return (sum(bool(v["trackable"]) for v in vs), len(vs)) if vs else None


def onset(rdir: Path, v: dict) -> dict:
    """What the simulation was doing when the error first left the bound: the
    plan phase, the support set, torque saturation, support-foot slip, the CoP
    against the sole box, and the clock state, over the 150 ms before it."""
    rdir = Path(rdir)
    tr = TrackedTrajectory.load(rdir / "trajectory.npz")
    log = np.load(rdir / "rollout.npz", allow_pickle=True)
    t0 = v["t_error_left_bound"]
    if t0 is None:
        return {}
    i = int(np.argmin(np.abs(tr.t - t0)))
    j = max(0, i - int(round(0.15 / float(tr.meta["control_dt"]))))
    names = tr.contacts
    phase_names = list(tr.meta.get("phase_names", []))
    ph = int(tr.phase[i])
    fp = tr.foot_pos.reshape(tr.n, len(names), 3)
    slip = {}
    for k, c in enumerate(names):
        sup = tr.support[j:i + 1, k] > 0.5
        if sup.any():
            idx = np.where(sup)[0]
            segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            slip[c] = float(max(np.linalg.norm(fp[j + sg, k, :2] - fp[j + sg[0], k, :2], axis=1).max() for sg in segs))
    off = tr.lam_off
    cop = {}
    for k, c in enumerate(names):
        if tr.contact_dims[k] == 6:
            w = tr.lam[j:i + 1, off[k]:off[k] + 6]
            m_ = w[:, 2] > 20.0
            if m_.any():
                cop[c] = [float(np.abs(-w[m_, 4] / w[m_, 2]).max()), float(np.abs(w[m_, 3] / w[m_, 2]).max())]
    return dict(t=float(t0), phase=(phase_names[ph] if ph < len(phase_names) else str(ph)),
                support=[c for k, c in enumerate(names) if tr.support[i, k] > 0.5],
                measured=[c for k, c in enumerate(names) if tr.measured[i, k] > 0.5],
                scheduled=[c for k, c in enumerate(names) if tr.scheduled[i, k] > 0.5],
                clock_frozen=bool(tr.frozen[i] > 0.5), t_ref=float(tr.t_ref[i]),
                tau_ratio_max_150ms=float(np.nanmax(log["tau_ratio"][j:i + 1])),
                qp_slack_max_150ms=float(np.nanmax(log["qp_slack"][j:i + 1])),
                support_slip_150ms=slip, cop_max_150ms=cop,
                base_speed=float(np.linalg.norm(tr.v[i, :3])),
                joint_speed_max=float(np.abs(tr.v[i, tr.act_idx]).max()))


# ----------------------------------------------------------- plan diagnostics
def plan_diagnostics(task: str, robot: str, archive, problem_yaml=None, ref=None) -> dict:
    """Properties of the SOLUTION that bear on trackability: landing impulses
    per foot vs the friction cone, the fraction of the plan spent within 1 %
    of a joint limit / at the speed or torque limit, the swing-foot approach
    height (terrain)."""
    import pinocchio as pin
    if ref is None:
        py = problem_yaml if (problem_yaml and Path(problem_yaml).exists()) else None
        ref = load_leap_reference(task, robot, archive, problem_yaml=py)
    r, sp = ref.robot, ref.spec
    m, d = r.model, r.data()
    impact_times = list(getattr(ref, "impact_times", []))
    out = dict(mu=float(sp.mu), lammin=float(sp.lammin), duration=float(ref.duration), impacts=[],
               junction=("plastic_impact" if impact_times else "continuity / impulse-free"))
    # A plan whose touchdowns are continuity junctions has no impact resets; the
    # same numbers are then reported AT the touchdown events (dv_max is the
    # reference's own velocity jump there: 0 by construction).
    for t_td in (impact_times or [float(t) for t, _ in ref.touchdown_events]):
        sm, spost = ref.eval(t_td - 1e-6), ref.eval(t_td + 1e-6)
        q, vm, vp = sm.q, sm.v, spost.v
        dv = float(np.abs(vp - vm).max())
        pin.crba(m, d, q)
        M = np.triu(d.M) + np.triu(d.M, 1).T
        pin.computeJointJacobians(m, d, q)
        pin.updateFramePlacements(m, d)
        feet = [c for c in r.contacts if c.name in spost.scheduled]
        Js = [pin.getFrameJacobian(m, d, c.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:c.dim] for c in feet]
        J = np.vstack(Js)
        Minv = np.linalg.inv(M)
        G = J @ Minv @ J.T
        # net impulse on the robot that produces the plan's jump, min-norm
        # (the per-foot split is not unique with two flat feet; the net and
        # the per-foot ratio of a point foot are)
        P = np.linalg.lstsq(J.T, M @ (vp - vm), rcond=None)[0]
        rows = []
        off = 0
        for c, Jc in zip(feet, Js):
            L = P[off:off + c.dim]
            off += c.dim
            vf = Jc @ vm
            rows.append(dict(foot=c.name, v_foot_minus=[float(x) for x in vf[:3]], impulse=[float(x) for x in L[:3]],
                             cone_ratio=float(np.hypot(L[0], L[1]) / max(L[2], 1e-9)) if L[2] > 1e-6 else float("nan"),
                             impulse_norm=float(np.linalg.norm(L[:3])), point_foot=(c.dim == 3)))
        net = np.sum([np.asarray(rw["impulse"]) for rw in rows], axis=0)
        impulsive = dv > 1e-3 and float(np.linalg.norm(net)) > 1e-3 * r.total_mass()
        out["impacts"].append(dict(t=float(t_td), dv_max=dv, feet=rows, impulsive=bool(impulsive),
                                   net_impulse=[float(x) for x in net],
                                   net_cone_ratio=(float(np.hypot(net[0], net[1]) / max(abs(net[2]), 1e-9))
                                                   if impulsive else float("nan")),
                                   delassus_cond=float(np.linalg.cond(G))))
    # limits
    act = r.act_idx
    lo, hi = r.q_lower, r.q_upper
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    vlim = r.velocity_limit * sp.vlim
    ulim = r.effort_limit * sp.ulim
    ts = np.arange(0.0, ref.duration, 0.002)
    Q, V, U = [], [], []
    for t in ts:
        s = ref.eval(float(t))
        Q.append(s.q[act])
        V.append(s.v[act])
        U.append(s.u)
    Q, V, U = np.array(Q), np.array(V), np.array(U)
    dist = np.minimum(Q - lo, hi - Q) / rng
    worst = np.argsort(dist.min(axis=0))[:3]
    out["limits"] = dict(
        joint_within_1pct_frac=float((dist < 0.01).any(axis=1).mean()),
        joint_min_dist_frac=float(dist.min()),
        joint_worst=[r.actuated_joints[i] for i in worst],
        speed_ratio_max=float((np.abs(V) / vlim).max()),
        speed_at_limit_frac=float(((np.abs(V) / vlim).max(axis=1) > 0.99).mean()),
        torque_ratio_max=float((np.abs(U) / ulim).max()),
        torque_at_limit_frac=float(((np.abs(U) / ulim).max(axis=1) > 0.99).mean()),
    )
    worst_cop, worst_t, single, tot = 0.0, 0.0, 0, 0
    for t in np.arange(0.0, ref.duration, 0.01):
        s = ref.eval(float(t))
        tot += 1
        if len(s.scheduled) == 1:
            single += 1
        for c in r.contacts:
            if c.dim == 6 and c.name in s.lam and s.lam[c.name][2] > 1.0:
                w = s.lam[c.name]
                u = max(abs(w[4] / w[2]) / max(sp.copx, 1e-9), abs(w[3] / w[2]) / max(sp.copy, 1e-9))
                if u > worst_cop:
                    worst_cop, worst_t = float(u), float(t)
    out["cop"] = dict(box=[float(sp.copx), float(sp.copy)], worst_utilisation=worst_cop, worst_t=worst_t,
                      single_support_frac=(single / max(tot, 1)))
    if task == "terrain":
        hs = []
        for i, md in enumerate(ref.modes):
            if md.kind != "swing":
                continue
            top = md.swing_to[2]
            fid = r.frame_id(md.swing_foot)
            t0 = ref.mode_starts[i]
            for s_ in (0.5, 0.8, 0.9):
                q = ref.eval(t0 + s_ * md.T - 1e-9).q
                pin.framesForwardKinematics(m, d, q)
                hs.append((s_, float(d.oMf[fid].translation[2] - top)))
        hs = np.array(hs)
        out["swing_height_above_stone"] = {f"s={s_:.1f}": dict(min=float(hs[hs[:, 0] == s_, 1].min()),
                                                                 median=float(np.median(hs[hs[:, 0] == s_, 1])))
                                           for s_ in (0.5, 0.8, 0.9)}
    return out


# ----------------------------------------------------------------- plots
def plot_case(rdir: Path, v: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = TrackedTrajectory.load(Path(rdir) / "trajectory.npz")
    act = tr.act_idx
    e = tr.q - tr.q_ref
    e[:, 3:6] = wrap(e[:, 3:6])
    T = tr.duration
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    axs[0].plot(tr.t, np.linalg.norm(e[:, :3], axis=1) * 1e3, lw=1.2, color="C0")
    axs[0].axhline(CRITERIA["bound_base_pos"] * 1e3, color="C3", ls="--", lw=1, label="bound")
    axs[0].axhline(CRITERIA["final_base_pos"] * 1e3, color="C2", ls=":", lw=1, label="final tolerance")
    axs[0].set_ylabel("base position error [mm]")
    axs[0].set_yscale("symlog", linthresh=10)
    axs[0].legend(loc="upper left", fontsize=8)
    axs[1].plot(tr.t, np.abs(e[:, 3:6]).max(axis=1), lw=1.2, color="C1")
    axs[1].axhline(CRITERIA["bound_att"], color="C3", ls="--", lw=1)
    axs[1].axhline(CRITERIA["final_att"], color="C2", ls=":", lw=1)
    axs[1].set_ylabel("base attitude error [rad]")
    axs[1].set_yscale("symlog", linthresh=0.1)
    axs[2].plot(tr.t, np.abs(e[:, act]).max(axis=1), lw=1.2, color="C4")
    axs[2].axhline(CRITERIA["final_joint"], color="C2", ls=":", lw=1)
    axs[2].set_ylabel("max joint error [rad]")
    axs[2].set_yscale("symlog", linthresh=0.1)
    axs[2].set_xlabel("wall time [s]")
    for ax in axs:
        ax.axvspan(T, tr.t[-1], color="0.9", zorder=0)
        for i in np.where(np.diff(tr.frozen) > 0.5)[0]:
            ax.axvline(tr.t[i], color="0.6", lw=0.6, ls="-.")
        ax.grid(alpha=0.3)
    ttl = f"{v['method']} / {v['task']} / {v['robot']}: {'TRACKABLE' if v['trackable'] else 'NOT trackable'}"
    if v["reason"]:
        ttl += f" ({v['reason']})"
    fig.suptitle(ttl + "\ngrey = terminal hold, dash-dot = reference clock waiting for a touchdown", fontsize=10)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- report
def fmt_mm(x):
    return "—" if x is None or not np.isfinite(x) else f"{x*1e3:.0f}"


def fmt(x, f="{:.3f}"):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def yn(b):
    return "**yes**" if b else "no"


def case_dirs(root: Path, robots=None) -> dict:
    """{robot: case dir} under `root` (one `<robot>/` per case; a case whose
    only runs are `rep<k>/` is represented by its first repeat)."""
    root = Path(root)
    order = list(robots) if robots else list(ALL_ROBOTS)
    out = {}
    for r in order:
        d = root / r
        if (d / "trajectory.npz").exists() and (d / "metrics.json").exists():
            out[r] = d
        elif d.is_dir() and repeat_dirs(d):
            out[r] = d
    return out


def write_report(task: str, root: Path, robots=None, *, plots: bool = True, title: str | None = None,
                 diagnostics: bool = True) -> dict:
    """Verdict per case (`verdict.json`), `REPORT.md`, `verdicts.json`,
    `plan_diagnostics.json` (+ `plots/`) under `root`. Returns {robot: verdict}."""
    root = Path(root)
    if robots is None:
        robots = [r for r in list(TASK_ROBOTS[task]) + [r for r in ALL_ROBOTS if r not in TASK_ROBOTS[task]]]
    cases = case_dirs(root, robots)
    rows, diags = {}, {}
    for robot, rdir in cases.items():
        main_dir = rdir if (rdir / "trajectory.npz").exists() else repeat_dirs(rdir)[0]
        v = verdict(main_dir)
        v["onset"] = onset(main_dir, v)
        v["repeats"] = repeats_of(rdir)
        v["case_dir"] = str(main_dir)
        rows[robot] = v
        if diagnostics:
            try:
                diags[robot] = plan_diagnostics(task, robot, v["archive"], v.get("problem_yaml"))
            except Exception as ex:  # noqa: BLE001
                diags[robot] = dict(error=str(ex))
        if plots:
            plot_case(main_dir, v, root / "plots" / f"{task}_{robot}.png")
    method = next((v["method"] for v in rows.values()), "leap")
    C = CRITERIA
    L = []
    L.append(f"# MuJoCo whole-body tracking of the {method} {task} solutions"
             + (f" -- {title}" if title else "") + "\n")
    L.append(f"Generated by `leaptrack.report.write_report` from `{root}/<robot>/`. Robots: "
             f"{', '.join(rows)}. One controller architecture per task "
             "(the whole-body inverse-dynamics QP of `leaptrack/wbc.py`, per-robot numbers in "
             "`leaptrack/controllers.py`), the optimizer's own robot model in MuJoCo (same URDF, "
             "pinocchio inertias, no motor model, joint order resolved by name), the problem's friction coefficient on "
             "every contact, the plastic-impact contact configuration of `scene.SETTINGS`.\n")
    L.append("## 1. Verdict\n")
    L.append("**Trackable** = (A) the error does not diverge (the robot does not fall and the simulation does not blow up; a "
             "transient excursion during a landing is bounded, and (B) certifies that it comes back) AND (B) the final state is "
             "reached at the end of the hold "
             f"(base <= {C['final_base_pos']*1e3:.0f} mm, attitude <= {C['final_att']:.2f} rad, joints <= {C['final_joint']:.2f} rad, "
             f"base speed <= {C['final_speed']:.2f} m/s, feet <= {C['final_foot']*1e3:.0f} mm from their terminal placement, "
             f"no obstacle contact, clearance >= 0). The `strict bound` column additionally asks that the error never exceed "
             f"{C['bound_base_pos']*1e3:.0f} mm / {C['bound_att']:.2f} rad at ANY instant.\n")
    n_ok = sum(v["trackable"] for v in rows.values())
    at_T = [r for r, v in rows.items() if v.get("judged_at_T")]
    L.append(f"**{n_ok} of {len(rows)} cases are trackable under these criteria.**"
             + (f" {', '.join(at_T)}: judged AT T (the end of the plan, no hold), with the tolerances of "
                f"`controllers.JUDGE_AT_T` (speed {', '.join(f'{JUDGE_AT_T[task][r].get('final_speed', CRITERIA['final_speed']):.2f} m/s' for r in at_T)}, "
                f"feet {', '.join(f'{JUDGE_AT_T[task][r].get('final_foot', CRITERIA['final_foot'])*1e3:.0f} mm' for r in at_T)}); "
                "its final columns are the state at T and its fall / bound columns cover the plan only." if at_T else "")
             + "\n")
    any_rep = any(v.get("repeats") for v in rows.values())
    if any_rep:
        L.append("The `trackable in repeats` column is the outcome of the case's `rep<k>/` runs (the same configuration "
                 "re-run in fresh interpreters); the other columns are the single run in `<robot>/` (or `rep1/`).\n")
    L.append("| robot | trackable | " + ("trackable in repeats | " if any_rep else "")
             + "(A) no divergence | (B) final state | strict bound | why not | max base err (whole run) [mm] | max attitude err [rad] | "
             "base RMSE (plan) [mm] | joint RMSE [rad] | final base / att / joint | final speed [m/s] | feet at end [mm] | "
             "touchdown vs plan [ms] | clock shift [ms] | torque-clipped ticks | QP fallbacks | strict bound first exceeded at [s] |")
    L.append("|---|:--:|" + (":--:|" if any_rep else "") + ":--:|:--:|:--:|---|--:|--:|--:|--:|---|--:|--:|---|--:|--:|--:|--:|")
    for r, v in rows.items():
        td = ", ".join(f"{k.split('@')[0][:5]} {x['dt']*1e3:+.0f}" for k, x in list(v["touchdowns"].items())[:4])
        if len(v["touchdowns"]) > 4:
            td += ", …"
        f = v["final"]
        rep = v.get("repeats")
        repcol = (f"{rep[0]} / {rep[1]} | " if rep else ("— | " if any_rep else ""))
        L.append(f"| {r}{' (at T)' if v.get('judged_at_T') else ''} | {yn(v['trackable'])} | {repcol}{yn(v['bounded'])} | {yn(v['final_state'])} | {yn(v['strict_bound'])} | {v['reason'] or '—'} | "
                 f"{fmt_mm(v['max_base_err'])} | {v['max_att_err']:.2f} | {fmt_mm(v['base_rmse'])} | {v['joint_rmse']:.3f} | "
                 f"{fmt_mm(f['base_pos'])} mm / {f['att']:.2f} rad / {f['joint_max']:.2f} rad | {f['base_speed']:.2f} | "
                 f"{fmt_mm(f['foot_max'])} | {td or '—'} | {v['time_shift']*1e3:+.0f} | {v['tau_clipped_frac']*100:.0f} % | "
                 f"{v['qp_fallbacks']}/{v['qp_solves']} | {fmt(v['t_error_left_bound'], '{:.2f}')} |")
    L.append("")
    ons = [(r, v["onset"]) for r, v in rows.items() if v.get("onset")]
    if ons:
        L.append("### 1.1 Where each excursion starts\n")
        L.append("The first instant the error leaves the strict bound (100 mm / 0.30 rad), and the 150 ms before it.\n")
        L.append("| robot | t [s] | plan phase | reference clock | scheduled / support / measured feet | max torque ratio (150 ms) | support-foot slip (150 ms) [mm] | max CoP offset (150 ms) [mm] | base speed [m/s] | max joint speed [rad/s] |")
        L.append("|---|--:|---|---|---|--:|---|---|--:|--:|")
        for r, o in ons:
            slip = ", ".join(f"{k.split('_')[0]} {x*1e3:.0f}" for k, x in o["support_slip_150ms"].items()) or "—"
            cop = ", ".join(f"{k.split('_')[0]} ({x[0]*1e3:.0f},{x[1]*1e3:.0f})" for k, x in o["cop_max_150ms"].items()) or "—"
            L.append(f"| {r} | {o['t']:.3f} | {o['phase']} | {o['t_ref']:.3f}{' (waiting)' if o['clock_frozen'] else ''} | "
                     f"{len(o['scheduled'])} / {len(o['support'])} / {len(o['measured'])} | {o['tau_ratio_max_150ms']:.2f} | {slip} | {cop} | "
                     f"{o['base_speed']:.2f} | {o['joint_speed_max']:.1f} |")
        L.append("")
    if diagnostics:
        L.append("## 2. What the plans ask of the physics (diagnostics of the solutions)\n")
        L.append("A touchdown modelled as a plastic impact (sticking projection of the pre-impact velocity onto the contact "
                 "constraints) requires an impulse from the ground; the tables give the impulse the plan's own velocity jump "
                 "requires (least-squares split over the landing feet; unique for point feet, the NET ratio for two flat feet), "
                 "whether it fits in the friction cone (ratio <= mu), the pre-impact foot velocity, and the fraction of the plan "
                 "spent at the limits.\n")
        L.append("| robot | mu | touchdown(s) at [s] | max |v+ − v−| [rad/s] (conditioning of the impact map) | pre-impact foot velocity [m/s] | tangential/normal impulse ratio per foot (net) | "
                 "plan within 1 % of a joint limit [% of time] (worst joints) | |v|/vlim max (at limit % t) | |u|/ulim max (at limit % t) | "
                 "planned CoP, worst fraction of the box (single support % of plan) |")
        L.append("|---|--:|---|--:|---|---|---|---|---|---|")
        for r in rows:
            dgn = diags.get(r)
            if not dgn or "error" in dgn:
                if dgn:
                    L.append(f"| {r} | | diagnostics failed: {dgn['error']} | | | | | | | |")
                continue
            imps = dgn["impacts"][:2]
            if not imps:
                L.append(f"| {r} | {dgn['mu']:.2f} | none | | | | | | | |")
                continue
            ts = ", ".join(f"{im['t']:.3f}" for im in dgn["impacts"]) + (" …" if len(dgn["impacts"]) > 2 else "")
            if dgn.get("junction") != "plastic_impact":
                ts += " (no impact map: continuity / impulse-free junctions)"
            dvs = ", ".join(f"{im['dv_max']:.2f}" for im in imps) + f" (Delassus cond {imps[0]['delassus_cond']:.0e})"
            imp0 = imps[0]
            if not imp0.get("impulsive", True):
                vf = "impulse-free touchdown (zero landing velocity)"
                ratios = "—"
            else:
                vf = " / ".join(f"[{x['v_foot_minus'][0]:+.2f},{x['v_foot_minus'][1]:+.2f},{x['v_foot_minus'][2]:+.2f}]"
                                for x in imp0["feet"])
                per = " / ".join((f"{x['cone_ratio']:.2f}" if (np.isfinite(x['cone_ratio']) and x['impulse_norm'] > 1e-2) else "n/a")
                                 for x in imp0["feet"])
                ratios = f"{per} (net **{imp0['net_cone_ratio']:.2f}**{' > mu' if imp0['net_cone_ratio'] > dgn['mu'] else ''})"
            lim = dgn["limits"]
            L.append(f"| {r} | {dgn['mu']:.2f} | {ts} | {dvs} | {vf} | {ratios} | "
                     f"{lim['joint_within_1pct_frac']*100:.0f} % ({', '.join(lim['joint_worst'][:2])}) | "
                     f"{lim['speed_ratio_max']:.2f} ({lim['speed_at_limit_frac']*100:.0f} %) | "
                     f"{lim['torque_ratio_max']:.2f} ({lim['torque_at_limit_frac']*100:.0f} %) | "
                     + (f"{dgn['cop']['worst_utilisation']:.2f} ({dgn['cop']['single_support_frac']*100:.0f} %) |"
                        if dgn.get("cop") and dgn["cop"]["worst_utilisation"] > 0 else f"— ({(dgn.get('cop') or {}).get('single_support_frac', 0)*100:.0f} %) |"))
        L.append("")
        terr = [(r, diags[r]) for r in rows if r in diags and "swing_height_above_stone" in diags[r]]
        if terr:
            L.append("Terrain swing-foot height above the target stone top in the plan (min / median over the steps): "
                     + "; ".join(f"{r}: " + ", ".join(f"at {k} {v_['min']*1e3:.1f}/{v_['median']*1e3:.1f} mm" for k, v_ in d_['swing_height_above_stone'].items())
                                 for r, d_ in terr) + ". A plan that skims the stone changes its touchdown instant by tens of ms "
                     "per millimetre of tracking error, which is what the reference clock absorbs.\n")
    L.append("## 3. Constraint violation of the simulated motion (dt = 0.01 s grid)\n")
    L.append("| robot | overall max (limiting family) | plan itself on the same grid |")
    L.append("|---|---|---|")
    for r, v in rows.items():
        vf = v.get("violations_fine") or {}
        vr = v.get("violations_fine_reference") or {}
        L.append(f"| {r} | {fmt(vf.get('overall'), '{:.2e}')} ({vf.get('limiting', '—')}) | {fmt(vr.get('overall'), '{:.2e}')} ({vr.get('limiting', '—')}) |")
    L.append("")
    L.append("## 4. Controller and simulation settings\n")
    clocks = sorted({str(v.get("clock")) for v in rows.values()})
    L.append("Architecture: `wbc.WholeBodyQP` (ProxQP; acceleration PD on every DOF, planned contact forces / torques "
             "as regularizers, unactuated dynamics + anchored support feet as equalities, friction pyramid / CoP box / torque box as "
             "inequalities) + `ContactPolicy` (measured contacts decide the support set) + the reference clock "
             f"`{', '.join(clocks)}` (`reference.TouchdownClock`: waits for a late touchdown on the pre-impact state and, where "
             "`early_jump` is on, jumps to the touchdown state on an early one). Control period 1 ms, MuJoCo step 0.1 ms, settle "
             f"0.05 s on the initial pose, terminal hold {TASK_OPTIONS[task]['hold']:.1f} s.\n")
    L.append("Task-level switches: " + json.dumps(TASK_GAINS[task]) + "; rollout options: "
             + json.dumps(TASK_OPTIONS[task]) + "; per-robot rollout options: " + json.dumps(ROBOT_OPTIONS.get(task, {})) + ".\n")
    L.append("Per-robot numbers: " + json.dumps(ROBOT_GAINS.get(task, {})) + " (everything else = `wbc.WbcGains()` defaults, recorded in every `metrics.json`).\n")
    if rows:
        mus = sorted({str(v.get("problem_mu")) for v in rows.values()})
        L.append(f"MuJoCo contact: friction = the problem's mu ({', '.join(mus)}), solref (2e-4 s, dampratio 20), solimp "
                 "(0.99, 0.999, 1e-4, 0.5, 2), elliptic cone, impratio 100, Newton solver, noslip 20, implicitfast at 1e-4 s. "
                 "Point feet are 1 mm spheres at the contact frame; flat feet are boxes of the yaml `foot_box` grown by 1 cm per "
                 "side (`scene.SOLE_MARGIN`), the QP's CoP box is the box grown by 0.5 cm. Stepping stones exist for a foot only "
                 "as top faces (`rollout.update_stone_masks`).\n")
    L.append("## 5. Summary\n")
    tr_ok = [r for r, v in rows.items() if v["trackable"]]
    tr_no = [r for r, v in rows.items() if not v["trackable"]]
    L.append(f"Trackable: {', '.join(tr_ok) if tr_ok else 'none'}. Not trackable: {', '.join(tr_no) if tr_no else 'none'}.\n")
    for r, v in rows.items():
        f = v["final"]
        line = (f"* **{r}**: {'trackable' if v['trackable'] else 'NOT trackable'}"
                + (f" ({v['reason']})" if v["reason"] else "")
                + f". Base RMSE {fmt_mm(v['base_rmse'])} mm over the plan, peak base error {fmt_mm(v['max_base_err'])} mm, "
                  f"final base error {fmt_mm(f['base_pos'])} mm, feet {fmt_mm(f['foot_max'])} mm from their terminal placement, "
                  f"reference clock shift {v['time_shift']*1e3:+.0f} ms, torque clipped {v['tau_clipped_frac']*100:.0f} % of the ticks."
                + (f" Repeats: trackable in {v['repeats'][0]} of {v['repeats'][1]} runs." if v.get("repeats") else ""))
        o = v.get("onset") or {}
        if o:
            line += (f" The error first left the strict bound at t = {o['t']:.2f} s (plan phase {o['phase']}, "
                     f"{len(o['support'])} support feet, torque ratio {o['tau_ratio_max_150ms']:.2f} in the preceding 150 ms).")
        L.append(line)
    L.append("")
    if plots:
        L.append("## 6. Figures\n")
        for r, v in rows.items():
            L.append(f"### {task} / {r}\n\n![](plots/{task}_{r}.png)\n")
    root.mkdir(parents=True, exist_ok=True)
    (root / "REPORT.md").write_text("\n".join(L))
    (root / "verdicts.json").write_text(json.dumps(rows, indent=1,
                                                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if diagnostics:
        (root / "plan_diagnostics.json").write_text(json.dumps(diags, indent=1))
    print(f"wrote {root / 'REPORT.md'}: {n_ok}/{len(rows)} trackable")
    for r, v in rows.items():
        print(f"  {task:7s} {r:13s} trackable={v['trackable']!s:5s} bounded={v['bounded']!s:5s} final={v['final_state']!s:5s} {v['reason']}")
    return rows
