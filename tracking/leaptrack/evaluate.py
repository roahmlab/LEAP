"""Judge one `TrackedTrajectory` against its reference: tracking error,
constraint violation of the simulated motion on a fine grid, final-state
error, physical checks.

Contents: the constraint families (`FAMILIES`, `UNITS`, `PROBLEM_FAMILIES`,
`enforcement_tables`, `ConstraintChecker`) and the tracking evaluation (`wrap_angles`,
`rms`, `vmax`, `_track_block`, `TOL`, `evaluate`, `evaluate_dir`, `print_summary`); the
task-keyed helpers `terminal_condition` and `clearance`. `evaluate_dir` reloads the LEAP
reference through `leaptrack.reference.load_leap_reference` (or takes it as an argument).

Constraint families (positive part of the violation, absolute residual for equalities):

| family | rows | unit |
|---|---|---|
| cone | sqrt(fx^2+fy^2+eps^2) - eps - mu fz, every foot with a measured force | N |
| normal | -fz | N |
| lam_floor | lammin - fz for the stage's stance feet (unless the stage drops the floor row) | N |
| cop, yaw | |ty| - copx fz, |tx| - copy fz ; |tz| - gammaz fz (dim-6 feet) | N·m |
| inactive_force | max |lam| on a foot the stage does NOT hold in contact | N |
| keepout | separating-face certificate of every collision capsule vs every obstacle box | m |
| selfcol | leap's smooth capsule-pair residual | m |
| clear | monitor_lb - z of every monitor frame the stage keeps off the ground | m |
| foot_pos | stance foot vs its commanded placement (position, + level sole for dim 6) | m, rad |
| foot_vel | no-slip J v of the feet the stage holds in contact | m/s, rad/s |
| box_q, box_v, box_att, base_z, planar, box_u | the boxes | rad, rad/s, rad, m, rad, N·m |
| inverse_dynamics | max |M a + h - S u - sum J^T lam| over the free DOFs (consistency check) | N, N·m |

Final-state error against the task's terminal condition (fence: the rest pose
`x_goal` and the landing footholds; terrain: the solution's own endpoint and the
pinned terminal footholds) at two instants: `at_T` (the first tick where the
reference clock reaches T) and `end` (the end of the hold). These six errors
(`base_pos_err, base_rot_err, joint_err_max, foot_err_max, base_speed,
joint_speed_max`) are what `leaptrack.score` turns into Table III's ratio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .geometry import selfcol_residual, separating_face_certificate
from .rollout import TrackedTrajectory, _json

EVAL_VERSION = 1

# ============================================================ constraint_check
FAMILIES = ("cone", "normal", "lam_floor", "cop", "yaw", "inactive_force",
            "keepout", "selfcol", "clear", "foot_pos", "foot_vel",
            "box_q", "box_v", "box_att", "base_z", "planar", "box_u", "inverse_dynamics")
UNITS = dict(cone="N", normal="N", lam_floor="N", cop="N·m", yaw="N·m", inactive_force="N",
             keepout="m", selfcol="m", clear="m", foot_pos="m / rad", foot_vel="m/s / rad/s", land_vz="m/s",
             box_q="rad", box_v="rad/s", box_att="rad", base_z="m", planar="m / rad", box_u="N·m",
             inverse_dynamics="N / N·m", vz_up="m/s", swing_z="m", bc="state units")
# families that are constraints of the PROBLEM (the headline max excludes the
# data-consistency row)
PROBLEM_FAMILIES = tuple(f for f in FAMILIES if f != "inverse_dynamics")


def enforcement_tables(plan):
    """Per knot (len N+1): the placement pins and the no-slip feet that knot
    ENFORCES, used for the interior rule."""
    enf_pins, enf_act = [], []
    for kn in plan.knots:
        pm = {}
        for src in (kn.stance_pins, kn.touchdown_pins, kn.start_pins):
            if src:
                pm.update(src)
        act = set(kn.active) | set(kn.noslip)
        if kn.touchdown_pins and kn.touchdown_vel not in (None, "zero"):
            act -= set(kn.touchdown_pins)
        enf_pins.append(pm)
        enf_act.append(act)
    if not plan.q0_free:
        enf_pins[0] = dict(plan.knots[0].targets)
        enf_act[0] = set(plan.knots[0].active)
    if plan.x_goal is not None:
        enf_pins.append(dict(plan.knots[-1].targets))
        enf_act.append(set(plan.knots[-1].active))
    else:
        enf_pins.append(dict(plan.terminal_foot_pins or {}))
        enf_act.append(set(plan.knots[-1].active) if plan.terminal_v_zero else set())
    return enf_pins, enf_act


class ConstraintChecker:
    def __init__(self, plan):
        self.plan = plan
        self.spec = spec = plan.spec
        self.robot = robot = plan.robot
        self.bench = plan.bench
        self.model = robot.model
        self.data = robot.data()
        self.nq, self.nv, self.nu = robot.nq, robot.nv, robot.nu
        self.contacts = [c.name for c in robot.contacts]
        self.dim_of = {c.name: int(c.dim) for c in robot.contacts}
        self.fid_of = {c.name: c.frame_id for c in robot.contacts}
        self.lam_off = np.concatenate([[0], np.cumsum([c.dim for c in robot.contacts])]).astype(int)
        self.ts = np.asarray(plan.ts, float)
        self.enf_pins, self.enf_act = enforcement_tables(plan)
        if self.bench == "fence":
            self.boxes = list(spec.obstacle_boxes)
            self.box_caps = list(robot.capsules.keys())
        else:
            self.boxes = list(spec.keepout_boxes)
            self.box_caps = list(spec.keepout_capsules)
        self.margin = float(spec.obstacle_margin)
        qlo, qhi = robot.q_lower, robot.q_upper
        self.qmid, self.qhalf = 0.5 * (qlo + qhi), 0.5 * (qhi - qlo)
        self.lock_idx = list(spec.planarlock_idx)
        self.att_idx = [i for i in (3, 4, 5) if i not in self.lock_idx]
        self.act_idx = np.asarray(robot.act_idx, int)
        self.free_rows = np.array([i for i in range(self.nv) if i not in set(self.lock_idx)], int)
        self.u_scale = (0.995 if self.bench == "fence" else 1.0) * spec.ulim
        self.monitors = list(plan.monitors or [])
        self.T = float(self.ts[-1])

    # ------------------------------------------------------------ markers
    def stage_at(self, t_ref: float) -> int:
        k = int(np.searchsorted(self.ts, t_ref, side="right") - 1)
        return min(max(k, 0), len(self.ts) - 2)

    def markers(self, t_ref: float) -> dict:
        """Which rows exist at reference time t_ref (interior rule)."""
        kn_list = self.plan.knots
        if t_ref >= self.T - 1e-9:                       # hold: the terminal rows
            k = len(kn_list) - 1
            kn = kn_list[-1]
            pins = dict(self.enf_pins[-1])
            slip = set(self.enf_act[-1])
            active = set(kn.active)
            terminal = True
        else:
            k = self.stage_at(t_ref)
            kn = kn_list[k]
            pins = {cn: tgt for cn, tgt in self.enf_pins[k].items() if cn in self.enf_pins[k + 1]}
            slip = self.enf_act[k] & self.enf_act[k + 1]
            active = set(kn.active)
            terminal = False
        return dict(k=k, knot=kn, pins=pins, slip=slip, active=active, terminal=terminal)

    # ------------------------------------------------------------ one sample
    def evaluate(self, t_ref: float, q, v, a=None, u=None, lam=None) -> dict:
        """Violation per family at one instant. `lam`: (sum dims,) stacked
        measured forces in contact order (or None: force rows skipped)."""
        spec, m, d = self.spec, self.model, self.data
        mk = self.markers(t_ref)
        kn = mk["knot"]
        out = {f: 0.0 for f in FAMILIES}

        def bump(name, val):
            out[name] = max(out[name], float(val))

        q = np.asarray(q, float)
        v = np.asarray(v, float)
        pin.computeJointJacobians(m, d, q)
        pin.updateFramePlacements(m, d)

        # ---- forces
        lam_c = {}
        if lam is not None:
            lam = np.asarray(lam, float)
            for i, cn in enumerate(self.contacts):
                lam_c[cn] = lam[self.lam_off[i]:self.lam_off[i + 1]]
            mu, eps, lammin = spec.mu, spec.friceps, spec.lammin
            for cn, f in lam_c.items():
                if not np.any(f):
                    continue
                fx, fy, fz = f[0], f[1], f[2]
                bump("cone", np.sqrt(fx**2 + fy**2 + eps**2) - eps - mu * fz)
                bump("normal", -fz)
                if len(f) == 6:
                    tx, ty, tz = f[3], f[4], f[5]
                    bump("cop", max(abs(ty) - spec.copx * fz, abs(tx) - spec.copy * fz))
                    bump("yaw", abs(tz) - spec.gammaz * fz)
                if cn not in mk["active"]:
                    bump("inactive_force", np.max(np.abs(f)))
            for cn in mk["active"]:
                if cn in kn.force_pins or cn in kn.floor_skip:
                    continue
                bump("lam_floor", lammin - lam_c[cn][2])

        # ---- geometry
        ends = {}
        for n, c in self.robot.capsules.items():
            A = d.oMf[self.robot.frame_id(c.frame)].act(c.a_local)
            B = d.oMf[self.robot.frame_id(c.b_frame)].act(c.b_local)
            ends[n] = (A, B, c.radius)
        for n in self.box_caps:
            A, B, rad = ends[n]
            for c, h in self.boxes:
                bump("keepout", separating_face_certificate(A, B, c, h, rad + self.margin)[0])
        for a_, b_, dsafe in spec.selfcol_pairs:
            A1, B1, _ = ends[a_]
            A2, B2, _ = ends[b_]
            bump("selfcol", selfcol_residual(A1, B1, A2, B2, dsafe)[0])
        if not mk["terminal"]:
            for fid, name in self.monitors:
                if self.bench == "fence" and not kn.flight_monitors and name in self.dim_of:
                    continue
                bump("clear", kn.monitor_lb - d.oMf[fid].translation[2])

        # ---- placement and no-slip
        for cn, tgt in mk["pins"].items():
            fid = self.fid_of[cn]
            bump("foot_pos", np.max(np.abs(d.oMf[fid].translation - np.asarray(tgt, float))))
            if self.dim_of[cn] == 6:
                bump("foot_pos", np.max(np.abs(pin.log3(d.oMf[fid].rotation))))
        for cn in mk["slip"]:
            J = pin.getFrameJacobian(m, d, self.fid_of[cn], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            rows = 6 if self.dim_of[cn] == 6 else 3
            fv = J[:rows] @ v
            bump("foot_vel", np.max(np.abs(fv)))
        # (touchdown-velocity rows of a "vz"/"xy" touchdown knot are knot
        # events: see `event_rows`)

        # ---- boxes
        qa = q[self.act_idx]
        bump("box_q", max(np.max(qa - (self.qmid + spec.qbox * self.qhalf)),
                          np.max((self.qmid - spec.qbox * self.qhalf) - qa)))
        va = v[self.act_idx]
        bump("box_v", np.max(np.abs(va) - spec.vlim * self.robot.velocity_limit))
        if self.att_idx and spec.attmax > 0:
            bump("box_att", np.max(np.abs(q[self.att_idx])) - spec.attmax)
        bump("base_z", -q[2])
        if self.lock_idx:
            bump("planar", np.max(np.abs(q[self.lock_idx])))
        if u is not None:
            bump("box_u", np.max(np.abs(np.asarray(u, float)) - self.u_scale * self.robot.effort_limit))

        # ---- data consistency: rigid inverse dynamics vs the logged (a, lam, u)
        if a is not None and u is not None and lam is not None:
            tau = pin.rnea(m, d, q, v, np.asarray(a, float)).copy()
            tau[self.act_idx] -= np.asarray(u, float)
            for cn, f in lam_c.items():
                if np.any(f):
                    J = pin.getFrameJacobian(m, d, self.fid_of[cn], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    tau -= J[:len(f)].T @ f
            bump("inverse_dynamics", np.max(np.abs(tau[self.free_rows])))
        return out

    # ------------------------------------------------------------ series
    def evaluate_series(self, t_ref, Q, V, A=None, U=None, LAM=None):
        """Row-wise `evaluate`; returns (families, N x nfam matrix)."""
        n = len(t_ref)
        M = np.zeros((n, len(FAMILIES)))
        for i in range(n):
            r = self.evaluate(float(t_ref[i]), Q[i], V[i],
                              None if A is None else A[i], None if U is None else U[i],
                              None if LAM is None else LAM[i])
            M[i] = [r[f] for f in FAMILIES]
        return list(FAMILIES), M

    def event_rows(self, t_ref, Q, V) -> dict:
        """Knot-only rows (`vz_up`: base vz >= vz_lb at the liftoff knot;
        `swing_z`: swing foot above z_lb at the clearance knots), each
        evaluated at the first sample whose reference clock reaches the knot.
        Returns {family: max violation} (0.0 when the plan has no such row)."""
        m, d = self.model, self.data
        out = {"vz_up": 0.0, "swing_z": 0.0, "land_vz": 0.0}
        t_ref = np.asarray(t_ref, float)
        for kn in self.plan.knots:
            td = bool(kn.touchdown_pins) and kn.touchdown_vel not in (None, "zero")
            if kn.vz_lb is None and kn.clear is None and not td:
                continue
            idx = np.where(t_ref >= kn.t - 1e-9)[0]
            if not idx.size:
                continue
            i = int(idx[0])
            q, v = np.asarray(Q[i], float), np.asarray(V[i], float)
            if kn.vz_lb is not None:
                out["vz_up"] = max(out["vz_up"], float(kn.vz_lb - v[2]))
            if kn.clear is not None:
                foot, zlb = kn.clear
                pin.framesForwardKinematics(m, d, q)
                out["swing_z"] = max(out["swing_z"], float(zlb - d.oMf[self.fid_of[foot]].translation[2]))
            if td:
                # a touchdown with a velocity cap (leap's landvz): xy rows for
                # "xy", and the vertical approach-speed cap for both
                pin.computeJointJacobians(m, d, q)
                pin.updateFramePlacements(m, d)
                for cn in kn.touchdown_pins:
                    J = pin.getFrameJacobian(m, d, self.fid_of[cn], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    fv = J[:3] @ v
                    if kn.touchdown_vel == "xy":
                        out["foot_vel_td"] = max(out.get("foot_vel_td", 0.0), float(np.max(np.abs(fv[:2]))))
                    if kn.land_vz > 0.0:
                        out["land_vz"] = max(out["land_vz"], float(-fv[2] - kn.land_vz))
        return out

    def unmodeled_force(self, q, lam, f_constraint) -> float:
        """max |f_constraint - sum J^T lam| over the free DOFs: the generalized
        force the simulator applied through constraints the optimizer's model
        does not have (joint-limit stops, non-foot contacts). N or N·m."""
        m, d = self.model, self.data
        q = np.asarray(q, float)
        pin.computeJointJacobians(m, d, q)
        pin.updateFramePlacements(m, d)
        r = np.asarray(f_constraint, float).copy()
        lam = np.asarray(lam, float)
        for i, cn in enumerate(self.contacts):
            f = lam[self.lam_off[i]:self.lam_off[i + 1]]
            if np.any(f):
                J = pin.getFrameJacobian(m, d, self.fid_of[cn], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                r -= J[:len(f)].T @ f
        return float(np.max(np.abs(r[self.free_rows])))

    def boundary_error(self, q, v, q_goal=None, v_goal=None) -> float:
        """`bc` analogue: max |x - x_goal| over the state (v only when q_goal is None)."""
        e = [np.abs(np.asarray(v, float) - (0.0 if v_goal is None else np.asarray(v_goal, float)))]
        if q_goal is not None:
            eq = np.asarray(q, float) - np.asarray(q_goal, float)
            eq[3:6] = (eq[3:6] + np.pi) % (2 * np.pi) - np.pi
            e.append(np.abs(eq))
        return float(np.max(np.concatenate(e)))


# ============================================== the benchmark registry's two answers
def terminal_condition(task: str, ref) -> dict:
    """{'q_goal': (nq,), 'v_goal': (nv,), 'foot_targets': {contact: (3,)}} --
    the problem's terminal condition the simulated end state is judged against
    (what `benchmarks.FenceBenchmark` / `TerrainBenchmark.terminal_condition`
    returned)."""
    nq = ref.nq
    if task == "fence":
        return dict(q_goal=np.asarray(ref.plan.x_goal[:nq], float),
                    v_goal=np.asarray(ref.plan.x_goal[nq:], float),
                    foot_targets={k: np.asarray(v, float) for k, v in ref.spec.feet2.items()},
                    q_goal_source="problem terminal rest pose (strictrest)")
    if task == "terrain":
        return dict(q_goal=np.asarray(ref.xs[-1, :nq], float), v_goal=np.zeros(ref.nv),
                    foot_targets={k: np.asarray(v, float) for k, v in ref.terminal_foot_targets.items()},
                    q_goal_source="solution final state (the terrain problem pins only the terminal footholds and v = 0)")
    raise KeyError(task)


def clearance(ref, q_samples):
    """Task-specific physical clearance of a q trajectory (fence: min
    capsule-to-box distance; terrain: topslack minus intrusion); None if N/A."""
    cs = ref.clearance_series(q_samples)
    return None if cs is None else float(np.min(cs))


# ============================================================ evaluate_tracking
def wrap_angles(e: np.ndarray, idx=slice(3, 6)) -> np.ndarray:
    e = np.array(e, float, copy=True)
    e[..., idx] = (e[..., idx] + np.pi) % (2 * np.pi) - np.pi
    return e


def rms(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def vmax(x) -> float:
    x = np.asarray(x, float)
    return float(np.max(np.abs(x))) if x.size else float("nan")


def _track_block(e_q, e_v, act, mask) -> dict:
    eq, ev = e_q[mask], e_v[mask]
    return dict(n=int(mask.sum()),
                base_pos_rmse=rms(eq[:, :3]), base_pos_max=vmax(eq[:, :3]),
                base_rot_rmse=rms(eq[:, 3:6]), base_rot_max=vmax(eq[:, 3:6]),
                joint_rmse=rms(eq[:, act]), joint_max=vmax(eq[:, act]),
                joint_vel_rmse=rms(ev[:, act]), joint_vel_max=vmax(ev[:, act]),
                base_vel_rmse=rms(ev[:, :3]), base_angvel_rmse=rms(ev[:, 3:6]),
                state_rmse=rms(np.hstack([eq, ev])))


TOL = 1e-3          # the benchmarks' accuracy criterion (max violation)


def evaluate(traj: TrackedTrajectory, ref, task: str, *, fine_dt: float = 0.01, clock: str = "ref",
             impact_window: float = 0.02):
    """Returns (summary dict -> evaluation.json, series dict -> evaluation.npz)."""
    meta = traj.meta
    robot = ref.robot
    act = traj.act_idx
    T = traj.duration
    in_ref = traj.ref_window()
    hold = ~in_ref
    names = traj.contacts
    nc = len(names)

    # ------------------------------------------------------------ tracking
    e_q = wrap_angles(traj.q - traj.q_ref)
    e_v = traj.v - traj.v_ref
    Qn, Vn = ref.sample(traj.t)
    e_q_nom = wrap_angles(traj.q - Qn)
    e_v_nom = traj.v - Vn
    in_ref_nom = traj.t <= T + 1e-9
    tracking = dict(controller_clock=_track_block(e_q, e_v, act, in_ref),
                    nominal_clock=_track_block(e_q_nom, e_v_nom, act, in_ref_nom),
                    hold=_track_block(e_q, e_v, act, hold) if hold.any() else None)
    phase_names = list(meta.get("phase_names", []))
    per_phase = {}
    for ph in sorted(set(traj.phase.tolist())):
        mask = (traj.phase == ph) & in_ref
        if mask.any():
            nm = phase_names[ph] if ph < len(phase_names) else str(ph)
            per_phase[nm] = _track_block(e_q, e_v, act, mask)
    # phases grouped by KIND (the alphabetic part of the name: a terrain walk's
    # support0..8 / swing1..8 become "support" / "swing"; the fence's four
    # phases are their own kinds)
    per_phase_kind = {}
    kinds = {}
    for ph in sorted(set(traj.phase.tolist())):
        nm = phase_names[ph] if ph < len(phase_names) else str(ph)
        kinds.setdefault("".join(ch for ch in nm if not ch.isdigit()).strip() or nm, []).append(ph)
    for kind, phs in kinds.items():
        mask = np.isin(traj.phase, phs) & in_ref
        if mask.any():
            per_phase_kind[kind] = _track_block(e_q, e_v, act, mask)
    # feet
    fp = traj.foot_pos.reshape(traj.n, nc, 3)
    fpr = traj.foot_pos_ref.reshape(traj.n, nc, 3)
    foot_err = np.linalg.norm(fp - fpr, axis=2)
    sup = traj.support > 0.5
    feet = {}
    for k, c in enumerate(names):
        sw = in_ref & ~sup[:, k]
        feet[c] = dict(rmse=rms(foot_err[in_ref, k]), max=vmax(foot_err[in_ref, k]),
                       swing_rmse=rms(foot_err[sw, k]) if sw.any() else float("nan"),
                       swing_max=vmax(foot_err[sw, k]) if sw.any() else float("nan"))
    feet["all"] = dict(rmse=rms(foot_err[in_ref]), max=vmax(foot_err[in_ref]))
    # torque
    tau_max = np.asarray(meta["effort_limit"], float)
    tau_max = np.where(np.isfinite(tau_max) & (tau_max > 0), tau_max, np.inf)
    e_u = traj.u - traj.u_ref
    torque = dict(rmse=rms(e_u[in_ref]), max=vmax(e_u[in_ref]),
                  rmse_normalized=rms((e_u / tau_max)[in_ref]),
                  ratio_max=float(np.max(np.abs(traj.u) / tau_max)),
                  clipped_frac=float(np.mean(np.any(np.abs(traj.u) >= tau_max - 1e-9, axis=1))),
                  ref_ratio_max=float(np.max(np.abs(traj.u_ref) / tau_max)),
                  effort_sim=float(np.trapezoid(np.sum(traj.u**2, axis=1)[in_ref], traj.t[in_ref])),
                  effort_ref=float(np.trapezoid(np.sum(traj.u_ref**2, axis=1)[in_ref], traj.t[in_ref])))
    # forces (force rows only, during the plan's stance, normalized by m g)
    mg = float(meta["total_mass"]) * 9.81
    off = traj.lam_off
    forces = {}
    sched = traj.scheduled > 0.5
    for k, c in enumerate(names):
        mask = in_ref & sched[:, k]
        f = traj.lam[:, off[k]:off[k] + 3]
        fr = traj.lam_ref[:, off[k]:off[k] + 3]
        forces[c] = dict(rmse_over_mg=rms((f - fr)[mask]) / mg if mask.any() else float("nan"),
                         fz_rmse_over_mg=rms((f[:, 2] - fr[:, 2])[mask]) / mg if mask.any() else float("nan"),
                         fz_max_sim=float(f[:, 2].max()), fz_max_ref=float(fr[:, 2].max()),
                         stance_ticks=int(mask.sum()))
    fz_tot = sum(traj.lam[:, off[k] + 2] for k in range(nc))
    fz_tot_ref = sum(traj.lam_ref[:, off[k] + 2] for k in range(nc))
    forces["total_fz_rmse_over_mg"] = rms((fz_tot - fz_tot_ref)[in_ref]) / mg

    # ------------------------------------------------------------ fine grid
    trf = traj.resample(fine_dt)
    checker = ConstraintChecker(ref.plan)
    tt = trf.t_ref if clock == "ref" else trf.t
    fams, M = checker.evaluate_series(tt, trf.q, trf.v, trf.a, trf.u, trf.lam)
    _, Mr = checker.evaluate_series(tt, trf.q_ref, trf.v_ref, trf.a_ref, trf.u_ref, trf.lam_ref)
    events = checker.event_rows(tt, trf.q, trf.v)
    events_ref = checker.event_rows(tt, trf.q_ref, trf.v_ref)
    inref_f = trf.ref_window()

    def fam_max(Mat, mask, ev):
        out = {f: (float(Mat[mask, i].max()) if mask.any() else 0.0) for i, f in enumerate(fams)}
        out.update({k: float(v) for k, v in ev.items()})
        prob = {k: v for k, v in out.items() if k in PROBLEM_FAMILIES or k in ev}
        out["overall"] = max(prob.values())
        out["limiting"] = max(prob, key=prob.get)
        return out
    # Impact windows: the plastic landing is an impulsive force spike that no
    # force row of the problem is meant to hold at, so every force family is
    # also reported with the samples within `impact_window` of a physical
    # contact-make event removed, and with the fraction of the window above
    # the benchmark's 1e-3 criterion.
    make = np.any((traj.measured[1:] > 0.5) & (traj.measured[:-1] < 0.5), axis=1)
    t_make = traj.t[1:][make]
    impact = np.zeros(trf.n, bool)
    for tm in t_make:
        impact |= np.abs(trf.t - tm) <= impact_window
    violations_fine = fam_max(M, inref_f, events)
    violations_fine_hold = fam_max(M, ~inref_f, {}) if (~inref_f).any() else None
    violations_fine_reference = fam_max(Mr, inref_f, events_ref)
    violations_fine_no_impact = fam_max(M, inref_f & ~impact, events) if (inref_f & ~impact).any() else None
    violations_frac_above_tol = {f: float(np.mean(M[inref_f, i] > TOL)) for i, f in enumerate(fams)}
    violations_frac_above_tol["any_problem_row"] = float(np.mean(
        np.max(M[inref_f][:, [fams.index(f) for f in PROBLEM_FAMILIES]], axis=1) > TOL))

    # ------------------------------------------------------------ final state
    tc = terminal_condition(task, ref)
    q_goal, v_goal, targets = tc["q_goal"], tc["v_goal"], tc["foot_targets"]
    idx_T = np.where(traj.t_ref >= T - 1e-9)[0]
    i_T = int(idx_T[0]) if idx_T.size else traj.n - 1

    def final_block(i: int) -> dict:
        q, v = traj.q[i], traj.v[i]
        out = dict(t=float(traj.t[i]), t_ref=float(traj.t_ref[i]),
                   speed_norm=float(np.linalg.norm(v - v_goal)), base_speed=float(np.linalg.norm(v[:3])),
                   joint_speed_max=vmax(v[act]))
        if q_goal is not None:
            e = wrap_angles(q - q_goal)
            out.update(base_pos_err=float(np.linalg.norm(e[:3])), base_rot_err=vmax(e[3:6]),
                       joint_err_max=vmax(e[act]), joint_err_rms=rms(e[act]),
                       terminal_err=checker.boundary_error(q, v, q_goal, v_goal))
        else:
            out.update(base_pos_err=float("nan"), base_rot_err=float("nan"), joint_err_max=float("nan"),
                       joint_err_rms=float("nan"), terminal_err=checker.boundary_error(q, v, None, v_goal))
        fe = {c: float(np.linalg.norm(fp[i, k] - targets[c])) for k, c in enumerate(names) if c in targets}
        out["foot_err"] = fe
        out["foot_err_max"] = max(fe.values()) if fe else float("nan")
        return out
    final_state = dict(at_T=final_block(i_T), end=final_block(traj.n - 1),
                       q_goal_source=tc.get("q_goal_source", "problem terminal state"))

    # ------------------------------------------------------------ physical
    step = max(1, int(round(0.005 / fine_dt))) if fine_dt < 0.005 else 1
    unmod = np.array([checker.unmodeled_force(trf.q[i], trf.lam[i], trf.f_constraint[i]) for i in range(trf.n)])
    physical = dict(clearance_sim=clearance(ref, trf.q[::step]),
                    unmodeled_force_max=float(unmod[inref_f].max()) if inref_f.any() else 0.0,
                    unmodeled_force_ticks_frac=float(np.mean(unmod[inref_f] > 1.0)) if inref_f.any() else 0.0,
                    clearance_ref=clearance(ref, trf.q_ref[::step]),
                    capsule_obstacle_hit_ticks=int(np.sum(traj.capsule_hits > 0)),
                    min_base_z=float(traj.q[:, 2].min()), min_base_z_ref=float(traj.q_ref[:, 2].min()))
    outcome = dict(success=bool(meta.get("success", False)), fell=bool(meta.get("fell", False)),
                   unstable=bool(meta.get("unstable", False)), time_shift=float(meta.get("time_shift", 0.0)),
                   frozen_ticks=int(np.sum(traj.frozen > 0.5)), simulated_s=float(traj.t[-1]))

    summary = dict(
        eval_version=EVAL_VERSION, benchmark=meta["benchmark"], method=meta["method"], robot=meta["robot"],
        archive=meta["archive"], controller=meta.get("controller"), duration=T, hold=float(meta["hold"]),
        control_dt=float(meta["control_dt"]), sim_dt=float(meta["sim_dt"]), fine_dt=float(fine_dt),
        constraint_clock=clock, n_fine=int(trf.n), n_fine_ref_window=int(inref_f.sum()),
        outcome=outcome, tracking=tracking, per_phase=per_phase, per_phase_kind=per_phase_kind,
        feet=feet, torque=torque, forces=forces,
        violations_fine=violations_fine, violations_fine_hold=violations_fine_hold,
        violations_fine_reference=violations_fine_reference, violations_fine_no_impact=violations_fine_no_impact,
        violations_frac_above_tol=violations_frac_above_tol, impact_window=float(impact_window),
        n_impact_events=int(len(t_make)), events=events,
        final_state=final_state, physical=physical,
        families=list(fams), family_units=UNITS, phase_names=phase_names, contacts=names, contact_dims=traj.contact_dims,
        actuated_joints=meta.get("actuated_joints"), gains=meta.get("gains"), mujoco=meta.get("mujoco"),
        determinism=meta.get("determinism"))
    series = dict(t=traj.t, t_ref=traj.t_ref, in_ref=in_ref, phase=traj.phase,
                  e_q=e_q, e_v=e_v, e_q_nom=e_q_nom, e_v_nom=e_v_nom, foot_err=foot_err,
                  support=traj.support, scheduled=traj.scheduled, measured=traj.measured,
                  u=traj.u, u_ref=traj.u_ref, lam=traj.lam, lam_ref=traj.lam_ref, act_idx=act,
                  t_fine=trf.t, t_ref_fine=tt, in_ref_fine=inref_f, impact_fine=impact, violations=M, violations_ref=Mr,
                  unmodeled_force=unmod,
                  families=np.array(fams), contacts=np.array(names), contact_dims=np.array(traj.contact_dims))
    return summary, series


def evaluate_dir(out_dir: Path, *, task: str | None = None, ref=None, archive=None, problem_yaml=None,
                 fine_dt: float = 0.01, clock: str = "ref", verbose: bool = True, impact_window: float = 0.02):
    """Evaluate `out_dir/trajectory.npz`; writes evaluation.json/.npz there.
    The reference is `ref` if given, else reloaded from the archive recorded
    in the trajectory (`archive` / `problem_yaml` override; the default problem
    yaml is the copy `track_archive` wrote next to the rollout)."""
    from .reference import load_leap_reference
    out_dir = Path(out_dir)
    path = out_dir / "trajectory.npz"
    if not path.exists():
        if verbose:
            print(f"no trajectory in {out_dir}")
        return None
    traj = TrackedTrajectory.load(path)
    task = task or traj.meta["benchmark"]
    if ref is None:
        py = problem_yaml or traj.meta.get("problem_yaml") or None
        if py is not None and not Path(py).exists():
            py = None
        ref = load_leap_reference(task, traj.meta["robot"], archive or traj.meta["archive"],
                                  problem_yaml=py, work_dir=out_dir)
    summary, series = evaluate(traj, ref, task, fine_dt=fine_dt, clock=clock, impact_window=impact_window)
    (out_dir / "evaluation.json").write_text(json.dumps(summary, indent=1, default=_json))
    np.savez_compressed(out_dir / "evaluation.npz", **series)
    if verbose:
        print_summary(summary)
        print(f"wrote {out_dir / 'evaluation.json'}  {out_dir / 'evaluation.npz'}")
    return summary


def print_summary(s: dict):
    tc, tn = s["tracking"]["controller_clock"], s["tracking"]["nominal_clock"]
    fe, fT = s["final_state"]["end"], s["final_state"]["at_T"]
    v, vr = s["violations_fine"], s["violations_fine_reference"]
    o = s["outcome"]
    print(f"[{s['benchmark']}/{s['method']}/{s['robot']}] {'SUCCESS' if o['success'] else ('FELL' if o['fell'] else 'FAIL')}"
          f"  T={s['duration']:.2f}s hold={s['hold']:.1f}s  ref clock shift {o['time_shift']*1e3:+.0f} ms")
    print(f"  tracking (controller clock): base {tc['base_pos_rmse']*1e3:.1f} mm / {tc['base_rot_rmse']:.4f} rad, "
          f"joints {tc['joint_rmse']:.4f} rad, joint vel {tc['joint_vel_rmse']:.3f} rad/s, feet {s['feet']['all']['rmse']*1e3:.1f} mm")
    print(f"  tracking (nominal clock):    base {tn['base_pos_rmse']*1e3:.1f} mm / {tn['base_rot_rmse']:.4f} rad, "
          f"joints {tn['joint_rmse']:.4f} rad")
    print(f"  torque: rmse {s['torque']['rmse']:.2f} N·m ({s['torque']['rmse_normalized']:.3f} of limit), "
          f"max ratio {s['torque']['ratio_max']:.2f}, clipped {s['torque']['clipped_frac']:.1%}; "
          f"total fz rmse {s['forces']['total_fz_rmse_over_mg']:.3f} mg")
    print(f"  fine grid dt={s['fine_dt']:g}: max violation {v['overall']:.2e} ({v['limiting']}); "
          f"reference on the same grid {vr['overall']:.2e} ({vr['limiting']}); "
          f"inverse-dynamics residual {v['inverse_dynamics']:.2e}")
    if s.get("violations_fine_no_impact"):
        vn = s["violations_fine_no_impact"]
        print(f"  excluding ±{s['impact_window']*1e3:.0f} ms around the {s['n_impact_events']} contact-make events: "
              f"{vn['overall']:.2e} ({vn['limiting']}); rows above 1e-3 on {s['violations_frac_above_tol']['any_problem_row']:.1%} of the window")
    top = sorted(((k, val) for k, val in v.items() if k in s["families"] or k in s["events"]),
                 key=lambda kv: -kv[1])[:6]
    print("  " + ", ".join(f"{k} {val:.1e}" for k, val in top))
    print(f"  final at T: base {fT['base_pos_err']*1e3:.1f} mm / {fT['base_rot_err']:.4f} rad, joints max {fT['joint_err_max']:.4f} rad, "
          f"speed {fT['speed_norm']:.2e}, feet {fT['foot_err_max']*1e3:.1f} mm, terminal_err {fT['terminal_err']:.2e}")
    print(f"  final at end: base {fe['base_pos_err']*1e3:.1f} mm / {fe['base_rot_err']:.4f} rad, joints max {fe['joint_err_max']:.4f} rad, "
          f"speed {fe['speed_norm']:.2e}, feet {fe['foot_err_max']*1e3:.1f} mm, terminal_err {fe['terminal_err']:.2e}")
    p = s["physical"]
    if p["clearance_sim"] is not None:
        print(f"  clearance sim/ref {p['clearance_sim']*1e3:+.1f}/{p['clearance_ref']*1e3:+.1f} mm, "
              f"capsule-obstacle hit ticks {p['capsule_obstacle_hit_ticks']}")
    print(f"  unmodeled constraint force (joint stops etc.): max {p['unmodeled_force_max']:.1f}, "
          f"above 1 N·m on {p['unmodeled_force_ticks_frac']:.1%} of the reference window")
