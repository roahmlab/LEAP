"""Plot, animate, independently re-verify and report a LEAP solution archive.

    python report_solution.py <archive.h5> --out <dir> [--dt 0.005] [--no-gif] [--urdf <path>]

Writes into <dir>:
    states.png, velocities.png, torques.png, forces.png, violations.png   the trajectory
    animation.gif                                                         a stick-figure replay (x-z projection)
    report.md                                                             the verification table and the solve summary

The verification is INDEPENDENT of the solver: the archive's node values are interpolated
here (barycentric on the Chebyshev-Gauss-Lobatto nodes, Bernstein for the force curve) and
every physical quantity is recomputed with pinocchio's Python bindings from the URDF the
archive names: the equations of motion on a fine time grid, the contact kinematics, the
friction cone, the hardware limits, the junction conditions (continuity and the plastic
impact map) and the rest conditions. Only numpy, h5py, matplotlib and pinocchio are needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import pinocchio as pin
except ImportError:  # the plots still work without pinocchio
    pin = None

GRAVITY = 9.81


# --- the archive ---------------------------------------------------------------------------

@dataclass
class Segment:
    index: int
    label: str
    phase: int
    t0: float
    T: float
    taus: np.ndarray          # (Nn,) ascending
    Q: np.ndarray             # (nq, Nn)
    V: np.ndarray
    A: np.ndarray
    U: np.ndarray             # (nu, Nn)
    lam_nodes: np.ndarray     # (nlam, Nn)
    lam_ctrl: np.ndarray | None   # (nlam, Nc) physical control polygon (already scaled)
    cons: np.ndarray          # (Nn, 7)
    contacts: list[dict]      # [{name, dim}]
    reset_after: str

    @property
    def degree(self) -> int:
        return self.taus.size - 1


@dataclass
class Solution:
    meta: dict
    segments: list[Segment]
    rows_g: np.ndarray
    rows_term: np.ndarray
    rows_ineq: np.ndarray
    terms: list[str]

    @property
    def total_time(self) -> float:
        return self.segments[-1].t0 + self.segments[-1].T


def load(path: Path) -> Solution:
    with h5py.File(path, "r") as f:
        meta = json.loads(f["/meta_json"][()].decode())
        segs = []
        for s in meta["segments"]:
            g = f[f"/segments/{s['index']:02d}"]
            lam_ctrl = None
            if "lam_ctrl" in g:
                lam_ctrl = g["lam_ctrl"][()] * g["lam_scale"][()][None, :]
            segs.append(Segment(
                index=s["index"], label=s.get("label", f"segment {s['index']}"), phase=s["parent_mode"], t0=s["t0"], T=s["T"],
                taus=g["taus"][()], Q=g["Q"][()], V=g["V"][()], A=g["A"][()], U=g["U"][()], lam_nodes=g["lam_nodes"][()],
                lam_ctrl=lam_ctrl, cons=g["cons"][()], contacts=s["active_contacts"], reset_after=s["reset_after"]))
        return Solution(meta=meta, segments=segs, rows_g=f["/rows/g"][()], rows_term=f["/rows/term"][()],
                        rows_ineq=f["/rows/ineq"][()], terms=list(meta["terms"]))


# --- interpolation ---------------------------------------------------------------------------

def barycentric_weights(n: int) -> np.ndarray:
    """Barycentric weights of the degree-n CGL nodes (ascending), up to a common factor."""
    w = np.array([(-1.0) ** k for k in range(n + 1)])
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def interp_weights(taus: np.ndarray, tau: float) -> np.ndarray:
    """Lagrange basis values at tau for the nodes `taus` (barycentric form)."""
    n = taus.size - 1
    w = barycentric_weights(n)
    d = tau - taus
    hit = np.where(np.abs(d) < 1e-14)[0]
    if hit.size:
        out = np.zeros_like(taus)
        out[hit[0]] = 1.0
        return out
    c = w / d
    return c / c.sum()


def diff_matrix(taus: np.ndarray) -> np.ndarray:
    """Spectral differentiation matrix on the nodes: (D X')' = dX/dtau."""
    n = taus.size
    w = barycentric_weights(n - 1)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (taus[i] - taus[j])
        D[i, i] = -D[i].sum()
    return D


def bernstein(nc: int, s: float) -> np.ndarray:
    n = nc - 1
    return np.array([math.comb(n, j) * s ** j * (1.0 - s) ** (n - j) for j in range(nc)])


def state_at(seg: Segment, tau: float):
    w = interp_weights(seg.taus, tau)
    q, v, a, u = seg.Q @ w, seg.V @ w, seg.A @ w, seg.U @ w
    if seg.lam_ctrl is not None:
        lam = seg.lam_ctrl @ bernstein(seg.lam_ctrl.shape[1], 0.5 * (tau + 1.0))
    else:
        lam = np.zeros(seg.lam_nodes.shape[0])
    return q, v, a, u, lam


def fine_grid(sol: Solution, dt: float):
    """(t, segment index, tau) on an equidistant physical-time grid, every segment sampled."""
    out = []
    for seg in sol.segments:
        n = max(2, int(round(seg.T / dt)) + 1)
        for k in range(n):
            tau = -1.0 + 2.0 * k / (n - 1)
            out.append((seg.t0 + 0.5 * (tau + 1.0) * seg.T, seg.index, tau))
    return out


# --- the robot (pinocchio) ----------------------------------------------------------------------

class Robot:
    def __init__(self, urdf: Path, meta: dict):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        names = [self.model.names[i] for i in range(1, self.model.njoints)]
        if names != list(meta["model"]["joint_names"]) or self.model.nq != meta["model"]["nq"]:
            raise RuntimeError(f"the URDF's joints {names} do not match the archive's {meta['model']['joint_names']}")
        self.actuated = list(meta["model"]["actuated_idx"])
        self.contact_frames = {c["name"]: self.model.getFrameId(c["name"]) for c in meta["model"]["contacts"]}
        self.contact_dim = {c["name"]: int(c["dim"]) for c in meta["model"]["contacts"]}
        self.monitors = []
        try:  # the robot yaml (embedded) names the ground keep-out frames
            import yaml
            ry = yaml.safe_load(meta["robot"]["yaml"])
            self.monitors = [(n, self.model.getFrameId(n)) for n in ry.get("monitor_frames", []) if self.model.existFrame(n)]
        except Exception:
            pass
        self.capsules = [(c["frame"], np.array(c["a"]), np.array(c["b"]), c["radius"]) for c in meta.get("collision", {}).get("capsules", [])]
        self.motor_model = meta["robot"].get("motor_model", "none")
        # a planar robot described by a 3-D URDF: the problem pins the out-of-plane
        # coordinates (`problem.planarlock`) and drops their dynamics rows, so those rows
        # are not part of the problem and are reported separately
        self.locked_rows = []
        try:
            import yaml
            py = yaml.safe_load(meta.get("problem", {}).get("yaml", "") or "") or {}
            for name in (py.get("problem", {}) or {}).get("planarlock", []) or []:
                if self.model.existJointName(name):
                    self.locked_rows.append(self.model.idx_vs[self.model.getJointId(name)])
        except Exception:
            self.locked_rows = []

    def contact_jacobian(self, q, name):
        fid = self.contact_frames[name]
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
        return J[: self.contact_dim[name]]  # [linear; angular] rows, force first

    def dynamics_residual(self, q, v, a, u, lam, contacts):
        """tau(q, v, a) - B u - sum_c J_c' lam_c (no motor loss)."""
        r = pin.rnea(self.model, self.data, q, v, a).copy()
        for j, i in enumerate(self.actuated):
            r[i] -= u[j]
        o = 0
        for c in contacts:
            d = int(c["dim"])
            r -= self.contact_jacobian(q, c["name"]).T @ lam[o:o + d]
            o += d
        return r

    def frame_pos_vel(self, q, v, fid):
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        p = self.data.oMf[fid].translation.copy()
        vel = pin.getFrameVelocity(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
        return p, np.array(vel.linear), np.array(vel.angular)

    def impact(self, q, vminus, contacts):
        """Plastic (sticking) impact of the named contacts: v+ = v- - M^-1 J' (J M^-1 J')^-1 J v-."""
        M = pin.crba(self.model, self.data, q)
        M = np.triu(M) + np.triu(M, 1).T
        J = np.vstack([self.contact_jacobian(q, c["name"]) for c in contacts])
        Minv = np.linalg.inv(M)
        G = J @ Minv @ J.T
        G += 1e-10 * np.eye(G.shape[0])
        return vminus - Minv @ J.T @ np.linalg.solve(G, J @ vminus)

    def chain_points(self, q):
        """World positions of every joint origin and its parent (the stick figure)."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        segs = []
        for j in range(1, self.model.njoints):
            p = self.model.parents[j]
            if p >= 1:
                segs.append((self.data.oMi[p].translation.copy(), self.data.oMi[j].translation.copy()))
        caps = []
        for frame, a, b, r in self.capsules:
            fid = self.model.getFrameId(frame)
            M = self.data.oMf[fid]
            caps.append((M.act(a), M.act(b), r))
        return segs, caps


# --- plots ----------------------------------------------------------------------------------------

def mode_spans(sol: Solution):
    spans = []
    for seg in sol.segments:
        spans.append((seg.t0, seg.t0 + seg.T, len(seg.contacts) > 0))
    return spans


def shade_modes(ax, sol: Solution):
    for t0, t1, stance in mode_spans(sol):
        ax.axvspan(t0, t1, color="#cfe3f3" if stance else "#fbe4c8", alpha=0.35, lw=0)
    for seg in sol.segments[1:]:
        ax.axvline(seg.t0, color="0.6", lw=0.4)


def sample_curves(sol: Solution, dt: float):
    rows = []
    for t, si, tau in fine_grid(sol, dt):
        q, v, a, u, lam = state_at(sol.segments[si], tau)
        rows.append((t, si, tau, q, v, a, u, lam))
    return rows


def plot_all(sol: Solution, rows, out: Path, robot: Robot | None, limits: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r[0] for r in rows])
    Q = np.array([r[3] for r in rows]).T
    V = np.array([r[4] for r in rows]).T
    U = np.array([r[6] for r in rows]).T
    names = sol.meta["model"]["joint_names"]
    act = sol.meta["model"]["actuated_idx"]
    nq = Q.shape[0]
    # states
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for i in range(min(6, nq)):
        if np.max(np.abs(Q[i])) > 1e-9:
            axes[0].plot(t, Q[i], label=names[i])
    axes[0].set_ylabel("base (m, rad)")
    axes[0].legend(ncol=3, fontsize=8)
    shade_modes(axes[0], sol)
    for i in act:
        axes[1].plot(t, Q[i], label=names[i])
    axes[1].set_ylabel("joint angles (rad)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(ncol=4, fontsize=8)
    shade_modes(axes[1], sol)
    fig.suptitle("Configuration (blue: stance, orange: flight; thin lines: element junctions)")
    fig.tight_layout()
    fig.savefig(out / "states.png", dpi=130)
    plt.close(fig)
    # velocities
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for i in range(min(6, nq)):
        if np.max(np.abs(V[i])) > 1e-9:
            axes[0].plot(t, V[i], label=names[i])
    axes[0].set_ylabel("base velocity (m/s, rad/s)")
    axes[0].legend(ncol=3, fontsize=8)
    shade_modes(axes[0], sol)
    for j, i in enumerate(act):
        line, = axes[1].plot(t, V[i], label=names[i])
        if "velocity" in limits:
            axes[1].axhline(limits["velocity"][j], color=line.get_color(), ls=":", lw=0.8)
            axes[1].axhline(-limits["velocity"][j], color=line.get_color(), ls=":", lw=0.8)
    axes[1].set_ylabel("joint velocity (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(ncol=4, fontsize=8)
    shade_modes(axes[1], sol)
    fig.suptitle("Velocities (dotted: URDF limits)")
    fig.tight_layout()
    fig.savefig(out / "velocities.png", dpi=130)
    plt.close(fig)
    # torques
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, i in enumerate(act):
        line, = ax.plot(t, U[j], label=names[i])
        if "effort" in limits:
            ax.axhline(limits["effort"][j], color=line.get_color(), ls=":", lw=0.8)
            ax.axhline(-limits["effort"][j], color=line.get_color(), ls=":", lw=0.8)
    ax.set_ylabel("torque (N m)")
    ax.set_xlabel("time (s)")
    ax.legend(ncol=4, fontsize=8)
    shade_modes(ax, sol)
    fig.suptitle("Joint torques (dotted: URDF effort limits)")
    fig.tight_layout()
    fig.savefig(out / "torques.png", dpi=130)
    plt.close(fig)
    # forces
    contacts = sol.meta["model"]["contacts"]
    fig, axes = plt.subplots(len(contacts), 1, figsize=(9, 3.2 * len(contacts)), sharex=True, squeeze=False)
    for ci, c in enumerate(contacts):
        ax = axes[ci, 0]
        tc, fc = [], []
        for r in rows:
            seg = sol.segments[r[1]]
            names_active = [a["name"] for a in seg.contacts]
            if c["name"] in names_active:
                o = sum(int(a["dim"]) for a in seg.contacts[: names_active.index(c["name"])])
                tc.append(r[0])
                fc.append(r[7][o:o + int(c["dim"])])
            else:
                tc.append(r[0])
                fc.append(np.zeros(int(c["dim"])))
        F = np.array(fc).T
        ax.plot(tc, F[0], label="fx")
        ax.plot(tc, F[1], label="fy")
        ax.plot(tc, F[2], label="fz")
        ax.plot(tc, np.hypot(F[0], F[1]), "k--", lw=0.8, label="|f_t|")
        if "mu" in limits:
            ax.plot(tc, limits["mu"] * F[2], "r:", lw=0.8, label="mu fz (cone)")
        ax.set_ylabel(f"{c['name']} force (N)")
        ax.legend(ncol=5, fontsize=8)
        shade_modes(ax, sol)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Contact forces (Bezier curve between the nodes)")
    fig.tight_layout()
    fig.savefig(out / "forces.png", dpi=130)
    plt.close(fig)


def plot_violations(sol: Solution, checks: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    # per-node table from the archive (the solver's own residuals at the zero-dual point)
    cols = ["dynamics", "foot", "boundary", "ineq", "other", "kin1", "kin2"]
    for ci, name in enumerate(cols):
        tt, vv = [], []
        for seg in sol.segments:
            for k, tau in enumerate(seg.taus):
                tt.append(seg.t0 + 0.5 * (tau + 1.0) * seg.T)
                vv.append(seg.cons[k, ci])
        vv = np.array(vv)
        if np.all(np.isnan(vv)) or np.nanmax(vv) <= 0:
            continue
        axes[0].semilogy(tt, np.maximum(vv, 1e-16), ".", ms=4, label=name)
    axes[0].axhline(1e-3, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("node violation (solver)")
    axes[0].legend(ncol=4, fontsize=8)
    shade_modes(axes[0], sol)
    for key, (tt, vv, label) in checks.items():
        axes[1].semilogy(tt, np.maximum(np.abs(vv), 1e-16), lw=1.0, label=label)
    axes[1].axhline(1e-3, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("fine-grid residual (pinocchio)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(ncol=3, fontsize=8)
    shade_modes(axes[1], sol)
    fig.suptitle("Constraint residuals: at the nodes (top) and recomputed on the fine grid (bottom); dashed: 1e-3")
    fig.tight_layout()
    fig.savefig(out / "violations.png", dpi=130)
    plt.close(fig)


def animate(sol: Solution, rows, robot: Robot, out: Path, fps: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    tgrid = np.array([r[0] for r in rows])
    frames_t = np.arange(0.0, sol.total_time + 1e-9, 1.0 / fps)
    frames = [rows[int(np.argmin(np.abs(tgrid - tf)))] for tf in frames_t]
    xs, zs = [], []
    for r in frames:
        segs, caps = robot.chain_points(r[3])
        for a, b in segs:
            xs += [a[0], b[0]]
            zs += [a[2], b[2]]
    xlim = (min(xs) - 0.3, max(xs) + 0.3)
    zlim = (-0.05, max(zs) + 0.25)
    fig, ax = plt.subplots(figsize=(5, 5 * (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])))
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="k", lw=1.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    chain_lines = []
    cap_lines = []
    title = ax.set_title("")
    trail, = ax.plot([], [], "-", color="0.6", lw=0.8)
    base_x, base_z = [], []

    def draw(i):
        nonlocal chain_lines, cap_lines
        r = frames[i]
        for ln in chain_lines + cap_lines:
            ln.remove()
        chain_lines, cap_lines = [], []
        segs, caps = robot.chain_points(r[3])
        for a, b, rad in caps:
            ln, = ax.plot([a[0], b[0]], [a[2], b[2]], "-", color="#4a90d9", alpha=0.35, lw=rad * 500.0 / (xlim[1] - xlim[0]), solid_capstyle="round")
            cap_lines.append(ln)
        for a, b in segs:
            ln, = ax.plot([a[0], b[0]], [a[2], b[2]], "o-", color="k", lw=2, ms=3)
            chain_lines.append(ln)
        base_x.append(r[3][0])
        base_z.append(r[3][2])
        trail.set_data(base_x, base_z)
        seg = sol.segments[r[1]]
        title.set_text(f"t = {r[0]:.2f} s   {seg.label}   {'stance' if seg.contacts else 'flight'}")
        return chain_lines + cap_lines + [trail, title]

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out / "animation.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)


# --- verification -------------------------------------------------------------------------------

def verify(sol: Solution, rows, robot: Robot, mu: float, tol: float):
    """Independent checks; returns (table rows, curves for the plot)."""
    results = []
    curves = {}
    weight = None
    # --- equations of motion on the fine grid
    tt, dyn, dyn_locked = [], [], []
    keep = np.array([i for i in range(robot.model.nv) if i not in robot.locked_rows], dtype=int)
    for t, si, tau, q, v, a, u, lam in rows:
        seg = sol.segments[si]
        r = robot.dynamics_residual(q, v, a, u, lam, seg.contacts)
        tt.append(t)
        dyn.append(np.max(np.abs(r[keep])))
        if robot.locked_rows:
            dyn_locked.append(np.max(np.abs(r[robot.locked_rows])))
        if weight is None:
            g = pin.rnea(robot.model, robot.data, q, np.zeros_like(v), np.zeros_like(a))
            weight = abs(g[2])
    dyn = np.array(dyn)
    curves["dyn"] = (tt, dyn, "equations of motion max|r| (N, N m)")
    note = "every stance and flight sample; torque and force curves interpolated as the solver does"
    if robot.locked_rows:
        note += (f"; the {len(robot.locked_rows)} planar-locked rows are not part of the problem (pinned coordinates, dynamics rows "
                 f"dropped) and are left out here: their residual is {max(dyn_locked):.2e}")
    results.append(("equations of motion on the fine grid, max |residual| (N or N m)", dyn.max(), tol, note))
    # --- kinematic consistency at the nodes (spectral derivative)
    kin = 0.0
    for seg in sol.segments:
        D = diff_matrix(seg.taus)
        a = 2.0 / seg.T
        kin = max(kin, np.max(np.abs(seg.V - a * seg.Q @ D.T)), np.max(np.abs(seg.A - a * seg.V @ D.T)))
    results.append(("v = dq/dt and a = dv/dt at the nodes (spectral derivative)", kin, tol, "the kinematic rows are constraints of the flow (part of the node gate), not an identity"))
    # --- contacts: stance foot on the floor and not sliding; flight: monitors above the floor
    tt_c, foot_h, foot_s = [], [], []
    clear = 0.0
    for t, si, tau, q, v, a, u, lam in rows:
        seg = sol.segments[si]
        for c in seg.contacts:
            p, pdot, _ = robot.frame_pos_vel(q, v, robot.contact_frames[c["name"]])
            if tt_c and t - tt_c[-1] > 3.0 * (rows[1][0] - rows[0][0]):  # a gap (flight): break the plotted line
                tt_c.append(t)
                foot_h.append(np.nan)
                foot_s.append(np.nan)
            tt_c.append(t)
            foot_h.append(abs(p[2]))
            foot_s.append(np.linalg.norm(pdot))
        if not seg.contacts:
            for name, fid in robot.monitors:
                p, _, _ = robot.frame_pos_vel(q, v, fid)
                clear = max(clear, -p[2])
    if tt_c:
        curves["foot_h"] = (tt_c, np.array(foot_h), "stance foot height |z| (m)")
        curves["foot_s"] = (tt_c, np.array(foot_s), "stance foot speed (m/s)")
        results.append(("stance contact height |z| on the fine grid (m)", np.nanmax(foot_h), tol, "the foot stays on the floor"))
        results.append(("stance contact speed on the fine grid (m/s)", np.nanmax(foot_s), tol, "no slip between the nodes"))
    results.append(("flight: monitor frames below the floor (m)", clear, tol, "ground keep-out of the monitor frames"))
    # --- force rows on the control points and along the curve
    cone, floor = 0.0, 0.0
    for seg in sol.segments:
        if seg.lam_ctrl is None:
            continue
        o = 0
        for c in seg.contacts:
            d = int(c["dim"])
            C = seg.lam_ctrl[o:o + d]
            cone = max(cone, np.max(np.hypot(C[0], C[1]) - mu * C[2]))
            floor = max(floor, np.max(-C[2]))
            o += d
    results.append(("friction cone |f_t| - mu fz on the control points (N)", cone, tol, f"mu = {mu}; convexity carries it to the whole curve"))
    results.append(("normal force -fz on the control points (N)", floor, tol, "no pulling"))
    # --- hardware limits on the fine grid
    eff = robot.model.effortLimit[robot.actuated]
    vel = robot.model.velocityLimit[robot.actuated]
    lo, hi = robot.model.lowerPositionLimit[robot.actuated], robot.model.upperPositionLimit[robot.actuated]
    over_u = max(np.max(np.abs(r[6]) - eff) for r in rows)
    over_v = max(np.max(np.abs(r[4][robot.actuated]) - vel) for r in rows)
    over_q = max(max(np.max(r[3][robot.actuated] - hi), np.max(lo - r[3][robot.actuated])) for r in rows)
    results.append(("torque above the URDF effort limit on the fine grid (N m)", over_u, tol, "the torque polynomial between the nodes"))
    results.append(("joint speed above the URDF limit on the fine grid (rad/s)", over_v, tol, ""))
    results.append(("joint angle outside the URDF range on the fine grid (rad)", over_q, tol, ""))
    # --- junctions and rests
    cont_q, cont_v, impact = 0.0, 0.0, 0.0
    for s0, s1 in zip(sol.segments[:-1], sol.segments[1:]):
        cont_q = max(cont_q, np.max(np.abs(s1.Q[:, 0] - s0.Q[:, -1])))
        if s0.reset_after == "continuity":
            cont_v = max(cont_v, np.max(np.abs(s1.V[:, 0] - s0.V[:, -1])))
        elif s0.reset_after == "plastic_impact":
            vplus = robot.impact(s0.Q[:, -1], s0.V[:, -1], s1.contacts)
            impact = max(impact, np.max(np.abs(s1.V[:, 0] - vplus)))
    results.append(("configuration continuity across every junction (m, rad)", cont_q, tol, ""))
    results.append(("velocity continuity across continuity junctions (m/s, rad/s)", cont_v, tol, ""))
    results.append(("plastic impact map v+ = Pi(q) v- at touchdowns (m/s, rad/s)", impact, tol, "sticking impact of the landing contacts, recomputed from the mass matrix"))
    rest = max(np.max(np.abs(sol.segments[0].V[:, 0])), np.max(np.abs(sol.segments[-1].V[:, -1])))
    results.append(("rest at both ends |v| (m/s, rad/s)", rest, tol, ""))
    return results, curves, weight


# --- report ---------------------------------------------------------------------------------------

def effort_integral(sol: Solution) -> float:
    total = 0.0
    for seg in sol.segments:
        n = seg.degree
        # Clenshaw-Curtis weights on the CGL nodes (ascending; symmetric)
        w = np.zeros(n + 1)
        for k in range(n + 1):
            theta = math.pi * k / n
            s = 0.0
            for j in range(1, n // 2 + 1):
                b = 1.0 if 2 * j < n else 0.5
                s += b / (4 * j * j - 1) * math.cos(2 * j * theta)
            c = 1.0 if 0 < k < n else 0.5
            w[k] = c * 2.0 / n * (1.0 - 2.0 * s)
        total += 0.5 * seg.T * float(np.sum(w * np.sum(seg.U ** 2, axis=0)))
    return total


def write_report(sol: Solution, results, out: Path, weight, has_gif: bool, extra: dict):
    meta = sol.meta
    lines = [f"# Solution report: {meta['problem']['name']}", ""]
    lines += [f"Archive generator `{meta['generator']}`, problem type `{meta['problem']['type']}`, robot URDF `{meta['robot']['urdf']}`.", ""]
    lines += ["## Solve summary", "", "| quantity | value |", "|---|---|"]
    m = meta["metrics"]
    lines += [f"| node gate (zero-dual max violation, solver) | {m['gate']:.3e} |",
              f"| fine-grid dynamics worst (solver) | {m['internode_dyn_max']:.3e} |",
              f"| elements / nodes | {len(sol.segments)} / {sum(s.taus.size for s in sol.segments)} |",
              f"| total time | {sol.total_time:.3f} s |",
              f"| control effort int ||u||^2 dt | {effort_integral(sol):.4e} |"]
    for k, v in extra.items():
        lines.append(f"| {k} | {v} |")
    if "run" in meta:
        lines.append(f"| solver wall | {meta['run']['wall_s']:.1f} s ({meta['run']['omp_threads']} threads) |")
    lines += ["", "## Independent verification (pinocchio, fine grid)", ""]
    lines += [f"Robot weight {weight:.1f} N. Each row is the worst value over the whole trajectory; pass = value <= threshold.", ""]
    lines += ["| check | worst | threshold | pass | note |", "|---|---|---|---|---|"]
    npass = 0
    for name, val, thr, note in results:
        ok = val <= thr
        npass += ok
        lines.append(f"| {name} | {val:.2e} | {thr:.0e} | {'yes' if ok else 'NO'} | {note} |")
    lines += ["", f"{npass} of {len(results)} checks pass.", ""]
    lines += ["## Figures", "", "![states](states.png)", "", "![velocities](velocities.png)", "", "![torques](torques.png)", "",
              "![forces](forces.png)", "", "![violations](violations.png)", ""]
    if has_gif:
        lines += ["![animation](animation.gif)", ""]
    (out / "report.md").write_text("\n".join(lines))
    return npass, len(results)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--mu", type=float, default=0.85, help="friction coefficient of the problem (for the cone check and plot)")
    ap.add_argument("--urdf", type=Path, default=None, help="override the URDF path recorded in the archive")
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    sol = load(a.archive)
    rows = sample_curves(sol, a.dt)
    robot = None
    limits = {"mu": a.mu}
    if pin is not None:
        urdf = a.urdf or Path(sol.meta["robot"]["urdf"])
        if not urdf.is_absolute() and not urdf.exists():
            urdf = a.archive.parent / urdf
        robot = Robot(urdf, sol.meta)
        limits["effort"] = robot.model.effortLimit[robot.actuated]
        limits["velocity"] = robot.model.velocityLimit[robot.actuated]
    plot_all(sol, rows, a.out, robot, limits)
    results, curves, weight = ([], {}, 0.0)
    if robot is not None:
        results, curves, weight = verify(sol, rows, robot, a.mu, a.tol)
        if robot.motor_model != "none":
            print("note: the robot has a motor model; the pinocchio residual here omits the motor loss")
    plot_violations(sol, curves, a.out)
    has_gif = False
    if robot is not None and not a.no_gif:
        animate(sol, rows, robot, a.out)
        has_gif = True
    npass, n = write_report(sol, results, a.out, weight, has_gif, {})
    print(f"report written to {a.out / 'report.md'}: {npass}/{n} verification checks pass")
    for name, val, thr, _ in results:
        print(f"  {'ok ' if val <= thr else 'NO '} {val:.2e} <= {thr:.0e}  {name}")
    return 0 if npass == n else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
