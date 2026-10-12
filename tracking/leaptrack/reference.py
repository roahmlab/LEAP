"""Reference trajectories for the tracking loop: the gait interface the
controller, rollout and evaluator expect, the LEAP archive loaders, and the
touchdown-aware reference clocks.

Contents: `RefSample`, `SceneBoxSpec`, FOOT_* / PHASE_NAMES, `ShootingReference` (the
gait-agnostic base: properties, `stage_at`, `scheduled_at`, `targets_at`, the Hermite
`eval`, `sample`, the generic `_hold_controls`), `FenceReference`, `TerrainReference`,
`TimeWarp` (the `--clock timewarp` option), the LEAP references (`_LeapEval`,
`LeapFenceReference`, `LeapTerrainReference`, `_grid_fields`, `_check_schedule`,
`load_leap_reference`) and the clocks (`TouchdownClock`, `make_clock`). The problem yaml
is read from the archive's own `meta_json["problem"]["yaml"]` by default
(`--problem-yaml` overrides), with its `robot:` line resolved by basename against
`new_src/robots/`.

What comes from where for a LEAP archive:
  * q, v, a, u, lam, active contacts, plastic-impact jumps ......... the archive
    (SPECTRAL: Chebyshev polynomials at CGL nodes, forces as Bezier control
    points; `leaptrack.archive.ReferenceTrajectory` evaluates them exactly)
  * scene geometry (fence box / stepping stones), friction, CoP box, lammin,
    planar lock, terminal footholds, terminal rest pose ............ the problem yaml
    (`leaptrack.spec_fence` / `spec_terrain`; the archive's mode schedule is
    CHECKED against it)
  * the knot-grid `plan` (only for the constraint checker of the evaluator and
    for the legacy `xs/ts` fields, sampled from the archive on the plan's grid)
    .................................................................. `leaptrack.plan`
    built with `impulse_free=False`, i.e. the touchdown rows leap itself states.

Past the final time the reference holds the archive's final state with the
archive's own final torque and contact forces (the problems end at rest).
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .archive import ReferenceTrajectory, load_solution
from .paths import robot_yaml_path, urdf_path
from .plan import fence_plan, terrain_plan
from .spec_fence import load_fence_spec
from .spec_terrain import load_terrain_spec

PHASE_NAMES = ("push", "flight", "land", "hold")
# per-foot phase states (RefSample.foot_phase[contact][0])
FOOT_STANCE, FOOT_SWING, FOOT_LANDED = 0, 1, 2

METHOD = "leap"
FINE_DT = 0.01        # knot grid of the legacy fields / constraint checker
YAML_SUFFIX = {"fence": "jump_fence", "terrain": "terrain_walk"}


@dataclass
class RefSample:
    t: float
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    u: np.ndarray                 # actuated torque (nu)
    lam: dict                     # contact name -> planned force/wrench (only scheduled feet)
    scheduled: list               # contact names in stance per the plan
    targets: dict                 # contact name -> planned world foothold (scheduled feet)
    phase: int                    # benchmark phase index (`ref.phase_names`)
    phase_s: float                # progress within the phase, [0, 1]
    stage: int                    # knot interval index (clipped)
    foot_phase: dict = field(default_factory=dict)   # contact -> (FOOT_* state, progress in [0, 1])
    phase_duration: float = 0.0  # seconds; zero when the phase clock is held


@dataclass
class SceneBoxSpec:
    """A static box the simulator must build: `collide_capsules` says whether
    the robot's collision capsules physically collide with it (an obstacle the
    problem keeps the capsules clear of) or not (a stepping stone the feet
    stand on, whose keep-out is an inset box judged, not simulated)."""
    name: str
    center: np.ndarray
    half: np.ndarray
    collide_capsules: bool = True


@dataclass
class ShootingReference:
    method: str
    robot_name: str
    npz_path: Path
    json_path: Path
    spec: object                  # spec_fence.FenceSpec | spec_terrain.TerrainSpec
    plan: object                  # plan.Plan
    ts: np.ndarray
    xs: np.ndarray
    us: np.ndarray                # N x nu          torque at the START of each stage
    lams: np.ndarray              # N x sum(dims)   contact forces at the START of each stage
    a1: np.ndarray                # N x nv  acceleration at the left knot of each stage
    a4: np.ndarray                # N x nv  acceleration at the right end of each stage
    settings: dict = field(default_factory=dict)
    # first-order hold archives: torque / force at the END of each stage. None =
    # zero-order hold (`us[k]` held over the stage).
    us_end: np.ndarray | None = None
    lams_end: np.ndarray | None = None
    stage_tolerance: float = 0.0  # opt-in right-limit lookup for switch-timing diagnostics

    benchmark: str = ""
    phase_names: tuple = ()
    clearance_label: str = "clearance"

    # ---------------------------------------------------------------- props
    @property
    def robot(self):
        return self.spec.robot

    @property
    def nq(self) -> int:
        return self.robot.nq

    @property
    def nv(self) -> int:
        return self.robot.nv

    @property
    def nu(self) -> int:
        return self.robot.nu

    @property
    def duration(self) -> float:
        return float(self.ts[-1])

    @property
    def contact_names(self) -> list:
        return [c.name for c in self.robot.contacts]

    @property
    def contact_dims(self) -> dict:
        return {c.name: int(c.dim) for c in self.robot.contacts}

    @property
    def lam_slices(self) -> dict:
        out, off = {}, 0
        for c in self.robot.contacts:
            out[c.name] = slice(off, off + c.dim)
            off += c.dim
        return out

    @property
    def planarlock(self) -> list:
        """Base joint NAMES the problem pins to zero."""
        m = self.robot.model
        names = list(m.names)[1:]
        return [names[i] for i in self.spec.planarlock_idx]

    def q0(self):
        return self.xs[0, :self.nq].copy(), self.xs[0, self.nq:].copy()

    def q_final(self):
        return self.xs[-1, :self.nq].copy()

    # ------------------------------------------------------- gait interface
    # Subclasses implement these.
    @property
    def phase_times(self) -> list:
        raise NotImplementedError

    def phase_at(self, t: float):
        """(phase index, progress in [0, 1])."""
        raise NotImplementedError

    def foot_phase_at(self, t: float) -> dict:
        """contact -> (FOOT_STANCE | FOOT_SWING | FOOT_LANDED, progress)."""
        raise NotImplementedError

    @property
    def touchdown_events(self) -> list:
        """[(t_touchdown, [feet])] in reference time, chronological: the
        instants at which the plan puts a swinging foot back on the ground
        (what the reference clocks wait for)."""
        raise NotImplementedError

    def scene_boxes(self) -> list:
        """[SceneBoxSpec] the simulator builds."""
        raise NotImplementedError

    @property
    def obstacle_boxes(self) -> list:
        """[(center, half)] boxes the collision capsules must keep clear of
        (fence: the fence; terrain: none physically -- the inset keep-out is
        judged by the constraint checker)."""
        return []

    def clearance_series(self, Q: np.ndarray) -> np.ndarray | None:
        """Per-sample physical clearance of a q trajectory (>= 0 means clear);
        None when the benchmark has no such notion."""
        return None

    @property
    def terminal_foot_targets(self) -> dict:
        """contact -> world target the feet must end on (empty if none)."""
        return {}

    terminal_foot_tol: float | None = None   # success requires the feet within this of the targets (None: not checked)

    def _hold_controls(self):
        """(u, lam row) held past the final knot: static equilibrium at the
        terminal pose. Default: min-norm floating-base least squares for the
        stance forces, inverse dynamics for the torque (the terrain seed's
        convention, robot-agnostic)."""
        if not hasattr(self, "_hold"):
            import pinocchio as pin
            r = self.robot
            m, d = r.model, r.data()
            q = self.xs[-1, :self.nq]
            active = self.scheduled_at(self.duration)
            tau = pin.rnea(m, d, q, np.zeros(self.nv), np.zeros(self.nv))
            pin.computeJointJacobians(m, d, q)
            pin.updateFramePlacements(m, d)
            unact = [i for i in range(r.nv) if i not in set(r.act_idx)]
            Js = {}
            for c in r.contacts:
                if c.name in active:
                    J = pin.getFrameJacobian(m, d, c.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    Js[c.name] = J[:c.dim]
            row = np.zeros(self.lams.shape[1])
            if Js:
                K = np.hstack([Js[n][:, unact].T for n in Js])
                sol = np.linalg.lstsq(K, tau[unact], rcond=None)[0]
                off = 0
                for n in Js:
                    dim = Js[n].shape[0]
                    row[self.lam_slices[n]] = sol[off:off + dim]
                    tau = tau - Js[n].T @ sol[off:off + dim]
                    off += dim
            self._hold = (tau[r.act_idx].copy(), row)
        return self._hold[0].copy(), self._hold[1]

    # ---------------------------------------------------------------- eval
    def stage_at(self, t: float) -> int:
        # `stage_tolerance` > 0 selects the right limit at a knot (opt-in).
        k = int(np.searchsorted(self.ts, t + self.stage_tolerance, side="right") - 1)
        return min(max(k, 0), len(self.ts) - 2)

    def scheduled_at(self, t: float) -> list:
        if t >= self.duration:
            return list(self.plan.stage_contacts[-1])
        return list(self.plan.stage_contacts[self.stage_at(t)])

    def targets_at(self, t: float) -> dict:
        if t >= self.duration:
            return dict(self.plan.knots[-1].targets)
        return dict(self.plan.knots[self.stage_at(t)].targets)

    def eval(self, t: float) -> RefSample:
        """Cubic Hermite between the knots of a shooting archive (not used by
        the LEAP references, which evaluate the spectral archive exactly)."""
        nq, nv = self.nq, self.nv
        if t >= self.duration:
            k = len(self.ts) - 2
            q = self.xs[-1, :nq].copy()
            v = np.zeros(nv)
            a = np.zeros(nv)
            u, lam_row = self._hold_controls()
        else:
            k = self.stage_at(t)
            t0, t1 = self.ts[k], self.ts[k + 1]
            h = t1 - t0
            s = (t - t0) / h
            q0, q1 = self.xs[k, :nq], self.xs[k + 1, :nq]
            v0, v1 = self.xs[k, nq:], self.xs[k + 1, nq:]
            a0, a1 = self.a1[k], self.a4[k]
            # cubic Hermite basis
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            d00 = 6 * s**2 - 6 * s
            d10 = 3 * s**2 - 4 * s + 1
            d01 = -6 * s**2 + 6 * s
            d11 = 3 * s**2 - 2 * s
            q = h00 * q0 + h10 * h * v0 + h01 * q1 + h11 * h * v1
            v = h00 * v0 + h10 * h * a0 + h01 * v1 + h11 * h * a1
            a = (d00 * v0 + d10 * h * a0 + d01 * v1 + d11 * h * a1) / h
            if self.us_end is None:                      # zero-order hold
                u = self.us[k].copy()
                lam_row = self.lams[k]
            else:                                        # first-order hold: linear in the stage
                u = (1.0 - s) * self.us[k] + s * self.us_end[k]
                lam_row = (1.0 - s) * self.lams[k] + s * self.lams_end[k]
        phase, s = self.phase_at(t)
        sched = self.scheduled_at(t)
        sl = self.lam_slices
        lam = {c: lam_row[sl[c]].copy() for c in sched}
        phase_times = getattr(self, 'phase_times', ())
        phase_duration = (float(phase_times[phase + 1] - phase_times[phase])
                          if phase + 1 < len(phase_times) else 0.0)
        return RefSample(t=float(t), q=q, v=v, a=a, u=u, lam=lam, scheduled=sched,
                         targets=self.targets_at(t), phase=phase, phase_s=float(s), stage=k,
                         foot_phase=self.foot_phase_at(t), phase_duration=phase_duration)

    def sample(self, times):
        """Vectorised q/v over an array of times (N x nq, N x nv)."""
        Q, V = [], []
        for t in times:
            r = self.eval(float(t))
            Q.append(r.q)
            V.append(r.v)
        return np.asarray(Q), np.asarray(V)


# ------------------------------------------------------------- fence
@dataclass
class FenceReference(ShootingReference):
    benchmark: str = "fence"
    phase_names: tuple = PHASE_NAMES
    clearance_label: str = "capsule-fence gap"

    @property
    def phase_times(self) -> list:
        s = self.spec
        return [0.0, s.t_stance, s.t_stance + s.t_air, s.t_stance + s.t_air + s.t_land]

    def phase_at(self, t: float):
        pt = self.phase_times
        if t >= pt[3] - 1e-9:
            return 3, 1.0
        for i in range(3):
            if t < pt[i + 1]:
                return i, float((t - pt[i]) / max(pt[i + 1] - pt[i], 1e-12))
        return 3, 1.0

    def foot_phase_at(self, t: float) -> dict:
        ph, s = self.phase_at(t)
        state = {0: FOOT_STANCE, 1: FOOT_SWING, 2: FOOT_LANDED, 3: FOOT_LANDED}[ph]
        return {c: (state, s) for c in self.contact_names}

    @property
    def touchdown_events(self) -> list:
        return [(self.phase_times[2], list(self.contact_names))]

    @property
    def obstacle_boxes(self) -> list:
        """[(center(3,), half(3,))] of the problem's fence boxes."""
        return list(self.spec.obstacle_boxes)

    def scene_boxes(self) -> list:
        return [SceneBoxSpec(f"fence{i}", np.asarray(c, float), np.asarray(h, float), True)
                for i, (c, h) in enumerate(self.obstacle_boxes)]

    def clearance_series(self, Q):
        from .geometry import fence_gap
        boxes = self.obstacle_boxes
        if not boxes:
            return None
        r = self.robot
        return np.array([fence_gap(r, np.zeros(1), np.hstack([q, np.zeros(r.nv)])[None], boxes, subdiv=1) for q in Q])

    @property
    def terminal_foot_targets(self) -> dict:
        return dict(self.spec.feet2)

    def _hold_controls(self):
        """Static equilibrium at the terminal pose: equal gravity share per
        foot (the seed's convention) and the inverse-dynamics torque."""
        if not hasattr(self, "_hold"):
            from .spec_fence import seed_force_torque
            q = self.xs[-1, :self.nq]
            lam, u = seed_force_torque(self.spec, q, np.zeros(self.nv), np.zeros(self.nv), True)
            row = np.zeros(self.lams.shape[1])
            for c, sl in self.lam_slices.items():
                row[sl] = lam[c]
            self._hold = (u, row)
        return self._hold[0].copy(), self._hold[1]


# ------------------------------------------------------------- terrain
@dataclass
class TerrainReference(ShootingReference):
    """The terrain counterpart of `FenceReference`: a chain of 2N+1 modes
    (support_0, then per step a swing mode and a support mode). Per foot:

        foot_phase[c] = (FOOT_SWING,  s)  in the swing mode whose swing foot is c
                      = (FOOT_LANDED, s)  in the support mode right after c touched down
                      = (FOOT_STANCE, s)  otherwise

    `touchdown_events` = the start of every support mode with touchdown feet;
    `scene_boxes` = the stepping stones (feet stand on them, the collision
    capsules do NOT collide with them); `clearance_series` = `topslack -
    intrusion`. There is no terminal pose pin in the terrain problem."""
    benchmark: str = "terrain"
    clearance_label: str = "stone-top slack minus capsule intrusion"
    terminal_foot_tol: float | None = 0.03

    def __post_init__(self):
        modes = self.spec.modes
        self._mode_starts = np.concatenate([[0.0], np.cumsum([m.T for m in modes])])
        names = []
        for i, m in enumerate(modes):
            names.append(f"support{i // 2}" if m.kind == "support" else f"swing{(i + 1) // 2}")
        self.phase_names = tuple(names)

    # ------------------------------------------------------------ modes
    @property
    def modes(self) -> list:
        return self.spec.modes

    @property
    def mode_starts(self) -> np.ndarray:
        return self._mode_starts

    def mode_at(self, t: float) -> int:
        n = len(self.modes)
        mi = int(np.searchsorted(self._mode_starts, t + 1e-9, side="right") - 1)
        return min(max(mi, 0), n - 1)

    @property
    def phase_times(self) -> list:
        return [float(x) for x in self._mode_starts]

    def phase_at(self, t: float):
        mi = self.mode_at(t)
        T = max(self.modes[mi].T, 1e-12)
        s = (t - self._mode_starts[mi]) / T
        return mi, float(min(max(s, 0.0), 1.0))

    def foot_phase_at(self, t: float) -> dict:
        mi, s = self.phase_at(t)
        md = self.modes[mi]
        out = {}
        for c in self.contact_names:
            if md.kind == "swing" and c == md.swing_foot:
                out[c] = (FOOT_SWING, s)
            elif md.kind == "support" and c in md.touchdown_feet:
                out[c] = (FOOT_LANDED, s)
            else:
                out[c] = (FOOT_STANCE, s)
        return out

    @property
    def touchdown_events(self) -> list:
        return [(float(self._mode_starts[i]), list(md.touchdown_feet))
                for i, md in enumerate(self.modes) if md.kind == "support" and md.touchdown_feet]

    # ------------------------------------------------------------ scene
    @property
    def pillars(self) -> list:
        """[(cx, cy, lx, ly, top)]."""
        return list(self.spec.pillars)

    def scene_boxes(self) -> list:
        out = []
        for i, (cx, cy, lx, ly, top) in enumerate(self.pillars):
            out.append(SceneBoxSpec(f"stone{i}", np.array([cx, cy, 0.5 * top]), np.array([0.5 * lx, 0.5 * ly, 0.5 * top]),
                                    collide_capsules=False))
        return out

    def clearance_series(self, Q):
        from .geometry import terrain_intrusion
        r = self.robot
        pil = self.pillars
        if not pil:
            return None
        return np.array([self.spec.topslack - terrain_intrusion(r, np.zeros(1), np.hstack([q, np.zeros(r.nv)])[None], pil, subdiv=1)
                         for q in Q])

    @property
    def terminal_foot_targets(self) -> dict:
        return {k: np.asarray(v, float) for k, v in (self.plan.terminal_foot_pins or {}).items()}


# ---------------------------------------------------------- LEAP references
class _LeapEval:
    """Archive-backed evaluation shared by the fence and terrain references.
    `self.leap` is the ReferenceTrajectory; the dataclass fields of the
    shooting reference (`ts, xs, us, lams, a1, a4`) hold the archive SAMPLED on
    the plan grid so that legacy consumers (`xs[-1]`, `q_final`) stay valid."""
    leap: ReferenceTrajectory = None

    # -- schedule from the archive -------------------------------------
    def scheduled_at(self, t: float) -> list:
        if t >= self.duration:
            return list(self.leap.modes[-1].active_contacts)
        return list(self.leap.active_contacts(t))

    def _lam_row(self, t: float) -> np.ndarray:
        row = np.zeros(self.lams.shape[1])
        if t >= self.duration:
            return self._hold_controls()[1]
        for c, w in self.leap.lam_map(t).items():
            sl = self.lam_slices[c]
            row[sl] = w[:sl.stop - sl.start]
        return row

    def _hold_controls(self):
        """Archive's own final torque and contact forces (rest at the end)."""
        if not hasattr(self, "_hold"):
            T = self.duration
            q, v, a, lam, u, s = self.leap.eval_full(T)
            row = np.zeros(self.lams.shape[1])
            for c, w in self.leap.lam_map(T).items():
                sl = self.lam_slices[c]
                row[sl] = w[:sl.stop - sl.start]
            self._hold = (np.asarray(u, float).copy(), row)
        return self._hold[0].copy(), self._hold[1]

    def eval(self, t: float) -> RefSample:
        nv = self.nv
        if t >= self.duration:
            k = len(self.ts) - 2
            q = self.q_final()
            v = np.zeros(nv)
            a = np.zeros(nv)
            u, lam_row = self._hold_controls()
        else:
            k = self.stage_at(t)
            q, v, a, _, u, _ = self.leap.eval_full(float(t))
            q, v, a, u = (np.asarray(x, float).copy() for x in (q, v, a, u))
            lam_row = self._lam_row(t)
        phase, s = self.phase_at(t)
        sched = self.scheduled_at(t)
        sl = self.lam_slices
        lam = {c: lam_row[sl[c]].copy() for c in sched}
        pt = self.phase_times
        phase_duration = float(pt[phase + 1] - pt[phase]) if phase + 1 < len(pt) else 0.0
        return RefSample(t=float(t), q=q, v=v, a=a, u=u, lam=lam, scheduled=sched,
                         targets=self.targets_at(t), phase=phase, phase_s=float(s), stage=k,
                         foot_phase=self.foot_phase_at(t), phase_duration=phase_duration)

    def sample(self, times):
        Q, V = [], []
        for t in times:
            r = self.eval(float(t))
            Q.append(r.q)
            V.append(r.v)
        return np.asarray(Q), np.asarray(V)

    def q0(self):
        q, v = self.leap.q0()
        return np.asarray(q, float).copy(), np.asarray(v, float).copy()

    def q_final(self):
        return np.asarray(self.leap.eval(self.duration)[0], float).copy()

    # -- extras for the report ------------------------------------------
    @property
    def impact_times(self) -> list:
        """Reference times of the archive's plastic-impact resets."""
        out = []
        for i, m in enumerate(self.leap.modes[:-1]):
            if self.leap.reset_kind_after(i) == "plastic_impact":
                out.append(float(m.t0 + m.T))
        return out


@dataclass
class LeapFenceReference(_LeapEval, FenceReference):
    leap: ReferenceTrajectory = None


@dataclass
class LeapTerrainReference(_LeapEval, TerrainReference):
    leap: ReferenceTrajectory = None


# -------------------------------------------------------------------- loader
def _grid_fields(leap: ReferenceTrajectory, plan, robot):
    """Sample the archive on the plan's knot grid: xs (N+1 x 2nq), us / lams at
    stage starts, a at both knots of every stage."""
    ts = np.asarray(plan.ts, float)
    nq, nv, nu = robot.nq, robot.nv, robot.nu
    nlam = sum(c.dim for c in robot.contacts)
    slices, off = {}, 0
    for c in robot.contacts:
        slices[c.name] = slice(off, off + c.dim)
        off += c.dim
    xs = np.zeros((len(ts), nq + nv))
    acc = np.zeros((len(ts), nv))
    us = np.zeros((len(ts) - 1, nu))
    lams = np.zeros((len(ts) - 1, nlam))
    for i, t in enumerate(ts):
        te = min(float(t), leap.duration)
        q, v, a, lam, u, s = leap.eval_full(te)
        xs[i, :nq], xs[i, nq:] = q, v
        acc[i] = a
        if i < len(ts) - 1:
            us[i] = u
            for c, w in leap.lam_map(te).items():
                sl = slices[c]
                lams[i, sl] = w[:sl.stop - sl.start]
    return ts, xs, us, lams, acc[:-1].copy(), acc[1:].copy()


def _check_schedule(leap: ReferenceTrajectory, plan, spec_duration: float, path: Path):
    """The archive's mode chain must be the problem's: same total duration, a
    segment boundary at every plan mode boundary, the same stance set at every
    stage midpoint."""
    if abs(leap.duration - spec_duration) > 1e-6:
        raise ValueError(f"{path}: archive duration {leap.duration} != problem {spec_duration}")
    bounds = np.array([m.t0 for m in leap.modes] + [leap.duration])
    ts = np.asarray(plan.ts)
    for k in range(len(ts) - 1):
        sched_plan = set(plan.stage_contacts[k])
        tm = 0.5 * (ts[k] + ts[k + 1])
        sched_arch = set(leap.active_contacts(float(tm)))
        if sched_plan != sched_arch:
            raise ValueError(f"{path}: stance set at t={tm:.4f}: archive {sorted(sched_arch)} "
                             f"!= problem {sorted(sched_plan)}")
        if k > 0 and set(plan.stage_contacts[k]) != set(plan.stage_contacts[k - 1]):
            if np.min(np.abs(bounds - ts[k])) > 1e-6:
                raise ValueError(f"{path}: no segment boundary at the problem's mode boundary t={ts[k]:.4f}")


_ROBOT_LINE = re.compile(r"^(robot:\s*)(\S+)(.*)$", re.MULTILINE)


def materialize_problem_yaml(text: str, dest: Path) -> Path:
    """Write `text` (a problem yaml) to `dest` with its `robot:` line pointing at
    `new_src/robots/<basename>` (absolute), so the spec loaders -- which resolve
    `robot:` relative to the yaml's directory -- find the robot description
    wherever the yaml came from (an archive echo, the old tree, anywhere)."""
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "robot" not in doc:
        raise ValueError("problem yaml has no top-level `robot:` key")
    target = robot_yaml_path(doc["robot"])
    if not target.exists():
        raise FileNotFoundError(f"robot yaml {target} (from `robot: {doc['robot']}`)")
    out, n = _ROBOT_LINE.subn(lambda m: f"{m.group(1)}{target}{m.group(3)}", text, count=1)
    if n != 1:
        raise ValueError("could not rewrite the `robot:` line of the problem yaml")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out)
    return dest


def load_leap_reference(task: str, robot: str, archive, *, problem_yaml=None, work_dir=None,
                        lam_interp: str = "bezier"):
    """Build a `LeapFenceReference` / `LeapTerrainReference` for `robot` from the
    solution archive `archive`. The problem yaml is the archive's own echo
    (`meta_json["problem"]["yaml"]`) unless `problem_yaml` (a path) is given;
    either way a copy with the `robot:` line resolved against `new_src/robots/`
    is written to `work_dir/<robot>_<suffix>.yaml` (a temp dir when None)."""
    path = Path(archive)
    if not path.exists():
        raise FileNotFoundError(path)
    sol = load_solution(path)
    # torque-eliminated archives reconstruct u(t) by inverse dynamics on the
    # URDF; the archive's path (relative to the old tree, or absolute into it)
    # is resolved by basename against new_src/assets/ (the same file)
    sol.meta["urdf"] = str(urdf_path(sol.urdf))
    if problem_yaml is not None:
        text = Path(problem_yaml).read_text()
        source = str(Path(problem_yaml).resolve())
    else:
        text = sol.meta["problem"]["yaml"]
        source = f"{path}:meta_json.problem.yaml"
    wd = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="leaptrack_"))
    yaml_path = materialize_problem_yaml(text, wd / f"{robot}_{YAML_SUFFIX[task]}.yaml")
    leap = ReferenceTrajectory(sol, lam_interp=lam_interp)
    if task == "fence":
        # allow_impact_cone: LEAP states the friction-cone row on the touchdown
        # impulse (problem.impactcone); the shooting ports cannot, so the parser
        # refuses the key by default.
        spec = load_fence_spec(yaml_path, allow_impact_cone=True)
        plan = fence_plan(spec, dt_target=FINE_DT, impulse_free=False)
        duration = spec.t_stance + spec.t_air + spec.t_land
        cls = LeapFenceReference
    elif task == "terrain":
        spec = load_terrain_spec(yaml_path)
        plan = terrain_plan(spec, dt_target=FINE_DT, kp=100.0, impulse_free=False)
        duration = float(sum(m.T for m in spec.modes))
        cls = LeapTerrainReference
    else:
        raise KeyError(task)
    r = spec.robot
    names = list(r.model.names)[1:]
    if list(sol.joint_names) != names:
        raise ValueError(f"{path}: archive joint order {sol.joint_names} != robot {names}")
    arch_contacts = [(c["name"], int(c["dim"])) for c in sol.contacts]
    if arch_contacts != [(c.name, int(c.dim)) for c in r.contacts]:
        raise ValueError(f"{path}: archive contacts {arch_contacts} != robot yaml")
    _check_schedule(leap, plan, duration, path)
    ts, xs, us, lams, a1, a4 = _grid_fields(leap, plan, r)
    ref = cls(method=METHOD, robot_name=robot, npz_path=path, json_path=path.with_suffix(".json"),
              spec=spec, plan=plan, ts=ts, xs=xs, us=us, lams=lams, a1=a1, a4=a4,
              settings=dict(source="leap archive", dt=FINE_DT, gate=float(sol.meta["metrics"]["gate"]),
                            lam_interp=lam_interp, problem_yaml=str(yaml_path), problem_yaml_source=source),
              leap=leap)
    return ref


# ------------------------------------------------------------------ clocks
class TimeWarp:
    """Reference clock that waits for (or jumps to) the physical touchdowns --
    the variant written for the impulse-free shooting plans (`--clock timewarp`).

    For every touchdown event of the plan (`ref.touchdown_events`, chronological):
    once the reference clock reaches the event with none of its feet touching,
    the reference is FROZEN at the touchdown knot (joint velocities and
    accelerations zeroed) until one of those feet touches; a foot of the event
    touching during the last part of its swing instead JUMPS the reference to
    the touchdown knot. The accumulated shift is reported, never hidden.
    """

    def __init__(self, ref: ShootingReference, early_phase: float = 0.8, freeze_mode: str = "hold",
                 continuous_coast: bool = False, resume_duration: float = .15,
                 early_jump: bool = True):
        if freeze_mode not in ("hold", "coast"):
            raise ValueError("freeze_mode must be 'hold' or 'coast'")
        self.ref = ref
        self.early_phase = float(early_phase)
        self.early_jump = bool(early_jump)
        self.freeze_mode = freeze_mode
        self.continuous_coast = continuous_coast
        self.resume_duration = float(resume_duration)
        if self.resume_duration <= 0:
            raise ValueError("resume_duration must be positive")
        self._freeze_start = None
        self._coast_offset = None
        self._resume_start = None
        self._resume_offset = None
        self.events = [(float(t), list(feet)) for t, feet in ref.touchdown_events]
        self.shift = 0.0
        self.next_event = 0
        self.frozen = False
        self.landed_events = 0

    @property
    def landed(self) -> bool:
        return self.next_event >= len(self.events)

    def sample(self, t: float, measured: set) -> RefSample:
        ref = self.ref
        tr = t - self.shift
        was_frozen = self.frozen
        self.frozen = False
        t_event = None
        feet = []
        if self.next_event < len(self.events):
            t_td, feet = self.events[self.next_event]
            touching = bool(measured & set(feet))
            fp = ref.foot_phase_at(tr)
            in_late_swing = any(fp[f][0] == FOOT_SWING and fp[f][1] >= self.early_phase for f in feet)
            if self.early_jump and tr < t_td - 1e-12 and in_late_swing and touching:
                self.shift = t - t_td              # early landing: jump ahead
                tr = t_td
                self.next_event += 1
            elif tr >= t_td - 1e-12:
                if touching:
                    if was_frozen:                  # resume exactly from the touchdown knot
                        self.shift = t - t_td
                        tr = t_td
                    self.next_event += 1
                else:                               # late landing: wait
                    if not was_frozen:
                        self._freeze_start = t - max(0., tr - t_td)
                        self._coast_offset = None
                    self.shift = t - t_td
                    tr = t_td
                    self.frozen = True
                    t_event = t_td
        smp = ref.eval(tr)
        if self.frozen:
            smp.phase_duration = 0.0
            # Waiting for a late foot. Airborne (the fence: no foot touching)
            # the reference HOLDS the touchdown pose with zero velocity
            # (`freeze_mode`); with other feet in support (a walk's single or
            # triple stance) it COASTS instead: the touchdown-knot motion
            # continues.
            supported = bool(measured - set(feet))
            mode = "coast" if supported else self.freeze_mode
            if mode == "hold":
                smp.v = np.zeros_like(smp.v)
            else:
                elapsed = t - self._freeze_start if self.continuous_coast else t - self.shift - t_event
                self._coast_offset = smp.v * elapsed
                smp.q = smp.q + self._coast_offset
            smp.a = np.zeros_like(smp.a)
        elif self.continuous_coast:
            if was_frozen and self._coast_offset is not None:
                self._resume_start = t
                self._resume_offset = self._coast_offset.copy()
            if self._resume_start is not None:
                # Quintic removal of the coasting displacement: position and
                # velocity stay continuous when the reference clock resumes.
                u = (t - self._resume_start) / self.resume_duration
                if u < 1.:
                    f = 1.-10.*u**3+15.*u**4-6.*u**5
                    df = (-30.*u**2+60.*u**3-30.*u**4)/self.resume_duration
                    ddf = (-60.*u+180.*u**2-120.*u**3)/self.resume_duration**2
                    smp.q += f*self._resume_offset
                    smp.v += df*self._resume_offset
                    smp.a += ddf*self._resume_offset
                else:
                    self._resume_start = None
        return smp


class TouchdownClock:
    """Reference clock that waits for (or jumps to) the physical touchdowns of a
    plan whose touchdowns may be PLASTIC IMPACTS (a velocity jump at the touchdown
    instant). Works with any reference exposing `touchdown_events`, `foot_phase_at`
    and `eval`.

    For every touchdown event of the plan (chronological):
      * the clock runs at wall time minus the accumulated shift;
      * EARLY landing (`early_jump`): a foot of the event touching during the
        last part of its own swing (progress >= `early_phase`) jumps the clock
        to the touchdown instant (post-impact side) -- the landing happened;
      * LATE landing: once the clock reaches the touchdown instant with none of
        the event's feet touching, the reference COASTS on the PRE-impact
        state: q = q(t_td^-), v = v(t_td^-), a = 0, i.e. the controller keeps
        the planned approach velocity as its velocity reference (the swing
        feet keep descending at the planned speed instead of being braked)
        until a foot touches, when the clock resumes at the post-impact
        touchdown state.
    The accumulated shift is reported (`shift`), never hidden. A wait longer
    than `max_wait` releases the clock (the robot has fallen)."""

    def __init__(self, ref, early_phase: float = 0.8, early_jump: bool = True, max_wait: float = 0.3,
                 rearm: float = 0.0):
        """`rearm` > 0 (seconds; 0 = off): if every foot of an event loses contact
        within `rearm` of the touchdown that consumed the event (an early jump, or a
        touch at / after the planned instant), the event is re-armed: the reference
        goes back to the pre-impact coast state of a late landing until a foot of the
        event touches again."""
        self.ref = ref
        self.early_phase = float(early_phase)
        self.early_jump = bool(early_jump)
        self.max_wait = float(max_wait)
        self.rearm = float(rearm)
        self.events = [(float(t), list(feet)) for t, feet in ref.touchdown_events]
        self.shift = 0.0
        self.next_event = 0
        self.frozen = False
        self._t_freeze = None
        self._consumed = None    # (event index, wall time of the touchdown that consumed it)
        self.rearms = []         # (t_td, wall time) of every re-armed event
        self.waits = []          # (t_td, wall seconds waited) per event

    @property
    def landed(self) -> bool:
        return self.next_event >= len(self.events)

    def sample(self, t: float, measured: set) -> RefSample:
        ref = self.ref
        tr = t - self.shift
        was_frozen = self.frozen
        self.frozen = False
        if self.rearm > 0.0 and self._consumed is not None:
            k, t_touch = self._consumed
            t_td, feet = self.events[k]
            if t - t_touch > self.rearm:
                self._consumed = None
            elif not (measured & set(feet)):
                # the touchdown did not hold: back to the pre-impact wait of event k
                self._consumed = None
                self.next_event = k
                self.rearms.append((t_td, t))
                self.shift = t - t_td
                self._t_freeze = t
                self.frozen = True
                smp = ref.eval(t_td - 1e-6)
                smp.a = np.zeros_like(smp.a)
                smp.phase_duration = 0.0
                return smp
        if self.next_event < len(self.events):
            t_td, feet = self.events[self.next_event]
            touching = bool(measured & set(feet))
            if tr < t_td - 1e-12:
                fp = ref.foot_phase_at(tr)
                late_swing = any(fp[f][0] == FOOT_SWING and fp[f][1] >= self.early_phase for f in feet)
                if self.early_jump and late_swing and touching:
                    self.shift = t - t_td
                    self.waits.append((t_td, tr - t_td))
                    self._consumed = (self.next_event, t)
                    self.next_event += 1
                    return ref.eval(t_td)
            else:
                if touching:
                    if was_frozen:
                        self.shift = t - t_td
                        self.waits.append((t_td, t - self._t_freeze))
                        tr = t_td
                    self._consumed = (self.next_event, t)
                    self.next_event += 1
                    return ref.eval(tr)
                if not was_frozen:
                    self._t_freeze = t - max(0.0, tr - t_td)
                tau = t - self._t_freeze
                if tau <= self.max_wait:
                    self.shift = t - t_td
                    self.frozen = True
                    # pre-impact side: configuration HELD at the touchdown
                    # pose, velocity = the planned approach velocity, a = 0.
                    smp = ref.eval(t_td - 1e-6)
                    smp.a = np.zeros_like(smp.a)
                    smp.phase_duration = 0.0
                    return smp
                # gave up waiting: release the clock at the accumulated shift
                self.waits.append((t_td, tau))
                self.next_event += 1
                tr = t - self.shift
        return ref.eval(tr)


LeapClock = TouchdownClock      # the name the earlier study used


def make_clock(kind: str, ref, gains, early_jump: bool):
    """`touchdown` (TouchdownClock), `timewarp` (TimeWarp, for impulse-free
    shooting plans) or `none`."""
    if kind in (None, "none"):
        return None
    if kind == "touchdown":
        return TouchdownClock(ref, early_phase=gains.early_contact_phase, early_jump=early_jump,
                              rearm=float(getattr(gains, "touchdown_rearm", 0.0)))
    if kind == "timewarp":
        return TimeWarp(ref, early_phase=gains.early_contact_phase, early_jump=early_jump)
    raise ValueError(f"unknown clock {kind!r} (touchdown | timewarp | none)")
