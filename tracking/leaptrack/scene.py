"""The MuJoCo scene: the ONE contact/solver configuration (plastic-impact
settings), the sole-contact wrench maps, and the robot model whose generalized
coordinates are EXACTLY the optimizer's (pinocchio's) `q`.

Contents: the MuJoCo settings (`MujocoSettings`, `SETTINGS`, `apply_options`,
`apply_contact_geom`, `describe`), the contact geometry (`wrench_map`,
`contact_projection`; used by `leaptrack.wbc`) and the model builder (`SceneBox`,
`RobotSim`, `patch_urdf`, `derive_body2joint`, `align_inertias_to_pinocchio`,
`robot_yaml`, `frame_anchor`, `capsule_endpoints_in_body`, `build`, `check_dynamics`;
`visual=True` is refused here, the rendering model is `figure2.scene`).
Robot descriptions are read from `new_src/robots/` and `new_src/assets/` (`leaptrack.paths`).

--- mujoco_settings ---------------------------------------------------------
Two goals, in this order:

1. **Highest fidelity.** Newton solver run to a tight tolerance with a large
   iteration budget, the elliptic friction cone (the optimizer's cone is the
   exact second-order cone; MuJoCo's pyramidal cone over-constrains the
   tangential impulse), a dense constraint Jacobian, no contact margin, no
   constraint-force override, and a small integrator step.

2. **A plastic (perfectly inelastic, sticking) impact.** The trajectory
   optimizer replaces each touchdown by the instantaneous projection

       v+ = v- - M^{-1} J^T (J M^{-1} J^T + eps I)^{-1} J v-        (eps = 1e-9)

   MuJoCo has no impulsive impact model; every contact is a soft constraint
   whose reference acceleration is a spring-damper parameterised by `solref =
   (timeconst, dampratio)`. The plastic limit is approached by dampratio >> 1
   (over-damped: the impact is absorbed by the damper), the smallest timeconst
   MuJoCo allows (2 * timestep), solimp with d0/dwidth close to 1 (a hard
   constraint), impratio >> 1 (the tangential velocity is also killed at
   touchdown) and noslip iterations (stance creep).

`SETTINGS` is the frozen choice.

--- robot_model -------------------------------------------------------------
  * the robot URDF loaded through `MjSpec.from_string` after `patch_urdf`:
    visuals stripped, mesh collision geoms dropped (every URDF geom is made
    non-colliding below), massless links given a 1e-9 kg inertia;
  * per-body inertials overwritten with pinocchio's fixed-joint-merged
    composite inertias (`align_inertias_to_pinocchio`);
  * ONE contact geom per optimizer contact: a sphere whose LOWEST point is the
    contact frame (dim-3 point foot) or a thin box whose bottom face is the sole
    plane with the yaml's `foot_box` half-extents grown by `SOLE_MARGIN`
    (dim-6 flat foot), plus a site at the exact frame, on the MuJoCo body of
    the frame's parent JOINT;
  * a ground plane and the problem's static boxes (the fence, or the stepping
    stones with one collision bit each), all carrying `SETTINGS`;
  * optionally the robot's `collision_capsules` as capsule geoms that collide
    with the OBSTACLE BOXES ONLY.

The floating base is the URDF's six explicit 1-DOF joints, so `qpos` and `q`
are the same vector space; MuJoCo orders joints in URDF DOCUMENT order, which
differs from pinocchio's traversal order for talos, so every quantity crossing
the interface goes through the name-resolved permutation `qadr`/`dofadr`.
Planar robots keep their `planarlock` joints; the rollout projects them to
zero after every step.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from .paths import ROBOTS_DIR
from .robot import Robot, load_robot

# =========================================================== mujoco_settings
_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_SOLVERS = {
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
}
_CONES = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}
_JACOBIANS = {
    "dense": mujoco.mjtJacobian.mjJAC_DENSE,
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    "auto": mujoco.mjtJacobian.mjJAC_AUTO,
}


@dataclass(frozen=True)
class MujocoSettings:
    # --- integration ---
    timestep: float = 1.0e-4
    integrator: str = "implicitfast"
    # --- constraint solver (fidelity) ---
    solver: str = "newton"
    iterations: int = 200          # MuJoCo default 100
    tolerance: float = 1.0e-12     # MuJoCo default 1e-8
    ls_iterations: int = 100       # MuJoCo default 50
    ls_tolerance: float = 1.0e-4   # MuJoCo default 0.01
    noslip_iterations: int = 20    # MuJoCo default 0 (off)
    noslip_tolerance: float = 1.0e-10
    cone: str = "elliptic"         # MuJoCo default pyramidal
    impratio: float = 100.0        # MuJoCo default 1
    jacobian: str = "dense"
    # --- contact model (plastic impact) ---
    # timeconst <= 0 means "the stiffest MuJoCo allows": 2 * timestep.
    solref_timeconst: float = 0.0
    solref_dampratio: float = 20.0 # >= 1 => no rebound; 20 = strongly over-damped
    solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 1.0e-4, 0.5, 2.0)
    friction: float = 1.0          # tangential; torsional/rolling left at MuJoCo defaults
    condim: int = 3                # point contact with sliding friction (dim-3 forces)
    margin: float = 0.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    @property
    def solref(self) -> tuple[float, float]:
        tc = self.solref_timeconst if self.solref_timeconst > 0 else 2.0 * self.timestep
        return (tc, self.solref_dampratio)

    def with_(self, **kw: Any) -> "MujocoSettings":
        return replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["solref_effective"] = list(self.solref)
        d["solimp"] = list(self.solimp)
        return d


# The frozen production choice.
SETTINGS = MujocoSettings()


def apply_options(spec: mujoco.MjSpec, s: MujocoSettings = SETTINGS) -> None:
    """Write the global `<option>` block of `spec` from `s`."""
    o = spec.option
    o.timestep = float(s.timestep)
    o.integrator = _INTEGRATORS[s.integrator]
    o.solver = _SOLVERS[s.solver]
    o.iterations = int(s.iterations)
    o.tolerance = float(s.tolerance)
    o.ls_iterations = int(s.ls_iterations)
    o.ls_tolerance = float(s.ls_tolerance)
    o.noslip_iterations = int(s.noslip_iterations)
    o.noslip_tolerance = float(s.noslip_tolerance)
    o.cone = _CONES[s.cone]
    o.impratio = float(s.impratio)
    o.jacobian = _JACOBIANS[s.jacobian]
    o.gravity = list(s.gravity)
    # Keep MuJoCo's refsafe clamp ON (timeconst >= 2 dt) and use NO global
    # override (o_solref / o_solimp / o_margin): every colliding geom is
    # configured explicitly by `apply_contact_geom`, so what is in the model is
    # exactly what runs.
    o.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    o.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)


def apply_contact_geom(geom, s: MujocoSettings = SETTINGS, *, collide: bool = True) -> None:
    """Configure one MjsGeom as a colliding surface with the plastic contact
    parameters. MuJoCo combines two geoms' solref/solimp by priority/average and
    takes the max friction, so give EVERY colliding geom the same values."""
    geom.contype = 1 if collide else 0
    geom.conaffinity = 1 if collide else 0
    geom.condim = int(s.condim)
    geom.friction = [float(s.friction), 0.005, 0.0001]  # MuJoCo default torsional/rolling
    geom.solref = list(s.solref)
    geom.solimp = list(s.solimp)
    geom.margin = float(s.margin)
    geom.gap = 0.0


def describe(s: MujocoSettings = SETTINGS) -> str:
    tc, dr = s.solref
    return (f"dt={s.timestep:g} {s.integrator} | {s.solver} it={s.iterations} tol={s.tolerance:g} "
            f"ls={s.ls_iterations}/{s.ls_tolerance:g} noslip={s.noslip_iterations} | "
            f"cone={s.cone} impratio={s.impratio:g} | solref=({tc:g},{dr:g}) "
            f"solimp={list(s.solimp)} mu={s.friction:g} condim={s.condim}")


# ========================================================== contact_geometry
def wrench_map(offsets):
    """Offsets from the foot frame origin; return [sum f; sum r cross f]."""
    blocks = []
    for x, y, z in np.asarray(offsets).reshape(-1, 3):
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        blocks.append(np.vstack([np.eye(3), skew]))
    return np.hstack(blocks) if blocks else np.zeros((6, 0))


def contact_projection(offsets, angular_velocity):
    """Project foot accelerations onto the motions constrained by contact.

    One point constrains three directions; a line leaves rocking about that
    line free. Include centripetal acceleration of the material contact points.
    """
    G = wrench_map(offsets).T
    inverse = np.linalg.pinv(G, rcond=1e-8)
    omega = np.asarray(angular_velocity)
    drift = np.cross(omega, np.cross(omega, offsets)).reshape(-1)
    return inverse @ G, inverse @ drift


# ============================================================== robot_model
# Point-foot sphere radius. The optimizer's foot is a POINT at the frame; a
# sphere of radius r whose bottom sits at the frame ROLLS as the shank tilts,
# so the frame moves ~r*sin(tilt) sideways and rises r*(1-cos(tilt)) relative
# to the physical contact point; 1 mm makes it negligible.
FOOT_RADIUS = 0.001
FOOT_BOX_HALF_HEIGHT = 0.004  # flat-foot box half height
# The simulated sole is the yaml `foot_box` GROWN by this much per side. The
# problems' CoP box (copx, copy) IS the yaml foot box; with the margin the
# optimizer's support polygon lies strictly inside the physical one. Point feet
# are unaffected.
SOLE_MARGIN = 0.01

# collision bitmasks: bit 1 = ground group (floor, feet), bit 2 = obstacle group
GROUND_BIT = 1
OBSTACLE_BIT = 2
STONE_BIT0 = 4          # scene boxes that are NOT capsule obstacles get one bit each: 4, 8, 16, ...


@dataclass
class SceneBox:
    name: str
    center: np.ndarray
    half: np.ndarray
    rgba: tuple = (0.55, 0.40, 0.28, 1.0)
    collide_capsules: bool = True     # obstacle (fence): capsules collide; False: a stepping stone (feet only, own bit)

    @property
    def top(self) -> float:
        return float(self.center[2] + self.half[2])


@dataclass
class RobotSim:
    """Compiled model + the bookkeeping every consumer needs."""
    name: str
    robot: Robot                        # pinocchio side (leaptrack.robot)
    model: mujoco.MjModel
    spec: mujoco.MjSpec
    qadr: np.ndarray                    # qpos address of pinocchio q[i]
    dofadr: np.ndarray                  # qvel address of pinocchio v[i]
    act_dofadr: np.ndarray              # qvel address of each actuated joint (u order)
    lock_idx: np.ndarray                # pinocchio indices of planarlocked DOFs
    foot_site: dict                     # contact name -> site id
    foot_geom: dict                     # contact name -> geom id
    contact_dims: dict                  # contact name -> 3 | 6
    base_body: int
    capsule_geoms: dict = field(default_factory=dict)   # capsule name -> geom id
    box_geoms: dict = field(default_factory=dict)       # scene box name -> geom id
    box_bits: dict = field(default_factory=dict)        # scene box name -> its own collision bit (stones), GROUND_BIT for obstacles
    foot_corners: dict = field(default_factory=dict)    # contact -> (4, 3) sole-plane corner offsets in the site frame (dim 6) or zeros (dim 3)
    floor_geom: int = -1
    settings: MujocoSettings = SETTINGS
    scene_boxes: list = field(default_factory=list)
    visual_pack: object = None

    # --- state in the OPTIMIZER layout ---
    def set_state(self, d: mujoco.MjData, q, v) -> None:
        d.qpos[self.qadr] = q
        d.qvel[self.dofadr] = v
        if self.lock_idx.size:
            d.qpos[self.qadr[self.lock_idx]] = 0.0
            d.qvel[self.dofadr[self.lock_idx]] = 0.0

    def get_state(self, d: mujoco.MjData):
        return d.qpos[self.qadr].copy(), d.qvel[self.dofadr].copy()

    def project_locks(self, d: mujoco.MjData) -> None:
        if self.lock_idx.size:
            d.qpos[self.qadr[self.lock_idx]] = 0.0
            d.qvel[self.dofadr[self.lock_idx]] = 0.0

    def apply_torque(self, d: mujoco.MjData, u) -> None:
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.act_dofadr] = u

    def feet_in_contact(self, d: mujoco.MjData, max_dist: float = 0.0) -> set:
        """Contact names whose geom has a contact record with dist <= max_dist
        (against anything: floor OR obstacle). `d.contact` must be fresh."""
        g2f = {g: f for f, g in self.foot_geom.items()}
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > max_dist:
                continue
            f = g2f.get(c.geom1) or g2f.get(c.geom2)
            if f is not None:
                out.add(f)
        return out

    def capsule_obstacle_contacts(self, d: mujoco.MjData) -> list:
        """(capsule name, box name, penetration depth) for every capsule-vs-box
        contact record at the current state."""
        g2c = {g: n for n, g in self.capsule_geoms.items()}
        g2b = {g: n for n, g in self.box_geoms.items()}
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            cap = g2c.get(a) or g2c.get(b)
            box = g2b.get(a) or g2b.get(b)
            if cap is not None and box is not None and c.dist <= 0.0:
                out.append((cap, box, float(-c.dist)))
        return out

    def contact_wrenches(self, d: mujoco.MjData) -> dict:
        """Net world wrench on each foot from MuJoCo's contact records: [f; t],
        torque about the foot site (the optimizer's contact frame origin)."""
        out = {f: np.zeros(6) for f in self.foot_geom}
        g2f = {g: f for f, g in self.foot_geom.items()}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            foot = g2f.get(c.geom1) or g2f.get(c.geom2)
            if foot is None:
                continue
            mujoco.mj_contactForce(self.model, d, i, f6)
            R = c.frame.reshape(3, 3)
            fw = R.T @ f6[:3]
            site = d.site_xpos[self.foot_site[foot]]
            tw = R.T @ f6[3:] + np.cross(c.pos - site, fw)
            w = np.concatenate([fw, tw])
            # mj_contactForce reports the force on geom2, in the contact frame.
            out[foot] += w if g2f.get(c.geom2) == foot else -w
        return out

    def foot_positions(self, d: mujoco.MjData) -> dict:
        return {f: d.site_xpos[s].copy() for f, s in self.foot_site.items()}

    def top_contact_points(self, d: mujoco.MjData) -> dict:
        """Measured ground/stone top contact offsets about each foot site."""
        points = {f: [] for f in self.foot_geom}
        g2f = {gid: f for f, gid in self.foot_geom.items()}
        for c in d.contact:
            foot = g2f.get(c.geom1) or g2f.get(c.geom2)
            if foot is not None and c.dist <= 0. and abs(c.frame[2]) > .99:
                points[foot].append(c.pos.copy() - d.site_xpos[self.foot_site[foot]])
        return {f: np.asarray(p).reshape(-1, 3) for f, p in points.items()}

    def foot_bottom_z(self, d: mujoco.MjData) -> dict:
        """Lowest point of the physical contact geom, including foot rotation.

        A point-foot sphere is offset from its site in the rotating foot
        frame. Its bottom equals the site height only while that frame is level.
        """
        out = {}
        for f, gid in self.foot_geom.items():
            size = self.model.geom_size[gid]
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = size[0]
            else:
                assert self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
                extent = np.abs(d.geom_xmat[gid].reshape(3, 3)[2]) @ size
            out[f] = float(d.geom_xpos[gid, 2] - extent)
        return out

    def set_foot_box_mask(self, foot: str, allowed_boxes) -> None:
        """Let `foot` collide with the floor and with the scene boxes named in
        `allowed_boxes` only (the "stones are top faces" model): the box bits
        of the other stones are cleared from the foot geom's contype. Obstacle
        boxes (fence) are on the ground bit and always collide."""
        bits = GROUND_BIT
        for b in allowed_boxes:
            bits |= self.box_bits.get(b, 0)
        self.model.geom_contype[self.foot_geom[foot]] = bits

    def total_mass(self) -> float:
        return float(self.model.body_mass.sum())


# ------------------------------------------------------------------ helpers
def _rot_rpy(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def patch_urdf(urdf_path: str | Path) -> str:
    """Visuals stripped, unresolvable/mesh collisions dropped, massless links
    given 1e-9 kg, remaining mesh paths absolutised."""
    urdf_dir = Path(urdf_path).resolve().parent
    root = ET.parse(str(urdf_path)).getroot()
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            link.remove(vis)
        if link.find("inertial") is None:
            inr = ET.SubElement(link, "inertial")
            ET.SubElement(inr, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inr, "mass", value="1e-9")
            ET.SubElement(inr, "inertia", ixx="1e-12", ixy="0", ixz="0",
                          iyy="1e-12", iyz="0", izz="1e-12")
        for el in list(link.findall("collision")):
            mesh = el.find("geometry/mesh")
            if mesh is None:
                continue
            fn = mesh.get("filename") or ""
            if not fn.startswith(("package://", "/")):
                fn = str(urdf_dir / fn)
                mesh.set("filename", fn)
            if fn.startswith("package://") or not Path(fn).exists():
                link.remove(el)
    return ET.tostring(root, encoding="unicode")


def derive_body2joint(urdf_path: str | Path) -> dict:
    """MuJoCo body name -> the non-fixed URDF joint whose child link it is."""
    root = ET.parse(str(urdf_path)).getroot()
    out = {}
    for joint in root.findall("joint"):
        if joint.get("type") in (None, "fixed"):
            continue
        child = joint.find("child")
        if child is not None:
            out[child.get("link")] = joint.get("name")
    return out


def align_inertias_to_pinocchio(model: mujoco.MjModel, robot: Robot) -> float:
    """Overwrite MuJoCo's body inertials with pinocchio's composite (fixed-joint
    merged) inertias. Returns the largest mass change."""
    pm = robot.model
    max_dm = 0.0
    for bname, jname in derive_body2joint(robot.urdf_path).items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        I = pm.inertias[pm.getJointId(jname)]
        if I.mass < 1e-6:
            continue
        max_dm = max(max_dm, abs(float(model.body_mass[bid]) - I.mass))
        model.body_mass[bid] = I.mass
        model.body_ipos[bid] = I.lever
        w, V = np.linalg.eigh(I.inertia)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        model.body_inertia[bid] = w
        model.body_iquat[bid] = _quat_from_mat(V)
        model.body_sameframe[bid] = 0
    return max_dm


def robot_yaml(name: str) -> dict:
    return yaml.safe_load(open(ROBOTS_DIR / f"{name}.yaml"))


def frame_anchor(robot: Robot, frame_name: str):
    """(parent joint name, placement of the frame in the parent joint frame) --
    the MuJoCo body of that joint has the joint frame as its body frame."""
    m = robot.model
    fid = m.getFrameId(frame_name)
    assert fid < m.nframes, frame_name
    fr = m.frames[fid]
    return m.names[fr.parentJoint], fr.placement


def capsule_endpoints_in_body(robot: Robot, cap):
    """World-fixed-in-body endpoints of a yaml capsule, expressed in the frame
    of the parent JOINT of `cap.frame` (= that MuJoCo body's frame). Uses FK
    at q = 0: the yaml guarantees both endpoints are body-fixed in cap.frame."""
    import pinocchio as pin
    m = robot.model
    d = m.createData()
    pin.framesForwardKinematics(m, d, np.zeros(m.nq))
    fa = m.getFrameId(cap.frame)
    fb = m.getFrameId(cap.b_frame)
    jid = m.frames[fa].parentJoint
    oMj = d.oMi[jid]
    a_w = d.oMf[fa].act(cap.a_local)
    b_w = d.oMf[fb].act(cap.b_local)
    return m.names[jid], oMj.actInv(a_w), oMj.actInv(b_w)


# ------------------------------------------------------------------ builder
def build(name: str, *,
          scene_boxes: list | None = None,
          planarlock: tuple = (),
          capsule_collision: bool = True,
          visual: bool = False,
          settings: MujocoSettings = SETTINGS,
          foot_radius: float = FOOT_RADIUS,
          foot_box_half_height: float = FOOT_BOX_HALF_HEIGHT,
          sole_margin: float = SOLE_MARGIN,
          floor_extent: float = 12.0) -> RobotSim:
    """Compile `robots/<name>.yaml`'s robot into a MuJoCo model (see module doc).

    `scene_boxes`: list of SceneBox (or dicts with name/center/half) -- the
    problem's obstacle boxes. `planarlock`: base joint names the problem pins to
    zero. `visual` (rendering dress-up) is not part of this package."""
    if visual:
        raise NotImplementedError("leaptrack.scene.build: rendering (visual=True) is not ported")

    robot = load_robot(ROBOTS_DIR / f"{name}.yaml")
    doc = robot_yaml(name)
    contact_dims = {c["frame"]: int(c["dim"]) for c in doc["contacts"]}
    foot_boxes = {c["frame"]: c.get("foot_box") for c in doc["contacts"]}

    spec = mujoco.MjSpec.from_string(patch_urdf(robot.urdf_path))
    apply_options(spec, settings)
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = True

    # every URDF geom: non-colliding (the optimizer models no such contacts)
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
        g.group = 3

    # floor
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [floor_extent, floor_extent, 0.1]
    apply_contact_geom(floor, settings)
    floor.contype = GROUND_BIT
    floor.conaffinity = GROUND_BIT

    # scene boxes. Obstacles (fence, `collide_capsules`): collide with feet
    # (ground bit) AND capsules (obstacle bit). Stones (`collide_capsules`
    # False): each gets ITS OWN bit so a foot's collision with each stone can
    # be switched per control tick (`set_foot_box_mask`); capsules never
    # collide with them (the problem's stone keep-out is an inset box judged
    # by the constraint checker, not simulated).
    boxes = []
    box_bits = {}
    n_stone = 0
    for sb in (scene_boxes or ()):
        if isinstance(sb, dict):
            sb = SceneBox(str(sb["name"]), np.asarray(sb["center"], float),
                          np.asarray(sb["half"], float), tuple(sb.get("rgba", SceneBox.rgba)),
                          bool(sb.get("collide_capsules", True)))
        boxes.append(sb)
        g = spec.worldbody.add_geom()
        g.name = f"box_{sb.name}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [float(x) for x in sb.center]
        g.size = [float(x) for x in sb.half]
        apply_contact_geom(g, settings)
        if sb.collide_capsules:
            g.contype = GROUND_BIT | OBSTACLE_BIT
            g.conaffinity = GROUND_BIT | OBSTACLE_BIT
            box_bits[sb.name] = GROUND_BIT
        else:
            bit = STONE_BIT0 << n_stone
            n_stone += 1
            assert bit < (1 << 31), "too many stones for per-stone collision bits"
            g.contype = 0
            g.conaffinity = bit
            box_bits[sb.name] = bit
        g.rgba = list(sb.rgba)
    all_stone_bits = 0
    for b in box_bits.values():
        if b != GROUND_BIT:
            all_stone_bits |= b

    # contact sites + geoms on the parent joint's body
    joint_body = {}
    for b in spec.bodies:
        for j in b.joints:
            joint_body[j.name] = b
    site_names, geom_names, foot_corners = {}, {}, {}
    for cname, dim in contact_dims.items():
        jname, plc = frame_anchor(robot, cname)
        body = joint_body[jname]
        R, p = plc.rotation, plc.translation
        s = body.add_site()
        s.name = f"{cname}_site"
        s.pos = list(p)
        s.quat = list(_quat_from_mat(R))
        s.size = [0.004, 0.004, 0.004]
        s.group = 4
        site_names[cname] = s.name
        cg = body.add_geom()
        cg.name = f"{cname}_contact"
        cg.quat = list(_quat_from_mat(R))
        if dim == 6:
            fb = foot_boxes[cname]
            h = foot_box_half_height
            cg.type = mujoco.mjtGeom.mjGEOM_BOX
            cg.size = [float(fb["x"]) + sole_margin, float(fb["y"]) + sole_margin, h]
            cg.pos = list(p + R @ np.array([0.0, 0.0, h]))
        else:
            cg.type = mujoco.mjtGeom.mjGEOM_SPHERE
            cg.size = [foot_radius, 0.0, 0.0]
            cg.pos = list(p + R @ np.array([0.0, 0.0, foot_radius]))
        apply_contact_geom(cg, settings)
        cg.contype = GROUND_BIT | all_stone_bits
        cg.conaffinity = GROUND_BIT
        cg.rgba = [0.1, 0.1, 0.1, 1.0]
        cg.group = 3
        geom_names[cname] = cg.name
        if dim == 6:
            fb = foot_boxes[cname]
            foot_corners[cname] = np.array([[sx * (float(fb["x"]) + sole_margin), sy * (float(fb["y"]) + sole_margin), 0.0]
                                            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
        else:
            foot_corners[cname] = np.zeros((0, 3))

    # collision capsules vs obstacle boxes only
    cap_names = {}
    if capsule_collision and boxes:
        for cap in robot.capsules.values():
            jname, a_b, b_b = capsule_endpoints_in_body(robot, cap)
            body = joint_body[jname]
            g = body.add_geom()
            g.name = f"capsule_{cap.name}"
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.fromto = [*a_b, *b_b]
            g.size = [float(cap.radius), 0.0, 0.0]
            apply_contact_geom(g, settings)
            g.contype = OBSTACLE_BIT      # capsule.contype & box.conaffinity != 0
            g.conaffinity = 0             # nothing else ever tests against it
            g.rgba = [0.2, 0.5, 0.9, 0.25]
            g.group = 3
            cap_names[cap.name] = g.name

    model = spec.compile()
    if boxes:
        # collision midphase off: every geom pair is tested directly
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    align_inertias_to_pinocchio(model, robot)
    # the optimizer's pinocchio model has no joint damping/friction/armature
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    model.dof_armature[:] = 0.0

    names = list(robot.model.names)[1:]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    missing = [n for n, j in zip(names, jids) if j < 0]
    if missing:
        raise KeyError(f"{name}: joints missing from the MuJoCo model: {missing}")
    qadr = np.array([model.jnt_qposadr[j] for j in jids], int)
    dofadr = np.array([model.jnt_dofadr[j] for j in jids], int)
    act_dofadr = dofadr[robot.act_idx]
    lock_idx = np.array([names.index(j) for j in planarlock], int)
    base_body = int(model.jnt_bodyid[jids[5]])
    return RobotSim(
        name=name, robot=robot, model=model, spec=spec, qadr=qadr, dofadr=dofadr,
        act_dofadr=act_dofadr, lock_idx=lock_idx,
        foot_site={c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for c, s in site_names.items()},
        foot_geom={c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for c, g in geom_names.items()},
        contact_dims=contact_dims, base_body=base_body,
        capsule_geoms={c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for c, g in cap_names.items()},
        box_geoms={b.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"box_{b.name}") for b in boxes},
        box_bits=box_bits, foot_corners=foot_corners,
        floor_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        settings=settings, scene_boxes=boxes, visual_pack=None)


# ------------------------------------------------------------------ gate
def check_dynamics(rs: RobotSim, n: int = 100, seed: int = 0) -> dict:
    """MuJoCo vs pinocchio, contacts/constraints disabled, at random states:
    max relative acceleration error, max contact-frame Jacobian error, max
    site-vs-frame position error, and capsule endpoint error."""
    import pinocchio as pin
    m, d = rs.model, mujoco.MjData(rs.model)
    pm = rs.robot.model
    pd = pm.createData()
    rng = np.random.default_rng(seed)
    nv = pm.nv
    act = rs.robot.act_idx
    lo = np.maximum(np.asarray(pm.lowerPositionLimit)[act], -1.2)
    hi = np.minimum(np.asarray(pm.upperPositionLimit)[act], 1.2)
    bad = lo > hi
    lo = np.where(bad, np.asarray(pm.lowerPositionLimit)[act], lo)
    hi = np.where(bad, np.asarray(pm.upperPositionLimit)[act], hi)
    a_rel, j_err, p_err, cap_err = 0.0, 0.0, 0.0, 0.0
    saved = int(m.opt.disableflags)
    m.opt.disableflags = saved | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    try:
        for _ in range(n):
            q = np.zeros(nv)
            q[0:2] = rng.uniform(-0.3, 0.3, 2)
            q[3:6] = rng.uniform(-0.4, 0.4, 3)
            q[act] = rng.uniform(lo, hi)
            q[2] = rng.uniform(0.5, 1.5)
            v = rng.uniform(-2.0, 2.0, nv)
            tau = rng.uniform(-30.0, 30.0, nv)
            a_pin = pin.aba(pm, pd, q, v, tau)
            rs.set_state(d, q, v) if not rs.lock_idx.size else None
            d.qpos[rs.qadr] = q
            d.qvel[rs.dofadr] = v
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[rs.dofadr] = tau
            mujoco.mj_forward(m, d)
            a_mj = d.qacc[rs.dofadr]
            a_rel = max(a_rel, float(np.linalg.norm(a_pin - a_mj) / (1.0 + np.linalg.norm(a_pin))))
            pin.computeJointJacobians(pm, pd, q)
            pin.framesForwardKinematics(pm, pd, q)
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            for cname, sid in rs.foot_site.items():
                fid = pm.getFrameId(cname)
                J = pin.getFrameJacobian(pm, pd, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                mujoco.mj_jacSite(m, d, jacp, jacr, sid)
                Jmj = np.vstack([jacp, jacr])[:, rs.dofadr]
                j_err = max(j_err, float(np.abs(J - Jmj).max()))
                p_err = max(p_err, float(np.abs(pd.oMf[fid].translation - d.site_xpos[sid]).max()))
            for cname, gid in rs.capsule_geoms.items():
                cap = rs.robot.capsules[cname]
                A = pd.oMf[pm.getFrameId(cap.frame)].act(cap.a_local)
                B = pd.oMf[pm.getFrameId(cap.b_frame)].act(cap.b_local)
                # MuJoCo capsule: center + half-length along local z
                R = d.geom_xmat[gid].reshape(3, 3)
                c = d.geom_xpos[gid]
                hl = m.geom_size[gid, 1]
                ends = np.array([c - hl * R[:, 2], c + hl * R[:, 2]])
                e1 = max(np.abs(ends[0] - A).max(), np.abs(ends[1] - B).max())
                e2 = max(np.abs(ends[0] - B).max(), np.abs(ends[1] - A).max())
                cap_err = max(cap_err, min(e1, e2))
    finally:
        m.opt.disableflags = saved
        d.qfrc_applied[:] = 0.0
    return dict(a_rel=a_rel, jac=j_err, site_pos=p_err, capsule=cap_err,
                mass_mj=float(m.body_mass.sum()), mass_pin=rs.robot.total_mass())
