"""robots/*.yaml -> RobotDescription (Python side).

Parses the same physical motor/saturation/lag schema as the C++ loader, plus
the simulator-owned `sim:` contact defaults and legacy
`actuation.motors:` command-model presets. Missing optional fields retain the
backward-compatible no-motor defaults.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class FootBox:
    """Physical CoP-rectangle half-extents of a flat foot (dim-6 contact), in
    the contact frame's x/y axes. The frame sits at the sole-rectangle center."""
    x: float
    y: float


@dataclass(frozen=True)
class SimConfig:
    foot_radius: float = 0.005          # sim collision sphere (optimizer feet are POINTS)
    foot_box_half_height: float = 0.004  # box-foot half height (dim-6 contacts);
                                         # the box's bottom face sits on the sole plane
    friction: float = 1.0               # ground friction (grip margin over the plan's mu)
    restitution: float = 0.0            # plastic impacts
    timestep: float = 5.0e-4            # <= 0.5 ms for stiff plastic contact + PD
    solref: tuple = (0.002, 1.0)        # dampratio 1 -> inelastic
    solimp: tuple = (0.95, 0.99, 0.001, 0.5, 2.0)
    fall_min_z: object = None           # per-robot survival thresholds for rollout
    fall_rpy_abs: object = None         # summaries (None -> the Go2 module defaults)
    camera: object = None               # dict(distance, azimuth, elevation) or None
    visual_mesh_xml: object = None      # donor MuJoCo xml with the render meshes
                                        # (Path or None; simulation/visual_meshes.py)


@dataclass(frozen=True)
class MotorPhysicsConfig:
    model: str = "none"
    coulomb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    viscous: np.ndarray = field(default_factory=lambda: np.zeros(0))
    armature: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))
    velocity_smoothing: float = 0.05

    def loss(self, v_actuated: np.ndarray, *, smooth: bool) -> np.ndarray:
        """Joint-side Fc*sign(v) + Fv*v + beta (armature lives in M)."""
        v = np.asarray(v_actuated, float)
        if self.model == "none":
            return np.zeros_like(v)
        if smooth:
            eps = self.velocity_smoothing
            sign = v / np.sqrt(v*v + eps*eps)
        else:
            sign = np.sign(v)
        return self.coulomb * sign + self.viscous * v + self.bias


@dataclass(frozen=True)
class CollisionCapsule:
    """One capsule of the robot's collision model (robot yaml
    `collision_capsules:`), as DECLARED: `to_frame` is still symbolic here (the
    C++ RobotModel resolves it against the URDF and the archive echoes the
    resolved offsets, so prefer pyleap.collision.CollisionModel when reading a
    solution)."""
    name: str
    frame: str
    radius: float
    a: tuple = (0.0, 0.0, 0.0)
    b: tuple = (0.0, 0.0, 0.0)
    to_frame: str = ""


@dataclass(frozen=True)
class SaturationConfig:
    enabled: bool = False
    enforce_in_optimization: bool = True
    effort_scale: float = 1.0
    speed_derating: bool = True
    knee_fraction: float = 0.6


@dataclass(frozen=True)
class LagConfig:
    enabled: bool = False
    time_constant: float = 0.004


@dataclass(frozen=True)
class ActuationConfig:
    motor_model: str = "none"
    motor: MotorPhysicsConfig = field(default_factory=MotorPhysicsConfig)
    saturation: SaturationConfig = field(default_factory=SaturationConfig)
    lag: LagConfig = field(default_factory=LagConfig)
    effort_limits: object = "from_urdf"   # "from_urdf" | np.ndarray (actuated order)
    velocity_limits: object = "from_urdf"
    motors: dict = field(default_factory=dict)  # name -> params (simulation/motors.py)


@dataclass(frozen=True)
class RobotDescription:
    name: str
    yaml_path: Path
    urdf: Path                           # resolved absolute
    base_type: str                       # floating_explicit_xyz_euler | fixed
    gravity: tuple
    joint_order: tuple                   # may be empty
    actuated_joints: tuple               # empty = all 1-DOF joints
    contact_frames: tuple
    contact_dims: tuple
    monitor_frames: tuple
    sim: SimConfig
    actuation: ActuationConfig
    contact_boxes: tuple = ()            # per contact: FootBox or None (aligned
                                         # with contact_frames; dim-6 flat feet)
    collision_capsules: tuple = ()       # CollisionCapsule per declared capsule

    def effort_limits(self) -> np.ndarray:
        if isinstance(self.actuation.effort_limits, np.ndarray):
            return self.actuation.effort_limits
        eff, _ = urdf_joint_limits(self.urdf, list(self.actuated_joints))
        return eff

    def velocity_limits(self) -> np.ndarray:
        if isinstance(self.actuation.velocity_limits, np.ndarray):
            return self.actuation.velocity_limits
        _, vel = urdf_joint_limits(self.urdf, list(self.actuated_joints))
        return vel


def urdf_joint_limits(urdf_path: str | Path,
                      joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(effort, velocity) limits per joint in `joint_names` order, straight from
    the URDF <limit> tags (no pinocchio needed). Empty names = every non-fixed
    joint in URDF order."""
    root = ET.parse(urdf_path).getroot()
    limits = {}
    order = []
    for j in root.iter("joint"):
        if j.get("type") in (None, "fixed"):
            continue
        lim = j.find("limit")
        limits[j.get("name")] = (
            float(lim.get("effort", "inf")) if lim is not None else np.inf,
            float(lim.get("velocity", "inf")) if lim is not None else np.inf,
        )
        order.append(j.get("name"))
    names = joint_names if joint_names else order
    missing = [n for n in names if n not in limits]
    if missing:
        raise KeyError(f"joints not in {urdf_path}: {missing}")
    eff = np.array([limits[n][0] for n in names])
    vel = np.array([limits[n][1] for n in names])
    return eff, vel


def _limits_field(v) -> object:
    if v is None or v == "from_urdf" or v == "urdf":
        return "from_urdf"
    return np.asarray(v, float)


def _motor_vector(v, n: int, key: str, *, nonnegative: bool) -> np.ndarray:
    if v is None:
        out = np.zeros(n)
    else:
        raw = np.atleast_1d(np.asarray(v, float))
        if len(raw) == 1:
            out = np.full(n, float(raw[0]))
        elif len(raw) == n:
            out = raw.copy()
        else:
            raise ValueError(f"{key}: expected a scalar or {n} values, got {len(raw)}")
    if not np.isfinite(out).all():
        raise ValueError(f"{key}: values must be finite")
    if nonnegative and np.any(out < 0):
        raise ValueError(f"{key}: values must be nonnegative")
    return out


def load_robot(yaml_path: str | Path) -> RobotDescription:
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        doc = yaml.safe_load(f) or {}
    return _robot_from_doc(doc, yaml_path)


def load_robot_text(text: str, *, base_dir: str | Path = ".",
                    name: str | None = None) -> RobotDescription:
    """Parse a robots/*.yaml document from TEXT -- e.g. the verbatim robot-yaml
    echo a solution archive carries under `meta["robot"]["yaml"]` -- without a
    file on disk. Relative paths in the document (urdf, sim.visual_mesh_xml)
    resolve against `base_dir`; when parsing an archive echo away from the
    original repo layout they may not point at real files, but everything else
    (contacts + foot boxes, sim block, actuation) is fully usable."""
    doc = yaml.safe_load(text) or {}
    pseudo = (Path(base_dir).resolve()
              / f"{name or doc.get('name', 'robot')}.yaml")
    return _robot_from_doc(doc, pseudo)


def _robot_from_doc(doc: dict, yaml_path: Path) -> RobotDescription:
    schema = int(doc.get("schema", 1))
    if schema != 1:
        raise ValueError(f"{yaml_path}: unsupported robot schema {schema}")
    urdf = (yaml_path.parent / doc["urdf"]).resolve()

    contacts = doc.get("contacts") or []
    sim_doc = doc.get("sim") or {}
    act_doc = doc.get("actuation") or {}
    actuated_joints = tuple(doc.get("actuated_joints") or ())
    if actuated_joints:
        n_act = len(actuated_joints)
    else:
        n_act = len(urdf_joint_limits(urdf, [])[0])
    sim_defaults = SimConfig()
    visual_xml = sim_doc.get("visual_mesh_xml")
    camera = sim_doc.get("camera")
    sim = SimConfig(
        foot_radius=float(sim_doc.get("foot_radius", sim_defaults.foot_radius)),
        foot_box_half_height=float(sim_doc.get(
            "foot_box_half_height", sim_defaults.foot_box_half_height)),
        friction=float(sim_doc.get("friction", sim_defaults.friction)),
        restitution=float(sim_doc.get("restitution", sim_defaults.restitution)),
        timestep=float(sim_doc.get("timestep", sim_defaults.timestep)),
        solref=tuple(sim_doc.get("solref", sim_defaults.solref)),
        solimp=tuple(sim_doc.get("solimp", sim_defaults.solimp)),
        fall_min_z=(float(sim_doc["fall_min_z"])
                    if sim_doc.get("fall_min_z") is not None else None),
        fall_rpy_abs=(float(sim_doc["fall_rpy_abs"])
                      if sim_doc.get("fall_rpy_abs") is not None else None),
        camera=(dict(camera) if camera else None),
        visual_mesh_xml=((yaml_path.parent / visual_xml).resolve()
                         if visual_xml else None),
    )
    motor_model = str(act_doc.get("motor_model", "none"))
    if motor_model not in ("none", "friction_inertia"):
        raise ValueError(f"{yaml_path}: unknown actuation.motor_model '{motor_model}'")
    mp = dict(act_doc.get("motor_parameters") or {})
    if motor_model == "none" and mp:
        raise ValueError(f"{yaml_path}: motor_parameters requires "
                         "motor_model: friction_inertia")
    velocity_smoothing = float(mp.get("velocity_smoothing", 0.05))
    if not np.isfinite(velocity_smoothing) or velocity_smoothing <= 0:
        raise ValueError("actuation.motor_parameters.velocity_smoothing must be positive")
    motor = MotorPhysicsConfig(
        model=motor_model,
        coulomb=_motor_vector(mp.get("coulomb"), n_act, "motor.coulomb",
                               nonnegative=True),
        viscous=_motor_vector(mp.get("viscous"), n_act, "motor.viscous",
                              nonnegative=True),
        armature=_motor_vector(mp.get("armature"), n_act, "motor.armature",
                                nonnegative=True),
        bias=_motor_vector(mp.get("bias"), n_act, "motor.bias",
                           nonnegative=False),
        velocity_smoothing=velocity_smoothing,
    )
    sat_doc = dict(act_doc.get("saturation") or {})
    saturation = SaturationConfig(
        enabled=bool(sat_doc.get("enabled", False)),
        enforce_in_optimization=bool(
            sat_doc.get("enforce_in_optimization", True)),
        effort_scale=float(sat_doc.get("effort_scale", 1.0)),
        speed_derating=bool(sat_doc.get("speed_derating", True)),
        knee_fraction=float(sat_doc.get("knee_fraction", 0.6)),
    )
    if not np.isfinite(saturation.effort_scale) or saturation.effort_scale <= 0:
        raise ValueError("actuation.saturation.effort_scale must be positive")
    if not 0 <= saturation.knee_fraction < 1:
        raise ValueError("actuation.saturation.knee_fraction must be in [0, 1)")
    lag_doc = dict(act_doc.get("lag") or {})
    lag = LagConfig(
        enabled=bool(lag_doc.get("enabled", False)),
        time_constant=float(lag_doc.get("time_constant", 0.004)),
    )
    if not np.isfinite(lag.time_constant) or lag.time_constant <= 0:
        raise ValueError("actuation.lag.time_constant must be positive")

    actuation = ActuationConfig(
        motor_model=motor_model,
        motor=motor,
        saturation=saturation,
        lag=lag,
        effort_limits=_limits_field(act_doc.get("effort_limits")),
        velocity_limits=_limits_field(act_doc.get("velocity_limits")),
        motors=dict(act_doc.get("motors") or {"ideal": {}}),
    )
    return RobotDescription(
        name=str(doc.get("name", yaml_path.stem)),
        yaml_path=yaml_path,
        urdf=urdf,
        base_type=str((doc.get("base") or {}).get("type", "fixed")),
        gravity=tuple(doc.get("gravity", (0.0, 0.0, -9.81))),
        joint_order=tuple(doc.get("joint_order") or ()),
        actuated_joints=actuated_joints,
        contact_frames=tuple(c["frame"] for c in contacts),
        contact_dims=tuple(int(c.get("dim", 3)) for c in contacts),
        contact_boxes=tuple(
            FootBox(x=float(c["foot_box"]["x"]), y=float(c["foot_box"]["y"]))
            if c.get("foot_box") else None
            for c in contacts),
        monitor_frames=tuple(doc.get("monitor_frames") or ()),
        collision_capsules=tuple(
            CollisionCapsule(
                name=str(c["name"]), frame=str(c["frame"]),
                radius=float(c["radius"]),
                a=tuple(float(x) for x in (c.get("a") or (0.0, 0.0, 0.0))),
                b=tuple(float(x) for x in (c.get("b") or (0.0, 0.0, 0.0))),
                to_frame=str(c.get("to_frame") or ""))
            for c in (doc.get("collision_capsules") or ())),
        sim=sim,
        actuation=actuation,
    )
