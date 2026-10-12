"""Load a robots/*.yaml + URDF into a pinocchio model, mirroring the C++
RobotModel conventions (explicit 6x1-DOF floating base, nq == nv, actuated
joints after the base, contact/capsule tables from the yaml).

Shared by every comparison method (aligator / crocoddyl / acados).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml


@dataclasses.dataclass
class Capsule:
    name: str
    frame: str            # parent frame of endpoint a
    a_local: np.ndarray   # endpoint a in `frame` coordinates
    b_frame: str          # frame whose ORIGIN is endpoint b (to_frame), or == frame
    b_local: np.ndarray   # endpoint b in `b_frame` coordinates
    radius: float


@dataclasses.dataclass
class Contact:
    name: str    # yaml `frame` (also used as the contact's name)
    frame_id: int
    dim: int     # 3 = point force, 6 = flat-foot wrench


@dataclasses.dataclass
class Robot:
    name: str
    yaml_path: Path
    urdf_path: Path
    model: pin.Model
    contacts: list          # list[Contact]
    actuated_joints: list   # joint names, order fixes u layout
    act_idx: np.ndarray     # v-indices of actuated joints (columns of S^T)
    effort_limit: np.ndarray    # per actuated joint (from URDF)
    velocity_limit: np.ndarray  # per actuated joint (from URDF)
    q_lower: np.ndarray     # per actuated joint (from URDF)
    q_upper: np.ndarray
    capsules: dict          # name -> Capsule
    gravity: np.ndarray

    @property
    def nq(self):
        return self.model.nq

    @property
    def nv(self):
        return self.model.nv

    @property
    def nu(self):
        return len(self.actuated_joints)

    def data(self):
        return self.model.createData()

    def frame_id(self, name: str) -> int:
        fid = self.model.getFrameId(name)
        assert fid < self.model.nframes, f"frame {name} not in model"
        return fid

    def total_mass(self) -> float:
        return float(sum(inert.mass for inert in self.model.inertias))


def _joint_v_index(model: pin.Model, joint_name: str) -> int:
    jid = model.getJointId(joint_name)
    return model.idx_vs[jid] if hasattr(model, "idx_vs") else model.joints[jid].idx_v


def load_robot(yaml_path) -> Robot:
    yaml_path = Path(yaml_path).resolve()
    spec = yaml.safe_load(open(yaml_path))
    urdf = (yaml_path.parent / spec["urdf"]).resolve()
    model = pin.buildModelFromUrdf(str(urdf))
    model.gravity.linear = np.asarray(spec.get("gravity", [0, 0, -9.81]), dtype=float)

    # sanity: the explicit floating-base layout the whole pipeline assumes
    order = spec["joint_order"]
    got = list(model.names)[1:]
    assert got == order, f"{yaml_path}: joint order mismatch\n yaml {order}\n urdf {got}"
    assert model.nq == model.nv, "explicit-euler base expected (nq == nv)"

    actuated = spec["actuated_joints"]
    act_idx = np.array([_joint_v_index(model, j) for j in actuated], dtype=int)

    # URDF limits, per actuated joint (position limits may be +-inf for continuous)
    eff = np.array([model.effortLimit[i] for i in act_idx])
    vel = np.array([model.velocityLimit[i] for i in act_idx])
    qlo = np.array([model.lowerPositionLimit[i] for i in act_idx])
    qhi = np.array([model.upperPositionLimit[i] for i in act_idx])

    contacts = []
    for c in spec["contacts"]:
        fname = c["frame"]
        contacts.append(Contact(name=fname, frame_id=model.getFrameId(fname),
                                dim=int(c["dim"])))

    capsules = {}
    for c in spec.get("collision_capsules", []):
        frame = c["frame"]
        if frame == "base":
            frame = "base"  # the URDF's floating-base body frame
        if "to_frame" in c:
            b_frame, b_local = c["to_frame"], np.zeros(3)
            a_local = np.asarray(c.get("a", [0, 0, 0]), dtype=float)
        else:
            b_frame, b_local = frame, np.asarray(c["b"], dtype=float)
            a_local = np.asarray(c["a"], dtype=float)
        capsules[c["name"]] = Capsule(c["name"], frame, a_local, b_frame, b_local,
                                      float(c["radius"]))

    return Robot(name=spec["name"], yaml_path=yaml_path, urdf_path=urdf,
                 model=model, contacts=contacts, actuated_joints=actuated,
                 act_idx=act_idx, effort_limit=eff, velocity_limit=vel,
                 q_lower=qlo, q_upper=qhi, capsules=capsules,
                 gravity=np.asarray(spec.get("gravity", [0, 0, -9.81]), dtype=float))


def capsule_endpoints(robot: Robot, data, cap: Capsule):
    """World endpoints of a capsule after framesForwardKinematics(data, q)."""
    fa = robot.model.getFrameId(cap.frame if cap.frame != "base" else "base")
    Ta = data.oMf[fa]
    a = Ta.act(cap.a_local)
    fb = robot.model.getFrameId(cap.b_frame if cap.b_frame != "base" else "base")
    Tb = data.oMf[fb]
    b = Tb.act(cap.b_local)
    return a, b
