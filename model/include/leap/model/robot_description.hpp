#pragma once

// RobotDescription: everything a URDF cannot say about a robot, read from robots/*.yaml.
//
//   urdf                 path (relative to the yaml) of the URDF
//   base.type            floating_explicit_xyz_euler | fixed
//   gravity              world gravity vector
//   joint_order          hard check of the URDF parse (q layout)
//   actuated_joints      columns of the actuation matrix B, in this order
//   contacts             the force-bearing frames (dim 2 / 3 point force, 6 flat-foot wrench)
//   monitor_frames       position-only frames read by ground keep-out rows
//   collision_capsules   the robot's capsule collision model (obstacle / self-collision rows)
//   actuation            motor model and the source of the effort / velocity limits
//
// The floating base is EXPLICIT: six 1-DOF joints (prismatic px, py, pz then revolute
// rx, ry, rz), so q is a vector space with nq == nv. Pinocchio's quaternion free-flyer
// is not used.

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace leap {

struct ContactFrameSpec {
  std::string frameName;
  int dim = 3;          // 2: world x, y rows; 3: x, y, z point force; 6: wrench [f; tau]
  double footBoxX = 0;  // dim 6 only: CoP rectangle half extents of the sole (informational)
  double footBoxY = 0;
  std::string collisionType;  // informational (simulator): "" when absent
  double collisionRadius = 0.0;
};

// Passive reflected joint-side motor model, in actuated-joint order:
//   loss(v, a) = coulomb * smoothSign(v) + viscous * v + armature * a + bias.
// The armature is folded into pinocchio's rotor inertia so it participates in every
// dynamics algorithm; the rest enters the dynamics residual as an extra actuated torque.
struct MotorModelConfig {
  std::string type = "none";  // "none" | "friction_inertia"
  std::vector<double> coulomb, viscous, armature, bias;  // empty = 0; one entry = broadcast
  double velocitySmoothing = 0.05;                       // rad/s, smooth-sign scale
};

struct MotorSaturationConfig {
  bool enabled = false;
  bool enforceInOptimization = true;
  double effortScale = 1.0;
  bool speedDerating = true;
  double kneeFraction = 0.6;
};

struct MotorLagConfig {
  bool enabled = false;
  double timeConstant = 0.004;
};

// One collision capsule: the segment between two body-fixed points swept by a sphere.
// The second endpoint is either `b` in the frame or the origin of `toFrame` (resolved
// at model construction and checked to be rigidly attached to `frameName`).
struct CollisionCapsuleSpec {
  std::string name;
  std::string frameName;
  Eigen::Vector3d a{0.0, 0.0, 0.0};
  Eigen::Vector3d b{0.0, 0.0, 0.0};
  std::string toFrame;
  double radius = 0.0;
};

// Where the explicit floating base's rotation coordinates live in q (an Euler chart is
// singular where its middle angle reaches +-pi/2; the chain order decides where).
struct BaseRotationLayout {
  int roll = 3, pitch = 4, yaw = 5;
  bool yawIsWorld = false;  // true when yaw is the outermost rotation
};

struct RobotDescription {
  std::string name;
  std::string yamlPath;  // "" when parsed from a string
  std::string sourceText;
  std::string baseType = "floating_explicit_xyz_euler";
  std::string urdfPath;  // resolved against the yaml's directory
  Eigen::Vector3d gravity{0.0, 0.0, -9.81};
  std::vector<std::string> expectedJointOrder;  // optional hard check
  std::vector<std::string> actuatedJoints;      // empty = every joint (fixed base)
  std::vector<ContactFrameSpec> contacts;
  std::vector<std::string> monitorFrames;
  std::vector<CollisionCapsuleSpec> capsules;
  BaseRotationLayout baseRot;
  MotorModelConfig motor;
  MotorSaturationConfig saturation;
  MotorLagConfig lag;
  std::string effortLimitsSource = "from_urdf";    // or "explicit"
  std::vector<double> effortLimits;                // explicit only, actuated order
  std::string velocityLimitsSource = "from_urdf";
  std::vector<double> velocityLimits;
};

RobotDescription loadRobotDescription(const std::string& yamlPath);
RobotDescription parseRobotDescription(const std::string& yamlText);

}  // namespace leap
