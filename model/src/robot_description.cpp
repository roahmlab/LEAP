#include "leap/model/robot_description.hpp"

#include <cmath>
#include <stdexcept>

#include "leap/io/yaml_doc.hpp"

namespace leap {

namespace {

[[noreturn]] void bad(const std::string& msg) { throw std::runtime_error("robot yaml: " + msg); }

std::vector<double> scalarOrVector(const YamlDoc& d, const std::string& key) {
  if (!d.has(key)) return {};
  if (d.seqSize(key) >= 0) return d.getVecD(key);
  return {d.getD(key, 0.0)};
}

void checkFinite(const std::vector<double>& values, const std::string& key, bool nonnegative) {
  for (double v : values) {
    if (!std::isfinite(v)) bad(key + " entries must be finite");
    if (nonnegative && v < 0.0) bad(key + " entries must be nonnegative");
  }
}

int indexOf(const std::vector<std::string>& names, const char* n) {
  for (size_t i = 0; i < names.size(); ++i)
    if (names[i] == n) return static_cast<int>(i);
  return -1;
}

bool readPoint(const YamlDoc& d, const std::string& key, Eigen::Vector3d& out) {
  if (!d.has(key)) return false;
  const std::vector<double> v = d.getVecD(key);
  if (v.size() != 3) bad(key + " must have 3 components");
  out = Eigen::Vector3d(v[0], v[1], v[2]);
  return true;
}

void readLimits(const YamlDoc& d, const std::string& dotted, std::string& source, std::vector<double>& vals) {
  if (!d.has(dotted)) return;
  if (d.seqSize(dotted) >= 0) {
    source = "explicit";
    vals = d.getVecD(dotted);
  } else {
    source = d.getS(dotted, "from_urdf");
    if (source != "from_urdf") bad(dotted + " must be 'from_urdf' or an explicit list, got '" + source + "'");
  }
}

RobotDescription parse(const YamlDoc& d, const std::string& yamlPath) {
  d.checkKeys("", {"schema", "name", "urdf", "base", "gravity", "joint_order", "actuated_joints", "contacts",
                   "monitor_frames", "collision_capsules", "actuation", "sim"});
  d.checkKeys("base", {"type"});
  d.checkKeys("actuation", {"motor_model", "motor_parameters", "saturation", "lag", "effort_limits",
                            "velocity_limits", "motors"});
  d.checkKeys("actuation.motor_parameters", {"coulomb", "viscous", "armature", "bias", "velocity_smoothing"});
  d.checkKeys("actuation.saturation",
              {"enabled", "enforce_in_optimization", "effort_scale", "speed_derating", "knee_fraction"});
  d.checkKeys("actuation.lag", {"enabled", "time_constant"});

  RobotDescription rd;
  if (d.getI("schema", 1) != 1) bad("unsupported schema " + std::to_string(d.getI("schema", 1)));
  rd.yamlPath = yamlPath;
  rd.sourceText = d.sourceText();
  rd.name = d.getS("name", "");
  if (rd.name.empty()) bad("missing 'name'");

  const std::string urdf = d.getS("urdf", "");
  if (urdf.empty()) bad("missing 'urdf'");
  rd.urdfPath = d.resolvePath(urdf);

  rd.baseType = d.getS("base.type", "");
  if (rd.baseType != "floating_explicit_xyz_euler" && rd.baseType != "fixed")
    bad("base.type must be floating_explicit_xyz_euler or fixed, got '" + rd.baseType + "'");

  const std::vector<double> g = d.getVecD("gravity");
  if (g.size() != 3) bad("gravity must have 3 components");
  rd.gravity = Eigen::Vector3d(g[0], g[1], g[2]);

  rd.expectedJointOrder = d.getVecS("joint_order");
  rd.actuatedJoints = d.getVecS("actuated_joints");
  rd.monitorFrames = d.getVecS("monitor_frames");

  // The base's Euler chart follows the declared joint order.
  const int ir = indexOf(rd.expectedJointOrder, "base_rx");
  const int ip = indexOf(rd.expectedJointOrder, "base_ry");
  const int iy = indexOf(rd.expectedJointOrder, "base_rz");
  if (ir >= 0 && ip >= 0 && iy >= 0) {
    rd.baseRot.roll = ir;
    rd.baseRot.pitch = ip;
    rd.baseRot.yaw = iy;
    rd.baseRot.yawIsWorld = (iy < ir && iy < ip);
  }

  const int nc = d.seqSize("contacts");
  for (int i = 0; i < nc; ++i) {
    const std::string base = "contacts." + std::to_string(i);
    d.checkKeys(base, {"frame", "dim", "collision", "foot_box"});
    d.checkKeys(base + ".collision", {"type", "radius"});
    d.checkKeys(base + ".foot_box", {"x", "y"});
    ContactFrameSpec c;
    c.frameName = d.getS(base + ".frame", "");
    if (c.frameName.empty()) bad(base + ": missing 'frame'");
    c.dim = d.getI(base + ".dim", 3);
    if (c.dim != 2 && c.dim != 3 && c.dim != 6) bad(base + ": dim must be 2, 3, or 6");
    c.collisionType = d.getS(base + ".collision.type", "");
    c.collisionRadius = d.getD(base + ".collision.radius", 0.0);
    if (d.has(base + ".foot_box")) {
      if (c.dim != 6) bad(base + ": foot_box requires dim 6");
      c.footBoxX = d.getD(base + ".foot_box.x", 0.0);
      c.footBoxY = d.getD(base + ".foot_box.y", 0.0);
      if (!(c.footBoxX > 0.0) || !(c.footBoxY > 0.0)) bad(base + ": foot_box x/y must be positive");
    }
    rd.contacts.push_back(c);
  }

  const int ncap = d.seqSize("collision_capsules");
  for (int i = 0; i < ncap; ++i) {
    const std::string base = "collision_capsules." + std::to_string(i);
    d.checkKeys(base, {"name", "frame", "a", "b", "to_frame", "radius"});
    CollisionCapsuleSpec c;
    c.name = d.getS(base + ".name", "");
    if (c.name.empty()) bad(base + ": missing 'name'");
    c.frameName = d.getS(base + ".frame", "");
    if (c.frameName.empty()) bad(base + ": missing 'frame'");
    c.toFrame = d.getS(base + ".to_frame", "");
    readPoint(d, base + ".a", c.a);
    const bool hasB = readPoint(d, base + ".b", c.b);
    if (c.toFrame.empty() && !hasB) bad(base + ": needs either 'b' (local endpoint) or 'to_frame'");
    if (!c.toFrame.empty() && hasB) bad(base + ": 'b' and 'to_frame' are mutually exclusive");
    c.radius = d.getD(base + ".radius", 0.0);
    if (!(c.radius > 0.0) || !std::isfinite(c.radius)) bad(base + ": radius must be positive and finite");
    rd.capsules.push_back(c);
  }

  MotorModelConfig& motor = rd.motor;
  motor.type = d.getS("actuation.motor_model", "none");
  if (motor.type != "none" && motor.type != "friction_inertia")
    bad("actuation.motor_model must be 'none' or 'friction_inertia', got '" + motor.type + "'");
  motor.coulomb = scalarOrVector(d, "actuation.motor_parameters.coulomb");
  motor.viscous = scalarOrVector(d, "actuation.motor_parameters.viscous");
  motor.armature = scalarOrVector(d, "actuation.motor_parameters.armature");
  motor.bias = scalarOrVector(d, "actuation.motor_parameters.bias");
  motor.velocitySmoothing = d.getD("actuation.motor_parameters.velocity_smoothing", motor.velocitySmoothing);
  checkFinite(motor.coulomb, "actuation.motor_parameters.coulomb", true);
  checkFinite(motor.viscous, "actuation.motor_parameters.viscous", true);
  checkFinite(motor.armature, "actuation.motor_parameters.armature", true);
  checkFinite(motor.bias, "actuation.motor_parameters.bias", false);
  if (!std::isfinite(motor.velocitySmoothing) || motor.velocitySmoothing <= 0.0)
    bad("actuation.motor_parameters.velocity_smoothing must be positive and finite");
  if (motor.type == "none" && d.has("actuation.motor_parameters"))
    bad("actuation.motor_parameters requires motor_model: friction_inertia");

  MotorSaturationConfig& sat = rd.saturation;
  sat.enabled = d.getB("actuation.saturation.enabled", sat.enabled);
  sat.enforceInOptimization = d.getB("actuation.saturation.enforce_in_optimization", sat.enforceInOptimization);
  sat.effortScale = d.getD("actuation.saturation.effort_scale", sat.effortScale);
  sat.speedDerating = d.getB("actuation.saturation.speed_derating", sat.speedDerating);
  sat.kneeFraction = d.getD("actuation.saturation.knee_fraction", sat.kneeFraction);
  if (!std::isfinite(sat.effortScale) || sat.effortScale <= 0.0)
    bad("actuation.saturation.effort_scale must be positive and finite");
  if (!std::isfinite(sat.kneeFraction) || sat.kneeFraction < 0.0 || sat.kneeFraction >= 1.0)
    bad("actuation.saturation.knee_fraction must be in [0, 1)");

  MotorLagConfig& lag = rd.lag;
  lag.enabled = d.getB("actuation.lag.enabled", lag.enabled);
  lag.timeConstant = d.getD("actuation.lag.time_constant", lag.timeConstant);
  if (!std::isfinite(lag.timeConstant) || lag.timeConstant <= 0.0)
    bad("actuation.lag.time_constant must be positive and finite");

  readLimits(d, "actuation.effort_limits", rd.effortLimitsSource, rd.effortLimits);
  readLimits(d, "actuation.velocity_limits", rd.velocityLimitsSource, rd.velocityLimits);
  return rd;
}

}  // namespace

RobotDescription loadRobotDescription(const std::string& yamlPath) {
  return parse(YamlDoc::fromFile(yamlPath), yamlPath);
}

RobotDescription parseRobotDescription(const std::string& yamlText) {
  return parse(YamlDoc::fromString(yamlText), "");
}

}  // namespace leap
