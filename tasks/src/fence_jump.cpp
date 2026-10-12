#include "leap/tasks/fence_jump.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "leap/problem/costs.hpp"
#include "leap/problem/self_collision.hpp"
#include "leap/spectral/bernstein.hpp"
#include "leap/problem/mesh_refinement.hpp"
#include "leap/solver/term_scales.hpp"
#include "leap/spectral/cgl_grid.hpp"

namespace leap {

namespace {

constexpr double kGravity = 9.81;
constexpr double kLevelTol = 1e-6;

double vzMinOf(double H) { return H > 0.0 ? std::sqrt(2.0 * kGravity * H) : 0.0; }

std::vector<int> range(int first, int count) {
  std::vector<int> v(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) v[static_cast<size_t>(i)] = first + i;
  return v;
}

int jointIndex(const RobotModel& m, const std::string& name, const char* what) {
  if (!m.hasJoint(name)) {
    std::string all;
    for (const std::string& n : m.jointNames()) all += (all.empty() ? "" : ", ") + n;
    throw std::runtime_error(std::string(what) + ": no joint '" + name + "' in this robot (joints: " + all + ")");
  }
  return m.jointIndex(name);
}

bool contains(const std::vector<std::string>& v, const std::string& s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

// The pose the yaml describes with the base height derived: joint angles by name, base
// pose zero, then the base dropped until the lowest contact frame sits on z = 0. The
// contacts must be level and every flat foot flat.
Eigen::VectorXd poseFromAngles(const RobotModel& m, ModelData& data, ModelEvalCache& cache,
                               const std::vector<PoseAngle>& poses, bool squat, const char* what) {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(m.nq());
  for (const PoseAngle& pa : poses) q(jointIndex(m, pa.joint, what)) = squat ? pa.squat : pa.stand;
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(m.nv());
  unsigned needs = kFootPos;
  for (int c = 0; c < m.nContacts(); ++c)
    if (m.contactDim(c) == 6) needs |= kFootRot;
  m.evalNode(data, q, zero, zero, needs, cache);
  double zMin = cache.footPos[0](2), zMax = zMin;
  for (int c = 1; c < m.nContacts(); ++c) {
    const double z = cache.footPos[static_cast<size_t>(c)](2);
    zMin = std::min(zMin, z);
    zMax = std::max(zMax, z);
  }
  if (zMax - zMin > kLevelTol) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s: contact frames are not level (heights span %.6f m); the pose angles must place every contact at one height", what, zMax - zMin);
    throw std::runtime_error(buf);
  }
  for (int c = 0; c < m.nContacts(); ++c) {
    if (m.contactDim(c) != 6) continue;
    const double tilt = cache.footRotLog[static_cast<size_t>(c)].norm();
    if (tilt > kLevelTol) {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%s: flat-foot contact '%s' sole is not flat (|log3(R)| = %.6f rad)", what, m.contactName(c).c_str(), tilt);
      throw std::runtime_error(buf);
    }
  }
  q(2) = -zMin;
  return q;
}

double smoothstep(double s) { return s * s * (3.0 - 2.0 * s); }
double smootherstep(double s) { return s * s * s * (10.0 + s * (-15.0 + 6.0 * s)); }
double bump(double s) { return 16.0 * s * s * (1.0 - s) * (1.0 - s); }

std::vector<double> gradedWidths(int k, double grade, bool gradeEnds, bool towardStart) {
  std::vector<double> w(static_cast<size_t>(k), 1.0);
  if (k > 1 && grade != 1.0) {
    for (int i = 0; i < k; ++i) {
      const int oneSided = towardStart ? k - 1 - i : i;
      const int exponent = gradeEnds ? std::min(i, k - 1 - i) : oneSided;
      w[static_cast<size_t>(i)] = std::pow(grade, exponent);
    }
  }
  double sum = 0.0;
  for (double x : w) sum += x;
  for (double& x : w) x /= sum;
  return w;
}

Eigen::VectorXd qAt(const Eigen::VectorXd& base, const Eigen::Vector2d& d) {
  Eigen::VectorXd q = base;
  q(0) = d.x();
  q(1) = d.y();
  return q;
}

std::vector<Eigen::VectorXd> bernsteinCoefficientRows(int degree) {
  const CglGrid g(degree);
  const Eigen::MatrixXd E = bernsteinBasisAtDegree(degree, g);
  const Eigen::MatrixXd Einv = E.inverse();
  std::vector<Eigen::VectorXd> W;
  for (int r = 0; r < Einv.rows(); ++r) W.push_back(Einv.row(r).transpose());
  return W;
}

// the rest-pose rows: a full configuration pin, or a box of half-width eps around it
std::shared_ptr<Constraint> makeRestPin(const RobotModel& m, const Eigen::VectorXd& target) {
  return std::make_shared<PinConstraint>(range(0, m.nq()), target, "pin_q_rest");
}

std::shared_ptr<Constraint> makeRestBox(const RobotModel& m, const Eigen::VectorXd& target, double eps) {
  const Eigen::VectorXd e = Eigen::VectorXd::Constant(m.nq(), eps);
  return std::make_shared<BoxConstraint>(Field::Q, range(0, m.nq()), target - e, target + e, "ineq_rest_box");
}

// base x (and y unless planar-locked) plus the actuated joints of `base`, at base xy `d`
std::shared_ptr<Constraint> makeJointConfigurationPin(const RobotModel& m, const Eigen::VectorXd& base, const Eigen::Vector2d& d,
                                                      bool yLocked) {
  std::vector<int> idx{0};
  if (!yLocked) idx.push_back(1);
  idx.insert(idx.end(), m.actuatedIdx().begin(), m.actuatedIdx().end());
  Eigen::VectorXd target(static_cast<int>(idx.size()));
  int o = 0;
  target(o++) = d.x();
  if (!yLocked) target(o++) = d.y();
  for (int j = 0; j < m.nu(); ++j) target(o + j) = base(m.actuatedIdx()[static_cast<size_t>(j)]);
  return std::make_shared<PinConstraint>(idx, target, "pin_q_joints");
}

// A flight row at global flight fraction `fraction`: the flight children are elements
// k0 .. k0+k1-1; a sample on a child boundary is owned by the left child's last node,
// anything else is an off-node point of the child it falls in.
void attachAtFlightFraction(TrajectoryProblem& P, int k0, int k1, double fraction, const std::shared_ptr<Constraint>& con) {
  constexpr double kBoundaryTol = 1e-12;
  const double childCoordinate = fraction * static_cast<double>(k1);
  const double boundary = std::round(childCoordinate);
  if (std::abs(childCoordinate - boundary) <= kBoundaryTol) {
    const int leftChild = std::clamp(static_cast<int>(boundary) - 1, 0, k1 - 1);
    P.elements[static_cast<size_t>(k0 + leftChild)].addNode(con, Where::LastNode);
  } else {
    const int child = std::clamp(static_cast<int>(std::floor(childCoordinate)), 0, k1 - 1);
    const double tau = 2.0 * (childCoordinate - child) - 1.0;
    P.elements[static_cast<size_t>(k0 + child)].addOffNode(con, {tau});
  }
}

bool splitElementDescending(const ElementSplit& a, const ElementSplit& b) { return a.element > b.element; }

}  // namespace

// --- validation -----------------------------------------------------------------------------

std::string validateFenceJumpSpec(const FenceJumpSpec& js) {
  if (js.degree < 2) return "degree must be >= 2";
  if (js.Nc < 2 || js.Nc > js.degree + 1) return "Nc must be in [2, degree + 1]";
  if (js.splitPerMode < 1 || js.splitPerMode > 16) return "split must be in [1, 16]";
  if (js.lsqNq < 0) return "lsqnq must be >= 0";
  if (!(js.Tstance > 0.0)) return "tstance must be > 0";
  if (!(js.Tair > 0.0)) return "tair must be > 0";
  if (js.hopHeight < 0.0) return "hop must be >= 0";
  if (!std::isfinite(js.lamMax) || js.lamMax < 0.0) return "lammax must be finite and >= 0";
  if (js.lamMax > 0.0 && js.lamMax <= js.lamMin) return "lammax must exceed lammin";
  if (!(js.copX > 0.0) || !(js.copY > 0.0)) return "copx/copy must be > 0";
  if (!(js.gammaZ > 0.0)) return "gammaz must be > 0";
  if (js.poses.empty()) return "poses is empty: the two rest poses define the jump";
  if (!std::isfinite(js.selfColMargin) || js.selfColMargin < 0.0) return "selfcolmargin must be finite and >= 0";
  if (js.selfColInterp < 0 || js.selfColInterp > 16) return "selfcolinterp must be in [0, 16]";
  if (js.dynamicsInterpNq < 0 || js.dynamicsInterpNq > 16) return "dynnq must be in [0, 16]";
  if (js.qBoxNq < 0 || js.qBoxNq > 16 || js.vBoxNq < 0 || js.vBoxNq > 16) return "qboxnq/vboxnq must be in [0, 16]";
  if (js.swingPathNq < 0 || js.swingPathNq > 16 || js.swingProgressNq < 0 || js.swingProgressNq > 16)
    return "swingpathnq/swingprogressnq must be in [0, 16]";
  if (!std::isfinite(js.liftoffXMax)) return "liftoffxmax must be finite";
  if (js.touchdownPosture && js.holdLandingJoints) return "landjointpin and holdlanding cannot both be enabled";
  if (!std::isfinite(js.touchdownPostureBlend) || js.touchdownPostureBlend < 0.0 || js.touchdownPostureBlend > 1.0)
    return "touchdown posture blend must be in [0, 1]";
  if (!std::isfinite(js.landVxyMax) || js.landVxyMax < 0.0) return "landvxy must be finite and >= 0";
  for (const FenceJumpSpec::SelfColPair& p : js.selfColPairs) {
    if (p.a.empty() || p.b.empty() || p.a == p.b) return "selfcolpairs entries want two distinct capsule names";
    if (!std::isfinite(p.margin) || (p.margin < 0.0 && p.margin != -1.0)) return "selfcolpairs per-pair margin must be >= 0 (or omitted)";
  }
  if (js.seedMode != 0 && js.seedMode != 1 && js.seedMode != 3)
    return "seedmode must be 0 (static hold), 1 (piecewise) or 3 (linear flight); the ballistic seed (2) is not provided";
  if (!js.splits.empty()) {
    if (js.splits.size() != 3) return "splits wants one entry per parent mode (3)";
    for (int k : js.splits)
      if (k < 1 || k > 16) return "splits entries must be in [1, 16]";
  }
  if (!js.degs.empty()) {
    if (js.degs.size() != 3) return "degs wants one entry per parent mode (3)";
    for (int dg : js.degs)
      if (dg < 2 || dg > 16) return "degs entries must be in [2, 16]";
    if (js.degs[0] != js.degs[2]) return "degs[0] and degs[2] must match (the stances share the force basis)";
    if (js.Nc > js.degs[0] + 1) return "Nc exceeds the stance degree + 1";
  }
  if (!js.lsqnqs.empty()) {
    if (js.lsqnqs.size() != 3) return "lsqnqs wants one entry per parent mode (3)";
    for (int n : js.lsqnqs)
      if (n < 0 || n > 32) return "lsqnqs entries must be in [0, 32]";
  }
  if (js.grade < 0.05 || js.grade > 5.0) return "hgrade must be in [0.05, 5.0]";
  if (js.uSamples < 0 || js.uSamples > 32) return "usamples must be in [0, 32]";
  if (js.uSamples > 0 && !js.deriveActuation) return "usamples requires implicitu";
  if (js.deriveActuation && js.uBernstein) return "implicitu eliminates the torque field; ubern needs it";
  if (js.deriveActuation && !js.planarLock.empty()) return "implicitu is not implemented with planarlock";
  const double vzNeed = vzMinOf(js.hopHeight);
  const double vzAvail = kGravity * js.Tair / 2.0;
  if (vzNeed > 1.05 * vzAvail)
    return "ballistically infeasible: hop height " + std::to_string(js.hopHeight) + " m demands liftoff vz >= " +
           std::to_string(vzNeed) + " m/s but a flight of tair=" + std::to_string(js.Tair) + " s supports " +
           std::to_string(vzAvail) + " m/s";
  if (js.frictionMu > 0.0 && js.step.norm() > 0.0) {
    const double vx = js.step.norm() / js.Tair;
    if (vx > 1.15 * js.frictionMu * vzAvail)
      return "step " + std::to_string(js.step.norm()) + " m in tair=" + std::to_string(js.Tair) + " s needs vx = " +
             std::to_string(vx) + " m/s, outside the friction cone at liftoff";
  }
  return "";
}

std::vector<std::string> fenceJumpSpecWarnings(const FenceJumpSpec& js) {
  std::vector<std::string> w;
  const double vzNeed = vzMinOf(js.hopHeight);
  const double vzAvail = kGravity * js.Tair / 2.0;
  if (vzNeed > 0.9 * vzAvail) w.push_back("ballistic margin < 10%");
  const double vx = js.step.norm() / js.Tair;
  if (js.frictionMu > 0.0 && vzAvail > 0.0 && vx / vzAvail > 0.8 * js.frictionMu) w.push_back("launch angle presses the friction cone");
  if (js.obstacles.empty()) w.push_back("no obstacles declared: this is a plain leap, not a fence jump");
  if (js.obstacles.interpPoints == 0 && !js.obstacles.empty()) w.push_back("obstacles.interp = 0: the clearance is enforced at the nodes only");
  if (js.velLimitScale <= 0.0) w.push_back("vlim <= 0: joint speed limits are not enforced");
  if (js.uLimitScale <= 0.0) w.push_back("ulim <= 0: torque limits are not enforced");
  return w;
}

// --- the builder ------------------------------------------------------------------------------

FenceJumpBuild buildFenceJump(const FenceJumpSpec& js, std::shared_ptr<const RobotModel> modelPtr) {
  {
    const std::string err = validateFenceJumpSpec(js);
    if (!err.empty()) throw std::invalid_argument("fence jump spec: " + err);
  }
  const RobotModel& m = *modelPtr;
  FenceJumpBuild out;
  out.problem.model = modelPtr;
  out.problem.preconditionForce = js.precond;

  // --- what the model must look like ---
  if (m.nContacts() < 1) throw std::runtime_error("fence jump: the robot yaml declares no contacts");
  const int dim = m.contactDim(0);
  if (dim != 3 && dim != 6) throw std::runtime_error("fence jump: contact dim must be 3 or 6");
  for (int c = 1; c < m.nContacts(); ++c)
    if (m.contactDim(c) != dim) throw std::runtime_error("fence jump: all contacts must share one dim");
  {
    const std::vector<std::string>& jn = m.jointNames();
    const char* want[6] = {"base_px", "base_py", "base_pz", "base_rx", "base_ry", "base_rz"};
    if (jn.size() < 6) throw std::runtime_error("fence jump: not a floating-base robot");
    for (int i = 0; i < 6; ++i)
      if (jn[static_cast<size_t>(i)] != want[i])
        throw std::runtime_error(std::string("fence jump: joint ") + std::to_string(i) + " is '" + jn[static_cast<size_t>(i)] + "', expected '" + want[i] + "'");
  }
  const int splitOf[3] = {js.splits.empty() ? js.splitPerMode : js.splits[0], js.splits.empty() ? js.splitPerMode : js.splits[1],
                          js.splits.empty() ? js.splitPerMode : js.splits[2]};
  const int degOf[3] = {js.degs.empty() ? js.degree : js.degs[0], js.degs.empty() ? js.degree : js.degs[1],
                        js.degs.empty() ? js.degree : js.degs[2]};
  const int lsqNqOf[3] = {js.lsqnqs.empty() ? js.lsqNq : js.lsqnqs[0], js.lsqnqs.empty() ? js.lsqNq : js.lsqnqs[1],
                          js.lsqnqs.empty() ? js.lsqNq : js.lsqnqs[2]};
  const double Tland = js.TstanceLand > 0.0 ? js.TstanceLand : js.Tstance;

  const ContactSet all = m.allContacts();
  const ContactSet none;
  const NodeDims dimsC = m.nodeDims(all);
  const int impactAll = m.ensureImpactSet(all);
  const int impactConeAll = js.impactCone ? m.ensureImpactConeSet(all) : -1;
  ModelDataPtr data = m.makeData();
  ModelEvalCache cache;
  m.initCache(cache);
  out.qStand = poseFromAngles(m, *data, cache, js.poses, false, "stand pose");
  out.qSquat = poseFromAngles(m, *data, cache, js.poses, true, "squat pose");

  std::vector<int> planarRows;
  if (!js.planarLock.empty()) {
    std::vector<char> locked(static_cast<size_t>(m.nv()), 0);
    for (const std::string& name : js.planarLock) locked[static_cast<size_t>(jointIndex(m, name, "planarLock"))] = 1;
    for (int row = 0; row < m.nv(); ++row)
      if (!locked[static_cast<size_t>(row)]) planarRows.push_back(row);
  }
  // dynamics: the actuated rows define u under torque elimination; planar robots drop
  // the rows a planar guide holds
  std::shared_ptr<Constraint> dynStance, dynAir;
  if (js.deriveActuation) {
    dynStance = std::make_shared<FloatingBaseDynamicsConstraint>(m, all);
    dynAir = std::make_shared<FloatingBaseDynamicsConstraint>(m, none);
  } else if (!planarRows.empty()) {
    dynStance = std::make_shared<SelectedRowsDynamicsConstraint>(m, all, planarRows);
    dynAir = std::make_shared<SelectedRowsDynamicsConstraint>(m, none, planarRows);
  } else {
    dynStance = std::make_shared<DynamicsConstraint>(m, all);
    dynAir = std::make_shared<DynamicsConstraint>(m, none);
  }

  // --- constraint objects ---
  std::vector<std::shared_ptr<Constraint>> feetVel, feetAcc;
  std::vector<int> lamZIdx, lamZComp;
  for (const ContactRef& cr : all.active) {
    const std::string& nm = m.contactName(cr.index);
    feetAcc.push_back(std::make_shared<FrameAccelerationConstraint>(cr.index, Eigen::VectorXd::Zero(cr.dim), "acc_" + nm));
    feetVel.push_back(std::make_shared<FrameVelocityConstraint>(cr.index, Eigen::VectorXd::Zero(cr.dim), "vel_" + nm));
    lamZIdx.push_back(dimsC.offLam() + all.lamOffset(cr.index) + 2);
    lamZComp.push_back(all.lamOffset(cr.index) + 2);
  }
  auto ineqLamZ = std::make_shared<LowerBoundConstraint>(lamZIdx, Eigen::VectorXd::Constant(static_cast<int>(lamZIdx.size()), js.lamMin), "ineq_lam_z");
  ineqLamZ->setForceOnly(true);
  std::shared_ptr<BoxConstraint> ineqLamZMax;
  if (js.lamMax > 0.0) {
    const int n = static_cast<int>(lamZComp.size());
    ineqLamZMax = std::make_shared<BoxConstraint>(Field::Lam, lamZComp, Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity()),
                                                  Eigen::VectorXd::Constant(n, js.lamMax), "ineq_lam_z_max");
  }
  std::shared_ptr<FrictionConeConstraint> fric;
  if (js.frictionMu > 0.0) fric = std::make_shared<FrictionConeConstraint>(all, js.frictionMu, js.frictionEps, "ineq_fric");
  std::shared_ptr<PinConstraint> pinLamLiftoff;
  if (js.liftoffFullWrench || !fric)
    pinLamLiftoff = std::make_shared<PinConstraint>(range(dimsC.offLam(), all.nLam()), Eigen::VectorXd::Zero(all.nLam()), "pin_lam_liftoff");
  else
    pinLamLiftoff = std::make_shared<PinConstraint>(lamZIdx, Eigen::VectorXd::Zero(static_cast<int>(lamZIdx.size())), "pin_lam_liftoff");
  pinLamLiftoff->setForceOnly(true);
  std::shared_ptr<PinConstraint> pinLamLand;
  if (js.landRamp) {
    pinLamLand = std::make_shared<PinConstraint>(range(dimsC.offLam(), all.nLam()), Eigen::VectorXd::Zero(all.nLam()), "pin_lam_land");
    pinLamLand->setForceOnly(true);
  }
  std::shared_ptr<ContactWrenchBoxConstraint> cop, tz;
  if (dim == 6) {
    cop = std::make_shared<ContactWrenchBoxConstraint>(all, std::vector<int>{4, 3}, std::vector<double>{js.copX, js.copY}, "ineq_cop");
    tz = std::make_shared<ContactWrenchBoxConstraint>(all, std::vector<int>{5}, std::vector<double>{js.gammaZ}, "ineq_tz");
  }
  std::vector<std::shared_ptr<Constraint>> contactGround;
  for (int c = 0; c < m.nContacts(); ++c) {
    contactGround.push_back(std::make_shared<FramePositionComponentConstraint>(c, 2, 0.0, "pos_ground_" + m.contactName(c)));
    if (m.contactDim(c) == 6) contactGround.push_back(std::make_shared<FrameOrientationConstraint>(c, Eigen::Vector3d::Zero(), "pos_flat_" + m.contactName(c)));
  }
  auto clearGround = std::make_shared<MonitorHeightLowerBoundConstraint>(m, Eigen::VectorXd::Zero(m.nMonitors()), "ineq_clear");
  Eigen::VectorXd stanceGroundLb = Eigen::VectorXd::Zero(m.nMonitors());
  for (int i = 0; i < m.nMonitors(); ++i)
    for (int c = 0; c < m.nContacts(); ++c)
      if (m.monitorName(i) == m.contactName(c)) stanceGroundLb(i) = -1.0;
  auto clearGroundStance = std::make_shared<MonitorHeightLowerBoundConstraint>(m, stanceGroundLb, "ineq_clear");
  auto clearAir = std::make_shared<MonitorHeightLowerBoundConstraint>(m, Eigen::VectorXd::Constant(m.nMonitors(), js.footClear), "ineq_clear_air");
  auto ineqBaseZ = std::make_shared<LowerBoundConstraint>(std::vector<int>{2}, Eigen::VectorXd::Zero(1), "ineq_base_z");
  auto effort = std::make_shared<ControlEffortCost>();
  std::shared_ptr<Cost> forceCost, velocityCost, accelerationCost, baseAttitudeCost, baseAngularVelocityCost;
  const double effortRef = m.actuatedEffortLimit().cwiseAbs().mean();
  const double velocityRef = std::max(m.actuatedVelocityLimit().cwiseAbs().mean(), 1e-6);
  if (js.forceCostWeight > 0.0) forceCost = std::make_shared<ContactForceCost>(js.forceCostWeight);
  if (js.jointVelocityWeight > 0.0) {
    const double w = js.jointVelocityWeight * (effortRef / velocityRef) * (effortRef / velocityRef);
    velocityCost = std::make_shared<JointVelocityCost>(m.actuatedIdx(), Eigen::VectorXd::Constant(m.nu(), w));
  }
  if (js.jointAccelerationWeight > 0.0) {
    const double accelerationRef = velocityRef / 0.25;
    const double w = js.jointAccelerationWeight * (effortRef / accelerationRef) * (effortRef / accelerationRef);
    accelerationCost = std::make_shared<JointAccelerationCost>(m.actuatedIdx(), Eigen::VectorXd::Constant(m.nu(), w));
  }
  std::vector<int> attitudeAxes;
  for (int i = 3; i < 6; ++i)
    if (!contains(js.planarLock, m.jointNames()[static_cast<size_t>(i)])) attitudeAxes.push_back(i);
  if (!attitudeAxes.empty() && js.baseAttitudeWeight > 0.0) {
    const int n = static_cast<int>(attitudeAxes.size());
    const double w = js.baseAttitudeWeight * (effortRef / 0.5) * (effortRef / 0.5);
    baseAttitudeCost = std::make_shared<JointPostureCost>(attitudeAxes, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Constant(n, w));
  }
  if (!attitudeAxes.empty() && js.baseAngularVelocityWeight > 0.0) {
    const int n = static_cast<int>(attitudeAxes.size());
    const double w = js.baseAngularVelocityWeight * (effortRef / 5.0) * (effortRef / 5.0);
    baseAngularVelocityCost = std::make_shared<JointVelocityCost>(attitudeAxes, Eigen::VectorXd::Constant(n, w));
  }
  auto vzUp = std::make_shared<LowerBoundConstraint>(std::vector<int>{dimsC.offV() + 2}, Eigen::VectorXd::Constant(1, vzMinOf(js.hopHeight)), "ineq_vz_up");
  std::vector<int> cpAll;
  for (int j = 0; j < js.Nc; ++j) cpAll.push_back(j);
  const std::vector<int> cpButLast(cpAll.begin(), cpAll.end() - 1);
  const std::vector<int> cpButFirst(cpAll.begin() + 1, cpAll.end());

  const Eigen::Vector2d X0 = Eigen::Vector2d::Zero();
  const Eigen::Vector2d X1 = js.step;
  const Eigen::VectorXd& landingPose = out.qStand;
  const Eigen::VectorXd qHold0 = qAt(out.qSquat, X0);
  const Eigen::VectorXd qHold2 = qAt(landingPose, X1);
  const bool basePyLocked = contains(js.planarLock, "base_py");
  auto pinRestV = std::make_shared<PinConstraint>(range(dimsC.offV(), m.nv()), Eigen::VectorXd::Zero(m.nv()), "pin_v_rest");
  auto pinRestA = std::make_shared<PinConstraint>(range(dimsC.offA(), dimsC.na), Eigen::VectorXd::Zero(dimsC.na), "pin_a_rest");

  // contactPosAll: full foot-position pins at every stance node
  std::vector<std::shared_ptr<Constraint>> stanceFeet0, stanceFeet2;
  if (js.contactPosAll) {
    for (int which = 0; which < 2; ++which) {
      const Eigen::VectorXd qRef = which == 0 ? qHold0 : qHold2;
      std::vector<std::shared_ptr<Constraint>>& dst = which == 0 ? stanceFeet0 : stanceFeet2;
      m.evalNode(*data, qRef, Eigen::VectorXd::Zero(m.nv()), Eigen::VectorXd::Zero(m.nv()), kFootPos, cache);
      for (int c = 0; c < m.nContacts(); ++c) {
        dst.push_back(std::make_shared<FramePositionConstraint>(c, cache.footPos[static_cast<size_t>(c)].head<3>(), "pos_stance_" + m.contactName(c)));
        if (m.contactDim(c) == 6) dst.push_back(std::make_shared<FrameOrientationConstraint>(c, Eigen::Vector3d::Zero(), "pos_flat_" + m.contactName(c)));
      }
    }
  }

  // --- the three parent modes -------------------------------------------------------------
  Element modes[3];
  Element& m0 = modes[0];
  Element& m1 = modes[1];
  Element& m2 = modes[2];
  // mode 0: push-off
  m0.label = "push_off";
  m0.phase = 0;
  m0.degree = degOf[0];
  m0.T = js.Tstance;
  m0.contacts = all;
  m0.forceControlPoints = js.Nc;
  m0.deriveTorque = js.deriveActuation;
  if (js.groundClearance) m0.addNode(clearGroundStance, Where::AllNodes);
  m0.addNode(dynStance, Where::FirstNode);
  if (js.strictRest) {
    if (js.restBoxEps > 0.0) m0.addNode(makeRestBox(m, qHold0, js.restBoxEps), Where::FirstNode);
    else m0.addNode(makeRestPin(m, qHold0), Where::FirstNode);
    m0.addNode(pinRestV, Where::FirstNode);
    if (js.endpointAccel) m0.addNode(pinRestA, Where::FirstNode);
    if (!js.rowOwnership) {
      for (const std::shared_ptr<Constraint>& c : contactGround) m0.addNode(c, Where::FirstNode);
      for (const std::shared_ptr<Constraint>& c : feetVel) m0.addNode(c, Where::FirstNode);
    } else if (js.restBoxEps > 0.0) {
      for (const std::shared_ptr<Constraint>& c : contactGround) m0.addNode(c, Where::FirstNode);
    }
  } else {
    m0.addNode(makeJointConfigurationPin(m, out.qSquat, X0, basePyLocked), Where::FirstNode);
    for (const std::shared_ptr<Constraint>& c : contactGround) m0.addNode(c, Where::FirstNode);
    for (const std::shared_ptr<Constraint>& c : feetVel) m0.addNode(c, Where::FirstNode);
    if (js.baseHeight) m0.addNode(ineqBaseZ, Where::FirstNode);
  }
  m0.addNode(dynStance, Where::Interior);
  for (const std::shared_ptr<Constraint>& c : feetVel) m0.addNode(c, Where::Interior);
  if (js.contactPosAll) for (const std::shared_ptr<Constraint>& c : stanceFeet0) m0.addNode(c, Where::Interior);
  if (js.stanceAcc) for (const std::shared_ptr<Constraint>& c : feetAcc) m0.addNode(c, Where::Interior);
  if (js.baseHeight) m0.addNode(ineqBaseZ, Where::Interior);
  m0.addNode(dynStance, Where::LastNode);
  for (const std::shared_ptr<Constraint>& c : feetVel) m0.addNode(c, Where::LastNode);
  if (js.contactPosAll) for (const std::shared_ptr<Constraint>& c : stanceFeet0) m0.addNode(c, Where::LastNode);
  m0.addNode(vzUp, Where::LastNode);
  if (js.liftoffXMax >= 0.0)
    m0.addNode(std::make_shared<BoxConstraint>(Field::Q, std::vector<int>{0}, Eigen::VectorXd::Constant(1, -std::numeric_limits<double>::infinity()),
                                               Eigen::VectorXd::Constant(1, out.qSquat(0) + js.liftoffXMax), "ineq_liftoff_x"),
               Where::LastNode);
  if (!attitudeAxes.empty()) {
    const int n = static_cast<int>(attitudeAxes.size());
    if (js.liftoffAttMax > 0.0)
      m0.addNode(std::make_shared<BoxConstraint>(Field::Q, attitudeAxes, Eigen::VectorXd::Constant(n, -js.liftoffAttMax), Eigen::VectorXd::Constant(n, js.liftoffAttMax), "box_liftoff_att"), Where::LastNode);
    if (js.liftoffOmegaMax > 0.0)
      m0.addNode(std::make_shared<BoxConstraint>(Field::V, attitudeAxes, Eigen::VectorXd::Constant(n, -js.liftoffOmegaMax), Eigen::VectorXd::Constant(n, js.liftoffOmegaMax), "box_liftoff_omega"), Where::LastNode);
  }
  if (js.minimizeEffort) m0.addCost(effort);
  if (forceCost) m0.addCost(forceCost);
  if (velocityCost) m0.addCost(velocityCost);
  if (accelerationCost) m0.addCost(accelerationCost);
  if (js.stanceNoSlipNq > 0) {
    const std::vector<double> taus = uniformInteriorTaus(js.stanceNoSlipNq);
    for (const std::shared_ptr<Constraint>& c : feetVel) m0.addOffNode(c, taus);
    if (js.stanceAcc) for (const std::shared_ptr<Constraint>& c : feetAcc) m0.addOffNode(c, taus);
  }
  // mode 1: flight
  m1.label = "flight";
  m1.phase = 1;
  m1.degree = degOf[1];
  m1.T = js.Tair;
  m1.contacts = none;
  m1.deriveTorque = js.deriveActuation;
  if (js.groundClearance) m1.addNode(clearGround, Where::AllNodes);
  m1.addNode(dynAir, Where::FirstNode);
  if (js.baseHeight) m1.addNode(ineqBaseZ, Where::FirstNode);
  m1.addNode(dynAir, Where::Interior);
  if (js.baseHeight) m1.addNode(ineqBaseZ, Where::Interior);
  m1.addNode(dynAir, Where::LastNode);
  if (js.landingFootPin) {
    m.evalNode(*data, qHold2, Eigen::VectorXd::Zero(m.nv()), Eigen::VectorXd::Zero(m.nv()), kFootPos, cache);
    for (int c = 0; c < m.nContacts(); ++c) {
      m1.addNode(std::make_shared<FramePositionConstraint>(c, cache.footPos[static_cast<size_t>(c)].head<3>(), "pos_land_" + m.contactName(c)), Where::LastNode);
      if (m.contactDim(c) == 6) m1.addNode(std::make_shared<FrameOrientationConstraint>(c, Eigen::Vector3d::Zero(), "pos_flat_" + m.contactName(c)), Where::LastNode);
    }
  } else {
    for (const std::shared_ptr<Constraint>& c : contactGround) m1.addNode(c, Where::LastNode);
  }
  if (js.touchdownPosture) {
    const std::vector<int>& actuated = m.actuatedIdx();
    Eigen::VectorXd target(static_cast<int>(actuated.size()));
    for (int j = 0; j < target.size(); ++j) target(j) = landingPose(actuated[static_cast<size_t>(j)]);
    if (js.touchdownPostureBlend >= 1.0 - 1e-12) {
      m1.addNode(std::make_shared<PinConstraint>(actuated, target, "pin_touchdown_joints"), Where::LastNode);
    } else {
      const Eigen::VectorXd half = Eigen::VectorXd::Constant(target.size(), 1.0 - js.touchdownPostureBlend);
      m1.addNode(std::make_shared<BoxConstraint>(Field::Q, actuated, target - half, target + half, "box_touchdown_joints"), Where::LastNode);
    }
  }
  if (js.zeroLandXY) {
    for (int c = 0; c < m.nContacts(); ++c) {
      auto velocity = std::make_shared<FrameVelocityConstraint>(c, Eigen::VectorXd::Zero(m.contactDim(c)), "pin_land_vxy_" + m.contactName(c));
      m1.addNode(std::make_shared<RowSliceConstraint>(velocity, 0), Where::LastNode);
      m1.addNode(std::make_shared<RowSliceConstraint>(velocity, 1), Where::LastNode);
    }
  } else if (js.landVxyMax > 0.0) {
    for (int c = 0; c < m.nContacts(); ++c) {
      for (int comp = 0; comp < 2; ++comp) {
        const std::string axis = comp == 0 ? "x" : "y";
        m1.addNode(std::make_shared<FrameVelocityBoundConstraint>(c, comp, 1.0, js.landVxyMax, "ineq_land_v" + axis + "_upper_" + m.contactName(c), dim), Where::LastNode);
        m1.addNode(std::make_shared<FrameVelocityBoundConstraint>(c, comp, -1.0, -js.landVxyMax, "ineq_land_v" + axis + "_lower_" + m.contactName(c), dim), Where::LastNode);
      }
    }
  }
  if (js.landVzMax > 0.0)
    for (int c = 0; c < m.nContacts(); ++c)
      m1.addNode(std::make_shared<FrameVelocityBoundConstraint>(c, 2, -1.0, -js.landVzMax, "ineq_land_vz_" + m.contactName(c), dim), Where::LastNode);
  if (js.impactCone) {
    std::vector<int> coneOff;
    for (const ContactRef& cr : all.active) coneOff.push_back(m.impactConeLamOffset(impactConeAll, cr.index));
    m1.addNode(std::make_shared<ImpactConeConstraint>(impactConeAll, coneOff, js.frictionMu, js.frictionEps, "ineq_impact_cone"), Where::LastNode);
  }
  if (js.minimizeEffort) m1.addCost(effort);
  if (velocityCost) m1.addCost(velocityCost);
  if (accelerationCost) m1.addCost(accelerationCost);
  // mode 2: landing
  m2.label = "landing";
  m2.phase = 2;
  m2.degree = degOf[2];
  m2.T = Tland;
  m2.contacts = all;
  m2.forceControlPoints = js.Nc;
  m2.deriveTorque = js.deriveActuation;
  if (js.groundClearance) m2.addNode(clearGroundStance, Where::AllNodes);
  m2.addNode(dynStance, Where::FirstNode);
  if (!js.rowOwnership) {
    for (const std::shared_ptr<Constraint>& c : contactGround) m2.addNode(c, Where::FirstNode);
    for (const std::shared_ptr<Constraint>& c : feetVel) m2.addNode(c, Where::FirstNode);
  }
  if (js.stanceAcc) for (const std::shared_ptr<Constraint>& c : feetAcc) m2.addNode(c, Where::FirstNode);
  if (js.baseHeight) m2.addNode(ineqBaseZ, Where::FirstNode);
  m2.addNode(dynStance, Where::Interior);
  for (const std::shared_ptr<Constraint>& c : feetVel) m2.addNode(c, Where::Interior);
  if (js.contactPosAll) for (const std::shared_ptr<Constraint>& c : stanceFeet2) m2.addNode(c, Where::Interior);
  if (js.stanceAcc) for (const std::shared_ptr<Constraint>& c : feetAcc) m2.addNode(c, Where::Interior);
  if (js.baseHeight) m2.addNode(ineqBaseZ, Where::Interior);
  m2.addNode(dynStance, Where::LastNode);
  if (js.strictRest) {
    if (js.restBoxEps > 0.0) m2.addNode(makeRestBox(m, qHold2, js.restBoxEps), Where::LastNode);
    else m2.addNode(makeRestPin(m, qHold2), Where::LastNode);
    m2.addNode(pinRestV, Where::LastNode);
    if (js.endpointAccel) m2.addNode(pinRestA, Where::LastNode);
    if (!js.rowOwnership) for (const std::shared_ptr<Constraint>& c : feetVel) m2.addNode(c, Where::LastNode);
    else if (js.restBoxEps > 0.0) for (const std::shared_ptr<Constraint>& c : contactGround) m2.addNode(c, Where::LastNode);
  } else {
    for (const std::shared_ptr<Constraint>& c : feetVel) m2.addNode(c, Where::LastNode);
    m2.addNode(makeJointConfigurationPin(m, landingPose, X1, basePyLocked), Where::LastNode);
  }
  if (js.holdLandingJoints) {
    Eigen::VectorXd target(m.nu());
    for (int j = 0; j < m.nu(); ++j) target(j) = landingPose(m.actuatedIdx()[static_cast<size_t>(j)]);
    auto hold = std::make_shared<PinConstraint>(m.actuatedIdx(), target, "pin_landing_joints");
    m2.addNode(hold, Where::FirstNode);
    m2.addNode(hold, Where::Interior);
  }
  if (js.landingXSlack >= 0.0)
    m2.addNode(std::make_shared<LowerBoundConstraint>(std::vector<int>{0}, Eigen::VectorXd::Constant(1, js.step.x() - js.landingXSlack), "ineq_landing_side"), Where::AllNodes);
  if (js.minimizeEffort) m2.addCost(effort);
  if (forceCost) m2.addCost(forceCost);
  if (velocityCost) m2.addCost(velocityCost);
  if (accelerationCost) m2.addCost(accelerationCost);
  if (js.stanceNoSlipNq > 0) {
    const std::vector<double> taus = uniformInteriorTaus(js.stanceNoSlipNq);
    for (const std::shared_ptr<Constraint>& c : feetVel) m2.addOffNode(c, taus);
    if (js.stanceAcc) for (const std::shared_ptr<Constraint>& c : feetAcc) m2.addOffNode(c, taus);
  }
  // the least-squares dynamics defect and the hard off-node dynamics rows, per mode
  for (int md = 0; md < 3; ++md) {
    const int bit = md == 1 ? 4 : 1;
    const std::shared_ptr<Constraint>& dyn = md == 1 ? dynAir : dynStance;
    if ((js.lsqModeMask & bit) != 0 && lsqNqOf[md] > 0 && js.lsqRho > 0.0) {
      if (js.lsqGauss) {
        std::vector<double> taus, weights;
        gaussLegendreInteriorTaus(lsqNqOf[md], taus, weights);
        modes[md].setDefectCost(dyn, taus, js.lsqRho, weights);
      } else {
        modes[md].setDefectCost(dyn, uniformInteriorTaus(lsqNqOf[md]), js.lsqRho);
      }
    }
    if ((js.lsqModeMask & bit) != 0 && js.dynamicsInterpNq > 0) modes[md].addOffNode(dyn, uniformInteriorTaus(js.dynamicsInterpNq));
  }
  if (js.actuatedSymmetry) {
    const std::vector<std::string>& names = m.jointNames();
    std::vector<int> left, right;
    for (int index : m.actuatedIdx()) {
      const std::string& name = names.at(static_cast<size_t>(index));
      if (name.rfind("left_", 0) != 0) continue;
      const std::string mate = "right_" + name.substr(5);
      if (!m.hasJoint(mate)) throw std::invalid_argument("fence jump actuated symmetry: no mate '" + mate + "'");
      left.push_back(index);
      right.push_back(m.jointIndex(mate));
    }
    if (left.empty() || 2 * left.size() != m.actuatedIdx().size()) throw std::invalid_argument("fence jump actuated symmetry requires paired left_*/right_* actuated joints");
    auto symmetry = std::make_shared<PairEqualityConstraint>(left, right, "pin_actuated_symmetry");
    m0.addNode(symmetry, Where::Interior);
    m0.addNode(symmetry, Where::LastNode);
    m1.addNode(symmetry, Where::Interior);
    m1.addNode(symmetry, Where::LastNode);
    m2.addNode(symmetry, Where::Interior);
  }
  // hardware boxes from the URDF, on every mode
  {
    const std::vector<int>& act = m.actuatedIdx();
    const int nu = m.nu();
    if (js.qBoxScale > 0.0) {
      const Eigen::VectorXd lo = m.lowerPositionLimit(), hi = m.upperPositionLimit();
      Eigen::VectorXd qlb(nu), qub(nu);
      for (int j = 0; j < nu; ++j) {
        const int i = act[static_cast<size_t>(j)];
        const double mid = 0.5 * (lo(i) + hi(i)), half = 0.5 * (hi(i) - lo(i));
        qlb(j) = mid - js.qBoxScale * half;
        qub(j) = mid + js.qBoxScale * half;
      }
      auto boxQ = std::make_shared<BoxConstraint>(Field::Q, act, qlb, qub, "box_q");
      for (Element& e : modes) {
        e.addNode(boxQ, Where::AllNodes);
        if (js.qBoxNq > 0) e.addOffNode(boxQ, uniformInteriorTaus(js.qBoxNq));
      }
    }
    if (js.velLimitScale > 0.0) {
      const Eigen::VectorXd vlim = js.velLimitScale * m.actuatedVelocityLimit();
      auto boxV = std::make_shared<BoxConstraint>(Field::V, act, -vlim, vlim, "box_v");
      for (Element& e : modes) {
        e.addNode(boxV, Where::AllNodes);
        if (js.vBoxNq > 0) e.addOffNode(boxV, uniformInteriorTaus(js.vBoxNq));
      }
    }
    if (js.uLimitScale > 0.0) {
      const Eigen::VectorXd ulim = js.uLimitScale * m.actuatedEffortLimit();
      if (js.uBernstein) {
        // Bernstein-certified torque box: the coefficients c = E^-1 u_nodes of each
        // element's torque polynomial are boxed (0.5 % margin for the enforcement leak)
        const double certMargin = 0.995;
        auto boxUCert = std::make_shared<BoxConstraint>(Field::U, range(0, nu), -certMargin * ulim, certMargin * ulim, "box_u");
        for (int md = 0; md < 3; ++md) modes[md].addFieldFunctionals(boxUCert, Field::U, bernsteinCoefficientRows(degOf[md]));
      } else {
        auto boxU = std::make_shared<BoxConstraint>(Field::U, range(0, nu), -ulim, ulim, "box_u");
        for (Element& e : modes) {
          e.addNode(boxU, Where::AllNodes);
          if (js.deriveActuation && js.uSamples > 0) e.addOffNode(boxU, uniformInteriorTaus(js.uSamples));
        }
      }
    }
    if (js.posePathSlack >= 0.0) {
      Eigen::VectorXd lower(nu), upper(nu);
      for (int j = 0; j < nu; ++j) {
        const int i = act[static_cast<size_t>(j)];
        lower(j) = std::min(out.qStand(i), out.qSquat(i)) - js.posePathSlack;
        upper(j) = std::max(out.qStand(i), out.qSquat(i)) + js.posePathSlack;
      }
      auto posePath = std::make_shared<BoxConstraint>(Field::Q, act, lower, upper, "box_pose_path");
      for (int md = 0; md < 3; ++md) modes[md].addFieldFunctionals(posePath, Field::Q, bernsteinCoefficientRows(degOf[md]));
    }
    if (js.attMax > 0.0 && !attitudeAxes.empty()) {
      const int n = static_cast<int>(attitudeAxes.size());
      auto boxAtt = std::make_shared<BoxConstraint>(Field::Q, attitudeAxes, Eigen::VectorXd::Constant(n, -js.attMax), Eigen::VectorXd::Constant(n, js.attMax), "box_att");
      for (Element& e : modes) e.addNode(boxAtt, Where::AllNodes);
    }
    if (js.attVelMax > 0.0 && !attitudeAxes.empty()) {
      const int n = static_cast<int>(attitudeAxes.size());
      auto boxAttV = std::make_shared<BoxConstraint>(Field::V, attitudeAxes, Eigen::VectorXd::Constant(n, -js.attVelMax), Eigen::VectorXd::Constant(n, js.attVelMax), "box_att_v");
      for (Element& e : modes) {
        e.addNode(boxAttV, Where::AllNodes);
        if (js.vBoxNq > 0) e.addOffNode(boxAttV, uniformInteriorTaus(js.vBoxNq));
      }
    }
    const MotorSaturationConfig& sat = m.description().saturation;
    if (sat.enabled && sat.enforceInOptimization) {
      auto env = std::make_shared<MotorEnvelopeConstraint>(m, sat, "ineq_motor_envelope");
      for (Element& e : modes) {
        e.addNode(env, Where::AllNodes);
        e.addOffNode(env, uniformInteriorTaus(4));
      }
    }
  }
  if (!js.planarLock.empty()) {
    std::vector<int> idx;
    for (const std::string& n : js.planarLock) idx.push_back(jointIndex(m, n, "planarLock"));
    auto pinPlanar = std::make_shared<PinConstraint>(idx, Eigen::VectorXd::Zero(static_cast<int>(idx.size())), "pin_planar");
    if (js.strictRest) {
      m0.addNode(pinPlanar, Where::Interior);
      m0.addNode(pinPlanar, Where::LastNode);
      m1.addNode(pinPlanar, Where::AllNodes);
      m2.addNode(pinPlanar, Where::FirstNode);
      m2.addNode(pinPlanar, Where::Interior);
    } else {
      for (Element& e : modes) e.addNode(pinPlanar, Where::AllNodes);
    }
  }
  // obstacles and self-collision on every mode (before the split: each child then
  // checks its own interior points)
  {
    TrajectoryProblem tmp;
    tmp.model = modelPtr;
    for (Element& e : modes) tmp.elements.push_back(e);
    attachObstacles(m, js.obstacles, tmp);
    for (int md = 0; md < 3; ++md) modes[md] = tmp.elements[static_cast<size_t>(md)];
  }
  if (!js.selfColPairs.empty()) {
    std::vector<std::pair<int, int>> pairs;
    std::vector<double> margins;
    for (const FenceJumpSpec::SelfColPair& p : js.selfColPairs) {
      pairs.push_back({m.capsuleIndex(p.a), m.capsuleIndex(p.b)});
      margins.push_back(p.margin >= 0.0 ? p.margin : js.selfColMargin);
    }
    auto selfCol = std::make_shared<CapsuleSelfCollisionConstraint>(m, pairs, js.selfColMargin, "ineq_selfcol", margins);
    for (Element& e : modes) {
      e.addNode(selfCol, Where::AllNodes);
      if (js.selfColInterp > 0) e.addOffNode(selfCol, uniformInteriorTaus(js.selfColInterp));
    }
  }

  // --- split every mode into its elements (stances graded toward the contact change) ---
  TrajectoryProblem& P = out.problem;
  for (int md = 0; md < 3; ++md) {
    const Element& mm = modes[md];
    const int k = splitOf[md];
    const bool graded = js.grade != 1.0 && md != 1 && k > 1;
    const std::vector<double> widths = gradedWidths(k, js.grade, js.gradeEnds, md == 2);
    for (int i = 0; i < k; ++i) {
      Element sub;
      sub.label = mm.label + (k > 1 ? "_" + std::to_string(i) : "");
      sub.phase = md;
      sub.degree = mm.degree;
      sub.T = graded ? mm.T * widths[static_cast<size_t>(i)] : mm.T / k;
      sub.contacts = mm.contacts;
      sub.forceControlPoints = mm.forceControlPoints;
      sub.deriveTorque = mm.deriveTorque;
      sub.costs = mm.costs;
      sub.pointTerms = mm.pointTerms;
      sub.defect = mm.defect;
      for (const NodeAttachment& a : mm.nodeTerms)
        for (Where w : mapWhereUniqueOwner(a.where, i, k, !a.con->readsOneSidedFields())) sub.addNode(a.con, w);
      P.elements.push_back(sub);
      if (i < k - 1) P.addReset(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
    }
    if (md == 0) P.addReset(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
    if (md == 1) P.addReset(std::make_shared<PlasticImpactReset>(impactAll, dimsC.nq, dimsC.nv));
  }
  const int k0 = splitOf[0], k1 = splitOf[1], k2 = splitOf[2];
  // force feasibility on the control points: liftoff structure on the LAST push-off
  // element only, the landing ramp (opt-in) on the FIRST landing element only, the
  // plain stance set everywhere else
  for (int e = 0; e < static_cast<int>(P.elements.size()); ++e) {
    Element& el = P.elements[static_cast<size_t>(e)];
    if (el.contacts.active.empty()) continue;
    const bool liftsOff = (e == k0 - 1);
    const bool cleanLift = liftsOff && js.pinLiftoffForce && js.Nc >= 2;
    const bool rampLand = pinLamLand && e == k0 + k1 && js.Nc >= 2;
    const std::vector<int>& coneCps = (cleanLift && js.liftoffFullWrench) ? cpButLast : cpAll;
    if (fric) el.addForceControlPoints(fric, coneCps);
    if (cop) el.addForceControlPoints(cop, coneCps);
    if (tz) el.addForceControlPoints(tz, coneCps);
    if (ineqLamZMax) el.addForceControlPoints(ineqLamZMax, cpAll);
    if (rampLand) {
      // the terrain's ramp: zero wrench at the first control point, the floor from the
      // second on (the cone rows stay on every point; the zero wrench satisfies them)
      el.addForceControlPoints(ineqLamZ, cpButFirst);
      el.addForceControlPoints(pinLamLand, {0});
      continue;
    }
    if (!cleanLift) {
      el.addForceControlPoints(ineqLamZ, cpAll);
      continue;
    }
    // tapered floor: fz >= lamMin (1 - j / (Nc - 1)) on control point j, zero at the last
    for (int j = 0; j + 1 < js.Nc; ++j) {
      const double frac = 1.0 - static_cast<double>(j) / static_cast<double>(js.Nc - 1);
      auto floor = std::make_shared<LowerBoundConstraint>(lamZIdx, Eigen::VectorXd::Constant(static_cast<int>(lamZIdx.size()), js.lamMin * frac), "ineq_lam_z");
      floor->setForceOnly(true);
      el.addForceControlPoints(floor, {j});
    }
    el.addForceControlPoints(pinLamLiftoff, {js.Nc - 1});
  }
  // mesh-invariant flight rows at global flight fractions (swing path / progress /
  // clearance): a sample on a child boundary is owned by the left child's last node
  if (js.swingPathNq > 0 || js.swingProgressNq > 0) {
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(m.nv());
    std::vector<Eigen::Vector2d> from(static_cast<size_t>(m.nContacts())), to(static_cast<size_t>(m.nContacts()));
    m.evalNode(*data, qHold0, zero, zero, kFootPos, cache);
    for (int c = 0; c < m.nContacts(); ++c) from[static_cast<size_t>(c)] = cache.footPos[static_cast<size_t>(c)].head<2>();
    m.evalNode(*data, qHold2, zero, zero, kFootPos, cache);
    for (int c = 0; c < m.nContacts(); ++c) to[static_cast<size_t>(c)] = cache.footPos[static_cast<size_t>(c)].head<2>();
    for (int i = 0; i < js.swingPathNq; ++i) {
      const double fraction = static_cast<double>(i + 1) / static_cast<double>(js.swingPathNq + 1);
      for (int c = 0; c < m.nContacts(); ++c) {
        const Eigen::Vector2d target = (1.0 - fraction) * from[static_cast<size_t>(c)] + fraction * to[static_cast<size_t>(c)];
        attachAtFlightFraction(P, k0, k1, fraction, std::make_shared<FramePositionConstraint>(c, target, "pin_swing_path_" + m.contactName(c) + "_" + std::to_string(i)));
      }
    }
    for (int i = 0; i < js.swingProgressNq; ++i) {
      const double fraction = static_cast<double>(i + 1) / static_cast<double>(js.swingProgressNq + 1);
      for (int c = 0; c < m.nContacts(); ++c) {
        const Eigen::Vector2d delta = to[static_cast<size_t>(c)] - from[static_cast<size_t>(c)];
        const double distance = delta.norm();
        if (distance <= 1e-12) continue;
        const Eigen::Vector2d direction = delta / distance;
        const Eigen::Vector2d target = from[static_cast<size_t>(c)] + fraction * delta;
        const Eigen::Vector3d normal(-direction.x(), -direction.y(), 0.0);
        attachAtFlightFraction(P, k0, k1, fraction, std::make_shared<FramePositionHalfspaceConstraint>(c, normal, -direction.dot(target), "ineq_swing_progress_" + m.contactName(c) + "_" + std::to_string(i)));
      }
    }
  }
  if (js.groundClearance && js.footClear > 0.0)
    for (double fraction : {0.25, 0.5, 0.75}) attachAtFlightFraction(P, k0, k1, fraction, clearAir);
  // the launch-state objective on the last push-off element only
  if (baseAttitudeCost) P.elements[static_cast<size_t>(k0 - 1)].addCost(baseAttitudeCost);
  if (baseAngularVelocityCost) P.elements[static_cast<size_t>(k0 - 1)].addCost(baseAngularVelocityCost);
  P.validate();

  // --- the cold seed -----------------------------------------------------------------------
  {
    const CompiledProblem prob = CompiledProblem::compile(P);
    const Layout& L = prob.layout();
    out.y0.setZero(L.n());
    const double apex = js.seedApex >= 0.0 ? js.seedApex : js.hopHeight;
    double totalSeedTime = P.totalTime();
    const int M = static_cast<int>(L.elem.size());
    for (int e = 0; e < M; ++e) {
      const ElementLayout& el = L.elem[static_cast<size_t>(e)];
      const Element& spec = P.elements[static_cast<size_t>(e)];
      const Eigen::VectorXd t = prob.grid(e).timeGrid(el.T);
      const int md = spec.phase;
      int sub = 0;
      for (int ee = 0; ee < e; ++ee)
        if (P.elements[static_cast<size_t>(ee)].phase == md) ++sub;
      const int kM = splitOf[md];
      ElementViewMutable X(out.y0.data(), el);
      for (int k = 0; k < el.Nn; ++k) {
        if (js.linearSeed) {
          const double fraction = totalSeedTime > 0.0 ? (el.t0 + t(k)) / totalSeedTime : 0.0;
          const double blend = smootherstep(std::clamp(fraction, 0.0, 1.0));
          X.Q.col(k) = (1.0 - blend) * qHold0 + blend * qHold2;
        } else if (js.seedMode == 0 || md == 0) {
          X.Q.col(k) = qHold0;
        } else if (md == 2) {
          X.Q.col(k) = qHold2;
        } else if (js.seedMode == 3) {
          const double local = el.T > 0.0 ? t(k) / el.T : 0.0;
          const double fraction = (static_cast<double>(sub) + local) / kM;
          X.Q.col(k) = (1.0 - fraction) * qHold0 + fraction * qHold2;
        } else {
          // flight: smoothstep blend squat@X0 -> stand@X1 with zero end slopes plus a
          // quartic base-z bump of height `apex` (polynomial of degree <= 4 in flight
          // time, so the CGL representation of degree >= 4 is exact per element)
          const double local = el.T > 0.0 ? t(k) / el.T : 0.0;
          const double sfrac = (static_cast<double>(sub) + local) / kM;
          const double blend = smoothstep(sfrac);
          X.Q.col(k) = (1.0 - blend) * qHold0 + blend * qHold2;
          X.Q(2, k) += apex * bump(sfrac);
        }
      }
      const Eigen::MatrixXd& D = prob.grid(e).D();
      const double a = prob.timeScale(e);
      X.V = a * X.Q * D.transpose();
      X.A = a * X.V * D.transpose();
    }
    // forces: the static support force of the element's first-node pose on every node,
    // fit exactly by the Bezier polygon (constant); the liftoff control point zeroed
    for (int e = 0; e < M; ++e) {
      const ElementLayout& el = L.elem[static_cast<size_t>(e)];
      const Element& spec = P.elements[static_cast<size_t>(e)];
      ElementViewMutable X(out.y0.data(), el);
      Eigen::MatrixXd nodeForce = Eigen::MatrixXd::Zero(el.dims.nlam, el.Nn);
      if (el.Nc > 0) {
        m.evalNode(*data, X.Q.col(0), Eigen::VectorXd::Zero(m.nv()), Eigen::VectorXd::Zero(m.nv()), kRnea, cache);
        const double supportFz = std::max(cache.tau(2) / std::max(m.nContacts(), 1), js.lamMin);
        Eigen::MatrixXd desired = Eigen::MatrixXd::Zero(el.dims.nlam, el.Nn);
        int o = 0;
        for (const ContactRef& cr : spec.contacts.active) {
          desired.row(o + 2).setConstant(supportFz);
          o += cr.dim;
        }
        Eigen::MatrixXd Cphys = el.E.completeOrthogonalDecomposition().solve(desired.transpose()).transpose();
        if (js.pinLiftoffForce && e == k0 - 1) Cphys.col(el.Nc - 1).setZero();
        X.C = Cphys * el.sigma.cwiseInverse().asDiagonal();
        nodeForce = Cphys * el.E.transpose();
      }
      if (!el.uStored) continue;
      for (int k = 0; k < el.Nn; ++k) {
        m.evalNode(*data, X.Q.col(k), X.V.col(k), X.A.col(k), kRnea | kFootPos, cache);
        for (int j = 0; j < m.nu(); ++j) {
          const int qi = m.actuatedIdx()[static_cast<size_t>(j)];
          double u = cache.tau(qi);
          int o = 0;
          for (const ContactRef& cr : spec.contacts.active) {
            u -= cache.footJ[static_cast<size_t>(cr.index)].col(qi).dot(nodeForce.col(k).segment(o, cr.dim));
            o += cr.dim;
          }
          X.U(j, k) = u;
        }
      }
    }
  }
  {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "nq=%d nv=%d nu=%d  contacts=%dx dim%d  monitors=%d capsules=%d  squat z=%.4f stand z=%.4f step=(%.3f, %.3f)  "
                  "vz_liftoff>=%.3f Tair=%.3f  elements=%d (mesh %dx d%d | %dx d%d | %dx d%d, grade %.3g)",
                  m.nq(), m.nv(), m.nu(), m.nContacts(), dim, m.nMonitors(), m.nCapsules(), out.qSquat(2), out.qStand(2), js.step.x(), js.step.y(),
                  vzMinOf(js.hopHeight), js.Tair, static_cast<int>(P.elements.size()), k0, degOf[0], k1, degOf[1], k2, degOf[2], js.grade);
    out.summary = buf;
  }
  return out;
}

// --- yaml ----------------------------------------------------------------------------------------

namespace {

const std::vector<std::string>& fenceProblemKeys() {
  static const std::vector<std::string> k = {
      "type", "tstance", "tair", "tstanceland", "hop", "step", "mu", "friceps", "lammin", "lammax", "wforce", "wvel", "wacc",
      "wbaseatt", "wbaseomega", "landvz", "copx", "copy", "gammaz", "footclear", "ulim", "vlim", "qbox", "qboxnq", "vboxnq",
      "swingpathnq", "swingprogressnq", "liftoffxmax", "attmax", "attvmax", "liftoffatt", "liftoffomega", "noslipnq", "lsqmodemask",
      "poses", "planarlock", "pinliftoff", "staticseed", "mineffort", "holdlanding", "landingxslack", "posepathslack", "strictrest",
      "endpointaccel", "baseheight", "groundclearance", "linearseed", "exactkin", "dynnq", "ownership", "liftofffull", "seedmode",
      "seedapex", "contactposall", "stanceacc", "landfootpin", "landjointpin", "zerolandxy", "landvxy", "impactcone", "landramp", "actsymmetry",
      "restbox", "ubern", "selfcolmargin", "selfcolinterp", "selfcolpairs", "implicitu", "usamples", "lsqgauss", "dyngauss", "colloc",
      "dtknot", "lamtaper", "alignknots", "middyn", "stancevel", "wjerk", "wlamrate", "lamramp", "lamrampcap", "lamramppow",
      "lamrampland"};
  return k;
}

// Problem keys of the reference tree whose devices this implementation does not carry.
void rejectUnsupported(const YamlDoc& d) {
  if (d.getB("problem.exactkin", false)) throw std::runtime_error("problem.exactkin is not supported");
  if (!d.getS("problem.colloc", "").empty() && d.getS("problem.colloc", "") != "none" && d.getS("problem.colloc", "") != "cgl")
    throw std::runtime_error("problem.colloc (local collocation) is not supported");
  if (d.getB("problem.dyngauss", false)) throw std::runtime_error("problem.dyngauss is not supported");
  if (d.getI("problem.middyn", 0) != 0 || d.getI("problem.stancevel", 2) != 2 || d.getD("problem.wjerk", 0.0) != 0.0 ||
      d.getD("problem.wlamrate", 0.0) != 0.0 || d.getD("problem.lamramp", 0.0) != 0.0)
    throw std::runtime_error("local-collocation problem keys (middyn, stancevel, wjerk, wlamrate, lamramp) are not supported");
  if (d.getI("solver.eventlsq", 0) != 0) throw std::runtime_error("solver.eventlsq is not supported");
}

}  // namespace

FenceJumpSpec fenceSpecFromYaml(const YamlDoc& d) {
  d.checkKeys("problem", fenceProblemKeys());
  rejectUnsupported(d);
  FenceJumpSpec js;
  js.Tstance = d.getD("problem.tstance", js.Tstance);
  js.Tair = d.getD("problem.tair", js.Tair);
  js.TstanceLand = d.getD("problem.tstanceland", js.TstanceLand);
  js.hopHeight = d.getD("problem.hop", js.hopHeight);
  {
    const std::vector<double> st = d.getVecD("problem.step");
    if (!st.empty()) {
      if (st.size() != 2) throw std::runtime_error("problem.step wants [dx, dy]");
      js.step = Eigen::Vector2d(st[0], st[1]);
    }
  }
  js.frictionMu = d.getD("problem.mu", js.frictionMu);
  js.frictionEps = d.getD("problem.friceps", js.frictionEps);
  js.lamMin = d.getD("problem.lammin", js.lamMin);
  js.lamMax = d.getD("problem.lammax", js.lamMax);
  js.forceCostWeight = d.getD("problem.wforce", js.forceCostWeight);
  js.jointVelocityWeight = d.getD("problem.wvel", js.jointVelocityWeight);
  js.jointAccelerationWeight = d.getD("problem.wacc", js.jointAccelerationWeight);
  js.baseAttitudeWeight = d.getD("problem.wbaseatt", js.baseAttitudeWeight);
  js.baseAngularVelocityWeight = d.getD("problem.wbaseomega", js.baseAngularVelocityWeight);
  js.landVzMax = d.getD("problem.landvz", js.landVzMax);
  js.copX = d.getD("problem.copx", js.copX);
  js.copY = d.getD("problem.copy", js.copY);
  js.gammaZ = d.getD("problem.gammaz", js.gammaZ);
  js.footClear = d.getD("problem.footclear", js.footClear);
  js.uLimitScale = d.getD("problem.ulim", js.uLimitScale);
  js.velLimitScale = d.getD("problem.vlim", js.velLimitScale);
  js.qBoxScale = d.getD("problem.qbox", js.qBoxScale);
  js.attMax = d.getD("problem.attmax", js.attMax);
  js.attVelMax = d.getD("problem.attvmax", js.attVelMax);
  js.liftoffAttMax = d.getD("problem.liftoffatt", js.liftoffAttMax);
  js.liftoffOmegaMax = d.getD("problem.liftoffomega", js.liftoffOmegaMax);
  js.stanceNoSlipNq = d.getI("problem.noslipnq", js.stanceNoSlipNq);
  js.lsqModeMask = d.getI("problem.lsqmodemask", js.lsqModeMask);
  js.pinLiftoffForce = d.getB("problem.pinliftoff", js.pinLiftoffForce);
  js.minimizeEffort = d.getB("problem.mineffort", js.minimizeEffort);
  js.holdLandingJoints = d.getB("problem.holdlanding", js.holdLandingJoints);
  js.landingXSlack = d.getD("problem.landingxslack", js.landingXSlack);
  js.posePathSlack = d.getD("problem.posepathslack", js.posePathSlack);
  js.planarLock = d.getVecS("problem.planarlock");
  const int nPoses = d.seqSize("problem.poses");
  if (nPoses <= 0) throw std::runtime_error("problem.poses is required (the two rest poses)");
  for (int i = 0; i < nPoses; ++i) {
    const std::string p = "problem.poses." + std::to_string(i);
    d.checkKeys(p, {"joint", "stand", "squat"});
    PoseAngle pa;
    pa.joint = d.getS(p + ".joint", "");
    if (pa.joint.empty()) throw std::runtime_error(p + ".joint is required");
    pa.stand = d.getD(p + ".stand", 0.0);
    pa.squat = d.getD(p + ".squat", 0.0);
    js.poses.push_back(pa);
  }
  js.degree = d.getI("solver.deg", js.degree);
  js.Nc = d.getI("solver.nc", js.Nc);
  js.splitPerMode = d.getI("solver.split", js.splitPerMode);
  js.lsqNq = d.getI("solver.lsqnq", js.lsqNq);
  js.precond = d.getB("solver.precond", js.precond);
  {
    const double rho = d.getD("solver.rho", d.getD("solver.rhoabs", 0.0));
    js.lsqRho = rho > 0.0 ? rho : d.getD("solver.wdyn", 0.0) * d.getD("solver.mumax", 0.0);
  }
  for (double x : d.getVecD("solver.splits")) js.splits.push_back(static_cast<int>(x));
  for (double x : d.getVecD("solver.degs")) js.degs.push_back(static_cast<int>(x));
  for (double x : d.getVecD("solver.lsqnqs")) js.lsqnqs.push_back(static_cast<int>(x));
  js.grade = d.getD("solver.hgrade", js.grade);
  js.gradeEnds = d.getB("solver.hgradeends", js.gradeEnds);
  js.strictRest = d.getB("problem.strictrest", js.strictRest);
  js.baseHeight = d.getB("problem.baseheight", js.baseHeight);
  js.groundClearance = d.getB("problem.groundclearance", js.groundClearance);
  js.endpointAccel = d.getB("problem.endpointaccel", js.endpointAccel);
  js.linearSeed = d.getB("problem.linearseed", js.linearSeed);
  js.dynamicsInterpNq = d.getI("problem.dynnq", d.getI("solver.dynnq", 0));
  js.rowOwnership = d.getB("problem.ownership", js.rowOwnership);
  js.liftoffFullWrench = d.getB("problem.liftofffull", js.liftoffFullWrench);
  js.seedMode = d.getB("problem.staticseed", false) ? 0 : js.seedMode;
  js.seedMode = d.getI("problem.seedmode", js.seedMode);
  js.seedApex = d.getD("problem.seedapex", js.seedApex);
  js.contactPosAll = d.getB("problem.contactposall", js.contactPosAll);
  js.stanceAcc = d.getB("problem.stanceacc", js.stanceAcc);
  js.qBoxNq = d.getI("problem.qboxnq", js.qBoxNq);
  js.vBoxNq = d.getI("problem.vboxnq", js.vBoxNq);
  js.swingPathNq = d.getI("problem.swingpathnq", js.swingPathNq);
  js.swingProgressNq = d.getI("problem.swingprogressnq", js.swingProgressNq);
  js.liftoffXMax = d.getD("problem.liftoffxmax", js.liftoffXMax);
  js.landingFootPin = d.getB("problem.landfootpin", js.landingFootPin);
  js.touchdownPosture = d.getB("problem.landjointpin", js.touchdownPosture);
  js.zeroLandXY = d.getB("problem.zerolandxy", js.zeroLandXY);
  js.landVxyMax = d.getD("problem.landvxy", js.landVxyMax);
  js.impactCone = d.getB("problem.impactcone", js.impactCone);
  js.landRamp = d.getB("problem.landramp", js.landRamp);
  js.actuatedSymmetry = d.getB("problem.actsymmetry", js.actuatedSymmetry);
  js.restBoxEps = d.getD("problem.restbox", js.restBoxEps);
  js.uBernstein = d.getB("problem.ubern", js.uBernstein);
  js.deriveActuation = d.getB("problem.implicitu", js.deriveActuation);
  js.uSamples = d.getI("problem.usamples", js.uSamples);
  js.lsqGauss = d.getB("problem.lsqgauss", js.lsqGauss);
  js.selfColMargin = d.getD("problem.selfcolmargin", js.selfColMargin);
  js.selfColInterp = d.getI("problem.selfcolinterp", js.selfColInterp);
  const int nSelf = d.seqSize("problem.selfcolpairs");
  for (int i = 0; i < nSelf; ++i) {
    const std::string p = "problem.selfcolpairs." + std::to_string(i);
    d.checkKeys(p, {"a", "b", "margin"});
    FenceJumpSpec::SelfColPair sp;
    sp.a = d.getS(p + ".a", "");
    sp.b = d.getS(p + ".b", "");
    if (sp.a.empty() || sp.b.empty()) throw std::runtime_error(p + " wants capsule names a and b");
    sp.margin = d.getD(p + ".margin", -1.0);
    js.selfColPairs.push_back(sp);
  }
  js.obstacles = obstaclesFromYaml(d);
  return js;
}

ObstacleScene obstaclesFromYaml(const YamlDoc& d) {
  ObstacleScene sc;
  if (!d.has("obstacles")) return sc;
  d.checkKeys("obstacles", {"margin", "capsules", "capsule_margins", "interp", "rounded", "spheres", "boxes", "smooth"});
  sc.margin = d.getD("obstacles.margin", sc.margin);
  sc.smoothEps = d.getD("obstacles.smooth", 0.0);
  if (!std::isfinite(sc.smoothEps) || sc.smoothEps < 0.0) throw std::runtime_error("obstacles.smooth must be finite and >= 0");
  sc.interpPoints = d.getI("obstacles.interp", sc.interpPoints);
  sc.exactCuboidOutside = d.getB("obstacles.rounded", false);
  sc.capsules = d.getVecS("obstacles.capsules");
  sc.capsuleMargins = d.getVecD("obstacles.capsule_margins");
  const int nSph = d.seqSize("obstacles.spheres");
  for (int i = 0; i < nSph; ++i) {
    const std::string p = "obstacles.spheres." + std::to_string(i);
    d.checkKeys(p, {"name", "center", "radius"});
    SphereObstacle s;
    s.name = d.getS(p + ".name", "sphere" + std::to_string(i));
    const std::vector<double> c = d.getVecD(p + ".center");
    if (c.size() != 3) throw std::runtime_error(p + ".center wants [x, y, z]");
    s.center = Eigen::Vector3d(c[0], c[1], c[2]);
    s.radius = d.getD(p + ".radius", 0.0);
    sc.spheres.push_back(s);
  }
  const int nBox = d.seqSize("obstacles.boxes");
  for (int i = 0; i < nBox; ++i) {
    const std::string p = "obstacles.boxes." + std::to_string(i);
    d.checkKeys(p, {"name", "center", "size", "rpy"});
    CuboidObstacle b;
    b.name = d.getS(p + ".name", "box" + std::to_string(i));
    const std::vector<double> c = d.getVecD(p + ".center");
    const std::vector<double> s = d.getVecD(p + ".size");
    if (c.size() != 3) throw std::runtime_error(p + ".center wants [x, y, z]");
    if (s.size() != 3) throw std::runtime_error(p + ".size wants [sx, sy, sz] (full sizes)");
    b.center = Eigen::Vector3d(c[0], c[1], c[2]);
    b.half = 0.5 * Eigen::Vector3d(s[0], s[1], s[2]);
    const std::vector<double> r = d.getVecD(p + ".rpy");
    if (!r.empty()) {
      if (r.size() != 3) throw std::runtime_error(p + ".rpy wants [roll, pitch, yaw]");
      b.rot = rotationFromRpy(r[0], r[1], r[2]);
    }
    sc.boxes.push_back(b);
  }
  return sc;
}

// --- the family -----------------------------------------------------------------------------------

FenceFamily::FenceFamily(const YamlDoc& d, const RobotDescription& rd, std::shared_ptr<const RobotModel> model)
    : type_("jump_fence_v2"), name_(d.getS("name", "unnamed")), model_(std::move(model)) {
  (void)rd;
  spec_ = fenceSpecFromYaml(d);
  objective_ = objectiveFromYaml(d);
  warnings_ = fenceJumpSpecWarnings(spec_);
  build_ = buildFenceJump(spec_, model_);
  attachObjective(build_.problem, objective_, *model_);
  current_ = build_.problem;
}

TrajectoryProblem FenceFamily::build() { return current_; }

std::vector<double> FenceFamily::objectiveNodeWeights(const CompiledProblem& prob) const {
  return leap::objectiveNodeWeights(prob, objective_.present ? objective_.quadrature : "clenshaw_curtis", objective_.present ? objective_.scale : 1.0);
}

TrajectoryProblem FenceFamily::obstacleVariant(const ObstacleVariant& v) {
  if (!level0_) throw std::runtime_error("fence: obstacle variants exist on level 0 only");
  FenceJumpSpec js = spec_;
  if (v.smoothEps >= 0.0) js.obstacles.smoothEps = v.smoothEps;
  if (v.rounded >= 0) js.obstacles.exactCuboidOutside = v.rounded != 0;
  if (v.smoothEps > 0.0) js.obstacles.exactCuboidOutside = false;
  if (v.zScale != 1.0) {
    for (CuboidObstacle& bx : js.obstacles.boxes) {
      bx.center.z() *= v.zScale;
      bx.half.z() *= v.zScale;
    }
    for (SphereObstacle& sp : js.obstacles.spheres) {
      sp.center.z() *= v.zScale;
      sp.radius *= v.zScale;
    }
  }
  return buildFenceJump(js, model_).problem;
}

Eigen::VectorXd FenceFamily::seed(const CompiledProblem& prob) {
  if (build_.y0.size() != prob.layout().n()) throw std::runtime_error("fence seed: layout mismatch");
  return build_.y0;
}

void FenceFamily::refine(const std::vector<ElementSplit>& splits) {
  std::vector<ElementSplit> sorted = splits;
  std::sort(sorted.begin(), sorted.end(), splitElementDescending);  // highest element first: lower indices stay valid
  pendingElements_.clear();
  pendingFractions_.clear();
  for (const ElementSplit& s : sorted) {
    current_ = splitElement(current_, s.element, s.fraction);
    pendingElements_.push_back(s.element);
    pendingFractions_.push_back(s.fraction);
  }
  level0_ = false;
}

double FenceFamily::prolong(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to,
                            Eigen::VectorXd& yTo) {
  const double err = prolongSplits(from, yFrom, pendingElements_, pendingFractions_, to, yTo);
  pendingElements_.clear();
  pendingFractions_.clear();
  return err;
}

Eigen::VectorXd FenceFamily::oracleTermScales(const CompiledProblem& prob) const {
  return physicalTermScales(prob, *model_, build_.qSquat);
}

}  // namespace leap
