#include "leap/tasks/contact_sequence.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>

#include "leap/problem/costs.hpp"
#include "leap/problem/mesh_refinement.hpp"
#include "leap/problem/self_collision.hpp"
#include "leap/spectral/bernstein.hpp"
#include "leap/spectral/cgl_grid.hpp"
#include "leap/tasks/fence_jump.hpp"  // obstaclesFromYaml
#include "leap/tasks/objective.hpp"
#include "leap/tasks/static_family.hpp"

namespace leap {

namespace {

constexpr double kGravity = 9.81;
constexpr double kPoseTol = 1e-6;

std::vector<int> range(int first, int count) {
  std::vector<int> v(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) v[static_cast<size_t>(i)] = first + i;
  return v;
}

bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
bool containsName(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

int jointIndex(const RobotModel& m, const std::string& name, const std::string& what) {
  if (!m.hasJoint(name)) {
    std::string all;
    for (const std::string& n : m.jointNames()) all += (all.empty() ? "" : ", ") + n;
    throw std::runtime_error(what + ": no joint '" + name + "' in this robot (joints: " + all + ")");
  }
  return m.jointIndex(name);
}

int contactIndexOf(const RobotModel& m, const std::string& name, const std::string& what) {
  for (int c = 0; c < m.nContacts(); ++c)
    if (m.contactName(c) == name) return c;
  std::string all;
  for (int c = 0; c < m.nContacts(); ++c) all += (all.empty() ? "" : ", ") + m.contactName(c);
  throw std::runtime_error(what + ": no contact '" + name + "' in this robot (contacts: " + all + ")");
}

std::string setKey(const ContactSet& cs) {
  std::string k;
  for (const ContactRef& cr : cs.active) k += std::to_string(cr.index) + ",";
  return k.empty() ? std::string("-") : k;
}

std::string setNames(const RobotModel& m, const ContactSet& cs) {
  std::string s;
  for (const ContactRef& cr : cs.active) s += (s.empty() ? "" : "+") + m.contactName(cr.index);
  return s.empty() ? std::string("flight") : s;
}


// Widths of k elements shrinking geometrically by `grade` toward the start, the end or
// both ends of a mode (normalized to sum 1).
std::vector<double> gradedWidths(int k, double grade, bool towardStart, bool towardEnd) {
  std::vector<double> w(static_cast<size_t>(k), 1.0);
  if (k > 1 && grade != 1.0 && (towardStart || towardEnd)) {
    for (int i = 0; i < k; ++i) {
      int exponent;
      if (towardStart && towardEnd) exponent = std::min(i, k - 1 - i);
      else if (towardStart) exponent = k - 1 - i;
      else exponent = i;
      w[static_cast<size_t>(i)] = std::pow(grade, exponent);
    }
  }
  double sum = 0.0;
  for (double x : w) sum += x;
  for (double& x : w) x /= sum;
  return w;
}

std::vector<Eigen::VectorXd> bernsteinCoefficientRows(int degree) {
  const CglGrid g(degree);
  const Eigen::MatrixXd Einv = bernsteinBasisAtDegree(degree, g).inverse();
  std::vector<Eigen::VectorXd> W;
  for (int r = 0; r < Einv.rows(); ++r) W.push_back(Einv.row(r).transpose());
  return W;
}

// A row at physical-time fraction `fraction` of a mode whose elements are
// P.elements[k0 .. k0 + k - 1] (any widths): on a child boundary the LEFT child's last
// node owns it, otherwise it is an off-node point of the child it falls in.
void attachAtModeFraction(TrajectoryProblem& P, int k0, int k, double fraction, const std::shared_ptr<Constraint>& con) {
  constexpr double kBoundaryTol = 1e-9;
  double T = 0.0;
  for (int i = 0; i < k; ++i) T += P.elements[static_cast<size_t>(k0 + i)].T;
  const double t = fraction * T;
  double t0 = 0.0;
  for (int i = 0; i < k; ++i) {
    Element& el = P.elements[static_cast<size_t>(k0 + i)];
    const double t1 = t0 + el.T;
    if (std::abs(t - t1) <= kBoundaryTol * T) {
      el.addNode(con, Where::LastNode);
      return;
    }
    if (t < t1 || i == k - 1) {
      const double tau = std::clamp(2.0 * (t - t0) / el.T - 1.0, -1.0 + 1e-6, 1.0 - 1e-6);
      el.addOffNode(con, {tau});
      return;
    }
    t0 = t1;
  }
}

// --- inverse kinematics of one event ---------------------------------------------------
// Damped least squares on the whole configuration: the named contacts on their targets
// (and level for a flat foot), the base near `baseTarget` and level, the locked joints
// at zero, the actuated joints near qRef (projected onto their URDF box).
struct IkResult {
  Eigen::VectorXd q;
  double posErr = 0.0, tiltErr = 0.0;
};

IkResult solvePoseIk(const RobotModel& m, ModelData& data, ModelEvalCache& cache, const Eigen::VectorXd& qRef,
                     const std::vector<int>& contacts, const std::vector<Eigen::Vector3d>& targets, const Eigen::Vector3d& baseTarget,
                     const std::vector<int>& lockIdx) {
  const int nv = m.nv();
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(nv);
  bool anyWrench = false;
  for (int c : contacts) anyWrench = anyWrench || m.contactDim(c) == 6;
  const unsigned needs = kFootPos | (anyWrench ? kFootRot : 0u);
  const int nc = static_cast<int>(contacts.size());
  int nWrench = 0;
  for (int c : contacts) nWrench += m.contactDim(c) == 6 ? 1 : 0;
  const int nAct = m.nu();
  const int rows = 3 * nc + 3 * nWrench + 3 + 3 + static_cast<int>(lockIdx.size()) + nAct;
  const double wPos = 1.0, wRot = 1.0, wAtt = 0.02, wBase = 0.02, wLock = 1.0, wReg = 0.01;
  Eigen::VectorXd q = qRef;
  q.head<3>() = baseTarget;
  for (int i : lockIdx) q(i) = 0.0;
  const Eigen::VectorXd qLo = m.lowerPositionLimit(), qHi = m.upperPositionLimit();
  Eigen::MatrixXd A(rows, nv);
  Eigen::VectorXd r(rows);
  for (int iter = 0; iter < 200; ++iter) {
    m.evalNode(data, q, zero, zero, needs, cache);
    A.setZero();
    r.setZero();
    int o = 0;
    for (int k = 0; k < nc; ++k) {
      const int c = contacts[static_cast<size_t>(k)];
      r.segment(o, 3) = wPos * (cache.footPos[static_cast<size_t>(c)].head(3) - targets[static_cast<size_t>(k)]);
      A.middleRows(o, 3) = wPos * cache.footJ[static_cast<size_t>(c)].topRows(3);
      o += 3;
    }
    for (int k = 0; k < nc; ++k) {
      const int c = contacts[static_cast<size_t>(k)];
      if (m.contactDim(c) != 6) continue;
      r.segment(o, 3) = wRot * cache.footRotLog[static_cast<size_t>(c)];
      A.middleRows(o, 3) = wRot * cache.footRotJ[static_cast<size_t>(c)];
      o += 3;
    }
    for (int i = 3; i < 6; ++i) {
      r(o) = wAtt * q(i);
      A(o, i) = wAtt;
      ++o;
    }
    for (int i = 0; i < 3; ++i) {
      r(o) = wBase * (q(i) - baseTarget(i));
      A(o, i) = wBase;
      ++o;
    }
    for (int i : lockIdx) {
      r(o) = wLock * q(i);
      A(o, i) = wLock;
      ++o;
    }
    for (int j = 0; j < nAct; ++j) {
      const int i = m.actuatedIdx()[static_cast<size_t>(j)];
      r(o) = wReg * (q(i) - qRef(i));
      A(o, i) = wReg;
      ++o;
    }
    const Eigen::VectorXd dq = A.completeOrthogonalDecomposition().solve(-r);
    const double step = dq.cwiseAbs().maxCoeff();
    q += (step > 0.3 ? 0.3 / step : 1.0) * dq;
    for (int j = 0; j < m.nu(); ++j) {
      const int i = m.actuatedIdx()[static_cast<size_t>(j)];
      q(i) = std::min(std::max(q(i), qLo(i)), qHi(i));
    }
    if (step < 1e-12) break;
  }
  IkResult out;
  out.q = q;
  m.evalNode(data, q, zero, zero, needs, cache);
  for (int k = 0; k < nc; ++k) {
    const int c = contacts[static_cast<size_t>(k)];
    out.posErr = std::max(out.posErr, (cache.footPos[static_cast<size_t>(c)].head(3) - targets[static_cast<size_t>(k)]).cwiseAbs().maxCoeff());
    if (m.contactDim(c) == 6) out.tiltErr = std::max(out.tiltErr, cache.footRotLog[static_cast<size_t>(c)].norm());
  }
  return out;
}

// The configuration of a named pose with the base at the origin, and its contact
// positions.
Eigen::VectorXd poseAtOrigin(const RobotModel& m, const std::vector<JointAngle>& angles, const std::string& what) {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(m.nq());
  for (const JointAngle& ja : angles) q(jointIndex(m, ja.joint, what)) = ja.angle;
  return q;
}

void contactPositions(const RobotModel& m, ModelData& data, ModelEvalCache& cache, const Eigen::VectorXd& q,
                      std::vector<Eigen::Vector3d>& pos, double& worstTilt) {
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(m.nv());
  unsigned needs = kFootPos;
  for (int c = 0; c < m.nContacts(); ++c)
    if (m.contactDim(c) == 6) needs |= kFootRot;
  m.evalNode(data, q, zero, zero, needs, cache);
  pos.resize(static_cast<size_t>(m.nContacts()));
  worstTilt = 0.0;
  for (int c = 0; c < m.nContacts(); ++c) {
    pos[static_cast<size_t>(c)] = cache.footPos[static_cast<size_t>(c)].head<3>();
    if (m.contactDim(c) == 6) worstTilt = std::max(worstTilt, cache.footRotLog[static_cast<size_t>(c)].norm());
  }
}

// Per-contact-set constraint bundle (shared objects: one penalty term per name).
struct SetBundle {
  std::shared_ptr<Constraint> dyn;
  std::map<int, std::shared_ptr<Constraint>> vel;        // vel_<c>
  std::map<int, std::shared_ptr<Constraint>> lamZFoot;   // one-row fz floor per contact
  std::map<int, std::shared_ptr<Constraint>> lamZero;    // whole-wrench zero pin per contact (liftoff)
  std::map<int, std::shared_ptr<Constraint>> lamZeroLand;
  std::shared_ptr<Constraint> ineqLamZ;                  // fz floor, every contact of the set
  std::shared_ptr<Constraint> fric, cop, tz;
  std::shared_ptr<Constraint> clearGround;               // monitors above the floor, the set's feet exempt
};

SetBundle makeBundle(const RobotModel& m, const ContactSet& cs, const NodeDims& dimsC, const std::vector<int>& planarRows,
                     const ContactSequenceSpec& spec) {
  SetBundle b;
  if (!planarRows.empty()) b.dyn = std::make_shared<SelectedRowsDynamicsConstraint>(m, cs, planarRows);
  else b.dyn = std::make_shared<DynamicsConstraint>(m, cs);
  std::vector<int> lamZIdx;
  bool anyWrench = false;
  for (const ContactRef& cr : cs.active) {
    const std::string& nm = m.contactName(cr.index);
    anyWrench = anyWrench || cr.dim == 6;
    b.vel.emplace(cr.index, std::make_shared<FrameVelocityConstraint>(cr.index, Eigen::VectorXd::Zero(cr.dim), "vel_" + nm));
    const int zIdx = dimsC.offLam() + cs.lamOffset(cr.index) + 2;
    lamZIdx.push_back(zIdx);
    auto floor = std::make_shared<LowerBoundConstraint>(std::vector<int>{zIdx}, Eigen::VectorXd::Constant(1, spec.lamMin), "ineq_lam_z");
    floor->setForceOnly(true);
    b.lamZFoot.emplace(cr.index, floor);
    auto zero = std::make_shared<PinConstraint>(range(dimsC.offLam() + cs.lamOffset(cr.index), cr.dim), Eigen::VectorXd::Zero(cr.dim), "pin_lam_liftoff");
    zero->setForceOnly(true);
    b.lamZero.emplace(cr.index, zero);
    auto zeroLand = std::make_shared<PinConstraint>(range(dimsC.offLam() + cs.lamOffset(cr.index), cr.dim), Eigen::VectorXd::Zero(cr.dim), "pin_lam_land");
    zeroLand->setForceOnly(true);
    b.lamZeroLand.emplace(cr.index, zeroLand);
  }
  if (!lamZIdx.empty()) {
    auto floorAll = std::make_shared<LowerBoundConstraint>(lamZIdx, Eigen::VectorXd::Constant(static_cast<int>(lamZIdx.size()), spec.lamMin), "ineq_lam_z");
    floorAll->setForceOnly(true);
    b.ineqLamZ = floorAll;
  }
  if (spec.frictionMu > 0.0 && !cs.active.empty()) b.fric = std::make_shared<FrictionConeConstraint>(cs, spec.frictionMu, spec.frictionEps, "ineq_fric");
  if (anyWrench) {
    b.cop = std::make_shared<ContactWrenchBoxConstraint>(cs, std::vector<int>{4, 3}, std::vector<double>{spec.copX, spec.copY}, "ineq_cop");
    b.tz = std::make_shared<ContactWrenchBoxConstraint>(cs, std::vector<int>{5}, std::vector<double>{spec.gammaZ}, "ineq_tz");
  }
  Eigen::VectorXd lb = Eigen::VectorXd::Zero(m.nMonitors());
  for (int i = 0; i < m.nMonitors(); ++i)
    for (const ContactRef& cr : cs.active)
      if (m.monitorName(i) == m.contactName(cr.index)) lb(i) = -1.0;
  b.clearGround = std::make_shared<MonitorHeightLowerBoundConstraint>(m, lb, "ineq_clear");
  return b;
}

// Force feasibility on the control points of one element: the cone rows everywhere,
// the floor on every contact except where it lifts (tapered to zero, then pinned at the
// last control point) or lands under the ramp (pinned at the first, floor from the second).
void finalizeForce(Element& el, const SetBundle& b, const std::vector<int>& lift, const std::vector<int>& land, int Nc, double lamMin,
                   const NodeDims& dimsC, const ContactSet& cs) {
  std::vector<int> cpAll, cpButFirst;
  for (int j = 0; j < Nc; ++j) {
    cpAll.push_back(j);
    if (j > 0) cpButFirst.push_back(j);
  }
  if (b.fric) el.addForceControlPoints(b.fric, cpAll);
  if (b.cop) el.addForceControlPoints(b.cop, cpAll);
  if (b.tz) el.addForceControlPoints(b.tz, cpAll);
  if (lift.empty() && land.empty()) {
    if (b.ineqLamZ) el.addForceControlPoints(b.ineqLamZ, cpAll);
    return;
  }
  for (const ContactRef& cr : cs.active) {
    const int c = cr.index;
    const bool lifting = contains(lift, c), landing = contains(land, c);
    if (lifting) {
      // tapered floor fz >= lamMin (1 - j / (Nc - 1)) on control point j, zero at the last
      const int zIdx = dimsC.offLam() + cs.lamOffset(c) + 2;
      for (int j = 0; j + 1 < Nc; ++j) {
        if (landing && j == 0) continue;
        const double frac = 1.0 - static_cast<double>(j) / static_cast<double>(Nc - 1);
        auto floor = std::make_shared<LowerBoundConstraint>(std::vector<int>{zIdx}, Eigen::VectorXd::Constant(1, lamMin * frac), "ineq_lam_z");
        floor->setForceOnly(true);
        el.addForceControlPoints(floor, {j});
      }
      el.addForceControlPoints(b.lamZero.at(c), {Nc - 1});
    } else {
      el.addForceControlPoints(b.lamZFoot.at(c), landing ? cpButFirst : cpAll);
    }
    if (landing) el.addForceControlPoints(b.lamZeroLand.at(c), {0});
  }
}

}  // namespace

// --- validation -----------------------------------------------------------------------------

std::string validateContactSequenceSpec(const ContactSequenceSpec& cs) {
  if (cs.modes.empty()) return "modes is empty: list the contact modes of the sequence";
  for (size_t i = 0; i < cs.modes.size(); ++i) {
    const ContactSequenceMode& md = cs.modes[i];
    if (!(md.T > 0.0) || !std::isfinite(md.T)) return "modes[" + std::to_string(i) + "].T must be > 0";
    for (size_t a = 0; a < md.contacts.size(); ++a)
      for (size_t b2 = a + 1; b2 < md.contacts.size(); ++b2)
        if (md.contacts[a] == md.contacts[b2]) return "modes[" + std::to_string(i) + "] lists contact '" + md.contacts[a] + "' twice";
    if (i > 0 && md.contacts == cs.modes[i - 1].contacts && md.contacts.empty())
      return "modes[" + std::to_string(i) + "] and the mode before are both flights: merge them";
    if (!md.shift.allFinite()) return "modes[" + std::to_string(i) + "].shift must be finite";
  }
  if (cs.poseStart.empty()) return "pose_start is empty: the joint angles of the initial pose are required";
  if (cs.restStart && cs.modes.front().contacts.empty()) return "the first mode has no contact, so it cannot start at rest (rest_start)";
  if (cs.restEnd && cs.modes.back().contacts.empty()) return "the last mode has no contact, so it cannot end at rest (rest_end)";
  if (cs.degree < 2) return "deg must be >= 2";
  if (cs.Nc < 2 || cs.Nc > cs.degree + 1) return "nc must be in [2, deg + 1]";
  if (cs.splitPerMode < 1 || cs.splitPerMode > 32) return "split must be in [1, 32]";
  if (cs.grade < 0.05 || cs.grade > 5.0) return "hgrade must be in [0.05, 5.0]";
  if (cs.lsqNq < 0 || cs.lsqNq > 32) return "lsqnq must be in [0, 32]";
  if (!(cs.lsqRho > 0.0) || !std::isfinite(cs.lsqRho))
    return "solver.rho must be > 0: the least-squares dynamics-defect weight and the unit of every penalty (3e6 for a 10 kg "
           "hopper, 3e8 for a humanoid)";
  if (cs.stanceNoSlipNq < 0 || cs.stanceNoSlipNq > 16) return "noslipnq must be in [0, 16]";
  if (!std::isfinite(cs.lamMin) || cs.lamMin < 0.0) return "lammin must be finite and >= 0";
  if (!(cs.copX > 0.0) || !(cs.copY > 0.0) || !(cs.gammaZ > 0.0)) return "copx, copy and gammaz must be > 0";
  if (!std::isfinite(cs.hop) || cs.hop < 0.0) return "hop must be finite and >= 0";
  if (!std::isfinite(cs.selfColMargin) || cs.selfColMargin < 0.0) return "selfcolmargin must be finite and >= 0";
  if (cs.selfColInterp < 0 || cs.selfColInterp > 16) return "selfcolinterp must be in [0, 16]";
  for (const ContactSequenceSpec::SelfColPair& p : cs.selfColPairs) {
    if (p.a.empty() || p.b.empty() || p.a == p.b) return "selfcolpairs entries want two distinct capsule names";
    if (!std::isfinite(p.margin) || (p.margin < 0.0 && p.margin != -1.0)) return "selfcolpairs per-pair margin must be >= 0 (or omitted)";
  }
  // a pure flight must be ballistically consistent with the hop it is asked for
  if (cs.hop > 0.0) {
    for (size_t i = 0; i < cs.modes.size(); ++i) {
      if (!cs.modes[i].contacts.empty()) continue;
      const double vzNeed = std::sqrt(2.0 * kGravity * cs.hop), vzAvail = kGravity * cs.modes[i].T / 2.0;
      if (vzNeed > 1.05 * vzAvail)
        return "modes[" + std::to_string(i) + "]: a hop of " + std::to_string(cs.hop) + " m needs liftoff vz >= " + std::to_string(vzNeed) +
               " m/s but a flight of T=" + std::to_string(cs.modes[i].T) + " s supports " + std::to_string(vzAvail) + " m/s";
    }
  }
  return "";
}

std::vector<std::string> contactSequenceSpecWarnings(const ContactSequenceSpec& cs) {
  std::vector<std::string> w;
  if (cs.velLimitScale <= 0.0) w.push_back("vlim <= 0: joint speed limits are not enforced");
  if (cs.uLimitScale <= 0.0) w.push_back("ulim <= 0: torque limits are not enforced");
  if (cs.frictionMu <= 0.0) w.push_back("mu <= 0: no friction cone");
  bool anyFlight = false;
  for (const ContactSequenceMode& md : cs.modes) anyFlight = anyFlight || md.contacts.empty();
  if (anyFlight && cs.hop <= 0.0) w.push_back("a flight mode without hop: no liftoff speed is required, the seed lifts the base by footclear only");
  if (!cs.obstacles.empty()) w.push_back("static obstacles: no continuation pre-stage (the seed should not cross them)");
  return w;
}

// --- the builder ------------------------------------------------------------------------------

ContactSequenceBuild buildContactSequence(const ContactSequenceSpec& cs, std::shared_ptr<const RobotModel> modelPtr) {
  {
    const std::string err = validateContactSequenceSpec(cs);
    if (!err.empty()) throw std::invalid_argument("contact sequence: " + err);
  }
  const RobotModel& m = *modelPtr;
  ContactSequenceBuild out;
  out.problem.model = modelPtr;
  out.problem.preconditionForce = cs.precond;
  if (m.nContacts() < 1) throw std::runtime_error("contact sequence: the robot yaml declares no contacts");
  {
    const std::vector<std::string>& jn = m.jointNames();
    const char* want[6] = {"base_px", "base_py", "base_pz", "base_rx", "base_ry", "base_rz"};
    if (jn.size() < 6) throw std::runtime_error("contact sequence: not a floating-base robot");
    for (int i = 0; i < 6; ++i)
      if (jn[static_cast<size_t>(i)] != want[i])
        throw std::runtime_error(std::string("contact sequence: joint ") + std::to_string(i) + " is '" + jn[static_cast<size_t>(i)] + "', expected '" + want[i] + "'");
  }
  const int nModes = static_cast<int>(cs.modes.size());
  const ContactSet all = m.allContacts();
  const NodeDims dimsC = m.nodeDims(all);

  // --- the contact sets, the landing / lifting contacts, the impact sets ---
  std::vector<ContactSet> sets(static_cast<size_t>(nModes));
  std::vector<std::vector<int>> lands(static_cast<size_t>(nModes)), lifts(static_cast<size_t>(nModes));
  for (int i = 0; i < nModes; ++i) {
    std::vector<std::string> names;
    for (int c = 0; c < m.nContacts(); ++c)
      if (containsName(cs.modes[static_cast<size_t>(i)].contacts, m.contactName(c))) names.push_back(m.contactName(c));
    for (const std::string& nm : cs.modes[static_cast<size_t>(i)].contacts) contactIndexOf(m, nm, "modes[" + std::to_string(i) + "].contacts");
    sets[static_cast<size_t>(i)] = names.empty() ? ContactSet() : m.contactSet(names);
  }
  for (int i = 0; i < nModes; ++i) {
    for (const ContactRef& cr : sets[static_cast<size_t>(i)].active) {
      if (i > 0 && !sets[static_cast<size_t>(i) - 1].contains(cr.index)) lands[static_cast<size_t>(i)].push_back(cr.index);
      if (i + 1 < nModes && !sets[static_cast<size_t>(i) + 1].contains(cr.index)) lifts[static_cast<size_t>(i)].push_back(cr.index);
    }
  }
  std::vector<int> impactSet(static_cast<size_t>(nModes), -1), coneSet(static_cast<size_t>(nModes), -1);
  for (int i = 1; i < nModes; ++i) {
    if (lands[static_cast<size_t>(i)].empty()) continue;
    impactSet[static_cast<size_t>(i)] = m.ensureImpactSet(sets[static_cast<size_t>(i)]);  // before makeData()
    if (cs.impactCone) coneSet[static_cast<size_t>(i)] = m.ensureImpactConeSet(sets[static_cast<size_t>(i)]);
  }
  ModelDataPtr data = m.makeData();
  ModelEvalCache cache;
  m.initCache(cache);

  // --- the poses and the placements ---
  std::vector<int> lockIdx;
  for (const std::string& n : cs.planarLock) lockIdx.push_back(jointIndex(m, n, "planarlock"));
  std::vector<int> planarRows;
  if (!lockIdx.empty())
    for (int row = 0; row < m.nv(); ++row)
      if (!contains(lockIdx, row)) planarRows.push_back(row);
  Eigen::VectorXd qStart = poseAtOrigin(m, cs.poseStart, "pose_start");
  std::vector<Eigen::Vector3d> feetStart;
  double tilt = 0.0;
  contactPositions(m, *data, cache, qStart, feetStart, tilt);
  if (tilt > kPoseTol) throw std::runtime_error("contact sequence: pose_start does not level a flat foot (sole tilt " + std::to_string(tilt) + " rad)");
  std::vector<Eigen::Vector3d> placed(static_cast<size_t>(m.nContacts()));
  if (cs.start.empty()) {
    double zMin = feetStart[0].z();
    for (const Eigen::Vector3d& p : feetStart) zMin = std::min(zMin, p.z());
    qStart(2) -= zMin;
    for (int c = 0; c < m.nContacts(); ++c) placed[static_cast<size_t>(c)] = feetStart[static_cast<size_t>(c)] - Eigen::Vector3d(0.0, 0.0, zMin);
  } else {
    std::vector<bool> given(static_cast<size_t>(m.nContacts()), false);
    for (const ContactPlacement& p : cs.start) {
      const int c = contactIndexOf(m, p.contact, "start");
      placed[static_cast<size_t>(c)] = p.pos;
      given[static_cast<size_t>(c)] = true;
    }
    // the base translation that puts the pose's feet on the given placements
    Eigen::Vector3d shift = Eigen::Vector3d::Zero();
    int n = 0;
    for (int c = 0; c < m.nContacts(); ++c)
      if (given[static_cast<size_t>(c)]) {
        shift += placed[static_cast<size_t>(c)] - feetStart[static_cast<size_t>(c)];
        ++n;
      }
    if (n == 0) throw std::runtime_error("contact sequence: start lists no contact");
    shift /= static_cast<double>(n);
    for (int c = 0; c < m.nContacts(); ++c) {
      if (!given[static_cast<size_t>(c)]) {
        placed[static_cast<size_t>(c)] = feetStart[static_cast<size_t>(c)] + shift;
        continue;
      }
      const double err = (feetStart[static_cast<size_t>(c)] + shift - placed[static_cast<size_t>(c)]).norm();
      if (err > 1e-4)
        throw std::runtime_error("contact sequence: the start placement of '" + m.contactName(c) + "' is " + std::to_string(err) +
                                 " m from where pose_start puts that foot (up to a base translation); make the placements and the "
                                 "pose consistent, or omit `start` to take the pose's own foot positions");
    }
    qStart.head<3>() += shift;
  }
  double zPlacedMean = 0.0;
  for (const Eigen::Vector3d& p : placed) zPlacedMean += p.z();
  zPlacedMean /= static_cast<double>(placed.size());
  const double zNominal = qStart(2) - zPlacedMean;  // the base height above the feet
  // placements during every mode
  std::vector<std::vector<Eigen::Vector3d>> placementAt(static_cast<size_t>(nModes));
  for (int i = 0; i < nModes; ++i) {
    const ContactSequenceMode& md = cs.modes[static_cast<size_t>(i)];
    for (int c : lands[static_cast<size_t>(i)]) placed[static_cast<size_t>(c)] += md.shift;
    for (const ContactPlacement& p : md.feet) {
      const int c = contactIndexOf(m, p.contact, "modes[" + std::to_string(i) + "].feet");
      if (!sets[static_cast<size_t>(i)].contains(c))
        throw std::runtime_error("contact sequence: modes[" + std::to_string(i) + "].feet places '" + p.contact + "', which is not active in that mode");
      placed[static_cast<size_t>(c)] = p.pos;
    }
    placementAt[static_cast<size_t>(i)] = placed;
  }
  // the end pose (optional): its base follows the final placements of the last mode's contacts
  const bool hasEndPose = !cs.poseEnd.empty();
  Eigen::VectorXd qEnd;
  if (hasEndPose) {
    qEnd = poseAtOrigin(m, cs.poseEnd, "pose_end");
    std::vector<Eigen::Vector3d> feetEnd;
    contactPositions(m, *data, cache, qEnd, feetEnd, tilt);
    if (tilt > kPoseTol) throw std::runtime_error("contact sequence: pose_end does not level a flat foot (sole tilt " + std::to_string(tilt) + " rad)");
    const ContactSet& last = sets[static_cast<size_t>(nModes) - 1];
    Eigen::Vector3d shift = Eigen::Vector3d::Zero();
    for (const ContactRef& cr : last.active) shift += placementAt.back()[static_cast<size_t>(cr.index)] - feetEnd[static_cast<size_t>(cr.index)];
    if (!last.active.empty()) shift /= static_cast<double>(last.active.size());
    for (const ContactRef& cr : last.active) {
      const double err = (feetEnd[static_cast<size_t>(cr.index)] + shift - placementAt.back()[static_cast<size_t>(cr.index)]).norm();
      if (err > 1e-4)
        throw std::runtime_error("contact sequence: pose_end puts '" + m.contactName(cr.index) + "' " + std::to_string(err) +
                                 " m from its final placement (up to a base translation); the rest pose at the end must stand on the "
                                 "final placements. Adjust pose_end or the placements, or omit pose_end (the end is then at rest in a free pose)");
    }
    qEnd.head<3>() += shift;
  }

  // --- the pose at every event (the start of every mode and the end), by IK ---
  out.qEvent.resize(static_cast<size_t>(nModes) + 1);
  double Ttotal = 0.0;
  for (const ContactSequenceMode& md : cs.modes) Ttotal += md.T;
  double tEvent = 0.0;
  double worstTilt = 0.0;
  for (int k = 0; k <= nModes; ++k) {
    if (k > 0) tEvent += cs.modes[static_cast<size_t>(k) - 1].T;
    if (k == 0 && cs.restStart) {
      out.qEvent[0] = qStart;
      continue;
    }
    if (k == nModes && hasEndPose && cs.restEnd) {
      out.qEvent[static_cast<size_t>(k)] = qEnd;
      continue;
    }
    // the contacts pinned at this instant: the union of the sets before and after
    std::vector<int> pinned;
    std::vector<Eigen::Vector3d> targets;
    const int before = k - 1, after = k < nModes ? k : -1;
    for (int c = 0; c < m.nContacts(); ++c) {
      if (after >= 0 && sets[static_cast<size_t>(after)].contains(c)) {
        pinned.push_back(c);
        targets.push_back(placementAt[static_cast<size_t>(after)][static_cast<size_t>(c)]);
      } else if (before >= 0 && sets[static_cast<size_t>(before)].contains(c)) {
        pinned.push_back(c);
        targets.push_back(placementAt[static_cast<size_t>(before)][static_cast<size_t>(c)]);
      }
    }
    const double frac = Ttotal > 0.0 ? tEvent / Ttotal : 0.0;
    const Eigen::VectorXd qRef = hasEndPose ? Eigen::VectorXd((1.0 - frac) * qStart + frac * qEnd) : qStart;
    Eigen::Vector3d baseTarget = qRef.head<3>();
    if (!pinned.empty()) {
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (const Eigen::Vector3d& t : targets) mean += t;
      mean /= static_cast<double>(targets.size());
      // keep the pose's own base offset over its feet (a hopper's base is not above its foot)
      Eigen::Vector3d refMean = Eigen::Vector3d::Zero();
      std::vector<Eigen::Vector3d> refFeet;
      double t2 = 0.0;
      contactPositions(m, *data, cache, qRef, refFeet, t2);
      for (int c : pinned) refMean += refFeet[static_cast<size_t>(c)];
      refMean /= static_cast<double>(pinned.size());
      baseTarget = qRef.head<3>() + (mean - refMean);
      baseTarget.z() = mean.z() + zNominal;
    }
    const IkResult ik = solvePoseIk(m, *data, cache, qRef, pinned, targets, baseTarget, lockIdx);
    out.qEvent[static_cast<size_t>(k)] = ik.q;
    out.ikWorstErr = std::max(out.ikWorstErr, ik.posErr);
    worstTilt = std::max(worstTilt, ik.tiltErr);
  }
  if (out.ikWorstErr > 0.05)
    throw std::runtime_error("contact sequence: the placements are not simultaneously reachable: inverse kinematics leaves " +
                             std::to_string(out.ikWorstErr) + " m of foot position error at one of the events (shorten the steps or change the poses)");

  // --- constraint objects ---
  std::map<std::string, SetBundle> bundles;
  for (int i = 0; i < nModes; ++i) {
    const ContactSet& s = sets[static_cast<size_t>(i)];
    if (bundles.find(setKey(s)) == bundles.end()) bundles.emplace(setKey(s), makeBundle(m, s, dimsC, planarRows, cs));
  }
  auto ineqBaseZ = std::make_shared<LowerBoundConstraint>(std::vector<int>{2}, Eigen::VectorXd::Zero(1), "ineq_base_z");
  auto pinRestV = std::make_shared<PinConstraint>(range(dimsC.offV(), m.nv()), Eigen::VectorXd::Zero(m.nv()), "pin_v_rest");
  auto effort = std::make_shared<ControlEffortCost>();
  std::vector<int> attitudeAxes;
  for (int i = 3; i < 6; ++i)
    if (!contains(lockIdx, i)) attitudeAxes.push_back(i);
  const std::vector<double> noSlipTaus = cs.stanceNoSlipNq > 0 ? uniformInteriorTaus(cs.stanceNoSlipNq) : std::vector<double>{};

  // --- the parent modes ---
  std::vector<Element> modes(static_cast<size_t>(nModes));
  for (int i = 0; i < nModes; ++i) {
    const ContactSequenceMode& md = cs.modes[static_cast<size_t>(i)];
    const ContactSet& s = sets[static_cast<size_t>(i)];
    const SetBundle& b = bundles.at(setKey(s));
    Element& e = modes[static_cast<size_t>(i)];
    e.label = md.label.empty() ? "mode" + std::to_string(i) + "_" + setNames(m, s) : md.label;
    e.phase = i;
    e.degree = cs.degree;
    e.T = md.T;
    e.contacts = s;
    e.forceControlPoints = s.active.empty() ? 0 : cs.Nc;
    const bool first = i == 0, last = i == nModes - 1;
    e.addNode(b.dyn, Where::AllNodes);
    if (cs.groundClearance) e.addNode(b.clearGround, Where::AllNodes);
    if (cs.baseHeight) {
      if (first) e.addNode(ineqBaseZ, Where::AllNodes);
      else {
        e.addNode(ineqBaseZ, Where::Interior);
        e.addNode(ineqBaseZ, Where::LastNode);
      }
    }
    // the stance contacts: where they stand, no slip (the first node of a mode is the
    // previous mode's last node, which already carries the rows: single ownership)
    for (const ContactRef& cr : s.active) {
      const int c = cr.index;
      const std::string& nm = m.contactName(c);
      auto pos = std::make_shared<FramePositionConstraint>(c, placementAt[static_cast<size_t>(i)][static_cast<size_t>(c)], "pos_stance_" + nm);
      std::shared_ptr<Constraint> flat;
      if (cr.dim == 6) flat = std::make_shared<FrameOrientationConstraint>(c, Eigen::Vector3d::Zero(), "pos_flat_" + nm);
      std::vector<Where> where{Where::Interior, Where::LastNode};
      if (first && !cs.restStart) where.push_back(Where::FirstNode);
      for (Where w : where) {
        e.addNode(pos, w);
        if (flat) e.addNode(flat, w);
        e.addNode(b.vel.at(c), w);
      }
      if (!noSlipTaus.empty()) e.addOffNode(b.vel.at(c), noSlipTaus);
    }
    if (first && cs.restStart) {
      e.addNode(std::make_shared<PinConstraint>(range(0, m.nq()), qStart, "pin_q_rest"), Where::FirstNode);
      e.addNode(pinRestV, Where::FirstNode);
    }
    if (last && cs.restEnd) {
      if (hasEndPose) e.addNode(std::make_shared<PinConstraint>(range(0, m.nq()), qEnd, "pin_q_rest"), Where::LastNode);
      e.addNode(pinRestV, Where::LastNode);
    }
    // the touchdown of the contacts that land in the next mode, at this mode's last node
    if (i + 1 < nModes) {
      const std::vector<int>& landing = lands[static_cast<size_t>(i) + 1];
      for (int c : landing) {
        const std::string& nm = m.contactName(c);
        e.addNode(std::make_shared<FramePositionConstraint>(c, placementAt[static_cast<size_t>(i) + 1][static_cast<size_t>(c)], "pos_land_" + nm), Where::LastNode);
        if (m.contactDim(c) == 6) e.addNode(std::make_shared<FrameOrientationConstraint>(c, Eigen::Vector3d::Zero(), "pos_flat_" + nm), Where::LastNode);
        if (cs.landVzMax > 0.0)
          e.addNode(std::make_shared<FrameVelocityBoundConstraint>(c, 2, -1.0, -cs.landVzMax, "ineq_land_vz_" + nm, m.contactDim(c)), Where::LastNode);
      }
      if (!landing.empty() && cs.impactCone) {
        const ContactSet& next = sets[static_cast<size_t>(i) + 1];
        std::vector<int> coneOff;
        for (const ContactRef& cr : next.active) coneOff.push_back(m.impactConeLamOffset(coneSet[static_cast<size_t>(i) + 1], cr.index));
        e.addNode(std::make_shared<ImpactConeConstraint>(coneSet[static_cast<size_t>(i) + 1], coneOff, cs.frictionMu, cs.frictionEps, "ineq_impact_cone"), Where::LastNode);
      }
      // a pure flight next: the liftoff speed of the hop
      if (sets[static_cast<size_t>(i) + 1].active.empty() && cs.hop > 0.0)
        e.addNode(std::make_shared<LowerBoundConstraint>(std::vector<int>{dimsC.offV() + 2}, Eigen::VectorXd::Constant(1, std::sqrt(2.0 * kGravity * cs.hop)), "ineq_vz_up"),
                  Where::LastNode);
    }
    if (cs.minimizeEffort) e.addCost(effort);
    if (cs.lsqNq > 0) e.setDefectCost(b.dyn, uniformInteriorTaus(cs.lsqNq), cs.lsqRho);
  }
  // hardware boxes, attitude, planar lock, motor envelope on every mode
  {
    const std::vector<int>& act = m.actuatedIdx();
    const int nu = m.nu();
    if (cs.qBoxScale > 0.0) {
      const Eigen::VectorXd lo = m.lowerPositionLimit(), hi = m.upperPositionLimit();
      Eigen::VectorXd qlb(nu), qub(nu);
      for (int j = 0; j < nu; ++j) {
        const int i = act[static_cast<size_t>(j)];
        const double mid = 0.5 * (lo(i) + hi(i)), half = 0.5 * (hi(i) - lo(i));
        qlb(j) = mid - cs.qBoxScale * half;
        qub(j) = mid + cs.qBoxScale * half;
      }
      auto boxQ = std::make_shared<BoxConstraint>(Field::Q, act, qlb, qub, "box_q");
      for (Element& e : modes) e.addNode(boxQ, Where::AllNodes);
    }
    if (cs.velLimitScale > 0.0) {
      const Eigen::VectorXd vlim = cs.velLimitScale * m.actuatedVelocityLimit();
      auto boxV = std::make_shared<BoxConstraint>(Field::V, act, -vlim, vlim, "box_v");
      for (Element& e : modes) e.addNode(boxV, Where::AllNodes);
    }
    if (cs.uLimitScale > 0.0) {
      const Eigen::VectorXd ulim = cs.uLimitScale * m.actuatedEffortLimit();
      if (cs.uBernstein) {
        auto boxUCert = std::make_shared<BoxConstraint>(Field::U, range(0, nu), -0.995 * ulim, 0.995 * ulim, "box_u");
        for (Element& e : modes) e.addFieldFunctionals(boxUCert, Field::U, bernsteinCoefficientRows(cs.degree));
      } else {
        auto boxU = std::make_shared<BoxConstraint>(Field::U, range(0, nu), -ulim, ulim, "box_u");
        for (Element& e : modes) e.addNode(boxU, Where::AllNodes);
      }
    }
    if (cs.attMax > 0.0 && !attitudeAxes.empty()) {
      const int n = static_cast<int>(attitudeAxes.size());
      auto boxAtt = std::make_shared<BoxConstraint>(Field::Q, attitudeAxes, Eigen::VectorXd::Constant(n, -cs.attMax), Eigen::VectorXd::Constant(n, cs.attMax), "box_att");
      for (Element& e : modes) e.addNode(boxAtt, Where::AllNodes);
    }
    const MotorSaturationConfig& sat = m.description().saturation;
    if (sat.enabled && sat.enforceInOptimization) {
      auto env = std::make_shared<MotorEnvelopeConstraint>(m, sat, "ineq_motor_envelope");
      for (Element& e : modes) {
        e.addNode(env, Where::AllNodes);
        e.addOffNode(env, uniformInteriorTaus(4));
      }
    }
    if (!lockIdx.empty()) {
      // the locked coordinates are pinned in q, v AND a: pinning q alone leaves the
      // spectral derivatives of the locked coordinates free (the dropped dynamics rows
      // do not see them)
      const int nl = static_cast<int>(lockIdx.size());
      std::vector<int> lockV, lockA;
      for (int i : lockIdx) {
        lockV.push_back(dimsC.offV() + i);
        lockA.push_back(dimsC.offA() + i);
      }
      auto pinPlanar = std::make_shared<PinConstraint>(lockIdx, Eigen::VectorXd::Zero(nl), "pin_planar");
      auto pinPlanarV = std::make_shared<PinConstraint>(lockV, Eigen::VectorXd::Zero(nl), "pin_planar_v");
      auto pinPlanarA = std::make_shared<PinConstraint>(lockA, Eigen::VectorXd::Zero(nl), "pin_planar_a");
      pinPlanarA->setOneSided(true);
      for (int i = 0; i < nModes; ++i) {
        Element& e = modes[static_cast<size_t>(i)];
        const bool skipFirst = i == 0 && cs.restStart, skipLast = i == nModes - 1 && cs.restEnd && hasEndPose;
        for (const std::shared_ptr<Constraint>& pin : {pinPlanar, pinPlanarV}) {
          if (!skipFirst && !skipLast) e.addNode(pin, Where::AllNodes);
          else {
            if (!skipFirst) e.addNode(pin, Where::FirstNode);
            e.addNode(pin, Where::Interior);
            if (!skipLast) e.addNode(pin, Where::LastNode);
          }
        }
        e.addNode(pinPlanarA, Where::AllNodes);
      }
    }
  }
  // static obstacles and self-collision on every mode (before the split: each child
  // then checks its own interior points)
  if (!cs.obstacles.empty()) {
    TrajectoryProblem tmp;
    tmp.model = modelPtr;
    for (Element& e : modes) tmp.elements.push_back(e);
    attachObstacles(m, cs.obstacles, tmp);
    for (int i = 0; i < nModes; ++i) modes[static_cast<size_t>(i)] = tmp.elements[static_cast<size_t>(i)];
  }
  if (!cs.selfColPairs.empty()) {
    std::vector<std::pair<int, int>> pairs;
    std::vector<double> margins;
    for (const ContactSequenceSpec::SelfColPair& p : cs.selfColPairs) {
      pairs.push_back({m.capsuleIndex(p.a), m.capsuleIndex(p.b)});
      margins.push_back(p.margin >= 0.0 ? p.margin : cs.selfColMargin);
    }
    auto selfCol = std::make_shared<CapsuleSelfCollisionConstraint>(m, pairs, cs.selfColMargin, "ineq_selfcol", margins);
    for (Element& e : modes) {
      e.addNode(selfCol, Where::AllNodes);
      if (cs.selfColInterp > 0) e.addOffNode(selfCol, uniformInteriorTaus(cs.selfColInterp));
    }
  }

  // --- split every mode into its elements (stances graded toward their contact changes) ---
  TrajectoryProblem& P = out.problem;
  std::vector<int> firstElem(static_cast<size_t>(nModes));
  const int k = cs.splitPerMode;
  for (int i = 0; i < nModes; ++i) {
    const Element& mm = modes[static_cast<size_t>(i)];
    const ContactSet& s = sets[static_cast<size_t>(i)];
    const bool changeAtStart = i > 0 && sets[static_cast<size_t>(i) - 1] != s;
    const bool changeAtEnd = i + 1 < nModes && sets[static_cast<size_t>(i) + 1] != s;
    const bool stance = !s.active.empty();
    const std::vector<double> widths = gradedWidths(k, cs.grade, stance && changeAtStart, stance && changeAtEnd);
    firstElem[static_cast<size_t>(i)] = static_cast<int>(P.elements.size());
    for (int j = 0; j < k; ++j) {
      Element sub;
      sub.label = mm.label + (k > 1 ? "_" + std::to_string(j) : "");
      sub.phase = i;
      sub.degree = mm.degree;
      sub.T = mm.T * widths[static_cast<size_t>(j)];
      sub.contacts = mm.contacts;
      sub.forceControlPoints = mm.forceControlPoints;
      sub.costs = mm.costs;
      sub.pointTerms = mm.pointTerms;
      sub.defect = mm.defect;
      for (const NodeAttachment& a : mm.nodeTerms)
        for (Where w : mapWhereUniqueOwner(a.where, j, k, !a.con->readsOneSidedFields())) sub.addNode(a.con, w);
      P.elements.push_back(sub);
      if (j < k - 1) P.addReset(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
    }
    if (i + 1 < nModes) {
      if (lands[static_cast<size_t>(i) + 1].empty()) P.addReset(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
      else P.addReset(std::make_shared<PlasticImpactReset>(impactSet[static_cast<size_t>(i) + 1], dimsC.nq, dimsC.nv));
    }
  }
  // the contact-force rows on every stance element; the swing-foot clearance rows
  for (int i = 0; i < nModes; ++i) {
    const ContactSet& s = sets[static_cast<size_t>(i)];
    const SetBundle& b = bundles.at(setKey(s));
    const int k0 = firstElem[static_cast<size_t>(i)];
    for (int j = 0; j < k; ++j) {
      Element& el = P.elements[static_cast<size_t>(k0 + j)];
      if (s.active.empty()) continue;
      finalizeForce(el, b, j == k - 1 ? lifts[static_cast<size_t>(i)] : std::vector<int>{},
                    (j == 0 && cs.landRamp) ? lands[static_cast<size_t>(i)] : std::vector<int>{}, cs.Nc, cs.lamMin, dimsC, s);
    }
    if (i + 1 < nModes && cs.groundClearance && cs.footClear > 0.0 && !lands[static_cast<size_t>(i) + 1].empty()) {
      // the feet that land next are in the air now: above footclear at the quarter points
      Eigen::VectorXd lb = Eigen::VectorXd::Zero(m.nMonitors());
      for (int mi = 0; mi < m.nMonitors(); ++mi) {
        for (const ContactRef& cr : s.active)
          if (m.monitorName(mi) == m.contactName(cr.index)) lb(mi) = -1.0;
        for (int c : lands[static_cast<size_t>(i) + 1])
          if (m.monitorName(mi) == m.contactName(c)) lb(mi) = cs.footClear;
      }
      auto clearAir = std::make_shared<MonitorHeightLowerBoundConstraint>(m, lb, "ineq_swing_z");
      for (double fraction : {0.25, 0.5, 0.75}) attachAtModeFraction(P, k0, k, fraction, clearAir);
    }
  }
  P.validate();

  // --- the cold seed: a quadratic Bezier in joint space through a mid-mode arc pose
  //     (the swing feet lifted), velocities and accelerations by spectral differentiation,
  //     the static support force split over the stance contacts, torques by inverse dynamics
  {
    const CompiledProblem prob = CompiledProblem::compile(P);
    const Layout& L = prob.layout();
    out.y0.setZero(L.n());
    const double apex = cs.seedApex >= 0.0 ? cs.seedApex : (cs.hop > 0.0 ? cs.hop : std::max(cs.footClear, 0.0));
    std::vector<Eigen::VectorXd> qArc(static_cast<size_t>(nModes));
    for (int i = 0; i < nModes; ++i) {
      const Eigen::VectorXd& q0 = out.qEvent[static_cast<size_t>(i)];
      const Eigen::VectorXd& q1 = out.qEvent[static_cast<size_t>(i) + 1];
      const ContactSet& s = sets[static_cast<size_t>(i)];
      const bool flight = s.active.empty();
      const std::vector<int>& landing = i + 1 < nModes ? lands[static_cast<size_t>(i) + 1] : std::vector<int>{};
      if (!flight && landing.empty()) {
        qArc[static_cast<size_t>(i)] = Eigen::VectorXd();  // plain stance: linear blend
        continue;
      }
      // the arc pose: stance feet on their placements, the swing feet lifted by `apex`
      // halfway between liftoff and landing, the base halfway (and lifted in a flight)
      std::vector<int> pinned;
      std::vector<Eigen::Vector3d> targets;
      std::vector<Eigen::Vector3d> feet0, feet1;
      double t0 = 0.0, t1 = 0.0;
      contactPositions(m, *data, cache, q0, feet0, t0);
      contactPositions(m, *data, cache, q1, feet1, t1);
      for (int c = 0; c < m.nContacts(); ++c) {
        if (s.contains(c)) {
          pinned.push_back(c);
          targets.push_back(placementAt[static_cast<size_t>(i)][static_cast<size_t>(c)]);
        } else if (contains(landing, c)) {
          pinned.push_back(c);
          targets.push_back(0.5 * (feet0[static_cast<size_t>(c)] + feet1[static_cast<size_t>(c)]) + Eigen::Vector3d(0.0, 0.0, apex));
        }
      }
      const Eigen::VectorXd qRef = 0.5 * (q0 + q1);
      Eigen::Vector3d baseTarget = qRef.head<3>();
      if (flight) baseTarget.z() += apex;
      const IkResult ik = solvePoseIk(m, *data, cache, qRef, pinned, targets, baseTarget, lockIdx);
      qArc[static_cast<size_t>(i)] = ik.q;
    }
    const int M = static_cast<int>(L.elem.size());
    for (int e = 0; e < M; ++e) {
      const ElementLayout& el = L.elem[static_cast<size_t>(e)];
      const int i = P.elements[static_cast<size_t>(e)].phase;
      const Eigen::VectorXd& q0 = out.qEvent[static_cast<size_t>(i)];
      const Eigen::VectorXd& q1 = out.qEvent[static_cast<size_t>(i) + 1];
      double tMode0 = 0.0;
      for (int ee = firstElem[static_cast<size_t>(i)]; ee < e; ++ee) tMode0 += P.elements[static_cast<size_t>(ee)].T;
      const double Tmode = cs.modes[static_cast<size_t>(i)].T;
      const Eigen::VectorXd t = prob.grid(e).timeGrid(el.T);
      ElementViewMutable X(out.y0.data(), el);
      for (int kk = 0; kk < el.Nn; ++kk) {
        const double sg = std::clamp((tMode0 + t(kk)) / Tmode, 0.0, 1.0);
        if (qArc[static_cast<size_t>(i)].size() > 0) {
          const Eigen::VectorXd ctrl = 2.0 * qArc[static_cast<size_t>(i)] - 0.5 * (q0 + q1);
          X.Q.col(kk) = (1.0 - sg) * (1.0 - sg) * q0 + 2.0 * sg * (1.0 - sg) * ctrl + sg * sg * q1;
        } else {
          X.Q.col(kk) = (1.0 - sg) * q0 + sg * q1;
        }
      }
      const Eigen::MatrixXd& D = prob.grid(e).D();
      const double a = prob.timeScale(e);
      X.V = a * X.Q * D.transpose();
      X.A = a * X.V * D.transpose();
    }
    const double weight = m.totalMass() * kGravity;
    for (int e = 0; e < M; ++e) {
      const ElementLayout& el = L.elem[static_cast<size_t>(e)];
      const Element& spec = P.elements[static_cast<size_t>(e)];
      const int i = spec.phase;
      ElementViewMutable X(out.y0.data(), el);
      Eigen::MatrixXd nodeForce = Eigen::MatrixXd::Zero(el.dims.nlam, el.Nn);
      if (el.Nc > 0) {
        const double supportFz = std::max(weight / static_cast<double>(spec.contacts.active.size()), cs.lamMin);
        Eigen::MatrixXd desired = Eigen::MatrixXd::Zero(el.dims.nlam, el.Nn);
        int o = 0;
        for (const ContactRef& cr : spec.contacts.active) {
          desired.row(o + 2).setConstant(supportFz);
          o += cr.dim;
        }
        Eigen::MatrixXd Cphys = el.E.completeOrthogonalDecomposition().solve(desired.transpose()).transpose();
        const bool lastOfMode = e == firstElem[static_cast<size_t>(i)] + k - 1;
        if (lastOfMode) {
          int oo = 0;
          for (const ContactRef& cr : spec.contacts.active) {
            if (contains(lifts[static_cast<size_t>(i)], cr.index)) Cphys.block(oo, el.Nc - 1, cr.dim, 1).setZero();
            oo += cr.dim;
          }
        }
        X.C = Cphys * el.sigma.cwiseInverse().asDiagonal();
        nodeForce = Cphys * el.E.transpose();
      }
      for (int kk = 0; kk < el.Nn; ++kk) {
        m.evalNode(*data, X.Q.col(kk), X.V.col(kk), X.A.col(kk), kRnea | kFootPos, cache);
        for (int j = 0; j < m.nu(); ++j) {
          const int qi = m.actuatedIdx()[static_cast<size_t>(j)];
          double u = cache.tau(qi);
          int o = 0;
          for (const ContactRef& cr : spec.contacts.active) {
            u -= cache.footJ[static_cast<size_t>(cr.index)].col(qi).dot(nodeForce.col(kk).segment(o, cr.dim));
            o += cr.dim;
          }
          X.U(j, kk) = u;
        }
      }
    }
  }
  {
    std::string seq;
    for (int i = 0; i < nModes; ++i) {
      char b[96];
      std::snprintf(b, sizeof(b), "%s%s (%.3f s)", i ? " -> " : "", setNames(m, sets[static_cast<size_t>(i)]).c_str(), cs.modes[static_cast<size_t>(i)].T);
      seq += b;
    }
    char buf[640];
    std::snprintf(buf, sizeof(buf),
                  "nq=%d nv=%d nu=%d  contacts=%d  monitors=%d capsules=%d\n  modes=%d (-> %d elements of degree %d, Nc %d, split %d, grade %.3g)  T=%.3f s\n"
                  "  base z=%.4f (start) %.4f (end)  IK residual %.2e m pos, %.2e rad sole tilt\n  sequence: %s",
                  m.nq(), m.nv(), m.nu(), m.nContacts(), m.nMonitors(), m.nCapsules(), nModes, static_cast<int>(P.elements.size()), cs.degree, cs.Nc, k,
                  cs.grade, Ttotal, out.qEvent.front()(2), out.qEvent.back()(2), out.ikWorstErr, worstTilt, seq.c_str());
    out.summary = buf;
  }
  return out;
}

// --- yaml ----------------------------------------------------------------------------------------

namespace {

Eigen::Vector3d vec3(const YamlDoc& d, const std::string& key, const char* what) {
  const std::vector<double> v = d.getVecD(key);
  if (v.size() != 3) throw std::runtime_error(key + " wants [x, y, z] (" + what + ")");
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

std::vector<JointAngle> anglesFromYaml(const YamlDoc& d, const std::string& key) {
  std::vector<JointAngle> out;
  const int n = d.seqSize(key);
  for (int i = 0; i < n; ++i) {
    const std::string p = key + "." + std::to_string(i);
    d.checkKeys(p, {"joint", "angle"});
    JointAngle ja;
    ja.joint = d.getS(p + ".joint", "");
    if (ja.joint.empty()) throw std::runtime_error(p + ".joint is required");
    ja.angle = d.getD(p + ".angle", 0.0);
    out.push_back(ja);
  }
  return out;
}

std::vector<ContactPlacement> placementsFromYaml(const YamlDoc& d, const std::string& key) {
  std::vector<ContactPlacement> out;
  const int n = d.seqSize(key);
  for (int i = 0; i < n; ++i) {
    const std::string p = key + "." + std::to_string(i);
    d.checkKeys(p, {"contact", "pos"});
    ContactPlacement cp;
    cp.contact = d.getS(p + ".contact", "");
    if (cp.contact.empty()) throw std::runtime_error(p + ".contact is required");
    cp.pos = vec3(d, p + ".pos", "a placement");
    out.push_back(cp);
  }
  return out;
}

}  // namespace

ContactSequenceSpec contactSequenceSpecFromYaml(const YamlDoc& d) {
  d.checkKeys("problem", {"type", "modes", "start", "pose_start", "pose_end", "rest_start", "rest_end", "mu", "friceps", "lammin", "landvz",
                          "copx", "copy", "gammaz", "impactcone", "landramp", "hop", "footclear", "ulim", "vlim", "qbox", "attmax", "ubern",
                          "baseheight", "groundclearance", "mineffort", "planarlock", "selfcolmargin", "selfcolinterp", "selfcolpairs",
                          "noslipnq", "seedapex"});
  ContactSequenceSpec cs;
  const int nModes = d.seqSize("problem.modes");
  if (nModes <= 0) throw std::runtime_error("problem.modes is required: the list of contact modes [{contacts: [...], T: ...}, ...]");
  for (int i = 0; i < nModes; ++i) {
    const std::string p = "problem.modes." + std::to_string(i);
    d.checkKeys(p, {"contacts", "T", "feet", "shift", "label"});
    ContactSequenceMode md;
    md.label = d.getS(p + ".label", "");
    md.contacts = d.getVecS(p + ".contacts");
    md.T = d.requireD(p + ".T");
    md.feet = placementsFromYaml(d, p + ".feet");
    if (d.has(p + ".shift")) md.shift = vec3(d, p + ".shift", "the landing shift");
    cs.modes.push_back(md);
  }
  cs.start = placementsFromYaml(d, "problem.start");
  cs.poseStart = anglesFromYaml(d, "problem.pose_start");
  cs.poseEnd = anglesFromYaml(d, "problem.pose_end");
  cs.restStart = d.getB("problem.rest_start", cs.restStart);
  cs.restEnd = d.getB("problem.rest_end", cs.restEnd);
  cs.frictionMu = d.getD("problem.mu", cs.frictionMu);
  cs.frictionEps = d.getD("problem.friceps", cs.frictionEps);
  cs.lamMin = d.getD("problem.lammin", cs.lamMin);
  cs.landVzMax = d.getD("problem.landvz", cs.landVzMax);
  cs.copX = d.getD("problem.copx", cs.copX);
  cs.copY = d.getD("problem.copy", cs.copY);
  cs.gammaZ = d.getD("problem.gammaz", cs.gammaZ);
  cs.impactCone = d.getB("problem.impactcone", cs.impactCone);
  cs.landRamp = d.getB("problem.landramp", cs.landRamp);
  cs.hop = d.getD("problem.hop", cs.hop);
  cs.footClear = d.getD("problem.footclear", cs.footClear);
  cs.uLimitScale = d.getD("problem.ulim", cs.uLimitScale);
  cs.velLimitScale = d.getD("problem.vlim", cs.velLimitScale);
  cs.qBoxScale = d.getD("problem.qbox", cs.qBoxScale);
  cs.attMax = d.getD("problem.attmax", cs.attMax);
  cs.uBernstein = d.getB("problem.ubern", cs.uBernstein);
  cs.baseHeight = d.getB("problem.baseheight", cs.baseHeight);
  cs.groundClearance = d.getB("problem.groundclearance", cs.groundClearance);
  cs.minimizeEffort = d.getB("problem.mineffort", cs.minimizeEffort);
  cs.planarLock = d.getVecS("problem.planarlock");
  cs.selfColMargin = d.getD("problem.selfcolmargin", cs.selfColMargin);
  cs.selfColInterp = d.getI("problem.selfcolinterp", cs.selfColInterp);
  const int nSelf = d.seqSize("problem.selfcolpairs");
  for (int i = 0; i < nSelf; ++i) {
    const std::string p = "problem.selfcolpairs." + std::to_string(i);
    d.checkKeys(p, {"a", "b", "margin"});
    ContactSequenceSpec::SelfColPair sp;
    sp.a = d.getS(p + ".a", "");
    sp.b = d.getS(p + ".b", "");
    if (sp.a.empty() || sp.b.empty()) throw std::runtime_error(p + " wants capsule names a and b");
    sp.margin = d.getD(p + ".margin", -1.0);
    cs.selfColPairs.push_back(sp);
  }
  cs.stanceNoSlipNq = d.getI("problem.noslipnq", cs.stanceNoSlipNq);
  cs.seedApex = d.getD("problem.seedapex", cs.seedApex);
  d.checkKeys("solver", {"deg", "nc", "split", "hgrade", "lsqnq", "rho", "precond"});
  cs.degree = d.getI("solver.deg", cs.degree);
  cs.Nc = d.getI("solver.nc", cs.Nc);
  cs.splitPerMode = d.getI("solver.split", cs.splitPerMode);
  cs.grade = d.getD("solver.hgrade", cs.grade);
  cs.lsqNq = d.getI("solver.lsqnq", cs.lsqNq);
  cs.lsqRho = d.getD("solver.rho", cs.lsqRho);
  cs.precond = d.getB("solver.precond", cs.precond);
  cs.obstacles = obstaclesFromYaml(d);
  return cs;
}

std::unique_ptr<ProblemFamily> makeContactSequenceFamily(const YamlDoc& d, std::shared_ptr<const RobotModel> model) {
  const ContactSequenceSpec cs = contactSequenceSpecFromYaml(d);
  ContactSequenceBuild b = buildContactSequence(cs, model);
  const ObjectiveSpec objective = objectiveFromYaml(d);
  attachObjective(b.problem, objective, *model);
  std::unique_ptr<StaticFamily> fam = std::make_unique<StaticFamily>(d.getS("name", "unnamed"), b.problem, b.y0, cs.lsqRho, "contact_sequence", b.summary);
  if (objective.present) fam->setObjectiveWeights(objective.quadrature, objective.scale);
  fam->setWarnings(contactSequenceSpecWarnings(cs));
  return fam;
}

}  // namespace leap
