#include "leap/tasks/terrain_walk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

#include "leap/problem/costs.hpp"
#include "leap/problem/mesh_refinement.hpp"
#include "leap/problem/self_collision.hpp"
#include "leap/spectral/bernstein.hpp"
#include "leap/spectral/cgl_grid.hpp"
#include "leap/solver/term_scales.hpp"

namespace leap {

namespace {

constexpr double kLevelTol = 1e-6;

std::vector<int> range(int first, int count) {
  std::vector<int> v(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) v[static_cast<size_t>(i)] = first + i;
  return v;
}

bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
bool containsName(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

int jointIndex(const RobotModel& m, const std::string& name, const char* what) {
  if (!m.hasJoint(name)) {
    std::string all;
    for (const std::string& n : m.jointNames()) all += (all.empty() ? "" : ", ") + n;
    throw std::runtime_error(std::string(what) + ": no joint '" + name + "' in this robot (joints: " + all + ")");
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

// Map an attachment's Where onto sub-element i of k. This is the reference
// implementation's plain map: an AllNodes row lands on BOTH sides of an interior
// junction (the junction quotient identifies the two nodes' q, v, so a q-only row is
// then stated twice there, with its own multiplier and penalty term on each side).
// It is the default (problem.junctionownership=plain); `unique` switches to
// mapWhereUniqueOwner of
// problem/mesh_refinement.hpp, which states such a row once.
std::vector<Where> mapWhere(Where where, int i, int k) {
  if (k <= 1) return {where};
  switch (where) {
    case Where::FirstNode: return i == 0 ? std::vector<Where>{Where::FirstNode} : std::vector<Where>{};
    case Where::LastNode: return i == k - 1 ? std::vector<Where>{Where::LastNode} : std::vector<Where>{};
    case Where::AllNodes: return {Where::AllNodes};
    case Where::Interior:
      if (i == 0) return {Where::Interior, Where::LastNode};
      if (i == k - 1) return {Where::FirstNode, Where::Interior};
      return {Where::AllNodes};
    case Where::OneNode: return {};
  }
  return {where};
}

// Per-contact-set constraint bundle (shared objects: one penalty term per name).
struct SetBundle {
  std::shared_ptr<Constraint> dyn;
  std::vector<std::shared_ptr<Constraint>> feet;     // vel + acc, per contact
  std::vector<std::shared_ptr<Constraint>> feetAcc;
  std::vector<std::shared_ptr<Constraint>> feetVel;
  std::shared_ptr<Constraint> ineqLamZ;              // fz floor, every in-set foot
  std::shared_ptr<Constraint> fric, cop, tz;
  std::map<int, std::shared_ptr<Constraint>> lamZFoot;     // one-row fz floor per foot
  std::map<int, std::shared_ptr<Constraint>> lamZeroFoot;  // dim-row zero pin per foot
};

std::string setKey(const ContactSet& cs) {
  std::string k;
  for (const ContactRef& cr : cs.active) k += std::to_string(cr.index) + ",";
  return k.empty() ? std::string("-") : k;
}

SetBundle makeBundle(const RobotModel& m, const ContactSet& cs, const NodeDims& dimsC, int dim, const TerrainWalkSpec& ws) {
  SetBundle b;
  if (ws.elimU) b.dyn = std::make_shared<FloatingBaseDynamicsConstraint>(m, cs);
  else b.dyn = std::make_shared<DynamicsConstraint>(m, cs);
  std::vector<int> lamZIdx;
  for (const ContactRef& cr : cs.active) {
    const std::string& nm = m.contactName(cr.index);
    auto acc = std::make_shared<FrameAccelerationConstraint>(cr.index, Eigen::VectorXd::Zero(cr.dim), "acc_" + nm);
    auto vel = std::make_shared<FrameVelocityConstraint>(cr.index, Eigen::VectorXd::Zero(cr.dim), "vel_" + nm);
    b.feet.push_back(vel);
    b.feet.push_back(acc);
    b.feetVel.push_back(vel);
    b.feetAcc.push_back(acc);
    const int zIdx = dimsC.offLam() + cs.lamOffset(cr.index) + 2;
    lamZIdx.push_back(zIdx);
    auto floor = std::make_shared<LowerBoundConstraint>(std::vector<int>{zIdx}, Eigen::VectorXd::Constant(1, ws.lamMin), "ineq_lam_z");
    floor->setForceOnly(true);
    b.lamZFoot.emplace(cr.index, floor);
    auto zero = std::make_shared<PinConstraint>(range(dimsC.offLam() + cs.lamOffset(cr.index), cr.dim), Eigen::VectorXd::Zero(cr.dim), "pin_lam_zero");
    zero->setForceOnly(true);
    b.lamZeroFoot.emplace(cr.index, zero);
  }
  if (!lamZIdx.empty()) {
    auto floorAll = std::make_shared<LowerBoundConstraint>(lamZIdx, Eigen::VectorXd::Constant(static_cast<int>(lamZIdx.size()), ws.lamMin), "ineq_lam_z");
    floorAll->setForceOnly(true);
    b.ineqLamZ = floorAll;
  }
  if (ws.frictionMu > 0.0 && !cs.active.empty()) b.fric = std::make_shared<FrictionConeConstraint>(cs, ws.frictionMu, ws.frictionEps, "ineq_fric");
  if (dim == 6 && !cs.active.empty()) {
    b.cop = std::make_shared<ContactWrenchBoxConstraint>(cs, std::vector<int>{4, 3}, std::vector<double>{ws.copX, ws.copY}, "ineq_cop");
    b.tz = std::make_shared<ContactWrenchBoxConstraint>(cs, std::vector<int>{5}, std::vector<double>{ws.gammaZ}, "ineq_tz");
  }
  return b;
}

// --- inverse kinematics of one gait event -------------------------------------------
// Damped least squares on the whole configuration: every named contact on its
// placement (and level for a flat foot), the base level, the planar-locked joints at
// zero, the base at the nominal height above the footholds, the pose near nominal.
struct IkResult {
  Eigen::VectorXd q;
  double posErr = 0.0, tiltErr = 0.0;
};

IkResult ikEvent(const RobotModel& m, ModelData& data, ModelEvalCache& cache, const Eigen::VectorXd& qRef,
                 const std::vector<int>& contacts, const std::vector<Eigen::Vector3d>& targets, double baseZ,
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
  const int rows = 3 * nc + 3 * nWrench + 3 + 1 + static_cast<int>(lockIdx.size()) + nAct;
  const double wPos = 1.0, wRot = 1.0, wAtt = 0.02, wZ = 0.02, wLock = 1.0, wReg = 0.01;

  Eigen::VectorXd q = qRef;
  {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& t : targets) mean += t;
    if (nc > 0) mean /= static_cast<double>(nc);
    q(0) = mean.x();
    q(1) = mean.y();
    q(2) = baseZ;
    for (int i : lockIdx) q(i) = 0.0;
  }
  const Eigen::VectorXd qLo = m.lowerPositionLimit(), qHi = m.upperPositionLimit();
  Eigen::MatrixXd A(rows, nv);
  Eigen::VectorXd r(rows);
  for (int iter = 0; iter < 150; ++iter) {
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
    r(o) = wZ * (q(2) - baseZ);
    A(o, 2) = wZ;
    ++o;
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
    // projected Gauss-Newton: the actuated joints stay inside their URDF box
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

// The nominal pose: the yaml's joint angles with the base dropped so the lowest
// contact frame sits on z = 0.
Eigen::VectorXd nominalPose(const RobotModel& m, ModelData& data, ModelEvalCache& cache, const std::vector<NominalAngle>& poses) {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(m.nq());
  for (const NominalAngle& pa : poses) q(jointIndex(m, pa.joint, "nominal pose")) = pa.angle;
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(m.nv());
  unsigned needs = kFootPos;
  for (int c = 0; c < m.nContacts(); ++c)
    if (m.contactDim(c) == 6) needs |= kFootRot;
  m.evalNode(data, q, zero, zero, needs, cache);
  double zMin = cache.footPos[0](2);
  for (int c = 1; c < m.nContacts(); ++c) zMin = std::min(zMin, cache.footPos[static_cast<size_t>(c)](2));
  q(2) = -zMin;
  return q;
}

std::shared_ptr<Constraint> placePin(const RobotModel& m, int c, const Eigen::Vector3d& p) {
  return std::make_shared<FramePositionConstraint>(c, p, "pin_foot_" + m.contactName(c));
}
std::shared_ptr<Constraint> flatPin(const RobotModel& m, int c) {
  return std::make_shared<FrameOrientationConstraint>(c, Eigen::Vector3d::Zero(), "pin_sole_" + m.contactName(c));
}

enum class ModeKind { Support, Lift, Flight, Approach };
struct ModePlan {
  ModeKind kind;
  int step;
};
struct ForcePlan {
  const SetBundle* b = nullptr;
  std::vector<int> lift, land;
};

// Force feasibility on the control points of one element. `lift` names the contacts
// whose force reaches zero at the end (floor stops one control point short, whole
// block pinned at the last), `land` those whose force starts at zero.
void finalizeForce(Element& el, const SetBundle& b, const std::vector<int>& lift, const std::vector<int>& land, int Nc) {
  std::vector<int> cpAll, cpButLast, cpButFirst, cpMiddle;
  for (int j = 0; j < Nc; ++j) {
    cpAll.push_back(j);
    if (j < Nc - 1) cpButLast.push_back(j);
    if (j > 0) cpButFirst.push_back(j);
    if (j > 0 && j < Nc - 1) cpMiddle.push_back(j);
  }
  if (b.fric) el.addForceControlPoints(b.fric, cpAll);
  if (b.cop) el.addForceControlPoints(b.cop, cpAll);
  if (b.tz) el.addForceControlPoints(b.tz, cpAll);
  if (lift.empty() && land.empty()) {
    if (b.ineqLamZ) el.addForceControlPoints(b.ineqLamZ, cpAll);
    return;
  }
  for (const std::pair<const int, std::shared_ptr<Constraint>>& kv : b.lamZFoot) {
    const bool lifting = contains(lift, kv.first);
    const bool landing = contains(land, kv.first);
    const std::vector<int>& sel = lifting && landing ? cpMiddle : lifting ? cpButLast : landing ? cpButFirst : cpAll;
    el.addForceControlPoints(kv.second, sel);
  }
  for (int idx : lift) el.addForceControlPoints(b.lamZeroFoot.at(idx), {Nc - 1});
  for (int idx : land) el.addForceControlPoints(b.lamZeroFoot.at(idx), {0});
}

double stanceZMean(const std::vector<Eigen::Vector3d>& at, const std::vector<int>& swing) {
  double z = 0.0;
  int n = 0;
  for (int c = 0; c < static_cast<int>(at.size()); ++c) {
    if (contains(swing, c)) continue;
    z += at[static_cast<size_t>(c)].z();
    ++n;
  }
  return n > 0 ? z / n : 0.0;
}

// Bernstein-hull rows of a degree-n nodal polynomial elevated to degree N: each row
// of Elev * E^-1 is one linear functional of the node values.
std::vector<Eigen::VectorXd> bernsteinHullRows(int n, int N) {
  const CglGrid grid(n);
  const Eigen::MatrixXd E = bernsteinBasisAtDegree(n, grid);
  Eigen::MatrixXd W = E.fullPivLu().solve(Eigen::MatrixXd::Identity(n + 1, n + 1));
  for (int m = n; m < N; ++m) {
    Eigen::MatrixXd El = Eigen::MatrixXd::Zero(m + 2, m + 1);
    for (int i = 0; i <= m + 1; ++i) {
      const double a = static_cast<double>(i) / static_cast<double>(m + 1);
      if (i >= 1) El(i, i - 1) += a;
      if (i <= m) El(i, i) += 1.0 - a;
    }
    W = El * W;
  }
  std::vector<Eigen::VectorXd> rows;
  for (int j = 0; j <= N; ++j) rows.push_back(W.row(j).transpose());
  return rows;
}

bool splitElementDescending(const ElementSplit& a, const ElementSplit& b) { return a.element > b.element; }

}  // namespace

// --- geometry helpers ------------------------------------------------------------------

bool terrainKeepOutBox(const TerrainPillar& p, double inset, CuboidObstacle& out) {
  const double top = p.top - inset;
  if (!(top > 1e-9)) return false;
  out.name = p.name;
  out.center = Eigen::Vector3d(p.cx, p.cy, 0.5 * top);
  out.half = Eigen::Vector3d(0.5 * p.lx, 0.5 * p.ly, 0.5 * top);
  out.rot = Eigen::Matrix3d::Identity();
  return true;
}

double terrainHeightAt(const std::vector<TerrainPillar>& pillars, double x, double y, double shrink) {
  double z = 0.0;
  for (const TerrainPillar& p : pillars) {
    if (std::abs(x - p.cx) > 0.5 * p.lx - shrink) continue;
    if (std::abs(y - p.cy) > 0.5 * p.ly - shrink) continue;
    z = std::max(z, p.top);
  }
  return z;
}

// --- validation -------------------------------------------------------------------------

namespace {
std::string checkPlacement(const TerrainWalkSpec& ws, const FootPlacement& f, const std::string& at) {
  const double h = terrainHeightAt(ws.pillars, f.pos.x(), f.pos.y());
  if (std::abs(f.pos.z() - h) > 1e-6)
    return at + "placement z = " + std::to_string(f.pos.z()) + " m is not on the terrain surface under it (" + std::to_string(h) +
           " m). Footholds stand on the floor or on a pillar's top face.";
  return "";
}
}  // namespace

std::string validateTerrainWalkSpec(const TerrainWalkSpec& ws) {
  if (!std::isfinite(ws.selfColMargin) || ws.selfColMargin < 0.0) return "selfcolmargin must be finite and >= 0";
  if (ws.selfColInterp < 0 || ws.selfColInterp > 16) return "selfcolinterp must be in [0, 16]";
  for (const TerrainWalkSpec::SelfColPair& p : ws.selfColPairs) {
    if (p.a.empty() || p.b.empty() || p.a == p.b) return "selfcolpairs entries want two distinct capsule names";
    if (!std::isfinite(p.margin) || (p.margin < 0.0 && p.margin != -1.0)) return "selfcolpairs per-pair margin must be >= 0 (or omitted)";
  }
  if (!std::isfinite(ws.baseZMinFrac)) return "basezfrac must be finite";
  if (!ws.attBox.empty() && ws.attBox.size() != 3) return "attbox wants 3 numbers [roll, pitch, yaw]";
  for (const TerrainWalkSpec::PostureBound& pb : ws.posture) {
    if (pb.joint.empty()) return "posture entries want a joint name";
    if (std::isnan(pb.lo) || std::isnan(pb.hi) || pb.lo > pb.hi) return "posture '" + pb.joint + "': wants lo <= hi";
  }
  if (ws.degree < 2) return "degree must be >= 2";
  if (ws.Nc < 2 || ws.Nc > ws.degree + 1) return "Nc must be in [2, degree+1=" + std::to_string(ws.degree + 1) + "] (got " + std::to_string(ws.Nc) + ")";
  if (ws.splitPerMode < 1) return "splitPerMode must be >= 1";
  if (ws.lsqNq < 0) return "lsqNq must be >= 0";
  if (ws.dynNq < 0 || ws.dynNq > 16) return "dynNq must be in [0, 16]";
  if (ws.attBoxNq < 0 || ws.attBoxNq > 16) return "attBoxNq must be in [0, 16]";
  if (ws.qBoxNq < 0 || ws.qBoxNq > 16) return "qBoxNq must be in [0, 16]";
  if (!(ws.Tstart > 0.0)) return "tstart must be > 0";
  if (ws.steps.empty()) return "steps is empty: a walk needs at least one step";
  if (ws.start.empty()) return "start is empty: every contact needs its initial placement (that IS the initial condition of this problem)";
  if (!(ws.copX > 0.0) || !(ws.copY > 0.0)) return "copx/copy must be > 0";
  if (!(ws.gammaZ > 0.0)) return "gammaz must be > 0";
  if (!std::isfinite(ws.approachFrac) || ws.approachFrac < 0.0 || ws.approachFrac > 0.5) return "approachfrac must be in [0, 0.5]";
  if (!std::isfinite(ws.liftFrac) || ws.liftFrac < 0.0 || ws.liftFrac > 0.5) return "liftfrac must be in [0, 0.5]";
  if (ws.approachFrac + ws.liftFrac >= 1.0) return "approachfrac + liftfrac must be < 1";
  for (const TerrainWalkSpec::ULimitJoint& uj : ws.uLimitJoint) {
    if (uj.joint.empty()) return "ulimjoint entries want a joint name";
    if (!std::isfinite(uj.scale) || uj.scale <= 0.0) return "ulimjoint '" + uj.joint + "': scale must be > 0";
  }
  for (size_t i = 0; i < ws.steps.size(); ++i) {
    const WalkStep& s = ws.steps[i];
    const std::string at = "step " + std::to_string(i + 1) + ": ";
    if (!(s.Tswing > 0.0)) return at + "tswing must be > 0";
    if (!(s.Tsupport > 0.0)) return at + "tsupport must be > 0";
    if (s.feet.empty()) return at + "no swing feet (a step moves at least one foot)";
  }
  for (const TerrainPillar& p : ws.pillars) {
    const std::string at = "pillar '" + p.name + "': ";
    if (!(p.lx > 0.0) || !(p.ly > 0.0)) return at + "size must be > 0";
    if (!(p.top > 0.0)) return at + "top must be > 0 (a pillar level with the floor is not terrain)";
  }
  if (ws.terrainTopSlack < 0.0) return "terrain topslack must be >= 0";
  if (ws.terrainTopInset < 0.0) return "terrain topinset must be >= 0";
  if (ws.terrainMargin < 0.0) return "terrain margin must be >= 0";
  for (const FootPlacement& f : ws.start) {
    const std::string e = checkPlacement(ws, f, "start '" + f.contact + "': ");
    if (!e.empty()) return e;
  }
  for (size_t i = 0; i < ws.steps.size(); ++i)
    for (const FootPlacement& f : ws.steps[i].feet) {
      const std::string e = checkPlacement(ws, f, "step " + std::to_string(i + 1) + " '" + f.contact + "': ");
      if (!e.empty()) return e;
    }
  return "";
}

std::vector<std::string> terrainWalkSpecWarnings(const TerrainWalkSpec& ws) {
  std::vector<std::string> w;
  if (ws.pillars.empty()) w.push_back("no pillars declared: this is a walk on flat ground, not on terrain");
  if (ws.terrainInterp == 0 && !ws.pillars.empty()) w.push_back("terrain interp = 0: the keep-out is enforced at the NODES only, so the polynomial may cut a step's corner between them");
  if (ws.velLimitScale <= 0.0) w.push_back("vlim <= 0: joint speed limits are NOT enforced");
  if (ws.uLimitScale <= 0.0) w.push_back("ulim <= 0: torque limits are NOT enforced");
  std::map<std::string, Eigen::Vector3d> at;
  for (const FootPlacement& f : ws.start) at[f.contact] = f.pos;
  double worstRise = 0.0, worstSpeed = 0.0;
  for (size_t i = 0; i < ws.steps.size(); ++i) {
    const WalkStep& s = ws.steps[i];
    for (const FootPlacement& f : s.feet) {
      std::map<std::string, Eigen::Vector3d>::iterator it = at.find(f.contact);
      if (it == at.end()) {
        w.push_back("step " + std::to_string(i + 1) + ": '" + f.contact + "' has no earlier placement");
        at[f.contact] = f.pos;
        continue;
      }
      const double travel = (f.pos - it->second).head(2).norm();
      worstSpeed = std::max(worstSpeed, travel / s.Tswing);
      worstRise = std::max(worstRise, std::abs(f.pos.z() - it->second.z()));
      if (f.pos.z() > 1e-9 && std::abs(terrainHeightAt(ws.pillars, f.pos.x(), f.pos.y(), 0.03) - f.pos.z()) > 1e-9)
        w.push_back("step " + std::to_string(i + 1) + " '" + f.contact + "': foothold is within 3 cm of the pillar edge");
      it->second = f.pos;
    }
  }
  if (worstSpeed > 1.5) w.push_back("fastest swing travels " + std::to_string(worstSpeed) + " m/s of foot speed");
  if (worstRise > 0.0) w.push_back("largest step height change " + std::to_string(worstRise) + " m");
  return w;
}

// --- the builder --------------------------------------------------------------------------

TerrainWalkBuild buildTerrainWalk(const TerrainWalkSpec& ws, std::shared_ptr<const RobotModel> modelPtr) {
  {
    const std::string err = validateTerrainWalkSpec(ws);
    if (!err.empty()) throw std::invalid_argument("terrain walk spec: " + err);
  }
  TerrainWalkBuild out;
  const RobotModel& m = *modelPtr;
  out.problem.model = modelPtr;
  out.problem.preconditionForce = ws.precond;
  const int nSteps = static_cast<int>(ws.steps.size());

  // --- what the model must look like ---
  if (m.nContacts() < 1) throw std::runtime_error("terrain walk: the robot yaml declares no contacts");
  const int dim = m.contactDim(0);
  if (dim != 3 && dim != 6) throw std::runtime_error("terrain walk: contact dim must be 3 (point force) or 6 (flat-foot wrench), got " + std::to_string(dim));
  for (int c = 1; c < m.nContacts(); ++c)
    if (m.contactDim(c) != dim)
      throw std::runtime_error("terrain walk: all contacts must share one dim (contact '" + m.contactName(c) + "' is dim " + std::to_string(m.contactDim(c)) + ")");
  {
    const std::vector<std::string>& jn = m.jointNames();
    const char* want[6] = {"base_px", "base_py", "base_pz", "base_rx", "base_ry", "base_rz"};
    if (jn.size() < 6) throw std::runtime_error("terrain walk: not a floating-base robot");
    for (int i = 0; i < 6; ++i)
      if (jn[static_cast<size_t>(i)] != want[i])
        throw std::runtime_error(std::string("terrain walk: joint ") + std::to_string(i) + " is '" + jn[static_cast<size_t>(i)] + "', expected '" + want[i] + "'");
  }

  // --- the gait, resolved against the contact table ---
  std::vector<Eigen::Vector3d> placedAt(static_cast<size_t>(m.nContacts()));
  std::vector<bool> placed(static_cast<size_t>(m.nContacts()), false);
  for (const FootPlacement& f : ws.start) {
    const int c = contactIndexOf(m, f.contact, "start placement");
    if (placed[static_cast<size_t>(c)]) throw std::runtime_error("terrain walk: contact '" + f.contact + "' is placed twice in `start`");
    placedAt[static_cast<size_t>(c)] = f.pos;
    placed[static_cast<size_t>(c)] = true;
  }
  for (int c = 0; c < m.nContacts(); ++c)
    if (!placed[static_cast<size_t>(c)])
      throw std::runtime_error("terrain walk: contact '" + m.contactName(c) + "' has no `start` placement. The initial stance of EVERY contact is a boundary condition of this problem.");
  std::vector<std::vector<int>> swingIdx(static_cast<size_t>(nSteps));
  std::vector<std::vector<Eigen::Vector3d>> swingTo(static_cast<size_t>(nSteps));
  std::vector<std::vector<double>> swingClearZ(static_cast<size_t>(nSteps));
  std::vector<std::vector<Eigen::Vector3d>> stanceAfter(static_cast<size_t>(nSteps) + 1);
  stanceAfter[0] = placedAt;
  for (int k = 0; k < nSteps; ++k) {
    const WalkStep& s = ws.steps[static_cast<size_t>(k)];
    std::vector<bool> moved(static_cast<size_t>(m.nContacts()), false);
    for (const FootPlacement& f : s.feet) {
      const int c = contactIndexOf(m, f.contact, "step " + std::to_string(k + 1));
      if (moved[static_cast<size_t>(c)]) throw std::runtime_error("terrain walk: step " + std::to_string(k + 1) + " moves contact '" + f.contact + "' twice");
      moved[static_cast<size_t>(c)] = true;
      swingIdx[static_cast<size_t>(k)].push_back(c);
      swingTo[static_cast<size_t>(k)].push_back(f.pos);
      swingClearZ[static_cast<size_t>(k)].push_back(f.clearZ * ws.clearZScale);
      placedAt[static_cast<size_t>(c)] = f.pos;
    }
    stanceAfter[static_cast<size_t>(k) + 1] = placedAt;
  }
  const ContactSet all = m.allContacts();
  const ContactSet none;
  const NodeDims dimsC = m.nodeDims(all);
  const int impactAll = m.ensureImpactSet(all);
  std::vector<ContactSet> support(static_cast<size_t>(nSteps));
  for (int k = 0; k < nSteps; ++k) {
    std::vector<std::string> names;
    for (int c = 0; c < m.nContacts(); ++c)
      if (!contains(swingIdx[static_cast<size_t>(k)], c)) names.push_back(m.contactName(c));
    support[static_cast<size_t>(k)] = names.empty() ? none : m.contactSet(names);
  }
  ModelDataPtr data = m.makeData();
  ModelEvalCache cache;
  m.initCache(cache);
  out.qNominal = nominalPose(m, *data, cache, ws.nominal);
  const double zNominal = out.qNominal(2);

  // --- constraint objects ---
  std::map<std::string, SetBundle> bundles;
  bundles.emplace(setKey(all), makeBundle(m, all, dimsC, dim, ws));
  for (int k = 0; k < nSteps; ++k) {
    const ContactSet& cs = support[static_cast<size_t>(k)];
    if (bundles.find(setKey(cs)) == bundles.end()) bundles.emplace(setKey(cs), makeBundle(m, cs, dimsC, dim, ws));
  }
  const SetBundle& bAll = bundles.at(setKey(all));
  auto clearGround = std::make_shared<MonitorHeightLowerBoundConstraint>(m, Eigen::VectorXd::Constant(m.nMonitors(), ws.monitorZ), "ineq_clear");
  double zBaseLo = 0.0;
  if (ws.baseZMinFrac > 0.0) {
    double zFoot = ws.start.front().pos.z();
    for (const FootPlacement& f : ws.start) zFoot = std::min(zFoot, f.pos.z());
    zBaseLo = zFoot + ws.baseZMinFrac * zNominal;
  }
  auto ineqBaseZ = std::make_shared<LowerBoundConstraint>(std::vector<int>{2}, Eigen::VectorXd::Constant(1, zBaseLo), "ineq_base_z");
  auto pinV0 = std::make_shared<PinConstraint>(range(dimsC.offV(), dimsC.nv), Eigen::VectorXd::Zero(dimsC.nv), "pin_v");
  auto effort = std::make_shared<ControlEffortCost>();
  std::shared_ptr<Cost> velCost;
  if (ws.wvel > 0.0) {
    const double u0 = m.actuatedEffortLimit().cwiseAbs().mean();
    const double v0 = std::max(m.actuatedVelocityLimit().cwiseAbs().mean(), 1e-6);
    const double w = ws.wvel * (u0 / v0) * (u0 / v0);
    velCost = std::make_shared<JointVelocityCost>(m.actuatedIdx(), Eigen::VectorXd::Constant(m.nu(), w));
  }
  const std::vector<double> lsqTaus = ws.lsqNq > 0 ? uniformInteriorTaus(ws.lsqNq) : std::vector<double>{};
  const std::vector<double> noSlipTaus = ws.stanceNoSlipNq > 0 ? uniformInteriorTaus(ws.stanceNoSlipNq) : std::vector<double>{};
  const std::vector<double> dynTaus = ws.dynNq > 0 ? uniformInteriorTaus(ws.dynNq) : std::vector<double>{};

  // --- the parent modes ---
  std::vector<Element> modes;
  std::vector<ModePlan> plan;
  std::vector<ForcePlan> forceOf;
  // a support phase: `start` at rest on the placements; `restEnd` at rest at its end
  for (int evt = 0; evt <= nSteps; ++evt) (void)evt;
  // helpers written out per mode kind (no lambdas): the skeleton shared by every mode
  struct ModeBuilder {
    const RobotModel& m;
    const TerrainWalkSpec& ws;
    const std::vector<ContactSet>& support;
    const std::vector<std::vector<Eigen::Vector3d>>& stanceAfter;
    const std::vector<std::vector<int>>& swingIdx;
    const std::vector<std::vector<Eigen::Vector3d>>& swingTo;
    const std::vector<std::vector<double>>& swingClearZ;
    const std::map<std::string, SetBundle>& bundles;
    const ContactSet& all;
    const SetBundle& bAll;
    std::shared_ptr<Constraint> clearGround, ineqBaseZ, pinV0;
    std::shared_ptr<Cost> effort, velCost;
    const std::vector<double>& lsqTaus;
    const std::vector<double>& noSlipTaus;
    const std::vector<double>& dynTaus;
    int dim;

    void addCosts(Element& el) const {
      el.addCost(effort);
      if (velCost) el.addCost(velCost);
    }
    void addLsq(Element& el, const std::shared_ptr<Constraint>& dyn, int modeBit) const {
      if ((ws.lsqModeMask & modeBit) == 0) return;
      if (lsqTaus.empty() || ws.lsqRho <= 0.0) return;
      el.setDefectCost(dyn, lsqTaus, ws.lsqRho);
    }
    void addDynInterp(Element& el, const std::shared_ptr<Constraint>& dyn) const {
      if (!dynTaus.empty()) el.addOffNode(dyn, dynTaus);
    }
    void landingRows(Element& el, int c, const Eigen::Vector3d& to) const {
      el.addNode(placePin(m, c, to), Where::LastNode);
      if (m.contactDim(c) == 6) el.addNode(flatPin(m, c), Where::LastNode);
      if (ws.zeroLand) {
        el.addNode(std::make_shared<FrameVelocityConstraint>(c, Eigen::VectorXd::Zero(m.contactDim(c)), "pin_land_v_" + m.contactName(c)), Where::LastNode);
      } else if (ws.zeroLandXY) {
        auto vpin = std::make_shared<FrameVelocityConstraint>(c, Eigen::VectorXd::Zero(m.contactDim(c)), "pin_land_vxy_" + m.contactName(c));
        el.addNode(std::make_shared<RowSliceConstraint>(vpin, 0), Where::LastNode);
        el.addNode(std::make_shared<RowSliceConstraint>(vpin, 1), Where::LastNode);
        if (ws.landVzMax > 0.0) el.addNode(softLand(c), Where::LastNode);
      } else if (ws.landVzMax > 0.0) {
        el.addNode(softLand(c), Where::LastNode);
      }
    }
    std::shared_ptr<Constraint> softLand(int c) const {
      return std::make_shared<FrameVelocityBoundConstraint>(c, 2, -1.0, -ws.landVzMax, "ineq_land_vz_" + m.contactName(c), dim);
    }
    void hoverRows(Element& el, int c, const Eigen::Vector3d& p, const std::vector<Where>& where) const {
      std::shared_ptr<Constraint> pin = placePin(m, c, p);
      auto zlb = std::make_shared<ContactHeightLowerBoundConstraint>(c, p.z(), "ineq_hover_z_" + m.contactName(c));
      for (Where w : where) {
        el.addNode(std::make_shared<RowSliceConstraint>(pin, 0), w);
        el.addNode(std::make_shared<RowSliceConstraint>(pin, 1), w);
        if (m.contactDim(c) == 6) el.addNode(flatPin(m, c), w);
        el.addNode(zlb, w);
      }
    }
    Element supportMode(double T, bool start, bool restEnd, const std::vector<Eigen::Vector3d>& at) const {
      Element el;
      el.degree = ws.degree;
      el.T = T;
      el.contacts = all;
      el.forceControlPoints = ws.Nc;
      el.deriveTorque = ws.elimU;
      el.addNode(clearGround, Where::AllNodes);
      if (ws.stancePin) {
        for (int c = 0; c < m.nContacts(); ++c) {
          el.addNode(placePin(m, c, at[static_cast<size_t>(c)]), Where::AllNodes);
          if (m.contactDim(c) == 6) el.addNode(flatPin(m, c), Where::AllNodes);
        }
      }
      el.addNode(bAll.dyn, Where::FirstNode);
      if (start) {
        if (!ws.stancePin) {
          for (int c = 0; c < m.nContacts(); ++c) {
            el.addNode(placePin(m, c, at[static_cast<size_t>(c)]), Where::FirstNode);
            if (m.contactDim(c) == 6) el.addNode(flatPin(m, c), Where::FirstNode);
          }
        }
        el.addNode(pinV0, Where::FirstNode);
        for (const std::shared_ptr<Constraint>& c : bAll.feetAcc) el.addNode(c, Where::FirstNode);
      } else {
        for (const std::shared_ptr<Constraint>& c : bAll.feet) el.addNode(c, Where::FirstNode);
      }
      el.addNode(ineqBaseZ, Where::FirstNode);
      el.addNode(bAll.dyn, Where::Interior);
      for (const std::shared_ptr<Constraint>& c : bAll.feet) el.addNode(c, Where::Interior);
      el.addNode(ineqBaseZ, Where::Interior);
      el.addNode(bAll.dyn, Where::LastNode);
      el.addNode(ineqBaseZ, Where::LastNode);
      if (restEnd) {
        el.addNode(pinV0, Where::LastNode);
        for (const std::shared_ptr<Constraint>& c : bAll.feetAcc) el.addNode(c, Where::LastNode);
      } else {
        for (const std::shared_ptr<Constraint>& c : bAll.feet) el.addNode(c, Where::LastNode);
      }
      addCosts(el);
      for (const std::shared_ptr<Constraint>& v : bAll.feetVel) el.addOffNode(v, noSlipTaus);
      if (ws.noSlipAcc)
        for (const std::shared_ptr<Constraint>& a : bAll.feetAcc) el.addOffNode(a, noSlipTaus);
      addLsq(el, bAll.dyn, 1);
      addDynInterp(el, bAll.dyn);
      return el;
    }
    Element swingHead(int k, double T) const {
      const ContactSet& cs = support[static_cast<size_t>(k)];
      const SetBundle& b = bundles.at(setKey(cs));
      Element el;
      el.degree = ws.degree;
      el.T = T;
      el.contacts = cs;
      el.forceControlPoints = cs.active.empty() ? 0 : ws.Nc;
      el.deriveTorque = ws.elimU;
      el.addNode(clearGround, Where::AllNodes);
      if (ws.stancePin) {
        const std::vector<Eigen::Vector3d>& at = stanceAfter[static_cast<size_t>(k)];
        for (const ContactRef& cr : cs.active) {
          el.addNode(placePin(m, cr.index, at[static_cast<size_t>(cr.index)]), Where::AllNodes);
          if (m.contactDim(cr.index) == 6) el.addNode(flatPin(m, cr.index), Where::AllNodes);
        }
      }
      el.addNode(b.dyn, Where::FirstNode);
      for (const std::shared_ptr<Constraint>& c : b.feet) el.addNode(c, Where::FirstNode);
      el.addNode(ineqBaseZ, Where::FirstNode);
      el.addNode(b.dyn, Where::Interior);
      for (const std::shared_ptr<Constraint>& c : b.feet) el.addNode(c, Where::Interior);
      el.addNode(ineqBaseZ, Where::Interior);
      el.addNode(b.dyn, Where::LastNode);
      el.addNode(ineqBaseZ, Where::LastNode);
      return el;
    }
    void swingTail(Element& el, int k) const {
      const ContactSet& cs = support[static_cast<size_t>(k)];
      const SetBundle& b = bundles.at(setKey(cs));
      addCosts(el);
      for (const std::shared_ptr<Constraint>& v : b.feetVel) el.addOffNode(v, noSlipTaus);
      if (ws.noSlipAcc)
        for (const std::shared_ptr<Constraint>& a : b.feetAcc) el.addOffNode(a, noSlipTaus);
      addLsq(el, b.dyn, cs.active.empty() ? 4 : 2);
      addDynInterp(el, b.dyn);
    }
    Element swingMode(int k, bool hasApproach, bool hasLift) const {
      const double Tsw = ws.steps[static_cast<size_t>(k)].Tswing;
      Element el = swingHead(k, Tsw * (1.0 - ws.approachFrac - ws.liftFrac));
      (void)hasLift;
      for (size_t j = 0; j < swingIdx[static_cast<size_t>(k)].size(); ++j) {
        const int c = swingIdx[static_cast<size_t>(k)][j];
        if (!hasApproach) landingRows(el, c, swingTo[static_cast<size_t>(k)][j]);
        const double cz = swingClearZ[static_cast<size_t>(k)][j];
        if (cz >= 0.0) el.addOffNode(std::make_shared<ContactHeightLowerBoundConstraint>(c, cz, "ineq_swing_z"), {-0.5, 0.0, 0.5});
      }
      swingTail(el, k);
      return el;
    }
    Element approachMode(int k) const {
      Element el = swingHead(k, ws.steps[static_cast<size_t>(k)].Tswing * ws.approachFrac);
      for (size_t j = 0; j < swingIdx[static_cast<size_t>(k)].size(); ++j) {
        const int c = swingIdx[static_cast<size_t>(k)][j];
        const Eigen::Vector3d& to = swingTo[static_cast<size_t>(k)][j];
        hoverRows(el, c, to, {Where::FirstNode, Where::Interior});
        landingRows(el, c, to);
      }
      swingTail(el, k);
      return el;
    }
    Element liftMode(int k) const {
      Element el = swingHead(k, ws.steps[static_cast<size_t>(k)].Tswing * ws.liftFrac);
      for (size_t j = 0; j < swingIdx[static_cast<size_t>(k)].size(); ++j) {
        const int c = swingIdx[static_cast<size_t>(k)][j];
        hoverRows(el, c, stanceAfter[static_cast<size_t>(k)][static_cast<size_t>(c)], {Where::Interior, Where::LastNode});
      }
      swingTail(el, k);
      return el;
    }
  };
  const ModeBuilder mb{m, ws, support, stanceAfter, swingIdx, swingTo, swingClearZ, bundles, all, bAll, clearGround, ineqBaseZ, pinV0,
                       effort, velCost, lsqTaus, noSlipTaus, dynTaus, dim};
  const bool hasApproach = ws.approachFrac > 0.0;
  const bool hasLift = ws.liftFrac > 0.0;
  std::vector<std::shared_ptr<ResetMap>> resets;
  modes.push_back(mb.supportMode(ws.Tstart, true, false, stanceAfter[0]));
  modes.back().label = "support_0";
  plan.push_back({ModeKind::Support, 0});
  forceOf.push_back({&bAll, swingIdx[0], {}});
  for (int k = 0; k < nSteps; ++k) {
    const SetBundle* bSwing = &bundles.at(setKey(support[static_cast<size_t>(k)]));
    resets.push_back(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
    if (hasLift) {
      modes.push_back(mb.liftMode(k));
      modes.back().label = "lift_" + std::to_string(k + 1);
      plan.push_back({ModeKind::Lift, k});
      forceOf.push_back({bSwing, {}, {}});
      resets.push_back(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
    }
    modes.push_back(mb.swingMode(k, hasApproach, hasLift));
    modes.back().label = "swing_" + std::to_string(k + 1);
    plan.push_back({ModeKind::Flight, k});
    forceOf.push_back({bSwing, {}, {}});
    if (hasApproach) {
      resets.push_back(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
      modes.push_back(mb.approachMode(k));
      modes.back().label = "approach_" + std::to_string(k + 1);
      plan.push_back({ModeKind::Approach, k});
      forceOf.push_back({bSwing, {}, {}});
    }
    if (ws.zeroLand && ws.landCont) resets.push_back(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
    else resets.push_back(std::make_shared<PlasticImpactReset>(impactAll, dimsC.nq, dimsC.nv));
    const bool last = k == nSteps - 1;
    modes.push_back(mb.supportMode(ws.steps[static_cast<size_t>(k)].Tsupport, false, last, stanceAfter[static_cast<size_t>(k) + 1]));
    modes.back().label = "support_" + std::to_string(k + 1);
    plan.push_back({ModeKind::Support, k + 1});
    forceOf.push_back({&bAll, last ? std::vector<int>{} : swingIdx[static_cast<size_t>(k) + 1], ws.landRamp ? swingIdx[static_cast<size_t>(k)] : std::vector<int>{}});
  }
  for (size_t s = 0; s < modes.size(); ++s) modes[s].phase = static_cast<int>(s);

  // --- hardware limit boxes from the URDF ---
  {
    const std::vector<int>& act = m.actuatedIdx();
    const int nu = m.nu();
    const std::vector<double> qTaus = ws.qBoxNq > 0 ? uniformInteriorTaus(ws.qBoxNq) : std::vector<double>{};
    if (ws.qBoxScale > 0.0) {
      const Eigen::VectorXd lo = m.lowerPositionLimit(), hi = m.upperPositionLimit();
      Eigen::VectorXd qlb(nu), qub(nu);
      for (int j = 0; j < nu; ++j) {
        const int i = act[static_cast<size_t>(j)];
        const double mid = 0.5 * (lo(i) + hi(i)), half = 0.5 * (hi(i) - lo(i));
        qlb(j) = mid - ws.qBoxScale * half;
        qub(j) = mid + ws.qBoxScale * half;
      }
      auto boxQ = std::make_shared<BoxConstraint>(Field::Q, act, qlb, qub, "box_q");
      for (Element& el : modes) {
        el.addNode(boxQ, Where::AllNodes);
        if (!qTaus.empty()) el.addOffNode(boxQ, qTaus);
      }
    }
    if (!ws.posture.empty()) {
      const int n = static_cast<int>(ws.posture.size());
      std::vector<int> idx;
      Eigen::VectorXd lo(n), hi(n);
      for (int j = 0; j < n; ++j) {
        const TerrainWalkSpec::PostureBound& pb = ws.posture[static_cast<size_t>(j)];
        idx.push_back(jointIndex(m, pb.joint, "posture"));
        lo(j) = pb.lo;
        hi(j) = pb.hi;
      }
      auto boxP = std::make_shared<BoxConstraint>(Field::Q, idx, lo, hi, "box_posture");
      for (Element& el : modes) {
        el.addNode(boxP, Where::AllNodes);
        if (!qTaus.empty()) el.addOffNode(boxP, qTaus);
      }
    }
    if (ws.baseZMinFrac > 0.0 && !qTaus.empty())
      for (Element& el : modes) el.addOffNode(ineqBaseZ, qTaus);
    const std::vector<double> vTaus = ws.vBoxNq > 0 ? uniformInteriorTaus(ws.vBoxNq) : std::vector<double>{};
    if (ws.velLimitScale > 0.0) {
      const Eigen::VectorXd vlim = ws.velLimitScale * m.actuatedVelocityLimit();
      auto boxV = std::make_shared<BoxConstraint>(Field::V, act, -vlim, vlim, "box_v");
      for (Element& el : modes) {
        el.addNode(boxV, Where::AllNodes);
        if (!vTaus.empty()) el.addOffNode(boxV, vTaus);
      }
    }
    if (!ws.baseVBox.empty()) {
      const std::vector<std::string>& jn = m.jointNames();
      std::vector<int> comps;
      std::vector<double> lims;
      for (int i = 0; i < 3 && i < static_cast<int>(ws.baseVBox.size()); ++i) {
        const double lim = ws.baseVBox[static_cast<size_t>(i)];
        if (!(lim > 0.0)) continue;
        if (containsName(ws.planarLock, jn[static_cast<size_t>(i)])) continue;
        comps.push_back(i);
        lims.push_back(lim);
      }
      if (!comps.empty()) {
        Eigen::VectorXd ub(static_cast<int>(lims.size()));
        for (int j = 0; j < ub.size(); ++j) ub(j) = lims[static_cast<size_t>(j)];
        auto boxVB = std::make_shared<BoxConstraint>(Field::V, comps, -ub, ub, "box_v_base");
        for (Element& el : modes) {
          el.addNode(boxVB, Where::AllNodes);
          if (!vTaus.empty()) el.addOffNode(boxVB, vTaus);
        }
      }
    }
    if (ws.uLimitScale > 0.0) {
      Eigen::VectorXd ulim = ws.uLimitScale * m.actuatedEffortLimit();
      for (const TerrainWalkSpec::ULimitJoint& uj : ws.uLimitJoint) {
        const int qi = jointIndex(m, uj.joint, "ulimjoint");
        const std::vector<int>::const_iterator it = std::find(act.begin(), act.end(), qi);
        if (it == act.end()) throw std::invalid_argument("terrain ulimjoint: '" + uj.joint + "' is not an actuated joint");
        ulim(static_cast<int>(it - act.begin())) *= uj.scale;
      }
      auto boxU = std::make_shared<BoxConstraint>(Field::U, range(0, nu), -ulim, ulim, "box_u");
      const std::vector<double> uTaus = ws.elimU ? (ws.elimUNq > 0 ? uniformInteriorTaus(ws.elimUNq) : std::vector<double>{})
                                                 : (ws.uBoxNq > 0 ? uniformInteriorTaus(ws.uBoxNq) : std::vector<double>{});
      for (Element& el : modes) {
        el.addNode(boxU, Where::AllNodes);
        if (!uTaus.empty()) el.addOffNode(boxU, uTaus);
      }
      if (!ws.elimU && ws.uHullDeg > 0) {
        // Bernstein-hull torque box: the degree-n nodal torque polynomial's Bernstein
        // coefficients, elevated to degree N, boxed (convex hull => the whole curve)
        const std::vector<Eigen::VectorXd> hullRows = bernsteinHullRows(ws.degree, std::max(ws.uHullDeg, ws.degree));
        auto boxUHull = std::make_shared<BoxConstraint>(Field::U, range(0, nu), -ulim, ulim, "box_u_hull");
        for (Element& el : modes) el.addFieldFunctionals(boxUHull, Field::U, hullRows);
      }
    }
    if (ws.attMax > 0.0 || !ws.attBox.empty()) {
      const std::vector<std::string>& jn = m.jointNames();
      std::vector<int> att;
      std::vector<double> attLim;
      for (int i = 3; i < 6; ++i) {
        const double lim = ws.attBox.empty() ? ws.attMax : ws.attBox[static_cast<size_t>(i - 3)];
        if (!(lim > 0.0)) continue;
        if (!containsName(ws.planarLock, jn[static_cast<size_t>(i)])) {
          att.push_back(i);
          attLim.push_back(lim);
        }
      }
      if (!att.empty()) {
        const int n = static_cast<int>(att.size());
        Eigen::VectorXd attUb(n);
        for (int j = 0; j < n; ++j) attUb(j) = attLim[static_cast<size_t>(j)];
        auto boxAtt = std::make_shared<BoxConstraint>(Field::Q, att, -attUb, attUb, "box_att");
        const std::vector<double> attTaus = ws.attBoxNq > 0 ? uniformInteriorTaus(ws.attBoxNq) : std::vector<double>{};
        for (Element& el : modes) {
          el.addNode(boxAtt, Where::AllNodes);
          if (!attTaus.empty()) el.addOffNode(boxAtt, attTaus);
        }
      }
    }
    const MotorSaturationConfig& sat = m.description().saturation;
    if (sat.enabled && sat.enforceInOptimization) {
      auto env = std::make_shared<MotorEnvelopeConstraint>(m, sat, "ineq_motor_envelope");
      const std::vector<double> motorTaus = ws.elimU ? (ws.elimUNq > 0 ? uniformInteriorTaus(ws.elimUNq) : std::vector<double>{}) : uniformInteriorTaus(4);
      for (Element& el : modes) {
        el.addNode(env, Where::AllNodes);
        if (!motorTaus.empty()) el.addOffNode(env, motorTaus);
      }
    }
  }
  if (!ws.planarLock.empty()) {
    std::vector<int> idx;
    for (const std::string& n : ws.planarLock) idx.push_back(jointIndex(m, n, "planarlock"));
    auto pinPlanar = std::make_shared<PinConstraint>(idx, Eigen::VectorXd::Zero(static_cast<int>(idx.size())), "pin_planar");
    for (Element& el : modes) el.addNode(pinPlanar, Where::AllNodes);
  }

  // --- the gait events (IK), needed before the obstacle window ---
  std::vector<int> lockIdx;
  for (const std::string& n : ws.planarLock) lockIdx.push_back(jointIndex(m, n, "planarlock"));
  const std::vector<int> allIdx = range(0, m.nContacts());
  out.qEvent.resize(static_cast<size_t>(nSteps) + 1);
  double worstTilt = 0.0;
  for (int e = 0; e <= nSteps; ++e) {
    const std::vector<Eigen::Vector3d>& at = stanceAfter[static_cast<size_t>(e)];
    double zMean = 0.0;
    for (const Eigen::Vector3d& p : at) zMean += p.z();
    zMean /= static_cast<double>(at.size());
    const IkResult ik = ikEvent(m, *data, cache, out.qNominal, allIdx, at, zMean + zNominal, lockIdx);
    out.qEvent[static_cast<size_t>(e)] = ik.q;
    out.ikWorstErr = std::max(out.ikWorstErr, ik.posErr);
    worstTilt = std::max(worstTilt, ik.tiltErr);
  }
  if (out.ikWorstErr > 0.05)
    throw std::runtime_error("terrain walk: the commanded placements are not simultaneously reachable -- inverse kinematics leaves " +
                             std::to_string(out.ikWorstErr) + " m of foot position error at one of the " + std::to_string(nSteps + 1) +
                             " gait events. Shorten the step, reduce the terrain height range, or deepen the nominal pose.");

  // --- terrain keep-out: one cuboid row per mode over that mode's window of pillars ---
  std::vector<int> capIdx;
  for (const std::string& nm : ws.terrainCapsules) capIdx.push_back(m.capsuleIndex(nm));
  if (capIdx.empty()) capIdx = range(0, m.nCapsules());
  std::string terrainNote;
  if (!ws.pillars.empty()) {
    if (m.nCapsules() == 0) throw std::runtime_error("terrain walk: the robot declares no collision_capsules, so the pillars cannot be made into obstacles");
    double dsafeMax = 0.0;
    for (int ci : capIdx) dsafeMax = std::max(dsafeMax, m.capsuleRadius(ci) + ws.terrainMargin);
    out.topInset = ws.terrainTopInset > 0.0 ? ws.terrainTopInset : dsafeMax + ws.terrainTopSlack;
    std::vector<CuboidObstacle> boxes;
    for (const TerrainPillar& p : ws.pillars) {
      CuboidObstacle b;
      if (!terrainKeepOutBox(p, out.topInset, b)) {
        ++out.pillarsDropped;
        continue;
      }
      if (ws.obstacleZScale != 1.0) {
        b.center.z() *= ws.obstacleZScale;
        b.half.z() *= ws.obstacleZScale;
      }
      boxes.push_back(b);
    }
    double reach = 0.0;
    for (const Eigen::VectorXd& q : out.qEvent) {
      std::vector<Eigen::Vector3d> cp;
      std::vector<Eigen::MatrixXd> cj;
      m.capsuleKinematics(*data, q, cp, cj);
      for (int ci : capIdx)
        for (int e = 0; e < 2; ++e) reach = std::max(reach, std::abs(cp[static_cast<size_t>(2 * ci + e)].x() - q(0)));
    }
    std::vector<std::pair<double, double>> modeX;
    for (const ModePlan& mp : plan) {
      if (mp.kind == ModeKind::Support) {
        const double x = out.qEvent[static_cast<size_t>(mp.step)](0);
        modeX.emplace_back(x, x);
      } else {
        const double x0 = out.qEvent[static_cast<size_t>(mp.step)](0), x1 = out.qEvent[static_cast<size_t>(mp.step) + 1](0);
        modeX.emplace_back(std::min(x0, x1), std::max(x0, x1));
      }
    }
    std::map<std::string, std::shared_ptr<CapsuleCuboidConstraint>> rowCache;
    int minBoxes = static_cast<int>(boxes.size()), maxBoxes = 0;
    for (size_t s = 0; s < modes.size(); ++s) {
      std::vector<CuboidObstacle> sub;
      std::string key;
      for (size_t j = 0; j < boxes.size(); ++j) {
        if (ws.windowPad > 0.0) {
          const double lo = modeX[s].first - reach - ws.windowPad;
          const double hi = modeX[s].second + reach + ws.windowPad;
          const double bl = boxes[j].center.x() - boxes[j].half.x() - dsafeMax;
          const double bh = boxes[j].center.x() + boxes[j].half.x() + dsafeMax;
          if (bh < lo || bl > hi) continue;
        }
        sub.push_back(boxes[j]);
        key += std::to_string(j) + ",";
      }
      minBoxes = std::min(minBoxes, static_cast<int>(sub.size()));
      maxBoxes = std::max(maxBoxes, static_cast<int>(sub.size()));
      if (sub.empty()) continue;
      std::map<std::string, std::shared_ptr<CapsuleCuboidConstraint>>::iterator it = rowCache.find(key);
      if (it == rowCache.end())
        it = rowCache.emplace(key, std::make_shared<CapsuleCuboidConstraint>(m, sub, capIdx, ws.terrainMargin, "ineq_obstacle_box", ws.obstacleRounded,
                                                                            std::vector<double>{}, ws.obstacleSmoothEps))
                 .first;
      modes[s].addNode(it->second, Where::AllNodes);
      if (ws.terrainInterp > 0) modes[s].addOffNode(it->second, uniformInteriorTaus(ws.terrainInterp));
    }
    out.obstacleRowsPerNode = maxBoxes * static_cast<int>(capIdx.size());
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "terrain: %d pillar(s) -> %d keep-out box(es) (%d too low to certify), inset %.4f m (dsafe %.4f, slack %.3f); capsules %d, "
                  "boxes/mode %d..%d (reach %.3f m, pad %.2f), %d row(s)/node, interp %d",
                  static_cast<int>(ws.pillars.size()), static_cast<int>(boxes.size()), out.pillarsDropped, out.topInset, dsafeMax, ws.terrainTopSlack,
                  static_cast<int>(capIdx.size()), minBoxes, maxBoxes, reach, ws.windowPad, out.obstacleRowsPerNode, ws.terrainInterp);
    terrainNote = buf;
  }
  if (!ws.posture.empty() || ws.baseZMinFrac > 0.0) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "posture: %d joint box(es), base z >= %.4f m (%.2f x nominal)", static_cast<int>(ws.posture.size()), zBaseLo, ws.baseZMinFrac);
    terrainNote += (terrainNote.empty() ? "" : "\n  ") + std::string(buf);
  }
  // --- self-collision (before the split: every sub-element checks its own interior) ---
  if (!ws.selfColPairs.empty()) {
    std::vector<std::pair<int, int>> pairs;
    std::vector<double> margins;
    for (const TerrainWalkSpec::SelfColPair& p : ws.selfColPairs) {
      int ia = -1, ib = -1;
      for (int ci = 0; ci < m.nCapsules(); ++ci) {
        if (m.capsuleName(ci) == p.a) ia = ci;
        if (m.capsuleName(ci) == p.b) ib = ci;
      }
      if (ia < 0 || ib < 0) throw std::invalid_argument("terrain self-collision: capsule pair '" + p.a + "' vs '" + p.b + "' not found in the robot's collision_capsules table");
      pairs.push_back({ia, ib});
      margins.push_back(p.margin >= 0.0 ? p.margin : ws.selfColMargin);
    }
    auto selfCol = std::make_shared<CapsuleSelfCollisionConstraint>(m, pairs, ws.selfColMargin, "ineq_selfcol", margins);
    const std::vector<double> selfTaus = ws.selfColInterp > 0 ? uniformInteriorTaus(ws.selfColInterp) : std::vector<double>{};
    for (Element& el : modes) {
      el.addNode(selfCol, Where::AllNodes);
      if (!selfTaus.empty()) el.addOffNode(selfCol, selfTaus);
    }
  }

  // --- mid-swing arc poses and the window hover poses (initial guess only) ---
  const double kHover = 0.04;
  std::vector<Eigen::VectorXd> qArcStep(static_cast<size_t>(nSteps)), qLiftStep, qPreStep;
  for (int k = 0; k < nSteps; ++k) {
    std::vector<Eigen::Vector3d> tg = stanceAfter[static_cast<size_t>(k)];
    for (size_t j = 0; j < swingIdx[static_cast<size_t>(k)].size(); ++j) {
      const int c = swingIdx[static_cast<size_t>(k)][j];
      const Eigen::Vector3d& from = stanceAfter[static_cast<size_t>(k)][static_cast<size_t>(c)];
      const Eigen::Vector3d& to = swingTo[static_cast<size_t>(k)][j];
      const double cz = swingClearZ[static_cast<size_t>(k)][j];
      Eigen::Vector3d apex = 0.5 * (from + to);
      apex.z() = std::max(cz, std::max(from.z(), to.z()) + 0.03) + 0.02;
      tg[static_cast<size_t>(c)] = apex;
    }
    const double zBase = stanceZMean(stanceAfter[static_cast<size_t>(k)], swingIdx[static_cast<size_t>(k)]) + zNominal;
    qArcStep[static_cast<size_t>(k)] = ikEvent(m, *data, cache, out.qNominal, allIdx, tg, zBase, lockIdx).q;
    if (hasLift) {
      std::vector<Eigen::Vector3d> up = stanceAfter[static_cast<size_t>(k)];
      for (int c : swingIdx[static_cast<size_t>(k)]) up[static_cast<size_t>(c)].z() += kHover;
      qLiftStep.push_back(ikEvent(m, *data, cache, out.qNominal, allIdx, up, zBase, lockIdx).q);
    }
    if (hasApproach) {
      std::vector<Eigen::Vector3d> up = stanceAfter[static_cast<size_t>(k) + 1];
      for (int c : swingIdx[static_cast<size_t>(k)]) up[static_cast<size_t>(c)].z() += kHover;
      qPreStep.push_back(ikEvent(m, *data, cache, out.qNominal, allIdx, up, zBase, lockIdx).q);
    }
  }
  // per-mode guess endpoints
  std::vector<Eigen::VectorXd> qaV, qbV, qArcV;
  for (const ModePlan& mp : plan) {
    const size_t k = static_cast<size_t>(mp.step);
    switch (mp.kind) {
      case ModeKind::Support:
        qaV.push_back(out.qEvent[k]);
        qbV.push_back(out.qEvent[k]);
        qArcV.emplace_back();
        break;
      case ModeKind::Lift:
        qaV.push_back(out.qEvent[k]);
        qbV.push_back(qLiftStep[k]);
        qArcV.emplace_back();
        break;
      case ModeKind::Flight:
        qaV.push_back(hasLift ? qLiftStep[k] : out.qEvent[k]);
        qbV.push_back(hasApproach ? qPreStep[k] : out.qEvent[k + 1]);
        qArcV.push_back(qArcStep[k]);
        break;
      case ModeKind::Approach:
        qaV.push_back(qPreStep[k]);
        qbV.push_back(out.qEvent[k + 1]);
        qArcV.emplace_back();
        break;
    }
  }

  // --- h-refinement into the final elements ---
  const int nMode = static_cast<int>(modes.size());
  std::vector<std::vector<double>> fr = ws.subFractions;
  if (fr.empty()) fr.assign(static_cast<size_t>(nMode), std::vector<double>(static_cast<size_t>(ws.splitPerMode), 1.0 / ws.splitPerMode));
  if (static_cast<int>(fr.size()) != nMode)
    throw std::invalid_argument("terrain subFractions: one entry per mode is required (" + std::to_string(nMode) + " modes, got " + std::to_string(fr.size()) + ")");
  for (std::vector<double>& f : fr) {
    if (f.empty()) f = {1.0};
    double sum = 0.0;
    for (double x : f) {
      if (!(x > 0.0)) throw std::invalid_argument("terrain subFractions: every fraction must be > 0");
      sum += x;
    }
    for (double& x : f) x /= sum;
  }
  out.subFractions = fr;
  TrajectoryProblem& P = out.problem;
  std::vector<int> segOrig, segSub, segKOf(static_cast<size_t>(nMode));
  std::vector<double> segFrac0, segFrac;
  std::vector<Eigen::VectorXd> nqa, nqb;
  for (int s = 0; s < nMode; ++s) {
    const Element& mm = modes[static_cast<size_t>(s)];
    const std::vector<double>& f = fr[static_cast<size_t>(s)];
    const int k = static_cast<int>(f.size());
    segKOf[static_cast<size_t>(s)] = k;
    double c0 = 0.0;
    for (int i = 0; i < k; ++i) {
      const double fi = f[static_cast<size_t>(i)];
      Element sub;
      sub.label = mm.label + (k > 1 ? "_" + std::to_string(i) : "");
      sub.phase = mm.phase;
      sub.degree = mm.degree;
      sub.T = mm.T * fi;
      sub.contacts = mm.contacts;
      sub.forceControlPoints = mm.forceControlPoints;
      sub.deriveTorque = mm.deriveTorque;
      sub.costs = mm.costs;
      sub.pointTerms = mm.pointTerms;
      sub.defect = mm.defect;
      for (const NodeAttachment& a : mm.nodeTerms) {
        const std::vector<Where> ws_ = ws.junctionOwnership == "unique"
                                           ? mapWhereUniqueOwner(a.where, i, k, !a.con->readsOneSidedFields())
                                           : mapWhere(a.where, i, k);
        for (Where w : ws_) sub.addNode(a.con, w);
      }
      P.elements.push_back(sub);
      segOrig.push_back(s);
      segSub.push_back(i);
      segFrac0.push_back(c0);
      segFrac.push_back(fi);
      const Eigen::VectorXd d = qbV[static_cast<size_t>(s)] - qaV[static_cast<size_t>(s)];
      nqa.push_back(qaV[static_cast<size_t>(s)] + d * c0);
      nqb.push_back(qaV[static_cast<size_t>(s)] + d * (c0 + fi));
      c0 += fi;
      if (i < k - 1) P.addReset(std::make_shared<ContinuityReset>(dimsC.nq, dimsC.nv));
    }
    if (s < nMode - 1) P.addReset(resets[static_cast<size_t>(s)]);
  }
  // force feasibility on the final elements: lift pins on the last sub-element of a
  // mode, land pins on the first
  for (size_t s = 0; s < P.elements.size(); ++s) {
    const ForcePlan& fp = forceOf[static_cast<size_t>(segOrig[s])];
    if (P.elements[s].contacts.active.empty()) continue;
    finalizeForce(P.elements[s], *fp.b, segSub[s] == segKOf[static_cast<size_t>(segOrig[s])] - 1 ? fp.lift : std::vector<int>{},
                  segSub[s] == 0 ? fp.land : std::vector<int>{}, ws.Nc);
  }
  P.validate();

  // --- the initial guess ---
  {
    const CompiledProblem prob = CompiledProblem::compile(P);
    const Layout& L = prob.layout();
    out.y0.setZero(L.n());
    const int M = static_cast<int>(L.elem.size());
    for (int s = 0; s < M; ++s) {
      const ElementLayout& el = L.elem[static_cast<size_t>(s)];
      const Eigen::VectorXd& qa = nqa[static_cast<size_t>(s)];
      const Eigen::VectorXd& qb = nqb[static_cast<size_t>(s)];
      const Eigen::VectorXd t = prob.grid(s).timeGrid(el.T);
      ElementViewMutable X(out.y0.data(), el);
      const int mo = segOrig[static_cast<size_t>(s)];
      if (qArcV[static_cast<size_t>(mo)].size() > 0) {
        // swing: quadratic Bezier through the mid-swing arc pose at the mode's midpoint
        const Eigen::VectorXd& q0 = qaV[static_cast<size_t>(mo)];
        const Eigen::VectorXd& q1 = qbV[static_cast<size_t>(mo)];
        const Eigen::VectorXd ctrl = 2.0 * qArcV[static_cast<size_t>(mo)] - 0.5 * (q0 + q1);
        for (int k = 0; k < el.Nn; ++k) {
          const double sg = segFrac0[static_cast<size_t>(s)] + segFrac[static_cast<size_t>(s)] * (t(k) / el.T);
          X.Q.col(k) = (1.0 - sg) * (1.0 - sg) * q0 + 2.0 * sg * (1.0 - sg) * ctrl + sg * sg * q1;
        }
      } else {
        for (int k = 0; k < el.Nn; ++k) X.Q.col(k) = qa + (qb - qa) * (t(k) / el.T);
      }
      const Eigen::MatrixXd& D = prob.grid(s).D();
      const double a = prob.timeScale(s);
      X.V = a * X.Q * D.transpose();
      X.A = a * X.V * D.transpose();
    }
    // forces from the floating-base inverse dynamics at every node, fit by the Bezier
    // polygon; then the joint torques from the fitted force
    for (int s = 0; s < M; ++s) {
      const ElementLayout& el = L.elem[static_cast<size_t>(s)];
      ElementViewMutable X(out.y0.data(), el);
      const ContactSet& cs = P.elements[static_cast<size_t>(s)].contacts;
      const std::vector<int>& unact = m.unactuatedIdx();
      Eigen::MatrixXd desired = Eigen::MatrixXd::Zero(el.dims.nlam, el.Nn);
      for (int k = 0; k < el.Nn && el.dims.nlam > 0; ++k) {
        m.evalNode(*data, X.Q.col(k), X.V.col(k), X.A.col(k), kRnea | kFootPos, cache);
        Eigen::MatrixXd K(static_cast<int>(unact.size()), el.dims.nlam);
        int o = 0;
        for (const ContactRef& cr : cs.active) {
          for (size_t r = 0; r < unact.size(); ++r) K.block(static_cast<int>(r), o, 1, cr.dim) = cache.footJ[static_cast<size_t>(cr.index)].col(unact[r]).transpose();
          o += cr.dim;
        }
        Eigen::VectorXd rhs(static_cast<int>(unact.size()));
        for (size_t r = 0; r < unact.size(); ++r) rhs(static_cast<int>(r)) = cache.tau(unact[r]);
        desired.col(k) = K.completeOrthogonalDecomposition().solve(rhs);
      }
      Eigen::MatrixXd nodeForce = desired;
      if (el.Nc > 0) {
        const Eigen::MatrixXd Cphys = el.E.completeOrthogonalDecomposition().solve(desired.transpose()).transpose();
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
          for (const ContactRef& cr : cs.active) {
            u -= cache.footJ[static_cast<size_t>(cr.index)].col(qi).dot(nodeForce.col(k).segment(o, cr.dim));
            o += cr.dim;
          }
          X.U(j, k) = u;
        }
      }
    }
  }
  {
    double tTotal = ws.Tstart;
    for (int k = 0; k < nSteps; ++k) tTotal += ws.steps[static_cast<size_t>(k)].Tswing + ws.steps[static_cast<size_t>(k)].Tsupport;
    const double travel = out.qEvent.back()(0) - out.qEvent.front()(0);
    double zLo = 0.0, zHi = 0.0;
    for (const TerrainPillar& p : ws.pillars) zHi = std::max(zHi, p.top);
    for (const FootPlacement& f : ws.start) zLo = std::min(zLo, f.pos.z());
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "nq=%d nv=%d nu=%d  contacts=%dx dim%d  monitors=%d capsules=%d\n"
                  "  steps=%d  modes=%d (-> %d elements)  T=%.2f s  base travel %.3f m (%.3f m/s)\n"
                  "  nominal base z=%.4f  terrain z in [%.3f, %.3f]  IK residual: %.2e m pos, %.2e rad sole tilt%s%s",
                  m.nq(), m.nv(), m.nu(), m.nContacts(), dim, m.nMonitors(), m.nCapsules(), nSteps, nMode, static_cast<int>(P.elements.size()), tTotal,
                  travel, travel / tTotal, zNominal, zLo, zHi, out.ikWorstErr, worstTilt, terrainNote.empty() ? "" : "\n  ", terrainNote.c_str());
    out.summary = buf;
  }
  if (worstTilt > kLevelTol && dim == 6)
    out.summary += "\n  ! IK could not level a sole (see the tilt above): the flat-sole placement rows will start infeasible";
  return out;
}

// --- prolongation ---------------------------------------------------------------------------

namespace {
// The old element of phase `phase` that contains global time t (the last one whose
// start does not exceed t).
int locateOldElement(const CompiledProblem& from, int phase, double t) {
  const Layout& L = from.layout();
  int j = -1;
  for (int e = 0; e < static_cast<int>(L.elem.size()); ++e) {
    if (from.spec().elements[static_cast<size_t>(e)].phase != phase) continue;
    if (j < 0 || L.elem[static_cast<size_t>(e)].t0 <= t) j = e;
  }
  if (j < 0) throw std::runtime_error("prolongTerrainMesh: the new mesh has a phase the old one lacks");
  return j;
}
}  // namespace

double prolongTerrainMesh(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to, Eigen::VectorXd& yTo) {
  const Layout& Lo = from.layout();
  const Layout& Ln = to.layout();
  yTo.setZero(Ln.n());
  Eigen::VectorXd q, v, a, lam;
  double err = 0.0;
  Eigen::VectorXd q2, v2, a2, lam2;
  for (int e = 0; e < static_cast<int>(Ln.elem.size()); ++e) {
    const ElementLayout& nl = Ln.elem[static_cast<size_t>(e)];
    const int phase = to.spec().elements[static_cast<size_t>(e)].phase;
    const Eigen::VectorXd tn = to.grid(e).timeGrid(nl.T);
    ElementViewMutable X(yTo.data(), nl);
    for (int k = 0; k < nl.Nn; ++k) {
      const double tIn = nl.t0 + std::clamp(tn(k), 1e-6 * nl.T, (1.0 - 1e-6) * nl.T);
      const int j = locateOldElement(from, phase, tIn);
      const ElementLayout& ol = Lo.elem[static_cast<size_t>(j)];
      const double tau = std::clamp(-1.0 + 2.0 * (nl.t0 + tn(k) - ol.t0) / ol.T, -1.0, 1.0);
      from.stateAt(yFrom, j, tau, q, v, a, lam);
      X.Q.col(k) = q;
      X.V.col(k) = v;
      X.A.col(k) = a;
      if (nl.uStored && ol.uStored) X.U.col(k) = from.storedU(yFrom, j, tau);
    }
    if (nl.Nc > 0) {
      const int j = locateOldElement(from, phase, nl.t0 + 0.5 * nl.T);
      const ElementLayout& ol = Lo.elem[static_cast<size_t>(j)];
      if (ol.Nc != nl.Nc) throw std::runtime_error("prolongTerrainMesh: the force bases differ");
      const ElementView Xo(yFrom.data(), ol);
      Eigen::MatrixXd Cp = Xo.C * ol.sigma.asDiagonal();
      const double s0 = std::clamp((nl.t0 - ol.t0) / ol.T, 0.0, 1.0);
      const double s1 = std::clamp((nl.t0 + nl.T - ol.t0) / ol.T, 0.0, 1.0);
      Eigen::MatrixXd left, right;
      if (s0 > 1e-12) {
        deCasteljauSplit(Cp, s0, left, right);
        Cp = right;
      }
      const double s1r = s0 < 1.0 - 1e-12 ? (s1 - s0) / (1.0 - s0) : 1.0;
      if (s1r < 1.0 - 1e-12) {
        deCasteljauSplit(Cp, s1r, left, right);
        Cp = left;
      }
      X.C = Cp * nl.sigma.cwiseInverse().asDiagonal();
    }
    // verification: the new polynomials reproduce the old ones
    for (int s = 0; s < 33; ++s) {
      const double tauN = -1.0 + 2.0 * s / 32.0;
      const double t = nl.t0 + 0.5 * (tauN + 1.0) * nl.T;
      const int j = locateOldElement(from, phase, std::clamp(t, nl.t0 + 1e-9, nl.t0 + nl.T - 1e-9));
      const ElementLayout& ol = Lo.elem[static_cast<size_t>(j)];
      const double tauO = std::clamp(-1.0 + 2.0 * (t - ol.t0) / ol.T, -1.0, 1.0);
      from.stateAt(yFrom, j, tauO, q, v, a, lam);
      to.stateAt(yTo, e, tauN, q2, v2, a2, lam2);
      err = std::max(err, (q - q2).cwiseAbs().maxCoeff());
      err = std::max(err, (v - v2).cwiseAbs().maxCoeff());
      err = std::max(err, (a - a2).cwiseAbs().maxCoeff());
      if (lam.size() > 0 && lam.size() == lam2.size()) err = std::max(err, (lam - lam2).cwiseAbs().maxCoeff());
      if (nl.uStored && ol.uStored) err = std::max(err, (from.storedU(yFrom, j, tauO) - to.storedU(yTo, e, tauN)).cwiseAbs().maxCoeff());
    }
  }
  return err;
}

// --- yaml -------------------------------------------------------------------------------------

namespace {

Eigen::Vector3d yamlVec(const YamlDoc& d, const std::string& key, int n) {
  const std::vector<double> v = d.getVecD(key);
  if (static_cast<int>(v.size()) != n) throw std::runtime_error("'" + key + "' needs " + std::to_string(n) + " numbers, got " + std::to_string(v.size()));
  Eigen::Vector3d out = Eigen::Vector3d::Zero();
  for (int i = 0; i < n; ++i) out(i) = v[static_cast<size_t>(i)];
  return out;
}

FootPlacement placementFromYaml(const YamlDoc& d, const std::string& base) {
  d.checkKeys(base, {"contact", "pos", "clearz"});
  FootPlacement f;
  f.contact = d.getS(base + ".contact", "");
  if (f.contact.empty()) throw std::runtime_error(base + ": 'contact' is required (a contact frame name)");
  f.pos = yamlVec(d, base + ".pos", 3);
  f.clearZ = d.getD(base + ".clearz", -1.0);
  return f;
}

}  // namespace

TerrainWalkSpec terrainSpecFromYaml(const YamlDoc& d) {
  d.checkKeys("problem", {"type", "tstart", "mu", "friceps", "lammin", "landvz", "copx", "copy", "gammaz", "monitorz", "ulim", "vlim", "qbox",
                          "attmax", "noslipnq", "lsqmodemask", "nominal", "planarlock", "start", "steps", "stancepin", "zeroland", "zerolandxy",
                          "landramp", "landcont", "basevbox", "wvel", "vboxnq", "noslipacc", "attnq", "qboxnq", "uboxnq", "uhull", "clearzscale",
                          "selfcolmargin", "selfcolinterp", "selfcolpairs", "posture", "basezfrac", "attbox", "approachfrac", "liftfrac",
                          "ulimjoint", "junctionownership"});
  d.checkKeys("terrain", {"margin", "topslack", "topinset", "interp", "windowpad", "capsules", "pillars"});
  TerrainWalkSpec ws;
  ws.Tstart = d.getD("problem.tstart", ws.Tstart);
  ws.frictionMu = d.getD("problem.mu", ws.frictionMu);
  ws.frictionEps = d.getD("problem.friceps", ws.frictionEps);
  ws.lamMin = d.getD("problem.lammin", ws.lamMin);
  ws.landVzMax = d.getD("problem.landvz", ws.landVzMax);
  ws.copX = d.getD("problem.copx", ws.copX);
  ws.copY = d.getD("problem.copy", ws.copY);
  ws.gammaZ = d.getD("problem.gammaz", ws.gammaZ);
  ws.monitorZ = d.getD("problem.monitorz", ws.monitorZ);
  ws.uLimitScale = d.getD("problem.ulim", ws.uLimitScale);
  ws.velLimitScale = d.getD("problem.vlim", ws.velLimitScale);
  ws.qBoxScale = d.getD("problem.qbox", ws.qBoxScale);
  ws.attMax = d.getD("problem.attmax", ws.attMax);
  ws.stanceNoSlipNq = d.getI("problem.noslipnq", ws.stanceNoSlipNq);
  ws.lsqModeMask = d.getI("problem.lsqmodemask", ws.lsqModeMask);
  ws.stancePin = d.getB("problem.stancepin", ws.stancePin);
  ws.zeroLand = d.getB("problem.zeroland", ws.zeroLand);
  ws.zeroLandXY = d.getB("problem.zerolandxy", ws.zeroLandXY);
  ws.landRamp = d.getB("problem.landramp", ws.landRamp);
  ws.landCont = d.getB("problem.landcont", ws.landCont);
  ws.junctionOwnership = d.getS("problem.junctionownership", ws.junctionOwnership);
  if (ws.junctionOwnership != "plain" && ws.junctionOwnership != "unique")
    throw std::runtime_error("problem.junctionownership must be 'plain' or 'unique'");
  if (d.has("problem.basevbox")) ws.baseVBox = d.getVecD("problem.basevbox");
  ws.approachFrac = d.getD("problem.approachfrac", ws.approachFrac);
  ws.liftFrac = d.getD("problem.liftfrac", ws.liftFrac);
  for (int i = 0, n = d.seqSize("problem.ulimjoint"); i < n; ++i) {
    const std::string b = "problem.ulimjoint." + std::to_string(i);
    d.checkKeys(b, {"joint", "scale"});
    TerrainWalkSpec::ULimitJoint uj;
    uj.joint = d.getS(b + ".joint", "");
    if (uj.joint.empty()) throw std::runtime_error(b + ": 'joint' is required");
    uj.scale = d.getD(b + ".scale", 1.0);
    ws.uLimitJoint.push_back(uj);
  }
  ws.wvel = d.getD("problem.wvel", ws.wvel);
  ws.vBoxNq = d.getI("problem.vboxnq", ws.vBoxNq);
  ws.attBoxNq = d.getI("problem.attnq", ws.attBoxNq);
  ws.qBoxNq = d.getI("problem.qboxnq", ws.qBoxNq);
  ws.uBoxNq = d.getI("problem.uboxnq", ws.uBoxNq);
  ws.uHullDeg = d.getI("problem.uhull", ws.uHullDeg);
  ws.clearZScale = d.getD("problem.clearzscale", ws.clearZScale);
  ws.noSlipAcc = d.getB("problem.noslipacc", ws.noSlipAcc);
  ws.planarLock = d.getVecS("problem.planarlock");
  ws.selfColMargin = d.getD("problem.selfcolmargin", ws.selfColMargin);
  ws.baseZMinFrac = d.getD("problem.basezfrac", ws.baseZMinFrac);
  if (d.has("problem.attbox")) ws.attBox = d.getVecD("problem.attbox");
  for (int i = 0, n = d.seqSize("problem.posture"); i < n; ++i) {
    const std::string b = "problem.posture." + std::to_string(i);
    d.checkKeys(b, {"joint", "lo", "hi", "center", "band"});
    TerrainWalkSpec::PostureBound pb;
    pb.joint = d.getS(b + ".joint", "");
    if (pb.joint.empty()) throw std::runtime_error(b + ": 'joint' is required");
    if (d.has(b + ".band")) {
      const double c = d.getD(b + ".center", 0.0), w = d.getD(b + ".band", 0.0);
      if (!(w >= 0.0)) throw std::runtime_error(b + ": 'band' must be >= 0");
      pb.lo = c - w;
      pb.hi = c + w;
    }
    if (d.has(b + ".lo")) pb.lo = d.getD(b + ".lo", pb.lo);
    if (d.has(b + ".hi")) pb.hi = d.getD(b + ".hi", pb.hi);
    ws.posture.push_back(pb);
  }
  ws.selfColInterp = d.getI("problem.selfcolinterp", ws.selfColInterp);
  for (int i = 0, n = d.seqSize("problem.selfcolpairs"); i < n; ++i) {
    const std::string b = "problem.selfcolpairs." + std::to_string(i);
    d.checkKeys(b, {"a", "b", "margin"});
    TerrainWalkSpec::SelfColPair sp;
    sp.a = d.getS(b + ".a", "");
    sp.b = d.getS(b + ".b", "");
    if (sp.a.empty() || sp.b.empty()) throw std::runtime_error(b + ": wants capsule names a and b");
    sp.margin = d.getD(b + ".margin", -1.0);
    ws.selfColPairs.push_back(sp);
  }
  for (int i = 0, n = d.seqSize("problem.nominal"); i < n; ++i) {
    const std::string b = "problem.nominal." + std::to_string(i);
    d.checkKeys(b, {"joint", "angle"});
    NominalAngle pa;
    pa.joint = d.getS(b + ".joint", "");
    if (pa.joint.empty()) throw std::runtime_error(b + ": 'joint' is required");
    pa.angle = d.getD(b + ".angle", 0.0);
    ws.nominal.push_back(pa);
  }
  for (int i = 0, n = d.seqSize("problem.start"); i < n; ++i) ws.start.push_back(placementFromYaml(d, "problem.start." + std::to_string(i)));
  const int nSteps = d.seqSize("problem.steps");
  if (nSteps <= 0) throw std::runtime_error("problem.steps: at least one step is required");
  for (int i = 0; i < nSteps; ++i) {
    const std::string b = "problem.steps." + std::to_string(i);
    d.checkKeys(b, {"tswing", "tsupport", "feet"});
    WalkStep s;
    s.Tswing = d.getD(b + ".tswing", s.Tswing);
    s.Tsupport = d.getD(b + ".tsupport", s.Tsupport);
    const int nf = d.seqSize(b + ".feet");
    if (nf <= 0) throw std::runtime_error(b + ".feet: a step moves at least one foot");
    for (int j = 0; j < nf; ++j) s.feet.push_back(placementFromYaml(d, b + ".feet." + std::to_string(j)));
    ws.steps.push_back(s);
  }
  ws.terrainMargin = d.getD("terrain.margin", ws.terrainMargin);
  ws.terrainTopSlack = d.getD("terrain.topslack", ws.terrainTopSlack);
  ws.terrainTopInset = d.getD("terrain.topinset", ws.terrainTopInset);
  ws.terrainInterp = d.getI("terrain.interp", ws.terrainInterp);
  ws.windowPad = d.getD("terrain.windowpad", ws.windowPad);
  ws.terrainCapsules = d.getVecS("terrain.capsules");
  for (int i = 0, n = d.seqSize("terrain.pillars"); i < n; ++i) {
    const std::string b = "terrain.pillars." + std::to_string(i);
    d.checkKeys(b, {"name", "center", "size", "top"});
    TerrainPillar p;
    p.name = d.getS(b + ".name", "p" + std::to_string(i));
    const Eigen::Vector3d c = yamlVec(d, b + ".center", 2);
    const Eigen::Vector3d s = yamlVec(d, b + ".size", 2);
    p.cx = c.x();
    p.cy = c.y();
    p.lx = s.x();
    p.ly = s.y();
    p.top = d.getD(b + ".top", 0.0);
    ws.pillars.push_back(p);
  }
  // transcription knobs (the `solver:` block)
  ws.degree = d.getI("solver.deg", ws.degree);
  ws.Nc = d.getI("solver.nc", ws.Nc);
  ws.splitPerMode = d.getI("solver.split", ws.splitPerMode);
  ws.lsqNq = d.getI("solver.lsqnq", ws.lsqNq);
  ws.precond = d.getB("solver.precond", ws.precond);
  {
    const double rho = d.getD("solver.rho", d.getD("solver.rhoabs", 0.0));
    ws.lsqRho = rho > 0.0 ? rho : d.getD("solver.wdyn", 0.0) * d.getD("solver.mumax", 0.0);
  }
  ws.elimU = d.getB("solver.elimu", ws.elimU);
  ws.elimUNq = d.getI("solver.elimuq", ws.elimUNq);
  ws.dynNq = d.getI("solver.dynnq", ws.dynNq);
  return ws;
}

// --- the family ---------------------------------------------------------------------------------

TerrainFamily::TerrainFamily(const YamlDoc& d, const RobotDescription& rd, std::shared_ptr<const RobotModel> model)
    : type_("terrain_walk"), name_(d.getS("name", "terrain_walk")), model_(std::move(model)) {
  (void)rd;
  ws_ = terrainSpecFromYaml(d);
  {
    const std::string err = validateTerrainWalkSpec(ws_);
    if (!err.empty()) throw std::invalid_argument("terrain_walk problem: " + err);
  }
  warnings_ = terrainWalkSpecWarnings(ws_);
  objective_ = objectiveFromYaml(d);
  current_ = rebuild(ws_);
  fractions_ = current_.subFractions;
  y0_ = current_.y0;
}

TerrainWalkSpec TerrainFamily::specWith(const std::vector<std::vector<double>>& fractions) const {
  TerrainWalkSpec ws = ws_;
  ws.subFractions = fractions;
  return ws;
}

TerrainWalkBuild TerrainFamily::rebuild(const TerrainWalkSpec& ws) const {
  TerrainWalkBuild b = buildTerrainWalk(ws, model_);
  attachObjective(b.problem, objective_, *model_);
  return b;
}

TrajectoryProblem TerrainFamily::build() { return current_.problem; }

std::vector<double> TerrainFamily::objectiveNodeWeights(const CompiledProblem& prob) const {
  return leap::objectiveNodeWeights(prob, objective_.present ? objective_.quadrature : "clenshaw_curtis", objective_.present ? objective_.scale : 1.0);
}

TrajectoryProblem TerrainFamily::obstacleVariant(const ObstacleVariant& v) {
  TerrainWalkSpec ws = specWith(fractions_);
  if (v.smoothEps >= 0.0) ws.obstacleSmoothEps = v.smoothEps;
  if (v.rounded >= 0) ws.obstacleRounded = v.rounded != 0;
  if (v.smoothEps > 0.0) ws.obstacleRounded = false;
  ws.obstacleZScale = v.zScale;
  return rebuild(ws).problem;
}

Eigen::VectorXd TerrainFamily::seed(const CompiledProblem& prob) {
  if (y0_.size() != prob.layout().n()) throw std::runtime_error("terrain seed: layout mismatch");
  return y0_;
}

void TerrainFamily::refine(const std::vector<ElementSplit>& splits) {
  const std::vector<Element>& elems = current_.problem.elements;
  std::vector<int> subIdx(elems.size(), 0);
  for (size_t s = 1; s < elems.size(); ++s) subIdx[s] = elems[s].phase == elems[s - 1].phase ? subIdx[s - 1] + 1 : 0;
  std::vector<ElementSplit> sorted = splits;
  std::sort(sorted.begin(), sorted.end(), splitElementDescending);
  std::vector<std::vector<double>> fr = fractions_;
  for (const ElementSplit& sp : sorted) {
    const int mode = elems[static_cast<size_t>(sp.element)].phase;
    const int i = subIdx[static_cast<size_t>(sp.element)];
    std::vector<double>& f = fr[static_cast<size_t>(mode)];
    const double fi = f[static_cast<size_t>(i)];
    f[static_cast<size_t>(i)] = fi * sp.fraction;
    f.insert(f.begin() + i + 1, fi * (1.0 - sp.fraction));
  }
  current_ = rebuild(specWith(fr));
  fractions_ = current_.subFractions;
}

double TerrainFamily::prolong(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to, Eigen::VectorXd& yTo) {
  return prolongTerrainMesh(from, yFrom, to, yTo);
}

Eigen::VectorXd TerrainFamily::oracleTermScales(const CompiledProblem& prob) const {
  return physicalTermScales(prob, *model_, current_.qNominal);
}

}  // namespace leap
