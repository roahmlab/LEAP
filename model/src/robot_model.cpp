#include "leap/model/robot_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/impulse-dynamics-derivatives.hpp>
#include <pinocchio/algorithm/impulse-dynamics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/proximal.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

#include "leap/model/robot_model_impl.hpp"

namespace leap {

// ------------------------------------------------------------------------------------
// ModelData
// ------------------------------------------------------------------------------------

ModelData::ModelData(const pinocchio::Model& m, size_t nContacts, size_t nImpactSets, size_t nConeSets)
    : data(m),
      impactData(nImpactSets, pinocchio::Data(m)),
      contactData(nImpactSets),
      coneData(nConeSets, pinocchio::Data(m)),
      coneContactData(nConeSets),
      fext(m.joints.size(), pinocchio::Force::Zero()) {
  kvq.setZero(6, m.nv);
  kaq.setZero(6, m.nv);
  kav.setZero(6, m.nv);
  kad.setZero(6, m.nv);
  J6.setZero(6, m.nv);
  dvel3.setZero(3, m.nv);
  corr.setZero(m.nv, m.nv);
  Jl3.assign(nContacts, Eigen::MatrixXd::Zero(3, m.nv));
  Jwang3.assign(nContacts, Eigen::MatrixXd::Zero(3, m.nv));
  p3.assign(nContacts, Eigen::Vector3d::Zero());
}

// ------------------------------------------------------------------------------------
// construction helpers
// ------------------------------------------------------------------------------------

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("RobotModel: " + msg); }

BodyPoint bodyPointOfFrame(const pinocchio::Model& model, const std::string& frameName, int dim) {
  if (!model.existFrame(frameName)) fail("unknown frame '" + frameName + "'");
  const pinocchio::FrameIndex fid = model.getFrameId(frameName);
  const pinocchio::Frame& fr = model.frames[fid];
  BodyPoint p;
  p.name = frameName;
  p.dim = dim;
  p.frame = fid;
  p.joint = fr.parentJoint;
  p.placement = fr.placement;
  return p;
}

Eigen::Vector3d localOffset(const pinocchio::Model& model, pinocchio::Data& data, const Eigen::VectorXd& q,
                            pinocchio::FrameIndex from, pinocchio::FrameIndex to) {
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  return data.oMf[from].actInv(data.oMf[to]).translation();
}

// Resolve the declared capsules: check the frames, turn a `toFrame` endpoint into a
// constant local offset (measured at several configurations: it must be body-fixed),
// and emit the two endpoint points per capsule.
void resolveCapsules(const std::vector<CollisionCapsuleSpec>& declared, const pinocchio::Model& model,
                     std::vector<CollisionCapsuleSpec>& out, std::vector<BodyPoint>& points) {
  if (declared.empty()) return;
  pinocchio::Data data(model);
  std::vector<Eigen::VectorXd> probes;
  const Eigen::VectorXd qN = pinocchio::neutral(model);
  probes.push_back(qN);
  for (int k = 0; k < 3; ++k) {
    Eigen::VectorXd dv(model.nv);
    for (int i = 0; i < model.nv; ++i) dv(i) = 0.35 * std::sin(1.0 + 0.7 * i + 2.1 * k) + 0.1 * (k + 1);
    probes.push_back(pinocchio::integrate(model, qN, dv));
  }
  for (const CollisionCapsuleSpec& cs : declared) {
    if (cs.name.empty()) fail("a collision capsule has no name");
    for (const CollisionCapsuleSpec& seen : out)
      if (seen.name == cs.name) fail("duplicate collision capsule '" + cs.name + "'");
    if (!(cs.radius > 0.0) || !std::isfinite(cs.radius))
      fail("collision capsule '" + cs.name + "' needs a positive finite radius");
    if (!model.existFrame(cs.frameName))
      fail("collision capsule '" + cs.name + "' references unknown frame '" + cs.frameName + "'");
    CollisionCapsuleSpec r = cs;
    const pinocchio::FrameIndex fid = model.getFrameId(cs.frameName);
    if (!cs.toFrame.empty()) {
      if (!model.existFrame(cs.toFrame))
        fail("collision capsule '" + cs.name + "' references unknown to_frame '" + cs.toFrame + "'");
      const pinocchio::FrameIndex tid = model.getFrameId(cs.toFrame);
      r.b = localOffset(model, data, probes.front(), fid, tid);
      for (size_t p = 1; p < probes.size(); ++p) {
        const Eigen::Vector3d o = localOffset(model, data, probes[p], fid, tid);
        if ((o - r.b).norm() > 1e-10)
          fail("collision capsule '" + cs.name + "': to_frame '" + cs.toFrame + "' is not body-fixed in '" +
               cs.frameName + "' (give an explicit 'b' offset instead)");
      }
    }
    if (!r.a.allFinite() || !r.b.allFinite()) fail("collision capsule '" + cs.name + "' has a non-finite endpoint");
    const pinocchio::Frame& fr = model.frames[fid];
    for (const Eigen::Vector3d& p : {r.a, r.b}) {
      BodyPoint bp;
      bp.name = cs.name;
      bp.dim = 3;
      bp.frame = fid;
      bp.joint = fr.parentJoint;
      bp.placement = fr.placement * pinocchio::SE3(Eigen::Matrix3d::Identity(), p);
      points.push_back(bp);
    }
    out.push_back(r);
  }
}

Eigen::VectorXd expandMotorValues(const std::vector<double>& values, int nu, const char* name) {
  if (values.empty()) return Eigen::VectorXd::Zero(nu);
  if (values.size() != 1 && static_cast<int>(values.size()) != nu)
    fail(std::string("motor ") + name + " must be a scalar or have nu=" + std::to_string(nu) + " entries");
  Eigen::VectorXd out(nu);
  if (values.size() == 1)
    out.setConstant(values.front());
  else
    for (int i = 0; i < nu; ++i) out(i) = values[static_cast<size_t>(i)];
  for (int i = 0; i < nu; ++i)
    if (!std::isfinite(out(i))) fail(std::string("motor ") + name + " entries must be finite");
  return out;
}

// World position + position-Jacobian of body-fixed points. Requires
// computeJointJacobians(q) to have run on `data`.
void evalPointFrames(const pinocchio::Model& model, pinocchio::Data& data, const std::vector<BodyPoint>& pts,
                     Eigen::MatrixXd& J6, std::vector<Eigen::Vector3d>& posOut,
                     std::vector<Eigen::MatrixXd>& jOut) {
  for (size_t i = 0; i < pts.size(); ++i) {
    const BodyPoint& bp = pts[i];
    const Eigen::Vector3d p = (data.oMi[bp.joint] * bp.placement).translation();
    J6.setZero();  // pinocchio writes the support columns only
    pinocchio::getJointJacobian(model, data, bp.joint, pinocchio::WORLD, J6);
    posOut[i] = p;
    for (int c = 0; c < model.nv; ++c) {
      const Eigen::Vector3d JwLin = J6.block<3, 1>(0, c);
      const Eigen::Vector3d JwAng = J6.block<3, 1>(3, c);
      jOut[i].col(c) = JwLin + JwAng.cross(p);
    }
  }
}

}  // namespace

// ------------------------------------------------------------------------------------
// construction
// ------------------------------------------------------------------------------------

RobotModel::RobotModel(const RobotDescription& rd) : impl_(std::make_unique<Impl>()) {
  Impl& im = *impl_;
  im.description = rd;
  pinocchio::urdf::buildModel(rd.urdfPath, im.model);
  im.model.gravity.linear(rd.gravity);
  im.model.gravity.angular(Eigen::Vector3d::Zero());
  im.jointNames.assign(im.model.names.begin() + 1, im.model.names.end());
  for (size_t j = 1; j < im.model.joints.size(); ++j)
    if (im.model.joints[j].nq() != 1 || im.model.joints[j].nv() != 1)
      fail("joint '" + im.model.names[j] + "' is not 1-DOF (continuous joints are not supported: use revolute)");

  if (!rd.expectedJointOrder.empty()) {
    bool ok = im.jointNames.size() == rd.expectedJointOrder.size();
    for (size_t i = 0; ok && i < im.jointNames.size(); ++i) ok = im.jointNames[i] == rd.expectedJointOrder[i];
    if (!ok) {
      std::string got;
      for (size_t i = 0; i < im.jointNames.size(); ++i) got += (i ? ", " : "") + im.jointNames[i];
      fail("joint order mismatch (q layout would differ). Parsed order: " + got);
    }
  }

  for (const ContactFrameSpec& c : rd.contacts) {
    if (c.dim != 2 && c.dim != 3 && c.dim != 6) fail("contact dim must be 2, 3, or 6");
    im.contacts.push_back(bodyPointOfFrame(im.model, c.frameName, c.dim));
  }
  for (const std::string& name : rd.monitorFrames) im.monitors.push_back(bodyPointOfFrame(im.model, name, 3));
  resolveCapsules(rd.capsules, im.model, im.capsules, im.capsulePoints);

  // actuation
  std::vector<std::string> actuated = rd.actuatedJoints;
  if (actuated.empty()) actuated = im.jointNames;  // fixed-base arm: every joint
  for (const std::string& name : actuated) {
    if (!im.model.existJointName(name)) fail("unknown actuated joint '" + name + "'");
    const pinocchio::JointIndex j = im.model.getJointId(name);
    im.actIdx.push_back(im.model.joints[j].idx_v());
  }
  const int nu = static_cast<int>(im.actIdx.size());
  im.B = Eigen::MatrixXd::Zero(im.model.nv, nu);
  for (int j = 0; j < nu; ++j) im.B(im.actIdx[static_cast<size_t>(j)], j) = 1.0;
  std::vector<char> isAct(static_cast<size_t>(im.model.nv), 0);
  for (int i : im.actIdx) isAct[static_cast<size_t>(i)] = 1;
  for (int i = 0; i < im.model.nv; ++i)
    if (!isAct[static_cast<size_t>(i)]) im.unactIdx.push_back(i);

  // limits (URDF unless the description gives explicit lists)
  im.effortLimit.resize(nu);
  im.velocityLimitU.resize(nu);
  for (int j = 0; j < nu; ++j) {
    im.effortLimit(j) = im.model.effortLimit(im.actIdx[static_cast<size_t>(j)]);
    im.velocityLimitU(j) = im.model.velocityLimit(im.actIdx[static_cast<size_t>(j)]);
  }
  if (rd.effortLimitsSource == "explicit") {
    if (static_cast<int>(rd.effortLimits.size()) != nu) fail("actuation.effort_limits must have nu entries");
    for (int j = 0; j < nu; ++j) im.effortLimit(j) = rd.effortLimits[static_cast<size_t>(j)];
  }
  if (rd.velocityLimitsSource == "explicit") {
    if (static_cast<int>(rd.velocityLimits.size()) != nu) fail("actuation.velocity_limits must have nu entries");
    for (int j = 0; j < nu; ++j) im.velocityLimitU(j) = rd.velocityLimits[static_cast<size_t>(j)];
  }

  // motor model
  im.motorType = rd.motor.type;
  if (im.motorType != "none" && im.motorType != "friction_inertia") fail("unknown motor model '" + im.motorType + "'");
  im.motorCoulomb = expandMotorValues(rd.motor.coulomb, nu, "coulomb");
  im.motorViscous = expandMotorValues(rd.motor.viscous, nu, "viscous");
  im.motorArmature = expandMotorValues(rd.motor.armature, nu, "armature");
  im.motorBias = expandMotorValues(rd.motor.bias, nu, "bias");
  im.motorVelocitySmoothing = rd.motor.velocitySmoothing;
  if (!std::isfinite(im.motorVelocitySmoothing) || im.motorVelocitySmoothing <= 0.0)
    fail("motor velocity smoothing must be positive and finite");
  for (int j = 0; j < nu; ++j) {
    if (im.motorCoulomb(j) < 0.0 || im.motorViscous(j) < 0.0 || im.motorArmature(j) < 0.0)
      fail("coulomb, viscous and armature coefficients must be nonnegative");
    if (im.motorType == "friction_inertia") im.model.armature(im.actIdx[static_cast<size_t>(j)]) = im.motorArmature(j);
  }
}

RobotModel::~RobotModel() = default;
RobotModel::RobotModel(RobotModel&&) noexcept = default;
RobotModel& RobotModel::operator=(RobotModel&&) noexcept = default;

// ------------------------------------------------------------------------------------
// accessors
// ------------------------------------------------------------------------------------

const RobotDescription& RobotModel::description() const { return impl_->description; }
int RobotModel::nq() const { return impl_->model.nq; }
int RobotModel::nv() const { return impl_->model.nv; }
int RobotModel::nu() const { return static_cast<int>(impl_->actIdx.size()); }
const std::vector<std::string>& RobotModel::jointNames() const { return impl_->jointNames; }

int RobotModel::jointIndex(const std::string& name) const {
  for (size_t i = 0; i < impl_->jointNames.size(); ++i)
    if (impl_->jointNames[i] == name) return static_cast<int>(i);
  fail("unknown joint '" + name + "'");
}

bool RobotModel::hasJoint(const std::string& name) const {
  for (const std::string& n : impl_->jointNames)
    if (n == name) return true;
  return false;
}

bool RobotModel::hasFrame(const std::string& name) const { return impl_->model.existFrame(name); }

const Eigen::MatrixXd& RobotModel::B() const { return impl_->B; }
const std::vector<int>& RobotModel::actuatedIdx() const { return impl_->actIdx; }
const std::vector<int>& RobotModel::unactuatedIdx() const { return impl_->unactIdx; }
const BaseRotationLayout& RobotModel::baseRotation() const { return impl_->description.baseRot; }

Eigen::VectorXd RobotModel::actuatedEffortLimit() const { return impl_->effortLimit; }
Eigen::VectorXd RobotModel::actuatedVelocityLimit() const { return impl_->velocityLimitU; }
Eigen::VectorXd RobotModel::lowerPositionLimit() const { return impl_->model.lowerPositionLimit; }
Eigen::VectorXd RobotModel::upperPositionLimit() const { return impl_->model.upperPositionLimit; }
Eigen::VectorXd RobotModel::velocityLimit() const {
  // the URDF vector in v order, with the description's explicit actuated limits applied
  Eigen::VectorXd lim = impl_->model.velocityLimit;
  for (size_t j = 0; j < impl_->actIdx.size(); ++j) lim(impl_->actIdx[j]) = impl_->velocityLimitU(static_cast<int>(j));
  return lim;
}
const Eigen::Vector3d& RobotModel::gravity() const { return impl_->description.gravity; }

double RobotModel::totalMass() const {
  double m = 0.0;
  for (size_t i = 1; i < impl_->model.inertias.size(); ++i) m += impl_->model.inertias[i].mass();
  return m;
}

const std::string& RobotModel::motorModelType() const { return impl_->motorType; }

Eigen::VectorXd RobotModel::motorLoss(const Eigen::Ref<const Eigen::VectorXd>& v, bool smooth) const {
  if (v.size() != nv()) throw std::invalid_argument("RobotModel::motorLoss: v has wrong size");
  Eigen::VectorXd loss = Eigen::VectorXd::Zero(nu());
  if (impl_->motorType == "none") return loss;
  const double eps = impl_->motorVelocitySmoothing;
  for (int j = 0; j < nu(); ++j) {
    const double x = v(impl_->actIdx[static_cast<size_t>(j)]);
    const double sign = smooth ? x / std::sqrt(x * x + eps * eps) : static_cast<double>((x > 0.0) - (x < 0.0));
    loss(j) = impl_->motorCoulomb(j) * sign + impl_->motorViscous(j) * x + impl_->motorBias(j);
  }
  return loss;
}

int RobotModel::nContacts() const { return static_cast<int>(impl_->contacts.size()); }
int RobotModel::contactDim(int contactIdx) const { return impl_->contacts.at(static_cast<size_t>(contactIdx)).dim; }
const std::string& RobotModel::contactName(int contactIdx) const {
  return impl_->contacts.at(static_cast<size_t>(contactIdx)).name;
}

int RobotModel::contactIndex(const std::string& name) const {
  for (size_t i = 0; i < impl_->contacts.size(); ++i)
    if (impl_->contacts[i].name == name) return static_cast<int>(i);
  fail("unknown contact frame '" + name + "'");
}

ContactSet RobotModel::contactSet(const std::vector<std::string>& names) const {
  ContactSet cs;
  for (const std::string& n : names) {
    const int i = contactIndex(n);
    cs.active.push_back({i, impl_->contacts[static_cast<size_t>(i)].dim});
  }
  return cs;
}

ContactSet RobotModel::allContacts() const {
  ContactSet cs;
  for (size_t i = 0; i < impl_->contacts.size(); ++i) cs.active.push_back({static_cast<int>(i), impl_->contacts[i].dim});
  return cs;
}

NodeDims RobotModel::nodeDims(const ContactSet& cs) const {
  NodeDims d;
  d.nq = nq();
  d.nv = nv();
  d.na = nv();
  d.nu = nu();
  d.nlam = cs.nLam();
  return d;
}

int RobotModel::nMonitors() const { return static_cast<int>(impl_->monitors.size()); }
const std::string& RobotModel::monitorName(int monitorIdx) const {
  return impl_->monitors.at(static_cast<size_t>(monitorIdx)).name;
}

int RobotModel::nCapsules() const { return static_cast<int>(impl_->capsules.size()); }
const std::string& RobotModel::capsuleName(int i) const { return impl_->capsules.at(static_cast<size_t>(i)).name; }
int RobotModel::capsuleIndex(const std::string& name) const {
  for (size_t i = 0; i < impl_->capsules.size(); ++i)
    if (impl_->capsules[i].name == name) return static_cast<int>(i);
  fail("unknown collision capsule '" + name + "'");
}
const std::string& RobotModel::capsuleFrame(int i) const { return impl_->capsules.at(static_cast<size_t>(i)).frameName; }
double RobotModel::capsuleRadius(int i) const { return impl_->capsules.at(static_cast<size_t>(i)).radius; }
Eigen::Vector3d RobotModel::capsuleLocalA(int i) const { return impl_->capsules.at(static_cast<size_t>(i)).a; }
Eigen::Vector3d RobotModel::capsuleLocalB(int i) const { return impl_->capsules.at(static_cast<size_t>(i)).b; }

// ------------------------------------------------------------------------------------
// impact registries and per-thread data
// ------------------------------------------------------------------------------------

int RobotModel::ensureImpactSet(const ContactSet& cs) const {
  for (size_t i = 0; i < impl_->impactSets.size(); ++i)
    if (impl_->impactSets[i] == cs) return static_cast<int>(i);
  ConstraintModelVec models;
  for (const ContactRef& cr : cs.active) {
    const BodyPoint& bp = impl_->contacts.at(static_cast<size_t>(cr.index));
    models.emplace_back(cr.dim == 6 ? pinocchio::CONTACT_6D : pinocchio::CONTACT_3D, impl_->model, bp.joint,
                        bp.placement, pinocchio::LOCAL);
  }
  impl_->impactSets.push_back(cs);
  impl_->impactModels.push_back(std::move(models));
  return static_cast<int>(impl_->impactSets.size()) - 1;
}

int RobotModel::ensureImpactConeSet(const ContactSet& cs) const {
  for (size_t i = 0; i < impl_->coneSets.size(); ++i)
    if (impl_->coneSets[i] == cs) return static_cast<int>(i);
  ConstraintModelVec models;
  for (const ContactRef& cr : cs.active) {
    const BodyPoint& bp = impl_->contacts.at(static_cast<size_t>(cr.index));
    models.emplace_back(cr.dim == 6 ? pinocchio::CONTACT_6D : pinocchio::CONTACT_3D, impl_->model, bp.joint,
                        bp.placement, pinocchio::LOCAL_WORLD_ALIGNED);
  }
  impl_->coneSets.push_back(cs);
  impl_->coneModels.push_back(std::move(models));
  return static_cast<int>(impl_->coneSets.size()) - 1;
}

int RobotModel::numImpactConeSets() const { return static_cast<int>(impl_->coneSets.size()); }

int RobotModel::impactConeLamOffset(int coneId, int contactIndex) const {
  const auto s = static_cast<size_t>(coneId);
  if (s >= impl_->coneSets.size()) return -1;
  return impl_->coneSets[s].lamOffset(contactIndex);
}

const ContactSet& RobotModel::impactSetContacts(int setId) const { return impl_->impactSets.at(static_cast<size_t>(setId)); }
const ContactSet& RobotModel::impactConeSetContacts(int coneId) const { return impl_->coneSets.at(static_cast<size_t>(coneId)); }

void ModelDataDeleter::operator()(ModelData* p) const { delete p; }

ModelDataPtr RobotModel::makeData() const {
  ModelDataPtr pd(new ModelData(impl_->model, impl_->contacts.size(), impl_->impactSets.size(),
                                impl_->coneSets.size()));
  for (size_t s = 0; s < impl_->impactSets.size(); ++s) {
    for (const auto& cm : impl_->impactModels[s]) pd->contactData[s].emplace_back(cm);
    pinocchio::initConstraintDynamics(impl_->model, pd->impactData[s], impl_->impactModels[s]);
  }
  for (size_t s = 0; s < impl_->coneSets.size(); ++s) {
    for (const auto& cm : impl_->coneModels[s]) pd->coneContactData[s].emplace_back(cm);
    pinocchio::initConstraintDynamics(impl_->model, pd->coneData[s], impl_->coneModels[s]);
  }
  return pd;
}

void RobotModel::initCache(ModelEvalCache& out) const {
  const int n = nv();
  out.tau.setZero(n);
  out.gdyn.setZero(n);
  out.dtau_dq.setZero(n, n);
  out.dtau_dv.setZero(n, n);
  out.M.setZero(n, n);
  out.dgdyn_dq.setZero(n, n);
  const size_t nc = impl_->contacts.size();
  out.footPos.resize(nc);
  out.footVel.resize(nc);
  out.footAcc.resize(nc);
  out.footJ.resize(nc);
  out.dvel_dq.resize(nc);
  out.dacc_dq.resize(nc);
  out.dacc_dv.resize(nc);
  out.footRotLog.assign(nc, Eigen::Vector3d::Zero());
  out.footRotJ.resize(nc);
  for (size_t i = 0; i < nc; ++i) {
    const int dim = impl_->contacts[i].dim;
    out.footPos[i].setZero(std::min(dim, 3));
    out.footVel[i].setZero(dim);
    out.footAcc[i].setZero(dim);
    out.footJ[i].setZero(dim, n);
    out.dvel_dq[i].setZero(dim, n);
    out.dacc_dq[i].setZero(dim, n);
    out.dacc_dv[i].setZero(dim, n);
    out.footRotJ[i].setZero(3, n);
  }
  const size_t nm = impl_->monitors.size();
  out.monPos.assign(nm, Eigen::Vector3d::Zero());
  out.monJ.assign(nm, Eigen::MatrixXd::Zero(3, n));
  const size_t ncp = impl_->capsulePoints.size();
  out.capPos.assign(ncp, Eigen::Vector3d::Zero());
  out.capJ.assign(ncp, Eigen::MatrixXd::Zero(3, n));
  out.comPos.assign(1, Eigen::Vector3d::Zero());
  out.comJ.assign(1, Eigen::MatrixXd::Zero(3, n));
  const size_t ncs = impl_->coneSets.size();
  out.impactLam.resize(ncs);
  out.impactLamDq.resize(ncs);
  out.impactLamDv.resize(ncs);
  for (size_t s = 0; s < ncs; ++s) {
    const int rows = impl_->coneSets[s].nLam();
    out.impactLam[s].setZero(rows);
    out.impactLamDq[s].setZero(rows, n);
    out.impactLamDv[s].setZero(rows, n);
  }
}

// ------------------------------------------------------------------------------------
// evalNode
// ------------------------------------------------------------------------------------

void RobotModel::evalNode(ModelData& pd, const Eigen::Ref<const Eigen::VectorXd>& q,
                          const Eigen::Ref<const Eigen::VectorXd>& v, const Eigen::Ref<const Eigen::VectorXd>& a,
                          const Eigen::Ref<const Eigen::VectorXd>& lam, const ContactSet& cs, unsigned needs,
                          ModelEvalCache& out) const {
  const Impl& im = *impl_;
  const pinocchio::Model& model = im.model;
  pinocchio::Data& data = pd.data;
  if (out.footPos.size() != im.contacts.size() || out.footRotLog.size() != im.contacts.size() ||
      out.monPos.size() != im.monitors.size() || out.capPos.size() != im.capsulePoints.size() ||
      out.tau.size() != nv() || out.impactLam.size() != im.coneSets.size())
    initCache(out);
  // gdyn / dgdyn_dq fold in -Jc' lam_c, so the contact Jacobians are required.
  if (!cs.active.empty() && (needs & (kRnea | kRneaDerivs))) needs |= kFootPos;

  // --- inverse dynamics -------------------------------------------------------------
  // computeRNEADerivatives also produces data.tau and M (upper triangle), so the
  // separate rnea / crba passes are skipped when the derivatives are requested.
  const bool tauMFromDerivs = (needs & kRneaDerivs) != 0;
  if ((needs & kRnea) && !tauMFromDerivs) out.tau = pinocchio::rnea(model, data, q, v, a);
  if (needs & kRneaDerivs) {
    pinocchio::computeRNEADerivatives(model, data, q, v, a);
    out.dtau_dq = data.dtau_dq;
    out.dtau_dv = data.dtau_dv;
    if (needs & kRnea) out.tau = data.tau;
    out.M = data.M.selfadjointView<Eigen::Upper>();
  }
  if ((needs & kMass) && !tauMFromDerivs) {
    pinocchio::crba(model, data, q);
    out.M = data.M.selfadjointView<Eigen::Upper>();
  }
  // Joint-side passive motor terms (reflected inertia is already inside pinocchio).
  if (im.motorType != "none") {
    const double eps2 = im.motorVelocitySmoothing * im.motorVelocitySmoothing;
    for (int j = 0; j < nu(); ++j) {
      const int i = im.actIdx[static_cast<size_t>(j)];
      const double x = v(i);
      const double den = x * x + eps2;
      if (needs & kRnea) {
        const double smoothSign = x / std::sqrt(den);
        out.tau(i) += im.motorCoulomb(j) * smoothSign + im.motorViscous(j) * x + im.motorBias(j);
      }
      if (needs & kRneaDerivs) {
        const double dsign = eps2 / std::pow(den, 1.5);
        out.dtau_dv(i, i) += im.motorCoulomb(j) * dsign + im.motorViscous(j);
      }
    }
  }

  // --- contact frames -----------------------------------------------------------------
  // WORLD joint derivatives plus an explicit shift to the body-fixed contact point p(q):
  //   pdot = v_O.lin + om x p,   pddot = a_O.lin + al x p + om x pdot.
  const bool wantFeet = needs & (kFootPos | kFootVel | kFootAcc | kFootDerivs | kFootRot);
  if (wantFeet) {
    pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);
    const int nvm = model.nv;
    for (size_t i = 0; i < im.contacts.size(); ++i) {
      const BodyPoint& ci = im.contacts[i];
      const pinocchio::JointIndex j = ci.joint;
      const Eigen::Vector3d p = (data.oMi[j] * ci.placement).translation();
      const pinocchio::Motion& vw = data.ov[j];
      const pinocchio::Motion& aw = data.oa[j];
      const Eigen::Vector3d om = vw.angular();
      const Eigen::Vector3d pdot = vw.linear() + om.cross(p);
      const Eigen::Vector3d al = aw.angular();
      const Eigen::Vector3d pddot = aw.linear() + al.cross(p) + om.cross(pdot);
      pd.p3[i] = p;

      // Non-ancestor columns of the WORLD derivative buffers are untouched by
      // pinocchio, so they must be zeroed between contacts.
      pd.kvq.setZero();
      pd.kaq.setZero();
      pd.kav.setZero();
      pd.kad.setZero();
      pinocchio::getJointAccelerationDerivatives(model, data, j, pinocchio::WORLD, pd.kvq, pd.kaq, pd.kav, pd.kad);
      Eigen::MatrixXd& Jl = pd.Jl3[i];
      for (int c = 0; c < nvm; ++c) {
        const Eigen::Vector3d JwLin = pd.kad.block<3, 1>(0, c);
        const Eigen::Vector3d JwAng = pd.kad.block<3, 1>(3, c);
        pd.Jwang3[i].col(c) = JwAng;
        Jl.col(c) = JwLin + JwAng.cross(p);
      }
      const int dim = ci.dim;
      const int nLin = std::min(dim, 3);
      const bool wrench = dim == 6;
      if (needs & (kFootPos | kFootDerivs)) {
        out.footPos[i] = p.head(nLin);
        out.footJ[i].topRows(nLin) = Jl.topRows(nLin);
        if (wrench) out.footJ[i].bottomRows(3) = pd.Jwang3[i];
      }
      if (needs & kFootVel) {
        out.footVel[i].head(nLin) = pdot.head(nLin);
        if (wrench) out.footVel[i].tail<3>() = om;
      }
      if (needs & kFootAcc) {
        out.footAcc[i].head(nLin) = pddot.head(nLin);
        if (wrench) out.footAcc[i].tail<3>() = al;
      }
      if (needs & kFootDerivs) {
        for (int c = 0; c < nvm; ++c) {
          const Eigen::Vector3d vqLin = pd.kvq.block<3, 1>(0, c);
          const Eigen::Vector3d vqAng = pd.kvq.block<3, 1>(3, c);
          pd.dvel3.col(c) = vqLin + vqAng.cross(p) + om.cross(Eigen::Vector3d(Jl.col(c)));
        }
        out.dvel_dq[i].topRows(nLin) = pd.dvel3.topRows(nLin);
        for (int c = 0; c < nvm; ++c) {
          const Eigen::Vector3d aqLin = pd.kaq.block<3, 1>(0, c);
          const Eigen::Vector3d aqAng = pd.kaq.block<3, 1>(3, c);
          const Eigen::Vector3d vqAng = pd.kvq.block<3, 1>(3, c);
          const Eigen::Vector3d daccQ = aqLin + aqAng.cross(p) + al.cross(Eigen::Vector3d(Jl.col(c))) +
                                        vqAng.cross(pdot) + om.cross(Eigen::Vector3d(pd.dvel3.col(c)));
          out.dacc_dq[i].col(c).head(nLin) = daccQ.head(nLin);
          const Eigen::Vector3d avLin = pd.kav.block<3, 1>(0, c);
          const Eigen::Vector3d avAng = pd.kav.block<3, 1>(3, c);
          const Eigen::Vector3d JwAng = pd.Jwang3[i].col(c);
          const Eigen::Vector3d daccV =
              avLin + avAng.cross(p) + JwAng.cross(pdot) + om.cross(Eigen::Vector3d(Jl.col(c)));
          out.dacc_dv[i].col(c).head(nLin) = daccV.head(nLin);
        }
        if (wrench) {
          out.dvel_dq[i].bottomRows(3) = pd.kvq.bottomRows(3);
          out.dacc_dq[i].bottomRows(3) = pd.kaq.bottomRows(3);
          out.dacc_dv[i].bottomRows(3) = pd.kav.bottomRows(3);
        }
      }
      if (wrench && (needs & kFootRot)) {
        const Eigen::Matrix3d R = (data.oMi[j] * ci.placement).rotation();
        double theta;
        const Eigen::Vector3d r = pinocchio::log3(R, theta);
        out.footRotLog[i] = r;
        Eigen::Matrix3d Jlog;
        pinocchio::Jlog3(theta, r, Jlog);  // maps LOCAL angular velocity
        out.footRotJ[i].noalias() = Jlog * R.transpose() * pd.Jwang3[i];
      }
    }
  }

  // --- monitor frames and capsule endpoints (position + Jacobian only) --------------
  // Must run before the fext RNEA pass below, which clobbers data.J.
  const bool wantMon = (needs & kMonitorPos) && !im.monitors.empty();
  const bool wantCap = (needs & kCapsulePos) && !im.capsulePoints.empty();
  if (wantMon || wantCap) {
    pinocchio::computeJointJacobians(model, data, q);
    if (wantMon) evalPointFrames(model, data, im.monitors, pd.J6, out.monPos, out.monJ);
    if (wantCap) evalPointFrames(model, data, im.capsulePoints, pd.J6, out.capPos, out.capJ);
  }
  if (needs & kComPos) {
    pinocchio::centerOfMass(model, data, q, false);
    pinocchio::jacobianCenterOfMass(model, data, q, false);
    out.comPos[0] = data.com[0];
    out.comJ[0] = data.Jcom;
  }

  // --- contact-folded dynamics residual ----------------------------------------------
  if (needs & kRnea) {
    out.gdyn = out.tau;
    int o = 0;
    for (const ContactRef& cr : cs.active) {
      out.gdyn.noalias() -= out.footJ[static_cast<size_t>(cr.index)].transpose() * lam.segment(o, cr.dim);
      o += cr.dim;
    }
  }
  if (needs & kRneaDerivs) {
    if (cs.active.empty()) {
      out.dgdyn_dq = out.dtau_dq;
    } else {
      // d(Jc' lam)/dq through computeRNEADerivatives with the contact forces as external
      // forces, plus the correction for fext being constant in the LOCAL joint frames
      // while lam is constant in the WORLD frame: Jl3' (Jw_ang.col(j) x lam3).
      std::fill(pd.fext.begin(), pd.fext.end(), pinocchio::Force::Zero());
      pd.corr.setZero();
      const int nvm = model.nv;
      int o = 0;
      for (const ContactRef& cr : cs.active) {
        const auto idx = static_cast<size_t>(cr.index);
        const BodyPoint& ci = im.contacts[idx];
        if (cr.dim <= 3) {
          Eigen::Vector3d lam3 = Eigen::Vector3d::Zero();
          lam3.head(cr.dim) = lam.segment(o, cr.dim);
          const pinocchio::Force fO(lam3, pd.p3[idx].cross(lam3));
          pd.fext[ci.joint] += data.oMi[ci.joint].actInv(fO);
          for (int c = 0; c < nvm; ++c) {
            const Eigen::Vector3d JwAng = pd.Jwang3[idx].col(c);
            pd.corr.col(c).noalias() += pd.Jl3[idx].transpose() * JwAng.cross(lam3);
          }
        } else {
          const Eigen::Vector3d f = lam.segment<3>(o);
          const Eigen::Vector3d tq = lam.segment<3>(o + 3);
          const pinocchio::Force fO(f, pd.p3[idx].cross(f) + tq);
          pd.fext[ci.joint] += data.oMi[ci.joint].actInv(fO);
          for (int c = 0; c < nvm; ++c) {
            const Eigen::Vector3d JwAng = pd.Jwang3[idx].col(c);
            pd.corr.col(c).noalias() += pd.Jl3[idx].transpose() * JwAng.cross(f);
            pd.corr.col(c).noalias() += pd.Jwang3[idx].transpose() * JwAng.cross(tq);
          }
        }
        o += cr.dim;
      }
      pinocchio::computeRNEADerivatives(model, data, q, v, a, pd.fext);
      out.dgdyn_dq = data.dtau_dq + pd.corr;
    }
  }
  if (needs & kImpactImpulse) {
    for (size_t s = 0; s < im.coneSets.size(); ++s)
      impactConeImpulse(pd, static_cast<int>(s), q, v, out.impactLam[s], &out.impactLamDq[s], &out.impactLamDv[s]);
  }
}

void RobotModel::evalNode(ModelData& pd, const Eigen::Ref<const Eigen::VectorXd>& q,
                          const Eigen::Ref<const Eigen::VectorXd>& v, const Eigen::Ref<const Eigen::VectorXd>& a,
                          unsigned needs, ModelEvalCache& out) const {
  evalNode(pd, q, v, a, Eigen::VectorXd(), ContactSet{}, needs, out);
}

// ------------------------------------------------------------------------------------
// kinematics helpers
// ------------------------------------------------------------------------------------

void RobotModel::framePositions(ModelData& pd, const Eigen::Ref<const Eigen::VectorXd>& q,
                                std::vector<Eigen::Vector3d>& contactPos) const {
  pinocchio::forwardKinematics(impl_->model, pd.data, q);
  contactPos.resize(impl_->contacts.size());
  for (size_t i = 0; i < impl_->contacts.size(); ++i) {
    const BodyPoint& bp = impl_->contacts[i];
    contactPos[i] = (pd.data.oMi[bp.joint] * bp.placement).translation();
  }
}

Eigen::Matrix3d RobotModel::contactRotation(ModelData& pd, const Eigen::Ref<const Eigen::VectorXd>& q,
                                            int contactIdx) const {
  pinocchio::forwardKinematics(impl_->model, pd.data, q);
  const BodyPoint& bp = impl_->contacts.at(static_cast<size_t>(contactIdx));
  return (pd.data.oMi[bp.joint] * bp.placement).rotation();
}

void RobotModel::capsuleKinematics(ModelData& pd, const Eigen::Ref<const Eigen::VectorXd>& q,
                                   std::vector<Eigen::Vector3d>& pos, std::vector<Eigen::MatrixXd>& J) const {
  const size_t n = impl_->capsulePoints.size();
  if (pos.size() != n) pos.assign(n, Eigen::Vector3d::Zero());
  if (J.size() != n) J.assign(n, Eigen::MatrixXd::Zero(3, nv()));
  for (size_t i = 0; i < n; ++i)
    if (J[i].rows() != 3 || J[i].cols() != nv()) J[i].setZero(3, nv());
  if (n == 0) return;
  pinocchio::computeJointJacobians(impl_->model, pd.data, q);
  evalPointFrames(impl_->model, pd.data, impl_->capsulePoints, pd.J6, pos, J);
}

Eigen::VectorXd RobotModel::inverseDynamicsTorque(ModelData& pd, const Eigen::Ref<const Eigen::VectorXd>& q,
                                                  const Eigen::Ref<const Eigen::VectorXd>& v,
                                                  const Eigen::Ref<const Eigen::VectorXd>& a,
                                                  const Eigen::Ref<const Eigen::VectorXd>& lam,
                                                  const ContactSet& cs) const {
  ModelEvalCache c;
  evalNode(pd, q, v, a, lam, cs, kRnea, c);
  Eigen::VectorXd u(nu());
  for (int j = 0; j < nu(); ++j) u(j) = c.gdyn(impl_->actIdx[static_cast<size_t>(j)]);
  return u;
}

// ------------------------------------------------------------------------------------
// plastic impacts
// ------------------------------------------------------------------------------------

// impulseDynamics / computeImpulseDynamicsDerivatives are tagged unsupported
// ([[deprecated]]) in pinocchio 3.9 but are the exact one-shot damped solve this
// method relies on.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

void RobotModel::impactVelocity(ModelData& pd, int setId, const Eigen::Ref<const Eigen::VectorXd>& q,
                                const Eigen::Ref<const Eigen::VectorXd>& v, Eigen::VectorXd& vplus) const {
  const auto s = static_cast<size_t>(setId);
  if (s >= pd.impactData.size()) fail("impact set registered after makeData()");
  pinocchio::Data& d = pd.impactData[s];
  const pinocchio::ProximalSettings settings(1e-12, impactProximalDamping(), 1);
  pinocchio::impulseDynamics(impl_->model, d, q, v, impl_->impactModels[s], pd.contactData[s], 0.0, settings);
  vplus = d.dq_after;
}

void RobotModel::impactDerivatives(ModelData& pd, int setId, const Eigen::Ref<const Eigen::VectorXd>& q,
                                   const Eigen::Ref<const Eigen::VectorXd>& v, Eigen::MatrixXd& dvdq,
                                   Eigen::MatrixXd& dvdv, Eigen::VectorXd& vplus) const {
  const auto s = static_cast<size_t>(setId);
  if (s >= pd.impactData.size()) fail("impact set registered after makeData()");
  pinocchio::Data& d = pd.impactData[s];
  // mu = 1e-9 damps the Delassus diagonal; max_iter = 1 keeps the one-shot solve the
  // derivatives assume.
  const pinocchio::ProximalSettings settings(1e-12, impactProximalDamping(), 1);
  const double rCoeff = 0.0;  // plastic impact
  pinocchio::impulseDynamics(impl_->model, d, q, v, impl_->impactModels[s], pd.contactData[s], rCoeff, settings);
  vplus = d.dq_after;
  // computeImpulseDynamicsDerivatives resets data.dtau_dq but not data.dvc_dq, whose
  // kinematic pass writes only the ancestor-joint columns.
  d.dvc_dq.setZero();
  pinocchio::computeImpulseDynamicsDerivatives(impl_->model, d, impl_->impactModels[s], pd.contactData[s], rCoeff,
                                               settings);
  dvdq = d.ddq_dq;
  dvdv = Eigen::MatrixXd::Identity(nv(), nv()) + d.ddq_dv;
}

void RobotModel::impactConeImpulse(ModelData& pd, int coneId, const Eigen::Ref<const Eigen::VectorXd>& q,
                                   const Eigen::Ref<const Eigen::VectorXd>& v, Eigen::VectorXd& lam,
                                   Eigen::MatrixXd* dLdq, Eigen::MatrixXd* dLdv) const {
  const auto s = static_cast<size_t>(coneId);
  if (s >= pd.coneData.size()) fail("impact-cone set registered after makeData()");
  pinocchio::Data& d = pd.coneData[s];
  const pinocchio::ProximalSettings settings(1e-12, impactProximalDamping(), 1);
  const double rCoeff = 0.0;
  pinocchio::impulseDynamics(impl_->model, d, q, v, impl_->coneModels[s], pd.coneContactData[s], rCoeff, settings);
  lam = d.impulse_c;
  if (dLdq != nullptr || dLdv != nullptr) {
    d.dvc_dq.setZero();
    pinocchio::computeImpulseDynamicsDerivatives(impl_->model, d, impl_->coneModels[s], pd.coneContactData[s], rCoeff,
                                                 settings);
    if (dLdq != nullptr) *dLdq = d.dlambda_dq;
    if (dLdv != nullptr) *dLdv = d.dlambda_dv;
  }
}

#pragma GCC diagnostic pop

}  // namespace leap
