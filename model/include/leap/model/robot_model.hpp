#pragma once

// RobotModel: a pinocchio model built from a URDF plus the robot description (contact
// table, monitor frames, collision capsules, actuation), behind one evaluation call.
//
// evalNode(q, v, a, lam, contacts, needs, cache) runs the rigid-body algorithms once per
// node for the union of quantities the node's constraints and costs declare (the
// ModelNeeds bit mask) and fills a ModelEvalCache. Constraints are then small dense
// consumers of the cache. The dynamics residual is folded here:
//     gdyn = M(q) a + C(q, v) v + g(q) [+ motor loss] - sum_c Jc(q)^T lam_c,
// so that no constraint has to know how a contact wrench enters the equations.
//
// The floating base is six explicit 1-DOF joints (px, py, pz, rx, ry, rz), so q is a
// vector space and nq == nv. Frame quantities are WORLD-aligned at the body-fixed
// contact point: for a dim-6 contact footVel = [pdot; omega], footAcc = [pddot; alpha],
// footJ = [Jlin; Jang] and footPos stays position-only, so footJ.topRows(3) is always
// d(footPos)/dq.

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/model/contact_set.hpp"
#include "leap/model/robot_description.hpp"

namespace leap {

enum ModelNeeds : unsigned {
  kMass = 1u << 0,           // M(q)
  kRnea = 1u << 1,           // tau = M a + C v + g, and gdyn = tau - sum Jc' lam_c
  kRneaDerivs = 1u << 2,     // dtau/dq, dtau/dv, dgdyn/dq (implies M)
  kFootPos = 1u << 3,        // contact-frame position + Jacobian, all contacts
  kFootVel = 1u << 4,        // contact-frame velocity
  kFootAcc = 1u << 5,        // contact-frame acceleration
  kFootDerivs = 1u << 6,     // dvel/dq, dacc/dq, dacc/dv
  kMonitorPos = 1u << 7,     // monitor-frame world position + Jacobian
  kComPos = 1u << 8,         // centre of mass + Jacobian
  kFootRot = 1u << 9,        // contact-frame orientation log3 + Jacobian (dim-6 contacts)
  kCapsulePos = 1u << 10,    // collision-capsule endpoint positions + Jacobians
  kImpactImpulse = 1u << 11, // plastic-impact impulse (+ derivatives) of every cone set
};

struct ModelEvalCache {
  // Dynamics. gdyn / dgdyn_dq fold in the active contacts' -Jc' lam_c term.
  Eigen::VectorXd tau, gdyn;
  Eigen::MatrixXd dtau_dq, dtau_dv, M, dgdyn_dq;
  // Contact frames, indexed by the model's contact table (see the header comment).
  std::vector<Eigen::VectorXd> footPos, footVel, footAcc;
  std::vector<Eigen::MatrixXd> footJ, dvel_dq, dacc_dq, dacc_dv;
  // Orientation of dim-6 contacts: log3(R_wf) and its q-Jacobian.
  std::vector<Eigen::Vector3d> footRotLog;
  std::vector<Eigen::MatrixXd> footRotJ;
  // Monitor frames (position only).
  std::vector<Eigen::Vector3d> monPos;
  std::vector<Eigen::MatrixXd> monJ;
  // Collision-capsule endpoints: entries 2i and 2i+1 are endpoints A and B of capsule i.
  std::vector<Eigen::Vector3d> capPos;
  std::vector<Eigen::MatrixXd> capJ;
  // Plastic-impact impulse of every registered cone set in world-aligned axes
  // (3 rows per point contact, 6 per flat foot, force first), and its partials.
  std::vector<Eigen::VectorXd> impactLam;
  std::vector<Eigen::MatrixXd> impactLamDq, impactLamDv;
  // Centre of mass (single entry).
  std::vector<Eigen::Vector3d> comPos;
  std::vector<Eigen::MatrixXd> comJ;
};

// Per-thread pinocchio scratch (opaque: pinocchio stays out of this header). Owned
// through ModelDataPtr, whose deleter is defined next to the class.
class ModelData;
struct ModelDataDeleter {
  void operator()(ModelData* p) const;
};
using ModelDataPtr = std::unique_ptr<ModelData, ModelDataDeleter>;

class RobotModel {
public:
  explicit RobotModel(const RobotDescription& rd);  // throws std::runtime_error
  ~RobotModel();
  RobotModel(RobotModel&&) noexcept;
  RobotModel& operator=(RobotModel&&) noexcept;
  RobotModel(const RobotModel&) = delete;
  RobotModel& operator=(const RobotModel&) = delete;

  const RobotDescription& description() const;
  int nq() const;
  int nv() const;
  int nu() const;
  // Joint names in q/v order (every joint is 1-DOF, so entry i labels q(i) and v(i)).
  const std::vector<std::string>& jointNames() const;
  int jointIndex(const std::string& name) const;  // index into q; throws if unknown
  bool hasJoint(const std::string& name) const;
  bool hasFrame(const std::string& name) const;

  // Actuation: B is nv x nu with column j = e_{actuatedIdx()[j]}.
  const Eigen::MatrixXd& B() const;
  const std::vector<int>& actuatedIdx() const;
  const std::vector<int>& unactuatedIdx() const;
  const BaseRotationLayout& baseRotation() const;

  // Limits in u order (size nu) and in q/v order (size nq / nv). A joint the URDF
  // leaves unbounded carries pinocchio's large sentinel. Effort and velocity limits are
  // the URDF's unless the robot yaml gives explicit `actuation.effort_limits` /
  // `velocity_limits` lists (actuated order), which override them in both orders.
  Eigen::VectorXd actuatedEffortLimit() const;
  Eigen::VectorXd actuatedVelocityLimit() const;
  Eigen::VectorXd lowerPositionLimit() const;
  Eigen::VectorXd upperPositionLimit() const;
  Eigen::VectorXd velocityLimit() const;
  double totalMass() const;
  const Eigen::Vector3d& gravity() const;

  // Motor model (joint-side passive loss, see RobotDescription).
  const std::string& motorModelType() const;
  Eigen::VectorXd motorLoss(const Eigen::Ref<const Eigen::VectorXd>& v, bool smooth = true) const;

  // Contact table.
  int nContacts() const;
  int contactDim(int contactIdx) const;
  const std::string& contactName(int contactIdx) const;
  int contactIndex(const std::string& name) const;  // throws if unknown
  ContactSet contactSet(const std::vector<std::string>& names) const;
  ContactSet allContacts() const;
  NodeDims nodeDims(const ContactSet& cs) const;

  // Monitor table.
  int nMonitors() const;
  const std::string& monitorName(int monitorIdx) const;

  // Collision capsules (resolved endpoints: a `toFrame` has become a local offset).
  int nCapsules() const;
  const std::string& capsuleName(int capsuleIdx) const;
  int capsuleIndex(const std::string& name) const;
  const std::string& capsuleFrame(int capsuleIdx) const;
  double capsuleRadius(int capsuleIdx) const;
  Eigen::Vector3d capsuleLocalA(int capsuleIdx) const;
  Eigen::Vector3d capsuleLocalB(int capsuleIdx) const;

  // Plastic-impact sets (the contacts that become active at a touchdown). Register
  // every set BEFORE makeData(): the per-thread data is sized per registered set.
  int ensureImpactSet(const ContactSet& cs) const;
  // The same contacts in world-aligned axes, used only to read the impact impulse.
  int ensureImpactConeSet(const ContactSet& cs) const;
  int numImpactConeSets() const;
  int impactConeLamOffset(int coneId, int contactIndex) const;
  const ContactSet& impactSetContacts(int setId) const;
  const ContactSet& impactConeSetContacts(int coneId) const;
  // The proximal damping mu of the one-shot impulse solve (shared with the exact
  // Hessian provider, which differentiates the same damped map).
  static double impactProximalDamping() { return 1e-9; }

  ModelDataPtr makeData() const;
  void initCache(ModelEvalCache& out) const;

  void evalNode(ModelData& data, const Eigen::Ref<const Eigen::VectorXd>& q,
                const Eigen::Ref<const Eigen::VectorXd>& v, const Eigen::Ref<const Eigen::VectorXd>& a,
                const Eigen::Ref<const Eigen::VectorXd>& lam, const ContactSet& cs, unsigned needs,
                ModelEvalCache& out) const;
  // Contact-free convenience (gdyn = tau).
  void evalNode(ModelData& data, const Eigen::Ref<const Eigen::VectorXd>& q,
                const Eigen::Ref<const Eigen::VectorXd>& v, const Eigen::Ref<const Eigen::VectorXd>& a,
                unsigned needs, ModelEvalCache& out) const;

  // Forward kinematics only: world position of the contact frames (and, for the
  // capsules, their endpoints) at q. Used by seeds and inverse kinematics.
  void framePositions(ModelData& data, const Eigen::Ref<const Eigen::VectorXd>& q,
                      std::vector<Eigen::Vector3d>& contactPos) const;
  Eigen::Matrix3d contactRotation(ModelData& data, const Eigen::Ref<const Eigen::VectorXd>& q,
                                  int contactIdx) const;
  // Capsule endpoint world positions + 3 x nv Jacobians at q, into caller storage.
  void capsuleKinematics(ModelData& data, const Eigen::Ref<const Eigen::VectorXd>& q,
                         std::vector<Eigen::Vector3d>& pos, std::vector<Eigen::MatrixXd>& J) const;

  // Plastic impact v+(q, v-) = v- + M^-1 J' Lambda with J v+ = 0 (damped Delassus
  // solve), and its exact partials.
  void impactVelocity(ModelData& data, int setId, const Eigen::Ref<const Eigen::VectorXd>& q,
                      const Eigen::Ref<const Eigen::VectorXd>& v, Eigen::VectorXd& vplus) const;
  void impactDerivatives(ModelData& data, int setId, const Eigen::Ref<const Eigen::VectorXd>& q,
                         const Eigen::Ref<const Eigen::VectorXd>& v, Eigen::MatrixXd& dvdq,
                         Eigen::MatrixXd& dvdv, Eigen::VectorXd& vplus) const;
  void impactConeImpulse(ModelData& data, int coneId, const Eigen::Ref<const Eigen::VectorXd>& q,
                         const Eigen::Ref<const Eigen::VectorXd>& v, Eigen::VectorXd& lam,
                         Eigen::MatrixXd* dLdq = nullptr, Eigen::MatrixXd* dLdv = nullptr) const;

  // Rigid-body quantities for seeds and reports (no cache): inverse dynamics with the
  // contact forces folded in, u = [tau - sum Jc' lam_c]_actuated.
  Eigen::VectorXd inverseDynamicsTorque(ModelData& data, const Eigen::Ref<const Eigen::VectorXd>& q,
                                        const Eigen::Ref<const Eigen::VectorXd>& v,
                                        const Eigen::Ref<const Eigen::VectorXd>& a,
                                        const Eigen::Ref<const Eigen::VectorXd>& lam,
                                        const ContactSet& cs) const;

  struct Impl;  // private; the exact-Hessian provider and the tests reach pinocchio through it
  const Impl& impl() const { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace leap
