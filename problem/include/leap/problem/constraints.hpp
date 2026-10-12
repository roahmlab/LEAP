#pragma once

// The constraint catalogue. Every term is a small dense consumer of the ModelEvalCache
// (which already folds the contact forces into the dynamics residual), so nothing here
// is robot-specific. Inequality rows read g <= 0.
//
//   dynamics      DynamicsConstraint (all nv rows), FloatingBaseDynamicsConstraint (the
//                 unactuated rows: torque elimination), SelectedRowsDynamicsConstraint
//                 (a subset of rows: planar robots described by a 3-D URDF)
//   contact frame FramePosition / FramePositionComponent / FramePositionHalfspace,
//                 FrameVelocity / FrameVelocityBound, FrameAcceleration, FrameOrientation,
//                 ContactHeightLowerBound
//   keep-out      MonitorHeightLowerBound (every monitor frame above a floor)
//   linear        Pin, PairEquality, LowerBound, Box (any field), MotorEnvelope, RowSlice
//   force         FrictionCone, ContactWrenchBox (CoP / yaw), ImpactCone (on the plastic
//                 impulse at the pre-impact node)
// Obstacles and self-collision live in obstacles.hpp / self_collision.hpp.

#include <memory>
#include <string>
#include <vector>

#include "leap/problem/constraint.hpp"

namespace leap {

// --- dynamics ---------------------------------------------------------------------------

// M(q) a + C(q, v) v + g(q) - sum_c Jc' lam_c - B u = 0, all nv rows.
class DynamicsConstraint final : public Constraint {
public:
  DynamicsConstraint(const RobotModel& m, ContactSet cs);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return nv_; }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override;
  void value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;
  const ContactSet& contacts() const { return cs_; }
  bool readsOneSidedFields() const override { return true; }

private:
  ContactSet cs_;
  std::vector<int> actIdx_;
  int nv_;
  std::string name_;
};

// The unactuated rows of the dynamics only. The actuated rows then DEFINE the torque
// u = [M a + h - Jc' lam]_act (inverse-dynamics torque elimination, see the layout).
class FloatingBaseDynamicsConstraint final : public Constraint {
public:
  FloatingBaseDynamicsConstraint(const RobotModel& m, ContactSet cs);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(unactIdx_.size()); }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override;
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  const ContactSet& contacts() const { return cs_; }
  bool readsOneSidedFields() const override { return true; }

  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  ContactSet cs_;
  std::vector<int> unactIdx_;
  std::string name_;
};

// The dynamics restricted to a subset of its rows (the coordinates a planar guide does
// not hold). Actuated rows among them subtract their u.
class SelectedRowsDynamicsConstraint final : public Constraint {
public:
  SelectedRowsDynamicsConstraint(const RobotModel& m, ContactSet cs, std::vector<int> selectedRows);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(rows_.size()); }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override;
  void value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  const ContactSet& contacts() const { return cs_; }
  const std::vector<int>& selectedRows() const { return rows_; }
  bool readsOneSidedFields() const override { return true; }

  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  ContactSet cs_;
  std::vector<int> rows_;
  std::vector<int> actSlot_;  // physical row -> u index or -1
  std::string name_;
};

// --- contact frames -----------------------------------------------------------------------

// footPos[contact] - ref = 0 (ref has min(dim, 3) rows or fewer: the leading rows).
class FramePositionConstraint final : public Constraint {
public:
  FramePositionConstraint(int contactIdx, Eigen::VectorXd ref, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(ref_.size()); }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override { return kFootPos; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;
  int contact() const { return contact_; }
  const Eigen::VectorXd& reference() const { return ref_; }

private:
  int contact_;
  Eigen::VectorXd ref_;
  std::string name_;
};

// One component of footPos: footPos[contact](comp) - ref = 0.
class FramePositionComponentConstraint final : public Constraint {
public:
  FramePositionComponentConstraint(int contactIdx, int component, double ref, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 1; }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override { return kFootPos; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  int contact_, component_;
  double ref_;
  std::string name_;
};

// normal . footPos[contact] - bound <= 0.
class FramePositionHalfspaceConstraint final : public Constraint {
public:
  FramePositionHalfspaceConstraint(int contactIdx, Eigen::Vector3d normal, double bound, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 1; }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kFootPos; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  int contact_;
  Eigen::Vector3d normal_;
  double bound_;
  std::string name_;
};

// Contact-frame height above a floor: lb - footPos[contact].z <= 0.
class ContactHeightLowerBoundConstraint final : public Constraint {
public:
  ContactHeightLowerBoundConstraint(int contactIdx, double lb, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 1; }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kFootPos; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  int contact_;
  double lb_;
  std::string name_;
};

// footVel[contact].head(n) - ref = 0 (no-slip; n = ref.size(), 6 rows pin [pdot; omega]).
class FrameVelocityConstraint final : public Constraint {
public:
  FrameVelocityConstraint(int contactIdx, Eigen::VectorXd ref, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(ref_.size()); }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override { return kFootVel | kFootPos | kFootDerivs; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;
  int contact() const { return contact_; }

private:
  int contact_;
  Eigen::VectorXd ref_;
  std::string name_;
};

// sign * (footVel[contact](comp) - bound) <= 0.
class FrameVelocityBoundConstraint final : public Constraint {
public:
  FrameVelocityBoundConstraint(int contactIdx, int comp, double sign, double bound, std::string name, int frameDim = 3);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 1; }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kFootVel | kFootPos | kFootDerivs; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  int contact_, comp_;
  double sign_, bound_;
  int frameDim_;
  std::string name_;
};

// footAcc[contact].head(n) - ref = 0.
class FrameAccelerationConstraint final : public Constraint {
public:
  FrameAccelerationConstraint(int contactIdx, Eigen::VectorXd ref, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(ref_.size()); }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override { return kFootAcc | kFootPos | kFootDerivs; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

  bool readsOneSidedFields() const override { return true; }

private:
  int contact_;
  Eigen::VectorXd ref_;
  std::string name_;
};

// log3(R_wf) - ref = 0 for a dim-6 contact (ref = (0, 0, yaw) is "sole flat on z-up ground").
class FrameOrientationConstraint final : public Constraint {
public:
  FrameOrientationConstraint(int contactIdx, Eigen::Vector3d ref, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 3; }
  Sense sense() const override { return Sense::Equality; }
  unsigned modelNeeds() const override { return kFootRot; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  int contact_;
  Eigen::Vector3d ref_;
  std::string name_;
};

// Every monitor frame above its floor: lb_i - monPos_i.z <= 0 (one row per monitor).
class MonitorHeightLowerBoundConstraint final : public Constraint {
public:
  MonitorHeightLowerBoundConstraint(const RobotModel& m, Eigen::VectorXd lb, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return nMon_; }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kMonitorPos; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  int nMon_, nContacts_;
  Eigen::VectorXd lb_;
  std::string name_;
};

// --- linear terms on the physical vector ----------------------------------------------------

// z[idx_i] - target_i = 0 (idx in [q; v; a; lam; u] coordinates).
class PinConstraint final : public Constraint {
public:
  PinConstraint(std::vector<int> idx, Eigen::VectorXd target, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(idx_.size()); }
  Sense sense() const override { return Sense::Equality; }
  void value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims&, const ModelEvalCache&, MatRef J) const override;
  bool readsOnlyForce() const override { return forceOnly_; }
  bool readsOneSidedFields() const override { return forceOnly_ || oneSided_; }
  const std::vector<int>& indices() const { return idx_; }
  const Eigen::VectorXd& target() const { return target_; }
  // Mark a pin whose indices all lie in the lam block (set by the builder so the term
  // may be attached to force control points).
  void setForceOnly(bool on) { forceOnly_ = on; }
  // Mark a pin on a one-sided field (a, lam or u): at an element junction the row then
  // lives on both junction nodes instead of the left node only.
  void setOneSided(bool on) { oneSided_ = on; }

private:
  std::vector<int> idx_;
  Eigen::VectorXd target_;
  std::string name_;
  bool forceOnly_ = false;
  bool oneSided_ = false;
};

// z[left_i] - z[right_i] = 0.
class PairEqualityConstraint final : public Constraint {
public:
  PairEqualityConstraint(std::vector<int> left, std::vector<int> right, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(left_.size()); }
  Sense sense() const override { return Sense::Equality; }
  void value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims&, const ModelEvalCache&, MatRef J) const override;

private:
  std::vector<int> left_, right_;
  std::string name_;
};

// lb_i - z[idx_i] <= 0.
class LowerBoundConstraint final : public Constraint {
public:
  LowerBoundConstraint(std::vector<int> idx, Eigen::VectorXd lb, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(idx_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  void value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims&, const ModelEvalCache&, MatRef J) const override;
  bool readsOnlyForce() const override { return forceOnly_; }
  bool readsOneSidedFields() const override { return forceOnly_; }
  void setForceOnly(bool on) { forceOnly_ = on; }

private:
  std::vector<int> idx_;
  Eigen::VectorXd lb_;
  std::string name_;
  bool forceOnly_ = false;
};

enum class Field { Q, V, A, Lam, U };
int fieldOffset(Field f, const NodeDims& d);
int fieldSize(Field f, const NodeDims& d);

// lb_i <= field[comp_i] <= ub_i as PHR rows: a lower row (lb - x <= 0) when lb is
// finite and an upper row (x - ub <= 0) when ub is finite.
class BoxConstraint final : public Constraint {
public:
  BoxConstraint(Field field, std::vector<int> comps, Eigen::VectorXd lb, Eigen::VectorXd ub, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(rows_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  void value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, MatRef J) const override;
  bool readsOnlyForce() const override { return field_ == Field::Lam; }
  bool readsOneSidedFields() const override { return field_ == Field::U || field_ == Field::A || field_ == Field::Lam; }
  Field field() const { return field_; }
  struct Row {
    int comp;
    double sign;   // -1 lower (lb - x), +1 upper (x - ub)
    double bound;
  };
  const std::vector<Row>& boundRows() const { return rows_; }

private:
  Field field_;
  std::vector<Row> rows_;
  std::string name_;
};

// Per-actuated-joint motor envelope (speed box, torque box and, with derating, the
// sloped motoring caps), every row divided by its limit.
class MotorEnvelopeConstraint final : public Constraint {
public:
  MotorEnvelopeConstraint(const RobotModel& m, const MotorSaturationConfig& cfg, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(actIdx_.size()) * (derate_ ? 6 : 4); }
  Sense sense() const override { return Sense::Inequality; }
  void value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, MatRef J) const override;
  bool readsOneSidedFields() const override { return true; }

private:
  std::vector<int> actIdx_;
  Eigen::VectorXd tauMax_, vMax_;
  bool derate_;
  double knee_;
  std::string name_;
};

// One row of another constraint as a standalone term (keeps the inner name).
class RowSliceConstraint final : public Constraint {
public:
  RowSliceConstraint(std::shared_ptr<const Constraint> inner, int row);
  const std::string& name() const override { return inner_->name(); }
  int rows(const NodeDims&) const override { return 1; }
  Sense sense() const override { return inner_->sense(); }
  unsigned modelNeeds() const override { return inner_->modelNeeds(); }
  void value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;
  bool readsOnlyForce() const override { return inner_->readsOnlyForce(); }
  bool readsOneSidedFields() const override { return inner_->readsOneSidedFields(); }

private:
  std::shared_ptr<const Constraint> inner_;
  int row_;
};

// --- contact forces and impulses --------------------------------------------------------------

// Smoothed Coulomb cone on every active contact's force (the leading 3 components of a
// wrench): sqrt(fx^2 + fy^2 + eps^2) - eps - mu fz <= 0 and -fz <= 0.
class FrictionConeConstraint final : public Constraint {
public:
  FrictionConeConstraint(const ContactSet& cs, double mu, double eps, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 2 * static_cast<int>(lamOff_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  void value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;
  bool readsOnlyForce() const override { return true; }
  bool readsOneSidedFields() const override { return true; }
  double mu() const { return mu_; }
  double epsilon() const { return eps_; }

private:
  static constexpr double kApexTol = 1e-9;
  std::vector<int> lamOff_;
  double mu_, eps_;
  std::string name_;
};

// Symmetric wrench-component bounds tied to the normal force of dim-6 contacts:
// +-lam(o + c) - k fz <= 0 (CoP rectangle with comps {4, 3}, yaw bound with {5}).
class ContactWrenchBoxConstraint final : public Constraint {
public:
  ContactWrenchBoxConstraint(const ContactSet& cs, std::vector<int> comps, std::vector<double> coefs, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 2 * static_cast<int>(comps_.size()) * static_cast<int>(lamOff_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  void value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, MatRef J) const override;
  bool readsOnlyForce() const override { return true; }
  bool readsOneSidedFields() const override { return true; }

private:
  std::vector<int> comps_;
  std::vector<double> coefs_;
  std::vector<int> lamOff_;
  std::string name_;
};

// Friction-cone admissibility of the plastic impact impulse at the pre-impact node.
class ImpactConeConstraint final : public Constraint {
public:
  ImpactConeConstraint(int coneId, std::vector<int> lamOff, double mu, double eps, std::string name);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return 2 * static_cast<int>(lamOff_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kImpactImpulse; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  static constexpr double kApexTol = 1e-9;
  int coneId_;
  std::vector<int> lamOff_;
  double mu_, eps_;
  std::string name_;
};

}  // namespace leap
