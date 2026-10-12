#include "leap/problem/constraints.hpp"

#include <cmath>
#include <stdexcept>

namespace leap {

namespace {

void requireHessians(const HessianScratch& s, const std::string& name) {
  if (s.hessians == nullptr)
    throw std::logic_error(name + ": exact contracted Hessian requested without a ModelHessians provider");
}

void packZ(const NodeQuantities& z, const NodeDims& d, HessianScratch& s) {
  s.z.resize(d.nP());
  z.toVector(d, s.z);
}

}  // namespace

// --- dynamics ---------------------------------------------------------------------------

DynamicsConstraint::DynamicsConstraint(const RobotModel& m, ContactSet cs)
    : cs_(std::move(cs)), actIdx_(m.actuatedIdx()), nv_(m.nv()), name_("dyn") {}

unsigned DynamicsConstraint::modelNeeds() const { return kRnea | kRneaDerivs | (cs_.nLam() > 0 ? kFootPos : 0u); }

void DynamicsConstraint::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache& c,
                               Eigen::Ref<Eigen::VectorXd> g) const {
  g = c.gdyn;
  for (size_t j = 0; j < actIdx_.size(); ++j) g(actIdx_[j]) -= z.u(static_cast<Eigen::Index>(j));
}

void DynamicsConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const {
  J.setZero();
  J.middleCols(d.offQ(), d.nq) = c.dgdyn_dq;
  J.middleCols(d.offV(), d.nv) = c.dtau_dv;
  J.middleCols(d.offA(), d.na) = c.M;
  int o = 0;
  for (const ContactRef& cr : cs_.active) {
    J.middleCols(d.offLam() + o, cr.dim) = -c.footJ[static_cast<size_t>(cr.index)].transpose();
    o += cr.dim;
  }
  for (size_t j = 0; j < actIdx_.size(); ++j) J(actIdx_[j], d.offU() + static_cast<int>(j)) = -1.0;
}

void DynamicsConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                              const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.hessians->addDynamicsHessian(d, cs_, s.z, w, H);
}

FloatingBaseDynamicsConstraint::FloatingBaseDynamicsConstraint(const RobotModel& m, ContactSet cs)
    : cs_(std::move(cs)), unactIdx_(m.unactuatedIdx()), name_("dyn") {}

unsigned FloatingBaseDynamicsConstraint::modelNeeds() const {
  return kRnea | kRneaDerivs | (cs_.nLam() > 0 ? kFootPos : 0u);
}

void FloatingBaseDynamicsConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                           Eigen::Ref<Eigen::VectorXd> g) const {
  for (size_t r = 0; r < unactIdx_.size(); ++r) g(static_cast<Eigen::Index>(r)) = c.gdyn(unactIdx_[r]);
}

void FloatingBaseDynamicsConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                              MatRef J) const {
  J.setZero();
  for (size_t r = 0; r < unactIdx_.size(); ++r) {
    const int rr = unactIdx_[r];
    const int i = static_cast<int>(r);
    J.row(i).segment(d.offQ(), d.nq) = c.dgdyn_dq.row(rr);
    J.row(i).segment(d.offV(), d.nv) = c.dtau_dv.row(rr);
    J.row(i).segment(d.offA(), d.na) = c.M.row(rr);
    int o = 0;
    for (const ContactRef& cr : cs_.active) {
      J.block(i, d.offLam() + o, 1, cr.dim) = -c.footJ[static_cast<size_t>(cr.index)].col(rr).transpose();
      o += cr.dim;
    }
  }
}

void FloatingBaseDynamicsConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                          const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  Eigen::VectorXd wFull = Eigen::VectorXd::Zero(d.nv);
  for (size_t r = 0; r < unactIdx_.size(); ++r) wFull(unactIdx_[r]) = w(static_cast<Eigen::Index>(r));
  s.hessians->addDynamicsHessian(d, cs_, s.z, wFull, H);
}

SelectedRowsDynamicsConstraint::SelectedRowsDynamicsConstraint(const RobotModel& m, ContactSet cs,
                                                               std::vector<int> selectedRows)
    : cs_(std::move(cs)), rows_(std::move(selectedRows)), actSlot_(static_cast<size_t>(m.nv()), -1), name_("dyn") {
  if (rows_.empty()) throw std::invalid_argument("SelectedRowsDynamicsConstraint: no rows selected");
  std::vector<char> seen(static_cast<size_t>(m.nv()), 0);
  for (int row : rows_) {
    if (row < 0 || row >= m.nv()) throw std::invalid_argument("SelectedRowsDynamicsConstraint: row out of range");
    if (seen[static_cast<size_t>(row)]) throw std::invalid_argument("SelectedRowsDynamicsConstraint: duplicate row");
    seen[static_cast<size_t>(row)] = 1;
  }
  const std::vector<int>& act = m.actuatedIdx();
  for (size_t j = 0; j < act.size(); ++j) actSlot_[static_cast<size_t>(act[j])] = static_cast<int>(j);
}

unsigned SelectedRowsDynamicsConstraint::modelNeeds() const {
  return kRnea | kRneaDerivs | (cs_.nLam() > 0 ? kFootPos : 0u);
}

void SelectedRowsDynamicsConstraint::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache& c,
                                           Eigen::Ref<Eigen::VectorXd> g) const {
  for (size_t r = 0; r < rows_.size(); ++r) {
    const int physical = rows_[r];
    g(static_cast<Eigen::Index>(r)) = c.gdyn(physical);
    const int slot = actSlot_[static_cast<size_t>(physical)];
    if (slot >= 0) g(static_cast<Eigen::Index>(r)) -= z.u(slot);
  }
}

void SelectedRowsDynamicsConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                              MatRef J) const {
  J.setZero();
  for (size_t r = 0; r < rows_.size(); ++r) {
    const int physical = rows_[r];
    const int i = static_cast<int>(r);
    J.row(i).segment(d.offQ(), d.nq) = c.dgdyn_dq.row(physical);
    J.row(i).segment(d.offV(), d.nv) = c.dtau_dv.row(physical);
    J.row(i).segment(d.offA(), d.na) = c.M.row(physical);
    int o = 0;
    for (const ContactRef& cr : cs_.active) {
      J.block(i, d.offLam() + o, 1, cr.dim) = -c.footJ[static_cast<size_t>(cr.index)].col(physical).transpose();
      o += cr.dim;
    }
    const int slot = actSlot_[static_cast<size_t>(physical)];
    if (slot >= 0) J(i, d.offU() + slot) = -1.0;
  }
}

void SelectedRowsDynamicsConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                          const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  Eigen::VectorXd wFull = Eigen::VectorXd::Zero(d.nv);
  for (size_t r = 0; r < rows_.size(); ++r) wFull(rows_[r]) = w(static_cast<Eigen::Index>(r));
  s.hessians->addDynamicsHessian(d, cs_, s.z, wFull, H);
}

// --- contact frames -----------------------------------------------------------------------

FramePositionConstraint::FramePositionConstraint(int contactIdx, Eigen::VectorXd ref, std::string name)
    : contact_(contactIdx), ref_(std::move(ref)), name_(std::move(name)) {}

void FramePositionConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                    Eigen::Ref<Eigen::VectorXd> g) const {
  g = c.footPos[static_cast<size_t>(contact_)].head(ref_.size()) - ref_;
}

void FramePositionConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const {
  J.setZero();
  J.middleCols(d.offQ(), d.nq) = c.footJ[static_cast<size_t>(contact_)].topRows(ref_.size());
}

void FramePositionConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                   const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.hessians->addFrameHessian(FrameTerm::Position, contact_, d, s.z, w, H);
}

FramePositionComponentConstraint::FramePositionComponentConstraint(int contactIdx, int component, double ref,
                                                                   std::string name)
    : contact_(contactIdx), component_(component), ref_(ref), name_(std::move(name)) {
  if (component_ < 0 || component_ >= 3) throw std::invalid_argument(name_ + ": position component must be in [0, 2]");
}

void FramePositionComponentConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                             Eigen::Ref<Eigen::VectorXd> g) const {
  g(0) = c.footPos[static_cast<size_t>(contact_)](component_) - ref_;
}

void FramePositionComponentConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                                MatRef J) const {
  J.setZero();
  J.row(0).segment(d.offQ(), d.nq) = c.footJ[static_cast<size_t>(contact_)].row(component_);
}

void FramePositionComponentConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.w3.setZero(3);
  s.w3(component_) = w(0);
  s.hessians->addFrameHessian(FrameTerm::Position, contact_, d, s.z, s.w3, H);
}

FramePositionHalfspaceConstraint::FramePositionHalfspaceConstraint(int contactIdx, Eigen::Vector3d normal, double bound,
                                                                   std::string name)
    : contact_(contactIdx), normal_(std::move(normal)), bound_(bound), name_(std::move(name)) {
  if (!normal_.allFinite() || normal_.norm() == 0.0 || !std::isfinite(bound_))
    throw std::invalid_argument(name_ + ": invalid position halfspace");
}

void FramePositionHalfspaceConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                             Eigen::Ref<Eigen::VectorXd> g) const {
  g(0) = normal_.dot(c.footPos[static_cast<size_t>(contact_)]) - bound_;
}

void FramePositionHalfspaceConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                                MatRef J) const {
  J.setZero();
  J.row(0).segment(d.offQ(), d.nq) = normal_.transpose() * c.footJ[static_cast<size_t>(contact_)].topRows(3);
}

void FramePositionHalfspaceConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d,
                                                            HessianScratch& s,
                                                            const Eigen::Ref<const Eigen::VectorXd>& w,
                                                            Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.w3 = w(0) * normal_;
  s.hessians->addFrameHessian(FrameTerm::Position, contact_, d, s.z, s.w3, H);
}

ContactHeightLowerBoundConstraint::ContactHeightLowerBoundConstraint(int contactIdx, double lb, std::string name)
    : contact_(contactIdx), lb_(lb), name_(std::move(name)) {}

void ContactHeightLowerBoundConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                              Eigen::Ref<Eigen::VectorXd> g) const {
  g(0) = lb_ - c.footPos[static_cast<size_t>(contact_)](2);
}

void ContactHeightLowerBoundConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                                 MatRef J) const {
  J.setZero();
  J.row(0).segment(d.offQ(), d.nq) = -c.footJ[static_cast<size_t>(contact_)].row(2);
}

void ContactHeightLowerBoundConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                             const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.w3.setZero(3);
  s.w3(2) = -w(0);
  s.hessians->addFrameHessian(FrameTerm::Position, contact_, d, s.z, s.w3, H);
}

FrameVelocityConstraint::FrameVelocityConstraint(int contactIdx, Eigen::VectorXd ref, std::string name)
    : contact_(contactIdx), ref_(std::move(ref)), name_(std::move(name)) {}

void FrameVelocityConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                    Eigen::Ref<Eigen::VectorXd> g) const {
  g = c.footVel[static_cast<size_t>(contact_)].head(ref_.size()) - ref_;
}

void FrameVelocityConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const {
  J.setZero();
  const auto f = static_cast<size_t>(contact_);
  J.middleCols(d.offQ(), d.nq) = c.dvel_dq[f].topRows(ref_.size());
  J.middleCols(d.offV(), d.nv) = c.footJ[f].topRows(ref_.size());
}

void FrameVelocityConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                   const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.hessians->addFrameHessian(FrameTerm::Velocity, contact_, d, s.z, w, H);
}

FrameVelocityBoundConstraint::FrameVelocityBoundConstraint(int contactIdx, int comp, double sign, double bound,
                                                           std::string name, int frameDim)
    : contact_(contactIdx), comp_(comp), sign_(sign), bound_(bound), frameDim_(frameDim), name_(std::move(name)) {}

void FrameVelocityBoundConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                         Eigen::Ref<Eigen::VectorXd> g) const {
  g(0) = sign_ * (c.footVel[static_cast<size_t>(contact_)](comp_) - bound_);
}

void FrameVelocityBoundConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                            MatRef J) const {
  J.setZero();
  const auto f = static_cast<size_t>(contact_);
  J.block(0, d.offQ(), 1, d.nq) = sign_ * c.dvel_dq[f].row(comp_);
  J.block(0, d.offV(), 1, d.nv) = sign_ * c.footJ[f].row(comp_);
}

void FrameVelocityBoundConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                        const Eigen::Ref<const Eigen::VectorXd>& w,
                                                        Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.w3.setZero(frameDim_);
  s.w3(comp_) = sign_ * w(0);
  s.hessians->addFrameHessian(FrameTerm::Velocity, contact_, d, s.z, s.w3, H);
}

FrameAccelerationConstraint::FrameAccelerationConstraint(int contactIdx, Eigen::VectorXd ref, std::string name)
    : contact_(contactIdx), ref_(std::move(ref)), name_(std::move(name)) {}

void FrameAccelerationConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                        Eigen::Ref<Eigen::VectorXd> g) const {
  g = c.footAcc[static_cast<size_t>(contact_)].head(ref_.size()) - ref_;
}

void FrameAccelerationConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                           MatRef J) const {
  J.setZero();
  const auto f = static_cast<size_t>(contact_);
  J.middleCols(d.offQ(), d.nq) = c.dacc_dq[f].topRows(ref_.size());
  J.middleCols(d.offV(), d.nv) = c.dacc_dv[f].topRows(ref_.size());
  J.middleCols(d.offA(), d.na) = c.footJ[f].topRows(ref_.size());
}

void FrameAccelerationConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                       const Eigen::Ref<const Eigen::VectorXd>& w,
                                                       Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.hessians->addFrameHessian(FrameTerm::Acceleration, contact_, d, s.z, w, H);
}

FrameOrientationConstraint::FrameOrientationConstraint(int contactIdx, Eigen::Vector3d ref, std::string name)
    : contact_(contactIdx), ref_(std::move(ref)), name_(std::move(name)) {}

void FrameOrientationConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                       Eigen::Ref<Eigen::VectorXd> g) const {
  g = c.footRotLog[static_cast<size_t>(contact_)] - ref_;
}

void FrameOrientationConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                          MatRef J) const {
  J.setZero();
  J.middleCols(d.offQ(), d.nq) = c.footRotJ[static_cast<size_t>(contact_)];
}

void FrameOrientationConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                      const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.hessians->addFrameHessian(FrameTerm::Orientation, contact_, d, s.z, w, H);
}

MonitorHeightLowerBoundConstraint::MonitorHeightLowerBoundConstraint(const RobotModel& m, Eigen::VectorXd lb,
                                                                     std::string name)
    : nMon_(m.nMonitors()), nContacts_(m.nContacts()), lb_(std::move(lb)), name_(std::move(name)) {
  if (lb_.size() != nMon_) throw std::invalid_argument(name_ + ": lb size != nMonitors");
}

void MonitorHeightLowerBoundConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                              Eigen::Ref<Eigen::VectorXd> g) const {
  for (int i = 0; i < nMon_; ++i) g(i) = lb_(i) - c.monPos[static_cast<size_t>(i)].z();
}

void MonitorHeightLowerBoundConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                                 MatRef J) const {
  J.setZero();
  for (int i = 0; i < nMon_; ++i) J.row(i).segment(d.offQ(), d.nq) = -c.monJ[static_cast<size_t>(i)].row(2);
}

void MonitorHeightLowerBoundConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d,
                                                             HessianScratch& s,
                                                             const Eigen::Ref<const Eigen::VectorXd>& w,
                                                             Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  packZ(z, d, s);
  s.w3.setZero(3);
  for (int i = 0; i < nMon_; ++i) {
    if (w(i) == 0.0) continue;
    s.w3(2) = -w(i);
    s.hessians->addFrameHessian(FrameTerm::Position, nContacts_ + i, d, s.z, s.w3, H);
  }
}

// --- linear terms --------------------------------------------------------------------------

PinConstraint::PinConstraint(std::vector<int> idx, Eigen::VectorXd target, std::string name)
    : idx_(std::move(idx)), target_(std::move(target)), name_(std::move(name)) {
  if (static_cast<int>(idx_.size()) != target_.size()) throw std::invalid_argument(name_ + ": idx/target size mismatch");
}

void PinConstraint::value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                          Eigen::Ref<Eigen::VectorXd> g) const {
  for (size_t i = 0; i < idx_.size(); ++i) g(static_cast<int>(i)) = z.at(d, idx_[i]) - target_(static_cast<int>(i));
}

void PinConstraint::jacobian(const NodeQuantities&, const NodeDims&, const ModelEvalCache&, MatRef J) const {
  J.setZero();
  for (size_t i = 0; i < idx_.size(); ++i) J(static_cast<int>(i), idx_[i]) = 1.0;
}

PairEqualityConstraint::PairEqualityConstraint(std::vector<int> left, std::vector<int> right, std::string name)
    : left_(std::move(left)), right_(std::move(right)), name_(std::move(name)) {
  if (left_.empty() || left_.size() != right_.size())
    throw std::invalid_argument(name_ + ": pair equality wants equally sized non-empty index lists");
  for (size_t i = 0; i < left_.size(); ++i)
    if (left_[i] < 0 || right_[i] < 0 || left_[i] == right_[i])
      throw std::invalid_argument(name_ + ": invalid pair equality index");
}

void PairEqualityConstraint::value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                                   Eigen::Ref<Eigen::VectorXd> g) const {
  for (size_t i = 0; i < left_.size(); ++i) g(static_cast<int>(i)) = z.at(d, left_[i]) - z.at(d, right_[i]);
}

void PairEqualityConstraint::jacobian(const NodeQuantities&, const NodeDims&, const ModelEvalCache&, MatRef J) const {
  J.setZero();
  for (size_t i = 0; i < left_.size(); ++i) {
    J(static_cast<int>(i), left_[i]) = 1.0;
    J(static_cast<int>(i), right_[i]) = -1.0;
  }
}

LowerBoundConstraint::LowerBoundConstraint(std::vector<int> idx, Eigen::VectorXd lb, std::string name)
    : idx_(std::move(idx)), lb_(std::move(lb)), name_(std::move(name)) {
  if (static_cast<int>(idx_.size()) != lb_.size()) throw std::invalid_argument(name_ + ": idx/lb size mismatch");
}

void LowerBoundConstraint::value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                                 Eigen::Ref<Eigen::VectorXd> g) const {
  for (size_t i = 0; i < idx_.size(); ++i) g(static_cast<int>(i)) = lb_(static_cast<int>(i)) - z.at(d, idx_[i]);
}

void LowerBoundConstraint::jacobian(const NodeQuantities&, const NodeDims&, const ModelEvalCache&, MatRef J) const {
  J.setZero();
  for (size_t i = 0; i < idx_.size(); ++i) J(static_cast<int>(i), idx_[i]) = -1.0;
}

int fieldOffset(Field f, const NodeDims& d) {
  switch (f) {
    case Field::Q: return d.offQ();
    case Field::V: return d.offV();
    case Field::A: return d.offA();
    case Field::Lam: return d.offLam();
    case Field::U: return d.offU();
  }
  return d.offQ();
}

int fieldSize(Field f, const NodeDims& d) {
  switch (f) {
    case Field::Q: return d.nq;
    case Field::V: return d.nv;
    case Field::A: return d.na;
    case Field::Lam: return d.nlam;
    case Field::U: return d.nu;
  }
  return d.nq;
}

BoxConstraint::BoxConstraint(Field field, std::vector<int> comps, Eigen::VectorXd lb, Eigen::VectorXd ub,
                             std::string name)
    : field_(field), name_(std::move(name)) {
  if (comps.size() != static_cast<size_t>(lb.size()) || comps.size() != static_cast<size_t>(ub.size()))
    throw std::invalid_argument(name_ + ": comps/lb/ub size mismatch");
  for (size_t i = 0; i < comps.size(); ++i) {
    const Eigen::Index k = static_cast<Eigen::Index>(i);
    if (lb(k) > ub(k)) throw std::invalid_argument(name_ + ": lb > ub");
    if (std::isfinite(lb(k))) rows_.push_back({comps[i], -1.0, lb(k)});
    if (std::isfinite(ub(k))) rows_.push_back({comps[i], +1.0, ub(k)});
  }
}

void BoxConstraint::value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                          Eigen::Ref<Eigen::VectorXd> g) const {
  const int off = fieldOffset(field_, d);
  for (size_t r = 0; r < rows_.size(); ++r)
    g(static_cast<int>(r)) = rows_[r].sign * (z.at(d, off + rows_[r].comp) - rows_[r].bound);
}

void BoxConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, MatRef J) const {
  J.setZero();
  const int off = fieldOffset(field_, d);
  for (size_t r = 0; r < rows_.size(); ++r) J(static_cast<int>(r), off + rows_[r].comp) = rows_[r].sign;
}

MotorEnvelopeConstraint::MotorEnvelopeConstraint(const RobotModel& m, const MotorSaturationConfig& cfg, std::string name)
    : actIdx_(m.actuatedIdx()),
      tauMax_(cfg.effortScale * m.actuatedEffortLimit()),
      vMax_(m.actuatedVelocityLimit()),
      derate_(cfg.speedDerating),
      knee_(cfg.kneeFraction),
      name_(std::move(name)) {
  if (!cfg.enabled) throw std::invalid_argument(name_ + ": saturation config is disabled");
  if (!(cfg.effortScale > 0.0) || !(knee_ >= 0.0 && knee_ < 1.0))
    throw std::invalid_argument(name_ + ": invalid saturation parameters");
  for (int j = 0; j < tauMax_.size(); ++j)
    if (!std::isfinite(tauMax_(j)) || tauMax_(j) <= 0.0 || !std::isfinite(vMax_(j)) || vMax_(j) <= 0.0)
      throw std::invalid_argument(name_ + ": finite positive effort/velocity limits are required");
}

void MotorEnvelopeConstraint::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&,
                                    Eigen::Ref<Eigen::VectorXd> g) const {
  int r = 0;
  for (int j = 0; j < static_cast<int>(actIdx_.size()); ++j) {
    const double v = z.v(actIdx_[static_cast<size_t>(j)]);
    const double u = z.u(j);
    const double tm = tauMax_(j), vm = vMax_(j);
    g(r++) = v / vm - 1.0;
    g(r++) = -v / vm - 1.0;
    g(r++) = u / tm - 1.0;
    g(r++) = -u / tm - 1.0;
    if (derate_) {
      const double invRamp = 1.0 / (1.0 - knee_);
      g(r++) = u / tm + invRamp * v / vm - invRamp;
      g(r++) = -u / tm - invRamp * v / vm - invRamp;
    }
  }
}

void MotorEnvelopeConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, MatRef J) const {
  J.setZero();
  int r = 0;
  for (int j = 0; j < static_cast<int>(actIdx_.size()); ++j) {
    const int vi = d.offV() + actIdx_[static_cast<size_t>(j)];
    const int ui = d.offU() + j;
    J(r++, vi) = 1.0 / vMax_(j);
    J(r++, vi) = -1.0 / vMax_(j);
    J(r++, ui) = 1.0 / tauMax_(j);
    J(r++, ui) = -1.0 / tauMax_(j);
    if (derate_) {
      const double velocityScale = 1.0 / ((1.0 - knee_) * vMax_(j));
      J(r, ui) = 1.0 / tauMax_(j);
      J(r++, vi) = velocityScale;
      J(r, ui) = -1.0 / tauMax_(j);
      J(r++, vi) = -velocityScale;
    }
  }
}

RowSliceConstraint::RowSliceConstraint(std::shared_ptr<const Constraint> inner, int row)
    : inner_(std::move(inner)), row_(row) {
  if (inner_ == nullptr || row_ < 0) throw std::invalid_argument("RowSliceConstraint: null inner / bad row");
}

void RowSliceConstraint::value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c,
                               Eigen::Ref<Eigen::VectorXd> g) const {
  const int n = inner_->rows(d);
  if (row_ >= n) throw std::out_of_range(inner_->name() + ": sliced row out of range");
  Eigen::VectorXd gin(n);
  inner_->value(z, d, c, gin);
  g(0) = gin(row_);
}

void RowSliceConstraint::jacobian(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c, MatRef J) const {
  const int n = inner_->rows(d);
  if (row_ >= n) throw std::out_of_range(inner_->name() + ": sliced row out of range");
  Eigen::MatrixXd Jin = Eigen::MatrixXd::Zero(n, J.cols());
  inner_->jacobian(z, d, c, Jin);
  J.row(0) = Jin.row(row_);
}

void RowSliceConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                              const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  Eigen::VectorXd win = Eigen::VectorXd::Zero(inner_->rows(d));
  win(row_) = w(0);
  inner_->addContractedHessian(z, d, s, win, H);
}

// --- contact forces and impulses --------------------------------------------------------------

FrictionConeConstraint::FrictionConeConstraint(const ContactSet& cs, double mu, double eps, std::string name)
    : mu_(mu), eps_(eps), name_(std::move(name)) {
  for (const ContactRef& cr : cs.active) {
    if (cr.dim != 3 && cr.dim != 6) throw std::invalid_argument(name_ + ": cone needs dim == 3 or dim == 6");
    lamOff_.push_back(cs.lamOffset(cr.index));
  }
}

void FrictionConeConstraint::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&,
                                   Eigen::Ref<Eigen::VectorXd> g) const {
  for (size_t f = 0; f < lamOff_.size(); ++f) {
    const int o = lamOff_[f];
    const double fx = z.lam(o), fy = z.lam(o + 1), fz = z.lam(o + 2);
    const double r = std::sqrt(fx * fx + fy * fy + eps_ * eps_);
    const int rc = 2 * static_cast<int>(f);
    g(rc) = r - eps_ - mu_ * fz;
    g(rc + 1) = -fz;
  }
}

void FrictionConeConstraint::jacobian(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&, MatRef J) const {
  J.setZero();
  for (size_t f = 0; f < lamOff_.size(); ++f) {
    const int o = lamOff_[f];
    const int col = d.offLam() + o;
    const double fx = z.lam(o), fy = z.lam(o + 1);
    const double r = std::sqrt(fx * fx + fy * fy + eps_ * eps_);
    const int rc = 2 * static_cast<int>(f);
    if (r > kApexTol) {
      J(rc, col) = fx / r;
      J(rc, col + 1) = fy / r;
    }
    J(rc, col + 2) = -mu_;
    J(rc + 1, col + 2) = -1.0;
  }
}

void FrictionConeConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch&,
                                                  const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  for (size_t f = 0; f < lamOff_.size(); ++f) {
    const double wc = w(2 * static_cast<int>(f));
    if (wc == 0.0) continue;
    const int o = lamOff_[f];
    const double fx = z.lam(o), fy = z.lam(o + 1);
    const double r2 = fx * fx + fy * fy + eps_ * eps_;
    const double r = std::sqrt(r2);
    if (r <= kApexTol) continue;
    const double inv = wc / (r2 * r);
    const int b = d.offLam() + o;
    H(b, b) += inv * (fy * fy + eps_ * eps_);
    H(b, b + 1) += inv * (-fx * fy);
    H(b + 1, b) += inv * (-fx * fy);
    H(b + 1, b + 1) += inv * (fx * fx + eps_ * eps_);
  }
}

ContactWrenchBoxConstraint::ContactWrenchBoxConstraint(const ContactSet& cs, std::vector<int> comps,
                                                       std::vector<double> coefs, std::string name)
    : comps_(std::move(comps)), coefs_(std::move(coefs)), name_(std::move(name)) {
  if (comps_.empty() || comps_.size() != coefs_.size())
    throw std::invalid_argument(name_ + ": comps/coefs must match and be nonempty");
  for (int c : comps_)
    if (c < 3 || c > 5) throw std::invalid_argument(name_ + ": comps must be torque rows {3, 4, 5}");
  for (double k : coefs_)
    if (!(k > 0.0)) throw std::invalid_argument(name_ + ": coefs must be positive");
  for (const ContactRef& cr : cs.active) {
    if (cr.dim != 6) throw std::invalid_argument(name_ + ": wrench box needs dim == 6 contacts");
    lamOff_.push_back(cs.lamOffset(cr.index));
  }
}

void ContactWrenchBoxConstraint::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&,
                                       Eigen::Ref<Eigen::VectorXd> g) const {
  int r = 0;
  for (int o : lamOff_) {
    const double fz = z.lam(o + 2);
    for (size_t k = 0; k < comps_.size(); ++k) {
      const double t = z.lam(o + comps_[k]);
      g(r++) = t - coefs_[k] * fz;
      g(r++) = -t - coefs_[k] * fz;
    }
  }
}

void ContactWrenchBoxConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&,
                                          MatRef J) const {
  J.setZero();
  int r = 0;
  for (int o : lamOff_) {
    const int cz = d.offLam() + o + 2;
    for (size_t k = 0; k < comps_.size(); ++k) {
      const int ct = d.offLam() + o + comps_[k];
      J(r, ct) = 1.0;
      J(r++, cz) = -coefs_[k];
      J(r, ct) = -1.0;
      J(r++, cz) = -coefs_[k];
    }
  }
}

ImpactConeConstraint::ImpactConeConstraint(int coneId, std::vector<int> lamOff, double mu, double eps, std::string name)
    : coneId_(coneId), lamOff_(std::move(lamOff)), mu_(mu), eps_(eps), name_(std::move(name)) {
  if (lamOff_.empty()) throw std::invalid_argument(name_ + ": impact cone needs at least one contact");
}

void ImpactConeConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                 Eigen::Ref<Eigen::VectorXd> g) const {
  const Eigen::VectorXd& L = c.impactLam.at(static_cast<size_t>(coneId_));
  for (size_t f = 0; f < lamOff_.size(); ++f) {
    const int o = lamOff_[f];
    const double lx = L(o), ly = L(o + 1), lz = L(o + 2);
    const double r = std::sqrt(lx * lx + ly * ly + eps_ * eps_);
    const int rc = 2 * static_cast<int>(f);
    g(rc) = r - eps_ - mu_ * lz;
    g(rc + 1) = -lz;
  }
}

void ImpactConeConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const {
  J.setZero();
  const auto s = static_cast<size_t>(coneId_);
  const Eigen::VectorXd& L = c.impactLam.at(s);
  const Eigen::MatrixXd& dq = c.impactLamDq.at(s);
  const Eigen::MatrixXd& dv = c.impactLamDv.at(s);
  const int nq = static_cast<int>(dq.cols());
  const int nv = static_cast<int>(dv.cols());
  for (size_t f = 0; f < lamOff_.size(); ++f) {
    const int o = lamOff_[f];
    const double lx = L(o), ly = L(o + 1);
    const double r = std::sqrt(lx * lx + ly * ly + eps_ * eps_);
    const double cx = r > kApexTol ? lx / r : 0.0;
    const double cy = r > kApexTol ? ly / r : 0.0;
    const int rc = 2 * static_cast<int>(f);
    J.block(rc, d.offQ(), 1, nq) = cx * dq.row(o) + cy * dq.row(o + 1) - mu_ * dq.row(o + 2);
    J.block(rc, d.offV(), 1, nv) = cx * dv.row(o) + cy * dv.row(o + 1) - mu_ * dv.row(o + 2);
    J.block(rc + 1, d.offQ(), 1, nq) = -dq.row(o + 2);
    J.block(rc + 1, d.offV(), 1, nv) = -dv.row(o + 2);
  }
}

// g = cone(Lambda(q, v)): H = sum_i w_i [ (dLambda/dz)' (d2 cone_i / dLambda2) (dLambda/dz)
//                                       + sum_r (d cone_i / dLambda_r) d2 Lambda_r / dz2 ].
void ImpactConeConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  requireHessians(s, name_);
  if (s.model == nullptr || s.data == nullptr) throw std::logic_error(name_ + ": the Hessian pass needs a model and its data");
  packZ(z, d, s);
  Eigen::VectorXd L;
  Eigen::MatrixXd dLdq, dLdv;
  s.model->impactConeImpulse(*s.data, coneId_, z.q, z.v, L, &dLdq, &dLdv);
  const int nq = static_cast<int>(dLdq.cols()), nv = static_cast<int>(dLdv.cols());
  Eigen::VectorXd wL = Eigen::VectorXd::Zero(L.size());
  Eigen::MatrixXd Jqv(L.size(), nq + nv);
  Jqv << dLdq, dLdv;
  Eigen::MatrixXd Hqv = Eigen::MatrixXd::Zero(nq + nv, nq + nv);
  for (size_t f = 0; f < lamOff_.size(); ++f) {
    const int o = lamOff_[f];
    const int rc = 2 * static_cast<int>(f);
    const double wc = w(rc), wz = w(rc + 1);
    const double lx = L(o), ly = L(o + 1);
    const double r2 = lx * lx + ly * ly + eps_ * eps_;
    const double r = std::sqrt(r2);
    const double cx = r > kApexTol ? lx / r : 0.0;
    const double cy = r > kApexTol ? ly / r : 0.0;
    wL(o) += wc * cx;
    wL(o + 1) += wc * cy;
    wL(o + 2) += -mu_ * wc - wz;
    if (wc == 0.0 || r <= kApexTol) continue;
    Eigen::Matrix2d Hc;
    Hc << ly * ly + eps_ * eps_, -lx * ly, -lx * ly, lx * lx + eps_ * eps_;
    Hc *= wc / (r2 * r);
    const Eigen::MatrixXd Jxy = Jqv.middleRows(o, 2);
    Hqv.noalias() += Jxy.transpose() * Hc * Jxy;
  }
  H.block(d.offQ(), d.offQ(), nq, nq) += Hqv.topLeftCorner(nq, nq);
  H.block(d.offQ(), d.offV(), nq, nv) += Hqv.topRightCorner(nq, nv);
  H.block(d.offV(), d.offQ(), nv, nq) += Hqv.bottomLeftCorner(nv, nq);
  H.block(d.offV(), d.offV(), nv, nv) += Hqv.bottomRightCorner(nv, nv);
  const Eigen::VectorXd wdv = Eigen::VectorXd::Zero(nv);
  s.hessians->addImpactHessian(s.model->impactConeSetContacts(coneId_), d, s.z, wdv, wL, H);
}

}  // namespace leap
