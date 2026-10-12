#include "leap/problem/costs.hpp"

#include <cmath>
#include <stdexcept>

namespace leap {

ControlEffortCost::ControlEffortCost(Eigen::VectorXd weights) : w_(std::move(weights)) {
  for (int j = 0; j < w_.size(); ++j)
    if (!(w_(j) >= 0.0) || !std::isfinite(w_(j))) throw std::invalid_argument("ControlEffortCost: weights must be finite and >= 0");
}
double ControlEffortCost::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const {
  if (w_.size() == 0) return 0.5 * z.u.squaredNorm();
  if (w_.size() != z.u.size()) throw std::invalid_argument("ControlEffortCost: weights do not match nu");
  return 0.5 * z.u.cwiseProduct(w_).dot(z.u);
}
void ControlEffortCost::gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                                 Eigen::Ref<Eigen::VectorXd> g) const {
  g.setZero();
  if (w_.size() == 0)
    g.segment(d.offU(), d.nu) = z.u;
  else
    g.segment(d.offU(), d.nu) = w_.cwiseProduct(z.u);
}
void ControlEffortCost::addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&,
                                   Eigen::MatrixXd& H) const {
  for (int i = 0; i < d.nu; ++i) H(d.offU() + i, d.offU() + i) += w_.size() == 0 ? 1.0 : w_(i);
}

JointVelocityCost::JointVelocityCost(std::vector<int> idx, Eigen::VectorXd w) : idx_(std::move(idx)), w_(std::move(w)) {
  if (static_cast<int>(idx_.size()) != w_.size()) throw std::invalid_argument("JointVelocityCost: idx/w size mismatch");
}
double JointVelocityCost::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const {
  double s = 0.0;
  for (size_t j = 0; j < idx_.size(); ++j) {
    const double v = z.v(idx_[j]);
    s += w_(static_cast<int>(j)) * v * v;
  }
  return 0.5 * s;
}
void JointVelocityCost::gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                                 Eigen::Ref<Eigen::VectorXd> g) const {
  g.setZero();
  for (size_t j = 0; j < idx_.size(); ++j) g(d.offV() + idx_[j]) = w_(static_cast<int>(j)) * z.v(idx_[j]);
}
void JointVelocityCost::addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&,
                                   Eigen::MatrixXd& H) const {
  for (size_t j = 0; j < idx_.size(); ++j) H(d.offV() + idx_[j], d.offV() + idx_[j]) += w_(static_cast<int>(j));
}

JointAccelerationCost::JointAccelerationCost(std::vector<int> idx, Eigen::VectorXd w)
    : idx_(std::move(idx)), w_(std::move(w)) {
  if (static_cast<int>(idx_.size()) != w_.size())
    throw std::invalid_argument("JointAccelerationCost: idx/w size mismatch");
}
double JointAccelerationCost::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const {
  double s = 0.0;
  for (size_t j = 0; j < idx_.size(); ++j) {
    const double a = z.a(idx_[j]);
    s += w_(static_cast<int>(j)) * a * a;
  }
  return 0.5 * s;
}
void JointAccelerationCost::gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                                     Eigen::Ref<Eigen::VectorXd> g) const {
  g.setZero();
  for (size_t j = 0; j < idx_.size(); ++j) g(d.offA() + idx_[j]) = w_(static_cast<int>(j)) * z.a(idx_[j]);
}
void JointAccelerationCost::addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&,
                                       Eigen::MatrixXd& H) const {
  for (size_t j = 0; j < idx_.size(); ++j) H(d.offA() + idx_[j], d.offA() + idx_[j]) += w_(static_cast<int>(j));
}

JointPostureCost::JointPostureCost(std::vector<int> idx, Eigen::VectorXd target, Eigen::VectorXd w)
    : idx_(std::move(idx)), target_(std::move(target)), w_(std::move(w)) {
  if (static_cast<int>(idx_.size()) != w_.size() || target_.size() != w_.size())
    throw std::invalid_argument("JointPostureCost: idx/target/w size mismatch");
}
double JointPostureCost::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const {
  double s = 0.0;
  for (size_t j = 0; j < idx_.size(); ++j) {
    const double e = z.q(idx_[j]) - target_(static_cast<int>(j));
    s += w_(static_cast<int>(j)) * e * e;
  }
  return 0.5 * s;
}
void JointPostureCost::gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                                Eigen::Ref<Eigen::VectorXd> g) const {
  g.setZero();
  for (size_t j = 0; j < idx_.size(); ++j)
    g(d.offQ() + idx_[j]) = w_(static_cast<int>(j)) * (z.q(idx_[j]) - target_(static_cast<int>(j)));
}
void JointPostureCost::addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&,
                                  Eigen::MatrixXd& H) const {
  for (size_t j = 0; j < idx_.size(); ++j) H(d.offQ() + idx_[j], d.offQ() + idx_[j]) += w_(static_cast<int>(j));
}

ContactForceCost::ContactForceCost(double weight) : weight_(weight) {
  if (!(weight >= 0.0) || !std::isfinite(weight)) throw std::invalid_argument("ContactForceCost: weight must be finite and >= 0");
}
double ContactForceCost::value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const {
  return 0.5 * weight_ * z.lam.squaredNorm();
}
void ContactForceCost::gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                                Eigen::Ref<Eigen::VectorXd> g) const {
  g.setZero();
  if (d.nlam > 0) g.segment(d.offLam(), d.nlam) = weight_ * z.lam;
}
void ContactForceCost::addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&,
                                  Eigen::MatrixXd& H) const {
  for (int i = 0; i < d.nlam; ++i) H(d.offLam() + i, d.offLam() + i) += weight_;
}

}  // namespace leap
