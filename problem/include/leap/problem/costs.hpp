#pragma once

// The cost catalogue: quadratic per-node terms (their Hessians are exact).

#include <vector>

#include "leap/problem/constraint.hpp"

namespace leap {

// l = 1/2 sum_j w_j u_j^2 (control effort; empty weights = 1/2 ||u||^2).
class ControlEffortCost final : public Cost {
public:
  ControlEffortCost() = default;
  explicit ControlEffortCost(Eigen::VectorXd weights);
  double value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const override;
  void gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                Eigen::Ref<Eigen::VectorXd> g) const override;
  void addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, Eigen::MatrixXd& H) const override;
  const Eigen::VectorXd& weights() const { return w_; }

private:
  Eigen::VectorXd w_;  // empty = ones
};

// l = 1/2 sum_j w_j v_{idx_j}^2 (joint path-length proxy; idx are v rows).
class JointVelocityCost final : public Cost {
public:
  JointVelocityCost(std::vector<int> idx, Eigen::VectorXd w);
  double value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const override;
  void gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                Eigen::Ref<Eigen::VectorXd> g) const override;
  void addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, Eigen::MatrixXd& H) const override;

private:
  std::vector<int> idx_;
  Eigen::VectorXd w_;
};

// l = 1/2 sum_j w_j a_{idx_j}^2.
class JointAccelerationCost final : public Cost {
public:
  JointAccelerationCost(std::vector<int> idx, Eigen::VectorXd w);
  double value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const override;
  void gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                Eigen::Ref<Eigen::VectorXd> g) const override;
  void addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, Eigen::MatrixXd& H) const override;

private:
  std::vector<int> idx_;
  Eigen::VectorXd w_;
};

// l = 1/2 sum_j w_j (q_{idx_j} - target_j)^2.
class JointPostureCost final : public Cost {
public:
  JointPostureCost(std::vector<int> idx, Eigen::VectorXd target, Eigen::VectorXd w);
  double value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const override;
  void gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                Eigen::Ref<Eigen::VectorXd> g) const override;
  void addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, Eigen::MatrixXd& H) const override;

private:
  std::vector<int> idx_;
  Eigen::VectorXd target_, w_;
};

// l = weight/2 ||lam||^2.
class ContactForceCost final : public Cost {
public:
  explicit ContactForceCost(double weight);
  double value(const NodeQuantities& z, const NodeDims&, const ModelEvalCache&) const override;
  void gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache&,
                Eigen::Ref<Eigen::VectorXd> g) const override;
  void addHessian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache&, Eigen::MatrixXd& H) const override;

private:
  double weight_;
};

}  // namespace leap
