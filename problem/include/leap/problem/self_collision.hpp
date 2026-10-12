#pragma once

// Self-collision: selected pairs of the robot's own capsules stay apart,
//     g = (dsafe^2 - D) / (2 dsafe) <= 0,   D = min_{t, s in [0,1]} |A_i + t u_i - A_j - s u_j|^2,
// dsafe = r_i + r_j + margin. The closest pair is Ericson's closed form; the gradient is
// the envelope theorem with (t*, s*) frozen.

#include <string>
#include <utility>
#include <vector>

#include "leap/problem/constraint.hpp"

namespace leap {

struct SegmentSegmentClosest {
  double t = 0.0, s = 0.0;
  Eigen::Vector3d w{0.0, 0.0, 0.0};  // P(t) - Q(s)
  double dist2 = 0.0;
  bool interiorT = false, interiorS = false;
};

SegmentSegmentClosest closestSegmentSegment(const Eigen::Vector3d& A1, const Eigen::Vector3d& B1,
                                            const Eigen::Vector3d& A2, const Eigen::Vector3d& B2);
double capsulePairResidual(const Eigen::Vector3d& A1, const Eigen::Vector3d& B1, const Eigen::Vector3d& A2,
                           const Eigen::Vector3d& B2, double dsafe);

class CapsuleSelfCollisionConstraint final : public Constraint {
public:
  CapsuleSelfCollisionConstraint(const RobotModel& m, std::vector<std::pair<int, int>> pairs, double margin,
                                 std::string name, std::vector<double> perPairMargin = {});
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(pairs_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kCapsulePos; }
  const std::string& rowLabel(int r) const { return labels_.at(static_cast<size_t>(r)); }
  const std::vector<std::pair<int, int>>& capsulePairs() const { return pairs_; }
  double margin() const { return margin_; }
  double dsafe(int r) const { return dsafe_.at(static_cast<size_t>(r)); }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  std::vector<std::pair<int, int>> pairs_;
  std::vector<int> cacheA1_, cacheB1_, cacheA2_, cacheB2_;
  std::vector<int> pointA1_, pointB1_, pointA2_, pointB2_;
  std::vector<std::string> labels_;
  std::vector<double> dsafe_;
  double margin_;
  std::string name_;
};

}  // namespace leap
