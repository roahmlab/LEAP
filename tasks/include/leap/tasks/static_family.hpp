#pragma once

// StaticFamily: any TrajectoryProblem built with the formulation API, plus its cold seed,
// as a ProblemFamily the staged pipeline can run. Refinement bisects elements with the
// generic splitElement / prolongSplits; there is no obstacle family (the pre-stage finds
// nothing to continue and phase A0 runs on the posed problem). The yaml problem type
// `contact_sequence` (tasks/contact_sequence.hpp) and the C++ example
// examples/hopper_jump.cpp both run through it.

#include <memory>
#include <string>
#include <vector>

#include "leap/problem/trajectory_problem.hpp"
#include "leap/solver/pipeline.hpp"

namespace leap {

class StaticFamily final : public ProblemFamily {
public:
  // `seed` is the level-0 iterate (primal + duals, usually zero duals) of the compiled
  // `problem`; `defectWeight` is the LS dynamics-defect rho the penalties are scaled by.
  // `type` names the problem kind in the logs and the archive (default "custom"),
  // `summary` the one-paragraph description the driver prints (default: generic).
  StaticFamily(std::string name, TrajectoryProblem problem, Eigen::VectorXd seed, double defectWeight, std::string type = "custom",
               std::string summary = "");
  const std::string& type() const override { return type_; }
  const std::string& name() const override { return name_; }
  std::shared_ptr<const RobotModel> model() const override { return current_.model; }
  TrajectoryProblem build() override { return current_; }
  TrajectoryProblem obstacleVariant(const ObstacleVariant& v) override;
  Eigen::VectorXd seed(const CompiledProblem& prob) override;
  void refine(const std::vector<ElementSplit>& splits) override;
  double prolong(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to,
                 Eigen::VectorXd& yTo) override;
  Eigen::VectorXd oracleTermScales(const CompiledProblem& prob) const override;
  std::vector<double> objectiveNodeWeights(const CompiledProblem& prob) const override;
  double defectWeight() const override { return defectWeight_; }
  int numElements() const override { return static_cast<int>(current_.elements.size()); }
  const std::vector<std::string>& warnings() const override { return warnings_; }
  std::string summary() const override;

  // The objective's quadrature ("clenshaw_curtis" | "sum") and scale (objective mode);
  // default Clenshaw-Curtis, scale 1.
  void setObjectiveWeights(std::string quadrature, double scale);
  void setWarnings(std::vector<std::string> warnings) { warnings_ = std::move(warnings); }

private:
  std::string type_, name_, summary_;
  TrajectoryProblem current_;
  Eigen::VectorXd y0_, qRef_;
  double defectWeight_;
  std::string objectiveQuadrature_ = "clenshaw_curtis";
  double objectiveScale_ = 1.0;
  std::vector<int> pendingElements_;
  std::vector<double> pendingFractions_;
  std::vector<std::string> warnings_;
};

}  // namespace leap
