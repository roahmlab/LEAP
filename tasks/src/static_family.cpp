#include "leap/tasks/static_family.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "leap/problem/mesh_refinement.hpp"
#include "leap/solver/objective_weights.hpp"
#include "leap/solver/term_scales.hpp"

namespace leap {

namespace {
bool splitElementDescending(const ElementSplit& a, const ElementSplit& b) { return a.element > b.element; }
}  // namespace

StaticFamily::StaticFamily(std::string name, TrajectoryProblem problem, Eigen::VectorXd seed, double defectWeight, std::string type,
                           std::string summary)
    : type_(std::move(type)), name_(std::move(name)), summary_(std::move(summary)), current_(std::move(problem)), y0_(std::move(seed)),
      defectWeight_(defectWeight) {
  current_.validate();
  if (current_.elements.empty()) throw std::invalid_argument("StaticFamily: the problem has no elements");
  const CompiledProblem prob = CompiledProblem::compile(current_);
  if (y0_.size() != prob.layout().n())
    throw std::invalid_argument("StaticFamily: the seed has " + std::to_string(y0_.size()) + " entries, the compiled problem " +
                                std::to_string(prob.layout().n()));
  const ElementView X(y0_.data(), prob.layout().elem[0]);
  qRef_ = X.Q.col(0);
}

TrajectoryProblem StaticFamily::obstacleVariant(const ObstacleVariant& v) {
  (void)v;  // no obstacle family: the posed problem is its own variant
  return current_;
}

Eigen::VectorXd StaticFamily::seed(const CompiledProblem& prob) {
  if (y0_.size() != prob.layout().n()) throw std::runtime_error("StaticFamily::seed: layout mismatch (the seed is for the level-0 mesh)");
  return y0_;
}

void StaticFamily::refine(const std::vector<ElementSplit>& splits) {
  std::vector<ElementSplit> sorted = splits;
  std::sort(sorted.begin(), sorted.end(), splitElementDescending);  // highest index first: lower indices stay valid
  pendingElements_.clear();
  pendingFractions_.clear();
  for (const ElementSplit& s : sorted) {
    current_ = splitElement(current_, s.element, s.fraction);
    pendingElements_.push_back(s.element);
    pendingFractions_.push_back(s.fraction);
  }
}

double StaticFamily::prolong(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to,
                             Eigen::VectorXd& yTo) {
  const double err = prolongSplits(from, yFrom, pendingElements_, pendingFractions_, to, yTo);
  pendingElements_.clear();
  pendingFractions_.clear();
  return err;
}

Eigen::VectorXd StaticFamily::oracleTermScales(const CompiledProblem& prob) const {
  return physicalTermScales(prob, *current_.model, qRef_);
}

std::vector<double> StaticFamily::objectiveNodeWeights(const CompiledProblem& prob) const {
  return leap::objectiveNodeWeights(prob, objectiveQuadrature_, objectiveScale_);
}

void StaticFamily::setObjectiveWeights(std::string quadrature, double scale) {
  objectiveQuadrature_ = std::move(quadrature);
  objectiveScale_ = scale;
}

std::string StaticFamily::summary() const {
  if (!summary_.empty()) return summary_;
  char buf[200];
  std::snprintf(buf, sizeof(buf), "%s: %d element(s), T = %.3f s, defect weight %.3g", name_.c_str(),
                static_cast<int>(current_.elements.size()), current_.totalTime(), defectWeight_);
  return buf;
}

}  // namespace leap
