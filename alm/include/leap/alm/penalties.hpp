#pragma once

// One penalty per constraint TERM (a named constraint on one element; kin1/kin2 per
// element; one shared "reset" term), expanded to a per-row vector mu for the evaluator.
//
// Policies, stepped once per window from the term's current max violation cv:
//   Constant   never changes
//   Stall      (Nocedal-Wright) mu <- min(growth mu, muMax) iff cv > ratio * cv_prev
//   Monotone   mu <- min(growth mu, muMax) every step

#include <vector>

#include <Eigen/Dense>

#include "leap/problem/compiled_problem.hpp"

namespace leap {

enum class PenaltyPolicy { Constant, Stall, Monotone };

struct PenaltyConfig {
  PenaltyPolicy policy = PenaltyPolicy::Stall;
  double mu0 = 10.0;
  double muMax = 3e6;
  double growth = 3.0;
  double stallRatio = 0.25;
};

class Penalties {
public:
  Penalties(const CompiledProblem& prob, const PenaltyConfig& cfg);

  void seed(double mu0);                               // every term to mu0, stall history cleared
  void update(const Eigen::VectorXd& perTermViol);     // one policy step per term
  void setPolicy(PenaltyPolicy p) { cfg_.policy = p; }
  void setMuMax(double m) { cfg_.muMax = m; }

  const Eigen::VectorXd& muRow() const { return muRow_; }
  double termMu(int term) const { return mu_[static_cast<size_t>(term)]; }
  double muMax() const { return cfg_.muMax; }
  double muRepresentative() const;  // the largest term penalty
  double muMin() const;
  int numTerms() const { return static_cast<int>(mu_.size()); }
  const PenaltyConfig& config() const { return cfg_; }

private:
  void rebuildRows();
  const CompiledProblem* prob_;
  PenaltyConfig cfg_;
  std::vector<double> mu_, cvPrev_;
  Eigen::VectorXd muRow_;
};

}  // namespace leap
