#pragma once

// First-order diagnostics of an evaluation in the flow's own coordinates x = E z: the
// reduced gradient of the augmented Lagrangian and the complementarity of the
// inequality rows, both relative to the PHR multiplier estimate w (which includes the
// penalty correction). A normalized residual, not a model-decrease test; it is what
// the objective mode's stop reads.

#include <algorithm>
#include <cmath>
#include <limits>

#include "leap/alm/evaluator.hpp"
#include "leap/alm/junction_quotient.hpp"

namespace leap {

struct FirstOrderResidual {
  double gradientInf = 0.0, gradientRel = 0.0;
  double complementarityInf = 0.0, complementarityRel = 0.0;
  double multiplierScale = 1.0;
};

inline FirstOrderResidual firstOrderResidual(const CompiledProblem& p, const JunctionQuotient& Q, const Evaluation& c) {
  FirstOrderResidual r;
  if (c.F.size() == 0 || !c.F.allFinite() || !c.phr.w.allFinite() || !c.g.allFinite()) {
    r.gradientInf = r.gradientRel = r.complementarityInf = r.complementarityRel = std::numeric_limits<double>::infinity();
    return r;
  }
  Eigen::VectorXd g;
  Q.reduceCovector(c.F.head(p.layout().np), g);
  r.gradientInf = g.size() ? g.lpNorm<Eigen::Infinity>() : 0.0;
  r.multiplierScale = std::max(1.0, c.phr.w.size() ? c.phr.w.lpNorm<Eigen::Infinity>() : 0.0);
  r.gradientRel = r.gradientInf / r.multiplierScale;
  for (int i = 0; i < p.layout().md; ++i)
    if (p.rowIneq()[static_cast<size_t>(i)]) r.complementarityInf = std::max(r.complementarityInf, std::abs(c.phr.w(i) * c.g(i)));
  r.complementarityRel = r.complementarityInf / r.multiplierScale;
  return r;
}

}  // namespace leap
