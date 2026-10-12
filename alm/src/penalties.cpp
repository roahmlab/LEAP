#include "leap/alm/penalties.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace leap {

Penalties::Penalties(const CompiledProblem& prob, const PenaltyConfig& cfg)
    : prob_(&prob),
      cfg_(cfg),
      mu_(static_cast<size_t>(prob.terms().size()), cfg.mu0),
      cvPrev_(static_cast<size_t>(prob.terms().size()), std::numeric_limits<double>::infinity()),
      muRow_(prob.layout().md) {
  if (!(cfg.mu0 > 0.0) || !(cfg.muMax > 0.0)) throw std::invalid_argument("Penalties: mu0 and muMax must be positive");
  rebuildRows();
}

void Penalties::seed(double mu0) {
  std::fill(mu_.begin(), mu_.end(), mu0);
  std::fill(cvPrev_.begin(), cvPrev_.end(), std::numeric_limits<double>::infinity());
  rebuildRows();
}

void Penalties::update(const Eigen::VectorXd& perTermViol) {
  if (cfg_.policy == PenaltyPolicy::Constant) return;
  if (perTermViol.size() != static_cast<Eigen::Index>(mu_.size()))
    throw std::invalid_argument("Penalties::update: one violation per term");
  for (size_t t = 0; t < mu_.size(); ++t) {
    const double cv = perTermViol(static_cast<Eigen::Index>(t));
    if (cfg_.policy == PenaltyPolicy::Monotone) {
      mu_[t] = std::min(cfg_.muMax, cfg_.growth * mu_[t]);
    } else {
      if (cv > cfg_.stallRatio * cvPrev_[t]) mu_[t] = std::min(cfg_.muMax, cfg_.growth * mu_[t]);
      cvPrev_[t] = cv;
    }
  }
  rebuildRows();
}

double Penalties::muRepresentative() const {
  double m = 0.0;
  for (double v : mu_) m = std::max(m, v);
  return m;
}

double Penalties::muMin() const {
  double m = std::numeric_limits<double>::infinity();
  for (double v : mu_) m = std::min(m, v);
  return m;
}

void Penalties::rebuildRows() {
  const std::vector<int>& t = prob_->rowTerm();
  for (Eigen::Index i = 0; i < muRow_.size(); ++i) muRow_(i) = mu_[static_cast<size_t>(t[static_cast<size_t>(i)])];
}

}  // namespace leap
