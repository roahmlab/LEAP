#pragma once

// The Powell-Hestenes-Rockafellar augmented Lagrangian, row by row. For row i with
// residual g_i, multiplier gamma_i and penalty mu_i:
//   equality:    c_eff = g,                    w = gamma + mu g,         active always,
//                psi = gamma g + mu g^2 / 2
//   inequality:  c_eff = max(g, -gamma / mu),  w = max(0, gamma + mu g), active iff w > 0,
//                psi = (max(0, gamma + mu g)^2 - gamma^2) / (2 mu)
//   L_A = f + sum_i psi_i,  grad_x L_A = grad f + J' w,  dL_A/dgamma = c_eff.

#include <vector>

#include <Eigen/Dense>

namespace leap {

struct PhrState {
  Eigen::VectorXd ceff, w;
  std::vector<char> active;        // the exact PHR active set
  std::vector<char> activeMetric;  // the anticipatory set used by the metric (margin added)
};

// Fills ceff, w, active (exact) and activeMetric (inequality rows within `margin` of
// switching on count as active).
inline void phrEffectiveResidual(const Eigen::VectorXd& g, const Eigen::VectorXd& gamma, const Eigen::VectorXd& mu,
                                 const std::vector<char>& ineq, double margin, PhrState& s) {
  const Eigen::Index m = g.size();
  s.ceff.resize(m);
  s.w.resize(m);
  s.active.assign(static_cast<size_t>(m), 1);
  s.activeMetric.assign(static_cast<size_t>(m), 1);
  for (Eigen::Index i = 0; i < m; ++i) {
    if (ineq[static_cast<size_t>(i)]) {
      const double pre = gamma(i) + mu(i) * g(i);
      s.active[static_cast<size_t>(i)] = pre > 0.0 ? 1 : 0;
      s.ceff(i) = std::max(g(i), -gamma(i) / mu(i));
      s.activeMetric[static_cast<size_t>(i)] = (pre > 0.0 || gamma(i) + mu(i) * (g(i) + margin) > 0.0) ? 1 : 0;
    } else {
      s.ceff(i) = g(i);
    }
    s.w(i) = gamma(i) + mu(i) * s.ceff(i);
  }
}

// sum_i psi_i.
inline double phrPenaltyValue(const Eigen::VectorXd& g, const Eigen::VectorXd& gamma, const Eigen::VectorXd& mu,
                              const std::vector<char>& ineq) {
  double val = 0.0;
  for (Eigen::Index i = 0; i < g.size(); ++i) {
    if (ineq[static_cast<size_t>(i)]) {
      const double p = std::max(0.0, gamma(i) + mu(i) * g(i));
      val += (p * p - gamma(i) * gamma(i)) / (2.0 * mu(i));
    } else {
      val += gamma(i) * g(i) + 0.5 * mu(i) * g(i) * g(i);
    }
  }
  return val;
}

}  // namespace leap
