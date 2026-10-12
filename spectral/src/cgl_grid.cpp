#include "leap/spectral/cgl_grid.hpp"

#include <cmath>
#include <stdexcept>

namespace leap {

CglGrid::CglGrid(int degree) : degree_(degree) {
  if (degree < 1) throw std::invalid_argument("CglGrid: degree must be >= 1");
  const int n = degree + 1;

  // Ascending CGL nodes: tau_k = -cos(pi k / N), so tau_0 = -1, tau_N = +1.
  nodes_.resize(n);
  for (int k = 0; k < n; ++k) nodes_(k) = -std::cos(M_PI * k / degree);
  nodes_(0) = -1.0;
  nodes_(degree) = 1.0;
  if (degree % 2 == 0) nodes_(degree / 2) = 0.0;

  // Barycentric weights of the CGL grid: w_k ~ (-1)^k, halved at both ends.
  bary_w_.resize(n);
  for (int k = 0; k < n; ++k) bary_w_(k) = (k % 2 == 0) ? 1.0 : -1.0;
  bary_w_(0) *= 0.5;
  bary_w_(degree) *= 0.5;

  // Differentiation matrix from the barycentric weights:
  //   D_ij = (w_j / w_i) / (x_i - x_j)   (i != j)
  //   D_ii = -sum_{j != i} D_ij          (negative-sum trick)
  D_.resize(n, n);
  for (int i = 0; i < n; ++i) {
    double rowsum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const double dij = (bary_w_(j) / bary_w_(i)) / (nodes_(i) - nodes_(j));
      D_(i, j) = dij;
      rowsum += dij;
    }
    D_(i, i) = -rowsum;
  }
}

Eigen::VectorXd CglGrid::timeGrid(double T) const {
  return (T / 2.0) * (nodes_.array() + 1.0).matrix();
}

Eigen::VectorXd CglGrid::interpWeights(double tau) const {
  const int n = numNodes();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);
  for (int k = 0; k < n; ++k) {
    if (std::abs(tau - nodes_(k)) < 1e-14) {
      w(k) = 1.0;
      return w;
    }
  }
  // Second (true) barycentric formula.
  double denom = 0.0;
  for (int k = 0; k < n; ++k) {
    w(k) = bary_w_(k) / (tau - nodes_(k));
    denom += w(k);
  }
  w /= denom;
  return w;
}

Eigen::VectorXd CglGrid::interpWeightsDerivative(double tau) const {
  // f'(tau) = sum_k dw_k f_k. On a node the derivative weights are the rows of D;
  // elsewhere differentiate the barycentric formula.
  const int n = numNodes();
  for (int k = 0; k < n; ++k) {
    if (std::abs(tau - nodes_(k)) < 1e-14) return D_.row(k).transpose();
  }
  Eigen::VectorXd a(n), da(n);
  double s = 0.0, ds = 0.0;
  for (int k = 0; k < n; ++k) {
    const double d = tau - nodes_(k);
    a(k) = bary_w_(k) / d;
    da(k) = -bary_w_(k) / (d * d);
    s += a(k);
    ds += da(k);
  }
  // w_k = a_k / s  ->  dw_k = (da_k s - a_k ds) / s^2
  Eigen::VectorXd dw(n);
  for (int k = 0; k < n; ++k) dw(k) = (da(k) * s - a(k) * ds) / (s * s);
  return dw;
}

Eigen::VectorXd clenshawCurtisWeights(int N) {
  Eigen::VectorXd w(N + 1);
  if (N == 0) {
    w(0) = 2.0;
    return w;
  }
  const double pi = 3.14159265358979323846;
  for (int k = 0; k <= N; ++k) {
    double s = 0.0;
    for (int j = 1; j <= N / 2; ++j) {
      const double bj = (2 * j == N) ? 1.0 : 2.0;
      s += bj / (4.0 * j * j - 1.0) * std::cos(2.0 * j * k * pi / N);
    }
    const double ck = (k == 0 || k == N) ? 1.0 : 2.0;
    w(k) = ck / N * (1.0 - s);
  }
  return w;
}

void gaussLegendre(int n, Eigen::VectorXd& nodes, Eigen::VectorXd& weights) {
  nodes.resize(n);
  weights.resize(n);
  for (int i = 0; i < n; ++i) {
    // Newton on the Legendre polynomial from the asymptotic root estimate.
    double x = std::cos(M_PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));
    double dp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0;
      double p1 = x;
      for (int k = 2; k <= n; ++k) {
        const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = static_cast<double>(n) * (x * p1 - p0) / (x * x - 1.0);
      const double dx = -p1 / dp;
      x += dx;
      if (std::abs(dx) < 1e-15) break;
    }
    const int idx = n - 1 - i;
    nodes(idx) = x;
    weights(idx) = 2.0 / ((1.0 - x * x) * dp * dp);
  }
}

}  // namespace leap
