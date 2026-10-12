#pragma once

// Chebyshev-Gauss-Lobatto (CGL) grid on tau in [-1, 1].
//
// Nodes are ASCENDING in time: node 0 <-> tau = -1 <-> t = 0 and node N <-> tau = +1
// <-> t = T. Every trajectory field of an element is a Lagrange interpolant through
// these nodes, so a field is stored by its nodal values and differentiated by the
// spectral differentiation matrix D (d/dtau at the nodes). The physical time
// derivative on an element of duration T is (2/T) * D.
//
// The object is immutable after construction and shared by every element of the same
// degree.

#include <Eigen/Dense>

namespace leap {

class CglGrid {
public:
  explicit CglGrid(int degree);

  int degree() const { return degree_; }
  int numNodes() const { return degree_ + 1; }

  // Nodes tau_k = -cos(pi k / N), ascending.
  const Eigen::VectorXd& nodes() const { return nodes_; }
  // Differentiation matrix: (d f / d tau)(tau_i) = sum_j D(i, j) f(tau_j).
  const Eigen::MatrixXd& D() const { return D_; }
  // Barycentric weights of the second barycentric formula.
  const Eigen::VectorXd& baryWeights() const { return bary_w_; }

  // Physical time of each node on an element of duration T: t = (T/2)(tau + 1).
  Eigen::VectorXd timeGrid(double T) const;

  // Interpolation weights w(tau) with f(tau) = sum_k w_k f(tau_k). Exact one-hot when
  // tau coincides with a node.
  Eigen::VectorXd interpWeights(double tau) const;

  // d/dtau of the interpolation weights: f'(tau) = sum_k dw_k f(tau_k).
  Eigen::VectorXd interpWeightsDerivative(double tau) const;

private:
  int degree_;
  Eigen::VectorXd nodes_;
  Eigen::MatrixXd D_;
  Eigen::VectorXd bary_w_;
};

// Clenshaw-Curtis quadrature weights on the N+1 CGL nodes of [-1, 1] (symmetric, so
// the ascending node order needs no permutation): int_{-1}^{1} f ~ sum_k w_k f(tau_k).
Eigen::VectorXd clenshawCurtisWeights(int degree);

// Gauss-Legendre nodes and weights on [-1, 1], ascending nodes.
void gaussLegendre(int n, Eigen::VectorXd& nodes, Eigen::VectorXd& weights);

}  // namespace leap
