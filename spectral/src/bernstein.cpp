#include "leap/spectral/bernstein.hpp"

#include <cmath>

namespace leap {

namespace {

Eigen::VectorXd binomialRow(int degree) {
  Eigen::VectorXd binom(degree + 1);
  binom(0) = 1.0;
  for (int j = 1; j <= degree; ++j)
    binom(j) = binom(j - 1) * static_cast<double>(degree - j + 1) / static_cast<double>(j);
  return binom;
}

}  // namespace

Eigen::VectorXd bernsteinBasisAt(int degree, double s) {
  const Eigen::VectorXd binom = binomialRow(degree);
  const double oms = 1.0 - s;
  Eigen::VectorXd b(degree + 1);
  for (int j = 0; j <= degree; ++j) b(j) = binom(j) * std::pow(s, j) * std::pow(oms, degree - j);
  return b;
}

Eigen::MatrixXd bernsteinBasisAtDegree(int ctrlDeg, const CglGrid& grid) {
  const int n = grid.numNodes();
  const Eigen::VectorXd& tau = grid.nodes();
  const Eigen::VectorXd binom = binomialRow(ctrlDeg);
  Eigen::MatrixXd B(n, ctrlDeg + 1);
  for (int k = 0; k < n; ++k) {
    const double s = 0.5 * (tau(k) + 1.0);
    const double oms = 1.0 - s;
    for (int j = 0; j <= ctrlDeg; ++j) B(k, j) = binom(j) * std::pow(s, j) * std::pow(oms, ctrlDeg - j);
  }
  return B;
}

Eigen::VectorXd bernsteinColumnScaling(const Eigen::MatrixXd& E) {
  Eigen::VectorXd sigma(E.cols());
  for (int j = 0; j < E.cols(); ++j) sigma(j) = 1.0 / std::sqrt(1.0 + E.col(j).squaredNorm());
  return sigma;
}

void deCasteljauSplit(const Eigen::MatrixXd& C, double s, Eigen::MatrixXd& left,
                      Eigen::MatrixXd& right) {
  const int n = static_cast<int>(C.cols());
  left.resize(C.rows(), n);
  right.resize(C.rows(), n);
  Eigen::MatrixXd P = C;
  left.col(0) = P.col(0);
  right.col(n - 1) = P.col(n - 1);
  for (int r = 1; r < n; ++r) {
    for (int j = 0; j < n - r; ++j) P.col(j) = (1.0 - s) * P.col(j) + s * P.col(j + 1);
    left.col(r) = P.col(0);
    right.col(n - 1 - r) = P.col(n - 1 - r);
  }
}

}  // namespace leap
