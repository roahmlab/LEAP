#pragma once

// The global backends of the condensed linear step: the reduced metric assembled into
// ONE matrix from the per-element dense local blocks (lower triangle, fixed pattern)
// and factorized as a whole,
//   sparse_cholmod   CHOLMOD supernodal Cholesky (pattern analysed once),
//   sparse_llt       Eigen simplicial Cholesky,
//   sparse_ldlt      Eigen simplicial LDLT (an indefinite metric is accepted),
//   dense            Eigen dense LDLT.

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "leap/linalg/element_assembly.hpp"

namespace leap {

class GlobalMetricSolver final : public ElementMetricAssembly {
public:
  GlobalMetricSolver(const CompiledProblem& prob, const JunctionQuotient& quotient, LinearBackend backend);
  ~GlobalMetricSolver() override;

  void beginAssembly() override;
  bool factor(double hTheta) override;
  bool solve(const Eigen::VectorXd& r, Eigen::VectorXd& dz) const override;
  double residualNorm(const Eigen::VectorXd& v, const Eigen::VectorXd& r) const override;
  long numFactorizations() const override { return nFactor_; }
  LinearBackend backend() const override { return backend_; }

private:
  struct Impl;
  // The reduced index of local column c of block b.
  int globalOf(const Block& b, int c) const;
  void assembleMatrix(double hTheta);

  LinearBackend backend_;
  std::vector<int> sepReduced_;  // separator id -> reduced coordinate
  Eigen::SparseMatrix<double> M_; // lower triangle of I + hTheta H
  Eigen::MatrixXd Mdense_;
  std::unique_ptr<Impl> impl_;
  long nFactor_ = 0;
  bool factored_ = false;
};

}  // namespace leap
