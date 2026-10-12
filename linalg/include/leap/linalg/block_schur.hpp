#pragma once

// The block-Schur backend of the condensed linear step  M dz = r,  M = I + h theta H:
//
//       M = [ D_1            C_1 ]            S = M_ss - sum_b C_b' D_b^-1 C_b
//           [      ...       ... ]
//           [          D_nb  C_nb]      solve:  z_b = D_b^-1 r_b,  r_s -= sum_b C_b' z_b,
//           [ C_1' ... C_nb' M_ss]              x_s = S^-1 r_s,    x_b = z_b - D_b^-1 C_b x_s
//
//   every D_b is a dense Cholesky (parallel over the elements), S a sparse Cholesky
//   (CHOLMOD when available and enabled, else Eigen). With allowIndefinite (the exact-
//   Newton metric) the dense factorizations are LDLT and S is factorized densely.

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "leap/linalg/element_assembly.hpp"

namespace leap {

class BlockSchurSolver final : public ElementMetricAssembly {
public:
  BlockSchurSolver(const CompiledProblem& prob, const JunctionQuotient& quotient, const LinearSolverOptions& opt);
  ~BlockSchurSolver() override;

  void beginAssembly() override;
  bool factor(double hTheta) override;
  bool solve(const Eigen::VectorXd& r, Eigen::VectorXd& dz) const override;
  double residualNorm(const Eigen::VectorXd& v, const Eigen::VectorXd& r) const override;
  long numFactorizations() const override { return nFactor_; }
  LinearBackend backend() const override { return LinearBackend::BlockSchur; }

private:
  struct Factor {
    Eigen::MatrixXd D, C;        // H blocks (views of Hloc, kept for the residual)
    Eigen::MatrixXd MD, MC, W;   // metric blocks and D^-1 C
    Eigen::LLT<Eigen::MatrixXd> llt;
    Eigen::LDLT<Eigen::MatrixXd> ldlt;  // allowIndefinite
  };
  struct SparseImpl;

  LinearSolverOptions opt_;
  std::vector<Factor> factors_;
  Eigen::MatrixXd Hss_;  // dense separator-separator part of H (small, block tridiagonal)
  Eigen::SparseMatrix<double> Sschur_;
  Eigen::LDLT<Eigen::MatrixXd> Sldlt_;  // allowIndefinite: dense LDLT of the separator system
  std::unique_ptr<SparseImpl> sparse_;
  double hTheta_ = 0.0;
  long nFactor_ = 0;
  bool factored_ = false;
};

}  // namespace leap
