#pragma once

// The condensed linear step  M dz = r  with  M = I + h theta H  in the reduced (junction
// quotient) coordinates: the interface every backend implements, and the backend menu.
//
//   block_schur     exact block-Schur factorization: dense Cholesky (LDLT for an
//                   indefinite exact-Newton metric) per element interior in parallel,
//                   sparse Cholesky (CHOLMOD when available and enabled, else Eigen) on
//                   the junction separators. The default. Alias: cholmod_sn.
//   sparse_cholmod  the whole reduced metric as one sparse matrix, CHOLMOD supernodal
//                   Cholesky (needs LEAP_HAS_CHOLMOD).
//   sparse_llt      the same with Eigen's simplicial Cholesky (serial, always built).
//   sparse_ldlt     Eigen's simplicial LDLT (serial): accepts an indefinite metric.
//   dense           one dense LDLT of the reduced metric (small problems, testing).
//
// Every backend receives the metric per ELEMENT through the MetricSink interface
// (dense blocks and weighted Jacobian rows over full decision indices), accumulated
// into one dense local matrix per element over [interior | separators] by
// ElementMetricAssembly; the backends differ in what they factorize.

#include <string>

#include <Eigen/Dense>

#include "leap/alm/evaluator.hpp"

namespace leap {

enum class LinearBackend { BlockSchur, SparseCholmod, SparseLLT, SparseLDLT, Dense };

const char* linearBackendName(LinearBackend b);
// Accepts the names above and the alias "cholmod_sn"; throws std::runtime_error on an
// unknown name or a backend that is not compiled in.
LinearBackend parseLinearBackend(const std::string& name);
bool linearBackendAvailable(LinearBackend b);

struct LinearSolverOptions {
  bool useCholmod = true;       // sparse factorizations through CHOLMOD when available
  bool allowIndefinite = false; // exact-Newton metric: LDLT (dense) instead of Cholesky, so an
                                // indefinite I + h H still yields a step (the merit test rejects it)
};

class LinearStep : public MetricSink {
public:
  ~LinearStep() override = default;
  virtual int reducedSize() const = 0;
  // Zero every block before a new metric is scattered in.
  virtual void beginAssembly() = 0;
  // M = I + hTheta H; returns false when the factorization fails (not SPD / zero pivot).
  virtual bool factor(double hTheta) = 0;
  // M dz = r for a REDUCED right-hand side (size reducedSize()).
  virtual bool solve(const Eigen::VectorXd& r, Eigen::VectorXd& dz) const = 0;
  // ||M v - r|| for diagnostics.
  virtual double residualNorm(const Eigen::VectorXd& v, const Eigen::VectorXd& r) const = 0;
  virtual long numFactorizations() const = 0;
  virtual LinearBackend backend() const = 0;
};

}  // namespace leap
