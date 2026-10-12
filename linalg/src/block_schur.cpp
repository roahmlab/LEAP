#include "leap/linalg/block_schur.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/SparseCholesky>
#ifdef LEAP_HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#ifdef LEAP_HAS_OPENMP
#include <omp.h>
#endif

namespace leap {

struct BlockSchurSolver::SparseImpl {
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> eigenLlt;
#ifdef LEAP_HAS_CHOLMOD
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>, Eigen::Lower> cholmodLlt;
#endif
  bool useCholmod = false;
  bool analyzed = false;
};

BlockSchurSolver::BlockSchurSolver(const CompiledProblem& prob, const JunctionQuotient& quotient,
                                   const LinearSolverOptions& opt)
    : ElementMetricAssembly(prob, quotient), opt_(opt), factors_(blocks_.size()), sparse_(std::make_unique<SparseImpl>()) {
  Hss_.setZero(nSep_, nSep_);
  sparse_->useCholmod = false;
#ifdef LEAP_HAS_CHOLMOD
  sparse_->useCholmod = opt_.useCholmod;
#endif
}

BlockSchurSolver::~BlockSchurSolver() = default;

void BlockSchurSolver::beginAssembly() {
  ElementMetricAssembly::beginAssembly();
  factored_ = false;
}

bool BlockSchurSolver::factor(double hTheta) {
  hTheta_ = hTheta;
  const int nb = static_cast<int>(blocks_.size());
  bool ok = true;
  // per element: the rank-k update of the stacked rows, then the dense interior Cholesky
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int bi = 0; bi < nb; ++bi) {
    Block& b = blocks_[static_cast<size_t>(bi)];
    Factor& f = factors_[static_cast<size_t>(bi)];
    foldRows(b);
    const int n = b.nInt;
    const int ns = static_cast<int>(b.separators.size());
    f.D = b.Hloc.topLeftCorner(n, n);
    f.C = b.Hloc.topRightCorner(n, ns);
    f.MD = hTheta * f.D;
    for (int i = 0; i < n; ++i) f.MD(i, i) += 1.0;
    f.MC = hTheta * f.C;
    if (opt_.allowIndefinite) {
      f.ldlt.compute(f.MD);
      if (f.ldlt.info() != Eigen::Success || !(f.ldlt.vectorD().array().abs() > 1e-300).all()) {
#ifdef LEAP_HAS_OPENMP
#pragma omp critical
#endif
        ok = false;
        continue;
      }
      f.W = f.ldlt.solve(f.MC);
      continue;
    }
    f.llt.compute(f.MD);
    if (f.llt.info() != Eigen::Success) {
#ifdef LEAP_HAS_OPENMP
#pragma omp critical
#endif
      ok = false;
      continue;
    }
    f.W = f.llt.solve(f.MC);
  }
  if (!ok) return false;
  // the separator-separator part, summed over the elements that touch each junction
  Hss_.setZero(nSep_, nSep_);
  for (const Block& b : blocks_) {
    const int ns = static_cast<int>(b.separators.size());
    for (int j = 0; j < ns; ++j)
      for (int i = 0; i < ns; ++i) Hss_(b.separators[static_cast<size_t>(i)], b.separators[static_cast<size_t>(j)]) += b.Hloc(b.nInt + i, b.nInt + j);
  }
  // Schur complement on the separators
  Eigen::MatrixXd S = hTheta * Hss_;
  for (int i = 0; i < nSep_; ++i) S(i, i) += 1.0;
  for (size_t bi = 0; bi < blocks_.size(); ++bi) {
    const Block& b = blocks_[bi];
    const Factor& f = factors_[bi];
    const int ns = static_cast<int>(b.separators.size());
    if (ns == 0 || b.interior.empty()) continue;
    const Eigen::MatrixXd update = f.MC.transpose() * f.W;
    for (int i = 0; i < ns; ++i)
      for (int j = 0; j < ns; ++j) S(b.separators[static_cast<size_t>(i)], b.separators[static_cast<size_t>(j)]) -= update(i, j);
  }
  if (opt_.allowIndefinite) {
    Sldlt_.compute(S);
    if (Sldlt_.info() != Eigen::Success || !(Sldlt_.vectorD().array().abs() > 1e-300).all()) return false;
    ++nFactor_;
    factored_ = true;
    return true;
  }
  // sparse factorization of the (block tridiagonal) separator system
  std::vector<Eigen::Triplet<double>> trip;
  for (int j = 0; j < nSep_; ++j)
    for (int i = j; i < nSep_; ++i)
      if (S(i, j) != 0.0) trip.emplace_back(i, j, S(i, j));
  Sschur_.resize(nSep_, nSep_);
  Sschur_.setFromTriplets(trip.begin(), trip.end());
  if (nSep_ > 0) {
#ifdef LEAP_HAS_CHOLMOD
    if (sparse_->useCholmod) {
      if (!sparse_->analyzed) {
        sparse_->cholmodLlt.analyzePattern(Sschur_);
        sparse_->analyzed = true;
      }
      sparse_->cholmodLlt.factorize(Sschur_);
      if (sparse_->cholmodLlt.info() != Eigen::Success) return false;
    } else
#endif
    {
      if (!sparse_->analyzed) {
        sparse_->eigenLlt.analyzePattern(Sschur_);
        sparse_->analyzed = true;
      }
      sparse_->eigenLlt.factorize(Sschur_);
      if (sparse_->eigenLlt.info() != Eigen::Success) return false;
    }
  }
  ++nFactor_;
  factored_ = true;
  return true;
}

bool BlockSchurSolver::solve(const Eigen::VectorXd& r, Eigen::VectorXd& dz) const {
  if (!factored_) return false;
  if (r.size() != nz_) throw std::invalid_argument("BlockSchurSolver::solve: wrong rhs size");
  const int nb = static_cast<int>(blocks_.size());
  // gather
  std::vector<Eigen::VectorXd> rb(static_cast<size_t>(nb)), zb(static_cast<size_t>(nb));
  Eigen::VectorXd rs(nSep_);
  for (int i = 0; i < nz_; ++i) {
    const Entry& e = entry_[static_cast<size_t>(i)];
    if (e.separator >= 0) rs(e.separator) = r(i);
  }
  for (int bi = 0; bi < nb; ++bi) {
    const Block& b = blocks_[static_cast<size_t>(bi)];
    rb[static_cast<size_t>(bi)].resize(static_cast<int>(b.interior.size()));
    for (size_t k = 0; k < b.interior.size(); ++k) rb[static_cast<size_t>(bi)](static_cast<int>(k)) = r(b.interior[k]);
  }
  // z_b = D_b^-1 r_b;  r_s -= C_b' z_b
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int bi = 0; bi < nb; ++bi) {
    const Factor& f = factors_[static_cast<size_t>(bi)];
    if (opt_.allowIndefinite)
      zb[static_cast<size_t>(bi)] = f.ldlt.solve(rb[static_cast<size_t>(bi)]);
    else
      zb[static_cast<size_t>(bi)] = f.llt.solve(rb[static_cast<size_t>(bi)]);
  }
  for (int bi = 0; bi < nb; ++bi) {
    const Block& b = blocks_[static_cast<size_t>(bi)];
    const Factor& f = factors_[static_cast<size_t>(bi)];
    if (b.separators.empty() || b.interior.empty()) continue;
    const Eigen::VectorXd t = f.MC.transpose() * zb[static_cast<size_t>(bi)];
    for (size_t k = 0; k < b.separators.size(); ++k) rs(b.separators[k]) -= t(static_cast<int>(k));
  }
  Eigen::VectorXd xs(nSep_);
  if (nSep_ > 0) {
    if (opt_.allowIndefinite) {
      xs = Sldlt_.solve(rs);
    } else {
#ifdef LEAP_HAS_CHOLMOD
      if (sparse_->useCholmod)
        xs = sparse_->cholmodLlt.solve(rs);
      else
#endif
        xs = sparse_->eigenLlt.solve(rs);
    }
  }
  dz.resize(nz_);
  for (int i = 0; i < nz_; ++i) {
    const Entry& e = entry_[static_cast<size_t>(i)];
    if (e.separator >= 0) dz(i) = xs(e.separator);
  }
  for (int bi = 0; bi < nb; ++bi) {
    const Block& b = blocks_[static_cast<size_t>(bi)];
    const Factor& f = factors_[static_cast<size_t>(bi)];
    Eigen::VectorXd xb = zb[static_cast<size_t>(bi)];
    if (!b.separators.empty() && !b.interior.empty()) {
      Eigen::VectorXd xsLocal(static_cast<int>(b.separators.size()));
      for (size_t k = 0; k < b.separators.size(); ++k) xsLocal(static_cast<int>(k)) = xs(b.separators[k]);
      xb.noalias() -= f.W * xsLocal;
    }
    for (size_t k = 0; k < b.interior.size(); ++k) dz(b.interior[k]) = xb(static_cast<int>(k));
  }
  return true;
}

double BlockSchurSolver::residualNorm(const Eigen::VectorXd& v, const Eigen::VectorXd& r) const {
  // M v in the partitioned form
  Eigen::VectorXd Mv = v;  // the identity part
  Eigen::VectorXd vs(nSep_);
  for (int i = 0; i < nz_; ++i) {
    const Entry& e = entry_[static_cast<size_t>(i)];
    if (e.separator >= 0) vs(e.separator) = v(i);
  }
  Eigen::VectorXd acc = hTheta_ * (Hss_ * vs);
  for (size_t bi = 0; bi < blocks_.size(); ++bi) {
    const Block& b = blocks_[bi];
    const Factor& f = factors_[bi];
    if (b.interior.empty()) continue;
    Eigen::VectorXd vb(static_cast<int>(b.interior.size()));
    for (size_t k = 0; k < b.interior.size(); ++k) vb(static_cast<int>(k)) = v(b.interior[k]);
    Eigen::VectorXd vsl(static_cast<int>(b.separators.size()));
    for (size_t k = 0; k < b.separators.size(); ++k) vsl(static_cast<int>(k)) = vs(b.separators[k]);
    const Eigen::VectorXd mb = hTheta_ * (f.D * vb + f.C * vsl);
    for (size_t k = 0; k < b.interior.size(); ++k) Mv(b.interior[k]) += mb(static_cast<int>(k));
    const Eigen::VectorXd ms = hTheta_ * (f.C.transpose() * vb);
    for (size_t k = 0; k < b.separators.size(); ++k) acc(b.separators[k]) += ms(static_cast<int>(k));
  }
  for (int i = 0; i < nz_; ++i) {
    const Entry& e = entry_[static_cast<size_t>(i)];
    if (e.separator >= 0) Mv(i) += acc(e.separator);
  }
  return (Mv - r).norm();
}

}  // namespace leap
