#include "leap/linalg/global_metric.hpp"

#include <stdexcept>

#include <Eigen/SparseCholesky>
#ifdef LEAP_HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#ifdef LEAP_HAS_OPENMP
#include <omp.h>
#endif

namespace leap {

struct GlobalMetricSolver::Impl {
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> llt;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower> ldlt;
#ifdef LEAP_HAS_CHOLMOD
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>, Eigen::Lower> cholmod;
#endif
  Eigen::LDLT<Eigen::MatrixXd> dense;
  bool analyzed = false;
  Eigen::Index analyzedNnz = -1;
  std::vector<Eigen::Triplet<double>> trip;
};

GlobalMetricSolver::GlobalMetricSolver(const CompiledProblem& prob, const JunctionQuotient& quotient, LinearBackend backend)
    : ElementMetricAssembly(prob, quotient), backend_(backend), impl_(std::make_unique<Impl>()) {
  if (backend_ == LinearBackend::BlockSchur) throw std::invalid_argument("GlobalMetricSolver: block_schur is BlockSchurSolver");
  if (!linearBackendAvailable(backend_)) throw std::runtime_error(std::string("linear backend '") + linearBackendName(backend_) + "' is not compiled in");
  sepReduced_.assign(static_cast<size_t>(nSep_), -1);
  for (int r = 0; r < nz_; ++r)
    if (entry_[static_cast<size_t>(r)].separator >= 0) sepReduced_[static_cast<size_t>(entry_[static_cast<size_t>(r)].separator)] = r;
  M_.resize(nz_, nz_);
}

GlobalMetricSolver::~GlobalMetricSolver() = default;

void GlobalMetricSolver::beginAssembly() {
  ElementMetricAssembly::beginAssembly();
  factored_ = false;
}

int GlobalMetricSolver::globalOf(const Block& b, int c) const {
  return c < b.nInt ? b.interior[static_cast<size_t>(c)] : sepReduced_[static_cast<size_t>(b.separators[static_cast<size_t>(c - b.nInt)])];
}

void GlobalMetricSolver::assembleMatrix(double hTheta) {
  const int nb = static_cast<int>(blocks_.size());
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int bi = 0; bi < nb; ++bi) foldRows(blocks_[static_cast<size_t>(bi)]);
  if (backend_ == LinearBackend::Dense) {
    Mdense_ = Eigen::MatrixXd::Identity(nz_, nz_);
    for (const Block& b : blocks_)
      for (int j = 0; j < b.nLoc; ++j) {
        const int gj = globalOf(b, j);
        for (int i = 0; i < b.nLoc; ++i) Mdense_(globalOf(b, i), gj) += hTheta * b.Hloc(i, j);
      }
    return;
  }
  // lower triangle (every entry of every block, zeros included: the pattern is fixed)
  std::vector<Eigen::Triplet<double>>& trip = impl_->trip;
  trip.clear();
  for (int i = 0; i < nz_; ++i) trip.emplace_back(i, i, 1.0);
  for (const Block& b : blocks_)
    for (int j = 0; j < b.nLoc; ++j) {
      const int gj = globalOf(b, j);
      for (int i = 0; i < b.nLoc; ++i) {
        const int gi = globalOf(b, i);
        if (gi >= gj) trip.emplace_back(gi, gj, hTheta * b.Hloc(i, j));
      }
    }
  M_.setFromTriplets(trip.begin(), trip.end());
}

bool GlobalMetricSolver::factor(double hTheta) {
  assembleMatrix(hTheta);
  Impl& s = *impl_;
  bool ok = false;
  switch (backend_) {
    case LinearBackend::Dense:
      s.dense.compute(Mdense_);
      ok = s.dense.info() == Eigen::Success && (s.dense.vectorD().array().abs() > 1e-300).all();
      break;
    case LinearBackend::SparseLLT:
      if (!s.analyzed || s.analyzedNnz != M_.nonZeros()) {
        s.llt.analyzePattern(M_);
        s.analyzed = true;
        s.analyzedNnz = M_.nonZeros();
      }
      s.llt.factorize(M_);
      ok = s.llt.info() == Eigen::Success;
      break;
    case LinearBackend::SparseLDLT:
      if (!s.analyzed || s.analyzedNnz != M_.nonZeros()) {
        s.ldlt.analyzePattern(M_);
        s.analyzed = true;
        s.analyzedNnz = M_.nonZeros();
      }
      s.ldlt.factorize(M_);
      ok = s.ldlt.info() == Eigen::Success && (s.ldlt.vectorD().array().abs() > 1e-300).all();
      break;
    case LinearBackend::SparseCholmod:
#ifdef LEAP_HAS_CHOLMOD
      if (!s.analyzed || s.analyzedNnz != M_.nonZeros()) {
        s.cholmod.analyzePattern(M_);
        s.analyzed = true;
        s.analyzedNnz = M_.nonZeros();
      }
      s.cholmod.factorize(M_);
      ok = s.cholmod.info() == Eigen::Success;
#endif
      break;
    case LinearBackend::BlockSchur: break;
  }
  if (!ok) return false;
  ++nFactor_;
  factored_ = true;
  return true;
}

bool GlobalMetricSolver::solve(const Eigen::VectorXd& r, Eigen::VectorXd& dz) const {
  if (!factored_) return false;
  if (r.size() != nz_) throw std::invalid_argument("GlobalMetricSolver::solve: wrong rhs size");
  const Impl& s = *impl_;
  switch (backend_) {
    case LinearBackend::Dense: dz = s.dense.solve(r); break;
    case LinearBackend::SparseLLT: dz = s.llt.solve(r); break;
    case LinearBackend::SparseLDLT: dz = s.ldlt.solve(r); break;
    case LinearBackend::SparseCholmod:
#ifdef LEAP_HAS_CHOLMOD
      dz = s.cholmod.solve(r);
#endif
      break;
    case LinearBackend::BlockSchur: return false;
  }
  return dz.allFinite();
}

double GlobalMetricSolver::residualNorm(const Eigen::VectorXd& v, const Eigen::VectorXd& r) const {
  if (backend_ == LinearBackend::Dense) return (Mdense_ * v - r).norm();
  const Eigen::VectorXd Mv = M_.selfadjointView<Eigen::Lower>() * v;
  return (Mv - r).norm();
}

}  // namespace leap
