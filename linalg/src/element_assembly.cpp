#include "leap/linalg/element_assembly.hpp"

#include <stdexcept>

namespace leap {

const char* linearBackendName(LinearBackend b) {
  switch (b) {
    case LinearBackend::BlockSchur: return "block_schur";
    case LinearBackend::SparseCholmod: return "sparse_cholmod";
    case LinearBackend::SparseLLT: return "sparse_llt";
    case LinearBackend::SparseLDLT: return "sparse_ldlt";
    case LinearBackend::Dense: return "dense";
  }
  return "?";
}

bool linearBackendAvailable(LinearBackend b) {
#ifdef LEAP_HAS_CHOLMOD
  (void)b;
  return true;
#else
  return b != LinearBackend::SparseCholmod;
#endif
}

LinearBackend parseLinearBackend(const std::string& name) {
  if (name == "cholmod_sn") return LinearBackend::BlockSchur;  // alias
  for (LinearBackend b : {LinearBackend::BlockSchur, LinearBackend::SparseCholmod, LinearBackend::SparseLLT, LinearBackend::SparseLDLT,
                          LinearBackend::Dense}) {
    if (name != linearBackendName(b)) continue;
    if (!linearBackendAvailable(b)) throw std::runtime_error("linear backend '" + name + "' is not compiled in (CHOLMOD missing)");
    return b;
  }
  throw std::runtime_error("unknown linear backend '" + name + "' (block_schur | sparse_cholmod | sparse_llt | sparse_ldlt | dense)");
}

ElementMetricAssembly::ElementMetricAssembly(const CompiledProblem& prob, const JunctionQuotient& quotient)
    : prob_(&prob), quotient_(&quotient) {
  const Layout& L = prob.layout();
  nz_ = quotient.reducedSize();
  const int M = static_cast<int>(L.elem.size());
  blocks_.resize(static_cast<size_t>(M));
  entry_.assign(static_cast<size_t>(nz_), Entry{});
  // separators: the reduced q and v of the last node of every element but the last
  std::vector<int> sepOfReduced(static_cast<size_t>(nz_), -1);
  for (int e = 0; e + 1 < M; ++e) {
    const ElementLayout& el = L.elem[static_cast<size_t>(e)];
    const int k = el.Nn - 1;
    for (int i = 0; i < el.dims.nq; ++i) {
      const int r = quotient.reducedOf(el.xQ(i, k));
      if (sepOfReduced[static_cast<size_t>(r)] < 0) sepOfReduced[static_cast<size_t>(r)] = nSep_++;
    }
    for (int i = 0; i < el.dims.nv; ++i) {
      const int r = quotient.reducedOf(el.xV(i, k));
      if (sepOfReduced[static_cast<size_t>(r)] < 0) sepOfReduced[static_cast<size_t>(r)] = nSep_++;
    }
  }
  for (int r = 0; r < nz_; ++r) {
    Entry& en = entry_[static_cast<size_t>(r)];
    en.element = quotient.ownerElement(r);
    en.separator = sepOfReduced[static_cast<size_t>(r)];
    if (en.separator < 0) {
      Block& b = blocks_[static_cast<size_t>(en.element)];
      en.interiorPos = static_cast<int>(b.interior.size());
      b.interior.push_back(r);
    }
  }
  // the separators each element touches: its own junction and the previous one
  for (int e = 0; e < M; ++e) {
    Block& b = blocks_[static_cast<size_t>(e)];
    b.sepSlot.assign(static_cast<size_t>(nSep_), -1);
    for (int j = std::max(0, e - 1); j <= std::min(e, M - 2); ++j) {
      const ElementLayout& ej = L.elem[static_cast<size_t>(j)];
      const int k = ej.Nn - 1;
      for (int i = 0; i < ej.dims.nq; ++i) {
        const int s = sepOfReduced[static_cast<size_t>(quotient.reducedOf(ej.xQ(i, k)))];
        if (b.sepSlot[static_cast<size_t>(s)] < 0) {
          b.sepSlot[static_cast<size_t>(s)] = static_cast<int>(b.separators.size());
          b.separators.push_back(s);
        }
      }
      for (int i = 0; i < ej.dims.nv; ++i) {
        const int s = sepOfReduced[static_cast<size_t>(quotient.reducedOf(ej.xV(i, k)))];
        if (b.sepSlot[static_cast<size_t>(s)] < 0) {
          b.sepSlot[static_cast<size_t>(s)] = static_cast<int>(b.separators.size());
          b.separators.push_back(s);
        }
      }
    }
    b.nInt = static_cast<int>(b.interior.size());
    b.nLoc = b.nInt + static_cast<int>(b.separators.size());
    b.Hloc.setZero(b.nLoc, b.nLoc);
    b.Jrows.resize(0, b.nLoc);
    b.wrows.resize(0);
    b.nRows = 0;
  }
}

void ElementMetricAssembly::beginAssembly() {
  for (Block& b : blocks_) {
    b.Hloc.setZero();
    b.nRows = 0;
  }
}

int ElementMetricAssembly::localColumn(const Block& b, int element, int globalIdx) const {
  const Entry& en = entry_[static_cast<size_t>(quotient_->reducedOf(globalIdx))];
  if (en.separator < 0) {
    // a coordinate of another element's interior: legal only for an all-zero column
    // (a reset's index list spans both node supports, its Jacobian reads q, v only)
    if (en.element != element) return -1;
    return en.interiorPos;
  }
  const int slot = b.sepSlot[static_cast<size_t>(en.separator)];
  if (slot < 0) throw std::logic_error("ElementMetricAssembly: element touches a separator it does not own");
  return b.nInt + slot;
}

void ElementMetricAssembly::resolveLocal(const Block& b, int element, const std::vector<int>& idx, std::vector<int>& local) const {
  local.resize(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) local[i] = localColumn(b, element, idx[i]);
}

void ElementMetricAssembly::addBlock(int element, const std::vector<int>& idx, const Eigen::MatrixXd& H) {
  const int n = static_cast<int>(idx.size());
  if (H.rows() != n || H.cols() != n) throw std::invalid_argument("ElementMetricAssembly::addBlock: block/index size mismatch");
  Block& b = blocks_[static_cast<size_t>(element)];
  std::vector<int> local;
  resolveLocal(b, element, idx, local);
  for (int j = 0; j < n; ++j) {
    const int lj = local[static_cast<size_t>(j)];
    if (lj < 0) {
      if (H.col(j).cwiseAbs().maxCoeff() != 0.0) throw std::logic_error("ElementMetricAssembly::addBlock: a term of one element couples the interior of another");
      continue;
    }
    for (int i = 0; i < n; ++i) {
      const int li = local[static_cast<size_t>(i)];
      if (li < 0) {
        if (H(i, j) != 0.0) throw std::logic_error("ElementMetricAssembly::addBlock: a term of one element couples the interior of another");
        continue;
      }
      b.Hloc(li, lj) += H(i, j);
    }
  }
}

void ElementMetricAssembly::addWeightedRows(int element, const std::vector<int>& idx, const Eigen::MatrixXd& J, const Eigen::VectorXd& w) {
  const int n = static_cast<int>(idx.size());
  const int rows = static_cast<int>(J.rows());
  if (J.cols() != n || w.size() != rows) throw std::invalid_argument("ElementMetricAssembly::addWeightedRows: size mismatch");
  if (rows == 0) return;
  Block& b = blocks_[static_cast<size_t>(element)];
  std::vector<int> local;
  resolveLocal(b, element, idx, local);
  if (b.nRows + rows > b.Jrows.rows()) {
    const int cap = std::max(2 * static_cast<int>(b.Jrows.rows()), b.nRows + rows);
    b.Jrows.conservativeResize(cap, b.nLoc);
    b.wrows.conservativeResize(cap);
  }
  b.Jrows.middleRows(b.nRows, rows).setZero();
  for (int c = 0; c < n; ++c) {
    const int lc = local[static_cast<size_t>(c)];
    if (lc < 0) {
      if (J.col(c).cwiseAbs().maxCoeff() != 0.0) throw std::logic_error("ElementMetricAssembly::addWeightedRows: a term of one element couples the interior of another");
      continue;
    }
    b.Jrows.block(b.nRows, lc, rows, 1) += J.col(c);
  }
  b.wrows.segment(b.nRows, rows) = w;
  b.nRows += rows;
}

void ElementMetricAssembly::foldRows(Block& b) const {
  if (b.nRows == 0) return;
  const Eigen::MatrixXd Jw = b.wrows.head(b.nRows).asDiagonal() * b.Jrows.topRows(b.nRows);
  b.Hloc.noalias() += b.Jrows.topRows(b.nRows).transpose() * Jw;
  b.nRows = 0;
}

}  // namespace leap
