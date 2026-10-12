#include "leap/alm/junction_quotient.hpp"

namespace leap {

JunctionQuotient::JunctionQuotient(const CompiledProblem& prob, bool shareJunctions) : enabled_(shareJunctions) {
  const Layout& L = prob.layout();
  np_ = L.np;
  md_ = L.md;
  // full index -> element owner
  std::vector<int> elemOf(static_cast<size_t>(np_), 0);
  for (const ElementLayout& el : L.elem)
    for (int i = 0; i < el.nx; ++i) elemOf[static_cast<size_t>(el.x0 + i)] = el.index;
  // union-find over the aliased pairs (head coordinate -> tail coordinate)
  std::vector<int> parent(static_cast<size_t>(np_));
  for (int i = 0; i < np_; ++i) parent[static_cast<size_t>(i)] = i;
  identityRow_.assign(static_cast<size_t>(md_), 0);
  resetIdentity_.assign(prob.resets().size(), 0);
  if (enabled_) {
    for (const ResetInfo& r : prob.resets()) {
      const ElementLayout& Lt = L.elem[static_cast<size_t>(r.tailElem)];
      const ElementLayout& Lh = L.elem[static_cast<size_t>(r.headElem)];
      const bool continuity = r.reset->kind() == ResetKind::Continuity;
      for (int i = 0; i < Lt.dims.nq; ++i) parent[static_cast<size_t>(Lh.xQ(i, 0))] = Lt.xQ(i, Lt.Nn - 1);
      if (continuity)
        for (int i = 0; i < Lt.dims.nv; ++i) parent[static_cast<size_t>(Lh.xV(i, 0))] = Lt.xV(i, Lt.Nn - 1);
      // identity rows: q rows always, v rows for continuity
      const int nq = Lt.dims.nq, nv = Lt.dims.nv;
      for (int i = 0; i < nq; ++i) {
        identityRow_[static_cast<size_t>(r.row0 + i)] = 1;
        identityRows_.push_back(r.row0 + i);
      }
      if (continuity) {
        for (int i = 0; i < nv; ++i) {
          identityRow_[static_cast<size_t>(r.row0 + nq + i)] = 1;
          identityRows_.push_back(r.row0 + nq + i);
        }
        resetIdentity_[static_cast<size_t>(r.index)] = 1;
      }
    }
  }
  // path compression: chains tail <- head <- next head
  for (int i = 0; i < np_; ++i) {
    int p = i;
    while (parent[static_cast<size_t>(p)] != p) p = parent[static_cast<size_t>(p)];
    parent[static_cast<size_t>(i)] = p;
  }
  // number the representatives in ascending full order
  reducedOf_.assign(static_cast<size_t>(np_), -1);
  for (int i = 0; i < np_; ++i) {
    if (parent[static_cast<size_t>(i)] != i) continue;
    reducedOf_[static_cast<size_t>(i)] = static_cast<int>(representative_.size());
    representative_.push_back(i);
    multiplicity_.push_back(1);
    owner_.push_back(elemOf[static_cast<size_t>(i)]);
  }
  for (int i = 0; i < np_; ++i) {
    if (parent[static_cast<size_t>(i)] == i) continue;
    const int r = reducedOf_[static_cast<size_t>(parent[static_cast<size_t>(i)])];
    reducedOf_[static_cast<size_t>(i)] = r;
    ++multiplicity_[static_cast<size_t>(r)];
  }
  nz_ = static_cast<int>(representative_.size());
}

void JunctionQuotient::reduceCovector(const Eigen::VectorXd& Fx, Eigen::VectorXd& Fz) const {
  Fz.setZero(nz_);
  for (int i = 0; i < np_; ++i) Fz(reducedOf_[static_cast<size_t>(i)]) += Fx(i);
}

void JunctionQuotient::scatter(const Eigen::VectorXd& z, Eigen::VectorXd& x) const {
  x.resize(np_);
  for (int i = 0; i < np_; ++i) x(i) = z(reducedOf_[static_cast<size_t>(i)]);
}

void JunctionQuotient::enforce(Eigen::VectorXd& y) const {
  if (!enabled_) return;
  for (int i = 0; i < np_; ++i) {
    const int rep = representative_[static_cast<size_t>(reducedOf_[static_cast<size_t>(i)])];
    if (rep != i) y(i) = y(rep);
  }
  for (int row : identityRows_) y(np_ + row) = 0.0;
}

void JunctionQuotient::project(Eigen::VectorXd& y) const {
  if (!enabled_) return;
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(nz_);
  for (int i = 0; i < np_; ++i) mean(reducedOf_[static_cast<size_t>(i)]) += y(i);
  for (int r = 0; r < nz_; ++r) mean(r) /= static_cast<double>(multiplicity_[static_cast<size_t>(r)]);
  for (int i = 0; i < np_; ++i) y(i) = mean(reducedOf_[static_cast<size_t>(i)]);
  for (int row : identityRows_) y(np_ + row) = 0.0;
}

}  // namespace leap
