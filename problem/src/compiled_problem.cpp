#include "leap/problem/compiled_problem.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "leap/spectral/bernstein.hpp"

namespace leap {

// --- TermTable ----------------------------------------------------------------------------

int TermTable::intern(const std::string& name) {
  const auto it = index_.find(name);
  if (it != index_.end()) return it->second;
  const int id = static_cast<int>(names_.size());
  names_.push_back(name);
  index_.emplace(name, id);
  return id;
}

int TermTable::find(const std::string& name) const {
  const auto it = index_.find(name);
  return it == index_.end() ? -1 : it->second;
}

std::string TermTable::stripped(const std::string& termName) {
  const size_t slash = termName.find('/');
  return slash == std::string::npos ? termName : termName.substr(slash + 1);
}

// --- compile ------------------------------------------------------------------------------

namespace {

void nodeSupport(const ElementLayout& L, int k, std::vector<int>& cols) {
  const NodeDims& d = L.dims;
  cols.clear();
  cols.reserve(static_cast<size_t>(L.supportSize()));
  for (int i = 0; i < d.nq; ++i) cols.push_back(L.xQ(i, k));
  for (int i = 0; i < d.nv; ++i) cols.push_back(L.xV(i, k));
  for (int i = 0; i < d.na; ++i) cols.push_back(L.xA(i, k));
  for (int j = 0; j < L.Nc; ++j)
    for (int i = 0; i < d.nlam; ++i) cols.push_back(L.xC(i, j));
  if (L.uStored)
    for (int i = 0; i < d.nu; ++i) cols.push_back(L.xU(i, k));
}

// Decision support of an element-wide point row: [q, v, a, (u)] per node, then all C.
void elementSupport(const ElementLayout& L, std::vector<int>& cols, int& nState) {
  const NodeDims& d = L.dims;
  nState = d.nq + d.nv + d.na + (L.uStored ? d.nu : 0);
  cols.clear();
  cols.reserve(static_cast<size_t>(nState * L.Nn + d.nlam * L.Nc));
  for (int k = 0; k < L.Nn; ++k) {
    for (int i = 0; i < d.nq; ++i) cols.push_back(L.xQ(i, k));
    for (int i = 0; i < d.nv; ++i) cols.push_back(L.xV(i, k));
    for (int i = 0; i < d.na; ++i) cols.push_back(L.xA(i, k));
    if (L.uStored)
      for (int i = 0; i < d.nu; ++i) cols.push_back(L.xU(i, k));
  }
  for (int j = 0; j < L.Nc; ++j)
    for (int i = 0; i < d.nlam; ++i) cols.push_back(L.xC(i, j));
}

}  // namespace

void CompiledProblem::offNodeWeights(int e, double tau, Eigen::VectorXd& W, Eigen::VectorXd& Wlam) const {
  const ElementLayout& L = layout_.elem[static_cast<size_t>(e)];
  W = grids_[static_cast<size_t>(e)]->interpWeights(tau);
  if (L.Nc > 0)
    Wlam = bernsteinBasisAt(L.Nc - 1, 0.5 * (tau + 1.0)).cwiseProduct(L.sigma);
  else
    Wlam.resize(0);
}

void CompiledProblem::buildPointInfo(int e, const PointAttachment& att, PointInfo& info) const {
  const ElementLayout& L = layout_.elem[static_cast<size_t>(e)];
  const NodeDims& d = L.dims;
  info.kind = att.kind;
  info.elem = e;
  info.con = att.con.get();
  info.rowsPerPoint = att.con->rows(d);
  info.needs = att.con->modelNeeds();
  if (att.kind == PointKind::OffNode) {
    info.taus = att.taus;
    info.nPoints = static_cast<int>(att.taus.size());
    info.W.resize(static_cast<size_t>(info.nPoints));
    info.Wlam.resize(static_cast<size_t>(info.nPoints));
    for (int p = 0; p < info.nPoints; ++p) offNodeWeights(e, att.taus[static_cast<size_t>(p)], info.W[p], info.Wlam[p]);
    elementSupport(L, info.zIdxAll, info.nState);
    return;
  }
  // field-only rows
  info.needs = 0;
  if (att.kind == PointKind::ForceControlPoints) {
    info.field = Field::Lam;
    info.fieldDim = d.nlam;
    info.nPoints = static_cast<int>(att.controlPoints.size());
    for (int j : att.controlPoints) {
      Eigen::VectorXd w = Eigen::VectorXd::Zero(L.Nc);
      w(j) = L.sigma(j);  // the physical control point sigma_j C_j
      info.Wf.push_back(w);
    }
  } else {
    info.field = att.field;
    info.fieldDim = fieldSize(att.field, d);
    info.nPoints = static_cast<int>(att.weights.size());
    info.Wf = att.weights;
  }
  const int nCols = info.field == Field::Lam ? L.Nc : L.Nn;
  std::vector<char> used(static_cast<size_t>(nCols), 0);
  for (const Eigen::VectorXd& w : info.Wf) {
    if (w.size() != nCols) throw std::invalid_argument("compile: functional weights of '" + att.con->name() + "' have the wrong length");
    for (int c = 0; c < nCols; ++c)
      if (w(c) != 0.0) used[static_cast<size_t>(c)] = 1;
  }
  std::vector<int> position(static_cast<size_t>(nCols), -1);
  for (int c = 0; c < nCols; ++c) {
    if (!used[static_cast<size_t>(c)]) continue;
    position[static_cast<size_t>(c)] = static_cast<int>(info.fieldColumns.size());
    info.fieldColumns.push_back(c);
  }
  info.pointColumns.resize(static_cast<size_t>(info.nPoints));
  for (int p = 0; p < info.nPoints; ++p)
    for (int c = 0; c < nCols; ++c)
      if (info.Wf[static_cast<size_t>(p)](c) != 0.0) info.pointColumns[static_cast<size_t>(p)].push_back(position[static_cast<size_t>(c)]);
  for (int c : info.fieldColumns) {
    for (int i = 0; i < info.fieldDim; ++i) {
      switch (info.field) {
        case Field::Q: info.fieldZIdx.push_back(L.xQ(i, c)); break;
        case Field::V: info.fieldZIdx.push_back(L.xV(i, c)); break;
        case Field::A: info.fieldZIdx.push_back(L.xA(i, c)); break;
        case Field::Lam: info.fieldZIdx.push_back(L.xC(i, c)); break;
        case Field::U: info.fieldZIdx.push_back(L.xU(i, c)); break;
      }
    }
  }
}

CompiledProblem CompiledProblem::compile(const TrajectoryProblem& spec) {
  spec.validate();
  CompiledProblem P;
  P.model_ = spec.model;
  P.spec_ = spec;
  const int M = static_cast<int>(spec.elements.size());
  Layout& Lay = P.layout_;
  Lay.elem.resize(static_cast<size_t>(M));

  // --- primal layout ---
  int x = 0;
  double t0 = 0.0;
  for (int e = 0; e < M; ++e) {
    const Element& el = spec.elements[static_cast<size_t>(e)];
    ElementLayout& L = Lay.elem[static_cast<size_t>(e)];
    L.index = e;
    L.dims = P.model_->nodeDims(el.contacts);
    L.Nn = el.degree + 1;
    L.Nc = el.contacts.active.empty() ? 0 : el.forceControlPoints;
    L.T = el.T;
    L.t0 = t0;
    t0 += el.T;
    L.uStored = !el.deriveTorque;
    L.x0 = x;
    int o = 0;
    L.offQ = o;
    o += L.dims.nq * L.Nn;
    L.offV = o;
    o += L.dims.nv * L.Nn;
    L.offA = o;
    o += L.dims.na * L.Nn;
    if (L.Nc > 0) {
      L.offLam = o;
      o += L.dims.nlam * L.Nc;
    }
    if (L.uStored) {
      L.offU = o;
      o += L.dims.nu * L.Nn;
    }
    L.nx = o;
    x += o;
    P.grids_.push_back(std::make_unique<CglGrid>(el.degree));
    if (L.Nc > 0) {
      L.E = bernsteinBasisAtDegree(L.Nc - 1, *P.grids_.back());
      L.sigma = spec.preconditionForce ? bernsteinColumnScaling(L.E) : Eigen::VectorXd::Ones(L.Nc);
    }
  }
  Lay.np = x;

  // --- row space ---
  int row = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int e = 0; e < M; ++e) {
    const Element& el = spec.elements[static_cast<size_t>(e)];
    ElementLayout& L = Lay.elem[static_cast<size_t>(e)];
    const std::string pre = "e" + std::to_string(e) + "/";
    L.nodeRow0.resize(static_cast<size_t>(L.Nn));
    L.nodeRows.resize(static_cast<size_t>(L.Nn));
    for (int k = 0; k < L.Nn; ++k) {
      NodeInfo info;
      info.elem = e;
      info.node = k;
      info.row0 = row;
      L.nodeRow0[static_cast<size_t>(k)] = row;
      int m = 0;
      if (el.deriveTorque) info.needs |= kRnea | kRneaDerivs | kFootPos;  // u = ID at the node
      for (const NodeAttachment& att : el.nodeTerms) {
        if (!att.appliesTo(k, L.Nn)) continue;
        ResolvedTerm t;
        t.con = att.con.get();
        t.rowOffset = m;
        t.rows = att.con->rows(L.dims);
        t.term = P.terms_.intern(pre + att.con->name());
        info.needs |= att.con->modelNeeds();
        const bool ineq = att.con->sense() == Sense::Inequality;
        for (int r = 0; r < t.rows; ++r) {
          P.rowTerm_.push_back(t.term);
          P.rowIneq_.push_back(ineq ? 1 : 0);
          P.rowInfo_.push_back({RowFamily::Node, t.term, e, k, nan, -1, r, ineq});
        }
        m += t.rows;
        info.terms.push_back(t);
      }
      for (const std::shared_ptr<Cost>& c : el.costs) info.needs |= c->modelNeeds();
      info.rows = m;
      L.nodeRows[static_cast<size_t>(k)] = m;
      row += m;
      nodeSupport(L, k, info.zIdx);
      P.nodes_.push_back(std::move(info));
    }
    // kinematic consistency rows
    L.kin1Row0 = row;
    const int tid1 = P.terms_.intern(pre + "kin1");
    for (int r = 0; r < L.dims.nq * L.Nn; ++r) {
      P.rowTerm_.push_back(tid1);
      P.rowIneq_.push_back(0);
      P.rowInfo_.push_back({RowFamily::Kinematic1, tid1, e, r / L.dims.nq, nan, -1, r % L.dims.nq, false});
    }
    row += L.dims.nq * L.Nn;
    L.kin2Row0 = row;
    const int tid2 = P.terms_.intern(pre + "kin2");
    for (int r = 0; r < L.dims.nq * L.Nn; ++r) {
      P.rowTerm_.push_back(tid2);
      P.rowIneq_.push_back(0);
      P.rowInfo_.push_back({RowFamily::Kinematic2, tid2, e, r / L.dims.nq, nan, -1, r % L.dims.nq, false});
    }
    row += L.dims.nq * L.Nn;
    // point rows
    L.pointRow0 = row;
    for (const PointAttachment& att : el.pointTerms) {
      PointInfo info;
      P.buildPointInfo(e, att, info);
      if (info.nPoints == 0) continue;
      info.term = P.terms_.intern(pre + att.con->name() + "_interp");
      info.row0 = row;
      const bool ineq = att.con->sense() == Sense::Inequality;
      for (int p = 0; p < info.nPoints; ++p) {
        const double tau = att.kind == PointKind::OffNode ? att.taus[static_cast<size_t>(p)] : nan;
        for (int r = 0; r < info.rowsPerPoint; ++r) {
          P.rowTerm_.push_back(info.term);
          P.rowIneq_.push_back(ineq ? 1 : 0);
          P.rowInfo_.push_back({RowFamily::Point, info.term, e, -1, tau, p, r, ineq});
        }
      }
      row += info.nPoints * info.rowsPerPoint;
      P.points_.push_back(std::move(info));
    }
    L.pointRows = row - L.pointRow0;
    // defect cost (no rows)
    if (el.defect.rho > 0.0 && !el.defect.taus.empty()) {
      DefectInfo di;
      di.elem = e;
      di.con = el.defect.dynamics.get();
      di.rowsPerPoint = di.con->rows(L.dims);
      di.nPoints = static_cast<int>(el.defect.taus.size());
      di.needs = di.con->modelNeeds();
      di.rho = el.defect.rho;
      di.W.resize(static_cast<size_t>(di.nPoints));
      di.Wlam.resize(static_cast<size_t>(di.nPoints));
      for (int p = 0; p < di.nPoints; ++p) {
        P.offNodeWeights(e, el.defect.taus[static_cast<size_t>(p)], di.W[p], di.Wlam[p]);
        di.wq.push_back(el.defect.weights.empty() ? 2.0 / static_cast<double>(di.nPoints) : el.defect.weights[static_cast<size_t>(p)]);
      }
      elementSupport(L, di.zIdxAll, di.nState);
      P.defects_.push_back(std::move(di));
    }
  }
  // resets
  const int resetTerm = P.terms_.intern("reset");
  for (int t = 0; t + 1 < M; ++t) {
    ResetInfo ri;
    ri.index = t;
    ri.reset = spec.resets[static_cast<size_t>(t)].get();
    ri.tailElem = t;
    ri.headElem = t + 1;
    ri.row0 = row;
    ri.rows = ri.reset->rows();
    ri.term = resetTerm;
    Lay.resetRow0.push_back(row);
    const ElementLayout& Lt = Lay.elem[static_cast<size_t>(t)];
    const ElementLayout& Lh = Lay.elem[static_cast<size_t>(t + 1)];
    std::vector<int> tail, head;
    nodeSupport(Lt, Lt.Nn - 1, tail);
    nodeSupport(Lh, 0, head);
    ri.zIdx = tail;
    ri.zIdx.insert(ri.zIdx.end(), head.begin(), head.end());
    for (int r = 0; r < ri.rows; ++r) {
      P.rowTerm_.push_back(resetTerm);
      P.rowIneq_.push_back(0);
      P.rowInfo_.push_back({RowFamily::Reset, resetTerm, t, -1, nan, -1, r, false});
    }
    row += ri.rows;
    P.resets_.push_back(std::move(ri));
  }
  Lay.md = row;
  return P;
}

void CompiledProblem::relocateOffNodeTaus(int pointIdx, const std::vector<double>& newTaus) {
  if (pointIdx < 0 || pointIdx >= static_cast<int>(points_.size()))
    throw std::out_of_range("relocateOffNodeTaus: point index out of range");
  PointInfo& info = points_[static_cast<size_t>(pointIdx)];
  if (info.kind != PointKind::OffNode) throw std::invalid_argument("relocateOffNodeTaus: not an off-node attachment");
  if (static_cast<int>(newTaus.size()) != info.nPoints)
    throw std::invalid_argument("relocateOffNodeTaus: the point count must not change");
  for (double t : newTaus)
    if (!(t > -1.0 && t < 1.0)) throw std::invalid_argument("relocateOffNodeTaus: tau must lie in (-1, 1)");
  info.taus = newTaus;
  for (int p = 0; p < info.nPoints; ++p) {
    offNodeWeights(info.elem, newTaus[static_cast<size_t>(p)], info.W[p], info.Wlam[p]);
    for (int r = 0; r < info.rowsPerPoint; ++r) rowInfo_[static_cast<size_t>(info.row0 + p * info.rowsPerPoint + r)].tau = newTaus[static_cast<size_t>(p)];
  }
}

void CompiledProblem::setNodeCostWeights(std::vector<double> w) {
  if (!w.empty() && w.size() != nodes_.size()) throw std::invalid_argument("setNodeCostWeights: one weight per node");
  nodeCostWeight_ = std::move(w);
}

void CompiledProblem::scaleDefectRho(double factor) {
  for (DefectInfo& d : defects_) d.rho *= factor;
}

void CompiledProblem::stateAt(const Eigen::VectorXd& y, int elem, double tau, Eigen::VectorXd& q, Eigen::VectorXd& v,
                              Eigen::VectorXd& a, Eigen::VectorXd& lam) const {
  const ElementLayout& L = layout_.elem[static_cast<size_t>(elem)];
  const ElementView X(y.data(), L);
  Eigen::VectorXd W, Wlam;
  offNodeWeights(elem, tau, W, Wlam);
  q.noalias() = X.Q * W;
  v.noalias() = X.V * W;
  a.noalias() = X.A * W;
  if (L.Nc > 0)
    lam.noalias() = X.C * Wlam;
  else
    lam.resize(0);
}

Eigen::VectorXd CompiledProblem::storedU(const Eigen::VectorXd& y, int elem, double tau) const {
  const ElementLayout& L = layout_.elem[static_cast<size_t>(elem)];
  if (!L.uStored) throw std::logic_error("storedU: the torque of this element is derived");
  const ElementView X(y.data(), L);
  return X.U * grids_[static_cast<size_t>(elem)]->interpWeights(tau);
}

Eigen::VectorXd CompiledProblem::physicalControlPoint(const Eigen::VectorXd& y, int elem, int j) const {
  const ElementLayout& L = layout_.elem[static_cast<size_t>(elem)];
  const ElementView X(y.data(), L);
  return L.sigma(j) * X.C.col(j);
}

Eigen::MatrixXd CompiledProblem::nodeForces(const Eigen::VectorXd& y, int elem) const {
  const ElementLayout& L = layout_.elem[static_cast<size_t>(elem)];
  if (L.Nc == 0) return Eigen::MatrixXd(L.dims.nlam, L.Nn);
  const ElementView X(y.data(), L);
  return (X.C * L.sigma.asDiagonal()) * L.E.transpose();
}

}  // namespace leap
