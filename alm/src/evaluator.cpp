#include "leap/alm/evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "leap/model/model_hessians.hpp"

#ifdef LEAP_HAS_OPENMP
#include <omp.h>
#endif

namespace leap {

namespace {

int threadCount() {
#ifdef LEAP_HAS_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int threadId() {
#ifdef LEAP_HAS_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

}  // namespace

// Per-thread scratch: model data, cache, dense temporaries.
struct AlmEvaluator::Scratch {
  explicit Scratch(const RobotModel& m) : data(m.makeData()) {
    m.initCache(cache);
    hs.model = &m;
    hs.data = data.get();
  }
  ModelDataPtr data;
  ModelEvalCache cache;
  HessianScratch hs;
  std::unique_ptr<ModelHessians> hessians;  // exact-Newton metric only
  Eigen::VectorXd q, v, a, lam, u, gphys, gtmp, wseg;
  Eigen::MatrixXd Jphys, Ju, S, T, Hphys, Hl, Jb, Jtmp;
};

AlmEvaluator::AlmEvaluator(const CompiledProblem& prob, const EvaluatorOptions& opt) : prob_(&prob), opt_(opt) {
  quotient_ = std::make_unique<JunctionQuotient>(prob, opt.shareJunctions);
  const int nt = std::max(1, opt.threads > 0 ? opt.threads : threadCount());
  for (int t = 0; t < nt; ++t) {
    scratch_.push_back(std::make_unique<Scratch>(prob.model()));
    if (!opt.gaussNewton) {
      scratch_.back()->hessians = std::make_unique<PinocchioHessians>(prob.model());
      scratch_.back()->hs.hessians = scratch_.back()->hessians.get();
    }
  }
  nodeBase_.resize(prob.layout().elem.size());
  int base = 0;
  for (size_t e = 0; e < prob.layout().elem.size(); ++e) {
    nodeBase_[e] = base;
    base += prob.layout().elem[e].Nn;
  }
  const size_t M = prob.layout().elem.size();
  pointsOfElem_.assign(M, {});
  defectsOfElem_.assign(M, {});
  resetsOfHead_.assign(M, {});
  for (size_t pi = 0; pi < prob.points().size(); ++pi) pointsOfElem_[static_cast<size_t>(prob.points()[pi].elem)].push_back(static_cast<int>(pi));
  for (size_t di = 0; di < prob.defects().size(); ++di) defectsOfElem_[static_cast<size_t>(prob.defects()[di].elem)].push_back(static_cast<int>(di));
  for (const ResetInfo& ri : prob.resets()) resetsOfHead_[static_cast<size_t>(ri.headElem)].push_back(ri.index);
}

AlmEvaluator::~AlmEvaluator() = default;

// --- lifts ------------------------------------------------------------------------------

void AlmEvaluator::torqueJacobian(const ElementLayout& L, const ModelEvalCache& c, Eigen::MatrixXd& Ju) const {
  // u_ID = [gdyn]_act: d/dq = dgdyn_dq, d/dv = dtau_dv, d/da = M, d/dlam = -Jc'.
  const NodeDims& d = L.dims;
  const std::vector<int>& act = prob_->model().actuatedIdx();
  const Element& el = prob_->spec().elements[static_cast<size_t>(L.index)];
  Ju.setZero(d.nu, d.nP());
  for (int j = 0; j < d.nu; ++j) {
    const int rr = act[static_cast<size_t>(j)];
    Ju.row(j).segment(d.offQ(), d.nq) = c.dgdyn_dq.row(rr);
    Ju.row(j).segment(d.offV(), d.nv) = c.dtau_dv.row(rr);
    Ju.row(j).segment(d.offA(), d.na) = c.M.row(rr);
  }
  int o = 0;
  for (const ContactRef& cr : el.contacts.active) {
    const Eigen::MatrixXd& Jc = c.footJ[static_cast<size_t>(cr.index)];
    for (int j = 0; j < d.nu; ++j) Ju.block(j, d.offLam() + o, 1, cr.dim) = -Jc.col(act[static_cast<size_t>(j)]).transpose();
    o += cr.dim;
  }
}

void AlmEvaluator::buildNodeLift(const ElementLayout& L, int k, const ModelEvalCache& c, bool torqueChain,
                                 Scratch& s, Eigen::MatrixXd& S) const {
  const NodeDims& d = L.dims;
  const int ns = L.supportSize();
  S.setZero(d.nP(), ns);
  for (int i = 0; i < d.nq; ++i) S(d.offQ() + i, i) = 1.0;
  for (int i = 0; i < d.nv; ++i) S(d.offV() + i, d.nq + i) = 1.0;
  for (int i = 0; i < d.na; ++i) S(d.offA() + i, d.nq + d.nv + i) = 1.0;
  const int offC = L.supportOffC();
  for (int j = 0; j < L.Nc; ++j) {
    const double w = L.E(k, j) * L.sigma(j);
    for (int i = 0; i < d.nlam; ++i) S(d.offLam() + i, offC + j * d.nlam + i) = w;
  }
  if (L.uStored) {
    const int offU = L.supportOffU();
    for (int i = 0; i < d.nu; ++i) S(d.offU() + i, offU + i) = 1.0;
  } else if (torqueChain) {
    torqueJacobian(L, c, s.Ju);
    S.middleRows(d.offU(), d.nu).noalias() = s.Ju * S;
  }
}

void AlmEvaluator::buildPointLift(const ElementLayout& L, const Eigen::VectorXd& W, const Eigen::VectorXd& Wlam,
                                  const ModelEvalCache& c, bool torqueChain, Scratch& s, Eigen::MatrixXd& T) const {
  // decision support: [q, v, a, (u)] per node (nState each), then every control point
  const NodeDims& d = L.dims;
  const int nState = d.nq + d.nv + d.na + (L.uStored ? d.nu : 0);
  const int nCols = nState * L.Nn + d.nlam * L.Nc;
  T.setZero(d.nP(), nCols);
  for (int k = 0; k < L.Nn; ++k) {
    const double w = W(k);
    const int base = k * nState;
    for (int i = 0; i < d.nq; ++i) T(d.offQ() + i, base + i) = w;
    for (int i = 0; i < d.nv; ++i) T(d.offV() + i, base + d.nq + i) = w;
    for (int i = 0; i < d.na; ++i) T(d.offA() + i, base + d.nq + d.nv + i) = w;
    if (L.uStored)
      for (int i = 0; i < d.nu; ++i) T(d.offU() + i, base + d.nq + d.nv + d.na + i) = w;
  }
  const int offC = nState * L.Nn;
  for (int j = 0; j < L.Nc; ++j)
    for (int i = 0; i < d.nlam; ++i) T(d.offLam() + i, offC + j * d.nlam + i) = Wlam(j);
  if (!L.uStored && torqueChain) {
    torqueJacobian(L, c, s.Ju);
    T.middleRows(d.offU(), d.nu).noalias() = s.Ju * T;
  }
}

// --- evaluate ---------------------------------------------------------------------------

void AlmEvaluator::evaluate(const Eigen::VectorXd& y, const Eigen::VectorXd& muRow, const EvaluationRequest& req,
                            Evaluation& out) {
  const Layout& L = prob_->layout();
  if (y.size() != L.n()) throw std::invalid_argument("AlmEvaluator::evaluate: y has the wrong size");
  const bool jac = req.jacobian;
  out.g.setZero(L.md);
  out.cost = 0.0;
  out.defectCost = 0.0;
  // per-element derived fields
  const int M = static_cast<int>(L.elem.size());
  out.nodeForce.resize(static_cast<size_t>(M));
  out.derivedU.resize(static_cast<size_t>(M));
  for (int e = 0; e < M; ++e) {
    const ElementLayout& el = L.elem[static_cast<size_t>(e)];
    out.nodeForce[static_cast<size_t>(e)] = prob_->nodeForces(y, e);
    if (!el.uStored) out.derivedU[static_cast<size_t>(e)].setZero(el.dims.nu, el.Nn);
  }
  const size_t nNodes = prob_->nodes().size();
  if (jac) {
    out.Jnode.resize(nNodes);
    out.Snode.resize(nNodes);
    out.Hcost.resize(nNodes);
    out.gcost.resize(nNodes);
    out.Hnode.resize(nNodes);
    out.Jpoint.resize(prob_->points().size());
    out.Hpoint.resize(prob_->points().size());
    out.Jdefect.resize(prob_->defects().size());
    out.Jreset.resize(prob_->resets().size());
    if (!opt_.gaussNewton) {
      out.JuNode.resize(nNodes);
      out.guCostNode.resize(nNodes);
      out.JuPoint.resize(prob_->points().size());
    }
  }
  out.rdefect.resize(prob_->defects().size());

  evaluateNodes(y, req, out);
  evaluateKinematics(y, out);
  evaluatePoints(y, req, out);
  evaluateDefects(y, req, out);
  evaluateResets(y, req, out);

  // PHR pass
  const Eigen::VectorXd gamma = y.tail(L.md);
  phrEffectiveResidual(out.g, gamma, muRow, prob_->rowIneq(), opt_.activeMargin, out.phr);
  out.perTermViol.setZero(prob_->terms().size());
  out.maxViol = 0.0;
  const std::vector<int>& rowTerm = prob_->rowTerm();
  const std::vector<char>& ineq = prob_->rowIneq();
  for (Eigen::Index i = 0; i < L.md; ++i) {
    const double vi = (opt_.ineqViolMax0 && ineq[static_cast<size_t>(i)]) ? std::max(0.0, out.g(i)) : std::abs(out.g(i));
    double& tv = out.perTermViol(rowTerm[static_cast<size_t>(i)]);
    tv = std::max(tv, vi);
    out.maxViol = std::max(out.maxViol, std::abs(out.phr.ceff(i)));
  }
  if (req.value) {
    out.penalty = phrPenaltyValue(out.g, gamma, muRow, ineq);
    out.merit = out.cost + out.defectCost + out.penalty;
  }
  if (req.gradient && jac) accumulateRhs(y, req, out);
}

void AlmEvaluator::evaluateNodes(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out) {
  const Layout& L = prob_->layout();
  const std::vector<NodeInfo>& nodes = prob_->nodes();
  const int nNodes = static_cast<int>(nodes.size());
  const std::vector<double>& ncw = prob_->nodeCostWeights();
  const bool jac = req.jacobian;
  nodeCostOf_.assign(static_cast<size_t>(nNodes), 0.0);
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(scratch_.size()))
#endif
  for (int ni = 0; ni < nNodes; ++ni) {
    Scratch& s = *scratch_[static_cast<size_t>(threadId())];
    const NodeInfo& info = nodes[static_cast<size_t>(ni)];
    const ElementLayout& el = L.elem[static_cast<size_t>(info.elem)];
    const Element& spec = prob_->spec().elements[static_cast<size_t>(info.elem)];
    const NodeDims& d = el.dims;
    const ElementView X(y.data(), el);
    const int k = info.node;
    s.q = X.Q.col(k);
    s.v = X.V.col(k);
    s.a = X.A.col(k);
    s.lam = out.nodeForce[static_cast<size_t>(info.elem)].col(k);
    unsigned needs = info.needs;
    if (!jac) needs &= ~(kRneaDerivs | kFootDerivs);
    prob_->model().evalNode(*s.data, s.q, s.v, s.a, s.lam, spec.contacts, needs, s.cache);
    if (el.uStored) {
      s.u = X.U.col(k);
    } else {
      s.u.resize(d.nu);
      const std::vector<int>& act = prob_->model().actuatedIdx();
      for (int j = 0; j < d.nu; ++j) s.u(j) = s.cache.gdyn(act[static_cast<size_t>(j)]);
      out.derivedU[static_cast<size_t>(info.elem)].col(k) = s.u;
    }
    const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
    // constraint values and physical Jacobians
    if (jac) s.Jphys.setZero(info.rows, d.nP());
    for (const ResolvedTerm& t : info.terms) {
      t.con->value(z, d, s.cache, out.g.segment(info.row0 + t.rowOffset, t.rows));
      if (jac) t.con->jacobian(z, d, s.cache, s.Jphys.middleRows(t.rowOffset, t.rows));
    }
    // costs
    const double cw = ncw.empty() ? 1.0 : ncw[static_cast<size_t>(ni)];
    double nodeCost = 0.0;
    if (jac) s.gphys.setZero(d.nP());
    for (const std::shared_ptr<Cost>& c : spec.costs) {
      nodeCost += c->value(z, d, s.cache);
      if (jac) {
        s.gtmp.resize(d.nP());
        c->gradient(z, d, s.cache, s.gtmp);
        s.gphys += s.gtmp;
      }
    }
    nodeCostOf_[static_cast<size_t>(ni)] = cw * nodeCost;
    if (!jac) continue;
    // lift
    buildNodeLift(el, k, s.cache, true, s, out.Snode[static_cast<size_t>(ni)]);
    const Eigen::MatrixXd& S = out.Snode[static_cast<size_t>(ni)];
    out.Jnode[static_cast<size_t>(ni)].noalias() = s.Jphys * S;
    out.gcost[static_cast<size_t>(ni)].noalias() = S.transpose() * (cw * s.gphys);
    if (!opt_.gaussNewton && !el.uStored) {
      out.JuNode[static_cast<size_t>(ni)] = s.Jphys.middleCols(d.offU(), d.nu);
      out.guCostNode[static_cast<size_t>(ni)] = cw * s.gphys.segment(d.offU(), d.nu);
    }
    if (!spec.costs.empty()) {
      s.Hphys.setZero(d.nP(), d.nP());
      for (const std::shared_ptr<Cost>& c : spec.costs) c->addHessian(z, d, s.cache, s.Hphys);
      out.Hcost[static_cast<size_t>(ni)] = cw * s.Hphys;
    } else {
      out.Hcost[static_cast<size_t>(ni)].resize(0, 0);
    }
  }
  // a fixed-order reduction
  for (double c : nodeCostOf_) out.cost += c;
}

void AlmEvaluator::evaluateKinematics(const Eigen::VectorXd& y, Evaluation& out) const {
  const Layout& L = prob_->layout();
  for (size_t e = 0; e < L.elem.size(); ++e) {
    const ElementLayout& el = L.elem[e];
    const ElementView X(y.data(), el);
    const Eigen::MatrixXd& D = prob_->grid(static_cast<int>(e)).D();
    const double a = prob_->timeScale(static_cast<int>(e));
    Eigen::Map<Eigen::MatrixXd> R1(out.g.data() + el.kin1Row0, el.dims.nq, el.Nn);
    Eigen::Map<Eigen::MatrixXd> R2(out.g.data() + el.kin2Row0, el.dims.nq, el.Nn);
    R1 = X.V - a * X.Q * D.transpose();
    R2 = X.A - a * X.V * D.transpose();
  }
}

void AlmEvaluator::evaluatePoints(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out) {
  const Layout& L = prob_->layout();
  const std::vector<PointInfo>& points = prob_->points();
  const int nInfo = static_cast<int>(points.size());
  const bool jac = req.jacobian;
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(scratch_.size()))
#endif
  for (int pi = 0; pi < nInfo; ++pi) {
    Scratch& s = *scratch_[static_cast<size_t>(threadId())];
    const PointInfo& info = points[static_cast<size_t>(pi)];
    const ElementLayout& el = L.elem[static_cast<size_t>(info.elem)];
    const Element& spec = prob_->spec().elements[static_cast<size_t>(info.elem)];
    const NodeDims& d = el.dims;
    const ElementView X(y.data(), el);
    const int rpp = info.rowsPerPoint;
    if (info.kind == PointKind::OffNode) {
      const int nCols = static_cast<int>(info.zIdxAll.size());
      if (jac) out.Jpoint[static_cast<size_t>(pi)].setZero(info.nPoints * rpp, nCols);
      const bool keepJu = jac && !opt_.gaussNewton && !el.uStored;
      if (keepJu) out.JuPoint[static_cast<size_t>(pi)].setZero(info.nPoints * rpp, d.nu);
      for (int p = 0; p < info.nPoints; ++p) {
        const Eigen::VectorXd& W = info.W[static_cast<size_t>(p)];
        const Eigen::VectorXd& Wlam = info.Wlam[static_cast<size_t>(p)];
        s.q.noalias() = X.Q * W;
        s.v.noalias() = X.V * W;
        s.a.noalias() = X.A * W;
        if (el.Nc > 0)
          s.lam.noalias() = X.C * Wlam;
        else
          s.lam.resize(0);
        unsigned needs = info.needs;
        if (!el.uStored) needs |= kRnea | kFootPos | (jac ? kRneaDerivs : 0u);
        if (!jac) needs &= ~(kRneaDerivs | kFootDerivs);
        prob_->model().evalNode(*s.data, s.q, s.v, s.a, s.lam, spec.contacts, needs, s.cache);
        if (el.uStored) {
          s.u.noalias() = X.U * W;
        } else {
          s.u.resize(d.nu);
          const std::vector<int>& act = prob_->model().actuatedIdx();
          for (int j = 0; j < d.nu; ++j) s.u(j) = s.cache.gdyn(act[static_cast<size_t>(j)]);
        }
        const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
        info.con->value(z, d, s.cache, out.g.segment(info.row0 + p * rpp, rpp));
        if (!jac) continue;
        s.Jb.setZero(rpp, d.nP());
        info.con->jacobian(z, d, s.cache, s.Jb);
        if (keepJu) out.JuPoint[static_cast<size_t>(pi)].middleRows(p * rpp, rpp) = s.Jb.middleCols(d.offU(), d.nu);
        buildPointLift(el, W, Wlam, s.cache, true, s, s.T);
        out.Jpoint[static_cast<size_t>(pi)].middleRows(p * rpp, rpp).noalias() = s.Jb * s.T;
      }
    } else {
      // field-only rows: the constraint reads one field at a linear functional
      const int nCols = static_cast<int>(info.fieldZIdx.size());
      if (jac) out.Jpoint[static_cast<size_t>(pi)].setZero(info.nPoints * rpp, nCols);
      s.q.setZero(d.nq);
      s.v.setZero(d.nv);
      s.a.setZero(d.na);
      s.lam.setZero(d.nlam);
      s.u.setZero(d.nu);
      for (int p = 0; p < info.nPoints; ++p) {
        const Eigen::VectorXd& Wf = info.Wf[static_cast<size_t>(p)];
        switch (info.field) {
          case Field::Q: s.q.noalias() = X.Q * Wf; break;
          case Field::V: s.v.noalias() = X.V * Wf; break;
          case Field::A: s.a.noalias() = X.A * Wf; break;
          case Field::Lam: s.lam.noalias() = X.C * Wf; break;
          case Field::U: s.u.noalias() = X.U * Wf; break;
        }
        const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
        info.con->value(z, d, s.cache, out.g.segment(info.row0 + p * rpp, rpp));
        if (!jac) continue;
        s.Jb.setZero(rpp, d.nP());
        info.con->jacobian(z, d, s.cache, s.Jb);
        const int off = fieldOffset(info.field, d);
        Eigen::MatrixXd& Jp = out.Jpoint[static_cast<size_t>(pi)];
        for (size_t t = 0; t < info.fieldColumns.size(); ++t) {
          const double w = Wf(info.fieldColumns[t]);
          if (w == 0.0) continue;
          Jp.block(p * rpp, static_cast<int>(t) * info.fieldDim, rpp, info.fieldDim) = w * s.Jb.middleCols(off, info.fieldDim);
        }
      }
    }
  }
}

void AlmEvaluator::evaluateDefects(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out) {
  const Layout& L = prob_->layout();
  const std::vector<DefectInfo>& defects = prob_->defects();
  const int nInfo = static_cast<int>(defects.size());
  const bool jac = req.jacobian;
  defectCostOf_.assign(static_cast<size_t>(nInfo), 0.0);
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(scratch_.size()))
#endif
  for (int di = 0; di < nInfo; ++di) {
    Scratch& s = *scratch_[static_cast<size_t>(threadId())];
    const DefectInfo& info = defects[static_cast<size_t>(di)];
    const ElementLayout& el = L.elem[static_cast<size_t>(info.elem)];
    const Element& spec = prob_->spec().elements[static_cast<size_t>(info.elem)];
    const NodeDims& d = el.dims;
    const ElementView X(y.data(), el);
    const int rpp = info.rowsPerPoint;
    const int nCols = static_cast<int>(info.zIdxAll.size());
    Eigen::VectorXd& r = out.rdefect[static_cast<size_t>(di)];
    r.setZero(info.nPoints * rpp);
    if (jac) out.Jdefect[static_cast<size_t>(di)].setZero(info.nPoints * rpp, nCols);
    double cost = 0.0;
    for (int p = 0; p < info.nPoints; ++p) {
      const Eigen::VectorXd& W = info.W[static_cast<size_t>(p)];
      const Eigen::VectorXd& Wlam = info.Wlam[static_cast<size_t>(p)];
      s.q.noalias() = X.Q * W;
      s.v.noalias() = X.V * W;
      s.a.noalias() = X.A * W;
      if (el.Nc > 0)
        s.lam.noalias() = X.C * Wlam;
      else
        s.lam.resize(0);
      unsigned needs = info.needs;
      if (!jac) needs &= ~(kRneaDerivs | kFootDerivs);
      prob_->model().evalNode(*s.data, s.q, s.v, s.a, s.lam, spec.contacts, needs, s.cache);
      if (el.uStored) {
        s.u.noalias() = X.U * W;
      } else {
        s.u.setZero(d.nu);  // the floating-base residual does not read u
      }
      const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
      info.con->value(z, d, s.cache, r.segment(p * rpp, rpp));
      cost += 0.5 * info.rho * info.wq[static_cast<size_t>(p)] * r.segment(p * rpp, rpp).squaredNorm();
      if (!jac) continue;
      s.Jb.setZero(rpp, d.nP());
      info.con->jacobian(z, d, s.cache, s.Jb);
      buildPointLift(el, W, Wlam, s.cache, false, s, s.T);
      out.Jdefect[static_cast<size_t>(di)].middleRows(p * rpp, rpp).noalias() = s.Jb * s.T;
    }
    defectCostOf_[static_cast<size_t>(di)] = cost;
  }
  for (double c : defectCostOf_) out.defectCost += c;
}

void AlmEvaluator::evaluateResets(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out) {
  const Layout& L = prob_->layout();
  const bool jac = req.jacobian;
  Scratch& s = *scratch_[0];
  for (const ResetInfo& ri : prob_->resets()) {
    const ElementLayout& Lt = L.elem[static_cast<size_t>(ri.tailElem)];
    const ElementLayout& Lh = L.elem[static_cast<size_t>(ri.headElem)];
    const ElementView Xt(y.data(), Lt), Xh(y.data(), Lh);
    const int kt = Lt.Nn - 1;
    const Eigen::VectorXd qt = Xt.Q.col(kt), vt = Xt.V.col(kt), at = Xt.A.col(kt);
    const Eigen::VectorXd lt = out.nodeForce[static_cast<size_t>(ri.tailElem)].col(kt);
    const Eigen::VectorXd ut = Lt.uStored ? Eigen::VectorXd(Xt.U.col(kt)) : Eigen::VectorXd(out.derivedU[static_cast<size_t>(ri.tailElem)].col(kt));
    const Eigen::VectorXd qh = Xh.Q.col(0), vh = Xh.V.col(0), ah = Xh.A.col(0);
    const Eigen::VectorXd lh = out.nodeForce[static_cast<size_t>(ri.headElem)].col(0);
    const Eigen::VectorXd uh = Lh.uStored ? Eigen::VectorXd(Xh.U.col(0)) : Eigen::VectorXd(out.derivedU[static_cast<size_t>(ri.headElem)].col(0));
    const NodeQuantities tail{qt, vt, at, lt, ut};
    const NodeQuantities head{qh, vh, ah, lh, uh};
    ri.reset->value(tail, head, prob_->model(), *s.data, out.g.segment(ri.row0, ri.rows));
    if (quotient_->enabled())
      for (int r = 0; r < ri.rows; ++r)
        if (quotient_->isIdentityRow(ri.row0 + r)) out.g(ri.row0 + r) = 0.0;
    if (!jac) continue;
    const int nPt = Lt.dims.nP(), nPh = Lh.dims.nP();
    s.Jphys.setZero(ri.rows, nPt + nPh);
    ri.reset->jacobian(tail, head, Lt.dims, Lh.dims, prob_->model(), *s.data, s.Jphys);
    const Eigen::MatrixXd& St = out.Snode[static_cast<size_t>(nodeIndex(ri.tailElem, kt))];
    const Eigen::MatrixXd& Sh = out.Snode[static_cast<size_t>(nodeIndex(ri.headElem, 0))];
    Eigen::MatrixXd& Jd = out.Jreset[static_cast<size_t>(ri.index)];
    Jd.resize(ri.rows, St.cols() + Sh.cols());
    Jd.leftCols(St.cols()).noalias() = s.Jphys.leftCols(nPt) * St;
    Jd.rightCols(Sh.cols()).noalias() = s.Jphys.rightCols(nPh) * Sh;
    if (quotient_->enabled())
      for (int r = 0; r < ri.rows; ++r)
        if (quotient_->isIdentityRow(ri.row0 + r)) Jd.row(r).setZero();
  }
}

void AlmEvaluator::accumulateRhs(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out) const {
  (void)y;
  const Layout& L = prob_->layout();
  out.F.setZero(L.n());
  Eigen::VectorXd& F = out.F;
  const Eigen::VectorXd& w = out.phr.w;
  // nodes: F[zIdx] -= gcost + J' w
  const std::vector<NodeInfo>& nodes = prob_->nodes();
  Eigen::VectorXd tmp;
  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    const NodeInfo& info = nodes[ni];
    tmp = out.gcost[ni];
    if (info.rows > 0) tmp.noalias() += out.Jnode[ni].transpose() * w.segment(info.row0, info.rows);
    for (size_t c = 0; c < info.zIdx.size(); ++c) F(info.zIdx[c]) -= tmp(static_cast<Eigen::Index>(c));
  }
  // kinematic rows: R1 = V - a Q D', R2 = A - a V D'
  for (size_t e = 0; e < L.elem.size(); ++e) {
    const ElementLayout& el = L.elem[e];
    const Eigen::MatrixXd& D = prob_->grid(static_cast<int>(e)).D();
    const double a = prob_->timeScale(static_cast<int>(e));
    const Eigen::Map<const Eigen::MatrixXd> S1(w.data() + el.kin1Row0, el.dims.nq, el.Nn);
    const Eigen::Map<const Eigen::MatrixXd> S2(w.data() + el.kin2Row0, el.dims.nq, el.Nn);
    Eigen::Map<Eigen::MatrixXd> FQ(F.data() + el.x0 + el.offQ, el.dims.nq, el.Nn);
    Eigen::Map<Eigen::MatrixXd> FV(F.data() + el.x0 + el.offV, el.dims.nv, el.Nn);
    Eigen::Map<Eigen::MatrixXd> FA(F.data() + el.x0 + el.offA, el.dims.na, el.Nn);
    FQ.noalias() += a * S1 * D;
    FV -= S1;
    FV.noalias() += a * S2 * D;
    FA -= S2;
  }
  // point rows
  const std::vector<PointInfo>& points = prob_->points();
  for (size_t pi = 0; pi < points.size(); ++pi) {
    const PointInfo& info = points[pi];
    const int rows = info.nPoints * info.rowsPerPoint;
    tmp.noalias() = out.Jpoint[pi].transpose() * w.segment(info.row0, rows);
    const std::vector<int>& cols = info.kind == PointKind::OffNode ? info.zIdxAll : info.fieldZIdx;
    for (size_t c = 0; c < cols.size(); ++c) F(cols[c]) -= tmp(static_cast<Eigen::Index>(c));
  }
  // defects: F -= Jf' (rho wq r)
  const std::vector<DefectInfo>& defects = prob_->defects();
  Eigen::VectorXd rw;
  for (size_t di = 0; di < defects.size(); ++di) {
    const DefectInfo& info = defects[di];
    rw = out.rdefect[di];
    for (int p = 0; p < info.nPoints; ++p)
      rw.segment(p * info.rowsPerPoint, info.rowsPerPoint) *= info.rho * info.wq[static_cast<size_t>(p)];
    tmp.noalias() = out.Jdefect[di].transpose() * rw;
    for (size_t c = 0; c < info.zIdxAll.size(); ++c) F(info.zIdxAll[c]) -= tmp(static_cast<Eigen::Index>(c));
  }
  // resets
  for (const ResetInfo& ri : prob_->resets()) {
    tmp.noalias() = out.Jreset[static_cast<size_t>(ri.index)].transpose() * w.segment(ri.row0, ri.rows);
    for (size_t c = 0; c < ri.zIdx.size(); ++c) F(ri.zIdx[c]) -= tmp(static_cast<Eigen::Index>(c));
  }
  if (req.freezeDuals)
    F.tail(L.md).setZero();
  else
    F.tail(L.md) = out.phr.ceff;
}

// --- metric -----------------------------------------------------------------------------

void AlmEvaluator::assembleMetric(const Eigen::VectorXd& y, const Evaluation& ev, const Eigen::VectorXd& muRow, MetricSink& sink) const {
  if (!opt_.gaussNewton) assembleExactHessians(y, ev, sink);
  const Layout& L = prob_->layout();
  const std::vector<char>& act = ev.phr.activeMetric;
  const std::vector<NodeInfo>& nodes = prob_->nodes();
  const std::vector<PointInfo>& points = prob_->points();
  const std::vector<DefectInfo>& defects = prob_->defects();
  const std::vector<ResetInfo>& resets = prob_->resets();
  const int M = static_cast<int>(L.elem.size());
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int e = 0; e < M; ++e) {
    Eigen::MatrixXd H, Ja;
    Eigen::VectorXd w;
    std::vector<int> idx;
    const ElementLayout& el = L.elem[static_cast<size_t>(e)];
    // nodes: S' Hcost S + Jact' diag(mu) Jact as one dense block over the node support
    for (int k = 0; k < el.Nn; ++k) {
      const size_t ni = static_cast<size_t>(nodeIndex(e, k));
      const NodeInfo& info = nodes[ni];
      const int ns = static_cast<int>(info.zIdx.size());
      H.setZero(ns, ns);
      bool any = false;
      if (ev.Hcost[ni].size() > 0) {
        const Eigen::MatrixXd& S = ev.Snode[ni];
        H.noalias() += S.transpose() * ev.Hcost[ni] * S;
        any = true;
      }
      if (info.rows > 0) {
        const Eigen::MatrixXd& J = ev.Jnode[ni];
        int na = 0;
        for (int r = 0; r < info.rows; ++r)
          if (act[static_cast<size_t>(info.row0 + r)]) ++na;
        if (na > 0) {
          Ja.resize(na, ns);
          w.resize(na);
          int i = 0;
          for (int r = 0; r < info.rows; ++r) {
            if (!act[static_cast<size_t>(info.row0 + r)]) continue;
            Ja.row(i) = J.row(r);
            w(i) = muRow(info.row0 + r);
            ++i;
          }
          H.noalias() += Ja.transpose() * (w.asDiagonal() * Ja);
          any = true;
        }
      }
      if (any) sink.addBlock(e, info.zIdx, H);
    }
    // kinematic rows, per joint j: block over [Q(j,:), V(j,:), A(j,:)]
    {
      const Eigen::MatrixXd& D = prob_->grid(e).D();
      const double a = prob_->timeScale(e);
      const double mu1 = muRow(el.kin1Row0), mu2 = muRow(el.kin2Row0);
      const Eigen::MatrixXd DtD = D.transpose() * D;
      const int n = el.Nn;
      H.setZero(3 * n, 3 * n);
      for (int l1 = 0; l1 < n; ++l1) {
        for (int l2 = 0; l2 < n; ++l2) {
          H(l1, l2) = a * a * mu1 * DtD(l1, l2);                                     // QQ
          H(l1, n + l2) = -a * mu1 * D(l2, l1);                                      // QV
          H(n + l1, l2) = -a * mu1 * D(l1, l2);                                      // VQ
          H(n + l1, n + l2) = a * a * mu2 * DtD(l1, l2) + (l1 == l2 ? mu1 : 0.0);   // VV
          H(n + l1, 2 * n + l2) = -a * mu2 * D(l2, l1);                              // VA
          H(2 * n + l1, n + l2) = -a * mu2 * D(l1, l2);                              // AV
          H(2 * n + l1, 2 * n + l2) = (l1 == l2 ? mu2 : 0.0);                        // AA
        }
      }
      idx.resize(static_cast<size_t>(3 * n));
      for (int j = 0; j < el.dims.nq; ++j) {
        for (int k = 0; k < n; ++k) {
          idx[static_cast<size_t>(k)] = el.xQ(j, k);
          idx[static_cast<size_t>(n + k)] = el.xV(j, k);
          idx[static_cast<size_t>(2 * n + k)] = el.xA(j, k);
        }
        sink.addBlock(e, idx, H);
      }
    }
    // point rows: the active rows with their penalties
    for (int pi : pointsOfElem_[static_cast<size_t>(e)]) {
      const PointInfo& info = points[static_cast<size_t>(pi)];
      const Eigen::MatrixXd& J = ev.Jpoint[static_cast<size_t>(pi)];
      const int rows = info.nPoints * info.rowsPerPoint;
      const int nc = static_cast<int>(J.cols());
      int na = 0;
      for (int r = 0; r < rows; ++r)
        if (act[static_cast<size_t>(info.row0 + r)]) ++na;
      if (na == 0) continue;
      Ja.resize(na, nc);
      w.resize(na);
      int i = 0;
      for (int r = 0; r < rows; ++r) {
        if (!act[static_cast<size_t>(info.row0 + r)]) continue;
        Ja.row(i) = J.row(r);
        w(i) = muRow(info.row0 + r);
        ++i;
      }
      sink.addWeightedRows(e, info.kind == PointKind::OffNode ? info.zIdxAll : info.fieldZIdx, Ja, w);
    }
    // defects: rho sum_m wq_m Jf_m' Jf_m
    for (int di : defectsOfElem_[static_cast<size_t>(e)]) {
      const DefectInfo& info = defects[static_cast<size_t>(di)];
      const Eigen::MatrixXd& J = ev.Jdefect[static_cast<size_t>(di)];
      w.resize(J.rows());
      for (int p = 0; p < info.nPoints; ++p)
        w.segment(p * info.rowsPerPoint, info.rowsPerPoint).setConstant(info.rho * info.wq[static_cast<size_t>(p)]);
      sink.addWeightedRows(e, info.zIdxAll, J, w);
    }
    // resets whose head node is in this element (they reach into the junction separator
    // and this element's interior only)
    for (int r : resetsOfHead_[static_cast<size_t>(e)]) {
      const ResetInfo& ri = resets[static_cast<size_t>(r)];
      if (quotient_->enabled() && quotient_->resetIsIdentity(ri.index)) continue;
      const Eigen::MatrixXd& J = ev.Jreset[static_cast<size_t>(ri.index)];
      w.resize(ri.rows);
      for (int rr = 0; rr < ri.rows; ++rr) w(rr) = muRow(ri.row0 + rr);
      sink.addWeightedRows(e, ri.zIdx, J, w);
    }
  }
}


// --- the exact-Newton additions: sum_i w_i d2 g_i / dz2 of every term, lifted ------------------
//
// Per node: the contracted Hessians of the node's terms with the PHR weights w of the
// evaluation, lifted through the node lift S (S' H S). Under torque elimination the lift
// is iterate dependent and its own curvature is added: with gu = (dg/du)' w + dl/du the
// weight of every eliminated torque, sum_j gu_j d2 u_ID,j / dz2 is the dynamics
// residual's contracted Hessian on the actuated rows. Per off-node point: the same
// through the point lift T. Per defect point: the dynamics residual's contracted
// Hessian with w = rho wq r. Per reset: the reset map's contracted Hessian over
// [z_tail; z_head], lifted through the two node lifts.
void AlmEvaluator::addTorqueLiftHessian(const ElementLayout& L, const NodeQuantities& z, const Eigen::VectorXd& gu, Scratch& s,
                                        Eigen::MatrixXd& H) const {
  if (gu.cwiseAbs().maxCoeff() == 0.0) return;
  const Element& el = prob_->spec().elements[static_cast<size_t>(L.index)];
  const std::vector<int>& act = prob_->model().actuatedIdx();
  Eigen::VectorXd wFull = Eigen::VectorXd::Zero(L.dims.nv);
  for (int j = 0; j < L.dims.nu; ++j) wFull(act[static_cast<size_t>(j)]) = gu(j);
  s.hs.z.resize(L.dims.nP());
  z.toVector(L.dims, s.hs.z);
  s.hs.hessians->addDynamicsHessian(L.dims, el.contacts, s.hs.z, wFull, H);
}

void AlmEvaluator::assembleExactHessians(const Eigen::VectorXd& y, const Evaluation& ev, MetricSink& sink) const {
  const Layout& L = prob_->layout();
  const std::vector<NodeInfo>& nodes = prob_->nodes();
  const std::vector<PointInfo>& points = prob_->points();
  const std::vector<DefectInfo>& defects = prob_->defects();
  const std::vector<ResetInfo>& resets = prob_->resets();
  const Eigen::VectorXd& w = ev.phr.w;
  const int M = static_cast<int>(L.elem.size());
#ifdef LEAP_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(scratch_.size()))
#endif
  for (int e = 0; e < M; ++e) {
    Scratch& s = *scratch_[static_cast<size_t>(threadId())];
    const ElementLayout& el = L.elem[static_cast<size_t>(e)];
    const Element& spec = prob_->spec().elements[static_cast<size_t>(e)];
    const NodeDims& d = el.dims;
    const ElementView X(y.data(), el);
    // nodes
    for (int k = 0; k < el.Nn; ++k) {
      const size_t ni = static_cast<size_t>(nodeIndex(e, k));
      const NodeInfo& info = nodes[ni];
      const bool liftCurvature = !el.uStored && (info.rows > 0 || !spec.costs.empty());
      if (info.rows == 0 && !liftCurvature) continue;
      s.q = X.Q.col(k);
      s.v = X.V.col(k);
      s.a = X.A.col(k);
      s.lam = ev.nodeForce[static_cast<size_t>(e)].col(k);
      s.u = el.uStored ? Eigen::VectorXd(X.U.col(k)) : Eigen::VectorXd(ev.derivedU[static_cast<size_t>(e)].col(k));
      const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
      s.Hphys.setZero(d.nP(), d.nP());
      for (const ResolvedTerm& t : info.terms) {
        s.wseg = w.segment(info.row0 + t.rowOffset, t.rows);
        if (s.wseg.cwiseAbs().maxCoeff() == 0.0) continue;
        t.con->addContractedHessian(z, d, s.hs, s.wseg, s.Hphys);
      }
      if (liftCurvature) {
        Eigen::VectorXd gu = ev.guCostNode[ni];
        if (info.rows > 0) gu.noalias() += ev.JuNode[ni].transpose() * w.segment(info.row0, info.rows);
        addTorqueLiftHessian(el, z, gu, s, s.Hphys);
      }
      if (s.Hphys.cwiseAbs().maxCoeff() == 0.0) continue;
      const Eigen::MatrixXd& S = ev.Snode[ni];
      s.Hl.noalias() = S.transpose() * s.Hphys * S;
      sink.addBlock(e, info.zIdx, s.Hl);
    }
    // off-node and field-functional points
    for (int pi : pointsOfElem_[static_cast<size_t>(e)]) {
      const PointInfo& info = points[static_cast<size_t>(pi)];
      const int rpp = info.rowsPerPoint;
      if (info.kind == PointKind::OffNode) {
        const int nCols = static_cast<int>(info.zIdxAll.size());
        s.Hl.setZero(nCols, nCols);
        bool any = false;
        for (int p = 0; p < info.nPoints; ++p) {
          s.wseg = w.segment(info.row0 + p * rpp, rpp);
          if (s.wseg.cwiseAbs().maxCoeff() == 0.0) continue;
          const Eigen::VectorXd& W = info.W[static_cast<size_t>(p)];
          const Eigen::VectorXd& Wlam = info.Wlam[static_cast<size_t>(p)];
          s.q.noalias() = X.Q * W;
          s.v.noalias() = X.V * W;
          s.a.noalias() = X.A * W;
          if (el.Nc > 0) s.lam.noalias() = X.C * Wlam;
          else s.lam.resize(0);
          unsigned needs = info.needs & ~(kRneaDerivs | kFootDerivs);
          if (!el.uStored) needs |= kRnea | kFootPos | kRneaDerivs;
          prob_->model().evalNode(*s.data, s.q, s.v, s.a, s.lam, spec.contacts, needs, s.cache);
          if (el.uStored) {
            s.u.noalias() = X.U * W;
          } else {
            s.u.resize(d.nu);
            const std::vector<int>& act = prob_->model().actuatedIdx();
            for (int j = 0; j < d.nu; ++j) s.u(j) = s.cache.gdyn(act[static_cast<size_t>(j)]);
          }
          const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
          s.Hphys.setZero(d.nP(), d.nP());
          info.con->addContractedHessian(z, d, s.hs, s.wseg, s.Hphys);
          if (!el.uStored) {
            const Eigen::VectorXd gu = ev.JuPoint[static_cast<size_t>(pi)].middleRows(p * rpp, rpp).transpose() * s.wseg;
            addTorqueLiftHessian(el, z, gu, s, s.Hphys);
          }
          if (s.Hphys.cwiseAbs().maxCoeff() == 0.0) continue;
          buildPointLift(el, W, Wlam, s.cache, true, s, s.T);
          s.Hl.noalias() += s.T.transpose() * s.Hphys * s.T;
          any = true;
        }
        if (any) sink.addBlock(e, info.zIdxAll, s.Hl);
      } else {
        // one field at a linear functional: H_lift(a, b) = Wf_a Wf_b H_field
        const int nCols = static_cast<int>(info.fieldZIdx.size());
        s.Hl.setZero(nCols, nCols);
        bool any = false;
        s.q.setZero(d.nq);
        s.v.setZero(d.nv);
        s.a.setZero(d.na);
        s.lam.setZero(d.nlam);
        s.u.setZero(d.nu);
        const int off = fieldOffset(info.field, d);
        for (int p = 0; p < info.nPoints; ++p) {
          s.wseg = w.segment(info.row0 + p * rpp, rpp);
          if (s.wseg.cwiseAbs().maxCoeff() == 0.0) continue;
          const Eigen::VectorXd& Wf = info.Wf[static_cast<size_t>(p)];
          switch (info.field) {
            case Field::Q: s.q.noalias() = X.Q * Wf; break;
            case Field::V: s.v.noalias() = X.V * Wf; break;
            case Field::A: s.a.noalias() = X.A * Wf; break;
            case Field::Lam: s.lam.noalias() = X.C * Wf; break;
            case Field::U: s.u.noalias() = X.U * Wf; break;
          }
          const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
          s.Hphys.setZero(d.nP(), d.nP());
          info.con->addContractedHessian(z, d, s.hs, s.wseg, s.Hphys);
          const Eigen::MatrixXd Hf = s.Hphys.block(off, off, info.fieldDim, info.fieldDim);
          if (Hf.cwiseAbs().maxCoeff() == 0.0) continue;
          for (size_t ta = 0; ta < info.fieldColumns.size(); ++ta) {
            const double wa = Wf(info.fieldColumns[ta]);
            if (wa == 0.0) continue;
            for (size_t tb = 0; tb < info.fieldColumns.size(); ++tb) {
              const double wb = Wf(info.fieldColumns[tb]);
              if (wb == 0.0) continue;
              s.Hl.block(static_cast<int>(ta) * info.fieldDim, static_cast<int>(tb) * info.fieldDim, info.fieldDim, info.fieldDim) += wa * wb * Hf;
            }
          }
          any = true;
        }
        if (any) sink.addBlock(e, info.fieldZIdx, s.Hl);
      }
    }
    // defect points: w = rho wq r
    for (int di : defectsOfElem_[static_cast<size_t>(e)]) {
      const DefectInfo& info = defects[static_cast<size_t>(di)];
      const int rpp = info.rowsPerPoint;
      const int nCols = static_cast<int>(info.zIdxAll.size());
      s.Hl.setZero(nCols, nCols);
      bool any = false;
      for (int p = 0; p < info.nPoints; ++p) {
        s.wseg = info.rho * info.wq[static_cast<size_t>(p)] * ev.rdefect[static_cast<size_t>(di)].segment(p * rpp, rpp);
        if (s.wseg.cwiseAbs().maxCoeff() == 0.0) continue;
        const Eigen::VectorXd& W = info.W[static_cast<size_t>(p)];
        const Eigen::VectorXd& Wlam = info.Wlam[static_cast<size_t>(p)];
        s.q.noalias() = X.Q * W;
        s.v.noalias() = X.V * W;
        s.a.noalias() = X.A * W;
        if (el.Nc > 0) s.lam.noalias() = X.C * Wlam;
        else s.lam.resize(0);
        prob_->model().evalNode(*s.data, s.q, s.v, s.a, s.lam, spec.contacts, info.needs & ~(kRneaDerivs | kFootDerivs), s.cache);
        if (el.uStored) s.u.noalias() = X.U * W;
        else s.u.setZero(d.nu);
        const NodeQuantities z{s.q, s.v, s.a, s.lam, s.u};
        s.Hphys.setZero(d.nP(), d.nP());
        info.con->addContractedHessian(z, d, s.hs, s.wseg, s.Hphys);
        if (s.Hphys.cwiseAbs().maxCoeff() == 0.0) continue;
        buildPointLift(el, W, Wlam, s.cache, false, s, s.T);
        s.Hl.noalias() += s.T.transpose() * s.Hphys * s.T;
        any = true;
      }
      if (any) sink.addBlock(e, info.zIdxAll, s.Hl);
    }
    // resets whose head node is in this element: the map's Hessian over [z_tail; z_head]
    for (int r : resetsOfHead_[static_cast<size_t>(e)]) {
      const ResetInfo& ri = resets[static_cast<size_t>(r)];
      if (quotient_->enabled() && quotient_->resetIsIdentity(ri.index)) continue;
      s.wseg = w.segment(ri.row0, ri.rows);
      if (s.wseg.cwiseAbs().maxCoeff() == 0.0) continue;
      const ElementLayout& Lt = L.elem[static_cast<size_t>(ri.tailElem)];
      const ElementLayout& Lh = L.elem[static_cast<size_t>(ri.headElem)];
      const ElementView Xt(y.data(), Lt), Xh(y.data(), Lh);
      const int kt = Lt.Nn - 1;
      const Eigen::VectorXd qt = Xt.Q.col(kt), vt = Xt.V.col(kt), at = Xt.A.col(kt);
      const Eigen::VectorXd lt = ev.nodeForce[static_cast<size_t>(ri.tailElem)].col(kt);
      const Eigen::VectorXd ut = Lt.uStored ? Eigen::VectorXd(Xt.U.col(kt)) : Eigen::VectorXd(ev.derivedU[static_cast<size_t>(ri.tailElem)].col(kt));
      const Eigen::VectorXd qh = Xh.Q.col(0), vh = Xh.V.col(0), ah = Xh.A.col(0);
      const Eigen::VectorXd lh = ev.nodeForce[static_cast<size_t>(ri.headElem)].col(0);
      const Eigen::VectorXd uh = Lh.uStored ? Eigen::VectorXd(Xh.U.col(0)) : Eigen::VectorXd(ev.derivedU[static_cast<size_t>(ri.headElem)].col(0));
      const NodeQuantities tail{qt, vt, at, lt, ut};
      const NodeQuantities head{qh, vh, ah, lh, uh};
      const int nPt = Lt.dims.nP(), nPh = Lh.dims.nP();
      s.Hphys.setZero(nPt + nPh, nPt + nPh);
      ri.reset->addContractedHessian(tail, head, Lt.dims, Lh.dims, s.hs, s.wseg, s.Hphys);
      if (s.Hphys.cwiseAbs().maxCoeff() == 0.0) continue;
      const Eigen::MatrixXd& St = ev.Snode[static_cast<size_t>(nodeIndex(ri.tailElem, kt))];
      const Eigen::MatrixXd& Sh = ev.Snode[static_cast<size_t>(nodeIndex(ri.headElem, 0))];
      const int nt = static_cast<int>(St.cols()), nh = static_cast<int>(Sh.cols());
      s.Hl.setZero(nt + nh, nt + nh);
      s.Hl.topLeftCorner(nt, nt).noalias() = St.transpose() * s.Hphys.topLeftCorner(nPt, nPt) * St;
      s.Hl.topRightCorner(nt, nh).noalias() = St.transpose() * s.Hphys.topRightCorner(nPt, nPh) * Sh;
      s.Hl.bottomLeftCorner(nh, nt) = s.Hl.topRightCorner(nt, nh).transpose();
      s.Hl.bottomRightCorner(nh, nh).noalias() = Sh.transpose() * s.Hphys.bottomRightCorner(nPh, nPh) * Sh;
      sink.addBlock(e, ri.zIdx, s.Hl);
    }
  }
}

double AlmEvaluator::zeroDualViolation(const Eigen::VectorXd& y, const Eigen::VectorXd& muRow, Evaluation& out) {
  Eigen::VectorXd y0 = y;
  y0.tail(prob_->layout().md).setZero();
  EvaluationRequest req;
  req.value = true;
  req.gradient = false;
  req.jacobian = false;
  evaluate(y0, muRow, req, out);
  return out.maxViol;
}

}  // namespace leap
