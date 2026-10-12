#include "leap/problem/mesh_refinement.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "leap/problem/reset_map.hpp"
#include "leap/spectral/bernstein.hpp"

namespace leap {

std::vector<Where> mapWhereUniqueOwner(Where where, int i, int k, bool qvOnly) {
  if (k <= 1) return {where};
  switch (where) {
    case Where::FirstNode: return i == 0 ? std::vector<Where>{Where::FirstNode} : std::vector<Where>{};
    case Where::LastNode: return i == k - 1 ? std::vector<Where>{Where::LastNode} : std::vector<Where>{};
    case Where::AllNodes:
      if (!qvOnly || i == 0) return {Where::AllNodes};
      return {Where::Interior, Where::LastNode};
    case Where::Interior:
      if (i == 0) return {Where::Interior, Where::LastNode};
      if (i == k - 1) return qvOnly ? std::vector<Where>{Where::Interior} : std::vector<Where>{Where::FirstNode, Where::Interior};
      return qvOnly ? std::vector<Where>{Where::Interior, Where::LastNode} : std::vector<Where>{Where::AllNodes};
    case Where::OneNode: return {};  // single-node rows are never split
  }
  return {where};
}

TrajectoryProblem splitElement(const TrajectoryProblem& parent, int element, double fraction) {
  if (element < 0 || element >= static_cast<int>(parent.elements.size())) throw std::out_of_range("splitElement: element out of range");
  if (!(fraction > 0.0 && fraction < 1.0)) throw std::invalid_argument("splitElement: fraction must lie in (0, 1)");
  const Element& src = parent.elements[static_cast<size_t>(element)];
  Element left, right;
  for (Element* child : {&left, &right}) {
    child->label = src.label;
    child->phase = src.phase;
    child->degree = src.degree;
    child->contacts = src.contacts;
    child->forceControlPoints = src.forceControlPoints;
    child->deriveTorque = src.deriveTorque;
    child->costs = src.costs;
    child->defect = src.defect;
  }
  left.label += "_l";
  right.label += "_r";
  left.T = fraction * src.T;
  right.T = (1.0 - fraction) * src.T;
  for (const NodeAttachment& a : src.nodeTerms) {
    const bool qvOnly = !a.con->readsOneSidedFields();
    for (Where w : mapWhereUniqueOwner(a.where, 0, 2, qvOnly)) left.addNode(a.con, w);
    for (Where w : mapWhereUniqueOwner(a.where, 1, 2, qvOnly)) right.addNode(a.con, w);
  }
  for (const PointAttachment& p : src.pointTerms) {
    if (p.kind == PointKind::ForceControlPoints && p.con->name() == "pin_lam_liftoff") {
      right.pointTerms.push_back(p);
    } else if (p.kind == PointKind::OffNode && p.con->name().rfind("dyn", 0) == 0) {
      PointAttachment l = p, r = p;
      l.taus.clear();
      r.taus.clear();
      for (double tau : p.taus) {
        const double s = std::clamp(0.5 * (tau + 1.0), 0.0, 1.0);
        if (s <= fraction) l.taus.push_back(2.0 * std::clamp(s / fraction, 0.0, 1.0) - 1.0);
        else r.taus.push_back(2.0 * std::clamp((s - fraction) / (1.0 - fraction), 0.0, 1.0) - 1.0);
      }
      if (!l.taus.empty()) left.pointTerms.push_back(l);
      if (!r.taus.empty()) right.pointTerms.push_back(r);
    } else {
      left.pointTerms.push_back(p);
      right.pointTerms.push_back(p);
    }
  }
  TrajectoryProblem refined;
  refined.model = parent.model;
  refined.preconditionForce = parent.preconditionForce;
  const int nq = parent.model->nq(), nv = parent.model->nv();
  for (int e = 0; e < static_cast<int>(parent.elements.size()); ++e) {
    if (e == element) {
      refined.elements.push_back(left);
      refined.elements.push_back(right);
      refined.addReset(std::make_shared<ContinuityReset>(nq, nv));
    } else {
      refined.elements.push_back(parent.elements[static_cast<size_t>(e)]);
    }
    if (e + 1 < static_cast<int>(parent.elements.size())) refined.addReset(parent.resets[static_cast<size_t>(e)]);
  }
  return refined;
}

double prolongElementSplit(const CompiledProblem& parent, const Eigen::VectorXd& yParent, const CompiledProblem& refined,
                           int element, double fraction, Eigen::VectorXd& yRefined) {
  const Layout& Lp = parent.layout();
  const Layout& Lr = refined.layout();
  if (Lr.elem.size() != Lp.elem.size() + 1) throw std::invalid_argument("prolongElementSplit: not a one-element split");
  yRefined.setZero(Lr.n());
  for (int oe = 0; oe < static_cast<int>(Lp.elem.size()); ++oe) {
    if (oe == element) continue;
    const int ne = oe < element ? oe : oe + 1;
    const ElementLayout& ol = Lp.elem[static_cast<size_t>(oe)];
    const ElementLayout& nl = Lr.elem[static_cast<size_t>(ne)];
    if (ol.nx != nl.nx) throw std::invalid_argument("prolongElementSplit: an unchanged element changed layout");
    yRefined.segment(nl.x0, nl.nx) = yParent.segment(ol.x0, ol.nx);
  }
  const ElementLayout& ol = Lp.elem[static_cast<size_t>(element)];
  Eigen::VectorXd q, v, a, lam;
  for (int side = 0; side < 2; ++side) {
    const int ce = element + side;
    const ElementLayout& cl = Lr.elem[static_cast<size_t>(ce)];
    ElementViewMutable X(yRefined.data(), cl);
    for (int node = 0; node < cl.Nn; ++node) {
      const double childTau = refined.grid(ce).nodes()(node);
      const double parentTau = side == 0 ? fraction * (childTau + 1.0) - 1.0 : fraction + (1.0 - fraction) * childTau;
      parent.stateAt(yParent, element, parentTau, q, v, a, lam);
      X.Q.col(node) = q;
      X.V.col(node) = v;
      X.A.col(node) = a;
      if (cl.uStored) X.U.col(node) = parent.storedU(yParent, element, parentTau);
    }
  }
  if (ol.Nc > 0) {
    const ElementView Xp(yParent.data(), ol);
    const Eigen::MatrixXd physical = Xp.C * ol.sigma.asDiagonal();
    Eigen::MatrixXd leftC, rightC;
    deCasteljauSplit(physical, fraction, leftC, rightC);
    ElementViewMutable XL(yRefined.data(), Lr.elem[static_cast<size_t>(element)]);
    ElementViewMutable XR(yRefined.data(), Lr.elem[static_cast<size_t>(element + 1)]);
    XL.C = leftC * Lr.elem[static_cast<size_t>(element)].sigma.cwiseInverse().asDiagonal();
    XR.C = rightC * Lr.elem[static_cast<size_t>(element + 1)].sigma.cwiseInverse().asDiagonal();
  }
  // verification: the children reproduce the parent polynomials
  double err = 0.0;
  Eigen::VectorXd q2, v2, a2, lam2;
  for (int side = 0; side < 2; ++side) {
    const int ce = element + side;
    for (int s = 0; s < 65; ++s) {
      const double childTau = -1.0 + 2.0 * s / 64.0;
      const double parentTau = side == 0 ? fraction * (childTau + 1.0) - 1.0 : fraction + (1.0 - fraction) * childTau;
      parent.stateAt(yParent, element, parentTau, q, v, a, lam);
      refined.stateAt(yRefined, ce, childTau, q2, v2, a2, lam2);
      err = std::max(err, (q - q2).cwiseAbs().maxCoeff());
      err = std::max(err, (v - v2).cwiseAbs().maxCoeff());
      err = std::max(err, (a - a2).cwiseAbs().maxCoeff());
      if (lam.size() > 0) err = std::max(err, (lam - lam2).cwiseAbs().maxCoeff());
      if (ol.uStored) err = std::max(err, (parent.storedU(yParent, element, parentTau) - refined.storedU(yRefined, ce, childTau)).cwiseAbs().maxCoeff());
    }
  }
  return err;
}

double prolongSplits(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const std::vector<int>& elements,
                     const std::vector<double>& fractions, const CompiledProblem& to, Eigen::VectorXd& yTo) {
  if (elements.size() != fractions.size()) throw std::invalid_argument("prolongSplits: elements / fractions size mismatch");
  std::unique_ptr<CompiledProblem> parent;
  Eigen::VectorXd y = yFrom;
  double err = 0.0;
  for (size_t i = 0; i < elements.size(); ++i) {
    const CompiledProblem& par = parent ? *parent : from;
    TrajectoryProblem childSpec = splitElement(par.spec(), elements[i], fractions[i]);
    std::unique_ptr<CompiledProblem> child = std::make_unique<CompiledProblem>(CompiledProblem::compile(childSpec));
    Eigen::VectorXd yChild;
    err = std::max(err, prolongElementSplit(par, y, *child, elements[i], fractions[i], yChild));
    y = yChild;
    parent = std::move(child);
  }
  if (y.size() != to.layout().n()) throw std::runtime_error("prolongSplits: the replayed splits do not match the target mesh");
  yTo = y;
  return err;
}

}  // namespace leap
