#include "leap/problem/trajectory_problem.hpp"

#include <stdexcept>

#include "leap/spectral/cgl_grid.hpp"

namespace leap {

bool NodeAttachment::appliesTo(int nodeIdx, int numNodes) const {
  switch (where) {
    case Where::FirstNode: return nodeIdx == 0;
    case Where::LastNode: return nodeIdx == numNodes - 1;
    case Where::Interior: return nodeIdx > 0 && nodeIdx < numNodes - 1;
    case Where::AllNodes: return true;
    case Where::OneNode: return nodeIdx == node;
  }
  return false;
}

Element& Element::addNode(std::shared_ptr<Constraint> con, Where where) {
  if (!con) throw std::invalid_argument("Element::addNode: null constraint");
  if (where == Where::OneNode) throw std::invalid_argument("Element::addNode: use addNodeAt for a single node");
  NodeAttachment a;
  a.con = std::move(con);
  a.where = where;
  nodeTerms.push_back(std::move(a));
  return *this;
}

Element& Element::addNodeAt(std::shared_ptr<Constraint> con, int node) {
  if (!con) throw std::invalid_argument("Element::addNodeAt: null constraint");
  NodeAttachment a;
  a.con = std::move(con);
  a.where = Where::OneNode;
  a.node = node;
  nodeTerms.push_back(std::move(a));
  return *this;
}

Element& Element::addOffNode(std::shared_ptr<Constraint> con, std::vector<double> taus) {
  if (!con) throw std::invalid_argument("Element::addOffNode: null constraint");
  if (taus.empty()) return *this;
  for (double t : taus)
    if (!(t > -1.0 && t < 1.0)) throw std::invalid_argument("Element::addOffNode: tau must lie in (-1, 1)");
  PointAttachment p;
  p.kind = PointKind::OffNode;
  p.con = std::move(con);
  p.taus = std::move(taus);
  pointTerms.push_back(std::move(p));
  return *this;
}

Element& Element::addForceControlPoints(std::shared_ptr<Constraint> con, std::vector<int> controlPoints) {
  if (!con) throw std::invalid_argument("Element::addForceControlPoints: null constraint");
  if (!con->readsOnlyForce())
    throw std::invalid_argument("Element::addForceControlPoints: '" + con->name() + "' does not read only the force");
  if (controlPoints.empty()) return *this;
  PointAttachment p;
  p.kind = PointKind::ForceControlPoints;
  p.con = std::move(con);
  p.controlPoints = std::move(controlPoints);
  pointTerms.push_back(std::move(p));
  return *this;
}

Element& Element::addFieldFunctionals(std::shared_ptr<Constraint> con, Field field,
                                      std::vector<Eigen::VectorXd> weights) {
  if (!con) throw std::invalid_argument("Element::addFieldFunctionals: null constraint");
  if (weights.empty()) return *this;
  PointAttachment p;
  p.kind = PointKind::FieldFunctionals;
  p.con = std::move(con);
  p.field = field;
  p.weights = std::move(weights);
  pointTerms.push_back(std::move(p));
  return *this;
}

Element& Element::addCost(std::shared_ptr<Cost> cost) {
  if (!cost) throw std::invalid_argument("Element::addCost: null cost");
  costs.push_back(std::move(cost));
  return *this;
}

Element& Element::setDefectCost(std::shared_ptr<Constraint> dynamics, std::vector<double> taus, double rho,
                                std::vector<double> weights) {
  if (!dynamics) throw std::invalid_argument("Element::setDefectCost: null dynamics");
  if (!weights.empty() && weights.size() != taus.size())
    throw std::invalid_argument("Element::setDefectCost: one weight per tau");
  defect.dynamics = std::move(dynamics);
  defect.taus = std::move(taus);
  defect.weights = std::move(weights);
  defect.rho = rho;
  return *this;
}

Element& TrajectoryProblem::addElement(const Element& e) {
  elements.push_back(e);
  return elements.back();
}

void TrajectoryProblem::addReset(std::shared_ptr<ResetMap> reset) {
  if (!reset) throw std::invalid_argument("TrajectoryProblem::addReset: null reset");
  resets.push_back(std::move(reset));
}

double TrajectoryProblem::totalTime() const {
  double T = 0.0;
  for (const Element& e : elements) T += e.T;
  return T;
}

std::vector<int> TrajectoryProblem::elementsOfPhase(int phase) const {
  std::vector<int> out;
  for (size_t i = 0; i < elements.size(); ++i)
    if (elements[i].phase == phase) out.push_back(static_cast<int>(i));
  return out;
}

void TrajectoryProblem::validate() const {
  if (!model) throw std::invalid_argument("TrajectoryProblem: no model");
  if (elements.empty()) throw std::invalid_argument("TrajectoryProblem: no elements");
  if (resets.size() + 1 != elements.size())
    throw std::invalid_argument("TrajectoryProblem: need exactly elements - 1 resets");
  for (size_t i = 0; i < elements.size(); ++i) {
    const Element& e = elements[i];
    const std::string tag = "element " + std::to_string(i) + " (" + e.label + ")";
    if (!(e.T > 0.0)) throw std::invalid_argument(tag + ": duration must be positive");
    if (e.degree < 1) throw std::invalid_argument(tag + ": degree must be >= 1");
    if (!e.contacts.active.empty() && e.forceControlPoints < 1)
      throw std::invalid_argument(tag + ": a stance element needs >= 1 force control point");
    for (const NodeAttachment& a : e.nodeTerms)
      if (a.where == Where::OneNode && (a.node < 0 || a.node >= e.numNodes()))
        throw std::invalid_argument(tag + ": node index out of range for '" + a.con->name() + "'");
    for (const PointAttachment& p : e.pointTerms) {
      if (p.kind == PointKind::ForceControlPoints) {
        if (e.contacts.active.empty()) throw std::invalid_argument(tag + ": force control points on an airborne element");
        for (int j : p.controlPoints)
          if (j < 0 || j >= e.forceControlPoints)
            throw std::invalid_argument(tag + ": control point index out of range for '" + p.con->name() + "'");
      }
      if (p.kind == PointKind::FieldFunctionals) {
        if (p.field == Field::Lam) throw std::invalid_argument(tag + ": use addForceControlPoints for the force");
        if (p.field == Field::U && e.deriveTorque)
          throw std::invalid_argument(tag + ": functionals of u need a stored torque field");
        for (const Eigen::VectorXd& w : p.weights)
          if (w.size() != e.numNodes()) throw std::invalid_argument(tag + ": functional weights must have Nn entries");
      }
    }
    if (e.defect.rho > 0.0 && e.defect.taus.empty()) throw std::invalid_argument(tag + ": defect cost needs taus");
  }
}

std::vector<double> uniformInteriorTaus(int n) {
  std::vector<double> t;
  for (int i = 0; i < n; ++i) t.push_back(-1.0 + 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n));
  return t;
}

void gaussLegendreInteriorTaus(int n, std::vector<double>& taus, std::vector<double>& weights) {
  taus.clear();
  weights.clear();
  if (n <= 0) return;
  Eigen::VectorXd x, w;
  gaussLegendre(n, x, w);
  for (int i = 0; i < n; ++i) {
    taus.push_back(x(i));
    weights.push_back(w(i));
  }
}

}  // namespace leap
