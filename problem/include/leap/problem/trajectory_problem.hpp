#pragma once

// TrajectoryProblem: the declarative description of one optimal control problem.
//
// A problem is a chain of ELEMENTS (Chebyshev segments) joined by RESET MAPS. An element
// carries a duration, an active contact set, a polynomial degree (Nn = degree + 1 CGL
// nodes), the number of Bezier control points of its contact force, and its terms:
//
//   addNode(con, where)             a constraint collocated at the nodes `where` selects
//   addOffNode(con, taus)           the same constraint at off-node points of the element
//                                   (interpolation rows; these points are MOVABLE, see
//                                   the relocation oracle of the solver)
//   addForceControlPoints(con, j)   a force-only constraint on the physical Bezier control
//                                   points j (convexity then certifies it along the curve)
//   addFieldFunctionals(con, f, W)  a constraint on linear functionals sum_k W_k field_k of
//                                   one nodal field (e.g. the Bernstein coefficients of u)
//   addCost(cost)                   an objective term summed over the nodes
//   setDefectCost(dyn, taus, rho)   the least-squares dynamics-defect cost between nodes
//
// Several elements may belong to one PHASE (contact mode): a phase split into elements
// is joined by continuity resets, a contact change by a plastic impact. The task
// builders (tasks/) fill a TrajectoryProblem from a problem yaml; a user may also do it
// by hand. CompiledProblem::compile turns it into the row space the solver flows.

#include <memory>
#include <string>
#include <vector>

#include "leap/model/robot_model.hpp"
#include "leap/problem/constraint.hpp"
#include "leap/problem/constraints.hpp"
#include "leap/problem/reset_map.hpp"

namespace leap {

// Where a node term applies within an element (ascending time: FirstNode = t0).
enum class Where { FirstNode, LastNode, Interior, AllNodes, OneNode };

struct NodeAttachment {
  std::shared_ptr<Constraint> con;
  Where where = Where::AllNodes;
  int node = -1;  // Where::OneNode: exactly this node
  bool appliesTo(int nodeIdx, int numNodes) const;
};

enum class PointKind { OffNode, ForceControlPoints, FieldFunctionals };

// Rows enforced away from the nodes. One attachment = one term "<name>_interp".
struct PointAttachment {
  PointKind kind = PointKind::OffNode;
  std::shared_ptr<Constraint> con;
  std::vector<double> taus;              // OffNode: points in (-1, 1)
  std::vector<int> controlPoints;        // ForceControlPoints: indices into 0..Nc-1
  Field field = Field::U;                // FieldFunctionals: which nodal field
  std::vector<Eigen::VectorXd> weights;  // FieldFunctionals: per row set, length Nn
};

// (rho/2) sum_m w_m ||r(tau_m)||^2 with r the dynamics residual at off-node points,
// the force taken from the Bezier curve. Pure cost: no multiplier rows.
struct DefectCost {
  std::shared_ptr<Constraint> dynamics;  // a dynamics constraint of the element's contact set
  std::vector<double> taus;
  std::vector<double> weights;  // empty = midpoint rule 2/n
  double rho = 0.0;             // 0 = no defect cost
};

struct Element {
  std::string label;
  int phase = 0;   // parent contact mode (bookkeeping, archives)
  double T = 1.0;  // duration
  ContactSet contacts;
  int degree = 4;
  int forceControlPoints = 0;  // Nc (ignored when the contact set is empty)
  bool deriveTorque = false;   // inverse-dynamics torque elimination: u is not a decision
  std::vector<NodeAttachment> nodeTerms;
  std::vector<PointAttachment> pointTerms;
  std::vector<std::shared_ptr<Cost>> costs;
  DefectCost defect;

  Element& addNode(std::shared_ptr<Constraint> con, Where where);
  Element& addNodeAt(std::shared_ptr<Constraint> con, int node);
  Element& addOffNode(std::shared_ptr<Constraint> con, std::vector<double> taus);
  Element& addForceControlPoints(std::shared_ptr<Constraint> con, std::vector<int> controlPoints);
  Element& addFieldFunctionals(std::shared_ptr<Constraint> con, Field field, std::vector<Eigen::VectorXd> weights);
  Element& addCost(std::shared_ptr<Cost> cost);
  Element& setDefectCost(std::shared_ptr<Constraint> dynamics, std::vector<double> taus, double rho,
                         std::vector<double> weights = {});
  int numNodes() const { return degree + 1; }
};

struct TrajectoryProblem {
  std::shared_ptr<const RobotModel> model;
  std::vector<Element> elements;
  std::vector<std::shared_ptr<ResetMap>> resets;  // elements.size() - 1
  bool preconditionForce = true;  // scale the force control points by sigma_j

  Element& addElement(const Element& e);
  void addReset(std::shared_ptr<ResetMap> reset);
  double totalTime() const;
  // Elements sharing the same `phase` index, in order.
  std::vector<int> elementsOfPhase(int phase) const;
  void validate() const;  // throws std::invalid_argument
};

// Uniformly spaced points in the open interval (-1, 1): tau_i = -1 + 2 (i + 1/2) / n.
std::vector<double> uniformInteriorTaus(int n);
// Gauss-Legendre points in (-1, 1) with their weights (sum 2).
void gaussLegendreInteriorTaus(int n, std::vector<double>& taus, std::vector<double>& weights);

}  // namespace leap
