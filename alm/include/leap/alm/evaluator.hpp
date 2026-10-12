#pragma once

// AlmEvaluator: everything one iterate y = [x; gamma] needs for one step of the flow.
//
//   evaluate(y, mu, request, out):
//     the residual vector g (every row of the compiled problem), the PHR quantities
//     c_eff / w / active sets, the merit L_A = f + defect + penalty, the per-term
//     violations, and (gradient requests) the flow right-hand side
//         F = [ -grad_x L_A ; c_eff or 0 ]
//     together with the Jacobian blocks in DECISION coordinates that the metric needs.
//   assembleMetric(out, mu, sink):
//     the Gauss-Newton metric H = H_cost + sum_active mu_i J_i' J_i + rho sum w_m Jf_m' Jf_m
//     (+ the contracted Hessians when exact Newton is requested), handed to a sink as
//     dense blocks over index sets.
//
// Decision coordinates reach the physical per-node vector through a LIFT: at node k,
// z = S_k x_support with S_k the identity on q, v, a (and u when stored), the Bernstein
// row E(k, :) sigma on the force control points, and d(u_ID)/d(support) on the torque
// rows when the torque is eliminated; at an off-node point tau, z = T x_elem with the
// barycentric weights on the nodal fields and the Bernstein basis on the control points.
// Jacobians are J_z T, Hessians T' H_z T, gradients T' g_z.

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "leap/alm/junction_quotient.hpp"
#include "leap/alm/phr.hpp"
#include "leap/problem/compiled_problem.hpp"

namespace leap {

struct EvaluationRequest {
  bool value = true;      // merit
  bool gradient = true;   // F (needs jacobian)
  bool jacobian = true;   // Jacobian blocks (derivative model passes)
  bool freezeDuals = true;  // F.tail(md) = 0 instead of c_eff
};

struct Evaluation {
  Eigen::VectorXd g;
  PhrState phr;
  double cost = 0.0, defectCost = 0.0, penalty = 0.0, merit = 0.0;
  double maxViol = 0.0;             // max |c_eff|
  Eigen::VectorXd perTermViol;      // raw max |g| (or max(0, g) on inequality rows) per term
  Eigen::VectorXd F;                // np + md
  // Derived per-element quantities.
  std::vector<Eigen::MatrixXd> nodeForce;  // nlam x Nn
  std::vector<Eigen::MatrixXd> derivedU;   // nu x Nn on torque-eliminated elements
  // Jacobian blocks in decision coordinates (jacobian requests).
  std::vector<Eigen::MatrixXd> Jnode;      // per node: rows x |zIdx|
  std::vector<Eigen::MatrixXd> Snode;      // per node: nP x |zIdx| (the lift)
  std::vector<Eigen::MatrixXd> Hcost;      // per node: nP x nP physical cost Hessian (weighted); empty when no cost
  std::vector<Eigen::VectorXd> gcost;      // per node: |zIdx| lifted, weighted cost gradient
  std::vector<Eigen::MatrixXd> Jpoint;     // per point info: (nPoints rows) x |cols|
  std::vector<Eigen::MatrixXd> Hpoint;     // per point info: exact contracted Hessian (only when requested)
  std::vector<Eigen::MatrixXd> Jdefect;    // per defect: (nPoints rows) x |zIdxAll|
  std::vector<Eigen::VectorXd> rdefect;    // per defect: residuals, point-major
  std::vector<Eigen::MatrixXd> Jreset;     // per reset: rows x |zIdx| (identity rows zeroed under the quotient)
  std::vector<Eigen::MatrixXd> Hnode;      // per node: exact contracted Hessians (only when requested), nP x nP
  // Torque elimination, exact-Newton only: the u columns of the physical Jacobians and
  // the u part of the weighted cost gradient, so that the lift's own curvature
  // sum_j (dg/du . w + dl/du)_j d2 u_ID,j / dz2 can be added in the Hessian pass.
  std::vector<Eigen::MatrixXd> JuNode;     // per node: rows x nu (empty when u is stored)
  std::vector<Eigen::VectorXd> guCostNode; // per node: nu (weighted cost gradient wrt u)
  std::vector<Eigen::MatrixXd> JuPoint;    // per off-node point info: (nPoints rows) x nu
};

// Receives the metric per ELEMENT, over FULL decision indices: a dense symmetric block
// (H += B over idx) or weighted Jacobian rows (H += J' diag(w) J over idx). `element` is
// the element every index belongs to (its own coordinates or a junction it touches);
// calls with different elements may run concurrently.
class MetricSink {
public:
  virtual ~MetricSink() = default;
  virtual void addBlock(int element, const std::vector<int>& idx, const Eigen::MatrixXd& B) = 0;
  virtual void addWeightedRows(int element, const std::vector<int>& idx, const Eigen::MatrixXd& J, const Eigen::VectorXd& w) = 0;
};

struct EvaluatorOptions {
  bool gaussNewton = true;         // drop the contracted Hessians
  double activeMargin = 0.0;       // anticipatory active set of the metric (row units)
  bool ineqViolMax0 = true;        // per-term violations count max(0, g) on inequality rows
  bool shareJunctions = true;      // the junction quotient (identity reset rows dropped)
  int threads = 0;                 // 0 = OpenMP default
};

class AlmEvaluator {
public:
  AlmEvaluator(const CompiledProblem& prob, const EvaluatorOptions& opt);
  ~AlmEvaluator();
  AlmEvaluator(const AlmEvaluator&) = delete;
  AlmEvaluator& operator=(const AlmEvaluator&) = delete;

  const CompiledProblem& problem() const { return *prob_; }
  const EvaluatorOptions& options() const { return opt_; }
  const JunctionQuotient& quotient() const { return *quotient_; }

  void evaluate(const Eigen::VectorXd& y, const Eigen::VectorXd& muRow, const EvaluationRequest& req,
                Evaluation& out);
  // The metric from the blocks of a jacobian evaluation at y: Gauss-Newton, or exact
  // Newton (the contracted Hessians of every term with the PHR weights of ev) when the
  // options say so. y must be the iterate ev was evaluated at.
  void assembleMetric(const Eigen::VectorXd& y, const Evaluation& ev, const Eigen::VectorXd& muRow, MetricSink& sink) const;

  // Zero-dual judge: max |c_eff| at [x; 0] and the per-term violations (values only).
  double zeroDualViolation(const Eigen::VectorXd& y, const Eigen::VectorXd& muRow, Evaluation& out);

  // Global node index of node k of element e.
  int nodeIndex(int elem, int k) const { return nodeBase_[static_cast<size_t>(elem)] + k; }

private:
  struct Scratch;
  void evaluateNodes(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out);
  void evaluatePoints(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out);
  void evaluateDefects(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out);
  void evaluateResets(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out);
  void evaluateKinematics(const Eigen::VectorXd& y, Evaluation& out) const;
  void accumulateRhs(const Eigen::VectorXd& y, const EvaluationRequest& req, Evaluation& out) const;
  void buildNodeLift(const ElementLayout& L, int k, const ModelEvalCache& c, bool torqueChain, Scratch& s,
                     Eigen::MatrixXd& S) const;
  void buildPointLift(const ElementLayout& L, const Eigen::VectorXd& W, const Eigen::VectorXd& Wlam,
                      const ModelEvalCache& c, bool torqueChain, Scratch& s, Eigen::MatrixXd& T) const;
  void torqueJacobian(const ElementLayout& L, const ModelEvalCache& c, Eigen::MatrixXd& Ju) const;
  void assembleExactHessians(const Eigen::VectorXd& y, const Evaluation& ev, MetricSink& sink) const;
  void addTorqueLiftHessian(const ElementLayout& L, const NodeQuantities& z, const Eigen::VectorXd& gu, Scratch& s,
                            Eigen::MatrixXd& H) const;

  const CompiledProblem* prob_;
  EvaluatorOptions opt_;
  std::unique_ptr<JunctionQuotient> quotient_;
  std::vector<std::unique_ptr<Scratch>> scratch_;  // one per thread
  std::vector<int> nodeBase_;
  std::vector<std::vector<int>> pointsOfElem_, defectsOfElem_, resetsOfHead_;
  std::vector<double> nodeCostOf_, defectCostOf_;  // per-node / per-defect costs, reduced in a fixed order
};

}  // namespace leap
