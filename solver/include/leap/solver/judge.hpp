#pragma once

// The judges (evaluate-only).
//
// Node gate: max |c_eff| over every row at zero multipliers.
// Fine grid: the whole trajectory sampled on t_k = k dt; every throughout constraint of
// the owning element (node terms attached AllNodes/Interior, plain off-node rows) is
// evaluated from the element's own polynomials (barycentric q, v, a, u; the Bezier
// force curve; inverse-dynamics torque when it is eliminated). Force control-point rows
// whose weights are one-hot are judged on the physical curve as well; a row set that
// leaves exactly one control point out (a ramp or taper pin next to it) is reported but
// not gated. Boundary pins ("pin*") and the mid-swing window ("ineq_swing_z") are
// node-only by nature and are skipped.

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/alm/evaluator.hpp"
#include "leap/alm/penalties.hpp"

namespace leap {

struct FineGridTerm {
  std::string name;
  bool ineq = false;
  double worst = 0.0;
  int elem = -1;
  double t = 0.0;
  int row = -1;
  bool controlPoint = false;
  bool partialCoverage = false;
};

struct FineGridReport {
  double dt = 0.0;
  int nSamples = 0;
  double worst = 0.0, worstGate = 0.0;
  std::string worstName, worstGateName;
  int worstElem = -1, worstGateElem = -1;
  double worstT = 0.0, worstGateT = 0.0;
  double worstDyn = 0.0, worstNonDyn = 0.0;
  std::vector<FineGridTerm> terms;  // worst first
  std::vector<double> elemWorst, elemWorstGate, elemWorstTau, elemWorstGateTau;
  std::vector<std::string> elemWorstName, elemWorstGateName;
};

FineGridReport fineGridViolations(const CompiledProblem& prob, const Eigen::VectorXd& y, double dt);
void printFineGrid(const FineGridReport& r, int maxTerms = 24);

// Diagnostic group of a constraint name (dynamics / foot / boundary / inequality / ...).
std::string constraintGroup(const std::string& name);
// Per-group zero-dual violations of an evaluation.
void printViolationsByGroup(const CompiledProblem& prob, const Evaluation& ev);

struct ElementIndicator {
  double ind = 0.0, node = 0.0, fine = 0.0, fineTau = 0.0;
  std::string worst;
};

struct Judgement {
  double gate = 0.0;     // zero-dual node gate (all rows)
  double fine = 0.0;     // gated fine-grid worst
  double fineAll = 0.0;  // ungated fine-grid worst
  std::string fineName;
  int fineElem = -1;
  double fineTau = 0.0, fineT = 0.0;
  std::vector<ElementIndicator> elements;
  bool met(double tol) const { return gate <= tol && fine <= tol; }
};

Judgement judgeLevel(AlmEvaluator& ev, const Penalties& pen, const Eigen::VectorXd& y, double dt, bool gated);

}  // namespace leap
