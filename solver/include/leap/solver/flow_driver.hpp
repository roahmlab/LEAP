#pragma once

// One frozen-dual gradient flow of the augmented Lagrangian, integrated in windows.
//
// Inside a window the multipliers gamma and penalties mu are frozen and the primal
// descends L_A by linearly implicit Levenberg-Marquardt steps,
//     (I + h H) dz = -h grad_z L_A(v),     v = y + beta (y - y_prev)   (Nesterov look-ahead),
// accepted only on STRICT decrease of L_A (backtracking alpha /= 4 along dz), with the
// ratio of actual to predicted decrease growing or shrinking h. A window ends at its
// pseudo-time budget, on the gradient certificate, or on a merit stall; at the boundary
// the damped multiplier update gamma += tau mu .* c_eff and the per-term penalty policy
// step are applied, the level stop predicate is consulted, and the run-level safety
// nets (keepBest, blow-up recovery, plateau, h-collapse) tick.

#include <string>

#include <Eigen/Dense>

#include <memory>

#include "leap/alm/evaluator.hpp"
#include "leap/alm/first_order.hpp"
#include "leap/alm/penalties.hpp"
#include "leap/linalg/linear_step.hpp"

namespace leap {

// Observer of every accepted step (the budget-study snapshot writer): `y` is the accepted
// iterate, `rawGate` the raw max violation of the flow's rows at y (max |g|, or max(0, g)
// on inequality rows, which is the zero-dual node gate of the flow's own problem).
class StepObserver {
public:
  virtual ~StepObserver() = default;
  virtual void onAcceptedStep(const Eigen::VectorXd& y, double rawGate, long steps) = 0;
};

struct FlowConfig {
  LinearBackend backend = LinearBackend::BlockSchur;
  bool cholmod = true;          // CHOLMOD for the sparse factorizations when available
  bool exactPositiveDefinite = false;  // exact-Newton metric: Cholesky (shrink h on failure) instead of LDLT
  bool entryMean = false;       // entry iterate: project onto the junction-group mean (the reference's
                                // `project`) instead of snapping every member to the representative
  double theta = 1.0;
  double h0 = 1e-4, hMin = 2e-5, hMax = 5.0;
  double hShrink = 0.5, ratioGrow = 2.0, ratioLow = 0.25, ratioHigh = 0.75;
  double dyMaxRel = 1.0;        // geometric backstop
  int solveRetries = 3;
  double window = 20.0;         // pseudo-time per window
  double sMax = 1000.0;         // pseudo-time budget
  long maxSteps = 4000;         // accepted-step cap (ends the run)
  int maxStepsPerWindow = 0;
  double tolFlow = 1e-6;        // gradient certificate and the 'converged' test
  double feasStop = 0.0;        // > 0: stop when max|c_eff| <= feasStop
  bool keepBest = true;         // return the lowest-max|c| window iterate
  int stallWindows = 12;        // max|c| plateau patience (0 = off)
  double stallRel = 1e-2;
  int hCollapseWindows = 4;     // 0 = off
  double blowupFactor = 10.0;   // 0 = off
  int blowupRetries = 3;
  bool momentum = true;         // Nesterov look-ahead
  bool dualUpdate = false;      // gamma += dualStep mu .* c_eff at every boundary
  double dualStep = 0.5;
  bool resetAfterDualUpdate = false;
  double meritStallRel = 1e-3;
  // Objective mode: a boundary whose cost moved by more than this relative amount is
  // progress, so it does not count toward the max|c| plateau (0 = off).
  double objectiveProgressRel = 0.0;
  bool verbose = true;
  std::string header;
  StepObserver* observer = nullptr;  // called after every accepted step (not owned)
};

struct FlowReport {
  double s = 0.0;
  long steps = 0, windows = 0;
  double maxViol = 0.0, gnorm = 0.0, hFinal = 0.0;
  bool converged = false, stalled = false, diverged = false, stopFired = false;
  long shrinks = 0, grows = 0, restarts = 0, backstops = 0, solveFails = 0, dualUpdates = 0, blowupRecoveries = 0,
       trials = 0, factorizations = 0;
  // the last boundary's objective value and first-order residuals (objective mode)
  double objective = 0.0, kktRel = 0.0, complementarityRel = 0.0;
  std::string stop;  // human-readable reason
};

// What a window boundary knows besides the iterate.
struct BoundaryInfo {
  long steps = 0, windows = 0;
  double s = 0.0;
  double maxViol = 0.0, gnorm = 0.0;
  double cost = 0.0;  // the weighted node-cost sum (the objective) at y
  FirstOrderResidual firstOrder;
};

// Consulted at every window boundary (after the diagnostic evaluation at y, before the
// multiplier update). Returning true ends the run with the current iterate.
class WindowStopCheck {
public:
  virtual ~WindowStopCheck() = default;
  virtual bool stop(const Eigen::VectorXd& y, const BoundaryInfo& boundary) = 0;
};

class FlowDriver {
public:
  FlowDriver(AlmEvaluator& evaluator, Penalties& penalties, const FlowConfig& cfg, WindowStopCheck* stopCheck = nullptr);
  FlowReport run(Eigen::VectorXd& y);

private:
  struct Nesterov {
    double t = 1.0;
    double tNext() const;
    double beta() const { return (t - 1.0) / tNext(); }
    void onAccept() { t = tNext(); }
    void reset() { t = 1.0; }
  };
  void evaluateAt(const Eigen::VectorXd& y, bool full, Evaluation& out);
  double reducedGradientNorm(const Evaluation& ev) const;

  // wall-time split of one run (printed when verbose)
  struct Timing {
    double evalFull = 0.0, evalValue = 0.0, metric = 0.0, factor = 0.0, solve = 0.0;
    long nEvalFull = 0, nEvalValue = 0, nMetric = 0;
  };
  Timing timing_;

  AlmEvaluator* ev_;
  Penalties* pen_;
  FlowConfig cfg_;
  WindowStopCheck* stop_;
  std::unique_ptr<LinearStep> solver_;
  Evaluation cache_, trial_;
};

}  // namespace leap
