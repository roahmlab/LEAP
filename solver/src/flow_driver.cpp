#include "leap/solver/flow_driver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "leap/linalg/block_schur.hpp"
#include "leap/linalg/global_metric.hpp"

namespace leap {

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::unique_ptr<LinearStep> makeLinearStep(const AlmEvaluator& ev, const FlowConfig& cfg) {
  if (cfg.backend == LinearBackend::BlockSchur) {
    LinearSolverOptions o;
    o.useCholmod = cfg.cholmod;
    o.allowIndefinite = !ev.options().gaussNewton && !cfg.exactPositiveDefinite;  // the exact metric need not be positive definite
    return std::make_unique<BlockSchurSolver>(ev.problem(), ev.quotient(), o);
  }
  return std::make_unique<GlobalMetricSolver>(ev.problem(), ev.quotient(), cfg.backend);
}
}  // namespace

double FlowDriver::Nesterov::tNext() const { return (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0; }

FlowDriver::FlowDriver(AlmEvaluator& evaluator, Penalties& penalties, const FlowConfig& cfg, WindowStopCheck* stopCheck)
    : ev_(&evaluator), pen_(&penalties), cfg_(cfg), stop_(stopCheck), solver_(makeLinearStep(evaluator, cfg)) {}

void FlowDriver::evaluateAt(const Eigen::VectorXd& y, bool full, Evaluation& out) {
  EvaluationRequest req;
  req.value = true;
  req.gradient = full;
  req.jacobian = full;
  req.freezeDuals = true;
  const double t0 = now();
  ev_->evaluate(y, pen_->muRow(), req, out);
  const double dt = now() - t0;
  if (full) {
    timing_.evalFull += dt;
    ++timing_.nEvalFull;
  } else {
    timing_.evalValue += dt;
    ++timing_.nEvalValue;
  }
}

double FlowDriver::reducedGradientNorm(const Evaluation& ev) const {
  Eigen::VectorXd Fz;
  ev_->quotient().reduceCovector(ev.F.head(ev_->problem().layout().np), Fz);
  return Fz.norm();
}

FlowReport FlowDriver::run(Eigen::VectorXd& y) {
  const Layout& L = ev_->problem().layout();
  const JunctionQuotient& Q = ev_->quotient();
  const int np = L.np, md = L.md;
  const double inf = std::numeric_limits<double>::infinity();
  FlowReport rep;
  if (cfg_.entryMean) Q.project(y); else Q.enforce(y);
  Eigen::VectorXd yPrev = y, v, dz, dx, yTrial, Fz;
  Nesterov mom;
  double s = 0.0, hCtrl = cfg_.h0;
  double meritY = inf;
  bool meritYValid = false;
  // keepBest
  Eigen::VectorXd snapY;
  double snapViol = inf, snapGnorm = inf;
  bool entrySnapped = false;
  // run-level bookkeeping
  double bestViol = inf, bestWindowViol = inf, hCollapseRef = inf, sLastMu = 0.0;
  int nStall = 0, nHCollapse = 0, blowupRecoveries = 0, solveFails = 0;
  double plateauObjRef = std::numeric_limits<double>::quiet_NaN();
  bool converged = false;
  if (cfg_.verbose) {
    if (!cfg_.header.empty()) std::printf("%s\n", cfg_.header.c_str());
    std::printf("  %10s %9s %11s %11s\n", "s", "mu_rep", "max|c_eff|", "|grad|");
  }

  while (s < cfg_.sMax - 1e-12 && !converged) {
    ++rep.windows;
    const double sEnd = std::min(s + cfg_.window, cfg_.sMax);
    int meritStall = 0;
    bool windowStationary = false;
    const long stepsAtWindowStart = rep.steps;

    while (s < sEnd - 1e-12) {
      const double h = std::min(hCtrl, sEnd - s);
      // look-ahead
      const double beta = cfg_.momentum ? mom.beta() : 0.0;
      v = y + beta * (y - yPrev);
      Q.enforce(v);
      // monotone restart (values only): L(v) must not exceed L(y)
      if ((v - y).squaredNorm() > 0.0) {
        evaluateAt(v, false, cache_);
        const double Lv = cache_.merit;
        if (!meritYValid) {
          evaluateAt(y, false, trial_);
          meritY = trial_.merit;
          meritYValid = true;
        }
        if (!std::isfinite(Lv) || (std::isfinite(meritY) && Lv > meritY)) {
          mom.reset();
          yPrev = y;
          v = y;
          ++rep.restarts;
        }
      }
      evaluateAt(v, true, cache_);
      rep.maxViol = cache_.maxViol;
      rep.gnorm = reducedGradientNorm(cache_);
      if (cfg_.keepBest && !entrySnapped) {
        snapViol = rep.maxViol;
        snapGnorm = rep.gnorm;
        snapY = y;
        entrySnapped = true;
      }
      if (rep.maxViol < cfg_.tolFlow && rep.gnorm < cfg_.tolFlow) {
        converged = true;
        break;
      }
      if (cfg_.feasStop > 0.0 && rep.maxViol <= cfg_.feasStop) {
        converged = true;
        break;
      }
      if (rep.gnorm < cfg_.tolFlow) {
        windowStationary = true;  // gradient certificate
        break;
      }
      // the linearly implicit step: (I + h theta H) dz = h F_z
      const double tm0 = now();
      solver_->beginAssembly();
      ev_->assembleMetric(v, cache_, pen_->muRow(), *solver_);
      Q.reduceCovector(cache_.F.head(np), Fz);
      const double tm1 = now();
      bool ok = solver_->factor(h * cfg_.theta);
      const double tm2 = now();
      if (ok) ok = solver_->solve(h * Fz, dz);
      const double tm3 = now();
      timing_.metric += tm1 - tm0;
      timing_.factor += tm2 - tm1;
      timing_.solve += tm3 - tm2;
      ++timing_.nMetric;
      rep.factorizations = solver_->numFactorizations();
      if (!ok || !dz.allFinite()) {
        if (cfg_.solveRetries > 0 && solveFails < cfg_.solveRetries && hCtrl > cfg_.hMin) {
          ++solveFails;
          ++rep.solveFails;
          hCtrl = std::max(cfg_.hMin, hCtrl * cfg_.hShrink);
          mom.reset();
          if (cfg_.verbose) std::printf("  [flow] linear solve failed at s=%.4e -> shrink h to %.2e and retry (%d/%d)\n", s, hCtrl, solveFails, cfg_.solveRetries);
          continue;
        }
        if (cfg_.verbose) std::printf("  [flow] linear solve failed at s=%.4e\n", s);
        rep.diverged = true;
        rep.stop = "linear solve failed";
        rep.s = s;
        if (cfg_.keepBest && snapY.size() > 0 && snapViol < rep.maxViol) {
          y = snapY;
          rep.maxViol = snapViol;
          rep.gnorm = snapGnorm;
        }
        Q.enforce(y);
        return rep;
      }
      solveFails = 0;
      Q.scatter(dz, dx);
      // geometric backstop: a step longer than the iterate itself is damped first
      {
        Eigen::VectorXd zCur;
        Eigen::VectorXd xv = v.head(np);
        Q.reduceCovector(xv, zCur);  // sums members: only representatives are needed, so rebuild
        double zn2 = 0.0;
        for (int r = 0; r < Q.reducedSize(); ++r) zn2 += xv(Q.representative(r)) * xv(Q.representative(r));
        if (dz.norm() > cfg_.dyMaxRel * std::sqrt(zn2)) {
          if (hCtrl > cfg_.hMin) {
            hCtrl = std::max(cfg_.hMin, hCtrl * cfg_.hShrink);
            mom.reset();
            ++rep.backstops;
            ++rep.restarts;
            continue;
          }
        }
      }
      // merit acceptance with backtracking along dz
      const double L0 = cache_.merit;
      const double fdy0 = Fz.dot(dz);
      const double dn20 = dz.squaredNorm();
      double alpha = 1.0, LTrial = std::numeric_limits<double>::quiet_NaN();
      bool accept = false;
      for (;;) {
        yTrial = v;
        yTrial.head(np) += alpha * dx;
        Q.enforce(yTrial);
        evaluateAt(yTrial, false, trial_);
        ++rep.trials;
        LTrial = trial_.merit;
        accept = std::isfinite(LTrial) && (LTrial < L0 || !std::isfinite(L0));
        if (accept || alpha <= 1e-10) break;
        alpha *= 0.25;
      }
      // predicted decrease of the Gauss-Newton model along alpha dz
      const double pred = alpha * (1.0 - 0.5 * alpha / cfg_.theta) * fdy0 + alpha * alpha * dn20 / (2.0 * h * cfg_.theta);
      double stepRatio = std::numeric_limits<double>::quiet_NaN();
      if (pred > 0.0 && std::isfinite(LTrial) && std::isfinite(L0)) stepRatio = (L0 - LTrial) / pred;
      if (!accept) {
        // merit floor: no damped step decreases L_A at these multipliers
        rep.stalled = true;
        rep.stop = "merit floor";
        s = cfg_.sMax;
        break;
      }
      meritY = LTrial;
      meritYValid = true;
      if (hCtrl < cfg_.hMin) hCtrl = cfg_.hMin;
      if (alpha < 1.0) hCtrl = std::max(cfg_.hMin, hCtrl * cfg_.hShrink);
      if (alpha <= 1e-6 ||
          (cfg_.meritStallRel > 0.0 && std::isfinite(L0) &&
           L0 - LTrial < cfg_.meritStallRel * (alpha * h) * std::max(std::abs(LTrial), 1e-12)))
        ++meritStall;
      else
        meritStall = 0;
      // accept
      yPrev = y;
      y = yTrial;
      mom.onAccept();
      s += alpha * h;
      ++rep.steps;
      if (cfg_.observer != nullptr)
        cfg_.observer->onAcceptedStep(y, trial_.perTermViol.size() > 0 ? trial_.perTermViol.maxCoeff() : trial_.maxViol, rep.steps);
      if (cfg_.dualUpdate && meritStall >= 3) {
        windowStationary = true;
        break;
      }
      if (cfg_.maxSteps > 0 && rep.steps >= cfg_.maxSteps) {
        rep.stalled = true;
        rep.stop = "step cap";
        s = cfg_.sMax;
        break;
      }
      if (cfg_.maxStepsPerWindow > 0 && rep.steps - stepsAtWindowStart >= cfg_.maxStepsPerWindow) break;
      // ratio step controller
      if (std::isfinite(stepRatio) && stepRatio > cfg_.ratioHigh && alpha >= 1.0) {
        hCtrl = std::min(cfg_.hMax, hCtrl * cfg_.ratioGrow);
        ++rep.grows;
      } else if (std::isfinite(stepRatio) && stepRatio < cfg_.ratioLow) {
        hCtrl = std::max(cfg_.hMin, hCtrl * cfg_.hShrink);
        ++rep.shrinks;
      }
      if (hCtrl > cfg_.hMax) hCtrl = cfg_.hMax;
      rep.hFinal = hCtrl;
    }
    (void)windowStationary;

    // --- window boundary -----------------------------------------------------------------
    evaluateAt(y, true, cache_);
    rep.maxViol = cache_.maxViol;
    rep.gnorm = reducedGradientNorm(cache_);
    if (rep.maxViol < cfg_.tolFlow && rep.gnorm < cfg_.tolFlow) converged = true;
    if (cfg_.feasStop > 0.0 && rep.maxViol <= cfg_.feasStop) {
      if (cfg_.verbose) std::printf("  [flow] feasibility stop: max|c| = %.3e <= %.3e at s=%.4f\n", rep.maxViol, cfg_.feasStop, s);
      converged = true;
    }
    rep.objective = cache_.cost;
    bool objProgress = false;
    if (cfg_.objectiveProgressRel > 0.0) {
      objProgress = std::isfinite(plateauObjRef) && std::abs(cache_.cost - plateauObjRef) > cfg_.objectiveProgressRel * std::max(1.0, std::abs(cache_.cost));
      plateauObjRef = cache_.cost;
    }
    bool stopFired = false;
    if (!converged && stop_ != nullptr) {
      BoundaryInfo bi;
      bi.steps = rep.steps;
      bi.windows = rep.windows;
      bi.s = s;
      bi.maxViol = rep.maxViol;
      bi.gnorm = rep.gnorm;
      bi.cost = cache_.cost;
      bi.firstOrder = firstOrderResidual(ev_->problem(), Q, cache_);
      rep.kktRel = bi.firstOrder.gradientRel;
      rep.complementarityRel = bi.firstOrder.complementarityRel;
      stopFired = stop_->stop(y, bi);
    }
    if (stopFired) {
      if (cfg_.verbose) std::printf("  [flow] stop predicate fired: max|c| = %.3e at s=%.4f\n", rep.maxViol, s);
      converged = true;
      rep.stopFired = true;
      if (cfg_.keepBest) {
        snapViol = rep.maxViol;
        snapGnorm = rep.gnorm;
        snapY = y;
      }
    }
    if (cfg_.keepBest && rep.maxViol < snapViol) {
      snapViol = rep.maxViol;
      snapGnorm = rep.gnorm;
      snapY = y;
    }
    // blow-up recovery
    bestWindowViol = std::min(bestWindowViol, rep.maxViol);
    if (cfg_.blowupFactor > 0.0 && !converged && rep.maxViol > cfg_.blowupFactor * bestWindowViol) {
      const bool canRecover = blowupRecoveries < cfg_.blowupRetries && snapY.size() > 0 && std::isfinite(snapViol) && hCtrl > cfg_.hMin;
      if (canRecover) {
        ++blowupRecoveries;
        ++rep.blowupRecoveries;
        if (cfg_.verbose)
          std::printf("  [flow] blow-up RECOVERY %d/%d at s=%.4f: max|c| = %.3e > %.1fx best %.3e -> roll back to %.3e\n",
                      blowupRecoveries, cfg_.blowupRetries, s, rep.maxViol, cfg_.blowupFactor, bestWindowViol, snapViol);
        y = snapY;
        yPrev = y;
        rep.maxViol = snapViol;
        rep.gnorm = snapGnorm;
        mom.reset();
        meritYValid = false;
        hCtrl = std::max(cfg_.hMin, hCtrl * cfg_.hShrink * cfg_.hShrink);
        continue;
      }
      if (cfg_.verbose) std::printf("  [flow] blow-up abort: max|c| = %.3e > %.1fx best %.3e at s=%.4f\n", rep.maxViol, cfg_.blowupFactor, bestWindowViol, s);
      rep.diverged = true;
      rep.stop = "blow-up";
      break;
    }
    blowupRecoveries = 0;
    // damped multiplier update
    if (cfg_.dualUpdate && !converged) {
      y.tail(md) += cfg_.dualStep * pen_->muRow().cwiseProduct(cache_.phr.ceff);
      Q.enforce(y);
      evaluateAt(y, true, cache_);
      ++rep.dualUpdates;
      meritYValid = false;
      if (cfg_.resetAfterDualUpdate) {
        mom.reset();
        yPrev = y;
        hCtrl = std::max(hCtrl, cfg_.h0);
      }
    }
    // penalty policy step on the window cadence
    if (s - sLastMu >= cfg_.window - 1e-12) {
      const Eigen::VectorXd muBefore = pen_->muRow();
      pen_->update(cache_.perTermViol);
      if (!(muBefore.array() == pen_->muRow().array()).all()) meritYValid = false;
      sLastMu = s;
    }
    if (cfg_.verbose)
      std::printf("  %10.4f %9.2e %11.3e %11.3e   h=%.2e steps=%ld\n", std::min(s, cfg_.sMax), pen_->muRepresentative(),
                  rep.maxViol, rep.gnorm, hCtrl, rep.steps);
    // max|c| plateau
    if (cfg_.stallWindows > 0 && !converged) {
      if (rep.maxViol < bestViol * (1.0 - cfg_.stallRel)) {
        bestViol = rep.maxViol;
        nStall = 0;
      } else if (objProgress) {
        nStall = 0;  // the objective still moves: max|c| sits at its floor while f descends
      } else if (++nStall >= cfg_.stallWindows) {
        rep.stalled = true;
        rep.stop = "max|c| plateau";
        if (cfg_.verbose)
          std::printf("  [flow] max|c| plateau (best %.3e, %d windows w/o %.0f%% improvement) at s=%.4f\n", bestViol, nStall, 100.0 * cfg_.stallRel, s);
        break;
      }
    }
    // h-collapse
    if (cfg_.hCollapseWindows > 0 && !converged) {
      if (hCtrl <= cfg_.hMin * 1.0001 && rep.maxViol >= hCollapseRef * (1.0 - cfg_.stallRel)) {
        if (++nHCollapse >= cfg_.hCollapseWindows) {
          rep.stalled = true;
          rep.stop = "h-collapse";
          if (cfg_.verbose) std::printf("  [flow] h-collapse abort: %d consecutive windows at h_min without progress at s=%.4f\n", nHCollapse, s);
          break;
        }
      } else {
        nHCollapse = 0;
      }
      hCollapseRef = rep.maxViol;
    }
  }
  rep.s = s;
  rep.converged = converged;
  rep.hFinal = hCtrl;
  if (rep.stop.empty()) rep.stop = converged ? (rep.stopFired ? "stop predicate" : "converged") : "pseudo-time budget";
  if (cfg_.keepBest && snapY.size() > 0 && snapViol < rep.maxViol) {
    y = snapY;
    rep.maxViol = snapViol;
    rep.gnorm = snapGnorm;
    if (cfg_.verbose) std::printf("  [flow] returning best window iterate (max|c|=%.3e)\n", snapViol);
  }
  Q.enforce(y);
  if (cfg_.verbose)
    std::printf("  [flow] time split: eval(full) %.2fs x%ld, eval(value) %.2fs x%ld, metric %.2fs, factor %.2fs, solve %.2fs x%ld\n",
                timing_.evalFull, timing_.nEvalFull, timing_.evalValue, timing_.nEvalValue, timing_.metric, timing_.factor, timing_.solve,
                timing_.nMetric);
  return rep;
}

}  // namespace leap
