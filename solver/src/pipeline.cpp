#include "leap/solver/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "leap/solver/relocation.hpp"
#include "leap/solver/objective_weights.hpp"
#include "leap/solver/snapshots.hpp"

namespace leap {

namespace {

double wallSince(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool startsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

// Zero-dual max|c| of y on the restoration rows (values only).
double posedViolation(Level& L, const Eigen::VectorXd& y, Evaluation& c) {
  return L.evaluator->zeroDualViolation(y, L.penalties->muRow(), c);
}

// Zero-dual violation of the obstacle family (-1 when the problem has none).
double obstacleFamilyViolation(Level& L, const Eigen::VectorXd& y) {
  Evaluation c;
  posedViolation(L, y, c);
  const CompiledProblem& prob = *L.restoration;
  double obs = -1.0;
  for (int t = 0; t < prob.terms().size(); ++t) {
    if (!startsWith(TermTable::stripped(prob.terms().name(t)), "ineq_obstacle")) continue;
    obs = std::max(obs, c.perTermViol(t));
  }
  return obs;
}

// The level stop predicate consulted at every phase-C window boundary.
class LevelStopCheck final : public WindowStopCheck {
public:
  LevelStopCheck(Level& L, const SolverRecipe& r) : L_(&L), r_(&r) {}
  // True when the last `true` was the judged criterion (gate and fine grid <= tol), as
  // opposed to a hand-over to the next mesh level.
  bool judged() const { return judged_; }
  bool stop(const Eigen::VectorXd& y, const BoundaryInfo& bi) override {
    judged_ = false;
    const double tol = r_->tol;
    if (bi.maxViol > 4.0 * r_->finefraccap * tol) return false;  // cheap pre-filter
    const double gate = L_->evaluator->zeroDualViolation(y, L_->penalties->muRow(), cache_);
    if (gate > tol) {
      if (gate <= r_->finefraccap * tol) {
        const FineGridReport fr = fineGridViolations(*L_->restoration, y, r_->finedt);
        const double fine = r_->finegate ? fr.worstGate : fr.worst;
        if (gate <= r_->finefrac * fine) {
          std::printf("  [flow] stop: node gate %.3e <= %.2g x fine-grid %.3e (%s elem %d t=%.3f): the mesh owns the remaining error -> refine\n",
                      gate, r_->finefrac, fine, fr.worstGateName.c_str(), fr.worstGateElem, fr.worstGateT);
          return true;
        }
      }
      return false;
    }
    const FineGridReport fr = fineGridViolations(*L_->restoration, y, r_->finedt);
    const double fine = r_->finegate ? fr.worstGate : fr.worst;
    if (fine > tol) {
      if (gate <= r_->finefrac * fine) {
        std::printf("  [flow] stop: gate %.3e holds and is <= %.2g x fine-grid %.3e (%s elem %d t=%.3f): hand over\n", gate,
                    r_->finefrac, fine, fr.worstGateName.c_str(), fr.worstGateElem, fr.worstGateT);
        return true;
      }
      if (r_->fineplateau > 0) {
        if (fine < 0.99 * bestFine_) {
          bestFine_ = fine;
          finePlateau_ = 0;
        } else if (++finePlateau_ >= r_->fineplateau) {
          std::printf("  [flow] stop: gate %.3e holds; fine-grid %.3e is a plateaued transcription floor (%d probes)\n", gate, fine,
                      r_->fineplateau);
          return true;
        }
      }
      std::printf("  [flow] gate %.3e holds; fine-grid %.3e > %.3e (%s elem %d t=%.3f): continuing\n", gate, fine, tol,
                  fr.worstGateName.c_str(), fr.worstGateElem, fr.worstGateT);
      return false;
    }
    judged_ = true;
    return true;  // judged
  }

private:
  Level* L_;
  const SolverRecipe* r_;
  Evaluation cache_;
  double bestFine_ = std::numeric_limits<double>::infinity();
  int finePlateau_ = 0;
  bool judged_ = false;
};

// The objective mode's stop: the level stop (feasibility / hand-over) AND, on a judged
// mesh with objstop = stationary, the first-order certificate (relative reduced gradient
// of L_A and relative complementarity <= kkttol) plus an objective plateau (relative
// change <= objtol over objwindows boundaries).
class ObjectiveStopCheck final : public WindowStopCheck {
public:
  ObjectiveStopCheck(Level& L, const SolverRecipe& r) : base_(L, r), r_(&r) {}
  bool stationary() const { return stationary_; }
  bool stop(const Eigen::VectorXd& y, const BoundaryInfo& bi) override {
    const double f = bi.cost;
    const double rel = std::isfinite(fPrev_) ? std::abs(f - fPrev_) / std::max(1.0, std::abs(f)) : std::numeric_limits<double>::infinity();
    plateau_ = rel <= r_->objtol ? plateau_ + 1 : 0;
    fPrev_ = f;
    const bool feasible = base_.stop(y, bi);
    const bool firstOrder = r_->kkttol > 0.0 && bi.firstOrder.gradientRel <= r_->kkttol && bi.firstOrder.complementarityRel <= r_->kkttol;
    std::printf("  [flow] objective: boundary f=%.9e relchange=%.3e plateau=%d/%d kktrel=%.3e comprel=%.3e feasible=%d judged=%d firstorder=%d\n",
                f, rel, plateau_, r_->objwindows - 1, bi.firstOrder.gradientRel, bi.firstOrder.complementarityRel, feasible ? 1 : 0,
                base_.judged() ? 1 : 0, firstOrder ? 1 : 0);
    stationary_ = false;
    if (!feasible) return false;
    if (!base_.judged()) return true;  // hand-over to the next mesh level, not a stationary success
    if (r_->objstop == "feasibility") return true;
    if (!firstOrder) return false;
    if (plateau_ < r_->objwindows - 1) return false;
    std::printf("  [flow] stop: first-order certificate -- feasibility, reduced Lagrangian gradient %.3e, complementarity %.3e, objective plateau over %d boundaries\n",
                bi.firstOrder.gradientRel, bi.firstOrder.complementarityRel, r_->objwindows);
    stationary_ = true;
    return true;
  }

private:
  LevelStopCheck base_;
  const SolverRecipe* r_;
  double fPrev_ = std::numeric_limits<double>::quiet_NaN();
  int plateau_ = 0;
  bool stationary_ = false;
};

FlowConfig sharedFlowConfig(const SolverRecipe& r) {
  FlowConfig c;
  c.backend = parseLinearBackend(r.backend);
  c.cholmod = r.cholmod;
  c.exactPositiveDefinite = r.exactpd;
  c.entryMean = r.junctionmean;
  c.theta = 1.0;
  c.h0 = r.h0;
  c.hMin = r.hmin;
  c.hMax = r.hmax;
  c.hShrink = r.hshrink;
  c.ratioGrow = r.hgrow;
  c.ratioLow = r.ratiolow;
  c.ratioHigh = r.ratiohigh;
  c.dyMaxRel = r.dymaxrel;
  c.solveRetries = r.solveretry;
  c.window = r.windowLength();
  c.tolFlow = r.tolflow;
  c.stallRel = r.stallrel;
  c.hCollapseWindows = r.hcollapse;
  c.blowupFactor = r.blowup;
  c.blowupRetries = r.blowupretry;
  c.momentum = !r.nomomentum;
  c.meritStallRel = r.meritstallrel;
  c.verbose = r.verbose;
  return c;
}

void printJudgement(const Judgement& J, double tol) {
  std::printf("[hp]   element indicators:");
  for (size_t e = 0; e < J.elements.size(); ++e)
    std::printf(" %zu:%.1e%s", e, J.elements[e].ind, J.elements[e].ind > tol ? "*" : "");
  std::printf("\n");
}

std::vector<ElementSplit> markElements(const Judgement& J, const SolverRecipe& r) {
  double maxInd = 0.0;
  for (const ElementIndicator& e : J.elements) maxInd = std::max(maxInd, e.ind);
  std::vector<std::pair<double, int>> marked;
  for (size_t e = 0; e < J.elements.size(); ++e)
    if (J.elements[e].ind > r.tol && J.elements[e].ind >= r.theta * maxInd)
      marked.emplace_back(J.elements[e].ind, static_cast<int>(e));
  std::sort(marked.begin(), marked.end());
  std::reverse(marked.begin(), marked.end());
  if (static_cast<int>(marked.size()) > r.maxsplits) marked.resize(static_cast<size_t>(r.maxsplits));
  std::vector<ElementSplit> splits;
  for (const std::pair<double, int>& m : marked) {
    const ElementIndicator& e = J.elements[static_cast<size_t>(m.second)];
    ElementSplit s;
    s.element = m.second;
    s.fraction = 0.5;
    if (e.fine >= e.node && e.fine > 0.0) s.fraction = std::clamp(0.5 * (e.fineTau + 1.0), r.splitlo, r.splithi);
    splits.push_back(s);
  }
  return splits;
}

std::string stopText(const FlowReport& rep) {
  char b[160];
  std::snprintf(b, sizeof(b), "%s (steps %ld, s %.3f, max|c| %.3e)", rep.stop.c_str(), rep.steps, rep.s, rep.maxViol);
  return b;
}

// Feeds one stage's accepted steps to the snapshot writer. The gate the writer wants is
// the POSED zero-dual node violation: phase C flows on the posed rows (the raw gate of
// the flow is it); the rungs and a smooth-certificate A0 flow on a variant problem, so
// the posed rows are evaluated once more per step (values only).
class StageObserver final : public StepObserver {
public:
  StageObserver(SnapshotWriter* w, Level& L, int level, const char* stage, bool posed)
      : w_(w), L_(&L), level_(level), stage_(stage), posed_(posed) {}
  void onAcceptedStep(const Eigen::VectorXd& y, double rawGate, long) override {
    if (w_ == nullptr) return;
    w_->onStep(*L_, level_, stage_, y, posed_ ? w_->posedGate(*L_, y) : rawGate);
  }

private:
  SnapshotWriter* w_;
  Level* L_;
  int level_;
  const char* stage_;
  bool posed_;
};

// The step controller's counters of one flow run (the reference tree's "adaptive:" line).
void printFlowCounters(const std::string& tag, const char* stage, const FlowReport& rep) {
  std::printf("%s [%s] adaptive: steps=%ld grows=%ld shrinks=%ld restarts=%ld backstops=%ld solvefails=%ld blowups=%ld\n",
              tag.c_str(), stage, rep.steps, rep.grows, rep.shrinks, rep.restarts, rep.backstops, rep.solveFails,
              rep.blowupRecoveries);
}

}  // namespace

Eigen::VectorXd ProblemFamily::oracleTermScales(const CompiledProblem& prob) const {
  (void)prob;
  return Eigen::VectorXd();
}

std::vector<double> ProblemFamily::objectiveNodeWeights(const CompiledProblem& prob) const {
  return leap::objectiveNodeWeights(prob, "clenshaw_curtis", 1.0);
}

TrajectoryProblem restorationTwin(const TrajectoryProblem& posed) {
  TrajectoryProblem twin = posed;
  for (Element& e : twin.elements) e.costs.clear();
  return twin;
}

EvaluatorOptions evaluatorOptions(const SolverRecipe& r) {
  EvaluatorOptions o;
  o.gaussNewton = r.gn;
  o.activeMargin = r.activemargin;
  o.ineqViolMax0 = r.ineqviol;
  o.shareJunctions = r.sharejunction;
  o.threads = r.threads;
  return o;
}

Level buildLevel(ProblemFamily& family, const SolverRecipe& r) {
  Level L;
  L.posedSpec = family.build();
  L.posed = std::make_unique<CompiledProblem>(CompiledProblem::compile(L.posedSpec));
  L.restoration = std::make_unique<CompiledProblem>(CompiledProblem::compile(restorationTwin(L.posedSpec)));
  if (L.posed->layout().np != L.restoration->layout().np || L.posed->layout().md != L.restoration->layout().md)
    throw std::runtime_error("buildLevel: the restoration twin's layout differs from the posed problem");
  if (r.objective == "posed") {
    if (L.posed->terms().size() != L.restoration->terms().size())
      throw std::runtime_error("buildLevel: the posed problem's terms differ from the restoration twin's");
    L.posed->setNodeCostWeights(family.objectiveNodeWeights(*L.posed));
  }
  L.evaluator = std::make_unique<AlmEvaluator>(*L.restoration, evaluatorOptions(r));
  L.posedEvaluator = std::make_unique<AlmEvaluator>(*L.posed, evaluatorOptions(r));
  PenaltyConfig pc;
  pc.policy = PenaltyPolicy::Stall;
  pc.mu0 = r.muc;
  pc.muMax = r.mumaxc;
  pc.growth = r.mugrow;
  pc.stallRatio = 0.25;
  L.penalties = std::make_unique<Penalties>(*L.restoration, pc);
  L.termScales = family.oracleTermScales(*L.restoration);
  return L;
}

// The evaluator options of the pure-feasibility LM stages (rungs, A0): Gauss-Newton
// unless the recipe asks for the exact metric everywhere.
EvaluatorOptions feasibilityStageOptions(const SolverRecipe& r) {
  EvaluatorOptions o = evaluatorOptions(r);
  if (!r.gn && r.gna0) o.gaussNewton = true;
  return o;
}

FlowConfig phaseCConfig(const SolverRecipe& r, int level) {
  FlowConfig c = sharedFlowConfig(r);
  c.h0 = level == 0 ? r.h0 : r.h0level;
  c.hMax = level == 0 ? r.hmax : r.hmaxlevel;
  c.sMax = r.csmax;
  c.maxSteps = (level == 0 || r.cstepslevel <= 0) ? r.csteps : r.cstepslevel;
  c.maxStepsPerWindow = r.cwindowsteps;
  c.stallWindows = r.stall;
  c.keepBest = true;
  c.dualUpdate = true;
  c.dualStep = r.dualstep;
  c.resetAfterDualUpdate = r.dualreset;
  c.feasStop = 0.0;
  if (r.objective == "posed") c.objectiveProgressRel = r.objtol;
  return c;
}

FlowConfig phaseA0Config(const SolverRecipe& r) {
  FlowConfig c = sharedFlowConfig(r);
  c.window = r.a0window > 0.0 ? r.a0window : c.window;
  c.sMax = r.a0smax;
  c.maxSteps = r.a0steps;
  c.stallWindows = r.a0stall;
  c.keepBest = r.a0keepbest;
  c.feasStop = r.a0gate;
  c.dualUpdate = false;
  c.momentum = !r.nomomentum && r.a0momentum;
  c.hCollapseWindows = 0;
  return c;
}

FlowConfig rungConfig(const SolverRecipe& r) {
  FlowConfig c = sharedFlowConfig(r);
  c.sMax = r.presmax;
  c.maxSteps = r.presteps;
  c.stallWindows = r.prestall;
  c.keepBest = true;
  c.feasStop = r.pregate;
  c.dualUpdate = true;
  c.dualStep = 1.0;
  c.momentum = false;
  return c;
}

namespace {

void runObstacleRungs(ProblemFamily& fam, Level& L, const SolverRecipe& r, const std::string& tag, Eigen::VectorXd& y,
                      SolveRecord& rec, SnapshotWriter* snap) {
  const auto t0 = std::chrono::steady_clock::now();
  const int md = L.restoration->layout().md;
  std::printf("%s [pre] obstacle-family continuation over %zu rung(s) (vertical scale", tag.c_str(), r.preladder.size());
  for (double s : r.preladder) std::printf(" %g", s);
  const double rungMuMax = r.premumax > 0.0 ? r.premumax : std::min(30.0 * r.premu, r.mumaxc);
  std::printf("): gate %.1e, smax %g, %d steps/rung, mu %.3g -> %.3g\n", r.pregate, r.presmax, r.presteps, r.premu, rungMuMax);
  Evaluation judgeCache;
  for (double sc : r.preladder) {
    ObstacleVariant v;
    v.zScale = sc;
    v.smoothEps = r.presmooth;
    const TrajectoryProblem variantSpec = restorationTwin(fam.obstacleVariant(v));
    CompiledProblem prob = CompiledProblem::compile(variantSpec);
    if (prob.layout().n() != L.restoration->layout().n() || prob.layout().md != L.restoration->layout().md)
      throw std::runtime_error("[pre] rung layout differs from the posed problem");
    AlmEvaluator ev(prob, feasibilityStageOptions(r));
    PenaltyConfig pc;
    pc.policy = PenaltyPolicy::Stall;
    pc.mu0 = r.premu;
    pc.muMax = rungMuMax;
    pc.growth = r.mugrow;
    Penalties pen(prob, pc);
    FlowConfig cfg = rungConfig(r);
    char hb[160];
    std::snprintf(hb, sizeof(hb), "%s [pre] rung scale %g", tag.c_str(), sc);
    cfg.header = hb;
    Eigen::VectorXd yy = y;
    yy.tail(md).setZero();
    StageObserver obs(snap, L, 0, "pre", /*posed=*/true);
    cfg.observer = snap != nullptr ? &obs : nullptr;
    FlowDriver flow(ev, pen, cfg, nullptr);
    const FlowReport rep = flow.run(yy);
    if (r.verbose) printFlowCounters(tag, "pre", rep);
    rec.preSteps += rep.steps;
    const bool keep = !rep.diverged && yy.allFinite();
    if (keep) y = yy;
    rec.preViol = posedViolation(L, y, judgeCache);
    std::printf("%s [pre] rung scale %g: %s%s; POSED zero-dual max|c| %.3e\n", tag.c_str(), sc, stopText(rep).c_str(),
                keep ? "" : " DIVERGED (iterate dropped)", rec.preViol);
    printViolationsByGroup(*L.restoration, judgeCache);
  }
  y.tail(md).setZero();
  rec.preWall += wallSince(t0);
  std::printf("%s [pre] done: %zu rung(s), %ld steps, posed max|c| %.3e, %.1f s\n", tag.c_str(), r.preladder.size(), rec.preSteps,
              rec.preViol, rec.preWall);
}

void runPreStage(ProblemFamily& fam, Level& L, const SolverRecipe& r, const std::string& tag, Eigen::VectorXd& y, SolveRecord& rec,
                 SnapshotWriter* snap) {
  if (r.pre == "none" || r.preladder.empty()) return;
  if (r.pre == "auto") {
    const double obs = obstacleFamilyViolation(L, y);
    std::printf("%s [pre] auto: seed's obstacle-family violation %.3e%s\n", tag.c_str(), obs,
                obs <= r.pregate ? ": rungs skipped" : ": running the rungs");
    if (obs <= r.pregate) return;
  }
  runObstacleRungs(fam, L, r, tag, y, rec, snap);
}

void runA0(ProblemFamily& fam, Level& L, const SolverRecipe& r, const std::string& tag, Eigen::VectorXd& y, SolveRecord& rec,
           SnapshotWriter* snap) {
  if (!r.a0) return;
  const auto t0 = std::chrono::steady_clock::now();
  const int md = L.restoration->layout().md;
  std::unique_ptr<CompiledProblem> variant;
  std::unique_ptr<AlmEvaluator> variantEv;
  AlmEvaluator* ev = L.evaluator.get();
  const CompiledProblem* prob = L.restoration.get();
  if (r.a0smooth > 0.0) {
    ObstacleVariant v;
    v.smoothEps = r.a0smooth;
    variant = std::make_unique<CompiledProblem>(CompiledProblem::compile(restorationTwin(fam.obstacleVariant(v))));
    if (variant->layout().n() != L.restoration->layout().n() || variant->layout().md != L.restoration->layout().md)
      throw std::runtime_error("[A0] smooth-certificate problem layout differs from the posed one");
    variantEv = std::make_unique<AlmEvaluator>(*variant, feasibilityStageOptions(r));
    ev = variantEv.get();
    prob = variant.get();
    std::printf("%s [A0] on the SMOOTH cuboid certificate (temperature %g); phase C on the posed one\n", tag.c_str(), r.a0smooth);
  } else if (!r.gn && r.gna0) {
    variantEv = std::make_unique<AlmEvaluator>(*L.restoration, feasibilityStageOptions(r));
    ev = variantEv.get();
  }
  PenaltyConfig pc;
  pc.policy = PenaltyPolicy::Constant;
  pc.mu0 = r.a0mu;
  pc.muMax = r.a0mu;
  Penalties pen0(*prob, pc);
  FlowConfig cfg = phaseA0Config(r);
  char hb[200];
  std::snprintf(hb, sizeof(hb), "%s phase A0: pure-feasibility LM flow (w=%g, feasStop=%g, s_max=%g)", tag.c_str(), r.a0mu, r.a0gate,
                cfg.sMax);
  cfg.header = hb;
  y.tail(md).setZero();
  StageObserver obs(snap, L, 0, "A0", /*posed=*/ev != L.evaluator.get());
  cfg.observer = snap != nullptr ? &obs : nullptr;
  FlowDriver flow(*ev, pen0, cfg, nullptr);
  const FlowReport rep = flow.run(y);
  if (r.verbose) printFlowCounters(tag, "A0", rep);
  y.tail(md).setZero();
  rec.a0Steps += rep.steps;
  rec.a0Viol = rep.maxViol;
  rec.a0Wall += wallSince(t0);
  std::printf("%s phase A0 done in %.1f s: %s\n", tag.c_str(), rec.a0Wall, stopText(rep).c_str());
}

// One phase C call: the level's opening call (reseed = true) or a relocation round
// (reseed = !mucarry). By default every call re-seeds every term's penalty at muc and
// clears the stall history.
FlowReport runPhaseC(Level& L, const SolverRecipe& r, int level, const std::string& tag, Eigen::VectorXd& y, double sMax,
                     int maxSteps, SolveRecord& rec, bool reseed, SnapshotWriter* snap) {
  FlowConfig cfg = phaseCConfig(r, level);
  if (sMax > 0.0) cfg.sMax = sMax;
  if (maxSteps > 0) cfg.maxSteps = maxSteps;
  StageObserver obs(snap, L, level, "C", /*posed=*/false);
  cfg.observer = snap != nullptr ? &obs : nullptr;
  const bool objective = r.objective == "posed";
  char hb[256];
  std::snprintf(hb, sizeof(hb), "%s phase C (level %d): frozen-dual LM windows (%.3g) + multiplier updates (tau %.2g), mu %.3g -> %.3g%s",
                tag.c_str(), level, cfg.window, cfg.dualStep, r.muc, r.mumaxc,
                objective ? ", on the POSED objective (first-order stop)" : "");
  cfg.header = hb;
  if (reseed) L.penalties->seed(r.muc);
  if (!objective) {
    LevelStopCheck stop(L, r);
    FlowDriver flow(*L.evaluator, *L.penalties, cfg, &stop);
    FlowReport rep = flow.run(y);
    if (r.verbose) printFlowCounters(tag, "C", rep);
    return rep;
  }
  ObjectiveStopCheck stop(L, r);
  FlowDriver flow(*L.posedEvaluator, *L.penalties, cfg, &stop);
  FlowReport rep = flow.run(y);
  if (r.verbose) printFlowCounters(tag, "C", rep);
  rec.objective = rep.objective;
  rec.kktRel = rep.kktRel;
  rec.complementarityRel = rep.complementarityRel;
  rec.objectiveStationary = stop.stationary();
  return rep;
}

}  // namespace

PipelineResult runPipeline(ProblemFamily& fam, const SolverRecipe& r, const std::string& tag, SnapshotWriter* snap) {
  const auto tSolve = std::chrono::steady_clock::now();
  PipelineResult out;
  SolveRecord& rec = out.record;
  Level cur = buildLevel(fam, r);
  Eigen::VectorXd y = fam.seed(*cur.posed);
  if (y.size() != cur.restoration->layout().n()) throw std::runtime_error("runPipeline: the seed has the wrong size");
  if (snap != nullptr) snap->begin(cur, y);
  std::printf("%s level 0: %d elements, np=%d md=%d, %d terms\n", tag.c_str(), cur.elements(), cur.restoration->layout().np,
              cur.restoration->layout().md, cur.restoration->terms().size());
  {
    Evaluation c0;
    const double v0 = posedViolation(cur, y, c0);
    std::printf("%s seed: zero-dual max|c| %.3e\n", tag.c_str(), v0);
    printViolationsByGroup(*cur.restoration, c0);
  }
  runPreStage(fam, cur, r, tag, y, rec, snap);
  runA0(fam, cur, r, tag, y, rec, snap);
  {
    Evaluation c0;
    const double v0 = posedViolation(cur, y, c0);
    std::printf("%s handoff to phase C: zero-dual max|c| %.3e (pre %ld steps, A0 %ld steps)\n", tag.c_str(), v0, rec.preSteps, rec.a0Steps);
  }

  double lastHandoffGate = std::numeric_limits<double>::infinity();
  int finalLevel = 0;
  for (int level = 0;; ++level) {
    finalLevel = level;
    const auto tLevel = std::chrono::steady_clock::now();
    LevelRecord lr;
    lr.level = level;
    lr.elements = cur.elements();
    {
      const Judgement J0 = judgeLevel(*cur.evaluator, *cur.penalties, y, r.finedt, r.finegate);
      lr.handoffGate = J0.gate;
      lr.handoffFine = J0.fine;
      std::printf("[hp] level %d start (%d elements, np %d, md %d): node gate %.3e, fine grid %.3e (%s elem %d)\n", level, lr.elements,
                  cur.restoration->layout().np, cur.restoration->layout().md, J0.gate, J0.fine, J0.fineName.c_str(), J0.fineElem);
    }
    const std::string lvTag = tag + (level > 0 ? " hp" + std::to_string(level) : "");
    FlowReport fr = runPhaseC(cur, r, level, lvTag, y, 0.0, 0, rec, /*reseed=*/true, snap);
    lr.cSteps += fr.steps;
    rec.lastFlow = fr;
    Judgement J = judgeLevel(*cur.evaluator, *cur.penalties, y, r.finedt, r.finegate);
    for (int round = 1; r.relocate && round <= r.rounds; ++round) {
      if (J.met(r.tol)) break;
      if (!fineGridWorstReachable(*cur.restoration, J.fineName, J.fineElem, J.fineTau, J.fine, r.reloctau, r.relocceil * r.tol, r.relocbox)) {
        std::printf("[hp] level %d round %d: fine-grid worst %s (elem %d, tau %+.2f, %.3e) is out of the relocation's reach%s\n", level,
                    round, J.fineName.c_str(), J.fineElem, J.fineTau, J.fine, J.gate > r.tol ? " (node gate still open)" : ": refining");
        break;
      }
      RelocationOptions ro;
      ro.nScan = r.relocscan;
      ro.minSep = r.relocminsep;
      ro.coverSep = r.reloccover;
      ro.moveTol = r.relocmovetol;
      ro.minScaled = r.relocfloor * r.tol;
      ro.maxScaled = r.relocceil * r.tol;
      ro.maxMoves = r.relocmax;
      ro.moveBoxRows = r.relocbox;
      ro.movableTau = r.reloctau;
      const RelocationReport rel = relocateOffNodePoints(*cur.restoration, y, ro, cur.termScales, false);
      // mirror the moved points into the posed twin (identical point order by construction)
      for (int i = 0; i < static_cast<int>(cur.restoration->points().size()); ++i) {
        const PointInfo& info = cur.restoration->points()[static_cast<size_t>(i)];
        if (info.kind != PointKind::OffNode || info.taus.empty()) continue;
        cur.posed->relocateOffNodeTaus(i, info.taus);
      }
      std::printf("[hp] level %d round %d: %d off-node point(s) moved (of %d row sets); worst between-node %s = %.3e (scaled %.3e) elem %d tau %+.2f\n",
                  level, round, rel.moved, rel.scanned, rel.worstName.c_str(), rel.worstRaw, rel.worstScaled, rel.worstElem, rel.worstTau);
      lr.moved += rel.moved;
      if (rel.moved == 0) break;
      ++lr.rounds;
      fr = runPhaseC(cur, r, level, lvTag + " r" + std::to_string(round), y, r.roundsmax, r.roundsteps, rec, /*reseed=*/!r.mucarry, snap);
      lr.cSteps += fr.steps;
      rec.lastFlow = fr;
      J = judgeLevel(*cur.evaluator, *cur.penalties, y, r.finedt, r.finegate);
    }
    lr.gate = J.gate;
    lr.fine = J.fine;
    lr.wall = wallSince(tLevel);
    std::printf("[hp] level %d done: %ld steps, node gate %.3e, fine grid %.3e (%s elem %d t=%.3f), %d round(s), %.1f s\n", level, lr.cSteps,
                J.gate, J.fine, J.fineName.c_str(), J.fineElem, J.fineT, lr.rounds, lr.wall);
    printJudgement(J, r.tol);
    if (J.met(r.tol)) {
      std::printf("[hp] criterion met: max(node gate, fine grid) = %.3e <= %.1e\n", std::max(J.gate, J.fine), r.tol);
      lr.stop = "met";
      rec.levels.push_back(lr);
      rec.met = true;
      rec.stop = "criterion met";
      break;
    }
    if (r.abortfail && level > 0 && J.gate > r.abortfactor * r.tol && J.gate > lastHandoffGate / r.abortimprove) {
      std::printf("[hp] LEVEL FAILED ALGEBRAICALLY: node gate %.3e (handoff %.3e): not a discretization error; stopping the refinement\n",
                  J.gate, lastHandoffGate);
      lr.stop = "algebraic failure";
      rec.levels.push_back(lr);
      rec.stop = "level failed algebraically";
      break;
    }
    if (level >= r.levels) {
      std::printf("[hp] level budget exhausted (%d refinement rounds)\n", r.levels);
      lr.stop = "level budget";
      rec.levels.push_back(lr);
      rec.stop = "level budget exhausted";
      break;
    }
    const std::vector<ElementSplit> splits = markElements(J, r);
    if (splits.empty()) {
      std::printf("[hp] nothing to mark (gate %.3e, fine %.3e): the remaining violation is not element-local\n", J.gate, J.fine);
      lr.stop = "nothing to mark";
      rec.levels.push_back(lr);
      rec.stop = "nothing to mark";
      break;
    }
    std::printf("[hp] level %d: bisecting %zu element(s):", level, splits.size());
    for (const ElementSplit& s : splits) {
      std::printf(" %d@%.2f", s.element, s.fraction);
      lr.marked.push_back(s.element);
    }
    std::printf("\n");
    lr.stop = "refine";
    rec.levels.push_back(lr);
    if (snap != nullptr) snap->flush(cur, level, "refine");  // the incumbent's vector lives on this mesh
    fam.refine(splits);
    Level nxt = buildLevel(fam, r);
    Eigen::VectorXd yNew = Eigen::VectorXd::Zero(nxt.restoration->layout().n());
    const double perr = fam.prolong(*cur.posed, y, *nxt.posed, yNew);
    yNew.tail(nxt.restoration->layout().md).setZero();
    std::printf("[hp] level %d -> %d: %d -> %d elements, prolongation error %.1e\n", level, level + 1, cur.elements(), nxt.elements(), perr);
    lastHandoffGate = J.gate;
    cur = std::move(nxt);
    y = std::move(yNew);
  }
  if (snap != nullptr) snap->finish(cur, finalLevel, y, snap->posedGate(cur, y));
  rec.solveWall = wallSince(tSolve);
  out.level = std::move(cur);
  out.y = std::move(y);
  return out;
}

}  // namespace leap
