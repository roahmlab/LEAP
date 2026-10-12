// leap_solve: one gradient-flow solver for every problem family this codebase
// transcribes (problem.type contact_sequence | jump_fence_v2 | terrain_walk). The problem yaml describes
// the task (robot, physics, transcription); the METHOD is the `unified:` block
// (solver/recipe.hpp), whose defaults are the paper's recipe.
//
//   usage: leap_solve <problem.yaml> [dotted.key=value ...]
//   e.g.:  OMP_NUM_THREADS=12 ./build/leap_solve problems/go2_jump_fence.yaml
//              solver.split=8 solver.hgrade=0.6 output.stem=results/sol_fence_go2
//
// Output: the solve log on stdout (the summary line starts with "[unified]", one
// "[unified-level]" line per refinement level) and the archive <output.stem>.h5
// (docs/solution_format.md). Exit codes: 0 ok, 2 configuration / robot / yaml error,
// 3 the problem is infeasible by construction.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "leap/io/solution_archive.hpp"
#include "leap/io/yaml_doc.hpp"
#include "leap/model/robot_description.hpp"
#include "leap/model/robot_model.hpp"
#include "leap/solver/judge.hpp"
#include "leap/solver/snapshots.hpp"
#include "leap/solver/pipeline.hpp"
#include "leap/solver/recipe.hpp"
#include "leap/spectral/cgl_grid.hpp"
#include "leap/tasks/contact_sequence.hpp"
#include "leap/tasks/fence_jump.hpp"
#include "leap/tasks/objective.hpp"
#include "leap/tasks/terrain_walk.hpp"

namespace {

int ompThreads() {
  const char* omp = std::getenv("OMP_NUM_THREADS");
  return omp ? std::atoi(omp) : 0;
}

double wallSince(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The accepted output keys; the ones this driver does not implement must stay at their
// off value.
void checkOutputKeys(const leap::YamlDoc& d) {
  d.checkKeys("output", {"stem", "archive", "internode_trace", "full_internode", "finegrid_dt", "finegrid_dt2",
                         "finegrid_csv", "judgeonly", "initial_guess", "initial_guess_only", "step_trace", "snapshots",
                         "snapshot_perdecade", "snapshot_tmin", "snapshot_tmax", "snapshot_seed"});
  if (d.getB("output.judgeonly", false)) throw std::runtime_error("output.judgeonly is not supported");
  if (!d.getS("output.initial_guess", "").empty() || d.getB("output.initial_guess_only", false))
    throw std::runtime_error("output.initial_guess export is not supported");
  if (!d.getS("output.step_trace", "").empty()) throw std::runtime_error("output.step_trace is not supported");
  if (!d.getS("output.finegrid_csv", "").empty()) throw std::runtime_error("output.finegrid_csv is not supported");
  d.checkKeys("input", {"initial_guess"});
  if (!d.getS("input.initial_guess", "").empty()) throw std::runtime_error("input.initial_guess is not supported (cold start only)");
}

std::unique_ptr<leap::ProblemFamily> makeFamily(const leap::YamlDoc& d, const leap::RobotDescription& rd,
                                                std::shared_ptr<const leap::RobotModel> model) {
  const std::string type = d.getS("problem.type", "");
  if (type == "contact_sequence") return leap::makeContactSequenceFamily(d, std::move(model));
  if (type == "jump_fence_v2") return std::make_unique<leap::FenceFamily>(d, rd, std::move(model));
  if (type == "terrain_walk") return std::make_unique<leap::TerrainFamily>(d, rd, std::move(model));
  throw std::runtime_error("problem.type must be contact_sequence, jump_fence_v2 or terrain_walk (got '" + type + "')");
}

int solve(const std::string& yamlPath, const std::vector<std::string>& overrides) {
  const auto tProcess = std::chrono::steady_clock::now();
  leap::YamlDoc d = leap::YamlDoc::fromFile(yamlPath);
  for (const std::string& ov : overrides) d.setOverride(ov);
  d.checkKeys("", {"schema", "name", "robot", "problem", "solver", "output", "input", "obstacles", "terrain", "unified", "objective"});
  checkOutputKeys(d);
  if (d.getI("schema", 1) != 1) throw std::runtime_error("schema: only version 1 is understood");
  const std::string name = d.getS("name", "unnamed");
  const std::string robotYaml = d.resolvePath(d.requireS("robot"));
  const leap::RobotDescription rd = leap::loadRobotDescription(robotYaml);
  std::shared_ptr<const leap::RobotModel> model = std::make_shared<leap::RobotModel>(rd);
  leap::SolverRecipe recipe = leap::recipeFromYaml(d);
  Eigen::setNbThreads(1);

  std::unique_ptr<leap::ProblemFamily> fam = makeFamily(d, rd, model);
  recipe.applyPenaltyScale(fam->defectWeight());
  const std::string type = fam->type();
  for (const std::string& w : fam->warnings()) std::printf("[leap_solve|%s] warning: %s\n", name.c_str(), w.c_str());
  const std::string tag = "[leap_solve|" + name + "]";
  std::printf("%s type=%s robot=%s threads=%d\n", tag.c_str(), type.c_str(), rd.name.c_str(), ompThreads());
  const std::string objSummary = leap::objectiveSummary(leap::objectiveFromYaml(d));
  std::printf("  recipe: %s\n  problem: %s\n  objective: %s (%s)\n", recipe.fingerprint().c_str(), fam->summary().c_str(), objSummary.c_str(),
              recipe.objective == "posed" ? "minimized in phase C" : "reported only");

  // the archive metadata (the final archive and every budget snapshot share it)
  leap::ArchiveRunInfo info;
  info.generator = "leap_solve";
  info.problemName = name;
  info.problemType = type;
  info.problemYamlText = d.sourceText();
  info.robotYamlText = rd.sourceText;
  info.overrides = d.overrides();
  info.solverFingerprint = recipe.fingerprint();
  info.configHashHex = leap::fnv1aHex(info.solverFingerprint);
  info.urdfPath = rd.urdfPath;
  info.motorModel = model->motorModelType();
  info.ompThreads = ompThreads();
  info.warnings = fam->warnings();

  // output.snapshots=<dir>: the budget-study candidates (solver/snapshots.hpp)
  std::unique_ptr<leap::SnapshotWriter> snapshots;
  if (!d.getS("output.snapshots", "").empty()) {
    leap::SnapshotOptions so;
    so.dir = d.getS("output.snapshots", "");
    so.perDecade = d.getD("output.snapshot_perdecade", so.perDecade);
    so.tMin = d.getD("output.snapshot_tmin", so.tMin);
    so.tMax = d.getD("output.snapshot_tmax", so.tMax);
    so.seed = d.getB("output.snapshot_seed", so.seed);
    snapshots = std::make_unique<leap::SnapshotWriter>(so, info);
  }

  leap::PipelineResult res = leap::runPipeline(*fam, recipe, tag, snapshots.get());
  leap::Level& L = res.level;
  const Eigen::VectorXd& y = res.y;
  const leap::SolveRecord& rec = res.record;

  // --- the judges: zero-dual node gate, fine grids ---
  const leap::CompiledProblem& prob = *L.restoration;
  leap::Evaluation ev;
  const double gate = L.evaluator->zeroDualViolation(y, L.penalties->muRow(), ev);
  leap::printViolationsByGroup(prob, ev);
  std::printf("  gate (zero-dual) max|c| = %.3e\n", gate);
  const double fgDt = d.getD("output.finegrid_dt", recipe.finedt);
  const leap::FineGridReport fr = leap::fineGridViolations(prob, y, fgDt);
  leap::printFineGrid(fr, 400);
  std::printf("[finegrid] dt=%g samples=%d worst=%.6e name=%s seg=%d t=%.4f worstGate=%.6e gateName=%s\n", fr.dt, fr.nSamples, fr.worst,
              fr.worstName.c_str(), fr.worstElem, fr.worstT, fr.worstGate, fr.worstGateName.c_str());
  for (const leap::FineGridTerm& t : fr.terms)
    std::printf("[finegrid] term=%s group=%s sense=%s worst=%.6e seg=%d t=%.4f row=%d cp=%d partial=%d\n", t.name.c_str(),
                leap::constraintGroup(leap::TermTable::stripped(t.name)).c_str(), t.ineq ? "ineq" : "eq", t.worst, t.elem, t.t, t.row,
                t.controlPoint ? 1 : 0, t.partialCoverage ? 1 : 0);
  const double fineGate = recipe.finegate ? fr.worstGate : fr.worst;
  double fine2 = 0.0;
  const double fgDt2 = d.getD("output.finegrid_dt2", 0.0);
  if (fgDt2 > 0.0) {
    const leap::FineGridReport fr2 = leap::fineGridViolations(prob, y, fgDt2);
    fine2 = recipe.finegate ? fr2.worstGate : fr2.worst;
    std::printf("[finegrid2] dt=%g samples=%d worst=%.6e worstGate=%.6e gateName=%s\n", fr2.dt, fr2.nSamples, fr2.worst, fr2.worstGate,
                fr2.worstGateName.c_str());
  }
  // the control effort int_0^T ||u||^2 dt (Clenshaw-Curtis on the nodal torques, the
  // inverse-dynamics torque where it is eliminated): reported, not minimized
  double effort = 0.0;
  {
    leap::ModelDataPtr data = model->makeData();
    for (int e = 0; e < prob.numElements(); ++e) {
      const leap::ElementLayout& el = prob.layout().elem[static_cast<size_t>(e)];
      const leap::ElementView X(y.data(), el);
      const Eigen::VectorXd cc = leap::clenshawCurtisWeights(prob.grid(e).degree());
      const Eigen::MatrixXd lam = el.Nc > 0 ? prob.nodeForces(y, e) : Eigen::MatrixXd(0, el.Nn);
      for (int k = 0; k < el.Nn; ++k) {
        const Eigen::VectorXd u = el.uStored ? Eigen::VectorXd(X.U.col(k))
                                             : model->inverseDynamicsTorque(*data, X.Q.col(k), X.V.col(k), X.A.col(k), lam.col(k),
                                                                            prob.spec().elements[static_cast<size_t>(e)].contacts);
        effort += 0.5 * el.T * cc(k) * u.squaredNorm();
      }
    }
    if (recipe.objective == "posed")
      std::printf("  control effort int_0^T ||u||^2 dt = %.6e (Clenshaw-Curtis); posed objective f = %.9e (%s), reduced-gradient %.3e, complementarity %.3e, stationary stop %d\n",
                  effort, rec.objective, objSummary.c_str(), rec.kktRel, rec.complementarityRel, rec.objectiveStationary ? 1 : 0);
    else
      std::printf("  objective int_0^T ||u||^2 dt = %.6e (Clenshaw-Curtis) [feasibility mode: reported, not minimized]\n", effort);
  }
  const bool met = gate <= recipe.tol && fineGate <= recipe.tol;
  const double processWall = wallSince(tProcess);

  // --- the one-line summary the suite tools parse ---
  std::printf("[unified] robot=%s type=%s pipeline=unified met=%d gate=%.6e fine=%.6e fineall=%.6e fine2=%.6e finename=%s pre=%ld a0=%ld levels=%zu csteps=",
              rd.name.c_str(), type.c_str(), met ? 1 : 0, gate, fineGate, fr.worst, fine2, fr.worstGateName.c_str(), rec.preSteps, rec.a0Steps,
              rec.levels.size());
  for (size_t i = 0; i < rec.levels.size(); ++i) std::printf("%s%ld", i ? "+" : "", rec.levels[i].cSteps);
  std::printf(" elements=");
  for (size_t i = 0; i < rec.levels.size(); ++i) std::printf("%s%d", i ? ">" : "", rec.levels[i].elements);
  std::printf(" rounds=");
  for (size_t i = 0; i < rec.levels.size(); ++i) std::printf("%s%d", i ? "," : "", rec.levels[i].rounds);
  std::printf(" total=%ld solvewall=%.1f prewall=%.1f a0wall=%.1f objective=%s effort=%.6e", rec.totalSteps(), rec.solveWall, rec.preWall,
              rec.a0Wall, recipe.objective.c_str(), effort);
  if (recipe.objective == "posed")
    std::printf(" objective_value=%.9e kktrel=%.6e comprel=%.6e stationary=%d", rec.objective, rec.kktRel, rec.complementarityRel,
                rec.objectiveStationary ? 1 : 0);
  std::printf(" stop=\"%s\"\n", rec.stop.c_str());
  for (const leap::LevelRecord& lr : rec.levels)
    std::printf("[unified-level] level=%d elements=%d steps=%ld rounds=%d moved=%d handoffgate=%.3e handofffine=%.3e gate=%.3e fine=%.3e wall=%.1f stop=\"%s\"\n",
                lr.level, lr.elements, lr.cSteps, lr.rounds, lr.moved, lr.handoffGate, lr.handoffFine, lr.gate, lr.fine, lr.wall, lr.stop.c_str());
  std::printf("%s SOLVE WALL %.1f s (solver) / %.1f s (process, judges included)\n", tag.c_str(), rec.solveWall, processWall);

  // --- archive ---
  const std::string stem = d.getS("output.stem", "sol_" + name);
  if (d.getB("output.archive", true)) {
    info.wallS = rec.solveWall;
    info.interNodeDynMax = fr.worstDyn;
    info.lastFlow = &rec.lastFlow;
    const std::string h5 = stem + ".h5";
    leap::writeSolutionArchive(prob, ev, y, L.penalties->muRow(), info, h5);
    std::printf("solution archive written to %s\n", h5.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 2 || argv[1][0] == '-') {
    std::fprintf(stderr, "usage: %s <problem.yaml> [dotted.key=value ...]\n", argv[0]);
    return 2;
  }
  try {
    std::vector<std::string> overrides;
    for (int i = 2; i < argc; ++i) overrides.emplace_back(argv[i]);
    return solve(argv[1], overrides);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "leap_solve: problem infeasible by construction: %s\n", e.what());
    return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "leap_solve: %s\n", e.what());
    return 2;
  }
}
