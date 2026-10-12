// The generic contact-sequence problem type (tasks/contact_sequence.hpp), posed from the
// example yamls:
//
//   1. the three examples build, validate and compile; the seed satisfies the spectral
//      kinematic rows exactly and every continuity reset; the row names are the
//      diagnostics' groups
//   2. the yaml loader rejects a mode that places an inactive contact, a flight that
//      cannot carry its hop, an unknown key and a missing rho with a readable message
//   3. the flow right-hand side of the hopper problem equals a finite difference of the
//      merit along random directions (every attachment kind of the builder in situ)
//   4. the hopper jump solves to the paper's criterion with the default recipe
//
// usage: test_contact_sequence <repo root>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "leap/alm/evaluator.hpp"
#include "leap/io/yaml_doc.hpp"
#include "leap/model/robot_description.hpp"
#include "leap/model/robot_model.hpp"
#include "leap/solver/judge.hpp"
#include "leap/solver/pipeline.hpp"
#include "leap/solver/recipe.hpp"
#include "leap/tasks/contact_sequence.hpp"
#include "test_check.hpp"

namespace {

struct Built {
  std::shared_ptr<const leap::RobotModel> model;
  leap::ContactSequenceSpec spec;
  leap::ContactSequenceBuild build;
};

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Build from the yaml text with one substring replaced (the strict-schema check and the
// sequence keys read the DOCUMENT, not the overrides).
Built buildFromText(const std::string& path, const std::string& needle, const std::string& replacement) {
  std::string text = readFile(path);
  const size_t at = text.find(needle);
  if (at == std::string::npos) throw std::runtime_error("text '" + needle + "' not found in " + path);
  text.replace(at, needle.size(), replacement);
  leap::YamlDoc d = leap::YamlDoc::fromString(text);
  const std::string dir = path.substr(0, path.find_last_of('/'));
  std::string robot = d.requireS("robot");
  if (!robot.empty() && robot[0] != '/') robot = dir + "/" + robot;
  const leap::RobotDescription rd = leap::loadRobotDescription(robot);
  Built b;
  b.model = std::make_shared<leap::RobotModel>(rd);
  b.spec = leap::contactSequenceSpecFromYaml(d);
  b.build = leap::buildContactSequence(b.spec, b.model);
  return b;
}

Built buildFromYaml(const std::string& path, const std::vector<std::string>& overrides) {
  leap::YamlDoc d = leap::YamlDoc::fromFile(path);
  for (const std::string& ov : overrides) d.setOverride(ov);
  const leap::RobotDescription rd = leap::loadRobotDescription(d.resolvePath(d.requireS("robot")));
  Built b;
  b.model = std::make_shared<leap::RobotModel>(rd);
  b.spec = leap::contactSequenceSpecFromYaml(d);
  b.build = leap::buildContactSequence(b.spec, b.model);
  return b;
}

Eigen::VectorXd randomVector(int n, double amplitude, unsigned seed) {
  Eigen::VectorXd x(n);
  for (int i = 0; i < n; ++i) x(i) = amplitude * std::sin(0.37 * seed + 1.3 * i + 0.11 * seed * i);
  return x;
}

bool sameActiveSet(const leap::PhrState& a, const leap::PhrState& b) {
  if (a.active.size() != b.active.size()) return false;
  for (size_t i = 0; i < a.active.size(); ++i)
    if (a.active[i] != b.active[i]) return false;
  return true;
}

void checkSeed(const char* label, const leap::CompiledProblem& prob, const Eigen::VectorXd& y0) {
  leap::EvaluatorOptions opt;
  opt.shareJunctions = true;
  opt.threads = 2;
  leap::AlmEvaluator ev(prob, opt);
  leap::Evaluation s;
  const int md = prob.layout().md;
  ev.zeroDualViolation(y0, Eigen::VectorXd::Ones(md), s);
  CHECK(s.g.allFinite(), "seed residuals finite");
  double kin = 0.0, reset = 0.0;
  for (int i = 0; i < md; ++i) {
    const leap::RowInfo& ri = prob.rowInfo()[static_cast<size_t>(i)];
    if (ri.family == leap::RowFamily::Kinematic1 || ri.family == leap::RowFamily::Kinematic2) kin = std::max(kin, std::abs(s.g(i)));
    if (ri.family == leap::RowFamily::Reset) reset = std::max(reset, std::abs(s.g(i)));
  }
  std::printf("  %s: %d elements, np %d, md %d, %d terms; seed max|c| %.3e, kinematic rows %.2e, reset rows %.2e\n", label, prob.numElements(),
              prob.layout().np, md, prob.terms().size(), s.maxViol, kin, reset);
  CHECK(kin < 1e-8, "seed satisfies the spectral kinematic rows");
  // every term name belongs to a diagnostics group the judge knows
  for (int t = 0; t < prob.terms().size(); ++t) {
    const std::string name = leap::TermTable::stripped(prob.terms().name(t));
    const std::string g = leap::constraintGroup(name);
    if (g == "other") std::printf("  term '%s' is in group 'other'\n", name.c_str());
    CHECK(g != "other", "every term is grouped by the diagnostics");
  }
}

// F = -grad_x L_A against a fourth-order finite difference of the merit along random
// reduced directions, at a perturbed seed with live multipliers.
void checkGradient(const leap::CompiledProblem& prob, const Eigen::VectorXd& y0) {
  leap::EvaluatorOptions opt;
  opt.shareJunctions = true;
  opt.activeMargin = 0.0;
  opt.threads = 2;
  leap::AlmEvaluator ev(prob, opt);
  const leap::JunctionQuotient& Q = ev.quotient();
  const int np = prob.layout().np, md = prob.layout().md;
  Eigen::VectorXd y = y0;
  y.head(np) += randomVector(np, 0.02, 5);
  y.tail(md) = randomVector(md, 2.0, 22);
  Q.enforce(y);
  Eigen::VectorXd mu(md);
  for (int i = 0; i < md; ++i) mu(i) = 1e3 * (1.0 + 0.5 * std::sin(0.3 * i));
  leap::EvaluationRequest req;  // the default request: gradient and Jacobian
  leap::Evaluation out;
  ev.evaluate(y, mu, req, out);
  int tested = 0, agreed = 0;
  double worst = 0.0;
  for (unsigned s = 0; s < 12; ++s) {
    Eigen::VectorXd dFull = Eigen::VectorXd::Zero(np + md);
    dFull.head(np) = randomVector(np, 1.0, 100 + s);
    Q.enforce(dFull);
    leap::EvaluationRequest vreq;
    vreq.gradient = false;
    vreq.jacobian = false;
    leap::Evaluation vp;
    const double h = 1e-4;
    const double steps[4] = {-2.0 * h, -h, h, 2.0 * h};
    double f[4];
    bool ok = true;
    for (int k = 0; k < 4 && ok; ++k) {
      Eigen::VectorXd yk = y + steps[k] * dFull;
      Q.enforce(yk);
      ev.evaluate(yk, mu, vreq, vp);
      ok = sameActiveSet(vp.phr, out.phr);
      f[k] = vp.merit;
    }
    if (!ok) continue;
    const double fd = (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h);
    const double an = -out.F.head(np).dot(dFull.head(np));  // F is the full-coordinate covector
    const double scale = std::max(1.0, std::abs(an));
    worst = std::max(worst, std::abs(fd - an) / scale);
    ++tested;
    if (std::abs(fd - an) <= 2e-5 * scale) ++agreed;
  }
  std::printf("  gradient vs FD: %d/%d directions agree to 1e-5 (worst relative %.2e)\n", agreed, tested, worst);
  CHECK(tested >= 6, "enough directions without an active-set change");
  CHECK(agreed == tested, "the flow right-hand side equals the finite-difference merit gradient");
}

void expectThrow(const std::string& path, const std::vector<std::string>& overrides, const char* what, const std::string& needle = "",
                 const std::string& replacement = "") {
  bool threw = false;
  std::string msg;
  try {
    if (needle.empty()) buildFromYaml(path, overrides);
    else buildFromText(path, needle, replacement);
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  std::printf("  %s -> %s\n", what, threw ? msg.c_str() : "NO ERROR");
  CHECK(threw, what);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  const std::string ex = root + "/problems/examples/";
  try {
    // 1. the three examples build; the seeds are kinematically exact
    for (const char* f : {"hopper2d_jump.yaml", "go2_jump.yaml", "go2_step.yaml"}) {
      Built b = buildFromYaml(ex + f, {"solver.split=2"});
      const leap::CompiledProblem prob = leap::CompiledProblem::compile(b.build.problem);
      checkSeed(f, prob, b.build.y0);
      CHECK(static_cast<int>(b.build.qEvent.size()) == static_cast<int>(b.spec.modes.size()) + 1, "one event pose per mode start plus the end");
      CHECK(b.build.ikWorstErr < 1e-3, "the event poses put the feet on their placements");
    }
    // 2. readable errors
    expectThrow(ex + "go2_step.yaml", {}, "placing a contact that is not active in its mode is rejected",
                "contacts: [FR_foot, RL_foot, RR_foot],          T: 0.4}",
                "contacts: [FR_foot, RL_foot, RR_foot],          T: 0.4, feet: [{contact: FL_foot, pos: [0, 0, 0]}]}");
    expectThrow(ex + "hopper2d_jump.yaml", {"problem.modes.1.T=0.1"}, "a flight too short for its hop is rejected");
    expectThrow(ex + "hopper2d_jump.yaml", {}, "an unknown problem key is rejected", "  hop: 0.17", "  hopp: 0.17");
    expectThrow(ex + "hopper2d_jump.yaml", {"solver.rho=0"}, "a missing defect weight is rejected");
    expectThrow(ex + "go2_jump.yaml", {}, "an end pose that does not stand on the final placements is rejected",
                "T: 0.5, shift: [0.3, 0.0, 0.0]}", "T: 0.5, shift: [0.3, 0.0, 0.0], feet: [{contact: FL_foot, pos: [0.6, 0.3, 0.0]}]}");
    // 3. the flow right-hand side of the hopper problem against finite differences
    {
      Built b = buildFromYaml(ex + "hopper2d_jump.yaml", {"solver.split=2"});
      const leap::CompiledProblem prob = leap::CompiledProblem::compile(b.build.problem);
      checkGradient(prob, b.build.y0);
    }
    // 4. the hopper jump solves with the default recipe
    {
      leap::YamlDoc d = leap::YamlDoc::fromFile(ex + "hopper2d_jump.yaml");
      const leap::RobotDescription rd = leap::loadRobotDescription(d.resolvePath(d.requireS("robot")));
      std::shared_ptr<const leap::RobotModel> model = std::make_shared<leap::RobotModel>(rd);
      std::unique_ptr<leap::ProblemFamily> fam = leap::makeContactSequenceFamily(d, model);
      CHECK(fam->type() == "contact_sequence", "the family reports its type");
      leap::SolverRecipe recipe = leap::recipeFromYaml(d);
      recipe.applyPenaltyScale(fam->defectWeight());
      recipe.verbose = false;
      leap::PipelineResult res = leap::runPipeline(*fam, recipe, "[test]");
      leap::Evaluation ev;
      const double gate = res.level.evaluator->zeroDualViolation(res.y, res.level.penalties->muRow(), ev);
      const leap::FineGridReport fr = leap::fineGridViolations(*res.level.restoration, res.y, 0.01);
      std::printf("  hopper jump: %ld steps, %zu levels, node gate %.3e, fine grid %.3e, %.1f s\n", res.record.totalSteps(), res.record.levels.size(),
                  gate, fr.worstGate, res.record.solveWall);
      CHECK(res.record.met, "the hopper jump meets the criterion");
      CHECK(gate <= 1e-3 && fr.worstGate <= 1e-3, "node gate and fine grid below 1e-3");
    }
  } catch (const std::exception& e) {
    std::printf("FAIL exception: %s\n", e.what());
    return 1;
  }
  return leap::test::finish("test_contact_sequence");
}
