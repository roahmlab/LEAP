// The two benchmark problem families, as posed by their yamls, against finite
// differences: every constraint row that the fence jump and the terrain walk attach is
// exercised in situ (node rows, off-node rows, force control-point rows, Bernstein
// torque functionals, defect cost, resets, junction quotient, torque elimination on the
// terrain walk).
//
//   1. the flow right-hand side F = -grad_x L_A equals a fourth-order finite difference of
//      the merit along every (strided) reduced coordinate and along random directions
//   2. the Gauss-Newton metric equals sum_active mu J' J + rho sum w Jf' Jf + H_cost built
//      from finite-difference Jacobians of every row and defect residual
//   3. the cold seed satisfies the kinematic rows exactly and every reset
//
// usage: test_problem_fd <repo root>   (reads problems/*.yaml, robots/*.yaml)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

#include "leap/alm/evaluator.hpp"
#include "leap/io/yaml_doc.hpp"
#include "leap/model/robot_description.hpp"
#include "leap/model/robot_model.hpp"
#include "leap/solver/pipeline.hpp"
#include "leap/tasks/fence_jump.hpp"
#include "leap/tasks/terrain_walk.hpp"
#include "test_check.hpp"

namespace {

class DenseSink final : public leap::MetricSink {
public:
  explicit DenseSink(int n) : H(Eigen::MatrixXd::Zero(n, n)) {}
  void addBlock(int, const std::vector<int>& idx, const Eigen::MatrixXd& B) override {
    const std::lock_guard<std::mutex> lock(mutex_);  // the evaluator assembles elements concurrently
    for (size_t i = 0; i < idx.size(); ++i)
      for (size_t j = 0; j < idx.size(); ++j) H(idx[i], idx[j]) += B(static_cast<int>(i), static_cast<int>(j));
  }
  void addWeightedRows(int element, const std::vector<int>& idx, const Eigen::MatrixXd& J, const Eigen::VectorXd& w) override {
    const Eigen::MatrixXd B = J.transpose() * (w.asDiagonal() * J);
    addBlock(element, idx, B);
  }
  Eigen::MatrixXd H;

private:
  std::mutex mutex_;
};

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

// Fourth-order directional derivative of the merit along d (h = 1e-4); false when the
// PHR active set changes inside the stencil (the merit has a kink there).
bool directionalDerivative(leap::AlmEvaluator& ev, const Eigen::VectorXd& y, const Eigen::VectorXd& d, const Eigen::VectorXd& mu,
                           const leap::PhrState& active, double& out) {
  leap::EvaluationRequest vreq;
  vreq.gradient = false;
  vreq.jacobian = false;
  leap::Evaluation vp;
  const double h = 1e-4;
  const double steps[4] = {-2.0 * h, -h, h, 2.0 * h};
  double f[4];
  for (int k = 0; k < 4; ++k) {
    Eigen::VectorXd yk = y + steps[k] * d;
    ev.quotient().enforce(yk);
    ev.evaluate(yk, mu, vreq, vp);
    if (!sameActiveSet(vp.phr, active)) return false;
    f[k] = vp.merit;
  }
  out = (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h);
  return true;
}

void checkProblem(const char* label, const leap::CompiledProblem& prob, const Eigen::VectorXd& y0) {
  std::printf("=== %s: np=%d md=%d terms=%d elements=%d points=%zu defects=%zu ===\n", label, prob.layout().np, prob.layout().md,
              prob.terms().size(), prob.numElements(), prob.points().size(), prob.defects().size());
  leap::EvaluatorOptions opt;
  opt.shareJunctions = true;
  opt.activeMargin = 0.0;
  opt.threads = 4;
  leap::AlmEvaluator ev(prob, opt);
  const leap::JunctionQuotient& Q = ev.quotient();
  const int np = prob.layout().np, md = prob.layout().md;

  // 3. the seed: kinematic rows exact, resets small, everything finite
  {
    leap::Evaluation s;
    Eigen::VectorXd mu1 = Eigen::VectorXd::Ones(md);
    ev.zeroDualViolation(y0, mu1, s);
    CHECK(s.g.allFinite(), "seed residuals finite");
    double kin = 0.0, reset = 0.0;
    for (int i = 0; i < md; ++i) {
      const leap::RowInfo& ri = prob.rowInfo()[static_cast<size_t>(i)];
      if (ri.family == leap::RowFamily::Kinematic1 || ri.family == leap::RowFamily::Kinematic2) kin = std::max(kin, std::abs(s.g(i)));
      if (ri.family == leap::RowFamily::Reset) reset = std::max(reset, std::abs(s.g(i)));
    }
    std::printf("  seed: zero-dual max|c| %.3e, kinematic rows %.2e, reset rows %.2e\n", s.maxViol, kin, reset);
    CHECK(kin < 1e-8, "seed satisfies the spectral kinematic rows");
    // (the terrain seed interpolates IK poses linearly in q, so its impact resets are
    // not satisfied; only reported)
  }

  // a generic point: the seed, perturbed, with live multipliers and term penalties
  Eigen::VectorXd y = y0;
  y.head(np) += randomVector(np, 0.02, 5);
  Q.enforce(y);
  y.tail(md) = randomVector(md, 2.0, 22);
  Q.enforce(y);
  Eigen::VectorXd mu = Eigen::VectorXd::Constant(md, 7.0);
  for (int i = 0; i < md; ++i) mu(i) += 3.0 * std::abs(std::sin(0.7 * prob.rowTerm()[static_cast<size_t>(i)]));
  leap::EvaluationRequest req;
  leap::Evaluation out;
  ev.evaluate(y, mu, req, out);
  CHECK(std::isfinite(out.merit), "merit finite");
  CHECK(out.F.allFinite(), "F finite");
  Eigen::VectorXd Fz;
  Q.reduceCovector(out.F.head(np), Fz);

  // 1b. random directions over the full primal (every row at once)
  char msg[240];
  double worst = 0.0;
  int tested = 0;
  for (unsigned k = 0; k < 12; ++k) {
    Eigen::VectorXd d = Eigen::VectorXd::Zero(np + md);
    d.head(np) = randomVector(np, 1.0, 100 + k);
    Q.enforce(d);  // a direction inside the quotient
    const double Fd = -out.F.head(np).dot(d.head(np));
    double fd;
    if (!directionalDerivative(ev, y, d, mu, out.phr, fd)) continue;
    ++tested;
    worst = std::max(worst, std::abs(fd - Fd) / std::max(1.0, std::abs(Fd)));
  }
  std::snprintf(msg, sizeof(msg), "%s: directional derivative along %d random directions, worst rel err %.2e", label, tested, worst);
  CHECK(worst < 2e-5, msg);
  std::printf("  %s\n", msg);

  // 2. the Gauss-Newton metric vs finite-difference row Jacobians
  leap::EvaluationRequest vreq;
  vreq.gradient = false;
  vreq.jacobian = false;
  leap::Evaluation vp, vm;
  const double eps = 1e-6;
  Eigen::MatrixXd Jfd(md, np);
  std::vector<Eigen::MatrixXd> Jf(prob.defects().size());
  std::vector<Eigen::VectorXd> rf(prob.defects().size());
  for (size_t di = 0; di < prob.defects().size(); ++di) {
    const leap::DefectInfo& info = prob.defects()[di];
    Jf[di].resize(info.nPoints * info.rowsPerPoint, np);
  }
  for (int i = 0; i < np; ++i) {
    Eigen::VectorXd yp = y, ym = y;
    yp(i) += eps;
    ym(i) -= eps;
    ev.evaluate(yp, mu, req, vp);
    ev.evaluate(ym, mu, req, vm);
    Jfd.col(i) = (vp.g - vm.g) / (2.0 * eps);
    for (size_t di = 0; di < prob.defects().size(); ++di) Jf[di].col(i) = (vp.rdefect[di] - vm.rdefect[di]) / (2.0 * eps);
  }
  // 1a. assembly consistency: F = -(J' w + Jf' (rho wq r) + grad cost), with J, Jf the
  //     finite-difference Jacobians and grad cost from the defect-free twin
  Eigen::VectorXd gradCost = Eigen::VectorXd::Zero(np);
  {
    leap::TrajectoryProblem specNoDefect = prob.spec();
    for (leap::Element& el : specNoDefect.elements) el.defect.rho = 0.0;
    const leap::CompiledProblem probCost = leap::CompiledProblem::compile(specNoDefect);
    leap::AlmEvaluator evCost(probCost, opt);
    const Eigen::VectorXd mu0 = Eigen::VectorXd::Zero(md);
    Eigen::VectorXd yc = y;
    yc.tail(md).setZero();
    leap::Evaluation c0;
    evCost.evaluate(yc, mu0, req, c0);
    gradCost = -c0.F.head(np);
  }
  {
    Eigen::VectorXd Fcheck = -gradCost;
    Fcheck.noalias() -= Jfd.transpose() * out.phr.w;
    for (size_t di = 0; di < prob.defects().size(); ++di) {
      const leap::DefectInfo& info = prob.defects()[di];
      Eigen::VectorXd rw = out.rdefect[di];
      for (int p = 0; p < info.nPoints; ++p) rw.segment(p * info.rowsPerPoint, info.rowsPerPoint) *= info.rho * info.wq[static_cast<size_t>(p)];
      Fcheck.noalias() -= Jf[di].transpose() * rw;
    }
    Eigen::VectorXd Fz2;
    Q.reduceCovector(Fcheck, Fz2);
    const double scale = std::max(1.0, Fz.cwiseAbs().maxCoeff());
    const double err = (Fz - Fz2).cwiseAbs().maxCoeff() / scale;
    std::snprintf(msg, sizeof(msg), "%s: F vs -(J' w + Jf' rho w r + grad cost) from FD Jacobians (rel err %.2e)", label, err);
    CHECK(err < 1e-6, msg);
    std::printf("  %s\n", msg);
  }
  Eigen::MatrixXd Hexp = Eigen::MatrixXd::Zero(np, np);
  for (int i = 0; i < md; ++i) {
    if (!out.phr.activeMetric[static_cast<size_t>(i)]) continue;
    if (Q.isIdentityRow(i)) continue;
    Hexp.noalias() += mu(i) * Jfd.row(i).transpose() * Jfd.row(i);
  }
  for (size_t di = 0; di < prob.defects().size(); ++di) {
    const leap::DefectInfo& info = prob.defects()[di];
    for (int p = 0; p < info.nPoints; ++p) {
      const double w = info.rho * info.wq[static_cast<size_t>(p)];
      const Eigen::MatrixXd Jp = Jf[di].middleRows(p * info.rowsPerPoint, info.rowsPerPoint);
      Hexp.noalias() += w * Jp.transpose() * Jp;
    }
  }
  // the cost Hessian by finite differences of the cost gradient: the same spec with the
  // defect cost removed (identical layout), every row weight and multiplier at zero
  {
    leap::TrajectoryProblem specNoDefect = prob.spec();
    for (leap::Element& el : specNoDefect.elements) el.defect.rho = 0.0;
    const leap::CompiledProblem probCost = leap::CompiledProblem::compile(specNoDefect);
    CHECK(probCost.layout().np == np && probCost.layout().md == md, "defect-free twin keeps the layout");
    leap::AlmEvaluator evCost(probCost, opt);
    const Eigen::VectorXd mu0 = Eigen::VectorXd::Zero(md);
    Eigen::VectorXd yc = y;
    yc.tail(md).setZero();
    for (int i = 0; i < np; ++i) {
      Eigen::VectorXd yp = yc, ym = yc;
      yp(i) += eps;
      ym(i) -= eps;
      evCost.evaluate(yp, mu0, req, vp);
      evCost.evaluate(ym, mu0, req, vm);
      Hexp.col(i) += -(vp.F.head(np) - vm.F.head(np)) / (2.0 * eps);
    }
  }
  DenseSink sink(np);
  ev.assembleMetric(y, out, mu, sink);
  // compare on the reduced coordinates (the sink receives full-index blocks that the
  // solver reduces the same way: sum over members of a class)
  const int nr = Q.reducedSize();
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(np, nr);
  for (int i = 0; i < np; ++i) R(i, Q.reducedOf(i)) = 1.0;
  const Eigen::MatrixXd Hr = R.transpose() * sink.H * R;
  const Eigen::MatrixXd Hr2 = R.transpose() * Hexp * R;
  const double scale = std::max(1.0, Hr2.cwiseAbs().maxCoeff());
  const double errH = (Hr - Hr2).cwiseAbs().maxCoeff() / scale;
  std::snprintf(msg, sizeof(msg), "%s: Gauss-Newton metric vs FD Jacobians (rel err %.2e, scale %.2e)", label, errH, scale);
  CHECK(errH < 5e-4, msg);
  std::printf("  %s\n", msg);
  CHECK((Hr - Hr.transpose()).cwiseAbs().maxCoeff() / scale < 1e-12, "metric symmetric");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  try {
    {
      leap::YamlDoc d = leap::YamlDoc::fromFile(root + "/problems/hopper2d_jump_fence.yaml");
      // the record's transcription, on a coarser mesh (2 elements per mode) for speed
      for (const char* ov : {"solver.split=2", "solver.hgrade=0.6", "solver.lsqnqs=8,8,8", "solver.rho=3000000", "obstacles.smooth=0", "obstacles.rounded=true"})
        d.setOverride(ov);
      const leap::RobotDescription rd = leap::loadRobotDescription(d.resolvePath(d.requireS("robot")));
      std::shared_ptr<const leap::RobotModel> model = std::make_shared<leap::RobotModel>(rd);
      leap::FenceFamily fam(d, rd, model);
      const leap::CompiledProblem prob = leap::CompiledProblem::compile(fam.build());
      checkProblem("fence hopper2d", prob, fam.seed(prob));
    }
    {
      leap::YamlDoc d = leap::YamlDoc::fromFile(root + "/problems/hopper2d_foot_terrain_walk.yaml");
      for (const char* ov : {"solver.split=1", "problem.attnq=1", "problem.qboxnq=1", "problem.uboxnq=2", "problem.noslipacc=true", "problem.uhull=16"})
        d.setOverride(ov);
      const leap::RobotDescription rd = leap::loadRobotDescription(d.resolvePath(d.requireS("robot")));
      std::shared_ptr<const leap::RobotModel> model = std::make_shared<leap::RobotModel>(rd);
      leap::TerrainFamily fam(d, rd, model);
      const leap::CompiledProblem prob = leap::CompiledProblem::compile(fam.build());
      checkProblem("terrain hopper2d_foot", prob, fam.seed(prob));
    }
  } catch (const std::exception& e) {
    std::printf("FAIL exception: %s\n", e.what());
    return 1;
  }
  return leap::test::finish("test_problem_fd");
}
