// The augmented-Lagrangian layer against finite differences, on a hand-built problem
// for the planar hopper: two stance elements (the second with torque elimination) and
// one flight element, continuity and plastic-impact resets, node terms of every family,
// off-node rows, force control-point rows, torque Bernstein functionals, a defect cost,
// and the effort cost.
//
//   1. the flow right-hand side F = -grad_x L_A equals the central difference of the merit
//   2. the Gauss-Newton metric equals H_cost + sum_active mu J' J + rho sum w Jf' Jf built
//      from a finite-difference Jacobian of every row
//   3. the block-Schur solve inverts M = I + h H (reduced coordinates, quotient on and off)
//
// usage: test_alm_fd <repo root>

#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

#include "leap/alm/evaluator.hpp"
#include "leap/linalg/block_schur.hpp"
#include "leap/linalg/global_metric.hpp"
#include "leap/problem/compiled_problem.hpp"
#include "leap/problem/costs.hpp"
#include "leap/problem/obstacles.hpp"
#include "leap/problem/self_collision.hpp"
#include "hopper_test_problem.hpp"
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

void checkProblem(const leap::CompiledProblem& prob, bool shareJunctions) {
  std::printf("--- quotient %s: np=%d md=%d terms=%d nodes=%d points=%zu defects=%zu ---\n",
              shareJunctions ? "on" : "off", prob.layout().np, prob.layout().md, prob.terms().size(),
              prob.numNodesTotal(), prob.points().size(), prob.defects().size());
  leap::EvaluatorOptions opt;
  opt.shareJunctions = shareJunctions;
  opt.activeMargin = 0.0;
  opt.threads = 2;
  leap::AlmEvaluator ev(prob, opt);
  const int np = prob.layout().np, md = prob.layout().md;
  Eigen::VectorXd y = randomVector(np + md, 0.3, 21);
  // a plausible state: base height and modest forces
  for (int e = 0; e < prob.numElements(); ++e) {
    const leap::ElementLayout& L = prob.layout().elem[static_cast<size_t>(e)];
    leap::ElementViewMutable X(y.data(), L);
    X.Q.row(2).array() += 0.6;
    if (L.Nc > 0) X.C.row(2).array() += 60.0;
  }
  ev.quotient().enforce(y);
  y.tail(md) = randomVector(md, 2.0, 22);  // live multipliers (identity rows re-zeroed below)
  ev.quotient().enforce(y);
  // one penalty per TERM (the metric relies on it), varying across terms
  Eigen::VectorXd mu = Eigen::VectorXd::Constant(md, 7.0);
  for (int i = 0; i < md; ++i) mu(i) += 3.0 * std::abs(std::sin(0.7 * prob.rowTerm()[static_cast<size_t>(i)]));

  leap::EvaluationRequest req;
  leap::Evaluation out;
  ev.evaluate(y, mu, req, out);
  CHECK(std::isfinite(out.merit), "merit finite");
  CHECK(out.F.allFinite(), "F finite");

  // 1. F = -grad_x L_A by central differences of the merit (reduced through the quotient
  //    when it is on: only representative coordinates are free, members follow)
  const leap::JunctionQuotient& Q = ev.quotient();
  leap::EvaluationRequest vreq;
  vreq.gradient = false;
  vreq.jacobian = false;
  leap::Evaluation vp, vm;
  Eigen::VectorXd Fz;
  Q.reduceCovector(out.F.head(np), Fz);
  const double eps = 1e-6;
  double worst = 0.0;
  for (int r = 0; r < Q.reducedSize(); ++r) {
    const int i = Q.representative(r);
    // fourth-order stencil with a 1e-4 step: the merit is ~1e6, so a 1e-6 central
    // difference would be dominated by round-off
    const double h = 1e-4;
    double f[4];
    const double steps[4] = {-2.0 * h, -h, h, 2.0 * h};
    for (int k = 0; k < 4; ++k) {
      Eigen::VectorXd yk = y;
      yk(i) += steps[k];
      Q.enforce(yk);
      ev.evaluate(yk, mu, vreq, vp);
      f[k] = vp.merit;
    }
    const double fd = -(f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h);
    const double scale = std::max(1.0, std::abs(Fz(r)));
    const double err = std::abs(fd - Fz(r)) / scale;
    if (err > worst) {
      worst = err;
      std::printf("  F worst so far at reduced %d (full %d): F %.6e fd %.6e merit %.3e\n", r, i, Fz(r), fd, out.merit);
    }
  }
  char msg[200];
  std::snprintf(msg, sizeof(msg), "flow rhs F vs finite-difference merit gradient (worst rel err %.2e)", worst);
  CHECK(worst < 2e-5, msg);

  // 2. the Gauss-Newton metric against a finite-difference Jacobian of every row
  Eigen::MatrixXd Jfd(md, np);
  for (int i = 0; i < np; ++i) {
    Eigen::VectorXd yp = y, ym = y;
    yp(i) += eps;
    ym(i) -= eps;
    ev.evaluate(yp, mu, vreq, vp);
    ev.evaluate(ym, mu, vreq, vm);
    Jfd.col(i) = (vp.g - vm.g) / (2.0 * eps);
  }
  // the defect residual Jacobian and cost Hessian by FD too
  Eigen::MatrixXd Hexp = Eigen::MatrixXd::Zero(np, np);
  for (int i = 0; i < md; ++i) {
    if (!out.phr.activeMetric[static_cast<size_t>(i)]) continue;
    if (shareJunctions && Q.isIdentityRow(i)) continue;
    Hexp.noalias() += mu(i) * Jfd.row(i).transpose() * Jfd.row(i);
  }
  for (size_t di = 0; di < prob.defects().size(); ++di) {
    const leap::DefectInfo& info = prob.defects()[di];
    const int rows = info.nPoints * info.rowsPerPoint;
    Eigen::MatrixXd Jf(rows, np);
    for (int i = 0; i < np; ++i) {
      Eigen::VectorXd yp = y, ym = y;
      yp(i) += eps;
      ym(i) -= eps;
      ev.evaluate(yp, mu, vreq, vp);
      ev.evaluate(ym, mu, vreq, vm);
      Jf.col(i) = (vp.rdefect[di] - vm.rdefect[di]) / (2.0 * eps);
    }
    for (int p = 0; p < info.nPoints; ++p)
      Hexp.noalias() += info.rho * info.wq[static_cast<size_t>(p)] *
                        Jf.middleRows(p * info.rowsPerPoint, info.rowsPerPoint).transpose() *
                        Jf.middleRows(p * info.rowsPerPoint, info.rowsPerPoint);
  }
  // cost Hessian: FD of the cost gradient (grad f = -F at zero multipliers and zero penalties
  // is not separable, so take the second difference of the cost value)
  {
    Eigen::VectorXd y0 = y;
    for (int i = 0; i < np; ++i) {
      for (int j = i; j < np; ++j) {
        // only the u blocks carry cost curvature; skip the rest cheaply
        bool uCoord = false;
        for (int e = 0; e < prob.numElements() && !uCoord; ++e) {
          const leap::ElementLayout& L = prob.layout().elem[static_cast<size_t>(e)];
          if (L.uStored && i >= L.x0 + L.offU && i < L.x0 + L.offU + L.dims.nu * L.Nn &&
              j >= L.x0 + L.offU && j < L.x0 + L.offU + L.dims.nu * L.Nn)
            uCoord = true;
        }
        if (!uCoord) continue;
        if (i == j) Hexp(i, i) += 1.0;  // 1/2||u||^2 on stored u
      }
    }
  }
  // element 1 has the effort cost on DERIVED u: its curvature is S_u' S_u; add from the lift
  for (size_t ni = 0; ni < prob.nodes().size(); ++ni) {
    const leap::NodeInfo& info = prob.nodes()[ni];
    const leap::ElementLayout& L = prob.layout().elem[static_cast<size_t>(info.elem)];
    if (L.uStored) continue;
    const Eigen::MatrixXd Su = out.Snode[ni].middleRows(L.dims.offU(), L.dims.nu);
    const Eigen::MatrixXd Hb = Su.transpose() * Su;
    for (size_t a = 0; a < info.zIdx.size(); ++a)
      for (size_t b = 0; b < info.zIdx.size(); ++b) Hexp(info.zIdx[a], info.zIdx[b]) += Hb(static_cast<int>(a), static_cast<int>(b));
  }
  DenseSink sink(np);
  ev.assembleMetric(y, out, mu, sink);
  // compare in reduced coordinates (E' H E)
  Eigen::MatrixXd Hz = Eigen::MatrixXd::Zero(Q.reducedSize(), Q.reducedSize());
  Eigen::MatrixXd Hzexp = Hz;
  for (int i = 0; i < np; ++i)
    for (int j = 0; j < np; ++j) {
      Hz(Q.reducedOf(i), Q.reducedOf(j)) += sink.H(i, j);
      Hzexp(Q.reducedOf(i), Q.reducedOf(j)) += Hexp(i, j);
    }
  const double hscale = std::max(1.0, Hzexp.cwiseAbs().maxCoeff());
  const double herr = (Hz - Hzexp).cwiseAbs().maxCoeff() / hscale;
  {
    Eigen::Index bi, bj;
    (Hz - Hzexp).cwiseAbs().maxCoeff(&bi, &bj);
    const int fi = Q.representative(static_cast<int>(bi)), fj = Q.representative(static_cast<int>(bj));
    std::printf("  worst metric entry at reduced (%ld, %ld) = full (%d, %d): assembled %.6e expected %.6e (scale %.3e)\n",
                static_cast<long>(bi), static_cast<long>(bj), fi, fj, Hz(bi, bj), Hzexp(bi, bj), hscale);
    for (int e = 0; e < prob.numElements(); ++e) {
      const leap::ElementLayout& L = prob.layout().elem[static_cast<size_t>(e)];
      for (int f : {fi, fj}) {
        if (f < L.x0 || f >= L.x0 + L.nx) continue;
        const int o = f - L.x0;
        const char* field = o < L.offV ? "Q" : (o < L.offA ? "V" : (L.offLam >= 0 && o >= L.offLam && (L.offU < 0 || o < L.offU) ? "C" : (L.offU >= 0 && o >= L.offU ? "U" : "A")));
        std::printf("    full %d -> element %d field %s local %d\n", f, e, field, o);
      }
    }
  }
  std::snprintf(msg, sizeof(msg), "Gauss-Newton metric vs FD Jacobians (rel err %.2e)", herr);
  CHECK(herr < 2e-5, msg);
  CHECK((Hz - Hz.transpose()).cwiseAbs().maxCoeff() < 1e-9 * hscale, "metric symmetric");

  // 3. the block-Schur solve
  leap::LinearSolverOptions lopt;
  leap::BlockSchurSolver solver(prob, Q, lopt);
  solver.beginAssembly();
  ev.assembleMetric(y, out, mu, solver);
  const double hTheta = 0.05;
  CHECK(solver.factor(hTheta), "block-Schur factorization succeeds");
  Eigen::VectorXd rhs = randomVector(Q.reducedSize(), 1.0, 31), dz;
  CHECK(solver.solve(rhs, dz), "block-Schur solve succeeds");
  const Eigen::MatrixXd Mz = Eigen::MatrixXd::Identity(Q.reducedSize(), Q.reducedSize()) + hTheta * Hz;
  const Eigen::VectorXd dzDense = Mz.ldlt().solve(rhs);
  const double serr = (dz - dzDense).norm() / std::max(1.0, dzDense.norm());
  std::snprintf(msg, sizeof(msg), "block-Schur solve vs dense solve (rel err %.2e, separators %d)", serr, solver.numSeparators());
  CHECK(serr < 1e-9, msg);
  CHECK(solver.residualNorm(dz, rhs) < 1e-8 * std::max(1.0, rhs.norm()), "block-Schur residual");

  // 4. the global backends on the same metric
  for (leap::LinearBackend b : {leap::LinearBackend::SparseLLT, leap::LinearBackend::SparseLDLT, leap::LinearBackend::Dense,
                                leap::LinearBackend::SparseCholmod}) {
    if (!leap::linearBackendAvailable(b)) continue;
    leap::GlobalMetricSolver gs(prob, Q, b);
    gs.beginAssembly();
    ev.assembleMetric(y, out, mu, gs);
    Eigen::VectorXd dzg;
    const bool okf = gs.factor(hTheta);
    const bool oks = okf && gs.solve(rhs, dzg);
    const double gerr = oks ? (dzg - dzDense).norm() / std::max(1.0, dzDense.norm()) : 1.0;
    std::snprintf(msg, sizeof(msg), "backend %s solve vs dense solve (rel err %.2e)", leap::linearBackendName(b), gerr);
    CHECK(okf && oks && gerr < 1e-9, msg);
    if (oks) CHECK(gs.residualNorm(dzg, rhs) < 1e-8 * std::max(1.0, rhs.norm()), "global backend residual");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <repo root>\n", argv[0]);
    return 2;
  }
  const std::string robots = std::string(argv[1]) + "/robots/";
  auto model = std::make_shared<leap::RobotModel>(leap::loadRobotDescription(robots + "hopper2d.yaml"));
  const leap::TrajectoryProblem spec = leap::test::buildHopperTestProblem(model);
  const leap::CompiledProblem prob = leap::CompiledProblem::compile(spec);
  checkProblem(prob, false);
  checkProblem(prob, true);
  return leap::test::finish("test_alm_fd");
}
