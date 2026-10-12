// The exact contracted Hessians against finite differences.
//
//   1. PinocchioHessians kernels, term by term, on hopper2d, hopper2d_foot, go2 and h1:
//      sum_i w_i d2 g_i / dz2 of the dynamics residual (point contacts and flat-foot
//      wrenches), of the frame position / velocity / acceleration / orientation rows, of
//      the monitor heights, of the impact cone and of the plastic impact map, versus
//      central differences of the analytic Jacobian's transpose contracted with w. The
//      dynamics, velocity and acceleration kernels are also compared with the CppAD tapes
//      recorded over pinocchio (an independent derivation of the same numbers).
//   2. In situ: the exact-Newton metric of three posed problems (the hand-built hopper
//      problem with torque elimination, impact cone and plastic reset; the fence hopper2d
//      with 2 elements per mode; the terrain hopper2d_foot with a flat foot and torque
//      elimination) against the directional finite difference of the flow right-hand side
//      H d = -(F(y + h d) - F(y - h d)) / 2h, with live multipliers on EVERY row.
//
// usage: test_hessian_fd <repo root>

#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

#include "hopper_test_problem.hpp"
#include "leap/alm/evaluator.hpp"
#include "leap/io/yaml_doc.hpp"
#include "leap/model/model_hessians.hpp"
#include "leap/model/robot_description.hpp"
#include "leap/model/robot_model.hpp"
#include "leap/problem/constraints.hpp"
#include "leap/problem/reset_map.hpp"
#include "leap/tasks/fence_jump.hpp"
#include "leap/tasks/terrain_walk.hpp"
#include "test_check.hpp"

namespace {

Eigen::VectorXd randomVector(int n, double amplitude, unsigned seed) {
  Eigen::VectorXd x(n);
  for (int i = 0; i < n; ++i) x(i) = amplitude * std::sin(0.37 * seed + 1.3 * i + 0.11 * seed * i);
  return x;
}

leap::NodeQuantities viewOf(const Eigen::VectorXd& z, const leap::NodeDims& d) {
  return leap::NodeQuantities{z.segment(d.offQ(), d.nq), z.segment(d.offV(), d.nv), z.segment(d.offA(), d.na),
                              z.segment(d.offLam(), d.nlam), z.segment(d.offU(), d.nu)};
}

double relErr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double& scale) {
  scale = std::max(1.0, B.cwiseAbs().maxCoeff());
  return (A - B).cwiseAbs().maxCoeff() / scale;
}

// J(z)' w by the analytic Jacobian of `con` at z.
Eigen::VectorXd jtw(const leap::RobotModel& m, leap::ModelData& data, leap::ModelEvalCache& cache, const leap::Constraint& con,
                    const leap::ContactSet& cs, const leap::NodeDims& d, const Eigen::VectorXd& z, const Eigen::VectorXd& w) {
  const leap::NodeQuantities zq = viewOf(z, d);
  m.evalNode(data, zq.q, zq.v, zq.a, zq.lam, cs, con.modelNeeds() | leap::kFootPos, cache);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(con.rows(d), d.nP());
  con.jacobian(zq, d, cache, J);
  return J.transpose() * w;
}

Eigen::MatrixXd fdHessian(const leap::RobotModel& m, leap::ModelData& data, leap::ModelEvalCache& cache, const leap::Constraint& con,
                          const leap::ContactSet& cs, const leap::NodeDims& d, const Eigen::VectorXd& z, const Eigen::VectorXd& w) {
  const double eps = 1e-6;
  Eigen::MatrixXd Hfd(d.nP(), d.nP());
  for (int i = 0; i < d.nP(); ++i) {
    Eigen::VectorXd zp = z, zm = z;
    zp(i) += eps;
    zm(i) -= eps;
    Hfd.col(i) = (jtw(m, data, cache, con, cs, d, zp, w) - jtw(m, data, cache, con, cs, d, zm, w)) / (2.0 * eps);
  }
  return Hfd;
}

void checkKernel(const char* label, const leap::RobotModel& m, leap::PinocchioHessians& hess, const leap::Constraint& con,
                 const leap::ContactSet& cs, const Eigen::VectorXd& z, double tol) {
  const leap::NodeDims d = m.nodeDims(cs);
  leap::ModelDataPtr data = m.makeData();
  leap::ModelEvalCache cache;
  m.initCache(cache);
  const int rows = con.rows(d);
  const Eigen::VectorXd w = randomVector(rows, 1.0, 7);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d.nP(), d.nP());
  leap::HessianScratch hs;
  hs.model = &m;
  hs.data = data.get();
  hs.hessians = &hess;
  con.addContractedHessian(viewOf(z, d), d, hs, w, H);
  const Eigen::MatrixXd Hfd = fdHessian(m, *data, cache, con, cs, d, z, w);
  double scale;
  const double err = relErr(H, Hfd, scale);
  char msg[200];
  std::snprintf(msg, sizeof(msg), "%s: contracted Hessian vs FD (rel err %.2e, scale %.2e)", label, err, scale);
  CHECK(err < tol, msg);
  CHECK((H - H.transpose()).cwiseAbs().maxCoeff() < 1e-9 * scale, "contracted Hessian symmetric");
  std::printf("  %s\n", msg);
}

// The analytic kernel against the CppAD tape of the same quantity.
void checkAgainstTape(const char* label, const leap::RobotModel& m, leap::PinocchioHessians& hess, leap::FrameTerm term, int contact,
                      const leap::ContactSet& cs, const Eigen::VectorXd& z) {
  const leap::NodeDims d = m.nodeDims(cs);
  const int rows = term == leap::FrameTerm::Position ? std::min(m.contactDim(contact), 3) : m.contactDim(contact);
  const Eigen::VectorXd w = randomVector(rows, 1.0, 11);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d.nP(), d.nP()), Ht = H;
  hess.addFrameHessian(term, contact, d, z, w, H);
  hess.addFrameHessianTaped(term, contact, d, z, w, Ht);
  double scale;
  const double err = relErr(H, Ht, scale);
  char msg[200];
  std::snprintf(msg, sizeof(msg), "%s: analytic kernel vs CppAD tape (rel err %.2e)", label, err);
  CHECK(err < 1e-10, msg);
  std::printf("  %s\n", msg);
}

void checkDynamicsAgainstTape(const leap::RobotModel& m, leap::PinocchioHessians& hess, const leap::ContactSet& cs,
                              const Eigen::VectorXd& z) {
  const leap::NodeDims d = m.nodeDims(cs);
  const Eigen::VectorXd w = randomVector(d.nv, 1.0, 13);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d.nP(), d.nP()), Ht = H;
  hess.addDynamicsHessian(d, cs, z, w, H);
  hess.addDynamicsHessianTaped(d, cs, z, w, Ht);
  double scale;
  const double err = relErr(H, Ht, scale);
  char msg[200];
  std::snprintf(msg, sizeof(msg), "dyn: analytic kernel vs CppAD tape (rel err %.2e)", err);
  CHECK(err < 1e-9, msg);
  std::printf("  %s\n", msg);
}

// The plastic impact reset's contracted Hessian over [z_tail; z_head] vs FD of its Jacobian.
void checkReset(const leap::RobotModel& m, leap::PinocchioHessians& hess, const leap::ContactSet& cs, const Eigen::VectorXd& zTail,
                double tol) {
  const int setId = m.ensureImpactSet(cs);
  const leap::PlasticImpactReset reset(setId, m.nq(), m.nv());
  const leap::ContactSet air;
  const leap::NodeDims dT = m.nodeDims(air), dH = m.nodeDims(cs);  // tail airborne, head in contact
  leap::ModelDataPtr data = m.makeData();
  Eigen::VectorXd zt = zTail.head(dT.nP());
  const Eigen::VectorXd zh = randomVector(dH.nP(), 0.3, 17);
  const int rows = reset.rows();
  const Eigen::VectorXd w = randomVector(rows, 1.0, 19);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dT.nP() + dH.nP(), dT.nP() + dH.nP());
  leap::HessianScratch hs;
  hs.model = &m;
  hs.data = data.get();
  hs.hessians = &hess;
  reset.addContractedHessian(viewOf(zt, dT), viewOf(zh, dH), dT, dH, hs, w, H);
  Eigen::MatrixXd Hfd(dT.nP() + dH.nP(), dT.nP() + dH.nP());
  const double eps = 1e-6;
  for (int i = 0; i < dT.nP() + dH.nP(); ++i) {
    Eigen::VectorXd xp(dT.nP() + dH.nP()), xm(dT.nP() + dH.nP());
    xp << zt, zh;
    xm = xp;
    xp(i) += eps;
    xm(i) -= eps;
    Eigen::MatrixXd Jp = Eigen::MatrixXd::Zero(rows, dT.nP() + dH.nP()), Jm = Jp;
    reset.jacobian(viewOf(xp.head(dT.nP()), dT), viewOf(xp.tail(dH.nP()), dH), dT, dH, m, *data, Jp);
    reset.jacobian(viewOf(xm.head(dT.nP()), dT), viewOf(xm.tail(dH.nP()), dH), dT, dH, m, *data, Jm);
    Hfd.col(i) = (Jp.transpose() * w - Jm.transpose() * w) / (2.0 * eps);
  }
  double scale;
  const double err = relErr(H, Hfd, scale);
  char msg[200];
  std::snprintf(msg, sizeof(msg), "plastic impact reset: contracted Hessian vs FD (rel err %.2e, scale %.2e)", err, scale);
  CHECK(err < tol, msg);
  std::printf("  %s\n", msg);
}

void checkRobot(const std::string& root, const char* robot) {
  std::printf("=== %s ===\n", robot);
  const leap::RobotDescription rd = leap::loadRobotDescription(root + "/robots/" + robot + ".yaml");
  leap::RobotModel m(rd);
  const leap::ContactSet all = m.allContacts();
  const int coneId = m.ensureImpactConeSet(all);
  const leap::NodeDims d = m.nodeDims(all);
  leap::PinocchioHessians hess(m);
  Eigen::VectorXd z = randomVector(d.nP(), 0.4, 3);
  z(2) += 0.6;
  for (const leap::ContactRef& cr : all.active) z(d.offLam() + all.lamOffset(cr.index) + 2) += 40.0;
  const double tol = 2e-5;
  const int dim = m.contactDim(0);
  const bool flat = dim == 6;
  checkKernel("dyn", m, hess, leap::DynamicsConstraint(m, all), all, z, tol);
  checkKernel("dyn (floating-base rows)", m, hess, leap::FloatingBaseDynamicsConstraint(m, all), all, z, tol);
  checkKernel("pos contact 0", m, hess, leap::FramePositionConstraint(0, Eigen::Vector3d(0.1, 0.0, 0.0), "pos"), all, z, tol);
  checkKernel("vel contact 0", m, hess, leap::FrameVelocityConstraint(0, Eigen::VectorXd::Zero(dim), "vel"), all, z, tol);
  checkKernel("acc contact 0", m, hess, leap::FrameAccelerationConstraint(0, Eigen::VectorXd::Zero(dim), "acc"), all, z, tol);
  if (flat) checkKernel("orientation contact 0", m, hess, leap::FrameOrientationConstraint(0, Eigen::Vector3d(0.0, 0.0, 0.1), "pos_flat"), all, z, tol);
  if (m.nMonitors() > 0)
    checkKernel("monitor heights", m, hess, leap::MonitorHeightLowerBoundConstraint(m, Eigen::VectorXd::Zero(m.nMonitors()), "ineq_clear"), all, z, tol);
  {
    std::vector<int> off;
    for (const leap::ContactRef& cr : all.active) off.push_back(m.impactConeLamOffset(coneId, cr.index));
    checkKernel("impact cone", m, hess, leap::ImpactConeConstraint(coneId, off, 0.8, 0.01, "ineq_impact_cone"), all, z, tol);
  }
  checkReset(m, hess, all, z, tol);
  // the tapes: an independent derivation of the dynamics / velocity / acceleration kernels
  checkDynamicsAgainstTape(m, hess, all, z);
  checkAgainstTape("vel contact 0", m, hess, leap::FrameTerm::Velocity, 0, all, z);
  checkAgainstTape("acc contact 0", m, hess, leap::FrameTerm::Acceleration, 0, all, z);
  std::printf("  tape calls: %ld (the cross-checks only)\n", hess.tapeCalls());
  CHECK(hess.tapeCalls() == 3, "the analytic path never falls back to a tape");
}

class DenseSink final : public leap::MetricSink {
public:
  explicit DenseSink(int n) : H(Eigen::MatrixXd::Zero(n, n)) {}
  void addBlock(int, const std::vector<int>& idx, const Eigen::MatrixXd& B) override {
    const std::lock_guard<std::mutex> lock(mutex_);
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

// The exact-Newton metric of a posed problem vs the finite difference of F, every
// multiplier live. y0 is the primal start (duals overwritten with random values).
void checkInSitu(const char* label, const leap::CompiledProblem& prob, const Eigen::VectorXd& y0, double primalNoise) {
  std::printf("=== %s: exact-Newton metric vs FD of F (np=%d md=%d) ===\n", label, prob.layout().np, prob.layout().md);
  leap::EvaluatorOptions opt;
  opt.gaussNewton = false;
  opt.activeMargin = 0.0;
  opt.threads = 4;
  leap::AlmEvaluator ev(prob, opt);
  const leap::JunctionQuotient& Q = ev.quotient();
  const int np = prob.layout().np, md = prob.layout().md;
  Eigen::VectorXd y = y0;
  y.head(np) += randomVector(np, primalNoise, 5);
  for (int e = 0; e < prob.numElements(); ++e) {
    const leap::ElementLayout& L = prob.layout().elem[static_cast<size_t>(e)];
    if (L.Nc == 0) continue;
    leap::ElementViewMutable X(y.data(), L);
    X.C.row(2).array() += 20.0;  // keep the forces off the cone apex
  }
  Q.enforce(y);
  y.tail(md) = randomVector(md, 2.0, 22);
  Q.enforce(y);
  Eigen::VectorXd mu = Eigen::VectorXd::Constant(md, 7.0);
  for (int i = 0; i < md; ++i) mu(i) += 3.0 * std::abs(std::sin(0.7 * prob.rowTerm()[static_cast<size_t>(i)]));
  leap::EvaluationRequest req;
  leap::Evaluation out;
  ev.evaluate(y, mu, req, out);
  CHECK(out.F.allFinite(), "F finite");
  DenseSink sink(np);
  ev.assembleMetric(y, out, mu, sink);
  double worst = 0.0;
  int tested = 0;
  leap::Evaluation vp, vm;
  for (unsigned k = 0; k < 8; ++k) {
    Eigen::VectorXd dir = Eigen::VectorXd::Zero(np + md);
    dir.head(np) = randomVector(np, 1.0, 100 + k);
    Q.enforce(dir);
    const double h = 1e-6;
    Eigen::VectorXd yp = y + h * dir, ym = y - h * dir;
    Q.enforce(yp);
    Q.enforce(ym);
    ev.evaluate(yp, mu, req, vp);
    ev.evaluate(ym, mu, req, vm);
    if (vp.phr.active != out.phr.active || vm.phr.active != out.phr.active) continue;
    const Eigen::VectorXd fd = -(vp.F.head(np) - vm.F.head(np)) / (2.0 * h);  // d(-F)/dy . d = H d
    const Eigen::VectorXd Hd = sink.H * dir.head(np);
    Eigen::VectorXd fdr, Hdr;
    Q.reduceCovector(fd, fdr);
    Q.reduceCovector(Hd, Hdr);
    const double scale = std::max(1.0, fdr.cwiseAbs().maxCoeff());
    worst = std::max(worst, (fdr - Hdr).cwiseAbs().maxCoeff() / scale);
    ++tested;
  }
  char msg[200];
  std::snprintf(msg, sizeof(msg), "%s: exact metric H d vs FD of F along %d directions (worst rel err %.2e)", label, tested, worst);
  CHECK(tested > 0 && worst < 1e-5, msg);
  std::printf("  %s\n", msg);
  const double asym = (sink.H - sink.H.transpose()).cwiseAbs().maxCoeff() / std::max(1.0, sink.H.cwiseAbs().maxCoeff());
  CHECK(asym < 1e-10, "exact metric symmetric");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  try {
    checkRobot(root, "hopper2d");
    checkRobot(root, "hopper2d_foot");
    checkRobot(root, "go2");
    checkRobot(root, "go2_motor");  // friction_inertia motor model: the Coulomb loss's (v, v) Hessian
    checkRobot(root, "h1");
    {
      auto model = std::make_shared<leap::RobotModel>(leap::loadRobotDescription(root + "/robots/hopper2d.yaml"));
      const leap::CompiledProblem prob = leap::CompiledProblem::compile(leap::test::buildHopperTestProblem(model));
      Eigen::VectorXd y0 = randomVector(prob.layout().n(), 0.3, 21);
      for (int e = 0; e < prob.numElements(); ++e) {
        const leap::ElementLayout& L = prob.layout().elem[static_cast<size_t>(e)];
        leap::ElementViewMutable X(y0.data(), L);
        X.Q.row(2).array() += 0.6;
        if (L.Nc > 0) X.C.row(2).array() += 60.0;
      }
      checkInSitu("hand-built hopper (torque elimination, impact cone, plastic reset)", prob, y0, 0.0);
    }
    {
      leap::YamlDoc d = leap::YamlDoc::fromFile(root + "/problems/hopper2d_jump_fence.yaml");
      for (const char* ov : {"solver.split=2", "solver.hgrade=0.6", "solver.lsqnqs=8,8,8", "solver.rho=3000000", "obstacles.smooth=0", "obstacles.rounded=true", "problem.impactcone=true"})
        d.setOverride(ov);
      const leap::RobotDescription rd = leap::loadRobotDescription(d.resolvePath(d.requireS("robot")));
      std::shared_ptr<const leap::RobotModel> model = std::make_shared<leap::RobotModel>(rd);
      leap::FenceFamily fam(d, rd, model);
      const leap::CompiledProblem prob = leap::CompiledProblem::compile(fam.build());
      checkInSitu("fence hopper2d", prob, fam.seed(prob), 0.01);
    }
    {
      leap::YamlDoc d = leap::YamlDoc::fromFile(root + "/problems/hopper2d_foot_terrain_walk.yaml");
      for (const char* ov : {"solver.split=1", "problem.attnq=1", "problem.qboxnq=1", "problem.uboxnq=2", "problem.noslipacc=true", "problem.uhull=16"})
        d.setOverride(ov);
      const leap::RobotDescription rd = leap::loadRobotDescription(d.resolvePath(d.requireS("robot")));
      std::shared_ptr<const leap::RobotModel> model = std::make_shared<leap::RobotModel>(rd);
      leap::TerrainFamily fam(d, rd, model);
      const leap::CompiledProblem prob = leap::CompiledProblem::compile(fam.build());
      checkInSitu("terrain hopper2d_foot (flat foot, torque elimination)", prob, fam.seed(prob), 0.01);
    }
  } catch (const std::exception& e) {
    std::printf("FAIL exception: %s\n", e.what());
    return 1;
  }
  return leap::test::finish("test_hessian_fd");
}
