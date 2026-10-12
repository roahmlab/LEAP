// Finite-difference check of every analytic derivative the RobotModel exposes, on
// three robots: the planar point-foot hopper (dim 3 contact, planar), Go2 (four dim-3
// contacts, capsules, monitors) and H1 (two dim-6 flat-foot contacts).
//
//   dynamics   dgdyn/dq, dgdyn/dv (= dtau/dv), dgdyn/da (= M), dgdyn/dlam (= -Jc')
//   frames     footJ (position), dvel/dq, dvel/dv, dacc/dq, dacc/dv, dacc/da,
//              footRotJ (dim 6), monitor and capsule endpoint Jacobians
//   impacts    d v+/dq, d v+/dv of the plastic impact map, and the impulse partials
//
// usage: test_model_fd <repo root>

#include <cmath>
#include <cstdio>
#include <string>

#include "leap/model/robot_model.hpp"
#include "test_check.hpp"

namespace {

double relErr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  const double scale = std::max(1.0, std::max(A.cwiseAbs().maxCoeff(), B.cwiseAbs().maxCoeff()));
  return (A - B).cwiseAbs().maxCoeff() / scale;
}

Eigen::VectorXd randomVector(int n, double amplitude, unsigned seed) {
  Eigen::VectorXd x(n);
  for (int i = 0; i < n; ++i) x(i) = amplitude * std::sin(0.37 * seed + 1.3 * i + 0.11 * seed * i);
  return x;
}

void checkRobot(const std::string& yaml, const char* label) {
  std::printf("--- %s ---\n", label);
  leap::RobotDescription rd = leap::loadRobotDescription(yaml);
  leap::RobotModel model(rd);
  const int nq = model.nq(), nv = model.nv();
  CHECK(nq == nv, "explicit floating base: nq == nv");
  const leap::ContactSet cs = model.allContacts();
  const int nlam = cs.nLam();
  const int setId = model.ensureImpactSet(cs);
  const int coneId = model.ensureImpactConeSet(cs);
  leap::ModelDataPtr data = model.makeData();

  Eigen::VectorXd q = randomVector(nq, 0.3, 1);
  q(2) += 0.5;  // base height
  const Eigen::VectorXd v = randomVector(nv, 0.8, 2);
  const Eigen::VectorXd a = randomVector(nv, 2.0, 3);
  const Eigen::VectorXd lam = randomVector(nlam, 40.0, 4);
  const unsigned needs = leap::kRnea | leap::kRneaDerivs | leap::kFootPos | leap::kFootVel | leap::kFootAcc |
                         leap::kFootDerivs | leap::kMonitorPos | leap::kCapsulePos | leap::kFootRot |
                         leap::kImpactImpulse;
  leap::ModelEvalCache c0;
  model.evalNode(*data, q, v, a, lam, cs, needs, c0);

  const double eps = 1e-6;
  const int nc = model.nContacts();
  Eigen::MatrixXd dg_dq(nv, nq), dg_dv(nv, nv), dg_da(nv, nv), dg_dlam(nv, nlam);
  std::vector<Eigen::MatrixXd> dpos_dq(nc), dvel_dq(nc), dvel_dv(nc), dacc_dq(nc), dacc_dv(nc), dacc_da(nc),
      drot_dq(nc);
  for (int i = 0; i < nc; ++i) {
    const int dim = model.contactDim(i);
    dpos_dq[i].resize(std::min(dim, 3), nq);
    dvel_dq[i].resize(dim, nq);
    dvel_dv[i].resize(dim, nv);
    dacc_dq[i].resize(dim, nq);
    dacc_dv[i].resize(dim, nv);
    dacc_da[i].resize(dim, nv);
    drot_dq[i].resize(3, nq);
  }
  std::vector<Eigen::MatrixXd> dmon_dq(model.nMonitors(), Eigen::MatrixXd(3, nq));
  std::vector<Eigen::MatrixXd> dcap_dq(2 * model.nCapsules(), Eigen::MatrixXd(3, nq));
  Eigen::MatrixXd dLam_dq(nlam, nq), dLam_dv(nlam, nv);
  Eigen::MatrixXd dvplus_dq(nv, nq), dvplus_dv(nv, nv);
  leap::ModelEvalCache cp, cm;
  const unsigned valueNeeds = leap::kRnea | leap::kFootPos | leap::kFootVel | leap::kFootAcc | leap::kMonitorPos |
                              leap::kCapsulePos | leap::kFootRot | leap::kImpactImpulse;

  // central differences in q
  for (int j = 0; j < nq; ++j) {
    Eigen::VectorXd qp = q, qm = q;
    qp(j) += eps;
    qm(j) -= eps;
    model.evalNode(*data, qp, v, a, lam, cs, valueNeeds, cp);
    model.evalNode(*data, qm, v, a, lam, cs, valueNeeds, cm);
    dg_dq.col(j) = (cp.gdyn - cm.gdyn) / (2 * eps);
    for (int i = 0; i < nc; ++i) {
      dpos_dq[i].col(j) = (cp.footPos[i] - cm.footPos[i]) / (2 * eps);
      dvel_dq[i].col(j) = (cp.footVel[i] - cm.footVel[i]) / (2 * eps);
      dacc_dq[i].col(j) = (cp.footAcc[i] - cm.footAcc[i]) / (2 * eps);
      drot_dq[i].col(j) = (cp.footRotLog[i] - cm.footRotLog[i]) / (2 * eps);
    }
    for (int m = 0; m < model.nMonitors(); ++m) dmon_dq[m].col(j) = (cp.monPos[m] - cm.monPos[m]) / (2 * eps);
    for (int k = 0; k < 2 * model.nCapsules(); ++k) dcap_dq[k].col(j) = (cp.capPos[k] - cm.capPos[k]) / (2 * eps);
    dLam_dq.col(j) = (cp.impactLam[coneId] - cm.impactLam[coneId]) / (2 * eps);
    Eigen::VectorXd vpP, vpM;
    model.impactVelocity(*data, setId, qp, v, vpP);
    model.impactVelocity(*data, setId, qm, v, vpM);
    dvplus_dq.col(j) = (vpP - vpM) / (2 * eps);
  }
  // in v
  for (int j = 0; j < nv; ++j) {
    Eigen::VectorXd vp = v, vm = v;
    vp(j) += eps;
    vm(j) -= eps;
    model.evalNode(*data, q, vp, a, lam, cs, valueNeeds, cp);
    model.evalNode(*data, q, vm, a, lam, cs, valueNeeds, cm);
    dg_dv.col(j) = (cp.gdyn - cm.gdyn) / (2 * eps);
    for (int i = 0; i < nc; ++i) {
      dvel_dv[i].col(j) = (cp.footVel[i] - cm.footVel[i]) / (2 * eps);
      dacc_dv[i].col(j) = (cp.footAcc[i] - cm.footAcc[i]) / (2 * eps);
    }
    dLam_dv.col(j) = (cp.impactLam[coneId] - cm.impactLam[coneId]) / (2 * eps);
    Eigen::VectorXd vpP, vpM;
    model.impactVelocity(*data, setId, q, vp, vpP);
    model.impactVelocity(*data, setId, q, vm, vpM);
    dvplus_dv.col(j) = (vpP - vpM) / (2 * eps);
  }
  // in a
  for (int j = 0; j < nv; ++j) {
    Eigen::VectorXd ap = a, am = a;
    ap(j) += eps;
    am(j) -= eps;
    model.evalNode(*data, q, v, ap, lam, cs, valueNeeds, cp);
    model.evalNode(*data, q, v, am, lam, cs, valueNeeds, cm);
    dg_da.col(j) = (cp.gdyn - cm.gdyn) / (2 * eps);
    for (int i = 0; i < nc; ++i) dacc_da[i].col(j) = (cp.footAcc[i] - cm.footAcc[i]) / (2 * eps);
  }
  // in lam
  for (int j = 0; j < nlam; ++j) {
    Eigen::VectorXd lp = lam, lm = lam;
    lp(j) += eps;
    lm(j) -= eps;
    model.evalNode(*data, q, v, a, lp, cs, leap::kRnea, cp);
    model.evalNode(*data, q, v, a, lm, cs, leap::kRnea, cm);
    dg_dlam.col(j) = (cp.gdyn - cm.gdyn) / (2 * eps);
  }

  const double tol = 2e-5;
  CHECK_NEAR(relErr(dg_dq, c0.dgdyn_dq), 0.0, tol, "dgdyn/dq");
  CHECK_NEAR(relErr(dg_dv, c0.dtau_dv), 0.0, tol, "dgdyn/dv");
  CHECK_NEAR(relErr(dg_da, c0.M), 0.0, tol, "dgdyn/da = M");
  Eigen::MatrixXd negJt(nv, nlam);
  {
    int o = 0;
    for (const leap::ContactRef& cr : cs.active) {
      negJt.middleCols(o, cr.dim) = -c0.footJ[static_cast<size_t>(cr.index)].transpose();
      o += cr.dim;
    }
  }
  CHECK_NEAR(relErr(dg_dlam, negJt), 0.0, tol, "dgdyn/dlam = -Jc'");
  for (int i = 0; i < nc; ++i) {
    const int dim = model.contactDim(i);
    const int nLin = std::min(dim, 3);
    CHECK_NEAR(relErr(dpos_dq[i], c0.footJ[i].topRows(nLin)), 0.0, tol, "footJ = d footPos / dq");
    CHECK_NEAR(relErr(dvel_dq[i], c0.dvel_dq[i]), 0.0, tol, "dvel/dq");
    CHECK_NEAR(relErr(dvel_dv[i], c0.footJ[i]), 0.0, tol, "dvel/dv = footJ");
    CHECK_NEAR(relErr(dacc_dq[i], c0.dacc_dq[i]), 0.0, tol, "dacc/dq");
    CHECK_NEAR(relErr(dacc_dv[i], c0.dacc_dv[i]), 0.0, tol, "dacc/dv");
    CHECK_NEAR(relErr(dacc_da[i], c0.footJ[i]), 0.0, tol, "dacc/da = footJ");
    if (dim == 6) CHECK_NEAR(relErr(drot_dq[i], c0.footRotJ[i]), 0.0, tol, "footRotJ");
  }
  for (int m = 0; m < model.nMonitors(); ++m) CHECK_NEAR(relErr(dmon_dq[m], c0.monJ[m]), 0.0, tol, "monitor J");
  for (int k = 0; k < 2 * model.nCapsules(); ++k) CHECK_NEAR(relErr(dcap_dq[k], c0.capJ[k]), 0.0, tol, "capsule J");
  // capsuleKinematics must agree with the cache
  {
    std::vector<Eigen::Vector3d> pos;
    std::vector<Eigen::MatrixXd> J;
    model.capsuleKinematics(*data, q, pos, J);
    for (int k = 0; k < 2 * model.nCapsules(); ++k) {
      CHECK_NEAR((pos[k] - c0.capPos[k]).norm(), 0.0, 1e-14, "capsuleKinematics position");
      CHECK_NEAR(relErr(J[k], c0.capJ[k]), 0.0, 1e-14, "capsuleKinematics Jacobian");
    }
  }
  // impacts
  {
    Eigen::MatrixXd dvdq, dvdv;
    Eigen::VectorXd vplus;
    model.impactDerivatives(*data, setId, q, v, dvdq, dvdv, vplus);
    CHECK_NEAR(relErr(dvplus_dq, dvdq), 0.0, 1e-4, "impact d v+/dq");
    CHECK_NEAR(relErr(dvplus_dv, dvdv), 0.0, tol, "impact d v+/dv");
    CHECK_NEAR(relErr(dLam_dq, c0.impactLamDq[coneId]), 0.0, 1e-4, "impulse dLam/dq");
    CHECK_NEAR(relErr(dLam_dv, c0.impactLamDv[coneId]), 0.0, tol, "impulse dLam/dv");
    // the impulse stops the landing feet: J v+ ~ 0 (up to the 1e-9 Delassus damping)
    leap::ModelEvalCache cplus;
    model.evalNode(*data, q, vplus, a, lam, cs, leap::kFootVel | leap::kFootPos, cplus);
    for (int i = 0; i < nc; ++i) CHECK_NEAR(cplus.footVel[i].norm(), 0.0, 1e-6, "post-impact contact velocity is zero");
  }
  // inverse dynamics torque helper
  {
    const Eigen::VectorXd u = model.inverseDynamicsTorque(*data, q, v, a, lam, cs);
    for (int j = 0; j < model.nu(); ++j)
      CHECK_NEAR(u(j), c0.gdyn(model.actuatedIdx()[static_cast<size_t>(j)]), 1e-12, "inverseDynamicsTorque");
  }
  std::printf("  nq=%d nu=%d contacts=%d (nlam=%d) monitors=%d capsules=%d mass=%.3f\n", nq, model.nu(), nc, nlam,
              model.nMonitors(), model.nCapsules(), model.totalMass());
}

// Explicit `actuation.effort_limits` / `velocity_limits` lists of a robot yaml override
// the URDF values.
void checkExplicitLimits(const std::string& yaml) {
  std::printf("--- explicit effort / velocity limits ---\n");
  leap::RobotDescription rd = leap::loadRobotDescription(yaml);
  const leap::RobotModel fromUrdf(rd);
  const int nu = fromUrdf.nu();
  rd.effortLimitsSource = "explicit";
  rd.velocityLimitsSource = "explicit";
  for (int j = 0; j < nu; ++j) {
    rd.effortLimits.push_back(10.0 + j);
    rd.velocityLimits.push_back(1.0 + 0.5 * j);
  }
  const leap::RobotModel model(rd);
  const Eigen::VectorXd ue = model.actuatedEffortLimit(), uv = model.actuatedVelocityLimit(), vv = model.velocityLimit();
  bool ok = ue.size() == nu && uv.size() == nu && vv.size() == model.nv();
  for (int j = 0; ok && j < nu; ++j) {
    const int i = model.actuatedIdx()[static_cast<size_t>(j)];
    ok = ue(j) == 10.0 + j && uv(j) == 1.0 + 0.5 * j && vv(i) == 1.0 + 0.5 * j;
  }
  CHECK(ok, "explicit lists override the URDF limits (u order and v order)");
  const Eigen::VectorXd vUrdf = fromUrdf.velocityLimit();
  ok = true;
  for (int i : model.unactuatedIdx()) ok = ok && vv(i) == vUrdf(i);
  CHECK(ok, "unactuated velocity limits keep the URDF value");
  ok = (fromUrdf.actuatedEffortLimit() - ue).cwiseAbs().maxCoeff() > 0.0;
  CHECK(ok, "the from_urdf model differs from the explicit one (the lists were applied)");
  rd.effortLimits.pop_back();
  bool threw = false;
  try {
    const leap::RobotModel bad(rd);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw, "an explicit list of the wrong size is rejected");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <repo root>\n", argv[0]);
    return 2;
  }
  const std::string robots = std::string(argv[1]) + "/robots/";
  checkRobot(robots + "hopper2d.yaml", "hopper2d");
  checkRobot(robots + "go2.yaml", "go2");
  checkRobot(robots + "go2_motor.yaml", "go2_motor (friction_inertia motor model)");
  checkRobot(robots + "h1.yaml", "h1");
  checkExplicitLimits(robots + "go2.yaml");
  return leap::test::finish("test_model_fd");
}
