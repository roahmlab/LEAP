// Finite-difference check of every constraint of the catalogue (value vs jacobian) and
// of the friction cone's closed-form contracted Hessian, on Go2 (point feet, capsules,
// monitors, an obstacle box) and H1 (flat feet: wrench boxes, orientation).
//
// usage: test_constraints_fd <repo root>

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "leap/model/robot_model.hpp"
#include "leap/problem/constraints.hpp"
#include "leap/problem/obstacles.hpp"
#include "leap/problem/self_collision.hpp"
#include "test_check.hpp"

namespace {

Eigen::VectorXd randomVector(int n, double amplitude, unsigned seed) {
  Eigen::VectorXd x(n);
  for (int i = 0; i < n; ++i) x(i) = amplitude * std::sin(0.37 * seed + 1.3 * i + 0.11 * seed * i);
  return x;
}

struct Point {
  Eigen::VectorXd q, v, a, lam, u;
};

leap::NodeQuantities view(const Point& p) { return leap::NodeQuantities{p.q, p.v, p.a, p.lam, p.u}; }

// Pack / unpack z = [q; v; a; lam; u].
Eigen::VectorXd pack(const Point& p) {
  Eigen::VectorXd z(p.q.size() + p.v.size() + p.a.size() + p.lam.size() + p.u.size());
  z << p.q, p.v, p.a, p.lam, p.u;
  return z;
}

Point unpack(const Eigen::VectorXd& z, const leap::NodeDims& d) {
  Point p;
  p.q = z.segment(d.offQ(), d.nq);
  p.v = z.segment(d.offV(), d.nv);
  p.a = z.segment(d.offA(), d.na);
  p.lam = z.segment(d.offLam(), d.nlam);
  p.u = z.segment(d.offU(), d.nu);
  return p;
}

void checkConstraint(const leap::Constraint& con, const leap::RobotModel& model, leap::ModelData& data,
                     const leap::ContactSet& cs, const Point& p0, const char* label) {
  const leap::NodeDims d = model.nodeDims(cs);
  const int m = con.rows(d);
  leap::ModelEvalCache c;
  const unsigned needs = con.modelNeeds();
  model.evalNode(data, p0.q, p0.v, p0.a, p0.lam, cs, needs, c);
  Eigen::VectorXd g0(m);
  con.value(view(p0), d, c, g0);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(m, d.nP());
  con.jacobian(view(p0), d, c, J);
  CHECK(g0.allFinite() && J.allFinite(), label);

  const Eigen::VectorXd z0 = pack(p0);
  const double eps = 1e-6;
  Eigen::MatrixXd Jfd(m, d.nP());
  Eigen::VectorXd gp(m), gm(m);
  // values-only needs for the FD evaluations (derivative flags are not needed)
  const unsigned valueNeeds = needs & ~(leap::kRneaDerivs | leap::kFootDerivs);
  for (int j = 0; j < d.nP(); ++j) {
    Eigen::VectorXd zp = z0, zm = z0;
    zp(j) += eps;
    zm(j) -= eps;
    const Point pp = unpack(zp, d), pm = unpack(zm, d);
    model.evalNode(data, pp.q, pp.v, pp.a, pp.lam, cs, valueNeeds | leap::kRnea * ((needs & leap::kRnea) ? 1 : 0), c);
    con.value(view(pp), d, c, gp);
    model.evalNode(data, pm.q, pm.v, pm.a, pm.lam, cs, valueNeeds | leap::kRnea * ((needs & leap::kRnea) ? 1 : 0), c);
    con.value(view(pm), d, c, gm);
    Jfd.col(j) = (gp - gm) / (2.0 * eps);
  }
  const double scale = std::max(1.0, J.cwiseAbs().maxCoeff());
  const double err = (J - Jfd).cwiseAbs().maxCoeff() / scale;
  char msg[256];
  std::snprintf(msg, sizeof(msg), "%s: jacobian vs FD (rel err %.2e, rows %d)", label, err, m);
  CHECK(err < 3e-5, msg);
}

void checkGo2(const std::string& robots) {
  std::printf("--- go2 ---\n");
  leap::RobotModel model(leap::loadRobotDescription(robots + "go2.yaml"));
  const leap::ContactSet cs = model.allContacts();
  const int coneId = model.ensureImpactConeSet(cs);
  const int nq = model.nq(), nv = model.nv();
  leap::ModelDataPtr data = model.makeData();
  const leap::NodeDims d = model.nodeDims(cs);
  Point p;
  p.q = randomVector(nq, 0.3, 1);
  p.q(2) += 0.4;
  p.v = randomVector(nv, 0.8, 2);
  p.a = randomVector(nv, 2.0, 3);
  p.lam = randomVector(d.nlam, 30.0, 4);
  p.u = randomVector(d.nu, 5.0, 5);

  checkConstraint(leap::DynamicsConstraint(model, cs), model, *data, cs, p, "dynamics");
  checkConstraint(leap::FloatingBaseDynamicsConstraint(model, cs), model, *data, cs, p, "floating-base dynamics");
  checkConstraint(leap::SelectedRowsDynamicsConstraint(model, cs, {0, 2, 4, 7, 8}), model, *data, cs, p,
                  "selected-rows dynamics");
  checkConstraint(leap::FramePositionConstraint(1, Eigen::Vector3d(0.1, 0.2, 0.0), "pos"), model, *data, cs, p,
                  "frame position");
  checkConstraint(leap::FramePositionComponentConstraint(2, 2, 0.05, "posz"), model, *data, cs, p, "frame position z");
  checkConstraint(leap::FramePositionHalfspaceConstraint(0, Eigen::Vector3d(-1.0, 0.0, 0.0), -0.2, "half"), model,
                  *data, cs, p, "frame halfspace");
  checkConstraint(leap::ContactHeightLowerBoundConstraint(3, 0.02, "swingz"), model, *data, cs, p, "contact height");
  checkConstraint(leap::FrameVelocityConstraint(1, Eigen::Vector3d::Zero(), "vel"), model, *data, cs, p,
                  "frame velocity");
  checkConstraint(leap::FrameVelocityBoundConstraint(2, 2, -1.0, -0.5, "vz"), model, *data, cs, p, "frame velocity bound");
  checkConstraint(leap::FrameAccelerationConstraint(0, Eigen::Vector3d::Zero(), "acc"), model, *data, cs, p,
                  "frame acceleration");
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(model.nMonitors(), 0.0);
  checkConstraint(leap::MonitorHeightLowerBoundConstraint(model, lb, "clear"), model, *data, cs, p, "monitor height");
  checkConstraint(leap::PinConstraint({0, 1, 5, nq + 2}, Eigen::Vector4d(0.1, 0.2, 0.3, 0.4), "pin"), model, *data,
                  cs, p, "pin");
  checkConstraint(leap::PairEqualityConstraint({6, 7}, {9, 10}, "pair"), model, *data, cs, p, "pair equality");
  checkConstraint(leap::LowerBoundConstraint({2, nq + 3}, Eigen::Vector2d(0.1, -1.0), "lb"), model, *data, cs, p,
                  "lower bound");
  {
    Eigen::VectorXd lo(3), hi(3);
    lo << -1.0, -INFINITY, -0.5;
    hi << 1.0, 2.0, INFINITY;
    checkConstraint(leap::BoxConstraint(leap::Field::V, {6, 7, 8}, lo, hi, "box_v"), model, *data, cs, p, "box");
  }
  {
    leap::MotorSaturationConfig sat;
    sat.enabled = true;
    sat.speedDerating = true;
    checkConstraint(leap::MotorEnvelopeConstraint(model, sat, "motor"), model, *data, cs, p, "motor envelope");
  }
  {
    auto inner = std::make_shared<leap::FrameVelocityConstraint>(1, Eigen::Vector3d::Zero(), "vel");
    checkConstraint(leap::RowSliceConstraint(inner, 2), model, *data, cs, p, "row slice");
  }
  checkConstraint(leap::FrictionConeConstraint(cs, 0.8, 0.01, "fric"), model, *data, cs, p, "friction cone");
  {
    std::vector<int> off;
    for (const leap::ContactRef& cr : cs.active) off.push_back(model.impactConeLamOffset(coneId, cr.index));
    checkConstraint(leap::ImpactConeConstraint(coneId, off, 0.8, 0.01, "impact"), model, *data, cs, p, "impact cone");
  }
  {
    leap::SphereObstacle sph;
    sph.name = "rock";
    sph.center = Eigen::Vector3d(0.2, 0.1, 0.1);
    sph.radius = 0.1;
    checkConstraint(leap::CapsuleSphereConstraint(model, {sph}, {}, 0.02, "obs"), model, *data, cs, p, "capsule-sphere");
    leap::CuboidObstacle box;
    box.name = "fence";
    box.center = Eigen::Vector3d(0.25, 0.0, 0.04);
    box.half = Eigen::Vector3d(0.02, 0.4, 0.04);
    checkConstraint(leap::CapsuleCuboidConstraint(model, {box}, {}, 0.02, "box"), model, *data, cs, p,
                    "capsule-cuboid (certificate)");
    checkConstraint(leap::CapsuleCuboidConstraint(model, {box}, {}, 0.02, "boxr", true), model, *data, cs, p,
                    "capsule-cuboid (rounded)");
    checkConstraint(leap::CapsuleCuboidConstraint(model, {box}, {}, 0.02, "boxs", false, {}, 0.003), model, *data, cs,
                    p, "capsule-cuboid (smooth)");
    std::vector<std::pair<int, int>> pairs{{model.capsuleIndex("FL_thigh"), model.capsuleIndex("FR_thigh")},
                                           {model.capsuleIndex("FL_calf"), model.capsuleIndex("RL_calf")}};
    checkConstraint(leap::CapsuleSelfCollisionConstraint(model, pairs, 0.02, "selfcol"), model, *data, cs, p,
                    "self-collision");
  }
  // the closed-form friction cone Hessian against FD of its Jacobian
  {
    leap::FrictionConeConstraint cone(cs, 0.8, 0.01, "fric");
    leap::ModelEvalCache c;
    const Eigen::VectorXd w = randomVector(cone.rows(d), 1.0, 7).cwiseAbs();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d.nP(), d.nP());
    leap::HessianScratch s;
    cone.addContractedHessian(view(p), d, s, w, H);
    const Eigen::VectorXd z0 = pack(p);
    const double eps = 1e-6;
    Eigen::MatrixXd Hfd = Eigen::MatrixXd::Zero(d.nP(), d.nP());
    Eigen::MatrixXd Jp = Eigen::MatrixXd::Zero(cone.rows(d), d.nP()), Jm = Jp;
    for (int j = 0; j < d.nP(); ++j) {
      Eigen::VectorXd zp = z0, zm = z0;
      zp(j) += eps;
      zm(j) -= eps;
      cone.jacobian(view(unpack(zp, d)), d, c, Jp);
      cone.jacobian(view(unpack(zm, d)), d, c, Jm);
      Hfd.col(j) = ((Jp - Jm) / (2.0 * eps)).transpose() * w;
    }
    CHECK_NEAR((H - Hfd).cwiseAbs().maxCoeff(), 0.0, 1e-6, "friction cone contracted Hessian");
  }
}

void checkH1(const std::string& robots) {
  std::printf("--- h1 ---\n");
  leap::RobotModel model(leap::loadRobotDescription(robots + "h1.yaml"));
  const leap::ContactSet cs = model.allContacts();
  const int nq = model.nq(), nv = model.nv();
  leap::ModelDataPtr data = model.makeData();
  const leap::NodeDims d = model.nodeDims(cs);
  Point p;
  p.q = randomVector(nq, 0.2, 11);
  p.q(2) += 1.0;
  p.v = randomVector(nv, 0.5, 12);
  p.a = randomVector(nv, 1.0, 13);
  p.lam = randomVector(d.nlam, 50.0, 14);
  p.u = randomVector(d.nu, 5.0, 15);
  checkConstraint(leap::DynamicsConstraint(model, cs), model, *data, cs, p, "dynamics (wrench)");
  checkConstraint(leap::FloatingBaseDynamicsConstraint(model, cs), model, *data, cs, p, "floating-base dynamics (wrench)");
  Eigen::VectorXd ref6 = Eigen::VectorXd::Zero(6);
  checkConstraint(leap::FrameVelocityConstraint(0, ref6, "vel6"), model, *data, cs, p, "frame velocity 6");
  checkConstraint(leap::FrameAccelerationConstraint(1, ref6, "acc6"), model, *data, cs, p, "frame acceleration 6");
  checkConstraint(leap::FrameOrientationConstraint(0, Eigen::Vector3d(0.0, 0.0, 0.1), "sole"), model, *data, cs, p,
                  "frame orientation");
  checkConstraint(leap::FrictionConeConstraint(cs, 0.8, 0.01, "fric"), model, *data, cs, p, "friction cone (wrench)");
  checkConstraint(leap::ContactWrenchBoxConstraint(cs, {4, 3}, {0.14, 0.015}, "cop"), model, *data, cs, p, "CoP box");
  checkConstraint(leap::ContactWrenchBoxConstraint(cs, {5}, {0.05}, "tz"), model, *data, cs, p, "yaw box");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <repo root>\n", argv[0]);
    return 2;
  }
  const std::string robots = std::string(argv[1]) + "/robots/";
  checkGo2(robots);
  checkH1(robots);
  return leap::test::finish("test_constraints_fd");
}
