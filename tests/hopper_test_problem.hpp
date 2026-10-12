#pragma once

// The hand-built planar-hopper problem shared by the augmented-Lagrangian tests: two
// stance elements (the second with torque elimination) and one flight element landing
// with a plastic impact, continuity and plastic-impact resets, node terms of every
// family, off-node rows, force control-point rows, torque Bernstein functionals, an
// impact cone, obstacle and self-collision rows, a defect cost and the effort cost.

#include <memory>

#include "leap/problem/compiled_problem.hpp"
#include "leap/problem/costs.hpp"
#include "leap/problem/obstacles.hpp"
#include "leap/problem/self_collision.hpp"
#include "leap/spectral/bernstein.hpp"

namespace leap::test {

inline leap::TrajectoryProblem buildHopperTestProblem(std::shared_ptr<leap::RobotModel> model) {
  using namespace leap;
  TrajectoryProblem P;
  P.model = model;
  const ContactSet stance = model->allContacts();
  const ContactSet air;
  const int nq = model->nq(), nv = model->nv(), nu = model->nu();
  const int impactSet = model->ensureImpactSet(stance);
  const int coneId = model->ensureImpactConeSet(stance);

  // element 0: stance, explicit torque
  Element e0;
  e0.label = "stance";
  e0.phase = 0;
  e0.T = 0.4;
  e0.contacts = stance;
  e0.degree = 3;
  e0.forceControlPoints = 3;
  e0.addNode(std::make_shared<DynamicsConstraint>(*model, stance), Where::AllNodes);
  e0.addNode(std::make_shared<FrameVelocityConstraint>(0, Eigen::Vector3d::Zero(), "vel_foot"), Where::Interior);
  e0.addNode(std::make_shared<FramePositionConstraint>(0, Eigen::Vector3d(0.0, 0.0, 0.0), "pos_foot"), Where::LastNode);
  e0.addNode(std::make_shared<PinConstraint>(std::vector<int>{0, 2}, Eigen::Vector2d(0.0, 0.5), "pin_q"), Where::FirstNode);
  e0.addNode(std::make_shared<MonitorHeightLowerBoundConstraint>(*model, Eigen::VectorXd::Zero(model->nMonitors()), "ineq_clear"), Where::AllNodes);
  {
    Eigen::VectorXd lo = Eigen::VectorXd::Constant(nu, -2.0), hi = Eigen::VectorXd::Constant(nu, 2.0);
    std::vector<int> comps;
    for (int j = 0; j < nu; ++j) comps.push_back(j);
    e0.addNode(std::make_shared<BoxConstraint>(Field::U, comps, lo, hi, "box_u"), Where::AllNodes);
  }
  e0.addOffNode(std::make_shared<FrameVelocityConstraint>(0, Eigen::Vector3d::Zero(), "vel_foot"), uniformInteriorTaus(2));
  e0.addForceControlPoints(std::make_shared<FrictionConeConstraint>(stance, 0.8, 0.01, "ineq_fric"), {0, 1, 2});
  {
    auto floor = std::make_shared<LowerBoundConstraint>(std::vector<int>{nq + nv + nv + 2}, Eigen::VectorXd::Constant(1, 5.0), "ineq_lam_z");
    floor->setForceOnly(true);
    e0.addForceControlPoints(floor, {0, 1});
  }
  {
    // Bernstein coefficients of u: weights = rows of E^-1 (E = Bernstein(deg) at the CGL nodes)
    CglGrid g(e0.degree);
    const Eigen::MatrixXd E = bernsteinBasisAtDegree(e0.degree, g);
    const Eigen::MatrixXd Einv = E.inverse();
    std::vector<Eigen::VectorXd> W;
    for (int r = 0; r < Einv.rows(); ++r) W.push_back(Einv.row(r).transpose());
    Eigen::VectorXd lo = Eigen::VectorXd::Constant(nu, -2.5), hi = Eigen::VectorXd::Constant(nu, 2.5);
    std::vector<int> comps;
    for (int j = 0; j < nu; ++j) comps.push_back(j);
    e0.addFieldFunctionals(std::make_shared<BoxConstraint>(Field::U, comps, lo, hi, "box_u_bern"), Field::U, W);
  }
  e0.addCost(std::make_shared<ControlEffortCost>());
  e0.setDefectCost(std::make_shared<DynamicsConstraint>(*model, stance), uniformInteriorTaus(3), 50.0);
  P.addElement(e0);

  // element 1: stance, torque eliminated
  Element e1;
  e1.label = "stance_elim";
  e1.phase = 0;
  e1.T = 0.3;
  e1.contacts = stance;
  e1.degree = 3;
  e1.forceControlPoints = 3;
  e1.deriveTorque = true;
  e1.addNode(std::make_shared<FloatingBaseDynamicsConstraint>(*model, stance), Where::AllNodes);
  e1.addNode(std::make_shared<FrameVelocityConstraint>(0, Eigen::Vector3d::Zero(), "vel_foot"), Where::AllNodes);
  {
    Eigen::VectorXd lo = Eigen::VectorXd::Constant(nu, -2.0), hi = Eigen::VectorXd::Constant(nu, 2.0);
    std::vector<int> comps;
    for (int j = 0; j < nu; ++j) comps.push_back(j);
    auto box = std::make_shared<BoxConstraint>(Field::U, comps, lo, hi, "box_u");
    e1.addNode(box, Where::AllNodes);
    e1.addOffNode(box, uniformInteriorTaus(2));
  }
  e1.addNode(std::make_shared<FrameVelocityBoundConstraint>(0, 2, 1.0, 0.1, "ineq_vz"), Where::LastNode);
  e1.addForceControlPoints(std::make_shared<FrictionConeConstraint>(stance, 0.8, 0.01, "ineq_fric"), {0, 1, 2});
  e1.addCost(std::make_shared<ControlEffortCost>());
  e1.setDefectCost(std::make_shared<FloatingBaseDynamicsConstraint>(*model, stance), uniformInteriorTaus(3), 50.0);
  P.addElement(e1);
  P.addReset(std::make_shared<ContinuityReset>(nq, nv));

  // element 2: flight, lands with a plastic impact onto element 3
  Element e2;
  e2.label = "flight";
  e2.phase = 1;
  e2.T = 0.25;
  e2.contacts = air;
  e2.degree = 3;
  e2.addNode(std::make_shared<DynamicsConstraint>(*model, air), Where::AllNodes);
  e2.addNode(std::make_shared<ContactHeightLowerBoundConstraint>(0, 0.0, "ineq_swing_z"), Where::Interior);
  {
    std::vector<int> off{model->impactConeLamOffset(coneId, 0)};
    e2.addNode(std::make_shared<ImpactConeConstraint>(coneId, off, 0.8, 0.01, "ineq_impact_cone"), Where::LastNode);
  }
  {
    CuboidObstacle box;
    box.name = "fence";
    box.center = Eigen::Vector3d(0.3, 0.0, 0.04);
    box.half = Eigen::Vector3d(0.02, 0.3, 0.04);
    auto obs = std::make_shared<CapsuleCuboidConstraint>(*model, std::vector<CuboidObstacle>{box}, std::vector<int>{}, 0.02, "ineq_obstacle_box", true);
    e2.addNode(obs, Where::AllNodes);
    e2.addOffNode(obs, uniformInteriorTaus(3));
    std::vector<std::pair<int, int>> pairs{{model->capsuleIndex("torso"), model->capsuleIndex("shank")}};
    auto sc = std::make_shared<CapsuleSelfCollisionConstraint>(*model, pairs, 0.02, "ineq_selfcol");
    e2.addOffNode(sc, uniformInteriorTaus(2));
  }
  e2.addCost(std::make_shared<ControlEffortCost>());
  e2.setDefectCost(std::make_shared<DynamicsConstraint>(*model, air), uniformInteriorTaus(3), 50.0);
  P.addElement(e2);
  P.addReset(std::make_shared<ContinuityReset>(nq, nv));

  Element e3;
  e3.label = "landing";
  e3.phase = 2;
  e3.T = 0.3;
  e3.contacts = stance;
  e3.degree = 3;
  e3.forceControlPoints = 3;
  e3.addNode(std::make_shared<DynamicsConstraint>(*model, stance), Where::AllNodes);
  e3.addNode(std::make_shared<FrameVelocityConstraint>(0, Eigen::Vector3d::Zero(), "vel_foot"), Where::AllNodes);
  e3.addNode(std::make_shared<PinConstraint>(std::vector<int>{nq + 0, nq + 1}, Eigen::Vector2d::Zero(), "pin_v"), Where::LastNode);
  e3.addCost(std::make_shared<ControlEffortCost>());
  P.addElement(e3);
  P.addReset(std::make_shared<PlasticImpactReset>(impactSet, nq, nv));
  return P;
}

}  // namespace leap::test
