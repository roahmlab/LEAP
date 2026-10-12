#pragma once

// Private pinocchio-facing side of RobotModel. Included only by the model sources,
// the exact-Hessian provider and the model tests.

#include <string>
#include <vector>

#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "leap/model/robot_model.hpp"

namespace leap {

// One body-fixed point: a contact frame, a monitor frame or a capsule endpoint.
struct BodyPoint {
  std::string name;
  int dim = 3;
  pinocchio::FrameIndex frame = 0;
  pinocchio::JointIndex joint = 0;
  pinocchio::SE3 placement = pinocchio::SE3::Identity();
};

using ConstraintModelVec = PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::RigidConstraintModel);
using ConstraintDataVec = PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::RigidConstraintData);

struct RobotModel::Impl {
  RobotDescription description;
  pinocchio::Model model;
  std::vector<std::string> jointNames;
  std::vector<BodyPoint> contacts;
  std::vector<BodyPoint> monitors;
  std::vector<CollisionCapsuleSpec> capsules;
  std::vector<BodyPoint> capsulePoints;  // [cap0.A, cap0.B, cap1.A, ...]
  Eigen::MatrixXd B;
  std::vector<int> actIdx, unactIdx;
  Eigen::VectorXd effortLimit, velocityLimitU;  // actuated order
  // motor model (expanded to nu entries)
  std::string motorType = "none";
  Eigen::VectorXd motorCoulomb, motorViscous, motorArmature, motorBias;
  double motorVelocitySmoothing = 0.05;
  // setup-time registries (complete before makeData())
  mutable std::vector<ContactSet> impactSets;
  mutable std::vector<ConstraintModelVec> impactModels;
  mutable std::vector<ContactSet> coneSets;
  mutable std::vector<ConstraintModelVec> coneModels;
};

class ModelData {
public:
  ModelData(const pinocchio::Model& m, size_t nContacts, size_t nImpactSets, size_t nConeSets);
  pinocchio::Data data;
  pinocchio::container::aligned_vector<pinocchio::Data> impactData;
  std::vector<ConstraintDataVec> contactData;
  pinocchio::container::aligned_vector<pinocchio::Data> coneData;
  std::vector<ConstraintDataVec> coneContactData;
  // evalNode scratch (allocation-free hot loop)
  PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext;
  Eigen::MatrixXd kvq, kaq, kav, kad;   // 6 x nv WORLD joint derivative buffers
  Eigen::MatrixXd J6;                   // 6 x nv point Jacobian scratch
  Eigen::MatrixXd dvel3;                // 3 x nv
  Eigen::MatrixXd corr;                 // nv x nv
  std::vector<Eigen::MatrixXd> Jl3;     // per contact: 3 x nv point Jacobian
  std::vector<Eigen::MatrixXd> Jwang3;  // per contact: 3 x nv world angular Jacobian
  std::vector<Eigen::Vector3d> p3;      // per contact: world contact point
};

}  // namespace leap
