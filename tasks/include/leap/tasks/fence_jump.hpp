#pragma once

// The fence jump (problem type jump_fence_v2): one rest-to-rest leap over an obstacle,
// three contact modes (push-off stance, flight, landing stance) joined by a continuity
// reset at liftoff and a plastic impact at touchdown, every limit from the URDF, the
// capsule model clear of the fence. Any legged robot the robot yaml can describe.
//
// Rows per mode (names as the diagnostics print them):
//   push-off  ineq_clear (monitors above the floor), dyn, pin_q_rest / pin_v_rest at the
//             first node, vel_<c> no-slip and pos_stance_<c> on the interior and last
//             nodes, ineq_base_z, ineq_vz_up at liftoff, no-slip at off-node points
//   flight    ineq_clear, dyn, ineq_base_z, pos_land_<c> (+ pos_flat_<c> for flat feet),
//             ineq_land_vz_<c>, ineq_impact_cone at the last node, ineq_clear_air at the
//             flight quarter points
//   landing   ineq_clear, dyn, ineq_base_z, vel_<c>, pos_stance_<c>, pin_q_rest /
//             pin_v_rest at the last node
//   every     box_q, box_v, the torque box (nodal box_u, or the Bernstein-coefficient
//   mode      certificate), box_att, pin_planar (planar robots), ineq_obstacle_box,
//             ineq_selfcol; the contact force rows on the Bezier control points
//             (ineq_fric, ineq_cop, ineq_tz, ineq_lam_z, the liftoff taper and pin)
// Each mode is split into `split` elements (stances graded toward the contact change).

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/io/yaml_doc.hpp"
#include "leap/model/robot_description.hpp"
#include "leap/problem/compiled_problem.hpp"
#include "leap/solver/pipeline.hpp"
#include "leap/tasks/objective.hpp"
#include "leap/solver/term_scales.hpp"
#include "leap/tasks/obstacle_scene.hpp"

namespace leap {

struct PoseAngle {
  std::string joint;
  double stand = 0.0;  // touchdown / rest pose downrange
  double squat = 0.0;  // rest pose at the start
};

struct FenceJumpSpec {
  // --- gait timing / geometry ---
  double Tstance = 0.5, Tair = 0.45, TstanceLand = 0.0;  // TstanceLand <= 0 -> Tstance
  double hopHeight = 0.18;                              // min liftoff vz = sqrt(2 g H)
  Eigen::Vector2d step{0.6, 0.0};                       // base (dx, dy) gained over the flight
  // --- contact model ---
  double frictionMu = 0.85, frictionEps = 1e-2;
  double lamMin = 5.0, lamMax = 0.0;
  double landVzMax = 0.5;
  double copX = 0.08, copY = 0.03, gammaZ = 0.05;
  double footClear = 0.05;  // mid-flight monitor clearance
  // --- objective (dimensionless, robot-scaled) ---
  bool minimizeEffort = true;
  double forceCostWeight = 0.0, jointVelocityWeight = 0.0, jointAccelerationWeight = 0.0;
  double baseAttitudeWeight = 0.0, baseAngularVelocityWeight = 0.0;
  // --- hardware limits (scale <= 0 turns a box off) ---
  double uLimitScale = 1.0, velLimitScale = 1.0, qBoxScale = 1.0;
  double attMax = 0.5, attVelMax = 0.0, liftoffAttMax = 0.0, liftoffOmegaMax = 0.0;
  // --- poses ---
  std::vector<PoseAngle> poses;
  double posePathSlack = -1.0;
  double landingXSlack = -1.0;
  bool holdLandingJoints = false;
  std::vector<std::string> planarLock;
  // --- obstacles ---
  ObstacleScene obstacles;
  // --- discretization ---
  int degree = 8, Nc = 7, splitPerMode = 2;
  std::vector<int> splits, degs, lsqnqs;  // per parent mode (empty = uniform)
  double grade = 1.0;                     // geometric grading of the stance elements
  bool gradeEnds = false;
  int lsqNq = 4, lsqModeMask = 7;
  double lsqRho = 0.0;
  bool lsqGauss = false;
  bool precond = true;
  int stanceNoSlipNq = 3;
  bool pinLiftoffForce = true;
  int dynamicsInterpNq = 0;
  // --- formulation flags ---
  bool strictRest = true, baseHeight = true, groundClearance = true, endpointAccel = false;
  bool linearSeed = false, rowOwnership = true, liftoffFullWrench = true;
  int seedMode = 1;  // 0 static hold, 1 piecewise squat / blend / stand, 3 linear flight chord
  double seedApex = -1.0;
  bool contactPosAll = false, stanceAcc = false;
  int qBoxNq = 0, vBoxNq = 0, swingPathNq = 0, swingProgressNq = 0;
  double liftoffXMax = -1.0;
  bool landingFootPin = true, touchdownPosture = false;
  double touchdownPostureBlend = 1.0;
  bool zeroLandXY = false;
  double landVxyMax = 0.0;
  bool impactCone = false, actuatedSymmetry = false;
  // Landing force ramp (problem.landramp): the whole wrench of every foot is pinned to
  // zero at the first control point of the first landing element and the normal-force
  // floor starts at the second, so the Bezier force grows from zero after the impact
  // (the terrain's `landramp`). The paper's fence problems have none: the floor
  // `lammin` holds from the first landing control point on and the impact is governed
  // by the impact cone and the touchdown-speed cap.
  bool landRamp = false;
  double restBoxEps = 0.0;
  bool uBernstein = false;
  bool deriveActuation = false;
  int uSamples = 0;
  // --- self-collision ---
  struct SelfColPair {
    std::string a, b;
    double margin = -1.0;
  };
  double selfColMargin = 0.0;
  int selfColInterp = 0;
  std::vector<SelfColPair> selfColPairs;
};

std::string validateFenceJumpSpec(const FenceJumpSpec& js);
std::vector<std::string> fenceJumpSpecWarnings(const FenceJumpSpec& js);

struct FenceJumpBuild {
  TrajectoryProblem problem;
  Eigen::VectorXd y0;  // the cold seed (duals zero)
  Eigen::VectorXd qStand, qSquat;
  std::string summary;
};

FenceJumpBuild buildFenceJump(const FenceJumpSpec& js, std::shared_ptr<const RobotModel> model);

// yaml -> spec (the `problem:`, `solver:` and `obstacles:` blocks).
FenceJumpSpec fenceSpecFromYaml(const YamlDoc& d);
ObstacleScene obstaclesFromYaml(const YamlDoc& d);

class FenceFamily final : public ProblemFamily {
public:
  FenceFamily(const YamlDoc& d, const RobotDescription& rd, std::shared_ptr<const RobotModel> model);
  const std::string& type() const override { return type_; }
  const std::string& name() const override { return name_; }
  std::shared_ptr<const RobotModel> model() const override { return model_; }
  TrajectoryProblem build() override;
  TrajectoryProblem obstacleVariant(const ObstacleVariant& v) override;
  Eigen::VectorXd seed(const CompiledProblem& prob) override;
  void refine(const std::vector<ElementSplit>& splits) override;
  double prolong(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to,
                 Eigen::VectorXd& yTo) override;
  Eigen::VectorXd oracleTermScales(const CompiledProblem& prob) const override;
  std::vector<double> objectiveNodeWeights(const CompiledProblem& prob) const override;
  double defectWeight() const override { return spec_.lsqRho; }
  int numElements() const override { return static_cast<int>(current_.elements.size()); }
  const std::vector<std::string>& warnings() const override { return warnings_; }
  std::string summary() const override { return build_.summary; }
  const FenceJumpSpec& spec() const { return spec_; }
  const ObjectiveSpec& objective() const { return objective_; }

private:
  std::string type_, name_;
  FenceJumpSpec spec_;
  ObjectiveSpec objective_;  // the yaml's `objective:` block (replaces the builder's costs when present)
  std::shared_ptr<const RobotModel> model_;
  FenceJumpBuild build_;      // level 0
  TrajectoryProblem current_; // the current mesh
  std::vector<int> pendingElements_;  // bisections since the last prolongation
  std::vector<double> pendingFractions_;
  std::vector<std::string> warnings_;
  bool level0_ = true;
};

}  // namespace leap
