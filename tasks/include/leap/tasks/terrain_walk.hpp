#pragma once

// The terrain walk (problem type terrain_walk): N steps across a field of rectangular
// pillars, for any legged robot the robot yaml can describe. The only boundary
// conditions are foot placements (where each contact is put down, plus a level sole for
// a flat foot) and rest at both ends; the pillars are both the surfaces the feet land
// on and, with their top faces lowered by the top inset, the keep-out boxes the capsule
// collision model must stay clear of.
//
// Mode chain for N steps (2N + 1 modes; a step that lifts every contact is a flight):
//   support_0  all contacts, starts at rest on the `start` placements
//     -- continuity (the step-1 swing feet let go) -->
//   [lift_1]   optional level vertical lift window (liftFrac of the swing)
//   swing_1    contacts = all \ swing(1); lands each swing foot on its placement
//   [approach_1] optional level vertical approach window (approachFrac)
//     -- plastic impact (or continuity when the landing is velocity-free and landCont) -->
//   support_1  ...  support_N ends at rest.
// Each mode is h-refined into sub-elements (uniform `split`, or the per-mode fractions
// of the adaptive mesh); the force pins of a liftoff / landing go to the sub-element
// that owns the event.
//
// Rows (names as the diagnostics print them): ineq_clear (monitors above the floor),
// dyn, pin_foot_<c> / pin_sole_<c> (stance pins and touchdown placements), pin_v
// (rest), vel_<c> / acc_<c> (no-slip), ineq_base_z, pin_land_v_<c> | pin_land_vxy_<c>
// | ineq_land_vz_<c> (touchdown), ineq_swing_z (mid-swing height), ineq_hover_z_<c>
// (windows), box_q, box_v, box_v_base, box_u (+ box_u_hull), box_att, box_posture,
// pin_planar, ineq_obstacle_box, ineq_selfcol, ineq_motor_envelope; on the force
// control points ineq_fric, ineq_cop, ineq_tz, ineq_lam_z, pin_lam_zero.

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/io/yaml_doc.hpp"
#include "leap/model/robot_description.hpp"
#include "leap/problem/compiled_problem.hpp"
#include "leap/problem/obstacles.hpp"
#include "leap/solver/pipeline.hpp"
#include "leap/tasks/objective.hpp"

namespace leap {

// One terrain block: an axis-aligned box standing on the floor.
struct TerrainPillar {
  std::string name;
  double cx = 0.0, cy = 0.0;  // footprint centre (m)
  double lx = 0.0, ly = 0.0;  // footprint FULL extents (m)
  double top = 0.0;           // top surface height (m, > 0)
};

// One commanded foot placement: the world position a contact frame is put down at.
struct FootPlacement {
  std::string contact;
  Eigen::Vector3d pos{0.0, 0.0, 0.0};
  double clearZ = -1.0;  // mid-swing height the foot must exceed (< 0 = none)
};

struct WalkStep {
  double Tswing = 0.4, Tsupport = 0.3;
  std::vector<FootPlacement> feet;  // the swing contacts and their targets
};

struct NominalAngle {
  std::string joint;
  double angle = 0.0;
};

struct TerrainWalkSpec {
  // --- gait ---
  double Tstart = 0.4;
  std::vector<FootPlacement> start;
  std::vector<WalkStep> steps;
  // --- contact model ---
  double frictionMu = 0.8, frictionEps = 1e-2;
  double lamMin = 5.0;
  double landVzMax = 0.5;
  bool stancePin = true;      // spot-pin every stance foot at every node
  bool zeroLand = true;       // zero touchdown velocity (impulse-free impact)
  bool zeroLandXY = false;    // tangential-only zero touchdown (ignored under zeroLand)
  bool landRamp = false;      // a landed foot's force starts at zero
  bool landCont = false;      // velocity-free touchdowns as continuity junctions
  // Junction ownership of the sub-element split (problem.junctionownership):
  //   "plain"  = the default map: an AllNodes row lands on BOTH sides of every interior
  //              junction, so a row reading only q, v is stated twice there (the junction
  //              quotient identifies the two nodes).
  //   "unique" = the generic rule of problem/mesh_refinement.hpp: such a row is stated
  //              once, on the left child's last node (what StaticFamily and the fence use).
  std::string junctionOwnership = "plain";
  double approachFrac = 0.0, liftFrac = 0.0;
  double copX = 0.08, copY = 0.03, gammaZ = 0.05;
  // --- keep-out ---
  double monitorZ = 0.0;
  struct SelfColPair {
    std::string a, b;
    double margin = -1.0;
  };
  double selfColMargin = 0.0;
  int selfColInterp = 0;
  std::vector<SelfColPair> selfColPairs;
  // --- hardware limits ---
  double uLimitScale = 1.0;
  struct ULimitJoint {
    std::string joint;
    double scale = 1.0;
  };
  std::vector<ULimitJoint> uLimitJoint;
  double velLimitScale = 1.0, qBoxScale = 1.0;
  double attMax = 0.5;
  std::vector<double> attBox;    // per-axis override of attMax (3 entries)
  std::vector<double> baseVBox;  // |base v_xyz| bounds (entries <= 0 unbounded)
  // --- posture ---
  struct PostureBound {
    std::string joint;
    double lo = -std::numeric_limits<double>::infinity();
    double hi = std::numeric_limits<double>::infinity();
  };
  std::vector<PostureBound> posture;
  double baseZMinFrac = 0.0;
  // --- objective ---
  double wvel = 0.0;  // joint-motion weight (dimensionless, robot-scaled)
  // --- poses / planarity ---
  std::vector<NominalAngle> nominal;
  std::vector<std::string> planarLock;
  // --- terrain ---
  std::vector<TerrainPillar> pillars;
  double terrainMargin = 0.03, terrainTopSlack = 0.02, terrainTopInset = 0.0;
  std::vector<std::string> terrainCapsules;
  int terrainInterp = 2;
  double windowPad = 0.0;
  // --- discretization ---
  int degree = 8, Nc = 7;
  bool elimU = false;  // inverse-dynamics torque elimination
  int elimUNq = 0;
  int vBoxNq = 0, attBoxNq = 0, qBoxNq = 0, uBoxNq = 0, uHullDeg = 0;
  double clearZScale = 1.0;
  int dynNq = 0;
  bool noSlipAcc = false;
  int splitPerMode = 2;
  std::vector<std::vector<double>> subFractions;  // per mode; empty = uniform split
  int lsqNq = 4, lsqModeMask = 7;
  double lsqRho = 0.0;
  bool precond = true;
  int stanceNoSlipNq = 3;
  // --- obstacle-family variants (the pipeline's pre-stage / A0) ---
  double obstacleSmoothEps = 0.0;
  bool obstacleRounded = false;
  double obstacleZScale = 1.0;
};

std::string validateTerrainWalkSpec(const TerrainWalkSpec& ws);
std::vector<std::string> terrainWalkSpecWarnings(const TerrainWalkSpec& ws);

struct TerrainWalkBuild {
  TrajectoryProblem problem;
  Eigen::VectorXd y0;
  Eigen::VectorXd qNominal;
  std::vector<Eigen::VectorXd> qEvent;  // IK stance pose at each gait event
  std::string summary;
  double ikWorstErr = 0.0, topInset = 0.0;
  int pillarsDropped = 0, obstacleRowsPerNode = 0;
  std::vector<std::vector<double>> subFractions;  // the fractions actually used (per mode)
};

// Throws std::invalid_argument when the spec is infeasible by construction and
// std::runtime_error on a bad robot / spec combination.
TerrainWalkBuild buildTerrainWalk(const TerrainWalkSpec& ws, std::shared_ptr<const RobotModel> model);

// The keep-out box of a pillar: its top face lowered by `inset`; false when nothing is left.
bool terrainKeepOutBox(const TerrainPillar& p, double inset, CuboidObstacle& out);
// Terrain surface height at (x, y): the tallest covering pillar, else the floor.
double terrainHeightAt(const std::vector<TerrainPillar>& pillars, double x, double y, double shrink = 0.0);

// Exact prolongation between two meshes of the same mode chain: every new element lies
// inside one old element of its mode; q, v, a, u are the old polynomials at the new
// nodes, the force polygon is the de Casteljau sub-polygon. Returns the measured error.
double prolongTerrainMesh(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to,
                          Eigen::VectorXd& yTo);

TerrainWalkSpec terrainSpecFromYaml(const YamlDoc& d);

class TerrainFamily final : public ProblemFamily {
public:
  TerrainFamily(const YamlDoc& d, const RobotDescription& rd, std::shared_ptr<const RobotModel> model);
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
  double defectWeight() const override { return ws_.lsqRho; }
  int numElements() const override { return static_cast<int>(current_.problem.elements.size()); }
  const std::vector<std::string>& warnings() const override { return warnings_; }
  std::string summary() const override { return current_.summary; }
  const TerrainWalkSpec& spec() const { return ws_; }
  const ObjectiveSpec& objective() const { return objective_; }

private:
  TerrainWalkSpec specWith(const std::vector<std::vector<double>>& fractions) const;
  TerrainWalkBuild rebuild(const TerrainWalkSpec& ws) const;  // build + the yaml objective

  std::string type_, name_;
  TerrainWalkSpec ws_;
  ObjectiveSpec objective_;
  std::shared_ptr<const RobotModel> model_;
  TerrainWalkBuild current_;
  Eigen::VectorXd y0_;  // the level-0 seed
  std::vector<std::vector<double>> fractions_;
  std::vector<std::string> warnings_;
};

}  // namespace leap
