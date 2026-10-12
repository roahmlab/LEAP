#pragma once

// The contact-sequence problem (problem type `contact_sequence`): Problem 1 of the paper
// posed from a yaml, for any legged robot the robot yaml can describe. The user
// prescribes the mode sequence (the active contact set and the duration of every mode),
// where each contact stands while it is active, the rest poses at both ends and the
// contact model; the builder attaches the paper's rows, splits every mode into graded
// elements and builds the cold seed. Jumps, steps, bounds, flights, partial supports:
// anything a sequence of contact sets expresses. It is the generic counterpart of the
// two benchmark builders (fence_jump, terrain_walk) without their task-specific devices
// (obstacle continuation, terrain pillars, swing windows, torque elimination).
//
// Rows per mode (names as the diagnostics print them):
//   every mode      dyn at every node; box_q, box_v, box_u (nodal, or the Bernstein
//                   coefficient certificate with `ubern`), box_att, pin_planar;
//                   ineq_clear (the monitor frames above the floor; a stance foot's own
//                   monitor is exempt), ineq_base_z, ineq_selfcol, the static obstacle
//                   rows; the effort cost; the least-squares dynamics defect
//   stance contact  pos_stance_<c> (+ pos_flat_<c> for a flat foot) and vel_<c> at the
//                   interior and last nodes and vel_<c> at `noslipnq` off-node points;
//                   on the Bezier control points ineq_fric, ineq_cop, ineq_tz and the
//                   normal-force floor ineq_lam_z
//   liftoff         a contact that is inactive in the next mode: its floor tapers to zero
//                   over the last element and its wrench is pinned to zero at the last
//                   control point (pin_lam_liftoff); when the next mode has no contact at
//                   all, ineq_vz_up (base vz >= sqrt(2 g hop)) at the last node
//   touchdown       a contact that becomes active in the next mode: pos_land_<c>
//                   (+ pos_flat_<c>), ineq_land_vz_<c> and, with `impactcone`, the
//                   friction-cone admissibility of the plastic impulse at the last node
//                   of the mode before; ineq_swing_z (`footclear`) for the swing feet at
//                   the quarter points of the mode (a point condition: the fine-grid
//                   judge and the relocation leave rows of that name alone, as on the
//                   terrain walk)
//   the ends        pin_q_rest / pin_v_rest at t = 0 (`rest_start`) and t = T (`rest_end`)
// Resets: a plastic impact of the new contact set where a contact becomes active,
// continuity otherwise (a pure liftoff is smooth).

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/io/yaml_doc.hpp"
#include "leap/model/robot_model.hpp"
#include "leap/problem/compiled_problem.hpp"
#include "leap/solver/pipeline.hpp"
#include "leap/tasks/obstacle_scene.hpp"

namespace leap {

struct ContactPlacement {
  std::string contact;
  Eigen::Vector3d pos{0.0, 0.0, 0.0};  // world position of the contact frame (m)
};

struct JointAngle {
  std::string joint;
  double angle = 0.0;
};

// One contact mode of the sequence.
struct ContactSequenceMode {
  std::string label;                   // optional (archive, diagnostics)
  std::vector<std::string> contacts;   // the active contacts (empty = flight)
  double T = 0.5;                      // duration (s)
  std::vector<ContactPlacement> feet;  // placements of contacts ACTIVE in this mode (default: unchanged)
  Eigen::Vector3d shift{0.0, 0.0, 0.0};  // added to the previous placement of every contact that LANDS here
};

struct ContactSequenceSpec {
  // --- the sequence ---
  std::vector<ContactSequenceMode> modes;
  std::vector<ContactPlacement> start;  // placements during the first mode (default: the FK of pose_start, lowest foot on z = 0)
  // --- the ends ---
  std::vector<JointAngle> poseStart;    // joint angles by name; the base is derived from the placements
  std::vector<JointAngle> poseEnd;      // empty: the end is at rest (zero velocity) in a free pose
  bool restStart = true, restEnd = true;
  // --- contact model ---
  double frictionMu = 0.8, frictionEps = 1e-2;
  double lamMin = 5.0;      // normal-force floor per contact in stance (N)
  double landVzMax = 0.5;   // touchdown speed cap (m/s; <= 0 off)
  double copX = 0.08, copY = 0.03, gammaZ = 0.05;  // flat-foot CoP box and yaw-torque bound
  bool impactCone = false;  // friction-cone admissibility of the plastic impulse
  bool landRamp = false;    // a landed contact's wrench starts at zero (pin at the first control point)
  // --- swings and flights ---
  double hop = 0.0;        // a pure flight: minimum liftoff base vz = sqrt(2 g hop) (<= 0 off)
  double footClear = 0.03; // swing feet above this height at the quarter points of the swing (<= 0 off)
  // --- hardware boxes (a scale <= 0 turns the box off) ---
  double uLimitScale = 1.0, velLimitScale = 1.0, qBoxScale = 1.0;
  double attMax = 0.5;     // |base roll, pitch, yaw| bound (rad; <= 0 off)
  bool uBernstein = false; // torque box on the Bernstein coefficients (certified between the nodes)
  bool baseHeight = true, groundClearance = true, minimizeEffort = true;
  std::vector<std::string> planarLock;  // joints pinned to zero at every node (sagittal robots)
  // --- collision ---
  struct SelfColPair {
    std::string a, b;
    double margin = -1.0;
  };
  double selfColMargin = 0.0;
  int selfColInterp = 0;
  std::vector<SelfColPair> selfColPairs;
  ObstacleScene obstacles;  // static spheres and cuboids (no continuation pre-stage)
  // --- transcription ---
  int degree = 4, Nc = 4;      // polynomial degree per element, force control points
  int splitPerMode = 4;        // elements per mode
  double grade = 0.6;          // geometric grading of a stance mode's elements toward its contact change
  int lsqNq = 6;               // least-squares dynamics-defect points per element
  double lsqRho = 0.0;         // the defect weight and the unit of every penalty (required > 0)
  bool precond = true;
  int stanceNoSlipNq = 3;      // off-node no-slip points per stance element
  // --- the seed ---
  double seedApex = -1.0;      // mid-swing / mid-flight lift of the seed (m; < 0 = hop, else footclear)
};

std::string validateContactSequenceSpec(const ContactSequenceSpec& cs);
std::vector<std::string> contactSequenceSpecWarnings(const ContactSequenceSpec& cs);

struct ContactSequenceBuild {
  TrajectoryProblem problem;
  Eigen::VectorXd y0;                   // the cold seed (duals zero)
  std::vector<Eigen::VectorXd> qEvent;  // the pose at the start of every mode and at the end
  double ikWorstErr = 0.0;
  std::string summary;
};

ContactSequenceBuild buildContactSequence(const ContactSequenceSpec& cs, std::shared_ptr<const RobotModel> model);

// yaml -> spec (the `problem:`, `solver:` and `obstacles:` blocks).
ContactSequenceSpec contactSequenceSpecFromYaml(const YamlDoc& d);

// The family `leap_solve` runs for problem.type contact_sequence: the built problem
// (with the yaml's `objective:` block attached when present) as a StaticFamily.
std::unique_ptr<ProblemFamily> makeContactSequenceFamily(const YamlDoc& d, std::shared_ptr<const RobotModel> model);

}  // namespace leap
