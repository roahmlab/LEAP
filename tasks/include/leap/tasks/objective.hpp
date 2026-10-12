#pragma once

// The user-defined objective of a problem yaml: a list of cost terms from the catalogue
// (problem/costs.hpp), composed by the user term by term,
//
//   objective:
//     quadrature: clenshaw_curtis   # the node costs are summed with (T_e / 2) x the
//                                   # Clenshaw-Curtis weights, so f is the quadrature of
//                                   # int_0^T l(t) dt (default); `sum` = the plain node sum
//     scale: 1.0                    # the solver minimizes scale x f (same minimizer; the
//                                   # merit's relative signals see a smaller objective)
//     terms:
//       - {type: effort, weight: 1.0}                          # 1/2 w ||u||^2  (or weights: [per actuated joint])
//       - {type: joint_velocity, weight: 0.5, joints: [...]}   # 1/2 w sum_j v_j^2  (default: the actuated joints)
//       - {type: joint_acceleration, weight: 0.1}             # 1/2 w sum_j a_j^2
//       - {type: posture, weight: 1.0, joints: [...], target: [...]}   # 1/2 w sum_j (q_j - target_j)^2
//       - {type: contact_force, weight: 1e-4}                 # 1/2 w ||lam||^2 on the stance elements
//     # every term may carry `phases: [k, ...]`: the parent contact modes that carry it
//     # (default: every element)
//
// When the block is present it REPLACES the task builder's own cost knobs (mineffort,
// wvel, wacc, ...); absent, the builder's costs stand (the paper's problems). What the
// solver does with the objective is the recipe's business (`unified.objective`): in the
// paper's feasibility mode the costs are only reported; in the posed-objective mode
// phase C descends f + the augmented penalty and stops on a first-order certificate.

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/io/yaml_doc.hpp"
#include "leap/model/robot_model.hpp"
#include "leap/solver/objective_weights.hpp"
#include "leap/problem/trajectory_problem.hpp"

namespace leap {

struct ObjectiveTermSpec {
  std::string type;                 // effort | joint_velocity | joint_acceleration | posture | contact_force
  double weight = 1.0;
  std::vector<double> weights;      // effort: per actuated joint (overrides weight)
  std::vector<std::string> joints;  // joint_velocity / joint_acceleration / posture: joint names (default: actuated)
  std::vector<double> target;       // posture: one per joint (default: zeros)
  std::vector<int> phases;          // empty = every element
};

struct ObjectiveSpec {
  bool present = false;
  std::string quadrature = "clenshaw_curtis";  // clenshaw_curtis | sum
  double scale = 1.0;
  std::vector<ObjectiveTermSpec> terms;
};

// The `objective:` block of a problem yaml (present = false when absent). Validates keys.
ObjectiveSpec objectiveFromYaml(const YamlDoc& d);

// One catalogue cost from a term spec (throws std::runtime_error on an unknown joint / type).
std::shared_ptr<Cost> makeObjectiveTerm(const ObjectiveTermSpec& t, const RobotModel& m);

// Replace the elements' costs by the objective's terms (per the phases filters).
void attachObjective(TrajectoryProblem& P, const ObjectiveSpec& spec, const RobotModel& m);

std::string objectiveSummary(const ObjectiveSpec& spec);

}  // namespace leap
