#pragma once

// Unit-free per-term scales for the relocation oracle: force rows in units of the
// robot's weight, moment rows of a tenth of it, torque rows of the effort limit, speed
// rows of the mean speed limit, everything else 1.

#include <Eigen/Dense>

#include "leap/model/robot_model.hpp"
#include "leap/problem/compiled_problem.hpp"

namespace leap {

Eigen::VectorXd physicalTermScales(const CompiledProblem& prob, const RobotModel& model, const Eigen::VectorXd& qRef);

}  // namespace leap
