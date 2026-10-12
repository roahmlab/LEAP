#pragma once

// ObstacleScene: the static obstacles (spheres and cuboids) a problem's robot must keep
// its capsule collision model clear of, and the one call that attaches them to every
// element of a problem (at the nodes, and at `interpPoints` off-node points).

#include <memory>
#include <string>
#include <vector>

#include "leap/problem/obstacles.hpp"
#include "leap/problem/trajectory_problem.hpp"

namespace leap {

struct ObstacleScene {
  std::vector<SphereObstacle> spheres;
  std::vector<CuboidObstacle> boxes;
  double margin = 0.02;
  std::vector<std::string> capsules;      // capsule names to enforce; empty = all
  std::vector<double> capsuleMargins;     // optional, parallel to `capsules`
  int interpPoints = 0;                   // off-node enforcement points per element
  bool exactCuboidOutside = false;        // rounded capsule-box distance outside the box
  double smoothEps = 0.0;                 // > 0: log-sum-exp surrogate of the cuboid certificate
  bool empty() const { return spheres.empty() && boxes.empty(); }
};

std::vector<int> enforcedCapsules(const RobotModel& m, const ObstacleScene& scene);
std::shared_ptr<CapsuleSphereConstraint> makeSphereObstacleConstraint(const RobotModel& m, const ObstacleScene& scene);
std::shared_ptr<CapsuleCuboidConstraint> makeCuboidObstacleConstraint(const RobotModel& m, const ObstacleScene& scene);
// Attach the scene to every element (spheres first, then cuboids).
void attachObstacles(const RobotModel& m, const ObstacleScene& scene, TrajectoryProblem& problem);

}  // namespace leap
