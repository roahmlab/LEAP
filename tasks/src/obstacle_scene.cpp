#include "leap/tasks/obstacle_scene.hpp"

#include <stdexcept>

namespace leap {

std::vector<int> enforcedCapsules(const RobotModel& m, const ObstacleScene& scene) {
  if (m.nCapsules() == 0)
    throw std::runtime_error("obstacles: the robot declares no collision_capsules, so there is no collision model");
  if (!scene.capsuleMargins.empty() && scene.capsuleMargins.size() != scene.capsules.size())
    throw std::runtime_error("obstacles: capsule_margins must be parallel to an explicit capsules list");
  std::vector<int> idx;
  for (const std::string& nm : scene.capsules) idx.push_back(m.capsuleIndex(nm));
  return idx;
}

std::shared_ptr<CapsuleSphereConstraint> makeSphereObstacleConstraint(const RobotModel& m, const ObstacleScene& scene) {
  if (scene.spheres.empty()) return nullptr;
  return std::make_shared<CapsuleSphereConstraint>(m, scene.spheres, enforcedCapsules(m, scene), scene.margin,
                                                   "ineq_obstacle", scene.capsuleMargins);
}

std::shared_ptr<CapsuleCuboidConstraint> makeCuboidObstacleConstraint(const RobotModel& m, const ObstacleScene& scene) {
  if (scene.boxes.empty()) return nullptr;
  return std::make_shared<CapsuleCuboidConstraint>(m, scene.boxes, enforcedCapsules(m, scene), scene.margin,
                                                   "ineq_obstacle_box", scene.exactCuboidOutside, scene.capsuleMargins,
                                                   scene.smoothEps);
}

void attachObstacles(const RobotModel& m, const ObstacleScene& scene, TrajectoryProblem& problem) {
  if (scene.empty()) return;
  std::vector<std::shared_ptr<Constraint>> cons;
  std::shared_ptr<CapsuleSphereConstraint> sph = makeSphereObstacleConstraint(m, scene);
  if (sph) cons.push_back(sph);
  std::shared_ptr<CapsuleCuboidConstraint> box = makeCuboidObstacleConstraint(m, scene);
  if (box) cons.push_back(box);
  const std::vector<double> taus = scene.interpPoints > 0 ? uniformInteriorTaus(scene.interpPoints) : std::vector<double>{};
  for (Element& e : problem.elements) {
    for (const std::shared_ptr<Constraint>& c : cons) {
      e.addNode(c, Where::AllNodes);
      if (!taus.empty()) e.addOffNode(c, taus);
    }
  }
}

}  // namespace leap
