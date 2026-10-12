#include "leap/tasks/objective.hpp"

#include <cmath>
#include <stdexcept>

#include "leap/problem/costs.hpp"

namespace leap {

namespace {

std::vector<int> jointRows(const ObjectiveTermSpec& t, const RobotModel& m) {
  std::vector<int> idx;
  if (t.joints.empty()) return m.actuatedIdx();
  for (const std::string& name : t.joints) idx.push_back(m.jointIndex(name));  // throws on an unknown joint
  return idx;
}

}  // namespace

ObjectiveSpec objectiveFromYaml(const YamlDoc& d) {
  ObjectiveSpec spec;
  if (!d.has("objective")) return spec;
  spec.present = true;
  d.checkKeys("objective", {"quadrature", "scale", "terms"});
  spec.quadrature = d.getS("objective.quadrature", spec.quadrature);
  if (spec.quadrature != "clenshaw_curtis" && spec.quadrature != "sum")
    throw std::runtime_error("objective.quadrature must be 'clenshaw_curtis' or 'sum'");
  spec.scale = d.getD("objective.scale", spec.scale);
  if (!(spec.scale > 0.0) || !std::isfinite(spec.scale)) throw std::runtime_error("objective.scale must be positive");
  const int n = d.seqSize("objective.terms");
  if (n <= 0) throw std::runtime_error("objective.terms must be a non-empty list");
  for (int i = 0; i < n; ++i) {
    const std::string p = "objective.terms." + std::to_string(i);
    d.checkKeys(p, {"type", "weight", "weights", "joints", "target", "phases"});
    ObjectiveTermSpec t;
    t.type = d.requireS(p + ".type");
    t.weight = d.getD(p + ".weight", t.weight);
    if (!(t.weight >= 0.0) || !std::isfinite(t.weight)) throw std::runtime_error(p + ".weight must be finite and >= 0");
    t.weights = d.getVecD(p + ".weights");
    t.joints = d.getVecS(p + ".joints");
    t.target = d.getVecD(p + ".target");
    for (double ph : d.getVecD(p + ".phases")) t.phases.push_back(static_cast<int>(ph));
    if (t.type != "effort" && t.type != "joint_velocity" && t.type != "joint_acceleration" && t.type != "posture" &&
        t.type != "contact_force")
      throw std::runtime_error(p + ".type '" + t.type + "' is not one of effort | joint_velocity | joint_acceleration | posture | contact_force");
    if (t.type != "effort" && !t.weights.empty()) throw std::runtime_error(p + ".weights is for the effort term only");
    if (t.type == "effort" && !t.joints.empty()) throw std::runtime_error(p + ".joints: the effort term is over the actuated joints (use weights)");
    if (t.type == "contact_force" && !t.joints.empty()) throw std::runtime_error(p + ".joints: the contact force term has no joints");
    if (t.type != "posture" && !t.target.empty()) throw std::runtime_error(p + ".target is for the posture term only");
    spec.terms.push_back(t);
  }
  return spec;
}

std::shared_ptr<Cost> makeObjectiveTerm(const ObjectiveTermSpec& t, const RobotModel& m) {
  if (t.type == "effort") {
    if (t.weights.empty()) return std::make_shared<ControlEffortCost>(Eigen::VectorXd::Constant(m.nu(), t.weight));
    if (static_cast<int>(t.weights.size()) != m.nu())
      throw std::runtime_error("objective: effort weights must have nu = " + std::to_string(m.nu()) + " entries (actuated order)");
    Eigen::VectorXd w(m.nu());
    for (int j = 0; j < m.nu(); ++j) w(j) = t.weights[static_cast<size_t>(j)];
    return std::make_shared<ControlEffortCost>(w);
  }
  if (t.type == "contact_force") return std::make_shared<ContactForceCost>(t.weight);
  const std::vector<int> idx = jointRows(t, m);
  const int n = static_cast<int>(idx.size());
  const Eigen::VectorXd w = Eigen::VectorXd::Constant(n, t.weight);
  if (t.type == "joint_velocity") return std::make_shared<JointVelocityCost>(idx, w);
  if (t.type == "joint_acceleration") return std::make_shared<JointAccelerationCost>(idx, w);
  if (t.type == "posture") {
    Eigen::VectorXd target = Eigen::VectorXd::Zero(n);
    if (!t.target.empty()) {
      if (static_cast<int>(t.target.size()) != n) throw std::runtime_error("objective: posture target must have one entry per joint");
      for (int j = 0; j < n; ++j) target(j) = t.target[static_cast<size_t>(j)];
    }
    return std::make_shared<JointPostureCost>(idx, target, w);
  }
  throw std::runtime_error("objective: unknown term type '" + t.type + "'");
}

void attachObjective(TrajectoryProblem& P, const ObjectiveSpec& spec, const RobotModel& m) {
  if (!spec.present) return;
  std::vector<std::shared_ptr<Cost>> costs;
  for (const ObjectiveTermSpec& t : spec.terms) costs.push_back(makeObjectiveTerm(t, m));
  for (Element& e : P.elements) {
    e.costs.clear();
    for (size_t i = 0; i < spec.terms.size(); ++i) {
      const ObjectiveTermSpec& t = spec.terms[i];
      bool carries = t.phases.empty();
      for (int ph : t.phases)
        if (ph == e.phase) carries = true;
      if (!carries) continue;
      if (t.type == "contact_force" && e.contacts.active.empty()) continue;
      e.addCost(costs[i]);
    }
  }
}

std::string objectiveSummary(const ObjectiveSpec& spec) {
  if (!spec.present) return "builder costs";
  std::string s = spec.quadrature + " x " + std::to_string(spec.scale) + ":";
  for (const ObjectiveTermSpec& t : spec.terms) {
    s += " " + t.type + "(w=" + std::to_string(t.weight);
    if (!t.phases.empty()) {
      s += ", phases";
      for (int p : t.phases) s += " " + std::to_string(p);
    }
    s += ")";
  }
  return s;
}

}  // namespace leap
