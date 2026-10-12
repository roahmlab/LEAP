#include "leap/solver/term_scales.hpp"

#include <algorithm>
#include <cmath>

namespace leap {

Eigen::VectorXd physicalTermScales(const CompiledProblem& prob, const RobotModel& model, const Eigen::VectorXd& qRef) {
  ModelDataPtr data = model.makeData();
  ModelEvalCache cache;
  model.initCache(cache);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(model.nv());
  model.evalNode(*data, qRef, zero, zero, kRnea, cache);
  const double weightN = std::abs(cache.tau(2));
  const double effortNm = model.actuatedEffortLimit().cwiseAbs().maxCoeff();
  const double vLim = std::max(model.actuatedVelocityLimit().cwiseAbs().mean(), 1.0);
  const double forceScale = 1.0 / std::max(weightN, 1.0);
  const double momentScale = 1.0 / std::max(0.1 * weightN, 1.0);
  const double effortScale = 1.0 / std::max(effortNm, 1.0);
  Eigen::VectorXd scales(prob.terms().size());
  for (int t = 0; t < prob.terms().size(); ++t) {
    const std::string name = TermTable::stripped(prob.terms().name(t));
    double scale = 1.0;
    if (name.rfind("dyn", 0) == 0 || name.rfind("ineq_lam_z", 0) == 0 || name.rfind("ineq_fric", 0) == 0 ||
        name.rfind("pin_lam_liftoff", 0) == 0 || name.rfind("pin_lam_zero", 0) == 0)
      scale = forceScale;
    else if (name.rfind("ineq_cop", 0) == 0 || name.rfind("ineq_tz", 0) == 0)
      scale = momentScale;
    else if (name.rfind("box_u", 0) == 0)
      scale = effortScale;
    else if (name.rfind("box_v", 0) == 0)
      scale = 1.0 / vLim;
    scales(t) = scale;
  }
  return scales;
}

}  // namespace leap
