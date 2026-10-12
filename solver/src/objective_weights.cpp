#include "leap/solver/objective_weights.hpp"

#include <stdexcept>

#include "leap/spectral/cgl_grid.hpp"

namespace leap {

std::vector<double> objectiveNodeWeights(const CompiledProblem& prob, const std::string& quadrature, double scale) {
  std::vector<double> w(prob.nodes().size(), scale);
  if (quadrature == "sum") return w;
  if (quadrature != "clenshaw_curtis") throw std::runtime_error("objective quadrature must be 'clenshaw_curtis' or 'sum'");
  std::vector<Eigen::VectorXd> cc(prob.layout().elem.size());
  for (size_t e = 0; e < cc.size(); ++e) cc[e] = clenshawCurtisWeights(prob.grid(static_cast<int>(e)).degree());
  for (size_t i = 0; i < prob.nodes().size(); ++i) {
    const NodeInfo& n = prob.nodes()[i];
    const ElementLayout& L = prob.layout().elem[static_cast<size_t>(n.elem)];
    w[i] = scale * 0.5 * L.T * cc[static_cast<size_t>(n.elem)](n.node);
  }
  return w;
}

}  // namespace leap
