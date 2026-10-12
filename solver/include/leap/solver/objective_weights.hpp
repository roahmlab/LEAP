#pragma once

// Per-node weights on the node costs that turn the node sum into the posed objective:
// `clenshaw_curtis` gives (T_e / 2) x the Clenshaw-Curtis weight of the node (the
// quadrature of int_0^T l(t) dt, exact for the element's degree), `sum` the plain node
// sum; `scale` multiplies every weight (the solver minimizes scale x f).

#include <string>
#include <vector>

#include "leap/problem/compiled_problem.hpp"

namespace leap {

std::vector<double> objectiveNodeWeights(const CompiledProblem& prob, const std::string& quadrature, double scale);

}  // namespace leap
