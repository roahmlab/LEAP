#pragma once

// Bernstein basis and Bezier curves on [-1, 1] (parameter s = (tau + 1)/2 in [0, 1]).
//
// A contact force on an element is a Bezier curve with Nc = degree + 1 control points
// C_0..C_degree:   lam(tau) = sum_j B_j(s) C_j.   The basis is nonnegative and sums to
// one, so any convex constraint satisfied by every control point holds along the whole
// curve (the certificate the friction cone and the force floor rely on).

#include <Eigen/Dense>

#include "leap/spectral/cgl_grid.hpp"

namespace leap {

// Basis values b(j) = C(degree, j) s^j (1 - s)^(degree - j), j = 0..degree.
Eigen::VectorXd bernsteinBasisAt(int degree, double s);

// Basis of control degree `ctrlDeg` sampled at the CGL nodes of `grid`:
// E(k, j) = B_j(s_k), s_k = (tau_k + 1)/2, shape numNodes x (ctrlDeg + 1). The node
// value of a Bezier curve is  value(k) = sum_j E(k, j) C_j.
Eigen::MatrixXd bernsteinBasisAtDegree(int ctrlDeg, const CglGrid& grid);

// Column scaling sigma(j) = 1 / sqrt(1 + ||E(:, j)||^2) of a forward basis E. The
// decision variable of a force control point is scaled by sigma so that the lifted
// Jacobian columns have comparable norms (a diagonal preconditioner of the force
// variables; see the layout documentation).
Eigen::VectorXd bernsteinColumnScaling(const Eigen::MatrixXd& E);

// de Casteljau subdivision at parameter s in [0, 1]: the control polygon of the SAME
// curve restricted to [0, s] (left) and [s, 1] (right). Control points are the columns
// of C (dim x (degree + 1)).
void deCasteljauSplit(const Eigen::MatrixXd& C, double s, Eigen::MatrixXd& left,
                      Eigen::MatrixXd& right);

}  // namespace leap
