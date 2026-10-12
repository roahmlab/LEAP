// Spectral layer: CGL nodes, differentiation matrix, barycentric interpolation,
// Clenshaw-Curtis and Gauss-Legendre quadrature, Bernstein basis and de Casteljau
// subdivision. Every check is against a closed-form polynomial identity.

#include <cmath>
#include <cstdio>

#include "leap/spectral/bernstein.hpp"
#include "leap/spectral/cgl_grid.hpp"
#include "test_check.hpp"

namespace {

double poly(double t) { return 1.0 + 2.0 * t - 3.0 * t * t + 0.5 * t * t * t; }
double dpoly(double t) { return 2.0 - 6.0 * t + 1.5 * t * t; }

}  // namespace

int main() {
  for (int deg : {1, 2, 4, 8, 16}) {
    leap::CglGrid g(deg);
    CHECK(g.numNodes() == deg + 1, "node count");
    CHECK(g.nodes()(0) == -1.0 && g.nodes()(deg) == 1.0, "end nodes are exactly -1 and +1");
    for (int k = 1; k <= deg; ++k) CHECK(g.nodes()(k) > g.nodes()(k - 1), "nodes ascend");
    if (deg >= 3) {
      // D differentiates the cubic exactly; the interpolant reproduces it.
      Eigen::VectorXd f(deg + 1);
      for (int k = 0; k <= deg; ++k) f(k) = poly(g.nodes()(k));
      Eigen::VectorXd df = g.D() * f;
      for (int k = 0; k <= deg; ++k) CHECK_NEAR(df(k), dpoly(g.nodes()(k)), 1e-11, "D of a cubic");
      for (double tau : {-0.93, -0.5, 0.1, 0.77}) {
        const Eigen::VectorXd w = g.interpWeights(tau);
        CHECK_NEAR(w.sum(), 1.0, 1e-13, "interpolation weights sum to one");
        CHECK_NEAR(w.dot(f), poly(tau), 1e-12, "barycentric interpolation of a cubic");
        CHECK_NEAR(g.interpWeightsDerivative(tau).dot(f), dpoly(tau), 1e-10, "derivative weights");
      }
      const Eigen::VectorXd wNode = g.interpWeights(g.nodes()(1));
      CHECK(wNode(1) == 1.0 && wNode.sum() == 1.0, "one-hot weights on a node");
      // Clenshaw-Curtis integrates the cubic exactly on [-1, 1]: int = 2 - 2 = 0.
      const Eigen::VectorXd cc = leap::clenshawCurtisWeights(deg);
      CHECK_NEAR(cc.sum(), 2.0, 1e-13, "Clenshaw-Curtis weights sum to 2");
      CHECK_NEAR(cc.dot(f), 2.0 - 2.0, 1e-12, "Clenshaw-Curtis integrates a cubic");
    }
    // time grid
    const Eigen::VectorXd t = g.timeGrid(0.7);
    CHECK(t(0) == 0.0 && std::abs(t(deg) - 0.7) < 1e-15, "time grid spans [0, T]");
  }

  // Gauss-Legendre: n points integrate degree 2n-1 exactly.
  for (int n : {1, 2, 3, 5, 8, 12}) {
    Eigen::VectorXd x, w;
    leap::gaussLegendre(n, x, w);
    CHECK_NEAR(w.sum(), 2.0, 1e-13, "GL weights sum to 2");
    for (int k = 1; k < n; ++k) CHECK(x(k) > x(k - 1), "GL nodes ascend");
    const int p = 2 * n - 1;
    double num = 0.0;
    for (int k = 0; k < n; ++k) num += w(k) * std::pow(x(k), p - 1);  // even power p-1
    const double exact = 2.0 / static_cast<double>(p);               // int x^(p-1), p-1 even
    CHECK_NEAR(num, exact, 1e-12, "GL integrates x^(2n-2) exactly");
  }

  // Bernstein: partition of unity, node evaluation, de Casteljau exactness.
  for (int ctrlDeg : {1, 3, 6}) {
    leap::CglGrid g(4);
    const Eigen::MatrixXd E = leap::bernsteinBasisAtDegree(ctrlDeg, g);
    CHECK(E.rows() == 5 && E.cols() == ctrlDeg + 1, "forward basis shape");
    for (int k = 0; k < 5; ++k) CHECK_NEAR(E.row(k).sum(), 1.0, 1e-14, "basis rows sum to one");
    CHECK(E(0, 0) == 1.0 && E(4, ctrlDeg) == 1.0, "end nodes pick the end control points");
    for (double s : {0.0, 0.3, 0.5, 1.0}) {
      const Eigen::VectorXd b = leap::bernsteinBasisAt(ctrlDeg, s);
      CHECK_NEAR(b.sum(), 1.0, 1e-14, "basis at s sums to one");
      CHECK(b.minCoeff() >= 0.0, "basis nonnegative");
    }
    const Eigen::VectorXd sigma = leap::bernsteinColumnScaling(E);
    for (int j = 0; j <= ctrlDeg; ++j)
      CHECK_NEAR(sigma(j), 1.0 / std::sqrt(1.0 + E.col(j).squaredNorm()), 1e-15, "sigma");
    // Subdivide a random curve at s* = 0.37 and compare both halves to the parent curve.
    Eigen::MatrixXd C = Eigen::MatrixXd::Random(3, ctrlDeg + 1);
    Eigen::MatrixXd L, R;
    leap::deCasteljauSplit(C, 0.37, L, R);
    for (double s : {0.0, 0.2, 0.5, 0.8, 1.0}) {
      const Eigen::Vector3d parentLeft = C * leap::bernsteinBasisAt(ctrlDeg, 0.37 * s);
      const Eigen::Vector3d child = L * leap::bernsteinBasisAt(ctrlDeg, s);
      CHECK_NEAR((parentLeft - child).norm(), 0.0, 1e-13, "left child reproduces the parent curve");
      const Eigen::Vector3d parentRight = C * leap::bernsteinBasisAt(ctrlDeg, 0.37 + 0.63 * s);
      const Eigen::Vector3d childR = R * leap::bernsteinBasisAt(ctrlDeg, s);
      CHECK_NEAR((parentRight - childR).norm(), 0.0, 1e-13, "right child reproduces the parent curve");
    }
  }
  return leap::test::finish("test_spectral");
}
