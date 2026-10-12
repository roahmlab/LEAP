#pragma once

// The formulation interface: one constraint term, one cost term, evaluated at a node.
//
// Both are written against the per-node PHYSICAL vector z = [q; v; a; lam; u]
// (NodeQuantities, sizes in NodeDims) and the ModelEvalCache the RobotModel filled for
// that node. They know nothing about how z maps onto decision variables (nodal values,
// Bezier force control points, eliminated torques): that is the transcription's job.
//
// A constraint declares
//   name()        the TERM it belongs to (one penalty per term; diagnostics group by it)
//   rows()        how many residual rows it emits
//   sense()       g = 0 (Equality) or g <= 0 (Inequality)
//   modelNeeds()  which model quantities it reads (ModelNeeds bit mask)
// and supplies value(), jacobian() (rows x nP, dense) and optionally the contracted
// Hessian sum_i w_i d2 g_i / dz2 for the exact-Newton metric (the default adds nothing,
// which is the Gauss-Newton treatment the production recipe uses).

#include <string>

#include <Eigen/Dense>

#include "leap/model/contact_set.hpp"
#include "leap/model/model_hessians.hpp"
#include "leap/model/robot_model.hpp"

namespace leap {

enum class Sense { Equality, Inequality };

// Writable view of a row block of a larger dense column-major matrix.
using MatRef = Eigen::Ref<Eigen::MatrixXd, 0, Eigen::OuterStride<>>;

// Non-owning views of one node's physical quantities.
struct NodeQuantities {
  Eigen::Ref<const Eigen::VectorXd> q, v, a, lam, u;

  // Coordinate i of the conceptual vector [q; v; a; lam; u].
  double at(const NodeDims& d, int i) const {
    if (i < d.offV()) return q(i);
    if (i < d.offA()) return v(i - d.offV());
    if (i < d.offLam()) return a(i - d.offA());
    if (i < d.offU()) return lam(i - d.offLam());
    return u(i - d.offU());
  }
  void toVector(const NodeDims& d, Eigen::Ref<Eigen::VectorXd> z) const {
    z.segment(d.offQ(), d.nq) = q;
    z.segment(d.offV(), d.nv) = v;
    z.segment(d.offA(), d.na) = a;
    if (d.nlam > 0) z.segment(d.offLam(), d.nlam) = lam;
    if (d.nu > 0) z.segment(d.offU(), d.nu) = u;
  }
};

// Per-thread scratch handed to the Hessian pass (the shared ModelEvalCache belongs to
// another node by then, so terms recompute what they need from z / q).
struct HessianScratch {
  const RobotModel* model = nullptr;
  ModelData* data = nullptr;
  ModelHessians* hessians = nullptr;  // null under Gauss-Newton
  Eigen::VectorXd z;                   // packed [q; v; a; lam; u]
  Eigen::VectorXd w3;
  Eigen::MatrixXd Jp;
  Eigen::VectorXd gq;
  std::vector<Eigen::Vector3d> capPos;
  std::vector<Eigen::MatrixXd> capJ;
};

class Constraint {
public:
  virtual ~Constraint() = default;
  virtual const std::string& name() const = 0;
  virtual int rows(const NodeDims& d) const = 0;
  virtual Sense sense() const = 0;
  virtual unsigned modelNeeds() const { return 0; }
  virtual void value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c,
                     Eigen::Ref<Eigen::VectorXd> g) const = 0;
  // J is rows x d.nP(), dense; implementations zero it first.
  virtual void jacobian(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c, MatRef J) const = 0;
  // H (nP x nP) += sum_i w_i d2 g_i / dz2. Default: nothing (Gauss-Newton).
  virtual void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                    const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
    (void)z;
    (void)d;
    (void)s;
    (void)w;
    (void)H;
  }
  // True when the term reads only the contact force (it can then be enforced on the
  // Bezier control points, where convexity certifies it along the whole curve).
  virtual bool readsOnlyForce() const { return false; }
  // True when the term reads a, lam or u: fields that are one-sided at an element
  // junction (a continuity reset equates q and v only), so a row of this term
  // attached at a junction lives on BOTH junction nodes. Rows reading only q and v
  // get a single owner (the left element's last node).
  virtual bool readsOneSidedFields() const { return false; }
};

class Cost {
public:
  virtual ~Cost() = default;
  virtual unsigned modelNeeds() const { return 0; }
  virtual double value(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c) const = 0;
  // g (size nP): implementations zero it first.
  virtual void gradient(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c,
                        Eigen::Ref<Eigen::VectorXd> g) const = 0;
  // H (nP x nP) += d2 l / dz2 (exact for the quadratic costs of the catalogue).
  virtual void addHessian(const NodeQuantities& z, const NodeDims& d, const ModelEvalCache& c,
                          Eigen::MatrixXd& H) const = 0;
};

}  // namespace leap
