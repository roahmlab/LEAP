#pragma once

// Exact contracted Hessians of the model-dependent terms, for the exact-Newton metric
// (the production recipe is Gauss-Newton and never asks for them).
//
// A constraint whose rows depend on the model asks a ModelHessians provider for
//     H += sum_i w_i d2 g_i / dz2
// on the node layout d (z = [q; v; a; lam; u]). PinocchioHessians answers everything in
// closed form (tests/test_hessian_fd.cpp pins every kernel against finite differences
// and, for the kinematic terms, against CppAD tapes recorded over pinocchio):
//   * the contact-folded dynamics residual g = tau(q, v, a) + loss(v) - sum_c Jc'(q) lam_c:
//     the four nv^3 tensors of pinocchio's second-order RNEA contracted with w, plus the
//     point kernel of the contact term over the contact joint's support chain (spatial
//     cross products of the world Jacobian columns; the moment rows of a flat-foot wrench
//     through the angular kernel);
//   * frame position (any point of the table: contacts, monitors, capsule endpoints),
//     frame velocity, frame acceleration (third-order point kinematics: the directional
//     derivative of the velocity kernel along v) and the log3 orientation rows of a
//     flat-foot contact;
//   * the plastic impact map (q, v-) -> [dv; Lambda] of a contact set: the adjoint of the
//     damped Delassus KKT system contracted with the second derivatives of its blocks
//     (used by the reset maps and by the impact-cone rows).
// The CppAD tapes of the earlier implementation are kept behind addDynamicsHessianTaped /
// addFrameHessianTaped for the tests (an independent derivation of the same numbers).
// One instance per thread (it owns pinocchio data, scratch and the tapes).

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "leap/model/contact_set.hpp"
#include "leap/model/robot_model.hpp"

namespace leap {

enum class FrameTerm { Position, Velocity, Acceleration, Orientation };

class ModelHessians {
public:
  virtual ~ModelHessians() = default;
  // H += sum_i w_i d2(gdyn_i - B u)/dz2 for the contact-folded dynamics residual.
  virtual void addDynamicsHessian(const NodeDims& d, const ContactSet& cs, const Eigen::VectorXd& z,
                                  const Eigen::VectorXd& w, Eigen::MatrixXd& H) = 0;
  // H += sum_i w_i d2(frame quantity_i)/dz2 for a point of the model's point table
  // (contacts, then monitors, then capsule endpoints). Velocity / acceleration /
  // orientation terms exist for contact frames only; w has the frame's row count
  // (min(dim, 3) for a position, dim for a velocity / acceleration, 3 for log3).
  virtual void addFrameHessian(FrameTerm term, int pointIdx, const NodeDims& d, const Eigen::VectorXd& z,
                               const Eigen::VectorXd& w, Eigen::MatrixXd& H) = 0;
  // The plastic impact of contact set `cs` at (q, v) = (z.q, z.v): the sticking impulse
  // Lambda (world-aligned axes, cs order, 3 rows per point contact and 6 per flat foot)
  // and the velocity jump dv = v+ - v solve the damped KKT system
  //     M dv = J' Lambda,   J dv + mu Lambda = -J v      (mu = the proximal damping).
  // H += sum_i wdv_i d2 dv_i / dz2 + sum_i wlam_i d2 Lambda_i / dz2 (q and v blocks).
  virtual void addImpactHessian(const ContactSet& cs, const NodeDims& d, const Eigen::VectorXd& z,
                                const Eigen::VectorXd& wdv, const Eigen::VectorXd& wlam, Eigen::MatrixXd& H) = 0;
};

class PinocchioHessians final : public ModelHessians {
public:
  explicit PinocchioHessians(const RobotModel& m);
  ~PinocchioHessians() override;
  PinocchioHessians(const PinocchioHessians&) = delete;
  PinocchioHessians& operator=(const PinocchioHessians&) = delete;

  void addDynamicsHessian(const NodeDims& d, const ContactSet& cs, const Eigen::VectorXd& z, const Eigen::VectorXd& w,
                          Eigen::MatrixXd& H) override;
  void addFrameHessian(FrameTerm term, int pointIdx, const NodeDims& d, const Eigen::VectorXd& z, const Eigen::VectorXd& w,
                       Eigen::MatrixXd& H) override;
  void addImpactHessian(const ContactSet& cs, const NodeDims& d, const Eigen::VectorXd& z, const Eigen::VectorXd& wdv,
                        const Eigen::VectorXd& wlam, Eigen::MatrixXd& H) override;

  // The same quantities through CppAD tapes recorded over the pinocchio algorithms
  // (dynamics with the contact wrenches as external forces; point position / velocity /
  // acceleration over [q; v; a]). Used by the tests as an independent oracle.
  void addDynamicsHessianTaped(const NodeDims& d, const ContactSet& cs, const Eigen::VectorXd& z, const Eigen::VectorXd& w,
                               Eigen::MatrixXd& H);
  void addFrameHessianTaped(FrameTerm term, int contactIdx, const NodeDims& d, const Eigen::VectorXd& z,
                            const Eigen::VectorXd& w, Eigen::MatrixXd& H);
  // How many calls so far went to the CppAD tapes.
  long tapeCalls() const;
  // The proximal damping mu of the impact map (the value RobotModel's impulse solve uses).
  static double impactDamping();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace leap
