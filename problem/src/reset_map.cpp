#include "leap/problem/reset_map.hpp"

#include <stdexcept>

namespace leap {

void ContinuityReset::value(const NodeQuantities& tail, const NodeQuantities& head, const RobotModel&, ModelData&,
                            Eigen::Ref<Eigen::VectorXd> r) const {
  r.head(nq_) = head.q - tail.q;
  r.tail(nv_) = head.v - tail.v;
}

void ContinuityReset::jacobian(const NodeQuantities&, const NodeQuantities&, const NodeDims& dTail,
                               const NodeDims& dHead, const RobotModel&, ModelData&, MatRef J) const {
  J.setZero();
  const int h0 = dTail.nP();
  for (int i = 0; i < nq_; ++i) {
    J(i, dTail.offQ() + i) = -1.0;
    J(i, h0 + dHead.offQ() + i) = 1.0;
  }
  for (int i = 0; i < nv_; ++i) {
    J(nq_ + i, dTail.offV() + i) = -1.0;
    J(nq_ + i, h0 + dHead.offV() + i) = 1.0;
  }
}

void PlasticImpactReset::value(const NodeQuantities& tail, const NodeQuantities& head, const RobotModel& model,
                               ModelData& data, Eigen::Ref<Eigen::VectorXd> r) const {
  r.head(nq_) = head.q - tail.q;
  Eigen::VectorXd vplus;
  model.impactVelocity(data, setId_, tail.q, tail.v, vplus);
  r.tail(nv_) = head.v - vplus;
}

void PlasticImpactReset::jacobian(const NodeQuantities& tail, const NodeQuantities&, const NodeDims& dTail,
                                  const NodeDims& dHead, const RobotModel& model, ModelData& data, MatRef J) const {
  J.setZero();
  const int h0 = dTail.nP();
  for (int i = 0; i < nq_; ++i) {
    J(i, dTail.offQ() + i) = -1.0;
    J(i, h0 + dHead.offQ() + i) = 1.0;
  }
  for (int i = 0; i < nv_; ++i) J(nq_ + i, h0 + dHead.offV() + i) = 1.0;
  Eigen::MatrixXd dvdq, dvdv;
  Eigen::VectorXd vplus;
  model.impactDerivatives(data, setId_, tail.q, tail.v, dvdq, dvdv, vplus);
  J.block(nq_, dTail.offQ(), nv_, nq_) = -dvdq;
  J.block(nq_, dTail.offV(), nv_, nv_) = -dvdv;
}

void PlasticImpactReset::addContractedHessian(const NodeQuantities& tail, const NodeQuantities&, const NodeDims& dTail,
                                              const NodeDims& dHead, HessianScratch& s,
                                              const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  (void)dHead;
  if (s.hessians == nullptr || s.model == nullptr)
    throw std::logic_error("PlasticImpactReset: exact contracted Hessian requested without a ModelHessians provider");
  const ContactSet& cs = s.model->impactSetContacts(setId_);
  s.z.resize(dTail.nP());
  tail.toVector(dTail, s.z);
  const Eigen::VectorXd wdv = -w.tail(nv_);
  const Eigen::VectorXd wlam = Eigen::VectorXd::Zero(cs.nLam());
  Eigen::MatrixXd Ht = Eigen::MatrixXd::Zero(dTail.nP(), dTail.nP());
  s.hessians->addImpactHessian(cs, dTail, s.z, wdv, wlam, Ht);
  H.topLeftCorner(dTail.nP(), dTail.nP()) += Ht;
}

}  // namespace leap
