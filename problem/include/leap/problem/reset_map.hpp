#pragma once

// Reset maps join the LAST node of one element (tail, its final time) to the FIRST node
// of the next (head, its initial time): nq + nv rows, configuration continuity plus a
// velocity map. The Jacobian spans [z_tail; z_head]; the head columns start at nP_tail.

#include <string>

#include "leap/problem/constraint.hpp"

namespace leap {

enum class ResetKind { Continuity, PlasticImpact };

class ResetMap {
public:
  virtual ~ResetMap() = default;
  int rows() const { return nq_ + nv_; }
  virtual ResetKind kind() const = 0;
  const char* kindName() const { return kind() == ResetKind::Continuity ? "continuity" : "plastic_impact"; }
  virtual void value(const NodeQuantities& tail, const NodeQuantities& head, const RobotModel& model,
                     ModelData& data, Eigen::Ref<Eigen::VectorXd> r) const = 0;
  // J is rows() x (nP_tail + nP_head); implementations zero it first.
  virtual void jacobian(const NodeQuantities& tail, const NodeQuantities& head, const NodeDims& dTail,
                        const NodeDims& dHead, const RobotModel& model, ModelData& data, MatRef J) const = 0;
  // H ((nP_tail + nP_head) square) += sum_i w_i d2 r_i / d[z_tail; z_head]2 for the
  // exact-Newton metric. Default: nothing (a continuity reset is linear).
  virtual void addContractedHessian(const NodeQuantities& tail, const NodeQuantities& head, const NodeDims& dTail,
                                    const NodeDims& dHead, HessianScratch& s, const Eigen::Ref<const Eigen::VectorXd>& w,
                                    Eigen::MatrixXd& H) const {
    (void)tail;
    (void)head;
    (void)dTail;
    (void)dHead;
    (void)s;
    (void)w;
    (void)H;
  }

protected:
  ResetMap(int nq, int nv) : nq_(nq), nv_(nv) {}
  int nq_, nv_;
};

// Smooth transition (liftoff, element junction): r = [q+ - q-; v+ - v-].
class ContinuityReset final : public ResetMap {
public:
  ContinuityReset(int nq, int nv) : ResetMap(nq, nv) {}
  ResetKind kind() const override { return ResetKind::Continuity; }
  void value(const NodeQuantities& tail, const NodeQuantities& head, const RobotModel&, ModelData&,
             Eigen::Ref<Eigen::VectorXd> r) const override;
  void jacobian(const NodeQuantities&, const NodeQuantities&, const NodeDims& dTail, const NodeDims& dHead,
                const RobotModel&, ModelData&, MatRef J) const override;
};

// Plastic impact: r = [q+ - q-; v+ - Pi(q-) v-], v+ the velocity after the sticking
// impact of the registered impact set (RobotModel::ensureImpactSet).
class PlasticImpactReset final : public ResetMap {
public:
  PlasticImpactReset(int impactSetId, int nq, int nv) : ResetMap(nq, nv), setId_(impactSetId) {}
  ResetKind kind() const override { return ResetKind::PlasticImpact; }
  int impactSetId() const { return setId_; }
  void value(const NodeQuantities& tail, const NodeQuantities& head, const RobotModel& model, ModelData& data,
             Eigen::Ref<Eigen::VectorXd> r) const override;
  void jacobian(const NodeQuantities& tail, const NodeQuantities& head, const NodeDims& dTail, const NodeDims& dHead,
                const RobotModel& model, ModelData& data, MatRef J) const override;
  // r_v = v+ - v- - dv(q-, v-): the velocity rows' weights contract -d2 dv / dz_tail2.
  void addContractedHessian(const NodeQuantities& tail, const NodeQuantities& head, const NodeDims& dTail,
                            const NodeDims& dHead, HessianScratch& s, const Eigen::Ref<const Eigen::VectorXd>& w,
                            Eigen::MatrixXd& H) const override;

private:
  int setId_;
};

}  // namespace leap
