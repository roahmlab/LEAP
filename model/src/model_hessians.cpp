#include "leap/model/model_hessians.hpp"

// pinocchio's CppAD scalar support must come before the algorithms.
#include <pinocchio/autodiff/cppad.hpp>

#include <cmath>
#include <stdexcept>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea-second-order-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/math/taylor-expansion.hpp>
#include <pinocchio/spatial/explog.hpp>

#include "leap/model/robot_model_impl.hpp"

// Conventions of the analytic kernels (pinned against finite differences and against
// the CppAD tapes by tests/test_hessian_fd.cpp).
//
// Second-order RNEA tensors (column-major nv x nv x nv, torque index first):
//   d2tau_dqdq(i, j, k) = d2 tau_i / dq_j dq_k,  d2tau_dvdv(i, j, k) = d2 tau_i / dv_j dv_k,
//   d2tau_dqdv(i, j, k) = d2 tau_i / dq_j dv_k,  dtau_dadq(i, j, k) = dM_ij / dq_k.
// Contracting the first index with w through a Map(t.data(), nv, nv * nv) gives the
// nv x nv block scattered into the node Hessian.
//
// Point kernel. A body-fixed point p(q) on joint jc; its support chain root -> jc has n
// dofs with world Jacobian columns J_i (spatial motions, fresh in data.J), i = 0..n-1 in
// chain order, so that dJ_l / dq_k = J_k ^ J_l for k < l (k an ancestor of l) and 0
// otherwise, and dp / dq_i = Jl_i = J_i.lin + J_i.ang x p. For a generalized velocity
// velw the body's spatial motion is m = sum_i velw_i J_i, the point velocity
// s = m.lin + m.ang x p and the angular velocity m.ang; with the suffix sums
// S_i = sum_{l >= i} velw_l J_l:
//   M_i = dm / dq_i = J_i ^ S_i,    G_kl = d2 m / dq_k dq_l = J_k ^ M_l   (k <= l),
//   P_kl = d2 p / dq_k dq_l = J_k.ang x Jl_l                               (k <= l),
//   ds / dq_i = M_i.lin + M_i.ang x p + m.ang x Jl_i,
//   T_kl = d2 (f3 . s) / dq_k dq_l = f3.(G_kl.lin + G_kl.ang x p) + f3.(M_l.ang x Jl_k)
//                                  + f3.(M_k.ang x Jl_l) + f3.(m.ang x P_kl),
//   Tang_kl = d2 (fa . m.ang) / dq_k dq_l = fa.(J_k.ang x M_l.ang),
//   Pw_kl = f3 . P_kl,   Yang(j, k) = d (fa . J_j.ang) / dq_k = fa.(J_k.ang x J_j.ang) [k < j].
// The dynamics contact term uses velw = w (the row weights act as a generalized
// velocity) and f3 = lam (qq block -T, (q, lam) block -ds/dq); the velocity frame term
// uses velw = v and f3 = w3 (qq block +T, (q, v) block Pw); the position term is Pw.
//
// The frame acceleration pddot = Jl(q) a + D_v s(q, v), where D_v is the directional
// derivative along v in q (the bias term is the quadratic form v' P v): its Hessian is
// Pw on (a, q), 2 Pw on (v, v), 2 T(velw = v) on (v, q) and T(velw = a) + D_v T(velw = v)
// on (q, q), with D_v of every building block from D_v J_i = A_i ^ J_i (A_i the prefix
// sum strictly before i) and D_v p = s. The angular rows of a flat foot follow the
// same pattern with omega = J.ang v.
//
// Plastic impact (world-aligned, contact set order; mu the proximal damping):
//   K x = b,  K = [[M, -J'], [-J, -mu I]],  x = [dv; Lambda],  b = [0; J v]
// For F = wdv . dv + wlam . Lambda with the adjoint z = K^-1 [wdv; wlam]:
//   F_k  = z . (b_k - K_k x),   F_kl = z . (b_kl - K_kl x - K_k x_l - K_l x_k),
// every block of which is a point kernel (J_k, J_kl against a generalized velocity),
// the mass-matrix derivatives of the second-order RNEA at (q, 0, dv) without gravity,
// and the first-order solution derivatives x_l = K^-1 (b_l - K_l x).

namespace leap {

namespace {

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using pinocchio::Motion;

using ADd = CppAD::AD<double>;
using ADModel = pinocchio::ModelTpl<ADd>;
using ADData = pinocchio::DataTpl<ADd>;
using ADVec = Eigen::Matrix<ADd, Eigen::Dynamic, 1>;
using ADVec3 = Eigen::Matrix<ADd, 3, 1>;
using ADForce = pinocchio::ForceTpl<ADd>;
using ADSE3 = pinocchio::SE3Tpl<ADd>;

// A residual r(z) recorded on a CppAD tape.
class TapeFunction {
public:
  virtual ~TapeFunction() = default;
  virtual ADVec eval(const ADVec& z) = 0;
};

CppAD::ADFun<double> recordTape(int nz, TapeFunction& f) {
  ADVec z(nz);
  for (int i = 0; i < nz; ++i) z[i] = 0.1 + 0.01 * i;  // a generic, branch-free point
  CppAD::Independent(z);
  ADVec r = f.eval(z);
  CppAD::ADFun<double> fun;
  fun.Dependent(z, r);
  fun.optimize();
  return fun;
}

std::vector<double> toStd(const Eigen::Ref<const VectorXd>& x) { return std::vector<double>(x.data(), x.data() + x.size()); }

void tapeHessian(CppAD::ADFun<double>& f, const Eigen::Ref<const VectorXd>& z, const Eigen::Ref<const VectorXd>& w, MatrixXd& H) {
  const int n = static_cast<int>(z.size());
  const std::vector<double> h = f.Hessian(toStd(z), toStd(w));
  H.resize(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) H(i, j) = h[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)];
}

// World point force at the body-fixed contact point as a LOCAL spatial force on the joint.
ADForce localContactForce(const ADSE3& oMi, const ADSE3& placement, const ADVec3& lam3) {
  const ADVec3 p = (oMi * placement).translation();
  const ADForce fO(lam3, p.cross(lam3));
  return oMi.actInv(fO);
}

// World wrench (force at the contact point + free torque, world axes) as a LOCAL spatial force.
ADForce localContactWrench(const ADSE3& oMi, const ADSE3& placement, const ADVec3& f, const ADVec3& tau) {
  const ADVec3 p = (oMi * placement).translation();
  const ADForce fO(f, p.cross(f) + tau);
  return oMi.actInv(fO);
}

// The dynamics residual g = rnea(q, v, a, fext(q, lam)) + loss(v) - B u on one contact layout.
class DynamicsTape final : public TapeFunction {
public:
  DynamicsTape(ADModel& adm, ADData& add, const NodeDims& d, const ContactSet& cs, const std::vector<BodyPoint>& contacts,
               const std::vector<int>& actIdx, const std::string& motorType, const VectorXd& coulomb, const VectorXd& viscous,
               const VectorXd& bias, double eps)
      : adm_(adm), add_(add), d_(d), cs_(cs), contacts_(contacts), actIdx_(actIdx), motorType_(motorType), coulomb_(coulomb),
        viscous_(viscous), bias_(bias), eps_(eps) {}
  ADVec eval(const ADVec& z) override {
    const ADVec q = z.segment(d_.offQ(), d_.nq), v = z.segment(d_.offV(), d_.nv), a = z.segment(d_.offA(), d_.na);
    PINOCCHIO_ALIGNED_STD_VECTOR(ADForce) fext(adm_.joints.size(), ADForce::Zero());
    if (!cs_.active.empty()) {
      pinocchio::forwardKinematics(adm_, add_, q);
      int o = 0;
      for (const ContactRef& cr : cs_.active) {
        const BodyPoint& ci = contacts_[static_cast<size_t>(cr.index)];
        if (cr.dim <= 3) {
          ADVec3 lam3 = ADVec3::Zero();
          lam3.head(cr.dim) = z.segment(d_.offLam() + o, cr.dim);
          fext[ci.joint] += localContactForce(add_.oMi[ci.joint], ci.placement.cast<ADd>(), lam3);
        } else {
          const ADVec3 f = z.segment(d_.offLam() + o, 3);
          const ADVec3 tq = z.segment(d_.offLam() + o + 3, 3);
          fext[ci.joint] += localContactWrench(add_.oMi[ci.joint], ci.placement.cast<ADd>(), f, tq);
        }
        o += cr.dim;
      }
    }
    ADVec r = pinocchio::rnea(adm_, add_, q, v, a, fext);
    for (size_t j = 0; j < actIdx_.size(); ++j) {
      const int i = actIdx_[j];
      if (motorType_ != "none") {
        const ADd x = v[i];
        const ADd sign = x / CppAD::sqrt(x * x + eps_ * eps_);
        r[i] += coulomb_(static_cast<int>(j)) * sign + viscous_(static_cast<int>(j)) * x + bias_(static_cast<int>(j));
      }
      if (d_.nu > 0) r[i] -= z[d_.offU() + static_cast<int>(j)];
    }
    return r;
  }

private:
  ADModel& adm_;
  ADData& add_;
  NodeDims d_;
  ContactSet cs_;
  const std::vector<BodyPoint>& contacts_;
  const std::vector<int>& actIdx_;
  std::string motorType_;
  VectorXd coulomb_, viscous_, bias_;
  double eps_;
};

// Point position / velocity / acceleration of one body point over [q; v; a]: dim <= 3 gives
// the first dim rows of the linear quantity, dim 6 gives [linear at p; world angular].
class PointTape final : public TapeFunction {
public:
  PointTape(ADModel& adm, ADData& add, const BodyPoint& pt, FrameTerm term, int nq, int nv)
      : adm_(adm), add_(add), pt_(pt), term_(term), nq_(nq), nv_(nv) {}
  ADVec eval(const ADVec& z) override {
    const ADVec q = z.segment(0, nq_), v = z.segment(nq_, nv_), a = z.segment(nq_ + nv_, nv_);
    pinocchio::forwardKinematics(adm_, add_, q, v, a);
    const ADSE3& oMi = add_.oMi[pt_.joint];
    const pinocchio::MotionTpl<ADd> vw = oMi.act(add_.v[pt_.joint]);
    const pinocchio::MotionTpl<ADd> aw = oMi.act(add_.a[pt_.joint]);
    const ADVec3 p = (oMi * pt_.placement.cast<ADd>()).translation();
    const ADVec3 om = vw.angular(), al = aw.angular();
    const ADVec3 pdot = vw.linear() + om.cross(p);
    const ADVec3 pddot = aw.linear() + al.cross(p) + om.cross(pdot);
    const int rows = pt_.dim == 6 ? 6 : std::min(pt_.dim, 3);
    ADVec r(term_ == FrameTerm::Position ? std::min(pt_.dim, 3) : rows);
    if (term_ == FrameTerm::Position) {
      r = p.head(r.size());
    } else if (pt_.dim == 6) {
      r.head(3) = term_ == FrameTerm::Velocity ? pdot : pddot;
      r.tail(3) = term_ == FrameTerm::Velocity ? om : al;
    } else {
      r = term_ == FrameTerm::Velocity ? pdot.head(r.size()) : pddot.head(r.size());
    }
    return r;
  }

private:
  ADModel& adm_;
  ADData& add_;
  BodyPoint pt_;
  FrameTerm term_;
  int nq_, nv_;
};

struct LayoutTape {
  ContactSet cs;
  CppAD::ADFun<double> fun;
};

Eigen::Matrix3d skew(const Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}

// The point kernel of one body point: the support chain's world Jacobian columns and,
// for a generalized velocity, the suffix sums and their directional derivatives.
struct PointKernel {
  int n = 0;
  const std::vector<int>* cols = nullptr;
  Vector3d p = Vector3d::Zero();
  std::vector<Motion> J;      // world Jacobian columns, chain order
  MatrixXd Jl;                // 3 x n point Jacobian columns
  // for a generalized velocity velw
  std::vector<Motion> S, M;   // S_i = sum_{l >= i} velw_l J_l,  M_i = J_i ^ S_i
  Motion m = Motion::Zero();  // S_0
  // directional derivatives along the velocity v used for velw (kernelDir)
  std::vector<Motion> A, DJ, DS, DM;  // A_i prefix strictly before i; D_v J_i, D_v S_i, D_v M_i
  MatrixXd DJl;                        // 3 x n: D_v Jl_i
  Vector3d s = Vector3d::Zero();      // D_v p = point velocity
  Motion Dm = Motion::Zero();          // D_v m

  void setup(const pinocchio::Data& data, const BodyPoint& pt, const std::vector<int>& support) {
    cols = &support;
    n = static_cast<int>(support.size());
    p = (data.oMi[pt.joint] * pt.placement).translation();
    J.resize(static_cast<size_t>(n));
    Jl.resize(3, n);
    for (int i = 0; i < n; ++i) {
      J[static_cast<size_t>(i)] = Motion(data.J.col(support[static_cast<size_t>(i)]));
      Jl.col(i) = J[static_cast<size_t>(i)].linear() + J[static_cast<size_t>(i)].angular().cross(p);
    }
  }
  void velocity(const VectorXd& velw) {
    S.resize(static_cast<size_t>(n));
    M.resize(static_cast<size_t>(n));
    S[static_cast<size_t>(n - 1)] = J[static_cast<size_t>(n - 1)] * velw((*cols)[static_cast<size_t>(n - 1)]);
    for (int i = n - 2; i >= 0; --i)
      S[static_cast<size_t>(i)] = S[static_cast<size_t>(i + 1)] + J[static_cast<size_t>(i)] * velw((*cols)[static_cast<size_t>(i)]);
    m = S[0];
    for (int i = 0; i < n; ++i) M[static_cast<size_t>(i)] = J[static_cast<size_t>(i)].cross(S[static_cast<size_t>(i)]);
  }
  // D_v of everything, for velw = v (velocity(v) must have run).
  void directional(const VectorXd& v) {
    A.resize(static_cast<size_t>(n));
    DJ.resize(static_cast<size_t>(n));
    DS.resize(static_cast<size_t>(n));
    DM.resize(static_cast<size_t>(n));
    DJl.resize(3, n);
    Motion acc = Motion::Zero();
    for (int i = 0; i < n; ++i) {
      A[static_cast<size_t>(i)] = acc;
      acc += J[static_cast<size_t>(i)] * v((*cols)[static_cast<size_t>(i)]);
    }
    s = m.linear() + m.angular().cross(p);
    for (int i = 0; i < n; ++i) {
      DJ[static_cast<size_t>(i)] = A[static_cast<size_t>(i)].cross(J[static_cast<size_t>(i)]);
      DJl.col(i) = DJ[static_cast<size_t>(i)].linear() + DJ[static_cast<size_t>(i)].angular().cross(p) + J[static_cast<size_t>(i)].angular().cross(s);
    }
    DS[static_cast<size_t>(n - 1)] = DJ[static_cast<size_t>(n - 1)] * v((*cols)[static_cast<size_t>(n - 1)]);
    for (int i = n - 2; i >= 0; --i)
      DS[static_cast<size_t>(i)] = DS[static_cast<size_t>(i + 1)] + DJ[static_cast<size_t>(i)] * v((*cols)[static_cast<size_t>(i)]);
    Dm = DS[0];
    for (int i = 0; i < n; ++i)
      DM[static_cast<size_t>(i)] = DJ[static_cast<size_t>(i)].cross(S[static_cast<size_t>(i)]) + J[static_cast<size_t>(i)].cross(DS[static_cast<size_t>(i)]);
  }

  // --- position ---
  double Pw(int k, int l, const Vector3d& f3) const {  // k <= l
    return f3.dot(J[static_cast<size_t>(k)].angular().cross(Vector3d(Jl.col(l))));
  }
  // --- velocity-dependent (velocity(velw) done) ---
  Vector3d dsdq(int i) const {
    const Motion& Mi = M[static_cast<size_t>(i)];
    return Mi.linear() + Mi.angular().cross(p) + m.angular().cross(Vector3d(Jl.col(i)));
  }
  Vector3d dsdqAng(int i) const { return M[static_cast<size_t>(i)].angular(); }
  double T(int k, int l, const Vector3d& f3) const {  // k <= l
    const Motion G = J[static_cast<size_t>(k)].cross(M[static_cast<size_t>(l)]);
    const Vector3d Pkl = J[static_cast<size_t>(k)].angular().cross(Vector3d(Jl.col(l)));
    return f3.dot(G.linear() + G.angular().cross(p)) + f3.dot(M[static_cast<size_t>(l)].angular().cross(Vector3d(Jl.col(k)))) +
           f3.dot(M[static_cast<size_t>(k)].angular().cross(Vector3d(Jl.col(l)))) + f3.dot(m.angular().cross(Pkl));
  }
  double Tang(int k, int l, const Vector3d& fa) const {  // k <= l
    return fa.dot(J[static_cast<size_t>(k)].angular().cross(M[static_cast<size_t>(l)].angular()));
  }
  double Yang(int j, int k, const Vector3d& fa) const {  // d (fa . J_j.ang) / dq_k
    if (k >= j) return 0.0;
    return fa.dot(J[static_cast<size_t>(k)].angular().cross(J[static_cast<size_t>(j)].angular()));
  }
  // --- directional (directional(v) done) ---
  double DT(int k, int l, const Vector3d& f3) const {  // D_v T_kl, k <= l
    const Motion& Jk = J[static_cast<size_t>(k)];
    const Motion& Ml = M[static_cast<size_t>(l)];
    const Motion& Mk = M[static_cast<size_t>(k)];
    const Motion G = Jk.cross(Ml);
    const Motion DG = DJ[static_cast<size_t>(k)].cross(Ml) + Jk.cross(DM[static_cast<size_t>(l)]);
    const Vector3d Jlk = Jl.col(k), Jll = Jl.col(l), DJlk = DJl.col(k), DJll = DJl.col(l);
    const Vector3d Pkl = Jk.angular().cross(Jll);
    const Vector3d DPkl = DJ[static_cast<size_t>(k)].angular().cross(Jll) + Jk.angular().cross(DJll);
    return f3.dot(DG.linear() + DG.angular().cross(p) + G.angular().cross(s)) +
           f3.dot(DM[static_cast<size_t>(l)].angular().cross(Jlk) + Ml.angular().cross(DJlk)) +
           f3.dot(DM[static_cast<size_t>(k)].angular().cross(Jll) + Mk.angular().cross(DJll)) +
           f3.dot(Dm.angular().cross(Pkl) + m.angular().cross(DPkl));
  }
  double DTang(int k, int l, const Vector3d& fa) const {  // k <= l
    const Motion DG = DJ[static_cast<size_t>(k)].cross(M[static_cast<size_t>(l)]) + J[static_cast<size_t>(k)].cross(DM[static_cast<size_t>(l)]);
    return fa.dot(DG.angular());
  }
  // d/dq_l of fa . d(D_v omega)/dv_j = fa.(G_lj + [l < j] ((J_l ^ B_lj) ^ J_j + A_j ^ (J_l ^ J_j))).ang,
  // B_lj = sum_{l < i < j} v_i J_i = S_{l+1} - S_j.
  double angularVQ(int j, int l, const Vector3d& fa) const {
    const int a = std::min(j, l), b = std::max(j, l);
    double val = fa.dot(J[static_cast<size_t>(a)].cross(M[static_cast<size_t>(b)]).angular());
    if (l < j) {
      const Motion B = S[static_cast<size_t>(l + 1)] - S[static_cast<size_t>(j)];
      const Motion& Jj = J[static_cast<size_t>(j)];
      const Motion& Jlm = J[static_cast<size_t>(l)];
      val += fa.dot((Jlm.cross(B).cross(Jj) + A[static_cast<size_t>(j)].cross(Jlm.cross(Jj))).angular());
    }
    return val;
  }
};

}  // namespace

struct PinocchioHessians::Impl {
  const RobotModel& robot;
  const RobotModel::Impl& rm;
  const pinocchio::Model& model;
  pinocchio::Data data;
  pinocchio::Model modelNoG;  // gravity-free copy: d(M(q) a)/dq without the gravity term
  pinocchio::Data dataNoG;
  std::vector<BodyPoint> points;  // contacts ++ monitors ++ capsule endpoints
  int nContacts = 0, nv = 0, nq = 0;
  std::vector<std::vector<int>> supp;  // per point: dof columns root -> joint
  // tapes (recorded on first use)
  ADModel adm;
  std::unique_ptr<ADData> add;
  std::vector<LayoutTape> dynTapes;
  std::vector<std::unique_ptr<CppAD::ADFun<double>>> velTapes, accTapes;  // per contact
  long tapeCalls = 0;
  // scratch
  RowVectorXd r;
  MatrixXd R, Htape;
  PointKernel K;
  VectorXd lastJacQ;

  explicit Impl(const RobotModel& m)
      : robot(m), rm(m.impl()), model(m.impl().model), data(m.impl().model), modelNoG(m.impl().model), dataNoG(m.impl().model),
        nv(m.impl().model.nv), nq(m.impl().model.nq), adm(m.impl().model.cast<ADd>()) {
    modelNoG.gravity.setZero();
    add = std::make_unique<ADData>(adm);
    points = rm.contacts;
    nContacts = static_cast<int>(points.size());
    points.insert(points.end(), rm.monitors.begin(), rm.monitors.end());
    points.insert(points.end(), rm.capsulePoints.begin(), rm.capsulePoints.end());
    for (const BodyPoint& pt : points) {
      std::vector<int> cols;
      for (pinocchio::JointIndex j : model.supports[pt.joint]) {
        if (j == 0) continue;
        for (int k = 0; k < model.joints[j].nv(); ++k) cols.push_back(model.joints[j].idx_v() + k);
      }
      supp.push_back(cols);
    }
    r.resize(nv * nv);
    R.resize(nv, nv);
    velTapes.resize(static_cast<size_t>(nContacts));
    accTapes.resize(static_cast<size_t>(nContacts));
  }

  void ensureJacobians(const Eigen::Ref<const VectorXd>& q) {
    if (lastJacQ.size() == q.size() && (lastJacQ.array() == q.array()).all()) return;
    pinocchio::computeJointJacobians(model, data, q);
    lastJacQ = q;
  }

  // Contract a tensor over its first (torque) index with w into R (nv x nv).
  void contract(const pinocchio::Data::Tensor3x& t, const VectorXd& w) {
    Eigen::Map<const MatrixXd> Tm(t.data(), nv, nv * nv);
    r.noalias() = w.transpose() * Tm;
    R = Eigen::Map<const MatrixXd>(r.data(), nv, nv);
  }

  CppAD::ADFun<double>& dynamicsTape(const NodeDims& d, const ContactSet& cs) {
    for (LayoutTape& lt : dynTapes)
      if (lt.cs == cs) return lt.fun;
    DynamicsTape tf(adm, *add, d, cs, rm.contacts, rm.actIdx, rm.motorType, rm.motorCoulomb, rm.motorViscous, rm.motorBias,
                    rm.motorVelocitySmoothing);
    dynTapes.push_back(LayoutTape{cs, recordTape(d.nP(), tf)});
    return dynTapes.back().fun;
  }

  CppAD::ADFun<double>& pointTape(FrameTerm term, int contactIdx) {
    std::vector<std::unique_ptr<CppAD::ADFun<double>>>& store = term == FrameTerm::Velocity ? velTapes : accTapes;
    std::unique_ptr<CppAD::ADFun<double>>& slot = store[static_cast<size_t>(contactIdx)];
    if (!slot) {
      PointTape tf(adm, *add, points[static_cast<size_t>(contactIdx)], term, nq, nv);
      slot = std::make_unique<CppAD::ADFun<double>>(recordTape(nq + 2 * nv, tf));
    }
    return *slot;
  }

  // --- the velocity / orientation / acceleration kernels of one contact -------------------

  // (q, q) += sum over a 6-vector weight [f3; fa] of the second derivatives of the point
  // velocity (and, dim 6, the angular velocity) at velw; the kernel must be set up and
  // velocity(velw) run.
  void addVelocityQQ(const NodeDims& d, const Vector3d& f3, const Vector3d& fa, bool wrench, double sign, MatrixXd& H) const {
    const std::vector<int>& cols = *K.cols;
    for (int i = 0; i < K.n; ++i)
      for (int j = i; j < K.n; ++j) {
        double val = K.T(i, j, f3);
        if (wrench) val += K.Tang(i, j, fa);
        val *= sign;
        H(d.offQ() + cols[static_cast<size_t>(i)], d.offQ() + cols[static_cast<size_t>(j)]) += val;
        if (j != i) H(d.offQ() + cols[static_cast<size_t>(j)], d.offQ() + cols[static_cast<size_t>(i)]) += val;
      }
  }

  void addOrientationHessian(size_t idx, const NodeDims& d, const Vector3d& w, MatrixXd& H) {
    const BodyPoint& pt = points[idx];
    const Eigen::Matrix3d Rw = (data.oMi[pt.joint] * pt.placement).rotation();
    double theta = 0.0;
    const Vector3d rlog = pinocchio::log3(Rw, theta);
    Eigen::Matrix3d Jlog;
    pinocchio::Jlog3(theta, rlog, Jlog);
    // d(Jlog3(r)' w)/dr with Jlog3 = alpha r r' + diag I + 1/2 [r]x (pinocchio's branches)
    const double prec = pinocchio::TaylorSeriesExpansion<double>::template precision<3>();
    double alpha, alphapOverTheta, dpOverTheta;
    if (theta < prec) {
      alpha = 1.0 / 12.0 + theta * theta / 720.0;
      alphapOverTheta = 1.0 / 360.0;
      dpOverTheta = -1.0 / 6.0;
    } else {
      const double st = std::sin(theta), ct = std::cos(theta);
      const double omc = 1.0 - ct;
      alpha = 1.0 / (theta * theta) - st / (2.0 * theta * omc);
      alphapOverTheta = (-2.0 / (theta * theta * theta) + 0.5 * (st + theta) / (theta * theta * omc)) / theta;
      dpOverTheta = 0.5 * (st - theta) / omc / theta;
    }
    const double rw = rlog.dot(w);
    Eigen::Matrix3d dm = alpha * (rw * Eigen::Matrix3d::Identity() + rlog * w.transpose()) + alphapOverTheta * rw * rlog * rlog.transpose() +
                         dpOverTheta * w * rlog.transpose() + 0.5 * skew(w);
    const Vector3d c = Rw * (Jlog.transpose() * w);
    const Eigen::Matrix3d B = Rw * dm;
    const Eigen::Matrix3d Arot = Jlog * Rw.transpose();
    const std::vector<int>& cols = *K.cols;
    for (int l = 0; l < K.n; ++l) {
      const Vector3d Jla = K.J[static_cast<size_t>(l)].angular();
      const Vector3d dcdq = Jla.cross(c) + B * (Arot * Jla);
      for (int k = 0; k < K.n; ++k) {
        const Vector3d Jka = K.J[static_cast<size_t>(k)].angular();
        double val = dcdq.dot(Jka);
        if (l < k) val += c.dot(Jla.cross(Jka));
        H(d.offQ() + cols[static_cast<size_t>(k)], d.offQ() + cols[static_cast<size_t>(l)]) += val;
      }
    }
  }

  void addAccelerationHessian(size_t idx, const NodeDims& d, const VectorXd& z, const Vector3d& f3, const Vector3d& fa, bool wrench,
                              MatrixXd& H) {
    const std::vector<int>& cols = *K.cols;
    const int n = K.n;
    const VectorXd a = z.segment(d.offA(), d.na);
    const VectorXd v = z.segment(d.offV(), d.nv);
    // (q, q): T(velw = a)
    K.velocity(a);
    addVelocityQQ(d, f3, fa, wrench, 1.0, H);
    // the rest at velw = v, with the directional derivatives along v
    K.velocity(v);
    K.directional(v);
    for (int k = 0; k < n; ++k) {
      const int qk = d.offQ() + cols[static_cast<size_t>(k)];
      const int vk = d.offV() + cols[static_cast<size_t>(k)];
      const int ak = d.offA() + cols[static_cast<size_t>(k)];
      for (int l = 0; l < n; ++l) {
        const int ql = d.offQ() + cols[static_cast<size_t>(l)];
        const int vl = d.offV() + cols[static_cast<size_t>(l)];
        const int lo = std::min(k, l), hi = std::max(k, l);
        // (a_k, q_l)
        double aq = K.Pw(lo, hi, f3);
        if (wrench) aq += K.Yang(k, l, fa);
        H(ak, ql) += aq;
        H(ql, ak) += aq;
        // (v_k, v_l)
        double vv = 2.0 * K.Pw(lo, hi, f3);
        if (wrench) vv += K.Yang(k, l, fa) + K.Yang(l, k, fa);
        H(vk, vl) += vv;
        // (v_k, q_l)
        double vq = 2.0 * K.T(lo, hi, f3);
        if (wrench) vq += K.angularVQ(k, l, fa);
        H(vk, ql) += vq;
        H(ql, vk) += vq;
        // (q_k, q_l): D_v T
        if (k <= l) {
          double qq = K.DT(k, l, f3);
          if (wrench) qq += K.DTang(k, l, fa);
          H(qk, ql) += qq;
          if (l != k) H(ql, qk) += qq;
        }
      }
    }
    (void)idx;
  }

  // --- the impact map ---------------------------------------------------------------------

  // Stacked world-aligned contact Jacobian of `cs` (rows in cs order; 3 per point
  // contact, 6 per flat foot) at the current data (fresh Jacobians).
  void stackedContactJacobian(const ContactSet& cs, MatrixXd& Jc) {
    const int nL = cs.nLam();
    Jc.setZero(nL, nv);
    int o = 0;
    for (const ContactRef& cr : cs.active) {
      const size_t idx = static_cast<size_t>(cr.index);
      K.setup(data, points[idx], supp[idx]);
      const std::vector<int>& cols = *K.cols;
      for (int i = 0; i < K.n; ++i) {
        Jc.block(o, cols[static_cast<size_t>(i)], std::min(cr.dim, 3), 1) = K.Jl.col(i).head(std::min(cr.dim, 3));
        if (cr.dim == 6) Jc.block<3, 1>(o + 3, cols[static_cast<size_t>(i)]) = K.J[static_cast<size_t>(i)].angular();
      }
      o += cr.dim;
    }
  }

  // Dsq(:, k) = d (J(q) velw) / dq_k (nL x nv) and Y(j, k) = d (J' f)_j / dq_k (nv x nv) for
  // the stacked contact Jacobian; T(k, l) += d2 (f . J velw) / dq_k dq_l (nv x nv).
  void impactKernels(const ContactSet& cs, const VectorXd& velw, const VectorXd& f, MatrixXd* Dsq, MatrixXd* Y, MatrixXd* T) {
    int o = 0;
    for (const ContactRef& cr : cs.active) {
      const size_t idx = static_cast<size_t>(cr.index);
      K.setup(data, points[idx], supp[idx]);
      K.velocity(velw);
      const std::vector<int>& cols = *K.cols;
      const int nLin = std::min(cr.dim, 3);
      Vector3d f3 = Vector3d::Zero(), fa = Vector3d::Zero();
      f3.head(nLin) = f.segment(o, nLin);
      if (cr.dim == 6) fa = f.segment<3>(o + 3);
      for (int i = 0; i < K.n; ++i) {
        const int ci = cols[static_cast<size_t>(i)];
        if (Dsq) {
          Dsq->block(o, ci, nLin, 1) += K.dsdq(i).head(nLin);
          if (cr.dim == 6) Dsq->block<3, 1>(o + 3, ci) += K.dsdqAng(i);
        }
        for (int j = 0; j < K.n; ++j) {
          const int cj = cols[static_cast<size_t>(j)];
          if (Y) {
            double y = K.Pw(std::min(i, j), std::max(i, j), f3);  // d (f3 . Jl_j) / dq_i
            if (cr.dim == 6) y += K.Yang(j, i, fa);
            (*Y)(cj, ci) += y;
          }
          if (T && i <= j) {
            double t = K.T(i, j, f3);
            if (cr.dim == 6) t += K.Tang(i, j, fa);
            (*T)(ci, cj) += t;
            if (j != i) (*T)(cj, ci) += t;
          }
        }
      }
      o += cr.dim;
    }
  }
};

PinocchioHessians::PinocchioHessians(const RobotModel& m) : impl_(std::make_unique<Impl>(m)) {}
PinocchioHessians::~PinocchioHessians() = default;
long PinocchioHessians::tapeCalls() const { return impl_->tapeCalls; }
double PinocchioHessians::impactDamping() { return RobotModel::impactProximalDamping(); }

void PinocchioHessians::addDynamicsHessianTaped(const NodeDims& d, const ContactSet& cs, const VectorXd& z, const VectorXd& w,
                                                MatrixXd& H) {
  Impl& s = *impl_;
  ++s.tapeCalls;
  tapeHessian(s.dynamicsTape(d, cs), z, w, s.Htape);
  H += s.Htape;
}

void PinocchioHessians::addFrameHessianTaped(FrameTerm term, int contactIdx, const NodeDims& d, const VectorXd& z,
                                             const VectorXd& w, MatrixXd& H) {
  Impl& s = *impl_;
  if (term == FrameTerm::Orientation || term == FrameTerm::Position)
    throw std::logic_error("PinocchioHessians: no tape for position / orientation terms");
  if (contactIdx < 0 || contactIdx >= s.nContacts) throw std::out_of_range("PinocchioHessians: tape terms exist for contact frames only");
  const int nx = s.nq + 2 * s.nv;
  ++s.tapeCalls;
  tapeHessian(s.pointTape(term, contactIdx), z.head(nx), w, s.Htape);
  H.topLeftCorner(nx, nx) += s.Htape;
  (void)d;
}

void PinocchioHessians::addDynamicsHessian(const NodeDims& d, const ContactSet& cs, const VectorXd& z, const VectorXd& w,
                                           MatrixXd& H) {
  Impl& s = *impl_;
  const int nv = s.nv;
  const Eigen::Ref<const VectorXd> q = z.segment(d.offQ(), d.nq);
  const Eigen::Ref<const VectorXd> v = z.segment(d.offV(), d.nv);
  const Eigen::Ref<const VectorXd> a = z.segment(d.offA(), d.na);
  // (a) RNEA part: one second-order pass (it also refreshes data.oMi / data.J)
  pinocchio::ComputeRNEASecondOrderDerivatives(s.model, s.data, q, v, a);
  s.lastJacQ = q;
  s.contract(s.data.d2tau_dqdq, w);
  H.block(d.offQ(), d.offQ(), nv, nv) += s.R;
  s.contract(s.data.d2tau_dvdv, w);
  H.block(d.offV(), d.offV(), nv, nv) += s.R;
  s.contract(s.data.d2tau_dqdv, w);
  H.block(d.offQ(), d.offV(), nv, nv) += s.R;
  H.block(d.offV(), d.offQ(), nv, nv) += s.R.transpose();
  s.contract(s.data.d2tau_dadq, w);
  H.block(d.offA(), d.offQ(), nv, nv) += s.R;
  H.block(d.offQ(), d.offA(), nv, nv) += s.R.transpose();
  // (b) the motor loss: coulomb * smoothsign(v) has a (v, v) diagonal second derivative
  if (s.rm.motorType != "none") {
    const double eps = s.rm.motorVelocitySmoothing;
    for (size_t j = 0; j < s.rm.actIdx.size(); ++j) {
      const int i = s.rm.actIdx[j];
      const double x = v(i), den = x * x + eps * eps;
      const double d2sign = -3.0 * x * eps * eps / (den * den * std::sqrt(den));
      H(d.offV() + i, d.offV() + i) += w(i) * s.rm.motorCoulomb(static_cast<int>(j)) * d2sign;
    }
  }
  // (c) contact part: g -= Jc' lam_c, so the kernel blocks are subtracted (velw = w)
  int o = 0;
  for (const ContactRef& cr : cs.active) {
    const size_t idx = static_cast<size_t>(cr.index);
    const int nLin = std::min(cr.dim, 3);
    Vector3d lam3 = Vector3d::Zero(), tq = Vector3d::Zero();
    lam3.head(nLin) = z.segment(d.offLam() + o, nLin);
    const bool wrench = cr.dim == 6;
    if (wrench) tq = z.segment<3>(d.offLam() + o + 3);
    s.K.setup(s.data, s.points[idx], s.supp[idx]);
    s.K.velocity(w);
    s.addVelocityQQ(d, lam3, tq, wrench, -1.0, H);
    const std::vector<int>& cols = s.supp[idx];
    for (int i = 0; i < s.K.n; ++i) {
      const int qi = d.offQ() + cols[static_cast<size_t>(i)];
      const Vector3d ds = s.K.dsdq(i);
      for (int rr = 0; rr < nLin; ++rr) {
        H(qi, d.offLam() + o + rr) -= ds(rr);
        H(d.offLam() + o + rr, qi) -= ds(rr);
      }
      if (wrench) {
        const Vector3d dsa = s.K.dsdqAng(i);
        for (int rr = 0; rr < 3; ++rr) {
          H(qi, d.offLam() + o + 3 + rr) -= dsa(rr);
          H(d.offLam() + o + 3 + rr, qi) -= dsa(rr);
        }
      }
    }
    o += cr.dim;
  }
}

void PinocchioHessians::addFrameHessian(FrameTerm term, int pointIdx, const NodeDims& d, const VectorXd& z, const VectorXd& w,
                                        MatrixXd& H) {
  Impl& s = *impl_;
  if (pointIdx < 0 || pointIdx >= static_cast<int>(s.points.size())) throw std::out_of_range("PinocchioHessians: point index out of range");
  const bool isContact = pointIdx < s.nContacts;
  if (term != FrameTerm::Position && !isContact)
    throw std::logic_error("PinocchioHessians: velocity / acceleration / orientation terms exist for contact frames only");
  const size_t idx = static_cast<size_t>(pointIdx);
  const int dim = s.points[idx].dim;
  const bool wrench = dim == 6;
  const int nLin = std::min(dim, 3);
  s.ensureJacobians(z.segment(d.offQ(), d.nq));
  s.K.setup(s.data, s.points[idx], s.supp[idx]);
  const std::vector<int>& cols = s.supp[idx];
  if (term == FrameTerm::Orientation) {
    if (!wrench) throw std::logic_error("PinocchioHessians: the orientation term needs a dim-6 contact");
    if (w.size() != 3) throw std::invalid_argument("PinocchioHessians: orientation weights must have 3 rows");
    s.addOrientationHessian(idx, d, Vector3d(w), H);
    return;
  }
  Vector3d f3 = Vector3d::Zero(), fa = Vector3d::Zero();
  const int nw = static_cast<int>(w.size());
  f3.head(std::min(nLin, nw)) = w.head(std::min(nLin, nw));
  if (wrench && nw >= 6) fa = w.segment<3>(3);
  if (term == FrameTerm::Position) {
    for (int i = 0; i < s.K.n; ++i)
      for (int j = i; j < s.K.n; ++j) {
        const double val = s.K.Pw(i, j, f3);
        H(d.offQ() + cols[static_cast<size_t>(i)], d.offQ() + cols[static_cast<size_t>(j)]) += val;
        if (j != i) H(d.offQ() + cols[static_cast<size_t>(j)], d.offQ() + cols[static_cast<size_t>(i)]) += val;
      }
    return;
  }
  if (term == FrameTerm::Acceleration) {
    s.addAccelerationHessian(idx, d, z, f3, fa, wrench, H);
    return;
  }
  // velocity: velw = v
  const VectorXd v = z.segment(d.offV(), d.nv);
  s.K.velocity(v);
  s.addVelocityQQ(d, f3, fa, wrench, 1.0, H);
  for (int i = 0; i < s.K.n; ++i)
    for (int j = 0; j < s.K.n; ++j) {
      const int qi = d.offQ() + cols[static_cast<size_t>(i)];
      const int vj = d.offV() + cols[static_cast<size_t>(j)];
      double val = s.K.Pw(std::min(i, j), std::max(i, j), f3);
      if (wrench) val += s.K.Yang(j, i, fa);
      H(qi, vj) += val;
      H(vj, qi) += val;
    }
}

void PinocchioHessians::addImpactHessian(const ContactSet& cs, const NodeDims& d, const VectorXd& z, const VectorXd& wdv,
                                         const VectorXd& wlam, MatrixXd& H) {
  Impl& s = *impl_;
  const int nv = s.nv, nq = s.nq;
  const int nL = cs.nLam();
  if (nL == 0) return;
  if (wdv.size() != nv || wlam.size() != nL) throw std::invalid_argument("PinocchioHessians::addImpactHessian: weight sizes");
  const VectorXd q = z.segment(d.offQ(), d.nq);
  const VectorXd v = z.segment(d.offV(), d.nv);
  const double mu = impactDamping();
  // mass matrix, Jacobians
  pinocchio::crba(s.model, s.data, q);
  MatrixXd M = s.data.M;
  M.triangularView<Eigen::StrictlyLower>() = M.transpose();
  pinocchio::computeJointJacobians(s.model, s.data, q);
  s.lastJacQ = q;
  MatrixXd Jc;
  s.stackedContactJacobian(cs, Jc);
  const Eigen::LLT<MatrixXd> Mllt(M);
  if (Mllt.info() != Eigen::Success) throw std::runtime_error("PinocchioHessians::addImpactHessian: mass matrix not SPD");
  const MatrixXd MinvJt = Mllt.solve(Jc.transpose());  // nv x nL
  MatrixXd Gd = Jc * MinvJt;
  for (int i = 0; i < nL; ++i) Gd(i, i) += mu;
  const Eigen::LLT<MatrixXd> Gllt(Gd);
  if (Gllt.info() != Eigen::Success) throw std::runtime_error("PinocchioHessians::addImpactHessian: Delassus matrix not SPD");
  // forward solution and adjoint
  const VectorXd lam = -Gllt.solve(Jc * v);
  const VectorXd dv = MinvJt * lam;
  const VectorXd vplus = v + dv;
  const VectorXd zlam = -Gllt.solve(wlam + Jc * Mllt.solve(wdv));
  const VectorXd zdv = Mllt.solve(wdv + Jc.transpose() * zlam);
  // mass-matrix derivatives at (q, 0, dv) without gravity: dM/dq (tensor) and d2(M dv)/dq2
  pinocchio::ComputeRNEASecondOrderDerivatives(s.modelNoG, s.dataNoG, q, VectorXd::Zero(nv), dv);
  Eigen::Map<const MatrixXd> Tm(s.dataNoG.d2tau_dadq.data(), nv, nv * nv);
  MatrixXd MkDv(nv, nq);  // column k = (dM/dq_k) dv
  for (int k = 0; k < nq; ++k) MkDv.col(k).noalias() = Tm.middleCols(k * nv, nv) * dv;
  s.contract(s.dataNoG.d2tau_dadq, zdv);
  const MatrixXd A = s.R;  // A(j, k) = sum_i zdv_i dM_ij / dq_k
  s.contract(s.dataNoG.d2tau_dqdq, zdv);
  const MatrixXd SO = s.R;  // sum_i zdv_i d2 (M dv)_i / dq_k dq_l
  // point kernels of the contact Jacobian (SO-RNEA refreshed dataNoG only: data.J is from computeJointJacobians above)
  MatrixXd DsqVplus = MatrixXd::Zero(nL, nv), DsqZdv = MatrixXd::Zero(nL, nv);
  MatrixXd Ylam = MatrixXd::Zero(nv, nv), Yzlam = MatrixXd::Zero(nv, nv);
  MatrixXd Tv = MatrixXd::Zero(nv, nv), Tzdv = MatrixXd::Zero(nv, nv), Tdv = MatrixXd::Zero(nv, nv);
  s.impactKernels(cs, vplus, lam, &DsqVplus, &Ylam, nullptr);   // J_k v+,  J_k' lam
  s.impactKernels(cs, zdv, zlam, &DsqZdv, &Yzlam, nullptr);     // J_k zdv, J_k' zlam
  s.impactKernels(cs, v, zlam, nullptr, nullptr, &Tv);          // zlam . J_kl v
  s.impactKernels(cs, zdv, lam, nullptr, nullptr, &Tzdv);       // lam . J_kl zdv
  s.impactKernels(cs, dv, zlam, nullptr, nullptr, &Tdv);        // zlam . J_kl dv
  // first-order solution derivatives: x_l = K^-1 (b_l - K_l x), x_{v_j} = K^-1 [0; J e_j]
  //   K [a; c] = [r1; r2]:  c = -Gd^-1 (r2 + J M^-1 r1),  a = M^-1 (r1 + J' c)
  const MatrixXd R1 = Ylam - MkDv;  // nv x nq
  const MatrixXd& R2 = DsqVplus;    // nL x nq
  const MatrixXd Lq = -Gllt.solve(R2 + Jc * Mllt.solve(R1));
  const MatrixXd DVq = Mllt.solve(R1 + Jc.transpose() * Lq);
  const MatrixXd Lv = -Gllt.solve(Jc);
  const MatrixXd DVv = MinvJt * Lv;
  // second derivatives
  //   C(k, l) = z . K_k x_l = (A' DVq - DsqZdv' Lq - Yzlam' DVq)(k, l)
  const MatrixXd C = A.transpose() * DVq - DsqZdv.transpose() * Lq - Yzlam.transpose() * DVq;
  const MatrixXd Hqq = Tv - SO + Tzdv + Tdv - C - C.transpose();
  //   (q_k, v_j) = Yzlam(j, k) - (A' DVv - DsqZdv' Lv - Yzlam' DVv)(k, j)
  const MatrixXd Hqv = Yzlam.transpose() - (A.transpose() * DVv - DsqZdv.transpose() * Lv - Yzlam.transpose() * DVv);
  H.block(d.offQ(), d.offQ(), nq, nq) += Hqq;
  H.block(d.offQ(), d.offV(), nq, nv) += Hqv;
  H.block(d.offV(), d.offQ(), nv, nq) += Hqv.transpose();
}

}  // namespace leap
