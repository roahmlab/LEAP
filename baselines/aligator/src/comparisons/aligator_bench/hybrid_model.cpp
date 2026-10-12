// Public Aligator user models; no solver implementation or modification.
#include "inverse_model.hpp"
#include <pinocchio/algorithm/impulse-dynamics.hpp>
#include <pinocchio/algorithm/impulse-dynamics-derivatives.hpp>
#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/proximal.hpp>
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/vector-space.hpp"

namespace hybrid {
using namespace inverse_rk4;
using Dynamics = aligator::context::ExplicitDynamics;
using DynamicsData = aligator::ExplicitDynamicsDataTpl<double>;
using Space = aligator::VectorSpaceTpl<double,Eigen::Dynamic>;
namespace bp = boost::python;

struct FOHData : BaseData {
  InverseData inner;
  Vector control;
  FOHData(const Fn &fn, const InverseRK4 &inv):BaseData(fn),inner(inv,inv.model),control(inv.nu) {}
};
// x=[q,v,F0/weight], u=[a1,a2,a3,a4,tau/limits,F1/weight].
struct FOHInverse : Fn {
  InverseRK4 inverse;
  Vector effort;
  double weight;
  int nv,nf,ntau;
  FOHInverse(const pinocchio::Model &m,const Eigen::VectorXi &act,
             const Eigen::VectorXi &frames,const Eigen::VectorXi &dims,
             const Vector &limits,double w,double h):
    Fn(2*m.nv+dims.sum(),4*m.nv+act.size()+dims.sum(),4*m.nv),
    inverse(2*m.nv,4*m.nv+act.size()+dims.sum(),m,act,frames,dims,h),
    effort(limits),weight(w),nv(m.nv),nf(dims.sum()),ntau(act.size()) {
    if(limits.size()!=ntau || w<=0 || (limits.array()<=0).any())
      throw std::invalid_argument("Invalid physical scales");
  }
  void inputs(Ref x,Ref u,int s,FOHData &d) const {
    const double c[4]={0,.5,.5,1};
    d.control.head(4*nv)=u.head(4*nv);
    d.control.segment(4*nv,ntau)=effort.array()*u.segment(4*nv,ntau).array();
    d.control.tail(nf)=weight*((1-c[s])*x.tail(nf)+c[s]*u.tail(nf));
  }
  void evaluate(const Ref &x,const Ref &u,BaseData &b) const override {
    auto &d=static_cast<FOHData&>(b);
    for(int s=0;s<4;++s) {
      inputs(x,u,s,d);inverse.stage(x.head(2*nv),d.control,s,d.inner,false);
    }
    d.value_=d.inner.value_/weight;
  }
  void computeJacobians(const Ref &x,const Ref &u,BaseData &b) const override {
    auto &d=static_cast<FOHData&>(b);
    d.inner.Jx().setZero();d.inner.Ju().setZero();d.Jx().setZero();d.Ju().setZero();
    const double c[4]={0,.5,.5,1};
    for(int s=0;s<4;++s) {
      inputs(x,u,s,d);inverse.stage(x.head(2*nv),d.control,s,d.inner,true);
      d.Jx().block(s*nv,0,nv,2*nv)=d.inner.Jx().middleRows(s*nv,nv)/weight;
      d.Jx().block(s*nv,2*nv,nv,nf)=(1-c[s])*d.inner.Ju().block(s*nv,4*nv+ntau,nv,nf);
      d.Ju().block(s*nv,0,nv,4*nv)=d.inner.Ju().block(s*nv,0,nv,4*nv)/weight;
      d.Ju().block(s*nv,4*nv,nv,ntau)=d.inner.Ju().block(s*nv,4*nv,nv,ntau)*effort.asDiagonal()/weight;
      d.Ju().block(s*nv,4*nv+ntau,nv,nf)=c[s]*d.inner.Ju().block(s*nv,4*nv+ntau,nv,nf);
    }
  }
  aligator::shared_ptr<BaseData> createData() const override {return std::make_shared<FOHData>(*this,inverse);}
};

using RCM = pinocchio::RigidConstraintModel;
using RCD = pinocchio::RigidConstraintData;
struct ImpactData : DynamicsData {
  pinocchio::Data pin;
  std::vector<RCD> contacts;
  ImpactData(const Dynamics &fn,const pinocchio::Model &m,const std::vector<RCM> &models):
    DynamicsData(fn),pin(m) {
    for(const auto &cm:models) contacts.emplace_back(cm);
    pinocchio::initConstraintDynamics(m,pin,models,contacts);
  }
};
struct Impact : Dynamics {
  pinocchio::Model model;
  std::vector<RCM> contacts;
  int nv,nf;
  Impact(const pinocchio::Model &m,const bp::object &cms,int forces,int controls):
    Dynamics(Space(2*m.nv+forces),controls),model(m),nv(m.nv),nf(forces) {
    if(m.nq!=m.nv) throw std::invalid_argument("Explicit-coordinate model required");
    for(int i=0;i<bp::len(cms);++i) contacts.push_back(bp::extract<RCM>(cms[i]));
  }
  void forward(const Ref &x,const Ref &u,DynamicsData &b) const override {
    auto &d=static_cast<ImpactData&>(b);
    pinocchio::ProximalSettings settings(1e-12,1e-9,1);
    pinocchio::impulseDynamics(model,d.pin,x.head(nv),x.segment(nv,nv),contacts,d.contacts,0.,settings);
    d.xnext_.head(nv)=x.head(nv);
    d.xnext_.segment(nv,nv)=d.pin.dq_after;
    d.xnext_.tail(nf)=u.tail(nf);
  }
  void dForward(const Ref &x,const Ref &u,DynamicsData &b) const override {
    auto &d=static_cast<ImpactData&>(b);
    forward(x,u,b);
    pinocchio::ProximalSettings settings(1e-12,1e-9,1);
    d.pin.dvc_dq.setZero();
    pinocchio::computeImpulseDynamicsDerivatives(model,d.pin,contacts,d.contacts,0.,settings);
    d.Jx().setZero();d.Ju().setZero();
    d.Jx().topLeftCorner(nv,nv).setIdentity();
    d.Jx().block(nv,0,nv,nv)=d.pin.ddq_dq;
    d.Jx().block(nv,nv,nv,nv)=Matrix::Identity(nv,nv)+d.pin.ddq_dv;
    d.Ju().bottomRightCorner(nf,nf).setIdentity();
  }
  aligator::shared_ptr<DynamicsData> createData() const override {return std::make_shared<ImpactData>(*this,model,contacts);}
};

struct RotationData : BaseData {
  pinocchio::Data pin;
  Matrix6 jac;
  RotationData(const Fn &fn,const pinocchio::Model &m):BaseData(fn),pin(m),jac(6,m.nv) {}
};
struct Rotation : Fn {
  pinocchio::Model model;
  int frame;
  Rotation(int nx,int nu,const pinocchio::Model &m,int f):Fn(nx,nu,3),model(m),frame(f) {}
  void evaluate(const Ref &x,const Ref &,BaseData &b) const override {
    auto &d=static_cast<RotationData&>(b);
    pinocchio::framesForwardKinematics(model,d.pin,x.head(model.nq));
    d.value_=pinocchio::log3(d.pin.oMf[frame].rotation());
  }
  void computeJacobians(const Ref &x,const Ref &,BaseData &b) const override {
    auto &d=static_cast<RotationData&>(b);
    pinocchio::computeJointJacobians(model,d.pin,x.head(model.nq));
    pinocchio::updateFramePlacements(model,d.pin);
    d.jac.setZero();pinocchio::getFrameJacobian(model,d.pin,frame,pinocchio::LOCAL,d.jac);
    Eigen::Matrix3d logjac;pinocchio::Jlog3(d.pin.oMf[frame].rotation(),logjac);
    d.Jx().setZero();d.Ju().setZero();
    d.Jx().leftCols(model.nv)=logjac*d.jac.bottomRows<3>();
  }
  aligator::shared_ptr<BaseData> createData() const override {return std::make_shared<RotationData>(*this,model);}
};
}
BOOST_PYTHON_MODULE(_hybrid_model) {
  namespace bp=boost::python;using namespace hybrid;
  bp::import("aligator.pyaligator");
  bp::class_<FOHInverse,bp::bases<Fn>>("FOHInverse",bp::init<const pinocchio::Model&,const Eigen::VectorXi&,const Eigen::VectorXi&,const Eigen::VectorXi&,const Vector&,double,double>())
    .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
  bp::class_<Impact,bp::bases<Dynamics>>("Impact",bp::init<const pinocchio::Model&,const bp::object&,int,int>())
    .def(aligator::python::PolymorphicMultiBaseVisitor<Dynamics>());
  bp::class_<Rotation,bp::bases<Fn>>("Rotation",bp::init<int,int,const pinocchio::Model&,int>())
    .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
}
