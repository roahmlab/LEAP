#pragma once
// User model only: native Pinocchio inverse dynamics at the four RK4 stages.
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "aligator/context.hpp"
#include "aligator/core/function-abstract.hpp"
#include "aligator/python/polymorphic.hpp"

namespace inverse_rk4 {
using Fn=aligator::context::StageFunction;
using BaseData=aligator::context::StageFunctionData;
using Ref=aligator::context::ConstVectorRef;
using Matrix=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;
using Matrix6=Eigen::Matrix<double,6,Eigen::Dynamic>;
struct InverseData : BaseData {
  pinocchio::Data pin;
  std::vector<pinocchio::Force> external;
  Vector q,v;
  Matrix dq,dv,mass;
  Matrix6 joint_jac,frame_jac,df;
  InverseData(const Fn &fn,const pinocchio::Model &m):BaseData(fn),pin(m),
    external(m.njoints,pinocchio::Force::Zero()),q(m.nq),v(m.nv),
    dq(m.nv,m.nv),dv(m.nv,m.nv),mass(m.nv,m.nv),joint_jac(6,m.nv),frame_jac(6,m.nv),df(6,m.nv) {}
};
struct InverseRK4 : Fn {
  pinocchio::Model model;
  Eigen::VectorXi act,frames,dims;
  double dt;
  int nv,ntau;
  InverseRK4(int nx,int nu,const pinocchio::Model &m,const Eigen::VectorXi &a,
             const Eigen::VectorXi &f,const Eigen::VectorXi &d,double h):
    Fn(nx,nu,4*m.nv),model(m),act(a),frames(f),dims(d),dt(h),nv(m.nv),ntau(a.size()) {
    if (m.nq!=m.nv || nx!=2*m.nv || f.size()!=d.size() ||
        nu!=4*m.nv+a.size()+d.sum() || h<=0)
      throw std::invalid_argument("InverseRK4 requires explicit coordinates and a positive step");
  }
  void stage(Ref x,Ref u,int s,InverseData &d,bool derivative) const {
    const double qcoef[4]={0,0,dt*dt/4,dt*dt/2};
    const double vcoef[4]={0,dt/2,dt/2,dt};
    const double time[4]={0,dt/2,dt/2,dt};
    d.q=x.head(nv)+time[s]*x.tail(nv);
    d.v=x.tail(nv);
    if(s>=2) d.q+=qcoef[s]*u.segment((s-2)*nv,nv);
    if(s>=1) d.v+=vcoef[s]*u.segment((s-1)*nv,nv);
    const Vector acceleration=u.segment(s*nv,nv);
    pinocchio::computeJointJacobians(model,d.pin,d.q);
    pinocchio::updateFramePlacements(model,d.pin);
    for(auto &f:d.external) f.setZero();
    int off=4*nv+ntau;
    for(int c=0;c<frames.size();++c) {
      const auto &frame=model.frames[frames[c]];
      const auto j=frame.parentJoint;
      const Eigen::Matrix3d Rt=d.pin.oMi[j].rotation().transpose();
      pinocchio::Force f=pinocchio::Force::Zero();
      f.linear()=Rt*u.segment<3>(off);
      f.angular()=frame.placement.translation().cross(f.linear());
      if(dims[c]==6) f.angular()+=Rt*u.segment<3>(off+3);
      d.external[j]+=f;
      off+=dims[c];
    }
    if(!derivative) {
      d.value_.segment(s*nv,nv)=pinocchio::rnea(model,d.pin,d.q,d.v,acceleration,d.external);
      for(int j=0;j<ntau;++j) d.value_[s*nv+act[j]]-=u[4*nv+j];
      return;
    }
    d.dq.setZero();d.dv.setZero();d.mass.setZero();
    pinocchio::computeRNEADerivatives(model,d.pin,d.q,d.v,acceleration,d.external,d.dq,d.dv,d.mass);
    d.mass=d.mass.selfadjointView<Eigen::Upper>();
    // RNEA treats external forces as constant in each joint's local frame.
    // Our controls are world-aligned; include the derivative of that rotation.
    off=4*nv+ntau;
    for(int c=0;c<frames.size();++c) {
      const auto &frame=model.frames[frames[c]];
      const auto j=frame.parentJoint;
      const Eigen::Matrix3d Rt=d.pin.oMi[j].rotation().transpose();
      const Eigen::Vector3d f=Rt*u.segment<3>(off);
      Eigen::Vector3d torque=Eigen::Vector3d::Zero();
      if(dims[c]==6) torque=Rt*u.segment<3>(off+3);
      d.joint_jac.setZero();
      pinocchio::getJointJacobian(model,d.pin,j,pinocchio::LOCAL,d.joint_jac);
      d.df.topRows<3>().noalias()=pinocchio::skew(f)*d.joint_jac.bottomRows<3>();
      d.df.bottomRows<3>().noalias()=pinocchio::skew(torque)*d.joint_jac.bottomRows<3>();
      d.df.bottomRows<3>().noalias()+=pinocchio::skew(frame.placement.translation())*d.df.topRows<3>();
      d.dq.noalias()-=d.joint_jac.transpose()*d.df;
      d.frame_jac.setZero();
      pinocchio::getFrameJacobian(model,d.pin,frames[c],pinocchio::LOCAL_WORLD_ALIGNED,d.frame_jac);
      d.Ju().block(s*nv,off,nv,dims[c])=-d.frame_jac.topRows(dims[c]).transpose();
      off+=dims[c];
    }
    d.Jx().block(s*nv,0,nv,nv)=d.dq;
    d.Jx().block(s*nv,nv,nv,nv)=time[s]*d.dq+d.dv;
    d.Ju().block(s*nv,s*nv,nv,nv)=d.mass;
    if(s>=2) d.Ju().block(s*nv,(s-2)*nv,nv,nv)+=qcoef[s]*d.dq;
    if(s>=1) d.Ju().block(s*nv,(s-1)*nv,nv,nv)+=vcoef[s]*d.dv;
    for(int j=0;j<ntau;++j) d.Ju()(s*nv+act[j],4*nv+j)=-1;
  }
  void evaluate(const Ref &x,const Ref &u,BaseData &b) const override {
    auto &d=static_cast<InverseData&>(b);
    for(int s=0;s<4;++s) stage(x,u,s,d,false);
  }
  void computeJacobians(const Ref &x,const Ref &u,BaseData &b) const override {
    auto &d=static_cast<InverseData&>(b);
    d.Jx().setZero();d.Ju().setZero();
    for(int s=0;s<4;++s) stage(x,u,s,d,true);
  }
  aligator::shared_ptr<BaseData> createData() const override {return std::make_shared<InverseData>(*this,model);}
};
}
