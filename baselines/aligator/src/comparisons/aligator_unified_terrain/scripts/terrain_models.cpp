// Public residual model only; the installed ALIGATOR optimizer is unchanged.
#include "../../aligator_bench/inverse_model.hpp"
namespace terrain_public {
using namespace inverse_rk4;
struct QuadraticData : BaseData {
  InverseData inner;
  Vector x,u;
  QuadraticData(const Fn &f,const InverseRK4 &inverse)
    :BaseData(f),inner(inverse,inverse.model),x(2*inverse.nv),u(inverse.nu) {}
};
struct QuadraticInverse : Fn {
  InverseRK4 inverse;
  int nv,nt,nf;
  double dt;
  QuadraticInverse(const pinocchio::Model &m,const Eigen::VectorXi &act,
    const Eigen::VectorXi &frames,const Eigen::VectorXi &dims,double h)
    :Fn(2*m.nv,3*m.nv+3*act.size()+3*dims.sum(),3*m.nv),
     inverse(2*m.nv,4*m.nv+act.size()+dims.sum(),m,act,frames,dims,h),
     nv(m.nv),nt(act.size()),nf(dims.sum()),dt(h) {}
  static void weights(double s,Eigen::Vector3d &q,Eigen::Vector3d &v,Eigen::Vector3d &a) {
    a << 2*s*s-3*s+1,4*s-4*s*s,2*s*s-s;
    v << 2*s*s*s/3-1.5*s*s+s,2*s*s-4*s*s*s/3,2*s*s*s/3-.5*s*s;
    q << s*s*s*s/6-.5*s*s*s+.5*s*s,2*s*s*s/3-s*s*s*s/3,s*s*s*s/6-s*s*s/6;
  }
  void point(const Ref &x,const Ref &u,int k,QuadraticData &d,bool deriv) const {
    const double s=.5*k;
    Eigen::Vector3d qw,vw,aw;weights(s,qw,vw,aw);
    d.x=x;d.x.head(nv)+=dt*s*x.tail(nv);d.u.setZero();
    for(int j=0;j<3;++j) {
      d.x.head(nv)+=dt*dt*qw[j]*u.segment(j*nv,nv);
      d.x.tail(nv)+=dt*vw[j]*u.segment(j*nv,nv);
      d.u.head(nv)+=aw[j]*u.segment(j*nv,nv);
    }
    const double hold[3]={(1-s)*(1-s),2*s*(1-s),s*s};
    for(int ep=0;ep<3;++ep) {
      d.u.segment(4*nv,nt)+=hold[ep]*u.segment(3*nv+ep*nt,nt);
      d.u.tail(nf)+=hold[ep]*u.segment(3*nv+3*nt+ep*nf,nf);
    }
    if(!deriv) {
      inverse.stage(d.x,d.u,0,d.inner,false);
      d.value_.segment(k*nv,nv)=d.inner.value_.head(nv);
      return;
    }
    d.inner.Jx_.setZero();d.inner.Ju_.setZero();
    inverse.stage(d.x,d.u,0,d.inner,true);
    const Matrix dq=d.inner.Jx_.topLeftCorner(nv,nv);
    const Matrix dv=d.inner.Jx_.block(0,nv,nv,nv);
    const Matrix mass=d.inner.Ju_.topLeftCorner(nv,nv);
    d.Jx_.block(k*nv,0,nv,nv)=dq;
    d.Jx_.block(k*nv,nv,nv,nv)=dt*s*dq+dv;
    for(int j=0;j<3;++j)
      d.Ju_.block(k*nv,j*nv,nv,nv)=dt*dt*qw[j]*dq+dt*vw[j]*dv+aw[j]*mass;
    const Matrix jt=d.inner.Ju_.block(0,4*nv,nv,nt);
    const Matrix jf=d.inner.Ju_.block(0,4*nv+nt,nv,nf);
    for(int ep=0;ep<3;++ep) {
      d.Ju_.block(k*nv,3*nv+ep*nt,nv,nt)=hold[ep]*jt;
      d.Ju_.block(k*nv,3*nv+3*nt+ep*nf,nv,nf)=hold[ep]*jf;
    }
  }
  void evaluate(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<QuadraticData&>(base);
    for(int k=0;k<3;++k)point(x,u,k,d,false);
  }
  void computeJacobians(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<QuadraticData&>(base);d.Jx_.setZero();d.Ju_.setZero();
    for(int k=0;k<3;++k)point(x,u,k,d,true);
  }
  aligator::shared_ptr<BaseData> createData() const override {
    return std::make_shared<QuadraticData>(*this,inverse);
  }
};

}
BOOST_PYTHON_MODULE(_terrain_models) {
  namespace bp=boost::python;
  using namespace terrain_public;
  bp::import("aligator.pyaligator");
  bp::class_<QuadraticInverse,bp::bases<Fn>>("QuadraticInverse",
      bp::init<const pinocchio::Model&,const Eigen::VectorXi&,const Eigen::VectorXi&,const Eigen::VectorXi&,double>())
      .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
}
