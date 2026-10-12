// Public modelling API only. The installed ALIGATOR solvers are unchanged.
#include "../../aligator_bench/inverse_model.hpp"
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/algorithm/impulse-dynamics.hpp>
#include <pinocchio/algorithm/impulse-dynamics-derivatives.hpp>
#include <pinocchio/algorithm/constrained-dynamics.hpp>

namespace fence_public {
using namespace inverse_rk4;
struct ImpulseConeData : BaseData {
  pinocchio::Data pin;
  std::vector<pinocchio::RigidConstraintData> contacts;
  ImpulseConeData(const Fn &fn,const pinocchio::Model &m,
                  const std::vector<pinocchio::RigidConstraintModel> &cms)
    :BaseData(fn),pin(m) {
    for(const auto &cm:cms) contacts.emplace_back(cm);
    pinocchio::initConstraintDynamics(m,pin,cms,contacts);
  }
};
// LEAPS ImpactConeConstraint: the impulse selected by the same one-shot
// damped plastic reset, expressed in world-aligned axes, on the airborne side.
// Raw mode is g<=0; positive mode returns max(g+shift,0) for FDDP/outer AL.
struct ImpulseCone : Fn {
  pinocchio::Model model;
  std::vector<pinocchio::RigidConstraintModel> contacts;
  Eigen::VectorXi offsets;
  double mu,eps;
  bool positive;
  Vector shift;
  ImpulseCone(int nu,const pinocchio::Model &m,const Eigen::VectorXi &frames,
              const Eigen::VectorXi &dims,double friction,double smoothing,
              bool clip,const Vector &dual)
    :Fn(2*m.nv,nu,2*frames.size()),model(m),offsets(frames.size()),
     mu(friction),eps(smoothing),positive(clip),shift(dual) {
    if(m.nq!=m.nv || frames.size()!=dims.size() || shift.size()!=nr ||
       !shift.allFinite() || !std::isfinite(mu) || !std::isfinite(eps) || mu<=0 || eps<0)
      throw std::invalid_argument("Invalid impulse cone dimensions/parameters");
    int off=0;
    for(int i=0;i<frames.size();++i) {
      if(frames[i]<0 || frames[i]>=int(m.nframes) || (dims[i]!=3 && dims[i]!=6))
        throw std::invalid_argument("Invalid impulse cone contact");
      const auto &fr=m.frames[frames[i]];
      contacts.emplace_back(dims[i]==6 ? pinocchio::CONTACT_6D : pinocchio::CONTACT_3D,
                            m,fr.parentJoint,fr.placement,pinocchio::LOCAL_WORLD_ALIGNED);
      offsets[i]=off;off+=dims[i];
    }
  }
  void evaluate(const Ref &x,const Ref &,BaseData &base) const override {
    auto &d=static_cast<ImpulseConeData&>(base);
    pinocchio::ProximalSettings settings(1e-12,1e-9,1);
    pinocchio::impulseDynamics(model,d.pin,x.head(model.nv),x.tail(model.nv),
                              contacts,d.contacts,0.,settings);
    for(int i=0;i<offsets.size();++i) {
      const int o=offsets[i];const auto &f=d.pin.impulse_c;
      d.value_[2*i]=std::sqrt(f[o]*f[o]+f[o+1]*f[o+1]+eps*eps)-eps-mu*f[o+2];
      d.value_[2*i+1]=-f[o+2];
    }
    d.value_+=shift;
    if(positive) d.value_=d.value_.cwiseMax(0.);
  }
  void computeJacobians(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<ImpulseConeData&>(base);
    evaluate(x,u,base);
    pinocchio::ProximalSettings settings(1e-12,1e-9,1);
    d.pin.dvc_dq.setZero();
    pinocchio::computeImpulseDynamicsDerivatives(model,d.pin,contacts,d.contacts,0.,settings);
    d.Jx_.setZero();d.Ju_.setZero();
    for(int i=0;i<offsets.size();++i) {
      const int o=offsets[i],n=model.nv;const auto &f=d.pin.impulse_c;
      const double norm=std::sqrt(f[o]*f[o]+f[o+1]*f[o+1]+eps*eps);
      const double cx=norm>1e-12 ? f[o]/norm : 0.,cy=norm>1e-12 ? f[o+1]/norm : 0.;
      for(int side=0;side<2;++side) {
        const Matrix &dl=side==0 ? d.pin.dlambda_dq : d.pin.dlambda_dv;
        if(!positive || d.value_[2*i]>0.)
          d.Jx_.block(2*i,side*n,1,n)=cx*dl.row(o)+cy*dl.row(o+1)-mu*dl.row(o+2);
        if(!positive || d.value_[2*i+1]>0.)
          d.Jx_.block(2*i+1,side*n,1,n)=-dl.row(o+2);
      }
    }
  }
  aligator::shared_ptr<BaseData> createData() const override {
    return std::make_shared<ImpulseConeData>(*this,model,contacts);
  }
};
struct CollocationData : BaseData {
  InverseData inner;
  Vector x,u;
  CollocationData(const Fn &f,const InverseRK4 &inverse)
    :BaseData(f),inner(inverse,inverse.model),x(2*inverse.nv),u(inverse.nu) {}
};
struct CollocationInverse : Fn {
  InverseRK4 inverse;
  int nv,nt,nf;
  double dt;
  CollocationInverse(const pinocchio::Model &m,const Eigen::VectorXi &act,
    const Eigen::VectorXi &frames,const Eigen::VectorXi &dims,double h)
    :Fn(2*m.nv,3*m.nv+2*act.size()+2*dims.sum(),3*m.nv),
     inverse(2*m.nv,4*m.nv+act.size()+dims.sum(),m,act,frames,dims,h),
     nv(m.nv),nt(act.size()),nf(dims.sum()),dt(h) {}
  static void weights(double s,Eigen::Vector3d &q,Eigen::Vector3d &v,Eigen::Vector3d &a) {
    a << 2*s*s-3*s+1,4*s-4*s*s,2*s*s-s;
    v << 2*s*s*s/3-1.5*s*s+s,2*s*s-4*s*s*s/3,2*s*s*s/3-.5*s*s;
    q << s*s*s*s/6-.5*s*s*s+.5*s*s,2*s*s*s/3-s*s*s*s/3,s*s*s*s/6-s*s*s/6;
  }
  void point(const Ref &x,const Ref &u,int k,CollocationData &d,bool deriv) const {
    const double s=.5*k;
    Eigen::Vector3d qw,vw,aw;weights(s,qw,vw,aw);
    d.x=x;d.x.head(nv)+=dt*s*x.tail(nv);d.u.setZero();
    for(int j=0;j<3;++j) {
      d.x.head(nv)+=dt*dt*qw[j]*u.segment(j*nv,nv);
      d.x.tail(nv)+=dt*vw[j]*u.segment(j*nv,nv);
      d.u.head(nv)+=aw[j]*u.segment(j*nv,nv);
    }
    d.u.segment(4*nv,nt)=(1-s)*u.segment(3*nv,nt)+s*u.segment(3*nv+nt,nt);
    d.u.tail(nf)=(1-s)*u.segment(3*nv+2*nt,nf)+s*u.tail(nf);
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
    d.Ju_.block(k*nv,3*nv,nv,nt)=(1-s)*jt;
    d.Ju_.block(k*nv,3*nv+nt,nv,nt)=s*jt;
    d.Ju_.block(k*nv,3*nv+2*nt,nv,nf)=(1-s)*jf;
    d.Ju_.block(k*nv,3*nv+2*nt+nf,nv,nf)=s*jf;
  }
  void evaluate(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<CollocationData&>(base);
    for(int k=0;k<3;++k)point(x,u,k,d,false);
  }
  void computeJacobians(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<CollocationData&>(base);d.Jx_.setZero();d.Ju_.setZero();
    for(int k=0;k<3;++k)point(x,u,k,d,true);
  }
  aligator::shared_ptr<BaseData> createData() const override {
    return std::make_shared<CollocationData>(*this,inverse);
  }
};
// Pull a residual back through the exact dense-state linear map.
struct MappedData : BaseData {
  aligator::shared_ptr<BaseData> inner;
  Vector x;
  MappedData(const Fn &f,const Fn &sub):BaseData(f),inner(sub.createData()),x(sub.ndx1) {}
};
struct MappedState : Fn {
  xyz::polymorphic<Fn> sub;
  Matrix A,B;
  MappedState(xyz::polymorphic<Fn> fn,const Matrix &aa,const Matrix &bb)
    :Fn(aa.cols(),bb.cols(),fn->nr),sub(fn),A(aa),B(bb) {}
  void evaluate(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<MappedData&>(base);d.x.noalias()=A*x+B*u;
    sub->evaluate(d.x,u,*d.inner);d.value_=d.inner->value_;
  }
  void computeJacobians(const Ref &,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<MappedData&>(base);sub->computeJacobians(d.x,u,*d.inner);
    d.Jx_.noalias()=d.inner->Jx_*A;
    d.Ju_.noalias()=d.inner->Jx_*B;d.Ju_+=d.inner->Ju_;
  }
  aligator::shared_ptr<BaseData> createData() const override {
    return std::make_shared<MappedData>(*this,*sub);
  }
};
struct GeometryData : BaseData {
  pinocchio::Data pin;
  pinocchio::GeometryData geom;
  Matrix6 j1,j2;
  GeometryData(const Fn &f,const pinocchio::Model &m,const pinocchio::GeometryModel &g)
    :BaseData(f),pin(m),geom(g),j1(6,m.nv),j2(6,m.nv) {
    for(auto &request:geom.distanceRequests) {
      request.gjk_tolerance=1e-12;request.epa_tolerance=1e-12;
      request.gjk_max_iterations=256;request.epa_max_iterations=256;
    }
  }
};
struct Geometry : Fn {
  pinocchio::Model model;
  pinocchio::GeometryModel geom;
  Vector margins;
  Geometry(int nu,const pinocchio::Model &m,const pinocchio::GeometryModel &g,const Vector &bounds)
    :Fn(2*m.nv,nu,g.collisionPairs.size()),model(m),geom(g),margins(bounds) {
    if(bounds.size()!=nr) throw std::invalid_argument("Collision margin dimensions");
  }
  void evaluate(const Ref &x,const Ref &,BaseData &base) const override {
    auto &d=static_cast<GeometryData&>(base);
    pinocchio::computeJointJacobians(model,d.pin,x.head(model.nq));
    pinocchio::updateFramePlacements(model,d.pin);
    pinocchio::updateGeometryPlacements(model,d.pin,geom,d.geom);
    for(int i=0;i<nr;++i) {
      pinocchio::computeDistance(geom,d.geom,i);
      d.value_[i]=std::max(0.,margins[i]-d.geom.distanceResults[i].min_distance);
    }
  }
  void computeJacobians(const Ref &,const Ref &,BaseData &base) const override {
    auto &d=static_cast<GeometryData&>(base);
    d.Jx_.setZero();d.Ju_.setZero();
    for(int i=0;i<nr;++i) {
      if(d.value_[i]<=0.) continue;
      const auto &pair=geom.collisionPairs[i];
      const auto &result=d.geom.distanceResults[i];
      const auto f1=geom.geometryObjects[pair.first].parentFrame;
      const auto f2=geom.geometryObjects[pair.second].parentFrame;
      d.j1.setZero();d.j2.setZero();
      pinocchio::getFrameJacobian(model,d.pin,f1,pinocchio::LOCAL_WORLD_ALIGNED,d.j1);
      pinocchio::getFrameJacobian(model,d.pin,f2,pinocchio::LOCAL_WORLD_ALIGNED,d.j2);
      const Eigen::Vector3d r1=result.nearest_points[0]-d.pin.oMf[f1].translation();
      const Eigen::Vector3d r2=result.nearest_points[1]-d.pin.oMf[f2].translation();
      d.j1.topRows<3>()-=pinocchio::skew(r1)*d.j1.bottomRows<3>();
      d.j2.topRows<3>()-=pinocchio::skew(r2)*d.j2.bottomRows<3>();
      d.Jx_.row(i).head(model.nv)=result.normal.transpose()*(d.j1.topRows<3>()-d.j2.topRows<3>());
    }
  }
  aligator::shared_ptr<BaseData> createData() const override {
    return std::make_shared<GeometryData>(*this,model,geom);
  }
};
struct SignedGeometry : Geometry {
  using Geometry::Geometry;
  void evaluate(const Ref &x,const Ref &u,BaseData &base) const override {
    Geometry::evaluate(x,u,base);auto &d=static_cast<GeometryData&>(base);
    for(int i=0;i<nr;++i)d.value_[i]=margins[i]-d.geom.distanceResults[i].min_distance;
  }
  void computeJacobians(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<GeometryData&>(base);const Vector saved=d.value_;
    d.value_.setOnes();Geometry::computeJacobians(x,u,base);d.value_=saved;
  }
};
}
BOOST_PYTHON_MODULE(_public_models) {
  namespace bp=boost::python;
  using namespace fence_public;
  bp::import("aligator.pyaligator");
  bp::class_<ImpulseCone,bp::bases<Fn>>("ImpulseCone",
      bp::init<int,const pinocchio::Model&,const Eigen::VectorXi&,const Eigen::VectorXi&,
               double,double,bool,const Vector&>())
      .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
  bp::class_<SignedGeometry,bp::bases<Fn>>("SignedGeometry",
      bp::init<int,const pinocchio::Model&,const pinocchio::GeometryModel&,const Vector&>())
      .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
  bp::class_<MappedState,bp::bases<Fn>>("MappedState",
      bp::init<xyz::polymorphic<Fn>,const Matrix&,const Matrix&>())
      .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
  bp::class_<Geometry,bp::bases<Fn>>("Geometry",
      bp::init<int,const pinocchio::Model&,const pinocchio::GeometryModel&,const Vector&>())
      .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
  bp::class_<CollocationInverse,bp::bases<Fn>>("CollocationInverse",
      bp::init<const pinocchio::Model&,const Eigen::VectorXi&,const Eigen::VectorXi&,const Eigen::VectorXi&,double>())
      .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
}
