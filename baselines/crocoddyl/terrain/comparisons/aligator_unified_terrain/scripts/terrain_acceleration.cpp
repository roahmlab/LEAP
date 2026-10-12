// Full classical contact acceleration through public Pinocchio/ALIGATOR models.
#include "../../aligator_bench/inverse_model.hpp"
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
namespace terrain_acceleration {
using namespace inverse_rk4;
struct AccelerationData : BaseData {
  pinocchio::Data pin;
  Matrix6 vq,aq,av,aa;
  AccelerationData(const Fn &fn,const pinocchio::Model &m)
    :BaseData(fn),pin(m),vq(6,m.nv),aq(6,m.nv),av(6,m.nv),aa(6,m.nv) {}
};
struct ContactAcceleration : Fn {
  pinocchio::Model model;
  Eigen::VectorXi frames,dims,active;
  int sample;
  ContactAcceleration(int nu,const pinocchio::Model &m,const Eigen::VectorXi &f,
      const Eigen::VectorXi &d,const Eigen::VectorXi &on,int index)
    :Fn(2*m.nv,nu,d.sum()),model(m),frames(f),dims(d),active(on),sample(index) {
    if(m.nq!=m.nv || f.size()!=d.size() || on.size()!=d.size() || index<0 || index>2)
      throw std::invalid_argument("Contact acceleration dimensions/sample");
  }
  void evaluate(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<AccelerationData&>(base);const int n=model.nv;
    pinocchio::forwardKinematics(model,d.pin,x.head(n),x.tail(n),u.segment(sample*n,n));
    pinocchio::updateFramePlacements(model,d.pin);d.value_.setZero();int row=0;
    for(int c=0;c<frames.size();++c) {
      if(active[c])d.value_.segment(row,dims[c])=pinocchio::getFrameClassicalAcceleration(model,d.pin,frames[c],pinocchio::LOCAL_WORLD_ALIGNED).toVector().head(dims[c]);
      row+=dims[c];
    }
  }
  void computeJacobians(const Ref &x,const Ref &u,BaseData &base) const override {
    auto &d=static_cast<AccelerationData&>(base);const int n=model.nv;
    pinocchio::computeForwardKinematicsDerivatives(model,d.pin,x.head(n),x.tail(n),u.segment(sample*n,n));
    pinocchio::updateFramePlacements(model,d.pin);d.Jx_.setZero();d.Ju_.setZero();int row=0;
    for(int c=0;c<frames.size();++c) {
      if(active[c]) {
        d.vq.setZero();d.aq.setZero();d.av.setZero();d.aa.setZero();
        pinocchio::getFrameAccelerationDerivatives(model,d.pin,frames[c],pinocchio::LOCAL,d.vq,d.aq,d.av,d.aa);
        const auto velocity=pinocchio::getFrameVelocity(model,d.pin,frames[c],pinocchio::LOCAL);
        d.aq.topRows<3>()+=pinocchio::skew(velocity.angular())*d.vq.topRows<3>()-pinocchio::skew(velocity.linear())*d.vq.bottomRows<3>();
        d.av.topRows<3>()+=pinocchio::skew(velocity.angular())*d.aa.topRows<3>()-pinocchio::skew(velocity.linear())*d.aa.bottomRows<3>();
        // Differentiate in LOCAL coordinates, then explicitly differentiate the
        // rotation into LOCAL_WORLD_ALIGNED coordinates.
        const Eigen::Matrix3d R=d.pin.oMf[frames[c]].rotation();
        const auto acc=pinocchio::getFrameClassicalAcceleration(model,d.pin,frames[c],pinocchio::LOCAL);
        d.Jx_.block(row,0,3,n)=R*d.aq.topRows<3>()-pinocchio::skew(R*acc.linear())*R*d.aa.bottomRows<3>();
        d.Jx_.block(row,n,3,n)=R*d.av.topRows<3>();
        d.Ju_.block(row,sample*n,3,n)=R*d.aa.topRows<3>();
        if(dims[c]==6) {
          d.Jx_.block(row+3,0,3,n)=R*d.aq.bottomRows<3>()-pinocchio::skew(R*acc.angular())*R*d.aa.bottomRows<3>();
          d.Jx_.block(row+3,n,3,n)=R*d.av.bottomRows<3>();
          d.Ju_.block(row+3,sample*n,3,n)=R*d.aa.bottomRows<3>();
        }
      }
      row+=dims[c];
    }
  }
  aligator::shared_ptr<BaseData> createData() const override {return std::make_shared<AccelerationData>(*this,model);}
};
}
BOOST_PYTHON_MODULE(_terrain_acceleration) {
  namespace bp=boost::python;using namespace terrain_acceleration;
  bp::import("aligator.pyaligator");
  bp::class_<ContactAcceleration,bp::bases<Fn>>("ContactAcceleration",
    bp::init<int,const pinocchio::Model&,const Eigen::VectorXi&,const Eigen::VectorXi&,const Eigen::VectorXi&,int>())
    .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
}
