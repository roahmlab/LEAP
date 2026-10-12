// Public Crocoddyl modeling adapters; no optimizer implementation or patch.
#include "../aligator_bench/inverse_model.hpp"
#include "geometry_helpers.hpp"
#include <pinocchio/algorithm/impulse-dynamics.hpp>
#include <pinocchio/algorithm/impulse-dynamics-derivatives.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <omp.h>

namespace impact_benchmark {
namespace bp=boost::python;
using namespace inverse_rk4;
using Residual=crocoddyl::ResidualModelAbstract;
using ResidualData=crocoddyl::ResidualDataAbstract;
using Action=crocoddyl::ActionModelAbstract;
using ActionData=crocoddyl::ActionDataAbstract;
using State=crocoddyl::StateAbstract;
using Costs=crocoddyl::CostModelSum;
using Collector=crocoddyl::DataCollectorAbstract;

struct BridgeData : ResidualData {
  aligator::shared_ptr<BaseData> inner;
  BridgeData(Residual *r,Collector *c,const Fn &f):ResidualData(r,c),inner(f.createData()) {}
};
struct ResidualBridge : Residual {
  xyz::polymorphic<Fn> fn;
  ResidualBridge(std::shared_ptr<State> s,xyz::polymorphic<Fn> f):Residual(s,f->nr,f->nu,true,true,true),fn(f) {}
  std::shared_ptr<crocoddyl::ResidualModelBase> cloneAsDouble() const override {return std::make_shared<ResidualBridge>(*this);}
  std::shared_ptr<crocoddyl::ResidualModelBase> cloneAsFloat() const override {throw std::runtime_error("float64 required");}
  void calc(const std::shared_ptr<ResidualData>& b,const Ref &x,const Ref &u) override {
    auto &d=static_cast<BridgeData&>(*b);fn->evaluate(x,u,*d.inner);d.r=d.inner->value_;
  }
  void calcDiff(const std::shared_ptr<ResidualData>& b,const Ref &x,const Ref &u) override {
    auto &d=static_cast<BridgeData&>(*b);fn->computeJacobians(x,u,*d.inner);d.Rx=d.inner->Jx_;d.Ru=d.inner->Ju_;
  }
  std::shared_ptr<ResidualData> createData(Collector *const c) override {return std::make_shared<BridgeData>(this,c,*fn);}
};

// Compose stock dynamics (LQR polynomial map or plastic impulse) with stock costs.
// Event dummy controls retain rectangular archives and are penalized to zero.
struct CombinedData : ActionData {
  Collector collector;
  std::shared_ptr<ActionData> inner;
  std::shared_ptr<crocoddyl::CostDataSum> costs;
  CombinedData(Action *a,Action &f,Costs &c):ActionData(a),inner(f.createData()),costs(c.createData(&collector)) {}
};
struct CombinedAction : Action {
  std::shared_ptr<Action> dynamics;
  std::shared_ptr<Costs> costs;
  CombinedAction(std::shared_ptr<Action> f,std::shared_ptr<Costs> c):Action(c->get_state(),c->get_nu(),c->get_nr()),dynamics(f),costs(c) {
    if(f->get_nu()!=0 && f->get_nu()!=get_nu())throw std::invalid_argument("Control dimensions");
  }
  std::shared_ptr<crocoddyl::ActionModelBase> cloneAsDouble() const override {return std::make_shared<CombinedAction>(*this);}
  std::shared_ptr<crocoddyl::ActionModelBase> cloneAsFloat() const override {throw std::runtime_error("float64 required");}
  void calc(const std::shared_ptr<ActionData>& b,const Ref &x,const Ref &u) override {
    auto &d=static_cast<CombinedData&>(*b);
    if(dynamics->get_nu())dynamics->calc(d.inner,x,u);else dynamics->calc(d.inner,x);
    costs->calc(d.costs,x,u);b->xnext=d.inner->xnext;b->cost=d.inner->cost+d.costs->cost;
  }
  void calcDiff(const std::shared_ptr<ActionData>& b,const Ref &x,const Ref &u) override {
    auto &d=static_cast<CombinedData&>(*b);
    if(dynamics->get_nu())dynamics->calcDiff(d.inner,x,u);else dynamics->calcDiff(d.inner,x);
    costs->calcDiff(d.costs,x,u);b->Fx=d.inner->Fx;b->Fu.setZero();
    b->Lx=d.costs->Lx+d.inner->Lx;b->Lxx=d.costs->Lxx+d.inner->Lxx;
    b->Lu=d.costs->Lu;b->Lxu=d.costs->Lxu;b->Luu=d.costs->Luu;
    if(dynamics->get_nu()) {b->Fu=d.inner->Fu;b->Lu+=d.inner->Lu;b->Lxu+=d.inner->Lxu;b->Luu+=d.inner->Luu;}
  }
  void calc(const std::shared_ptr<ActionData>& b,const Ref &x) override {calc(b,x,Vector::Zero(get_nu()));}
  void calcDiff(const std::shared_ptr<ActionData>& b,const Ref &x) override {calcDiff(b,x,Vector::Zero(get_nu()));}
  std::shared_ptr<ActionData> createData() override {return std::make_shared<CombinedData>(this,*dynamics,*costs);}
};

using RCM=pinocchio::RigidConstraintModel;
using RCD=pinocchio::RigidConstraintData;
struct ConeData : BaseData {
  pinocchio::Data pin;
  std::vector<RCD> cds;
  ConeData(const Fn &f,const pinocchio::Model &m,const std::vector<RCM>& cms):BaseData(f),pin(m) {
    for(const auto &c:cms)cds.emplace_back(c);pinocchio::initConstraintDynamics(m,pin,cms,cds);
  }
};
struct ImpulseCone : Fn {
  pinocchio::Model model;
  std::vector<RCM> cms;
  double mu,eps;
  ImpulseCone(int nu,const pinocchio::Model &m,const bp::object &contacts,double friction,double smoothing)
    :Fn(2*m.nv,nu,2*bp::len(contacts)),model(m),mu(friction),eps(smoothing) {
    for(int i=0;i<bp::len(contacts);++i)cms.push_back(bp::extract<RCM>(contacts[i]));
  }
  void evaluate(const Ref &x,const Ref &,BaseData &b) const override {
    auto &d=static_cast<ConeData&>(b);
    pinocchio::ProximalSettings settings(1e-12,1e-9,1);
    pinocchio::impulseDynamics(model,d.pin,x.head(model.nq),x.tail(model.nv),cms,d.cds,0.,settings);
    int off=0;
    for(size_t i=0;i<cms.size();++i) {
      auto f=d.pin.impulse_c.segment<3>(off);
      d.value_[2*i]=std::sqrt(f[0]*f[0]+f[1]*f[1]+eps*eps)-eps-mu*f[2];
      d.value_[2*i+1]=-f[2];off+=(cms[i].type==pinocchio::CONTACT_3D ? 3 : 6);
    }
  }
  void computeJacobians(const Ref &,const Ref &,BaseData &b) const override {
    auto &d=static_cast<ConeData&>(b);d.pin.dvc_dq.setZero();
    pinocchio::ProximalSettings settings(1e-12,1e-9,1);
    pinocchio::computeImpulseDynamicsDerivatives(model,d.pin,cms,d.cds,0.,settings);
    d.Jx_.setZero();d.Ju_.setZero();int off=0;
    for(size_t i=0;i<cms.size();++i) {
      auto f=d.pin.impulse_c.segment<3>(off);double norm=std::sqrt(f[0]*f[0]+f[1]*f[1]+eps*eps);
      Eigen::RowVector3d grad(f[0]/norm,f[1]/norm,-mu);
      d.Jx_.block(2*i,0,1,model.nv)=grad*d.pin.dlambda_dq.middleRows(off,3);
      d.Jx_.block(2*i,model.nv,1,model.nv)=grad*d.pin.dlambda_dv.middleRows(off,3);
      d.Jx_.block(2*i+1,0,1,model.nv)=-d.pin.dlambda_dq.row(off+2);
      d.Jx_.block(2*i+1,model.nv,1,model.nv)=-d.pin.dlambda_dv.row(off+2);off+=(cms[i].type==pinocchio::CONTACT_3D ? 3 : 6);
    }
  }
  aligator::shared_ptr<BaseData> createData() const override {return std::make_shared<ConeData>(*this,model,cms);}
};

// Exactly LEAP's rounded-outside / separating-face-inside obstacle residual,
// and its normalized squared capsule self-distance. No extra face certificate.
struct GeometryData : BaseData {
  pinocchio::Data pin;
  std::vector<Eigen::Vector3d> points;
  std::vector<Matrix> jac;
  GeometryData(const Fn &f,const pinocchio::Model &m,int count):BaseData(f),pin(m),points(count),jac(count,Matrix(3,m.nv)) {}
};
struct Geometry : Fn {
  pinocchio::Model model;
  Matrix caps,boxes,pairs;
  double margin;
  Geometry(int nu,const pinocchio::Model &m,const Matrix &c,const Matrix &b,const Matrix &p,double gap)
    :Fn(2*m.nv,nu,c.rows()*b.rows()+p.rows()),model(m),caps(c),boxes(b),pairs(p),margin(gap) {}
  void evaluate(const Ref &x,const Ref &,BaseData &base) const override {
    using namespace benchmark_geometry;
    auto &d=static_cast<GeometryData&>(base);d.Jx_.setZero();d.Ju_.setZero();
    pinocchio::computeJointJacobians(model,d.pin,x.head(model.nq));pinocchio::updateFramePlacements(model,d.pin);
    Matrix6 J(6,model.nv);
    for(int c=0;c<caps.rows();++c)for(int end=0;end<2;++end) {
      int fid=int(caps(c,end));Eigen::Vector3d local=caps.block(c,3+3*end,1,3).transpose();
      d.points[2*c+end]=d.pin.oMf[fid].act(local);J.setZero();
      pinocchio::getFrameJacobian(model,d.pin,fid,pinocchio::LOCAL_WORLD_ALIGNED,J);
      d.jac[2*c+end]=J.topRows<3>()-pinocchio::skew(d.points[2*c+end]-d.pin.oMf[fid].translation())*J.bottomRows<3>();
    }
    int row=0;
    for(int c=0;c<caps.rows();++c)for(int k=0;k<boxes.rows();++k,++row) {
      CuboidObstacle box;box.center=boxes.row(k).head<3>();box.half=boxes.row(k).tail<3>();
      const auto &a=d.points[2*c],&b=d.points[2*c+1];double safe=caps(c,2)+margin;
      auto close=closestSegmentBox(a,b,box);
      if(close.dist2>1e-20) {
        double dist=std::sqrt(close.dist2);d.value_[row]=safe-dist;
        d.Jx_.row(row).head(model.nv)=-(close.delta/dist).transpose()*((1-close.t)*d.jac[2*c]+close.t*d.jac[2*c+1]);
      } else {
        auto sep=separateSegmentBox(a,b,box,bottomFaceUnreachable(box,safe));d.value_[row]=safe-sep.offset;
        d.Jx_.row(row).head(model.nv)=-sep.normal.transpose()*d.jac[2*c+sep.end];
      }
    }
    for(int k=0;k<pairs.rows();++k,++row) {
      int i=int(pairs(k,0)),j=int(pairs(k,1));double safe=pairs(k,2);
      auto close=closestSegmentSegment(d.points[2*i],d.points[2*i+1],d.points[2*j],d.points[2*j+1]);
      d.value_[row]=(safe*safe-close.dist2)/(2*safe);
      d.Jx_.row(row).head(model.nv)=-close.w.transpose()/safe*((1-close.t)*d.jac[2*i]+close.t*d.jac[2*i+1]-(1-close.s)*d.jac[2*j]-close.s*d.jac[2*j+1]);
    }
  }
  void computeJacobians(const Ref &,const Ref &,BaseData &) const override {} // computed with the value
  aligator::shared_ptr<BaseData> createData() const override {return std::make_shared<GeometryData>(*this,model,2*caps.rows());}
};

struct RequestedStop {};
struct Recorder : crocoddyl::CallbackAbstract {
  bp::object callback;
  explicit Recorder(bp::object cb):callback(cb) {}
  void operator()(crocoddyl::SolverAbstract &s) override {
    PyGILState_STATE gil=PyGILState_Ensure();bool stop=false;
    try {stop=bp::extract<bool>(callback(bp::ptr(&s)));}catch(...) {PyGILState_Release(gil);throw;}
    PyGILState_Release(gil);if(stop)throw RequestedStop();
  }
};
bp::tuple run(crocoddyl::SolverFDDP &s,const std::vector<Vector>& xs,const std::vector<Vector>& us,int n,double reg,bp::object cb,int threads) {
  omp_set_dynamic(0);omp_set_num_threads(threads);s.setCallbacks({std::make_shared<Recorder>(cb)});
  PyThreadState *state=PyEval_SaveThread();bool ok=false,stopped=false;
  try {ok=s.solve(xs,us,n,false,reg);}catch(const RequestedStop&) {stopped=true;}
  catch(...) {PyEval_RestoreThread(state);s.setCallbacks({});throw;}
  PyEval_RestoreThread(state);s.setCallbacks({});return bp::make_tuple(ok,stopped);
}
}
BOOST_PYTHON_MODULE(_impact_native) {
  using namespace impact_benchmark;
  bp::import("crocoddyl");bp::import("aligator.pyaligator");
  bp::class_<ResidualBridge,bp::bases<Residual>,std::shared_ptr<ResidualBridge>,boost::noncopyable>("ResidualBridge",bp::init<std::shared_ptr<State>,xyz::polymorphic<Fn>>())
    .def("createData",&ResidualBridge::createData)
    .def("calc",&ResidualBridge::calc)
    .def("calcDiff",&ResidualBridge::calcDiff);
  bp::class_<CombinedAction,bp::bases<Action>,std::shared_ptr<CombinedAction>,boost::noncopyable>("CombinedAction",bp::init<std::shared_ptr<Action>,std::shared_ptr<Costs>>())
    .def("createData",&CombinedAction::createData)
    .def("calc",static_cast<void(CombinedAction::*)(const std::shared_ptr<ActionData>&,const Ref&,const Ref&)>(&CombinedAction::calc))
    .def("calcDiff",static_cast<void(CombinedAction::*)(const std::shared_ptr<ActionData>&,const Ref&,const Ref&)>(&CombinedAction::calcDiff));
  bp::class_<ImpulseCone,bp::bases<Fn>>("ImpulseCone",bp::init<int,const pinocchio::Model&,const bp::object&,double,double>()).def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
  bp::class_<Geometry,bp::bases<Fn>>("Geometry",bp::init<int,const pinocchio::Model&,const Matrix&,const Matrix&,const Matrix&,double>()).def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
  bp::def("run",&run);
}
