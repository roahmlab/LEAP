// Observation only: the installed Aligator FDDP implementation is unchanged.
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <chrono>
#include <fstream>
#include <memory>
#include <cstring>
#include <omp.h>
#include "aligator/core/callback-base.hpp"
#include "aligator/solvers/fddp/solver-fddp.hpp"
namespace {
namespace bp=boost::python;
using Clock=std::chrono::steady_clock;
using Vectors=std::vector<Eigen::VectorXd>;
struct Deadline {};
struct CostStop {};
struct Trace: aligator::CallbackBaseTpl<double> {
  Clock::time_point start;
  double limit,threshold,overhead=0.;
  size_t width,capacity,count=0;
  bool enabled;
  std::vector<double> data;
  Trace(double seconds,double cost,size_t n,size_t cap,bool on):limit(seconds),threshold(cost),width(n+9),capacity(cap),enabled(on) {
    // Preallocate and touch pages before the solver timer starts.
    if(enabled) data.resize(width*capacity);
  }
  double elapsed() const {return std::chrono::duration<double>(Clock::now()-start).count();}
  void capture(const Results &r,double kind,double at) {
    if(!enabled)return;
    auto begin=Clock::now();
    if(count==capacity)throw std::runtime_error("Trace capacity exhausted");
    double *p=data.data()+count++*width;
    p[0]=at;p[1]=at-overhead;p[2]=overhead;p[3]=r.num_iters;p[4]=kind;
    p[5]=r.traj_cost_;p[6]=r.prim_infeas;p[7]=r.dual_infeas;p[8]=r.merit_value_;
    p+=9;
    for(const auto &x:r.xs){std::memcpy(p,x.data(),x.size()*sizeof(double));p+=x.size();}
    for(const auto &u:r.us){std::memcpy(p,u.data(),u.size()*sizeof(double));p+=u.size();}
    overhead+=std::chrono::duration<double>(Clock::now()-begin).count();
  }
  void call(const Workspace &,const Results &r) override {
    double at=elapsed();capture(r,1.,at);
    if(limit>0 && elapsed()-overhead>=limit)throw Deadline();
    if(threshold>0 && r.traj_cost_<threshold && r.prim_infeas<1e-7)throw CostStop();
  }
};
bp::dict run(aligator::SolverFDDPTpl<double> &solver,const aligator::TrajOptProblemTpl<double> &problem,
             const Vectors &xs,const Vectors &us,double limit,double threshold,const std::string &path,bool enabled) {
  size_t n=0;for(const auto &x:xs)n+=x.size();for(const auto &u:us)n+=u.size();
  auto cb=std::make_shared<Trace>(limit,threshold,n,solver.max_iters+3,enabled);
  omp_set_dynamic(0);omp_set_num_threads(solver.getNumThreads());
  solver.registerCallback("iteration_trace",cb);
  bool conv=false,timeout=false,cost_stop=false;std::string error;
  cb->start=Clock::now();
  PyThreadState *thread=PyEval_SaveThread();
  try {conv=solver.run(problem,xs,us);} catch(const Deadline &){timeout=true;}
  catch(const CostStop &){cost_stop=true;} catch(const std::exception &e){error=e.what();}
  catch(...){error="Unknown native solver exception";}
  double wall=cb->elapsed();
  double solverCopy=cb->overhead;
  // FDDP has convergence branches before invokeCallbacks(): preserve the return
  // state explicitly, including when no regular iteration callback occurred.
  cb->capture(solver.results_,2.,wall);
  PyEval_RestoreThread(thread);
  solver.removeCallback("iteration_trace");
  if(enabled){
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(cb->data.data()),cb->count*cb->width*sizeof(double));
    if(!f)throw std::runtime_error("Could not write complete trace: "+path);
  }
  bp::dict d;d["solve_wall_s"]=wall;d["trace_copy_s"]=cb->overhead;d["in_solve_copy_s"]=solverCopy;d["solve_compute_s"]=wall-solverCopy;
  d["rows"]=cb->count;d["width"]=cb->width;d["converged"]=conv;d["timeout"]=timeout;
  d["cost_stop"]=cost_stop;d["error"]=error;return d;
}
}
BOOST_PYTHON_MODULE(_trace_run){bp::import("aligator.pyaligator");bp::def("run",&run);}
