// Stock FDDP, preallocated primal snapshots, no filesystem I/O during run().
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <chrono>
#include <fstream>
#include <memory>
#include <omp.h>
#include "aligator/context.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/core/callback-base.hpp"
#include "aligator/solvers/fddp/solver-fddp.hpp"
namespace bp=boost::python;
using Clock=std::chrono::steady_clock;
using Vectors=std::vector<Eigen::VectorXd>;
struct Stop {};
struct Trace: aligator::CallbackBaseTpl<double> {
  std::vector<double> primal,metrics;
  size_t width,capacity,count=0;
  double overhead=0,limit,threshold;
  bool timeout=false,cost_stop=false;
  Clock::time_point start;
  Trace(size_t w,size_t n,double seconds,double cost):primal(w*n),metrics(8*n),width(w),capacity(n),limit(seconds),threshold(cost) {}
  void capture(const Vectors &xs,const Vectors &us,double iter,double cost,double prim,double dual) {
    auto begin=Clock::now();
    if(count>=capacity) throw std::runtime_error("Trace capacity exhausted");
    double raw=std::chrono::duration<double>(begin-start).count();
    double* dst=primal.data()+count*width;
    for(const auto &v:xs) {std::copy(v.data(),v.data()+v.size(),dst);dst+=v.size();}
    for(const auto &v:us) {std::copy(v.data(),v.data()+v.size(),dst);dst+=v.size();}
    auto m=metrics.data()+8*count;
    m[0]=iter;m[1]=raw;m[2]=raw-overhead;m[4]=cost;m[5]=prim;m[6]=dual;
    m[7]=std::chrono::duration<double>(Clock::now()-begin).count();
    overhead+=m[7];m[3]=overhead;++count;
  }
  void call(const Workspace &,const Results &r) override {
    capture(r.xs,r.us,double(r.num_iters),r.traj_cost_,r.prim_infeas,r.dual_infeas);
    timeout=limit>0 && metrics[8*(count-1)+2]>=limit;
    cost_stop=r.traj_cost_<threshold && r.prim_infeas<1e-7;
    if(timeout || cost_stop) throw Stop();
  }
};
bp::tuple run(aligator::SolverFDDPTpl<double> &solver,const aligator::TrajOptProblemTpl<double> &problem,const Vectors &xs,const Vectors &us,double limit,double threshold,const std::string &prefix,int capacity) {
  size_t width=0;for(auto &v:xs)width+=v.size();for(auto &v:us)width+=v.size();
  auto cb=std::make_shared<Trace>(width,capacity+2,limit,threshold);
  omp_set_dynamic(0);omp_set_num_threads(int(solver.getNumThreads()));
  solver.registerCallback("__iteration_trace__",cb);
  bool conv=false;
  cb->start=Clock::now();
  PyThreadState *thread=PyEval_SaveThread();
  try {conv=solver.run(problem,xs,us);} catch(const Stop &) {} catch(...) {
    PyEval_RestoreThread(thread);solver.removeCallback("__iteration_trace__");throw;
  }
  const double raw=std::chrono::duration<double>(Clock::now()-cb->start).count();
  const double capture_s=cb->overhead;
  // Some native exit branches skip callbacks. Retain the returned primal too.
  cb->capture(solver.results_.xs,solver.results_.us,-1,solver.results_.traj_cost_,solver.results_.prim_infeas,solver.results_.dual_infeas);
  PyEval_RestoreThread(thread);solver.removeCallback("__iteration_trace__");
  for(auto suffix:{".primal.bin",".metrics.bin"}) {
    std::ofstream f(prefix+suffix,std::ios::binary);
    bool p=std::string(suffix)==".primal.bin";
    f.write(reinterpret_cast<const char*>(p?cb->primal.data():cb->metrics.data()),sizeof(double)*cb->count*(p?width:8));
    if(!f)throw std::runtime_error("Failed to write trace");
  }
  return bp::make_tuple(conv,cb->timeout,cb->cost_stop,raw,capture_s,cb->count,width);
}
BOOST_PYTHON_MODULE(_trace_run) {bp::import("aligator.pyaligator");bp::def("run_fddp",&run);}
