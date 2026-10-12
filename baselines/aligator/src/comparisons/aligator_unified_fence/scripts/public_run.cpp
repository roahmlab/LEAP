// GIL-releasing, wall-clock-capped entry points around the installed
// Aligator solvers. The solver implementations are the installed library;
// this file only pins the OpenMP team, releases the GIL and registers a
// stock CallbackBase that raises after a fixed number of seconds so that
// the sole solver call returns its last accepted iterate at the cap.
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <chrono>
#include <exception>
#include <memory>
#include <omp.h>
#include "aligator/context.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/core/callback-base.hpp"
#include "aligator/solvers/fddp/solver-fddp.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"

namespace {
namespace bp = boost::python;
using Problem = aligator::TrajOptProblemTpl<double>;
using Vectors = std::vector<Eigen::VectorXd>;

struct TimeLimitExceeded : std::exception {
  const char *what() const noexcept override { return "wall-clock limit"; }
};

struct CostLimitReached : std::exception {};

struct TimeLimit : aligator::CallbackBaseTpl<double> {
  std::chrono::steady_clock::time_point start;
  double limit,cost_limit;
  bool cost_reached=false;
  std::size_t calls = 0;
  explicit TimeLimit(double seconds,double threshold)
      : start(std::chrono::steady_clock::now()), limit(seconds),cost_limit(threshold) {}
  void call(const Workspace &, const Results &result) override {
    if(result.traj_cost_ < cost_limit && result.prim_infeas < 1e-7) {
      cost_reached=true; throw CostLimitReached();
    }
    ++calls;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (limit > 0 && elapsed > limit) throw TimeLimitExceeded();
  }
};

template <class Solver, class Runner>
bp::tuple guarded(Solver &solver, double limit,double cost_limit, Runner runner) {
  omp_set_dynamic(0);
  omp_set_num_threads(static_cast<int>(solver.getNumThreads()));
  auto cb = std::make_shared<TimeLimit>(limit,cost_limit);
  solver.registerCallback("__wall_clock_limit__", cb);
  PyThreadState *thread = PyEval_SaveThread();
  bool conv = false, timed_out = false;
  try {
    conv = runner();
  } catch (const CostLimitReached &) {
    // Stop orchestration and audit the accepted primal independently.
  } catch (const TimeLimitExceeded &) {
    timed_out = true;
  } catch (...) {
    PyEval_RestoreThread(thread);
    (void)solver.removeCallback("__wall_clock_limit__");
    throw;
  }
  PyEval_RestoreThread(thread);
  (void)solver.removeCallback("__wall_clock_limit__");
  return bp::make_tuple(conv, timed_out, cb->cost_reached);
}

bp::tuple run_fddp(aligator::SolverFDDPTpl<double> &solver,
                   const Problem &problem, const Vectors &xs,
                   const Vectors &us, double limit,double cost_limit) {
  return guarded(solver, limit,cost_limit, [&] { return solver.run(problem, xs, us); });
}

bp::tuple run_proxddp(aligator::SolverProxDDPTpl<double> &solver,
                      const Problem &problem, const Vectors &xs,
                      const Vectors &us, double limit,double cost_limit) {
  const Vectors vs, lams;
  return guarded(solver, limit,cost_limit,
                 [&] { return solver.run(problem, xs, us, vs, lams); });
}
}  // namespace

BOOST_PYTHON_MODULE(_public_run) {
  bp::import("aligator.pyaligator");
  bp::def("run_fddp", &run_fddp);
  bp::def("run_proxddp", &run_proxddp);
}
