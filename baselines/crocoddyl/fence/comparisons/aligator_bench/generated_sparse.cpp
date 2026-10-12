// Public Aligator StageFunction adapter for stateless CasADi-generated C.
// No solver implementation is defined or changed here.
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include "aligator/context.hpp"
#include "aligator/core/function-abstract.hpp"
#include "aligator/python/polymorphic.hpp"

using Fn = aligator::context::StageFunction;
using Data = aligator::context::StageFunctionData;
using Ref = aligator::context::ConstVectorRef;
using Eval = int (*)(const double **, double **, long long *, double *, int);
using Work = int (*)(long long *, long long *, long long *, long long *);
namespace sparse_adapter {
struct Library {
  void *handle;
  Eval value, jac;
  std::vector<long long> jx_index,ju_index;
  long long na=0, nr=0, ni=0, nw=0;
  Library(const std::string &path, const std::string &name) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
    value = reinterpret_cast<Eval>(dlsym(handle, name.c_str()));
    jac = reinterpret_cast<Eval>(dlsym(handle, (name+"_jac").c_str()));
    if (!value || !jac) throw std::runtime_error("Missing generated function");
    using Sparsity = const long long *(*)(long long);
    auto sparsity=reinterpret_cast<Sparsity>(dlsym(handle,(name+"_jac_sparsity_out").c_str()));
    if (!sparsity) throw std::runtime_error("Missing Jacobian sparsity");
    for (int output=0;output<2;++output) {
      const auto *sp=sparsity(output);
      auto &index=output==0 ? jx_index : ju_index;
      const long long rows=sp[0],cols=sp[1];
      if (sp[2]==1) {
        for (long long i=0;i<rows*cols;++i) index.push_back(i);
      } else {
        const auto *col=sp+2;
        const auto *row=col+cols+1;
        for (long long c=0;c<cols;++c)
          for (long long i=col[c];i<col[c+1];++i) index.push_back(c*rows+row[i]);
      }
    }
    for (auto suffix : {"_work", "_jac_work"}) {
      auto w = reinterpret_cast<Work>(dlsym(handle,(name+suffix).c_str()));
      if (!w) throw std::runtime_error("Missing work sizes");
      long long a,b,c,d; w(&a,&b,&c,&d);
      na=std::max(na,a); nr=std::max(nr,b); ni=std::max(ni,c); nw=std::max(nw,d);
    }
  }
  ~Library() { dlclose(handle); }
};
struct GeneratedData : Data {
  std::vector<const double *> args;
  std::vector<double *> results;
  std::vector<long long> iw;
  std::vector<double> work,jx,ju;
  GeneratedData(const Fn &fn, const Library &lib) : Data(fn),
    args(lib.na,nullptr), results(lib.nr,nullptr), iw(lib.ni), work(lib.nw),jx(lib.jx_index.size()),ju(lib.ju_index.size()) {}
};
struct GeneratedResidual : Fn {
  std::shared_ptr<Library> lib;
  Eigen::VectorXd params;
  GeneratedResidual(int nx, int nu, int nr, const std::string &path,
                    const std::string &name, const Eigen::VectorXd &p)
    : Fn(nx,nu,nr), lib(std::make_shared<Library>(path,name)), params(p) {}
  void call(Ref x, Ref u, Data &data, bool derivative) const {
    auto &d=static_cast<GeneratedData &>(data);
    d.args[0]=x.data(); d.args[1]=u.data(); d.args[2]=params.data();
    d.results[0]=derivative ? d.jx.data() : d.value_.data();
    if (derivative) d.results[1]=d.ju.data();
    int status=(derivative ? lib->jac : lib->value)(d.args.data(),d.results.data(),d.iw.data(),d.work.data(),0);
    if (status) throw std::runtime_error("Generated residual evaluation failed");
    if (derivative) {
      d.Jx().setZero(); d.Ju().setZero();
      for (std::size_t i=0;i<d.jx.size();++i) d.Jx().data()[lib->jx_index[i]]=d.jx[i];
      for (std::size_t i=0;i<d.ju.size();++i) d.Ju().data()[lib->ju_index[i]]=d.ju[i];
    }
  }
  void evaluate(const Ref &x, const Ref &u, Data &data) const override { call(x,u,data,false); }
  void computeJacobians(const Ref &x, const Ref &u, Data &data) const override { call(x,u,data,true); }
  aligator::shared_ptr<Data> createData() const override {
    return std::make_shared<GeneratedData>(*this,*lib);
  }
};
} // namespace sparse_adapter
BOOST_PYTHON_MODULE(_generated_sparse) {
  using sparse_adapter::GeneratedResidual;
  namespace bp=boost::python;
  bp::import("aligator.pyaligator");
  bp::class_<GeneratedResidual,bp::bases<Fn>>("GeneratedResidual",
    bp::init<int,int,int,const std::string &,const std::string &,const Eigen::VectorXd &>())
    .def(aligator::python::PolymorphicMultiBaseVisitor<Fn>());
}
