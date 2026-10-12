#!/usr/bin/env python3
"""Impact-cone fence benchmark; stock FDDP, preparation and physical records."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from physics import ROOT,ROBOTS

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.set_defaults(family='fence')
    ap.add_argument('--prepare-only',action='store_true',help='Build, archive and audit the cold seed; never instantiate an optimizer')
    ap.add_argument('--robot',choices=ROBOTS,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cpus',required=True,help='1 to 12 comma-separated distinct physical CPU IDs')
    ap.add_argument('--wall-seconds',type=float,default=86400.)
    ap.add_argument('--solver-seconds',type=float,default=14400.)
    ap.add_argument('--iterations',type=int,default=10000)
    ap.add_argument('--rounds',type=int,default=40)
    ap.add_argument('--dt',type=float,default=.01)
    ap.add_argument('--reg-init',type=float,default=1e-3)
    ap.add_argument('--th-stop',type=float,default=1e-12)
    ap.set_defaults(hold='linear')
    args=ap.parse_args(argv)
    cpus=[int(v) for v in args.cpus.split(',')]
    if not 1<=len(cpus)<=12 or len(set(cpus))!=len(cpus):ap.error('Use at most 12 distinct physical CPUs')
    if min(args.wall_seconds,args.solver_seconds,args.dt,args.reg_init,args.iterations,args.rounds)<=0:ap.error('Positive budgets/settings required')
    if args.output.exists():ap.error('Output must be a new directory')
    allowed=os.sched_getaffinity(0)
    if not set(cpus)<=allowed:ap.error('Requested CPUs are outside the available affinity')
    physical=[]
    for cpu in cpus:
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.append(((topology/'physical_package_id').read_text(),(topology/'core_id').read_text()))
    if len(set(physical))!=len(cpus):ap.error('CPU list repeats a physical core through SMT')
    os.sched_setaffinity(0,cpus);os.environ['OMP_NUM_THREADS']=str(len(cpus));args.output.mkdir(parents=True)
    start=time.perf_counter()
    import numpy as np,crocoddyl,pinocchio,aligator
    from physics import build,audit
    from trace import Trace,atomic_json
    from _impact_native import run
    trace=None;record={};xs=us=None;built=None
    try:
        t=time.perf_counter();built=build(args.robot,args.dt);build_seconds=time.perf_counter()-t
        built.problem.nthreads=len(cpus)
        physical_inputs=[built.info.path,built.info.r.yaml_path,built.info.r.urdf_path,built.info.seed.path,built.generated_library]
        record=dict(stock_components=['SolverFDDP','ShootingProblem','ActionModelLQR','ActionModelImpulseFwdDynamics','CostModelSum','CostModelResidual','ActivationModelQuadraticBarrier'],schema='crocoddyl-impact-cone-v1',family=args.family,robot=args.robot,settings=vars(args)|{'output':str(args.output)},
          command=[sys.executable,str(Path(__file__).resolve()),*(argv if argv is not None else sys.argv[1:])],
          versions=dict(crocoddyl=crocoddyl.__version__,pinocchio=pinocchio.__version__,model_api=aligator.__version__),
          crocoddyl_module=crocoddyl.__file__,
          crocoddyl_library_sha256={p:digest(p) for p in sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libcrocoddyl.so' in line))},
          crocoddyl_release_commit='174c6082f0aef4dca80fa767f47fe025d6c4fef1',
          solver='unmodified installed crocoddyl.SolverFDDP',model_dependency='ALIGATOR public modeling API only; no ALIGATOR optimization',
          objective='feasibility restoration; physical objective identically zero',geometry=built.geometry,
          input_sha256={str(p.relative_to(ROOT)):digest(p) for p in physical_inputs},
          seed_sha256=built.info.seed.sha256,cpus=cpus,nthreads=int(built.problem.nthreads),
          host=os.uname().nodename,machine_load=list(os.getloadavg()),
          mesh=built.mesh,hold=built.hold,build_s=build_seconds,symbolic_digest=built.symbolic_digest,
          impact_cone=True,stance_acceleration=False,fence_stance_acceleration_note='Reference LEAP fence stanceAcc=false',calls=[])
        # Archive every dependency used by the physical factory and verifier.
        source_files=list(HERE.glob('*.hpp'))+list(HERE.glob('*.md'))+list(HERE.glob('*.py'))+list(HERE.glob('*.cpp'))+list(HERE.glob('*.so'))+list(HERE.glob('*.sh'))+list(HERE.glob('*.patch'))
        for folder in ['common','aligator_bench','aligator_unified_fence/scripts']:
            source_files.extend(p for p in (ROOT/'comparisons'/folder).iterdir() if p.suffix in ('.py','.cpp','.hpp','.so'))
        import shutil
        record['source_sha256']={}
        for p in source_files:
            relative=p.relative_to(ROOT/'comparisons');destination=args.output/'executed_source'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination);record['source_sha256'][str(relative)]=digest(p)
        for p in physical_inputs:
            destination=args.output/'inputs'/p.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
        atomic_json(args.output/'protocol.json',record)
        xs,us=built.xs,built.us
        np.savez_compressed(args.output/'initial.npz',xs=np.array(xs),us=np.array(us),mesh_json=json.dumps(built.mesh))
        trace=Trace(args.output,built,start,args.wall_seconds,args.solver_seconds);trace.build_seconds=build_seconds
        trace.write(xs,us,event='initial')
        if args.prepare_only:
            record.update(prepared=True,optimization_started=False,initial_audit=trace.last_report)
            atomic_json(args.output/'prepared.json',record)
            print(json.dumps(dict(robot=args.robot,prepared=True,optimization_started=False)),flush=True)
            return 0
        for outer in range(args.rounds):
            if trace.reason:break
            trace.round=outer
            solver=crocoddyl.SolverFDDP(built.problem)
            solver.th_stop=args.th_stop
            if hasattr(solver,'th_minImprove'):solver.th_minImprove=1e-16
            if hasattr(solver,'reg_min'):solver.reg_min=1e-9
            trace.last_end=time.perf_counter();before=trace.solver_seconds
            converged,stopped=run(solver,xs,us,args.iterations,args.reg_init,trace,len(cpus))
            trace.solver_seconds+=time.perf_counter()-trace.last_end
            xs=[np.array(x).copy() for x in solver.xs];us=[np.array(u).copy() for u in solver.us]
            call=dict(round=outer,native_converged=bool(converged),callback_stopped=bool(stopped),native_iteration=int(solver.iter),solver_s=trace.solver_seconds-before,cost=float(solver.cost),native_stop=float(solver.stop))
            record['calls'].append(call)
            # The terminal iterate is archived even if no native callback fired.
            if not trace.reason:trace.write(xs,us,solver,event='call_return')
            atomic_json(args.output/'calls.json',record['calls'])
            # Stop when the native solver converges to a nonzero restoration minimum.
            if converged and not trace.passed:
                trace.reason='native_converged_infeasible';break
            if not trace.reason and solver.iter+1<args.iterations:
                trace.reason='native_stopped_infeasible';break
        final=audit(built,xs,us)
        np.savez_compressed(args.output/'solution.npz',xs=np.array(xs),us=np.array(us),mesh_json=json.dumps(built.mesh))
        record.update(passed=final['passed'],audit=final,stop_reason=trace.reason or 'iteration_budget',solver_s=trace.solver_seconds,wall_s=time.perf_counter()-start,logging_s=trace.overhead,iterations_saved=trace.index,solution_sha256=digest(args.output/'solution.npz'))
        atomic_json(args.output/'result.json',record)
        print(json.dumps({k:record[k] for k in ['family','robot','passed','stop_reason','solver_s','wall_s','iterations_saved']}),flush=True)
        return 0 if final['passed'] else 2
    except BaseException:
        record.update(passed=False,error=traceback.format_exc(),wall_s=time.perf_counter()-start)
        atomic_json(args.output/'failure.json',record)
        raise
    finally:
        if trace is not None:trace.close()

if __name__=='__main__':raise SystemExit(main())
