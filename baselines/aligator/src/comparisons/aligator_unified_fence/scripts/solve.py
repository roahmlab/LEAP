#!/usr/bin/env python3
"""Native Aligator continuation from the supplied unified fence initial guess."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import sys

os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='FALSE')


def write_json(path, value):
    def clean(v):
        if hasattr(v,'tolist'):return clean(v.tolist())
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,float) and not __import__('math').isfinite(v): return None
        return v
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def save(path, mesh, xs, us):
    import numpy as np
    np.savez_compressed(path,xs=np.array(xs),controls=np.concatenate(us),
                        offsets=np.r_[0,np.cumsum([len(u) for u in us])],mesh_json=json.dumps(mesh))


def safe_audit(info,mesh,xs,us):
    from verify import audit
    try:
        return audit(info,mesh,xs,us)
    except Exception as exc:
        return dict(passed=False,audit_error=repr(exc))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot',required=True)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--dt',type=float,default=.005)
    ap.add_argument('--threads',type=int,default=12)
    ap.add_argument('--core-offset',type=int,default=0)
    ap.add_argument('--iterations',type=int,default=1000)
    ap.add_argument('--call-seconds',type=float,default=180.)
    ap.add_argument('--rounds',type=int,default=4)
    ap.add_argument('--reg-init',type=float,default=1e-3)
    ap.add_argument('--resume',type=Path,help='A finite candidate from this experiment; preserve its seed provenance')
    ap.add_argument('--direct',action='store_true')
    ap.add_argument('--restoration-only',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args()
    if args.out.exists(): ap.error('Output directory must be new')
    if min(args.dt,args.threads,args.iterations,args.call_seconds,args.rounds,args.reg_init)<=0: ap.error('Positive settings required')
    args.out.mkdir(parents=True)
    import shutil
    (args.out/'executed_source').mkdir()
    for source in Path(__file__).parent.iterdir():
        if source.is_file() and source.suffix in ('.py','.cpp'):
            shutil.copy2(source,args.out/'executed_source'/source.name)
    from seed import ROOT, ROBOTS
    if args.robot not in ROBOTS: ap.error('Expected one of '+', '.join(ROBOTS))
    from aligator_bench.solve_fence_uniform import pin_physical_cores
    cpus=pin_physical_cores(args.threads,args.core_offset)
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    import numpy as np
    import aligator as al
    import pinocchio as pin
    from aligator_bench import _uniform_run
    from problem import inputs,build,configure_data
    from verify import audit
    started=time.perf_counter()
    info,mesh,xs,us,imported=inputs(args.robot,args.dt)
    if args.resume is not None:
        from transfer import transfer
        args.resume=args.resume.resolve()
        xs,us=transfer(info,mesh,args.resume)
    save(args.out/'initial.npz',mesh,xs,us)
    np.savez_compressed(args.out/'imported_seed_fields.npz',**{key:np.array([v[key] for v in imported]) for key in imported[0]})
    record=dict(robot=args.robot,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],
        aligator_version=al.__version__,pinocchio_version=pin.__version__,cpus=cpus,
        seed_sha256=info.seed.sha256,seed_node_import_error=info.seed.check_nodes(),
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [info.path,info.r.yaml_path,info.r.urdf_path]},
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir()
                 if p.is_file() and p.suffix in ('.py','.cpp')},
        stages=len(mesh),state_dimension=2*info.r.nv,
        stored_primal_coordinates=sum(len(x) for x in xs)+sum(len(u) for u in us),calls=[])
    from test_models import check
    record['preflight']=check(args.robot)
    if args.resume is not None:
        record['parent_candidate_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
        record['initialization']='mesh continuation from a candidate whose lineage starts at the supplied raw seed'
    else:
        record['initialization']='supplied raw seed, without projection or control clipping'
    write_json(args.out/'preflight.json',record['preflight'])
    write_json(args.out/'config.json',record)
    print(f'{args.robot}: imported original seed; {len(mesh)} stages',flush=True)
    # The untouched initial decision variables enter the first native call.
    # Only returned finite trajectories can initialize subsequent calls.
    recipe=[] if args.direct else [(scale,True,1e-3) for scale in [.25,.5,.75,1.]]
    recipe += [(1.,args.restoration_only,1e-5)]*args.rounds
    best=None
    for j,(scale,restoration,barrier) in enumerate(recipe):
        t0=time.perf_counter()
        built=build(info,mesh,xs,obstacle_scale=scale,restoration=restoration,barrier=barrier)
        if restoration:
            solver=al.SolverFDDP(1e-8,al.VERBOSE,args.reg_init,args.iterations)
            solver.force_initial_condition=True
            solver.reg_min,solver.reg_max=1e-8,1e14
            run=_uniform_run.run_fddp
        else:
            solver=al.SolverProxDDP(1e-7,.01,args.iterations,al.VERBOSE)
            solver.max_al_iters=1000
            solver.reg_min,solver.reg_max=1e-9,1e14
            run=_uniform_run.run_proxddp
        solver.setNumThreads(args.threads)
        solver.setup(built.problem)
        configured=configure_data(solver.workspace.problem_data,info.prox)
        history=al.HistoryCallback(solver,False,True)
        solver.registerCallback('history',history)
        call=dict(index=j,solver='FDDP' if restoration else 'ProxDDP',obstacle_scale=scale,
                  barrier=barrier if restoration else None,build_s=time.perf_counter()-t0,
                  native_force_data_configured=configured)
        if args.preflight_only:
            built.problem.evaluate(xs,us,solver.workspace.problem_data)
            built.problem.computeDerivatives(xs,us,solver.workspace.problem_data)
            call['evaluated']=True
            record['calls'].append(call); break
        print(f'call {j}: {call["solver"]}, obstacle height {scale}, build {call["build_s"]:.2f}s',flush=True)
        t0=time.perf_counter()
        try:
            converged,timeout,_=run(solver,built.problem,xs,us,args.call_seconds)
            call.update(converged=bool(converged),timeout=bool(timeout))
        except Exception as exc:
            call.update(converged=False,timeout=False,error=repr(exc))
        call.update(solve_s=time.perf_counter()-t0,iterations=int(solver.results.num_iters),
                    cost=float(solver.results.traj_cost),
                    native_primal=float(solver.results.prim_infeas) if hasattr(solver.results,'prim_infeas')
                    else float(getattr(solver.results,'primal_infeas',float('nan'))))
        newxs=[np.array(x).copy() for x in solver.results.xs]
        newus=[np.array(u).copy() for u in solver.results.us]
        finite=all(np.isfinite(x).all() for x in newxs+newus)
        call['finite']=finite
        save(args.out/f'call_{j:02d}.npz',mesh,newxs,newus)
        if finite:
            xs,us=newxs,newus
            if scale==1.:
                call['audit']=safe_audit(info,mesh,xs,us)
                score=max(call['audit'].get('node_max',float('inf')),call['audit'].get('fine_max',float('inf')),
                          max(call['audit'].get('shooting',{}).values(),default=0.))
                if best is None or score<best[0]:
                    best=(score,j,[x.copy() for x in xs],[u.copy() for u in us])
                else:
                    # Retain the lowest independently measured violation;
                    # a failed hard-constraint call must not destroy it.
                    xs,us=best[2],best[3]
        arrays={}
        for name in dir(history):
            if name.startswith('_'): continue
            try:
                arr=np.asarray(getattr(history,name))
                if arr.ndim==1 and arr.dtype.kind in 'fiu': arrays[name]=arr
            except (ValueError,TypeError): pass
        np.savez_compressed(args.out/f'history_{j:02d}.npz',**arrays)
        record['calls'].append(call)
        write_json(args.out/'result.json',record)
        print(json.dumps(call),flush=True)
        if call.get('audit',{}).get('passed'): break
        del solver,history,built
    if best is not None:
        record['selected_call']=best[1]
        xs,us=best[2],best[3]
    save(args.out/'solution.npz',mesh,xs,us)
    record['audit']=safe_audit(info,mesh,xs,us)
    record['passed']=record['audit']['passed']
    record['process_s']=time.perf_counter()-started
    record['solve_s']=sum(c.get('solve_s',0.) for c in record['calls'])
    write_json(args.out/'result.json',record)
    print(json.dumps({k:record[k] for k in ['robot','passed','solve_s','process_s']}),flush=True)
    return 0 if record['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
