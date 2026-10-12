#!/usr/bin/env python3
"""Current impact-cone cold-start experiment; never imports historical models."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get('LEAP_REPO',HERE.parents[2])).resolve()
BENCH=REPO/'comparisons/aligator_unified_fence'
SCRIPTS=BENCH/'scripts'
DEFAULT_OUT=Path(os.environ.get('ALIGATOR_FENCE_OUT',HERE.parents[3]/'results/fence'))
ROBOTS=('hopper2d','go2','h1','g1_29dof','talos')
def write(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist())+'\n')
def bootstrap():
    os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
    sys.path[:0]=[str(HERE),str(SCRIPTS),str(BENCH.parent),str(BENCH.parents[1])]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run_robot(args):
    bootstrap()
    from aligator_bench.solve_fence_uniform import pin_physical_cores
    cpus=pin_physical_cores(12,args.core_offset)
    os.environ['OMP_NUM_THREADS']='12'
    import numpy as np
    import aligator as al
    import pinocchio as pin
    import _trace_run
    from public_problem import cold,symbolic,build
    from verify_public import audit
    from solve import save
    start=time.perf_counter();out=args.out
    out.mkdir(parents=True,exist_ok=False);phase=out/'00_current_impact_cone';phase.mkdir()
    settings=dict(dt=args.dt,tolerance=1e-10,reg_init=1e-3,reg_min=1e-10,iterations=args.iterations,
                  threads=12,rounds=args.rounds,call_seconds=args.call_seconds,outer_al=args.outer_al)
    plan=dict(command=[sys.executable,*sys.argv],created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),robot=args.robot,budget_s=args.budget,settings=settings,core_offset=args.core_offset,
              formulation='current unified-impact-cone',initialization='exact LEAP cold archive sampling',
              clock='native wall time minus measured in-solve snapshot copying',store_iterations=True)
    write(out/'plan.json',plan)
    info,mesh,xs,us,fields=cold(args.robot,args.dt)
    assert info.spec.impact_cone
    save(phase/'initial.npz',mesh,xs,us)
    np.savez_compressed(phase/'imported_seed_fields.npz',**{key:np.array([v[key] for v in fields]) for key in fields[0]})
    inputs=[info.path,info.r.yaml_path,info.r.urdf_path,info.seed.path]
    archive=out/'reference_inputs';archive.mkdir()
    for i,p in enumerate(inputs):shutil.copy2(p,archive/f'{i}_{p.name}')
    record=dict(robot=args.robot,phase=0,settings=settings,cpus=cpus,calls=[],mesh=mesh,
        seed_path=str(info.seed.path),seed_sha256=info.seed.sha256,seed_node_import_error=info.seed.check_nodes(),
        inputs={str(p):digest(p) for p in inputs},aligator_version=al.__version__,pinocchio_version=pin.__version__,
        initialization=plan['initialization'],benchmark=plan['formulation'],
        environment=dict(platform=platform.platform(),python=sys.version,loadavg=os.getloadavg(),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            cpu_topology={str(cpu):{key:(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')/key).read_text().strip() for key in ('physical_package_id','core_id')} for cpu in cpus}),
        native_extension_hashes={str(p):digest(p) for p in list(SCRIPTS.glob('*.so'))+list(HERE.glob('*.so'))})
    shutil.copytree(SCRIPTS,phase/'model_source',ignore=shutil.ignore_patterns('build','__pycache__','*.so'))
    record['initial_audit']=audit(info,mesh,xs,us)
    sym=symbolic(info);record['symbolic_digest']=sym.digest
    write(phase/'config.json',record)
    used=0.;wall=0.;duals=None;best=None
    for index in range(args.rounds):
        remaining=args.budget-used
        if remaining<=0:break
        cap=min(args.call_seconds,remaining)
        setup=time.perf_counter()
        if args.outer_al:
            from outer_al import build as build_al,update
            built=build_al(info,mesh,sym,duals)
            write(phase/f'duals_before_{index:02d}.json',built.duals)
        else:built=build(info,mesh,sym)
        solver=al.SolverFDDP(settings['tolerance'],al.QUIET,settings['reg_init'],args.iterations)
        solver.force_initial_condition=True;solver.reg_min=1e-10;solver.reg_max=1e14
        solver.setNumThreads(12);solver.setup(built.problem)
        setup_s=time.perf_counter()-setup
        save(phase/f'input_{index:02d}.npz',mesh,xs,us)
        print(f'{args.robot} call={index} cap={cap:.3f}s compute_elapsed={used:.3f}s',flush=True)
        stats=dict(_trace_run.run(solver,built.problem,xs,us,cap,1e-7/(10**index),str(phase/f'trace_{index:02d}.bin'),True))
        stats.update(index=index,start_s=used,cap_s=cap,setup_s=setup_s,native_iterations=int(solver.results.num_iters),
                     native_final_cost=float(solver.results.traj_cost),cost_threshold=1e-7/(10**index))
        used+=stats['solve_compute_s'];wall+=stats['solve_wall_s']
        xs=[np.array(x).copy() for x in solver.results.xs];us=[np.array(u).copy() for u in solver.results.us]
        save(phase/f'call_{index:02d}.npz',mesh,xs,us)
        stats['audit']=audit(info,mesh,xs,us)
        score=max(stats['audit'].get('node_max',float('inf')),stats['audit'].get('fine_max',float('inf')))
        if best is None or score<best[0]:best=(score,index,xs,us)
        if args.outer_al:
            duals=update(built,xs,us);write(phase/f'duals_after_{index:02d}.json',duals)
        record['calls'].append(stats);record.update(solve_s=used,solve_wall_s=wall,selected_call=best[1])
        write(phase/'result.json',record)
        save(phase/'solution.npz',mesh,best[2],best[3])
        summary=dict(robot=args.robot,budget_s=args.budget,solve_compute_s=used,solve_wall_s=wall,phases=[str(phase)],
            complete=bool(stats['audit'].get('passed') or index==args.rounds-1 or used>=args.budget),
            smoke=bool(args.smoke),postprocessed=False,returned_audit=stats['audit'],process_s=time.perf_counter()-start)
        write(out/'summary.json',summary)
        print(json.dumps(stats),flush=True)
        if stats['error']:raise RuntimeError(stats['error'])
        if not stats['audit'].get('finite',False):raise RuntimeError('Nonfinite returned primal')
        if stats['audit']['passed']:break
        del solver,built
    if args.postprocess:
        subprocess.run([sys.executable,str(HERE/'postprocess.py'),str(out),'--workers','12'],check=True)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot',choices=ROBOTS);ap.add_argument('--all',action='store_true')
    ap.add_argument('--out',type=Path);ap.add_argument('--core-offset',type=int,default=48)
    ap.add_argument('--budget',type=float,default=1200.);ap.add_argument('--call-seconds',type=float,default=300.)
    ap.add_argument('--rounds',type=int,default=4);ap.add_argument('--iterations',type=int,default=10000)
    ap.add_argument('--dt',type=float,default=.005);ap.add_argument('--outer-al',action='store_true')
    ap.add_argument('--postprocess',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if min(args.budget,args.call_seconds,args.rounds,args.iterations,args.dt)<=0:ap.error('Positive settings required')
    if args.all:
        root=args.out or DEFAULT_OUT/'runs';root.mkdir(parents=True,exist_ok=False);children=[]
        for i,robot in enumerate(ROBOTS):
            cmd=[sys.executable,str(HERE/'run.py'),'--robot',robot,'--out',str(root/robot),'--core-offset',str(args.core_offset+12*i),
                 '--budget',str(args.budget),'--call-seconds',str(args.call_seconds),'--rounds',str(args.rounds),'--iterations',str(args.iterations),'--dt',str(args.dt),
                 '--postprocess' if args.postprocess else '--no-postprocess']
            if args.outer_al:cmd+=['--outer-al']
            if args.smoke:cmd+=['--smoke']
            log=(root/f'{robot}.log').open('w');children.append((robot,subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
        codes={}
        for robot,proc,log in children:codes[robot]=proc.wait();log.close()
        write(root/'exit_codes.json',codes)
        if any(codes.values()):raise SystemExit(1)
    elif args.robot:
        args.out=args.out or DEFAULT_OUT/'runs'/args.robot;run_robot(args)
    else:ap.error('Specify --robot or --all')
if __name__=='__main__':main()
