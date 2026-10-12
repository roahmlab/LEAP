#!/usr/bin/env python3
"""Continue an existing cold lineage within the user's enlarged solve budget."""
import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from run import bootstrap,write,digest,HERE,SCRIPTS,DEFAULT_OUT
bootstrap()

def main(args):
    import numpy as np
    import aligator as al
    import _trace_run
    from public_problem import cold,symbolic
    from outer_al import build,update
    from verify_public import audit
    from solve import save
    run=args.run;s=json.loads((run/'summary.json').read_text());plan=json.loads((run/'plan.json').read_text())
    s['budget_s']=args.budget;plan['budget_s']=args.budget
    write(run/'plan.json',plan)
    if s['returned_audit'].get('passed'):
        write(run/'summary.json',s)
        if args.postprocess and not s.get('postprocessed'):
            subprocess.run([sys.executable,str(HERE/'postprocess.py'),str(run),'--workers','12'],check=True)
        return
    prior=json.loads((Path(s['phases'][-1])/'result.json').read_text())
    cpus=prior['cpus'];os.sched_setaffinity(0,cpus);os.environ['OMP_NUM_THREADS']='12'
    robot=s['robot'];dt=prior['settings']['dt']
    info,mesh,_,_,_=cold(robot,dt);sym=symbolic(info)
    if args.continue_duals:
        if not prior['calls'][-1]['audit'].get('finite'):raise RuntimeError('Multiplier continuation needs a finite returned candidate')
        parent=Path(s['phases'][-1])/f"call_{prior['calls'][-1]['index']:02d}.npz"
    elif s.get('postprocessed') and (run/'best_observed_solution.npz').exists():parent=run/'best_observed_solution.npz'
    else:parent=Path(s['phases'][-1])/'solution.npz'
    z=np.load(parent);xs=[v.copy() for v in z['xs']];us=[v.copy() for v in np.split(z['controls'],z['offsets'][1:-1])]
    if not all(np.isfinite(v).all() for v in xs+us):raise RuntimeError('No finite continuation input')
    index_phase=len(s['phases']);phase=run/f'{index_phase:02d}_budget_extension';phase.mkdir()
    shutil.copytree(SCRIPTS,phase/'model_source',ignore=shutil.ignore_patterns('build','__pycache__','*.so'))
    shutil.copy2(Path(__file__),phase/'extension_runner_source.py')
    shutil.copy2(parent,phase/'parent_candidate.npz')
    save(phase/'initial.npz',mesh,xs,us)
    reg_init=.1 if robot=='h1' else 1e-3;reg_min=1e-4 if robot=='h1' else 1e-6
    if args.continue_duals:
        reg_init=prior['calls'][-1]['reg_init'];reg_min=prior['calls'][-1]['reg_min']
    settings=dict(dt=dt,tolerance=1e-10,reg_init=reg_init,reg_min=reg_min,iterations=10000,threads=12,
                  outer_al=True,dual_step=args.dual_step,geometry_weight=args.geometry_weight,clearance_weight=1.,call_seconds=300.)
    record=dict(robot=robot,phase=index_phase,settings=settings,cpus=cpus,calls=[],mesh=mesh,
                seed_sha256=info.seed.sha256,parent_candidate=str(parent),parent_candidate_sha256=digest(parent),
                initialization='own finite candidate from the same cold lineage on the identical mesh',
                reason='User increased total cap to four hours; projected outer AL is the validated public orchestration',
                aligator_version=al.__version__)
    write(phase/'config.json',record)
    s['phases'].append(str(phase));s.update(complete=False,postprocessed=False,termination='running')
    write(run/'summary.json',s)
    used=s['solve_compute_s'];wall=s['solve_wall_s'];phase_used=0.;duals=None;best=None;start=time.perf_counter()
    if args.continue_duals:
        previous_phase=Path(s['phases'][-2]);dual_path=previous_phase/f"duals_after_{prior['calls'][-1]['index']:02d}.json"
        shutil.copy2(dual_path,phase/'parent_duals.json')
        duals=[{key:np.asarray(value,dtype=float) for key,value in row.items()} for row in json.loads(dual_path.read_text())]
        record.update(parent_duals=str(dual_path),parent_duals_sha256=digest(dual_path))
        write(phase/'config.json',record)
    previous=([x.copy() for x in xs],[u.copy() for u in us])
    call_index=0
    while used<args.budget:
        setup=time.perf_counter();built=build(info,mesh,sym,duals,geometry_weight=args.geometry_weight)
        write(phase/f'duals_before_{call_index:02d}.json',built.duals)
        solver=al.SolverFDDP(1e-10,al.QUIET,reg_init,10000)
        solver.force_initial_condition=True;solver.reg_min=reg_min;solver.reg_max=1e14
        solver.setNumThreads(12);solver.setup(built.problem)
        setup_s=time.perf_counter()-setup;cap=min(300.,args.budget-used)
        save(phase/f'input_{call_index:02d}.npz',mesh,xs,us)
        print(f'{robot} outer_AL call={call_index} cap={cap:.3f}s elapsed={used:.3f}s reg_min={reg_min:g}',flush=True)
        stats=dict(_trace_run.run(solver,built.problem,xs,us,cap,-1.,str(phase/f'trace_{call_index:02d}.bin'),True))
        stats.update(dual_step=args.dual_step,geometry_weight=args.geometry_weight,index=call_index,start_s=used,cap_s=cap,setup_s=setup_s,reg_init=reg_init,reg_min=reg_min,
                     native_iterations=int(solver.results.num_iters),native_final_cost=float(solver.results.traj_cost))
        used+=stats['solve_compute_s'];phase_used+=stats['solve_compute_s'];wall+=stats['solve_wall_s']
        xx=[np.array(x).copy() for x in solver.results.xs];uu=[np.array(u).copy() for u in solver.results.us]
        save(phase/f'call_{call_index:02d}.npz',mesh,xx,uu)
        try:stats['audit']=audit(info,mesh,xx,uu)
        except Exception as exc:stats['audit']=dict(passed=False,finite=False,error=repr(exc))
        if stats['audit'].get('finite'):
            xs,us=xx,uu;previous=([x.copy() for x in xs],[u.copy() for u in us])
            score=max(stats['audit']['node_max'],stats['audit']['fine_max'])
            if best is None or score<best[0]:best=(score,call_index,xs,us)
            duals=update(built,xs,us,args.dual_step)
            write(phase/f'duals_after_{call_index:02d}.json',duals)
        else:
            # Preserve the failure and charge its solve time; restart only from
            # this phase's own preceding finite input, with stronger regularization.
            xs,us=previous;duals=None;reg_min=min(reg_min*10,1e6);reg_init=max(reg_init,reg_min)
            stats['recovery']='preceding own finite input, zero outer multipliers, regularization minimum multiplied by ten'
        record['calls'].append(stats);record.update(solve_s=phase_used,selected_call=best[1] if best else -1)
        write(phase/'result.json',record)
        save(phase/'solution.npz',mesh,best[2] if best else xs,best[3] if best else us)
        s.update(solve_compute_s=used,solve_wall_s=wall,returned_audit=stats['audit'],
                 complete=bool(stats['audit'].get('passed') or used>=args.budget),
                 termination='physical_success' if stats['audit'].get('passed') else 'budget_exhausted' if used>=args.budget else 'running',
                 extension_process_s=time.perf_counter()-start)
        write(run/'summary.json',s);print(json.dumps(stats),flush=True)
        if stats['audit'].get('passed'):break
        if args.max_calls and call_index+1>=args.max_calls and stats['audit'].get('finite'):
            s['termination']='phase_call_limit';write(run/'summary.json',s);break
        del solver,built;call_index+=1
    if args.postprocess:
        subprocess.run([sys.executable,str(HERE/'postprocess.py'),str(run),'--workers','12'],check=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    ap.add_argument('--budget',type=float,default=14400.)
    ap.add_argument('--geometry-weight',type=float,default=1.)
    ap.add_argument('--dual-step',type=float,default=.25)
    ap.add_argument('--continue-duals',action='store_true')
    ap.add_argument('--max-calls',type=int,default=0,help='Optional phase boundary; total solve cap is unchanged')
    ap.add_argument('--postprocess',action=argparse.BooleanOptionalAction,default=True)
    main(ap.parse_args())
