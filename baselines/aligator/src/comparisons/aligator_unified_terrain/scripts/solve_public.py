#!/usr/bin/env python3
"""Unmodified ALIGATOR FDDP with full-order public-API terrain models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
from seed import ROOT
from io_utils import write_json,save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot',required=True)
    ap.add_argument('--store-iterations',action='store_true',help='Preallocated native snapshots; write only after the solver returns')
    ap.add_argument('--reduced-dynamics',action='store_true')
    ap.add_argument('--defect-rounds',type=int,default=1)
    ap.add_argument('--defect-reference',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--touchdown-weight',type=float,default=1.,help='Cost weight on original zero touchdown velocity rows')
    ap.add_argument('--carry-native-duals',action='store_true',help='Carry ProxDDP multipliers and penalty between returned native calls')
    ap.add_argument('--native-dual-state',type=Path)
    ap.add_argument('--all-constraints',action='store_true',help='Register every physical family through native ProxDDP constraints')
    ap.add_argument('--state-constraints',action='store_true',help='Native ProxDDP state/contact/geometry constraints in addition to full restoration cost')
    ap.add_argument('--solver',choices=['fddp','proxddp'],default='fddp')
    ap.add_argument('--mu-init',type=float,default=.01)
    ap.add_argument('--control-scale',type=float,default=1.,help='Physical controls = scale * internal solver controls; archives retain physical values')
    ap.add_argument('--native-acceleration',action='store_true',help='Equivalent native Pinocchio classical-acceleration model')
    ap.add_argument('--sparse-cost',action='store_true',help='Equivalent sparse unit-weight public residual cost model')
    ap.add_argument('--identity-rounds',type=int,default=0,help='Temporary identity reset calls, only when zeroland is required')
    ap.add_argument('--inverse-weight',type=float,default=1.)
    ap.add_argument('--acceleration-weight',type=float,default=1.)
    ap.add_argument('--sway-kind',choices=['base','com'],default='base')
    ap.add_argument('--sway-guide',type=float,default=0.,help='Temporary lateral LIPM reference residual weight')
    ap.add_argument('--guide-rounds',type=int,default=0)
    ap.add_argument('--smooth',type=float,default=0.,help='Conservative terrain face smoothing, in metres')
    ap.add_argument('--hold',choices=['linear','quadratic'],default='linear')
    ap.add_argument('--dt',type=float,default=.01)
    ap.add_argument('--threads',type=int,default=12)
    ap.add_argument('--core-offset',type=int,default=48)
    ap.add_argument('--iterations',type=int,default=10000)
    ap.add_argument('--call-seconds',type=float,default=300)
    ap.add_argument('--cost-threshold',type=float,help='Explicit public callback threshold, used by recorded-call replay')
    ap.add_argument('--rounds',type=int,default=4)
    ap.add_argument('--reg-init',type=float,default=1e-3)
    ap.add_argument('--tolerance',type=float,default=1e-10)
    ap.add_argument('--obstacle-scale',type=float,default=1.,help='Fixed terrain-bound continuation level, in (0,1]')
    ap.add_argument('--continuation',action='store_true')
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--reg-min',type=float)
    ap.add_argument('--outer-al',action='store_true')
    ap.add_argument('--dual-state',type=Path)
    ap.add_argument('--dual-step',type=float,default=.5)
    ap.add_argument('--geometry-weight',type=float,default=1.)
    ap.add_argument('--clearance-weight',type=float,default=1.)
    args=ap.parse_args()
    if (args.state_constraints or args.all_constraints) and (args.solver!='proxddp' or args.outer_al):ap.error('Native state constraints require ProxDDP without external multipliers')
    if args.carry_native_duals and args.solver!='proxddp':ap.error('Native multiplier continuation requires ProxDDP')
    if args.native_dual_state and (not args.carry_native_duals or not args.resume):ap.error('native-dual-state requires carry-native-duals and resume')
    if args.native_dual_state and args.native_dual_state.resolve().parent!=args.resume.resolve().parent:ap.error('Native primal and dual resumes must belong to the same run')
    if args.cost_threshold is not None and args.cost_threshold<0:ap.error('cost-threshold must be nonnegative')
    if args.defect_rounds<0:ap.error('defect-rounds must be nonnegative')
    if args.reduced_dynamics and (args.solver!='fddp' or args.hold!='quadratic' or args.control_scale!=1 or args.outer_al or args.guide_rounds or args.identity_rounds or args.continuation or args.obstacle_scale!=1):ap.error('Reduced dynamics requires quadratic FDDP with unit input scale and no other temporary continuation')
    if args.defect_reference and (not args.reduced_dynamics or not args.resume):ap.error('Defect reference requires reduced resume')
    if args.reduced_dynamics and not args.resume and args.defect_rounds<1:ap.error('Cold reduction must preserve the mandatory seed in a defect phase')
    if args.out.exists():ap.error('Output directory must be new')
    if min(args.dt,args.threads,args.iterations,args.call_seconds,args.reg_init,args.control_scale,args.mu_init)<=0:ap.error('Positive solver and mesh settings required')
    if min(args.geometry_weight,args.clearance_weight,args.inverse_weight,args.acceleration_weight,args.touchdown_weight)<=0:ap.error('Penalty weights must be positive')
    if not 0<args.dual_step<=1:ap.error('dual-step must be in (0,1]')
    if args.dual_state and (not args.outer_al or not args.resume):ap.error('dual-state requires outer-al and resume')
    if args.dual_state and args.dual_state.resolve().parent!=args.resume.resolve().parent:ap.error('Dual and primal resumes must belong to the same run')
    if min(args.sway_guide,args.guide_rounds,args.rounds,args.identity_rounds)<0 or args.rounds+args.guide_rounds+args.identity_rounds+(args.defect_rounds if args.reduced_dynamics else 0)<1:ap.error('Invalid guide/final round counts')
    if args.sway_guide and (args.outer_al or args.continuation):ap.error('Sway guidance is a separate temporary full-geometry cost')
    if args.identity_rounds and (args.outer_al or args.continuation or args.guide_rounds):ap.error('Identity reset continuation must be used separately')
    if args.smooth<0:ap.error('Smoothing must be nonnegative')
    if args.outer_al and args.native_acceleration:ap.error('Native acceleration does not implement shifted outer multipliers')
    if args.outer_al and args.smooth:ap.error('Smooth terrain and outer AL are separate orchestration options')
    if not 0<args.obstacle_scale<=1:ap.error('obstacle-scale must be in (0,1]')
    if args.continuation and args.obstacle_scale!=1:ap.error('The continuation ladder ends at full geometry')
    if args.outer_al and (args.continuation or args.obstacle_scale!=1):ap.error('Use outer AL with full geometry')
    if args.store_iterations and (args.solver!='fddp' or args.reduced_dynamics):ap.error('Iteration capture currently supports full-coordinate FDDP')
    args.out.mkdir(parents=True)
    from seed import ROOT,ALL_ROBOTS
    if args.robot not in ALL_ROBOTS:ap.error('Expected one of '+', '.join(ALL_ROBOTS))
    from aligator_bench.solve_fence_uniform import pin_physical_cores
    cpus=pin_physical_cores(args.threads,args.core_offset)
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    import numpy as np
    import aligator as al
    import pinocchio as pin
    import _public_run as _uniform_run
    from public_problem import cold,symbolic,build
    from verify_public import audit
    start=time.perf_counter()
    info,mesh,xs,us,fields=cold(args.robot,args.dt,args.hold,args.smooth,args.native_acceleration)
    if args.identity_rounds and not info.spec.zeroland:ap.error('Identity reset continuation requires original zeroland=true')
    if args.resume:
        from transfer_public import transfer
        xs,us=transfer(info,mesh,args.resume)
    save(args.out/'initial.npz',mesh,xs,us)
    np.savez_compressed(args.out/'imported_seed_fields.npz',**{key:np.array([v[key] for v in fields]) for key in fields[0]})
    record=dict(robot=args.robot,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],cpus=cpus,
        aligator_version=al.__version__,pinocchio_version=pin.__version__,
        seed_sha256=info.seed.sha256,seed_node_import_error=info.seed.check_nodes(),
        initialization='exact supplied cold seed sampling' if not args.resume else 'exact polynomial continuation of own candidate',
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [info.path,info.r.yaml_path,info.r.urdf_path]},
        stages=len(mesh),nx=len(xs[0]),nu=len(us[0]),calls=[])
    record['external_sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'comparisons/aligator_bench').iterdir() if p.suffix in ('.py','.hpp','.cpp')}
    record['public_extension_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'comparisons/aligator_unified_fence/scripts').glob('_public*.so')}
    record['terrain_extension_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('_terrain_*.so')}
    record['common_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'comparisons/common').glob('*.py')}
    record['aligator_library_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(sys.prefix)/'lib').glob('libaligator.so*')}
    if args.native_dual_state:record['parent_native_dual_sha256']=hashlib.sha256(args.native_dual_state.read_bytes()).hexdigest()
    if args.dual_state:record['parent_dual_sha256']=hashlib.sha256(args.dual_state.read_bytes()).hexdigest()
    if args.resume:record['parent_candidate_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
    (args.out/'executed_source').mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.cpp'):shutil.copy2(p,args.out/'executed_source'/p.name)
    print(f'{args.robot}: {record["initialization"]}; {len(mesh)} stages',flush=True)
    record['initial_audit']=audit(info,mesh,xs,us)
    if args.sway_guide:
        from problem import lateral_reference
        info.sway=lateral_reference(info);write_json(args.out/'lateral_reference.json',info.sway)
    sym=symbolic(info);record['symbolic_digest']=sym.digest;record['generated_library_sha256']=hashlib.sha256(sym.library.read_bytes()).hexdigest()
    defect_values=None
    if args.reduced_dynamics:
        import reduction
        defect_values=reduction.defects(info,mesh,xs,us)
        if args.defect_reference:
            archive=np.load(args.defect_reference)
            if not np.array_equal(archive['times'],[s['t'] for s in mesh]) or archive['values'].shape!=defect_values.shape:raise ValueError('Defect reference does not match mesh')
            defect_values=archive['values'].copy()
            record['parent_defect_sha256']=hashlib.sha256(args.defect_reference.read_bytes()).hexdigest()
        np.savez_compressed(args.out/'dynamics_defects.npz',values=defect_values,times=[s['t'] for s in mesh],seed_sha256=info.seed.sha256)
    write_json(args.out/'config.json',record)
    scales=([.25,.5,.75] if args.continuation else [])+[args.obstacle_scale]*(args.rounds+args.guide_rounds+args.identity_rounds+(args.defect_rounds if args.reduced_dynamics else 0))
    duals=None
    if args.outer_al:
        from outer_al import initialize,update
        duals=initialize(info,mesh,sym)
        if args.dual_state:
            loaded=json.loads(args.dual_state.read_text())
            if len(loaded)!=len(duals):raise ValueError('Dual stage count differs')
            for expected,actual in zip(duals,loaded):
                for key in expected:
                    value=np.asarray(actual[key],dtype=float)
                    if value.shape!=expected[key].shape or not np.isfinite(value).all():raise ValueError('Dual shape/value differs')
                    expected[key]=value
    native_state=None
    if args.native_dual_state:
        import native_duals
        native_state=native_duals.load(args.native_dual_state,args.resume)
    best=None;built=None;previous_scale=None;previous_guide=None;previous_identity=None;previous_defect=None
    for j,scale in enumerate(scales):
        t=time.perf_counter()
        guide=args.sway_guide if j<args.guide_rounds else 0.
        identity=j<args.identity_rounds
        defect_scale=1. if args.reduced_dynamics and j<args.defect_rounds else 0.
        print(f'call {j}: building stock {args.solver} problem, terrain scale {scale}',flush=True)
        if built is None or scale!=previous_scale or args.outer_al or guide!=previous_guide or identity!=previous_identity or defect_scale!=previous_defect:
            built=build(info,mesh,sym,obstacle_scale=scale,geometry_weight=args.geometry_weight,clearance_weight=args.clearance_weight,duals=duals,sway_guide=guide,inverse_weight=args.inverse_weight,acceleration_weight=args.acceleration_weight,identity_impacts=identity,sparse_cost=args.sparse_cost,control_scale=args.control_scale,hard=args.state_constraints,touchdown_weight=args.touchdown_weight,sway_kind=args.sway_kind,hard_all=args.all_constraints,reduction=None if defect_values is None else defect_scale*defect_values)
            previous_scale=scale;previous_guide=guide;previous_identity=identity;previous_defect=defect_scale
        native_layout=None;native_warm=False;native_mu=args.mu_init
        if args.solver=='proxddp':
            import native_duals
            native_layout=native_duals.signature(info,mesh,sym,built.problem,args)
            if native_state is not None:
                if native_state['layout']!=native_layout:
                    if j==0 and args.native_dual_state:raise ValueError('Native multiplier layout differs')
                    native_state=None
                else:native_warm=True;native_mu=native_state['mu']
        if args.solver=='fddp':
            solver=al.SolverFDDP(args.tolerance,al.VERBOSE,args.reg_init,args.iterations)
            solver.force_initial_condition=True;run=_uniform_run.run_fddp
        else:
            solver=al.SolverProxDDP(args.tolerance,mu_init=native_mu,max_iters=args.iterations,verbose=al.VERBOSE)
            solver.reg_init=args.reg_init;run=_uniform_run.run_proxddp
        if args.reg_min is not None:solver.reg_min=args.reg_min
        solver.setNumThreads(args.threads);solver.setup(built.problem)
        if native_warm:native_duals.restore(solver,native_state,native_layout)
        history=al.HistoryCallback(solver,False,True);solver.registerCallback('history',history)
        call=dict(index=j,dynamics_defect_scale=defect_scale,solver_control_dimensions=sorted(set(st.nu for st in built.problem.stages)),solver_control_variables=sum(st.nu for st in built.problem.stages),native_warm_start=native_warm,native_mu_init=native_mu if args.solver=='proxddp' else None,native_constraint_rows=sum(st.num_dual-st.ndx2 for st in built.problem.stages),native_dual_rows=sum(st.num_dual for st in built.problem.stages),solver='FDDP' if args.solver=='fddp' else 'ProxDDP',identity_impacts=identity,inverse_weight=args.inverse_weight,acceleration_weight=args.acceleration_weight,sway_guide=guide,obstacle_scale=scale,geometry_weight=args.geometry_weight,clearance_weight=args.clearance_weight,build_s=time.perf_counter()-t)
        t=time.perf_counter()
        try:
            cost_threshold=1e-6 if scale<1. else 1e-7/(10**max(0,j-(3 if args.continuation else 0)))
            solver_us=reduction.pack(built,us) if args.reduced_dynamics else [u/args.control_scale for u in us]
            if args.reduced_dynamics:
                projected=reduction.expand(built,xs,solver_us)
                call['incoming_control_projection']=float(max(np.max(abs(a-b)) for a,b in zip(projected,us)))
                if j==0 and args.defect_rounds and not args.defect_reference and call['incoming_control_projection']>1e-7:raise ValueError('Defect phase failed to preserve incoming primal')
            if args.cost_threshold is not None:cost_threshold=args.cost_threshold
            if args.store_iterations:
                sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'along_the_way'))
                import _trace_run
                capture_start=time.perf_counter()
                converged,timeout,cost_stop,raw_s,capture_s,count,width=_trace_run.run_fddp(solver,built.problem,xs,solver_us,args.call_seconds,cost_threshold,str(args.out/f'trace_{j:02d}'),args.iterations)
                call['trace']=dict(raw_s=raw_s,capture_s=capture_s,solver_s=raw_s-capture_s,count=count,width=width,wrapper_s=time.perf_counter()-capture_start,columns=['native_iteration','raw_s','solver_s','capture_cumulative_s','native_cost','native_primal_infeasibility','native_dual_infeasibility','capture_step_s'])
            else:
                converged,timeout,cost_stop=run(solver,built.problem,xs,solver_us,args.call_seconds,cost_threshold)
            call.update(converged=bool(converged),timeout=bool(timeout),cost_stop=bool(cost_stop),cost_threshold=cost_threshold)
        except Exception as exc:call['error']=repr(exc)
        call.update(solve_s=time.perf_counter()-t,iterations=int(solver.results.num_iters),cost=float(solver.results.traj_cost))
        if args.store_iterations and 'trace' in call:call['solve_s']=call['trace']['solver_s']
        xx=[np.array(v).copy() for v in solver.results.xs];uu=[args.control_scale*np.array(v).copy() for v in solver.results.us]
        if args.reduced_dynamics:uu=reduction.expand(built,xx,uu)
        call['finite']=all(np.isfinite(v).all() for v in xx+uu)
        save(args.out/f'call_{j:02d}.npz',mesh,xx,uu)
        if args.solver=='proxddp' and call['finite']:
            saved_native=native_duals.capture(solver,native_layout)
            if saved_native is not None:native_duals.save(args.out/f'native_duals_{j:02d}.npz',saved_native,args.out/f'call_{j:02d}.npz')
            native_state=saved_native if args.carry_native_duals else None
        if call['finite']:
            xs,us=xx,uu
            call['audit']=audit(info,mesh,xs,us)
            if scale==1. and guide==0. and not identity and defect_scale==0.:
                score=max(call['audit']['node_max'],call['audit']['fine_max'])
                if best is None or score<best[0]:best=(score,j,xs,us)
        if args.outer_al and call['finite']:
            duals=update(built,xs,us,args.dual_step)
            write_json(args.out/f'duals_{j:02d}.json',duals)
        arrays={}
        for name in dir(history):
            if name.startswith('_'):continue
            try:
                arr=np.asarray(getattr(history,name))
                if arr.ndim==1 and arr.dtype.kind in 'fiu':arrays[name]=arr
            except (TypeError,ValueError):pass
        np.savez_compressed(args.out/f'history_{j:02d}.npz',**arrays)
        record['calls'].append(call);write_json(args.out/'result.json',record)
        print(json.dumps(call),flush=True)
        if call.get('audit',{}).get('passed') and scale==1. and guide==0. and not identity and defect_scale==0.:break
        del solver,history
    if best:
        _,record['selected_call'],xs,us=best
    save(args.out/'solution.npz',mesh,xs,us)
    record['audit']=audit(info,mesh,xs,us);record['passed']=bool(record['audit']['passed'] and best is not None)
    record['solve_s']=sum(c['solve_s'] for c in record['calls']);record['process_s']=time.perf_counter()-start
    write_json(args.out/'result.json',record)
    print(json.dumps({k:record[k] for k in ['robot','passed','solve_s','process_s']}),flush=True)
    return 0 if record['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
