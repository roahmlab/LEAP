#!/usr/bin/env python3
"""Cold-start lifted RK4 fence problem, one installed Aligator solver.run call.

The imported symbolic helpers construct model expressions and bound schedules;
no acados optimizer is instantiated or called. Generated C is a user residual
model through Aligator's public StageFunction API, not a solver modification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'comparisons'))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from aligator_bench.resources import physical_cores


def generate(model, directory, extra=(), optimized=False):
    import casadi as ca
    import fcntl
    # A distinct cache avoids overwriting a shared library mapped by a run
    # using the earlier scalar-temporary generator.
    # Unoptimized C for the largest straight-line Jacobians keeps code generation
    # bounded; solver time is deliberately secondary in this accuracy benchmark.
    opt='-O0' if model.x.numel()>=48 and not optimized else '-O1'
    cache=('chunked_sparse_o1_v1' if optimized else 'chunked_sparse_o0_v1') if model.x.numel()>=48 else 'sparse_v1'
    directory=directory/cache
    directory.mkdir(parents=True, exist_ok=True)
    funcs = []
    for name, y in [('running', model.cost_y_expr), ('terminal', model.cost_y_expr_e), *extra]:
        args = [model.x, model.u, model.p]
        if model.x.numel()>=48:
            # Bound each generated C function's size. MX calls retain the
            # SX blocks and their AD Jacobians as separate C functions.
            inputs=[ca.MX.sym('input_'+str(i),v.numel()) for i,v in enumerate(args)]
            blocks=[]
            for first in range(0,int(y.numel()),8):
                block=ca.Function(name+'_block_'+str(first),args,[y[first:min(first+8,int(y.numel()))]])
                blocks.append(block(*inputs))
            y=ca.vertcat(*blocks)
            args=inputs
        funcs.append(ca.Function(name, args, [ca.densify(y)]))
        funcs.append(ca.Function(name+'_jac', args,
                     [ca.jacobian(y,args[0]),ca.jacobian(y,args[1])]))
    digest = hashlib.sha256(''.join(f.serialize() for f in funcs).encode()).hexdigest()
    receipt=directory/'sha256'
    library=directory/'model.so'
    with (directory/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.exists() or not receipt.exists() or receipt.read_text()!=digest:
            previous=Path.cwd()
            os.chdir(directory)
            try:
                gen=ca.CodeGenerator('model.c',{'avoid_stack':True,'codegen_scalars':False})
                for f in funcs: gen.add(f)
                gen.generate()
                temporary=f'model.{os.getpid()}.so'
                subprocess.run(['cc',opt,'-fPIC','-shared','model.c','-o',temporary,'-lm'],check=True)
                # Never truncate a shared library another process has mapped.
                os.replace(temporary,'model.so')
                receipt.write_text(digest)
            finally: os.chdir(previous)
    return library, funcs, digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dt',type=float,default=.005)
    ap.add_argument('--threads',type=int,default=12,choices=range(1,17))
    ap.add_argument('--iterations',type=int,default=10000)
    ap.add_argument('--tol',type=float,default=1e-10)
    ap.add_argument('--smooth',type=float,default=.005)
    ap.add_argument('--seed-lift',type=float,default=.2)
    ap.add_argument('--seed',choices=['boundary','benchmark','ballistic','footpath','centroidal'],default='benchmark')
    ap.add_argument('--foot-height',type=float,default=.3)
    ap.add_argument('--foot-cross',type=float,default=.6)
    ap.add_argument('--tuck',type=float,default=1.)
    ap.add_argument('--geometry-weight',type=float,default=1.)
    ap.add_argument('--control-regularization',type=float,default=1e-6,help='Fixed control residual coefficient; zero requests pure feasibility')
    ap.add_argument('--knee-tuck',type=float,default=0.)
    ap.add_argument('--hip-swing',type=float,default=0.)
    ap.add_argument('--asymmetry',type=float,default=0.,help='Fixed opposite left/right flight joint seed perturbation')
    ap.add_argument('--body-pitch',type=float,default=0.,help='Analytic Biped5 trunk-pitch seed; compensate hip angles to preserve foot paths')
    ap.add_argument('--seed-inverse',action='store_true',help='Initialize torques by inverse dynamics of the chosen analytic seed')
    ap.add_argument('--matrix-orientation',action='store_true',help='Nonsingular R-I residual for zero foot-orientation pins')
    ap.add_argument('--native-inverse',action='store_true',help='Evaluate RK4 inverse dynamics with native Pinocchio derivatives')
    ap.add_argument('--inverse-scale',type=float,default=1.,help='Fixed inverse-dynamics residual scaling; physical verification remains unscaled')
    ap.add_argument('--forward-forces',action='store_true',help='Existing IntegratorRK2 with native torque/contact-force ODE')
    ap.add_argument('--compile-optimized',action='store_true',help='Optimize generated large model C with -O1 (longer compilation)')
    ap.add_argument('--compact-cost',action='store_true',help='Remove identically zero residual rows; preserves the objective')
    ap.add_argument('--reg-init',type=float,default=1e-3)
    ap.add_argument('--reg-min',type=float,default=1e-6)
    ap.add_argument('--mu-init',type=float,default=1e-3)
    ap.add_argument('--hard-inverse',action='store_true')
    ap.add_argument('--hard-all',action='store_true')
    ap.add_argument('--plastic-impact',action='store_true')
    ap.add_argument('--constraints',action='store_true',help='Enforce raw residuals with native BoxConstraint sets')
    ap.add_argument('--normalize-forces',action='store_true',help='Fixed native-constraint force/torque normalization; unscaled final audit')
    ap.add_argument('--solver',choices=['proxddp','fddp'],default='proxddp')
    ap.add_argument('--acceptance',choices=['armijo','nonmonotone','filter'],default='armijo',help='Existing ProxDDP step-acceptance strategy')
    ap.add_argument('--linear-rollout',action='store_true',help='Existing ProxDDP linear trial rollout')
    ap.add_argument('--lq-solver',choices=['serial','parallel','stagedense'],default='serial',help='Existing ProxDDP LQ backend')
    args=ap.parse_args()
    if min(args.dt,args.iterations,args.tol,args.reg_init,args.reg_min,args.mu_init,args.geometry_weight,args.inverse_scale)<=0:
        ap.error('time step, iteration limit, tolerances, regularization, and weights must be positive')
    if args.smooth<0:
        ap.error('smoothing must be nonnegative')
    if args.control_regularization<0:
        ap.error('control regularization must be nonnegative')
    if args.solver=='fddp' and (args.constraints or args.hard_all or args.hard_inverse):
        ap.error('FDDP handles the penalty formulation, not path constraint sets')
    if args.linear_rollout and args.solver!='proxddp':
        ap.error('linear rollout is a ProxDDP setting')
    if args.acceptance!='armijo' and args.solver!='proxddp':
        ap.error('step-acceptance selection is a ProxDDP setting')
    if args.lq_solver!='serial' and args.solver!='proxddp':
        ap.error('LQ backend selection is a ProxDDP setting')
    if args.lq_solver=='parallel' and args.threads<2:
        ap.error('the parallel LQ backend requires at least two threads')
    if args.matrix_orientation and args.hard_all:
        ap.error('matrix orientation supports penalties, native box constraints, or hard inverse dynamics')
    if args.native_inverse and (args.hard_all or args.hard_inverse or args.plastic_impact):
        ap.error('native inverse supports impulse-free penalties or native box constraints')
    if args.inverse_scale!=1. and not args.native_inverse:
        ap.error('inverse scaling requires native inverse dynamics')
    if args.normalize_forces and not (args.constraints and args.native_inverse):
        ap.error('force normalization requires native inverse and native constraints')
    if args.body_pitch and args.robot!='biped5':
        ap.error('body-pitch seed is implemented for biped5')
    if args.forward_forces and (args.native_inverse or args.constraints or args.hard_all or args.hard_inverse or args.plastic_impact):
        ap.error('forward forces supports the impulse-free penalty formulation')
    if args.compact_cost and (args.hard_all or args.hard_inverse):
        ap.error('compact cost supports penalties or native box constraints')
    if args.seed == 'boundary' and (args.seed_inverse or args.knee_tuck or args.hip_swing or args.asymmetry or args.body_pitch):
        ap.error('boundary interpolation forbids informative seed modifiers')
    if args.seed == 'boundary' and any(args.out.with_suffix(suffix).exists() for suffix in ['.json', '.config.json', '.initial.npz']):
        ap.error('boundary run refuses to overwrite an existing attempt')
    cpus=physical_cores(args.threads)
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    import aligator as al
    import casadi as ca
    import numpy as np
    from common.spec_fence import load_fence_spec
    from common.plan import fence_plan
    from acados_bench import builders
    from acados_bench.fence_lifted import build_lifted
    from aligator_bench.verify_fence_accuracy import verify
    from aligator_bench._generated_sparse import GeneratedResidual
    from aligator_bench import native

    started=time.perf_counter()
    args.out=args.out.resolve()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(rk4=True,cold_start=True,single_call=True)
    record=dict(settings=settings,cpus=cpus,threads=args.threads,
                solver='aligator.SolverProxDDP' if args.solver=='proxddp' else 'aligator.SolverFDDP',aligator_version=al.__version__,
                solve_calls=1,initialization='analytic benchmark seed plus fixed flight lift; zero duals')
    record['command']=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]
    args.out.with_suffix('.config.json').write_text(json.dumps(record,indent=2)+'\n')
    spec=load_fence_spec(ROOT/f'showcase_v2/fence/problems/{args.robot}_jump_fence.yaml')
    pl=fence_plan(spec,dt_target=args.dt,impulse_free=True)
    if args.plastic_impact:
        from acados_bench.fence_impact import add_impact_stage
        add_impact_stage(pl)
    directory=ROOT/'comparisons/.aligator_generated'/f'{args.robot}_{args.smooth}{"_impact" if args.plastic_impact else ""}{"_g"+str(args.geometry_weight) if args.geometry_weight != 1. else ""}'
    if args.constraints:
        directory=directory.with_name(directory.name+'_constraints')
    if args.matrix_orientation:
        directory=directory.with_name(directory.name+'_matrixrot')
    if args.native_inverse:
        directory=directory.with_name(directory.name+'_nativeinv')
    if args.forward_forces:
        directory=directory.with_name(directory.name+'_forwardforces')
    if args.compact_cost:
        directory=directory.with_name(directory.name+'_compact_r'+str(args.control_regularization))
    cfg=SimpleNamespace(smooth=args.smooth,rk4=True,plastic_impact=args.plastic_impact,dyn_scale=1.,
                        hard_endpoints=False,hard_joints=False,exact_hessian=False,
                        condensing=None,lm=1e-3,native_max_iter=args.iterations,
                        huber=0.,planar_reduction=False)
    built=builders.build(pl,directory)
    if args.plastic_impact:
        b=built.stage_bounds[pl.impact_stage]
        b['lbu'][:pl.robot.nu]=0.
        b['ubu'][:pl.robot.nu]=0.
    ocp=build_lifted(built,directory,cfg)
    model=ocp.model
    regularizer_size=int(model.u.numel())
    if args.matrix_orientation:
        from aligator_bench.orientation import matrix_orientation
        matrix_orientation(model,built)
    if args.native_inverse:
        from aligator_bench._inverse_rk4 import InverseRK4
        model.cost_y_expr[built.bound_dim:built.bound_dim+built.inverse_dim]=0
        def inverse_fn(k):
            residual=InverseRK4(2*pl.robot.nv,int(model.u.numel()),pl.robot.model,
                             np.asarray(pl.robot.act_idx,dtype=np.int32),
                             np.array([c.frame_id for c in pl.robot.contacts],dtype=np.int32),
                             np.array([c.dim for c in pl.robot.contacts],dtype=np.int32),pl.dts[k])
            return residual if args.inverse_scale==1. else al.linear_compose(
                residual,np.eye(residual.nr)*args.inverse_scale,np.zeros(residual.nr))
    if args.forward_forces:
        from aligator_bench._force_ode import ForceODE
        model.cost_y_expr[built.bound_dim:built.bound_dim+built.inverse_dim]=0
        full_u=model.u
        reduced_u=ca.SX.sym('torque_and_contact_force',built.sym.nuu)
        model.cost_y_expr=ca.substitute(model.cost_y_expr,full_u,
            ca.vertcat(ca.SX.zeros(4*pl.robot.nv),reduced_u))
        model.u=reduced_u
        ode=ForceODE(pl.robot.model,np.asarray(pl.robot.act_idx,dtype=np.int32),
                     np.array([c.frame_id for c in pl.robot.contacts],dtype=np.int32),
                     np.array([c.dim for c in pl.robot.contacts],dtype=np.int32))
        record['integration']='aligator.IntegratorRK2; independently checked with RK4'
    if args.geometry_weight != 1.:
        for name,i in built.sym.row.items():
            if name.startswith(('keepout_', 'selfcol_', 'mon_')):
                model.cost_y_expr[i] *= np.sqrt(args.geometry_weight)
    if args.compact_cost:
        first=built.bound_dim+built.inverse_dim
        model.cost_y_expr[first:first+regularizer_size]*=args.control_regularization/1e-6
        for attr in ['cost_y_expr','cost_y_expr_e']:
            expression=getattr(model,attr)
            keep=[i for i in range(expression.numel()) if not expression[i].is_zero()]
            setattr(model,attr,expression[keep])
    print(f'{args.robot}: generating residual C and exact first derivatives',flush=True)
    extra=[]
    if args.constraints:
        np_=int(built.sym.p.numel())
        zraw=built.bound_check(model.x,model.u[4*pl.robot.nv:],model.p[:np_])
        invargs=[model.x,model.u,model.p[np_]]
        if args.plastic_impact:
            invargs.append(model.p[-1])
        termraw=ca.vertcat(built.terminal_check(model.x,model.p[:np_]))
        if args.matrix_orientation:
            zraw[built.matrix_rotation_rows]=0
            termraw[built.matrix_rotation_terminal_rows]=0
        raw=zraw if args.native_inverse else ca.vertcat(zraw,built.inverse_check(*invargs))
        extra=[('constraints',raw),('terminal_constraints',termraw)]
        if args.matrix_orientation and built.matrix_rotation_expr.numel():
            extra.extend([('orientation',built.matrix_rotation_expr),
                          ('terminal_orientation',built.matrix_rotation_terminal_expr)])
    library, funcs, digest=generate(model,directory,extra,optimized=args.compile_optimized)
    nx,nu=int(model.x.numel()),int(model.u.numel())
    constraint_scales=np.ones(built.bound_dim)
    if args.normalize_forces:
        import pinocchio as pin
        weight=float(pin.computeTotalMass(pl.robot.model))*9.81
        for name,i in built.sym.row.items():
            if name.startswith(('cone','copA','copB','yaw')):
                constraint_scales[i]=1/weight
        control_offset=built.bound_dim-built.sym.nuu
        effort=np.maximum(1.,np.abs(built.stage_bounds[0]['ubu'][:pl.robot.nu]))
        constraint_scales[control_offset:control_offset+pl.robot.nu]=1/effort
        constraint_scales[control_offset+pl.robot.nu:]=1/weight
    space=al.manifolds.VectorSpace(nx)
    stages=[]
    z=np.zeros(nu)
    if not args.forward_forces:
        linear=ca.Function('linear',[model.p],[ca.jacobian(model.disc_dyn_expr,model.x),
                                               ca.jacobian(model.disc_dyn_expr,model.u)])
    for k in range(pl.nsteps):
        p=built.lifted_params[k]
        fn=GeneratedResidual(nx,nu,int(model.cost_y_expr.numel()),str(library),'running',p)
        cost=al.CostStack(space,nu)
        weights=np.ones(fn.nr)
        first_control=built.bound_dim+built.inverse_dim
        if not args.compact_cost:
            weights[first_control:first_control+regularizer_size]=(args.control_regularization/1e-6)**2
        cost.addCost(al.QuadraticResidualCost(space,fn,np.diag(weights)),
                     1e-6 if args.constraints else 1.)
        if args.native_inverse:
            inv=inverse_fn(k)
            cost.addCost(al.QuadraticResidualCost(space,inv,np.eye(inv.nr)),1.)
        if args.forward_forces:
            dyn=al.dynamics.IntegratorRK2(ode,pl.dts[k])
        else:
            A,B=linear(p)
            dyn=al.dynamics.LinearDiscreteDynamics(np.asarray(A),np.asarray(B),np.zeros(nx))
        st=al.StageModel(cost,dyn)
        if args.constraints:
            rawfn=GeneratedResidual(nx,nu,int(raw.numel()),str(library),'constraints',p)
            off=np_+1
            ni=0 if args.native_inverse else built.inverse_dim
            lo=np.concatenate([p[off:off+built.bound_dim],np.zeros(ni)])
            hi=np.concatenate([p[off+built.bound_dim:off+2*built.bound_dim],np.zeros(ni)])
            if args.normalize_forces:
                rawfn=al.linear_compose(rawfn,np.diag(constraint_scales),np.zeros(rawfn.nr))
                lo*=constraint_scales;hi*=constraint_scales
            st.addConstraint(rawfn,al.constraints.BoxConstraint(lo,hi))
            if args.native_inverse:
                st.addConstraint(inv,al.constraints.EqualityConstraintSet())
            if args.matrix_orientation and built.matrix_rotation_expr.numel():
                rot=GeneratedResidual(nx,nu,int(built.matrix_rotation_expr.numel()),str(library),'orientation',p)
                st.addConstraint(rot,al.constraints.EqualityConstraintSet())
        elif args.hard_all:
            st.addConstraint(fn[list(range(built.bound_dim+built.inverse_dim))],
                             al.constraints.EqualityConstraintSet())
        elif args.hard_inverse:
            begin=built.bound_dim
            rows=list(range(begin,begin+built.inverse_dim))
            st.addConstraint(fn[rows],al.constraints.EqualityConstraintSet())
        stages.append(st)
    fn=GeneratedResidual(nx,nu,int(model.cost_y_expr_e.numel()),str(library),'terminal',built.lifted_params[-1])
    terminal=al.QuadraticResidualCost(space,fn,np.eye(fn.nr))
    problem=al.TrajOptProblem(pl.x0,stages,terminal)
    if args.constraints:
        p=built.lifted_params[-1]
        rawfn=GeneratedResidual(nx,nu,int(termraw.numel()),str(library),'terminal_constraints',p)
        off=np_+1
        lo=p[off:off+built.terminal_bound_dim]
        hi=p[off+built.bound_dim:off+built.bound_dim+built.terminal_bound_dim]
        problem.addTerminalConstraint(rawfn,al.constraints.BoxConstraint(lo,hi))
        if args.matrix_orientation and built.matrix_rotation_terminal_expr.numel():
            rot=GeneratedResidual(nx,nu,int(built.matrix_rotation_terminal_expr.numel()),str(library),'terminal_orientation',p)
            problem.addTerminalConstraint(rot,al.constraints.EqualityConstraintSet())
    if args.hard_all:
        problem.addTerminalConstraint(fn,al.constraints.EqualityConstraintSet())
    if args.seed == 'boundary':
        from aligator_bench.boundary_seed import boundary_seed
        xs, controls = boundary_seed(pl.ts, pl.x0, pl.x_goal, nu)
        us = list(controls)
        record['initialization'] = 'linear interpolation of prescribed full boundary states; all controls and initial duals zero'
        np.savez_compressed(args.out.with_suffix('.initial.npz'), xs=xs, us=controls,
                            ts=pl.ts, x0=pl.x0, x_goal=pl.x_goal)
    else:
        # Analytical cold start, independent of every other optimization run.
        if args.seed == 'centroidal':
            from aligator_bench.centroidal_seed import centroidal_seed
            xs=centroidal_seed(pl,height=args.foot_height,cross=args.foot_cross)
            record['initialization']='analytic foot-path joints, ballistic COM, constant flight angular momentum by quadrature; zero duals'
        elif args.seed == 'footpath':
            from aligator_bench.footpath_seed import footpath_seed
            xs=footpath_seed(pl,height=args.foot_height,cross=args.foot_cross)
            record['initialization']='closed-form foot-path IK and Hermite/ballistic seed; zero duals'
        elif args.seed == 'ballistic':
            from acados_bench.fence_seed import ballistic_seed
            xs=ballistic_seed(pl,tuck=args.tuck)
            record['initialization']='analytic ballistic/Hermite planar seed; zero duals'
        else:
            xs=np.array(pl.xs_init,copy=True)
            phase=np.clip((pl.ts-spec.t_stance)/spec.t_air,0,1)
            xs[:,2]+=args.seed_lift*np.sin(np.pi*phase)**2
            xs[:,pl.robot.nq+2]+=args.seed_lift*np.pi/spec.t_air*np.sin(2*np.pi*phase)
        phase=np.clip((pl.ts-spec.t_stance)/spec.t_air,0,1)
        if args.body_pitch:
            t=np.asarray(pl.ts);duration=float(t[-1])
            pitch=args.body_pitch*np.sin(np.pi*t/duration)**2
            dpitch=args.body_pitch*np.pi/duration*np.sin(2*np.pi*t/duration)
            xs[:,4]+=pitch;xs[:,pl.robot.nq+4]+=dpitch
            for name in ['left_hip','right_hip']:
                index=pl.robot.model.joints[pl.robot.model.getJointId(name)].idx_q
                xs[:,index]-=pitch;xs[:,pl.robot.nq+index]-=dpitch
        bump=16*phase**2*(1-phase)**2
        dbump=32*phase*(1-phase)*(1-2*phase)/spec.t_air
        for j,name in enumerate(pl.robot.model.names):
            amplitude=args.knee_tuck if 'knee' in name else args.hip_swing if 'hip' in name else 0.
            side=1 if 'left' in name else -1 if 'right' in name else 0
            if 'knee' in name:
                amplitude+=side*args.asymmetry
            elif 'hip' in name:
                amplitude-=.5*side*args.asymmetry
            if amplitude:
                index=pl.robot.model.joints[j].idx_q
                xs[:,index]+=amplitude*bump
                xs[:,pl.robot.nq+index]+=amplitude*dbump
        acc=np.tile(np.diff(xs[:,pl.robot.nq:],axis=0)/np.maximum(np.asarray(pl.dts)[:,None],1e-12),(1,4))
        if args.plastic_impact:
            acc[pl.impact_stage]=0.
        us=[]
        for k in range(pl.nsteps):
            b=built.stage_bounds[k]
            control=built.us_init_full[k].copy()
            if args.seed_inverse:
                from common.spec_fence import seed_force_torque
                r=pl.robot
                lam,tau=seed_force_torque(spec,xs[k,:r.nq],xs[k,r.nq:],acc[k,:r.nv],
                                         bool(pl.knots[k].active) and not pl.knots[k].force_pins)
                control=np.concatenate([tau,*[lam.get(c.name,np.zeros(c.dim)) for c in r.contacts]])
                if k==getattr(pl,'impact_stage',-1):
                    control[:]=0.
            control=np.clip(control,b['lbu'],b['ubu'])
            us.append(control if args.forward_forces else np.concatenate([acc[k],control]))
    record['seed_sha256']=hashlib.sha256(xs.tobytes()+np.asarray(us).tobytes()).hexdigest()
    # Audit generated value/Jacobian adapter before entering the solver.
    rng=np.random.default_rng(21)
    worst=0.
    if args.forward_forces:
        data=ode.createData()
        for k in [0,pl.nsteps//2]:
            x=xs[k]+rng.normal(size=nx)*1e-4
            u=us[k]+rng.normal(size=nu)*1e-4
            ode.forward(x,u,data);ode.dForward(x,u,data)
            direction=rng.normal(size=nx+nu);direction/=np.linalg.norm(direction)
            analytic=data.Jx@direction[:nx]+data.Ju@direction[nx:]
            h=1e-6
            ode.forward(x+h*direction[:nx],u+h*direction[nx:],data);plus=data.xdot.copy()
            ode.forward(x-h*direction[:nx],u-h*direction[nx:],data);minus=data.xdot.copy()
            error=float(np.max(np.abs((plus-minus)/(2*h)-analytic))/(1+np.max(np.abs(analytic))))
            worst=max(worst,error)
            assert error<1e-5,error
    if args.native_inverse:
        dtvar=ca.SX.sym('audit_dt')
        invexpr=built.inverse_check(model.x,model.u,dtvar)
        invjac=ca.Function('audit_inverse_jac',[model.x,model.u,dtvar],
                           [ca.jacobian(invexpr,model.x),ca.jacobian(invexpr,model.u)])
        for k in [0,pl.nsteps//2]:
            fn=inverse_fn(k);data=fn.createData()
            x=xs[k]+rng.normal(size=nx)*1e-3
            u=us[k]+rng.normal(size=nu)*.1
            fn.evaluate(x,u,data);fn.computeJacobians(x,u,data)
            expected=np.asarray(built.inverse_check(x,u,pl.dts[k])).ravel()*args.inverse_scale
            jx,ju=invjac(x,u,pl.dts[k])
            np.testing.assert_allclose(data.value,expected,rtol=1e-10,atol=1e-8)
            np.testing.assert_allclose(data.Jx,np.asarray(jx)*args.inverse_scale,rtol=1e-9,atol=1e-8)
            np.testing.assert_allclose(data.Ju,np.asarray(ju)*args.inverse_scale,rtol=1e-9,atol=1e-8)
    # Exact rest poses matter: a perturbed-pose derivative test can miss a
    # singular expression at an exactly identity foot rotation.
    for k in [0,pl.nsteps//2,pl.nsteps]:
        terminal=k==pl.nsteps
        value,jac=funcs[2:4] if terminal else funcs[:2]
        x=xs[k];u=us[min(k,pl.nsteps-1)];p=built.lifted_params[k]
        assert np.isfinite(np.asarray(value(x,u,p))).all()
        assert all(np.isfinite(np.asarray(j)).all() for j in jac(x,u,p))
    for k in [0,pl.nsteps//2]:
        p=built.lifted_params[k]
        fun=GeneratedResidual(nx,nu,int(model.cost_y_expr.numel()),str(library),'running',p)
        d=fun.createData()
        x=xs[k]+rng.normal(size=nx)*1e-4
        u=us[k]+rng.normal(size=nu)*1e-4
        fun.evaluate(x,u,d);fun.computeJacobians(x,u,d)
        expected=np.asarray(funcs[0](x,u,p)).ravel()
        jx,ju=funcs[1](x,u,p)
        assert np.allclose(d.value,expected,rtol=1e-10,atol=1e-9)
        assert np.allclose(d.Jx,np.asarray(jx),rtol=1e-10,atol=1e-9)
        assert np.allclose(d.Ju,np.asarray(ju),rtol=1e-10,atol=1e-9)
        direction=rng.normal(size=nx+nu);direction/=np.linalg.norm(direction)
        h=1e-6
        fd=(np.asarray(funcs[0](x+h*direction[:nx],u+h*direction[nx:],p)).ravel()-
            np.asarray(funcs[0](x-h*direction[:nx],u-h*direction[nx:],p)).ravel())/(2*h)
        ana=d.Jx@direction[:nx]+d.Ju@direction[nx:]
        err=float(np.max(np.abs(fd-ana))/(1+np.max(np.abs(ana))))
        worst=max(worst,err)
        assert err<1e-5,err
    if args.solver=='fddp':
        from aligator_bench import _fddp_run
        solver=al.SolverFDDP(args.tol,verbose=al.VERBOSE,reg_init=args.reg_init,max_iters=args.iterations)
        runner=_fddp_run.run
        solver.force_initial_condition=True
    else:
        strategy={'armijo':al.SA_LINESEARCH_ARMIJO,'nonmonotone':al.SA_LINESEARCH_NONMONOTONE,
                  'filter':al.SA_FILTER}[args.acceptance]
        solver=al.SolverProxDDP(args.tol,mu_init=args.mu_init,max_iters=args.iterations,
                               verbose=al.VERBOSE,sa_strategy=strategy)
        solver.max_al_iters=1000
        solver.bcl_params.mu_lower_bound=1e-14
        if args.linear_rollout:
            solver.rollout_type=al.ROLLOUT_LINEAR
        solver.linear_solver_choice=getattr(al,'LQ_SOLVER_'+args.lq_solver.upper())
        runner=native.run
    solver.reg_init=args.reg_init
    solver.reg_min=args.reg_min
    solver.reg_max=1e14
    solver.setNumThreads(args.threads)
    solver.setup(problem)
    record.update(build_s=time.perf_counter()-started,generated_sha256=digest,
                  derivative_relative_error=worst,n_stages=pl.nsteps)
    print(f'{args.robot}: entering sole {record["solver"]}.run call',flush=True)
    start=time.perf_counter()
    try:
        conv=runner(solver,problem,list(xs),us)
    except Exception as exc:
        record.update(error=str(exc),solve_s=time.perf_counter()-start,passed=False)
        args.out.write_text(json.dumps(record,indent=2)+'\n')
        raise
    record.update(solve_s=time.perf_counter()-start,converged=bool(conv),
                  iterations=int(solver.results.num_iters),
                  penalty_cost=float(solver.results.traj_cost),
                  primal_infeasibility=float(solver.results.primal_infeas),
                  dual_infeasibility=float(solver.results.dual_infeas))
    xs=np.array(list(solver.results.xs));controls=np.array(list(solver.results.us))
    if args.forward_forces:
        us=controls;acc=[]
        data=ode.createData()
        def rhs(x,u):
            ode.forward(x,u,data)
            return np.asarray(data.xdot).copy()
        for k,h in enumerate(pl.dts):
            x=xs[k];u=us[k]
            k1=rhs(x,u);k2=rhs(x+h*k1/2,u);k3=rhs(x+h*k2/2,u);k4=rhs(x+h*k3,u)
            acc.append(np.concatenate([v[pl.robot.nv:] for v in [k1,k2,k3,k4]]))
        acc=np.asarray(acc)
    else:
        acc=controls[:,:4*pl.robot.nv];us=controls[:,4*pl.robot.nv:]
    np.savez_compressed(args.out.with_suffix('.npz'),xs=xs,us_full=us,acc=acc,ts=pl.ts)
    record['passed']=False
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    try:
        if not np.isfinite(xs).all() or not np.isfinite(controls).all():
            raise ValueError('Aligator returned a nonfinite trajectory')
        judged=verify(args.out)
    except Exception as exc:
        record.update(error=f'Independent verification failed: {type(exc).__name__}: {exc}',
                      process_wall_s=time.perf_counter()-started)
        args.out.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)
        return 2
    record.update(independent=judged,passed=judged['passed'],violations_fine=judged['violations_fine'],
                  process_wall_s=time.perf_counter()-started)
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    return 0 if judged['passed'] else 2


if __name__=='__main__':
    sys.exit(main())
