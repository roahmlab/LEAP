"""Terrain restoration with public ALIGATOR models and unmodified FDDP."""
from types import SimpleNamespace
import numpy as np
import aligator as al
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from seed import ROOT
from problem import inputs, geometry
from common.plan import _monitors
from aligator_bench._hybrid_model import Impact
from aligator_bench._generated_sparse import GeneratedResidual
from aligator_bench.solve_fence_lifted import generate
from _public_models import Geometry, SignedGeometry, CollocationInverse, MappedState

def dimensions(r):
    nv,nt,nf=r.nv,r.nu,sum(c.dim for c in r.contacts)
    count=3 if getattr(r,'hold','linear')=='quadratic' else 2
    return nv,nt,nf,3*nv+count*(nt+nf)


def cold(robot,dt,hold='linear',smooth=0.,native_acceleration=False):
    info,mesh,xs,_,fields=inputs(robot,dt)
    info.r.hold=hold;info.r.smooth=smooth;info.r.native_acceleration=native_acceleration
    # The generated residual library is keyed by the DRIVER robot name (a study
    # variant of a robot has its own cone / CoP / keep-out expressions).
    info.r.library_name=robot
    from verify_public import controls_from_samples
    nv,nt,nf,nu=dimensions(info.r)
    us=[]
    for step in mesh:
        if step['kind']!='flow':
            control=np.zeros(nu)
            if step['kind']=='initial':control[:nv]=xs[1][:nv]
            us.append(control);continue
        t,h=step['t'],step['h']
        element=info.seed.element(t)
        samples=[info.seed.sample(t+c*h,element=element) for c in [0.,.5,1.]]
        us.append(controls_from_samples(info.r,[(v['q'],v['v'],v['a'],v['u'],v['lambda']) for v in samples]))
    return info,mesh,xs,us,fields


def symbolic(info):
    r,s=info.r,info.spec
    nv,nt,nf,nu=dimensions(r)
    nx=2*nv
    x,u=ca.SX.sym('x',nx),ca.SX.sym('u',nu)
    model=cpin.Model(r.model);data=model.createData()
    cpin.computeJointJacobians(model,data,x[:nv]);cpin.updateFramePlacements(model,data)
    rows,names=[],[]
    def add(expr,name):
        expr=ca.vec(expr)
        for j in range(expr.numel()): rows.append(expr[j]);names.append(f'{name}_{j}')
    count=(nu-3*nv)//(nt+nf)
    add(x,'x')
    for ep in range(count):add(u[3*nv+ep*nt:3*nv+(ep+1)*nt],f'tau{ep}')
    for ep in range(count):add(u[3*nv+count*nt+ep*nf:3*nv+count*nt+(ep+1)*nf],f'force{ep}')
    for c in r.contacts:
        pose=data.oMf[c.frame_id]
        add(pose.translation,f'pos_{c.name}')
        if c.dim==6: add((pose.rotation-ca.SX.eye(3))/np.sqrt(2.),f'rot_{c.name}')
        J=cpin.getFrameJacobian(model,data,c.frame_id,pin.LOCAL_WORLD_ALIGNED)[:c.dim,:]
        add(J@x[nv:],f'vel_{c.name}')
        for j in range(3):
            expression=ca.SX.zeros(c.dim) if getattr(r,'native_acceleration',False) else J@u[j*nv:(j+1)*nv]+ca.jtimes(J@x[nv:],x[:nv],x[nv:])
            add(expression,f'acc{j}_{c.name}')
    for fid,name in _monitors(r): add(data.oMf[fid].translation[2],f'mon_{name}')
    for endpoint in range(count):
        offset=3*nv+count*nt+endpoint*nf
        for c in r.contacts:
            f=u[offset:offset+c.dim];offset+=c.dim
            add(ca.sqrt(f[0]**2+f[1]**2+s.friceps**2)-s.friceps-s.mu*f[2],f'cone{endpoint}_{c.name}')
            if c.dim==6:
                add(ca.vertcat(f[4]-s.copx*f[2],-f[4]-s.copx*f[2],
                              f[3]-s.copy*f[2],-f[3]-s.copy*f[2],
                              f[5]-s.gammaz*f[2],-f[5]-s.gammaz*f[2]),f'wrench{endpoint}_{c.name}')
    for name in s.keepout_capsules:
        cap=r.capsules[name]
        A=data.oMf[r.frame_id(cap.frame)].act(ca.SX(cap.a_local))
        B=data.oMf[r.frame_id(cap.b_frame)].act(ca.SX(cap.b_local))
        for j,(center,half) in enumerate(s.keepout_boxes):
            faces=[ca.fmin(sign*(A[k]-center[k])-half[k],sign*(B[k]-center[k])-half[k])
                   for k,sign in [(0,1),(0,-1),(1,1),(1,-1),(2,1)]]
            best=faces[0]
            for value in faces[1:]:best=ca.fmax(best,value)
            add(cap.radius+s.obstacle_margin-best,f'keepout_{name}_{j}')
    raw=ca.vertcat(*rows);nr=raw.numel()
    p=ca.SX.sym('p',3*nr)
    residual=p[2*nr:]*(ca.fmax(p[:nr]-raw,0)-ca.fmax(raw-p[nr:2*nr],0))
    obj=SimpleNamespace(x=x,u=u,p=p,cost_y_expr=residual,cost_y_expr_e=residual)
    directory=ROOT/'comparisons/aligator_unified_terrain/generated'/(f'{getattr(r,"library_name",r.name)}_collocation_physical_v2'+('_quadratic' if count==3 else '')+('_native_acceleration' if getattr(r,'native_acceleration',False) else ''))
    library,funcs,digest=generate(obj,directory,[("constraints",raw)],optimized=True)
    return SimpleNamespace(library=library,funcs=funcs,digest=digest,names=names,
                           nr=nr,row={name:j for j,name in enumerate(names)},nu=nu)



def bounds(info,step,sym,fraction=0.,obstacle_scale=1.,geometry_weight=1.,sway_guide=0.,touchdown_weight=1.):
    from problem import ramp
    r,s=info.r,info.spec;nv,nt,nf,nu=dimensions(r)
    lo,hi,w=np.full(sym.nr,-1e20),np.full(sym.nr,1e20),np.zeros(sym.nr)
    def vec(prefix,l,h,weight=1.):
        l,h=np.atleast_1d(l),np.atleast_1d(h)
        for j in range(max(len(l),len(h))):
            i=sym.row[f'{prefix}_{j}'];lo[i],hi[i],w[i]=l[j%len(l)],h[j%len(h)],weight
    def row(name,l,h,weight=1.):
        i=sym.row[name];lo[i],hi[i],w[i]=l,h,weight
    mode,kind=step['mode'],step['kind'];md=s.modes[mode];t=step['t']+fraction*step['h']
    mid,half=.5*(r.q_lower+r.q_upper),.5*(r.q_upper-r.q_lower)
    qlo={int(k):mid[j]-s.qbox*half[j] for j,k in enumerate(r.act_idx)};qhi={int(k):mid[j]+s.qbox*half[j] for j,k in enumerate(r.act_idx)}
    # Named posture boxes (box_posture, terrain_walk.cpp:1007-1024) share the q
    # row with box_q: the residual row carries the intersection of the two boxes.
    for _,k,plo,phi in s.posture:
        qlo[k]=max(qlo.get(k,-1e20),plo if np.isfinite(plo) else -1e20);qhi[k]=min(qhi.get(k,1e20),phi if np.isfinite(phi) else 1e20)
    for k in sorted(qlo):row(f'x_{k}',qlo[k],qhi[k])
    for j,k in enumerate(r.act_idx):
        row(f'x_{nv+k}',-s.vlim*r.velocity_limit[j],s.vlim*r.velocity_limit[j])
    # Base linear-velocity box (box_v_base, terrain_walk.cpp:1039-1064).
    for k,lim in s.basev_limits.items():row(f'x_{nv+k}',-lim,lim)
    row('x_2',0.,1e20)
    if sway_guide>0 and 1 not in s.planarlock_idx:
        target=float(np.interp(t,info.sway['times'],info.sway['base_y']));row('x_1',target,target,sway_guide)
    # Attitude box: attmax on every axis, or the per-axis attbox (terrain_walk.cpp:1125-1150).
    for k,lim in s.att_limits.items():row(f'x_{k}',-lim,lim)
    for k in s.planarlock_idx:row(f'x_{k}',0.,0.)
    if abs(t)<1e-11 or abs(t-info.starts[-1])<1e-11:
        for k in range(nv):row(f'x_{nv+k}',0.,0.)
    count=(nu-3*nv)//(nt+nf)
    for ep in range(count):
        if kind=='flow':vec(f'tau{ep}',-s.effort_box,s.effort_box)   # ulim (x ulimjoint scale) x URDF limits
        offset=0;te=step['t']+ep/(count-1)*step['h']
        for c in r.contacts:
            if c.name in md.active and kind=='flow':
                row(f'force{ep}_{offset+2}',s.lammin*ramp(info,mode,c.name,te),1e20)
                vec(f'cone{ep}_{c.name}',-1e20,0.)
                if c.dim==6:vec(f'wrench{ep}_{c.name}',np.full(6,-1e20),np.zeros(6))
                pinned=(c.name in md.land_feet and abs(te-info.starts[mode])<1e-10) or (c.name in md.lift_feet and abs(te-info.starts[mode+1])<1e-10)
                if pinned:
                    for j in range(c.dim):row(f'force{ep}_{offset+j}',0.,0.)
            else:
                for j in range(c.dim):row(f'force{ep}_{offset+j}',0.,0.)
            offset+=c.dim
    for c in r.contacts:
        if c.name in md.active:
            vec(f'pos_{c.name}',md.targets[c.name],md.targets[c.name])
            if c.dim==6:vec(f'rot_{c.name}',np.zeros(9),np.zeros(9))
            vec(f'vel_{c.name}',np.zeros(c.dim),np.zeros(c.dim))
            if kind=='flow':vec(f'acc{int(round(2*fraction))}_{c.name}',np.zeros(c.dim),np.zeros(c.dim))
        if kind in ('impact','junction') and md.landing and c.name==md.swing_foot:
            # Touchdown rows at the landing mode's last instant (terrain_walk.cpp:820-856),
            # whether the event is a plastic impact or a landcont continuity junction.
            vec(f'pos_{c.name}',md.swing_to,md.swing_to)
            if c.dim==6:vec(f'rot_{c.name}',np.zeros(9),np.zeros(9))
            if s.zeroland:vec(f'vel_{c.name}',np.zeros(c.dim),np.zeros(c.dim),np.sqrt(touchdown_weight))
            else:
                if s.zerolandxy:
                    row(f'vel_{c.name}_0',0.,0.);row(f'vel_{c.name}_1',0.,0.)
                if s.landvz>0:row(f'vel_{c.name}_2',-s.landvz,1e20)
        if kind=='flow' and md.window is not None and c.name==md.swing_foot:
            # Hover rows of the approach / lift window (terrain_walk.cpp:858-871): xy
            # pinned on the placement, a dim-6 sole level, foot at or above its height.
            # At the approach's last instant the touchdown rows above (full pin) apply too.
            hv=md.hover_target
            row(f'pos_{c.name}_0',hv[0],hv[0]);row(f'pos_{c.name}_1',hv[1],hv[1]);row(f'pos_{c.name}_2',hv[2],1e20)
            if c.dim==6:vec(f'rot_{c.name}',np.zeros(9),np.zeros(9))
    if any(abs(t-tt)<1e-10 for tt in info.clear_times[mode]):row(f'pos_{md.swing_foot}_2',md.clearz,1e20)
    for _,name in _monitors(r):vec(f'mon_{name}',s.monitorz,1e20)
    for name in sym.names:
        if name.startswith('keepout_'):row(name,-1e20,(1-obstacle_scale)*.2,0. if getattr(r,'smooth',0)>0 else np.sqrt(geometry_weight))
    return np.r_[lo,hi,w]


def build(info,mesh,sym,obstacle_scale=1.,geometry_weight=1.,inverse_weight=1.,hard=False,foot_guide=0.,posture_guide=0.,clearance_weight=1.,duals=None,sway_guide=0.,acceleration_weight=1.,identity_impacts=False,sparse_cost=False,control_scale=1.,touchdown_weight=1.,sway_kind='base',hard_all=False,reduction=None):
    from verify_public import weights
    r=info.r;nv,nt,nf,nu=dimensions(r);nx=2*nv;space=al.manifolds.VectorSpace(nx)
    if sway_guide>0 and not hasattr(info,'sway'):
        from problem import lateral_reference
        info.sway=lateral_reference(info)
    hard=hard or hard_all
    def input_coordinates(fn):
        if control_scale!=1.:
            from _terrain_controls import ControlScaledFunction
            return ControlScaledFunction(fn,control_scale)
        return fn
    def cost_with_units(cost):
        if control_scale!=1.:
            from _terrain_controls import ControlScaledCost
            return ControlScaledCost(cost,control_scale)
        return cost
    def residual_cost(fn):
        if sparse_cost:
            from _terrain_cost import SparseUnitResidualCost
            return SparseUnitResidualCost(space,fn)
        return al.QuadraticResidualCost(space,fn,np.eye(fn.nr))
    geom,rows=geometry(info);margins=np.array([v for _,v in rows]);stages=[];checks=[];reducers=[]
    terrain=None
    if getattr(r,'smooth',0)>0:
        from _terrain_smooth import SmoothTerrain
        caps=np.array([[r.frame_id(r.capsules[name].frame),r.frame_id(r.capsules[name].b_frame),r.capsules[name].radius,*r.capsules[name].a_local,*r.capsules[name].b_local] for name in info.spec.keepout_capsules])
        boxes=np.array([np.r_[c,h] for c,h in info.spec.keepout_boxes]).reshape(-1,6)
        terrain=SmoothTerrain(nu,r.model,caps,boxes,info.spec.obstacle_margin,r.smooth,.2*(1-obstacle_scale))
    for k,step in enumerate(mesh):
        h,kind=step['h'],step['kind'];cost=al.CostStack(space,nu);native_constraints=[]
        A,B=np.eye(nx),np.zeros((nx,nu))
        if kind=='initial':
            A[:]=0.;B[:nv,:nv]=np.eye(nv)
            # Every dummy control except the freely allocated initial q is zero.
            weight=np.eye(nu);weight[:nv,:nv]=0.
            cost.addCost(al.QuadraticControlCost(space,np.zeros(nu),weight),1.)
            dyn=al.dynamics.LinearDiscreteDynamics(A,B*control_scale,np.zeros(nx))
            stages.append(al.StageModel(cost_with_units(cost),dyn));checks.append({});reducers.append(None);continue
        geo=Geometry(nu,r.model,geom,margins) if len(rows) else None
        al_points=[];inv=None
        signed=SignedGeometry(nu,r.model,geom,margins) if (duals is not None or hard) and len(rows) else None
        for j,fraction in enumerate([0.,.5,1.] if kind=='flow' else [0.]):
            p=bounds(info,step,sym,fraction,obstacle_scale,geometry_weight,sway_guide if sway_kind=='base' else 0.,touchdown_weight)
            for row,name in enumerate(sym.names):
                if name.startswith('mon_'):p[2*sym.nr+row]*=np.sqrt(clearance_weight)
                if name.startswith('acc'):p[2*sym.nr+row]*=np.sqrt(acceleration_weight)
            original=p.copy()
            if duals is not None:
                p[:sym.nr]-=duals[k]['physical'][j];p[sym.nr:2*sym.nr]-=duals[k]['physical'][j]
                if len(rows):geo=Geometry(nu,r.model,geom,margins+duals[k]['geometry'][j])
            fn=GeneratedResidual(nx,nu,sym.nr,str(sym.library),'running',p)
            Aa,Bb=np.eye(nx),np.zeros((nx,nu));qw,vw,_=weights(fraction)
            Aa[:nv,nv:]=h*fraction*np.eye(nv)
            for a in range(3):
                Bb[:nv,a*nv:(a+1)*nv]=h*h*qw[a]*np.eye(nv)
                Bb[nv:,a*nv:(a+1)*nv]=h*vw[a]*np.eye(nv)
            if duals is not None:
                raw=GeneratedResidual(nx,nu,sym.nr,str(sym.library),'constraints',original)
                al_points.append(dict(raw=raw,A=Aa,B=Bb,p=original))
            if hard:
                raw=GeneratedResidual(nx,nu,sym.nr,str(sym.library),'constraints',original)
                selected=[]
                for i,name in enumerate(sym.names):
                    if original[2*sym.nr+i]<=0 and not name.startswith('keepout_'):continue
                    if not hard_all and not name.startswith(('x_','pos_','rot_','vel_','mon_','keepout_')):continue
                    if getattr(r,'native_acceleration',False) and name.startswith('acc'):continue
                    if fraction and name.startswith(('tau','force','cone','wrench')):continue
                    selected.append(i)
                if selected:
                    fn_state=input_coordinates(raw[selected])
                    mapped=MappedState(fn_state,Aa,Bb*control_scale) if fraction else fn_state
                    native_constraints.append((mapped,al.constraints.BoxConstraint(original[:sym.nr][selected],original[sym.nr:2*sym.nr][selected])))
                if signed is not None:
                    mapped=MappedState(signed,Aa,Bb*control_scale) if fraction else signed
                    native_constraints.append((mapped,al.constraints.BoxConstraint(np.full(signed.nr,-1e20),np.zeros(signed.nr))))

            acceleration=[]
            if kind=='flow' and getattr(r,'native_acceleration',False):
                from _terrain_acceleration import ContactAcceleration
                acc=ContactAcceleration(nu,r.model,np.array([c.frame_id for c in r.contacts],np.int32),np.array([c.dim for c in r.contacts],np.int32),np.array([c.name in info.spec.modes[step['mode']].active for c in r.contacts],np.int32),j)
                acceleration=[(acc,acceleration_weight)]
                if hard_all:
                    rows_acc=[];offset=0
                    for contact in r.contacts:
                        if contact.name in info.spec.modes[step['mode']].active:rows_acc.extend(range(offset,offset+contact.dim))
                        offset+=contact.dim
                    if rows_acc:
                        fn_acc=input_coordinates(acc[rows_acc]);mapped=MappedState(fn_acc,Aa,Bb*control_scale) if fraction else fn_acc
                        native_constraints.append((mapped,al.constraints.EqualityConstraintSet()))
            guide_cost=[]
            if sway_guide>0 and sway_kind=='com':
                target=float(np.interp(step['t']+fraction*h,info.sway['times'],info.sway['com_y']))
                com=al.CenterOfMassTranslationResidual(nx,nu,r.model,np.array([0.,target,0.]))[[1]]
                guide_cost=[(com,sway_guide**2)]
            for residual,weight in [(fn,1.)]+acceleration+guide_cost+([(geo,geometry_weight)] if geo else [])+([(terrain,geometry_weight)] if terrain is not None else []):
                mapped=MappedState(residual,Aa,Bb) if fraction else residual
                cost.addCost(residual_cost(mapped),weight)
        if kind=='flow':
            inverse_type=CollocationInverse
            if getattr(r,'hold','linear')=='quadratic':
                from _terrain_models import QuadraticInverse
                inverse_type=QuadraticInverse
            inv=inverse_type(r.model,np.array(r.act_idx,np.int32),np.array([c.frame_id for c in r.contacts],np.int32),np.array([c.dim for c in r.contacts],np.int32),h)
            inv_cost=al.linear_compose(inv,np.eye(inv.nr),duals[k]['inverse']) if duals is not None else inv
            if reduction is None:cost.addCost(residual_cost(inv_cost),inverse_weight)
            if hard_all:native_constraints.append((input_coordinates(inv),al.constraints.EqualityConstraintSet()))
            A[:nv,nv:]=h*np.eye(nv)
            for j,value in enumerate([1,2,0]):B[:nv,j*nv:(j+1)*nv]=value*h*h/6*np.eye(nv)
            for j,value in enumerate([1,4,1]):B[nv:,j*nv:(j+1)*nv]=value*h/6*np.eye(nv)
        else:cost.addCost(al.QuadraticControlCost(space,np.zeros(nu),np.eye(nu)),1.)
        dyn=Impact(r.model,list(info.cms[step['mode']+1]),0,nu) if kind=='impact' and not identity_impacts else al.dynamics.LinearDiscreteDynamics(A,B*control_scale,np.zeros(nx))
        reducer=None
        if reduction is not None and kind=='flow':
            from reduction import inverse_and_columns
            from _terrain_reduced import ReducedCost
            inverse,free,eliminated=inverse_and_columns(info,step)
            reducer=ReducedCost(cost,inverse,free,eliminated,reduction[k])
            cost=reducer;dyn=al.dynamics.LinearDiscreteDynamics(A,B[:,free],np.zeros(nx))
        reducers.append(reducer)
        stage=al.StageModel(cost_with_units(cost),dyn)
        for function,constraint in native_constraints:stage.addConstraint(function,constraint)
        stages.append(stage);checks.append(dict(fn=fn,p=p,points=al_points,inv=inv,signed=signed))
    # Last flow checks terminal foot placements, boxes, and zero velocity as well.
    target=np.r_[np.zeros(nv),-duals[-1]['terminal']] if duals is not None else np.zeros(nx)
    term=al.QuadraticStateCost(space,0,target,np.diag(np.r_[np.zeros(nv),np.ones(nv)]))
    problem=al.TrajOptProblem(np.zeros(nx),stages,term)
    return SimpleNamespace(problem=problem,checks=checks,reducers=reducers,info=info,mesh=mesh,sym=sym,dimensions=(nx,nu),obstacle_scale=obstacle_scale,duals=duals)
