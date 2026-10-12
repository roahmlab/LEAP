"""Multiplier orchestration around stock ALIGATOR FDDP.

Scaled multipliers are projected for inequalities. Each inner problem is a
public CostStack; no solver method, line search, or Riccati sweep is replaced.
"""
from types import SimpleNamespace
import numpy as np
import aligator as al
from public_problem import dimensions,bounds,geometry,impulse_cone
from aligator_bench._generated_sparse import GeneratedResidual
from aligator_bench._hybrid_model import Impact
from _public_models import Geometry,SignedGeometry,CollocationInverse,MappedState
from verify_public import weights


def build(info,mesh,sym,duals=None,geometry_weight=1.,clearance_weight=1.):
    r,s=info.r,info.spec
    nv,nt,nf,nu=dimensions(r);nx=2*nv
    space=al.manifolds.VectorSpace(nx)
    geom,rows=geometry(info,1.)
    margins=np.array([margin for _,margin in rows]);ng=len(rows)
    if duals is None:
        duals=[dict(physical=np.zeros((3 if st['kind']=='flow' else 1,sym.nr)),
                    geometry=np.zeros((3 if st['kind']=='flow' else 1,ng)),
                    inverse=np.zeros(3*nv),
                    impulse=np.zeros(2*len(r.contacts)) if st['kind']=='impact' and s.impact_cone else np.zeros(0)) for st in mesh]
        duals.append(dict(terminal=np.zeros(nx)))
    stages,checks=[],[]
    for k,st in enumerate(mesh):
        h,kind=st['h'],st['kind'];p=bounds(info,st,sym)
        for row,name in enumerate(sym.names):
            if name.startswith('mon_'):p[2*sym.nr+row]*=np.sqrt(clearance_weight)
        lo,hi,w=np.split(p,3);cost=al.CostStack(space,nu)
        raw=GeneratedResidual(nx,nu,sym.nr,str(sym.library),'constraints',p)
        signed=SignedGeometry(nu,r.model,geom,margins)
        points=[]
        for j,fraction in enumerate([0.,.5,1.] if kind=='flow' else [0.]):
            # Equality target shifts by -lambda/rho. A single signed lambda
            # also represents the active side of a two-sided interval.
            shift=duals[k]['physical'][j]
            fn=GeneratedResidual(nx,nu,sym.nr,str(sym.library),'running',np.r_[lo-shift,hi-shift,w])
            geo=Geometry(nu,r.model,geom,margins+duals[k]['geometry'][j])
            Aa,Bb=np.eye(nx),np.zeros((nx,nu))
            if fraction:
                qw,vw,_=weights(fraction);Aa[:nv,nv:]=h*fraction*np.eye(nv)
                for a in range(3):
                    Bb[:nv,a*nv:(a+1)*nv]=h*h*qw[a]*np.eye(nv)
                    Bb[nv:,a*nv:(a+1)*nv]=h*vw[a]*np.eye(nv)
                fn=MappedState(fn,Aa,Bb);geo=MappedState(geo,Aa,Bb)
            cost.addCost(al.QuadraticResidualCost(space,fn,np.eye(sym.nr)),1.)
            cost.addCost(al.QuadraticResidualCost(space,geo,np.eye(ng)),geometry_weight)
            points.append((Aa,Bb))
        inv=None
        if kind=='flow':
            inv=CollocationInverse(r.model,np.array(r.act_idx,np.int32),np.array([c.frame_id for c in r.contacts],np.int32),np.array([c.dim for c in r.contacts],np.int32),h)
            shifted=al.linear_compose(inv,np.eye(inv.nr),duals[k]['inverse'])
            cost.addCost(al.QuadraticResidualCost(space,shifted,np.eye(inv.nr)),1.)
        else:
            cost.addCost(al.QuadraticControlCost(space,np.zeros(nu),np.eye(nu)),1.)
        impulse=None
        if kind=='impact':
            dyn=Impact(r.model,list(info.cms[1]),0,nu)
            if s.impact_cone:
                impulse=impulse_cone(info)
                shifted=impulse_cone(info,positive=True,shift=duals[k]['impulse'])
                cost.addCost(al.QuadraticResidualCost(space,shifted,np.eye(shifted.nr)),1.)
        else:
            A,B=np.eye(nx),np.zeros((nx,nu));A[:nv,nv:]=h*np.eye(nv)
            for j,value in enumerate([1,2,0]):B[:nv,j*nv:(j+1)*nv]=value*h*h/6*np.eye(nv)
            for j,value in enumerate([1,4,1]):B[nv:,j*nv:(j+1)*nv]=value*h/6*np.eye(nv)
            dyn=al.dynamics.LinearDiscreteDynamics(A,B,np.zeros(nx))
        stages.append(al.StageModel(cost,dyn))
        checks.append(dict(raw=raw,signed=signed,inv=inv,impulse=impulse,points=points,lo=lo,hi=hi,w=w))
    goal=np.r_[s.q_hold2,np.zeros(nv)]
    term=al.QuadraticStateCost(space,0,goal-duals[-1]['terminal'],np.eye(nx))
    problem=al.TrajOptProblem(np.r_[s.q_hold0,np.zeros(nv)],stages,term)
    return SimpleNamespace(problem=problem,checks=checks,duals=duals,goal=goal)


def update(built,xs,us,step=1.):
    """lambda/rho <- projection(lambda/rho + raw constraint)."""
    for k,check in enumerate(built.checks):
        raw,geo,inv=check['raw'],check['signed'],check['inv']
        rd,gd=raw.createData(),geo.createData()
        for j,(A,B) in enumerate(check['points']):
            x=A@xs[k]+B@us[k]
            raw.evaluate(x,us[k],rd);value=np.array(rd.value)
            old=built.duals[k]['physical'][j]
            # Relaxation permits damped outer updates. At step=1 this is the
            # standard interval-projection augmented-Lagrangian update.
            new=value+old-np.clip(value+old,check['lo'],check['hi'])
            built.duals[k]['physical'][j]=(1-step)*old+step*new*(check['w']>0)
            geo.evaluate(x,us[k],gd)
            old=built.duals[k]['geometry'][j]
            built.duals[k]['geometry'][j]=(1-step)*old+step*np.maximum(0,old+np.array(gd.value))
        if inv is not None:
            data=inv.createData();inv.evaluate(xs[k],us[k],data)
            built.duals[k]['inverse']+=step*np.array(data.value)
        if check['impulse'] is not None:
            fn=check['impulse'];data=fn.createData();fn.evaluate(xs[k],us[k],data)
            old=built.duals[k]['impulse']
            built.duals[k]['impulse']=(1-step)*old+step*np.maximum(0.,old+np.array(data.value))
    built.duals[-1]['terminal']+=step*(xs[-1]-built.goal)
    return built.duals
