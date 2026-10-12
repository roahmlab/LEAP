"""Full-order fence through ALIGATOR's public residual-model interface.

Shooting state q/v; FOH torque/wrench controls and auxiliary acceleration samples.
Quadratic acceleration is integrated exactly for q/v propagation.
"""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from seed import ROOT
from inputs import inputs
from common.plan import _monitors
from aligator_bench._generated_sparse import GeneratedResidual
from aligator_bench.solve_fence_lifted import generate


def dimensions(r):
    nv,nt,nf=r.nv,r.nu,sum(c.dim for c in r.contacts)
    return nv,nt,nf,3*nv+2*nt+2*nf


def cold(robot,dt):
    info,mesh,xs,_,fields=inputs(robot,dt)
    nv,nt,nf,nu=dimensions(info.r)
    us=[]
    for step in mesh:
        if step['kind']!='flow': us.append(np.zeros(nu));continue
        t,h=step['t'],step['h']
        element=info.seed.element(t)
        samples=[info.seed.sample(t+c*h,element=element) for c in [0.,.5,1.]]
        us.append(np.r_[np.concatenate([v['a'] for v in samples]),
                        samples[0]['u'],samples[-1]['u'],samples[0]['lambda'],samples[-1]['lambda']])
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
    add(x,'x');add(u[3*nv:3*nv+nt],'tau0');add(u[3*nv+nt:3*nv+2*nt],'tau1')
    add(u[3*nv+2*nt:3*nv+2*nt+nf],'force0');add(u[-nf:],'force1')
    for c in r.contacts:
        pose=data.oMf[c.frame_id]
        add(pose.translation,f'pos_{c.name}')
        if c.dim==6: add(cpin.log3(pose.rotation),f'rot_{c.name}')
        J=cpin.getFrameJacobian(model,data,c.frame_id,pin.LOCAL_WORLD_ALIGNED)[:c.dim,:]
        add(J@x[nv:],f'vel_{c.name}')
    for fid,name in _monitors(r): add(data.oMf[fid].translation[2],f'mon_{name}')
    for endpoint in [0,1]:
        offset=3*nv+2*nt+endpoint*nf
        for c in r.contacts:
            f=u[offset:offset+c.dim];offset+=c.dim
            add(ca.sqrt(f[0]**2+f[1]**2+s.friceps**2)-s.friceps-s.mu*f[2],f'cone{endpoint}_{c.name}')
            if c.dim==6:
                add(ca.vertcat(f[4]-s.copx*f[2],-f[4]-s.copx*f[2],
                              f[3]-s.copy*f[2],-f[3]-s.copy*f[2],
                              f[5]-s.gammaz*f[2],-f[5]-s.gammaz*f[2]),f'wrench{endpoint}_{c.name}')
    raw=ca.vertcat(*rows);nr=raw.numel()
    p=ca.SX.sym('p',3*nr)
    residual=p[2*nr:]*(ca.fmax(p[:nr]-raw,0)-ca.fmax(raw-p[nr:2*nr],0))
    obj=SimpleNamespace(x=x,u=u,p=p,cost_y_expr=residual,cost_y_expr_e=residual)
    directory=ROOT/'comparisons/crocoddyl_impact_cone/generated'/f'{r.name}_collocation_physical_v2'
    library,funcs,digest=generate(obj,directory,[("constraints",raw)],optimized=True)
    return SimpleNamespace(library=library,funcs=funcs,digest=digest,names=names,
                           nr=nr,row={name:j for j,name in enumerate(names)},nu=nu)


def bounds(info,step,sym,foot_guide=0.):
    r,s=info.r,info.spec
    nv,nt,nf,nu=dimensions(r)
    lo,hi,w=np.full(sym.nr,-1e20),np.full(sym.nr,1e20),np.zeros(sym.nr)
    def setrow(name,l,h,weight=1.):
        i=sym.row[name];lo[i],hi[i],w[i]=l,h,weight
    def setvec(prefix,l,h,weight=1.):
        l,h=np.atleast_1d(l),np.atleast_1d(h)
        for j in range(max(len(l),len(h))):setrow(f'{prefix}_{j}',l[j%len(l)],h[j%len(h)],weight)
    mode,kind,t,h=step['mode'],step['kind'],step['t'],step['h']
    mid,half=.5*(r.q_lower+r.q_upper),.5*(r.q_upper-r.q_lower)
    for j,k in enumerate(r.act_idx):
        setrow(f'x_{k}',mid[j]-s.qbox*half[j],mid[j]+s.qbox*half[j])
        setrow(f'x_{nv+k}',-s.vlim*r.velocity_limit[j],s.vlim*r.velocity_limit[j])
    setrow('x_2',0.,1e20)
    for k in [3,4,5]:
        if k not in s.planarlock_idx and s.attmax>0:setrow(f'x_{k}',-s.attmax,s.attmax)
    for k in s.planarlock_idx:setrow(f'x_{k}',0.,0.)
    for ep in [0,1]:
        setvec(f'tau{ep}',-.995*s.ulim*r.effort_limit,.995*s.ulim*r.effort_limit)
        offset=0
        for c in r.contacts:
            if mode!=1 and kind=='flow':
                floor=s.lammin
                if mode==0:floor*=np.clip((s.t_stance-(t+ep*h))/(s.t_stance-info.taper_start),0.,1.)
                setrow(f'force{ep}_{offset+2}',floor,1e20)
                setrow(f'cone{ep}_{c.name}_0',-1e20,0.)
                if c.dim==6:setvec(f'wrench{ep}_{c.name}',np.full(6,-1e20),np.zeros(6))
                if mode==0 and ep==1 and abs(t+h-s.t_stance)<1e-10:
                    for j in range(c.dim):setrow(f'force{ep}_{offset+j}',0.,0.)
            else:
                for j in range(c.dim):setrow(f'force{ep}_{offset+j}',0.,0.)
            offset+=c.dim
    target=s.feet0 if mode==0 else s.feet2
    for c in r.contacts:
        if mode!=1 or kind=='impact':
            setvec(f'pos_{c.name}',target[c.name],target[c.name])
            if c.dim==6:setvec(f'rot_{c.name}',np.zeros(3),np.zeros(3))
        if mode!=1:setvec(f'vel_{c.name}',np.zeros(c.dim),np.zeros(c.dim))
        if kind=='impact' and s.landvz>0:setrow(f'vel_{c.name}_2',-s.landvz,1e20)
        if foot_guide>0 and mode==1 and kind=='flow':
            fraction=float(np.clip((t+.5*h-s.t_stance)/s.t_air,0.,1.))
            early=min(fraction/.85,1.);blend=early**3*(10-15*early+6*early**2)
            target=(1-blend)*np.array(s.feet0[c.name])+blend*np.array(s.feet2[c.name])
            height=max([center[2]+half[2] for center,half in s.obstacle_boxes]+[s.footclear])+.1
            target[2]+=16*height*fraction**2*(1-fraction)**2
            setvec(f'pos_{c.name}',target,target,foot_guide)
            if c.dim==6:setvec(f'rot_{c.name}',np.zeros(3),np.zeros(3),.2*foot_guide)
    if kind=='liftoff':setrow(f'x_{nv+2}',s.vz_liftoff,1e20)
    contact_names=[c.name for c in r.contacts]
    for _,name in _monitors(r):
        if mode!=1 and name in contact_names:continue
        setrow(f'mon_{name}_0',max(0.,s.footclear) if mode==1 else 0.,1e20)
    return np.r_[lo,hi,w]
