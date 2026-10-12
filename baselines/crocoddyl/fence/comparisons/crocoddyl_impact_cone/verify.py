"""Independent physical audit of the public-API shooting transcription."""
import numpy as np
import pinocchio as pin
from common.geometry import seg_seg_closest, separating_face_certificate
from common.plan import _monitors
from geometry_reference import axis_box_distance,Physics


def weights(s):
    a=np.array([2*s*s-3*s+1,4*s-4*s*s,2*s*s-s])
    v=np.array([2*s**3/3-1.5*s*s+s,2*s*s-4*s**3/3,2*s**3/3-.5*s*s])
    q=np.array([s**4/6-.5*s**3+.5*s*s,2*s**3/3-s**4/3,s**4/6-s**3/6])
    return q,v,a


def dense(r,x,u,h,s):
    nv,nt,nf=r.nv,r.nu,sum(c.dim for c in r.contacts)
    rates=u[:3*nv].reshape(3,nv)
    qw,vw,aw=weights(s)
    q=x[:nv]+h*s*x[nv:]+h*h*(qw@rates)
    v=x[nv:]+h*(vw@rates);a=aw@rates
    tau=(1-s)*u[3*nv:3*nv+nt]+s*u[3*nv+nt:3*nv+2*nt]
    force=(1-s)*u[3*nv+2*nt:3*nv+2*nt+nf]+s*u[-nf:]
    return q,v,a,tau,force


def inverse(info,q,v,a,tau,force):
    r=info.r;d=r.data()
    pin.computeJointJacobians(r.model,d,q);pin.updateFramePlacements(r.model,d)
    generalized=info.S@tau;offset=0
    for c in r.contacts:
        J=pin.getFrameJacobian(r.model,d,c.frame_id,pin.LOCAL_WORLD_ALIGNED)[:c.dim]
        generalized+=J.T@force[offset:offset+c.dim];offset+=c.dim
    return pin.rnea(r.model,d,q,v,a)-generalized


def audit(info,mesh,xs,us,fine_dt=.01):
    r,s=info.r,info.spec;nv=r.nv
    if not all(np.isfinite(x).all() for x in xs+us):return dict(passed=False,finite=False)
    tables={'node':{},'fine':{}};locations={};self_distance={'node':0.,'fine':0.}
    data=r.data();reset=Physics(info)
    monitors=_monitors(r);contact_names=[c.name for c in r.contacts]
    def bump(group,name,value,t):
        raw=np.asarray(value)
        if not np.isfinite(raw).all():raise ValueError(f'Nonfinite physical {name}')
        val=max(0.,float(np.max(raw)))
        if val>tables[group].get(name,-1.):
            tables[group][name]=val;locations[f'{group}:{name}']=float(t)
    def rows(group,q,v,a,tau,force,mode,t,kind='flow'):
        pin.computeJointJacobians(r.model,data,q);pin.updateFramePlacements(r.model,data)
        mid,half=.5*(r.q_lower+r.q_upper),.5*(r.q_upper-r.q_lower)
        bump(group,'box_q',np.r_[mid-s.qbox*half-q[r.act_idx],q[r.act_idx]-mid-s.qbox*half],t)
        bump(group,'box_v',abs(v[r.act_idx])-s.vlim*r.velocity_limit,t)
        bump(group,'box_u',abs(tau)-.995*s.ulim*r.effort_limit,t)
        bump(group,'base_z',-q[2],t)
        idx=[j for j in [3,4,5] if j not in s.planarlock_idx]
        if idx and s.attmax>0:bump(group,'box_att',abs(q[idx])-s.attmax,t)
        if s.planarlock_idx:bump(group,'planar',abs(q[s.planarlock_idx]),t)
        caps={}
        for name,c in r.capsules.items():
            A=data.oMf[r.frame_id(c.frame)].act(c.a_local);B=data.oMf[r.frame_id(c.b_frame)].act(c.b_local)
            caps[name]=(A,B)
            for center,halfbox in s.obstacle_boxes:
                distance=axis_box_distance(A,B,center,halfbox)
                gap=c.radius+s.obstacle_margin
                value=gap-distance if distance>1e-10 else separating_face_certificate(A,B,center,halfbox,gap)[0]
                bump(group,'fence',value,t)
        for ca,cb,safe in s.selfcol_pairs:
            _,_,dist2=seg_seg_closest(*caps[ca],*caps[cb])
            bump(group,'self_collision',(safe*safe-dist2)/(2*safe),t)
            self_distance[group]=max(self_distance[group],float(safe-np.sqrt(max(0.,dist2))))
        for fid,name in monitors:
            if mode!=1 and name in contact_names:continue
            bump(group,'clear',(max(0.,s.footclear) if mode==1 else 0.)-data.oMf[fid].translation[2],t)
        off=0;generalized=info.S@tau
        for c in r.contacts:
            pose=data.oMf[c.frame_id]
            J=pin.getFrameJacobian(r.model,data,c.frame_id,pin.LOCAL_WORLD_ALIGNED)[:c.dim].copy()
            if mode!=1 or kind=='impact':
                target=s.feet0[c.name] if mode==0 else s.feet2[c.name]
                bump(group,'foot_position',abs(pose.translation-target),t)
                if c.dim==6:bump(group,'foot_orientation',abs(pin.log3(pose.rotation)),t)
            if mode!=1:bump(group,'foot_velocity',abs(J@v),t)
            if kind=='impact' and s.landvz>0:bump(group,'landing_speed',-s.landvz-(J@v)[2],t)
            f=force[off:off+c.dim];off+=c.dim;generalized+=J.T@f
            if mode!=1 and kind!='terminal':
                bump(group,'normal',-f[2],t)
                floor=s.lammin if mode==2 else s.lammin*np.clip((s.t_stance-t)/(s.t_stance-info.taper_start),0.,1.)
                bump(group,'force_floor',floor-f[2],t)
                bump(group,'cone',np.sqrt(f[0]**2+f[1]**2+s.friceps**2)-s.friceps-s.mu*f[2],t)
                if c.dim==6:
                    bump(group,'cop',[abs(f[4])-s.copx*f[2],abs(f[3])-s.copy*f[2]],t)
                    bump(group,'yaw',abs(f[5])-s.gammaz*f[2],t)
            else:bump(group,'inactive_force',abs(f),t)
        if kind=='liftoff':
            bump(group,'liftoff_speed',s.vz_liftoff-v[2],t)
        if mode==0 and abs(t-s.t_stance)<1e-10:bump(group,'liftoff_force',abs(force),t)
        if a is not None:
            bump(group,'dynamics',abs(pin.rnea(r.model,data,q,v,a)-generalized),t)
    effort=0.
    for k,step in enumerate(mesh):
        x,u=xs[k],us[k];t,h,mode,kind=step['t'],step['h'],step['mode'],step['kind']
        if kind!='flow':
            expected=x if kind=='liftoff' else reset.reset(x)
            if kind=='impact':
                impulse=impact_impulse(info,x)
                offset=0
                for contact in r.contacts:
                    f=impulse[offset:offset+3];offset+=contact.dim
                    bump('node','impact_cone',np.sqrt(f[0]**2+f[1]**2+s.friceps**2)-s.friceps-s.mu*f[2],t)
                    bump('node','impact_normal',-f[2],t)
            bump('node','continuity' if kind=='liftoff' else 'impact_reset',abs(xs[k+1]-expected),t)
            bump('node','event_dummy_controls',abs(u),t)
            rows('node',x[:nv],x[nv:],None,np.zeros(r.nu),np.zeros(sum(c.dim for c in r.contacts)),mode,t,kind)
            continue
        for ss in [0.,.5,1.]:
            q,v,a,tau,f=dense(r,x,u,h,ss)
            rows('node',q,v,a,tau,f,mode,t+ss*h)
        bump('node','continuity',abs(np.r_[q,v]-xs[k+1]),t+h)
        for tt in np.arange(int(np.ceil((t-1e-12)/fine_dt)),int(np.floor((t+h+1e-12)/fine_dt))+1)*fine_dt:
            q,v,a,tau,f=dense(r,x,u,h,float(np.clip((tt-t)/h,0.,1.)))
            rows('fine',q,v,a,tau,f,mode,float(tt))
        nt=r.nu;nv=r.nv;ta=u[3*nv:3*nv+nt];tb=u[3*nv+nt:3*nv+2*nt]
        effort+=h*(ta@ta+ta@tb+tb@tb)/3
    bump('node','initial_state',abs(xs[0]-np.r_[s.q_hold0,np.zeros(nv)]),0.)
    rows('node',xs[-1][:nv],xs[-1][nv:],None,np.zeros(r.nu),np.zeros(sum(c.dim for c in r.contacts)),2,info.seed.sol.duration,'terminal')
    bump('node','terminal_state',abs(xs[-1]-np.r_[s.q_hold2,np.zeros(nv)]),info.seed.sol.duration)
    maxima={group:max(table.values(),default=0.) for group,table in tables.items()}
    return dict(passed=max(maxima.values())<=1e-3,finite=True,audit_version='impact-cone-rounded-v1',self_distance_diagnostic=self_distance,node=tables['node'],fine=tables['fine'],
                node_max=maxima['node'],fine_max=maxima['fine'],fine_dt=fine_dt,worst_times=locations,
                torque_integral=float(effort),kinematics='dq/dt=v and dv/dt=a exactly by polynomial integration')


def impact_impulse(info,x):
    """Independent Pinocchio impulse, with LEAP's world-aligned conventions."""
    r=info.r;data=r.data();cms=info.cms[1]
    cds=pin.StdVec_RigidConstraintData([c.createData() for c in cms])
    pin.initConstraintDynamics(r.model,data,cms,cds)
    pin.impulseDynamics(r.model,data,x[:r.nv],x[r.nv:],cms,cds,0.,pin.ProximalSettings(1e-12,1e-9,1))
    return data.impulse_c.copy()
