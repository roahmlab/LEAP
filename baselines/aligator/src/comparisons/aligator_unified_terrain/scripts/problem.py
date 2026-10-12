"""Load the exact terrain inputs and preserve both sides of every mode event."""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
import coal
from seed import Seed, ROOT, problem_yaml
from common.spec_terrain import load_terrain_spec
from common.plan import make_rcms


def inputs(robot, dt):
    seed=Seed(robot)
    path=problem_yaml(robot)
    # windows=True: the approachfrac / liftfrac swing windows become their own
    # modes (same contact set as the flight), as in the C++ mode plan.
    spec=load_terrain_spec(path,windows=True);r=spec.robot
    if seed.sol.nq!=r.nq or seed.sol.nv!=r.nv or not np.array_equal(seed.sol.actuated_idx,r.act_idx):
        raise ValueError('Archived seed model dimensions/actuation differ')
    if seed.sol.joint_names!=list(r.model.names)[1:]:raise ValueError('Archived seed joint order differs')
    if [(c['name'],c['dim']) for c in seed.sol.contacts]!=[(c.name,c.dim) for c in r.contacts]:
        raise ValueError('Archived seed contact order differs')
    S=np.zeros((r.nv,r.nu));S[r.act_idx,np.arange(r.nu)]=1.
    starts=np.r_[0.,np.cumsum([m.T for m in spec.modes])]
    assert abs(starts[-1]-seed.sol.duration)<1e-9
    cms=[pin.StdVec_RigidConstraintModel(make_rcms(r,[c.name for c in r.contacts],m.targets,kp=0.))
         if m.kind=='support' else None for m in spec.modes]
    info=SimpleNamespace(seed=seed,spec=spec,r=r,S=S,cms=cms,path=path,starts=starts)
    info.elements=[];info.clear_times={}
    for mode,md in enumerate(spec.modes):
        elements=[i for i,m in enumerate(seed.sol.modes) if starts[mode]-1e-9<=m.t0<starts[mode+1]-1e-9]
        info.elements.append(elements)
        # ineq_swing_z lives in the FLIGHT mode only (terrain_walk.cpp:881-889); clearz is -1 on the windows.
        info.clear_times[mode]=[seed.sol.modes[i].t0+f*seed.sol.modes[i].T for i in elements for f in [.25,.5,.75]] if md.kind=='swing' and md.window is None and md.clearz>=0 else []
        for i in elements:
            assert set(seed.sol.modes[i].lam_slices)==set(md.active),(robot,mode,i)
    mesh=[dict(kind='initial',mode=0,t=0.,h=0.)];xs=[np.zeros(2*r.nv)];imported=[seed.sample(0.)]
    for mode,md in enumerate(spec.modes):
        a,b=starts[mode:mode+2]
        edges=np.unique(np.round(np.r_[a,b,np.arange(0.,starts[-1],dt),seed.starts,
                         np.arange(0.,starts[-1],.01),info.clear_times[mode]],13))
        edges=edges[(edges>=a-1e-11)&(edges<=b+1e-11)]
        for t,end in zip(edges[:-1],edges[1:]):
            value=seed.sample(t)
            mesh.append(dict(kind='flow',mode=mode,t=float(t),h=float(end-t)))
            imported.append(value);xs.append(np.r_[value['q'],value['v']])
        if mode<len(spec.modes)-1:
            value=seed.sample(b,side='left')
            # Touchdown event: a plastic-impact reset, or -- zeroland + landcont
            # (terrain_walk.cpp:962-966) -- a continuity junction. Every other
            # mode boundary (liftoff, flight -> approach, lift -> flight) is a junction.
            plastic=md.landing and not (spec.zeroland and spec.landcont)
            mesh.append(dict(kind='impact' if plastic else 'junction',mode=mode,t=float(b),h=0.))
            imported.append(value);xs.append(np.r_[value['q'],value['v']])
    value=seed.sample(starts[-1]);imported.append(value);xs.append(np.r_[value['q'],value['v']])
    return info,mesh,xs,[],imported


def ramp(info, mode, contact, t):
    """Nonnegative endpoint floor on force ramps; judge skips ramp element floor."""
    md=info.spec.modes[mode];els=info.elements[mode];a,b=info.starts[mode:mode+2]
    first,last=info.seed.sol.modes[els[0]],info.seed.sol.modes[els[-1]]
    factor=1.
    if contact in md.land_feet:factor=min(factor,max(0.,(t-a)/first.T))
    if contact in md.lift_feet:factor=min(factor,max(0.,(b-t)/last.T))
    return factor


def skip_floor(info,mode,contact,t):
    md=info.spec.modes[mode];els=info.elements[mode]
    first,last=info.seed.sol.modes[els[0]],info.seed.sol.modes[els[-1]]
    return (contact in md.land_feet and t<=first.t0+first.T+1e-11) or (contact in md.lift_feet and t>=last.t0-1e-11)


def geometry(info, obstacle_scale=1.):
    """Self-collision only. Exact terrain face certificates are generated separately."""
    r=info.r;geom=pin.GeometryModel();d=r.data();q=info.seed.sample(0.)['q']
    pin.framesForwardKinematics(r.model,d,q)
    for name,cap in r.capsules.items():
        fid=r.frame_id(cap.frame);frame=r.model.frames[fid];a=cap.a_local
        b=d.oMf[fid].actInv(d.oMf[r.frame_id(cap.b_frame)].act(cap.b_local))
        check=r.data();pin.framesForwardKinematics(r.model,check,q+.03*np.sin(np.arange(r.nv)+1))
        assert np.max(abs(check.oMf[fid].actInv(check.oMf[r.frame_id(cap.b_frame)].act(cap.b_local))-b))<1e-9
        axis=b-a;rot=pin.Quaternion.FromTwoVectors(np.array([0.,0.,1.]),axis).matrix()
        geom.addGeometryObject(pin.GeometryObject(name,frame.parentJoint,fid,frame.placement*pin.SE3(rot,.5*(a+b)),coal.Capsule(cap.radius,float(np.linalg.norm(axis)))))
    ids={g.name:i for i,g in enumerate(geom.geometryObjects)};rows=[]
    for a,b,safe in info.spec.selfcol_pairs:
        geom.addCollisionPair(pin.CollisionPair(ids[a],ids[b]));rows.append((f'self_{a}_{b}',safe-r.capsules[a].radius-r.capsules[b].radius))
    return geom,rows


def reset(info,x,mode):
    r=info.r;d=r.data();cms=info.cms[mode+1]
    cds=pin.StdVec_RigidConstraintData([cm.createData() for cm in cms])
    pin.initConstraintDynamics(r.model,d,cms,cds)
    pin.impulseDynamics(r.model,d,x[:r.nv],x[r.nv:],cms,cds,0.,pin.ProximalSettings(1e-12,1e-9,1))
    return np.r_[x[:r.nv],d.dq_after.copy()]


def lateral_reference(info):
    """Fixed-height LIPM reference for a temporary cost, never a primal initializer.

    Solve -y'' + omega^2 y = omega^2 zmp, y'(0)=y'(T)=0, by tridiagonal
    linear algebra. The ZMP moves between the original active support feet.
    """
    r,s=info.r,info.spec;T=info.starts[-1];times=np.linspace(0.,T,int(np.ceil(T/.005))+1);h=times[1]
    q=info.seed.sample(0.)['q'];data=r.data();com=np.asarray(pin.centerOfMass(r.model,data,q)).copy()
    ground=np.mean([p[2] for p in s.modes[0].targets.values()]);height=max(.1,float(com[2]-ground));omega2=9.81/height
    def center(md):return float(np.mean([md.targets[n][1] for n in md.active]))
    zmp=[]
    for t in times:
        mode=min(int(np.searchsorted(info.starts,t+1e-11)-1),len(s.modes)-1);md=s.modes[mode]
        if md.kind=='swing':zmp.append(center(md));continue
        left=center(s.modes[mode-1]) if mode else center(md)
        right=center(s.modes[mode+1]) if mode+1<len(s.modes) else center(md)
        fraction=(t-info.starts[mode])/md.T;zmp.append((1-fraction)*left+fraction*right)
    zmp=np.array(zmp);count=len(times);lower=np.full(count-1,-1/h**2);upper=lower.copy();lower[-1]*=2;upper[0]*=2
    diagonal=np.full(count,2/h**2+omega2);rhs=omega2*zmp
    for i in range(1,count):
        factor=lower[i-1]/diagonal[i-1];diagonal[i]-=factor*upper[i-1];rhs[i]-=factor*rhs[i-1]
    y=np.empty(count);y[-1]=rhs[-1]/diagonal[-1]
    for i in range(count-2,-1,-1):y[i]=(rhs[i]-upper[i]*y[i+1])/diagonal[i]
    if not np.isfinite(y).all() or y.min()<zmp.min()-1e-8 or y.max()>zmp.max()+1e-8:raise ValueError('Invalid lateral reference')
    return dict(times=times,base_y=y-(com[1]-q[1]),com_y=y,zmp=zmp,height=height)
