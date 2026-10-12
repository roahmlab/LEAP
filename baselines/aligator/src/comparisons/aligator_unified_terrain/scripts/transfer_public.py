"""Exact mesh prolongation of this experiment's own polynomial candidate."""
import json
from pathlib import Path
import numpy as np
from verify_public import dense,controls_from_samples


def transfer(info,newmesh,path):
    path=Path(path)
    archive=np.load(path);oldmesh=json.loads(str(archive['mesh_json']))
    oldxs=list(archive['xs']);oldus=list(np.split(archive['controls'],archive['offsets'][1:-1]))
    count=3 if getattr(info.r,'hold','linear')=='quadratic' else 2
    nu=3*info.r.nv+count*(info.r.nu+sum(c.dim for c in info.r.contacts))
    if oldmesh==newmesh and len(oldus[0])==nu:return oldxs,oldus
    xs,us=[],[]
    for step in newmesh:
        t,h,mode=step['t'],step['h'],step['mode']
        if step['kind']!='flow':
            k=next(k for k,s in enumerate(oldmesh) if s['kind']==step['kind'] and s['mode']==mode and abs(s['t']-t)<1e-10)
            xs.append(oldxs[k].copy())
            control=np.zeros(nu);control[:min(nu,len(oldus[k]))]=oldus[k][:min(nu,len(oldus[k]))]
            us.append(control);continue
        candidates=[k for k,s in enumerate(oldmesh) if s['kind']=='flow' and s['mode']==mode
                    and s['t']-1e-11<=t and t+h<=s['t']+s['h']+1e-11]
        if not candidates:raise ValueError('New interval is not a subdivision of an old interval')
        k=candidates[-1];old=oldmesh[k]
        pts=[dense(info.r,oldxs[k],oldus[k],old['h'],np.clip((t+c*h-old['t'])/old['h'],0.,1.)) for c in [0.,.5,1.]]
        xs.append(np.r_[pts[0][0],pts[0][1]])
        us.append(controls_from_samples(info.r,pts))
    xs.append(oldxs[-1].copy())
    return xs,us
