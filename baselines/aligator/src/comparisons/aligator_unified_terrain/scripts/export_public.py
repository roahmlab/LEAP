#!/usr/bin/env python3
"""Export the actual polynomial path for the repository's H5 trajectory viewer.

The original NPZ remains the authoritative ALIGATOR primal archive.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import json
from pathlib import Path
import h5py
import numpy as np
from public_problem import cold
from verify_public import dense,held_coefficients
from pyleap.solution import load_solution
from pyleap.interp import bary_row,bary_weights


def export(directory):
    record=json.loads((directory/'result.json').read_text())
    fresh=directory/'recheck_0.01.json'
    if fresh.exists():record['audit']=json.loads(fresh.read_text());record['passed']=record['audit']['passed']
    info,mesh,_,_,_=cold(record['robot'],record['settings']['dt']);r=info.r
    archive=np.load(directory/'solution.npz');mesh=json.loads(str(archive['mesh_json']));xs=list(archive['xs']);us=list(np.split(archive['controls'],archive['offsets'][1:-1]))
    meta=copy.deepcopy(info.seed.sol.meta)
    for key in ['segments','terms','row_layout','internode','metrics','run','solver','parameterization','overrides']:
        meta.pop(key,None)
    meta.update(generator='aligator_unified_terrain_visualization',segments=[],
                solver=dict(backend='ALIGATOR '+record['calls'][record.get('selected_call',-1)]['solver'],version=record['aligator_version'],outer_al=record['settings'].get('outer_al',False)),
                metrics=dict(gate=record['audit']['node_max'],fine=record['audit']['fine_max'],passed=record['passed']),
                export=dict(authoritative_primal='solution.npz',seed_sha256=info.seed.sha256,
                            note='q/v/a and linear or quadratic torque/wrench curves; no LEAP duals or row layout'))
    meta['parameterization']=dict(derived_actuation=False,control_hold=record['settings'].get('hold','linear'))
    taus=-np.cos(np.linspace(0,np.pi,5));fractions=(taus+1)/2
    flow=[(k,s) for k,s in enumerate(mesh) if s['kind']=='flow']
    path=directory/'trajectory.h5'
    with h5py.File(path,'w') as file:
        file.attrs.update(schema_version=1,generator=meta['generator'])
        for j,(k,step) in enumerate(flow):
            samples=[dense(r,xs[k],us[k],step['h'],float(s)) for s in fractions]
            active=[dict(name=c.name,dim=c.dim) for c in r.contacts if c.name in info.spec.modes[step['mode']].active]
            ids=[];offset=0
            for c in r.contacts:
                if c.name in info.spec.modes[step['mode']].active:ids.extend(range(offset,offset+c.dim))
                offset+=c.dim
            nf=sum(c['dim'] for c in active)
            tau_ctrl,force_ctrl=held_coefficients(r,us[k]);count=len(tau_ctrl)
            reset='none' if j==len(flow)-1 else ('plastic_impact' if mesh[k+1]['kind']=='impact' else 'continuity')
            meta['segments'].append(dict(index=j,parent_mode=step['mode'],t0=step['t'],T=step['h'],Nn=5,degree=4,Nlam=count if nf else 0,
                dims=dict(nq=r.nq,nv=r.nv,na=r.nv,nu=r.nu,nlam=nf),active_contacts=active,reset_after=reset))
            group=file.create_group(f'segments/{j:02d}');group['taus']=taus
            for pos,key in enumerate(['Q','V','A','U']):group[key]=np.array([sample[pos] for sample in samples]).T
            full=np.array([sample[4] for sample in samples]).T
            group['lam_nodes']=full[ids] if nf else np.zeros((0,5));group['aligator_full_wrench']=full
            if nf:group['lam_ctrl']=force_ctrl.T[ids];group['lam_scale']=np.ones(count)
        file['meta_json']=json.dumps(meta)
    # Degree-four q and lower-degree v/a/u must round-trip exactly.
    loaded=load_solution(path);error=0.
    assert not loaded.derived_actuation,'Export must use its actual torque polynomial'
    for segment,(k,step) in zip(loaded.modes,flow):
        for fraction in [.17,.63]:
            row=bary_row(segment.taus,bary_weights(segment.taus),2*fraction-1)
            expected=dense(r,xs[k],us[k],step['h'],fraction)
            for matrix,value in zip([segment.Q,segment.V,segment.A,segment.U],expected):
                error=max(error,float(np.max(abs(matrix@row-value))))
            if segment.lam_ctrl_physical is not None and segment.lam_ctrl_physical.size:
                count=segment.lam_ctrl_physical.shape[1]
                hold=np.array([(1-fraction)**2,2*fraction*(1-fraction),fraction**2]) if count==3 else np.array([1-fraction,fraction])
                force=segment.lam_ctrl_physical@hold
                error=max(error,float(np.max(abs(force-np.concatenate([expected[4][sum(cc.dim for cc in r.contacts[:j]):sum(cc.dim for cc in r.contacts[:j+1])] for j,c in enumerate(r.contacts) if c.name in info.spec.modes[step['mode']].active])))))
    if error>1e-7:raise AssertionError(f'H5 round-trip error {error}')
    print(json.dumps(dict(path=str(path),segments=len(flow),round_trip_max=error)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directories',nargs='+',type=Path)
    for directory in ap.parse_args().directories:export(directory)
