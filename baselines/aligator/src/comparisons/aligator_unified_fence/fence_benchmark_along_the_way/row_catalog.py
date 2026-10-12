#!/usr/bin/env python3
"""Give each stored constraint row its contact/capsule identity and component."""
import json
from pathlib import Path
import sys
from run import bootstrap,DEFAULT_OUT,write
bootstrap()
import numpy as np
from public_problem import cold
from physical_audit import audit


def catalog(phase):
    record=json.loads((phase/'result.json').read_text())
    info,mesh,_,_,_=cold(record['robot'],record['settings']['dt'])
    archive=np.load(phase/'initial.npz')
    xs=list(archive['xs']);us=list(np.split(archive['controls'],archive['offsets'][1:-1]))
    rows={};offsets={}
    def labels(group,family,value,t,entity):
        key=f'{group}/{family}';start=offsets.get(key,0);count=np.size(value)
        rows.setdefault(key,[]).append(dict(start=start,count=count,time=float(t),entity=entity))
        offsets[key]=start+count
    audit(info,mesh,xs,us,label_sink=labels)
    result=dict(robot=record['robot'],schema_version=1,
                semantics='Blocks index columns of diagnostics/<grid>/<family>/raw; local component is column minus block start',
                contacts=[dict(name=c.name,dimension=c.dim) for c in info.r.contacts],
                act_idx=list(map(int,info.r.act_idx)),families=rows)
    # Check identity ordering against an existing shard whenever available.
    files=sorted((phase/'diagnostics').glob('*.h5'))
    if files:
        import h5py
        with h5py.File(files[0],'r') as f:
            for key,blocks in rows.items():
                times=np.concatenate([np.repeat(b['time'],b['count']) for b in blocks])
                components=np.concatenate([np.arange(b['count']) for b in blocks])
                np.testing.assert_array_equal(times,f[f'{key}/times'][:])
                np.testing.assert_array_equal(components,f[f'{key}/components'][:])
    write(phase/'constraint_catalog.json',result)

if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT_OUT
    for path in sorted((root/'runs').glob('*/*/result.json')):
        catalog(path.parent)
        print(path.parent,'catalogued',flush=True)
