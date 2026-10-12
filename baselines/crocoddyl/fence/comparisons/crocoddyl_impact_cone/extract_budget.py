#!/usr/bin/env python3
"""Extract a current or best-so-far candidate available within a fixed budget."""
import argparse,json
from pathlib import Path
import h5py,numpy as np

def select(records,seconds,clock,best):
    key={'wall':'available_wall_s','solver':'solver_s'}[clock]
    eligible=[r for r in records if r[key]<=seconds]
    if not eligible:raise ValueError('No candidate was recorded within the requested budget')
    return min(eligible,key=lambda r:r['maximum_violation']) if best else eligible[-1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--seconds',type=float,required=True)
    ap.add_argument('--clock',choices=['wall','solver'],default='wall');ap.add_argument('--best',action='store_true');ap.add_argument('--output',type=Path)
    args=ap.parse_args();records=[json.loads(line) for line in (args.directory/'iterations.jsonl').read_text().splitlines()]
    chosen=select(records,args.seconds,args.clock,args.best)
    if args.output:
        if args.output.exists():ap.error('Output exists')
        with h5py.File(args.directory/'iterations.h5','r',swmr=True) as f:
            np.savez_compressed(args.output,xs=f['xs'][chosen['index']],us=f['us'][chosen['index']],mesh_json=f.attrs['mesh_json'],metrics_json=json.dumps(chosen))
    print(json.dumps(chosen,indent=2))
if __name__=='__main__':main()
