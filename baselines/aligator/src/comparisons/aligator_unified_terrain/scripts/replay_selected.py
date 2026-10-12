"""Replay a selected trajectory's ALIGATOR-only lineage from the mandatory seed.

Saved native iteration counts replace wall limits for each recorded call.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse
import json
import subprocess
import sys
from seed import ROOT
from io_utils import write_json

folder=Path(__file__).resolve().parents[1]

def plan(path,through=None):
    d=json.loads((path/'result.json').read_text());calls=d['calls']
    if through is None:through=d.get('selected_call')
    if through is not None:calls=[c for c in calls if c['index']<=through]
    parent=d['settings'].get('resume');result=[]
    if parent:
        parent=Path(parent)
        if not parent.is_absolute():parent=ROOT/parent
        pd=json.loads((parent.parent/'result.json').read_text())
        limit=int(parent.stem.split('_')[-1]) if parent.stem.startswith('call_') else pd['selected_call']
        result+=plan(parent.parent,limit)
    for c in calls:
        result.append(dict(source=str(path),robot=d['robot'],settings=d['settings'],call=c['index'],obstacle_scale=c['obstacle_scale'],sway_guide=c.get('sway_guide',0.),identity_impacts=c.get('identity_impacts',False),dynamics_defect_scale=c.get('dynamics_defect_scale',0.),cost_threshold=c.get('cost_threshold'),iterations=c['iterations']+int(bool(c.get('timeout') or c.get('cost_stop')))))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--robot',required=True);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--core-offset',type=int,default=0);ap.add_argument('--threads',type=int,default=12);ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();selected=json.loads((folder/'reports/selected.json').read_text());phases=plan(folder/selected[args.robot]);commands=[];previous=None
    for j,phase in enumerate(phases):
        destination=args.out/f'phase_{j:02d}';s=phase['settings']
        command=[sys.executable,str(Path(__file__).with_name('solve_public.py')),'--robot',args.robot,'--out',str(destination),
                 '--threads',str(args.threads),'--core-offset',str(args.core_offset),'--dt',str(s['dt']),
                 '--hold',s.get('hold','linear'),'--obstacle-scale',str(phase['obstacle_scale']),'--rounds','0' if phase['sway_guide'] or phase['identity_impacts'] or phase['dynamics_defect_scale'] else '1','--iterations',str(phase['iterations']),'--call-seconds','86400']
        for key in ['sway_kind','touchdown_weight','solver','mu_init','control_scale','inverse_weight','acceleration_weight','smooth','reg_init','tolerance','reg_min','geometry_weight','clearance_weight']:
            if s.get(key) is not None:command.extend(['--'+key.replace('_','-'),str(s[key])])
        if phase['cost_threshold'] is not None:command.extend(['--cost-threshold',str(phase['cost_threshold'])])
        if phase['sway_guide']:command.extend(['--sway-guide',str(phase['sway_guide']),'--guide-rounds','1'])
        if phase['identity_impacts']:command.extend(['--identity-rounds','1'])
        if s.get('reduced_dynamics'):
            command.extend(['--reduced-dynamics','--defect-rounds','1' if phase['dynamics_defect_scale'] else '0'])
            if phase['call']>0 or s.get('defect_reference'):command.extend(['--defect-reference',str(previous.parent/'dynamics_defects.npz')])
        if s.get('all_constraints'):command.append('--all-constraints')
        if s.get('state_constraints'):command.append('--state-constraints')
        if s.get('native_acceleration'):command.append('--native-acceleration')
        if s.get('sparse_cost'):command.append('--sparse-cost')
        if previous:command.extend(['--resume',str(previous)])
        if s.get('carry_native_duals'):
            command.append('--carry-native-duals')
            if phase['call']>0 or s.get('native_dual_state'):command.extend(['--native-dual-state',str(previous.parent/'native_duals_00.npz')])
        if s.get('outer_al'):
            command.extend(['--outer-al','--dual-step',str(s.get('dual_step',.5))])
            if phase['call']>0 or s.get('dual_state'):
                command.extend(['--dual-state',str(previous.parent/'duals_00.json')])
        commands.append(command);previous=destination/'solution.npz'
    if args.dry_run:print(json.dumps(commands,indent=2));raise SystemExit(0)
    if args.out.exists():ap.error('Use a new output directory')
    args.out.mkdir(parents=True);write_json(args.out/'replay_plan.json',dict(phases=phases,commands=commands))
    for j,command in enumerate(commands):
        with (args.out/f'phase_{j:02d}.log').open('w') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode
        if code not in [0,2]:raise RuntimeError(f'Phase {j} failed with exit {code}')
    last=json.loads((previous.parent/'result.json').read_text());write_json(args.out/'replay_result.json',dict(final=str(previous),passed=last['passed'],audit=last['audit']))
    raise SystemExit(0 if last['passed'] else 2)
