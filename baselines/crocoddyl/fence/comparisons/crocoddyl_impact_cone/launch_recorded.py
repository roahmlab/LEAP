#!/usr/bin/env python3
"""Launch each robot on its preselected twelve physical cores and collect outcomes."""
import argparse,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    allocation=json.loads((args.output/'allocation.json').read_text())
    jobs=[]
    for robot,cpus in allocation.items():
        command=[str(HERE/'run_python.sh'),str(HERE/'solve.py'),'--robot',robot,
                 '--output',str(args.output/robot),'--cpus',','.join(map(str,cpus)),
                 '--solver-seconds','14400','--wall-seconds','86400']
        log=(args.output/f'{robot}.log').open('w')
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((robot,p,log,command))
    while True:
        records=[dict(robot=r,pid=p.pid,exit_code=p.poll(),command=c) for r,p,log,c in jobs]
        temporary=args.output/'manifest.json.tmp'
        temporary.write_text(json.dumps(records,indent=2)+'\n')
        temporary.replace(args.output/'manifest.json')
        if all(r['exit_code'] is not None for r in records):break
        time.sleep(5)
    for r,p,log,c in jobs:log.close()
    return 0 if all(r['exit_code']==0 for r in records) else 2

if __name__=='__main__':raise SystemExit(main())
