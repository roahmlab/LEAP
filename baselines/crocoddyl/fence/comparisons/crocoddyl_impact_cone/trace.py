"""Crash-readable iteration archive and timing, including rejected native steps."""
from pathlib import Path
import json,os,time,hashlib,resource
import h5py
import numpy as np
from physics import audit

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def finite(value):
    try:
        v=float(value)
        return v if np.isfinite(v) else None
    except (TypeError,ValueError):return None

class Trace:
    def __init__(self,folder,built,start,budget,solver_budget=14400.):
        self.solver_budget=solver_budget
        self.folder=Path(folder);self.built=built;self.start=start;self.budget=budget
        self.index=0;self.round=0;self.overhead=0.;self.build_seconds=0.;self.last_end=time.perf_counter();self.solver_seconds=0.
        self.last_digest=None;self.best=np.inf;self.passed=False;self.reason=None;self.last_report=None
        self.file=h5py.File(self.folder/'iterations.h5','w',libver='latest')
        self.file.attrs['schema']='crocoddyl-impact-cone-trace-v2'
        self.file.attrs['mesh_json']=json.dumps(built.mesh)
        self.file.attrs['physical_objective']='none (feasibility restoration)'
        self.file.attrs['hold']=built.hold
        self.file.attrs['state_layout']='q,v; explicit nq=nv coordinates'
        self.file.attrs['control_layout']='a0,am,a1; torque Bernstein coefficients; all-contact wrench Bernstein coefficients'
        r=built.info.r
        self.file.attrs['joint_names']=json.dumps(list(r.model.names)[1:])
        self.file.attrs['contacts']=json.dumps([dict(name=c.name,dim=c.dim) for c in r.contacts])
        self.file.create_dataset('actuated_idx',data=r.act_idx)
        shape_x=np.array(built.xs).shape;shape_u=np.array(built.us).shape
        self.xs=self.file.create_dataset('xs',shape=(0,*shape_x),maxshape=(None,*shape_x),chunks=(1,*shape_x),compression='lzf',dtype='f8')
        self.us=self.file.create_dataset('us',shape=(0,*shape_u),maxshape=(None,*shape_u),chunks=(1,*shape_u),compression='lzf',dtype='f8')
        string=h5py.string_dtype('utf-8')
        self.records=self.file.create_dataset('records_json',shape=(0,),maxshape=(None,),chunks=(64,),dtype=string)
        nv,nt,nf=r.nv,r.nu,sum(c.dim for c in r.contacts)
        self.file.attrs['nv']=nv;self.file.attrs['nt']=nt;self.file.attrs['nf']=nf
        self.file.attrs['timing_note']='solver_s excludes entire Python recorder; step_solver_s is its increment; wall includes preparation and recording'
        self.decoded={}
        for name,shape in [('q',(shape_x[0],nv)),('v',(shape_x[0],nv)),('a',(shape_u[0],3,nv)),('u',(shape_u[0],2,nt)),('lambda',(shape_u[0],2,nf))]:
            self.decoded[name]=self.file.create_dataset(name,shape=(0,*shape),maxshape=(None,*shape),chunks=(1,*shape),compression='lzf',dtype='f8')
        self.previous_solver_seconds=0.
        self.file.attrs['committed_iterations']=0
        self.file.attrs['best_index']=-1
        self.file.swmr_mode=True
        self.jsonl=open(self.folder/'iterations.jsonl','a',buffering=1)

    def write(self,xs,us,solver=None,event='iteration',entered=None):
        entered=time.perf_counter() if entered is None else entered
        if event=='iteration':self.solver_seconds+=entered-self.last_end
        xs=np.array(xs);us=np.array(us)
        report=audit(self.built,list(xs),list(us))
        if not report.get('finite'):raise FloatingPointError('Nonfinite native candidate')
        self.last_report=report;violation=max(report['node_max'],report['fine_max'])
        digest=hashlib.sha256(xs.tobytes()+us.tobytes()).hexdigest();changed=digest!=self.last_digest
        metrics={}
        if solver is not None:
            for key in ['cost','stop','stepLength','preg','dreg','dV','dVexp','ffeas','gfeas','hfeas','iter','isFeasible']:
                if hasattr(solver,key):metrics[key]=finite(getattr(solver,key))
        else:
            metrics['cost']=float(self.built.problem.calc(list(xs),list(us)))
        row=dict(index=self.index,event=event,round=self.round,wall_s=entered-self.start,
                 solver_s=self.solver_seconds,step_solver_s=self.solver_seconds-self.previous_solver_seconds,previous_logging_s=self.overhead,build_s=self.build_seconds,
                 trajectory_changed=changed,trajectory_sha256=digest,maximum_violation=violation,
                 node_max=report['node_max'],fine_max=report['fine_max'],node=report['node'],fine=report['fine'],
                 worst_times=report['worst_times'],physical_objective=0.,torque_integral=report['torque_integral'],
                 restoration_cost=metrics.get('cost'),native=metrics,passed=report['passed'],
                 optimality_note='Physical objective is constant zero; native stopping metric is not a physical KKT certificate.')
        usage=resource.getrusage(resource.RUSAGE_SELF)
        row.update(cpu_user_s=usage.ru_utime,cpu_system_s=usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024,machine_load=list(os.getloadavg()))
        if violation<self.best:self.best=violation;self.file.attrs['best_index']=self.index
        row['best_maximum_violation']=self.best
        for dataset,array in [(self.xs,xs),(self.us,us)]:
            dataset.resize(self.index+1,axis=0);dataset[self.index]=array
        nv,nt,nf=self.built.info.r.nv,self.built.info.r.nu,sum(c.dim for c in self.built.info.r.contacts)
        arrays=dict(q=xs[:,:nv],v=xs[:,nv:],a=us[:,:3*nv].reshape(-1,3,nv),u=us[:,3*nv:3*nv+2*nt].reshape(-1,2,nt))
        arrays['lambda']=us[:,3*nv+2*nt:].reshape(-1,2,nf)
        for name,array in arrays.items():
            dataset=self.decoded[name];dataset.resize(self.index+1,axis=0);dataset[self.index]=array
        self.records.resize(self.index+1,axis=0)
        self.file.flush()
        row['audit_and_archive_s']=time.perf_counter()-entered
        row['available_wall_s']=time.perf_counter()-self.start
        self.records[self.index]=json.dumps(row,allow_nan=False)
        self.file.flush();self.file.attrs.modify('committed_iterations',self.index+1);self.file.flush()
        self.jsonl.write(json.dumps(row,allow_nan=False)+'\n')
        self.passed=report['passed'];self.reason='physical_gate' if self.passed else ('solver_budget' if self.solver_seconds>=self.solver_budget else ('wall_budget' if time.perf_counter()-self.start>=self.budget else None))
        atomic_json(self.folder/'status.json',dict(index=self.index,round=self.round,maximum_violation=violation,node_max=report['node_max'],fine_max=report['fine_max'],best_maximum_violation=self.best,solver_s=self.solver_seconds,wall_s=time.perf_counter()-self.start,passed=self.passed,stop_reason=self.reason,trajectory_file='iterations.h5',trajectory_index=self.index))
        if self.index%10==0 or self.reason:print(f'iteration={self.index} round={self.round} max={violation:.6e} solve={self.solver_seconds:.2f}s wall={time.perf_counter()-self.start:.2f}s stop={self.reason}',flush=True)
        self.previous_solver_seconds=self.solver_seconds
        self.index+=1;self.last_digest=digest;self.overhead+=time.perf_counter()-entered;self.last_end=time.perf_counter()
        return self.reason is not None

    def __call__(self,solver):
        entered=time.perf_counter()
        return self.write(solver.xs,solver.us,solver,entered=entered)
    def close(self):self.file.close();self.jsonl.close()
