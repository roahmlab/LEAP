#!/usr/bin/env python3
"""Audit every stored iterate after optimization; resume completed HDF5 shards."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from run import bootstrap,write
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
bootstrap()
import numpy as np
import h5py

CACHE={}
INNER_CACHE={}


def initialize(cpus):
    # One worker per allocated physical core; extra offline cores do not change solver affinity.
    identity=mp.current_process()._identity
    os.sched_setaffinity(0,[cpus[(identity[-1]-1)%len(cpus)]])


def context(robot,dt):
    key=(robot,dt)
    if key not in CACHE:
        import aligator as al
        from public_problem import cold,symbolic,build
        info,mesh,seed_xs,seed_us,_=cold(robot,dt)
        CACHE[("seed",robot,dt)]=(seed_xs,seed_us)
        built=build(info,mesh,symbolic(info),obstacle_scale=1.,geometry_weight=1.,clearance_weight=1.)
        CACHE[key]=(info,mesh,built,al.TrajOptData(built.problem))
    return CACHE[key]


def optimality(built,data,xs,us):
    """KKT residual for the common zero-dual, full-geometry restoration objective.

    Choose shooting costates by backward adjoint recursion. This cancels all
    free-state stationarity rows; remaining control stationarity is reported.
    It is a local diagnostic, not distance to a global minimizer.
    """
    problem=built.problem
    objective=float(problem.evaluate(xs,us,data,1))
    problem.term_cost.computeGradients(xs[-1],np.zeros(problem.term_cost.nu),data.term_cost)
    adj=np.array(data.term_cost.Lx).copy();stationarity=0.;defect=0.
    for k in range(len(us)-1,-1,-1):
        stage=problem.stages[k];sd=data.stage_data[k]
        stage.computeFirstOrderDerivatives(xs[k],us[k],sd)
        dyn=sd.dynamics_data;cost=sd.cost_data
        grad=np.asarray(cost.Lu)+np.asarray(dyn.Ju).T@adj
        stationarity=max(stationarity,float(np.max(np.abs(grad))))
        defect=max(defect,float(np.max(np.abs(np.asarray(dyn.xnext)-xs[k+1]))))
        adj=np.asarray(cost.Lx)+np.asarray(dyn.Jx).T@adj
    defect=max(defect,float(np.max(np.abs(xs[0]-problem.x0_init))))
    return dict(restoration_objective=objective,restoration_stationarity_inf=stationarity,
                shooting_defect_inf=defect,restoration_kkt_inf=max(stationarity,defect))


def inner_optimality(record,phase,index,info,mesh,xs,us):
    """Recompute the actual fixed-multiplier inner objective and its KKT norm."""
    import aligator as al
    from public_problem import symbolic
    from outer_al import build
    key=(str(phase),index)
    if INNER_CACHE.get('key')!=key:
        values=json.loads((phase/f'duals_before_{index:02d}.json').read_text())
        duals=[{name:np.asarray(value,dtype=float) for name,value in row.items()} for row in values]
        call=record['calls'][index];settings=record['settings']
        built=build(info,mesh,symbolic(info),duals,
                    geometry_weight=call.get('geometry_weight',settings.get('geometry_weight',1.)),
                    clearance_weight=settings.get('clearance_weight',1.))
        INNER_CACHE.clear();INNER_CACHE.update(key=key,built=built,data=al.TrajOptData(built.problem))
    return optimality(INNER_CACHE['built'],INNER_CACHE['data'],xs,us)


def add_inner_fields(metric,opt=None,error=''):
    names={'restoration_objective':'inner_objective','restoration_stationarity_inf':'inner_stationarity_inf',
           'shooting_defect_inf':'inner_shooting_defect_inf','restoration_kkt_inf':'inner_kkt_inf'}
    if opt is None:
        opt=metric;error=metric.get('optimality_error','')
    for source,target in names.items():metric[target]=opt[source]
    metric['inner_optimality_valid']=not bool(error)
    metric['inner_optimality_error']=error
    return metric


def sampled_fields(info,mesh,xs,us):
    """Explicit physical arrays on flow collocation nodes and global .01s ticks.

    Repeated boundary times retain their owning stage, including both sides of
    discontinuities. Event primal states remain in the raw trace.
    """
    from physical_audit import dense
    result={}
    for grid in ('node','fine'):
        times=[];stages=[];modes=[];values=[]
        for k,step in enumerate(mesh):
            if step['kind']!='flow':continue
            t,h=step['t'],step['h']
            ticks=[t,t+h/2,t+h] if grid=='node' else np.arange(int(np.ceil((t-1e-12)/.01)),int(np.floor((t+h+1e-12)/.01))+1)*.01
            for tt in ticks:
                values.append(dense(info.r,xs[k],us[k],h,float(np.clip((tt-t)/h,0,1))))
                times.append(tt);stages.append(k);modes.append(step['mode'])
        result[grid]=dict(times=np.array(times),stage=np.array(stages),mode=np.array(modes),
                         **{name:np.array([v[j] for v in values]) for j,name in enumerate(('q','v','a','u','lambda'))})
    return result


from acceptance import THRESHOLD, metric_gate

def process_chunk(task):
    phase,index,begin,end,dest,budget,threshold=task
    phase=Path(phase);dest=Path(dest)
    record=json.loads((phase/'result.json').read_text());call=record['calls'][index]
    if dest.exists():
        with h5py.File(dest,'r+') as f:
            cached=json.loads(f['metrics_json'][()] if 'metrics_json' in f else f.attrs['metrics_json'])
            if 'inner_objective' not in cached[0]:
                if record['settings'].get('outer_al'):raise RuntimeError('Old AL diagnostic shards require recomputation')
                for m in cached:add_inner_fields(m)
                f['metrics_json'][()]=json.dumps(cached)
            for m in cached:metric_gate(m,threshold)
            f['metrics_json'][()]=json.dumps(cached)
            return cached
    info,mesh,built,data=context(record['robot'],record['settings']['dt'])
    initial=np.load(phase/f'input_{index:02d}.npz')
    nx=initial['xs'].shape[1];ns=initial['xs'].size;offsets=initial['offsets']
    raw=np.memmap(phase/f'trace_{index:02d}.bin',dtype='<f8',mode='r',shape=(call['rows'],call['width']))
    from physical_audit import audit
    collected={};schema={};metrics=[];trajectories={};trajectory_layout={}
    for sample in range(begin,end):
        if sample==-1:
            xs=[x.copy() for x in initial['xs']]
            us=[u.copy() for u in np.split(initial['controls'],offsets[1:-1])]
            meta=np.array([0,0,0,-1,0,np.nan,np.nan,np.nan,np.nan])
        else:
            row=raw[sample];meta=row[:9]
            xs=list(row[9:9+ns].reshape(-1,nx));us=list(np.split(row[9+ns:],offsets[1:-1]))
        values={};layout={}
        def sink(group,name,value,t):
            key=f'{group}/{name}';arr=np.asarray(value).reshape(-1)
            values.setdefault(key,[]).append(arr)
            if not metrics:
                entry=layout.setdefault(key,dict(times=[],components=[],occurrences=[]))
                occurrence=len(values[key])-1
                entry['times'].extend([float(t)]*len(arr));entry['components'].extend(range(len(arr)))
                entry['occurrences'].extend([occurrence]*len(arr))
        primal_finite=all(np.isfinite(x).all() for x in xs+us)
        audit_error=''
        try:
            result=audit(info,mesh,xs,us,row_sink=sink)
            if not result.get('finite'):raise ValueError('Nonfinite primal')
        except Exception as exc:
            audit_error=repr(exc)
            values={};layout={}
            sx,su=CACHE[('seed',record['robot'],record['settings']['dt'])]
            template=audit(info,mesh,sx,su,row_sink=sink)
            values={key:[np.full_like(v,np.nan,dtype=float) for v in parts] for key,parts in values.items()}
            result=dict(node={k:float('inf') for k in template['node']},fine={k:float('inf') for k in template['fine']},
                        node_max=float('inf'),fine_max=float('inf'),passed=False,torque_integral=float('nan'))
        for grid,fields in sampled_fields(info,mesh,xs,us).items():
            for name,array in fields.items():
                key=f'trajectory/{grid}/{name}'
                if name in ('times','stage','mode'):trajectory_layout[key]=array
                else:trajectories.setdefault(key,[]).append(array)
        for key,parts in values.items():collected.setdefault(key,[]).append(np.concatenate(parts))
        if not metrics:schema=layout
        at=call['start_s']+meta[1]
        metric=dict(phase=record['phase'],call=index,sample=sample,kind=int(meta[4]),native_iteration=int(meta[3]),
                    time_s=float(at),call_time_s=float(meta[1]),call_wall_s=float(meta[0]),copy_before_s=float(meta[2]),
                    native_cost=float(meta[5]),native_primal_pre_step=float(meta[6]),native_dual_pre_step=float(meta[7]),
                    within_budget=bool(at<=budget),node_max=result['node_max'],fine_max=result['fine_max'],
                    max_violation=max(result['node_max'],result['fine_max']),passed=result['passed'],
                    torque_integral=result['torque_integral'],shard=str(dest),shard_row=len(metrics))
        metric.update(primal_finite=primal_finite,audit_valid=not bool(audit_error),audit_error=audit_error)
        try:
            if not primal_finite:raise ValueError('Nonfinite primal')
            opt=optimality(built,data,xs,us)
            if not all(np.isfinite(v) for v in opt.values()):raise ValueError('Nonfinite optimality diagnostic')
            metric.update(opt,optimality_valid=True,optimality_error='')
        except Exception as exc:
            metric.update({key:float('nan') for key in ('restoration_objective','restoration_stationarity_inf','shooting_defect_inf','restoration_kkt_inf')})
            metric.update(optimality_valid=False,optimality_error=repr(exc))
        if not record['settings'].get('outer_al'):
            add_inner_fields(metric)
        else:
            try:
                if not primal_finite:raise ValueError('Nonfinite primal')
                opt=inner_optimality(record,phase,index,info,mesh,xs,us)
                if not all(np.isfinite(v) for v in opt.values()):raise ValueError('Nonfinite inner optimality diagnostic')
                add_inner_fields(metric,opt)
            except Exception as exc:
                empty={key:float('nan') for key in ('restoration_objective','restoration_stationarity_inf','shooting_defect_inf','restoration_kkt_inf')}
                add_inner_fields(metric,empty,repr(exc))
        for group in ['node','fine']:
            for name,value in result[group].items():metric[f'{group}:{name}']=value
        metric_gate(metric,threshold)
        metrics.append(metric)
    temporary=dest.with_suffix('.tmp.h5')
    with h5py.File(temporary,'w') as f:
        f.create_dataset('metrics_json',data=json.dumps(metrics))
        f.attrs['schema_version']=2
        for key,rows in trajectories.items():f.create_dataset(key,data=np.array(rows),compression='lzf',shuffle=True)
        for key,array in trajectory_layout.items():f.create_dataset(key,data=array)
        f.attrs['row_semantics']='inequality signed excess; equality absolute residual; violation = maximum(value,0)'
        for key,rows in collected.items():
            group=f.require_group(key)
            group.create_dataset('raw',data=np.array(rows),compression='lzf',shuffle=True)
            for name,array in schema[key].items():group.create_dataset(name,data=array)
    temporary.replace(dest)
    return metrics


def postprocess(directory,workers=12,chunk_size=16,offline_cpus=None):
    summary=json.loads((directory/'summary.json').read_text());budget=summary['budget_s'];threshold=THRESHOLD
    summary['acceptance_threshold']=threshold
    if 'postprocess_s' in summary:
        summary.setdefault('prior_postprocessing',[]).append(dict(duration_s=summary['postprocess_s'],samples=summary.get('samples'),
            last_sample_phase=summary.get('last_within_budget',{}).get('phase')))
    tasks=[];copies=0.;setup_total=0.;call_count=0;copy_offsets={};last_record=None
    for phase_name in summary['phases']:
        phase=Path(phase_name);record=json.loads((phase/'result.json').read_text());last_record=record
        dest=phase/'diagnostics';dest.mkdir(exist_ok=True)
        for call in record['calls']:
            setup_total+=call.get('setup_s',0.);call_count+=1
            copy_offsets[(record['phase'],call['index'])]=copies
            native=np.memmap(phase/f'trace_{call["index"]:02d}.bin',dtype='<f8',mode='r',shape=(call['rows'],call['width']))
            copies+=float(native[-1,2])  # Final capture starts after solver.run returns.
            del native
            # The call input is saved explicitly, including continuation jumps.
            for begin in range(-1,call['rows'],chunk_size):
                end=min(begin+chunk_size,call['rows'])
                tasks.append((str(phase),call['index'],begin,end,str(dest/f'call_{call["index"]:02d}_{begin+1:06d}.h5'),budget,threshold))
    processing_cpus=last_record['cpus'] if offline_cpus is None else list(offline_cpus)
    if not 1<=workers<=len(processing_cpus):raise ValueError('One worker per allocated offline physical core at most')
    topology=[]
    for cpu in processing_cpus:
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology.append(((path/'physical_package_id').read_text(),(path/'core_id').read_text()))
    if len(set(topology))!=len(processing_cpus):raise ValueError('Offline CPUs must be distinct physical cores')
    summary['postprocess_cpus']=processing_cpus;summary['postprocess_workers']=workers
    os.sched_setaffinity(0,processing_cpus)
    start=time.perf_counter();metrics=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(processing_cpus,)) as pool:
        for i,rows in enumerate(pool.map(process_chunk,tasks)):
            metrics.extend(rows)
            print(f'{summary["robot"]}: postprocessed shard {i+1}/{len(tasks)} ({len(metrics)} samples)',flush=True)
    metrics.sort(key=lambda x:(x['phase'],x['call'],x['sample']))
    best=float('inf');previous=0.;copy_offset=0.;last_call=None
    for m in metrics:
        m['within_budget']=m['time_s']<=budget
        key=(m['phase'],m['call'])
        if key!=last_call:
            copy_offset=copy_offsets[key]
            last_call=key
        m['time_minus_copy_s']=m['time_s']
        m['time_with_copy_s']=m['time_s']+copy_offset+m['copy_before_s']
        m['step_s']=m['time_s']-previous;previous=m['time_s']
        if m['within_budget']:best=min(best,m['max_violation'])
        m['best_so_far_max']=best
    fields=list(dict.fromkeys(k for m in metrics for k in m))
    with (directory/'iterations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(metrics)
    eligible=[m for m in metrics if m['within_budget']]
    deadline=eligible[-1];best_row=min(eligible,key=lambda m:m['max_violation'])
    last_finite=next((m for m in reversed(eligible) if m['primal_finite'] and m['audit_valid']),None)
    exports=[('deadline_solution',deadline),('best_observed_solution',best_row)]
    if last_finite:exports.append(('last_finite_solution',last_finite))
    for name,m in exports:
        phase=Path(summary['phases'][m['phase']]);initial=np.load(phase/f'input_{m["call"]:02d}.npz')
        if m['sample']==-1:xs=initial['xs'];controls=initial['controls']
        else:
            rec=json.loads((phase/'result.json').read_text());call=rec['calls'][m['call']]
            raw=np.memmap(phase/f'trace_{m["call"]:02d}.bin',dtype='<f8',mode='r',shape=(call['rows'],call['width']))
            row=raw[m['sample']];ns=initial['xs'].size
            xs=row[9:9+ns].reshape(initial['xs'].shape);controls=row[9+ns:]
        np.savez_compressed(directory/f'{name}.npz',xs=xs,controls=controls,offsets=initial['offsets'],mesh_json=initial['mesh_json'])
    summary.update(achieved_feasibility=any(m['passed'] for m in eligible),call_count=call_count,recorded_call_setup_s=setup_total,postprocessed=True,samples=len(metrics),eligible_samples=len(eligible),trace_copy_s=copies,
                   trace_copy_percent=100*copies/summary['solve_wall_s'],postprocess_s=time.perf_counter()-start,
                   last_within_budget=deadline,best_observed=best_row,last_finite=last_finite,
                   invalid_audit_samples=sum(not m['audit_valid'] for m in metrics),
                   deadline_overrun_s=max(0.,summary['solve_compute_s']-budget),
                   first_pass_time_s=next((m['time_s'] for m in eligible if m['passed']),None),
                   first_strict_1e3_pass_time_s=next((m['time_s'] for m in eligible if m['passed_strict_1e3']),None))
    from row_catalog import catalog
    for phase in summary['phases']:
        if not (Path(phase)/'constraint_catalog.json').exists():catalog(Path(phase))
    summary['postprocess_s']=time.perf_counter()-start
    write(directory/'summary.json',summary)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    ap.add_argument('--workers',type=int,default=12);ap.add_argument('--chunk-size',type=int,default=16)
    ap.add_argument('--offline-cpus',type=lambda s:[int(v) for v in s.split(',')],help='Optional distinct physical CPUs for untimed offline processing only')
    args=ap.parse_args();postprocess(args.directory,args.workers,args.chunk_size,args.offline_cpus)
