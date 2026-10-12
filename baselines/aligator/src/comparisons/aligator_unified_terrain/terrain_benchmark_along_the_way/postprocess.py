#!/usr/bin/env python3
"""Audit every captured ALIGATOR iterate of a run offline and export the dataset.

For each traced native call the RAM snapshots (trace_JJ.primal.bin / .metrics.bin written by
../along_the_way/_trace_run after solver.run returned) are decoded into the shooting primal
(states x=[q,v] at every stage, controls [a0,am,a1,tau..,F..]) and, per iterate:
  * the independent physical audit (../scripts/verify_public.py): every constraint family at
    the collocation/event nodes and on the global 0.01 s grid, every component violation kept;
  * the physical trajectory q, v, a, u (torque), lambda (contact force/wrench) sampled at the
    collocation nodes and on the 0.01 s grid;
  * first-order optimality of the restoration objective FDDP actually minimized in that call
    (adjoint recursion on the public problem: control stationarity, shooting defects).
Results: diagnostics/*.h5 shards, iterations.csv, summary.json, exported candidates, plots,
REPORT.md. Completed shards are reused (delete diagnostics/ to recompute).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

from atw_common import (HERE, ROOT, STRICT_GATE, LOOSE_GATE, METRIC_COLUMNS, bootstrap, write_json, read_json, phase_calls, digest)
bootstrap()
import numpy as np
import h5py

CACHE = {}
KKT_KEYS = ('restoration_objective', 'restoration_stationarity_inf', 'shooting_defect_inf', 'restoration_kkt_inf')


def initialize(cpus):
    identity = mp.current_process()._identity
    os.sched_setaffinity(0, [cpus[(identity[-1] - 1) % len(cpus)]])


def context(phase, problem=True):
    """Problem inputs (+ the public ALIGATOR problem when `problem`) of a phase, cached per process."""
    phase = Path(phase)
    if CACHE.get('phase') != str(phase) or (problem and 'built' not in CACHE):
        from public_problem import cold, symbolic, build
        record = read_json(phase / 'result.json')
        s = record['settings']
        for key in ('smooth', 'native_acceleration', 'sparse_cost', 'outer_al', 'reduced_dynamics', 'state_constraints', 'all_constraints', 'sway_guide', 'identity_rounds'):
            if s.get(key):
                raise RuntimeError(f'{phase}: postprocess does not model setting {key}={s[key]}')
        if s.get('control_scale', 1.) != 1. or s.get('obstacle_scale', 1.) != 1.:
            raise RuntimeError(f'{phase}: unit control/obstacle scale expected')
        info, mesh, seed_xs, seed_us, _ = cold(record['robot'], s['dt'], s.get('hold', 'linear'))
        initial = np.load(phase / 'initial.npz')
        assert json.loads(str(initial['mesh_json'])) == mesh, 'phase mesh differs from the reconstructed mesh'
        CACHE.clear()
        CACHE.update(phase=str(phase), record=record, info=info, mesh=mesh, seed=(seed_xs, seed_us), offsets=initial['offsets'], xshape=initial['xs'].shape)
        if problem:
            import aligator as al
            built = build(info, mesh, symbolic(info), obstacle_scale=1., geometry_weight=s.get('geometry_weight', 1.),
                          clearance_weight=s.get('clearance_weight', 1.), inverse_weight=s.get('inverse_weight', 1.),
                          acceleration_weight=s.get('acceleration_weight', 1.), touchdown_weight=s.get('touchdown_weight', 1.))
            CACHE.update(built=built, data=al.TrajOptData(built.problem))
    return CACHE


def optimality(built, data, xs, us):
    """KKT residual of the call's restoration objective at (xs, us): costates from the
    backward adjoint recursion cancel every state row; the remaining control gradient and
    the shooting defects are reported. Local diagnostic, not a global optimality gap."""
    problem = built.problem
    objective = float(problem.evaluate(xs, us, data, 1))
    problem.term_cost.computeGradients(xs[-1], np.zeros(problem.term_cost.nu), data.term_cost)
    adj = np.array(data.term_cost.Lx).copy()
    stationarity = defect = 0.
    for k in range(len(us) - 1, -1, -1):
        stage = problem.stages[k]
        sd = data.stage_data[k]
        stage.computeFirstOrderDerivatives(xs[k], us[k], sd)
        dyn, cost = sd.dynamics_data, sd.cost_data
        grad = np.asarray(cost.Lu) + np.asarray(dyn.Ju).T @ adj
        stationarity = max(stationarity, float(np.max(np.abs(grad))))
        defect = max(defect, float(np.max(np.abs(np.asarray(dyn.xnext) - xs[k + 1]))))
        adj = np.asarray(cost.Lx) + np.asarray(dyn.Jx).T @ adj
    defect = max(defect, float(np.max(np.abs(xs[0] - problem.x0_init))))
    return dict(restoration_objective=objective, restoration_stationarity_inf=stationarity,
                shooting_defect_inf=defect, restoration_kkt_inf=max(stationarity, defect))


def sampled_fields(info, mesh, xs, us):
    """q, v, a, u, lambda on the flow collocation nodes (s = 0, 1/2, 1 of every flow stage)
    and on the global 0.01 s ticks owned by each flow stage (both sides of a boundary)."""
    from verify_public import dense
    result = {}
    for grid in ('node', 'fine'):
        times, stages, modes, values = [], [], [], []
        for k, step in enumerate(mesh):
            if step['kind'] != 'flow':
                continue
            t, h = step['t'], step['h']
            ticks = [t, t + h / 2, t + h] if grid == 'node' else np.arange(int(np.ceil((t - 1e-12) / .01)), int(np.floor((t + h + 1e-12) / .01)) + 1) * .01
            for tt in ticks:
                values.append(dense(info.r, xs[k], us[k], h, float(np.clip((tt - t) / h, 0., 1.))))
                times.append(float(tt)); stages.append(k); modes.append(step['mode'])
        result[grid] = dict(times=np.array(times), stage=np.array(stages), mode=np.array(modes),
                            **{name: np.array([v[j] for v in values]) for j, name in enumerate(('q', 'v', 'a', 'u', 'lambda'))})
    return result


def snapshot_primal(phase, call, index, ctx):
    """(xs, us, meta) of snapshot `index` of call `call` (-1 = the call's input)."""
    phase = Path(phase)
    if index < 0:
        # solve_public feeds a call the previous call's returned primal when it is finite,
        # otherwise the last finite one (or the phase input).
        source = phase / 'initial.npz'
        for earlier in ctx['record']['calls'][:call['index']]:
            if earlier.get('finite'):
                source = phase / f"call_{earlier['index']:02d}.npz"
        z = np.load(source)
        xs = [x.copy() for x in z['xs']]
        us = [u.copy() for u in np.split(z['controls'], z['offsets'][1:-1])]
        return xs, us, None
    tr = call['trace']
    primal = np.memmap(phase / f"trace_{call['index']:02d}.primal.bin", dtype=np.float64, mode='r', shape=(tr['count'], tr['width']))
    metrics = np.fromfile(phase / f"trace_{call['index']:02d}.metrics.bin", dtype=np.float64).reshape(-1, 8)
    row = np.array(primal[index]); nx = int(np.prod(ctx['xshape']))
    xs = list(row[:nx].reshape(ctx['xshape']))
    us = list(np.split(row[nx:] * ctx['record']['settings'].get('control_scale', 1.), ctx['offsets'][1:-1]))
    return xs, us, metrics[index].copy()


def process_chunk(task):
    phase, call_index, begin, end, dest, cpus = task
    phase, dest = Path(phase), Path(dest)
    if dest.exists():
        with h5py.File(dest, 'r') as f:
            return json.loads(f['metrics_json'][()])
    ctx = context(phase)
    from verify_public import audit
    record = ctx['record']
    call = record['calls'][call_index]
    info, mesh, built, data = ctx['info'], ctx['mesh'], ctx['built'], ctx['data']
    collected, schema, metrics, trajectories, layout_arrays = {}, {}, [], {}, {}
    for sample in range(begin, end):
        xs, us, meta = snapshot_primal(phase, call, sample, ctx)
        values, layout = {}, {}
        def sink(group, name, value, t):
            key = f'{group}/{name}'
            arr = np.asarray(value).reshape(-1)
            values.setdefault(key, []).append(arr)
            if not metrics:
                entry = layout.setdefault(key, dict(times=[], components=[], occurrences=[]))
                entry['times'].extend([float(t)] * len(arr)); entry['components'].extend(range(len(arr)))
                entry['occurrences'].extend([len(values[key]) - 1] * len(arr))
        primal_finite = all(np.isfinite(x).all() for x in xs + us)
        audit_error = ''
        try:
            if not primal_finite:
                raise ValueError('Nonfinite primal')
            result = audit(info, mesh, xs, us, row_sink=sink)
            if not result.get('finite'):
                raise ValueError('Audit reported a nonfinite primal')
        except Exception as exc:  # noqa: BLE001
            audit_error = repr(exc); values, layout = {}, {}
            sx, su = ctx['seed']
            template = audit(info, mesh, sx, su, row_sink=sink)
            values = {key: [np.full_like(v, np.nan, dtype=float) for v in parts] for key, parts in values.items()}
            result = dict(node={k: float('inf') for k in template['node']}, fine={k: float('inf') for k in template['fine']},
                          node_max=float('inf'), fine_max=float('inf'), torque_integral=float('nan'))
        try:
            fields = sampled_fields(info, mesh, xs, us)
        except Exception:  # noqa: BLE001
            fields = None
        if fields is not None:
            for grid, table in fields.items():
                for name, array in table.items():
                    key = f'trajectory/{grid}/{name}'
                    if name in ('times', 'stage', 'mode'):
                        layout_arrays[key] = array
                    else:
                        trajectories.setdefault(key, []).append(array)
        for key, parts in values.items():
            collected.setdefault(key, []).append(np.concatenate(parts))
        if not metrics:
            schema = layout
        metric = dict(phase=int(phase.name.split('_')[1]), call=call_index, sample=sample,
                      kind=0 if sample < 0 else (2 if meta[0] < 0 else 1),
                      native_iteration=-1 if sample < 0 else int(meta[0]),
                      call_solver_s=0. if sample < 0 else float(meta[2]), call_raw_s=0. if sample < 0 else float(meta[1]),
                      capture_cumulative_s=0. if sample < 0 else float(meta[3]), capture_step_s=0. if sample < 0 else float(meta[7]),
                      native_cost=float('nan') if sample < 0 else float(meta[4]),
                      native_primal_infeasibility=float('nan') if sample < 0 else float(meta[5]),
                      native_dual_infeasibility=float('nan') if sample < 0 else float(meta[6]),
                      node_max=result['node_max'], fine_max=result['fine_max'], max_violation=max(result['node_max'], result['fine_max']),
                      torque_integral=result['torque_integral'], primal_finite=primal_finite, audit_valid=not audit_error, audit_error=audit_error,
                      shard=str(dest.relative_to(dest.parents[1])), shard_row=len(metrics))
        try:
            if not primal_finite:
                raise ValueError('Nonfinite primal')
            opt = optimality(built, data, xs, us)
            if not all(np.isfinite(v) for v in opt.values()):
                raise ValueError('Nonfinite optimality diagnostic')
            metric.update(opt, optimality_valid=True, optimality_error='')
        except Exception as exc:  # noqa: BLE001
            metric.update({k: float('nan') for k in KKT_KEYS}, optimality_valid=False, optimality_error=repr(exc))
        for group in ('node', 'fine'):
            for name, value in result[group].items():
                metric[f'{group}:{name}'] = value
        metrics.append(metric)
    temporary = dest.with_suffix('.tmp.h5')
    with h5py.File(temporary, 'w') as f:
        f.attrs['schema_version'] = 1
        f.attrs['row_semantics'] = 'component violations >= 0 (inequalities: positive excess, equalities: |residual|); 0 = satisfied'
        f.attrs['trajectory_semantics'] = 'q,v,a: nq/nv generalized coordinates; u: actuator torque (actuated joints, robot yaml order); lambda: contact force/wrench, contacts in robot yaml order, LOCAL_WORLD_ALIGNED, [fx,fy,fz(,tx,ty,tz)]'
        f.create_dataset('metrics_json', data=json.dumps(metrics))
        for key, rows in trajectories.items():
            f.create_dataset(key, data=np.array(rows), compression='lzf', shuffle=True)
        for key, array in layout_arrays.items():
            f.create_dataset(key, data=array)
        for key, rows in collected.items():
            g = f.require_group(key)
            g.create_dataset('raw', data=np.array(rows), compression='lzf', shuffle=True)
            for name, array in schema[key].items():
                g.create_dataset(name, data=np.array(array))
    temporary.replace(dest)
    return metrics


def catalog(phase, ctx=None):
    """Block layout of the audit rows: (family, time, component count) in audit order."""
    ctx = ctx or context(phase, problem=False)
    from verify_public import audit
    sx, su = ctx['seed']
    blocks = {}
    def sink(group, name, value, t):
        key = f'{group}/{name}'
        blocks.setdefault(key, []).append(dict(start=sum(b['count'] for b in blocks.get(key, [])), count=int(np.size(value)), time=float(t)))
    audit(ctx['info'], ctx['mesh'], sx, su, row_sink=sink)
    r = ctx['info'].r
    return dict(robot=ctx['record']['robot'], dt=ctx['record']['settings']['dt'], hold=ctx['record']['settings'].get('hold', 'linear'),
                contacts=[dict(name=c.name, dim=c.dim) for c in r.contacts], joint_names=list(r.model.names)[1:],
                actuated_joints=list(r.actuated_joints), act_idx=[int(i) for i in r.act_idx],
                monitor_frames=[n for _, n in __import__('common.plan', fromlist=['_monitors'])._monitors(r)],
                keepout_capsules=list(ctx['info'].spec.keepout_capsules), keepout_boxes=len(ctx['info'].spec.keepout_boxes),
                selfcol_pairs=[list(p[:2]) for p in ctx['info'].spec.selfcol_pairs],
                mesh=ctx['mesh'], row_order='within one (family, time) block the audit loops contacts / capsules x boxes / self-collision pairs / monitor frames in the orders listed here; components are the vector rows of one evaluation (see README)',
                families=blocks)


def export_candidate(phase, call, sample, out, ctx):
    xs, us, _ = snapshot_primal(phase, call, sample, ctx)
    np.savez_compressed(out, xs=np.array(xs), controls=np.concatenate(us), offsets=ctx['offsets'], mesh_json=json.dumps(ctx['mesh']),
                        phase=str(phase), call=call['index'], sample=sample)


def postprocess(directory, workers, chunk_size):
    directory = Path(directory)
    experiment = read_json(directory / 'experiment.json')
    cpus = list(experiment['cpus'])
    if not 1 <= workers <= len(cpus):
        raise ValueError('one worker per allocated physical core at most')
    os.sched_setaffinity(0, cpus)
    started = time.perf_counter()
    tasks, call_offsets = [], {}
    for p in experiment['phases']:
        if p.get('failed'):
            continue
        phase = directory / f"phase_{p['index']:02d}"
        record, calls = phase_calls(phase)
        (phase / 'diagnostics').mkdir(exist_ok=True)
        within = 0.
        for call in calls:
            call_offsets[(p['index'], call['index'])] = dict(solver=p['offset_s'] + within, raw=p['raw_offset_s'] + sum(c['trace']['raw_s'] for c in calls if c['index'] < call['index']))
            within += call['trace']['solver_s']
            for begin in range(-1, call['trace']['count'], chunk_size):
                end = min(begin + chunk_size, call['trace']['count'])
                tasks.append((str(phase), call['index'], begin, end, str(phase / 'diagnostics' / f"phase_{p['index']:02d}_call_{call['index']:02d}_{begin + 1:06d}.h5"), cpus))
    metrics = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=initialize, initargs=(cpus,)) as pool:
        for i, rows in enumerate(pool.map(process_chunk, tasks)):
            metrics.extend(rows)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"{experiment['robot']}: shard {i + 1}/{len(tasks)} ({len(metrics)} snapshots)", flush=True)
    metrics.sort(key=lambda m: (m['phase'], m['call'], m['sample']))
    previous = None
    for m in metrics:
        off = call_offsets[(m['phase'], m['call'])]
        if m['kind'] == 2:  # the returned snapshot is copied after the native clock stopped; use the call's exit clock
            phase = directory / f"phase_{m['phase']:02d}"
            tr = read_json(phase / 'result.json')['calls'][m['call']]['trace']
            m['call_solver_s'], m['call_raw_s'] = tr['solver_s'], tr['raw_s']
        m['time_s'] = off['solver'] + m['call_solver_s']
        m['raw_time_s'] = off['raw'] + m['call_raw_s']
        m['step_s'] = 0. if previous is None or m['kind'] == 0 else m['time_s'] - previous['time_s']
        m['raw_step_s'] = 0. if previous is None or m['kind'] == 0 else m['raw_time_s'] - previous['raw_time_s']
        m['within_budget'] = m['time_s'] <= experiment['budget_s']
        m['passed_strict_1e3'] = bool(m['audit_valid'] and m['max_violation'] <= STRICT_GATE)
        m['passed_2e3'] = bool(m['audit_valid'] and m['max_violation'] <= LOOSE_GATE)
        previous = m
    best = float('inf')
    for m in metrics:
        if m['audit_valid'] and m['within_budget']:
            best = min(best, m['max_violation'])
        m['best_so_far_max'] = best
    fields = list(dict.fromkeys(k for m in metrics for k in m))
    with (directory / 'iterations.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(metrics)
    # The shards were written before the cumulative clocks existed: store the completed rows too.
    by_shard = {}
    for m in metrics:
        by_shard.setdefault((m['phase'], m['shard']), []).append(m)
    for (phase_index, shard), rows in by_shard.items():
        path = directory / f'phase_{phase_index:02d}' / shard   # shard is relative to the phase directory
        with h5py.File(path, 'r+') as f:
            f['metrics_json'][()] = json.dumps(sorted(rows, key=lambda m: m['shard_row']))
    valid = [m for m in metrics if m['audit_valid']]
    returned = [m for m in metrics if m['kind'] == 2]
    final = returned[-1] if returned else metrics[-1]
    best_row = min(valid, key=lambda m: m['max_violation']) if valid else None
    first_strict = next((m for m in metrics if m['passed_strict_1e3']), None)
    first_loose = next((m for m in metrics if m['passed_2e3']), None)
    exports = dict(final_solution=final, best_observed_solution=best_row, first_pass_1e3_solution=first_strict, first_pass_2e3_solution=first_loose)
    for name, m in exports.items():
        if m is None:
            continue
        phase = directory / f"phase_{m['phase']:02d}"
        ctx = context(phase, problem=False)
        export_candidate(phase, ctx['record']['calls'][m['call']], m['sample'], directory / f'{name}.npz', ctx)
    # LEAP-archive (HDF5) view of the final candidate through the existing exporter, when it round-trips.
    last_phase = directory / f"phase_{experiment['phases'][-1]['index']:02d}"
    try:
        from export_public import export
        if not (last_phase / 'trajectory.h5').exists():
            export(last_phase)
        shutil.copy2(last_phase / 'trajectory.h5', directory / 'final_solution.h5')
    except Exception as exc:  # noqa: BLE001
        (directory / 'final_solution.h5.error.txt').write_text(repr(exc) + '\n')
    # The catalog depends only on (robot, dt, hold): phases with the same settings share one.
    catalogs = {}
    for p in experiment['phases']:
        if p.get('failed'):
            continue
        phase = directory / f"phase_{p['index']:02d}"
        rec = read_json(phase / 'result.json'); key = (rec['robot'], rec['settings']['dt'], rec['settings'].get('hold', 'linear'))
        if (phase / 'constraint_catalog.json').exists():
            catalogs.setdefault(key, phase / 'constraint_catalog.json'); continue
        if key in catalogs:
            shutil.copy2(catalogs[key], phase / 'constraint_catalog.json')
        else:
            write_json(phase / 'constraint_catalog.json', catalog(phase)); catalogs[key] = phase / 'constraint_catalog.json'
    families = sorted({k.split(':', 1)[1] for k in fields if k.startswith(('node:', 'fine:'))})
    summary = dict(robot=experiment['robot'], protocol=experiment['protocol'], budget_s=experiment['budget_s'], strict_gate=STRICT_GATE, loose_gate=LOOSE_GATE,
                   solve_s=experiment['solve_s'], raw_s=experiment['raw_s'], capture_s=experiment['capture_s'], capture_percent=100 * experiment['capture_s'] / max(experiment['raw_s'], 1e-12),
                   termination=experiment.get('termination'), snapshots=len(metrics), accepted_updates=sum(1 for m in metrics if m['kind'] == 1),
                   native_calls=len(call_offsets), phases=len(experiment['phases']), invalid_audits=sum(not m['audit_valid'] for m in metrics),
                   final=final, final_passed_1e3=final['passed_strict_1e3'], final_passed_2e3=final['passed_2e3'],
                   best_observed=best_row, first_pass_1e3=first_strict, first_pass_2e3=first_loose,
                   first_pass_1e3_time_s=None if first_strict is None else first_strict['time_s'], first_pass_2e3_time_s=None if first_loose is None else first_loose['time_s'],
                   families=families, postprocess_s=time.perf_counter() - started, postprocess_workers=workers, postprocess_cpus=cpus,
                   exports={k: f'{k}.npz' for k, v in exports.items() if v is not None})
    write_json(directory / 'summary.json', summary)
    from report import robot_report
    robot_report(directory)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--chunk-size', type=int, default=16)
    a = ap.parse_args()
    postprocess(a.directory, a.workers, a.chunk_size)
