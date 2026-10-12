#!/usr/bin/env python3
"""Offline physical audit of EVERY stored candidate of one run (the along-the-way data).

    run_python.sh postprocess.py <robot dir> [--workers 12] [--fine-dt 0.01]

Reads iterations.h5 / iterations.jsonl / protocol.json, audits every stored candidate
with the independent judge (tw_verify.audit: node gate + global 0.01 s grid, every
physical family) in a pool of single-threaded workers (resumable: audits.jsonl is a
checkpoint), then writes
  audits.jsonl      one audit per stored candidate (stored_index, per-family node/fine maxima, ...)
  progress.jsonl    one row per recorded callback = the scalar record + its candidate's audit
  progress.csv      the same, flat (per-family node.<family> / fine.<family> columns)
  decoded.h5        q, v, a, u, lambda of every stored candidate (see README)
  summary.json      first pass, best candidate, final candidate, clocks, overheads
  progress.png      max violation vs solver time (and per family group)
The solve itself is never touched; rerunning only fills missing audits.
"""
from pathlib import Path
import argparse, csv, json, os, sys, time
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import h5py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_state = {}


def _init(robot, dt, hold, h5path):
    import tw_residuals
    info, mesh, *_ = tw_residuals.cold(robot, dt, hold)
    _state.update(info=info, mesh=mesh, path=h5path)


def _audit_chunk(indices, fine_dt):
    import tw_verify
    out = []
    with h5py.File(_state['path'], 'r') as f:
        for i in indices:
            xs = list(f['xs'][i]); us = list(f['us'][i])
            t0 = time.perf_counter(); rep = tw_verify.audit(_state['info'], _state['mesh'], xs, us, fine_dt)
            if not rep.get('finite'):
                out.append(dict(stored_index=int(i), finite=False, passed=False, maximum_violation=None)); continue
            out.append(dict(stored_index=int(i), finite=True, maximum_violation=max(rep['node_max'], rep['fine_max']), node_max=rep['node_max'], fine_max=rep['fine_max'],
                            node=rep['node'], fine=rep['fine'], worst_times=rep['worst_times'], torque_integral=rep['torque_integral'], passed=bool(rep['passed']),
                            audit_s=time.perf_counter() - t0, audit_version=rep['audit_version'], fine_dt=fine_dt))
    return out


def load_records(folder):
    return [json.loads(line) for line in (folder / 'iterations.jsonl').read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--fine-dt', type=float, default=.01)
    ap.add_argument('--chunk', type=int, default=4)
    ap.add_argument('--no-decode', action='store_true')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    folder = args.directory; start = time.perf_counter()
    protocol = json.loads((folder / 'protocol.json').read_text()); robot, dt, hold = protocol['robot'], protocol['settings']['dt'], protocol['hold']
    h5path = folder / 'iterations.h5'
    with h5py.File(h5path, 'r') as f:
        committed = int(f.attrs['committed_iterations']); mesh = json.loads(f.attrs['mesh_json'])
    records = load_records(folder)
    audits_path = folder / 'audits.jsonl'
    done = {}
    if audits_path.exists():
        for line in audits_path.read_text().splitlines():
            if line.strip(): a = json.loads(line); done[a['stored_index']] = a
    todo = [i for i in range(committed) if i not in done]
    print(f'{robot}: {committed} stored candidates, {len(records)} records, {len(todo)} audits to run on {args.workers} workers', flush=True)
    if todo:
        chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init, initargs=(robot, dt, hold, str(h5path))) as pool, audits_path.open('a') as out:
            finished = 0
            for result in pool.map(_audit_chunk, chunks, [args.fine_dt] * len(chunks)):
                for a in result:
                    done[a['stored_index']] = a; out.write(json.dumps(a, allow_nan=False) + '\n')
                out.flush(); finished += len(result)
                if finished % (args.chunk * args.workers * 5) < args.chunk or finished == len(todo):
                    print(f'  audited {finished}/{len(todo)} ({time.perf_counter() - start:.0f} s)', flush=True)
    # cross-check the inline audits of the solve against the offline ones (same judge)
    mismatch = 0.
    for rec in records:
        if rec.get('audit') and rec['stored_index'] in done and done[rec['stored_index']].get('maximum_violation') is not None:
            mismatch = max(mismatch, abs(rec['audit']['maximum_violation'] - done[rec['stored_index']]['maximum_violation']))
    # progress rows
    families = sorted({k for a in done.values() if a.get('finite') for g in ('node', 'fine') for k in a[g]})
    rows = []; best = np.inf; first_pass = None
    for rec in records:
        a = done.get(rec['stored_index'])
        row = dict(index=rec['index'], event=rec['event'], round=rec['round'], native_iter=rec.get('native_iter'), solver_s=rec['solver_s'], wall_s=rec['wall_s'],
                   available_wall_s=rec.get('available_wall_s'), step_solver_s=rec.get('step_solver_s'), stored_index=rec['stored_index'], trajectory_changed=rec['trajectory_changed'],
                   restoration_cost=rec.get('restoration_cost'), plausible_pass=rec.get('plausible_pass'), inline_audit=bool(rec.get('audit')), record_s=rec.get('record_s'))
        for key in ['stop', 'stepLength', 'preg', 'dreg', 'dV', 'dVexp', 'ffeas', 'isFeasible']: row['native_' + key] = (rec.get('native') or {}).get(key)
        if a is not None and a.get('finite'):
            row.update(maximum_violation=a['maximum_violation'], node_max=a['node_max'], fine_max=a['fine_max'], passed=a['passed'], torque_integral=a['torque_integral'])
            row['node'] = a['node']; row['fine'] = a['fine']; row['worst_times'] = a['worst_times']
            if a['maximum_violation'] < best: best = a['maximum_violation']
            if a['passed'] and first_pass is None: first_pass = row
        else:
            row.update(maximum_violation=None, node_max=None, fine_max=None, passed=None, torque_integral=None, node={}, fine={}, worst_times={})
        row['best_so_far'] = None if not np.isfinite(best) else float(best)
        rows.append(row)
    with (folder / 'progress.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False) + '\n')
    scalar = [k for k in rows[0] if k not in ('node', 'fine', 'worst_times')] if rows else []
    with (folder / 'progress.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(scalar + [f'node.{k}' for k in families] + [f'fine.{k}' for k in families])
        for row in rows: w.writerow([row[k] for k in scalar] + [row['node'].get(k) for k in families] + [row['fine'].get(k) for k in families])
    # decoded arrays
    if not args.no_decode:
        import tw_residuals, tw_verify
        info, *_ = tw_residuals.cold(robot, dt, hold); r = info.r
        with h5py.File(h5path, 'r') as f, h5py.File(folder / 'decoded.h5', 'w') as g:
            for key in f.attrs: g.attrs[key] = f.attrs[key]
            g.attrs['note'] = ('per stored candidate: q, v (K+1, nv) shooting states; a (K, 3, nv) acceleration samples at s = 0, 1/2, 1; '
                               'u (K, count, nt) torque Bernstein coefficients; lambda (K, count, nf) all-contact wrench Bernstein coefficients '
                               '(LOCAL_WORLD_ALIGNED, [fx fy fz (tx ty tz)] per contact in the contacts order); event stages carry dummy controls. '
                               'record_of_stored maps a stored candidate to the first record index that produced it; see README for dense reconstruction.')
            first = tw_verify.decode(r, f['xs'][0], f['us'][0]); n = committed
            sets = {k: g.create_dataset(k if k != 'lam' else 'lambda', shape=(n, *v.shape), chunks=(1, *v.shape), compression='lzf', dtype='f8') for k, v in first.items()}
            for i in range(n):
                dec = tw_verify.decode(r, f['xs'][i], f['us'][i])
                for k, v in dec.items(): sets[k][i] = v
            record_of = np.full(n, -1, dtype=np.int64)
            for rec in records:
                if record_of[rec['stored_index']] < 0: record_of[rec['stored_index']] = rec['index']
            g.create_dataset('record_of_stored', data=record_of)
            g.create_dataset('stage_t', data=np.array([s['t'] for s in mesh])); g.create_dataset('stage_h', data=np.array([s['h'] for s in mesh]))
            g.create_dataset('stage_kind', data=np.array([s['kind'] for s in mesh], dtype=h5py.string_dtype()))
            g.create_dataset('stage_mode', data=np.array([s['mode'] for s in mesh]))
    # summary
    result = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').exists() else None
    audited = [row for row in rows if row['maximum_violation'] is not None]
    best_row = min(audited, key=lambda row: row['maximum_violation']) if audited else None
    final_row = rows[-1] if rows else None
    iterations = [row for row in rows if row['event'] == 'iteration']
    summary = dict(robot=robot, records=len(rows), stored_candidates=committed, native_iterations=len(iterations),
                   accepted_steps=sum(1 for row in iterations if row['trajectory_changed']),
                   passed_final=bool(final_row['passed']) if final_row and final_row['passed'] is not None else None,
                   first_pass=None if first_pass is None else {k: first_pass[k] for k in ['index', 'stored_index', 'round', 'native_iter', 'solver_s', 'wall_s', 'available_wall_s', 'maximum_violation', 'node_max', 'fine_max']},
                   best={k: best_row[k] for k in ['index', 'stored_index', 'round', 'native_iter', 'solver_s', 'wall_s', 'maximum_violation', 'node_max', 'fine_max']} | dict(node=best_row['node'], fine=best_row['fine']) if best_row else None,
                   final={k: final_row[k] for k in ['index', 'stored_index', 'round', 'native_iter', 'solver_s', 'wall_s', 'maximum_violation', 'node_max', 'fine_max', 'passed']} | dict(node=final_row['node'], fine=final_row['fine']) if final_row else None,
                   seed=dict(maximum_violation=rows[0]['maximum_violation'], node_max=rows[0]['node_max'], fine_max=rows[0]['fine_max']) if rows else None,
                   solver_s=rows[-1]['solver_s'] if rows else None, stop_reason=result and result.get('stop_reason'), result_passed=result and result.get('passed'),
                   solve_wall_s=result and result.get('wall_s'), inline_logging_s=result and result.get('logging_s'), inline_audit_s=result and result.get('inline_audit_s'),
                   inline_vs_offline_audit_max_abs_diff=mismatch, offline_audit_wall_s=time.perf_counter() - start, fine_dt=args.fine_dt,
                   calls=result and result.get('calls'), machine=protocol.get('host'), cpus=protocol.get('cpus'))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if not args.no_plot:
        try:
            plot(folder, rows, robot)
        except Exception as exc:  # plotting is a convenience, never a failure of the data
            print('plot skipped:', exc, flush=True)
    print(json.dumps({k: summary[k] for k in ['robot', 'records', 'stored_candidates', 'passed_final', 'first_pass', 'stop_reason', 'solver_s', 'inline_vs_offline_audit_max_abs_diff']}), flush=True)


def plot(folder, rows, robot):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from tw_verify import FAMILY_GROUP
    audited = [row for row in rows if row['maximum_violation'] is not None and row['event'] == 'iteration']
    if not audited: return
    t = np.array([row['solver_s'] for row in audited]); v = np.array([row['maximum_violation'] for row in audited])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]; ax.semilogy(t, np.maximum(v, 1e-12), lw=.8, label='max violation (node, fine grid)')
    ax.semilogy(t, np.maximum([row['node_max'] for row in audited], 1e-12), lw=.6, alpha=.7, label='node gate')
    ax.semilogy(t, np.maximum([row['fine_max'] for row in audited], 1e-12), lw=.6, alpha=.7, label='fine grid dt 0.01')
    ax.axhline(1e-3, color='k', ls='--', lw=.8, label='criterion 1e-3'); ax.set_xlabel('solver time [s]'); ax.set_ylabel('violation (raw units)'); ax.set_title(f'{robot}: crocoddyl FDDP restoration'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]; groups = sorted(set(FAMILY_GROUP.values()))
    for group in groups:
        fams = [f for f, g in FAMILY_GROUP.items() if g == group]
        series = np.array([max([row['node'].get(f, 0.) for f in fams] + [row['fine'].get(f, 0.) for f in fams] + [0.]) for row in audited])
        if series.max() > 0: ax.semilogy(t, np.maximum(series, 1e-12), lw=.7, label=group)
    ax.axhline(1e-3, color='k', ls='--', lw=.8); ax.set_xlabel('solver time [s]'); ax.set_title('per constraint group'); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder / 'progress.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
