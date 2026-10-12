"""The along-the-way recorder: every FDDP candidate is archived at the callback, the
physical audit is NOT run per iteration (that is postprocess.py, offline).

At each native callback the recorder copies (xs, us) into a chunked HDF5 archive
(only when the candidate changed), reads the native scalars (cost, stopping metric,
step length, regularization, expected/actual decrease, gap feasibility) and the
clocks, and appends one JSON record. It runs the full physical audit inline only
(a) when the native restoration cost is small enough that a pass is possible at all
    (cost <= 1/2 * rows * 1e-6: every unit-weight row enters the cost as 1/2 viol^2,
    so a single node row above 1e-3 already exceeds this), rate limited so that the
    inline audits cost at most ~10 % of the solver time,
(b) every `periodic_s` seconds of solver time, for live monitoring (status.json),
(c) at the initial candidate and at every native call return.
A passing inline audit stops the solve at the next callback. solver_s excludes the
whole recorder (timed from callback entry to exit); the exact first-pass candidate is
recovered offline from the audits of every stored candidate.
"""
from pathlib import Path
import json, os, time, hashlib, resource
import h5py
import numpy as np
from physics import audit

SCHEMA = 'crocoddyl-terrain-progress-v1'
NATIVE_KEYS = ['cost', 'stop', 'stepLength', 'preg', 'dreg', 'dV', 'dVexp', 'ffeas', 'gfeas', 'hfeas', 'iter', 'isFeasible']


def atomic_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)


def finite(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class Recorder:
    def __init__(self, stage_dir, status_dir, built, start, solver_budget, wall_budget,
                 periodic_s=60., min_gap_s=2., flush_s=5., status_every=10):
        self.stage = Path(stage_dir); self.status_dir = Path(status_dir); self.built = built; self.start = start
        self.solver_budget = solver_budget; self.wall_budget = wall_budget
        self.periodic_s = periodic_s; self.min_gap_s = min_gap_s; self.flush_s = flush_s; self.status_every = status_every
        self.index = 0; self.stored = 0; self.round = 0
        self.solver_seconds = 0.; self.previous_solver_seconds = 0.; self.overhead = 0.; self.audit_seconds = 0.; self.audits = 0
        self.build_seconds = 0.; self.last_end = time.perf_counter(); self.last_flush = time.perf_counter()
        self.best = np.inf; self.best_index = -1; self.best_stored = -1; self.passed = False; self.reason = None; self.last_report = None
        self.last_audit_solver = -np.inf; self.last_audit_duration = 0.; self.last_periodic = 0.
        self.last_digest = None; self.last_stored_index = -1
        self.necessary_cost = .5 * built.posed_rows * 1e-6
        r = built.info.r; d = built.dimensions
        self.file = h5py.File(self.stage / 'iterations.h5', 'w', libver='latest')
        a = self.file.attrs
        a['schema'] = SCHEMA; a['robot'] = built.robot; a['mesh_json'] = json.dumps(built.mesh)
        a['physical_objective'] = 'none (feasibility restoration)'; a['hold'] = built.hold
        a['state_layout'] = 'x = [q; v], explicit floating base (nq == nv), joint order joint_names'
        a['control_layout'] = ('flow stage u = [a0, am, a1 (3 nv, acceleration samples at s = 0, 1/2, 1) | torque Bernstein '
                               f'coefficients ({d["count"]} x nt) | all-contact wrench Bernstein coefficients ({d["count"]} x nf)]; '
                               'event stages (initial / junction / impact) carry dummy controls (the initial stage: u[:nv] = free q(0))')
        for key in ['nx', 'nu', 'nv', 'nt', 'nf', 'count']: a[key] = d[key]
        a['joint_names'] = json.dumps(list(r.model.names)[1:])
        a['contacts'] = json.dumps([dict(name=c.name, dim=c.dim, frame_id=int(c.frame_id)) for c in r.contacts])
        a['posed_rows'] = built.posed_rows; a['necessary_cost_for_pass'] = self.necessary_cost
        a['timing_note'] = ('solver_s excludes the entire recorder (callback entry to exit); step_solver_s is its increment since the previous '
                            'record; wall_s / available_wall_s include preparation and recording; audit_s is inline audit time (also excluded from solver_s)')
        self.file.create_dataset('actuated_idx', data=np.asarray(r.act_idx))
        shape_x = np.array(built.xs).shape; shape_u = np.array(built.us).shape
        self.xs = self.file.create_dataset('xs', shape=(0, *shape_x), maxshape=(None, *shape_x), chunks=(1, *shape_x), compression='lzf', dtype='f8')
        self.us = self.file.create_dataset('us', shape=(0, *shape_u), maxshape=(None, *shape_u), chunks=(1, *shape_u), compression='lzf', dtype='f8')
        self.records = self.file.create_dataset('records_json', shape=(0,), maxshape=(None,), chunks=(256,), dtype=h5py.string_dtype('utf-8'))
        a['committed_iterations'] = 0; a['committed_records'] = 0; a['best_index'] = -1; a['best_stored_index'] = -1
        self.file.swmr_mode = True
        self.jsonl = open(self.stage / 'iterations.jsonl', 'a', buffering=1)

    def _store(self, xs, us):
        i = self.stored
        self.xs.resize(i + 1, axis=0); self.xs[i] = xs
        self.us.resize(i + 1, axis=0); self.us[i] = us
        self.stored = i + 1
        return i

    def _flush(self):
        self.file.flush()
        self.file.attrs.modify('committed_iterations', self.stored); self.file.attrs.modify('committed_records', self.index)
        self.file.flush(); self.last_flush = time.perf_counter()

    def _audit(self, xs, us, trigger):
        t0 = time.perf_counter()
        report = audit(self.built, list(xs), list(us))
        duration = time.perf_counter() - t0
        if not report.get('finite'): raise FloatingPointError('Nonfinite native candidate')
        violation = max(report['node_max'], report['fine_max'])
        self.audit_seconds += duration; self.audits += 1
        self.last_audit_solver = self.solver_seconds; self.last_audit_duration = duration; self.last_periodic = self.solver_seconds
        self.last_report = report; self.passed = bool(report['passed'])
        return dict(trigger=trigger, audit_s=duration, maximum_violation=violation, node_max=report['node_max'], fine_max=report['fine_max'],
                    node=report['node'], fine=report['fine'], worst_times=report['worst_times'],
                    torque_integral=report['torque_integral'], passed=report['passed'])

    def record(self, xs, us, solver=None, event='iteration', entered=None):
        entered = time.perf_counter() if entered is None else entered
        if event in ('iteration', 'call_return'): self.solver_seconds += entered - self.last_end
        xs = np.array([np.asarray(x, dtype=float) for x in xs]); us = np.array([np.asarray(u, dtype=float) for u in us])
        if not (np.isfinite(xs).all() and np.isfinite(us).all()): raise FloatingPointError('Nonfinite native candidate')
        digest = hashlib.sha256(xs.tobytes() + us.tobytes()).hexdigest(); changed = digest != self.last_digest
        metrics = {}
        if solver is not None:
            for key in NATIVE_KEYS:
                if hasattr(solver, key): metrics[key] = finite(getattr(solver, key))
        else:
            metrics['cost'] = float(self.built.problem.calc(list(xs), list(us)))
        if changed or self.last_stored_index < 0:
            stored_index = self._store(xs, us); self.last_digest = digest; self.last_stored_index = stored_index
        else:
            stored_index = self.last_stored_index
        row = dict(index=self.index, event=event, round=self.round, native_iter=metrics.get('iter'),
                   wall_s=entered - self.start, solver_s=self.solver_seconds, step_solver_s=self.solver_seconds - self.previous_solver_seconds,
                   stored_index=stored_index, trajectory_changed=changed, trajectory_sha256=digest,
                   restoration_cost=metrics.get('cost'), native=metrics, previous_logging_s=self.overhead, previous_audit_s=self.audit_seconds,
                   build_s=self.build_seconds, physical_objective=0.)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        row.update(cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime, peak_rss_bytes=usage.ru_maxrss * 1024, machine_load=list(os.getloadavg()))
        cost = metrics.get('cost')
        plausible = cost is not None and cost <= self.necessary_cost
        gap = self.solver_seconds - self.last_audit_solver
        trigger = None
        if event != 'iteration': trigger = event
        elif plausible and gap >= max(self.min_gap_s, 9. * self.last_audit_duration): trigger = 'gate'
        elif self.solver_seconds - self.last_periodic >= max(self.periodic_s, 10. * self.last_audit_duration): trigger = 'periodic'   # monitoring audits cost <= 10 % of the solver time
        if trigger is not None:
            row['audit'] = self._audit(xs, us, trigger)
            if row['audit']['maximum_violation'] < self.best:
                self.best = row['audit']['maximum_violation']; self.best_index = self.index; self.best_stored = stored_index
                self.file.attrs.modify('best_index', self.best_index); self.file.attrs.modify('best_stored_index', self.best_stored)
            self.last_status_audit = dict(index=self.index, **{k: row['audit'][k] for k in ['trigger', 'maximum_violation', 'node_max', 'fine_max', 'passed', 'audit_s']})
        else:
            row['audit'] = None
        row['plausible_pass'] = bool(plausible)
        row['best_audited_violation'] = None if not np.isfinite(self.best) else float(self.best)
        self.reason = 'physical_gate' if self.passed else ('solver_budget' if self.solver_seconds >= self.solver_budget else
                                                            ('wall_budget' if time.perf_counter() - self.start >= self.wall_budget else None))
        row['stop_reason'] = self.reason
        row['record_s'] = time.perf_counter() - entered
        row['available_wall_s'] = time.perf_counter() - self.start
        text = json.dumps(row, allow_nan=False)
        self.records.resize(self.index + 1, axis=0); self.records[self.index] = text
        self.jsonl.write(text + '\n')
        if self.reason or event != 'iteration' or time.perf_counter() - self.last_flush >= self.flush_s: self._flush()
        if self.reason or trigger is not None or self.index % self.status_every == 0:
            status = dict(index=self.index, stored=self.stored, round=self.round, event=event, native_iter=metrics.get('iter'),
                          restoration_cost=cost, native_stop=metrics.get('stop'), solver_s=self.solver_seconds, wall_s=time.perf_counter() - self.start,
                          logging_s=self.overhead, audit_s=self.audit_seconds, audits=self.audits, plausible_pass=bool(plausible),
                          last_audit=getattr(self, 'last_status_audit', None), best_audited_violation=row['best_audited_violation'],
                          best_index=self.best_index, passed=self.passed, stop_reason=self.reason, trajectory_file='iterations.h5', staged_at=str(self.stage))
            atomic_json(self.status_dir / 'status.json', status)
        if self.reason or trigger is not None or self.index % 50 == 0:
            worst = '' if not row['audit'] else f" audit[{trigger}] max={row['audit']['maximum_violation']:.3e} ({row['audit']['audit_s']:.1f}s)"
            print(f"record={self.index} round={self.round} iter={metrics.get('iter')} cost={cost} stop={metrics.get('stop')} "
                  f"solver={self.solver_seconds:.1f}s wall={time.perf_counter() - self.start:.1f}s{worst} reason={self.reason}", flush=True)
        self.previous_solver_seconds = self.solver_seconds
        self.index += 1
        self.overhead += time.perf_counter() - entered; self.last_end = time.perf_counter()
        return self.reason is not None

    def __call__(self, solver):
        entered = time.perf_counter()
        return self.record(solver.xs, solver.us, solver, entered=entered)

    def close(self):
        self._flush(); self.file.close(); self.jsonl.close()
