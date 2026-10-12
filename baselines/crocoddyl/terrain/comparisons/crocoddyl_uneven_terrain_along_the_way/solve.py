#!/usr/bin/env python3
"""One uneven-terrain robot: stock FDDP from the LEAP cold seed, every candidate archived.

    run_python.sh solve.py --robot go2 --output <new dir> --cpus 40,53,54,...  [--solver-seconds 14400]

Exit 0 = the physical gate passed (max(node, fine grid) <= 1e-3 on the inline audit),
2 = stopped without a pass (budget, native convergence to an infeasible restoration
minimum, native stop), other = error (failure.json). The trajectory archive is written
to a LOCAL staging directory during the solve and moved into --output at the end
(staged.json in --output points to it meanwhile).
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys, time, traceback
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tw_inputs import ROOT, ROBOTS, ALL_ROBOTS

DEFAULT_STAGE = Path(os.environ.get('CROCODDYL_STAGE_ROOT', HERE.parents[2] / 'stage'))


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot', choices=ALL_ROBOTS, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cpus', required=True, help='1 to 12 comma-separated distinct physical CPU ids')
    ap.add_argument('--stage-root', type=Path, default=DEFAULT_STAGE, help='local directory for the per-iteration archive during the solve')
    ap.add_argument('--wall-seconds', type=float, default=86400.)
    ap.add_argument('--solver-seconds', type=float, default=14400.)
    ap.add_argument('--iterations', type=int, default=10000)
    ap.add_argument('--rounds', type=int, default=40)
    ap.add_argument('--dt', type=float, default=.01)
    ap.add_argument('--reg-init', type=float, default=1e-3)
    ap.add_argument('--th-stop', type=float, default=1e-12)
    ap.add_argument('--hold', choices=['linear', 'quadratic'], default='quadratic')
    ap.add_argument('--periodic-audit-seconds', type=float, default=60., help='solver seconds between monitoring audits')
    ap.add_argument('--prepare-only', action='store_true', help='build, archive and audit the cold seed; never instantiate an optimizer')
    args = ap.parse_args(argv)
    cpus = [int(v) for v in args.cpus.split(',')]
    if not 1 <= len(cpus) <= 12 or len(set(cpus)) != len(cpus): ap.error('Use at most 12 distinct physical CPUs')
    if min(args.wall_seconds, args.solver_seconds, args.dt, args.reg_init, args.iterations, args.rounds) <= 0: ap.error('Positive budgets/settings required')
    if args.output.exists(): ap.error('Output must be a new directory')
    if not set(cpus) <= os.sched_getaffinity(0): ap.error('Requested CPUs are outside the available affinity')
    physical = []
    for cpu in cpus:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.append(((topology / 'physical_package_id').read_text(), (topology / 'core_id').read_text()))
    if len(set(physical)) != len(cpus): ap.error('CPU list repeats a physical core through SMT')
    os.sched_setaffinity(0, cpus); os.environ['OMP_NUM_THREADS'] = str(len(cpus)); args.output.mkdir(parents=True)
    stage = args.stage_root / f'{args.robot}_{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}'; stage.mkdir(parents=True)
    start = time.perf_counter()
    import numpy as np, crocoddyl, pinocchio, aligator
    from physics import build, audit, STOCK_COMPONENTS, PUBLIC_MODEL_ADAPTERS
    from trace import Recorder, atomic_json, SCHEMA
    from _impact_native import run
    recorder = None; record = {}; xs = us = None; built = None
    try:
        t = time.perf_counter(); built = build(args.robot, args.dt, args.hold); build_seconds = time.perf_counter() - t
        built.problem.nthreads = len(cpus)
        physical_inputs = [built.info.path, built.info.r.yaml_path, built.info.r.urdf_path, built.info.seed.path, built.generated_library]
        record = dict(schema=SCHEMA, family='terrain', robot=args.robot, settings=vars(args) | {'output': str(args.output), 'stage_root': str(args.stage_root)},
                      command=[sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
                      stock_components=STOCK_COMPONENTS, public_model_adapters=PUBLIC_MODEL_ADAPTERS,
                      versions=dict(crocoddyl=crocoddyl.__version__, pinocchio=pinocchio.__version__, model_api=aligator.__version__, numpy=np.__version__),
                      crocoddyl_module=crocoddyl.__file__,
                      crocoddyl_library_sha256={p: digest(p) for p in sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libcrocoddyl.so' in line))},
                      crocoddyl_release_commit='174c6082f0aef4dca80fa767f47fe025d6c4fef1',
                      solver='unmodified installed crocoddyl.SolverFDDP', model_dependency='public residual/model adapters only; no other optimizer is instantiated',
                      objective='feasibility restoration; physical objective identically zero',
                      geometry=built.geometry, self_collision=built.self_collision,
                      input_sha256={str(p.relative_to(ROOT)): digest(p) for p in physical_inputs},
                      seed_sha256=built.info.seed.sha256, cpus=cpus, nthreads=int(built.problem.nthreads),
                      host=os.uname().nodename, machine_load=list(os.getloadavg()), start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      mesh=built.mesh, stages=built.stages, dimensions=built.dimensions, posed_rows=built.posed_rows,
                      hold=built.hold, build_s=build_seconds, symbolic_digest=built.symbolic_digest,
                      stance_acceleration=True, touchdown=dict(zeroland=built.info.spec.zeroland, zerolandxy=built.info.spec.zerolandxy, landramp=built.info.spec.landramp, landvz=built.info.spec.landvz,
                                                               landcont=built.info.spec.landcont),
                      windows=dict(approachfrac=built.info.spec.approachfrac, liftfrac=built.info.spec.liftfrac, modes=len(built.info.spec.modes),
                                   window_modes=sum(1 for m in built.info.spec.modes if m.window)),
                      boxes=dict(att_limits={str(k): v for k, v in built.info.spec.att_limits.items()}, posture=[[p[0], int(p[1]), p[2], p[3]] for p in built.info.spec.posture],
                                 basev_limits={str(k): v for k, v in built.info.spec.basev_limits.items()}, ulim_joint=built.info.spec.ulim_joint),
                      staged_at=str(stage), calls=[])
        source_files = list(HERE.glob('*.py')) + list(HERE.glob('*.md')) + list(HERE.glob('*.sh'))
        for folder in ['common', 'aligator_bench', 'aligator_unified_fence/scripts', 'aligator_unified_terrain/scripts', 'crocoddyl_impact_cone', 'crocoddyl_unified']:
            source_files.extend(p for p in (ROOT / 'comparisons' / folder).iterdir() if p.suffix in ('.py', '.cpp', '.hpp', '.so', '.sh', '.patch'))
        record['source_sha256'] = {}
        for p in source_files:
            relative = p.relative_to(ROOT / 'comparisons'); destination = args.output / 'executed_source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, destination); record['source_sha256'][str(relative)] = digest(p)
        for p in physical_inputs:
            destination = args.output / 'inputs' / p.relative_to(ROOT); destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, destination)
        atomic_json(args.output / 'protocol.json', record)
        atomic_json(args.output / 'staged.json', dict(stage=str(stage), note='iterations.h5 / iterations.jsonl live here until the solve ends'))
        xs, us = built.xs, built.us
        np.savez_compressed(args.output / 'initial.npz', xs=np.array(xs), us=np.array(us), mesh_json=json.dumps(built.mesh))
        recorder = Recorder(stage, args.output, built, start, args.solver_seconds, args.wall_seconds, periodic_s=args.periodic_audit_seconds)
        recorder.build_seconds = build_seconds
        recorder.record(xs, us, event='initial')
        if args.prepare_only:
            record.update(prepared=True, optimization_started=False, initial_audit=recorder.last_report)
            atomic_json(args.output / 'prepared.json', record)
            print(json.dumps(dict(robot=args.robot, prepared=True, optimization_started=False, seed_node_max=recorder.last_report['node_max'], seed_fine_max=recorder.last_report['fine_max'])), flush=True)
            return 0
        for outer in range(args.rounds):
            if recorder.reason: break
            recorder.round = outer
            solver = crocoddyl.SolverFDDP(built.problem)
            solver.th_stop = args.th_stop
            if hasattr(solver, 'th_minImprove'): solver.th_minImprove = 1e-16
            if hasattr(solver, 'reg_min'): solver.reg_min = 1e-9
            recorder.last_end = time.perf_counter(); before = recorder.solver_seconds
            converged, stopped = run(solver, xs, us, args.iterations, args.reg_init, recorder, len(cpus))
            returned = time.perf_counter()
            xs = [np.array(x).copy() for x in solver.xs]; us = [np.array(u).copy() for u in solver.us]
            # The returned candidate is archived (and audited) even when no callback fired for it.
            recorder.record(xs, us, solver, event='call_return', entered=returned)
            call = dict(round=outer, native_converged=bool(converged), callback_stopped=bool(stopped), native_iterations=int(solver.iter),
                        solver_s=recorder.solver_seconds - before, cost=float(solver.cost), native_stop=float(solver.stop),
                        audit=recorder.last_report and dict(maximum_violation=max(recorder.last_report['node_max'], recorder.last_report['fine_max']), passed=recorder.last_report['passed']))
            record['calls'].append(call); atomic_json(args.output / 'calls.json', record['calls'])
            print(json.dumps(call), flush=True)
            if recorder.reason: break
            # Native convergence without a passing audit ends the solve.
            if converged and not recorder.passed:
                recorder.reason = 'native_converged_infeasible'; break
            if solver.iter + 1 < args.iterations:
                recorder.reason = 'native_stopped_infeasible'; break
        final = audit(built, xs, us)
        np.savez_compressed(args.output / 'solution.npz', xs=np.array(xs), us=np.array(us), mesh_json=json.dumps(built.mesh))
        record.update(passed=final['passed'], audit=final, stop_reason=recorder.reason or 'iteration_budget', solver_s=recorder.solver_seconds,
                      wall_s=time.perf_counter() - start, logging_s=recorder.overhead, inline_audit_s=recorder.audit_seconds, inline_audits=recorder.audits,
                      records=recorder.index, stored_candidates=recorder.stored, best_audited_violation=None if not np.isfinite(recorder.best) else float(recorder.best),
                      best_audited_index=recorder.best_index, solution_sha256=digest(args.output / 'solution.npz'), end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic_json(args.output / 'result.json', record)
        print(json.dumps({k: record[k] for k in ['family', 'robot', 'passed', 'stop_reason', 'solver_s', 'wall_s', 'logging_s', 'inline_audit_s', 'records', 'stored_candidates']}), flush=True)
        return 0 if final['passed'] else 2
    except BaseException:
        record.update(passed=False, error=traceback.format_exc(), wall_s=time.perf_counter() - start)
        atomic_json(args.output / 'failure.json', record)
        raise
    finally:
        if recorder is not None:
            recorder.close()
            for name in ['iterations.h5', 'iterations.jsonl']:
                if (stage / name).exists(): shutil.move(str(stage / name), str(args.output / name))
            try: stage.rmdir()
            except OSError: pass
            if (args.output / 'staged.json').exists(): (args.output / 'staged.json').unlink()
            print(f'archive moved to {args.output}', flush=True)


if __name__ == '__main__': raise SystemExit(main())
