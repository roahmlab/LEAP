#!/usr/bin/env python3
"""Solve one uneven-terrain instance with stock ALIGATOR FDDP from LEAP's cold seed,
capturing every accepted iterate (states + controls + native scalars) in RAM, on 12
pinned physical cores; then audit every iterate offline (postprocess.py).

Protocols (recorded in experiment.json):
  recorded_lineage  go2 / g1_29dof / talos: the recipe selected in ../reports/selected.json,
                    replayed from the cold seed with its saved native update counts
                    (../scripts/replay_selected.py); if the replayed lineage does not pass the
                    physical gate, further native calls of the last phase's settings continue
                    from the lineage's own candidate until the gate or the budget.
  cold_ladder       hopper2d_foot (no recorded recipe): the driver's baseline cold start
                    (linear holds, dt 0.01, 4 x 300 s), then -- only if still failing -- the
                    two continuation devices the other robots' recipes used: exact degree
                    elevation to quadratic holds, then exact 0.005 s mesh refinement.
Every phase is one ../scripts/solve_public.py invocation (--store-iterations); nothing about the
solver is modified. Solve time = native wall time minus the measured in-solve snapshot copy;
setup, transfers, audits and file writes are outside it.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from atw_common import (HERE, SCRIPTS, TERRAIN, ROOT, PY, DEFAULT_ROOT, LEAP_BUDGET, ROBOTS, STRICT_GATE,
                    bootstrap, digest, write_json, read_json, cpu_list, machine_record, phase_calls, driver_robot)

CALL_SECONDS = 300.
ROUNDS = 4


def lineage_commands(robot, out, threads):
    selected = read_json(TERRAIN / 'reports/selected.json')
    if robot not in selected:
        return None
    text = subprocess.check_output([str(PY), str(SCRIPTS / 'replay_selected.py'), '--robot', robot, '--out', str(out),
                                    '--core-offset', '0', '--threads', str(threads), '--dry-run'], cwd=ROOT, text=True)
    return [c + ['--store-iterations'] for c in json.loads(text)]


def base_command(robot, out, threads):
    return [str(PY), str(SCRIPTS / 'solve_public.py'), '--robot', robot, '--out', str(out), '--threads', str(threads),
            '--core-offset', '0', '--store-iterations']


def ladder(robot, out, threads):
    """The cold protocol of a robot without a recorded recipe, as (name, settings) rungs."""
    return [('cold_linear', dict(dt=.01, hold='linear')),
            ('quadratic_elevation', dict(dt=.01, hold='quadratic')),
            ('mesh_refinement_0.005', dict(dt=.005, hold='quadratic'))]


def set_option(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(value)
    else:
        cmd += [flag, str(value)]
    return cmd


def drop_option(cmd, flag, has_value=True):
    cmd = list(cmd)
    while flag in cmd:
        i = cmd.index(flag)
        del cmd[i:i + 1 + int(has_value)]
    return cmd


def run_phase(cmd, out, cpus, log):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    full = ['taskset', '-c', ','.join(map(str, cpus))] + cmd
    with open(log, 'w') as f:
        f.write('# ' + ' '.join(full) + '\n')
        f.flush()
        code = subprocess.run(full, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return code


def phase_summary(phase):
    record, calls = phase_calls(phase)
    solve = sum(c['trace']['solver_s'] for c in calls)
    raw = sum(c['trace']['raw_s'] for c in calls)
    return dict(solve_s=solve, raw_s=raw, capture_s=raw - solve, calls=len(calls), all_calls=len(record['calls']),
                errors=[c['error'] for c in record['calls'] if 'error' in c], passed=bool(record.get('passed')),
                selected_call=record.get('selected_call'), settings=record['settings'],
                final_audit=dict(node_max=record['audit']['node_max'], fine_max=record['audit']['fine_max']) if record.get('audit', {}).get('finite', True) and 'node_max' in record.get('audit', {}) else None,
                process_s=record.get('process_s'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot', choices=ROBOTS, required=True)
    ap.add_argument('--out', type=Path, help='New run directory (default <data root>/runs/<robot>)')
    ap.add_argument('--cpus', required=True, help='Logical CPU list of 12 distinct physical cores, e.g. 16-27')
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--budget', type=float, default=14400., help='Cap on cumulative solve seconds (stop early at the gate)')
    ap.add_argument('--smoke', action='store_true', help='Three native updates of the first phase only; validation data, not results')
    ap.add_argument('--postprocess', action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    cpus = cpu_list(args.cpus)
    from atw_common import physical_core_ids
    if len(cpus) != args.threads or len(set(physical_core_ids(cpus))) != args.threads:
        ap.error(f'--cpus must name exactly {args.threads} distinct physical cores')
    out = (args.out or DEFAULT_ROOT / 'runs' / args.robot).resolve()
    if out.exists():
        ap.error(f'Use a new output directory: {out}')
    out.mkdir(parents=True)
    bootstrap()
    from atw_common import reference_inputs
    started = time.time()
    driver = driver_robot(args.robot)     # the terrain driver's robot key (problem yaml + seed archive)
    lineage = lineage_commands(driver, out, args.threads)
    protocol = 'recorded_lineage' if lineage else 'cold_ladder'
    record = dict(robot=args.robot, driver_robot=driver, protocol=protocol, budget_s=args.budget, threads=args.threads, cpus=cpus,
                  smoke=args.smoke, strict_gate=STRICT_GATE, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)),
                  command=[sys.executable, *sys.argv], call_seconds=CALL_SECONDS, rounds=ROUNDS, phases=[],
                  clock='native FDDP wall time minus measured in-solve snapshot copying; setup, transfers, audits and I/O excluded')
    inputs = reference_inputs(args.robot)
    ref = out / 'reference_inputs'
    ref.mkdir()
    record['inputs'] = {}
    for name, path in inputs.items():
        shutil.copy2(path, ref / path.name)
        record['inputs'][name] = dict(path=str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path), sha256=digest(path))
    leap_seed = next(iter(sorted((LEAP_BUDGET / args.robot).glob('seed_terrain_*.h5'))), LEAP_BUDGET / args.robot / 'missing.h5')
    if leap_seed.exists():
        record['leap_budget_seed'] = dict(path=str(leap_seed), sha256=digest(leap_seed), identical_to_seed=digest(leap_seed) == record['inputs']['seed']['sha256'])
    record['machine_start'] = machine_record(cpus)
    snap = out / 'source_snapshot'
    shutil.copytree(HERE, snap / 'terrain_benchmark_along_the_way', ignore=shutil.ignore_patterns('__pycache__', '*.tmp'))
    shutil.copytree(SCRIPTS, snap / 'scripts', ignore=shutil.ignore_patterns('__pycache__', 'build', '*.so'))
    shutil.copytree(TERRAIN / 'along_the_way', snap / 'along_the_way', ignore=shutil.ignore_patterns('__pycache__', 'build', '*.so'))
    record['native_extensions'] = {str(p.relative_to(ROOT)): digest(p) for p in list((TERRAIN / 'along_the_way').glob('*.so')) + list(SCRIPTS.glob('*.so')) + list((ROOT / 'comparisons/aligator_unified_fence/scripts').glob('_public*.so'))}
    write_json(out / 'experiment.json', record)

    elapsed = raw = 0.
    passed = False
    index = 0

    def execute(cmd, kind, note=''):
        nonlocal elapsed, raw, passed, index
        phase = out / f'phase_{index:02d}'
        cmd = set_option(cmd, '--out', phase)
        if args.smoke:
            cmd = set_option(set_option(set_option(cmd, '--iterations', 3), '--rounds', 1), '--call-seconds', 60)
        t0 = time.time()
        code = run_phase(cmd, phase, cpus, out / f'phase_{index:02d}.log')
        entry = dict(index=index, kind=kind, note=note, command=cmd, exit_code=code, offset_s=elapsed, raw_offset_s=raw,
                     wall_start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0)), process_s=time.time() - t0,
                     loadavg_1_5_15=os.getloadavg())
        if code not in (0, 2) or not (phase / 'result.json').exists():
            entry['failed'] = True
            record['phases'].append(entry)
            write_json(out / 'experiment.json', record)
            raise RuntimeError(f'{args.robot}: phase {index} failed with exit {code}; see {out}/phase_{index:02d}.log')
        entry.update(phase_summary(phase))
        elapsed += entry['solve_s']
        raw += entry['raw_s']
        passed = entry['passed']
        record['phases'].append(entry)
        record.update(solve_s=elapsed, raw_s=raw, capture_s=raw - elapsed, passed=passed)
        write_json(out / 'experiment.json', record)
        print(f"{args.robot}: phase {index} ({kind}) done: {entry['calls']} traced calls, solve {entry['solve_s']:.1f} s, "
              f"cumulative {elapsed:.1f}/{args.budget:.0f} s, passed={passed}, exit {code}", flush=True)
        index += 1
        return phase

    def budgeted(cmd):
        remaining = args.budget - elapsed
        rounds = max(1, min(ROUNDS, int(-(-remaining // CALL_SECONDS))))
        return set_option(set_option(cmd, '--rounds', rounds), '--call-seconds', min(CALL_SECONDS, remaining))

    try:
        if lineage:
            for cmd in (lineage[:1] if args.smoke else lineage):
                if elapsed >= args.budget:
                    break
                execute(cmd, 'lineage', 'recorded recipe phase (saved native update count)')
                if passed and cmd is lineage[-1]:
                    break
            last = out / f'phase_{index - 1:02d}'
            while not passed and elapsed < args.budget and not args.smoke:
                cmd = list(record['phases'][-1]['command'])
                cmd = drop_option(cmd, '--cost-threshold')
                cmd = set_option(cmd, '--iterations', 10000)
                cmd = set_option(cmd, '--resume', last / 'solution.npz')
                cmd = budgeted(cmd)
                last = execute(cmd, 'continuation', 'last lineage settings continued from the lineage candidate')
        else:
            previous = None
            rungs = ladder(driver, out, args.threads)
            for name, settings in (rungs[:1] if args.smoke else rungs):
                if passed or elapsed >= args.budget:
                    break
                cmd = base_command(driver, out, args.threads) + ['--dt', str(settings['dt']), '--hold', settings['hold']]
                if previous is not None:
                    cmd += ['--resume', str(previous / 'solution.npz')]
                cmd = budgeted(cmd)
                previous = execute(cmd, f'ladder:{name}', 'cold baseline' if previous is None else 'continuation of the previous rung candidate')
            while not passed and elapsed < args.budget and not args.smoke:
                cmd = budgeted(set_option(list(record['phases'][-1]['command']), '--resume', previous / 'solution.npz'))
                previous = execute(cmd, f'ladder:{rungs[-1][0]}:repeat', 'last rung repeated from its own candidate')
        record['termination'] = 'physical_gate' if passed else ('budget' if elapsed >= args.budget else 'protocol_complete')
    except RuntimeError as exc:
        record['termination'] = 'error'
        record['error'] = str(exc)
        print(exc, file=sys.stderr, flush=True)
    record['machine_end'] = dict(loadavg_1_5_15=os.getloadavg())
    record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    record['process_s'] = time.time() - started
    record['completed'] = record['termination'] != 'error'
    write_json(out / 'experiment.json', record)
    if args.postprocess and record['completed']:
        subprocess.run(['taskset', '-c', ','.join(map(str, cpus)), str(PY), str(HERE / 'postprocess.py'), str(out), '--workers', str(args.threads)], cwd=ROOT, check=True)
    return 0 if record['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
