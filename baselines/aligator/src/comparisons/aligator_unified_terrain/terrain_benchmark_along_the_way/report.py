#!/usr/bin/env python3
"""Per-robot plots + REPORT.md, and the aggregate RESULTS.md / results.csv / combined plots."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from atw_common import HERE, DEFAULT_ROOT, ROBOTS, STRICT_GATE, LOOSE_GATE, write_json, read_json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows_of(directory):
    with (Path(directory) / 'iterations.csv').open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ('shard', 'audit_error', 'optimality_error'):
                continue
            if v in ('True', 'False'):
                r[k] = v == 'True'
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def clip(values):
    return np.maximum(np.array(values, dtype=float), 1e-16)


def robot_report(directory):
    directory = Path(directory)
    rows = rows_of(directory)
    experiment = read_json(directory / 'experiment.json')
    summary = read_json(directory / 'summary.json')
    valid = [r for r in rows if r['audit_valid']]
    t = [r['time_s'] for r in valid]
    boundaries = [p['offset_s'] for p in experiment['phases'][1:]]
    # 1. maximum violation vs solve time
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy(t, clip([r['node_max'] for r in valid]), lw=.9, label='nodes / events')
    ax.semilogy(t, clip([r['fine_max'] for r in valid]), lw=.9, label='global 0.01 s grid')
    ax.semilogy(t, clip([r['max_violation'] for r in valid]), lw=1.3, color='k', label='max of both')
    ax.axhline(STRICT_GATE, color='r', ls='--', lw=.8, label='gate 1e-3')
    for b in boundaries:
        ax.axvline(b, color='gray', ls=':', lw=.7)
    ax.set(xlabel='cumulative ALIGATOR solve time excluding snapshot copies (s)', ylabel='max physical constraint violation', title=f"{experiment['robot']}: {experiment['protocol']}")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(directory / 'max_violation_time.png', dpi=160); fig.savefig(directory / 'max_violation_time.pdf'); plt.close(fig)
    # 2. per family (max of node and fine), the eight worst at the end
    fams = summary['families']
    last = valid[-1]
    worst = sorted(fams, key=lambda f: -max(last.get(f'node:{f}', 0) or 0, last.get(f'fine:{f}', 0) or 0))[:8]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for f in worst:
        ax.semilogy(t, clip([max(r.get(f'node:{f}', 0) or 0, r.get(f'fine:{f}', 0) or 0) for r in valid]), lw=.9, label=f)
    ax.axhline(STRICT_GATE, color='r', ls='--', lw=.8)
    for b in boundaries:
        ax.axvline(b, color='gray', ls=':', lw=.7)
    ax.set(xlabel='cumulative solve time (s)', ylabel='max violation per family (node or fine)', title=f"{experiment['robot']}: worst families at the end")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(directory / 'families_time.png', dpi=160); plt.close(fig)
    # 3. optimality of the restoration objective + native scalars
    opt = [r for r in valid if r['optimality_valid']]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogy([r['time_s'] for r in opt], clip([r['restoration_objective'] for r in opt]), lw=.9, label='restoration objective (call\'s own weights)')
    ax.semilogy([r['time_s'] for r in opt], clip([r['restoration_stationarity_inf'] for r in opt]), lw=.9, label='||control stationarity||_inf (adjoint)')
    ax.semilogy([r['time_s'] for r in opt], clip([r['shooting_defect_inf'] for r in opt]), lw=.9, label='||shooting defect||_inf')
    for b in boundaries:
        ax.axvline(b, color='gray', ls=':', lw=.7)
    ax.set(xlabel='cumulative solve time (s)', title='first-order optimality (recomputed on the saved primal)'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    ax = axes[1]
    nat = [r for r in valid if r['kind'] == 1]
    ax.semilogy([r['time_s'] for r in nat], clip([r['native_cost'] for r in nat]), lw=.9, label='native traj_cost (post-update)')
    ax.semilogy([r['time_s'] for r in nat], clip([r['native_primal_infeasibility'] for r in nat]), lw=.9, label='native prim_infeas (pre-update)')
    ax.semilogy([r['time_s'] for r in nat], clip([r['native_dual_infeasibility'] for r in nat]), lw=.9, label='native dual_infeas (pre-update)')
    for b in boundaries:
        ax.axvline(b, color='gray', ls=':', lw=.7)
    ax.set(xlabel='cumulative solve time (s)', title='FDDP native scalars at the callbacks'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(directory / 'optimality_time.png', dpi=160); plt.close(fig)
    # 4. step wall time
    steps = [r for r in valid if r['kind'] == 1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot([r['time_s'] for r in steps], [r['step_s'] for r in steps], lw=.8)
    ax.set(xlabel='cumulative solve time (s)', ylabel='seconds per accepted FDDP update', title=f"{experiment['robot']}: per-step wall time (12 threads)")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(directory / 'step_time.png', dpi=160); plt.close(fig)
    fin = summary['final']
    lines = [f"# {experiment['robot']} -- ALIGATOR uneven terrain, iterates along the way", '',
             f"Protocol `{experiment['protocol']}`; termination `{experiment.get('termination')}`; budget {experiment['budget_s']:.0f} s.",
             f"Solve time {summary['solve_s']:.2f} s (raw native wall {summary['raw_s']:.2f} s, snapshot copies {summary['capture_s']:.3f} s = {summary['capture_percent']:.3f} %), "
             f"{summary['native_calls']} native calls in {summary['phases']} phases, {summary['accepted_updates']} accepted updates, {summary['snapshots']} audited snapshots.", '',
             f"Final returned candidate: max violation {fin['max_violation']:.6e} (nodes {fin['node_max']:.6e}, 0.01 s grid {fin['fine_max']:.6e}); gate 1e-3 {'PASSED' if fin['passed_strict_1e3'] else 'NOT met'}; 2e-3 {'passed' if fin['passed_2e3'] else 'not met'}.",
             f"First iterate <= 1e-3: {'none' if summary['first_pass_1e3'] is None else f'{summary['first_pass_1e3_time_s']:.2f} s (phase {int(summary['first_pass_1e3']['phase'])}, call {int(summary['first_pass_1e3']['call'])}, snapshot {int(summary['first_pass_1e3']['sample'])})'}; "
             f"first <= 2e-3: {'none' if summary['first_pass_2e3'] is None else f'{summary['first_pass_2e3_time_s']:.2f} s'}; best observed {summary['best_observed']['max_violation']:.6e} at {summary['best_observed']['time_s']:.2f} s.", '',
             '| phase | kind | calls | updates | solve s | cumulative s | node max | fine max | passed |', '|---|---|--:|--:|--:|--:|--:|--:|:--:|']
    for p in experiment['phases']:
        upd = sum(1 for r in rows if r['phase'] == p['index'] and r['kind'] == 1)
        fa = p.get('final_audit') or {}
        lines.append(f"| {p['index']} | {p['kind']} | {p.get('calls', 0)} | {upd} | {p.get('solve_s', 0):.2f} | {p['offset_s'] + p.get('solve_s', 0):.2f} | {fa.get('node_max', float('nan')):.3e} | {fa.get('fine_max', float('nan')):.3e} | {p.get('passed')} |")
    lines += ['', '![max violation](max_violation_time.png)', '', '![families](families_time.png)', '', '![optimality](optimality_time.png)', '', '![step time](step_time.png)', '',
              'Worst families of the final candidate (max of node / fine):', '']
    for f in worst:
        lines.append(f"- `{f}`: {max(last.get(f'node:{f}', 0) or 0, last.get(f'fine:{f}', 0) or 0):.3e}")
    lines += ['', 'Data: `iterations.csv` (one row per snapshot), `phase_*/diagnostics/*.h5` (trajectories + every component violation), `summary.json`, `*_solution.npz`. Layout in the root README.md.']
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def aggregate(root):
    root = Path(root)
    runs = root / 'runs'
    records, lines = [], ['# ALIGATOR on the uneven terrain benchmark: results with every iterate recorded', '',
                          'Stock ALIGATOR 0.19.1 `SolverFDDP` on the public restoration formulation of `comparisons/aligator_unified_terrain` '
                          '(the same constraint families as LEAP unified, see FORMULATION_EQUIVALENCE.md), started from LEAP\'s cold seed archive (bitwise the LEAP budget experiment\'s seed), '
                          '12 pinned physical cores per solve. Solve time = native wall time minus the measured in-solve snapshot copy; setup, transfers, audits and I/O are excluded.', '',
                          f'Gate: max(node/event violation, global dt = 0.01 s grid violation) <= {STRICT_GATE:g} (LEAP\'s criterion); the 2e-3 reading is reported alongside.', '',
                          '| robot | protocol | phases / calls / updates | solve time (s) | final max violation (node / fine) | first <= 1e-3 (s) | first <= 2e-3 (s) | best observed | outcome |',
                          '|---|---|---|--:|---|--:|--:|--:|---|']
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig2, ax2 = plt.subplots(figsize=(8.5, 5))
    for robot in ROBOTS:
        d = runs / robot
        if not (d / 'summary.json').exists():
            lines.append(f'| {robot} | — | — | — | — | — | — | — | pending |')
            continue
        s = read_json(d / 'summary.json'); fin = s['final']
        outcome = 'PASS' if fin['passed_strict_1e3'] else ('PASS at 2e-3' if fin['passed_2e3'] else ('pass observed earlier' if s['first_pass_1e3'] else 'FAIL'))
        f1 = '—' if s['first_pass_1e3_time_s'] is None else f"{s['first_pass_1e3_time_s']:.1f}"
        f2 = '—' if s['first_pass_2e3_time_s'] is None else f"{s['first_pass_2e3_time_s']:.1f}"
        lines.append(f"| [{robot}](runs/{robot}/REPORT.md) | {s['protocol']} | {s['phases']} / {s['native_calls']} / {s['accepted_updates']} | {s['solve_s']:.1f} | {fin['max_violation']:.3e} ({fin['node_max']:.2e} / {fin['fine_max']:.2e}) | {f1} | {f2} | {s['best_observed']['max_violation']:.3e} | {outcome} |")
        records.append(dict(robot=robot, protocol=s['protocol'], phases=s['phases'], native_calls=s['native_calls'], accepted_updates=s['accepted_updates'], solve_s=s['solve_s'], raw_s=s['raw_s'], capture_s=s['capture_s'],
                            final_max_violation=fin['max_violation'], final_node_max=fin['node_max'], final_fine_max=fin['fine_max'], final_passed_1e3=fin['passed_strict_1e3'], final_passed_2e3=fin['passed_2e3'],
                            first_pass_1e3_time_s=s['first_pass_1e3_time_s'], first_pass_2e3_time_s=s['first_pass_2e3_time_s'], best_observed_max_violation=s['best_observed']['max_violation'], termination=s['termination'], outcome=outcome))
        rows = [r for r in rows_of(d) if r['audit_valid']]
        ax.semilogy([r['time_s'] for r in rows], clip([r['max_violation'] for r in rows]), lw=1, label=robot)
        opt = [r for r in rows if r['optimality_valid']]
        ax2.semilogy([r['time_s'] for r in opt], clip([r['restoration_stationarity_inf'] for r in opt]), lw=1, label=robot)
    ax.axhline(STRICT_GATE, color='r', ls='--', lw=.8, label='gate 1e-3'); ax.set(xlabel='cumulative ALIGATOR solve time (s), 12 physical cores', ylabel='max physical constraint violation (nodes and 0.01 s grid)', title='ALIGATOR FDDP, uneven terrain: violation of the current iterate')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(root / 'max_violation_time.png', dpi=160); fig.savefig(root / 'max_violation_time.pdf'); plt.close(fig)
    ax2.set(xlabel='cumulative ALIGATOR solve time (s)', ylabel='||restoration-objective control stationarity||_inf', title='first-order optimality of the restoration objective'); ax2.grid(alpha=.25); ax2.legend(); fig2.tight_layout(); fig2.savefig(root / 'stationarity_time.png', dpi=160); plt.close(fig2)
    lines += ['', '![max violation vs time](max_violation_time.png)', '', '![stationarity vs time](stationarity_time.png)', '',
              'Curves show the violation of the CURRENT iterate at every accepted FDDP update (not a best-so-far envelope); a restart (new native call, new hold or mesh) can raise it. '
              '"first <= 1e-3" is the first audited iterate meeting the gate, whether or not the solver stopped there; the outcome column judges the final returned candidate. '
              'The objective is a feasibility restoration (no torque effort), so the stationarity column is that of the restoration problem, not of LEAP\'s objective. See README.md for the data layout and units.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'results.json', records)
    if records:
        with (root / 'results.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    shutil.copy2(HERE / 'README.md', root / 'README.md')
    if (HERE / 'FORMULATION_EQUIVALENCE.md').exists():
        shutil.copy2(HERE / 'FORMULATION_EQUIVALENCE.md', root / 'FORMULATION_EQUIVALENCE.md')
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--robot', help='Only rebuild this robot\'s REPORT.md')
    a = ap.parse_args()
    if a.robot:
        robot_report(a.root / 'runs' / a.robot)
    else:
        for r in ROBOTS:
            if (a.root / 'runs' / r / 'summary.json').exists():
                robot_report(a.root / 'runs' / r)
        aggregate(a.root)
