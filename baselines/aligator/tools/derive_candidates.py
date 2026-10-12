#!/usr/bin/env python3
"""Final iterate and compute-budget candidates of one Aligator run folder.

    python tools/derive_candidates.py <fence|terrain> <robot> <run_dir> [--out DIR]

Reads <run_dir>/iterations.csv (written by postprocess.py) and, for the terrain task,
<run_dir>/experiment.json. Prints one JSON object: the final iterate's candidate id
(phase:call:sample) and its recorded solve time, the number of samples within the 14400 s
budget and the number of distinct candidates. With --out, writes DIR/plan.json: the candidate
retained at every budget of the grid (budget 0, then 24 budgets per decade from
0.042169650342858224 s to 14398.791664302998 s), in the format read by
new_src/tracking/scripts/budget_study.py plan --baselines DIR/plan.json.

Selection: at budget b, the valid sample with the smallest node violation recorded at solve
time <= b (earliest on a tie); from the run's last recorded time on, its last valid sample
(end rule "last"). Samples with within_budget = False are excluded. Terrain robot names are
the tracker's (g1: g1_29dof).

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

SECTION = 'terrain_scalability'   # robots, labels, joint counts, plan table, criterion
DATA = 'data_terrain'             # the three recorded budget experiments

# CLOCKS: what each method's recorded time measures
CLOCKS = {
    'fence': {
        'leap': 'steps/wall - steps/trace_overhead; includes pipeline work since trace opened',
        'crocoddyl': 'solver_s; native solve calls excluding complete Python recorder',
        'aligator': 'time_s; cumulative native solve excluding snapshot copying; excludes setup, audits and outer updates',
    },
    'terrain': {
        'leap': 'steps/wall - steps/trace_overhead; includes pipeline work since trace opened',
        'crocoddyl': 'solver_s; native solve calls with the whole Python recorder subtracted',
        'aligator': 'time_s; cumulative native solve excluding snapshot copying; excludes setup, audits and outer updates',
    },
}
PER_DECADE = 24
T_MIN = 0.0415324792265892      # budget_grid argument; the first positive budget of the grid is 0.042169650342858224 s
T_MAX = 14398.791664302998
SELECTION = ("best valid node violation among samples recorded at time <= budget, earliest tie "
             "(the budget figure's --mode best); at the run's last recorded time and beyond, the run's last valid "
             "iterate within budget (the same rule for every method)")

def load_series(root, robot, method):
    """Read scalar records only, retaining invalid audits as unavailable samples."""
    run = Path(root) / robot
    initial = None
    if method == 'leap':
        source = run / f'trace_fence_{robot}.h5'
        with h5py.File(source, 'r') as f:
            t = f['steps/wall'][:] - f['steps/trace_overhead'][:]
            y = f['steps/gate'][:]
            ids = [str(i) for i in f['steps/step'][:]]
            # Verify that gate really is the maximum of all posed constraint terms.
            np.testing.assert_allclose(y, f['steps/term_viol'][:].max(axis=1),
                                       rtol=1e-9, atol=1e-12)
            seeds = [e for e in json.loads(f['events_json'][()])
                     if e['kind'] == 'seed' and e['step'] == 0 and e['level'] == 0]
            if len(seeds) != 1:
                raise ValueError(f'Expected one cold-seed event: {source}')
            seed = seeds[0]
            initial = dict(source='events_json: kind=seed, step=0, level=0',
                           recorded_event_wall_s=seed['wall'],
                           node_violation=seed['v1'], plotted_budget_s=0.,
                           convention='Cold input is available before any optimization step')
            # The seed event audits the pre-optimization input. Its emission wall
            # is provenance, not solver effort needed to produce that input.
            t = np.r_[0., t]
            y = np.r_[float(seed['v1']), y]
            ids.insert(0, 'seed')
        valid = np.isfinite(y)
        excluded = 0
    elif method == 'crocoddyl':
        source = run / 'iterations.h5'
        with h5py.File(source, 'r', swmr=True) as f:
            n = int(f.attrs['committed_iterations'])
            rows = [json.loads(s) for s in f['records_json'][:n]]
        excluded = sum(float(r['solver_s']) > 14400 for r in rows)
        rows = [r for r in rows if float(r['solver_s']) <= 14400]
        t = np.array([r['solver_s'] for r in rows])
        y = np.array([r['node_max'] for r in rows])
        ids = [str(r['index']) for r in rows]
        valid = np.isfinite(y)
    else:
        source = run / 'iterations.csv'
        with source.open(newline='') as f:
            rows = list(csv.DictReader(f))
        excluded = sum(r['within_budget'].lower() != 'true' for r in rows)
        rows = [r for r in rows if r['within_budget'].lower() == 'true']
        t = np.array([float(r['time_s']) for r in rows])
        y = np.array([float(r['node_max']) for r in rows])
        ids = [f"{r['phase']}:{r['call']}:{r['sample']}" for r in rows]
        valid = np.array([r['audit_valid'].lower() == 'true' and
                          r['primal_finite'].lower() == 'true' for r in rows]) & np.isfinite(y)
    return compose(t, y, valid, ids, source, excluded, initial)


def compose(t, y, valid, ids, source, excluded=0, initial=None):
    """Validate one recorded series and carry its best valid candidate forward.

    This is the figure's selection rule, shared with the uneven-terrain
    counterpart: every sample is available from its own recorded budget onwards,
    a failed audit can never improve the incumbent, and chronological ties are
    preserved.
    """
    t, y, valid = np.asarray(t, float), np.asarray(y, float), np.asarray(valid, bool)
    if not len(t) or not valid.any():
        raise ValueError(f'No usable samples: {source}')
    if not np.isfinite(t).all() or (t < 0).any() or (np.diff(t) < -1e-9).any():
        raise ValueError(f'Invalid or decreasing budget clock: {source}')
    if (y[valid] < 0).any():
        raise ValueError(f'Negative constraint violation: {source}')
    # Preserve chronological ties. Failed samples cannot improve an incumbent.
    best = np.minimum.accumulate(np.where(valid, y, np.inf))
    best[~np.isfinite(best)] = np.nan
    return dict(time=t, violation=np.where(valid, y, np.nan), best=best,
                valid=valid, ids=list(ids), source=str(Path(source).resolve()),
                excluded=int(excluded), initial=initial)


def number(text):
    """A recorded scalar, or NaN where the run stored no audit for that sample."""
    return float(text) if text not in ('', 'nan', 'None', None) else float('nan')


def reported(value):
    """A scalar for the JSON provenance; an unaudited sample is reported as null."""
    value = float(value)
    return value if np.isfinite(value) else None


def run_directory(config, method, robot):
    """The recorded run of one robot; `run_directories` names the exceptions."""
    cfg = config[DATA]
    directory = Path(cfg['roots'][method]) / cfg['run_directories'][method].get(robot, robot)
    if not directory.is_dir():
        raise ValueError(f'Missing recorded run: {directory}')
    return directory


def aligator_series(config, robot):
    """One row per stored primal: the call inputs, the accepted updates, the returns."""
    run = run_directory(config, 'aligator', robot)
    source = run / 'iterations.csv'
    with source.open(newline='') as f:
        rows = list(csv.DictReader(f))
    excluded = sum(r['within_budget'].lower() != 'true' for r in rows)
    rows = [r for r in rows if r['within_budget'].lower() == 'true']
    t = np.array([float(r['time_s']) for r in rows])
    y = np.array([number(r['node_max']) for r in rows])
    ids = [f"{r['phase']}:{r['call']}:{r['sample']}" for r in rows]
    valid = np.array([r['audit_valid'].lower() == 'true' and
                      r['primal_finite'].lower() == 'true' for r in rows]) & np.isfinite(y)
    series = compose(t, y, valid, ids, source, excluded)
    overall = np.array([number(r['max_violation']) if v else np.nan
                        for r, v in zip(rows, valid)])
    experiment = json.loads((run / 'experiment.json').read_text())
    series.update(plan=experiment['inputs']['problem_yaml']['path'],
                  threads=len(experiment['cpus']),
                  outcome=dict(criterion_met=any(r['passed_strict_1e3'].lower() == 'true'
                                                 for r in rows),
                               stop='best observed', final_node=reported(y[-1]),
                               final_fine_grid=reported(number(rows[-1]['fine_max'])),
                               best_node_and_fine_grid=reported(np.nanmin(overall)),
                               field='iterations.csv max_violation / passed_strict_1e3'))
    return series


def incumbent_indices(series):
    """`best_idx[i]`: index of the retained candidate after sample i (the figure's
    best-so-far rule made explicit): the valid sample with the smallest violation among
    samples 0..i, the earliest on a tie; -1 while no valid sample exists."""
    y, valid = np.asarray(series["violation"], float), np.asarray(series["valid"], bool)
    best_idx = np.full(len(y), -1, int)
    cur, cur_v = -1, np.inf
    for i in range(len(y)):
        if valid[i] and np.isfinite(y[i]) and y[i] < cur_v:
            cur, cur_v = i, y[i]
        best_idx[i] = cur
    # must agree with the figure's own running minimum
    best = np.asarray(series["best"], float)
    check = np.array([y[j] if j >= 0 else np.nan for j in best_idx])
    if not np.allclose(np.nan_to_num(check, nan=-1.), np.nan_to_num(best, nan=-1.)):
        raise AssertionError(f"incumbent selection disagrees with the figure's best-so-far: {series['source']}")
    return best_idx


def budget_grid(all_series, per_decade=16, t_min=None, t_max=None):
    """Budget zero (the cold seed) plus log-spaced budgets from the first positive
    recorded time of any series to the last recorded time of any series. One grid for
    every task, method and robot, so a budget read off any panel is the same budget."""
    positive = np.concatenate([np.asarray(d["time"])[np.asarray(d["time"]) > 0] for d in all_series])
    left = float(positive.min()) if t_min is None else float(t_min)
    right = max(float(d["time"][-1]) for d in all_series) if t_max is None else float(t_max)
    lo, hi = np.floor(np.log10(left) * per_decade) / per_decade, np.log10(right)
    exps = np.arange(lo, hi + 1e-9, 1.0 / per_decade)
    grid = 10.0 ** exps
    grid = grid[(grid >= left * (1 - 1e-12)) & (grid <= right * (1 + 1e-12))]
    if grid.size == 0 or grid[-1] < right * (1 - 1e-9):
        grid = np.r_[grid, right]
    return np.r_[0.0, grid]


def last_candidate(series):
    """The run's last VALID recorded sample within budget (the figure's series already
    drops over-budget rows): the same rule for every method, LEAP's final archive included."""
    ids = list(series["ids"])
    valid = np.asarray(series["valid"], bool) & np.isfinite(np.asarray(series["violation"], float))
    j = int(np.flatnonzero(valid)[-1])
    return dict(id=ids[j], time_s=float(series["time"][j]), node_violation=float(series["violation"][j]))


def plan_series(series, grid, task, method, robot, returned=None):
    """`returned`: the run's returned solution (`returned_candidate`). When given, it is
    the candidate at the run's last recorded time (and hence along the dotted tail
    beyond), in place of the best-so-far sample: the curve ends at what the method
    actually delivered. A returned solution
    recorded past the series' last time (over budget) is appended at its own time."""
    """Which candidate each grid budget retains, and the distinct candidates to roll out.

    The budgets beyond the series' own last recorded time keep the final incumbent
    (the figure's dotted tail), which is included once with the run's own last time.
    """
    t = np.asarray(series["time"], float)
    best_idx = incumbent_indices(series)
    ids = list(series["ids"])
    y = np.asarray(series["violation"], float)
    budgets, cands = [], {}
    last = float(t[-1])
    points = [float(b) for b in grid if b <= last * (1 + 1e-12)]
    if not points or points[-1] < last * (1 - 1e-9):
        points.append(last)
    extra = None
    if returned is not None and returned["id"] not in ids:
        extra = returned
        points = [b for b in points if b < last * (1 - 1e-9)] + [float(returned["time_s"])]
        last = float(returned["time_s"])
    for b in points:
        i = int(np.searchsorted(t, b * (1 + 1e-12), side="right") - 1)
        j = int(best_idx[i]) if i >= 0 else -1
        if returned is not None and b >= last * (1 - 1e-9):
            if extra is not None:
                cid = extra["id"]
                budgets.append(dict(budget_s=b, candidate=cid))
                cands[cid] = dict(id=cid, index=None, recorded_time_s=float(extra["time_s"]),
                                  node_violation=float(extra["node_violation"]),
                                  dir=candidate_dir_name(method, cid), budgets=[b], beyond_series=True)
                continue
            j = ids.index(returned["id"])
        cid = ids[j] if j >= 0 else None
        budgets.append(dict(budget_s=b, candidate=cid))
        if cid is not None and cid not in cands:
            cands[cid] = dict(id=cid, index=j, recorded_time_s=float(t[j]), node_violation=float(y[j]),
                              dir=candidate_dir_name(method, cid), budgets=[])
        if cid is not None:
            cands[cid]["budgets"].append(b)
    for c in cands.values():
        c["role"] = "incumbent"
    if returned is not None:
        rid = returned["id"]
        cands[rid]["role"] = "returned" if len(cands[rid]["budgets"]) == 1 else "incumbent+returned"
    # The run's own returned solution (its last recorded sample) is rolled out too, even
    # when the best-so-far rule retains an earlier sample.
    last_id = ids[-1]
    if last_id in cands:
        cands[last_id]["role"] = "incumbent+final_iterate"
    elif valid_last(series):
        cands[last_id] = dict(id=last_id, index=len(ids) - 1, recorded_time_s=last, node_violation=float(y[-1]),
                              dir=candidate_dir_name(method, last_id), budgets=[], role="final_iterate")
    return dict(final_iterate=last_id, returned=None if returned is None else returned["id"],
                returned_time_s=None if returned is None else float(returned["time_s"]),task=task, method=method, robot=robot, source=series["source"],
                samples=int(len(t)), last_time_s=last, first_positive_time_s=float(t[t > 0].min()) if (t > 0).any() else None,
                budgets=budgets, candidates=list(cands.values()))


def valid_last(series):
    return bool(np.asarray(series["valid"], bool)[-1]) and np.isfinite(float(series["violation"][-1]))


def candidate_dir_name(method, cid):
    """A filesystem name for a sample id: `seed`, `step_000123` (LEAP), `rec_000045`
    (Crocoddyl), `p00_c00_s000189` / `p00_c00_sm1` (Aligator `phase:call:sample`)."""
    if cid == "seed":
        return "seed"
    if method == "leap":
        return f"step_{int(float(cid)):06d}"
    if method == "crocoddyl":
        return f"rec_{int(cid):06d}"
    if method == "aligator":
        phase, call, sample = (int(x) for x in cid.split(":"))
        s = f"s{sample:06d}" if sample >= 0 else f"sm{-sample}"
        return f"p{phase:02d}_c{call:02d}_{s}"
    raise KeyError(method)


# --------------------------------------------------------------------------- CLI
def recorded_series(task, robot, run_dir):
    run_dir = Path(run_dir).resolve()
    if task == 'fence':
        # load_series reads <root>/<robot>/iterations.csv
        if run_dir.name == robot:
            return load_series(run_dir.parent, robot, 'aligator')
        root = Path(tempfile.mkdtemp(prefix='aligator_series_'))
        os.symlink(run_dir, root / robot)
        return load_series(root, robot, 'aligator')
    config = {DATA: dict(roots=dict(aligator=str(run_dir.parent)), run_directories=dict(aligator={robot: run_dir.name}))}
    return aligator_series(config, robot)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task', choices=('fence', 'terrain'))
    ap.add_argument('robot')
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--out', type=Path, default=None, help='write DIR/plan.json')
    args = ap.parse_args(argv)
    series = recorded_series(args.task, args.robot, args.run_dir)
    grid = budget_grid([series], per_decade=PER_DECADE, t_min=T_MIN, t_max=T_MAX)
    returned = last_candidate(series)
    p = plan_series(series, grid, args.task, 'aligator', args.robot, returned=returned)
    summary = dict(task=args.task, robot=args.robot, run_dir=str(args.run_dir.resolve()), final_iterate=p['final_iterate'],
                   final_iterate_time_s=p['returned_time_s'], final_iterate_node_violation=returned['node_violation'],
                   samples=p['samples'], excluded_over_budget=int(series['excluded']), distinct_candidates=len(p['candidates']))
    print(json.dumps(summary, indent=1))
    if args.out is not None:
        plan = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), per_decade=PER_DECADE,
                    budget_grid_s=[float(b) for b in grid], n_budgets=int(len(grid)), n_candidates=len(p['candidates']),
                    selection=SELECTION, end_rule='last', clocks=CLOCKS, series=[p])
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'plan.json').write_text(json.dumps(plan, indent=1))
        print(f'wrote {args.out / "plan.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
