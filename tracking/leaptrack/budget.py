"""The budget study (Fig. 3): roll out the solver's snapshots.

The candidates are the solution archives that `leap_solve ... output.snapshots=<dir>`
writes (new_src/solver/snapshots.hpp): the incumbent (smallest node gate so far) at every
crossing of the budget grid, before every mesh change, and the final iterate. Every
candidate is tracked by the rollout of `rollout.track_archive`.

    snaps  = load_snapshots(snap_dir)              # snapshots.csv -> list of candidates
    grid   = budget_grid(24, 0.0421696..., 14398.79)  # 0, then 24 budgets per decade
    series = plan_series("terrain", "go2", snap_dir, grid)   # which candidate each budget retains
    rows   = run_series(series, out_root, python, jobs, cores)  # one rollout per distinct candidate
    collect(plan, out_root) -> results.json / results.csv

Selection rule: at budget b the retained candidate is the valid recorded sample with the
smallest node violation among those recorded at wall <= b, earliest on a tie; from the
run's end on, the run's last iterate (the final archive), which the figure draws as a
dotted tail. Budget 0 is the cold seed.

Baselines: `recorded_series` copies the Crocoddyl / Aligator series of a candidate plan
(the plan.json `new_src/baselines/<method>/tools/derive_candidates.py --out` writes: which
recorded candidate every budget retains) into the plan, with the run folder of each. `run_series` rolls every distinct
candidate out through `scripts/track.py --method <m> --candidate <id> --run-dir <run>`
with the same `extra` flags as the LEAP candidates, into
`<out>/<task>/<method>/<robot>/<dir>/` (`rec_000060`, `p03_c32_s000152`); `collect` and
the figure read them like the LEAP rows (candidate = the recorded id string).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACKING = HERE.parent
TRACK_SCRIPT = TRACKING / "scripts" / "track.py"

# the budget grid: 24 per decade from T_MIN to T_MAX, 135 budgets with budget 0
PER_DECADE = 24
T_MIN = 0.042169650342858224
T_MAX = 14398.791664302998

THRESHOLDS = {"base_pos_err": 0.050, "base_rot_err": 0.10, "joint_err_max": 0.20,
              "foot_err_max": 0.040, "base_speed": 0.10, "joint_speed_max": 0.10}


def budget_grid(per_decade=PER_DECADE, t_min=T_MIN, t_max=T_MAX):
    """Budget 0, then 10^(k / per_decade) from the decade step at or below t_min to t_max
    (t_max appended when it is not on the grid): series.budget_grid of the reference."""
    lo, hi = np.floor(np.log10(t_min) * per_decade) / per_decade, np.log10(t_max)
    exps = np.arange(lo, hi + 1e-9, 1.0 / per_decade)
    grid = 10.0 ** exps
    grid = grid[(grid >= t_min * (1 - 1e-12)) & (grid <= t_max * (1 + 1e-12))]
    if grid.size == 0 or grid[-1] < t_max * (1 - 1e-9):
        grid = np.r_[grid, t_max]
    return np.r_[0.0, grid]


def load_snapshots(snap_dir: Path) -> list[dict]:
    """The rows of snapshots.csv (index, file, wall_s, step, stage, level, gate, why), with
    absolute paths. The last row is always the run's final iterate."""
    snap_dir = Path(snap_dir)
    rows = []
    with (snap_dir / "snapshots.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(dict(index=int(r["index"]), file=str(snap_dir / r["file"]), wall_s=float(r["wall_s"]),
                             step=int(r["step"]), stage=r["stage"], level=int(r["level"]), gate=float(r["gate"]), why=r["why"]))
    if not rows:
        raise ValueError(f"{snap_dir}: no snapshots")
    return rows


def load_summary(snap_dir: Path) -> dict:
    out = {}
    p = Path(snap_dir) / "summary.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def plan_series(task: str, robot: str, snap_dir: Path, grid, method: str = "leap") -> dict:
    """Which candidate each budget of `grid` retains for one run, and the distinct candidates."""
    snaps = load_snapshots(snap_dir)
    summary = load_summary(snap_dir)
    final = snaps[-1]
    last_time = float(summary.get("wall_s", final["wall_s"]))
    gates = np.array([s["gate"] for s in snaps])
    walls = np.array([s["wall_s"] for s in snaps])
    valid = np.isfinite(gates)
    budgets, used = [], {}
    for b in grid:
        if b >= last_time and b > 0:
            cand = final
        else:
            ok = np.where(valid & (walls <= b + 1e-12))[0]
            if ok.size == 0:  # the seed is recorded at wall 0, so this only happens without a seed
                budgets.append(dict(budget_s=float(b), candidate=None))
                continue
            j = ok[np.argmin(gates[ok])]  # argmin returns the earliest on a tie
            cand = snaps[j]
        budgets.append(dict(budget_s=float(b), candidate=cand["index"]))
        used.setdefault(cand["index"], dict(candidate=cand["index"], file=cand["file"], recorded_time_s=cand["wall_s"],
                                             step=cand["step"], stage=cand["stage"], level=cand["level"],
                                             node_violation=cand["gate"], why=cand["why"], budgets=[]))
        used[cand["index"]]["budgets"].append(float(b))
    if final["index"] not in used:
        used[final["index"]] = dict(candidate=final["index"], file=final["file"], recorded_time_s=final["wall_s"], step=final["step"],
                                    stage=final["stage"], level=final["level"], node_violation=final["gate"], why=final["why"],
                                    budgets=[])
    return dict(task=task, robot=robot, method=method, snapshots_dir=str(snap_dir), last_time_s=last_time,
                final_iterate=final["index"], n_snapshots=len(snaps), budgets=budgets,
                candidates=[used[k] for k in sorted(used)])


def make_plan(series: list[dict], grid, per_decade=PER_DECADE, t_min=T_MIN, t_max=T_MAX) -> dict:
    return dict(per_decade=per_decade, t_min=t_min, t_max=t_max, n_budgets=int(len(grid)), budgets=[float(b) for b in grid],
                selection="valid recorded sample with the smallest node gate among those recorded at wall <= budget, earliest on a "
                          "tie; from the run's end on, the run's last iterate",
                series=series)


def is_recorded(s: dict) -> bool:
    """A baseline series (recorded Crocoddyl / Aligator candidates), not LEAP snapshots."""
    return s.get("method", "leap") != "leap"


def recorded_series(baseline_plan, methods=("crocoddyl", "aligator"), tasks=None, robots=None) -> list[dict]:
    """The Crocoddyl / Aligator series of a candidate plan (`baseline_plan`: a plan.json
    written by `new_src/baselines/<method>/tools/derive_candidates.py --out`), each
    candidate keyed by its recorded id (`candidate`), with the run folder (`run_dir`)."""
    from .recorded import baseline_series
    path = Path(baseline_plan)
    plan = json.loads(path.read_text())
    out = []
    for s in baseline_series(methods=tuple(methods), plan=plan):
        if (tasks and s["task"] not in tasks) or (robots and s["robot"] not in robots):
            continue
        cands = [dict(c, candidate=str(c["id"])) for c in s["candidates"]]
        out.append(dict(s, kind="recorded", baseline_plan=str(path),
                        final_iterate=str(s["final_iterate"]), candidates=cands,
                        budgets=[dict(b, candidate=None if b["candidate"] is None else str(b["candidate"]))
                                 for b in s["budgets"]]))
    return out


def candidate_dir(out_root: Path, s: dict, cand: dict) -> Path:
    if is_recorded(s):
        return Path(out_root) / s["task"] / s["method"] / s["robot"] / cand["dir"]
    return Path(out_root) / s["task"] / s["method"] / s["robot"] / f"cand_{cand['candidate']:04d}"


def ratio_of(evaluation: dict, tag: str):
    """max_i e_i / e_bar_i of one final-state block (Table III's quantity)."""
    block = evaluation["final_state"][tag]
    ratios = {k: float(block[k]) / t for k, t in THRESHOLDS.items()}
    worst = max(ratios, key=ratios.get)
    return ratios[worst], worst, ratios


def _rollout_cmd(python, task, robot, archive, out_dir, core, extra):
    cmd = []
    if core is not None:
        cmd += ["taskset", "-c", str(core)]
    cmd += [str(python), str(TRACK_SCRIPT), "--task", task, "--robot", robot, "--archive", str(archive), "--out", str(out_dir)]
    cmd += list(extra or [])
    return cmd


def _recorded_cmd(python, s, cand, out_dir, core, extra):
    """track.py for one recorded baseline candidate (the same `extra` flags as LEAP's)."""
    cmd = []
    if core is not None:
        cmd += ["taskset", "-c", str(core)]
    cmd += [str(python), str(TRACK_SCRIPT), "--task", s["task"], "--robot", s["robot"], "--method", s["method"],
            "--candidate", str(cand["candidate"]), "--run-dir", str(s["run_dir"]), "--out", str(Path(out_dir).resolve())]
    cmd += list(extra or [])
    return cmd


def run_series(plan: dict, out_root: Path, python: Path, jobs: int = 8, cores: list[int] | None = None, extra=None,
               only=None, redo: bool = False, verbose: bool = True, methods=None) -> list[dict]:
    """One rollout per distinct candidate of every series (a subprocess each, pinned to a
    core when `cores` is given), LEAP snapshots and recorded baselines alike. `only`:
    {(task, robot)}; `methods`: the methods to run (default all). Returns the rows
    (status + scalars), also written per candidate as summary.json."""
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    jobs_list = []
    for s in plan["series"]:
        if only and (s["task"], s["robot"]) not in only:
            continue
        if methods and s.get("method", "leap") not in methods:
            continue
        for cand in s["candidates"]:
            d = candidate_dir(out_root, s, cand)
            if not redo and (d / "summary.json").exists():
                continue
            jobs_list.append((s, cand, d))
    if verbose:
        print(f"[budget] {len(jobs_list)} rollout(s) to run with {jobs} job(s)")
    slots = list(cores) if cores else [None] * jobs
    free = list(range(len(slots)))

    def work(item):
        s, cand, d = item
        slot = free.pop(0)
        try:
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            if is_recorded(s):
                cmd = _recorded_cmd(python, s, cand, d, slots[slot], extra)
                # track.py writes its own run.log into the case folder; the console goes next to it
                with (d / "stdout.log").open("w") as log:
                    log.write(" ".join(cmd) + "\n")
                    log.flush()
                    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(TRACKING))
            else:
                cmd = _rollout_cmd(python, s["task"], s["robot"], cand["file"], d, slots[slot], extra)
                with (d / "run.log").open("w") as log:
                    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(TRACKING))
            row = summarize_candidate(s, cand, d, rc, time.time() - t0)
            (d / "summary.json").write_text(json.dumps(row, indent=1) + "\n")
            if verbose:
                label = f"{cand['candidate']:4d}" if isinstance(cand["candidate"], int) else f"{s['method']} {cand['candidate']}"
                print(f"[budget] {s['task']} {s['robot']} cand {label} (wall {cand['recorded_time_s']:9.3f} s, gate "
                      f"{cand['node_violation']:.2e}): {row['status']} ratio_end={row.get('terminal_ratio_end')} "
                      f"ratio_at_T={row.get('terminal_ratio_at_T')} [{row['rollout_wall_s']:.0f} s]", flush=True)
            return row
        finally:
            free.append(slot)

    rows = []
    with ThreadPoolExecutor(max_workers=len(slots)) as ex:
        for row in ex.map(work, jobs_list):
            rows.append(row)
    return rows


def summarize_candidate(s: dict, cand: dict, d: Path, rc: int, wall: float) -> dict:
    if is_recorded(s):
        row = dict(task=s["task"], method=s["method"], robot=s["robot"], candidate=cand["candidate"], dir=cand["dir"],
                   role=cand.get("role", "incumbent"), run_dir=s["run_dir"], recorded_time_s=cand["recorded_time_s"],
                   node_violation=cand["node_violation"], budgets=cand["budgets"], rollout_wall_s=wall, returncode=rc)
        m = d / "metrics.json"
        if m.exists():
            row["archive"] = json.loads(m.read_text()).get("archive")
    else:
        row = dict(task=s["task"], method=s["method"], robot=s["robot"], candidate=cand["candidate"], archive=cand["file"],
                   recorded_time_s=cand["recorded_time_s"], step=cand["step"], stage=cand["stage"], level=cand["level"],
                   node_violation=cand["node_violation"], budgets=cand["budgets"], rollout_wall_s=wall, returncode=rc)
    ev = d / "evaluation.json"
    if rc != 0 or not ev.exists():
        row["status"] = "failed"
        return row
    evaluation = json.loads(ev.read_text())
    try:
        for tag in ("end", "at_T"):
            r, worst, per = ratio_of(evaluation, tag)
            row[f"terminal_ratio_{tag}"] = r
            row[f"terminal_worst_{tag}"] = worst
            row[f"terminal_ratios_{tag}"] = per
    except Exception as e:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"no final-state block: {e}"
        return row
    m = d / "metrics.json"
    if m.exists():
        metrics = json.loads(m.read_text())
        for k in ("fell", "unstable", "success", "time_shift_s", "sim_time_s"):
            if k in metrics:
                row[k] = metrics[k]
    row["status"] = "ok"
    return row


def collect(plan: dict, out_root: Path) -> dict:
    rows = []
    for s in plan["series"]:
        for cand in s["candidates"]:
            d = candidate_dir(out_root, s, cand)
            p = d / "summary.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
            else:
                rows.append(dict(task=s["task"], method=s["method"], robot=s["robot"], candidate=cand["candidate"], status="pending",
                                 recorded_time_s=cand["recorded_time_s"], node_violation=cand["node_violation"], budgets=cand["budgets"]))
    results = dict(plan=str(Path(out_root) / "plan.json"), collected_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rows=rows)
    (Path(out_root) / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    with (Path(out_root) / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "robot", "candidate", "status", "recorded_time_s", "step", "stage", "level", "node_violation",
                    "terminal_ratio_end", "terminal_worst_end", "terminal_ratio_at_T", "terminal_worst_at_T", "fell", "rollout_wall_s"])
        for r in rows:
            w.writerow([r.get(k, "") for k in ("task", "method", "robot", "candidate", "status", "recorded_time_s", "step", "stage", "level",
                                                "node_violation", "terminal_ratio_end", "terminal_worst_end", "terminal_ratio_at_T",
                                                "terminal_worst_at_T", "fell", "rollout_wall_s")])
    return results


def default_python() -> Path:
    """The simulation interpreter: $LEAP_SIM_PYTHON, else the one running this module."""
    return Path(os.environ.get("LEAP_SIM_PYTHON", sys.executable))
