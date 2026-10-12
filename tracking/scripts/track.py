#!/usr/bin/env python
"""Track one LEAP archive, or a whole task, in MuJoCo with the task's whole-body QP.

    PY=<venv with mujoco, proxsuite, pinocchio>/bin/python
    export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

    # one archive -> <out>/{rollout.npz, metrics.json, trajectory.npz, evaluation.json, evaluation.npz, run.log}
    taskset -c 64 $PY track.py --task fence --robot go2 --archive sol_fence_go2.h5 --out tracked/fence/go2
    # the record naming sol_<task>_<robot>[_<tag>].h5 in a directory
    $PY track.py --task fence --robot go2 --archive-dir records/fence --archive-tag rerun --out tracked/fence/go2
    # a whole task: one subprocess per robot, each pinned to one core -> <out>/<robot>/
    $PY track.py --task terrain --all --archive-dir records/terrain --out tracked/terrain --jobs 5 --cores 64-99
    # k repeats of a case in fresh interpreters -> <out>/rep<k>/ (or <out>/<robot>/rep<k>/ with --all)
    $PY track.py --task fence --robot h1 --archive sol_fence_h1.h5 --out tracked/fence/h1 --repeat 3 --cores 64-66

    # a BASELINE solution (never re-solved): one candidate of a run folder written by
    # new_src/baselines/<method>/run.sh (its final iterate id: see that method's README), or 'best'
    $PY track.py --task fence --robot h1 --method crocoddyl --run-dir <run folder> --candidate 60 --out tracked/fence/crocoddyl/h1
    $PY track.py --task fence --robot h1 --method aligator --run-dir <run folder> --candidate 3:32:152 --out <dir>

The rollout is `leaptrack.rollout.track_archive` (LEAP) or
`leaptrack.recorded.track_recorded` (Crocoddyl / Aligator: the same `rollout.run`,
evaluator and output files).

`--method crocoddyl|aligator` reads the baseline's stored solution through
`leaptrack.recorded` from the run folder `--run-dir` (written by
`new_src/baselines/<method>/run.sh`); `--candidate` is a sample id of that run
(Crocoddyl: the record index, e.g. 60; Aligator: phase:call:sample) or `best`, the run's
own selection rule. Both are required; `--all` / `--robots` do not apply.
The controller is the task's and robot's, exactly as for LEAP (`controllers.py`).

The problem yaml is read from the archive (`meta_json["problem"]["yaml"]`) unless
`--problem-yaml` is given. Rollout options: `--hold`, `--settle`, `--control-dt`,
`--duration` (no hold, no evaluation), `--clock touchdown|timewarp|none`,
`--early-jump 0|1` (default: the task / robot rule of `controllers.py`),
`--friction` (a DIAGNOSTIC override of the problem's mu), `--no-capsule-collision`,
`--approach-ring`, and every `wbc.WbcGains` field as a flag (experiments).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent                      # new_src/tracking
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from leaptrack.controllers import ALL_ROBOTS, TASKS, TASK_ROBOTS, gains_for  # noqa: E402

CLOCKS = ("touchdown", "timewarp", "none")
METHODS = ("leap", "crocoddyl", "aligator")
ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="egl")


def archive_path(task, robot, archive_dir, tag):
    return Path(archive_dir) / (f"sol_{task}_{robot}" + (f"_{tag}" if tag else "") + ".h5")


# ------------------------------------------------------------------- gains
def add_gain_args(ap):
    from leaptrack.wbc import WbcGains
    for f in fields(WbcGains):
        if f.name == "mu":
            ap.add_argument("--mu", type=float, default=None)
        elif isinstance(f.default, bool):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(f.default, (int, float)):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None)


def gain_overrides(args) -> dict:
    from leaptrack.wbc import WbcGains
    return {f.name: getattr(args, f.name, None) for f in fields(WbcGains) if hasattr(args, f.name)}


# ------------------------------------------------------------------- cores
def physical_cpus():
    seen, cpus = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((p / "physical_package_id").read_text(), (p / "core_id").read_text())
        if key not in seen:
            seen.add(key)
            cpus.append(cpu)
    return cpus


def parse_cores(spec: str) -> list:
    """'64-83,90,92-95' -> [64, ..., 83, 90, 92, ..., 95]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ------------------------------------------------------------------- suite
def run_jobs(jobs: list, cores: list, extra_args: list):
    """`jobs`: [(task, robot, archive, out_dir)] -- or (task, robot, None, out_dir,
    job_args) for a recorded baseline (`job_args` = its --method / --candidate /
    --run-dir flags) -- each run as `track.py` in its own interpreter pinned to one
    core of `cores`; long cases first. Returns one row per job (exit code, wall, the
    metrics' headline numbers)."""
    from concurrent.futures import ThreadPoolExecutor
    import queue
    order = {"talos": 0, "h1": 1, "g1_29dof": 2, "go2": 3, "hopper2d_foot": 4, "hopper2d": 5}
    q = queue.Queue()
    for j in sorted(jobs, key=lambda j: order.get(j[1], 9)):
        q.put(j)
    env = dict(os.environ, **ENV)
    rows = []

    def worker(core):
        res = []
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return res
            task, robot, archive, dest = job[:4]
            job_args = list(job[4]) if len(job) > 4 else []
            # absolute paths: the subprocess runs in scripts/ (cwd=HERE), so a relative
            # --out or archive would otherwise resolve against a different directory
            dest = Path(dest).resolve()
            if archive is not None:
                archive = Path(archive).resolve()
            dest.mkdir(parents=True, exist_ok=True)
            cmd = ["taskset", "-c", str(core), sys.executable, str(HERE / "track.py"),
                   "--task", task, "--robot", robot]
            cmd += (["--archive", str(archive)] if archive is not None else []) + ["--out", str(dest)]
            cmd += job_args + extra_args
            t0 = time.perf_counter()
            with (dest / "stdout.log").open("w") as lg:
                lg.write(" ".join(cmd) + "\n")
                lg.flush()
                code = subprocess.run(cmd, cwd=str(HERE), env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
            row = dict(task=task, robot=robot, archive=str(archive), out=str(dest), core=core,
                       exit_code=code, process_wall_s=time.perf_counter() - t0)
            if job_args:
                row.update(method=job_args[job_args.index("--method") + 1] if "--method" in job_args else None,
                           job_args=job_args)
            if (dest / "metrics.json").exists():
                mt = json.loads((dest / "metrics.json").read_text())
                row.update(success=mt["success"], fell=mt["fell"], base_xyz_rmse=mt["base_xyz_rmse"],
                           final_base_err=mt["final"]["base_xyz_err"], time_shift=mt["time_shift"], wall_s=mt["wall_s"])
                if job_args:
                    row.update(candidate=mt.get("candidate"), archive=mt.get("archive"))
            if (dest / "evaluation.json").exists():
                from leaptrack.score import terminal_ratio
                tag, r, worst, _ = terminal_ratio(dest / "evaluation.json", task, robot)
                row.update(ratio=r, ratio_tag=tag, limiting=worst)
            print(json.dumps(row), flush=True)
            res.append(row)

    with ThreadPoolExecutor(max_workers=len(cores)) as pool:
        for group in pool.map(worker, cores):
            rows.extend(group)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--robot", choices=ALL_ROBOTS)
    ap.add_argument("--all", action="store_true", help="every robot of the task's suite (controllers.TASK_ROBOTS)")
    ap.add_argument("--robots", default=None, help="comma list of robots for the suite")
    ap.add_argument("--archive", default=None, help="the solution archive (.h5) of ONE robot")
    ap.add_argument("--archive-dir", default=None, help="directory holding sol_<task>_<robot>[_<tag>].h5")
    ap.add_argument("--archive-tag", default=None)
    ap.add_argument("--problem-yaml", default=None, help="override the problem yaml embedded in the archive")
    ap.add_argument("--method", choices=METHODS, default="leap",
                    help="whose solution: leap (default; an archive) or a recorded baseline (crocoddyl, aligator; "
                         "read from its run folder, never re-solved)")
    ap.add_argument("--candidate", default=None,
                    help="baselines (required): a sample id of the run (Crocoddyl e.g. 60, Aligator e.g. 3:32:152) "
                         "or best (the run's own selection rule)")
    ap.add_argument("--run-dir", default=None,
                    help="baselines (required): the run folder written by new_src/baselines/<method>/run.sh")
    ap.add_argument("--out", required=True, help="case dir (one robot) or suite root (--all / --robots: <out>/<robot>/)")
    ap.add_argument("--repeat", type=int, default=1, help="k > 1: run each case k times into <case>/rep<k>/")
    ap.add_argument("--jobs", type=int, default=5, help="parallel subprocesses (suite / repeats)")
    ap.add_argument("--cores", default=None, help="explicit core list for the subprocesses, e.g. 64-83 (default: physical cores from --core-offset)")
    ap.add_argument("--core-offset", type=int, default=64, help="index into the physical-core list")
    ap.add_argument("--hold", type=float, default=None)
    ap.add_argument("--settle", type=float, default=None)
    ap.add_argument("--control-dt", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None, help="simulate only this long (no hold, no evaluation)")
    ap.add_argument("--clock", default=None, choices=CLOCKS)
    ap.add_argument("--early-jump", type=lambda x: x.lower() in ("1", "true", "yes"), default=None,
                    help="clock jumps to the touchdown state on an early landing (default: task / robot rule)")
    ap.add_argument("--friction", type=float, default=None, help="DIAGNOSTIC: MuJoCo friction instead of the problem's mu")
    ap.add_argument("--no-capsule-collision", action="store_true")
    ap.add_argument("--approach-ring", type=float, default=None)
    ap.add_argument("--no-evaluate", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    add_gain_args(ap)
    args = ap.parse_args(argv)

    baseline = args.method != "leap"
    if baseline and (args.archive or args.archive_dir or args.problem_yaml):
        ap.error("--archive / --archive-dir / --problem-yaml are LEAP options; a baseline is read from its run folder")
    if not baseline and (args.candidate or args.run_dir):
        ap.error("--candidate / --run-dir need --method crocoddyl|aligator")
    if baseline:
        if args.all or args.robots:
            ap.error("--method crocoddyl|aligator tracks one run folder: give --robot, not --all / --robots")
        if not args.run_dir or not args.candidate:
            ap.error("--method crocoddyl|aligator needs --run-dir <run folder> and --candidate <id|best>")
    multi = args.all or args.robots or args.repeat > 1
    if multi:
        robots = ([r for r in args.robots.split(",") if r] if args.robots
                  else (list(TASK_ROBOTS[args.task]) if args.all else [args.robot]))
        if not all(robots):
            ap.error("--robot, --robots or --all")
        if not baseline:
            if args.archive and len(robots) > 1:
                ap.error("--archive names one archive; use --archive-dir for several robots")
            if not args.archive and not args.archive_dir:
                ap.error("--archive or --archive-dir")
        extra = []
        for k in ("hold", "settle", "control_dt", "duration", "problem_yaml", "friction", "clock", "approach_ring"):
            if getattr(args, k) is not None:
                extra += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        if args.early_jump is not None:
            extra += ["--early-jump", "1" if args.early_jump else "0"]
        for flag in ("no_capsule_collision", "no_evaluate"):
            if getattr(args, flag):
                extra.append(f"--{flag.replace('_', '-')}")
        for k, v in gain_overrides(args).items():
            if v is not None:
                extra += [f"--{k.replace('_', '-')}", str(v)]
        jobs = []
        for robot in robots:
            if baseline:
                job_args = ["--method", args.method, "--candidate", args.candidate,
                            "--run-dir", str(Path(args.run_dir).resolve())]
                case = Path(args.out) / robot if (args.all or args.robots) else Path(args.out)
                reps = [case / f"rep{k}" for k in range(1, args.repeat + 1)] if args.repeat > 1 else [case]
                jobs += [(args.task, robot, None, c, job_args) for c in reps]
                continue
            arch = Path(args.archive) if args.archive else archive_path(args.task, robot, args.archive_dir, args.archive_tag)
            if not arch.exists():
                raise SystemExit(f"missing archive {arch}")
            case = Path(args.out) / robot if (args.all or args.robots) else Path(args.out)
            if args.repeat > 1:
                jobs += [(args.task, robot, arch, case / f"rep{k}") for k in range(1, args.repeat + 1)]
            else:
                jobs.append((args.task, robot, arch, case))
        cores = parse_cores(args.cores) if args.cores else physical_cpus()[args.core_offset:args.core_offset + args.jobs]
        cores = cores[:max(1, args.jobs)]
        if not cores:
            raise SystemExit("no cores to run on")
        rows = run_jobs(jobs, cores, extra)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "suite_summary.json").write_text(json.dumps(rows, indent=1))
        return 0 if all(r["exit_code"] == 0 for r in rows) else 2

    # ------------------------------------------------------------ one rollout
    if not args.robot:
        ap.error("--robot (or --all / --robots)")
    if baseline:
        from leaptrack.recorded import track_recorded
        kw = dict(gains=gains_for(args.task, args.robot, gain_overrides(args)), verbose=not args.quiet,
                  clock=args.clock or "touchdown", early_jump=args.early_jump, control_dt=args.control_dt,
                  duration=args.duration, friction=args.friction, capsule_collision=not args.no_capsule_collision,
                  evaluate_after=not args.no_evaluate)
        for k in ("hold", "settle", "approach_ring"):
            if getattr(args, k) is not None:
                kw[k] = getattr(args, k)
        track_recorded(args.task, args.robot, args.method, args.out, args.candidate, run_dir=args.run_dir, **kw)
        return 0
    if args.archive:
        arch = Path(args.archive)
    elif args.archive_dir:
        arch = archive_path(args.task, args.robot, args.archive_dir, args.archive_tag)
    else:
        ap.error("--archive or --archive-dir")
    from leaptrack.rollout import track_archive
    gains = gains_for(args.task, args.robot, gain_overrides(args))
    kw = dict(problem_yaml=args.problem_yaml, gains=gains, verbose=not args.quiet,
              clock=args.clock or "touchdown", early_jump=args.early_jump, control_dt=args.control_dt,
              duration=args.duration, friction=args.friction, capsule_collision=not args.no_capsule_collision,
              evaluate_after=not args.no_evaluate)
    if args.hold is not None:
        kw["hold"] = args.hold
    if args.settle is not None:
        kw["settle"] = args.settle
    if args.approach_ring is not None:
        kw["approach_ring"] = args.approach_ring
    track_archive(args.task, args.robot, arch, args.out, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
