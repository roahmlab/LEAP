#!/usr/bin/env python3
"""Shared paths, robot table and helpers of the terrain along-the-way experiment."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TERRAIN = HERE.parent                      # comparisons/aligator_unified_terrain
SCRIPTS = TERRAIN / 'scripts'
ROOT = TERRAIN.parents[1]                  # src/
PY = Path(os.environ.get('ALIGATOR_PYTHON', sys.executable))
DEFAULT_ROOT = Path(os.environ.get('ALIGATOR_TERRAIN_OUT', ROOT.parent / 'results/terrain'))
LEAP_BUDGET = Path(os.environ.get('LEAP_BUDGET_EXPERIMENT_DIR', ROOT.parent / 'leap_budget_experiment_uneven_terrain'))
# The terrain benchmark's robots. 'h1' is solved on the h1 problem of
# new_src/problems/h1_terrain_walk.yaml (driver key h1_ap15_ank07_att02_torso05) and
# 'g1_29dof_repaired' on the g1 problem of new_src/problems/g1_29dof_terrain_walk.yaml.
ROBOTS = ('hopper2d_foot', 'go2', 'g1_29dof', 'g1_29dof_repaired', 'talos', 'h1')
# Dataset robot name -> the terrain driver's robot key (scripts/seed.py: problem yaml + seed
# archive). Identity unless listed.
DRIVER_ROBOT = {'h1': 'h1_ap15_ank07_att02_torso05'}


def driver_robot(robot):
    return DRIVER_ROBOT.get(robot, robot)
STRICT_GATE = 1e-3       # LEAP's criterion: max violation at nodes AND on the dt=0.01 grid
LOOSE_GATE = 2e-3        # the "around 1e-3" reading adopted by the fence experiment; reported alongside
METRIC_COLUMNS = ['native_iteration', 'raw_s', 'solver_s', 'capture_cumulative_s', 'native_cost',
                  'native_primal_infeasibility', 'native_dual_infeasibility', 'capture_step_s']


def bootstrap():
    """Import paths: this folder, then the terrain driver scripts (seed / problem / public_problem /
    verify_public / export_public), then the comparisons package root (common.*) and the repo root
    (pyleap). The fence driver directory, which shadows several of those module names, is only
    APPENDED (seed.py does that itself for the shared _public_* extensions)."""
    os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    front = [str(HERE), str(SCRIPTS)]
    back = [str(ROOT / 'comparisons'), str(ROOT), str(ROOT / 'comparisons/aligator_unified_fence/scripts')]
    for p in reversed(front):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for p in back:
        if p not in sys.path:
            sys.path.append(p)


def reference_inputs(robot):
    """Benchmark input files of a robot (problem yaml, robot yaml, URDF, LEAP cold seed archive)."""
    bootstrap()
    import yaml
    from seed import problem_yaml, seed_archive
    robot = driver_robot(robot)
    yaml_path = Path(problem_yaml(robot)).resolve()
    doc = yaml.safe_load(open(yaml_path))
    robot_yaml = (yaml_path.parent / doc['robot']).resolve()
    urdf = (robot_yaml.parent / yaml.safe_load(open(robot_yaml))['urdf']).resolve()
    seed = seed_archive(robot)
    return dict(problem_yaml=yaml_path, robot_yaml=robot_yaml, urdf=urdf, seed=seed)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def clean(v):
        if hasattr(v, 'tolist'):
            return clean(v.tolist())
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and v != v:
            return None
        if isinstance(v, Path):
            return str(v)
        return v
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(clean(value), indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def cpu_list(spec):
    """'16-27,30' -> [16,...,27,30]."""
    out = []
    for part in str(spec).split(','):
        a, b = (part.split('-') + [part])[:2]
        out.extend(range(int(a), int(b) + 1))
    return out


def physical_core_ids(cpus):
    keys = []
    for cpu in cpus:
        top = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        keys.append(((top / 'physical_package_id').read_text().strip(), (top / 'core_id').read_text().strip()))
    return keys


def machine_record(cpus):
    import platform
    import subprocess
    rec = dict(hostname=platform.node(), platform=platform.platform(), python=sys.version,
               loadavg_1_5_15=os.getloadavg(), cpus=list(cpus), physical_cores=physical_core_ids(cpus),
               env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_DYNAMIC')})
    try:
        rec['lscpu'] = subprocess.check_output(['lscpu'], text=True)
    except Exception as exc:  # noqa: BLE001
        rec['lscpu_error'] = repr(exc)
    try:
        rec['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        rec['git_status_short'] = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    except Exception as exc:  # noqa: BLE001
        rec['git_error'] = repr(exc)
    return rec


def phase_calls(phase):
    """(call record, trace dict) for every traced call of a phase directory."""
    record = read_json(Path(phase) / 'result.json')
    return record, [c for c in record['calls'] if 'trace' in c]
