"""leaptrack: MuJoCo whole-body tracking of LEAP solution archives (the paper's
Table III protocol), self-contained.

Runtime dependencies:
numpy, h5py, pyyaml, mujoco, pinocchio, proxsuite (matplotlib for the report
plots). Robot descriptions: `new_src/robots/`, `new_src/assets/`.

    from leaptrack.rollout import track_archive
    metrics = track_archive(task, robot, archive, out_dir)
    from leaptrack.score import terminal_ratio, THRESHOLDS, JUDGE_AT_T
    tag, ratio, worst, per_metric = terminal_ratio(out_dir / "evaluation.json", task, robot)
    from leaptrack.controllers import TASKS, TASK_ROBOTS

This package `__init__` imports nothing, so `leaptrack.score` (standard
library only) can be used on a machine without the simulation stack.
"""

# One thread per math library (see `leaptrack.determinism`): these variables only
# take effect if set before numpy is first imported, and importing any leaptrack
# module runs this file first. Standard library only.
import os as _os
import sys as _sys

if "numpy" not in _sys.modules:
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_k] = "1"
