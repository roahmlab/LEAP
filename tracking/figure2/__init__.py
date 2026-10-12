"""figure2: renders the paper's Fig. 2, the MuJoCo tracking time-lapse of the fence and
terrain suites. Entry point: `scripts/figure2.py`.
"""
import os as _os

# Headless rendering.
_os.environ.setdefault("MUJOCO_GL", "egl")
