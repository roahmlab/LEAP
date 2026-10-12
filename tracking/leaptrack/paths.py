"""Where the robot descriptions live for the tracking package.

The robot yamls and URDFs are `new_src/robots/` and `new_src/assets/`: every path in
this package is resolved by BASENAME against those two directories (a problem yaml's
`robot:` line and an archive's `robot.urdf` may be relative or absolute).
"""
from __future__ import annotations

from pathlib import Path

NEW_SRC = Path(__file__).resolve().parents[2]        # new_src/
ROBOTS_DIR = NEW_SRC / "robots"
ASSETS_DIR = NEW_SRC / "assets"


def robot_yaml_path(name_or_path) -> Path:
    """`new_src/robots/<basename>.yaml` for a robot name or any yaml path."""
    p = Path(str(name_or_path))
    name = p.stem if p.suffix == ".yaml" else p.name
    return ROBOTS_DIR / f"{name}.yaml"


def urdf_path(name_or_path) -> Path:
    """`new_src/assets/<basename>.urdf` for a robot name or any urdf path."""
    p = Path(str(name_or_path))
    name = p.stem if p.suffix == ".urdf" else p.name
    return ASSETS_DIR / f"{name}.urdf"
