"""Thin Python wrapper around the C++ solver `leap_solve`.

    from leap_solve import solve, load_archive
    r = solve("problems/hopper2d_jump_fence.yaml",
              overrides={"solver.split": 8, "solver.hgrade": 0.6, "output.stem": "results/sol_hopper2d"},
              threads=12, cpus="0-11")
    print(r.met, r.gate, r.fine, r.steps, r.wall)
    sol = load_archive(r.archive)          # dict of numpy arrays + the meta_json blob
    Q0 = sol["segments/00/Q"]              # joint positions of element 0 at its nodes

The wrapper only builds the command line, runs the binary, parses the `[unified]` summary
line and reads the HDF5 archive; every number comes from the C++ solver.

    python leap_solve.py <problem.yaml> [key=value ...]     # command-line use
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

HERE = Path(__file__).resolve().parent
DEFAULT_BIN = HERE.parent / "build" / "leap_solve"


@dataclass
class SolveResult:
    returncode: int
    log: str
    summary: Dict[str, str] = field(default_factory=dict)
    archive: Optional[Path] = None

    @property
    def met(self) -> bool:
        return self.summary.get("met") == "1"

    @property
    def gate(self) -> float:
        return float(self.summary.get("gate", "nan"))

    @property
    def fine(self) -> float:
        return float(self.summary.get("fine", "nan"))

    @property
    def steps(self) -> int:
        return int(self.summary.get("total", "-1"))

    @property
    def wall(self) -> float:
        return float(self.summary.get("solvewall", "nan"))


def _format_override(key: str, value) -> str:
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    if isinstance(value, (list, tuple)):
        return f"{key}={','.join(str(v) for v in value)}"
    return f"{key}={value}"


def solve(problem_yaml: Union[str, Path], overrides: Optional[Dict[str, object]] = None, threads: int = 0,
          cpus: Optional[str] = None, binary: Union[str, Path] = DEFAULT_BIN, log_path: Optional[Union[str, Path]] = None,
          echo: bool = False) -> SolveResult:
    """Run one solve. `overrides` are dotted yaml keys (`solver.split`, `unified.levels`, ...)."""
    binary = Path(binary)
    if not binary.exists():
        raise FileNotFoundError(f"leap_solve binary not found at {binary} (build it with cmake)")
    cmd = []
    if cpus:
        cmd += ["taskset", "-c", cpus]
    cmd += [str(binary), str(problem_yaml)]
    cmd += [_format_override(k, v) for k, v in (overrides or {}).items()]
    env = dict(os.environ)
    if threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if echo:
            sys.stdout.write(line)
    proc.wait()
    log = "".join(lines)
    if log_path:
        Path(log_path).write_text(log)
    result = SolveResult(returncode=proc.returncode, log=log)
    for line in lines:
        if line.startswith("[unified]"):
            result.summary = dict(re.findall(r"(\w+)=(\S+)", line))
    m = re.search(r"solution archive written to (\S+)", log)
    if m:
        result.archive = Path(m.group(1))
    return result


def load_archive(path: Union[str, Path]) -> Dict[str, object]:
    """Every numeric dataset of the archive as numpy arrays, plus `meta` (the parsed JSON)."""
    import h5py  # noqa: WPS433 (optional dependency)

    out: Dict[str, object] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.dtype.kind in "fiub":
                    out[name] = obj[()]
                elif name == "meta_json":
                    out["meta"] = json.loads(obj[()].decode())
        f.visititems(visit)
    return out


def main(argv: Iterable[str]) -> int:
    argv = list(argv)
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    overrides = {}
    for kv in argv[1:]:
        k, _, v = kv.partition("=")
        overrides[k] = v
    r = solve(argv[0], overrides, echo=True)
    if r.summary:
        print(f"met={r.met} gate={r.gate:.3e} fine={r.fine:.3e} steps={r.steps} wall={r.wall:.1f}s archive={r.archive}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
