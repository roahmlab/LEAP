#!/usr/bin/env python
"""Are two tracked case directories the same rollout?

    python compare_rollouts.py <case_a> <case_b> [<case_c> ...]   # every case against the first
    python compare_rollouts.py --reps <dir_with_rep1..repK>

IDENTICAL means: every array of rollout.npz, trajectory.npz and evaluation.npz has the same
shape and bits (NaN == NaN), and evaluation.json is equal. Ignored, because they name the
run rather than the rollout: in trajectory.npz's metadata a path into the case directory
itself (the copied problem yaml); in that metadata and in evaluation.json the location of
the solution archive (`archive`, `archive_dir`: the same file reached by another path);
and the controller profile's old name `wp4`, read as `impact_aware`
(`controllers.PROFILE_ALIASES`). The files themselves are not compared byte for byte (the
npz containers carry zip timestamps); metrics.json is not compared (it holds the wall
time).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FILES = ("rollout.npz", "trajectory.npz", "evaluation.npz")


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def same_array(x, y) -> bool:
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


PATH_KEYS = ("archive", "archive_dir")
PROFILE_ALIASES = {"wp4": "impact_aware"}


def normalize(d: dict) -> dict:
    """Drop the archive's location; read the profile's old name as the new one."""
    out = {k: v for k, v in d.items() if k not in PATH_KEYS}
    g = out.get("gains")
    if isinstance(g, dict) and g.get("config_profile") in PROFILE_ALIASES:
        out["gains"] = dict(g, config_profile=PROFILE_ALIASES[g["config_profile"]])
    return out


def strip_case(meta: dict, case: Path) -> dict:
    root = str(case.resolve())
    return normalize({k: v for k, v in meta.items() if not (isinstance(v, str) and v.startswith(root))})


def compare(a: Path, b: Path):
    diffs = []
    present = [n for n in FILES + ("evaluation.json",) if (a / n).exists() or (b / n).exists()]
    if "rollout.npz" not in present:
        return [f"no rollout.npz in {a} or {b}: nothing to compare"]
    for name in FILES:
        pa, pb = a / name, b / name
        if not pa.exists() and not pb.exists():
            continue
        if not (pa.exists() and pb.exists()):
            diffs.append(f"{name}: missing on one side")
            continue
        xa, xb = arrays(pa), arrays(pb)
        for k in sorted(set(xa) | set(xb)):
            if k not in xa or k not in xb:
                diffs.append(f"{name}:{k} missing on one side")
            elif k == "meta" and xa[k].dtype.kind == "U":
                # trajectory metadata: JSON; a path into the case directory itself
                # (the copied problem yaml) names the run, not the rollout
                if strip_case(json.loads(str(xa[k])), a) != strip_case(json.loads(str(xb[k])), b):
                    diffs.append(f"{name}:{k}")
            elif not same_array(xa[k], xb[k]):
                diffs.append(f"{name}:{k}")
    ja, jb = a / "evaluation.json", b / "evaluation.json"
    if ja.exists() or jb.exists():
        if not (ja.exists() and jb.exists()) or normalize(json.loads(ja.read_text())) != normalize(json.loads(jb.read_text())):
            diffs.append("evaluation.json")
    return diffs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--reps":
        root = Path(argv[1])
        cases = sorted((p for p in root.iterdir() if p.name.startswith("rep") and p.name[3:].isdigit()),
                       key=lambda p: int(p.name[3:]))
    else:
        cases = [Path(x) for x in argv]
    if len(cases) < 2:
        raise SystemExit("need at least two case directories")
    bad = 0
    for c in cases[1:]:
        d = compare(cases[0], c)
        bad += bool(d)
        print(f"{'IDENTICAL' if not d else 'DIFFERENT':10s} {cases[0]} vs {c}" + (f": {len(d)} items differ, e.g. {d[:4]}" if d else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
