#!/usr/bin/env python
"""Make the paper's Fig. 2 (tracking time-lapse, fence and terrain) from tracked rollouts.

    PY=<venv with mujoco, pinocchio, h5py, PyYAML, numpy, Pillow>/bin/python
    export MUJOCO_GL=egl
    $PY fetch_figure2_assets.py                     # visual meshes into figure2/assets/ (once)
    $PY track.py --task fence   --all --archive-dir records/fence   --out tracked/fence
    $PY track.py --task terrain --all --archive-dir records/terrain --out tracked/terrain
    $PY figure2.py --fence tracked/fence --terrain tracked/terrain --out fig2
    $PY figure2.py --fence tracked/fence --terrain tracked/terrain --out fig2 --reuse

System requirement: pdfinfo (poppler-utils) on PATH.

Inputs:
    --fence, --terrain   the --out folders of `track.py --task <task> --all`
                         (<robot>/rollout.npz, metrics.json, problem yaml)
    archives             the `archive` each case's metrics.json records, or
                         sol_<task>_<robot>[_<tag>].h5 in --fence-archive-dir / --terrain-archive-dir
                         (tag: --fence-archive-tag / --terrain-archive-tag)
    --assets             visual mesh root (default: figure2/assets/, filled by
                         fetch_figure2_assets.py), checked against figure2/assets/SHA256SUMS

Outputs, under --out:
    combined/leap_tracking_timelapse_combined.{png,pdf}            Fig. 2
    fence/leap_fence_tracking_timelapse_transposed.{png,pdf}
    terrain/leap_terrain_tracking_timelapse_transposed.{png,pdf}
    src_fence/{isometric,side}_dense_gradient/, src_terrain/{isometric,side}/
        per-view renders: per robot PNG, PDF and JSON sidecar; the view's combined
        figure; manifest.json; validation.json
    a manifest.json in each figure folder

Options: --figures fence terrain combined (default: all three); --reuse composes from the
renders under --out and renders nothing (fails if one is missing or stale); --force-render
re-renders every view.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent                      # new_src/tracking
# First on the path, so `figure2` is the package new_src/tracking/figure2/, not this file.
sys.path.insert(0, str(PKG_ROOT))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MUJOCO_GL", "egl")

FIGURES = ("fence", "terrain", "combined")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fence", type=Path, required=True,
                    help="tracked fence suite: the --out folder of track.py --task fence --all")
    ap.add_argument("--terrain", type=Path, required=True,
                    help="tracked terrain suite: the --out folder of track.py --task terrain --all")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--fence-archive-dir", type=Path,
                    help="read sol_fence_<robot>[_<tag>].h5 here instead of each case's metrics.json archive")
    ap.add_argument("--fence-archive-tag")
    ap.add_argument("--terrain-archive-dir", type=Path,
                    help="read sol_terrain_<robot>[_<tag>].h5 here instead of each case's metrics.json archive")
    ap.add_argument("--terrain-archive-tag")
    ap.add_argument("--assets", type=Path, default=PKG_ROOT/"figure2"/"assets",
                    help="visual mesh root (default: figure2/assets/, filled by fetch_figure2_assets.py)")
    ap.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES),
                    help="which figures to make (default: all three; 'combined' is Fig. 2)")
    ap.add_argument("--reuse", action="store_true",
                    help="compose from the saved renders under --out; fail if one is missing or stale")
    ap.add_argument("--force-render", action="store_true",
                    help="re-render every view even when validated renders exist")
    args = ap.parse_args()
    if args.reuse and args.force_render:
        ap.error("--reuse and --force-render exclude each other")

    from figure2 import combined, fence, terrain
    from figure2.scene import check_visual_assets
    from figure2.style import load_style

    if not args.reuse:
        # Check every mesh pack before rendering.
        robots = [r for r, _ in fence.LAYOUT["rows"]] + [r for r, _ in terrain.LAYOUT["rows"]]
        root = load_style(assets=args.assets)["scene"]["visual_assets"]
        try:
            check_visual_assets(robots, root)
        except FileNotFoundError as exc:
            sys.exit(f"[figure2] {exc}")

    out = args.out.resolve()
    assets = ["--assets", str(args.assets)]
    flags = ["--reuse"] if args.reuse else []

    def archive(task):
        d, tag = getattr(args, f"{task}_archive_dir"), getattr(args, f"{task}_archive_tag")
        return (["--archive-dir", str(d)] if d else []) + (["--archive-tag", tag] if tag else [])

    t_all = time.time()
    if "fence" in args.figures:
        t0 = time.time()
        fence.run(fence.build_parser().parse_args(
            ["--data", str(args.fence), "--source", str(out/"src_fence"), "--out", str(out/"fence")]
            + archive("fence") + assets + flags + (["--force-render"] if args.force_render else [])))
        print(f"[figure2] fence figure: {time.time()-t0:.0f} s", flush=True)
    if "terrain" in args.figures:
        t0 = time.time()
        terrain.run(terrain.build_parser().parse_args(
            ["--data", str(args.terrain), "--source", str(out/"src_terrain"), "--out", str(out/"terrain")]
            + archive("terrain") + assets + flags + (["--force-render"] if args.force_render else [])))
        print(f"[figure2] terrain figure: {time.time()-t0:.0f} s", flush=True)
    if "combined" in args.figures:
        t0 = time.time()
        # Views rendered above in this run are not forced again.
        force = args.force_render and not {"fence", "terrain"} <= set(args.figures)
        rename = lambda xs, task: [x.replace("--archive", f"--{task}-archive") for x in xs]  # noqa: E731
        combined.run(combined.build_parser().parse_args(
            ["--fence-data", str(args.fence), "--terrain-data", str(args.terrain),
             "--fence-source", str(out/"src_fence"), "--terrain-source", str(out/"src_terrain"),
             "--out", str(out/"combined")]
            + rename(archive("fence"), "fence") + rename(archive("terrain"), "terrain")
            + assets + flags + (["--force-render"] if force else [])))
        print(f"[figure2] combined figure (Fig. 2): {time.time()-t0:.0f} s", flush=True)
    print(f"[figure2] done in {time.time()-t_all:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
