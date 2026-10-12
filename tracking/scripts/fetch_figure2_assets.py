#!/usr/bin/env python
"""Download the visual mesh files of Fig. 2 and verify them against figure2/assets/SHA256SUMS.

    python scripts/fetch_figure2_assets.py              # into figure2/assets/
    python scripts/fetch_figure2_assets.py --dest DIR   # into DIR (then: figure2.py --assets DIR)
    python scripts/fetch_figure2_assets.py --check      # verify only, download nothing

Files: the entries of figure2/assets/SHA256SUMS (<pack>/<path>), written to <dest>/<pack>/<path>:
    go2_menagerie/    google-deepmind/mujoco_menagerie @ 0059d4335f8156206f63a35662313385f7ad6d74, unitree_go2/
    h1_menagerie/     google-deepmind/mujoco_menagerie @ 0059d4335f8156206f63a35662313385f7ad6d74, unitree_h1/
    talos_menagerie/  google-deepmind/mujoco_menagerie @ 0059d4335f8156206f63a35662313385f7ad6d74, pal_talos/
    g1_description/   unitreerobotics/unitree_ros @ 5994d4faef0a9cadd3287f8de0199a67eeb2a259, robots/g1_description/
from https://raw.githubusercontent.com/<repo>/<commit>/<folder>/<path>. Files already present
and verified are skipped. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "figure2" / "assets"           # new_src/tracking/figure2/assets
SUMS = ASSETS / "SHA256SUMS"
MENAGERIE = ("https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
             "0059d4335f8156206f63a35662313385f7ad6d74")
UNITREE_ROS = ("https://raw.githubusercontent.com/unitreerobotics/unitree_ros/"
               "5994d4faef0a9cadd3287f8de0199a67eeb2a259")
SOURCES = {
    "go2_menagerie": f"{MENAGERIE}/unitree_go2",
    "h1_menagerie": f"{MENAGERIE}/unitree_h1",
    "talos_menagerie": f"{MENAGERIE}/pal_talos",
    "g1_description": f"{UNITREE_ROS}/robots/g1_description",
}
RETRIES = 3
TIMEOUT_S = 60
JOBS = 4


def read_sums(path: Path) -> list[tuple[str, str]]:
    """[(relative path, sha256)] of a sha256sum-format file."""
    entries = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        if rel.split("/", 1)[0] not in SOURCES:
            raise SystemExit(f"{path}: no source for {rel}")
        entries.append((rel, digest.lower()))
    return entries


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def url(rel: str) -> str:
    pack, sub = rel.split("/", 1)
    return f"{SOURCES[pack]}/{quote(sub)}"


def check(rel: str, digest: str, dest: Path) -> str:
    target = dest / rel
    if not target.is_file():
        return "missing"
    return "ok" if sha256(target) == digest else "sha256 mismatch"


def fetch(rel: str, digest: str, dest: Path) -> str:
    """Download one file unless it is present and verified; place it only once verified."""
    target = dest / rel
    if target.is_file() and sha256(target) == digest:
        return "present"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    error = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url(rel), timeout=TIMEOUT_S) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out)
            error = None
            break
        except (urllib.error.URLError, OSError) as exc:      # HTTPError is a URLError
            error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                break
            time.sleep(2 ** attempt)
    if error is not None:
        part.unlink(missing_ok=True)
        return f"download failed ({error})"
    if sha256(part) != digest:
        part.unlink()
        return "sha256 mismatch"
    part.replace(target)
    return "fetched"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=ASSETS,
                    help=f"asset root to fill (default: {ASSETS})")
    ap.add_argument("--check", action="store_true", help="verify the files under --dest, download nothing")
    args = ap.parse_args(argv)
    entries = read_sums(SUMS)
    dest = args.dest.resolve()
    work = check if args.check else fetch
    with ThreadPoolExecutor(max_workers=1 if args.check else JOBS) as pool:
        results = list(pool.map(lambda e: (e[0], work(e[0], e[1], dest)), entries))
    failed = [(rel, status) for rel, status in results if status not in ("ok", "present", "fetched")]
    for rel, status in results:
        if status == "fetched":
            print(f"fetched {rel}", flush=True)
    for rel, status in failed:
        print(f"FAILED {rel}: {status}", file=sys.stderr, flush=True)
    fetched = sum(status == "fetched" for _, status in results)
    good = len(results) - len(failed)
    print(f"{good}/{len(results)} files verified under {dest}"
          + ("" if args.check else f" ({fetched} downloaded, {good - fetched} already present)"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
