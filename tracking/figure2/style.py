"""Loads `style.json` and resolves its package-relative paths (`text.font`,
`scene.visual_assets`).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE/'style.json'


def _resolve(path, base=HERE):
    p = Path(path)
    return str(p if p.is_absolute() else (base/p).resolve())


def load_style(path=DEFAULT_CONFIG, *, overlay=None, assets=None):
    """The style at `path` with its package-relative paths made absolute.

    `overlay` updates `tracking_overlay` (the tracked folders and archive lookup of
    one task); `assets` replaces `scene.visual_assets` (the visual mesh root).
    """
    cfg = json.loads(Path(path).read_text())
    cfg['text']['font'] = _resolve(cfg['text']['font'])
    if assets is not None:
        cfg['scene']['visual_assets'] = str(Path(assets).resolve())
    else:
        cfg['scene']['visual_assets'] = _resolve(cfg['scene']['visual_assets'])
    if overlay:
        cfg['tracking_overlay'].update(overlay)
    return cfg
