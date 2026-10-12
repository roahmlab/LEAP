"""Grafts the visual mesh geoms of a donor MuJoCo model onto the same-named bodies of a
MuJoCo spec, as non-colliding geoms (rendering only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

VISUAL_GROUP = 2      # rendered by default (MjvOption shows groups 0-2)
HIDDEN_GROUP = 3      # collision primitives: present but not rendered


@dataclass(frozen=True)
class MeshAsset:
    name: str
    file: Path
    scale: tuple


@dataclass(frozen=True)
class VisualGeom:
    body: str             # donor body name == target body name
    mesh: str
    pos: tuple
    quat: tuple           # (w, x, y, z)
    rgba: tuple


@dataclass(frozen=True)
class VisualPack:
    xml: Path
    meshes: dict          # mesh name -> MeshAsset
    geoms: tuple          # tuple[VisualGeom, ...]

    def bodies(self) -> set:
        return {g.body for g in self.geoms}


def _mesh_files(spec: mujoco.MjSpec, xml: Path) -> dict:
    """Mesh name -> (absolute file, scale) from the donor SPEC (the compiled
    model keeps only the name, which MuJoCo derives from the file stem)."""
    meshdir = Path(spec.compiler.meshdir or ".")
    base = xml.parent / meshdir if not meshdir.is_absolute() else meshdir
    out = {}
    for m in spec.meshes:
        if not m.file:
            continue
        f = Path(m.file)
        f = f if f.is_absolute() else base / f
        name = m.name or f.stem      # MuJoCo's default name is the file stem
        out[name] = MeshAsset(name=name, file=f, scale=tuple(np.asarray(m.scale, float)))
    return out


def load_pack(xml_path: str | Path) -> VisualPack:
    """Read a donor model's non-colliding mesh geoms.

    Placement is taken from the SPEC (i.e. the XML), never from the compiled
    model: the compiler re-centres every mesh on its own CoM and folds the
    compensating offset into `geom_pos`/`geom_quat`. The compiled model is used
    only to classify which (body, mesh) pairs
    are the donor's *visual* geoms, since that follows from its default classes.
    """
    xml = Path(xml_path).resolve()
    if not xml.exists():
        raise FileNotFoundError(
            f"visual pack {xml} not found -- fetch it with scripts/fetch_figure2_assets.py")
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.compiler.discardvisual = False
    files = _mesh_files(spec, xml)
    rgba_of = {m.name: tuple(np.asarray(m.rgba, float)) for m in spec.materials}
    model = spec.compile()        # resolves default classes (visual vs collision)

    visual_pairs = set()
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[gid] or model.geom_conaffinity[gid]:
            continue              # a colliding mesh is part of the donor's physics
        visual_pairs.add((
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                              int(model.geom_bodyid[gid])),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                              int(model.geom_dataid[gid]))))

    geoms = []
    seen_placements = set()
    for g in spec.geoms:
        body = g.parent.name if g.parent is not None else ""
        if not g.meshname or (body, g.meshname) not in visual_pairs:
            continue
        # Some donors (Unitree's g1_23dof.xml) carry a visual AND a collision
        # geom of the SAME mesh at the SAME pose on one body; the pair-based
        # classification above cannot tell them apart, so drop exact-duplicate
        # placements -- grafting one copy renders identically.
        placement = (body, g.meshname, tuple(np.asarray(g.pos, float)),
                     tuple(np.asarray(g.quat, float)))
        if placement in seen_placements:
            continue
        seen_placements.add(placement)
        if g.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT:
            raise NotImplementedError(
                f"{xml}: geom on '{body}' uses a non-quat orientation "
                f"({g.alt.type}); only pos/quat placement is grafted")
        if g.meshname not in files:
            raise KeyError(f"{xml}: mesh '{g.meshname}' has no resolvable file")
        rgba = rgba_of.get(g.material, tuple(np.asarray(g.rgba, float)))
        geoms.append(VisualGeom(body=body, mesh=g.meshname,
                                pos=tuple(np.asarray(g.pos, float)),
                                quat=tuple(np.asarray(g.quat, float)),
                                rgba=rgba))
    if not geoms:
        raise ValueError(f"{xml}: no non-colliding mesh geoms to graft")
    used = {g.mesh for g in geoms}
    return VisualPack(xml=xml, meshes={k: v for k, v in files.items() if k in used},
                      geoms=tuple(geoms))


def attach(spec: mujoco.MjSpec, pack: VisualPack, *,
           rgba: tuple | None = None,
           prefix: str = "vis_",
           hide_existing: bool = True) -> int:
    """Add `pack`'s mesh geoms to same-named bodies of `spec`, in place.

    `rgba` overrides every material colour (used for the translucent ghost).
    `hide_existing` moves the pre-existing geoms of the DRESSED bodies to the
    non-rendered group so the collision primitives stop poking through the
    meshes; contacts are unaffected (group is a rendering filter only) and geoms
    on other bodies -- notably the world's floor and props -- are left alone.
    Returns the number of geoms added. Raises if no body names match.
    """
    body_by_name = {b.name: b for b in spec.bodies}
    matched = pack.bodies() & set(body_by_name)
    if not matched:
        raise KeyError(f"visual pack {pack.xml.name} shares no body names with the "
                       f"model (pack: {sorted(pack.bodies())})")

    if hide_existing:
        for name in matched:
            for g in body_by_name[name].geoms:
                g.group = HIDDEN_GROUP

    for name in sorted({g.mesh for g in pack.geoms if g.body in matched}):
        asset = pack.meshes[name]
        m = spec.add_mesh()
        m.name = prefix + name
        m.file = str(asset.file)
        m.scale = list(asset.scale)

    n = 0
    names_used: set[str] = set()
    for g in pack.geoms:
        if g.body not in matched:
            continue
        geom = body_by_name[g.body].add_geom()
        name = f"{prefix}{g.body}_{g.mesh}"
        k = 2
        while name in names_used:   # same mesh twice on one body (distinct pose)
            name = f"{prefix}{g.body}_{g.mesh}_{k}"
            k += 1
        names_used.add(name)
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = prefix + g.mesh
        geom.pos = list(g.pos)
        geom.quat = list(g.quat)
        geom.rgba = list(rgba if rgba is not None else g.rgba)
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = VISUAL_GROUP
        n += 1

    missing = sorted(pack.bodies() - matched)
    if missing:
        import warnings
        warnings.warn(f"visual pack bodies absent from the model (skipped): {missing}")
    return n
