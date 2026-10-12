"""Builds the MuJoCo scene of a timelapse: the robot dressed for rendering (visual meshes,
floor, lights, contact geoms), the problem's boxes, one copy of the robot per drawn pose,
and the bounds the camera is fitted to. Checks the visual mesh files before rendering.
"""
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from leaptrack.paths import ROBOTS_DIR  # noqa: E402
from leaptrack.robot import load_robot  # noqa: E402
from leaptrack.scene import (FOOT_BOX_HALF_HEIGHT, FOOT_RADIUS, GROUND_BIT, OBSTACLE_BIT,  # noqa: E402
                             SETTINGS, SOLE_MARGIN, STONE_BIT0, RobotSim, SceneBox, _quat_from_mat,
                             align_inertias_to_pinocchio, apply_contact_geom, apply_options,
                             frame_anchor, patch_urdf, robot_yaml)
from . import visual_meshes  # noqa: E402


FETCH_SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'fetch_figure2_assets.py'
DEFAULT_ASSETS = Path(__file__).resolve().parent/'assets'


def fetch_command(assets):
    """The command that downloads the visual mesh files into the asset root `assets`."""
    dest = '' if Path(assets).resolve() == DEFAULT_ASSETS else f' --dest {Path(assets).resolve()}'
    return f'python {FETCH_SCRIPT}{dest}'


def visual_pack_xml(name, assets):
    """The donor model of `name`'s visual meshes under the asset root, or None.

    `sim.visual_mesh_xml` in `new_src/robots/<name>.yaml` names it relative to the
    robots folder (`../assets/go2_menagerie/go2.xml`); its folder and file name are
    looked up under `assets` (`<assets>/go2_menagerie/go2.xml`).
    """
    pack = (robot_yaml(name).get('sim', {}) or {}).get('visual_mesh_xml')
    if pack is None or (isinstance(pack, str) and pack.strip().lower() in ('', 'none')):
        return None
    pack = Path(pack)
    return Path(assets)/pack.parent.name/pack.name


def check_visual_assets(robots, assets, sums=Path(__file__).resolve().parent/'assets'/'SHA256SUMS'):
    """Check that the visual packs `robots` need are present before anything is rendered.

    Raises if a pack's donor model or one of the files `sums` lists for it is missing
    under `assets`.
    """
    assets, packs, missing = Path(assets), set(), []
    for robot in robots:
        xml = visual_pack_xml(robot, assets)
        if xml is not None:
            packs.add(xml.parent.name)
            if not xml.exists():
                missing.append(str(xml))
    for line in (sums.read_text().splitlines() if sums.exists() else []):
        rel = line.split(maxsplit=1)[1]
        if rel.split('/')[0] not in packs:
            continue
        if not (assets/rel).exists():
            missing.append(str(assets/rel))
    if missing:
        names = sorted({str(Path(m).relative_to(assets)) for m in missing})
        raise FileNotFoundError(f'{len(names)} visual mesh files missing under {assets} ('
                                + ', '.join(names[:3]) + (', ...' if len(names) > 3 else '')
                                + f'); fetch them with: {fetch_command(assets)}')


def build(name, *, scene_boxes=None, planarlock=(), settings=SETTINGS, assets,
          foot_radius=FOOT_RADIUS, foot_box_half_height=FOOT_BOX_HALF_HEIGHT,
          sole_margin=SOLE_MARGIN, floor_extent=12.0):
    """Compile `robots/<name>.yaml`'s robot dressed for rendering: the donor visual
    meshes when the yaml names a pack (`visual_pack_xml`), else the URDF collision
    primitives in a neutral gray; floor, lights, the scene boxes and the contact geoms.
    """
    robot = load_robot(ROBOTS_DIR/f"{name}.yaml")
    doc = robot_yaml(name)
    contact_dims = {c["frame"]: int(c["dim"]) for c in doc["contacts"]}
    foot_boxes = {c["frame"]: c.get("foot_box") for c in doc["contacts"]}

    spec = mujoco.MjSpec.from_string(patch_urdf(robot.urdf_path))
    apply_options(spec, settings)
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = True

    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.znear = 0.005
    pack = None
    pack_xml = visual_pack_xml(name, assets)
    if pack_xml is not None:
        if not pack_xml.exists():
            raise FileNotFoundError(f"{name}: visual pack {pack_xml} missing; fetch it with: "
                                    f"{fetch_command(assets)}")
        pack = visual_meshes.load_pack(pack_xml)

    # every URDF geom: non-colliding (the optimizer models no such contacts)
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
        if pack is not None:
            g.group = visual_meshes.HIDDEN_GROUP
        else:
            g.rgba = [0.62, 0.64, 0.68, 1.0]
            g.group = 2

    # floor
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [floor_extent, floor_extent, 0.1]
    apply_contact_geom(floor, settings)
    floor.contype = GROUND_BIT
    floor.conaffinity = GROUND_BIT
    tex = spec.add_texture()
    tex.name = "grid"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.width = tex.height = 512
    tex.rgb1 = [0.76, 0.78, 0.81]
    tex.rgb2 = [0.58, 0.61, 0.66]
    mat = spec.add_material()
    mat.name = "grid"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [24, 24]
    mat.reflectance = 0.0
    floor.material = "grid"
    light = spec.worldbody.add_light()
    light.pos = [1.0, -2.0, 3.0]
    light.dir = [-0.3, 0.6, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.castshadow = True
    light.diffuse = [0.8, 0.8, 0.8]
    light2 = spec.worldbody.add_light()
    light2.pos = [-1.0, 2.0, 3.0]
    light2.dir = [0.3, -0.6, -1.0]
    light2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light2.castshadow = False
    light2.diffuse = [0.35, 0.35, 0.35]

    # scene boxes: obstacles (fence) on the ground and obstacle bits, stones on a
    # bit of their own each
    boxes = []
    box_bits = {}
    n_stone = 0
    for sb in (scene_boxes or ()):
        if isinstance(sb, dict):
            sb = SceneBox(str(sb["name"]), np.asarray(sb["center"], float),
                          np.asarray(sb["half"], float), tuple(sb.get("rgba", SceneBox.rgba)),
                          bool(sb.get("collide_capsules", True)))
        boxes.append(sb)
        g = spec.worldbody.add_geom()
        g.name = f"box_{sb.name}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [float(x) for x in sb.center]
        g.size = [float(x) for x in sb.half]
        apply_contact_geom(g, settings)
        if sb.collide_capsules:
            g.contype = GROUND_BIT | OBSTACLE_BIT
            g.conaffinity = GROUND_BIT | OBSTACLE_BIT
            box_bits[sb.name] = GROUND_BIT
        else:
            bit = STONE_BIT0 << n_stone
            n_stone += 1
            assert bit < (1 << 31), "too many stones for per-stone collision bits"
            g.contype = 0
            g.conaffinity = bit
            box_bits[sb.name] = bit
        g.rgba = list(sb.rgba)
    all_stone_bits = 0
    for b in box_bits.values():
        if b != GROUND_BIT:
            all_stone_bits |= b

    # contact sites + geoms on the parent joint's body (rendered: group 2)
    joint_body = {}
    for b in spec.bodies:
        for j in b.joints:
            joint_body[j.name] = b
    site_names, geom_names, foot_corners = {}, {}, {}
    for cname, dim in contact_dims.items():
        jname, plc = frame_anchor(robot, cname)
        body = joint_body[jname]
        R, p = plc.rotation, plc.translation
        s = body.add_site()
        s.name = f"{cname}_site"
        s.pos = list(p)
        s.quat = list(_quat_from_mat(R))
        s.size = [0.004, 0.004, 0.004]
        s.group = 4
        site_names[cname] = s.name
        cg = body.add_geom()
        cg.name = f"{cname}_contact"
        cg.quat = list(_quat_from_mat(R))
        if dim == 6:
            fb = foot_boxes[cname]
            h = foot_box_half_height
            cg.type = mujoco.mjtGeom.mjGEOM_BOX
            cg.size = [float(fb["x"]) + sole_margin, float(fb["y"]) + sole_margin, h]
            cg.pos = list(p + R @ np.array([0.0, 0.0, h]))
        else:
            cg.type = mujoco.mjtGeom.mjGEOM_SPHERE
            cg.size = [foot_radius, 0.0, 0.0]
            cg.pos = list(p + R @ np.array([0.0, 0.0, foot_radius]))
        apply_contact_geom(cg, settings)
        cg.contype = GROUND_BIT | all_stone_bits
        cg.conaffinity = GROUND_BIT
        cg.rgba = [0.1, 0.1, 0.1, 1.0]
        cg.group = 2
        geom_names[cname] = cg.name
        if dim == 6:
            fb = foot_boxes[cname]
            foot_corners[cname] = np.array([[sx * (float(fb["x"]) + sole_margin), sy * (float(fb["y"]) + sole_margin), 0.0]
                                            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
        else:
            foot_corners[cname] = np.zeros((0, 3))

    if pack is not None:
        visual_meshes.attach(spec, pack, hide_existing=False)

    model = spec.compile()
    if boxes:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    align_inertias_to_pinocchio(model, robot)
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    model.dof_armature[:] = 0.0

    names = list(robot.model.names)[1:]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    missing = [n for n, j in zip(names, jids) if j < 0]
    if missing:
        raise KeyError(f"{name}: joints missing from the MuJoCo model: {missing}")
    qadr = np.array([model.jnt_qposadr[j] for j in jids], int)
    dofadr = np.array([model.jnt_dofadr[j] for j in jids], int)
    act_dofadr = dofadr[robot.act_idx]
    lock_idx = np.array([names.index(j) for j in planarlock], int)
    base_body = int(model.jnt_bodyid[jids[5]])
    return RobotSim(
        name=name, robot=robot, model=model, spec=spec, qadr=qadr, dofadr=dofadr,
        act_dofadr=act_dofadr, lock_idx=lock_idx,
        foot_site={c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for c, s in site_names.items()},
        foot_geom={c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for c, g in geom_names.items()},
        contact_dims=contact_dims, base_body=base_body,
        capsule_geoms={},
        box_geoms={b.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"box_{b.name}") for b in boxes},
        box_bits=box_bits, foot_corners=foot_corners,
        floor_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        settings=settings, scene_boxes=boxes, visual_pack=pack)


def build_scene(ref, *, assets, friction=None):
    """The MuJoCo model for a reference: robot + floor + the problem's boxes, dressed
    for rendering."""
    boxes = [SceneBox(sb.name, np.asarray(sb.center, float), np.asarray(sb.half, float),
                      collide_capsules=bool(sb.collide_capsules)) for sb in ref.scene_boxes()]
    settings = SETTINGS.with_(friction=float(ref.spec.mu if friction is None else friction))
    return build(ref.robot_name, scene_boxes=boxes, planarlock=tuple(ref.planarlock),
                 settings=settings, assets=assets)


def build_model(ref, solid_rgba, copies, cfg):
    rs = build_scene(ref, assets=cfg['scene']['visual_assets'])
    spec = rs.spec
    for g in spec.geoms:
        g.material = ''
        g.rgba = cfg['scene']['floor_rgba'] if g.name == 'floor' else (
            cfg['scene']['fence_rgba'] if g.parent == spec.worldbody else solid_rgba)
    # world-body obstacle geoms are identified by name as well
    box_names = {b.name for b in ref.scene_boxes()}
    for g in spec.geoms:
        if g.name in box_names:
            g.rgba = cfg['scene']['fence_rgba']
    for state, rgba in copies.items():
        ghost = mujoco.MjSpec.from_string(patch_urdf(rs.robot.urdf_path))
        ghost.compiler.discardvisual = False
        for g in ghost.geoms:
            g.contype = g.conaffinity = 0
            g.material = ''
            g.rgba = rgba
            g.group = visual_meshes.HIDDEN_GROUP if rs.visual_pack else 2
        if rs.visual_pack:
            visual_meshes.attach(ghost, rs.visual_pack, rgba=rgba, prefix='visual_', hide_existing=False)
        spec.attach(ghost, prefix=state + '_', frame=spec.worldbody.add_frame())
    floor_style = cfg.get('tracking_timelapse', {}).get('floor_surface')
    if floor_style:
        material = spec.add_material()
        material.name = 'timelapse_floor'
        material.rgba = cfg['scene']['floor_rgba']
        material.reflectance = floor_style['reflectance']
        material.specular = floor_style['specular']
        material.shininess = floor_style['shininess']
    for light, style in zip(spec.lights, cfg['scene']['lights']):
        light.pos, light.dir = style['pos'], style['dir']
        light.diffuse, light.ambient, light.castshadow = style['diffuse'], style['ambient'], style['shadow']
    spec.visual.global_.offwidth = cfg['video']['width']
    spec.visual.global_.offheight = cfg['video']['height']
    spec.visual.global_.fovy = cfg['camera']['fovy']
    spec.visual.quality.shadowsize = cfg['scene']['shadowsize']
    spec.visual.map.znear = cfg['scene']['znear']
    sky = spec.add_texture()
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    sky.width = sky.height = 32
    sky.rgb1 = sky.rgb2 = cfg['scene']['background']
    model = spec.compile()
    model.geom_matid[:] = -1
    if floor_style:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        model.geom_matid[floor_id] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'timelapse_floor')
    names = list(rs.robot.model.names)[1:]
    addresses = {}
    for key, prefix in [('solution', ''), *((name, name + '_') for name in copies)]:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix+n) for n in names]
        if min(ids) < 0 or any(int(model.jnt_type[i]) not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)) for i in ids):
            raise ValueError(f'Expected named scalar joints: {key} {list(zip(names, ids))} types={model.jnt_type}')
        addresses[key] = model.jnt_qposadr[ids]
    return model, addresses


def trajectory_bounds(model, data, addresses, trajectories, static_poses=None):
    """Fit visible robot geometry across every supplied paired configuration."""
    for name, q in (static_poses or {}).items():
        data.qpos[addresses[name]] = q
    lengths = {len(q) for q in trajectories.values()}
    if len(lengths) != 1:
        raise ValueError('Paired trajectories must have equal length')
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    visible = (model.geom_bodyid != 0) & (model.geom_group < 3)
    radius = model.geom_rbound[visible, None]
    for i in range(next(iter(lengths))):
        for name, q in trajectories.items():
            data.qpos[addresses[name]] = q[i]
        mujoco.mj_forward(model, data)
        xyz = data.geom_xpos[visible]
        lo = np.minimum(lo, (xyz-radius).min(axis=0))
        hi = np.maximum(hi, (xyz+radius).max(axis=0))
    return lo, hi


def scene_option():
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[:3] = 1
    option.sitegroup[:] = 0
    return option
