"""Renders one camera view of every robot of a task: per robot a PNG/PDF panel and a JSON
sidecar, plus the view's combined figure and `manifest.json`.

    python -m figure2.render --config <style.json> --variant side_dense_gradient --out <dir> [--robots ...]
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .style import DEFAULT_CONFIG, load_style
from .colors import sample_pose_colors
from .scene import build_model, mujoco, scene_option, trajectory_bounds
from .data import load_run, sha256
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
VARIANTS = ('side_dense_gradient', 'isometric_dense_gradient')


def select_samples(run, cfg):
    """Select exact ticks within configured stance, flight and landing phases."""
    s = cfg['tracking_timelapse']
    if cfg.get('tracking_overlay', {}).get('benchmark') == 'terrain':
        terrain = s.get('terrain_sampling', {})
        density = terrain.get('density_by_robot', {}).get(getattr(run, 'robot', ''),
                                                         terrain.get('default_density', 1.0))
        if not np.isfinite(density) or density <= 0:
            raise ValueError('Terrain sampling density must be positive and finite')
        count = max(2, round((s.get('terrain_pose_count', 5)-1)*density)+1)
        requested = np.linspace(0., run.ref.duration, count)
        indices = np.array([np.argmin(np.abs(run.samples['t']-t)) for t in requested], int)
        if not np.all(np.diff(indices) > 0):
            raise ValueError('Selected terrain snapshots must be distinct and ordered')
        return indices, requested
    fence = s.get('fence_sampling', {})
    profile = fence.get('profiles', {}).get(s.get('fence_sampling_profile', 'side'), s)
    density = fence.get('density_by_robot', {}).get(getattr(run, 'robot', ''),
                                                       fence.get('default_density', 1.0))
    if not np.isfinite(density) or density <= 0:
        raise ValueError('Fence sampling density must be positive and finite')
    selected = {}
    for phase in ('stance', 'flight', 'landing'):
        fractions = np.asarray(profile.get(phase+'_fractions', []), float)
        if np.any((fractions <= 0) | (fractions >= 1)) or np.any(np.diff(fractions) <= 0):
            raise ValueError(f'{phase}_fractions must be strictly increasing within (0, 1)')
        selected[phase] = fractions
    takeoff = run.ref.spec.t_stance
    touchdown = takeoff+run.ref.spec.t_air
    requested = np.r_[0., selected['stance']*takeoff,
        takeoff+selected['flight']*run.ref.spec.t_air,
        touchdown+selected['landing']*(run.ref.duration-touchdown), run.ref.duration]
    # Scale total pose count, retaining both endpoints and the phase distribution.
    count = max(2, int(np.floor(len(requested)*density + 0.5)))
    requested = np.interp(np.linspace(0., len(requested)-1, count),
                          np.arange(len(requested)), requested)
    indices = np.array([np.argmin(np.abs(run.samples['t']-t)) for t in requested], int)
    if not np.all(np.diff(indices) > 0):
        raise ValueError('Selected snapshots must be distinct and ordered')
    return indices, requested


def pose_colors(cfg):
    s = cfg['tracking_timelapse']
    initial = [*cfg['states']['initial']['rgba'][:3], s['endpoint_alpha']]
    final = [*cfg['states']['goal']['rgba'][:3], s['endpoint_alpha']]
    motion = [*cfg['methods'][s['motion_color_role']]['rgba'][:3], s['motion_alpha']]
    return initial, motion, final


def render_scene(run, cfg):
    """Render all selected configurations together, then crop by segmentation.

    No pose receives an artificial translation or scale. The crop includes every
    visible robot geometry and obstacle, plus the finite floor when configured.
    The orthographic view direction is configured for each figure variant.
    """
    s = cfg['tracking_timelapse']
    local = deepcopy(cfg)
    local['video']['width'] = local['video']['height'] = s['render_size']
    indices, requested = select_samples(run, cfg)
    colors, progress = sample_pose_colors(cfg, run.samples['t'][indices])
    copies = {f'pose{i}': colors[i].tolist() for i in range(1, len(indices)-1)}
    copies['final'] = colors[-1].tolist()
    model, adr = build_model(run.ref, colors[0].tolist(), copies, local)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor_id < 0:
        raise ValueError('Expected ground floor geometry is missing')
    model.geom_group[floor_id] = 0
    model.geom_rgba[floor_id] = cfg['scene']['floor_rgba']
    rendered_floor_rgba = model.geom_rgba[floor_id].copy()
    box_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'box_'+box.name) for box in run.ref.scene_boxes()]
    if not box_ids or min(box_ids) < 0:
        raise ValueError('Expected fence geometry is missing')
    rendered_fence_rgba = model.geom_rgba[box_ids].copy()
    if not np.allclose(rendered_fence_rgba, cfg['scene']['fence_rgba'], rtol=0, atol=1e-7):
        raise ValueError('Fence color differs from the shared configuration')
    data = mujoco.MjData(model)
    names = ['solution', *copies]
    configurations = {name: run.samples['q'][i] for name, i in zip(names, indices)}
    bounds = trajectory_bounds(model, data, adr, {k: q[None, :] for k, q in configurations.items()})
    # Include the complete physical obstacle in initial camera fitting too.
    for box in run.ref.scene_boxes():
        bounds = (np.minimum(bounds[0], np.asarray(box.center)-np.asarray(box.half)),
                  np.maximum(bounds[1], np.asarray(box.center)+np.asarray(box.half)))
    lo, hi = bounds
    floor_style = s.get('floor_surface')
    if floor_style:
        # A finite ground patch exposes its edges against the sky background.
        # Its top remains at the original physical ground height.
        padding = floor_style['padding_m']
        model.geom_pos[floor_id, :2] = (lo[:2]+hi[:2])/2
        model.geom_sameframe[floor_id] = 0
        model.geom_size[floor_id, :2] = (hi[:2]-lo[:2])/2+padding
        lo = lo.copy()
        hi = hi.copy()
        lo[:2] -= padding
        hi[:2] += padding
        lo[2] = min(lo[2], model.geom_pos[floor_id, 2])
        # re-create MjData after moving the patch, then restore the tracked
        # configurations
        data = mujoco.MjData(model)
        for name, q in configurations.items():
            data.qpos[adr[name]] = q
    az, el = np.deg2rad([s['azimuth'], s['elevation']])
    right = np.array([np.sin(az), -np.cos(az), 0.])
    up = np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
    span = max(np.abs(right) @ (hi-lo), np.abs(up) @ (hi-lo))*s['camera_margin']
    model.vis.global_.orthographic = s['orthographic']
    model.vis.global_.fovy = span
    cam = mujoco.MjvCamera()
    cam.orthographic = s['orthographic']
    cam.lookat[:] = (lo+hi)/2
    cam.distance = np.linalg.norm(hi-lo)+cfg['camera']['minimum_distance']
    cam.azimuth, cam.elevation = s['azimuth'], s['elevation']
    option = scene_option()
    mujoco.mj_forward(model, data)
    # segmentation rendering without multisampling
    model.vis.quality.offsamples = 0
    with mujoco.Renderer(model, s['render_size'], s['render_size']) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = s['render_shadows']
        rgb = renderer.render().copy()
        # Reflections are appearance, not additional physical robot geometry.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        # ID colors must not alpha-blend when many translucent poses overlap.
        # The RGB image above retains the configured pose transparency.
        for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
            geom.rgba[3] = 1.0
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    geom_id, object_type = segmentation[..., 0], segmentation[..., 1]
    valid = (object_type == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom_id >= 0)
    ids = np.clip(geom_id, 0, model.ngeom-1)
    mask = valid & (model.geom_type[ids] != int(mujoco.mjtGeom.mjGEOM_PLANE)) & (model.geom_group[ids] < 3)
    if floor_style:
        mask |= valid & (geom_id == floor_id)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('No visible robot/obstacle pixels')
    if min(xx.min(), yy.min()) == 0 or xx.max() == rgb.shape[1]-1 or yy.max() == rgb.shape[0]-1:
        raise ValueError('Initial camera clips geometry; increase camera_margin')
    raw = np.array([xx.min(), yy.min(), xx.max()+1, yy.max()+1], int)
    pad = max(s['crop_padding_min_px'], round(max(raw[2:]-raw[:2])*s['crop_padding_fraction']))
    crop = np.r_[np.maximum(0, raw[:2]-pad), np.minimum(s['render_size'], raw[2:]+pad)].astype(int)
    x0, y0, x1, y1 = crop
    image = Image.fromarray(rgb[y0:y1, x0:x1])
    audit = dict(run.audit, selected_indices=indices.tolist(), requested_times=requested.tolist(),
        selected_times=run.samples['t'][indices].tolist(), selected_q=run.samples['q'][indices].tolist(),
        interval='initial to plan endpoint; post-plan hold excluded',
        initial_pose='first tracked configuration after the recorded settling stage',
        final_pose='tracked configuration at the nearest plan-end tick; not the desired target',
        final_target_max_error=float(np.max(np.abs(run.samples['q'][indices[-1]]-(run.ref.eval(run.ref.duration).q if cfg.get('tracking_overlay', {}).get('benchmark') == 'terrain' else run.ref.spec.q_hold2)))),
        pose_rgba=colors.tolist(), time_progress=progress.tolist(),
        color_encoding=s.get('color_mode', 'method_color'),
        rendered_floor_rgba=rendered_floor_rgba.tolist(),
        floor_position=model.geom_pos[floor_id].tolist(), floor_half_size=model.geom_size[floor_id].tolist(),
        fence_rgba=cfg['scene']['fence_rgba'], rendered_fence_rgba=rendered_fence_rgba.tolist(), projection='orthographic', camera_angles=[s['azimuth'], s['elevation']],
        world_bounds=[lo.tolist(), hi.tolist()], orthographic_span=float(span),
        silhouette_bbox=raw.tolist(), crop_bbox=crop.tolist(), crop_padding_px=pad,
        scene_size=list(image.size), config=cfg)
    return image, audit


def legend_entries(cfg):
    s = cfg['tracking_timelapse']
    initial, motion, final = pose_colors(cfg)
    return [(s['initial_label'], initial), (s['motion_label'], motion),
            (s['final_label'], final), (s['obstacle_label'], cfg['scene']['fence_rgba'])]


def add_legend(image, cfg):
    s = cfg['tracking_timelapse']
    if s.get('color_mode') == 'time_gradient':
        return add_gradient_legend(image, cfg)
    font = ImageFont.truetype(cfg['text']['font'], s['legend_font_size'])
    measure = ImageDraw.Draw(image)
    widths = [s['swatch_size']+s['label_gap']+measure.textlength(label, font=font)
              for label, _ in legend_entries(cfg)]
    total = sum(widths)+s['legend_gap']*(len(widths)-1)
    # Fit the legend to narrow natural-aspect panels rather than widening them.
    scale = min(1., (image.width-2*s['crop_padding_min_px'])/total)
    strip = Image.new('RGB', (int(np.ceil(total))+2*s['crop_padding_min_px'], s['legend_height']), tuple(s['background']))
    draw = ImageDraw.Draw(strip)
    x, y = s['crop_padding_min_px'], (s['legend_height']-s['legend_font_size'])//2
    for (label, rgba), width in zip(legend_entries(cfg), widths):
        color = tuple(round(255*v) for v in rgba[:3])
        draw.rectangle((x, y, x+s['swatch_size'], y+s['swatch_size']), fill=color)
        draw.text((x+s['swatch_size']+s['label_gap'], y), label, font=font, fill=tuple(cfg['text']['foreground']))
        x += width+s['legend_gap']
    if scale < 1:
        strip = strip.resize((round(strip.width*scale), round(strip.height*scale)), Image.Resampling.LANCZOS)
    out = Image.new('RGB', (image.width, image.height+strip.height), tuple(s['background']))
    out.paste(image, (0, 0))
    out.paste(strip, ((image.width-strip.width)//2, image.height))
    return out


def add_gradient_legend(image, cfg):
    """Continuous endpoint-color ramp with a separate, unchanged obstacle swatch."""
    s = cfg['tracking_timelapse']
    font = ImageFont.truetype(cfg['text']['font'], s['legend_font_size'])
    pad, gap = s['gradient_legend_padding'], s['legend_gap']
    measure = ImageDraw.Draw(image)
    obstacle_width = s['swatch_size']+s['label_gap']+measure.textlength(s['obstacle_label'], font=font)
    width = min(s['gradient_bar_width'], int(image.width-2*pad-gap-obstacle_width))
    if width < 2:
        raise ValueError('Figure too narrow for the time-gradient legend')
    strip = Image.new('RGB', (image.width, s['gradient_legend_height']), tuple(s['background']))
    x = round((image.width-width-gap-obstacle_width)/2)
    y = pad+s['legend_font_size']+s['label_gap']
    start = np.asarray(cfg['states']['initial']['rgba'][:3])
    finish = np.asarray(cfg['states']['goal']['rgba'][:3])
    u = np.linspace(0., 1., width)[:, None]
    ramp = np.rint(255*((1-u)*start+u*finish)).astype(np.uint8)
    strip.paste(Image.fromarray(np.repeat(ramp[None, :, :], s['gradient_bar_height'], axis=0)), (x, y))
    draw = ImageDraw.Draw(strip)
    foreground = tuple(cfg['text']['foreground'])
    draw.text((x, pad), s['gradient_label'], font=font, fill=foreground)
    label_y = y+s['gradient_bar_height']+s['label_gap']
    draw.text((x, label_y), s['gradient_initial_label'], font=font, fill=foreground)
    draw.text((x+width-draw.textlength(s['gradient_final_label'], font=font), label_y),
              s['gradient_final_label'], font=font, fill=foreground)
    ox = x+width+gap
    draw.rectangle((ox, y, ox+s['swatch_size'], y+s['swatch_size']),
                   fill=tuple(round(255*v) for v in cfg['scene']['fence_rgba'][:3]))
    draw.text((ox+s['swatch_size']+s['label_gap'], y), s['obstacle_label'], font=font, fill=foreground)
    out = Image.new('RGB', (image.width, image.height+strip.height), tuple(s['background']))
    out.paste(image, (0, 0))
    out.paste(strip, (0, image.height))
    return out


def make_panel(scene, audit, cfg):
    s = cfg['tracking_timelapse']
    font = ImageFont.truetype(cfg['text']['font'], s['font_size'])
    time_font = ImageFont.truetype(cfg['text']['font'], s['time_font_size'])
    columns = min(s.get('time_label_columns', len(audit['selected_times'])), len(audit['selected_times']))
    rows = int(np.ceil(len(audit['selected_times'])/columns))
    panel = Image.new('RGB', (scene.width, scene.height+s['title_height']+rows*s['time_row_height']), tuple(s['background']))
    panel.paste(scene, (0, s['title_height']))
    draw = ImageDraw.Draw(panel)
    title = cfg['robots'][audit['robot']]
    draw.text(((panel.width-draw.textlength(title, font=font))/2, (s['title_height']-s['font_size'])//2),
              title, font=font, fill=tuple(cfg['text']['foreground']))
    # Use a compact chronological strip. Labels never require extra scene space.
    audit['timestamp_label_boxes'] = []
    for i, t in enumerate(audit['selected_times']):
        text = f'{t:.2f} s'
        x = (i%columns+.5)*panel.width/columns-draw.textlength(text, font=time_font)/2
        x = max(s['crop_padding_min_px'], min(x, panel.width-draw.textlength(text, font=time_font)-s['crop_padding_min_px']))
        y = s['title_height']+scene.height+(i//columns)*s['time_row_height']+(s['time_row_height']-s['time_font_size'])//2
        audit['timestamp_label_boxes'].append(list(draw.textbbox((x, y), text, font=time_font)))
        draw.text((x, y), text, font=time_font, fill=tuple(cfg['text']['foreground']))
    boxes = audit['timestamp_label_boxes']
    if labels_overlap(boxes):
        raise ValueError('Timestamp labels overlap; increase figure width or reduce time_font_size')
    return panel


def labels_overlap(boxes):
    """Allow multiple timestamp rows, but reject intersecting text rectangles."""
    return any(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
               for i, a in enumerate(boxes) for b in boxes[i+1:])


def save_figure(image, stem, cfg):
    dpi = cfg['tracking_timelapse']['dpi']
    image.save(stem.with_suffix('.png'), dpi=(dpi, dpi))
    image.save(stem.with_suffix('.pdf'), 'PDF', resolution=dpi,
               quality=cfg['tracking_timelapse']['pdf_jpeg_quality'],
               subsampling=cfg['tracking_timelapse']['pdf_subsampling'])


def render_view(config, out, variant, robots=None):
    """Render one camera view of every robot in `robots` into `out`.

    `config` is a style file (`style.json` layout); the view's variant block is
    applied, then per robot: `<robot>_tracking_timelapse.{png,pdf,json}`, and the
    view's combined figure and `manifest.json` (the configuration, which decides
    whether the renders are still valid; `scaled_grid.renders_fresh`).
    """
    out = Path(out)
    cfg = load_style(config)
    s = cfg['tracking_timelapse']
    s.update(deepcopy(s.get('variants', {}).get(variant, {})))
    s['variant'] = variant
    robots = ['g1_29dof' if r == 'g1' else r for r in (robots or cfg['robots'])]
    out.mkdir(parents=True, exist_ok=True)
    scenes, audits = {}, {}
    for robot in robots:
        run = load_run(robot, cfg, work_dir=out)
        scene, audit = render_scene(run, cfg)
        panel = make_panel(scene, audit, cfg)
        stem = out/f'{robot}_tracking_timelapse'
        save_figure(add_legend(panel, cfg), stem, cfg)
        audit['outputs'] = {p.name: sha256(p) for p in (stem.with_suffix('.png'), stem.with_suffix('.pdf'))}
        stem.with_suffix('.json').write_text(json.dumps(audit, indent=2)+'\n')
        scenes[robot] = scene
        audits[robot] = audit
        print(f'{robot}: saved tightly cropped figure {scene.width} x {scene.height}; times {audit["selected_times"]}', flush=True)
    rows = [[r for r in row if r in robots] for row in s['combined_rows']]
    rows = [row for row in rows if row]
    if sorted(r for row in rows for r in row) != sorted(robots):
        raise ValueError('combined_rows must include every selected robot exactly once')
    rendered_rows = []
    for row in rows:
        aspect_sum = sum(scenes[r].width/scenes[r].height for r in row)
        height = round((s['combined_width_px']-s['combined_gutter']*(len(row)-1))/aspect_sum)
        panels = []
        for robot in row:
            scene = scenes[robot]
            resized = scene.resize((round(scene.width*height/scene.height), height), Image.Resampling.LANCZOS)
            panels.append(make_panel(resized, deepcopy(audits[robot]), cfg))
        # Panels in a row share a scene height but not a caption height: a robot
        # sampled at a lower density needs fewer timestamp rows. Size the row to
        # the tallest panel so no caption is cropped.
        row_image = Image.new('RGB', (sum(p.width for p in panels)+s['combined_gutter']*(len(row)-1), max(p.height for p in panels)), tuple(s['background']))
        x = 0
        for panel in panels:
            row_image.paste(panel, (x, 0))
            x += panel.width+s['combined_gutter']
        rendered_rows.append(row_image)
    combined = Image.new('RGB', (max(r.width for r in rendered_rows),
        sum(r.height for r in rendered_rows)+s['combined_row_gap']*(len(rendered_rows)-1)), tuple(s['background']))
    y = 0
    for row in rendered_rows:
        combined.paste(row, ((combined.width-row.width)//2, y))
        y += row.height+s['combined_row_gap']
    combined = add_legend(combined, cfg)
    combined_stem = out/s['combined_name']
    save_figure(combined, combined_stem, cfg)
    (out/'manifest.json').write_text(json.dumps(dict(robots=robots, config=cfg,
        combined_scale='independent per-robot scale; each panel preserves its own x-z aspect ratio',
        combined_rows=rows, combined_size=list(combined.size),
        combined_outputs={p.name: sha256(p) for p in (combined_stem.with_suffix('.png'), combined_stem.with_suffix('.pdf'))},
        figures=[f'{r}_tracking_timelapse' for r in robots]), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robots', nargs='+')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    render_view(args.config, args.out, args.variant, args.robots)


if __name__ == '__main__':
    main()
