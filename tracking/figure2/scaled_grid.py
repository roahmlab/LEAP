"""Grid layout of the composed figures: loads a view's validated renders (rendering them
when missing or stale), trims each panel to its silhouette, scales the panels to one
shared unit (`hip`: pixels per hip height, `metre`: pixels per metre, `fit`: fill each
column), packs them with row labels, and draws the time-gradient legend.
"""
from copy import deepcopy
from statistics import median
import json

from PIL import Image, ImageChops, ImageDraw, ImageFont
from .data import sha256
from .render import add_gradient_legend, render_view
from .verify import verify

SCALE_JOINT = 'base_pz'
# What makes a render stale. style.json is shared with figures that have nothing to
# do with this one, so a blanket comparison of the whole style would invalidate
# every render whenever an unrelated figure gains a setting. These are the blocks
# the render path reads at all -- render.py, scene.py, data.py and colors.py, by
# inspection of every cfg[...] there -- split by how they
# are read. The renderer indexes LOOSE_KEYS by fixed name, so a value it recorded
# must not have changed, but an entry ADDED later (a new palette color, another
# robot) cannot have affected it. It reads STRICT_KEYS through `.get(name, default)`,
# where an added key DOES change what it draws, so those must match exactly.
LOOSE_KEYS = ('camera', 'data', 'methods', 'robots', 'scene', 'states', 'text', 'video')
STRICT_KEYS = ('tracking_timelapse', 'tracking_overlay')


def unchanged(recorded, current):
    """True when every value `recorded` holds is still present and equal in `current`."""
    if isinstance(recorded, dict):
        return isinstance(current, dict) and all(
            k in current and unchanged(v, current[k]) for k, v in recorded.items())
    return recorded == current


def view_config(cfg, variant, densities=None):
    """The shared style with this view's pose-density multipliers applied."""
    base = deepcopy(cfg)
    density = base['tracking_timelapse']['fence_sampling']['density_by_robot']
    for robot, factor in (densities or {}).get(variant, {}).items():
        if robot not in density:
            raise ValueError(f'No shared sampling density for {robot}')
        density[robot] *= factor
    return base


def renders_fresh(directory, resolved, args):
    """Whether `directory`'s renders were made with a configuration still in force.

    Renders are stale when something the RENDERER reads has changed; edits
    elsewhere in the shared style are reported and ignored (see LOOSE/STRICT_KEYS).
    """
    manifest_path = directory/'manifest.json'
    recorded = json.loads(manifest_path.read_text())['config'] if manifest_path.exists() else {}
    fresh = (not args.force_render and bool(recorded)
             and all(recorded.get(k) == resolved.get(k) for k in STRICT_KEYS)
             and all(unchanged(recorded.get(k), resolved.get(k)) for k in LOOSE_KEYS))
    ignored = sorted(k for k in set(recorded) | set(resolved)
                     if k not in STRICT_KEYS and recorded.get(k) != resolved.get(k))
    if fresh and ignored:
        print(f'{directory.name}: reusing renders; style.json differs in '
              f'{", ".join(ignored)}, which the renderer does not read', flush=True)
    return fresh


def source_renders(directory, variant, robots, cfg, args, densities=None):
    """Render (or reuse) one view's per-robot panels, trimmed to their silhouettes."""
    base = view_config(cfg, variant, densities)
    resolved = deepcopy(base)
    resolved['tracking_timelapse'].update(deepcopy(base['tracking_timelapse']['variants'][variant]))
    resolved['tracking_timelapse']['variant'] = variant
    manifest_path = directory/'manifest.json'
    if not renders_fresh(directory, resolved, args):
        if args.reuse:
            raise ValueError(f'No up-to-date renders in {directory}; rerun without --reuse')
        # Any figure-specific sampling is this figure's own, so the renders and
        # the style they were made with live beside it, not in a shared view.
        directory.mkdir(parents=True, exist_ok=True)
        style = directory/'style_used.json'
        style.write_text(json.dumps(base, indent=2)+'\n')
        render_view(style, directory, variant, robots)
        if json.loads(manifest_path.read_text())['config'] != resolved:
            raise ValueError(f'Stale configuration in {directory}')
    return load_panels(directory, robots, resolved, args), {'path': str(manifest_path), 'sha256': sha256(manifest_path)}


def load_panels(directory, robots, resolved, args):
    """Validate one view's finished renders and trim each panel to its silhouette.

    Only the scene rectangle is kept: the renderer's title above it and the
    timestamp strip below it are cropped away.
    """
    verify(directory)
    panels = {}
    for robot in robots:
        stem = directory/f'{robot}_tracking_timelapse'
        audit = json.loads(stem.with_suffix('.json').read_text())
        width, height = audit['scene_size']
        top = resolved['tracking_timelapse']['title_height']
        with Image.open(stem.with_suffix('.png')) as panel:
            scene = panel.crop((0, top, width, top+height))
        # Trim the renderer's uniform padding back to the drawn silhouette.
        crop, silhouette, pad = audit['crop_bbox'], audit['silhouette_bbox'], args.trim_pad
        box = (max(silhouette[0]-pad, crop[0])-crop[0], max(silhouette[1]-pad, crop[1])-crop[1],
               min(silhouette[2]+pad, crop[2])-crop[0], min(silhouette[3]+pad, crop[3])-crop[1])
        scene = scene.crop(box)
        # Orthographic projection: one metre is the same number of pixels
        # everywhere in a render, so the panel's physical extent is exact.
        metres_per_pixel = audit['orthographic_span']/audit['config']['tracking_timelapse']['render_size']
        hip = audit['selected_q'][0][audit['joint_names'].index(SCALE_JOINT)]
        if not hip > 0:
            raise ValueError(f'{robot}: non-positive {SCALE_JOINT} at the initial pose')
        panels[robot] = dict(scene=scene, metres_per_pixel=metres_per_pixel, hip=hip,
                             robot=robot, variant=resolved['tracking_timelapse']['variant'], poses=len(audit['selected_times']),
                             trim_box=list(box), scene_size=list(scene.size),
                             # Panel extent in the two units a scale mode can share:
                             # the robot's own hip heights, and metres of world.
                             hips_wide=scene.width*metres_per_pixel/hip,
                             hips_tall=scene.height*metres_per_pixel/hip,
                             metres_wide=scene.width*metres_per_pixel,
                             metres_tall=scene.height*metres_per_pixel)
    return panels


def gutters(args, columns):
    """Horizontal spacing before each column but the first.

    `args.gutter` is one number, or one per gap when a figure needs a wider gap
    somewhere -- between two groups of columns, say -- than between the columns
    within a group.
    """
    if not isinstance(args.gutter, (list, tuple)):
        return [args.gutter]*(columns-1)
    if len(args.gutter) != columns-1:
        raise ValueError(f'{columns} columns need {columns-1} gutters, got {len(args.gutter)}')
    return list(args.gutter)


def plan_layout(cells, args):
    """Size every panel in the grid `cells[row][column]` and return the geometry.

    Each panel gains `fit` (1 unless it was scaled down to fit its column),
    `scale` (pixels per unit of the shared scale) and `size`.
    """
    columns = list(zip(*cells))
    gaps = gutters(args, len(columns))
    # The unit the figure holds fixed: hip heights equalise the ROBOTS, metres
    # equalise the WORLD. Everything below is the same calculation in either.
    unit = 'metres' if args.scale == 'metre' else 'hips'
    wide, tall = unit+'_wide', unit+'_tall'
    if args.scale == 'fit':
        # Legacy behaviour: every panel fills its column, so robots differ in size.
        widths = [max(p[wide] for p in column) for column in columns]
        for column, width in zip(columns, widths):
            for panel in column:
                panel['fit'] = width/panel[wide]
        outliers = []
    else:
        extents = [p[wide] for row in cells for p in row]
        limit = args.overflow_factor*median(extents)
        outliers = [p for row in cells for p in row if p[wide] > limit]
        capped = {id(p) for p in outliers}
        regular = [p[wide] for row in cells for p in row if id(p) not in capped]
        if not regular:
            raise ValueError('Every panel is an envelope outlier; raise --overflow-factor')
        # A column of nothing but outliers (one robot per column, and that robot
        # is the outlier) falls back to the widest column the others support.
        widths = [max([p[wide] for p in column if id(p) not in capped], default=max(regular))
                  for column in columns]
        for column, width in zip(columns, widths):
            for panel in column:
                # Everything shares one scale; only an outlier is shrunk to fit.
                panel['fit'] = min(1., width/panel[wide])
    # One scale for the whole figure, from the available width.
    usable = args.width-args.margin-sum(gaps)
    pixels_per_unit = usable/sum(widths)
    for row in cells:
        for panel in row:
            panel['scale'] = pixels_per_unit*panel['fit']
            panel['size'] = (max(1, round(panel[wide]*panel['scale'])),
                             max(1, round(panel[tall]*panel['scale'])))
    column_widths = [max(p['size'][0] for p in column) for column in columns]
    row_heights = [max(p['size'][1] for p in row) for row in cells]
    capped = {id(p) for p in outliers}
    shared = [p['scale'] for row in cells for p in row if id(p) not in capped]
    if args.scale != 'fit' and max(shared)-min(shared) > 1e-9:
        raise ValueError(f'Uncapped panels must share one pixels-per-{unit[:-1]}')
    # `pixels_per_hip`/`hip_px` are the scale in whatever unit this mode shares;
    # the names are the ones the hip-scaled figures' manifests already record.
    return dict(column_widths=column_widths, row_heights=row_heights,
                pixels_per_hip=pixels_per_unit, outliers=outliers, hip_px=shared[0],
                unit=unit, unit_px=shared[0], gutters=gaps)


def compose(cells, plan, args, cfg, text_px, row_labels=None, column_labels=None):
    """Paste every panel into the grid, with rotated row labels and column labels."""
    s = cfg['tracking_timelapse']
    background, foreground = tuple(s['background']), tuple(cfg['text']['foreground'])
    column_widths, row_heights = plan['column_widths'], plan['row_heights']
    gaps = plan['gutters']
    label_height = round(text_px*args.column_label_factor) if column_labels else 0
    width = args.margin+sum(column_widths)+sum(gaps)
    grid = Image.new('RGB', (width, sum(row_heights)+args.gap*(len(row_heights)-1)+label_height),
                     background)
    font = ImageFont.truetype(cfg['text']['font'], text_px)
    draw = ImageDraw.Draw(grid)
    y = 0
    for index, (row, height) in enumerate(zip(cells, row_heights)):
        if row_labels:
            if draw.textlength(row_labels[index], font=font) > height:
                raise ValueError(f'Row label "{row_labels[index]}" does not fit its {height} px row')
            label = Image.new('RGB', (height, args.margin), background)
            ImageDraw.Draw(label).text((height/2, args.margin/2), row_labels[index],
                                       font=font, fill=foreground, anchor='mm')
            grid.paste(label.rotate(90, expand=True), (0, y))
        x = args.margin
        for column, (panel, column_width) in enumerate(zip(row, column_widths)):
            fitted = panel['scene'].resize(panel['size'], Image.Resampling.LANCZOS)
            grid.paste(fitted, (x+(column_width-fitted.width)//2, y+(height-fitted.height)//2))
            x += column_width+(gaps[column] if column < len(gaps) else 0)
        y += height+args.gap
    for index, label in enumerate(column_labels or []):
        x = column_origin(args, plan, index)+column_widths[index]/2
        draw.text((x, grid.height-label_height/2), label, font=font, fill=foreground, anchor='mm')
    return grid


def column_origin(args, plan, index):
    """Left edge of one column in the composed grid, in pixels."""
    return args.margin+sum(plan['column_widths'][:index])+sum(plan['gutters'][:index])


def trim(image, background):
    """The image cropped to its drawn content, with no surrounding margin or frame."""
    bounds = ImageChops.difference(image, Image.new('RGB', image.size, background)).getbbox()
    return (image.crop(bounds) if bounds is not None else image), bounds


def add_legend(grid, args, cfg, text_px, label, width_fraction):
    """The shared time-gradient legend, redrawn at this figure's point size."""
    s = cfg['tracking_timelapse']
    legend_cfg = deepcopy(cfg)
    legend_cfg['tracking_timelapse'].update(s['variants'][args.legend_variant])
    bar_height, label_gap, pad = round(.7*text_px), round(.45*text_px), round(.5*text_px)
    legend_cfg['tracking_timelapse'].update(
        gradient_label=label, legend_font_size=text_px, swatch_size=text_px,
        label_gap=label_gap, legend_gap=round(1.2*text_px),
        gradient_bar_width=round(grid.width*width_fraction), gradient_bar_height=bar_height,
        gradient_legend_padding=pad,
        gradient_legend_height=2*pad+2*text_px+2*label_gap+bar_height)
    legend = add_gradient_legend(Image.new('RGB', (grid.width, 0)), legend_cfg)
    combined = Image.new('RGB', (grid.width, grid.height+legend.height), tuple(s['background']))
    combined.paste(legend, (0, 0))
    combined.paste(grid, (0, legend.height))
    return trim(combined, tuple(s['background']))
