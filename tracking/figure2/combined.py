"""Fig. 2: a row per robot (Hopper, Go2, H1, G1, Talos), the uneven-terrain and the
fence-jump side views as two columns under a title each, all panels at one
pixels-per-metre.

    python -m figure2.combined --fence-data <tracked fence suite> --terrain-data <tracked terrain suite> \\
        --fence-source <dir>/src_fence --terrain-source <dir>/src_terrain --out <dir>/combined
"""
import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
from .style import DEFAULT_CONFIG, load_style
from .data import sha256
from . import scaled_grid
from .render import save_figure
from . import fence as fence_figure
from . import terrain as terrain_figure

LAYOUT = {
    # Rows, top to bottom: the two benchmarks' keys for one robot, and the label
    # of its row. The terrain hopper stands on a foot ("hopper2d_foot"), the
    # fence hopper on a point. These are the short names the scalability figures
    # and the paper's tables use (`<benchmark>_scalability.robot_labels`), not
    # the source figures' full product names: the label is set rotated in the
    # row's own height, and a side-view Go2 row is not "Unitree Go2" tall.
    'rows': [
        ('hopper2d_foot', 'hopper2d', 'Hopper'),
        ('go2', 'go2', 'Go2'),
        ('h1', 'h1', 'H1'),
        ('g1_29dof', 'g1_29dof', 'G1'),
        ('talos', 'talos', 'Talos'),
    ],
    # Column groups, left to right, in the order the paper introduces them. The
    # view names are deliberately not drawn -- isometric then side, as in both
    # source figures -- but the benchmark titles are, because a reader cannot
    # infer from a panel which of the two it belongs to.
    'groups': [
        dict(title='Uneven terrain', scenario='terrain', views=['side']),
        dict(title='Fence jump', scenario='fence', views=['side_dense_gradient']),
    ],
    'width_px': 1800,
    # One IEEE column.
    'column_in': 3.5,
    'text_pt': 8.0,
    'label_strip_factor': 1.4,
    'title_strip_factor': 1.7,
    # The gap BETWEEN the two benchmarks, as a multiple of the gap between two
    # views of one. With one view each there is only the former.
    'group_gutter_factor': 2.0,
    # Fraction of the way from the text color to white for the rule under each
    # benchmark title: present enough to bracket its columns, faint enough not
    # to compete with the robots.
    'rule_lightness': 0.55,
    'overflow_factor': 2.0,
    'trim_pad_px': 4,
    # Drawn only with --legend. One swatch serves both benchmarks' obstacles:
    # they share `scene.fence_rgba`.
    'legend_label': 'Elapsed time within each robot’s plan',
    'obstacle_label': 'Fence and stepping stones',
    'legend_width_fraction': 0.45,
    'combined_name': 'leap_tracking_timelapse_combined',
}


def scenario_panels(scenario, view, robots, args):
    """One benchmark's panels for one view, from its own figure's source renders.

    Both loaders are handed the style EXACTLY as their own figure resolves it --
    the terrain one builds its own -- because that configuration is what decides
    whether the saved renders are still valid. A label this figure changes for
    its own legend must not reach them, or it would invalidate, and silently
    re-render, the other figures' sources.
    """
    if scenario == 'terrain':
        overlay = terrain_figure.terrain_overlay(args.terrain_data, args.terrain_archive_dir,
                                                 args.terrain_archive_tag)
        return terrain_figure.source_renders(args.terrain_source/view, view, robots, args, overlay)
    style = fence_figure.fence_style(args.config, args.fence_data, args.fence_archive_dir,
                                     args.fence_archive_tag, args.assets)
    return scaled_grid.source_renders(args.fence_source/view, view, robots, style, args,
                                      fence_figure.LAYOUT['sample_density'])


def add_group_titles(grid, plan, args, cfg, text_px, groups):
    """A title per benchmark above the grid, centred over the columns it owns."""
    s = cfg['tracking_timelapse']
    background, foreground = tuple(s['background']), tuple(cfg['text']['foreground'])
    height = round(text_px*LAYOUT['title_strip_factor'])
    strip = Image.new('RGB', (grid.width, height), background)
    draw = ImageDraw.Draw(strip)
    font = ImageFont.truetype(cfg['text']['font'], text_px)
    rule = tuple(round(v+(255-v)*LAYOUT['rule_lightness']) for v in foreground)
    spans = []
    for group in groups:
        first, last = group['columns'][0], group['columns'][-1]
        left = scaled_grid.column_origin(args, plan, first)
        right = scaled_grid.column_origin(args, plan, last)+plan['column_widths'][last]
        if draw.textlength(group['title'], font=font) > right-left:
            raise ValueError(f'Title "{group["title"]}" does not fit its columns')
        draw.text(((left+right)/2, .42*height), group['title'], font=font,
                  fill=foreground, anchor='mm')
        # The rule brackets the group's columns and stops short of the panels.
        y = round(.74*height)
        draw.rectangle((left, y, right-1, y+max(1, round(.05*text_px))), fill=rule)
        spans.append([int(left), int(right)])
    out = Image.new('RGB', (grid.width, grid.height+height), background)
    out.paste(strip, (0, 0))
    out.paste(grid, (0, height))
    return out, spans


def add_input_arguments(parser):
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--fence-data', type=Path, required=True,
                        help='Tracked fence suite: the --out folder of track.py --task fence --all.')
    parser.add_argument('--terrain-data', type=Path, required=True,
                        help='Tracked terrain suite: the --out folder of track.py --task terrain --all.')
    parser.add_argument('--fence-archive-dir', type=Path)
    parser.add_argument('--fence-archive-tag')
    parser.add_argument('--terrain-archive-dir', type=Path)
    parser.add_argument('--terrain-archive-tag')
    parser.add_argument('--assets', type=Path,
                        help='Visual mesh root (default: figure2/assets; see assets/ASSETS.md).')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_arguments(parser)
    parser.add_argument('--terrain-source', type=Path, required=True,
                        help="Directory holding the terrain figure's per-view renders.")
    parser.add_argument('--fence-source', type=Path, required=True,
                        help="Directory holding the fence figure's per-view renders.")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=LAYOUT['width_px'],
                        help='Export width in pixels; the figure is drawn to fill it.')
    parser.add_argument('--column-in', type=float, default=LAYOUT['column_in'],
                        help='Printed width in inches; label point sizes are set for it.')
    parser.add_argument('--scale', choices=('metre', 'hip', 'fit'), default='metre',
                        help='metre: one pixels-per-metre everywhere (the robots keep their '
                             'relative sizes). hip: one pixels-per-hip-height, as in the '
                             'one-column figures. fit: fill each column.')
    parser.add_argument('--overflow-factor', type=float, default=LAYOUT['overflow_factor'])
    parser.add_argument('--text-pt', type=float, default=LAYOUT['text_pt'],
                        help='Printed size of every label, in points at --column-in wide.')
    parser.add_argument('--trim-pad', type=int, default=LAYOUT['trim_pad_px'],
                        help='Background pixels kept around each silhouette, in render pixels.')
    parser.add_argument('--gutter', type=int, help='Horizontal spacing; defaults to the shared style.')
    parser.add_argument('--gap', type=int, help='Vertical spacing; defaults to the shared style.')
    parser.add_argument('--legend', action='store_true',
                        help='Draw the time-gradient legend and obstacle swatch above the grid. '
                             'Off: the caption carries what they said, in space the figure keeps.')
    parser.add_argument('--reuse', action='store_true',
                        help='Fail instead of rendering when a source render is missing or stale.')
    parser.add_argument('--force-render', action='store_true',
                        help='Re-render every view even when validated renders already exist.')
    return parser


def run(args):
    # The shared style supplies spacing, typography and the legend; the single
    # obstacle swatch has to name both benchmarks' obstacles.
    cfg = load_style(args.config, assets=args.assets)
    cfg['tracking_timelapse']['obstacle_label'] = LAYOUT['obstacle_label']
    s = cfg['tracking_timelapse']
    rows, groups = LAYOUT['rows'], LAYOUT['groups']
    if args.gap is None:
        args.gap = s['combined_row_gap']
    gutter = s['combined_gutter'] if args.gutter is None else args.gutter
    # Points of printed text -> pixels of this export.
    text_px = max(1, round(args.text_pt*args.width/(args.column_in*72.)))
    args.margin = round(text_px*LAYOUT['label_strip_factor'])
    args.column_label_factor = 0.
    args.legend_variant = 'isometric_dense_gradient'
    args.out.mkdir(parents=True, exist_ok=True)

    # Columns are the groups' views end to end; each group records which it owns.
    columns, panels, sources = [], {}, {}
    for group in groups:
        group['columns'] = list(range(len(columns), len(columns)+len(group['views'])))
        robots = [row[0] if group['scenario'] == 'terrain' else row[1] for row in rows]
        for view in group['views']:
            key = (group['scenario'], view)
            panels[key], sources[key] = scenario_panels(group['scenario'], view, robots, args)
            columns.append((key, robots))
    # One wide gap between the benchmarks, the shared narrow one within each.
    boundaries = {group['columns'][-1] for group in groups[:-1]}
    args.gutter = [round(gutter*(LAYOUT['group_gutter_factor'] if column in boundaries else 1))
                   for column in range(len(columns)-1)]

    cells = [[panels[key][robots[index]] for key, robots in columns]
             for index, _ in enumerate(rows)]
    plan = scaled_grid.plan_layout(cells, args)
    grid = scaled_grid.compose(cells, plan, args, cfg, text_px,
                               row_labels=[label for _, _, label in rows])
    titled, spans = add_group_titles(grid, plan, args, cfg, text_px, groups)
    if args.legend:
        combined, content_bounds = scaled_grid.add_legend(titled, args, cfg, text_px,
                                                          LAYOUT['legend_label'],
                                                          LAYOUT['legend_width_fraction'])
    else:
        combined, content_bounds = scaled_grid.trim(titled, tuple(s['background']))
    stem = args.out/LAYOUT['combined_name']
    save_figure(combined, stem, cfg)
    with Image.open(stem.with_suffix('.png')) as exported:
        exported.load()
        assert exported.size == combined.size
    subprocess.run(['pdfinfo', str(stem.with_suffix('.pdf'))], check=True, capture_output=True)

    capped = {id(p) for p in plan['outliers']}
    drawn = {f'{scenario}/{view}': {
        robot: dict(size=panels[(scenario, view)][robot]['size'],
                    poses=panels[(scenario, view)][robot]['poses'],
                    metres_wide=panels[(scenario, view)][robot]['metres_wide'],
                    metres_tall=panels[(scenario, view)][robot]['metres_tall'],
                    hip_m=panels[(scenario, view)][robot]['hip'],
                    fraction=panels[(scenario, view)][robot]['fit'],
                    capped=id(panels[(scenario, view)][robot]) in capped)
        for robot in robots} for (scenario, view), robots in columns}
    report = dict(config=cfg, layout=LAYOUT, scale_mode=args.scale,
                  rows=[list(row) for row in rows], columns=[list(key) for key, _ in columns],
                  row_heights=plan['row_heights'], column_widths=plan['column_widths'],
                  gutters=plan['gutters'], title_spans=spans,
                  text_px=text_px, text_pt=args.text_pt, row_label_width_px=args.margin,
                  scale_unit=plan['unit'], pixels_per_unit=plan['unit_px'],
                  scale_note=f'every uncapped panel draws one {plan["unit"][:-1]} as '
                             f'{plan["unit_px"]:.1f} px',
                  capped_panels=[{'robot': p['robot'], 'variant': p['variant'],
                                  'fraction': p['fit']} for p in plan['outliers']],
                  legend=args.legend,
                  panels=drawn, size=list(combined.size), content_bounds=content_bounds,
                  printed_size_in=[args.column_in, args.column_in*combined.height/combined.width],
                  poses_only='tracked configurations; no LEAP reference, titles or timestamps drawn',
                  sources={f'{scenario}/{view}': source
                           for (scenario, view), source in sources.items()},
                  outputs={p.name: sha256(p) for p in
                           (stem.with_suffix('.png'), stem.with_suffix('.pdf'))}, passed=True)
    (args.out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved and validated {stem}.png and .pdf', flush=True)
    print(f'{combined.width} x {combined.height} px; {args.column_in:.2f} x '
          f'{report["printed_size_in"][1]:.2f} in printed at {args.width/args.column_in:.0f} dpi; '
          f'labels {args.text_pt:.1f} pt; one {plan["unit"][:-1]} = {plan["unit_px"]:.0f} px', flush=True)
    for (scenario, view), robots in columns:
        print(f'  {scenario}/{view}: poses ' + ', '.join(
            f'{r}={drawn[f"{scenario}/{view}"][r]["poses"]}' for r in robots), flush=True)
    for entry in report['capped_panels']:
        print(f'  scaled down to fit: {entry["robot"]} {entry["variant"]} at '
              f'{entry["fraction"]*100:.0f}% of the common scale', flush=True)
    return report


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
