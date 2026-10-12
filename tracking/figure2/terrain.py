"""Terrain figure: a row per robot (Hopper, Go2, H1, G1, Talos), an isometric and a side
view column, one pixels-per-hip-height, time-gradient legend; `terrain_config` resolves
the style of one terrain view.

    python -m figure2.terrain --data <tracked terrain suite> --source <dir>/src_terrain --out <dir>/terrain
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from PIL import Image
from .style import load_style
from .data import sha256
from . import scaled_grid
from .render import render_view, save_figure

ROBOTS = {'hopper2d_foot': 'Hopper', 'go2': 'Go2', 'h1': 'H1',
          'g1_29dof': 'G1 (29 DoF)', 'talos': 'Talos'}
COMBINED_ROWS = [['hopper2d_foot', 'go2'], ['h1', 'g1_29dof', 'talos']]
# Camera views: name -> style variant.
VIEWS = {'side': dict(variant='side_dense_gradient'),
         'isometric': dict(variant='isometric_dense_gradient')}
# Terrain pose sampling: density scales the 10 intervals of 11 poses (0.5 -> 6 poses,
# both endpoints kept).
TIMELAPSE_CONFIG = {
    "terrain_sampling": {
        "default_density": 1.0,
        "density_by_robot": {
            "h1": 0.5,
            "talos": 0.5
        }
    }
}


def terrain_overlay(data, archive_dir=None, archive_tag=None):
    """`tracking_overlay` entries naming one terrain suite's tracked folder and archives."""
    return dict(data_root=str(Path(data).resolve()),
                archive_dir=str(Path(archive_dir).resolve()) if archive_dir else None,
                archive_tag=archive_tag or None)


def terrain_config(view_name, density_scale=None, *, overlay=None, assets=None):
    """The shared style resolved for one terrain view; returns (cfg, variant).

    `density_scale` multiplies `terrain_sampling.density_by_robot` per robot for
    figures that carry their own, thinner sampling; endpoints are always kept.
    """
    view = dict(VIEWS[view_name])
    variant = view.pop('variant')
    cfg = load_style(assets=assets)
    cfg['robots'] = dict(ROBOTS)
    cfg['tracking_overlay'].update(benchmark='terrain', **(overlay or {}))
    if variant:
        cfg['tracking_timelapse'].update(deepcopy(cfg['tracking_timelapse']['variants'][variant]))
    cfg['tracking_timelapse'].update(orthographic=True,
        terrain_pose_count=11, endpoint_alpha=1.,
        obstacle_label='Stepping stones', initial_label='Initial tracked pose',
        final_label='Final tracked pose', motion_label='LEAP tracking',
        combined_rows=deepcopy(COMBINED_ROWS),
        combined_width_px=3000, combined_gutter=16, combined_row_gap=12,
        combined_name='leap_terrain_tracking_timelapse_'+view_name,
        crop_padding_fraction=0.008, crop_padding_min_px=8,
        minimum_crop_occupancy=0.9, time_label_columns=6,
        variants={}, title_height=52, time_row_height=40)
    cfg['tracking_timelapse'].update(deepcopy(TIMELAPSE_CONFIG))
    cfg['tracking_timelapse'].update(view)
    sampling = cfg['tracking_timelapse']['terrain_sampling']
    for robot, factor in (density_scale or {}).items():
        sampling.setdefault('density_by_robot', {})[robot] = (
            sampling.get('density_by_robot', {}).get(robot, sampling.get('default_density', 1.))*factor)
    return cfg, variant


LAYOUT = {
    # Rows, top to bottom; the complete product name of each robot.
    'rows': [
        ('hopper2d_foot', 'Hopper'),
        ('go2', 'Unitree Go2'),
        ('h1', 'Unitree H1'),
        ('g1_29dof', 'Unitree G1'),
        ('talos', 'PAL Talos'),
    ],
    # Columns, left to right, as `VIEWS` names them. Not drawn.
    'columns': ['isometric', 'side'],
    # Multipliers on the terrain `terrain_sampling` density per view. The shared
    # terrain sampling is already thinned to 6 poses for H1 and Talos, so no
    # further thinning is applied by default; endpoints are always kept.
    'sample_density': {'isometric': {}, 'side': {}},
    'width_px': 1800,
    'column_in': 3.5,
    'text_pt': 8.0,
    'label_strip_factor': 1.5,
    'overflow_factor': 2.0,
    'trim_pad_px': 6,
    'legend_label': 'Elapsed time within each robot’s plan',
    'legend_width_fraction': 0.38,
    'combined_name': 'leap_terrain_tracking_timelapse_transposed',
}


def source_renders(directory, view, robots, args, overlay):
    """Render (or reuse) one terrain view with this figure's sampling, then load its panels."""
    cfg, variant = terrain_config(view, LAYOUT['sample_density'].get(view), overlay=overlay,
                                  assets=args.assets)
    # The renderer records the variant name it was invoked with in its manifest.
    expected = deepcopy(cfg)
    expected['tracking_timelapse']['variant'] = variant
    manifest_path = directory/'manifest.json'
    # Same staleness rule as the fence figure: only the style blocks the renderer
    # actually reads can invalidate a render (`scaled_grid.renders_fresh`).
    if not scaled_grid.renders_fresh(directory, expected, args):
        if args.reuse:
            raise ValueError(f'No up-to-date renders in {directory}; rerun without --reuse')
        directory.mkdir(parents=True, exist_ok=True)
        style = directory/'resolved_style.json'
        style.write_text(json.dumps(cfg, indent=2)+'\n')
        render_view(style, directory, variant, robots)
        if json.loads(manifest_path.read_text())['config'] != expected:
            raise ValueError(f'Stale configuration in {directory}')
    panels = scaled_grid.load_panels(directory, robots, expected, args)
    for panel in panels.values():
        panel['variant'] = variant
    return panels, {'path': str(manifest_path), 'sha256': sha256(manifest_path)}


def add_input_arguments(parser):
    parser.add_argument('--data', type=Path, required=True,
                        help='Tracked terrain suite: the --out folder of track.py --task terrain --all.')
    parser.add_argument('--archive-dir', type=Path,
                        help='Read sol_terrain_<robot>[_<tag>].h5 here instead of the archive each '
                             "case's metrics.json records.")
    parser.add_argument('--archive-tag')
    parser.add_argument('--assets', type=Path,
                        help='Visual mesh root (default: figure2/assets; see assets/ASSETS.md).')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_arguments(parser)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=LAYOUT['width_px'])
    parser.add_argument('--column-in', type=float, default=LAYOUT['column_in'])
    parser.add_argument('--scale', choices=('hip', 'fit'), default='hip')
    parser.add_argument('--overflow-factor', type=float, default=LAYOUT['overflow_factor'])
    parser.add_argument('--text-pt', type=float, default=LAYOUT['text_pt'])
    parser.add_argument('--trim-pad', type=int, default=LAYOUT['trim_pad_px'])
    parser.add_argument('--gutter', type=int)
    parser.add_argument('--gap', type=int)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--force-render', action='store_true')
    return parser


def run(args):
    # The shared style supplies spacing, typography and the legend; the obstacle
    # swatch takes the terrain label.
    cfg = load_style(assets=args.assets)
    cfg['tracking_timelapse']['obstacle_label'] = 'Stepping stones'
    s = cfg['tracking_timelapse']
    overlay = terrain_overlay(args.data, args.archive_dir, args.archive_tag)
    rows, columns = LAYOUT['rows'], LAYOUT['columns']
    robots = [robot for robot, _ in rows]
    if args.gutter is None:
        args.gutter = s['combined_gutter']
    if args.gap is None:
        args.gap = s['combined_row_gap']
    text_px = max(1, round(args.text_pt*args.width/(args.column_in*72.)))
    args.margin = round(text_px*LAYOUT['label_strip_factor'])
    args.column_label_factor = 0.
    args.out.mkdir(parents=True, exist_ok=True)

    panels, sources = {}, []
    for view in columns:
        panels[view], source = source_renders(args.source/view, view, robots, args, overlay)
        sources.append(source)
    args.legend_variant = panels[columns[0]][robots[0]]['variant']
    cells = [[panels[column][robot] for column in columns] for robot in robots]
    plan = scaled_grid.plan_layout(cells, args)
    grid = scaled_grid.compose(cells, plan, args, cfg, text_px, row_labels=[name for _, name in rows])
    combined, content_bounds = scaled_grid.add_legend(grid, args, cfg, text_px,
                                                      LAYOUT['legend_label'], LAYOUT['legend_width_fraction'])
    stem = args.out/LAYOUT['combined_name']
    save_figure(combined, stem, cfg)
    with Image.open(stem.with_suffix('.png')) as exported:
        exported.load()
        assert exported.size == combined.size
    subprocess.run(['pdfinfo', str(stem.with_suffix('.pdf'))], check=True, capture_output=True)

    capped = {id(p) for p in plan['outliers']}
    drawn = {c: {r: dict(size=panels[c][r]['size'], poses=panels[c][r]['poses'],
                         hips_wide=panels[c][r]['hips_wide'], hips_tall=panels[c][r]['hips_tall'],
                         fraction=panels[c][r]['fit'], capped=id(panels[c][r]) in capped)
                 for r in robots} for c in columns}
    report = dict(config=cfg, layout=LAYOUT, scale_mode=args.scale, rows=robots,
                  row_labels=[n for _, n in rows], columns=columns, column_labels=[],
                  row_heights=plan['row_heights'], column_widths=plan['column_widths'],
                  text_px=text_px, text_pt=args.text_pt, row_label_width_px=args.margin,
                  pixels_per_hip=plan['pixels_per_hip'],
                  hip_heights_m={r: panels[columns[0]][r]['hip'] for r in robots},
                  scale_note=f'every uncapped panel draws one hip height as {plan["hip_px"]:.1f} px',
                  capped_panels=[{'column': p['variant'], 'robot': p['robot'], 'fraction': p['fit']}
                                 for p in plan['outliers']],
                  panels=drawn, size=list(combined.size), content_bounds=content_bounds,
                  printed_size_in=[args.column_in, args.column_in*combined.height/combined.width],
                  poses_only='tracked configurations; no LEAP reference, titles or timestamps drawn',
                  sources=sources, outputs={p.name: sha256(p) for p in
                  (stem.with_suffix('.png'), stem.with_suffix('.pdf'))}, passed=True)
    (args.out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved and validated {stem}.png and .pdf', flush=True)
    print(f'{combined.width} x {combined.height} px; {args.column_in:.2f} x '
          f'{report["printed_size_in"][1]:.2f} in printed at {args.width/args.column_in:.0f} dpi; '
          f'labels {args.text_pt:.1f} pt; one hip height = {plan["hip_px"]:.0f} px', flush=True)
    for column in columns:
        print(f'  {column}: poses ' + ', '.join(f'{r}={drawn[column][r]["poses"]}' for r in robots), flush=True)
    for entry in report['capped_panels']:
        print(f'  scaled down to fit: {entry["robot"]} {entry["column"]} at {entry["fraction"]*100:.0f}% of the common scale', flush=True)
    return report


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
