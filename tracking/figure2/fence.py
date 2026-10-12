"""Fence figure: a row per robot (Hopper, Go2, H1, G1, Talos), an isometric and a side
view column, one pixels-per-hip-height, time-gradient legend.

    python -m figure2.fence --data <tracked fence suite> --source <dir>/src_fence --out <dir>/fence
"""
from pathlib import Path
import argparse
import json
import subprocess

from PIL import Image
from .style import DEFAULT_CONFIG, load_style
from .data import sha256
from . import scaled_grid
from .render import save_figure

# Layout for the transposed figure.
LAYOUT = {
    # Rows, top to bottom; the complete product name of each robot.
    'rows': [
        ('hopper2d', 'Hopper'),
        ('go2', 'Unitree Go2'),
        ('h1', 'Unitree H1'),
        ('g1_29dof', 'Unitree G1'),
        ('talos', 'PAL Talos'),
    ],
    # Columns, left to right. Their names are intentionally not drawn.
    'columns': ['isometric_dense_gradient', 'side_dense_gradient'],
    # Multipliers on the shared `fence_sampling.density_by_robot`, per view. The
    # isometric camera stacks the poses along the line of sight and the two
    # tallest humanoids overlap heavily from the side, so those panels are
    # thinned; the endpoints are always kept, only intermediate poses are lost.
    'sample_density': {
        'isometric_dense_gradient': {'go2': .75, 'h1': .75, 'g1_29dof': .75, 'talos': .75},
        'side_dense_gradient': {'g1_29dof': .75, 'talos': .75},
    },
    'width_px': 1800,
    'column_in': 3.5,
    # Typography is specified in PRINTED points, not pixels: this figure is sized
    # for one IEEE column, where the pixel sizes inherited from the much wider
    # source figure would set every label at about 4 pt.
    'text_pt': 8.0,
    'label_strip_factor': 1.5,
    # A panel wider than this multiple of the median panel width (both measured
    # in robot hip heights) is an envelope outlier: it sets no column width and
    # is scaled down to fit instead of forcing every other robot to shrink.
    'overflow_factor': 2.0,
    'trim_pad_px': 6,
    'legend_label': 'Elapsed time within each robot’s plan',
    'legend_width_fraction': 0.38,
    'combined_name': 'leap_fence_tracking_timelapse_transposed',
}


def fence_overlay(data, archive_dir=None, archive_tag=None):
    """`tracking_overlay` entries naming one fence suite's tracked folder and archives."""
    return dict(benchmark='fence', data_root=str(Path(data).resolve()),
                archive_dir=str(Path(archive_dir).resolve()) if archive_dir else None,
                archive_tag=archive_tag or None)


def fence_style(config=DEFAULT_CONFIG, data=None, archive_dir=None, archive_tag=None, assets=None):
    """The style the fence renders are made (and checked for staleness) with."""
    return load_style(config, overlay=fence_overlay(data, archive_dir, archive_tag), assets=assets)


def add_input_arguments(parser):
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data', type=Path, required=True,
                        help='Tracked fence suite: the --out folder of track.py --task fence --all.')
    parser.add_argument('--archive-dir', type=Path,
                        help='Read sol_fence_<robot>[_<tag>].h5 here instead of the archive each '
                             "case's metrics.json records.")
    parser.add_argument('--archive-tag')
    parser.add_argument('--assets', type=Path,
                        help='Visual mesh root (default: figure2/assets; see assets/ASSETS.md).')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_arguments(parser)
    parser.add_argument('--source', type=Path, required=True,
                        help="Directory holding this figure's per-view render subdirectories.")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=LAYOUT['width_px'],
                        help='Export width in pixels; the figure is drawn to fill it.')
    parser.add_argument('--column-in', type=float, default=LAYOUT['column_in'],
                        help='Printed width in inches; label point sizes are set for it.')
    parser.add_argument('--scale', choices=('hip', 'fit'), default='hip',
                        help='hip: one pixels-per-hip-height for every robot. fit: fill each column.')
    parser.add_argument('--overflow-factor', type=float, default=LAYOUT['overflow_factor'])
    parser.add_argument('--text-pt', type=float, default=LAYOUT['text_pt'],
                        help='Printed size of every label, in points at --column-in wide.')
    parser.add_argument('--trim-pad', type=int, default=LAYOUT['trim_pad_px'],
                        help='Background pixels kept around each silhouette, in render pixels.')
    parser.add_argument('--gutter', type=int, help='Horizontal spacing; defaults to the shared style.')
    parser.add_argument('--gap', type=int, help='Vertical spacing; defaults to the shared style.')
    parser.add_argument('--reuse', action='store_true',
                        help='Fail instead of rendering when the source renders are missing or stale.')
    parser.add_argument('--force-render', action='store_true',
                        help='Re-render both views even when validated renders already exist.')
    return parser


def run(args):
    cfg = fence_style(args.config, args.data, args.archive_dir, args.archive_tag, args.assets)
    s = cfg['tracking_timelapse']
    rows, columns = LAYOUT['rows'], LAYOUT['columns']
    robots = [robot for robot, _ in rows]
    if args.gutter is None:
        args.gutter = s['combined_gutter']
    if args.gap is None:
        args.gap = s['combined_row_gap']
    # Points of printed text -> pixels of this export.
    text_px = max(1, round(args.text_pt*args.width/(args.column_in*72.)))
    args.margin = round(text_px*LAYOUT['label_strip_factor'])
    args.column_label_factor = 0.
    args.legend_variant = columns[0]
    args.out.mkdir(parents=True, exist_ok=True)

    panels, sources = {}, []
    for variant in columns:
        panels[variant], source = scaled_grid.source_renders(
            args.source/variant, variant, robots, cfg, args, LAYOUT['sample_density'])
        sources.append(source)
    cells = [[panels[column][robot] for column in columns] for robot in robots]
    plan = scaled_grid.plan_layout(cells, args)
    grid = scaled_grid.compose(cells, plan, args, cfg, text_px,
                               row_labels=[name for _, name in rows])
    combined, content_bounds = scaled_grid.add_legend(grid, args, cfg, text_px,
                                                      LAYOUT['legend_label'],
                                                      LAYOUT['legend_width_fraction'])
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
                  capped_panels=[{'column': p['variant'], 'robot': p['robot'],
                                  'fraction': p['fit']} for p in plan['outliers']],
                  panels=drawn, size=list(combined.size), content_bounds=content_bounds,
                  printed_size_in=[args.column_in, args.column_in*combined.height/combined.width],
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
        print(f'  scaled down to fit: {entry["robot"]} {entry["column"]} at '
              f'{entry["fraction"]*100:.0f}% of the common scale', flush=True)
    return report


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
