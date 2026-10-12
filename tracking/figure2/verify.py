"""Validates one view's renders: selected poses and times,
colors, camera, crop, timestamp labels, PNG sizes, PDFs (needs `pdfinfo`).

    python -m figure2.verify --out <view folder>
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from .render import labels_overlap
from .colors import sample_pose_colors


def verify(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    cfg = manifest['config']
    reports = []
    for robot in manifest['robots']:
        stem = directory/f'{robot}_tracking_timelapse'
        m = json.loads(stem.with_suffix('.json').read_text())
        with np.load(m['sources']['rollout']['path'], allow_pickle=False) as z:
            indices = np.asarray(m['selected_indices'])
            np.testing.assert_array_equal(z['q'][indices], m['selected_q'])
            np.testing.assert_array_equal(z['t'][indices], m['selected_times'])
            nearest = [np.argmin(np.abs(z['t']-t)) for t in m['requested_times']]
            np.testing.assert_array_equal(indices, nearest)
        np.testing.assert_allclose(m['rendered_fence_rgba'], np.broadcast_to(cfg['scene']['fence_rgba'], np.asarray(m['rendered_fence_rgba']).shape), rtol=0, atol=1e-7)
        if 'rendered_floor_rgba' in m:
            np.testing.assert_allclose(m['rendered_floor_rgba'], cfg['scene']['floor_rgba'], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(m['pose_rgba'][0][:3], cfg['states']['initial']['rgba'][:3])
        np.testing.assert_array_equal(m['pose_rgba'][-1][:3], cfg['states']['goal']['rgba'][:3])
        if cfg['tracking_timelapse'].get('color_mode') == 'time_gradient':
            expected_colors, expected_progress = sample_pose_colors(cfg, m['selected_times'])
            np.testing.assert_allclose(m['pose_rgba'], expected_colors, rtol=0, atol=1e-12)
            np.testing.assert_allclose(m['time_progress'], expected_progress, rtol=0, atol=1e-12)
        if not m['config']['tracking_timelapse']['orthographic']:
            raise ValueError('Figure is not orthographic')
        np.testing.assert_allclose(m.get('camera_angles', [90., 0.]),
            [cfg['tracking_timelapse']['azimuth'], cfg['tracking_timelapse']['elevation']])
        raw, crop = np.array(m['silhouette_bbox']), np.array(m['crop_bbox'])
        if np.any(raw[:2] < crop[:2]) or np.any(raw[2:] > crop[2:]):
            raise ValueError('Crop clips visible geometry')
        occupancy = (raw[2:]-raw[:2])/(crop[2:]-crop[:2])
        if np.min(occupancy) < cfg['tracking_timelapse']['minimum_crop_occupancy']:
            raise ValueError(f'Excess blank border: {robot} {occupancy}')
        boxes = m['timestamp_label_boxes']
        if labels_overlap(boxes):
            raise ValueError('Timestamp overlap')
        with Image.open(stem.with_suffix('.png')) as image:
            image.load()
            if image.width != m['scene_size'][0]:
                raise ValueError('Unexpected figure width')
        subprocess.run(['pdfinfo', str(stem.with_suffix('.pdf'))], check=True, capture_output=True)
        reports.append(dict(robot=robot, poses=len(indices), crop_occupancy=occupancy.tolist(), passed=True))
    combined = directory/cfg['tracking_timelapse']['combined_name']
    with Image.open(combined.with_suffix('.png')) as image:
        image.load()
        if list(image.size) != manifest['combined_size']:
            raise ValueError('Incorrect combined dimensions')
    subprocess.run(['pdfinfo', str(combined.with_suffix('.pdf'))], check=True, capture_output=True)
    result = dict(passed=True, figures=len(reports)+1, selected_poses=sum(r['poses'] for r in reports), results=reports)
    (directory/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Validated {len(reports)} individual figures and combined PNG/PDF: source poses, colors, crop, labels and PDFs.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    verify(parser.parse_args().out)
