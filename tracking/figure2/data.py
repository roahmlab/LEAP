"""Reads one tracked case (`<data_root>/<robot>/rollout.npz`, `metrics.json`), loads the
LEAP archive it tracked (the `archive` in metrics.json, or `sol_<task>_<robot>[_<tag>].h5`
in `tracking_overlay.archive_dir`) with the case's problem yaml (`<robot>_jump_fence.yaml`
/ `<robot>_terrain_walk.yaml` in the case folder, else the archive's copy), and checks
every recorded q_ref against the archive (tolerance 1e-8).
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from leaptrack.reference import YAML_SUFFIX, load_leap_reference


@dataclass
class TrackingRun:
    robot: str
    ref: object
    samples: dict
    metrics: dict
    audit: dict


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_recording(directory, method, task, robot):
    """One saved tracking run's metrics and paired samples, with the identity and
    sample checks every renderer relies on."""
    metrics_path, rollout_path = directory/'metrics.json', directory/'rollout.npz'
    metrics = json.loads(metrics_path.read_text())
    if (metrics['method'], metrics['benchmark'], metrics['robot']) != (method, task, robot):
        raise ValueError(f'Unexpected tracking identity in {metrics_path}')
    with np.load(rollout_path, allow_pickle=False) as z:
        samples = {k: z[k].copy() for k in ('t', 't_ref', 'q', 'q_ref', 'phase', 'frozen')}
        duration = float(z['duration'])
    t = samples['t']
    if len(t) < 2 or t[0] != 0 or not np.all(np.diff(t) > 0):
        raise ValueError(f'Invalid recording times in {rollout_path}')
    if any(len(v) != len(t) or not np.isfinite(v).all() for v in samples.values()):
        raise ValueError(f'Invalid sample arrays in {rollout_path}')
    return metrics, samples, duration, metrics_path, rollout_path


def run_directory(robot, cfg, data_root):
    directory = Path(data_root or cfg['tracking_overlay']['data_root'])
    return directory.resolve() / robot


def archive_path(task, robot, metrics, cfg):
    """`sol_<task>_<robot>[_<tag>].h5` in `tracking_overlay.archive_dir` when set,
    else the archive the case's metrics.json records."""
    overlay = cfg['tracking_overlay']
    if overlay.get('archive_dir'):
        tag = overlay.get('archive_tag')
        return Path(overlay['archive_dir']).resolve()/(f'sol_{task}_{robot}' + (f'_{tag}' if tag else '') + '.h5')
    if not metrics.get('archive'):
        raise ValueError(f'{task}/{robot}: metrics.json records no archive; pass an archive directory')
    return Path(metrics['archive'])


def load_run(robot, cfg, data_root=None, work_dir=None):
    """Fail on invalid samples or mismatched reference; never substitute a plan.

    q_ref is what the controller actually received, at its recorded t_ref.
    Validate every q_ref against the run's archive. Display the logged samples,
    preserving touchdown clock behavior. `work_dir` receives the problem yaml the
    reference is built from (`leaptrack.reference.materialize_problem_yaml`).
    """
    task = cfg['tracking_overlay'].get('benchmark', 'fence')
    directory = run_directory(robot, cfg, data_root)
    metrics, samples, duration, metrics_path, rollout_path = read_recording(
        directory, 'leap', task, robot)
    t, q, qr = samples['t'], samples['q'], samples['q_ref']
    archive = archive_path(task, robot, metrics, cfg)
    own_yaml = directory/f'{robot}_{YAML_SUFFIX[task]}.yaml'
    ref = load_leap_reference(task, robot, archive, problem_yaml=own_yaml if own_yaml.exists() else None,
                              work_dir=work_dir)
    if q.shape != qr.shape or q.shape != (len(t), ref.nq) or not np.isclose(duration, ref.duration):
        raise ValueError('Tracking samples and archive dimensions/duration disagree')
    archive_q = np.array([ref.leap.eval(float(min(tr, ref.duration)))[0] for tr in samples['t_ref']])
    error = float(np.max(np.abs(archive_q-qr)))
    if error > 1e-8:
        raise ValueError(f'Reference mismatch for {robot}: tracking={error}')
    paths = dict(rollout=rollout_path, metrics=metrics_path, archive=archive,
                 problem=Path(ref.settings['problem_yaml']))
    audit = dict(robot=robot, samples=len(t), plan_duration=duration, recorded_duration=float(t[-1]),
        archive_q_ref_max_error=error,
        reference_clock_shift_range=[float(np.min(samples['t_ref']-t)), float(np.max(samples['t_ref']-t))],
        sources={key: dict(path=str(path.resolve()), sha256=sha256(path)) for key, path in paths.items()},
        problem_yaml_source=ref.settings['problem_yaml_source'],
        joint_names=list(ref.robot.model.names)[1:],
        alignment='paired q and q_ref at the same recorded control tick; q_ref follows t_ref',
        validation='every recorded q_ref checked against the LEAP archive; tolerance 1e-8')
    return TrackingRun(robot, ref, samples, metrics, audit)
