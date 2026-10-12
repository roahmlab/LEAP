"""Time-based pose colors shared by both timelapse camera views.
"""
import numpy as np


def sample_pose_colors(cfg, times):
    """Interpolate endpoint sRGB by actual elapsed time, never by sample index."""
    s = cfg['tracking_timelapse']
    times = np.asarray(times, float)
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Pose times must be finite, strictly increasing samples')
    if not np.isfinite(times).all():
        raise ValueError('Non-finite pose time')
    start = np.asarray(cfg['states']['initial']['rgba'][:3])
    finish = np.asarray(cfg['states']['goal']['rgba'][:3])
    progress = (times-times[0])/(times[-1]-times[0])
    if s.get('color_mode') == 'time_gradient':
        rgb = (1-progress[:, None])*start + progress[:, None]*finish
    else:
        rgb = np.tile(cfg['methods'][s['motion_color_role']]['rgba'][:3], (len(times), 1))
    rgb[0], rgb[-1] = start, finish
    alpha = np.full((len(times), 1), s['motion_alpha'])
    alpha[0] = alpha[-1] = s['endpoint_alpha']
    return np.column_stack((rgb, alpha)), progress
