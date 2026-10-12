"""Unit tests of the pose sampling, the time colors and the timestamp layout.

    cd new_src/tracking && python -m unittest figure2.test_sampling
"""
from types import SimpleNamespace
import unittest
import numpy as np
from .render import select_samples, labels_overlap


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.run = SimpleNamespace(ref=SimpleNamespace(spec=SimpleNamespace(t_stance=2., t_air=4.), duration=10.),
                                   samples={'t': np.linspace(0., 10., 1001)})

    def test_dense_three_phases(self):
        s = {'stance_fractions': [.15, .5, .85], 'flight_fractions': [.25, .5, .75],
             'landing_fractions': [.15, .5, .85]}
        indices, targets = select_samples(self.run, {'tracking_timelapse': s})
        expected = [0., .3, 1., 1.7, 3., 4., 5., 6.6, 8., 9.4, 10.]
        np.testing.assert_allclose(targets, expected)
        np.testing.assert_allclose(self.run.samples['t'][indices], expected)

    def test_original_side_sampling(self):
        _, targets = select_samples(self.run, {'tracking_timelapse': {'flight_fractions': [.25, .5, .75]}})
        np.testing.assert_allclose(targets, [0., 3., 4., 5., 10.])

    def test_terrain_samples_saved_ticks(self):
        cfg = {'tracking_overlay': {'benchmark': 'terrain'},
               'tracking_timelapse': {'terrain_pose_count': 5}}
        indices, targets = select_samples(self.run, cfg)
        np.testing.assert_allclose(targets, [0., 2.5, 5., 7.5, 10.])
        np.testing.assert_array_equal(indices, [0, 250, 500, 750, 1000])

    def test_fence_density_and_terrain_isolation(self):
        from .style import load_style, DEFAULT_CONFIG
        cfg = load_style(DEFAULT_CONFIG)
        cfg['tracking_timelapse']['fence_sampling_profile'] = 'dense'
        for robot, count in [('hopper2d', 11), ('go2', 11), ('h1', 11),
                             ('g1_29dof', 11), ('talos', 6)]:
            self.run.robot = robot
            indices, targets = select_samples(self.run, cfg)
            self.assertEqual(len(indices), count)
            np.testing.assert_allclose(targets[[0, -1]], [0., 10.])
            self.assertTrue(np.all(np.diff(indices) > 0))
        cfg['tracking_overlay']['benchmark'] = 'terrain'
        cfg['tracking_timelapse']['terrain_pose_count'] = 20
        indices, targets = select_samples(self.run, cfg)
        self.assertEqual(len(indices), 20)
        np.testing.assert_allclose(targets, np.linspace(0., 10., 20))

    def test_terrain_density_config(self):
        from copy import deepcopy
        from .terrain import TIMELAPSE_CONFIG
        settings = deepcopy(TIMELAPSE_CONFIG)
        cfg = {'tracking_overlay': {'benchmark': 'terrain'},
               'tracking_timelapse': {'terrain_pose_count': 11, **settings}}
        for robot, count in [('hopper2d_foot', 11), ('go2', 11), ('g1_29dof', 11), ('talos', 6)]:
            self.run.robot = robot
            indices, targets = select_samples(self.run, cfg)
            self.assertEqual(len(indices), count)
            np.testing.assert_allclose(targets[[0, -1]], [0., 10.])
        cfg['tracking_overlay']['benchmark'] = 'fence'
        cfg['tracking_timelapse']['flight_fractions'] = [.25, .5, .75]
        self.assertEqual(len(select_samples(self.run, cfg)[0]), 5)

    def test_timestamp_rows(self):
        self.assertFalse(labels_overlap([[0, 0, 10, 10], [0, 15, 10, 25]]))
        self.assertTrue(labels_overlap([[0, 0, 10, 10], [5, 5, 15, 15]]))


class TimeColorTests(unittest.TestCase):
    def test_gradient_uses_time_not_sample_number(self):
        from .colors import sample_pose_colors
        cfg = {'states': {'initial': {'rgba': [.35, .35, .4, 1.]},
                          'goal': {'rgba': [0., .62, .45, 1.]}},
               'tracking_timelapse': {'color_mode': 'time_gradient', 'motion_alpha': .24, 'endpoint_alpha': 1.}}
        colors, progress = sample_pose_colors(cfg, [0., .1, .9, 1.])
        np.testing.assert_allclose(progress, [0., .1, .9, 1.])
        start, end = np.array([.35, .35, .4]), np.array([0., .62, .45])
        np.testing.assert_allclose(colors[:, :3], [(1-t)*start+t*end for t in progress])
        np.testing.assert_array_equal(colors[:, 3], [1., .24, .24, 1.])


if __name__ == '__main__':
    unittest.main()
