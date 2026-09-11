"""
test_noise_filter.py - Comprehensive unit tests for noise filtering algorithms in lidar_workbench.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lidar_workbench.noise_filter import (
    _estimate_surface_grid,
    dbscan_outlier_removal,
    low_point_removal,
    surface_noise_removal,
    multipath_reflection_removal,
    bilateral_filter,
    elevation_window_filter,
    _apply_pipeline,
)


class TestNoiseFilters(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a 50x50m sloping terrain patch
        self.n_ground = 2000
        self.xs = np.random.uniform(0, 50, self.n_ground)
        self.ys = np.random.uniform(0, 50, self.n_ground)
        # Ground equation: Z = 100 + 0.1*X + 0.05*Y + small roughness
        self.zs = 100.0 + 0.1 * self.xs + 0.05 * self.ys + np.random.normal(0, 0.02, self.n_ground)

    def test_estimate_surface_grid(self):
        """Surface estimation should accurately approximate the sloping terrain."""
        surf_z = _estimate_surface_grid(self.xs, self.ys, self.zs, grid_size=2.0)
        self.assertEqual(len(surf_z), len(self.zs))
        # Differences should be within a few cm of the true ground
        diffs = np.abs(self.zs - surf_z)
        self.assertLess(np.median(diffs), 0.1)

    def test_dbscan_outlier_removal_above(self):
        """DBSCAN above should only cluster and remove small clusters above height_above_surface."""
        # Add 5 aerial noise points (small cluster) at +15m
        high_noise_x = np.array([25.0, 25.1, 25.2, 25.3, 25.4])
        high_noise_y = np.array([25.0, 25.1, 25.2, 25.3, 25.4])
        high_noise_z = np.array([120.0, 120.1, 119.9, 120.2, 120.0])

        all_xs = np.concatenate([self.xs, high_noise_x])
        all_ys = np.concatenate([self.ys, high_noise_y])
        all_zs = np.concatenate([self.zs, high_noise_z])

        keep, outlier = dbscan_outlier_removal(
            all_xs, all_ys, all_zs,
            eps=1.0,
            min_samples=2,
            min_cluster_size=10,
            mode="above",
            height_above_surface=2.0,
        )

        # Ground points must NOT be touched
        self.assertTrue(np.all(keep[:self.n_ground]))
        # High noise points should be flagged as outliers
        self.assertTrue(np.all(outlier[self.n_ground:]))

    def test_dbscan_outlier_removal_below(self):
        """DBSCAN below should only cluster and remove small clusters below depth_below_surface."""
        # Add 4 underground multipath points at -10m
        low_noise_x = np.array([10.0, 10.1, 10.2, 10.3])
        low_noise_y = np.array([10.0, 10.1, 10.2, 10.3])
        low_noise_z = np.array([90.0, 90.1, 89.9, 90.2])

        all_xs = np.concatenate([self.xs, low_noise_x])
        all_ys = np.concatenate([self.ys, low_noise_y])
        all_zs = np.concatenate([self.zs, low_noise_z])

        keep, outlier = dbscan_outlier_removal(
            all_xs, all_ys, all_zs,
            eps=1.0,
            min_samples=2,
            min_cluster_size=10,
            mode="below",
            depth_below_surface=2.0,
        )

        # Ground points must be kept
        self.assertTrue(np.all(keep[:self.n_ground]))
        # Pit noise points should be flagged as outliers
        self.assertTrue(np.all(outlier[self.n_ground:]))

    def test_low_point_removal(self):
        """Low point filter should flag isolated pit points below surrounding neighbors."""
        # Add 3 pit points 5m below surrounding ground
        pit_x = np.array([15.0, 30.0, 42.0])
        pit_y = np.array([15.0, 30.0, 42.0])
        pit_z = 100.0 + 0.1 * pit_x + 0.05 * pit_y - 5.0

        all_xs = np.concatenate([self.xs, pit_x])
        all_ys = np.concatenate([self.ys, pit_y])
        all_zs = np.concatenate([self.zs, pit_z])

        keep, outlier = low_point_removal(
            all_xs, all_ys, all_zs,
            search_radius=3.0,
            below_threshold=1.0,
            above_threshold=10.0,
        )

        # All 3 pit points should be flagged as outliers
        self.assertTrue(np.all(outlier[self.n_ground:]))
        # Almost all ground points should be kept (>99%)
        self.assertGreater(keep[:self.n_ground].sum() / self.n_ground, 0.99)

    def test_surface_noise_removal(self):
        """Surface noise filter should flag points within tolerance of the surface."""
        # Test near_surface mode
        keep, outlier = surface_noise_removal(
            self.xs, self.ys, self.zs,
            grid_size=2.0,
            tolerance=0.05,
            mode="near_surface",
        )
        self.assertEqual(len(keep), self.n_ground)
        self.assertEqual(len(outlier), self.n_ground)

    def test_elevation_window_filter(self):
        """Elevation window filter should flag points outside absolute and relative bounds."""
        # Points from 100m to 108m
        # 1. Absolute limits: min_z=102.0
        keep, outlier = elevation_window_filter(
            self.xs, self.ys, self.zs,
            min_z=102.0,
        )
        expected_out = self.zs < 102.0
        np.testing.assert_array_equal(outlier, expected_out)

        # 2. Relative height limits: min_height=-0.5, max_height=10.0
        # Add a high cloud point at +30m and a pit at -5m
        cloud_x = np.array([25.0])
        cloud_y = np.array([25.0])
        cloud_z = np.array([135.0])

        pit_x = np.array([26.0])
        pit_y = np.array([26.0])
        pit_z = np.array([95.0])

        all_xs = np.concatenate([self.xs, cloud_x, pit_x])
        all_ys = np.concatenate([self.ys, cloud_y, pit_y])
        all_zs = np.concatenate([self.zs, cloud_z, pit_z])

        keep, outlier = elevation_window_filter(
            all_xs, all_ys, all_zs,
            min_height_above_ground=-1.0,
            max_height_above_ground=20.0,
        )
        # Cloud and pit must be flagged
        self.assertTrue(outlier[-2])  # cloud
        self.assertTrue(outlier[-1])  # pit
        # Ground points must be kept
        self.assertTrue(np.all(keep[:self.n_ground]))

    def test_bilateral_filter(self):
        """Bilateral filter should smooth points while preserving shapes."""
        sx, sy, sz = bilateral_filter(
            self.xs[:100], self.ys[:100], self.zs[:100],
            spatial_sigma=0.5,
            range_sigma=0.1,
            knn=10,
        )
        self.assertEqual(len(sx), 100)
        self.assertEqual(len(sy), 100)
        self.assertEqual(len(sz), 100)
        # Smoothed Z should stay close to original Z
        self.assertLess(np.median(np.abs(sz - self.zs[:100])), 0.2)
        self.assertLess(np.max(np.abs(sz - self.zs[:100])), 1.0)

    def test_apply_pipeline_with_elevation_window_and_dbscan(self):
        """Pipeline execution should chain elevation_window and dbscan correctly."""
        # Add 1 cloud point at Z=200
        cloud_x = np.array([10.0])
        cloud_y = np.array([10.0])
        cloud_z = np.array([200.0])

        data = {
            "x": np.concatenate([self.xs, cloud_x]),
            "y": np.concatenate([self.ys, cloud_y]),
            "z": np.concatenate([self.zs, cloud_z]),
        }

        pipeline = [
            {
                "type": "elevation_window",
                "max_z": 150.0,
            },
            {
                "type": "dbscan_above",
                "eps": 1.0,
                "min_samples": 5,
                "min_cluster_size": 10,
                "height_above_surface": 2.0,
            }
        ]

        keep = _apply_pipeline(data, pipeline)
        # Cloud point at index -1 should be removed
        self.assertFalse(keep[-1])
        # Ground points should be kept
        self.assertTrue(np.all(keep[:self.n_ground]))


if __name__ == "__main__":
    unittest.main()
