"""Unit tests for LiDAR Strip Adjustment (StripAdj) engine."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from lidar_workbench.strip_adjustment import (
    StripAdjustmentSolver,
    StripCorrection,
    TieSurface,
    Trajectory,
    apply_strip_corrections,
    extract_tie_planes,
)
from lidar_workbench.gui.strip_adjustment_dialog import StripAdjustmentDialog


class TestStripAdjustment(unittest.TestCase):

    def test_trajectory_parsing_and_interp(self):
        # Create a synthetic trajectory file in OPALS/Riegl format: [t, x, y, z, roll, pitch, yaw]
        content = (
            "100.0 500000.0 5200000.0 1200.0 0.0 0.0 90.0\n"
            "110.0 500500.0 5200000.0 1200.0 0.5 -0.2 90.0\n"
            "120.0 501000.0 5200000.0 1200.0 1.0 -0.4 90.0\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            traj = Trajectory.from_file(tmp_path)
            self.assertEqual(len(traj.times), 3)
            self.assertAlmostEqual(traj.t_min, 100.0)
            self.assertAlmostEqual(traj.t_max, 120.0)

            # Query at t = 105.0 (midway between 100 and 110)
            q_times = np.array([105.0])
            qx, qy, qz = traj.interpolate_position(q_times)
            self.assertAlmostEqual(qx[0], 500250.0)
            self.assertAlmostEqual(qy[0], 5200000.0)
            self.assertAlmostEqual(qz[0], 1200.0)

            r, p, y = traj.interpolate_attitude(q_times)
            self.assertAlmostEqual(r[0], 0.25)
            self.assertAlmostEqual(p[0], -0.10)
            self.assertAlmostEqual(y[0], 90.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_step1_xyz_solver(self):
        # 2 strips: Strip 1 (reference, fixed), Strip 2 has known offset: dx=0.08, dy=-0.05, dz=0.12
        # Generate simulated tie planes on flat and sloping surfaces
        true_dx, true_dy, true_dz = 0.08, -0.05, 0.12
        tie_surfaces = []
        rng = np.random.default_rng(42)

        for _ in range(200):
            cx = rng.uniform(100.0, 300.0)
            cy = rng.uniform(100.0, 300.0)
            cz = rng.uniform(50.0, 80.0)
            cb = np.array([cx, cy, cz])

            # Random normal with upward Z component
            nx = rng.uniform(-0.3, 0.3)
            ny = rng.uniform(-0.3, 0.3)
            nz = math.sqrt(max(0.1, 1.0 - nx*nx - ny*ny))
            normal_b = np.array([nx, ny, nz])
            normal_b /= np.linalg.norm(normal_b)

            # In Strip 2: point is shifted by true offset
            # pa is in Strip 2, cb is in Strip 1
            pa = cb + np.array([true_dx, true_dy, true_dz]) + normal_b * rng.normal(0, 0.005)

            tie_surfaces.append(TieSurface(
                strip_a=2,
                strip_b=1,
                point_a=pa,
                centroid_b=cb,
                normal_b=normal_b,
                time_a=100.0,
                time_b=200.0,
                roughness=0.01,
            ))

        solver = StripAdjustmentSolver(strip_ids=[1, 2], ref_strip_id=1)
        shifts = solver.solve_step1_xyz(tie_surfaces)

        # Strip 1 is reference -> shift ~ 0
        self.assertAlmostEqual(shifts[1][0], 0.0, delta=0.01)
        self.assertAlmostEqual(shifts[1][1], 0.0, delta=0.01)
        self.assertAlmostEqual(shifts[1][2], 0.0, delta=0.01)

        # Strip 2 shift should compensate true offset (dx ~ -0.08, dy ~ 0.05, dz ~ -0.12)
        # because point_corrected = pa + shift -> pa - true_offset = cb
        self.assertAlmostEqual(shifts[2][0], -true_dx, delta=0.02)
        self.assertAlmostEqual(shifts[2][1], -true_dy, delta=0.02)
        self.assertAlmostEqual(shifts[2][2], -true_dz, delta=0.02)

    def test_step2_boresight_solver(self):
        # 2 strips flown in opposite directions (e.g. along Y axis)
        # Strip 2 has a known roll misalignment of +0.03 deg (scissors tilt across X)
        true_roll_deg = 0.03
        true_roll_rad = math.radians(true_roll_deg)
        sensor_z = 500.0
        tie_surfaces = []
        rng = np.random.default_rng(101)

        for _ in range(250):
            # Points spread across track: x in [-150, 150]
            x = rng.uniform(-150.0, 150.0)
            y = rng.uniform(0.0, 500.0)
            z = 50.0 + rng.uniform(-5.0, 5.0)
            cb = np.array([x, y, z])
            normal_b = np.array([0.0, 0.0, 1.0])  # horizontal ground

            # Roll rotates around along-track axis (Y): dz = x * sin(roll)
            dz = x * math.sin(true_roll_rad)
            pa = cb + np.array([0.0, 0.0, dz]) + normal_b * rng.normal(0, 0.003)

            tie_surfaces.append(TieSurface(
                strip_a=2,
                strip_b=1,
                point_a=pa,
                centroid_b=cb,
                normal_b=normal_b,
                time_a=100.0,
                time_b=200.0,
                roughness=0.01,
            ))

        # Trajectories 500m high along nadir x=0
        traj1 = Trajectory(np.array([190.0, 210.0]), np.array([0.0, 0.0]), np.array([0.0, 500.0]), np.array([sensor_z, sensor_z]))
        traj2 = Trajectory(np.array([90.0, 110.0]), np.array([0.0, 0.0]), np.array([500.0, 0.0]), np.array([sensor_z, sensor_z]))

        solver = StripAdjustmentSolver(strip_ids=[1, 2], trajectories={1: traj1, 2: traj2}, ref_strip_id=1)
        angles = solver.solve_step2_boresight(tie_surfaces)

        # Strip 1 reference -> 0
        self.assertAlmostEqual(angles[1][0], 0.0, delta=0.005)
        # Strip 2 roll should be corrective angle (~ +0.03 deg to undo negative tilt)
        self.assertAlmostEqual(angles[2][0], true_roll_deg, delta=0.01)

    def test_step3_drift_solver(self):
        # 2 strips where Strip 2 has a time-dependent vertical drift:
        # dz(t) drifts from 0.0 at t=100 to 0.10m at t=160
        tie_surfaces = []
        rng = np.random.default_rng(77)

        for _ in range(150):
            t = rng.uniform(100.0, 160.0)
            true_drift = (t - 100.0) / 60.0 * 0.10  # 0 to 10cm drift
            cb = np.array([rng.uniform(0, 100), rng.uniform(0, 500), 50.0])
            normal_b = np.array([0.0, 0.0, 1.0])
            pa = cb + np.array([0.0, 0.0, true_drift])

            tie_surfaces.append(TieSurface(
                strip_a=2,
                strip_b=1,
                point_a=pa,
                centroid_b=cb,
                normal_b=normal_b,
                time_a=t,
                time_b=t,
                roughness=0.01,
            ))

        solver = StripAdjustmentSolver(strip_ids=[1, 2], ref_strip_id=1)
        drift_results = solver.solve_step3_drift(tie_surfaces, time_knot_interval=15.0)

        self.assertIn(2, drift_results)
        nodes = drift_results[2]
        self.assertGreaterEqual(len(nodes), 3)
        # First node near t=100 should be ~ -0.00 (corrective)
        self.assertAlmostEqual(nodes[0][3], 0.0, delta=0.03)
        # Last node near t=160 should be ~ -0.10 (corrective)
        self.assertAlmostEqual(nodes[-1][3], -0.10, delta=0.03)

    def test_apply_strip_corrections(self):
        xs = np.array([10.0, 20.0, 30.0])
        ys = np.array([50.0, 50.0, 50.0])
        zs = np.array([100.0, 100.0, 100.0])
        times = np.array([1.0, 2.0, 3.0])

        corr = StripCorrection(
            strip_id=1,
            dx=0.05,
            dy=-0.02,
            dz=0.10,
            droll=0.0,
            dpitch=0.0,
            dyaw=0.0,
            drift_nodes=[(1.0, 0, 0, 0.01), (3.0, 0, 0, 0.03)],
        )

        out_x, out_y, out_z = apply_strip_corrections(xs, ys, zs, times, corr)

        # Check translation
        self.assertAlmostEqual(out_x[0], 10.05)
        self.assertAlmostEqual(out_y[0], 49.98)
        # Z = 100.0 + 0.10 (shift) + 0.01 (drift at t=1.0) = 100.11
        self.assertAlmostEqual(out_z[0], 100.11)
        # Z at t=2.0 (midway) = 100.0 + 0.10 + 0.02 = 100.12
        self.assertAlmostEqual(out_z[1], 100.12)
        # Z at t=3.0 = 100.0 + 0.10 + 0.03 = 100.13
        self.assertAlmostEqual(out_z[2], 100.13)

    def test_strip_adjustment_dialog_init_and_table(self):
        """Test StripAdjustmentDialog initialization and table population."""
        dlg = StripAdjustmentDialog(tile_manager=None, database=None, project_dir=".")
        self.assertEqual(dlg._strip_table.columnCount(), 8)
        self.assertEqual(dlg._results_table.columnCount(), 7)
        self.assertTrue(dlg._step1_chk.isChecked())
        self.assertTrue(dlg._step2_chk.isChecked())
        self.assertFalse(dlg._step3_chk.isChecked())

        # Feed synthetic strips
        dlg._strip_data_map = {
            1: {
                "x": np.array([10.0, 20.0, 30.0]),
                "y": np.array([50.0, 50.0, 50.0]),
                "z": np.array([100.0, 100.0, 100.0]),
                "gps_time": np.array([100.0, 101.0, 102.0]),
            },
            2: {
                "x": np.array([15.0, 25.0, 35.0]),
                "y": np.array([52.0, 52.0, 52.0]),
                "z": np.array([100.1, 100.1, 100.1]),
                "gps_time": np.array([200.0, 201.0, 202.0]),
            },
        }
        dlg._populate_strip_table()

        self.assertEqual(dlg._strip_table.rowCount(), 2)
        self.assertEqual(dlg._strip_table.item(0, 1).text(), "1")
        self.assertEqual(dlg._strip_table.item(0, 2).text(), "Topo (VUX-160)")
        self.assertEqual(dlg._strip_table.item(0, 3).text(), "3")
        self.assertEqual(dlg._strip_table.item(1, 1).text(), "2")
        self.assertEqual(dlg._strip_table.item(1, 2).text(), "Topo (VUX-160)")
        self.assertEqual(dlg._strip_table.item(1, 3).text(), "3")
        self.assertEqual(dlg._ref_strip_combo.count(), 2)

        # Test results handling
        mock_results = {
            "strip_ids": [1, 2],
            "ref_strip_id": 1,
            "corrections": {
                1: StripCorrection(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                2: StripCorrection(2, 0.05, -0.02, 0.08, 0.015, -0.008, 0.002),
            },
            "total_ties": 150,
            "pair_summaries": [],
            "initial_rmse": 0.095,
            "initial_mean_dz": 0.080,
            "final_rmse": 0.018,
            "final_mean_dz": 0.002,
        }

        dlg._on_calculation_finished(mock_results)

        self.assertTrue(dlg._apply_btn.isEnabled())
        self.assertEqual(dlg._results_table.rowCount(), 2)
        # Strip 1 is Ref
        self.assertIn("Ref", dlg._results_table.item(0, 0).text())
        self.assertEqual(dlg._results_table.item(0, 1).text(), "+0.000")
        # Strip 2 has shifts and boresight
        self.assertEqual(dlg._results_table.item(1, 0).text(), "Strip 2")
        self.assertEqual(dlg._results_table.item(1, 1).text(), "+0.050")
        self.assertEqual(dlg._results_table.item(1, 2).text(), "-0.020")
        self.assertEqual(dlg._results_table.item(1, 3).text(), "+0.080")
        self.assertEqual(dlg._results_table.item(1, 4).text(), "+0.0150")

        # Diag label contains improvement stats
        self.assertIn("81.1% improvement", dlg._diag_label.text())

        dlg.close()

    def test_strip_adjustment_worker_tile_native(self):
        """Test _StripAdjustmentWorker streaming across multiple tiles."""
        from unittest.mock import MagicMock
        from lidar_workbench.gui.strip_adjustment_dialog import _StripAdjustmentWorker

        # Create synthetic tile with overlapping strips 1 and 2
        rng = np.random.default_rng(42)
        n = 300
        xs = rng.uniform(0.0, 6.0, n * 2)
        ys = rng.uniform(0.0, 6.0, n * 2)
        zs = np.full(n * 2, 100.0)
        # Strip 2 has +0.06m height offset
        zs[n:] += 0.06
        psids = np.array([1] * n + [2] * n, dtype=np.uint16)
        ts = np.array([100.0 + i * 0.1 for i in range(n * 2)])

        mock_tm = MagicMock()
        mock_tm.load_tile_points_full.return_value = {
            "x": xs, "y": ys, "z": zs, "point_source_id": psids, "gps_time": ts,
        }

        worker = _StripAdjustmentWorker(
            strip_ids=[1, 2],
            trajectories={},
            step1_enabled=True,
            step2_enabled=False,
            step3_enabled=False,
            ref_strip_id=1,
            tile_manager=mock_tm,
            tile_ids=["tile_0001", "tile_0002"],
        )

        results_holder = []
        errors_holder = []
        worker.finished.connect(lambda res: results_holder.append(res))
        worker.error.connect(lambda err: errors_holder.append(err))

        worker.run()

        self.assertEqual(len(errors_holder), 0, f"Worker errored: {errors_holder}")
        self.assertEqual(len(results_holder), 1)
        res = results_holder[0]
        self.assertGreater(res["total_ties"], 0)
        corr2 = res["corrections"][2]
        # Should compensate +0.06m offset
        self.assertAlmostEqual(corr2.dz, -0.06, delta=0.015)
        self.assertLess(res["final_rmse"], res["initial_rmse"])


if __name__ == "__main__":
    unittest.main()
