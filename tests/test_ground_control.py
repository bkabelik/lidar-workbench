"""Unit tests for Ground Control Points (GCP) parsing and elevation validation."""

import sys
from pathlib import Path
import unittest
import numpy as np
import tempfile
import os

# Ensure parent of lidar_workbench is on sys.path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from PySide6.QtWidgets import QApplication
# Ensure a QApplication exists for widget testing
app = QApplication.instance() or QApplication(sys.argv)

from lidar_workbench.gui.ground_control_dialog import _safe_float, _GroundControlWorker, GroundControlDialog


class TestGroundControl(unittest.TestCase):
    def test_safe_float(self):
        # Standard decimal point
        self.assertEqual(_safe_float("123.456"), 123.456)
        self.assertEqual(_safe_float(123.456), 123.456)
        self.assertEqual(_safe_float(100), 100.0)

        # European decimal comma
        self.assertEqual(_safe_float("123,456"), 123.456)
        self.assertEqual(_safe_float(" 532456,789 "), 532456.789)
        self.assertEqual(_safe_float("-12,5"), -12.5)

        # Negative and zero
        self.assertEqual(_safe_float("0,0"), 0.0)
        self.assertEqual(_safe_float("-0,25"), -0.25)

        # Invalid
        with self.assertRaises(ValueError):
            _safe_float("abc")
        with self.assertRaises(ValueError):
            _safe_float("")

    def test_gcp_elevation_comparison(self):
        # Create synthetic ground points around (100, 200) with true elevation Z=50.0
        n_pts = 100
        rng = np.random.default_rng(42)
        xs = 100.0 + rng.uniform(-3.0, 3.0, n_pts)
        ys = 200.0 + rng.uniform(-3.0, 3.0, n_pts)
        zs = 50.0 + rng.normal(0.0, 0.02, n_pts)
        nearby_xyz = np.column_stack((xs, ys, zs))

        # Test GCP with input elevation Z_in = 50.15 (GCP is 0.15m higher than point cloud)
        gx, gy, gz = 100.0, 200.0, 50.15
        z_cloud, dz, z_std = _GroundControlWorker._compare_gcp_to_surface(gx, gy, gz, nearby_xyz)

        self.assertAlmostEqual(z_cloud, 50.0, delta=0.02)
        self.assertAlmostEqual(dz, 0.15, delta=0.02)
        self.assertLess(z_std, 0.05)

    def test_gcp_elevation_sloping_terrain(self):
        # Synthetic ground points on a steep 25% slope over a 5m radius
        # Slope equation: Z = 100.0 + 0.20*(x - 500.0) - 0.15*(y - 600.0)
        rng = np.random.default_rng(99)
        n_pts = 300
        xs = 500.0 + rng.uniform(-5.0, 5.0, n_pts)
        ys = 600.0 + rng.uniform(-5.0, 5.0, n_pts)
        zs = 100.0 + 0.20 * (xs - 500.0) - 0.15 * (ys - 600.0) + rng.normal(0, 0.02, n_pts)
        nearby_xyz = np.column_stack((xs, ys, zs))

        # Check GCP at center (500.0, 600.0) with true ground elevation = 100.0
        # Survey GCP input has dz = -0.06m (100.0 - 0.06 = 99.94)
        gx, gy, gz = 500.0, 600.0, 99.94
        z_cloud, dz, z_std = _GroundControlWorker._compare_gcp_to_surface(gx, gy, gz, nearby_xyz)

        # Must recover ground elevation accurately without slope-induced bias
        self.assertAlmostEqual(z_cloud, 100.0, delta=0.02)
        self.assertAlmostEqual(dz, -0.06, delta=0.02)
        self.assertLess(z_std, 0.05)

    def test_gcp_worker_full_run(self):
        # Generate point cloud tile data
        rng = np.random.default_rng(123)
        n = 500
        xs = rng.uniform(500000.0, 500500.0, n)
        ys = rng.uniform(5300000.0, 5300500.0, n)
        zs = np.full(n, 250.0) + rng.normal(0, 0.01, n)
        cls = np.full(n, 2, dtype=np.int32)  # Ground

        tile_data = {
            "x": xs,
            "y": ys,
            "z": zs,
            "classification": cls,
        }

        # GCP at (500250.0, 5300250.0, 250.50) -> dz should be ~ +0.50m
        # GCP outside bounds -> warning "No points within..."
        params = {
            "points": [
                ("GCP_IN", 500250.0, 5300250.0, 250.50),
                ("GCP_OUT", 600000.0, 6000000.0, 250.50),
            ],
            "classes": {2},
            "radius": 50.0,
        }

        worker = _GroundControlWorker(tile_data, "gcp", params)
        results = []
        worker.finished_gcp.connect(lambda res: results.extend(res))
        worker._run_gcp()

        self.assertEqual(len(results), 2)
        r_in = results[0]
        self.assertEqual(r_in["name"], "GCP_IN")
        self.assertIsNotNone(r_in["dz"])
        self.assertAlmostEqual(r_in["dz"], 0.50, delta=0.05)

        r_out = results[1]
        self.assertEqual(r_out["name"], "GCP_OUT")
        self.assertIsNone(r_out["dz"])
        self.assertIn("No points within", r_out["warning"])

    def test_german_survey_csv_and_swap_xy(self):
        # Test German / Austrian surveyor CSV with commas:
        # Pkt;Rechtswert;Hochwert;Hoehe
        content = (
            "Pkt;Rechtswert;Hochwert;Hoehe\n"
            "101;532456,12;5342123,45;345,67\n"
            "102;532500,00;5342200,00;346,10\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            dlg = GroundControlDialog(tile_data={})
            dlg._gcp_sep_combo.setCurrentText("Semicolon (;)")
            dlg._parse_gcp_csv(tmp_path)

            # Check that German columns were auto-detected
            self.assertEqual(dlg._gcp_name_col.currentText(), "Pkt")
            self.assertEqual(dlg._gcp_x_col.currentText(), "Rechtswert")
            self.assertEqual(dlg._gcp_y_col.currentText(), "Hochwert")
            self.assertEqual(dlg._gcp_z_col.currentText(), "Hoehe")

            # Check parsed points with decimal comma conversion
            self.assertEqual(len(dlg._gcp_points), 2)
            p1 = dlg._gcp_points[0]
            self.assertEqual(p1[0], "101")
            self.assertAlmostEqual(p1[1], 532456.12)
            self.assertAlmostEqual(p1[2], 5342123.45)
            self.assertAlmostEqual(p1[3], 345.67)

            # Test Swap X & Y
            dlg._on_gcp_swap_xy()
            self.assertEqual(dlg._gcp_x_col.currentText(), "Hochwert")
            self.assertEqual(dlg._gcp_y_col.currentText(), "Rechtswert")

            p1_swapped = dlg._gcp_points[0]
            self.assertAlmostEqual(p1_swapped[1], 5342123.45)  # now X
            self.assertAlmostEqual(p1_swapped[2], 532456.12)   # now Y

            dlg.close()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

    def test_gcp_table_selective_use_and_sorting(self):
        from PySide6.QtCore import Qt
        from lidar_workbench.gui.ground_control_dialog import _NumericTableWidgetItem

        # Test _NumericTableWidgetItem comparison directly
        it_neg = _NumericTableWidgetItem(-0.50)
        it_zero = _NumericTableWidgetItem(0.0)
        it_pos = _NumericTableWidgetItem(1.25)
        it_none = _NumericTableWidgetItem(None)

        self.assertTrue(it_neg < it_zero)
        self.assertTrue(it_zero < it_pos)
        self.assertTrue(it_pos < it_none)  # None sorts last
        self.assertFalse(it_none < it_neg)

        # Create dialog and populate with mock results including an outlier
        dlg = GroundControlDialog(tile_data={})
        results = [
            {"name": "P1", "x": 10.0, "y": 20.0, "z_in": 100.10, "z_cloud": 100.0, "dz": 0.10, "n_nearby": 15},
            {"name": "P2", "x": 11.0, "y": 21.0, "z_in": 100.12, "z_cloud": 100.0, "dz": 0.12, "n_nearby": 18},
            {"name": "P3", "x": 12.0, "y": 22.0, "z_in": 100.08, "z_cloud": 100.0, "dz": 0.08, "n_nearby": 12},
            {"name": "P4_outlier", "x": 13.0, "y": 23.0, "z_in": 105.50, "z_cloud": 100.0, "dz": 5.50, "n_nearby": 5},
            {"name": "P5_missing", "x": 14.0, "y": 24.0, "z_in": 100.0, "z_cloud": None, "dz": None, "n_nearby": 0, "warning": "No points"},
        ]

        dlg._on_gcp_finished(results)
        self.assertEqual(dlg._gcp_table.rowCount(), 5)

        # 4 valid points active, median = median(0.08, 0.10, 0.12, 5.50) = 0.110
        self.assertAlmostEqual(dlg._gcp_shift, 0.110, places=3)

        # Test disabling outliers with threshold 0.50m
        dlg._gcp_outlier_spin.setValue(0.50)
        dlg._on_gcp_disable_outliers()

        # P4_outlier has dz=5.50 > 0.50 -> should be disabled
        # Active points are now P1 (0.10), P2 (0.12), P3 (0.08) -> median = 0.100
        self.assertAlmostEqual(dlg._gcp_shift, 0.100, places=3)

        # Verify sorting by ΔZ column (column 6) ascending
        dlg._gcp_table.sortItems(6, Qt.AscendingOrder)
        # Row 0 should be P3 (0.08)
        self.assertEqual(dlg._gcp_table.item(0, 1).text(), "P3")
        # Row 1 should be P1 (0.10)
        self.assertEqual(dlg._gcp_table.item(1, 1).text(), "P1")
        # Row 2 should be P2 (0.12)
        self.assertEqual(dlg._gcp_table.item(2, 1).text(), "P2")
        # Row 3 should be P4_outlier (5.50)
        self.assertEqual(dlg._gcp_table.item(3, 1).text(), "P4_outlier")
        # Row 4 should be P5_missing (None / warning)
        self.assertEqual(dlg._gcp_table.item(4, 1).text(), "P5_missing")

        # Manually uncheck P2 (row 2)
        dlg._gcp_table.item(2, 0).setCheckState(Qt.Unchecked)
        # Active are now P1 (0.10) and P3 (0.08) -> median = 0.090
        self.assertAlmostEqual(dlg._gcp_shift, 0.090, places=3)

        # Test visual check sync: inspect P4_outlier (index 3 in results)
        dlg._vis_index = 3
        dlg._update_vis_nav()
        self.assertFalse(dlg._vis_use_chk.isChecked())

        # Re-enable via visual check checkbox
        dlg._vis_use_chk.setChecked(True)
        # Now active are P1 (0.10), P3 (0.08), P4 (5.50) -> median = 0.100
        self.assertAlmostEqual(dlg._gcp_shift, 0.100, places=3)

        # Test Deselect All
        dlg._on_gcp_deselect_all()
        self.assertIsNone(dlg._gcp_shift)
        self.assertFalse(dlg._gcp_apply_btn.isEnabled())

        # Test Select All
        dlg._on_gcp_select_all()
        # All 4 valid points selected again -> median = 0.110
        self.assertAlmostEqual(dlg._gcp_shift, 0.110, places=3)

        # Test _on_vis_goto signal emission
        emitted = []
        dlg.visualize_point.connect(lambda x, y, z, cz, lbl: emitted.append((x, y, z, cz, lbl)))
        dlg._vis_index = 0
        dlg._on_vis_goto()
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][4], "P1")
        self.assertAlmostEqual(emitted[0][2], 100.10)   # Z_in
        self.assertAlmostEqual(emitted[0][3], 100.00)   # Z_cloud

        dlg.close()
