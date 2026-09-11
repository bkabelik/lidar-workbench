"""
Tests for Ground Control House Roofs and Ground Patches:
  - parse_roof_point_name (6-digit 010101, delimited 01_02_03, etc.)
  - cluster_roof_points_spatially (10m radius connected components)
  - sort_vertices_angular (polar angle CW/CCW sorting to fix zigzag/hourglass polygons)
  - _GroundControlWorker roof & ground plane fitting via RANSAC
  - _Roof2ViewInspector interactive nudging, plotting, and reset
  - GroundControlDialog table editing & live regrouping
"""

import os
import sys
import unittest
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QTableWidgetItem, QMessageBox
from PySide6.QtCore import Qt

from lidar_workbench.gui.ground_control_dialog import (
    parse_roof_point_name,
    cluster_roof_points_spatially,
    sort_vertices_angular,
    _GroundControlWorker,
    _Roof2ViewInspector,
    GroundControlDialog,
)

_app = QApplication.instance() or QApplication(sys.argv)


class TestRoofHelpers(unittest.TestCase):

    def test_parse_roof_point_name_6digit(self):
        # 010101 -> GCP 01, House 01, Point 01
        h, p = parse_roof_point_name("010101")
        self.assertEqual(h, "0101")
        self.assertEqual(p, "01")

        h2, p2 = parse_roof_point_name("020403")
        self.assertEqual(h2, "0204")
        self.assertEqual(p2, "03")

    def test_parse_roof_point_name_delimited_3part(self):
        # 01_02_03 or 1-2-4 or 01.02.03
        h, p = parse_roof_point_name("01_02_03")
        self.assertEqual(h, "01_02")
        self.assertEqual(p, "03")

        h2, p2 = parse_roof_point_name("GCP1-H2-P4")
        self.assertEqual(h2, "GCP1_H2")
        self.assertEqual(p2, "P4")

        h3, p3 = parse_roof_point_name("1.2.3")
        self.assertEqual(h3, "1_2")
        self.assertEqual(p3, "3")

    def test_parse_roof_point_name_delimited_2part(self):
        h, p = parse_roof_point_name("ROOF1_P2")
        self.assertEqual(h, "ROOF1")
        self.assertEqual(p, "P2")

        h2, p2 = parse_roof_point_name("H3-1")
        self.assertEqual(h2, "H3")
        self.assertEqual(p2, "1")

    def test_parse_roof_point_name_fallback(self):
        h, p = parse_roof_point_name("SOLO_POINT")
        self.assertEqual(h, "SOLO_POINT")
        self.assertEqual(p, "1")

    def test_cluster_roof_points_spatially(self):
        # Two distinct patches separated by 40 meters
        patch_a = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])
        patch_b = np.array([[40.0, 40.0], [44.0, 40.0], [44.0, 44.0], [40.0, 44.0]])
        pts = np.vstack([patch_a, patch_b])

        labels = cluster_roof_points_spatially(pts, radius=10.0)
        self.assertEqual(len(labels), 8)
        # First 4 must be in same cluster
        self.assertEqual(len(set(labels[:4])), 1)
        # Last 4 must be in same cluster
        self.assertEqual(len(set(labels[4:])), 1)
        # The two clusters must be different
        self.assertNotEqual(labels[0], labels[4])

    def test_cluster_roof_points_empty_and_single(self):
        empty_lbls = cluster_roof_points_spatially(np.empty((0, 2)), radius=10.0)
        self.assertEqual(len(empty_lbls), 0)

        single_lbls = cluster_roof_points_spatially(np.array([[5.0, 5.0]]), radius=10.0)
        self.assertEqual(list(single_lbls), [0])

    def test_sort_vertices_angular(self):
        # Crossed/zigzag square corners: (0, 10), (10, 0), (10, 10), (0, 0)
        # This zigzag creates an hourglass polygon.
        zigzag = np.array([
            [0.0, 10.0, 5.0],
            [10.0, 0.0, 5.0],
            [10.0, 10.0, 5.0],
            [0.0, 0.0, 5.0],
        ])

        sorted_v, order = sort_vertices_angular(zigzag, clockwise=False)
        # In CCW order around center (5, 5):
        # (0,0) angle -135°, (10,0) angle -45°, (10,10) angle +45°, (0,10) angle +135°
        expected_xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        np.testing.assert_allclose(sorted_v[:, :2], expected_xy, atol=1e-6)

        # Clockwise test: reverse order
        sorted_cw, order_cw = sort_vertices_angular(zigzag, clockwise=True)
        np.testing.assert_allclose(sorted_cw[0, :2], [0.0, 10.0], atol=1e-6)


class TestRoofWorkerRANSAC(unittest.TestCase):

    def test_worker_roof_ransac_fit(self):
        np.random.seed(42)
        # Generate sloped roof plane: z = 50 + 0.2*x - 0.1*y
        xx, yy = np.meshgrid(np.linspace(10, 20, 20), np.linspace(30, 40, 20))
        zz = 50.0 + 0.2 * xx - 0.1 * yy + np.random.normal(0, 0.01, xx.shape)
        cloud_x = xx.ravel()
        cloud_y = yy.ravel()
        cloud_z = zz.ravel()
        cloud_cls = np.full(len(cloud_x), 6, dtype=np.uint8)  # Class 6 = Building

        tile_data = {
            "x": cloud_x,
            "y": cloud_y,
            "z": cloud_z,
            "classification": cloud_cls,
        }

        # Input roof vertices (tilted on the same plane, slightly offset in Z by +0.10 m)
        roof_verts = [
            (12.0, 32.0, 50.0 + 0.2 * 12.0 - 0.1 * 32.0 + 0.10),
            (18.0, 32.0, 50.0 + 0.2 * 18.0 - 0.1 * 32.0 + 0.10),
            (18.0, 38.0, 50.0 + 0.2 * 18.0 - 0.1 * 38.0 + 0.10),
            (12.0, 38.0, 50.0 + 0.2 * 12.0 - 0.1 * 38.0 + 0.10),
        ]

        params = {
            "surfaces": [("ROOF_01", roof_verts, "roof")],
            "radius": 15.0,
        }

        worker = _GroundControlWorker(tile_data, "roofs", params)
        results = []
        worker.finished_roofs.connect(results.extend)
        worker.run()

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["name"], "ROOF_01")
        self.assertEqual(res["type"], "roof")
        self.assertIsNotNone(res["dx"])
        self.assertIsNotNone(res["dy"])
        self.assertIsNotNone(res["dz"])
        self.assertGreater(res["n_nearby"], 50)
        self.assertLess(res["plane_rmse"], 0.05)
        # Since input GCP roof is +0.10m above cloud along z, the shift to add to the
        # point cloud to bring it up to the GCP roof must be positive (~ +0.10m).
        self.assertAlmostEqual(res["dz"], +0.10, delta=0.04)


class TestRoof2ViewInspector(unittest.TestCase):

    def test_inspector_load_patch_and_nudging(self):
        inspector = _Roof2ViewInspector()

        verts = np.array([
            [0.0, 0.0, 10.0],
            [5.0, 0.0, 10.0],
            [5.0, 5.0, 10.0],
            [0.0, 5.0, 10.0],
        ])
        pts = np.array([
            [1.0, 1.0, 10.0],
            [2.0, 2.0, 10.0],
            [3.0, 3.0, 10.0],
            [4.0, 4.0, 10.0],
        ])
        inl = np.array([True, True, True, True])

        patch_dict = {
            "name": "0101",
            "type": "roof",
            "verts": verts,
            "local_pts": pts,
            "inlier_mask": inl,
            "cloud_normal": np.array([0.0, 0.0, 1.0]),
            "cloud_mean": np.array([2.5, 2.5, 10.0]),
            "dx": 0.05,
            "dy": -0.03,
            "dz": 0.08,
            "base_dx": 0.05,
            "base_dy": -0.03,
            "base_dz": 0.08,
            "shift_mag": 0.099,
            "plane_rmse": 0.012,
            "n_nearby": 4,
            "n_verts": 4,
        }

        inspector.load_patch(patch_dict)

        self.assertAlmostEqual(inspector._spin_dx.value(), 0.05, places=3)
        self.assertAlmostEqual(inspector._spin_dy.value(), -0.03, places=3)
        self.assertAlmostEqual(inspector._spin_dz.value(), 0.08, places=3)

        # Track offset_changed signal
        emitted = []
        inspector.offset_changed.connect(lambda name, dx, dy, dz: emitted.append((name, dx, dy, dz)))

        # Nudge +X (default step is 0.05)
        inspector._nudge("x", +1)
        self.assertAlmostEqual(inspector._spin_dx.value(), 0.10, places=3)
        self.assertTrue(len(emitted) > 0)
        self.assertEqual(emitted[-1][0], "0101")
        self.assertAlmostEqual(emitted[-1][1], 0.10, places=3)

        # Nudge -Z
        inspector._nudge("z", -1)
        self.assertAlmostEqual(inspector._spin_dz.value(), 0.03, places=3)
        self.assertAlmostEqual(emitted[-1][3], 0.03, places=3)

        # Reset to auto-fit
        inspector._reset_to_autofit()
        self.assertAlmostEqual(inspector._spin_dx.value(), 0.05, places=3)
        self.assertAlmostEqual(inspector._spin_dy.value(), -0.03, places=3)
        self.assertAlmostEqual(inspector._spin_dz.value(), 0.08, places=3)


class TestDialogTableEditing(unittest.TestCase):

    def test_table_editing_regroups_surfaces(self):
        dlg = GroundControlDialog({})
        # Populate raw records representing 2 houses: 0101 and 0102
        dlg._roof_raw_records = [
            {"name": "010101", "raw_id": "0101", "x": 0.0, "y": 0.0, "z": 10.0},
            {"name": "010102", "raw_id": "0101", "x": 5.0, "y": 0.0, "z": 10.0},
            {"name": "010103", "raw_id": "0101", "x": 5.0, "y": 5.0, "z": 10.0},
            {"name": "010104", "raw_id": "0101", "x": 0.0, "y": 5.0, "z": 10.0},
            {"name": "010201", "raw_id": "0102", "x": 20.0, "y": 20.0, "z": 12.0},
            {"name": "010202", "raw_id": "0102", "x": 25.0, "y": 20.0, "z": 12.0},
            {"name": "010203", "raw_id": "0102", "x": 25.0, "y": 25.0, "z": 12.0},
        ]
        dlg._detect_and_group_roof_points()

        # Both patches 0101 (4 pts) and 0102 (3 pts) should be detected
        self.assertEqual(len(dlg._roof_surfaces), 2)
        surface_ids = [s[0] for s in dlg._roof_surfaces]
        self.assertIn("0101", surface_ids)
        self.assertIn("0102", surface_ids)

        # Operator edits point 4's House No. to move it to 0102
        # (Row 3 is point 010104)
        dlg._roof_points_table.item(3, 1).setText("0102")
        # itemChanged triggers _rebuild_roof_surfaces_from_table
        # Now 0101 has 3 pts, 0102 has 4 pts
        surf_map = {s[0]: s[1] for s in dlg._roof_surfaces}
        self.assertEqual(len(surf_map["0101"]), 3)
        self.assertEqual(len(surf_map["0102"]), 4)

    def test_dismiss_roof_patch_excludes_from_shift(self):
        dlg = GroundControlDialog({})
        results = [
            {
                "name": "ROOF_A",
                "type": "roof",
                "n_verts": 4,
                "n_nearby": 80,
                "dx": 0.10, "dy": 0.20, "dz": 0.30,
                "shift_mag": float(np.sqrt(0.1**2 + 0.2**2 + 0.3**2)),
                "used": True,
            },
            {
                "name": "ROOF_B",
                "type": "roof",
                "n_verts": 4,
                "n_nearby": 90,
                "dx": 0.30, "dy": 0.40, "dz": 0.50,
                "shift_mag": float(np.sqrt(0.3**2 + 0.4**2 + 0.5**2)),
                "used": True,
            },
        ]
        dlg._on_roofs_finished(results)

        # Both are active: mean dx = 0.20, dy = 0.30, dz = 0.40
        self.assertIsNotNone(dlg._roof_shift)
        self.assertAlmostEqual(dlg._roof_shift[0], 0.20, places=3)
        self.assertAlmostEqual(dlg._roof_shift[1], 0.30, places=3)
        self.assertAlmostEqual(dlg._roof_shift[2], 0.40, places=3)
        self.assertTrue(dlg._roof_apply_btn.isEnabled())

        # Dismiss ROOF_A by unchecking column 0 in table
        dlg._roof_table.item(0, 0).setCheckState(Qt.Unchecked)

        self.assertFalse(dlg._roof_results[0]["used"])
        # Now only ROOF_B contributes to shift
        self.assertIsNotNone(dlg._roof_shift)
        self.assertAlmostEqual(dlg._roof_shift[0], 0.30, places=3)
        self.assertAlmostEqual(dlg._roof_shift[1], 0.40, places=3)
        self.assertAlmostEqual(dlg._roof_shift[2], 0.50, places=3)

        # Dismiss ROOF_B too
        dlg._roof_table.item(1, 0).setCheckState(Qt.Unchecked)
        self.assertFalse(dlg._roof_results[1]["used"])
        # All dismissed -> no shift to apply
        self.assertIsNone(dlg._roof_shift)
        self.assertFalse(dlg._roof_apply_btn.isEnabled())

    def test_inspector_use_toggled_syncs_dialog(self):
        dlg = GroundControlDialog({})
        results = [
            {
                "name": "ROOF_01",
                "type": "roof",
                "n_verts": 4,
                "n_nearby": 50,
                "dx": 0.05, "dy": 0.05, "dz": 0.05,
                "shift_mag": 0.086,
                "used": True,
            },
        ]
        dlg._on_roofs_finished(results)
        self.assertTrue(dlg._roof_inspector._use_chk.isChecked())

        # Uncheck "Use in Shift" inside the Inspector
        dlg._roof_inspector._use_chk.setChecked(False)

        # Table checkbox should now be Unchecked
        self.assertEqual(dlg._roof_table.item(0, 0).checkState(), Qt.Unchecked)
        self.assertFalse(dlg._roof_results[0]["used"])
        self.assertIn("[DISMISSED / EXCLUDED]", dlg._roof_inspector._title_label.text())
        self.assertIsNone(dlg._roof_shift)

    def test_delete_roof_patch(self):
        dlg = GroundControlDialog({})
        dlg._roof_raw_records = [
            {"name": "010101", "raw_id": "0101", "x": 0.0, "y": 0.0, "z": 10.0},
            {"name": "010102", "raw_id": "0101", "x": 5.0, "y": 0.0, "z": 10.0},
            {"name": "010103", "raw_id": "0101", "x": 5.0, "y": 5.0, "z": 10.0},
            {"name": "010104", "raw_id": "0101", "x": 0.0, "y": 5.0, "z": 10.0},
            {"name": "010201", "raw_id": "0102", "x": 20.0, "y": 20.0, "z": 12.0},
            {"name": "010202", "raw_id": "0102", "x": 25.0, "y": 20.0, "z": 12.0},
            {"name": "010203", "raw_id": "0102", "x": 25.0, "y": 25.0, "z": 12.0},
        ]
        dlg._detect_and_group_roof_points()
        self.assertEqual(len(dlg._roof_surfaces), 2)

        results = [
            {"name": "0101", "type": "roof", "n_verts": 4, "n_nearby": 30, "dx": 0.1, "dy": 0.1, "dz": 0.1, "shift_mag": 0.17, "used": True},
            {"name": "0102", "type": "roof", "n_verts": 3, "n_nearby": 40, "dx": 0.2, "dy": 0.2, "dz": 0.2, "shift_mag": 0.34, "used": True},
        ]
        dlg._on_roofs_finished(results)
        self.assertEqual(dlg._roof_table.rowCount(), 2)

        # Delete patch 0101
        dlg.delete_roof_patch("0101", confirm=False)

        # 0101 should be completely gone from points, surfaces, table, and results
        self.assertEqual(dlg._roof_points_table.rowCount(), 3)
        self.assertEqual(len(dlg._roof_surfaces), 1)
        self.assertEqual(dlg._roof_surfaces[0][0], "0102")
        self.assertEqual(dlg._roof_table.rowCount(), 1)
        self.assertEqual(dlg._roof_table.item(0, 1).text(), "0102")
        self.assertEqual(len(dlg._roof_results), 1)
        self.assertEqual(dlg._roof_results[0]["name"], "0102")

    def test_delete_roof_points(self):
        dlg = GroundControlDialog({})
        dlg._roof_raw_records = [
            {"name": "010101", "raw_id": "0101", "x": 0.0, "y": 0.0, "z": 10.0},
            {"name": "010102", "raw_id": "0101", "x": 5.0, "y": 0.0, "z": 10.0},
            {"name": "010103", "raw_id": "0101", "x": 5.0, "y": 5.0, "z": 10.0},
            {"name": "010104", "raw_id": "0101", "x": 0.0, "y": 5.0, "z": 10.0},
            {"name": "010201", "raw_id": "0102", "x": 20.0, "y": 20.0, "z": 12.0},
            {"name": "010202", "raw_id": "0102", "x": 25.0, "y": 20.0, "z": 12.0},
            {"name": "010203", "raw_id": "0102", "x": 25.0, "y": 25.0, "z": 12.0},
        ]
        dlg._detect_and_group_roof_points()
        self.assertEqual(len(dlg._roof_surfaces), 2)
        self.assertEqual(dlg._roof_points_table.rowCount(), 7)

        # Select row 0 (point 010101)
        dlg._roof_points_table.selectRow(0)
        dlg._on_delete_roof_points(confirm=False)

        # 0101 now has 3 vertices left (still a valid surface)
        self.assertEqual(dlg._roof_points_table.rowCount(), 6)
        surf_map = {s[0]: s[1] for s in dlg._roof_surfaces}
        self.assertEqual(len(surf_map["0101"]), 3)

        # Delete another point from 0101 (row 0 is now 010102)
        dlg._roof_points_table.selectRow(0)
        dlg._on_delete_roof_points(confirm=False)

        # 0101 now has 2 vertices, so it drops out of _roof_surfaces!
        self.assertEqual(dlg._roof_points_table.rowCount(), 5)
        surf_ids = [s[0] for s in dlg._roof_surfaces]
        self.assertNotIn("0101", surf_ids)
        self.assertIn("0102", surf_ids)

    def test_roof_shift_sign_cloud_higher_than_roof(self):
        """When the point cloud is higher (+0.07m) than the GCP roof, the shift

        applied to the point cloud MUST be negative (-0.07m) to lower it.
        """
        xx, yy = np.meshgrid(np.linspace(10, 20, 10), np.linspace(30, 40, 10))
        # Flat roof in LiDAR point cloud at Z = 100.07
        cloud_z = np.full(xx.shape, 100.07)
        cloud_cls = np.full(xx.size, 6, dtype=np.uint8)

        tile_data = {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": cloud_z.ravel(),
            "classification": cloud_cls,
        }

        # True surveyor GCP house roof vertices at Z = 100.00
        roof_verts = [
            (12.0, 32.0, 100.00),
            (18.0, 32.0, 100.00),
            (18.0, 38.0, 100.00),
            (12.0, 38.0, 100.00),
        ]

        worker = _GroundControlWorker(tile_data, "roofs", {
            "surfaces": [("ROOF_TEST", roof_verts, "roof")],
            "radius": 15.0,
        })
        results = []
        worker.finished_roofs.connect(results.extend)
        worker.run()

        self.assertEqual(len(results), 1)
        res = results[0]
        # Shift must be negative -0.07m to lower the point cloud to the GCP roof
        self.assertAlmostEqual(res["dz"], -0.07, delta=0.005)
        # Applying shift to point cloud brings it to 100.00 (the GCP elevation)
        shifted_cloud_z = tile_data["z"] + res["dz"]
        np.testing.assert_allclose(shifted_cloud_z, 100.00, atol=0.01)

    def test_tile_selection_dialog(self):
        """Test _TileSelectionDialog selection, filtering, and bulk operations."""
        from lidar_workbench.gui.ground_control_dialog import _TileSelectionDialog

        all_tiles = [f"tile_{i:02d}" for i in range(10)]
        init_sel = ["tile_01", "tile_03"]
        dlg = _TileSelectionDialog(all_tiles, init_sel)

        self.assertEqual(dlg.get_selected_tile_ids(), ["tile_01", "tile_03"])

        # Select all
        dlg._select_all()
        self.assertEqual(len(dlg.get_selected_tile_ids()), 10)

        # Deselect all
        dlg._deselect_all()
        self.assertEqual(len(dlg.get_selected_tile_ids()), 0)

        # Filter
        dlg._apply_filter("tile_05")
        dlg._select_all()  # only visible tile_05 selected
        self.assertEqual(dlg.get_selected_tile_ids(), ["tile_05"])

    def test_ground_control_multi_tile_scope(self):
        """Test GroundControlDialog multi-tile target scope resolution and button text."""
        all_tiles = ["tile_01", "tile_02", "tile_03", "tile_04"]
        sel_tiles = ["tile_02", "tile_03"]
        cur_tile = "tile_01"

        class MockTM:
            def tile_ids(self):
                return all_tiles

        dlg = GroundControlDialog(
            {}, tile_ids=sel_tiles, current_tile_id=cur_tile, tile_manager=MockTM()
        )

        self.assertEqual(dlg._all_tile_ids, all_tiles)
        self.assertEqual(dlg._selected_tile_ids, sel_tiles)
        self.assertEqual(dlg._current_tile_id, cur_tile)

        # Combo options should include all, selected, current, custom
        combo_data = [dlg._roof_scope_combo.itemData(i) for i in range(dlg._roof_scope_combo.count())]
        self.assertIn("all", combo_data)
        self.assertIn("selected", combo_data)
        self.assertIn("current", combo_data)
        self.assertIn("custom", combo_data)

        # Check 'all' scope
        dlg._roof_scope_combo.setCurrentIndex(dlg._roof_scope_combo.findData("all"))
        self.assertEqual(dlg.get_target_tile_ids(), all_tiles)

        # Check 'selected' scope
        dlg._roof_scope_combo.setCurrentIndex(dlg._roof_scope_combo.findData("selected"))
        self.assertEqual(dlg.get_target_tile_ids(), sel_tiles)

        # Check 'current' scope
        dlg._roof_scope_combo.setCurrentIndex(dlg._roof_scope_combo.findData("current"))
        self.assertEqual(dlg.get_target_tile_ids(), [cur_tile])

        # Check 'custom' scope
        dlg._custom_tile_ids = ["tile_04"]
        dlg._roof_scope_combo.setCurrentIndex(dlg._roof_scope_combo.findData("custom"))
        self.assertEqual(dlg.get_target_tile_ids(), ["tile_04"])

        # Check synchronization between GCP and Roof scope combos
        self.assertEqual(dlg._gcp_scope_combo.currentData(), "custom")

        # Test Apply button text dynamic update
        dlg._roof_shift = (0.01, -0.02, -0.07)
        dlg._update_apply_buttons_text()
        self.assertIn("Custom Tiles (1)", dlg._roof_apply_btn.text())
        self.assertIn("+0.010", dlg._roof_apply_btn.text())
        self.assertIn("-0.070", dlg._roof_apply_btn.text())

        # Switch back to 'all'
        dlg._roof_scope_combo.setCurrentIndex(dlg._roof_scope_combo.findData("all"))
        self.assertIn("All Project Tiles (4)", dlg._roof_apply_btn.text())

    def test_apply_shift_signal_emits_target_tile_ids(self):
        """Test that shift_applied emits (dx, dy, dz, target_tile_ids)."""
        from unittest.mock import patch
        all_tiles = ["tile_01", "tile_02", "tile_03"]

        class MockTM:
            def tile_ids(self):
                return all_tiles

        dlg = GroundControlDialog(
            {}, tile_ids=all_tiles, current_tile_id="tile_01", tile_manager=MockTM()
        )
        dlg._roof_shift = (0.05, -0.03, 0.08)

        emitted_args = []
        dlg.shift_applied.connect(lambda *args: emitted_args.append(args))

        # Accept confirmation dialog
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            dlg._on_apply_roofs()

        self.assertEqual(len(emitted_args), 1)
        dx, dy, dz, tids = emitted_args[0]
        self.assertAlmostEqual(dx, 0.05)
        self.assertAlmostEqual(dy, -0.03)
        self.assertAlmostEqual(dz, 0.08)
        self.assertEqual(tids, all_tiles)

    def test_main_window_apply_ground_control_shift_multi_tile(self):
        """Test _apply_ground_control_shift across multiple tiles."""
        from unittest.mock import MagicMock
        from lidar_workbench.gui.main_window import MainWindow

        mock_win = MagicMock()
        mock_win._editor.tile_id = "tile_01"
        tile_data_store = {
            "tile_01": {"x": np.array([10.0]), "y": np.array([20.0]), "z": np.array([30.0])},
            "tile_02": {"x": np.array([40.0]), "y": np.array([50.0]), "z": np.array([60.0])},
        }

        def mock_load(tid):
            return tile_data_store.get(tid)

        written_tiles = {}
        def mock_write(tid, data):
            written_tiles[tid] = data
            return True

        mock_win._tm.load_tile_points_full.side_effect = mock_load
        mock_win._write_tile_data_to_las.side_effect = mock_write

        # Call unbound method with mock_win
        MainWindow._apply_ground_control_shift(
            mock_win, dx=0.1, dy=-0.2, dz=0.3, tile_ids=["tile_01", "tile_02"]
        )

        # Verify both tiles shifted
        self.assertIn("tile_01", written_tiles)
        self.assertIn("tile_02", written_tiles)
        np.testing.assert_allclose(written_tiles["tile_01"]["x"], 10.1)
        np.testing.assert_allclose(written_tiles["tile_01"]["y"], 19.8)
        np.testing.assert_allclose(written_tiles["tile_01"]["z"], 30.3)
        np.testing.assert_allclose(written_tiles["tile_02"]["x"], 40.1)
        np.testing.assert_allclose(written_tiles["tile_02"]["y"], 49.8)
        np.testing.assert_allclose(written_tiles["tile_02"]["z"], 60.3)

        # Verify tile_01 (active tile) was reloaded in editor
        mock_win._editor.open_tile.assert_called_with("tile_01")
        mock_win._multi_load_for_edit.assert_called_with(written_tiles["tile_01"])
        mock_win._mark_project_dirty.assert_called_once()
        mock_win._regenerate_dtm.assert_called_once()



if __name__ == "__main__":
    unittest.main()


