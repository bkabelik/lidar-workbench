"""
Unit tests verifying GUI stability against crashes, race conditions,
selection retention, and safe painter handling.
"""

import sys
import unittest
import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from lidar_workbench.config import TileStatus
from lidar_workbench.gui.tile_list_widget import TileListWidget
from lidar_workbench.gui.view_dtm import ViewDTM


class TestTileListStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_selection_preserved_during_update_and_refresh(self):
        """Verify that tile selection is preserved across update_tile_status and set_tiles."""
        w = TileListWidget()
        tiles = [
            {"id": "tile_0001", "point_count": 1000, "status": TileStatus.IMPORTED},
            {"id": "tile_0324", "point_count": 14643835, "status": TileStatus.IMPORTED},
            {"id": "tile_0325", "point_count": 500000, "status": TileStatus.IMPORTED},
        ]
        w.set_tiles(tiles)
        self.assertEqual(w.get_selected_tile_ids(), [])

        # User selects tile_0324
        ok = w.select_tile("tile_0324")
        self.assertTrue(ok)
        self.assertEqual(w.get_selected_tile_ids(), ["tile_0324"])

        # Background filter completes on tile_0324
        w.update_tile_status("tile_0324", TileStatus.FILTERED)
        self.assertEqual(w.get_selected_tile_ids(), ["tile_0324"])

        # Background filter batch finishes and triggers full refresh
        tiles[1]["status"] = TileStatus.FILTERED
        w.set_tiles(tiles)
        self.assertEqual(w.get_selected_tile_ids(), ["tile_0324"])

    def test_multi_selection_preserved(self):
        """Verify multiple selected tiles are preserved across set_tiles."""
        w = TileListWidget()
        tiles = [
            {"id": "tile_A", "point_count": 100, "status": TileStatus.IMPORTED},
            {"id": "tile_B", "point_count": 200, "status": TileStatus.IMPORTED},
            {"id": "tile_C", "point_count": 300, "status": TileStatus.IMPORTED},
        ]
        w.set_tiles(tiles)
        count = w.select_tiles(["tile_A", "tile_C"])
        self.assertEqual(count, 2)
        self.assertEqual(set(w.get_selected_tile_ids()), {"tile_A", "tile_C"})

        # Re-set tiles
        w.set_tiles(tiles)
        self.assertEqual(set(w.get_selected_tile_ids()), {"tile_A", "tile_C"})

    def test_unknown_status_group_created_dynamically(self):
        """Verify update_tile_status handles uninitialized status groups without crashing."""
        w = TileListWidget()
        tiles = [
            {"id": "tile_0001", "point_count": 1000, "status": TileStatus.IMPORTED},
        ]
        w.set_tiles(tiles)
        w.select_tile("tile_0001")

        # Move to BATHY status (which currently has 0 tiles)
        w.update_tile_status("tile_0001", TileStatus.BATHY)
        self.assertEqual(w.get_selected_tile_ids(), ["tile_0001"])


class TestViewDTMStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_generate_dtm_skips_without_ground(self):
        """Verify generate_dtm returns early without attempting grid interpolation when no ground exists."""
        v = ViewDTM()
        n = 1000
        xs = np.linspace(0, 100, n)
        ys = np.linspace(0, 100, n)
        zs = np.sin(xs)
        # Class 1 (unclassified) - no ground points (class 2)
        cls_arr = np.ones(n, dtype=np.uint8)

        v.load_points({"x": xs, "y": ys, "z": zs, "classification": cls_arr})
        # Call generate_dtm - should safely skip
        v.generate_dtm(ground_class=2)
        self.assertIsNone(v._dtm_grid_z)


if __name__ == "__main__":
    unittest.main()

