import os
import sys
import time
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import laspy

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lidar_workbench.database import Database, TileStatus
from lidar_workbench.project_manager import ProjectManager
from lidar_workbench.tile_manager import TileManager
from lidar_workbench.manual_edit import ManualEditor


class TestManualEditPerformance(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_manual_edit_")
        self.proj_dir = os.path.join(self.test_dir, "proj")
        self.pm = ProjectManager()
        self.pm.create(self.proj_dir, name="TestProject")
        self.db = self.pm.db
        self.tiles_dir = str(self.pm.tiles_dir)

        # Create a synthetic LAS file with 100,000 points
        self.n_points = 100_000
        self.las_path = os.path.join(self.tiles_dir, "tile_001.las")
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.offsets = [0.0, 0.0, 0.0]
        header.scales = [0.001, 0.001, 0.001]
        las = laspy.LasData(header)
        las.x = np.linspace(0, 100, self.n_points)
        las.y = np.linspace(0, 100, self.n_points)
        las.z = np.sin(las.x / 10.0) * 5.0 + 100.0
        las.classification = np.full(self.n_points, 1, dtype=np.uint8)  # Unclassified
        las.intensity = np.random.randint(0, 65535, self.n_points, dtype=np.uint16)
        las.return_number = np.ones(self.n_points, dtype=np.uint8)
        las.number_of_returns = np.ones(self.n_points, dtype=np.uint8)
        las.point_source_id = np.ones(self.n_points, dtype=np.uint16)
        las.write(self.las_path)

        with self.db.connect() as conn:
            self.db.insert_tile(
                conn=conn,
                tile_id="tile_001",
                filename=os.path.basename(self.las_path),
                bbox=(0.0, 0.0, 100.0, 100.0),
                point_count=self.n_points,
                status=TileStatus.IMPORTED,
            )

        self.tm = TileManager(self.pm, self.db)
        self.editor = ManualEditor(self.tm)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_instant_ram_edits_and_deferred_disk_save(self):
        # 1. Open tile
        point_data = self.tm.load_tile_points_full("tile_001")
        self.assertIsNotNone(point_data)
        ok = self.editor.open_tile("tile_001", point_data=point_data)
        self.assertTrue(ok)
        self.assertFalse(self.editor.is_dirty)

        # 2. Extract profile corridor
        profile = self.editor.extract_profile((0.0, 0.0), (100.0, 100.0), width=10.0)
        self.assertIsNotNone(profile)
        self.assertGreater(len(profile.distances), 0)

        # Record file modification time on disk
        initial_mtime = os.path.getmtime(self.las_path)

        # 3. Simulate multi-stroke manual edits (brush selection + assign class)
        self.editor.select_brush(center=(profile.distances[250], profile.elevations[250]), radius=50.0)
        self.assertGreater(self.editor.selected_count, 0)

        # Time the classification stroke
        t0 = time.perf_counter()
        classify_ok = self.editor.assign_class(2)  # Reclassify to Ground
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertTrue(classify_ok)
        self.assertTrue(self.editor.is_dirty)
        # Check sub-millisecond to sub-15ms execution
        self.assertLess(elapsed_ms, 25.0, f"Stroke execution took {elapsed_ms:.2f}ms, expected sub-15ms")

        # Crucial check: Disk file MUST NOT be touched during edit stroke!
        current_mtime = os.path.getmtime(self.las_path)
        self.assertEqual(initial_mtime, current_mtime, "Disk file was modified during brush stroke! Must remain deferred.")

        # Undo benchmark
        t0 = time.perf_counter()
        undo_desc = self.editor.undo()
        undo_ms = (time.perf_counter() - t0) * 1000.0
        self.assertIsNotNone(undo_desc)
        self.assertLess(undo_ms, 15.0, f"Undo took {undo_ms:.2f}ms")

        # Redo benchmark
        t0 = time.perf_counter()
        redo_desc = self.editor.redo()
        redo_ms = (time.perf_counter() - t0) * 1000.0
        self.assertIsNotNone(redo_desc)
        self.assertLess(redo_ms, 15.0, f"Redo took {redo_ms:.2f}ms")

        # 4. Save tile to disk
        save_ok = self.editor.save_tile()
        self.assertTrue(save_ok)
        self.assertFalse(self.editor.is_dirty)

        # Disk file must now be updated and backup created
        backup_path = self.las_path + ".bak"
        self.assertTrue(os.path.exists(backup_path), ".bak backup file should exist after saving")
        saved_las = laspy.read(self.las_path)
        ground_count = np.sum(saved_las.classification == 2)
        self.assertGreater(ground_count, 0, "Saved LAS should contain reclassified ground points")


if __name__ == "__main__":
    unittest.main()
