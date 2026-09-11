"""
test_point_qc.py - Unit tests for LiDAR Point QC and Map Viewer.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import rasterio

from lidar_workbench.point_qc import (
    batch_export_all_strip_differences,
    generate_density_raster,
    generate_max_strip_difference_raster,
    generate_spacing_raster,
    generate_strip_difference_raster,
    generate_tile_grid_vector,
)
from lidar_workbench.gui.point_qc_layers import (
    LayerGroup,
    RasterLayer,
    VectorFeature,
    VectorLayer,
    load_layer_workspace,
    save_layer_workspace,
)


class MockDatabase:
    def __init__(self, tiles, db_path="/tmp/mock_project/project.db"):
        self._tiles = tiles
        self.db_path = db_path

    def get_all_tiles(self):
        return self._tiles

    def get_tiles_in_bbox(self, min_x, min_y, max_x, max_y):
        hits = []
        for t in self._tiles:
            if not (
                t["max_x"] < min_x
                or t["min_x"] > max_x
                or t["max_y"] < min_y
                or t["min_y"] > max_y
            ):
                hits.append(t)
        return hits


class MockProjectManager:
    def __init__(self, root):
        self.project_root = Path(root)


class MockTileManager:
    def __init__(self, data_map, root):
        self._data_map = data_map
        self._pm = MockProjectManager(root)

    def load_tile_points_full(self, tile_id):
        return self._data_map.get(tile_id)


class TestPointQC(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)

        # Create two 100x100m synthetic tiles
        # Tile 1: (500000, 6000000) to (500100, 6000100)
        # Tile 2: (500100, 6000000) to (500200, 6000100)
        np.random.seed(42)
        n1 = 500
        x1 = np.random.uniform(500000, 500100, n1)
        y1 = np.random.uniform(6000000, 6000100, n1)
        z1 = np.full(n1, 100.0) + np.random.normal(0, 0.05, n1)
        fl1 = np.random.choice([1, 2], n1)

        n2 = 300
        x2 = np.random.uniform(500100, 500200, n2)
        y2 = np.random.uniform(6000000, 6000100, n2)
        z2 = np.full(n2, 102.0) + np.random.normal(0, 0.05, n2)
        fl2 = np.full(n2, 2)

        self.mock_tiles = [
            {
                "id": "tile_01",
                "filename": "tile_01.las",
                "min_x": 500000.0,
                "max_x": 500100.0,
                "min_y": 6000000.0,
                "max_y": 6000100.0,
                "point_count": n1,
                "status": "raw",
                "crs_epsg": 25833,
            },
            {
                "id": "tile_02",
                "filename": "tile_02.las",
                "min_x": 500100.0,
                "max_x": 500200.0,
                "min_y": 6000000.0,
                "max_y": 6000100.0,
                "point_count": n2,
                "status": "raw",
                "crs_epsg": 25833,
            },
        ]

        self.data_map = {
            "tile_01": {
                "x": x1,
                "y": y1,
                "z": z1,
                "point_source_id": fl1,
                "classification": np.ones(n1, dtype=np.uint8),
                "return_number": np.ones(n1, dtype=np.uint8),
            },
            "tile_02": {
                "x": x2,
                "y": y2,
                "z": z2,
                "point_source_id": fl2,
                "classification": np.ones(n2, dtype=np.uint8),
                "return_number": np.ones(n2, dtype=np.uint8),
            },
        }

        self.db = MockDatabase(self.mock_tiles, db_path=str(self.root / "project.db"))
        self.tm = MockTileManager(self.data_map, root=self.root)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_density_raster_generation(self):
        """Test generation of point density GeoTIFF."""
        out_tif = self.root / "test_density.tif"
        stats = generate_density_raster(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_path=out_tif,
        )

        self.assertTrue(out_tif.exists())
        self.assertGreater(stats["active_cells"], 0)
        self.assertGreater(stats["mean_density"], 0.0)

        # Inspect with rasterio
        with rasterio.open(out_tif) as src:
            self.assertEqual(src.crs.to_epsg(), 25833)
            self.assertEqual(src.nodata, -9999.0)
            data = src.read(1)
            self.assertEqual(data.shape, (20, 40))

    def test_spacing_raster_generation(self):
        """Test derivation of nominal point spacing from density."""
        dens_tif = self.root / "test_density.tif"
        generate_density_raster(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_path=dens_tif,
        )

        sp_tif = self.root / "test_spacing.tif"
        sp_stats = generate_spacing_raster(dens_tif, sp_tif)
        self.assertTrue(sp_tif.exists())
        self.assertGreater(sp_stats["active_cells"], 0)
        self.assertGreater(sp_stats["mean_spacing"], 0.0)

    def test_strip_difference_raster(self):
        """Test calculation of strip height differences (dZ)."""
        out_tif = self.root / "strip_diff.tif"
        # Tile 1 has both strip 1 and strip 2 points
        stats = generate_strip_difference_raster(
            tile_manager=self.tm,
            database=self.db,
            strip_a=1,
            strip_b=2,
            cell_size=5.0,
            output_path=out_tif,
        )

        self.assertTrue(out_tif.exists())
        self.assertIn("rmse_dz", stats)
        with rasterio.open(out_tif) as src:
            diff_data = src.read(1)
            valid = diff_data != src.nodata
            if np.any(valid):
                # Diff should be around 0 +- 0.2m because z1 has same base
                self.assertLess(abs(stats["mean_dz"]), 0.2)

    def test_tile_grid_vector_generation(self):
        """Test creation of GeoJSON tile grid with uniform square sizing and noise filtering."""
        # Add a noise tile to database to ensure it gets excluded
        noise_tile = {
            "id": "tile_01_noise",
            "filename": "tile_01_noise.las",
            "min_x": 500010.0,
            "max_x": 500030.0,
            "min_y": 6000010.0,
            "max_y": 6000030.0,
            "point_count": 12,
            "status": "NOISE",
            "crs_epsg": 25833,
        }
        self.mock_tiles.append(noise_tile)

        out_geojson = self.root / "tile_grid.geojson"
        res = generate_tile_grid_vector(self.db, out_geojson, grid_mode="loaded_only")

        self.assertTrue(out_geojson.exists())
        # Noise tile must be excluded, leaving exactly 2 survey tiles
        self.assertEqual(res["tile_count"], 2)

        with open(out_geojson, "r", encoding="utf-8") as f:
            doc = json.load(f)

        self.assertEqual(doc["type"], "FeatureCollection")
        self.assertEqual(len(doc["features"]), 2)
        f0 = doc["features"][0]
        self.assertEqual(f0["properties"]["tile_id"], "tile_01")
        self.assertEqual(f0["geometry"]["type"], "Polygon")
        # Assert uniform square size
        p = f0["properties"]
        self.assertEqual(p["max_x"] - p["min_x"], p["tile_size"])
        self.assertEqual(p["max_y"] - p["min_y"], p["tile_size"])

    def test_tile_grid_all_together_mode_and_overlap(self):
        """Test 'all_together' contiguous grid matrix generation with 5m overlap."""
        # Create DB with 200m tiles and 5m overlap
        t1 = {
            "id": "tile_0000",
            "filename": "tile_0000.las",
            "min_x": 500000.0,
            "max_x": 500200.0,
            "min_y": 6000000.0,
            "max_y": 6000200.0,
            "point_count": 1000,
            "status": "LOADED",
        }
        # Next col at 500000 + 195 = 500195
        t2 = {
            "id": "tile_0001",
            "filename": "tile_0001.las",
            "min_x": 500195.0,
            "max_x": 500395.0,
            "min_y": 6000000.0,
            "max_y": 6000200.0,
            "point_count": 1500,
            "status": "LOADED",
        }
        # Next row at 6000000 + 195 = 6000195 (skip col 0, row 1; put tile at col 1, row 1)
        t3 = {
            "id": "tile_0011",
            "filename": "tile_0011.las",
            "min_x": 500195.0,
            "max_x": 500395.0,
            "min_y": 6000195.0,
            "max_y": 6000395.0,
            "point_count": 800,
            "status": "LOADED",
        }
        db = MockDatabase([t1, t2, t3], db_path=str(self.root / "overlap_proj.db"))
        out_geojson = self.root / "grid_all_together.geojson"

        res = generate_tile_grid_vector(
            db,
            out_geojson,
            grid_mode="all_together",
            tile_size=200.0,
            overlap=5.0,
        )

        # 2 cols x 2 rows = 4 cells total in contiguous matrix
        self.assertEqual(res["tile_count"], 4)
        with open(out_geojson, "r", encoding="utf-8") as f:
            doc = json.load(f)

        features = doc["features"]
        self.assertEqual(len(features), 4)

        # Verify every cell has exact 200m x 200m dimensions
        for feat in features:
            props = feat["properties"]
            self.assertEqual(props["max_x"] - props["min_x"], 200.0)
            self.assertEqual(props["max_y"] - props["min_y"], 200.0)
            self.assertEqual(props["tile_size"], 200.0)
            self.assertEqual(props["overlap"], 5.0)

        # 3 loaded tiles and 1 empty cell
        loaded = [f for f in features if f["properties"]["has_data"]]
        empty = [f for f in features if not f["properties"]["has_data"]]
        self.assertEqual(len(loaded), 3)
        self.assertEqual(len(empty), 1)
        self.assertEqual(empty[0]["properties"]["status"], "EMPTY")
        self.assertTrue(empty[0]["properties"]["tile_id"].startswith("grid_"))

    def test_raster_layer_and_decimation(self):
        """Test RasterLayer open, query, and decimated rendering."""
        out_tif = self.root / "test_density.tif"
        generate_density_raster(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_path=out_tif,
        )

        layer = RasterLayer(out_tif)
        self.assertTrue(layer.is_open)
        self.assertIsNotNone(layer.bounds)

        # Test value lookup
        qx = float(self.data_map["tile_01"]["x"][0])
        qy = float(self.data_map["tile_01"]["y"][0])
        val = layer.get_value_at(qx, qy)
        self.assertIsNotNone(val)
        self.assertGreater(val, 0.0)

        # Test viewport render
        res = layer.render_viewport((500000, 6000000, 500200, 6000100), target_size=(50, 50))
        self.assertIsNotNone(res)
        rgba, bounds = res
        self.assertEqual(rgba.ndim, 3)
        self.assertEqual(rgba.shape[2], 4)

        layer.close()
        self.assertFalse(layer.is_open)

    def test_vector_layer_and_queries(self):
        """Test VectorLayer loading and spatial hit tests."""
        out_geojson = self.root / "tile_grid.geojson"
        generate_tile_grid_vector(self.db, out_geojson)

        vl = VectorLayer(out_geojson)
        self.assertEqual(len(vl.features), 2)

        # Query point inside tile 1
        feat = vl.find_feature_at(500050.0, 6000050.0)
        self.assertIsNotNone(feat)
        self.assertEqual(feat.fid, "tile_01")

        # Query box intersecting both tiles
        hits = vl.find_features_in_bbox(500050.0, 6000050.0, 500150.0, 6000050.0)
        self.assertEqual(len(hits), 2)

    def test_layer_group_and_workspace_persistence(self):
        """Test saving and loading layer groups from SSD."""
        out_geojson = self.root / "tile_grid.geojson"
        generate_tile_grid_vector(self.db, out_geojson)

        grp1 = LayerGroup("QC Rasters")
        grp2 = LayerGroup("Vector Grids")
        vl = VectorLayer(out_geojson)
        grp2.add_layer(vl)

        ws_file = self.root / "workspace.json"
        save_layer_workspace([grp1, grp2], ws_file)
        self.assertTrue(ws_file.exists())

        loaded_grps = load_layer_workspace(ws_file)
        self.assertEqual(len(loaded_grps), 2)
        self.assertEqual(loaded_grps[1].name, "Vector Grids")
        self.assertEqual(len(loaded_grps[1].children), 1)
        self.assertEqual(loaded_grps[1].children[0].name, "tile_grid")

    def test_point_qc_window_initialization(self):
        """Test PointQCWindow instantiation and signal emissions."""
        from PySide6.QtWidgets import QApplication
        from lidar_workbench.gui.point_qc_window import PointQCWindow

        app = QApplication.instance() or QApplication([])

        win = PointQCWindow(
            tile_manager=self.tm,
            database=self.db,
            project_dir=str(self.root),
        )

        self.assertIsNotNone(win)
        self.assertEqual(len(win.groups), 3)

        # Test selecting a tile and jump signal
        jump_received = []
        win.jump_to_tile.connect(lambda tid: jump_received.append(tid))

        win._set_selected_tiles(["tile_01"])
        win._on_jump_to_first_selected()
        self.assertEqual(jump_received, ["tile_01"])

        # Test bulk load signal
        bulk_received = []
        win.bulk_load_tiles.connect(lambda tids: bulk_received.append(tids))

        win._set_selected_tiles(["tile_01", "tile_02"])
        win._on_bulk_load_selected()
        self.assertEqual(bulk_received, [["tile_01", "tile_02"]])

        win.close()

    def test_drag_and_drop_layers(self):
        """Test drag-and-drop raster and vector layer batch loading."""
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QMimeData, QUrl
        from lidar_workbench.gui.point_qc_window import PointQCWindow

        app = QApplication.instance() or QApplication([])

        win = PointQCWindow(
            tile_manager=self.tm,
            database=self.db,
            project_dir=str(self.root),
        )

        self.assertTrue(win.acceptDrops())
        self.assertTrue(win._plot_widget.acceptDrops())
        self.assertTrue(win._tree.acceptDrops())

        # Create test raster and vector
        dens_tif = self.root / "drop_density.tif"
        generate_density_raster(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_path=dens_tif,
        )

        out_geojson = self.root / "drop_grid.geojson"
        generate_tile_grid_vector(self.db, out_geojson)

        # Test mime validator
        mime_valid = QMimeData()
        mime_valid.setUrls([QUrl.fromLocalFile(str(dens_tif)), QUrl.fromLocalFile(str(out_geojson))])
        self.assertTrue(win._is_valid_drop(mime_valid))

        mime_invalid = QMimeData()
        mime_invalid.setUrls([QUrl.fromLocalFile(str(self.root / "test.txt"))])
        self.assertFalse(win._is_valid_drop(mime_invalid))

        # Test load_dropped_files
        count = win.load_dropped_files([dens_tif, out_geojson])
        self.assertEqual(count, 2)

        # Verify raster added to QC Rasters group and vector to Vector Grids
        qc_group = win.groups[0]
        vec_group = win.groups[1]
        self.assertTrue(any(l.name == "drop_density" for l in qc_group.layers))
        self.assertTrue(any(l.name == "drop_grid" for l in vec_group.layers))

        win.close()

    def test_generate_max_strip_difference_raster(self):
        """Test calculation of pixel-wise max distance across all overlapping strips."""
        out_tif = self.root / "max_strip_diff.tif"
        stats = generate_max_strip_difference_raster(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_path=out_tif,
        )

        self.assertTrue(out_tif.exists())
        self.assertIn("max_dz", stats)
        self.assertIn("overlap_cells", stats)
        self.assertGreater(stats["overlap_cells"], 0)

        with rasterio.open(out_tif) as src:
            diff_data = src.read(1)
            valid = (diff_data != src.nodata) & np.isfinite(diff_data)
            self.assertTrue(np.any(valid))
            # Max strip distance must be strictly non-negative
            self.assertTrue(np.all(diff_data[valid] >= 0.0))

    def test_batch_export_all_strip_differences(self):
        """Test batch export of all overlapping strip difference pairs."""
        # Ensure tiles have flightline_sensor_types
        self.mock_tiles[0]["flightline_sensor_types"] = json.dumps({"1": "topo", "2": "topo"})
        self.mock_tiles[1]["flightline_sensor_types"] = json.dumps({"2": "topo"})

        out_dir = self.root / "batch_strip_diffs"
        res = batch_export_all_strip_differences(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_dir=out_dir,
        )

        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["total_pairs"], 1)
        for p in res["output_paths"]:
            self.assertTrue(Path(p).exists())

    def test_raster_viewport_rendering_bounds_inversion_and_colormap(self):
        """Test that RasterLayer handles inverted viewport coordinates (vy0 > vy1) and diverging vs sequential colormaps."""
        # Generate pairwise strip diff raster
        pair_tif = self.root / "strip_1_vs_2_5.0m.tif"
        generate_strip_difference_raster(
            tile_manager=self.tm,
            database=self.db,
            strip_a=1,
            strip_b=2,
            cell_size=5.0,
            output_path=pair_tif,
        )

        layer_pair = RasterLayer(pair_tif)
        self.assertTrue(layer_pair.is_open)
        self.assertTrue(layer_pair.is_diverging)
        self.assertEqual(layer_pair.colormap_name, "coolwarm")

        # Test rendering with inverted bounds: vy0 > vy1 (Qt view_rect.bottom() > view_rect.top())
        b = layer_pair.bounds
        res_inverted = layer_pair.render_viewport((b[0], b[3], b[2], b[1]), target_size=(200, 200))
        self.assertIsNotNone(res_inverted)
        rgba, bounds = res_inverted
        self.assertEqual(len(rgba.shape), 3)
        self.assertEqual(rgba.shape[2], 4)

        # Generate max strip diff raster
        max_tif = self.root / "max_strip_difference_5.0m.tif"
        generate_max_strip_difference_raster(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_path=max_tif,
        )

        layer_max = RasterLayer(max_tif)
        self.assertTrue(layer_max.is_open)
        # Max difference is non-negative, so it should not be diverging
        self.assertFalse(layer_max.is_diverging)
        self.assertEqual(layer_max.colormap_name, "turbo")
        self.assertGreaterEqual(layer_max.vmin, 0.0)

    def test_raster_min_max_value_modification_and_presets(self):
        """Test user interactive modification of raster min/max color stretch values and presets."""
        from lidar_workbench.gui.point_qc_window import PointQCWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])

        win = PointQCWindow(project_dir=self.root, tile_manager=self.tm, database=self.db)
        out_tif = self.root / "max_diff_test.tif"
        generate_max_strip_difference_raster(
            tile_manager=self.tm,
            database=self.db,
            cell_size=5.0,
            output_path=out_tif,
        )

        rl = win.add_raster_layer(out_tif, group_name="QC Rasters")
        # Select the layer in the tree
        qc_item = win._tree.topLevelItem(0).child(0)
        win._tree.setCurrentItem(qc_item)

        # 1. Modify min and max value (e.g. 0.0m min and 0.3m max)
        win._vmin_spin.setValue(0.0)
        win._vmax_spin.setValue(0.3)
        self.assertAlmostEqual(rl.vmin, 0.0, places=3)
        self.assertAlmostEqual(rl.vmax, 0.3, places=3)

        # 2. Select preset
        idx = win._preset_combo.findText("0.0 m to 0.5 m")
        self.assertGreaterEqual(idx, 0)
        win._preset_combo.setCurrentIndex(idx)
        self.assertAlmostEqual(rl.vmin, 0.0, places=3)
        self.assertAlmostEqual(rl.vmax, 0.5, places=3)

        # 3. Auto Range
        win._auto_range_btn.click()
        self.assertIsNotNone(rl.vmin)
        self.assertIsNotNone(rl.vmax)

        win.close()


if __name__ == "__main__":
    unittest.main()

