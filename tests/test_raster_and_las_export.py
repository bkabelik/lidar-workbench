"""Unit tests for raster vertical alignment, CRS embedding, and LAS export."""
import os
import sys
import tempfile
import unittest
import numpy as np
from pathlib import Path

# Ensure package root is in sys.path
_pkg_root = str(Path(__file__).resolve().parents[1].parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
_app_root = str(Path(__file__).resolve().parents[1])
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from export_manager import (
    RasterGrid,
    ExportConfig,
    _write_ascii_grid,
    _write_geotiff,
    compute_core_bbox,
)


class TestRasterAndLasExport(unittest.TestCase):
    def test_ascii_and_geotiff_orientation_and_crs(self):
        """Test that row 0 in exported raster corresponds to the North edge."""
        # Create a 4x4 grid where row 0 (South) has value 10, and row 3 (North) has value 40
        # In RasterGrid convention: row 0 is ymin (South), row nrows-1 is ymax (North).
        grid_data = np.zeros((4, 4), dtype=np.float32)
        grid_data[0, :] = 10.0  # South
        grid_data[1, :] = 20.0
        grid_data[2, :] = 30.0
        grid_data[3, :] = 40.0  # North

        grid = RasterGrid(
            data=grid_data,
            xllcorner=100.0,
            yllcorner=200.0,
            cellsize=1.0,
            nodata=-9999.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            asc_path = Path(tmpdir) / "test.asc"
            tif_path = Path(tmpdir) / "test.tif"

            # Write ASCII grid with EPSG 25833 WKT
            wkt_25833 = 'PROJCS["ETRS89 / UTM zone 33N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101]],AUTHORITY["EPSG","25833"]]'
            _write_ascii_grid(grid, asc_path, crs_wkt=wkt_25833)

            # Check that .prj was created
            prj_path = asc_path.with_suffix(".prj")
            self.assertTrue(prj_path.is_file(), "ASCII companion .prj file should exist")
            prj_text = prj_path.read_text()
            self.assertIn("25833", prj_text)

            # Read ASCII grid text lines
            lines = asc_path.read_text().strip().split("\n")
            # Headers are first 6 lines
            data_lines = [l.strip() for l in lines[6:]]
            self.assertEqual(len(data_lines), 4)
            # First data line corresponds to North (row 3 in RasterGrid, value 40.0)
            first_row_vals = [float(x) for x in data_lines[0].split()]
            self.assertEqual(first_row_vals, [40.0, 40.0, 40.0, 40.0])
            # Last data line corresponds to South (row 0 in RasterGrid, value 10.0)
            last_row_vals = [float(x) for x in data_lines[-1].split()]
            self.assertEqual(last_row_vals, [10.0, 10.0, 10.0, 10.0])

            # Write GeoTIFF with EPSG 25833
            _write_geotiff(grid, tif_path, crs_epsg=25833, crs_wkt=wkt_25833)
            self.assertTrue(tif_path.is_file())
            tif_prj = tif_path.with_suffix(".prj")
            self.assertTrue(tif_prj.is_file(), "GeoTIFF companion .prj file should exist")

            # Check with rasterio if available
            try:
                import rasterio
                with rasterio.open(tif_path) as src:
                    self.assertEqual(src.crs.to_epsg(), 25833)
                    # src.read(1)[0, 0] is top-left (North-West), should be 40.0
                    raster_data = src.read(1)
                    self.assertAlmostEqual(raster_data[0, 0], 40.0, places=4)
                    self.assertAlmostEqual(raster_data[-1, 0], 10.0, places=4)
            except ImportError:
                pass

    def test_adjacent_tiles_raster_continuity(self):
        """Test that two vertically adjacent tiles match at the boundary after export."""
        # Tile 1: South tile, y from 0 to 100
        # Tile 2: North tile, y from 100 to 200
        # Linear ramp: z = y
        y1 = np.linspace(0.5, 99.5, 100)
        grid1_data = np.repeat(y1[:, None], 100, axis=1).astype(np.float32)

        y2 = np.linspace(100.5, 199.5, 100)
        grid2_data = np.repeat(y2[:, None], 100, axis=1).astype(np.float32)

        grid_south = RasterGrid(data=grid1_data, xllcorner=0.0, yllcorner=0.0, cellsize=1.0, nodata=-9999.0)
        grid_north = RasterGrid(data=grid2_data, xllcorner=0.0, yllcorner=100.0, cellsize=1.0, nodata=-9999.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            p_south = Path(tmpdir) / "south.asc"
            p_north = Path(tmpdir) / "north.asc"
            _write_ascii_grid(grid_south, p_south)
            _write_ascii_grid(grid_north, p_north)

            lines_s = p_south.read_text().strip().split("\n")[6:]
            lines_n = p_north.read_text().strip().split("\n")[6:]

            # The North edge of South tile is row 0 of lines_s (since written North-first)
            north_edge_of_south = float(lines_s[0].split()[0])
            # The South edge of North tile is row -1 of lines_n
            south_edge_of_north = float(lines_n[-1].split()[0])

            # In continuous elevation z = y, south tile top cell is at y=99.5, north tile bottom cell is at y=100.5
            # Difference should be exactly 1.0 (continuous terrain), NOT inverted!
            self.assertAlmostEqual(south_edge_of_north - north_edge_of_south, 1.0, places=2)

    def test_compute_core_bbox_disjoint_tiles(self):
        """Test that compute_core_bbox does not clip non-overlapping tiles."""
        # Non-overlapping tile with tile_size_m=200, width=195m (already partitioned)
        bbox = (1000.0, 2000.0, 1195.0, 2195.0)
        global_bbox = (0.0, 0.0, 10000.0, 10000.0)
        core = compute_core_bbox(bbox, global_bbox, overlap_m=5.0, tile_size_m=200.0)
        # Bbox should remain untouched!
        self.assertEqual(core, bbox)

    def test_compute_core_bbox_overlapping_tiles(self):
        """Test that compute_core_bbox correctly trims overlapping tiles."""
        # Overlapping tile with width 205m (stride 200m + 5m overlap)
        bbox = (1000.0, 2000.0, 1205.0, 2205.0)
        global_bbox = (0.0, 0.0, 10000.0, 10000.0)
        core = compute_core_bbox(bbox, global_bbox, overlap_m=5.0, tile_size_m=200.0)
        # Should trim 2.5m on each side
        self.assertEqual(core, (1002.5, 2002.5, 1202.5, 2202.5))

    def test_tile_list_widget_export_las_signal(self):
        """Test that TileListWidget has export_las_requested signal and can emit it."""
        import sys
        root = Path(__file__).resolve().parents[1].parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from PySide6.QtWidgets import QApplication
        from lidar_workbench.gui.tile_list_widget import TileListWidget

        app = QApplication.instance() or QApplication([])
        widget = TileListWidget()

        emitted = []
        widget.export_las_requested.connect(lambda tids: emitted.append(tids))
        widget.export_las_requested.emit(["tile_0001", "tile_0002"])

        self.assertEqual(emitted, [["tile_0001", "tile_0002"]])

    def test_compute_hillshade_no_zero_border(self):
        """Test that compute_hillshade computes valid slopes at border pixels without 181 flat artifact."""
        from export_manager import compute_hillshade
        # Create a sloping plane: z = 2.0 * x + 1.0 * y
        x = np.arange(10, dtype=np.float32)
        y = np.arange(10, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        z = 2.0 * xx + 1.0 * yy

        grid = RasterGrid(data=z, xllcorner=0.0, yllcorner=0.0, cellsize=1.0, nodata=-9999.0)
        hs = compute_hillshade(grid, azimuth=315.0, altitude=45.0)

        # On flat zero slope (dzdx=0, dzdy=0), hs would be ~180.605
        # On this strong slope, border pixels should NOT be 180.605!
        self.assertFalse(np.isclose(hs.data[0, :], 180.60512, atol=1e-2).all(), "Top border should not be flat 181")
        self.assertFalse(np.isclose(hs.data[-1, :], 180.60512, atol=1e-2).all(), "Bottom border should not be flat 181")
        self.assertFalse(np.isclose(hs.data[:, 0], 180.60512, atol=1e-2).all(), "Left border should not be flat 181")
        self.assertFalse(np.isclose(hs.data[:, -1], 180.60512, atol=1e-2).all(), "Right border should not be flat 181")

    def test_geotiff_aux_pam_statistics(self):
        """Test that GeoTIFF export writes STATISTICS tags and .aux.xml locking 0-255 scale."""
        data = np.full((10, 10), 150.0, dtype=np.float32)
        grid = RasterGrid(data=data, xllcorner=0.0, yllcorner=0.0, cellsize=1.0, nodata=-9999.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir) / "test_hs.tif"
            _write_geotiff(grid, tif_path, crs_epsg=25833)

            # Check .aux.xml file exists and has 0 to 255
            aux_path = Path(str(tif_path) + ".aux.xml")
            self.assertTrue(aux_path.is_file(), "Companion .aux.xml must be created")
            aux_text = aux_path.read_text()
            self.assertIn("STATISTICS_MINIMUM", aux_text)
            self.assertIn("STATISTICS_MAXIMUM", aux_text)
            self.assertIn(">0<", aux_text)
            self.assertIn(">255<", aux_text)

            # Check .qml file exists and has limits=None
            qml_path = tif_path.with_suffix(".qml")
            self.assertTrue(qml_path.is_file(), "Companion .qml must be created")
            qml_text = qml_path.read_text()
            self.assertIn("<limits>None</limits>", qml_text)
            self.assertIn("<minValue>0</minValue>", qml_text)
            self.assertIn("<maxValue>255</maxValue>", qml_text)

            # Check rasterio tags
            try:
                import rasterio
                with rasterio.open(str(tif_path)) as src:
                    tags = src.tags(1)
                    self.assertEqual(tags.get("STATISTICS_MINIMUM"), "0")
                    self.assertEqual(tags.get("STATISTICS_MAXIMUM"), "255")
            except ImportError:
                pass

    def test_compute_core_bbox_disjoint_tile_jitter(self):
        """Test that tiles with slight coordinate jitter (e.g. 195.15m) are NOT clipped."""
        # Non-overlapping tile with tile_size_m=200, overlap_m=5.0 (stride=195m)
        # Bbox width is 195.15m (> 195.10m by 5cm due to point quantization)
        bbox = (1000.0, 2000.0, 1195.15, 2195.15)
        global_bbox = (0.0, 0.0, 10000.0, 10000.0)
        core = compute_core_bbox(bbox, global_bbox, overlap_m=5.0, tile_size_m=200.0)
        self.assertEqual(core, bbox, "Disjoint tile with 195.15m width must not be clipped!")

    def test_dsm_highest_point_in_pixel(self):
        """Test that DSM binning picks the exact highest point (max-Z) in each pixel."""
        from export_manager import _bin_max

        # Create a 2x2 grid with cell size 1.0
        grid = RasterGrid(
            data=np.empty((2, 2), dtype=np.float64),
            xllcorner=0.0,
            yllcorner=0.0,
            cellsize=1.0,
            nodata=-9999.0,
        )

        # In pixel (col=0, row=0): three points (ground=100.0, branch=115.0, canopy=128.5)
        px = np.array([0.2, 0.5, 0.8, 1.5], dtype=np.float64)
        py = np.array([0.2, 0.5, 0.8, 1.5], dtype=np.float64)
        pz = np.array([100.0, 128.5, 115.0, 200.0], dtype=np.float64)

        binned = _bin_max(px, py, pz, grid, fill_voids=False)
        self.assertAlmostEqual(binned[0, 0], 128.5, places=4, msg="Highest point in pixel must be selected")
        self.assertAlmostEqual(binned[1, 1], 200.0, places=4)

    def test_dsm_void_filling_interpolates_water_and_gaps(self):
        """Test that DSM void filling smoothly interpolates lake/water gaps at surface level."""
        from export_manager import _bin_max

        # Create a 20x20 grid with cell size 1.0 (x from 0 to 20, y from 0 to 20)
        grid = RasterGrid(
            data=np.empty((20, 20), dtype=np.float64),
            xllcorner=0.0,
            yllcorner=0.0,
            cellsize=1.0,
            nodata=-9999.0,
        )

        # Points along perimeter representing shoreline/terrain at elevation 150.0m
        # Inside (x from 5 to 15, y from 5 to 15) is an unreflected water body (0 returns)
        xs = []
        ys = []
        zs = []
        for x in range(20):
            for y in range(20):
                if x < 5 or x > 15 or y < 5 or y > 15:
                    xs.append(x + 0.5)
                    ys.append(y + 0.5)
                    zs.append(150.0)

        px = np.array(xs, dtype=np.float64)
        py = np.array(ys, dtype=np.float64)
        pz = np.array(zs, dtype=np.float64)

        # Without void filling: lake center (10, 10) must be nodata
        no_fill = _bin_max(px, py, pz, grid, fill_voids=False)
        self.assertEqual(no_fill[10, 10], grid.nodata)

        # With void filling: lake center (10, 10) must be smoothly interpolated to water surface level 150.0m
        filled = _bin_max(px, py, pz, grid, fill_voids=True)
        self.assertNotEqual(filled[10, 10], grid.nodata)
        self.assertAlmostEqual(filled[10, 10], 150.0, places=1)
        # Entire grid must have no nodata holes remaining
        self.assertFalse((filled == grid.nodata).any())

    def test_export_las_tile_clean_naming_and_filtering(self):
        """Test that export_las_tile exports clean filenames and respects class filter."""
        from export_manager import export_las_tile
        import laspy

        with tempfile.TemporaryDirectory() as tmpdir:
            src_las = Path(tmpdir) / "tile_0133_ground.las"
            out_dir = Path(tmpdir) / "exported"

            # Create synthetic LAS file with points of class 2 (ground) and class 5 (high veg)
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.offsets = [0.0, 0.0, 0.0]
            header.scales = [0.01, 0.01, 0.01]
            las = laspy.LasData(header)
            las.x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
            las.y = np.array([10.0, 20.0, 30.0], dtype=np.float64)
            las.z = np.array([100.0, 110.0, 120.0], dtype=np.float64)
            las.classification = np.array([2, 5, 2], dtype=np.uint8)
            las.write(str(src_las))

            job = {
                "tile_id": "tile_0133",
                "las_path": str(src_las),
                "bbox": (0.0, 0.0, 100.0, 100.0),
            }
            g_bbox = (0.0, 0.0, 1000.0, 1000.0)

            # 1. Export all points: should produce tile_0133.las with 3 points
            out1 = export_las_tile(job, str(out_dir), g_bbox, overlap_m=0.0)
            self.assertIsNotNone(out1)
            p1 = Path(out1)
            self.assertEqual(p1.name, "tile_0133.las")
            with laspy.open(p1) as r:
                d = r.read()
                self.assertEqual(len(d.x), 3)

            # 2. Export ground only: should produce tile_0133_ground.las with 2 points
            out2 = export_las_tile(job, str(out_dir), g_bbox, overlap_m=0.0, classes={2})
            self.assertIsNotNone(out2)
            p2 = Path(out2)
            self.assertEqual(p2.name, "tile_0133_ground.las")
            with laspy.open(p2) as r:
                d = r.read()
                self.assertEqual(len(d.x), 2)
                self.assertTrue(np.all(d.classification == 2))


if __name__ == "__main__":
    unittest.main()
