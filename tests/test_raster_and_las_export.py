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
    compute_hillshade,
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

    def test_adjacent_tiles_neighbor_buffering_dtm_seamless(self):
        """Test that adjacent tiles rasterized with neighbor buffering produce a continuous DTM."""
        import laspy
        from export_manager import _attach_neighbor_paths, _rasterize_tile_job, ExportConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            p_west = Path(tmpdir) / "tile_0001.las"
            p_east = Path(tmpdir) / "tile_0002.las"

            # Create synthetic sloping terrain: z = 100.0 + 0.5 * x + 0.2 * y
            # West tile: bbox [0, 0, 50, 50]. Points stop at x=48.5 (1.5m gap before border at 50)
            # East tile: bbox [50, 0, 100, 50]. Points start at x=51.5 (1.5m gap after border at 50)
            def make_las(path, x_range, y_range):
                xs_grid, ys_grid = np.meshgrid(x_range, y_range)
                xs = xs_grid.ravel()
                ys = ys_grid.ravel()
                zs = 100.0 + 0.5 * xs + 0.2 * ys
                hdr = laspy.LasHeader(point_format=3, version="1.2")
                hdr.offsets = [0.0, 0.0, 0.0]
                hdr.scales = [0.01, 0.01, 0.01]
                las = laspy.LasData(hdr)
                las.x = xs
                las.y = ys
                las.z = zs
                las.classification = np.full(len(xs), 2, dtype=np.uint8)
                las.write(str(path))

            make_las(p_west, np.linspace(1.0, 48.5, 30), np.linspace(1.0, 49.0, 30))
            make_las(p_east, np.linspace(51.5, 99.0, 30), np.linspace(1.0, 49.0, 30))

            jobs = [
                {"tile_id": "tile_0001", "las_path": str(p_west), "bbox": (0.0, 0.0, 50.0, 50.0)},
                {"tile_id": "tile_0002", "las_path": str(p_east), "bbox": (50.0, 0.0, 100.0, 50.0)},
            ]
            global_bbox = (0.0, 100.0, 0.0, 50.0)
            config = ExportConfig(output_dir=tmpdir, resolution=1.0, mode="dtm")

            # Attach neighbor paths
            buffered_jobs = _attach_neighbor_paths(jobs, buffer_m=15.0)
            self.assertEqual(len(buffered_jobs[0]["neighbor_las_paths"]), 1)
            self.assertEqual(len(buffered_jobs[1]["neighbor_las_paths"]), 1)

            # Rasterize both tiles
            res_west = _rasterize_tile_job(buffered_jobs[0], global_bbox, config)
            res_east = _rasterize_tile_job(buffered_jobs[1], global_bbox, config)

            west_data = res_west["data"]  # shape (50, 50), row 0 is South, col 0 is West
            east_data = res_east["data"]

            # Border is between col -1 of west (x = 49.5) and col 0 of east (x = 50.5)
            # Distance is dx = 1.0 m, slope is dz/dx = 0.5.
            # Expected step across border = 0.5 * 1.0 = 0.5 m
            step = east_data[:, 0] - west_data[:, -1]
            # Verify across middle rows (away from outer boundaries) that elevation step is ~0.5m
            mid_slice = slice(5, 45)
            self.assertTrue(np.allclose(step[mid_slice], 0.5, atol=0.05),
                            f"Elevation step across border should be ~0.5m, got: {step[mid_slice]}")

    def test_export_tile_footprints_shp_and_gpkg(self):
        """Test that export_tile_footprints produces valid Shapefile and GeoPackage with QGIS style."""
        import shapefile
        import sqlite3
        from export_manager import export_tile_footprints

        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {
                    "tile_id": "tile_0001",
                    "bbox": (100.0, 200.0, 300.0, 400.0),
                    "dtm_file": "tile_0001_dtm.asc",
                    "dsm_file": "",
                    "hs_file": "tile_0001_dtm_hillshade.tif",
                    "las_file": "tile_0001.las",
                    "cellsize": 1.0,
                    "epsg": 25833,
                },
                {
                    "tile_id": "tile_0002",
                    "bbox": (300.0, 200.0, 500.0, 400.0),
                    "dtm_file": "tile_0002_dtm.asc",
                    "dsm_file": "",
                    "hs_file": "tile_0002_dtm_hillshade.tif",
                    "las_file": "tile_0002.las",
                    "cellsize": 1.0,
                    "epsg": 25833,
                },
            ]
            wkt_25833 = 'PROJCS["ETRS89 / UTM zone 33N",GEOGCS["ETRS89"]]'
            files = export_tile_footprints(
                records,
                output_dir=tmpdir,
                stem="tile_grid",
                formats="both",
                crs_epsg=25833,
                crs_wkt=wkt_25833,
            )
            file_names = [Path(f).name for f in files]

            # 1. Shapefile verification
            self.assertIn("tile_grid.shp", file_names)
            self.assertIn("tile_grid.dbf", file_names)
            self.assertIn("tile_grid.shx", file_names)
            self.assertIn("tile_grid.prj", file_names)
            self.assertIn("tile_grid.qml", file_names)

            sf = shapefile.Reader(str(Path(tmpdir) / "tile_grid.shp"))
            self.assertEqual(len(sf.shapes()), 2)
            self.assertEqual(sf.shapes()[0].shapeType, shapefile.POLYGON)
            rec0 = sf.record(0)
            self.assertEqual(rec0["tile_id"], "tile_0001")
            self.assertAlmostEqual(rec0["width_m"], 200.0)

            # Check that adjacent tiles touch with NO OVERLAP
            # tile 1 max_x == tile 2 min_x == 300.0
            self.assertAlmostEqual(rec0["max_x"], 300.0)
            rec1 = sf.record(1)
            self.assertEqual(rec1["tile_id"], "tile_0002")
            self.assertAlmostEqual(rec1["min_x"], 300.0)

            # 2. GeoPackage verification
            self.assertIn("tile_grid.gpkg", file_names)
            gpkg_path = Path(tmpdir) / "tile_grid.gpkg"
            conn = sqlite3.connect(str(gpkg_path))
            cur = conn.cursor()
            cur.execute("SELECT tile_id, min_x, max_x FROM tile_footprints ORDER BY tile_id")
            rows = cur.fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], "tile_0001")
            self.assertAlmostEqual(rows[0][2], 300.0)
            self.assertEqual(rows[1][0], "tile_0002")
            self.assertAlmostEqual(rows[1][1], 300.0)
            conn.close()

    def test_dtm_parallel_export_with_footprints_and_no_overlap(self):
        """Test that export_dtm_parallel clips overlapping tiles and creates non-overlapping footprints."""
        import laspy
        from export_manager import export_dtm_parallel, ExportConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            p_west = Path(tmpdir) / "tile_0001.las"
            p_east = Path(tmpdir) / "tile_0002.las"

            # Create two overlapping tiles: stride=190m, overlap=10m (each tile is 200m wide)
            # West tile: [0, 0, 200, 200]
            # East tile: [190, 0, 390, 200]
            def make_las(path, x_min, x_max):
                xs_grid, ys_grid = np.meshgrid(np.linspace(x_min + 1, x_max - 1, 20), np.linspace(1, 199, 20))
                xs = xs_grid.ravel()
                ys = ys_grid.ravel()
                zs = 100.0 + 0.1 * xs
                hdr = laspy.LasHeader(point_format=3, version="1.2")
                hdr.offsets = [0.0, 0.0, 0.0]
                hdr.scales = [0.01, 0.01, 0.01]
                las = laspy.LasData(hdr)
                las.x = xs
                las.y = ys
                las.z = zs
                las.classification = np.full(len(xs), 2, dtype=np.uint8)
                las.write(str(path))

            make_las(p_west, 0.0, 200.0)
            make_las(p_east, 190.0, 390.0)

            jobs = [
                {"tile_id": "tile_0001", "las_path": str(p_west), "bbox": (0.0, 0.0, 200.0, 200.0)},
                {"tile_id": "tile_0002", "las_path": str(p_east), "bbox": (190.0, 0.0, 390.0, 200.0)},
            ]
            global_bbox = (0.0, 390.0, 0.0, 200.0)
            config = ExportConfig(
                output_dir=str(Path(tmpdir) / "out"),
                resolution=1.0,
                mode="dtm",
                compute_hillshade=True,
                clip_overlap=True,
                overlap_m=10.0,
                tile_size_m=200.0,
                export_footprints=True,
                footprint_format="both",
                crs_epsg=25833,
            )

            written = export_dtm_parallel(jobs, global_bbox, config, workers=1)
            file_names = [p.name for p in written]

            # Rasters and hillshades created
            self.assertIn("tile_0001_dtm.asc", file_names)
            self.assertIn("tile_0002_dtm.asc", file_names)
            self.assertIn("tile_0001_dtm_hillshade.tif", file_names)
            self.assertIn("tile_0002_dtm_hillshade.tif", file_names)

            # Footprint vector files created
            self.assertIn("tile_footprints_dtm.shp", file_names)
            self.assertIn("tile_footprints_dtm.gpkg", file_names)

            # Read West and East DTM grids and verify NO overlap
            asc_west = Path(tmpdir) / "out" / "tile_0001_dtm.asc"
            asc_east = Path(tmpdir) / "out" / "tile_0002_dtm.asc"
            lines_w = asc_west.read_text().splitlines()[:6]
            lines_e = asc_east.read_text().splitlines()[:6]

            def parse_header(lines):
                return {l.split()[0].lower(): float(l.split()[1]) for l in lines}

            hw = parse_header(lines_w)
            he = parse_header(lines_e)

            # West tile: xllcorner=0, ncols=195 (trimmed 5m on east border: 200 - 5 = 195)
            # East tile: xllcorner=195, ncols=195 (trimmed 5m on west border: 190 + 5 = 195)
            self.assertAlmostEqual(hw["xllcorner"], 0.0)
            self.assertAlmostEqual(hw["ncols"], 195.0)
            self.assertAlmostEqual(he["xllcorner"], 195.0)
            self.assertAlmostEqual(he["ncols"], 195.0)

            # They meet EXACTLY at x = 195.0 with ZERO overlap!
            self.assertAlmostEqual(hw["xllcorner"] + hw["ncols"] * hw["cellsize"], he["xllcorner"])

    def test_hillshade_relief_orientation_and_no_inversion(self):
        """
        Verify that compute_hillshade applies correct Horn/GIS illumination so that
        depressions (such as river beds and valleys) are not inverted into ridges.
        """
        # 1. East-West River Trench
        # South to North (rows 0 to 4):
        # row 0: 10m (high ground south)
        # row 1: 8m  (south bank sloping down to north)
        # row 2: 0m  (river bed)
        # row 3: 8m  (north bank sloping down to south)
        # row 4: 10m (high ground north)
        ew_data = np.array([
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [ 8.0,  8.0,  8.0,  8.0,  8.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0],
            [ 8.0,  8.0,  8.0,  8.0,  8.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
        ], dtype=float)

        grid_ew = RasterGrid(data=ew_data, xllcorner=0.0, yllcorner=0.0, cellsize=1.0, nodata=-9999.0)
        hs_ew = compute_hillshade(grid_ew, azimuth=315.0, altitude=45.0)

        # In standard GeoTIFF image orientation (row 0 = North, row 4 = South):
        img_ew = np.flipud(hs_ew.data)
        north_bank_val = img_ew[1, 2]  # Upper bank (facing south into river)
        south_bank_val = img_ew[3, 2]  # Lower bank (facing north into river)

        # Under NW lighting (azimuth 315):
        # North bank slopes away from the NW light -> in shadow (dark)
        # South bank faces toward the NW light -> illuminated (bright)
        self.assertLess(north_bank_val, south_bank_val,
                        f"North bank ({north_bank_val}) should be darker than South bank ({south_bank_val})")

        # 2. North-South River Trench
        # West to East (cols 0 to 4):
        ns_data = np.array([
            [10.0, 8.0, 0.0, 8.0, 10.0],
            [10.0, 8.0, 0.0, 8.0, 10.0],
            [10.0, 8.0, 0.0, 8.0, 10.0],
            [10.0, 8.0, 0.0, 8.0, 10.0],
            [10.0, 8.0, 0.0, 8.0, 10.0],
        ], dtype=float)

        grid_ns = RasterGrid(data=ns_data, xllcorner=0.0, yllcorner=0.0, cellsize=1.0, nodata=-9999.0)
        hs_ns = compute_hillshade(grid_ns, azimuth=315.0, altitude=45.0)

        img_ns = np.flipud(hs_ns.data)
        west_bank_val = img_ns[2, 1]  # West bank (facing east into river)
        east_bank_val = img_ns[2, 3]  # East bank (facing west into river)

        # Under NW lighting:
        # West bank faces east (away from light) -> in shadow (dark)
        # East bank faces west (towards light) -> illuminated (bright)
        self.assertLess(west_bank_val, east_bank_val,
                        f"West bank ({west_bank_val}) should be darker than East bank ({east_bank_val})")


if __name__ == "__main__":
    unittest.main()


