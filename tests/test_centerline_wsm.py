import unittest
import numpy as np
import tempfile
from pathlib import Path

from centerline_wsm import (
    RiverCenterline,
    slice_cross_section_points,
    detect_section_water_level,
    enforce_downstream_monotonicity,
    interpolate_anchor_sections,
    rasterize_water_surface_model,
)

class TestCenterlineWSM(unittest.TestCase):
    def setUp(self):
        # 2D S-curve river centerline from (0, 0) to (100, 200)
        t = np.linspace(0, 1, 50)
        x = 50.0 * np.sin(t * np.pi)
        y = t * 200.0
        self.verts = np.column_stack((x, y))
        self.centerline = RiverCenterline(self.verts)

    def test_centerline_properties(self):
        self.assertGreater(self.centerline.total_length, 200.0)
        stations = self.centerline.sample_stations(spacing=10.0)
        self.assertGreaterEqual(len(stations), 20)
        self.assertAlmostEqual(stations[0], 0.0)
        self.assertAlmostEqual(stations[-1], self.centerline.total_length)

        # Test evaluate
        pos, tangent, normal = self.centerline.evaluate(stations[5])
        self.assertEqual(pos.shape, (2,))
        self.assertEqual(tangent.shape, (2,))
        self.assertEqual(normal.shape, (2,))
        # Tangent and normal must be orthogonal unit vectors
        self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=5)
        dot = float(np.dot(tangent, normal))
        self.assertAlmostEqual(dot, 0.0, places=5)

    def test_slicing_and_detection(self):
        # Create a synthetic channel at station s = 50
        s = 50.0
        pos, tangent, normal = self.centerline.evaluate(s)
        # Sliced channel:
        # Left bank dirt (NIR): offset in [-20, -6], Z in [102, 100.2]
        # Water channel: offset in [-6, +6], no NIR points, green bathy points at Z in [97, 99.8] (surface at 100.0)
        # Right bank dirt (NIR): offset in [+6, +20], Z in [100.2, 102]
        np.random.seed(42)
        n_left = 60
        off_left = np.random.uniform(-20, -6, n_left)
        z_left = 100.0 + 0.1 * (np.abs(off_left) - 6.0) + np.random.normal(0, 0.02, n_left)
        st_left = np.ones(n_left, dtype=np.int32) # NIR=1

        n_right = 60
        off_right = np.random.uniform(6, 20, n_right)
        z_right = 100.0 + 0.1 * (np.abs(off_right) - 6.0) + np.random.normal(0, 0.02, n_right)
        st_right = np.ones(n_right, dtype=np.int32) # NIR=1

        n_bathy = 50
        off_bathy = np.random.uniform(-5, 5, n_bathy)
        # green bathy points from bed 97m to water surface scatter 99.9m
        z_bathy = np.random.uniform(97.0, 99.9, n_bathy)
        st_bathy = np.full(n_bathy, 2, dtype=np.int32) # Bathy=2

        offsets = np.concatenate([off_left, off_bathy, off_right])
        elevations = np.concatenate([z_left, z_bathy, z_right])
        sensor_types = np.concatenate([st_left, st_bathy, st_right])

        # Test automated detection
        w_z, zl, zr = detect_section_water_level(offsets, elevations, sensor_types)
        # Water surface should be detected around 100.0m (+/- 0.15m)
        self.assertAlmostEqual(w_z, 100.0, delta=0.20)
        self.assertAlmostEqual(zl, 100.0, delta=0.25)
        self.assertAlmostEqual(zr, 100.0, delta=0.25)

    def test_monotonicity_and_anchors(self):
        stations = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        # Auto levels have small noisy bumps (100.0, 99.8, 99.9 (error!), 99.5, 99.2, 99.0)
        auto_levels = np.array([100.0, 99.8, 99.9, 99.5, 99.2, 99.0])
        locked_mask = np.zeros(len(stations), dtype=bool)

        mono = enforce_downstream_monotonicity(auto_levels, locked_mask)
        # Verify strictly non-increasing
        diffs = np.diff(mono)
        self.assertTrue((diffs <= 1e-9).all())

        # Test locking an anchor at station 30m with Z = 99.3m
        locked_mask[3] = True
        auto_levels[3] = 99.3
        interp = interpolate_anchor_sections(stations, auto_levels, locked_mask)
        self.assertAlmostEqual(interp[3], 99.3)
        self.assertTrue((np.diff(interp) <= 1e-9).all())

    def test_rasterization(self):
        stations = np.linspace(0, 100, 11)
        w_levels = np.linspace(100.0, 95.0, 11)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tif = Path(tmpdir) / "test_wsm.tif"
            surf, georef = rasterize_water_surface_model(
                self.centerline, stations, w_levels,
                corridor_width=30.0, output_path=out_tif, resolution=1.0, data_epsg=25832,
            )
            self.assertTrue(out_tif.is_file())
            self.assertEqual(surf.ndim, 2)
            self.assertFalse(np.isnan(surf).all())
            # Minimum water level is 95.0, maximum is 100.0
            valid = surf[~np.isnan(surf)]
            self.assertGreaterEqual(float(np.min(valid)), 94.9)
            self.assertLessEqual(float(np.max(valid)), 100.1)

    def test_stitch_contiguous_waterways(self):
        from centerline_wsm import stitch_contiguous_waterways
        # Two connected segments of a river named "Danube"
        seg1 = {
            "id": 1,
            "name": "Danube",
            "waterway": "river",
            "vertices": np.array([[0.0, 0.0], [100.0, 50.0]]),
            "length_m": 111.8,
        }
        seg2 = {
            "id": 2,
            "name": "Danube",
            "waterway": "river",
            "vertices": np.array([[100.0, 50.0], [200.0, 100.0]]),
            "length_m": 111.8,
        }
        merged = stitch_contiguous_waterways([seg1, seg2], max_gap=10.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "Danube")
        self.assertEqual(len(merged[0]["vertices"]), 3)
        self.assertAlmostEqual(merged[0]["length_m"], 223.6, places=1)

    def test_elevation_independence_and_formats(self):
        from centerline_wsm import load_centerline_from_file

        # 1. Direct 3D array (X, Y, Z) into RiverCenterline
        verts_3d = np.array([
            [10.0, 20.0, 150.0],
            [20.0, 40.0, 149.0],
            [30.0, 60.0, 148.0],
        ])
        cl = RiverCenterline(verts_3d)
        self.assertEqual(cl.vertices.shape, (3, 2))
        self.assertAlmostEqual(cl.vertices[0, 0], 10.0)
        self.assertAlmostEqual(cl.vertices[0, 1], 20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 2. 3D CSV with X, Y, Z
            csv_path = Path(tmpdir) / "centerline_3d.csv"
            np.savetxt(csv_path, verts_3d, delimiter=",", header="X,Y,Z", comments="#")
            loaded_csv = load_centerline_from_file(csv_path)
            self.assertEqual(loaded_csv.shape, (3, 2))
            np.testing.assert_allclose(loaded_csv, verts_3d[:, :2])

            # 3. 3D GeoJSON with LineString coordinates [X, Y, Z]
            import json
            geojson_path = Path(tmpdir) / "centerline_3d.geojson"
            gj_data = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[10.0, 20.0, 150.0], [20.0, 40.0, 149.0], [30.0, 60.0, 148.0]],
                    },
                    "properties": {"name": "River Test"}
                }]
            }
            with open(geojson_path, "w") as f:
                json.dump(gj_data, f)
            loaded_gj = load_centerline_from_file(geojson_path)
            self.assertEqual(loaded_gj.shape, (3, 2))
            np.testing.assert_allclose(loaded_gj, verts_3d[:, :2])

    def test_custom_station_and_guided_revisit(self):
        from centerline_wsm import insert_custom_station

        # Test project_point_to_station on straight line from (0,0) to (0, 100)
        straight = RiverCenterline(np.array([[0.0, 0.0], [0.0, 100.0]]))
        s_proj = straight.project_point_to_station(5.0, 42.0)
        self.assertAlmostEqual(s_proj, 42.0, places=3)

        # Test insert_custom_station
        stations = np.array([0.0, 20.0, 40.0])
        levels = np.array([100.0, 99.0, 98.0])
        locked = np.array([True, False, True])
        cached = [None, None, None]

        # Insert at 10.0m with Z = 99.8m
        st_new, lv_new, lk_new, ca_new, idx = insert_custom_station(
            stations, levels, locked, cached, new_s=10.0, new_z=99.8
        )
        self.assertEqual(idx, 1)
        self.assertEqual(len(st_new), 4)
        self.assertEqual(st_new[1], 10.0)
        self.assertEqual(lv_new[1], 99.8)
        self.assertTrue(lk_new[1])

        # Test guided revisit: a flat reservoir pool above a weir
        # Station 0: 105.0m (locked anchor)
        # Station 40: 100.0m (locked anchor below weir)
        # In between at Station 20: point cloud actually has a flat pool at 104.95m!
        off_l = np.linspace(-15, -5, 12)
        z_l = np.linspace(106, 104.95, 12)
        st_l = np.ones(12)

        off_r = np.linspace(5, 15, 12)
        z_r = np.linspace(104.95, 106, 12)
        st_r = np.ones(12)

        off_b = np.linspace(-4, 4, 8)
        z_b = np.full(8, 104.95)
        st_b = np.full(8, 2)

        sec_pool = {
            "offset": np.concatenate([off_l, off_b, off_r]),
            "z": np.concatenate([z_l, z_b, z_r]),
            "sensor_type": np.concatenate([st_l, st_b, st_r]),
            "classification": None,
        }
        st_test = np.array([0.0, 20.0, 40.0])
        lv_test = np.array([105.0, 102.5, 100.0]) # naive linear guess is 102.5m
        lk_test = np.array([True, False, True])
        ca_test = [None, sec_pool, None]

        revisited = interpolate_anchor_sections(st_test, lv_test, lk_test, cached_sections=ca_test)
        # Section 20 should snap to the actual reservoir pool at ~104.95m, NOT the naive average 102.5m!
        self.assertAlmostEqual(revisited[1], 104.95, delta=0.15)
        self.assertEqual(revisited[0], 105.0)
        self.assertEqual(revisited[2], 100.0)

if __name__ == "__main__":
    unittest.main()
