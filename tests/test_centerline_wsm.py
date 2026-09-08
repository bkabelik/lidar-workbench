import unittest
import numpy as np
import tempfile
from pathlib import Path

from centerline_wsm import (
    RiverCenterline,
    clip_polyline_to_bbox,
    clip_segment_to_bbox,
    detect_embankment_extents,
    detect_section_water_level,
    drape_centerline_water_surface,
    enforce_downstream_monotonicity,
    interpolate_anchor_sections,
    polyline_length_in_bbox,
    project_points_to_centerline,
    rasterize_water_surface_model,
    remove_station,
    slice_cross_section_points,
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

    def test_polyline_clipping_and_bbox_length(self):
        # Segment crossing box
        clipped = clip_segment_to_bbox([0.0, 5.0], [10.0, 5.0], 2.0, 0.0, 8.0, 10.0)
        self.assertIsNotNone(clipped)
        c0, c1 = clipped
        self.assertTrue(np.allclose(c0, [2.0, 5.0]))
        self.assertTrue(np.allclose(c1, [8.0, 5.0]))

        # Segment outside box
        self.assertIsNone(clip_segment_to_bbox([0.0, 15.0], [10.0, 15.0], 2.0, 0.0, 8.0, 10.0))

        # Polyline length in bbox
        v = np.array([[0.0, 5.0], [5.0, 5.0], [10.0, 5.0]])
        l = polyline_length_in_bbox(v, (2.0, 0.0, 8.0, 10.0))
        self.assertAlmostEqual(l, 6.0)

        # Polyline clipping
        chains = clip_polyline_to_bbox(v, (2.0, 0.0, 8.0, 10.0), buffer_m=0.0)
        self.assertEqual(len(chains), 1)
        self.assertTrue(np.allclose(chains[0][0], [2.0, 5.0]))
        self.assertTrue(np.allclose(chains[0][-1], [8.0, 5.0]))

        # S-curve centerline clipping
        sc_chains = clip_polyline_to_bbox(self.verts, (0.0, 50.0, 60.0, 150.0), buffer_m=0.0)
        self.assertGreaterEqual(len(sc_chains), 1)
        for ch in sc_chains:
            self.assertTrue(np.all(ch[:, 0] >= -1e-3))
            self.assertTrue(np.all(ch[:, 0] <= 60.0 + 1e-3))
            self.assertTrue(np.all(ch[:, 1] >= 50.0 - 1e-3))
            self.assertTrue(np.all(ch[:, 1] <= 150.0 + 1e-3))

    def test_drape_centerline_water_surface(self):
        # Create 5 stations along centerline with:
        # - High ground banks (Z=120m) at offset +/- 10m to 20m
        # - High tree canopy (Z=135m) at offset 0m
        # - Riverbed at Z=97.0 - 0.2*i
        # - Water surface at Z=98.5 - 0.2*i
        # - Station 2 has an artificial high bridge / canopy spike
        stations = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        cached_sections = []
        for i, s in enumerate(stations):
            # Bank ground points at offset +/- 15m, Z=120m
            off_bank = np.array([-18.0, -15.0, -12.0, 12.0, 15.0, 18.0])
            z_bank = np.full(6, 120.0)

            # High tree canopy directly above river
            off_tree = np.array([-0.5, 0.0, 0.5])
            z_tree = np.full(3, 135.0)

            true_surf = 98.5 - 0.2 * i
            true_bed = 97.0 - 0.2 * i

            if i == 2:
                # Spike: dense canopy where laser did not reach water
                off_chan = np.array([-0.2, 0.0, 0.2])
                z_chan = np.full(3, 130.0)
            else:
                # In channel: riverbed points (97m) and water surface points (98.5m)
                off_chan = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, -0.8, -0.2, 0.3, 0.9])
                z_chan = np.array([
                    true_bed, true_bed + 0.1, true_bed,
                    true_surf - 0.1, true_surf, true_surf - 0.05,
                    true_surf, true_surf - 0.15, true_surf,
                ])

            all_off = np.concatenate([off_bank, off_tree, off_chan])
            all_z = np.concatenate([z_bank, z_tree, z_chan])
            cached_sections.append({"offset": all_off, "z": all_z})

        levels = drape_centerline_water_surface(
            self.centerline,
            stations,
            cached_sections=cached_sections,
            center_strip_width=1.5,
            max_water_depth=3.5,
            outlier_threshold_m=0.60,
            median_window=3,
        )

        self.assertEqual(len(levels), 5)
        # Should be strictly monotonic downhill
        self.assertTrue((np.diff(levels) <= 1e-9).all())
        # Should ignore the 120m banks and 135m canopy completely
        self.assertLess(float(levels.max()), 100.0)
        self.assertGreater(float(levels.min()), 97.0)
        # Station 0 should be around 98.5
        self.assertAlmostEqual(levels[0], 98.5, delta=0.25)
        # Station 2 spike should be filtered out by median filter
        self.assertAlmostEqual(levels[2], 98.1, delta=0.35)

    def test_centerline_trim(self):
        # Original centerline total length is > 200m
        orig_len = self.centerline.total_length
        self.assertGreater(orig_len, 200.0)

        # Trim reach [50m, 150m]
        trimmed = self.centerline.trim(50.0, 150.0)
        self.assertAlmostEqual(trimmed.total_length, 100.0, places=1)

        # Station 0 on trimmed centerline must match station 50 on original
        pos_orig_50, tan_orig_50, _ = self.centerline.evaluate(50.0)
        pos_trim_0, tan_trim_0, _ = trimmed.evaluate(0.0)
        self.assertTrue(np.allclose(pos_orig_50, pos_trim_0, atol=1e-3))
        self.assertTrue(np.allclose(tan_orig_50, tan_trim_0, atol=1e-3))

        # Station 100 on trimmed centerline must match station 150 on original
        pos_orig_150, _, _ = self.centerline.evaluate(150.0)
        pos_trim_end, _, _ = trimmed.evaluate(100.0)
        self.assertTrue(np.allclose(pos_orig_150, pos_trim_end, atol=1e-3))

    def test_project_points_to_centerline(self):
        # Sample points along centerline at known stations s=20, 60, 110, with small transverse offsets
        test_stations = [20.0, 60.0, 110.0]
        test_xs = []
        test_ys = []
        for s in test_stations:
            p, _, n = self.centerline.evaluate(s)
            # Offset by 2m in normal direction
            pt = p + 2.0 * n
            test_xs.append(pt[0])
            test_ys.append(pt[1])

        proj_s = project_points_to_centerline(self.centerline, np.array(test_xs), np.array(test_ys))
        self.assertEqual(len(proj_s), 3)
        for expected, actual in zip(test_stations, proj_s):
            self.assertAlmostEqual(expected, actual, delta=0.5)

    def test_topo_priority_over_bathy_water_surface(self):
        # Station with both bathymetric green returns (submerged, bed=96.0m)
        # and Topo NIR returns (reflecting on surface=99.5m)
        offsets = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, -0.8, -0.2, 0.3, 0.9])
        # Green bathy points at bed
        bathy_mask = np.array([True, True, True, False, False, False, True, True, False])
        # Topo NIR points at surface
        topo_mask = ~bathy_mask

        z = np.empty(len(offsets), dtype=np.float64)
        z[bathy_mask] = np.random.uniform(95.8, 96.2, bathy_mask.sum())
        z[topo_mask] = np.random.uniform(99.4, 99.6, topo_mask.sum())

        st = np.ones(len(offsets), dtype=np.int32)  # NIR=1
        st[bathy_mask] = 2  # Bathy=2

        # 1. detect_section_water_level should pick the Topo water surface (~99.5m), NOT bed (~96.0m)
        w_z, _, _ = detect_section_water_level(offsets, z, sensor_types=st)
        self.assertAlmostEqual(w_z, 99.5, delta=0.20)

        # 2. drape_centerline_water_surface should also pick Topo (~99.5m)
        cached_sections = [{"offset": offsets, "z": z, "sensor_type": st}]
        stations = np.array([50.0])
        levels = drape_centerline_water_surface(
            self.centerline,
            stations,
            cached_sections=cached_sections,
            center_strip_width=2.0,
            max_water_depth=5.0,
        )
        self.assertAlmostEqual(levels[0], 99.5, delta=0.20)

        # 3. If only bathy returns exist (no topo returns in channel), it should fall back to bathy
        st_bathy_only = np.full(len(offsets), 2, dtype=np.int32)
        cached_bathy_only = [{"offset": offsets, "z": z, "sensor_type": st_bathy_only}]
        levels_fallback = drape_centerline_water_surface(
            self.centerline,
            stations,
            cached_sections=cached_bathy_only,
            center_strip_width=2.0,
            max_water_depth=5.0,
        )
        # In this case it uses the upper envelope of bathy
        self.assertAlmostEqual(levels_fallback[0], float(np.percentile(z, 85.0)), delta=0.20)

    def test_remove_station(self):
        stations = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        levels = np.array([100.0, 99.8, 99.5, 99.2, 99.0])
        locked = np.array([True, False, False, True, False])
        cached = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]

        # Remove section at index 2 (station 20.0m - e.g. under a bridge)
        up_s, up_l, up_lk, up_c, new_idx = remove_station(stations, levels, locked, cached, idx=2)

        self.assertEqual(len(up_s), 4)
        self.assertEqual(len(up_l), 4)
        self.assertEqual(len(up_lk), 4)
        self.assertEqual(len(up_c), 4)
        # Verify 20.0m was removed and station 30.0m is now at index 2
        np.testing.assert_array_equal(up_s, [0.0, 10.0, 30.0, 40.0])
        np.testing.assert_array_equal(up_l, [100.0, 99.8, 99.2, 99.0])
        np.testing.assert_array_equal(up_lk, [True, False, True, False])
        self.assertEqual(up_c[2]["id"], 3)
        self.assertEqual(new_idx, 2)

        # Remove the last station (index 3 of remaining 4)
        up_s2, up_l2, up_lk2, up_c2, new_idx2 = remove_station(up_s, up_l, up_lk, up_c, idx=3)
        self.assertEqual(len(up_s2), 3)
        np.testing.assert_array_equal(up_s2, [0.0, 10.0, 30.0])
        # new_idx2 should clamp to 2 (last valid index)
        self.assertEqual(new_idx2, 2)

        # Removing when only 1 section remains should raise ValueError
        single_s = np.array([10.0])
        single_l = np.array([99.0])
        single_lk = np.array([True])
        single_c = [{"id": 0}]
        with self.assertRaises(ValueError):
            remove_station(single_s, single_l, single_lk, single_c, idx=0)

    def test_detect_embankment_extents(self):
        # Channel from -20m to +20m
        # River bed at -5m to +5m, elevation 98.0m
        # Water surface at 100.0m
        # Embankments rise linearly from +/- 5m (elevation 98m) to +/- 10m (elevation 102m)
        # Touch-point with water surface (100m) is exactly at +/- 7.5m
        offsets = np.linspace(-20.0, 20.0, 81)
        elevations = np.zeros_like(offsets)
        for i, u in enumerate(offsets):
            if abs(u) <= 5.0:
                elevations[i] = 98.0
            else:
                # rise 0.8m per meter: at 7.5m, 98 + 0.8*(7.5 - 5) = 100.0m
                elevations[i] = 98.0 + 0.8 * (abs(u) - 5.0)

        # Margin = 2.0m: left_ext should be -7.5 - 2.0 = -9.5m, right_ext should be 7.5 + 2.0 = 9.5m
        l_ext, r_ext = detect_embankment_extents(
            offsets, elevations, water_z=100.0, corridor_width=40.0, margin=2.0
        )
        self.assertAlmostEqual(l_ext, -9.5, delta=0.5)
        self.assertAlmostEqual(r_ext, 9.5, delta=0.5)

        # Flat bathy bed across full corridor: clamps to corridor bounds
        flat_elevations = np.full_like(offsets, 98.0)
        l_flat, r_flat = detect_embankment_extents(
            offsets, flat_elevations, water_z=100.0, corridor_width=40.0, margin=2.0
        )
        self.assertEqual(l_flat, -20.0)
        self.assertEqual(r_flat, 20.0)

        # Dry land (all points above water surface): falls back to corridor bounds
        dry_elevations = np.full_like(offsets, 105.0)
        l_dry, r_dry = detect_embankment_extents(
            offsets, dry_elevations, water_z=100.0, corridor_width=40.0, margin=2.0
        )
        self.assertEqual(l_dry, -20.0)
        self.assertEqual(r_dry, 20.0)

    def test_rasterization_with_embankment_and_tilt(self):
        stations = np.array([0.0, 20.0, 40.0])
        w_levels = np.array([100.0, 99.0, 98.0])
        # Trimmed bank offsets: -8m to +8m instead of full 30m corridor (-15m to +15m)
        left_offsets = np.array([-8.0, -8.0, -8.0])
        right_offsets = np.array([8.0, 8.0, 8.0])
        # Add a 0.4m cross-stream tilt: left bank is 0.2m higher, right bank is 0.2m lower
        z_left = w_levels + 0.20
        z_right = w_levels - 0.20

        with tempfile.TemporaryDirectory() as tmpdir:
            out_tif = Path(tmpdir) / "test_wsm_tilt.tif"
            surf, georef = rasterize_water_surface_model(
                self.centerline, stations, w_levels,
                corridor_width=30.0, output_path=out_tif, resolution=1.0, data_epsg=25832,
                left_offsets=left_offsets,
                right_offsets=right_offsets,
                water_levels_left=z_left,
                water_levels_right=z_right,
            )
            self.assertTrue(out_tif.is_file())
            valid = surf[~np.isnan(surf)]
            self.assertGreater(len(valid), 0)
            # Max elevation should reach ~100.2m, min elevation should reach ~97.8m
            self.assertAlmostEqual(float(np.max(valid)), 100.2, delta=0.15)
            self.assertAlmostEqual(float(np.min(valid)), 97.8, delta=0.15)


if __name__ == "__main__":
    unittest.main()
