"""
test_osm_and_gui_features.py - Unit tests for OSM basemap, WSM 2D Map, LiDAR color modes, and manual editing brush/rect sizing.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QSplitter

from lidar_workbench.gui.osm_basemap import (
    fetch_osm_tile,
    fetch_osm_mosaic,
    OSMBasemapWorker,
)
from lidar_workbench.gui.point_qc_layers import (
    LayerGroup,
    OSMBasemapLayer,
    load_layer_workspace,
    save_layer_workspace,
)
from lidar_workbench.gui.view_profile import ViewProfile
from lidar_workbench.gui.view_dtm import ViewDTM
from lidar_workbench.gui.multi_view_widget import MultiViewWidget
from lidar_workbench.gui.water_surface_dialog import WaterSurfaceDialog, _WSM2DMapView
from centerline_wsm import RiverCenterline


class TestOSMAndGuiFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_osm_basemap_layer_serialization(self):
        """Test OSMBasemapLayer serialization and deserialization."""
        osm_layer = OSMBasemapLayer(name="OpenStreetMap", visible=True, opacity=0.85, crs="EPSG:25833")
        self.assertEqual(osm_layer.opacity, 0.85)

        data = osm_layer.to_dict()
        self.assertEqual(data["type"], "basemap_osm")
        self.assertEqual(data["opacity"], 0.85)
        self.assertEqual(data["crs"], "EPSG:25833")

        restored = OSMBasemapLayer.from_dict(data)
        self.assertEqual(restored.name, "OpenStreetMap")
        self.assertEqual(restored.opacity, 0.85)
        self.assertEqual(restored.crs, "EPSG:25833")
        self.assertTrue(restored.visible)

    def test_osm_tile_cache(self):
        """Test local cache read for OSM tiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cdir = Path(tmpdir)
            tile_path = cdir / "14" / "8852" / "5634.png"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_img = Image.new("RGBA", (256, 256), (100, 150, 200, 255))
            dummy_img.save(tile_path)

            loaded = fetch_osm_tile(14, 8852, 5634, cache_dir=cdir)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.size, (256, 256))

    def test_profile_and_dtm_color_modes(self):
        """Test color modes on ViewProfile and ViewDTM."""
        prof = ViewProfile()
        prof.resize(800, 400)
        dtm = ViewDTM()

        # Generate sample points
        n = 100
        x = np.linspace(0, 50, n)
        y = np.sin(x)
        z = np.linspace(100, 105, n)
        cls = np.random.choice([2, 6, 9], n)
        intensity = np.random.uniform(10, 255, n)
        return_num = np.random.choice([1, 2, 3], n)
        flightline = np.random.choice([1, 2], n)

        prof.set_profile_data(
            distances=x, elevations=z, classifications=cls,
            intensities=intensity,
            return_numbers=return_num,
            point_source_ids=flightline,
        )

        # ProfileView width adjusting should be False by default
        self.assertFalse(prof._width_adjusting)
        self.assertTrue(prof._base_dirty)

        # Trigger base pixmap rendering
        prof._render_base_pixmap()
        self.assertFalse(prof._base_dirty)
        self.assertIsNotNone(prof._base_pixmap)

        # Render profile view to test paintEvent including HUD overlay
        pixmap = QPixmap(800, 400)
        prof.render(pixmap)
        self.assertFalse(pixmap.isNull())

        modes = ["class", "height", "intensity", "return_number", "flightline"]
        for m in modes:
            prof.set_colour_mode(m)
            self.assertEqual(prof._colour_mode, m)

        # Test brush and rect sizes on ViewProfile
        prof.set_brush_radius(1.5)
        self.assertAlmostEqual(prof._brush_radius, 1.5)
        prof.set_rect_size(2.0, 1.0)
        self.assertAlmostEqual(prof._rect_width, 2.0)
        self.assertAlmostEqual(prof._rect_height, 1.0)

        # DTM View tests
        dtm.load_points({
            "x": x, "y": y, "z": z,
            "classification": cls,
            "intensity": intensity,
            "return_number": return_num,
            "point_source_id": flightline,
        })
        for m in modes:
            dtm.set_colour_mode(m)
            self.assertEqual(dtm._colour_mode, m)

        prof.close()
        dtm.close()

    def test_multi_view_widget_controls(self):
        """Test that MultiViewWidget exposes radius, w, h, corridor spinboxes and syncs them."""
        with patch("lidar_workbench.gui.view_3d.View3D._init_renderer"):
            mv = MultiViewWidget()
            self.assertTrue(hasattr(mv, "_brush_size_spin"))
            self.assertTrue(hasattr(mv, "_rect_w_spin"))
            self.assertTrue(hasattr(mv, "_rect_h_spin"))
            self.assertTrue(hasattr(mv, "_corridor_spin"))

            # Changing spinboxes updates profile view
            mv._brush_size_spin.setValue(0.75)
            self.assertAlmostEqual(mv._view_profile._brush_radius, 0.75)

            mv._rect_w_spin.setValue(0.8)
            mv._rect_h_spin.setValue(0.4)
            self.assertAlmostEqual(mv._view_profile._rect_width, 0.4)
            self.assertAlmostEqual(mv._view_profile._rect_height, 0.2)

            # Changing corridor width spinbox
            mv._corridor_spin.setValue(3.5)
            self.assertAlmostEqual(mv._view_profile._total_width, 3.5)

            # Test colour mode selector
            mv._colour_combo.setCurrentIndex(mv._colour_combo.findData("flightline"))
            self.assertEqual(mv._view_profile._colour_mode, "flightline")
            self.assertEqual(mv._view_dtm._colour_mode, "flightline")

            mv.close()

    def test_wsm_2d_map_and_buttons(self):
        """Test WSM 2D Map View, Intensity raster, and vertical layout above profile view."""
        t = np.linspace(0, 1, 30)
        x = 50.0 * np.sin(t * np.pi)
        y = t * 150.0
        centerline = RiverCenterline(np.column_stack((x, y)))

        map_view = _WSM2DMapView(centerline=centerline, crs_epsg=25833)
        self.assertIsNotNone(map_view)
        self.assertTrue(hasattr(map_view, "_raster_item"))
        self.assertTrue(hasattr(map_view, "_raster_combo"))

        # Feed points with intensity
        n = 5000
        px = np.random.uniform(0, 100, n)
        py = np.random.uniform(0, 150, n)
        pz = np.random.uniform(98, 102, n)
        p_int = np.random.uniform(10, 255, n)
        map_view.set_channel_points(px, py, pz, intensities=p_int)

        # Verify raster item is shown
        self.assertTrue(map_view._raster_item.isVisible())
        self.assertIsNotNone(map_view._raster_item.image)

        # Switch to elevation mode
        idx_elev = map_view._raster_combo.findData("elevation_0.5")
        self.assertGreaterEqual(idx_elev, 0)
        map_view._raster_combo.setCurrentIndex(idx_elev)
        self.assertTrue(map_view._raster_item.isVisible())

        # Update cross-sections
        stations = np.array([0.0, 30.0, 60.0])
        locked = np.array([True, False, True])
        map_view.set_sections(centerline, stations, locked, corridor_width=20.0)
        self.assertIsNotNone(map_view._sections_curve_normal.xData)

        # Create dialog (without parent or full run) to verify button labels & vertical splitter
        dlg = WaterSurfaceDialog(
            project_dir=tempfile.gettempdir(),
            load_points_func=lambda: None,
            data_epsg=25833,
        )
        dlg._full_centerline = centerline
        self.assertEqual(dlg._save_btn.text(), "💾 Save All Profiles…")
        self.assertEqual(dlg._load_btn.text(), "📂 Load All Profiles…")
        self.assertTrue(dlg._map_2d is not None)

        # Find the splitter holding _map_2d and verify vertical orientation (2D map above profile view)
        splitters = dlg.findChildren(QSplitter)
        wsm_splitter = None
        for s in splitters:
            if s.indexOf(dlg._map_2d) >= 0:
                wsm_splitter = s
                break
        self.assertIsNotNone(wsm_splitter)
        self.assertEqual(wsm_splitter.orientation(), Qt.Vertical)
        self.assertEqual(wsm_splitter.indexOf(dlg._map_2d), 0)  # 2D map at index 0 (top)

        dlg.close()
        map_view.close()


if __name__ == "__main__":
    unittest.main()
