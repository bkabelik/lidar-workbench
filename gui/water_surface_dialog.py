"""
LiDAR Workbench — River Water Surface Model (WSM) Generator Dialog.

Provides a semi-automatic, human-in-the-loop workflow:
  1. Load or draw a 2D river centerline polyline
  2. Automatically cut orthogonal cross-sections every N metres
  3. Detect water levels from NIR shoreline voids and green bathy returns
  4. Interactively review sections (step, jump every 10th section, drag water line)
  5. Lock key anchors and smoothly interpolate intermediate sections
  6. Export a 3D horizontal water surface model as a GeoTIFF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..centerline_wsm import (
    RiverCenterline,
    detect_section_water_level,
    enforce_downstream_monotonicity,
    interpolate_anchor_sections,
    load_centerline_from_file,
    rasterize_water_surface_model,
    slice_cross_section_points,
)

logger = logging.getLogger("lidar_workbench.gui.water_surface_dialog")


class _CrossSectionCanvas(QWidget):
    """
    2D interactive side-view canvas for a single river cross-section.

    Plots transverse offset [-W/2, +W/2] vs Elevation (Z).
    Draws a horizontal water surface line that can be dragged up/down with the mouse.
    """

    water_level_changed = Signal(float)  # emitted when user drags or edits the water line

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(400, 260)
        self.setMouseTracking(True)

        self._offsets: Optional[np.ndarray] = None
        self._elevations: Optional[np.ndarray] = None
        self._sensor_types: Optional[np.ndarray] = None
        self._classes: Optional[np.ndarray] = None

        self._corridor_width: float = 40.0
        self._water_z: float = 100.0
        self._is_locked: bool = False

        # View transform
        self._offset_x: float = 0.0
        self._offset_z: float = 0.0
        self._scale_x: float = 8.0  # pixels per metre
        self._scale_z: float = 25.0

        # Interaction
        self._dragging_water_line: bool = False
        self._hovering_water_line: bool = False
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None

    def set_data(
        self,
        offsets: np.ndarray,
        elevations: np.ndarray,
        sensor_types: Optional[np.ndarray],
        classes: Optional[np.ndarray],
        corridor_width: float,
        water_z: float,
        is_locked: bool = False,
    ):
        self._offsets = offsets
        self._elevations = elevations
        self._sensor_types = sensor_types
        self._classes = classes
        self._corridor_width = max(5.0, float(corridor_width))
        self._water_z = float(water_z)
        self._is_locked = bool(is_locked)

        self.fit_view()
        self.update()

    def set_water_level(self, z: float, is_locked: Optional[bool] = None):
        self._water_z = float(z)
        if is_locked is not None:
            self._is_locked = is_locked
        self.update()

    def fit_view(self):
        w = max(100, self.width())
        h = max(100, self.height())

        margin_x = 40.0
        margin_y = 30.0

        half_w = self._corridor_width * 0.5
        range_x = self._corridor_width * 1.1

        if self._elevations is not None and len(self._elevations) > 0:
            min_z = float(np.min(self._elevations))
            max_z = float(np.max(self._elevations))
            if not np.isnan(self._water_z):
                min_z = min(min_z, self._water_z - 1.0)
                max_z = max(max_z, self._water_z + 1.0)
        else:
            min_z = self._water_z - 3.0
            max_z = self._water_z + 3.0

        range_z = max(1.5, max_z - min_z) * 1.2
        mid_z = (min_z + max_z) * 0.5

        self._scale_x = (w - margin_x * 2.0) / max(1.0, range_x)
        self._scale_z = (h - margin_y * 2.0) / max(1.0, range_z)

        # Center in widget
        self._offset_x = (w * 0.5)
        self._offset_z = (h * 0.5) + mid_z * self._scale_z

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_view()

    def _world_to_widget(self, off: float, z: float) -> Tuple[float, float]:
        px = self._offset_x + off * self._scale_x
        py = self._offset_z - z * self._scale_z
        return px, py

    def _widget_to_world(self, px: float, py: float) -> Tuple[float, float]:
        off = (px - self._offset_x) / self._scale_x
        z = (self._offset_z - py) / self._scale_z
        return off, z

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#161a22"))

        # Grid lines & centerline marker
        painter.setPen(QPen(QColor("#2d333b"), 1, Qt.DashLine))
        cx, _ = self._world_to_widget(0.0, 0.0)
        painter.drawLine(int(cx), 0, int(cx), h)

        # Labels
        painter.setPen(QColor("#768390"))
        painter.setFont(QFont("sans-serif", 9))
        painter.drawText(int(cx) + 5, 20, "Centerline (0.0 m)")

        # Bank limits
        half_w = self._corridor_width * 0.5
        lx, _ = self._world_to_widget(-half_w, 0.0)
        rx, _ = self._world_to_widget(half_w, 0.0)
        painter.setPen(QPen(QColor("#3d444d"), 1, Qt.DotLine))
        painter.drawLine(int(lx), 0, int(lx), h)
        painter.drawLine(int(rx), 0, int(rx), h)
        painter.drawText(int(lx) + 5, h - 10, f"-{half_w:.1f} m (Left)")
        painter.drawText(int(rx) - 75, h - 10, f"+{half_w:.1f} m (Right)")

        # Render points
        if self._offsets is not None and len(self._offsets) > 0:
            is_bathy = (self._sensor_types == 2) if self._sensor_types is not None else np.zeros(len(self._offsets), dtype=bool)

            for i in range(len(self._offsets)):
                off = self._offsets[i]
                z = self._elevations[i]
                px, py = self._world_to_widget(off, z)

                if is_bathy[i]:
                    painter.setPen(QColor("#26e6e6"))  # Cyan for bathy
                    painter.setBrush(QColor("#26e6e6"))
                    pt_size = 3
                else:
                    painter.setPen(QColor("#d29922"))  # Amber for NIR / topo
                    painter.setBrush(QColor("#d29922"))
                    pt_size = 3

                painter.drawEllipse(QPointF(px, py), pt_size, pt_size)

        # Draw Horizontal Water Surface Line
        if not np.isnan(self._water_z):
            _, wy = self._world_to_widget(0.0, self._water_z)
            water_py = int(round(wy))

            line_color = QColor("#d29922") if self._is_locked else QColor("#58a6ff")
            pen_width = 3 if (self._hovering_water_line or self._dragging_water_line) else 2
            painter.setPen(QPen(line_color, pen_width, Qt.SolidLine))

            # Draw across corridor
            painter.drawLine(int(lx), water_py, int(rx), water_py)

            # Draw draggable handle in the middle
            handle_radius = 6 if (self._hovering_water_line or self._dragging_water_line) else 5
            painter.setBrush(line_color)
            painter.drawEllipse(QPointF(cx, water_py), handle_radius, handle_radius)

            # Draw Water Level Tag
            status_tag = " [LOCKED ANCHOR]" if self._is_locked else " [Auto]"
            tag_text = f"Water Z: {self._water_z:.3f} m{status_tag}"
            painter.setFont(QFont("sans-serif", 10, QFont.Bold))
            painter.setPen(line_color)
            painter.drawText(int(lx) + 10, water_py - 8, tag_text)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            _, wy = self._world_to_widget(0.0, self._water_z)
            if abs(event.position().y() - wy) <= 12:
                self._dragging_water_line = True
                self.setCursor(Qt.SizeVerCursor)
                return
            else:
                # Direct click-to-place: click anywhere on the section to place the horizontal water line!
                _, new_z = self._widget_to_world(event.position().x(), event.position().y())
                self._water_z = round(new_z, 3)
                self._is_locked = True
                self._dragging_water_line = True
                self.setCursor(Qt.SizeVerCursor)
                self.water_level_changed.emit(self._water_z)
                self.update()
                return
        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging_water_line:
            _, new_z = self._widget_to_world(event.position().x(), event.position().y())
            self._water_z = round(new_z, 3)
            self._is_locked = True
            self.water_level_changed.emit(self._water_z)
            self.update()
            return
        elif self._panning and self._pan_start is not None:
            dy = event.position().y() - self._pan_start.y()
            self._offset_z += dy
            self._pan_start = event.position()
            self.update()
            return

        # Check hover on water line
        if not np.isnan(self._water_z):
            _, wy = self._world_to_widget(0.0, self._water_z)
            near = abs(event.position().y() - wy) <= 12
            if near != self._hovering_water_line:
                self._hovering_water_line = near
                self.setCursor(Qt.SizeVerCursor if near else Qt.ArrowCursor)
                self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._dragging_water_line:
            self._dragging_water_line = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        # Vertical zoom on mouse wheel
        delta = event.angleDelta().y()
        zoom = 1.15 if delta > 0 else 0.85
        self._scale_z = max(5.0, min(200.0, self._scale_z * zoom))
        self.update()


class _StationRibbon(QWidget):
    """Horizontal colored ribbon displaying all cross-section stations."""

    station_clicked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setMouseTracking(True)
        self._n_stations: int = 0
        self._current_station: int = 0
        self._locked_mask: Optional[np.ndarray] = None

    def set_data(self, n_stations: int, current_station: int, locked_mask: np.ndarray):
        self._n_stations = max(0, n_stations)
        self._current_station = current_station
        self._locked_mask = locked_mask
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#1c2128"))

        if self._n_stations <= 0:
            painter.end()
            return

        tick_w = w / self._n_stations

        for i in range(self._n_stations):
            x = i * tick_w
            is_cur = (i == self._current_station)
            is_locked = bool(self._locked_mask[i]) if self._locked_mask is not None else False

            if is_cur:
                col = QColor("#ffffff")
            elif is_locked:
                col = QColor("#d29922")  # Gold for user-approved anchor
            else:
                col = QColor("#58a6ff")  # Cyan for auto

            rect = QRectF(x, 4, max(2.0, tick_w - 1.0), h - 8)
            painter.fillRect(rect, col)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._n_stations > 0:
            idx = int(event.position().x() / (self.width() / self._n_stations))
            idx = max(0, min(self._n_stations - 1, idx))
            self.station_clicked.emit(idx)


class WaterSurfaceDialog(QDialog):
    """
    Interactive River Centerline Water Surface Model Generator.
    """

    wsm_generated = Signal(str)  # emitted when GeoTIFF is saved, passes file path

    def __init__(
        self,
        project_dir: str | Path,
        load_points_func,
        data_epsg: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._project_dir = Path(project_dir)
        self._load_points_func = load_points_func
        self._data_epsg = data_epsg

        self.setWindowTitle("River Water Surface Model Generator (Centerline)")
        self.setMinimumSize(880, 680)

        # Centerline and cross-section state
        self._centerline: Optional[RiverCenterline] = None
        self._stations: np.ndarray = np.array([])
        self._water_levels: np.ndarray = np.array([])
        self._locked_mask: np.ndarray = np.array([], dtype=bool)
        self._cached_sections: List[Optional[dict]] = []

        self._curr_idx: int = 0
        self._corridor_width: float = 40.0
        self._section_spacing: float = 10.0

        # Point cloud cache
        self._all_xs: Optional[np.ndarray] = None
        self._all_ys: Optional[np.ndarray] = None
        self._all_zs: Optional[np.ndarray] = None
        self._all_st: Optional[np.ndarray] = None
        self._all_cls: Optional[np.ndarray] = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── 1. Setup Header ──
        setup_group = QGroupBox("1. River Centerline & Corridor Setup")
        sf = QFormLayout(setup_group)

        centerline_row = QHBoxLayout()
        self._centerline_path_edit = QLineEdit()
        self._centerline_path_edit.setPlaceholderText("Select centerline polyline (Shapefile, GeoJSON, CSV)...")
        self._centerline_path_edit.setReadOnly(True)
        centerline_row.addWidget(self._centerline_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_centerline)
        centerline_row.addWidget(browse_btn)

        self._osm_btn = QPushButton("🌐 Fetch River from OSM…")
        self._osm_btn.setToolTip("Automatically query OpenStreetMap / WaterwayMap for rivers in this tile's bounding box")
        self._osm_btn.setStyleSheet("font-weight: bold; background-color: #1f6feb; color: white;")
        self._osm_btn.clicked.connect(self._on_fetch_osm_waterways)
        centerline_row.addWidget(self._osm_btn)

        sf.addRow("Centerline:", centerline_row)

        self._river_combo = QComboBox()
        self._river_combo.setVisible(False)
        self._river_combo.currentIndexChanged.connect(self._on_river_combo_selected)
        sf.addRow("OSM Rivers:", self._river_combo)

        params_row = QHBoxLayout()
        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(5.0, 500.0)
        self._width_spin.setValue(40.0)
        self._width_spin.setSuffix(" m")
        params_row.addWidget(QLabel("Corridor Width:"))
        params_row.addWidget(self._width_spin)

        self._spacing_spin = QDoubleSpinBox()
        self._spacing_spin.setRange(2.0, 200.0)
        self._spacing_spin.setValue(10.0)
        self._spacing_spin.setSuffix(" m")
        params_row.addWidget(QLabel("Section Spacing:"))
        params_row.addWidget(self._spacing_spin)

        self._gen_btn = QPushButton("▶ Generate Cross-Sections")
        self._gen_btn.setStyleSheet("font-weight: bold; background-color: #238636; color: white; padding: 4px 12px;")
        self._gen_btn.clicked.connect(self._on_generate_sections)
        params_row.addWidget(self._gen_btn)

        sf.addRow(params_row)
        layout.addWidget(setup_group)

        # ── 2. Interactive Cross-Section Reviewer ──
        review_group = QGroupBox("2. Cross-Section Review & Anchor Adjustment")
        rf = QVBoxLayout(review_group)

        # Ribbon showing all stations
        self._ribbon = _StationRibbon()
        self._ribbon.station_clicked.connect(self._go_to_station)
        rf.addWidget(self._ribbon)

        # Canvas
        self._canvas = _CrossSectionCanvas()
        self._canvas.water_level_changed.connect(self._on_canvas_water_level_changed)
        rf.addWidget(self._canvas)

        # Controls under canvas
        ctrl_layout = QHBoxLayout()

        self._btn_prev10 = QPushButton("<< Jump -10")
        self._btn_prev10.clicked.connect(lambda: self._step_station(-10))
        ctrl_layout.addWidget(self._btn_prev10)

        self._btn_prev = QPushButton("< Prev (A)")
        self._btn_prev.clicked.connect(lambda: self._step_station(-1))
        ctrl_layout.addWidget(self._btn_prev)

        self._station_label = QLabel("Station: 0.0 m (Section 0 of 0)")
        self._station_label.setAlignment(Qt.AlignCenter)
        self._station_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        ctrl_layout.addWidget(self._station_label)

        self._btn_next = QPushButton("Next > (D)")
        self._btn_next.clicked.connect(lambda: self._step_station(1))
        ctrl_layout.addWidget(self._btn_next)

        self._btn_next10 = QPushButton("Jump +10 >>")
        self._btn_next10.clicked.connect(lambda: self._step_station(10))
        ctrl_layout.addWidget(self._btn_next10)

        rf.addLayout(ctrl_layout)

        # Elevation edit & lock anchor row
        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("Water Elevation:"))

        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(-1000.0, 9000.0)
        self._z_spin.setDecimals(3)
        self._z_spin.setValue(100.0)
        self._z_spin.setSuffix(" m")
        self._z_spin.valueChanged.connect(self._on_spin_z_changed)
        edit_layout.addWidget(self._z_spin)

        self._lock_btn = QPushButton("★ Lock as Approved Anchor (Space)")
        self._lock_btn.setCheckable(True)
        self._lock_btn.clicked.connect(self._on_toggle_lock)
        edit_layout.addWidget(self._lock_btn)

        self._interp_btn = QPushButton("↺ Interpolate Intermediate Sections")
        self._interp_btn.setToolTip("Smoothly re-interpolates all auto sections between your approved anchors.")
        self._interp_btn.clicked.connect(self._on_interpolate_anchors)
        edit_layout.addWidget(self._interp_btn)

        rf.addLayout(edit_layout)

        # Custom section insertion row
        sec_mgmt_layout = QHBoxLayout()
        self._btn_add_station = QPushButton("➕ Add Section at Station…")
        self._btn_add_station.setToolTip("Insert a new cross-section at an exact station distance (e.g. at a weir, power plant, or rapid)")
        self._btn_add_station.clicked.connect(self._on_add_station_dialog)
        sec_mgmt_layout.addWidget(self._btn_add_station)

        self._btn_import_drawn = QPushButton("✏ Insert Section from Map Profile Line")
        self._btn_import_drawn.setToolTip("Drag a profile line across the river on the 2D map, then click here to add it as an anchor!")
        self._btn_import_drawn.clicked.connect(self._on_import_drawn_profile_section)
        sec_mgmt_layout.addWidget(self._btn_import_drawn)

        rf.addLayout(sec_mgmt_layout)
        layout.addWidget(review_group)

        # ── 3. Bottom Actions ──
        bottom_layout = QHBoxLayout()
        self._status_label = QLabel("Ready. Load or draw a centerline to begin.")
        bottom_layout.addWidget(self._status_label)

        self._export_btn = QPushButton("💾 Export WSM GeoTIFF & Apply to Bathymetry")
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet("font-weight: bold; background-color: #1f6feb; color: white; padding: 6px 16px;")
        self._export_btn.clicked.connect(self._on_export_geotiff)
        bottom_layout.addWidget(self._export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)

        layout.addLayout(bottom_layout)

    def set_centerline_polyline(self, vertices: np.ndarray):
        """Set centerline directly from polyline vertices (e.g. drawn on DTM)."""
        try:
            self._centerline = RiverCenterline(vertices)
            self._centerline_path_edit.setText(f"[Drawn Polyline: {len(vertices)} vertices, {self._centerline.total_length:.1f} m]")
            self._status_label.setText(f"Centerline loaded: {self._centerline.total_length:.1f} m. Click 'Generate Cross-Sections'.")
        except Exception as e:
            QMessageBox.critical(self, "Centerline Error", str(e))

    def _on_browse_centerline(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select River Centerline File",
            str(self._project_dir),
            "Polyline Files (*.shp *.geojson *.json *.csv *.txt);;All Files (*)",
        )
        if not fn:
            return
        try:
            verts = load_centerline_from_file(fn)
            self._centerline = RiverCenterline(verts)
            self._centerline_path_edit.setText(fn)
            self._status_label.setText(f"Loaded {len(verts)} vertices ({self._centerline.total_length:.1f} m).")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load centerline: {e}")

    def _on_fetch_osm_waterways(self):
        if not self._ensure_points_loaded():
            return

        if not self._data_epsg:
            from PySide6.QtWidgets import QInputDialog
            epsg, ok = QInputDialog.getInt(
                self, "Project EPSG",
                "Enter project EPSG code for reprojection (e.g. 25832, 25833, 31256):",
                25832, 1000, 99999
            )
            if not ok:
                return
            self._data_epsg = epsg

        min_x = float(np.min(self._all_xs))
        max_x = float(np.max(self._all_xs))
        min_y = float(np.min(self._all_ys))
        max_y = float(np.max(self._all_ys))
        bbox = (min_x, min_y, max_x, max_y)

        self._status_label.setText("Querying OpenStreetMap / Overpass API for rivers in bounding box...")
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from ..centerline_wsm import fetch_osm_waterways
            ways = fetch_osm_waterways(bbox, data_epsg=self._data_epsg)
            if not ways:
                QMessageBox.information(
                    self, "No Waterways Found",
                    f"No river or stream centerlines found in OpenStreetMap for this bounding box.\n\n"
                    f"Extent: X[{min_x:.0f}, {max_x:.0f}], Y[{min_y:.0f}, {max_y:.0f}] (EPSG:{self._data_epsg})."
                )
                self._status_label.setText("No waterways found in OSM.")
                return

            self._osm_ways = ways
            self._river_combo.blockSignals(True)
            self._river_combo.clear()
            for w in ways:
                self._river_combo.addItem(f"{w['name']} ({w['waterway'].capitalize()}, {w['length_m']:.0f} m)")
            self._river_combo.blockSignals(False)
            self._river_combo.setVisible(True)

            # Select first (longest)
            self._on_river_combo_selected(0)
            self._status_label.setText(f"Found {len(ways)} waterway(s) in OpenStreetMap. Selected: {ways[0]['name']}.")
        except Exception as exc:
            logger.exception("OSM fetch failed")
            QMessageBox.critical(self, "OSM Query Failed", f"Could not fetch rivers from OpenStreetMap:\n{exc}")
            self._status_label.setText(f"OSM query failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_river_combo_selected(self, index: int):
        if not hasattr(self, "_osm_ways") or not (0 <= index < len(self._osm_ways)):
            return
        selected_way = self._osm_ways[index]
        self._centerline = RiverCenterline(selected_way["vertices"])
        self._centerline_path_edit.setText(
            f"[OSM: {selected_way['name']} ({selected_way['waterway']}), {self._centerline.total_length:.1f} m]"
        )
        self._status_label.setText(
            f"Active centerline: {selected_way['name']} ({self._centerline.total_length:.1f} m). Click '▶ Generate Cross-Sections'."
        )

    def _ensure_points_loaded(self) -> bool:
        if self._all_xs is not None and len(self._all_xs) > 0:
            return True

        self._status_label.setText("Loading point cloud points...")
        data = self._load_points_func()
        if data is None or "x" not in data or len(data["x"]) == 0:
            QMessageBox.warning(self, "No Points", "No point cloud points available.")
            return False

        self._all_xs = data["x"]
        self._all_ys = data["y"]
        self._all_zs = data["z"]
        self._all_st = data.get("sensor_type")
        self._all_cls = data.get("classification")
        return True

    def _on_generate_sections(self):
        if self._centerline is None:
            QMessageBox.warning(self, "Missing Centerline", "Please select or draw a river centerline first.")
            return

        if not self._ensure_points_loaded():
            return

        self._corridor_width = self._width_spin.value()
        self._section_spacing = self._spacing_spin.value()

        self._stations = self._centerline.sample_stations(self._section_spacing)
        n_sec = len(self._stations)
        self._water_levels = np.full(n_sec, np.nan, dtype=np.float64)
        self._locked_mask = np.zeros(n_sec, dtype=bool)
        self._cached_sections = [None] * n_sec

        self._status_label.setText(f"Slicing and detecting water level on {n_sec} cross-sections...")

        # Detect water levels automatically
        pos, tangent, normal = self._centerline.evaluate(self._stations)

        for i in range(n_sec):
            sec = slice_cross_section_points(
                self._all_xs, self._all_ys, self._all_zs,
                center_pos=pos[i],
                normal=normal[i],
                tangent=tangent[i],
                corridor_width=self._corridor_width,
                slice_thickness=2.0,
                sensor_types=self._all_st,
                classes=self._all_cls,
            )
            self._cached_sections[i] = sec
            w_z, zl, zr = detect_section_water_level(
                sec["offset"], sec["z"], sec["sensor_type"], sec["classification"]
            )
            self._water_levels[i] = w_z

        # Apply initial monotonic downhill pass
        self._water_levels = enforce_downstream_monotonicity(self._water_levels, self._locked_mask)

        # Lock first and last sections as initial anchors
        if n_sec > 0:
            self._locked_mask[0] = True
            self._locked_mask[-1] = True

        self._curr_idx = 0
        self._update_current_view()
        self._export_btn.setEnabled(True)
        self._status_label.setText(f"Generated {n_sec} cross-sections. Review sections and adjust anchors.")

    def _get_section_data(self, idx: int) -> dict:
        if self._cached_sections[idx] is None:
            pos, tangent, normal = self._centerline.evaluate(self._stations[idx])
            sec = slice_cross_section_points(
                self._all_xs, self._all_ys, self._all_zs,
                center_pos=pos,
                normal=normal,
                tangent=tangent,
                corridor_width=self._corridor_width,
                slice_thickness=2.0,
                sensor_types=self._all_st,
                classes=self._all_cls,
            )
            self._cached_sections[idx] = sec
        return self._cached_sections[idx]

    def _update_current_view(self):
        if len(self._stations) == 0:
            return

        idx = self._curr_idx
        sec = self._get_section_data(idx)
        w_z = self._water_levels[idx]
        is_locked = bool(self._locked_mask[idx])

        self._station_label.setText(
            f"Station: {self._stations[idx]:.1f} m  (Section {idx + 1} of {len(self._stations)})"
        )
        self._z_spin.blockSignals(True)
        self._z_spin.setValue(w_z)
        self._z_spin.blockSignals(False)

        self._lock_btn.blockSignals(True)
        self._lock_btn.setChecked(is_locked)
        self._lock_btn.setText("★ Approved Anchor (Locked)" if is_locked else "☆ Lock as Approved Anchor")
        self._lock_btn.setStyleSheet(
            "font-weight: bold; background-color: #d29922; color: black;"
            if is_locked else ""
        )
        self._lock_btn.blockSignals(False)

        self._canvas.set_data(
            sec["offset"], sec["z"], sec["sensor_type"], sec["classification"],
            corridor_width=self._corridor_width,
            water_z=w_z,
            is_locked=is_locked,
        )

        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _go_to_station(self, idx: int):
        if 0 <= idx < len(self._stations):
            self._curr_idx = idx
            self._update_current_view()

    def _step_station(self, delta: int):
        if len(self._stations) == 0:
            return
        new_idx = max(0, min(len(self._stations) - 1, self._curr_idx + delta))
        self._go_to_station(new_idx)

    def _on_canvas_water_level_changed(self, new_z: float):
        if len(self._stations) == 0:
            return
        self._water_levels[self._curr_idx] = new_z
        self._locked_mask[self._curr_idx] = True
        self._z_spin.blockSignals(True)
        self._z_spin.setValue(new_z)
        self._z_spin.blockSignals(False)
        self._lock_btn.setChecked(True)
        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _on_spin_z_changed(self, val: float):
        if len(self._stations) == 0:
            return
        self._water_levels[self._curr_idx] = val
        self._locked_mask[self._curr_idx] = True
        self._canvas.set_water_level(val, is_locked=True)
        self._lock_btn.setChecked(True)
        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _on_toggle_lock(self, checked: bool):
        if len(self._stations) == 0:
            return
        self._locked_mask[self._curr_idx] = checked
        self._canvas.set_water_level(self._water_levels[self._curr_idx], is_locked=checked)
        self._lock_btn.setText("★ Approved Anchor (Locked)" if checked else "☆ Lock as Approved Anchor")
        self._lock_btn.setStyleSheet(
            "font-weight: bold; background-color: #d29922; color: black;"
            if checked else ""
        )
        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _on_interpolate_anchors(self):
        if len(self._stations) < 2:
            return
        self._status_label.setText("Re-visiting in-between sections and fitting to real point cloud data...")
        self._water_levels = interpolate_anchor_sections(
            self._stations, self._water_levels, self._locked_mask,
            cached_sections=self._cached_sections, z_search_window=1.5,
        )
        self._update_current_view()
        self._status_label.setText("Re-evaluated intermediate sections against point cloud data between approved anchors.")

    def _on_add_station_dialog(self):
        if self._centerline is None or len(self._stations) == 0:
            QMessageBox.information(self, "No Centerline", "Please generate initial cross-sections first.")
            return

        from PySide6.QtWidgets import QInputDialog
        curr_s = float(self._stations[self._curr_idx]) if len(self._stations) > 0 else 0.0
        s_val, ok = QInputDialog.getDouble(
            self, "Add Cross-Section Station",
            f"Enter station distance along river (0 to {self._centerline.total_length:.1f} m):",
            curr_s, 0.0, float(self._centerline.total_length), 2
        )
        if not ok:
            return

        pos, tangent, normal = self._centerline.evaluate(s_val)
        sec = slice_cross_section_points(
            self._all_xs, self._all_ys, self._all_zs,
            center_pos=pos, normal=normal, tangent=tangent,
            corridor_width=self._corridor_width, slice_thickness=2.0,
            sensor_types=self._all_st, classes=self._all_cls,
        )
        w_z, _, _ = detect_section_water_level(sec["offset"], sec["z"], sec["sensor_type"], sec["classification"])
        if np.isnan(w_z):
            w_z = float(self._water_levels[self._curr_idx])

        from ..centerline_wsm import insert_custom_station
        self._stations, self._water_levels, self._locked_mask, self._cached_sections, new_idx = insert_custom_station(
            self._stations, self._water_levels, self._locked_mask, self._cached_sections,
            new_s=s_val, new_z=w_z, new_sec_data=sec,
        )
        self._go_to_station(new_idx)
        self._status_label.setText(f"Inserted custom section at station {s_val:.1f} m as a Locked Anchor.")

    def _on_import_drawn_profile_section(self):
        if self._centerline is None or len(self._stations) == 0:
            QMessageBox.information(self, "No Centerline", "Please generate initial cross-sections first.")
            return

        parent = self.parent()
        start_xy = getattr(parent, "_profile_start", None)
        end_xy = getattr(parent, "_profile_end", None)

        if start_xy is None or end_xy is None:
            QMessageBox.information(
                self, "No Profile Line",
                "No profile line has been drawn on the map yet.\n\n"
                "To use this feature:\n"
                "1. Click and drag a line across the river on the 2D map view\n"
                "2. Click this button to insert that cross-section as an anchor!"
            )
            return

        mx = (start_xy[0] + end_xy[0]) * 0.5
        my = (start_xy[1] + end_xy[1]) * 0.5
        s_val = float(self._centerline.project_point_to_station(mx, my))

        pos, tangent, normal = self._centerline.evaluate(s_val)
        sec = slice_cross_section_points(
            self._all_xs, self._all_ys, self._all_zs,
            center_pos=pos, normal=normal, tangent=tangent,
            corridor_width=self._corridor_width, slice_thickness=2.0,
            sensor_types=self._all_st, classes=self._all_cls,
        )
        w_z, _, _ = detect_section_water_level(sec["offset"], sec["z"], sec["sensor_type"], sec["classification"])
        if np.isnan(w_z):
            w_z = float(self._water_levels[self._curr_idx])

        from ..centerline_wsm import insert_custom_station
        self._stations, self._water_levels, self._locked_mask, self._cached_sections, new_idx = insert_custom_station(
            self._stations, self._water_levels, self._locked_mask, self._cached_sections,
            new_s=s_val, new_z=w_z, new_sec_data=sec,
        )
        self._go_to_station(new_idx)
        self._status_label.setText(
            f"Inserted custom section from drawn map line at station {s_val:.1f} m as a Locked Anchor."
        )

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_A, Qt.Key_Left):
            self._step_station(-1)
        elif key in (Qt.Key_D, Qt.Key_Right):
            self._step_station(1)
        elif key == Qt.Key_W:
            self._z_spin.setValue(self._z_spin.value() + 0.05)
        elif key == Qt.Key_S:
            self._z_spin.setValue(self._z_spin.value() - 0.05)
        elif key == Qt.Key_Space:
            self._on_toggle_lock(not self._locked_mask[self._curr_idx])
        else:
            super().keyPressEvent(event)

    def _on_export_geotiff(self):
        if len(self._stations) < 2 or self._centerline is None:
            return

        default_fn = str(self._project_dir / "water_surface_model.tif")
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Save Water Surface Model GeoTIFF",
            default_fn,
            "GeoTIFF Files (*.tif *.tiff)",
        )
        if not fn:
            return

        try:
            self._status_label.setText("Rasterizing 3D Water Surface Model...")
            # Make sure interpolation is clean before rasterizing
            levels = interpolate_anchor_sections(self._stations, self._water_levels, self._locked_mask)
            surf, georef = rasterize_water_surface_model(
                centerline=self._centerline,
                stations=self._stations,
                water_levels=levels,
                corridor_width=self._corridor_width,
                output_path=fn,
                resolution=1.0,
                data_epsg=self._data_epsg,
            )
            self._status_label.setText(f"WSM GeoTIFF saved successfully: {Path(fn).name}")
            self.wsm_generated.emit(fn)
            QMessageBox.information(
                self,
                "WSM Generated",
                f"Water Surface Model saved successfully:\n{fn}\n\n"
                f"Resolution: 1.0 m\nDimensions: {georef['width']} x {georef['height']}\n"
                f"This surface is ready for Snell's Law refraction and river corridor cropping.",
            )
        except Exception as e:
            logger.exception("WSM export failed")
            QMessageBox.critical(self, "Export Error", f"Failed to export GeoTIFF: {e}")

