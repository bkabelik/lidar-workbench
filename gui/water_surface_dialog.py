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
    clip_polyline_to_bbox,
    detect_embankment_extents,
    detect_section_water_level,
    drape_centerline_water_surface,
    enforce_downstream_monotonicity,
    interpolate_anchor_sections,
    load_centerline_from_file,
    load_water_surface_sections,
    polyline_length_in_bbox,
    rasterize_water_surface_model,
    save_water_surface_sections,
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
    water_section_changed = Signal(float, float, float, float, float)  # z_mid, z_left, z_right, off_left, off_right

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
        self._water_z_left: float = 100.0
        self._water_z_right: float = 100.0
        self._left_off: float = -20.0
        self._right_off: float = 20.0
        self._is_locked: bool = False
        self._tilt_mode: bool = False

        # View transform
        self._offset_x: float = 0.0
        self._offset_z: float = 0.0
        self._scale_x: float = 8.0  # pixels per metre
        self._scale_z: float = 25.0

        # Interaction
        self._drag_target: Optional[str] = None  # None, 'left', 'right', 'center', 'line'
        self._hover_target: Optional[str] = None
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
        left_off: Optional[float] = None,
        right_off: Optional[float] = None,
        water_z_left: Optional[float] = None,
        water_z_right: Optional[float] = None,
        tilt_mode: bool = False,
    ):
        self._offsets = offsets
        self._elevations = elevations
        self._sensor_types = sensor_types
        self._classes = classes
        self._corridor_width = max(5.0, float(corridor_width))
        self._water_z = float(water_z)
        self._is_locked = bool(is_locked)
        self._tilt_mode = bool(tilt_mode)

        half_w = self._corridor_width * 0.5
        self._left_off = float(left_off) if left_off is not None else -half_w
        self._right_off = float(right_off) if right_off is not None else half_w
        self._water_z_left = float(water_z_left) if water_z_left is not None else self._water_z
        self._water_z_right = float(water_z_right) if water_z_right is not None else self._water_z

        self.fit_view()
        self.update()

    def set_water_level(self, z: float, is_locked: Optional[bool] = None):
        dz = float(z) - self._water_z
        self._water_z = float(z)
        self._water_z_left = float(self._water_z_left + dz)
        self._water_z_right = float(self._water_z_right + dz)
        if is_locked is not None:
            self._is_locked = is_locked
        self.update()

    def set_water_geometry(
        self,
        z_center: float,
        z_left: Optional[float] = None,
        z_right: Optional[float] = None,
        left_off: Optional[float] = None,
        right_off: Optional[float] = None,
        is_locked: Optional[bool] = None,
    ):
        self._water_z = float(z_center)
        self._water_z_left = float(z_left) if z_left is not None else self._water_z
        self._water_z_right = float(z_right) if z_right is not None else self._water_z
        if left_off is not None:
            self._left_off = float(left_off)
        if right_off is not None:
            self._right_off = float(right_off)
        if is_locked is not None:
            self._is_locked = is_locked
        self.update()

    def set_tilt_mode(self, enabled: bool):
        self._tilt_mode = bool(enabled)
        self.update()

    def fit_view(self):
        w = max(100, self.width())
        h = max(100, self.height())

        margin_x = 40.0
        margin_y = 30.0

        range_x = self._corridor_width * 1.1

        if self._elevations is not None and len(self._elevations) > 0:
            min_z = float(np.min(self._elevations))
            max_z = float(np.max(self._elevations))
            for wz in (self._water_z, self._water_z_left, self._water_z_right):
                if not np.isnan(wz):
                    min_z = min(min_z, wz - 1.0)
                    max_z = max(max_z, wz + 1.0)
        else:
            w_ref = self._water_z if not np.isnan(self._water_z) else 100.0
            min_z = w_ref - 3.0
            max_z = w_ref + 3.0

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
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()

            # Background
            painter.fillRect(0, 0, w, h, QColor("#161a22"))

            # Grid lines & centerline marker
            painter.setPen(QPen(QColor("#2d333b"), 1, Qt.DashLine))
            cx, _ = self._world_to_widget(0.0, 0.0)
            painter.drawLine(int(cx), 0, int(cx), h)

            # Labels font
            f_norm = painter.font()
            f_norm.setPointSize(9)
            painter.setFont(f_norm)
            painter.setPen(QColor("#768390"))
            painter.drawText(int(cx) + 5, 20, "Centerline (0.0 m)")

            # Corridor limits
            half_w = self._corridor_width * 0.5
            lx, _ = self._world_to_widget(-half_w, 0.0)
            rx, _ = self._world_to_widget(half_w, 0.0)
            painter.setPen(QPen(QColor("#3d444d"), 1, Qt.DotLine))
            painter.drawLine(int(lx), 0, int(lx), h)
            painter.drawLine(int(rx), 0, int(rx), h)
            painter.drawText(int(lx) + 5, h - 10, f"-{half_w:.1f} m (Left)")
            painter.drawText(int(rx) - 75, h - 10, f"+{half_w:.1f} m (Right)")

            # Render LiDAR points
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
            else:
                f_italic = painter.font()
                f_italic.setPointSize(11)
                f_italic.setItalic(True)
                painter.setFont(f_italic)
                painter.setPen(QColor("#8b949e"))
                painter.drawText(
                    self.rect(),
                    Qt.AlignCenter,
                    "No LiDAR points at this station\n(Widen corridor or check if station is within survey extent)",
                )

            # Draw Water Surface Line (ending at embankments)
            if not np.isnan(self._water_z):
                plx, ply = self._world_to_widget(self._left_off, self._water_z_left)
                prx, pry = self._world_to_widget(self._right_off, self._water_z_right)
                pcx, pcy = self._world_to_widget(0.0, self._water_z)

                line_color = QColor("#d29922") if self._is_locked else QColor("#58a6ff")
                pen_width = 3 if self._drag_target or self._hover_target else 2
                painter.setPen(QPen(line_color, pen_width, Qt.SolidLine))

                # Water surface line within embankment bounds
                painter.drawLine(int(round(plx)), int(round(ply)), int(round(prx)), int(round(pry)))

                # Faded dotted extension to corridor edges if water section is narrower than corridor
                painter.setPen(QPen(QColor("#484f58"), 1, Qt.DashLine))
                if plx > lx + 4:
                    painter.drawLine(int(round(lx)), int(round(ply)), int(round(plx)), int(round(ply)))
                if prx < rx - 4:
                    painter.drawLine(int(round(prx)), int(round(pry)), int(round(rx)), int(round(pry)))

                # Embankment vertical markers
                painter.setPen(QPen(QColor("#3fb950"), 2, Qt.SolidLine))
                painter.drawLine(int(round(plx)), int(round(ply)) - 8, int(round(plx)), int(round(ply)) + 8)
                painter.drawLine(int(round(prx)), int(round(pry)) - 8, int(round(prx)), int(round(pry)) + 8)

                # Center handle
                painter.setPen(QPen(Qt.white, 1.5))
                painter.setBrush(line_color)
                c_rad = 6 if self._hover_target == "center" or self._drag_target == "center" else 5
                painter.drawEllipse(QPointF(pcx, pcy), c_rad, c_rad)

                # Endpoint handles (always interactive, highlighted in tilt mode)
                l_rad = 7 if (self._hover_target == "left" or self._drag_target == "left") else (6 if self._tilt_mode else 4)
                r_rad = 7 if (self._hover_target == "right" or self._drag_target == "right") else (6 if self._tilt_mode else 4)

                l_col = QColor("#f778ba") if self._tilt_mode else line_color
                r_col = QColor("#f778ba") if self._tilt_mode else line_color
                if self._drag_target == "left":
                    l_col = QColor("#ff7b72")
                if self._drag_target == "right":
                    r_col = QColor("#ff7b72")

                painter.setBrush(l_col)
                painter.drawEllipse(QPointF(plx, ply), l_rad, l_rad)

                painter.setBrush(r_col)
                painter.drawEllipse(QPointF(prx, pry), r_rad, r_rad)

                # Info Tag
                status_tag = " [LOCKED ANCHOR]" if self._is_locked else " [Auto]"
                diff_z = self._water_z_right - self._water_z_left
                span_w = max(0.1, self._right_off - self._left_off)
                if abs(diff_z) > 0.005:
                    slope_pct = (diff_z / span_w) * 100.0
                    tilt_info = f" | Tilt: L {self._water_z_left:.2f}m / R {self._water_z_right:.2f}m ({slope_pct:+.1f}%)"
                else:
                    tilt_info = " [Horizontal]"
                tag_text = f"Water Z: {self._water_z:.3f} m (Width: {span_w:.1f} m: [{self._left_off:.1f}, +{self._right_off:.1f}]){tilt_info}{status_tag}"

                f_bold = painter.font()
                f_bold.setPointSize(10)
                f_bold.setBold(True)
                painter.setFont(f_bold)
                painter.setPen(line_color)
                tag_y = int(min(ply, pry)) - 10
                painter.drawText(max(10, int(plx)), max(35, tag_y), tag_text)
        finally:
            painter.end()

    def _hit_test(self, px: float, py: float) -> Optional[str]:
        if np.isnan(self._water_z):
            return None

        plx, ply = self._world_to_widget(self._left_off, self._water_z_left)
        prx, pry = self._world_to_widget(self._right_off, self._water_z_right)
        pcx, pcy = self._world_to_widget(0.0, self._water_z)

        # 1. Left endpoint handle
        if np.hypot(px - plx, py - ply) <= 12:
            return "left"

        # 2. Right endpoint handle
        if np.hypot(px - prx, py - pry) <= 12:
            return "right"

        # 3. Center handle
        if np.hypot(px - pcx, py - pcy) <= 12:
            return "center"

        # 4. Line segment between endpoints
        dx = prx - plx
        dy = pry - ply
        l2 = dx * dx + dy * dy
        if l2 > 1e-6:
            t = max(0.0, min(1.0, ((px - plx) * dx + (py - ply) * dy) / l2))
            proj_x = plx + t * dx
            proj_y = ply + t * dy
            if np.hypot(px - proj_x, py - proj_y) <= 10:
                return "line"

        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            target = self._hit_test(event.position().x(), event.position().y())
            if target is not None:
                self._drag_target = target
                if target in ("left", "right"):
                    self.setCursor(Qt.SizeAllCursor if self._tilt_mode else Qt.SizeHorCursor)
                else:
                    self.setCursor(Qt.SizeVerCursor)
                self.update()
                return
            else:
                # Direct click anywhere on section: place horizontal water line at that elevation
                _, new_z = self._widget_to_world(event.position().x(), event.position().y())
                self._water_z = round(new_z, 3)
                self._water_z_left = self._water_z
                self._water_z_right = self._water_z
                self._is_locked = True
                self._drag_target = "center"
                self.setCursor(Qt.SizeVerCursor)
                self._emit_change()
                self.update()
                return
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        mx, my = event.position().x(), event.position().y()

        if self._drag_target is not None:
            world_off, world_z = self._widget_to_world(mx, my)
            half_w = self._corridor_width * 0.5

            if self._drag_target == "left":
                self._left_off = round(max(-half_w, min(-0.5, world_off)), 2)
                if self._tilt_mode:
                    self._water_z_left = round(world_z, 3)
                    self._water_z = round(0.5 * (self._water_z_left + self._water_z_right), 3)

            elif self._drag_target == "right":
                self._right_off = round(min(half_w, max(0.5, world_off)), 2)
                if self._tilt_mode:
                    self._water_z_right = round(world_z, 3)
                    self._water_z = round(0.5 * (self._water_z_left + self._water_z_right), 3)

            elif self._drag_target in ("center", "line"):
                dz = round(world_z, 3) - self._water_z
                self._water_z = round(world_z, 3)
                self._water_z_left = round(self._water_z_left + dz, 3)
                self._water_z_right = round(self._water_z_right + dz, 3)

            self._is_locked = True
            self._emit_change()
            self.update()
            return

        elif self._panning and self._pan_start is not None:
            dx = event.position().x() - self._pan_start.x()
            dy = event.position().y() - self._pan_start.y()
            self._offset_x += dx
            self._offset_z += dy
            self._pan_start = event.position()
            self.update()
            return

        # Hover test
        target = self._hit_test(mx, my)
        if target != self._hover_target:
            self._hover_target = target
            if target in ("left", "right"):
                self.setCursor(Qt.SizeAllCursor if self._tilt_mode else Qt.SizeHorCursor)
            elif target in ("center", "line"):
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._drag_target is not None:
            self._drag_target = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
        elif event.button() in (Qt.MiddleButton, Qt.RightButton) and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _emit_change(self):
        self.water_level_changed.emit(self._water_z)
        self.water_section_changed.emit(
            self._water_z, self._water_z_left, self._water_z_right, self._left_off, self._right_off
        )

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # Double click to reset view
        self.fit_view()
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        # Zoom centered at the mouse cursor
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        pos = event.position()
        mx, my = pos.x(), pos.y()
        world_off, world_z = self._widget_to_world(mx, my)

        new_scale_x = max(0.1, min(500.0, self._scale_x * factor))
        new_scale_z = max(0.1, min(500.0, self._scale_z * factor))

        self._scale_x = new_scale_x
        self._scale_z = new_scale_z
        self._offset_x = mx - world_off * self._scale_x
        self._offset_z = my + world_z * self._scale_z
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
        try:
            w, h = self.width(), self.height()
            painter.fillRect(0, 0, w, h, QColor("#1c2128"))

            if self._n_stations <= 0:
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
        finally:
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
        project_bbox: Optional[Tuple[float, float, float, float]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._project_dir = Path(project_dir)
        self._load_points_func = load_points_func
        self._data_epsg = data_epsg
        self._project_bbox = project_bbox

        crs_tag = f"EPSG:{data_epsg}" if data_epsg else "CRS: Not set"
        self.setWindowTitle(f"River Water Surface Model Generator (Centerline) — [{crs_tag}]")
        self.setMinimumSize(880, 680)

        # Centerline and cross-section state
        self._full_centerline: Optional[RiverCenterline] = None
        self._centerline: Optional[RiverCenterline] = None
        self._bathy_s_range: Optional[Tuple[float, float]] = None
        self._stations: np.ndarray = np.array([])
        self._water_levels: np.ndarray = np.array([])
        self._water_z_left: np.ndarray = np.array([])
        self._water_z_right: np.ndarray = np.array([])
        self._left_offsets: np.ndarray = np.array([])
        self._right_offsets: np.ndarray = np.array([])
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
        crs_tag = f"EPSG:{self._data_epsg}" if self._data_epsg else "Unknown CRS"
        setup_group = QGroupBox(f"1. River Centerline & Corridor Setup ({crs_tag})")
        self._setup_group = setup_group
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

        bathy_reach_row = QHBoxLayout()
        self._limit_bathy_cb = QCheckBox("Limit cross-sections to bathymetry reach")
        self._limit_bathy_cb.setChecked(True)
        self._limit_bathy_cb.setToolTip(
            "Automatically restrict cross-sections to the survey reach where bathymetric points exist,\n"
            "skipping upstream/downstream areas that only contain topo LiDAR."
        )
        self._limit_bathy_cb.toggled.connect(self._on_limit_bathy_toggled)
        bathy_reach_row.addWidget(self._limit_bathy_cb)

        self._bathy_reach_label = QLabel("")
        self._bathy_reach_label.setStyleSheet("color: #58a6ff; font-weight: bold;")
        bathy_reach_row.addWidget(self._bathy_reach_label)
        bathy_reach_row.addStretch()
        sf.addRow("Survey Extent:", bathy_reach_row)

        params_row = QHBoxLayout()
        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(5.0, 500.0)
        self._width_spin.setValue(40.0)
        self._width_spin.setSuffix(" m")
        self._width_spin.setToolTip("Full slicing corridor width. Set wide enough to capture whole channel and bends.")
        params_row.addWidget(QLabel("Corridor Width:"))
        params_row.addWidget(self._width_spin)

        self._spacing_spin = QDoubleSpinBox()
        self._spacing_spin.setRange(2.0, 200.0)
        self._spacing_spin.setValue(10.0)
        self._spacing_spin.setSuffix(" m")
        params_row.addWidget(QLabel("Section Spacing:"))
        params_row.addWidget(self._spacing_spin)

        self._embank_spin = QDoubleSpinBox()
        self._embank_spin.setRange(0.0, 50.0)
        self._embank_spin.setValue(2.5)
        self._embank_spin.setSuffix(" m")
        self._embank_spin.setToolTip("Buffer distance past the embankment touch-point where the water surface terminates")
        params_row.addWidget(QLabel("Bank Margin:"))
        params_row.addWidget(self._embank_spin)

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
        self._canvas.water_section_changed.connect(self._on_canvas_water_section_changed)
        rf.addWidget(self._canvas)

        # Controls under canvas
        ctrl_layout = QHBoxLayout()

        self._btn_prev10 = QPushButton("<< -10")
        self._btn_prev10.setToolTip("Jump back 10 cross-sections")
        self._btn_prev10.clicked.connect(lambda: self._step_station(-10))
        ctrl_layout.addWidget(self._btn_prev10)

        self._btn_prev5 = QPushButton("< -5")
        self._btn_prev5.setToolTip("Jump back 5 cross-sections")
        self._btn_prev5.clicked.connect(lambda: self._step_station(-5))
        ctrl_layout.addWidget(self._btn_prev5)

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

        self._btn_next5 = QPushButton("+5 >")
        self._btn_next5.setToolTip("Jump forward 5 cross-sections")
        self._btn_next5.clicked.connect(lambda: self._step_station(5))
        ctrl_layout.addWidget(self._btn_next5)

        self._btn_next10 = QPushButton("+10 >>")
        self._btn_next10.setToolTip("Jump forward 10 cross-sections")
        self._btn_next10.clicked.connect(lambda: self._step_station(10))
        ctrl_layout.addWidget(self._btn_next10)

        rf.addLayout(ctrl_layout)

        # Elevation edit, tilt, & lock anchor row
        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("Water Elevation:"))

        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(-1000.0, 9000.0)
        self._z_spin.setDecimals(3)
        self._z_spin.setValue(100.0)
        self._z_spin.setSuffix(" m")
        self._z_spin.valueChanged.connect(self._on_spin_z_changed)
        edit_layout.addWidget(self._z_spin)

        self._tilt_btn = QPushButton("📐 Tilt / Modify Endpoints")
        self._tilt_btn.setCheckable(True)
        self._tilt_btn.setToolTip("Enable interactive tilt and endpoint adjustment (drag pink handles on canvas)")
        self._tilt_btn.toggled.connect(self._on_toggle_tilt_mode)
        edit_layout.addWidget(self._tilt_btn)

        self._level_btn = QPushButton("➡ Level (Horizontal)")
        self._level_btn.setToolTip("Reset this section back to a level horizontal water surface")
        self._level_btn.clicked.connect(self._on_level_horizontal)
        edit_layout.addWidget(self._level_btn)

        self._lock_btn = QPushButton("★ Lock as Approved Anchor (Space)")
        self._lock_btn.setCheckable(True)
        self._lock_btn.clicked.connect(self._on_toggle_lock)
        edit_layout.addWidget(self._lock_btn)

        self._interp_btn = QPushButton("↺ Interpolate Intermediate Sections")
        self._interp_btn.setToolTip("Smoothly re-interpolates all auto sections between your approved anchors.")
        self._interp_btn.clicked.connect(self._on_interpolate_anchors)
        edit_layout.addWidget(self._interp_btn)

        rf.addLayout(edit_layout)

        # Custom section insertion & removal row
        sec_mgmt_layout = QHBoxLayout()
        self._btn_add_station = QPushButton("➕ Add Section at Station…")
        self._btn_add_station.setToolTip("Insert a new cross-section at an exact station distance (e.g. at a weir, power plant, or rapid)")
        self._btn_add_station.clicked.connect(self._on_add_station_dialog)
        sec_mgmt_layout.addWidget(self._btn_add_station)

        self._btn_import_drawn = QPushButton("✏ Insert Section from Map Profile Line")
        self._btn_import_drawn.setToolTip("Drag a profile line across the river on the 2D map, then click here to add it as an anchor!")
        self._btn_import_drawn.clicked.connect(self._on_import_drawn_profile_section)
        sec_mgmt_layout.addWidget(self._btn_import_drawn)

        self._btn_remove_sec = QPushButton("🗑 Remove Current Section (Del)")
        self._btn_remove_sec.setToolTip(
            "Remove the currently selected section (e.g. under bridges, power lines, or where no water data exists).\n"
            "Surrounding sections will be smoothly bridged during interpolation and export."
        )
        self._btn_remove_sec.setStyleSheet("color: #f85149; font-weight: bold;")
        self._btn_remove_sec.clicked.connect(self._on_remove_current_section)
        sec_mgmt_layout.addWidget(self._btn_remove_sec)

        rf.addLayout(sec_mgmt_layout)
        layout.addWidget(review_group)

        # ── 3. Bottom Actions ──
        bottom_layout = QHBoxLayout()
        self._status_label = QLabel("Ready. Load or draw a centerline to begin.")
        bottom_layout.addWidget(self._status_label)

        self._load_btn = QPushButton("📂 Load Profile…")
        self._load_btn.setToolTip("Load saved water surface cross-sections and anchors from JSON")
        self._load_btn.clicked.connect(self._on_load_sections)
        bottom_layout.addWidget(self._load_btn)

        self._save_btn = QPushButton("💾 Save Profile…")
        self._save_btn.setToolTip("Save current cross-sections, anchors, and tilt adjustments to JSON")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_sections)
        bottom_layout.addWidget(self._save_btn)

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
            self._full_centerline = RiverCenterline(vertices)
            self._centerline = self._full_centerline
            self._centerline_path_edit.setText(f"[Drawn Polyline: {len(vertices)} vertices, {self._centerline.total_length:.1f} m]")
            self._status_label.setText(f"Centerline loaded: {self._centerline.total_length:.1f} m. Click 'Generate Cross-Sections'.")
            self._update_bathy_reach()
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
            self._full_centerline = RiverCenterline(verts)
            self._centerline = self._full_centerline
            self._centerline_path_edit.setText(fn)
            self._status_label.setText(f"Loaded {len(verts)} vertices ({self._centerline.total_length:.1f} m).")
            self._update_bathy_reach()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load centerline: {e}")

    def _on_fetch_osm_waterways(self):
        # Determine bbox: prefer project_bbox, otherwise load points to get extent
        if self._project_bbox is not None:
            bbox = self._project_bbox
            min_x, min_y, max_x, max_y = bbox
        else:
            if not self._ensure_points_loaded():
                return
            min_x = float(np.min(self._all_xs))
            max_x = float(np.max(self._all_xs))
            min_y = float(np.min(self._all_ys))
            max_y = float(np.max(self._all_ys))
            bbox = (min_x, min_y, max_x, max_y)

        if not self._data_epsg:
            from PySide6.QtWidgets import QInputDialog
            epsg, ok = QInputDialog.getInt(
                self, "Project EPSG",
                "Enter project EPSG code for reprojection (e.g. 25832, 25833, 31256):",
                25833, 1000, 99999
            )
            if not ok:
                return
            self._data_epsg = epsg
            self.setWindowTitle(f"River Water Surface Model Generator (Centerline) — [EPSG:{self._data_epsg}]")
            if hasattr(self, "_setup_group"):
                self._setup_group.setTitle(f"1. River Centerline & Corridor Setup (EPSG:{self._data_epsg})")

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
                l_in = w.get("length_inside_m", 0.0)
                l_tot = w.get("length_m", 0.0)
                if l_in > 0:
                    label = f"{w['name']} ({w['waterway'].capitalize()} — {l_in:.0f} m in project, {l_tot:.0f} m total)"
                else:
                    label = f"{w['name']} ({w['waterway'].capitalize()} — {l_tot:.0f} m total)"
                self._river_combo.addItem(label)
            self._river_combo.blockSignals(False)
            self._river_combo.setVisible(True)

            # Auto-select the top-ranked river (ranked by coverage inside project)
            self._on_river_combo_selected(0)
            top_name = ways[0]["name"]
            top_len = ways[0].get("length_inside_m", ways[0]["length_m"])
            self._status_label.setText(
                f"Found {len(ways)} waterway(s) in OpenStreetMap. Auto-selected '{top_name}' ({top_len:.0f} m in project)."
            )
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
        verts = selected_way["vertices"]

        # Clip centerline to survey/project bounding box if available
        bbox = self._project_bbox
        if bbox is None and self._all_xs is not None and len(self._all_xs) > 0:
            bbox = (
                float(np.min(self._all_xs)),
                float(np.min(self._all_ys)),
                float(np.max(self._all_xs)),
                float(np.max(self._all_ys)),
            )

        if bbox is not None and len(verts) >= 2:
            chains = clip_polyline_to_bbox(verts, bbox, buffer_m=30.0)
            if chains:
                # Pick longest continuous sub-polyline inside the survey
                chains.sort(key=lambda c: np.sum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))), reverse=True)
                verts = chains[0]

        self._full_centerline = RiverCenterline(verts)
        self._centerline = self._full_centerline
        # Clear cached points so points for this new corridor will be loaded
        self._all_xs = None
        self._cached_sections = []
        self._stations = np.array([])
        self._water_levels = np.array([])
        self._bathy_s_range = None
        self._centerline_path_edit.setText(
            f"[OSM: {selected_way['name']} ({selected_way['waterway']}), {self._centerline.total_length:.1f} m clipped to survey]"
        )
        self._status_label.setText(
            f"Active centerline: {selected_way['name']} ({self._centerline.total_length:.1f} m in project). Click '▶ Generate Cross-Sections'."
        )
        self._update_bathy_reach()

    def _update_bathy_reach(self):
        """Detect bathymetric points range along centerline and configure reach trimming."""
        if self._full_centerline is None:
            self._bathy_s_range = None
            self._bathy_reach_label.setText("")
            return

        if self._all_xs is None or len(self._all_xs) == 0:
            return

        if self._all_st is None or not (self._all_st == 2).any():
            self._bathy_s_range = None
            self._bathy_reach_label.setText("(No bathymetry points in corridor)")
            self._limit_bathy_cb.setEnabled(False)
            self._centerline = self._full_centerline
            return

        # Subsample bathy points up to 20,000 for fast projection
        bathy_mask = (self._all_st == 2)
        b_xs = self._all_xs[bathy_mask]
        b_ys = self._all_ys[bathy_mask]
        n_b = len(b_xs)
        if n_b > 20000:
            step = n_b // 20000
            b_xs = b_xs[::step]
            b_ys = b_ys[::step]

        from ..centerline_wsm import project_points_to_centerline
        b_s = project_points_to_centerline(self._full_centerline, b_xs, b_ys)
        if len(b_s) == 0:
            self._bathy_s_range = None
            self._bathy_reach_label.setText("")
            return

        # 0.5% and 99.5% quantiles with 15m margin to capture the full bathy survey
        s_min = max(0.0, float(np.percentile(b_s, 0.5)) - 15.0)
        s_max = min(self._full_centerline.total_length, float(np.percentile(b_s, 99.5)) + 15.0)

        if s_max - s_min >= 5.0:
            self._bathy_s_range = (s_min, s_max)
            self._limit_bathy_cb.setEnabled(True)
            reach_len = s_max - s_min
            total_len = self._full_centerline.total_length
            self._bathy_reach_label.setText(
                f"Bathy reach: {s_min:.0f} m – {s_max:.0f} m ({reach_len:.0f} m of {total_len:.0f} m)"
            )
            if self._limit_bathy_cb.isChecked():
                self._centerline = self._full_centerline.trim(s_min, s_max)
            else:
                self._centerline = self._full_centerline
        else:
            self._bathy_s_range = None
            self._centerline = self._full_centerline

    def _on_limit_bathy_toggled(self, checked: bool):
        if self._full_centerline is None:
            return
        if checked and self._bathy_s_range is not None:
            s_min, s_max = self._bathy_s_range
            self._centerline = self._full_centerline.trim(s_min, s_max)
            self._status_label.setText(f"Trimmed centerline to bathy reach: {self._centerline.total_length:.1f} m.")
        else:
            self._centerline = self._full_centerline
            self._status_label.setText(f"Full centerline active: {self._centerline.total_length:.1f} m.")

    def _ensure_points_loaded(self, force_reload: bool = False) -> bool:
        if not force_reload and self._all_xs is not None and len(self._all_xs) > 0:
            return True

        self._status_label.setText("Loading point cloud points along river corridor...")
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            import inspect
            sig = inspect.signature(self._load_points_func)
            if "centerline_verts" in sig.parameters:
                verts = (self._full_centerline.vertices if self._full_centerline is not None
                         else (self._centerline.vertices if self._centerline else None))
                data = self._load_points_func(centerline_verts=verts)
            else:
                data = self._load_points_func()
        finally:
            QApplication.restoreOverrideCursor()

        if data is None or "x" not in data or len(data["x"]) == 0:
            QMessageBox.warning(
                self, "No Points",
                "No point cloud points available for this corridor.\n"
                f"Please verify project CRS (EPSG:{self._data_epsg}) matches your data."
            )
            return False

        self._all_xs = data["x"]
        self._all_ys = data["y"]
        self._all_zs = data["z"]
        self._all_st = data.get("sensor_type")
        self._all_cls = data.get("classification")

        if len(self._all_xs) > 100:
            order = np.argsort(self._all_xs)
            self._all_xs = self._all_xs[order]
            self._all_ys = self._all_ys[order]
            self._all_zs = self._all_zs[order]
            if self._all_st is not None:
                self._all_st = self._all_st[order]
            if self._all_cls is not None:
                self._all_cls = self._all_cls[order]
            self._points_sorted_x = True
        else:
            self._points_sorted_x = False

        self._status_label.setText(f"Loaded {len(self._all_xs):,} points along corridor.")
        self._update_bathy_reach()
        return True

    def _on_generate_sections(self):
        if self._centerline is None:
            QMessageBox.warning(self, "Missing Centerline", "Please select or draw a river centerline first.")
            return

        if not self._ensure_points_loaded():
            return

        # If limiting to bathymetry reach and reach is available, ensure centerline is trimmed
        if self._limit_bathy_cb.isChecked() and self._bathy_s_range is not None and self._full_centerline is not None:
            s_min, s_max = self._bathy_s_range
            self._centerline = self._full_centerline.trim(s_min, s_max)

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
        sections_with_points = 0
        sorted_x = getattr(self, "_points_sorted_x", False)

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
                sorted_x=sorted_x,
            )
            self._cached_sections[i] = sec
            n_pts = len(sec["offset"])
            if n_pts > 0:
                sections_with_points += 1

        if sections_with_points == 0:
            QMessageBox.warning(
                self, "No Points Along Centerline",
                "None of the cross-sections intersected point cloud points!\n\n"
                f"Points extent: X[{self._all_xs.min():.0f}, {self._all_xs.max():.0f}], "
                f"Y[{self._all_ys.min():.0f}, {self._all_ys.max():.0f}]\n"
                f"Centerline extent: X[{pos[:,0].min():.0f}, {pos[:,0].max():.0f}], "
                f"Y[{pos[:,1].min():.0f}, {pos[:,1].max():.0f}]\n\n"
                f"Please verify that project CRS (EPSG:{self._data_epsg}) matches your data coordinates."
            )

        # Estimate clean water surface profile by draping the centerline over the river channel:
        # Isolates central channel strip (+/- 1.5 m), eliminates high tree canopy / bridges
        # using riverbed baseline (Z_bed + max_water_depth), applies running median filter
        # to remove spikes, and enforces downstream monotonicity.
        self._water_levels = drape_centerline_water_surface(
            centerline=self._centerline,
            stations=self._stations,
            corridor_width=self._corridor_width,
            cached_sections=self._cached_sections,
            center_strip_width=1.5,
            max_water_depth=3.5,
            outlier_threshold_m=0.60,
            median_window=5,
        )

        # Initialize tilted levels and embankment trimmed offsets
        from ..centerline_wsm import detect_embankment_extents
        bank_margin = self._embank_spin.value()
        self._water_z_left = self._water_levels.copy()
        self._water_z_right = self._water_levels.copy()
        self._left_offsets = np.zeros(n_sec, dtype=np.float64)
        self._right_offsets = np.zeros(n_sec, dtype=np.float64)

        for i in range(n_sec):
            sec = self._cached_sections[i]
            if sec is not None and len(sec["offset"]) > 0:
                l_off, r_off = detect_embankment_extents(
                    sec["offset"], sec["z"], self._water_levels[i],
                    corridor_width=self._corridor_width, margin=bank_margin,
                )
            else:
                half_w = self._corridor_width * 0.5
                l_off, r_off = -half_w, half_w
            self._left_offsets[i] = l_off
            self._right_offsets[i] = r_off

        # Lock first and last sections as initial anchors
        if n_sec > 0:
            self._locked_mask[0] = True
            self._locked_mask[-1] = True

        self._curr_idx = 0
        self._update_current_view()
        self._save_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._status_label.setText(
            f"Generated {n_sec} cross-sections ({sections_with_points} with LiDAR points). Review sections and adjust anchors."
        )

    def _get_section_data(self, idx: int) -> dict:
        if self._cached_sections[idx] is None:
            if not self._ensure_points_loaded():
                return {"offset": np.array([]), "z": np.array([])}
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
                sorted_x=getattr(self, "_points_sorted_x", False),
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
        n_pts = len(sec["offset"]) if sec and "offset" in sec else 0

        self._station_label.setText(
            f"Station: {self._stations[idx]:.1f} m  (Section {idx + 1} of {len(self._stations)}) — {n_pts} pts"
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

        half_w = self._corridor_width * 0.5
        l_off = float(self._left_offsets[idx]) if len(self._left_offsets) > idx else -half_w
        r_off = float(self._right_offsets[idx]) if len(self._right_offsets) > idx else half_w
        z_l = float(self._water_z_left[idx]) if len(self._water_z_left) > idx else w_z
        z_r = float(self._water_z_right[idx]) if len(self._water_z_right) > idx else w_z

        self._canvas.set_data(
            sec["offset"], sec["z"], sec["sensor_type"], sec["classification"],
            corridor_width=self._corridor_width,
            water_z=w_z,
            is_locked=is_locked,
            left_off=l_off,
            right_off=r_off,
            water_z_left=z_l,
            water_z_right=z_r,
            tilt_mode=self._tilt_btn.isChecked(),
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

    def _on_toggle_tilt_mode(self, checked: bool):
        self._canvas.set_tilt_mode(checked)
        if checked:
            self._tilt_btn.setText("📐 Tilting Mode (Active)")
            self._tilt_btn.setStyleSheet("font-weight: bold; background-color: #f0883e; color: black;")
            self._status_label.setText(
                "Tilt mode active: Drag left/right handles on canvas to adjust bank endpoints and tilt."
            )
        else:
            self._tilt_btn.setText("📐 Tilt / Modify Endpoints")
            self._tilt_btn.setStyleSheet("")
            self._status_label.setText("Tilt mode inactive. Endpoints adjust horizontally.")

    def _on_level_horizontal(self):
        if len(self._stations) == 0:
            return
        idx = self._curr_idx
        w_z = float(self._water_levels[idx])
        self._water_z_left[idx] = w_z
        self._water_z_right[idx] = w_z
        self._locked_mask[idx] = True
        self._canvas.set_water_geometry(
            z_center=w_z,
            z_left=w_z,
            z_right=w_z,
            left_off=self._left_offsets[idx],
            right_off=self._right_offsets[idx],
            is_locked=True,
        )
        self._lock_btn.setChecked(True)
        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)
        self._status_label.setText(f"Reset section at {self._stations[idx]:.1f} m to level horizontal ({w_z:.3f} m).")

    def _on_canvas_water_level_changed(self, new_z: float):
        if len(self._stations) == 0:
            return
        idx = self._curr_idx
        dz = new_z - self._water_levels[idx]
        self._water_levels[idx] = new_z
        self._water_z_left[idx] += dz
        self._water_z_right[idx] += dz
        self._locked_mask[idx] = True
        self._z_spin.blockSignals(True)
        self._z_spin.setValue(new_z)
        self._z_spin.blockSignals(False)
        self._lock_btn.setChecked(True)
        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _on_canvas_water_section_changed(
        self, z_center: float, z_left: float, z_right: float, left_off: float, right_off: float
    ):
        if len(self._stations) == 0:
            return
        idx = self._curr_idx
        self._water_levels[idx] = z_center
        self._water_z_left[idx] = z_left
        self._water_z_right[idx] = z_right
        self._left_offsets[idx] = left_off
        self._right_offsets[idx] = right_off
        self._locked_mask[idx] = True

        self._z_spin.blockSignals(True)
        self._z_spin.setValue(z_center)
        self._z_spin.blockSignals(False)

        self._lock_btn.blockSignals(True)
        self._lock_btn.setChecked(True)
        self._lock_btn.setText("★ Approved Anchor (Locked)")
        self._lock_btn.setStyleSheet("font-weight: bold; background-color: #d29922; color: black;")
        self._lock_btn.blockSignals(False)

        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _on_spin_z_changed(self, val: float):
        if len(self._stations) == 0:
            return
        idx = self._curr_idx
        dz = val - self._water_levels[idx]
        self._water_levels[idx] = val
        self._water_z_left[idx] += dz
        self._water_z_right[idx] += dz
        self._locked_mask[idx] = True
        self._canvas.set_water_geometry(
            z_center=val,
            z_left=self._water_z_left[idx],
            z_right=self._water_z_right[idx],
            left_off=self._left_offsets[idx],
            right_off=self._right_offsets[idx],
            is_locked=True,
        )
        self._lock_btn.setChecked(True)
        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _on_toggle_lock(self, checked: bool):
        if len(self._stations) == 0:
            return
        idx = self._curr_idx
        self._locked_mask[idx] = checked
        self._canvas.set_water_geometry(
            z_center=self._water_levels[idx],
            z_left=self._water_z_left[idx],
            z_right=self._water_z_right[idx],
            left_off=self._left_offsets[idx],
            right_off=self._right_offsets[idx],
            is_locked=checked,
        )
        self._lock_btn.setText("★ Approved Anchor (Locked)" if checked else "☆ Lock as Approved Anchor")
        self._lock_btn.setStyleSheet(
            "font-weight: bold; background-color: #d29922; color: black;"
            if checked else ""
        )
        self._ribbon.set_data(len(self._stations), self._curr_idx, self._locked_mask)

    def _apply_interpolation(self):
        if len(self._stations) < 2:
            return
        old_levels = self._water_levels.copy()
        self._water_levels = interpolate_anchor_sections(
            self._stations, self._water_levels, self._locked_mask,
            cached_sections=self._cached_sections, z_search_window=1.5,
        )
        from ..centerline_wsm import detect_embankment_extents
        bank_margin = self._embank_spin.value()
        for i in range(len(self._stations)):
            if not self._locked_mask[i]:
                self._water_z_left[i] = self._water_levels[i]
                self._water_z_right[i] = self._water_levels[i]
                sec = self._cached_sections[i]
                if sec is not None and len(sec.get("offset", [])) > 0:
                    l_off, r_off = detect_embankment_extents(
                        sec["offset"], sec["z"], self._water_levels[i],
                        corridor_width=self._corridor_width, margin=bank_margin,
                    )
                    self._left_offsets[i] = l_off
                    self._right_offsets[i] = r_off
            else:
                # Preserve intentional cross-stream tilt
                if i < len(old_levels) and not np.isnan(old_levels[i]):
                    dz = self._water_levels[i] - old_levels[i]
                    self._water_z_left[i] += dz
                    self._water_z_right[i] += dz

    def _on_interpolate_anchors(self):
        if len(self._stations) < 2:
            return
        self._status_label.setText("Re-visiting in-between sections and fitting to real point cloud data...")
        self._apply_interpolation()
        self._update_current_view()
        self._status_label.setText("Re-evaluated intermediate sections against point cloud data between approved anchors.")

    def _insert_station_state(self, s_val: float, w_z: float, sec: dict) -> int:
        from ..centerline_wsm import insert_custom_station, detect_embankment_extents
        bank_margin = self._embank_spin.value()
        if sec is not None and len(sec.get("offset", [])) > 0:
            l_off, r_off = detect_embankment_extents(
                sec["offset"], sec["z"], w_z,
                corridor_width=self._corridor_width, margin=bank_margin,
            )
        else:
            half_w = self._corridor_width * 0.5
            l_off, r_off = -half_w, half_w

        n_prev = len(self._stations)
        self._stations, self._water_levels, self._locked_mask, self._cached_sections, new_idx = insert_custom_station(
            self._stations, self._water_levels, self._locked_mask, self._cached_sections,
            new_s=s_val, new_z=w_z, new_sec_data=sec,
        )

        if len(self._stations) == n_prev:
            self._water_z_left[new_idx] = w_z
            self._water_z_right[new_idx] = w_z
            self._left_offsets[new_idx] = l_off
            self._right_offsets[new_idx] = r_off
        else:
            self._water_z_left = np.insert(self._water_z_left, new_idx, w_z)
            self._water_z_right = np.insert(self._water_z_right, new_idx, w_z)
            self._left_offsets = np.insert(self._left_offsets, new_idx, l_off)
            self._right_offsets = np.insert(self._right_offsets, new_idx, r_off)

        return new_idx

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

        new_idx = self._insert_station_state(s_val, w_z, sec)
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

        new_idx = self._insert_station_state(s_val, w_z, sec)
        self._go_to_station(new_idx)
        self._status_label.setText(
            f"Inserted custom section from drawn map line at station {s_val:.1f} m as a Locked Anchor."
        )

    def _on_remove_current_section(self):
        if len(self._stations) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "Cannot remove the only remaining section.")
            return

        idx = self._curr_idx
        s_val = float(self._stations[idx])

        from ..centerline_wsm import remove_station
        self._stations, self._water_levels, self._locked_mask, self._cached_sections, new_idx = remove_station(
            self._stations, self._water_levels, self._locked_mask, self._cached_sections, idx
        )
        if idx < len(self._water_z_left):
            self._water_z_left = np.delete(self._water_z_left, idx)
            self._water_z_right = np.delete(self._water_z_right, idx)
            self._left_offsets = np.delete(self._left_offsets, idx)
            self._right_offsets = np.delete(self._right_offsets, idx)

        # If removing the first or last section, ensure the new boundary stays locked
        if len(self._stations) >= 2:
            if idx == 0 and not self._locked_mask[0]:
                self._locked_mask[0] = True
            elif idx >= len(self._stations) and not self._locked_mask[-1]:
                self._locked_mask[-1] = True

        # Re-interpolate intermediate sections immediately to smoothly bridge the gap
        self._apply_interpolation()

        self._curr_idx = new_idx
        self._update_current_view()
        self._status_label.setText(
            f"Removed section at station {s_val:.1f} m. {len(self._stations)} sections remaining."
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
            if len(self._stations) > 0 and 0 <= self._curr_idx < len(self._locked_mask):
                self._on_toggle_lock(not self._locked_mask[self._curr_idx])
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._on_remove_current_section()
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
            # Make sure interpolation and all bank/tilt arrays are synchronized before rasterizing
            self._apply_interpolation()

            surf, georef = rasterize_water_surface_model(
                centerline=self._centerline,
                stations=self._stations,
                water_levels=self._water_levels,
                corridor_width=self._corridor_width,
                output_path=fn,
                resolution=1.0,
                data_epsg=self._data_epsg,
                left_offsets=self._left_offsets if len(self._left_offsets) == len(self._stations) else None,
                right_offsets=self._right_offsets if len(self._right_offsets) == len(self._stations) else None,
                water_levels_left=self._water_z_left if len(self._water_z_left) == len(self._stations) else None,
                water_levels_right=self._water_z_right if len(self._water_z_right) == len(self._stations) else None,
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

    def _on_save_sections(self):
        if len(self._stations) == 0:
            QMessageBox.warning(self, "No Sections", "No cross-sections to save.")
            return

        default_fn = str(self._project_dir / "water_surface_profile.json")
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Save Water Surface Profile Sections",
            default_fn,
            "JSON Profile Files (*.json *.wsp.json);;All Files (*)",
        )
        if not fn:
            return

        try:
            save_water_surface_sections(
                file_path=fn,
                stations=self._stations,
                water_levels=self._water_levels,
                locked_mask=self._locked_mask,
                water_levels_left=self._water_z_left,
                water_levels_right=self._water_z_right,
                left_offsets=self._left_offsets,
                right_offsets=self._right_offsets,
                corridor_width=self._corridor_width,
                section_spacing=self._section_spacing,
                bank_margin=self._embank_spin.value(),
                data_epsg=self._data_epsg,
                centerline=self._centerline,
            )
            self._status_label.setText(f"Saved {len(self._stations)} sections to {Path(fn).name}.")
            QMessageBox.information(
                self, "Profile Saved",
                f"Water surface profile saved successfully:\n{fn}\n\n"
                f"Contains {len(self._stations)} cross-sections with water levels, anchors, tilt, and embankment offsets."
            )
        except Exception as e:
            logger.exception("Failed to save water surface profile")
            QMessageBox.critical(self, "Save Error", f"Could not save profile: {e}")

    def _on_load_sections(self):
        default_fn = str(self._project_dir / "water_surface_profile.json")
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Load Water Surface Profile Sections",
            default_fn,
            "JSON Profile Files (*.json *.wsp.json);;All Files (*)",
        )
        if not fn:
            return

        try:
            data = load_water_surface_sections(fn)

            # Restore centerline if present in profile file
            if data.get("centerline_vertices") is not None and len(data["centerline_vertices"]) >= 2:
                verts = data["centerline_vertices"]
                self._full_centerline = RiverCenterline(verts)
                self._centerline = self._full_centerline
                self._centerline_path_edit.setText(f"[Loaded Profile: {len(verts)} vertices, {self._centerline.total_length:.1f} m]")

            # Restore parameters
            self._corridor_width = data["corridor_width"]
            self._section_spacing = data["section_spacing"]
            self._width_spin.blockSignals(True)
            self._width_spin.setValue(self._corridor_width)
            self._width_spin.blockSignals(False)
            self._spacing_spin.blockSignals(True)
            self._spacing_spin.setValue(self._section_spacing)
            self._spacing_spin.blockSignals(False)
            self._embank_spin.blockSignals(True)
            self._embank_spin.setValue(data["bank_margin"])
            self._embank_spin.blockSignals(False)

            if data.get("data_epsg") and not self._data_epsg:
                self._data_epsg = data["data_epsg"]

            # Restore section arrays
            self._stations = data["stations"]
            self._water_levels = data["water_levels"]
            self._water_z_left = data["water_levels_left"]
            self._water_z_right = data["water_levels_right"]
            self._left_offsets = data["left_offsets"]
            self._right_offsets = data["right_offsets"]
            self._locked_mask = data["locked_mask"]
            self._cached_sections = [None] * len(self._stations)

            self._curr_idx = 0
            self._save_btn.setEnabled(True)
            self._export_btn.setEnabled(True)
            self._update_current_view()
            self._status_label.setText(f"Loaded {len(self._stations)} sections from {Path(fn).name}.")
            QMessageBox.information(
                self, "Profile Loaded",
                f"Loaded {len(self._stations)} cross-sections from:\n{fn}\n\n"
                f"Corridor width: {self._corridor_width:.1f} m, Bank margin: {data['bank_margin']:.1f} m\n"
                f"Anchors locked: {int(np.sum(self._locked_mask))} of {len(self._stations)}."
            )
        except Exception as e:
            logger.exception("Failed to load water surface profile")
            QMessageBox.critical(self, "Load Error", f"Could not load profile: {e}")

