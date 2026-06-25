"""
LiDAR Workbench — DTM View (2D Top-Down).

Displays a colour-coded DTM raster with overlaid point classes and
supports interactive profile-line drawing.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..config import get_class_color
from ..dtm_generator import generate_dtm

logger = logging.getLogger("lidar_workbench.gui.view_dtm")


class ViewDTM(QWidget):
    """
    2D top-down DTM view with point overlay and profile-line interaction.

    The user can:
        - Pan and zoom the DTM raster.
        - See ground points colour-coded by class.
        - Draw a profile line by click-dragging.

    Signals:
        profile_line_defined(start_xy, end_xy):
            Emitted when the user finishes drawing a profile line.
        point_hovered(x, y, class_code):
            Emitted when the mouse hovers over a point.
    """

    profile_line_defined = Signal(tuple, tuple)
    point_hovered = Signal(float, float, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

        # Data
        self._dtm_grid_x: Optional[np.ndarray] = None
        self._dtm_grid_y: Optional[np.ndarray] = None
        self._dtm_grid_z: Optional[np.ndarray] = None
        self._dtm_bbox: Tuple[float, float, float, float] = (0, 0, 1, 1)

        self._points_x: Optional[np.ndarray] = None
        self._points_y: Optional[np.ndarray] = None
        self._points_class: Optional[np.ndarray] = None

        # View transform
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._scale: float = 1.0  # pixels per CRS unit

        # Profile drawing state
        self._drawing_profile: bool = False
        self._profile_start: Optional[Tuple[float, float]] = None
        self._profile_end: Optional[Tuple[float, float]] = None

        # Profile corridor (shown after profile is defined)
        self._corridor_start: Optional[Tuple[float, float]] = None
        self._corridor_end: Optional[Tuple[float, float]] = None
        self._corridor_width: float = 5.0

        # Rendered DTM image (cached)
        self._dtm_pixmap: Optional[QPixmap] = None

    # ── public API ─────────────────────────────────────────────────

    def load_points(
        self,
        data: dict,
        ground_class: int = 2,
    ) -> None:
        """
        Load point data for 2D top-down display.

        DTM generation is **not** performed here — it is a batch operation
        done after classification.  This view shows a simple 2D point
        scatter coloured by elevation for fast interactive browsing.

        Args:
            data: Dict with keys ``x, y, z, classification``.
            ground_class: ASPRS code for ground (default 2).  Unused during
                          interactive viewing; used only by batch DTM export.
        """
        xs = data["x"]
        ys = data["y"]
        zs = data["z"]
        cls = data.get("classification", np.zeros(len(xs), dtype=np.uint8))

        self._points_x = xs
        self._points_y = ys
        self._points_z = zs
        self._points_class = cls

        # Subsample for fast 2D scatter rendering
        n = len(xs)
        if n > 200_000:
            step = max(1, n // 200_000)
            self._points_x = xs[::step]
            self._points_y = ys[::step]
            self._points_z = zs[::step]
            self._points_class = cls[::step]

        # Clear any cached DTM
        self._dtm_grid_x = None
        self._dtm_grid_y = None
        self._dtm_grid_z = None

        self._render_scatter()
        self._fit_view()
        self.update()

    def clear(self) -> None:
        """Clear all data."""
        self._dtm_grid_x = None
        self._dtm_grid_y = None
        self._dtm_grid_z = None
        self._points_x = None
        self._points_y = None
        self._points_class = None
        self._dtm_pixmap = None
        self._profile_start = None
        self._profile_end = None
        self._corridor_start = None
        self._corridor_end = None
        self.update()

    def set_profile_corridor(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        width: float,
    ) -> None:
        """
        Set the profile corridor to display as a shaded band on the DTM.

        Args:
            start, end: Profile line endpoints in CRS coords.
            width:      Corridor full-width in meters.
        """
        self._corridor_start = start
        self._corridor_end = end
        self._corridor_width = width
        self.update()

    # ── coordinate transforms ──────────────────────────────────────

    def _world_to_widget(self, wx: float, wy: float) -> QPointF:
        """Convert world coordinates to widget pixel coordinates."""
        px = (wx - self._offset_x) * self._scale + self.width() / 2
        py = (wy - self._offset_y) * self._scale + self.height() / 2
        return QPointF(px, py)

    def _widget_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert widget pixel coordinates to world coordinates."""
        wx = (px - self.width() / 2) / self._scale + self._offset_x
        wy = (py - self.height() / 2) / self._scale + self._offset_y
        return wx, wy

    def _fit_view(self) -> None:
        """Fit the DTM bbox to the widget."""
        if self._dtm_grid_x is None or self._dtm_grid_y is None:
            return
        x_min, x_max, y_min, y_max = self._dtm_bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        self._offset_x = cx
        self._offset_y = cy

        w = self.width() or 1
        h = self.height() or 1
        scale_x = w / (x_max - x_min) * 0.9 if (x_max - x_min) > 0 else 1.0
        scale_y = h / (y_max - y_min) * 0.9 if (y_max - y_min) > 0 else 1.0
        self._scale = min(scale_x, scale_y)

    # ── rendering ──────────────────────────────────────────────────

    def _render_scatter(self) -> None:
        """Pre-render a fast 2D height-coloured point scatter as a QPixmap."""
        if self._points_x is None or len(self._points_x) == 0:
            self._dtm_pixmap = None
            return

        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            w, h = 400, 300

        xs = self._points_x
        ys = self._points_y
        zs = self._points_z

        # Normalise coordinates to pixel space
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_pad = (x_max - x_min) * 0.02 or 1.0
        y_pad = (y_max - y_min) * 0.02 or 1.0
        px = ((xs - x_min + x_pad) / (x_max - x_min + 2 * x_pad) * (w - 1)).astype(np.int32)
        py = ((y_max - ys + y_pad) / (y_max - y_min + 2 * y_pad) * (h - 1)).astype(np.int32)
        # Clip
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

        # Height colour map
        z_min, z_max = float(zs.min()), float(zs.max())
        if z_max <= z_min:
            z_norm = np.full_like(zs, 0.5)
        else:
            z_norm = (zs - z_min) / (z_max - z_min)

        # RGB array
        img = np.zeros((h, w, 3), dtype=np.uint8)
        r = np.clip((z_norm - 0.5) * 4.0, 0, 1) + np.clip((z_norm - 0.75) * 4.0, 0, 1)
        g = np.clip(z_norm * 4.0, 0, 1) * (z_norm <= 0.5) + np.clip((1 - z_norm) * 4.0, 0, 1) * (z_norm > 0.5)
        b = np.clip((0.25 - z_norm) * 4.0, 0, 1) + np.clip((0.5 - z_norm) * 4.0, 0, 1) * (z_norm > 0.25)
        b = np.clip(b, 0, 1)
        cr = (r * 255).astype(np.uint8)
        cg = (g * 255).astype(np.uint8)
        cb = (b * 255).astype(np.uint8)

        img[py, px] = np.column_stack((cr, cg, cb))

        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        self._dtm_pixmap = QPixmap.fromImage(qimg.copy())

    def _render_dtm(self) -> None:
        """Pre-render the DTM raster as a QPixmap with hillshade relief."""
        if self._dtm_grid_z is None:
            self._dtm_pixmap = None
            return

        z = self._dtm_grid_z
        ny, nx = z.shape
        if ny < 2 or nx < 2:
            self._dtm_pixmap = None
            return

        z_valid = z[~np.isnan(z)]
        if len(z_valid) == 0:
            self._dtm_pixmap = None
            return

        z_min, z_max = z_valid.min(), z_valid.max()
        if z_max <= z_min:
            z_norm = np.full_like(z, 0.5, dtype=np.float64)
        else:
            z_norm = (z - z_min) / (z_max - z_min)

        # ── hillshade ───────────────────────────────────────────
        # Compute slope and aspect from the DTM grid
        # Cell size in CRS units (approximate)
        dx = (self._dtm_bbox[1] - self._dtm_bbox[0]) / max(nx - 1, 1)
        dy = (self._dtm_bbox[3] - self._dtm_bbox[2]) / max(ny - 1, 1)
        cell_size = min(dx, dy) or 1.0

        dz_dx = np.zeros_like(z)
        dz_dy = np.zeros_like(z)
        # Central differences for interior, forward/backward for edges
        dz_dx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * cell_size)
        dz_dx[:, 0] = (z[:, 1] - z[:, 0]) / cell_size
        dz_dx[:, -1] = (z[:, -1] - z[:, -2]) / cell_size

        dz_dy[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2 * cell_size)
        dz_dy[0, :] = (z[1, :] - z[0, :]) / cell_size
        dz_dy[-1, :] = (z[-1, :] - z[-2, :]) / cell_size

        # Slope (radians)
        slope = np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))

        # Aspect (radians, 0 = south, increasing east → standard GIS)
        aspect = np.arctan2(dz_dy, -dz_dx)
        aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)

        # Sun parameters (NW light, 45° above horizon)
        sun_azimuth = math.radians(315.0)   # NW
        sun_altitude = math.radians(45.0)   # 45° above horizon
        sun_zenith = math.pi / 2 - sun_altitude

        # Hillshade = cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth-aspect)
        hs = (np.cos(sun_zenith) * np.cos(slope)
              + np.sin(sun_zenith) * np.sin(slope)
              * np.cos(sun_azimuth - aspect))
        hs = np.where(np.isnan(z), 0.0, np.clip(hs, 0.0, 1.0))

        # ── combine elevation colour + hillshade ─────────────────
        # Elevation colours: green→yellow→brown
        t = z_norm
        r_el = np.clip(t * 180 + 40, 0, 255).astype(np.float64)
        g_el = np.clip((1 - t) * 160 + 40, 0, 255).astype(np.float64)
        b_el = np.clip((1 - t) * 100 + 20, 0, 255).astype(np.float64)

        # Blend: hillshade modulates brightness (50% base + 50% shaded)
        blend = 0.4 + 0.6 * hs
        r = np.clip(r_el * blend, 0, 255).astype(np.uint8)
        g = np.clip(g_el * blend, 0, 255).astype(np.uint8)
        b = np.clip(b_el * blend, 0, 255).astype(np.uint8)

        img = np.zeros((ny, nx, 4), dtype=np.uint8)
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        img[:, :, 3] = np.where(np.isnan(z), 0, 255).astype(np.uint8)

        qimg = QImage(img.data, nx, ny, QImage.Format_RGBA8888)
        self._dtm_pixmap = QPixmap.fromImage(qimg.copy())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        # Draw DTM raster
        if self._dtm_pixmap is not None and not self._dtm_pixmap.isNull():
            x_min, x_max, y_min, y_max = self._dtm_bbox
            top_left = self._world_to_widget(x_min, y_min)
            bottom_right = self._world_to_widget(x_max, y_max)
            target_rect = QRectF(top_left, bottom_right)
            painter.drawPixmap(target_rect.toRect(), self._dtm_pixmap)

        # Draw point overlay (if zoomed in enough)
        if self._points_x is not None and self._scale > 0.05:
            painter.setPen(Qt.NoPen)
            n = len(self._points_x)
            # Downsample for performance
            step = max(1, n // 20_000)
            for i in range(0, n, step):
                pt = self._world_to_widget(self._points_x[i], self._points_y[i])
                cls = self._points_class[i] if self._points_class is not None else 0
                r, g, b = get_class_color(int(cls))
                color = QColor(int(r * 255), int(g * 255), int(b * 255), 180)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(pt, 2, 2)

        # Draw profile line
        if self._profile_start is not None:
            pen = QPen(QColor("#ff4444"), 2, Qt.DashLine)
            painter.setPen(pen)
            p1 = self._world_to_widget(*self._profile_start)
            if self._profile_end is not None:
                p2 = self._world_to_widget(*self._profile_end)
                painter.drawLine(p1, p2)

        # Draw corridor band (semi-transparent shaded band)
        if self._corridor_start is not None and self._corridor_end is not None:
            sx, sy = self._corridor_start
            ex, ey = self._corridor_end
            dx, dy = ex - sx, ey - sy
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                # Perpendicular unit vector (rotate 90°)
                px, py = -dy / length, dx / length
                half_w = self._corridor_width / 2.0

                # Four corners of the corridor band
                p1 = self._world_to_widget(sx + px * (-half_w), sy + py * (-half_w))
                p2 = self._world_to_widget(sx + px * half_w, sy + py * half_w)
                p3 = self._world_to_widget(ex + px * half_w, ey + py * half_w)
                p4 = self._world_to_widget(ex + px * (-half_w), ey + py * (-half_w))

                path = QPainterPath()
                path.moveTo(p1)
                path.lineTo(p2)
                path.lineTo(p3)
                path.lineTo(p4)
                path.closeSubpath()

                # Fill with semi-transparent light blue
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(80, 140, 220, 50))
                painter.drawPath(path)

                # Outline
                painter.setPen(QPen(QColor(80, 140, 220, 140), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)

        # Crosshair at center
        pen = QPen(QColor("#444"), 1, Qt.DotLine)
        painter.setPen(pen)
        cx = self.width() / 2
        cy = self.height() / 2
        painter.drawLine(cx - 10, cy, cx + 10, cy)
        painter.drawLine(cx, cy - 10, cx, cy + 10)

        painter.end()

    # ── mouse events ───────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            wx, wy = self._widget_to_world(event.position().x(), event.position().y())
            self._drawing_profile = True
            self._profile_start = (wx, wy)
            self._profile_end = (wx, wy)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        wx, wy = self._widget_to_world(event.position().x(), event.position().y())
        if self._drawing_profile:
            self._profile_end = (wx, wy)
            self.update()
        else:
            # Emit hover info
            if self._points_x is not None:
                self.point_hovered.emit(wx, wy, 0)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drawing_profile:
            self._drawing_profile = False
            wx, wy = self._widget_to_world(event.position().x(), event.position().y())
            self._profile_end = (wx, wy)
            if self._profile_start is not None and self._profile_end is not None:
                dx = self._profile_end[0] - self._profile_start[0]
                dy = self._profile_end[1] - self._profile_start[1]
                if dx * dx + dy * dy > 1.0:  # minimum length
                    self.profile_line_defined.emit(self._profile_start, self._profile_end)
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom in/out."""
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self._scale *= factor
        self._scale = max(0.001, min(self._scale, 1000.0))
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_dtm()
