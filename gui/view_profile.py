"""
LiDAR Workbench — Profile View (2D Side View).

Displays a 2D scatter plot of points along a profile line, coloured
by ASPRS class, with a DTM reference line and interactive selection
tools (line-above/below, rectangle, brush).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..config import get_class_color

logger = logging.getLogger("lidar_workbench.gui.view_profile")

# Selection tool modes
SELECT_NONE = "none"
SELECT_LINE_ABOVE = "line_above"
SELECT_LINE_BELOW = "line_below"
SELECT_RECTANGLE = "rectangle"
SELECT_BRUSH = "brush"


class ViewProfile(QWidget):
    """
    2D profile side-view widget.

    Shows distance-on-profile (X axis) vs. elevation (Y axis).
    Supports four selection modes and reports selection masks.

    Signals:
        selection_changed(mask: np.ndarray):
            Emitted when the user completes a selection operation.
            The mask is a boolean array over the profile points.
        selection_mode_changed(mode: str):
            Emitted when the active selection tool changes.
    """

    selection_changed = Signal(np.ndarray)
    selection_mode_changed = Signal(str)
    profile_width_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

        # Data
        self._distances: Optional[np.ndarray] = None
        self._elevations: Optional[np.ndarray] = None
        self._classifications: Optional[np.ndarray] = None
        self._dtm_distances: Optional[np.ndarray] = None
        self._dtm_elevations: Optional[np.ndarray] = None

        # View transform
        self._offset_x: float = 0.0    # distance offset
        self._offset_y: float = 0.0    # elevation offset
        self._scale_x: float = 1.0     # pixels per meter
        self._scale_y: float = 1.0

        # Selection state
        self._select_mode: str = SELECT_BRUSH
        self._selecting: bool = False
        self._sel_start: Optional[Tuple[float, float]] = None
        self._sel_end: Optional[Tuple[float, float]] = None
        self._brush_radius: float = 2.0
        self._current_mask: Optional[np.ndarray] = None

        # Width-adjust mode: after profile is loaded, scroll adjusts width
        # until the user clicks to confirm and enter selection mode.
        self._width_adjusting: bool = False
        self._total_width: float = 5.0   # current corridor width (m)

    # ── public API ─────────────────────────────────────────────────

    def set_profile_data(
        self,
        distances: np.ndarray,
        elevations: np.ndarray,
        classifications: np.ndarray,
    ) -> None:
        """
        Load profile point data.  After loading, the view enters
        "width-adjust" mode: scroll adjusts corridor width, and the
        first left-click confirms the width and enters selection mode.

        Args:
            distances:       Distance along profile (meters).
            elevations:      Point elevations.
            classifications: ASPRS class codes.
        """
        self._distances = distances
        self._elevations = elevations
        self._classifications = classifications
        self._current_mask = None
        self._width_adjusting = True  # enter width-adjust mode
        self._fit_view()
        self.update()

    def set_dtm_reference(
        self,
        dtm_distances: np.ndarray,
        dtm_elevations: np.ndarray,
    ) -> None:
        """
        Set the DTM reference line data.

        Args:
            dtm_distances:  Distance samples along the profile.
            dtm_elevations: DTM elevations at each sample.
        """
        self._dtm_distances = dtm_distances
        self._dtm_elevations = dtm_elevations
        self.update()

    def set_profile_width(self, width: float) -> None:
        """Set the initial corridor width (called before profile data is loaded)."""
        self._total_width = max(0.5, width)

    def set_selection_mode(self, mode: str) -> None:
        """Set the active selection tool. Exits width-adjust mode if active."""
        if mode not in (SELECT_NONE, SELECT_LINE_ABOVE, SELECT_LINE_BELOW,
                         SELECT_RECTANGLE, SELECT_BRUSH):
            logger.warning("Unknown selection mode: %s", mode)
            return
        self._width_adjusting = False  # confirm width when user picks a tool
        self._select_mode = mode
        self.selection_mode_changed.emit(mode)
        logger.debug("Profile selection mode: %s", mode)

    def set_brush_radius(self, radius: float) -> None:
        """Set the brush selection radius in meters."""
        self._brush_radius = max(0.1, radius)

    def set_selection_mask(self, mask: np.ndarray) -> None:
        """Apply an externally-computed selection mask."""
        self._current_mask = mask
        self.update()

    def clear(self) -> None:
        """Clear all data."""
        self._distances = None
        self._elevations = None
        self._classifications = None
        self._dtm_distances = None
        self._dtm_elevations = None
        self._current_mask = None
        self.update()

    # ── coordinate transforms ──────────────────────────────────────

    def _world_to_widget(self, d: float, z: float) -> QPointF:
        px = (d - self._offset_x) * self._scale_x + self.width() / 2
        py = -(z - self._offset_y) * self._scale_y + self.height() / 2
        return QPointF(px, py)

    def _widget_to_world(self, px: float, py: float) -> Tuple[float, float]:
        d = (px - self.width() / 2) / self._scale_x + self._offset_x
        z = -(py - self.height() / 2) / self._scale_y + self._offset_y
        return d, z

    def _fit_view(self) -> None:
        """Auto-fit to data bounds."""
        if self._distances is None or len(self._distances) == 0:
            return
        d_min, d_max = self._distances.min(), self._distances.max()
        z_min, z_max = self._elevations.min(), self._elevations.max()

        pad_d = (d_max - d_min) * 0.1 if d_max > d_min else 1.0
        pad_z = (z_max - z_min) * 0.1 if z_max > z_min else 1.0

        self._offset_x = (d_min + d_max) / 2
        self._offset_y = (z_min + z_max) / 2

        w = self.width() or 1
        h = self.height() or 1
        self._scale_x = w / (d_max - d_min + 2 * pad_d) * 0.9 if (d_max - d_min) > 0 else 1.0
        self._scale_y = h / (z_max - z_min + 2 * pad_z) * 0.9 if (z_max - z_min) > 0 else 1.0

    # ── painting ───────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        # Axes
        painter.setPen(QPen(QColor("#555"), 1))
        origin = self._world_to_widget(0, 0)
        painter.drawLine(0, int(origin.y()), self.width(), int(origin.y()))
        painter.drawLine(int(origin.x()), 0, int(origin.x()), self.height())

        # DTM reference line
        if self._dtm_distances is not None and self._dtm_elevations is not None:
            pen = QPen(QColor("#8B4513"), 2)
            painter.setPen(pen)
            path = QPainterPath()
            pt = self._world_to_widget(self._dtm_distances[0], self._dtm_elevations[0])
            path.moveTo(pt)
            for i in range(1, len(self._dtm_distances)):
                pt = self._world_to_widget(self._dtm_distances[i], self._dtm_elevations[i])
                path.lineTo(pt)
            painter.drawPath(path)

        # Point cloud
        if self._distances is not None and len(self._distances) > 0:
            n = len(self._distances)
            step = max(1, n // 30_000)

            for i in range(0, n, step):
                pt = self._world_to_widget(self._distances[i], self._elevations[i])
                cls = self._classifications[i] if self._classifications is not None else 0
                r, g, b = get_class_color(int(cls))

                if self._current_mask is not None and self._current_mask[i]:
                    # Highlight selected points
                    color = QColor(255, 50, 50, 220)
                    radius = 3.5
                else:
                    color = QColor(int(r * 255), int(g * 255), int(b * 255), 200)
                    radius = 2.0

                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(pt, radius, radius)

        # Selection preview (while drawing)
        if self._selecting and self._sel_start is not None and self._sel_end is not None:
            painter.setPen(QPen(QColor("#ff4444"), 1, Qt.DashLine))

            if self._select_mode in (SELECT_LINE_ABOVE, SELECT_LINE_BELOW):
                p1 = self._world_to_widget(*self._sel_start)
                p2 = self._world_to_widget(*self._sel_end)
                painter.drawLine(p1, p2)
            elif self._select_mode == SELECT_RECTANGLE:
                p1 = self._world_to_widget(*self._sel_start)
                p2 = self._world_to_widget(*self._sel_end)
                rect = QRectF(p1, p2).normalized()
                painter.drawRect(rect)
            elif self._select_mode == SELECT_BRUSH:
                pt = self._world_to_widget(*self._sel_end)
                r = self._brush_radius * self._scale_x
                painter.drawEllipse(pt, r, r)

        # Width-adjust mode overlay
        if self._width_adjusting:
            painter.setPen(QColor("#ffcc00"))
            painter.drawText(
                10, 25,
                f"Width: {self._total_width:.1f} m — scroll to adjust, click to confirm"
            )

        painter.end()

    # ── mouse events ───────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._width_adjusting:
            # Any click exits width-adjust mode → enter selection mode
            self._width_adjusting = False
            self.profile_width_changed.emit(self._total_width)  # final width
            self.update()
            return  # don't start selection on the confirming click

        if event.button() == Qt.LeftButton:
            wx, wy = self._widget_to_world(event.position().x(), event.position().y())
            self._selecting = True
            self._sel_start = (wx, wy)
            self._sel_end = (wx, wy)

            # Brush: select immediately
            if self._select_mode == SELECT_BRUSH:
                self._compute_brush_selection(wx, wy)
                self._selecting = False

            self.update()
        elif event.button() == Qt.RightButton:
            # Cancel selection
            self._selecting = False
            self._current_mask = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        wx, wy = self._widget_to_world(event.position().x(), event.position().y())
        if self._selecting:
            self._sel_end = (wx, wy)
            self.update()
        elif self._select_mode == SELECT_BRUSH and event.buttons() & Qt.LeftButton:
            # Continuous brush painting
            self._compute_brush_selection(wx, wy, additive=True)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._selecting:
            return

        wx, wy = self._widget_to_world(event.position().x(), event.position().y())
        self._sel_end = (wx, wy)
        self._selecting = False

        if self._select_mode == SELECT_LINE_ABOVE:
            self._compute_line_selection(above=True)
        elif self._select_mode == SELECT_LINE_BELOW:
            self._compute_line_selection(above=False)
        elif self._select_mode == SELECT_RECTANGLE:
            self._compute_rect_selection()

        # Don't emit on brush — it's continuous
        if self._select_mode != SELECT_BRUSH and self._current_mask is not None:
            self.selection_changed.emit(self._current_mask.copy())

        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._width_adjusting:
            # In width-adjust mode: scroll changes corridor width
            direction = 1.0 if event.angleDelta().y() > 0 else -1.0
            self._total_width = max(0.5, self._total_width + direction * 1.0)
            self.profile_width_changed.emit(self._total_width)
        else:
            # Normal mode: scroll zooms
            factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            self._scale_x *= factor
            self._scale_y *= factor
            self._scale_x = max(0.001, min(self._scale_x, 10000.0))
            self._scale_y = max(0.001, min(self._scale_y, 10000.0))
            self.update()

    # ── selection computation ──────────────────────────────────────

    def _compute_line_selection(self, above: bool) -> None:
        if self._distances is None or self._sel_start is None or self._sel_end is None:
            return

        d1, z1 = self._sel_start
        d2, z2 = self._sel_end

        if abs(d2 - d1) < 1e-9:
            mask = self._distances >= d1 if above else self._distances < d1
        else:
            m = (z2 - z1) / (d2 - d1)
            b = z1 - m * d1
            line_z = m * self._distances + b
            mask = self._elevations > line_z if above else self._elevations < line_z

        self._current_mask = mask
        logger.debug("Line selection: %d points %s line", mask.sum(),
                      "above" if above else "below")

    def _compute_rect_selection(self) -> None:
        if self._distances is None or self._sel_start is None or self._sel_end is None:
            return

        d1, z1 = self._sel_start
        d2, z2 = self._sel_end
        d_min, d_max = sorted([d1, d2])
        z_min, z_max = sorted([z1, z2])

        self._current_mask = (
            (self._distances >= d_min)
            & (self._distances <= d_max)
            & (self._elevations >= z_min)
            & (self._elevations <= z_max)
        )
        logger.debug("Rect selection: %d points", self._current_mask.sum())

    def _compute_brush_selection(
        self, d: float, z: float, additive: bool = False
    ) -> None:
        if self._distances is None:
            return

        d_dist = self._distances - d
        e_dist = self._elevations - z
        new_mask = np.sqrt(d_dist * d_dist + e_dist * e_dist) <= self._brush_radius

        if additive and self._current_mask is not None:
            self._current_mask = self._current_mask | new_mask
        else:
            self._current_mask = new_mask

        self.selection_changed.emit(self._current_mask.copy())
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep view centered; don't re-fit (preserve user zoom)
