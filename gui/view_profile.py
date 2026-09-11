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

logger = logging.getLogger("lidar_workbench.gui.view_profile")

# Selection tool modes
SELECT_NONE = "none"
SELECT_LINE_ABOVE = "line_above"
SELECT_LINE_BELOW = "line_below"
SELECT_RECTANGLE = "rectangle"
SELECT_BRUSH = "brush"
SELECT_RECT_BRUSH = "rect_brush"

_FL_PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.50, 0.90], [0.10, 0.80, 0.10], [0.90, 0.60, 0.10],
    [0.70, 0.10, 0.90], [0.10, 0.80, 0.80], [0.90, 0.10, 0.70], [0.60, 0.60, 0.10],
    [0.90, 0.50, 0.50], [0.20, 0.60, 0.20], [0.50, 0.50, 0.90], [0.80, 0.80, 0.20],
])


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
    brush_size_changed = Signal(float)
    rect_size_changed = Signal(float, float)
    point_hovered = Signal(int, float, float, float, int, int)  # idx, x, y, z, class, intensity
    point_picked = Signal(int, float, float, float, int, int)   # idx, x, y, z, class, intensity (click)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Data
        self._distances: Optional[np.ndarray] = None
        self._elevations: Optional[np.ndarray] = None
        self._classifications: Optional[np.ndarray] = None
        self._intensities: Optional[np.ndarray] = None
        self._return_numbers: Optional[np.ndarray] = None
        self._point_source_ids: Optional[np.ndarray] = None
        self._profile_indices: Optional[np.ndarray] = None  # indices into the parent tile
        self._xs: Optional[np.ndarray] = None  # original X coords
        self._ys: Optional[np.ndarray] = None  # original Y coords
        self._zs: Optional[np.ndarray] = None  # original Z coords
        self._dtm_distances: Optional[np.ndarray] = None
        self._dtm_elevations: Optional[np.ndarray] = None
        self._class_visibility: Optional[np.ndarray] = None

        # View transform
        self._offset_x: float = 0.0    # distance offset
        self._offset_y: float = 0.0    # elevation offset
        self._scale_x: float = 1.0     # pixels per meter
        self._scale_y: float = 1.0

        # Colour mode: class, height, intensity, return_number, flightline
        self._colour_mode: str = "class"

        # Selection state: fine default sizes (0.6m radius, 1.0m x 0.4m rect)
        self._select_mode: str = SELECT_BRUSH
        self._selecting: bool = False
        self._sel_start: Optional[Tuple[float, float]] = None
        self._sel_end: Optional[Tuple[float, float]] = None
        self._brush_radius: float = 0.6
        self._rect_width: float = 0.5    # half-width in meters for rect_brush (1.0m total)
        self._rect_height: float = 0.2   # half-height in meters for rect_brush (0.4m total)
        self._current_mask: Optional[np.ndarray] = None
        self._cursor_world: Optional[Tuple[float, float]] = None  # mouse pos in world coords

        # Base rendering cache for buttery smooth mouse movement
        self._base_pixmap: Optional[QPixmap] = None
        self._base_dirty: bool = True
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None

        # Width-adjust mode (optional, Ctrl+Scroll to adjust corridor width)
        self._width_adjusting: bool = False
        self._total_width: float = 5.0   # current corridor width (m)

        # Point Info tool state
        self._point_info_active: bool = False
        self._picked_point: Optional[Tuple[float, float, float, int, int, int, float]] = None
        # (x, y, z, class, intensity, index, distance_on_profile)

    def _invalidate_base(self) -> None:
        self._base_dirty = True
        self.update()

    @property
    def point_info_active(self) -> bool:
        """Whether the Point Info tool is active."""
        return self._point_info_active

    @point_info_active.setter
    def point_info_active(self, value: bool) -> None:
        """Enable/disable Point Info tool. Clears the picked-point marker when off."""
        self._point_info_active = value
        if not value:
            self._picked_point = None
            self.update()

    # ── public API ─────────────────────────────────────────────────

    def set_profile_data(
        self,
        distances: np.ndarray,
        elevations: np.ndarray,
        classifications: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        xs: Optional[np.ndarray] = None,
        ys: Optional[np.ndarray] = None,
        zs: Optional[np.ndarray] = None,
        return_numbers: Optional[np.ndarray] = None,
        point_source_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Load profile point data.  After loading, the view enters
        "width-adjust" mode: scroll adjusts corridor width, and the
        first left-click confirms the width and enters selection mode.
        """
        self._distances = distances
        self._elevations = elevations
        self._classifications = classifications
        self._intensities = intensities
        self._return_numbers = return_numbers
        self._point_source_ids = point_source_ids
        self._profile_indices = indices
        self._xs = xs
        self._ys = ys
        self._zs = zs
        self._current_mask = None
        self._width_adjusting = False  # default to normal zoom mode
        self._fit_view()
        self._invalidate_base()

    def set_colour_mode(self, mode: str) -> None:
        """Set point cloud colouring mode (class, height, intensity, return_number, flightline)."""
        self._colour_mode = mode
        self._invalidate_base()

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
        self._invalidate_base()

    def set_profile_width(self, width: float) -> None:
        """Set the initial corridor width (called before profile data is loaded)."""
        self._total_width = max(0.5, width)

    def set_selection_mode(self, mode: str) -> None:
        """Set the active selection tool."""
        if mode not in (SELECT_NONE, SELECT_LINE_ABOVE, SELECT_LINE_BELOW,
                         SELECT_RECTANGLE, SELECT_BRUSH, SELECT_RECT_BRUSH):
            logger.warning("Unknown selection mode: %s", mode)
            return
        self._width_adjusting = False
        self._select_mode = mode
        self._cursor_world = None  # reset cursor on mode change
        self.selection_mode_changed.emit(mode)
        logger.debug("Profile selection mode: %s", mode)

    def set_brush_radius(self, radius: float, emit: bool = True) -> None:
        """Set the brush selection radius in meters."""
        self._brush_radius = max(0.05, float(radius))
        if emit:
            self.brush_size_changed.emit(self._brush_radius)
        self.update()

    def set_rect_size(self, width: float, height: float, emit: bool = True) -> None:
        """Set the rect_brush half-size in meters."""
        self._rect_width = max(0.05, float(width))
        self._rect_height = max(0.05, float(height))
        if emit:
            self.rect_size_changed.emit(self._rect_width, self._rect_height)
        self.update()

    def set_selection_mask(self, mask: np.ndarray) -> None:
        """Apply an externally-computed selection mask."""
        self._current_mask = mask
        self._invalidate_base()

    def set_class_visibility(self, visibility: np.ndarray) -> None:
        """Set which ASPRS classes are visible (bool array indexed by class code)."""
        self._class_visibility = visibility
        self._invalidate_base()

    def clear(self) -> None:
        """Clear all data."""
        self._distances = None
        self._elevations = None
        self._classifications = None
        self._dtm_distances = None
        self._dtm_elevations = None
        self._current_mask = None
        self._base_pixmap = None
        self._base_dirty = True
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
        """Auto-fit to data bounds, clipping extreme outliers via percentiles."""
        if self._distances is None or len(self._distances) == 0:
            return
        d_min, d_max = self._distances.min(), self._distances.max()

        # Use 2nd / 98th percentiles for elevation to clip noise outliers
        # (points far above/below ground that would otherwise flatten the profile)
        n = len(self._elevations)
        if n >= 50:
            z_min = float(np.percentile(self._elevations, 2))
            z_max = float(np.percentile(self._elevations, 98))
        else:
            z_min, z_max = float(self._elevations.min()), float(self._elevations.max())

        pad_d = (d_max - d_min) * 0.1 if d_max > d_min else 1.0
        pad_z = (z_max - z_min) * 0.1 if z_max > z_min else 1.0

        self._offset_x = (d_min + d_max) / 2
        self._offset_y = (z_min + z_max) / 2

        w = self.width() or 1
        h = self.height() or 1
        data_w = d_max - d_min + 2 * pad_d if (d_max - d_min) > 0 else 1.0
        data_h = z_max - z_min + 2 * pad_z if (z_max - z_min) > 0 else 1.0
        # Uniform scale — preserves true proportions (1m = same pixels on both axes)
        scale = min(w / data_w, h / data_h) * 0.9
        self._scale_x = scale
        self._scale_y = scale

    # ── painting ───────────────────────────────────────────────────

    def _render_base_pixmap(self) -> None:
        """Render axes, DTM reference curve, and point cloud into cached QPixmap."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        # Prepare numpy RGBA array (H, W, 4) initialized to dark background #1a1a2e
        img = np.empty((h, w, 4), dtype=np.uint8)
        img[:, :, 0] = 0x1A
        img[:, :, 1] = 0x1A
        img[:, :, 2] = 0x2E
        img[:, :, 3] = 0xFF

        # Draw point cloud points
        if self._distances is not None and len(self._distances) > 0:
            px = np.round((self._distances - self._offset_x) * self._scale_x + w / 2).astype(np.int32)
            py = np.round(-(self._elevations - self._offset_y) * self._scale_y + h / 2).astype(np.int32)

            vis = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if self._class_visibility is not None and self._classifications is not None:
                vis = vis & self._class_visibility[self._classifications]

            if np.any(vis):
                vis_idx = np.where(vis)[0]
                vis_px = px[vis_idx]
                vis_py = py[vis_idx]

                mode = self._colour_mode
                n_vis = len(vis_idx)

                # Compute RGB arrays
                if mode == "height":
                    z_vals = self._elevations[vis_idx]
                    z_min, z_max = float(self._elevations.min()), float(self._elevations.max())
                    z_span = max(1e-6, z_max - z_min)
                    t = np.clip((z_vals - z_min) / z_span, 0.0, 1.0)
                    cr = ((np.clip((t - 0.5) * 4, 0, 1) + np.clip((t - 0.75) * 4, 0, 1)) * 255).astype(np.uint8)
                    cg = ((np.clip(t * 4, 0, 1) * (t <= 0.5) + np.clip((1 - t) * 4, 0, 1) * (t > 0.5)) * 255).astype(np.uint8)
                    cb = ((np.clip((0.5 - t) * 4, 0, 1)) * 255).astype(np.uint8)
                elif mode == "intensity" and self._intensities is not None and len(self._intensities) > 0:
                    i_vals = self._intensities[vis_idx]
                    i_min, i_max = float(self._intensities.min()), float(self._intensities.max())
                    i_span = max(1e-6, i_max - i_min)
                    t = (np.clip((i_vals - i_min) / i_span, 0.0, 1.0) * 255).astype(np.uint8)
                    cr = cg = cb = t
                elif mode == "return_number" and self._return_numbers is not None and len(self._return_numbers) > 0:
                    rn_vals = self._return_numbers[vis_idx]
                    rn_palette = {1: (51, 178, 51), 2: (178, 178, 51), 3: (178, 102, 51), 4: (178, 51, 51)}
                    cr = np.full(n_vis, 102, dtype=np.uint8)
                    cg = np.full(n_vis, 51, dtype=np.uint8)
                    cb = np.full(n_vis, 178, dtype=np.uint8)
                    for r_code, rgb in rn_palette.items():
                        m = rn_vals == r_code
                        if np.any(m):
                            cr[m], cg[m], cb[m] = rgb
                elif mode == "flightline" and self._point_source_ids is not None and len(self._point_source_ids) > 0:
                    fl_vals = self._point_source_ids[vis_idx]
                    fl_unique = np.unique(self._point_source_ids)
                    fl_map = {fl_val: k for k, fl_val in enumerate(fl_unique)}
                    cr = np.zeros(n_vis, dtype=np.uint8)
                    cg = np.zeros(n_vis, dtype=np.uint8)
                    cb = np.zeros(n_vis, dtype=np.uint8)
                    for fl_val, k in fl_map.items():
                        m = fl_vals == fl_val
                        if np.any(m):
                            fl_col = _FL_PALETTE[k % len(_FL_PALETTE)]
                            cr[m] = int(fl_col[0] * 255)
                            cg[m] = int(fl_col[1] * 255)
                            cb[m] = int(fl_col[2] * 255)
                else:  # mode == "class"
                    cls_vals = self._classifications[vis_idx] if self._classifications is not None else np.zeros(n_vis, dtype=np.int32)
                    cr = np.zeros(n_vis, dtype=np.uint8)
                    cg = np.zeros(n_vis, dtype=np.uint8)
                    cb = np.zeros(n_vis, dtype=np.uint8)
                    for code in np.unique(cls_vals):
                        m = cls_vals == code
                        r, g, b = get_class_color(int(code))
                        cr[m] = int(r * 255)
                        cg[m] = int(g * 255)
                        cb[m] = int(b * 255)

                # Identify selected points mask
                is_sel = self._current_mask[vis_idx] if self._current_mask is not None else None

                # Stamp unselected points (2x2 square)
                if is_sel is not None and np.any(is_sel):
                    unsel = ~is_sel
                    u_px, u_py = vis_px[unsel], vis_py[unsel]
                    u_r, u_g, u_b = cr[unsel], cg[unsel], cb[unsel]
                else:
                    u_px, u_py = vis_px, vis_py
                    u_r, u_g, u_b = cr, cg, cb

                for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    cx = np.clip(u_px + dx, 0, w - 1)
                    cy = np.clip(u_py + dy, 0, h - 1)
                    img[cy, cx, 0] = u_r
                    img[cy, cx, 1] = u_g
                    img[cy, cx, 2] = u_b
                    img[cy, cx, 3] = 255

                # Stamp selected points in bright red (#ff3232, 4x4)
                if is_sel is not None and np.any(is_sel):
                    s_px, s_py = vis_px[is_sel], vis_py[is_sel]
                    for dx in range(-1, 3):
                        for dy in range(-1, 3):
                            cx = np.clip(s_px + dx, 0, w - 1)
                            cy = np.clip(s_py + dy, 0, h - 1)
                            img[cy, cx, 0] = 255
                            img[cy, cx, 1] = 50
                            img[cy, cx, 2] = 50
                            img[cy, cx, 3] = 255

        # Convert img array to QPixmap
        qimg = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)

        # Draw axes and DTM curve on top using QPainter
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            # Coordinate Axes
            painter.setPen(QPen(QColor("#444444"), 1))
            origin = self._world_to_widget(0, 0)
            painter.drawLine(0, int(origin.y()), w, int(origin.y()))
            painter.drawLine(int(origin.x()), 0, int(origin.x()), h)

            # DTM reference line
            if self._dtm_distances is not None and self._dtm_elevations is not None and len(self._dtm_distances) > 1:
                pen = QPen(QColor("#D2691E"), 2)
                painter.setPen(pen)
                path = QPainterPath()
                pt0 = self._world_to_widget(self._dtm_distances[0], self._dtm_elevations[0])
                path.moveTo(pt0)
                for i in range(1, len(self._dtm_distances)):
                    pt = self._world_to_widget(self._dtm_distances[i], self._dtm_elevations[i])
                    path.lineTo(pt)
                painter.drawPath(path)
        finally:
            painter.end()

        self._base_pixmap = pixmap
        self._base_dirty = False

    def paintEvent(self, event) -> None:
        if self._base_dirty or self._base_pixmap is None or self._base_pixmap.size() != self.size():
            self._render_base_pixmap()

        painter = QPainter(self)
        try:
            if self._base_pixmap is not None:
                painter.drawPixmap(0, 0, self._base_pixmap)

            # Selection preview (while dragging line or rect)
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
                    rx = self._brush_radius * self._scale_x
                    ry = self._brush_radius * self._scale_y
                    painter.drawEllipse(pt, rx, ry)
                elif self._select_mode == SELECT_RECT_BRUSH:
                    d, z = self._sel_end
                    p1 = self._world_to_widget(d - self._rect_width, z - self._rect_height)
                    p2 = self._world_to_widget(d + self._rect_width, z + self._rect_height)
                    rect = QRectF(p1, p2).normalized()
                    painter.drawRect(rect)

            # Persistent cursor indicator for click-to-place tools (brush / rect_brush)
            if (not self._selecting
                    and self._cursor_world is not None
                    and self._select_mode in (SELECT_BRUSH, SELECT_RECT_BRUSH)):
                painter.setPen(QPen(QColor("#ffaa00"), 1, Qt.DashLine))
                if self._select_mode == SELECT_BRUSH:
                    pt = self._world_to_widget(*self._cursor_world)
                    rx = self._brush_radius * self._scale_x
                    ry = self._brush_radius * self._scale_y
                    painter.drawEllipse(pt, rx, ry)
                elif self._select_mode == SELECT_RECT_BRUSH:
                    d, z = self._cursor_world
                    p1 = self._world_to_widget(d - self._rect_width, z - self._rect_height)
                    p2 = self._world_to_widget(d + self._rect_width, z + self._rect_height)
                    rect = QRectF(p1, p2).normalized()
                    painter.drawRect(rect)

            # Point Info picked-point marker
            if self._picked_point is not None and self._point_info_active:
                px, py, pz, cls, intens, idx, d_val = self._picked_point
                z_val = pz
                pt = self._world_to_widget(d_val, z_val)
                cx, cy = pt.x(), pt.y()
                r = 8
                painter.setPen(QPen(QColor(0, 255, 128, 220), 2.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), r, r)
                painter.drawLine(QPointF(cx - r - 4, cy), QPointF(cx + r + 4, cy))
                painter.drawLine(QPointF(cx, cy - r - 4), QPointF(cx, cy + r + 4))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 255, 128, 255)))
                painter.drawEllipse(QPointF(cx, cy), 3, 3)
        finally:
            if painter.isActive():
                painter.end()

    # ── mouse events ───────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Point Info tool: click always picks a point
        if self._point_info_active and event.button() in (Qt.LeftButton, Qt.RightButton):
            wx, wy = self._widget_to_world(event.position().x(), event.position().y())
            self._pick_nearest_point(wx, wy)
            return

        # Middle button: start panning
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            wx, wy = self._widget_to_world(event.position().x(), event.position().y())
            self._selecting = True
            self._sel_start = (wx, wy)
            self._sel_end = (wx, wy)

            # Brush / Rect Brush: select immediately on click
            if self._select_mode == SELECT_BRUSH:
                self._compute_brush_selection(wx, wy)
                self._selecting = False
            elif self._select_mode == SELECT_RECT_BRUSH:
                self._compute_rect_brush_selection(wx, wy)
                self._selecting = False

            self.update()
        elif event.button() == Qt.RightButton:
            # Cancel selection
            self._selecting = False
            self._current_mask = None
            self._invalidate_base()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Panning with middle mouse button
        if self._panning and self._pan_start is not None:
            dx = event.position().x() - self._pan_start.x()
            dy = event.position().y() - self._pan_start.y()
            self._pan_start = event.position()
            self._offset_x -= dx / self._scale_x
            self._offset_y += dy / self._scale_y
            self._invalidate_base()
            return

        wx, wy = self._widget_to_world(event.position().x(), event.position().y())
        self._cursor_world = (wx, wy)

        if self._selecting:
            self._sel_end = (wx, wy)
            self.update()
        elif self._select_mode == SELECT_BRUSH and event.buttons() & Qt.LeftButton:
            # Continuous brush painting
            self._compute_brush_selection(wx, wy, additive=True)
        elif self._select_mode == SELECT_RECT_BRUSH and event.buttons() & Qt.LeftButton:
            # Continuous rect brush painting
            self._compute_rect_brush_selection(wx, wy, additive=True)
        else:
            # Emit hover info for nearest point (only when Point Info is off)
            if not self._point_info_active:
                self._emit_hover(wx, wy)

        # Repaint for cursor indicator in brush/rect_brush modes (fast cached blit!)
        if self._select_mode in (SELECT_BRUSH, SELECT_RECT_BRUSH):
            self.update()

    def _pick_nearest_point(self, d: float, z: float) -> None:
        """Find the nearest profile point to (d, z) and emit point_picked."""
        if self._distances is None or len(self._distances) == 0:
            return

        d_dist = self._distances - d
        e_dist = self._elevations - z
        dists = d_dist * d_dist + e_dist * e_dist
        nearest = int(np.argmin(dists))

        if dists[nearest] > (self._brush_radius * 3) ** 2:
            return

        idx = int(self._profile_indices[nearest]) if self._profile_indices is not None else int(nearest)
        px = float(self._xs[nearest]) if self._xs is not None else 0.0
        py = float(self._ys[nearest]) if self._ys is not None else 0.0
        pz = float(self._elevations[nearest])
        cls = int(self._classifications[nearest]) if self._classifications is not None else 0
        intens = int(self._intensities[nearest]) if self._intensities is not None else 0

        self._picked_point = (px, py, pz, cls, intens, idx, d)
        self.update()

        self.point_picked.emit(idx, px, py, pz, cls, intens)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            event.accept()
            return

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

        if self._select_mode not in (SELECT_BRUSH, SELECT_RECT_BRUSH) and self._current_mask is not None:
            self.selection_changed.emit(self._current_mask.copy())

        self._invalidate_base()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ShiftModifier:
            # Shift+Scroll: dynamically resize active brush or rect tool
            factor = 1.15 if event.angleDelta().y() > 0 else 0.85
            if self._select_mode == SELECT_BRUSH:
                self.set_brush_radius(self._brush_radius * factor)
            elif self._select_mode == SELECT_RECT_BRUSH:
                self.set_rect_size(self._rect_width * factor, self._rect_height * factor)
            event.accept()
            return

        if event.modifiers() & Qt.ControlModifier:
            # Ctrl+Scroll: dynamically adjust corridor width
            direction = 1.0 if event.angleDelta().y() > 0 else -1.0
            self._total_width = max(0.5, self._total_width + direction * 0.5)
            self.profile_width_changed.emit(self._total_width)
            event.accept()
            return

        # Normal mode: scroll zooms smoothly
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self._scale_x *= factor
        self._scale_y *= factor
        self._scale_x = max(0.001, min(self._scale_x, 10000.0))
        self._scale_y = max(0.001, min(self._scale_y, 10000.0))
        self._invalidate_base()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_BracketLeft:
            if self._select_mode == SELECT_BRUSH:
                self.set_brush_radius(self._brush_radius * 0.85)
            elif self._select_mode == SELECT_RECT_BRUSH:
                self.set_rect_size(self._rect_width * 0.85, self._rect_height * 0.85)
            event.accept()
            return
        elif key == Qt.Key_BracketRight:
            if self._select_mode == SELECT_BRUSH:
                self.set_brush_radius(self._brush_radius * 1.15)
            elif self._select_mode == SELECT_RECT_BRUSH:
                self.set_rect_size(self._rect_width * 1.15, self._rect_height * 1.15)
            event.accept()
            return
        super().keyPressEvent(event)

    # ── selection computation ──────────────────────────────────────

    def _compute_line_selection(self, above: bool) -> None:
        if self._distances is None or self._sel_start is None or self._sel_end is None:
            return

        d1, z1 = self._sel_start
        d2, z2 = self._sel_end

        d_min, d_max = sorted([d1, d2])
        in_range = (self._distances >= d_min) & (self._distances <= d_max)

        if abs(d2 - d1) < 1e-9:
            mask = self._distances >= d1 if above else self._distances < d1
            mask = mask & in_range
        else:
            m = (z2 - z1) / (d2 - d1)
            b = z1 - m * d1
            line_z = m * self._distances + b
            mask = self._elevations > line_z if above else self._elevations < line_z
            mask = mask & in_range

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
        new_mask = (d_dist * d_dist + e_dist * e_dist) <= (self._brush_radius * self._brush_radius)

        if additive and self._current_mask is not None:
            self._current_mask = self._current_mask | new_mask
        else:
            self._current_mask = new_mask

        self.selection_changed.emit(self._current_mask.copy())
        self._invalidate_base()

    def _compute_rect_brush_selection(
        self, d: float, z: float, additive: bool = False
    ) -> None:
        if self._distances is None:
            return

        new_mask = (
            (self._distances >= d - self._rect_width)
            & (self._distances <= d + self._rect_width)
            & (self._elevations >= z - self._rect_height)
            & (self._elevations <= z + self._rect_height)
        )

        if additive and self._current_mask is not None:
            self._current_mask = self._current_mask | new_mask
        else:
            self._current_mask = new_mask

        self.selection_changed.emit(self._current_mask.copy())
        self._invalidate_base()

    def _emit_hover(self, d: float, z: float) -> None:
        """Find the nearest profile point and emit point_hovered signal."""
        if self._distances is None or len(self._distances) == 0:
            return

        d_dist = self._distances - d
        e_dist = self._elevations - z
        dists = d_dist * d_dist + e_dist * e_dist
        nearest = int(np.argmin(dists))

        # Only emit if within reasonable distance in world coords
        if dists[nearest] > (self._brush_radius * 3) ** 2:
            return

        idx = int(self._profile_indices[nearest]) if self._profile_indices is not None else int(nearest)
        px = float(self._xs[nearest]) if self._xs is not None else 0.0
        py = float(self._ys[nearest]) if self._ys is not None else 0.0
        pz = float(self._elevations[nearest])
        cls = int(self._classifications[nearest]) if self._classifications is not None else 0
        intens = int(self._intensities[nearest]) if self._intensities is not None else 0

        self.point_hovered.emit(idx, px, py, pz, cls, intens)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._base_dirty = True

    def leaveEvent(self, event) -> None:
        """Clear cursor indicator when mouse leaves the widget."""
        self._cursor_world = None
        self.update()
