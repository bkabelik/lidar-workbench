"""
LiDAR Workbench — DTM View (2D Top-Down).

Displays a colour-coded DTM raster or a class-coloured 2D point scatter,
and supports interactive profile-line drawing plus pan/zoom navigation.

Rendering is done into cached ``QPixmap`` images (one for the DTM raster,
one for the point scatter) so that panning and zooming only translate/scale
an image instead of drawing hundreds of thousands of individual points per
frame.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from ..config import get_class_color
from ..dtm_generator import generate_dtm

logger = logging.getLogger("lidar_workbench.gui.view_dtm")

# Maximum long-side pixel dimension of the cached scatter image.
_SCATTER_MAX_DIM = 2048

_FL_PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.50, 0.90], [0.10, 0.80, 0.10], [0.90, 0.60, 0.10],
    [0.70, 0.10, 0.90], [0.10, 0.80, 0.80], [0.90, 0.10, 0.70], [0.60, 0.60, 0.10],
    [0.90, 0.50, 0.50], [0.20, 0.60, 0.20], [0.50, 0.50, 0.90], [0.80, 0.80, 0.20],
])


class ViewDTM(QWidget):
    """
    2D top-down DTM view with point overlay and profile-line interaction.

    The user can:
        - Pan (middle-button drag) and zoom (scroll wheel) the view.
        - See ground points colour-coded by class (point mode) or the
          interpolated DTM raster (DTM mode).
        - Draw a profile line by left-click-dragging.

    Class visibility is independent of the other views and is controlled
    externally via :meth:`set_class_visibility`.

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
        self._points_z: Optional[np.ndarray] = None
        self._points_class: Optional[np.ndarray] = None
        self._points_intensity: Optional[np.ndarray] = None
        self._points_return_number: Optional[np.ndarray] = None
        self._points_source_id: Optional[np.ndarray] = None

        # Original (unsampled) arrays for DTM generation
        self._points_x_orig: Optional[np.ndarray] = None
        self._points_y_orig: Optional[np.ndarray] = None
        self._points_z_orig: Optional[np.ndarray] = None
        self._points_class_orig: Optional[np.ndarray] = None

        # Colour mode: class, height, intensity, return_number, flightline
        self._colour_mode: str = "class"

        # Class visibility: None = all classes visible
        self._class_visibility: Optional[np.ndarray] = None

        # View transform
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._scale: float = 1.0  # pixels per CRS unit

        # Display toggle: show DTM raster (True) or point scatter (False)
        self._show_dtm: bool = False

        # Profile drawing state
        self._drawing_profile: bool = False
        self._profile_start: Optional[Tuple[float, float]] = None
        self._profile_end: Optional[Tuple[float, float]] = None

        # Profile corridor (shown after profile is defined)
        self._corridor_start: Optional[Tuple[float, float]] = None
        self._corridor_end: Optional[Tuple[float, float]] = None
        self._corridor_width: float = 5.0

        # Panning state
        self._panning: bool = False
        self._pan_last: Optional[QPointF] = None

        # Cached rendered images
        self._dtm_pixmap: Optional[QPixmap] = None
        self._scatter_pixmap: Optional[QPixmap] = None
        self._scatter_bbox: Optional[Tuple[float, float, float, float]] = None

    # ── public API ─────────────────────────────────────────────────

    def load_points(self, data: dict) -> None:
        """
        Load point data for 2D top-down display.

        DTM generation is **not** performed here — it is a batch operation
        done after classification.  This view shows a class-coloured 2D
        point scatter for fast interactive browsing.

        Args:
            data: Dict with keys ``x, y, z, classification``.
        """
        xs = data["x"]
        ys = data["y"]
        zs = data["z"]
        cls = data.get("classification")
        if cls is None:
            cls = np.zeros(len(xs), dtype=np.uint8)

        self._points_x_orig = xs
        self._points_y_orig = ys
        self._points_z_orig = zs
        self._points_class_orig = cls

        intens = data.get("intensity")
        rn = data.get("return_number")
        psid = data.get("point_source_id")

        # Subsample for fast 2D scatter rendering
        n = len(xs)
        if n > 200_000:
            step = max(1, n // 200_000)
            self._points_x = xs[::step]
            self._points_y = ys[::step]
            self._points_z = zs[::step]
            self._points_class = cls[::step]
            self._points_intensity = intens[::step] if intens is not None else None
            self._points_return_number = rn[::step] if rn is not None else None
            self._points_source_id = psid[::step] if psid is not None else None
        else:
            self._points_x = xs
            self._points_y = ys
            self._points_z = zs
            self._points_class = cls
            self._points_intensity = intens
            self._points_return_number = rn
            self._points_source_id = psid

        # Clear any cached DTM (regenerated explicitly via generate_dtm)
        self._dtm_grid_x = None
        self._dtm_grid_y = None
        self._dtm_grid_z = None
        self._dtm_pixmap = None

        self._render_scatter()
        self._fit_view()
        self.update()

    def set_colour_mode(self, mode: str) -> None:
        """Set point cloud 2D scatter colouring mode."""
        self._colour_mode = mode
        self._render_scatter()
        self.update()

    def generate_dtm(self, ground_class: int = 2) -> None:
        """Generate a DTM from the currently loaded ground-classified points."""
        if self._points_x_orig is None:
            return
        cls = self._points_class_orig
        if cls is None or not (cls == ground_class).any():
            logger.debug("No ground points (class %d) available for DTM, skipping", ground_class)
            return
        try:
            xs = self._points_x_orig
            ys = self._points_y_orig
            zs = self._points_z_orig
            cls = self._points_class_orig
            gx, gy, gz, bbox = generate_dtm(xs, ys, zs, cls, ground_class=ground_class)
            self._dtm_grid_x = gx
            self._dtm_grid_y = gy
            self._dtm_grid_z = gz
            self._dtm_bbox = bbox
            self._render_dtm()
            self._show_dtm = True  # auto-switch to DTM view
            self._fit_view()
            self.update()
            logger.info("DTM generated: %dx%d grid", gz.shape[0], gz.shape[1])
        except Exception as exc:
            logger.warning("DTM generation failed: %s", exc)

    def toggle_dtm(self) -> None:
        """Toggle between DTM raster and point scatter view."""
        self._show_dtm = not self._show_dtm
        self.update()

    def set_class_visibility(self, visibility: Optional[np.ndarray]) -> None:
        """Set which ASPRS classes are visible (bool array indexed by class code)."""
        self._class_visibility = visibility
        self._render_scatter()
        self.update()

    def contextMenuEvent(self, event) -> None:
        """Right-click context menu."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        if self._show_dtm and self._dtm_pixmap is not None:
            label = "🟢 Show Points"
        else:
            label = "🗻 Show DTM"
        toggle_action = menu.addAction(label)
        toggle_action.triggered.connect(self.toggle_dtm)
        menu.addSeparator()
        gen_action = menu.addAction("Generate DTM from Ground")
        gen_action.triggered.connect(lambda: self.generate_dtm())
        menu.exec(event.globalPos())

    def clear(self) -> None:
        """Clear all data (class visibility preference is preserved)."""
        self._dtm_grid_x = None
        self._dtm_grid_y = None
        self._dtm_grid_z = None
        self._points_x = None
        self._points_y = None
        self._points_z = None
        self._points_class = None
        self._points_x_orig = None
        self._points_y_orig = None
        self._points_z_orig = None
        self._points_class_orig = None
        self._dtm_pixmap = None
        self._scatter_pixmap = None
        self._scatter_bbox = None
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
        """Convert world coordinates to widget pixel coordinates (North is up / nadir)."""
        px = (wx - self._offset_x) * self._scale + self.width() / 2
        py = -(wy - self._offset_y) * self._scale + self.height() / 2
        return QPointF(px, py)

    def _widget_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert widget pixel coordinates to world coordinates (North is up / nadir)."""
        wx = (px - self.width() / 2) / self._scale + self._offset_x
        wy = self._offset_y - (py - self.height() / 2) / self._scale
        return wx, wy

    def _bbox_to_widget_rect(self, bbox: Tuple[float, float, float, float]) -> QRectF:
        """Convert a world-space bbox ``(x_min, x_max, y_min, y_max)`` to a widget rect."""
        x_min, x_max, y_min, y_max = bbox
        p1 = self._world_to_widget(x_min, y_min)
        p2 = self._world_to_widget(x_max, y_max)
        return QRectF(p1, p2).normalized()

    def _fit_view(self) -> None:
        """Fit the view to the point extent (falling back to the DTM bbox)."""
        if self._points_x is not None and len(self._points_x) > 0:
            x_min, x_max = float(self._points_x.min()), float(self._points_x.max())
            y_min, y_max = float(self._points_y.min()), float(self._points_y.max())
        elif self._dtm_grid_x is not None and self._dtm_grid_y is not None:
            x_min, x_max, y_min, y_max = self._dtm_bbox
        else:
            return

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
        """Pre-render the class-coloured 2D point scatter as a QPixmap."""
        if self._points_x is None or len(self._points_x) == 0:
            self._scatter_pixmap = None
            self._scatter_bbox = None
            return

        xs = self._points_x
        ys = self._points_y
        cls = self._points_class

        zs = self._points_z
        intens = self._points_intensity
        rn = self._points_return_number
        psid = self._points_source_id

        # Apply class visibility filter
        if cls is not None and self._class_visibility is not None:
            vis = self._class_visibility[np.asarray(cls, dtype=np.int64)]
            xs = xs[vis]
            ys = ys[vis]
            cls = cls[vis]
            if zs is not None:
                zs = zs[vis]
            if intens is not None:
                intens = intens[vis]
            if rn is not None:
                rn = rn[vis]
            if psid is not None:
                psid = psid[vis]

        if len(xs) == 0:
            self._scatter_pixmap = None
            self._scatter_bbox = None
            return

        # World bbox with a small pad
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_pad = (x_max - x_min) * 0.02 or 1.0
        y_pad = (y_max - y_min) * 0.02 or 1.0
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        span_x = (x_max - x_min) or 1.0
        span_y = (y_max - y_min) or 1.0
        long_side = max(span_x, span_y)
        res = long_side / _SCATTER_MAX_DIM  # world units per pixel
        w = max(2, int(round(span_x / res)))
        h = max(2, int(round(span_y / res)))

        col = ((xs - x_min) / span_x * (w - 1)).astype(np.int32)
        row = ((y_max - ys) / span_y * (h - 1)).astype(np.int32)
        col = np.clip(col, 0, w - 1)
        row = np.clip(row, 0, h - 1)

        # Transparent RGBA canvas
        img = np.zeros((h, w, 4), dtype=np.uint8)

        def _stamp(rows, cols, rgb):
            rows = np.clip(rows, 0, h - 1)
            cols = np.clip(cols, 0, w - 1)
            img[rows, cols, 0] = rgb[0]
            img[rows, cols, 1] = rgb[1]
            img[rows, cols, 2] = rgb[2]
            img[rows, cols, 3] = 255

        mode = self._colour_mode

        if mode == "height" and zs is not None and len(zs) > 0:
            z_min, z_max = float(zs.min()), float(zs.max())
            z_span = max(1e-6, z_max - z_min)
            t = np.clip((zs - z_min) / z_span, 0.0, 1.0)
            cr = ((np.clip((t - 0.5) * 4, 0, 1) + np.clip((t - 0.75) * 4, 0, 1)) * 255).astype(np.uint8)
            cg = ((np.clip(t * 4, 0, 1) * (t <= 0.5) + np.clip((1 - t) * 4, 0, 1) * (t > 0.5)) * 255).astype(np.uint8)
            cb = ((np.clip((0.5 - t) * 4, 0, 1)) * 255).astype(np.uint8)
            for k in range(len(zs)):
                rgb = (int(cr[k]), int(cg[k]), int(cb[k]))
                _stamp(row[k], col[k], rgb)
                _stamp(row[k], col[k] + 1, rgb)
                _stamp(row[k] + 1, col[k], rgb)
                _stamp(row[k] + 1, col[k] + 1, rgb)
        elif mode == "intensity" and intens is not None and len(intens) > 0:
            i_min, i_max = float(intens.min()), float(intens.max())
            i_span = max(1e-6, i_max - i_min)
            vals = (np.clip((intens - i_min) / i_span, 0.0, 1.0) * 255).astype(np.uint8)
            for k in range(len(intens)):
                v = int(vals[k])
                rgb = (v, v, v)
                _stamp(row[k], col[k], rgb)
                _stamp(row[k], col[k] + 1, rgb)
                _stamp(row[k] + 1, col[k], rgb)
                _stamp(row[k] + 1, col[k] + 1, rgb)
        elif mode == "return_number" and rn is not None and len(rn) > 0:
            rn_palette = {1: (51, 178, 51), 2: (178, 178, 51), 3: (178, 102, 51), 4: (178, 51, 51)}
            for r_val in np.unique(rn):
                m = rn == r_val
                rgb = rn_palette.get(int(r_val), (102, 51, 178))
                rr = row[m]
                cc = col[m]
                _stamp(rr, cc, rgb)
                _stamp(rr, cc + 1, rgb)
                _stamp(rr + 1, cc, rgb)
                _stamp(rr + 1, cc + 1, rgb)
        elif mode == "flightline" and psid is not None and len(psid) > 0:
            for idx, fl_val in enumerate(np.unique(psid)):
                m = psid == fl_val
                fl_col = _FL_PALETTE[idx % len(_FL_PALETTE)]
                rgb = (int(fl_col[0] * 255), int(fl_col[1] * 255), int(fl_col[2] * 255))
                rr = row[m]
                cc = col[m]
                _stamp(rr, cc, rgb)
                _stamp(rr, cc + 1, rgb)
                _stamp(rr + 1, cc, rgb)
                _stamp(rr + 1, cc + 1, rgb)
        elif cls is not None:
            for code in np.unique(cls):
                m = cls == code
                r, g, b = get_class_color(int(code))
                rgb = (int(r * 255), int(g * 255), int(b * 255))
                rr = row[m]
                cc = col[m]
                # Draw a small 2x2 block so points remain visible when zoomed out
                _stamp(rr, cc, rgb)
                _stamp(rr, cc + 1, rgb)
                _stamp(rr + 1, cc, rgb)
                _stamp(rr + 1, cc + 1, rgb)
        else:
            rr = row
            cc = col
            _stamp(rr, cc, (200, 200, 200))
            _stamp(rr, cc + 1, (200, 200, 200))
            _stamp(rr + 1, cc, (200, 200, 200))
            _stamp(rr + 1, cc + 1, (200, 200, 200))

        qimg = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888)
        self._scatter_pixmap = QPixmap.fromImage(qimg.copy())
        self._scatter_bbox = (x_min, x_max, y_min, y_max)

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

        # Flip vertically so row 0 is North (top of map / nadir view from aircraft)
        z_north = np.ascontiguousarray(np.flipud(z))

        z_valid = z_north[~np.isnan(z_north)]
        if len(z_valid) == 0:
            self._dtm_pixmap = None
            return

        z_min, z_max = z_valid.min(), z_valid.max()
        if z_max <= z_min:
            z_norm = np.full_like(z_north, 0.5, dtype=np.float64)
        else:
            z_norm = (z_north - z_min) / (z_max - z_min)

        # ── hillshade ───────────────────────────────────────────
        # Compute slope and aspect from the DTM grid (North-Up)
        # Cell size in CRS units (approximate)
        dx = (self._dtm_bbox[1] - self._dtm_bbox[0]) / max(nx - 1, 1)
        dy = (self._dtm_bbox[3] - self._dtm_bbox[2]) / max(ny - 1, 1)
        cell_size = min(dx, dy) or 1.0

        # Differences in world coordinates:
        # col increases to East (+X): dz_dx
        # row increases to South (-Y), so row-1 is North (+Y): dz_dy
        dz_dx = np.zeros_like(z_north)
        dz_dy = np.zeros_like(z_north)

        dz_dx[:, 1:-1] = (z_north[:, 2:] - z_north[:, :-2]) / (2 * cell_size)
        dz_dx[:, 0] = (z_north[:, 1] - z_north[:, 0]) / cell_size
        dz_dx[:, -1] = (z_north[:, -1] - z_north[:, -2]) / cell_size

        dz_dy[1:-1, :] = (z_north[:-2, :] - z_north[2:, :]) / (2 * cell_size)
        dz_dy[0, :] = (z_north[0, :] - z_north[1, :]) / cell_size
        dz_dy[-1, :] = (z_north[-2, :] - z_north[-1, :]) / cell_size

        # Upward terrain surface normal: N = (-dz_dx, -dz_dy, 1) / |N|
        norm_len = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
        n_x = -dz_dx / norm_len
        n_y = -dz_dy / norm_len
        n_z = 1.0 / norm_len

        # Sun vector from NW (azimuth 315° CW from North, altitude 45° above horizon)
        # Vector pointing UP towards sun:
        # X: West is negative -> sin(315°) * cos(45°) = -0.5
        # Y: North is positive -> cos(315°) * cos(45°) = +0.5
        # Z: Upward is positive -> sin(45°) = sqrt(2)/2
        sun_alt = math.radians(45.0)
        sun_az = math.radians(315.0)
        sun_vec = np.array([
            math.sin(sun_az) * math.cos(sun_alt),
            math.cos(sun_az) * math.cos(sun_alt),
            math.sin(sun_alt),
        ], dtype=np.float64)
        sun_vec /= np.linalg.norm(sun_vec)

        # Lambertian hillshade = N . S
        hs = n_x * sun_vec[0] + n_y * sun_vec[1] + n_z * sun_vec[2]
        hs = np.where(np.isnan(z_north), 0.0, np.clip(hs, 0.0, 1.0))

        # ── combine elevation colour + hillshade ─────────────────
        # Elevation colours: green→yellow→brown
        t = z_norm
        r_el = np.clip(t * 180 + 40, 0, 255).astype(np.float64)
        g_el = np.clip((1 - t) * 160 + 40, 0, 255).astype(np.float64)
        b_el = np.clip((1 - t) * 100 + 20, 0, 255).astype(np.float64)

        # Blend: hillshade modulates brightness (40% base + 60% shaded)
        blend = 0.4 + 0.6 * hs
        r = np.clip(r_el * blend, 0, 255).astype(np.uint8)
        g = np.clip(g_el * blend, 0, 255).astype(np.uint8)
        b = np.clip(b_el * blend, 0, 255).astype(np.uint8)

        img = np.zeros((ny, nx, 4), dtype=np.uint8)
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        img[:, :, 3] = np.where(np.isnan(z_north), 0, 255).astype(np.uint8)

        qimg = QImage(img.data, nx, ny, QImage.Format_RGBA8888)
        self._dtm_pixmap = QPixmap.fromImage(qimg.copy())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#1a1a2e"))

            show_raster = self._show_dtm and self._dtm_pixmap is not None

            if show_raster:
                # DTM raster (hillshade) — smooth scaling looks better here
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                target_rect = self._bbox_to_widget_rect(self._dtm_bbox)
                painter.drawPixmap(
                    target_rect, self._dtm_pixmap,
                    QRectF(0, 0, self._dtm_pixmap.width(), self._dtm_pixmap.height()),
                )
                painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            elif self._scatter_pixmap is not None and self._scatter_bbox is not None:
                # Class-coloured point scatter
                target_rect = self._bbox_to_widget_rect(self._scatter_bbox)
                painter.drawPixmap(
                    target_rect, self._scatter_pixmap,
                    QRectF(0, 0, self._scatter_pixmap.width(), self._scatter_pixmap.height()),
                )

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

            # North compass indicator (top-right corner: nadir top-down view)
            nx_pos = self.width() - 28
            ny_pos = 36
            painter.setPen(Qt.NoPen)
            # Red North needle
            painter.setBrush(QColor("#e74c3c"))
            n_arrow = QPainterPath()
            n_arrow.moveTo(nx_pos, ny_pos - 16)
            n_arrow.lineTo(nx_pos - 5, ny_pos)
            n_arrow.lineTo(nx_pos, ny_pos - 4)
            n_arrow.closeSubpath()
            painter.drawPath(n_arrow)

            # Grey South needle
            painter.setBrush(QColor("#bdc3c7"))
            s_arrow = QPainterPath()
            s_arrow.moveTo(nx_pos, ny_pos - 16)
            s_arrow.lineTo(nx_pos + 5, ny_pos)
            s_arrow.lineTo(nx_pos, ny_pos - 4)
            s_arrow.closeSubpath()
            painter.drawPath(s_arrow)

            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawText(int(nx_pos - 4), int(ny_pos - 19), "N")
        finally:
            if painter.isActive():
                painter.end()

    # ── mouse events ───────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            wx, wy = self._widget_to_world(event.position().x(), event.position().y())
            self._drawing_profile = True
            self._profile_start = (wx, wy)
            self._profile_end = (wx, wy)
            self.update()
        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            if self._pan_last is not None:
                dx = event.position().x() - self._pan_last.x()
                dy = event.position().y() - self._pan_last.y()
                self._offset_x -= dx / self._scale
                self._offset_y += dy / self._scale
            self._pan_last = event.position()
            self.update()
            return

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
        elif event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_last = None
            self.unsetCursor()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom in/out, centred on the cursor position."""
        px = event.position().x()
        py = event.position().y()
        wx, wy = self._widget_to_world(px, py)

        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self._scale = max(0.001, min(self._scale * factor, 1000.0))

        # Keep the world point under the cursor stationary
        self._offset_x = wx - (px - self.width() / 2) / self._scale
        self._offset_y = wy + (py - self.height() / 2) / self._scale
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Re-render scatter at the same world resolution (no-op unless the
        # cached image was never built); keep the user's current zoom/pan.
        if self._scatter_pixmap is None:
            self._render_scatter()
