"""
LiDAR Workbench — 3D Point Cloud View.

Renders a coloured point cloud using QPainter with a software 3D
perspective projection (no OpenGL dependency).  Orbit / pan / zoom
are handled via mouse events — identical interaction model to the
DTM and Profile views.

Colour modes:
    - ``"class"`` — by ASPRS classification code
    - ``"height"`` — rainbow ramp by elevation
    - ``"intensity"`` — greyscale by LiDAR intensity
    - ``"return_number"`` — coloured by return number
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from ..config import MAX_POINTS_PER_VIEW, get_class_color

logger = logging.getLogger("lidar_workbench.gui.view_3d")


# ── colour helpers ─────────────────────────────────────────────────

def _height_colours(z: np.ndarray) -> np.ndarray:
    """Map elevation to a 'terrain' rainbow (blue → green → yellow → red)."""
    n = len(z)
    colors = np.zeros((n, 3), dtype=np.float32)
    if z.max() <= z.min():
        colors[:] = (0.5, 0.5, 0.5)
        return colors
    t = (z - z.min()) / (z.max() - z.min())
    r = np.clip((t - 0.5) * 4.0, 0.0, 1.0) + np.clip((t - 0.75) * 4.0, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)
    g = np.clip(t * 4.0, 0.0, 1.0) * (t <= 0.5) + np.clip((1.0 - t) * 4.0, 0.0, 1.0) * (t > 0.5)
    g = np.clip(g, 0.0, 1.0)
    b = np.clip((0.5 - t) * 4.0, 0.0, 1.0)
    colors[:, 0] = r
    colors[:, 1] = g
    colors[:, 2] = b
    return colors


# ── View3D ─────────────────────────────────────────────────────────

class View3D(QWidget):
    """
    3D point cloud view using software rendering (QPainter).

    Mouse controls:
        - **Left-drag** → orbit
        - **Middle-drag** or **Ctrl+left-drag** → pan
        - **Scroll-wheel** → zoom

    Supports four colour modes:
        - ``"class"`` — by ASPRS classification code
        - ``"height"`` — rainbow ramp by elevation
        - ``"intensity"`` — greyscale by LiDAR intensity
        - ``"return_number"`` — coloured by return number
    """

    _COLOUR_MODES = ("class", "height", "intensity", "return_number")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._point_data: Optional[dict] = None
        self._colour_mode: str = "class"
        self._has_geometry = False

        # Camera state (trackball)
        self._azimuth: float = -45.0    # degrees
        self._elevation: float = 35.0    # degrees
        self._distance: float = 500.0    # world units
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._target_z: float = 0.0

        # Mouse interaction state
        self._last_mouse_pos: Optional[QPointF] = None
        self._mouse_mode: str = "none"  # "orbit" | "pan" | "none"

        # Cached projected points for fast repaint
        self._proj_x: Optional[np.ndarray] = None
        self._proj_y: Optional[np.ndarray] = None
        self._proj_colors: Optional[np.ndarray] = None
        self._proj_depths: Optional[np.ndarray] = None

        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)

    # ── public API ─────────────────────────────────────────────────

    def load_point_cloud(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        classifications: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        return_numbers: Optional[np.ndarray] = None,
    ) -> None:
        """Load a point cloud, auto-downsampling if >MAX_POINTS_PER_VIEW."""
        n = len(xs)
        if n == 0:
            logger.debug("View3D: empty point cloud, skipping")
            return

        if n > MAX_POINTS_PER_VIEW:
            step = max(1, n // MAX_POINTS_PER_VIEW)
            indices = np.arange(0, n, step)
            xs = xs[indices]
            ys = ys[indices]
            zs = zs[indices]
            if classifications is not None:
                classifications = classifications[indices]
            if intensities is not None:
                intensities = intensities[indices]
            if return_numbers is not None:
                return_numbers = return_numbers[indices]
            logger.debug("Downsampled %d → %d points for 3D view", n, len(xs))

        self._point_data = {
            "x": xs, "y": ys, "z": zs,
            "classification": classifications,
            "intensity": intensities,
            "return_number": return_numbers,
        }

        # Auto-fit camera to data
        self._target_x = float(xs.mean())
        self._target_y = float(ys.mean())
        self._target_z = float(zs.mean())
        extent = float(np.ptp(zs)) or float(np.ptp(xs)) or 1.0
        self._distance = extent * 3.0

        self._rebuild_geometry()

    def load_point_cloud_colored(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Load a point cloud with pre-computed per-point RGB colours (filter previews)."""
        n = len(xs)
        if n == 0:
            return

        if n > MAX_POINTS_PER_VIEW:
            step = max(1, n // MAX_POINTS_PER_VIEW)
            idx = np.arange(0, n, step)
            xs, ys, zs = xs[idx], ys[idx], zs[idx]
            colors = colors[idx]

        self._point_data = {
            "x": xs, "y": ys, "z": zs,
            "classification": None, "intensity": None, "return_number": None,
        }
        self._colour_mode = "_custom"

        self._target_x = float(xs.mean())
        self._target_y = float(ys.mean())
        self._target_z = float(zs.mean())
        extent = float(np.ptp(zs)) or float(np.ptp(xs)) or 1.0
        self._distance = extent * 3.0

        self._project_and_cache(xs, ys, zs, np.asarray(colors, dtype=np.float32))

    def set_colour_mode(self, mode: str) -> None:
        """Change the point colouring mode."""
        if mode not in self._COLOUR_MODES:
            logger.warning("Unknown colour mode: %s", mode)
            return
        self._colour_mode = mode
        if self._point_data is not None:
            self._rebuild_geometry()

    def highlight_points(
        self,
        indices: np.ndarray,
        colour: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    ) -> None:
        """Recolour a subset of points (boolean mask or int indices)."""
        if self._point_data is None:
            return
        colors = self._compute_colours()
        if indices.dtype == bool:
            colors[indices] = colour
        else:
            colors[indices] = colour
        d = self._point_data
        self._project_and_cache(d["x"], d["y"], d["z"], colors)

    def clear(self) -> None:
        """Remove all geometry."""
        self._point_data = None
        self._has_geometry = False
        self._proj_x = None
        self._proj_y = None
        self._proj_colors = None
        self._proj_depths = None
        self.update()

    @property
    def has_geometry(self) -> bool:
        return self._has_geometry

    # ── painting ───────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(26, 26, 46))

        if self._proj_x is None or self._proj_y is None or self._proj_colors is None or len(self._proj_x) == 0:
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data loaded")
            painter.end()
            return

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        # Sort by depth (far first — painter's algorithm)
        if self._proj_depths is not None:
            order = np.argsort(-self._proj_depths)
        else:
            order = np.arange(len(self._proj_x))

        # Cap draw count for performance (draw ~50k points max)
        max_draw = min(len(order), 50000)
        step = max(1, len(order) // max_draw)
        draw_idx = order[::step]

        # Convert to screen pixels
        sx = self._proj_x[draw_idx] * (min(w, h) * 0.4) + cx
        sy = -self._proj_y[draw_idx] * (min(w, h) * 0.4) + cy

        # Clamp to visible area
        visible = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)

        # Draw points in batch — using small dots
        size = 2.0
        for i, vi in enumerate(visible):
            if not vi:
                continue
            r, g, b = self._proj_colors[draw_idx[i]]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(int(r * 255), int(g * 255), int(b * 255), 220))
            painter.drawEllipse(QPointF(sx[i], sy[i]), size, size)

        painter.end()

    # ── mouse interaction ──────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = event.position()
        if event.button() == Qt.MiddleButton:
            self._mouse_mode = "pan"
        elif event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self._mouse_mode = "pan"
            else:
                self._mouse_mode = "orbit"
        else:
            self._mouse_mode = "none"

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse_pos is None:
            return
        dx = event.position().x() - self._last_mouse_pos.x()
        dy = event.position().y() - self._last_mouse_pos.y()
        self._last_mouse_pos = event.position()

        if self._mouse_mode == "orbit":
            self._azimuth += dx * 0.5
            self._elevation = max(-89.0, min(89.0, self._elevation - dy * 0.5))
            self._reproject()
            self.update()
        elif self._mouse_mode == "pan":
            a = math.radians(self._azimuth)
            e = math.radians(self._elevation)
            pan = self._distance * 0.002
            # Camera right vector in world XY plane
            right_x = math.cos(a)
            right_y = -math.sin(a)
            self._target_x += right_x * (-dx * pan)
            self._target_y += right_y * (-dx * pan)
            # Camera up-ish (elevation affects Z only for pan)
            self._target_z += dy * pan
            self._reproject()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._mouse_mode = "none"
        self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._distance = max(1.0, min(1e6, self._distance * factor))
        self._reproject()
        self.update()

    # ── internals ──────────────────────────────────────────────────

    def _rebuild_geometry(self) -> None:
        """Compute colours and project."""
        if self._point_data is None:
            return
        colors = self._compute_colours()
        d = self._point_data
        self._project_and_cache(d["x"], d["y"], d["z"], colors)

    def _compute_colours(self) -> np.ndarray:
        """Compute per-point RGB colours for the current mode."""
        if self._point_data is None:
            return np.zeros((0, 3), dtype=np.float32)

        n = len(self._point_data["x"])
        colors = np.zeros((n, 3), dtype=np.float32)
        mode = self._colour_mode

        if mode == "class":
            cls = self._point_data["classification"]
            if cls is not None:
                for code in np.unique(cls):
                    mask = cls == code
                    colors[mask] = get_class_color(int(code))
            else:
                colors[:] = (0.5, 0.5, 0.5)

        elif mode == "height":
            colors = _height_colours(self._point_data["z"])

        elif mode == "intensity":
            intens = self._point_data["intensity"]
            if intens is not None and intens.max() > intens.min():
                t = (intens.astype(np.float64) - intens.min()) / (intens.max() - intens.min())
                colors = np.column_stack((t.astype(np.float32), t.astype(np.float32), t.astype(np.float32)))
            else:
                colors[:] = (0.5, 0.5, 0.5)

        elif mode == "return_number":
            rn = self._point_data["return_number"]
            palette = {
                1: (0.2, 0.7, 0.2), 2: (0.7, 0.7, 0.2),
                3: (0.7, 0.4, 0.2), 4: (0.7, 0.2, 0.2),
                5: (0.4, 0.2, 0.7),
            }
            if rn is not None:
                for r, col in palette.items():
                    colors[rn == r] = col
            else:
                colors[:] = (0.5, 0.5, 0.5)

        else:
            colors[:] = (0.5, 0.5, 0.5)

        return colors

    def _project_and_cache(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Update raw data and reproject."""
        self._point_data = {
            "x": xs, "y": ys, "z": zs,
            "classification": self._point_data.get("classification") if self._point_data else None,
            "intensity": self._point_data.get("intensity") if self._point_data else None,
            "return_number": self._point_data.get("return_number") if self._point_data else None,
        }
        self._has_geometry = len(xs) > 0
        self._reproject_core(xs, ys, zs, colors)

    def _reproject(self) -> None:
        """Re-project existing point data with current camera."""
        if self._point_data is None:
            return
        colors = self._compute_colours() if self._colour_mode != "_custom" else (
            self._proj_colors if self._proj_colors is not None else self._compute_colours()
        )
        self._reproject_core(
            self._point_data["x"],
            self._point_data["y"],
            self._point_data["z"],
            colors,
        )

    def _reproject_core(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Apply perspective projection to raw data."""
        n = len(xs)
        if n == 0:
            self._proj_x = None
            self._proj_y = None
            self._proj_colors = None
            self._proj_depths = None
            return

        # Translate to camera-relative coordinates
        tx = xs - self._target_x
        ty = ys - self._target_y
        tz = zs - self._target_z

        # Build camera basis vectors
        a = math.radians(self._azimuth)
        e = math.radians(self._elevation)

        # Camera direction (toward target)
        cd_x = math.cos(e) * math.sin(a)
        cd_y = math.cos(e) * math.cos(a)
        cd_z = math.sin(e)

        # Camera right (perpendicular to direction, in XY plane)
        cr_x = math.cos(a)
        cr_y = -math.sin(a)
        cr_z = 0.0

        # Camera up
        cu_x = -math.sin(e) * math.sin(a)
        cu_y = -math.sin(e) * math.cos(a)
        cu_z = math.cos(e)

        # View-space coordinates
        v_right = tx * cr_x + ty * cr_y + tz * cr_z
        v_up = tx * cu_x + ty * cu_y + tz * cu_z
        v_fwd = tx * cd_x + ty * cd_y + tz * cd_z

        # Distance from camera
        cam_dist = self._distance - v_fwd
        cam_dist = np.maximum(cam_dist, 0.01)

        # Perspective divide
        fov = 1.0
        proj_x = v_right * fov / cam_dist
        proj_y = v_up * fov / cam_dist

        self._proj_x = proj_x.astype(np.float32)
        self._proj_y = proj_y.astype(np.float32)
        self._proj_colors = np.asarray(colors, dtype=np.float32)
        # Depth: negative for sorting (far first)
        self._proj_depths = -v_fwd.astype(np.float32)
        self._has_geometry = True
