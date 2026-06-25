"""
LiDAR Workbench — 3D Point Cloud View.

GPU-accelerated point cloud rendering via Open3D ``OffscreenRenderer``,
displayed through a custom ``_PreviewView`` QWidget.  Identical approach
to the preview dialog — no software QPainter per-point loops.

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

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from ..config import ASPRS_CLASS_COLORS, FALLBACK_CLASS_COLOR

logger = logging.getLogger("lidar_workbench.gui.view_3d")

try:
    import open3d as o3d
    import open3d.visualization.rendering as o3d_render
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None
    o3d_render = None


# ── _PreviewView (shared with preview_dialog — keep in sync) ───────

class _PreviewView(QWidget):
    """QWidget that paints a stored QPixmap, with mouse orbit callbacks."""

    def __init__(self, parent=None, orbit_callback=None):
        super().__init__(parent)
        self._pixmap = None
        self._orbit_cb = orbit_callback
        self._mouse_last = None
        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)

    def set_pixmap(self, pm: QPixmap):
        self._pixmap = pm
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return
        from PySide6.QtGui import QPainter
        from PySide6.QtCore import QRectF
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        src = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        dst = QRectF(0, 0, self.width(), self.height())
        p.drawPixmap(dst, self._pixmap, src)
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._mouse_last = (event.position().x(), event.position().y())

    def mouseMoveEvent(self, event):
        if self._mouse_last is None or self._orbit_cb is None:
            return
        x, y = event.position().x(), event.position().y()
        dx, dy = x - self._mouse_last[0], y - self._mouse_last[1]
        self._mouse_last = (x, y)
        self._orbit_cb("orbit", dx, dy)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._mouse_last = None

    def wheelEvent(self, event):
        if self._orbit_cb is None:
            return
        self._orbit_cb("zoom", event.angleDelta().y() / 120.0, 0.0)


# ── View3D ─────────────────────────────────────────────────────────

class View3D(QWidget):
    """GPU-accelerated 3D view using Open3D OffscreenRenderer."""

    COLOUR_MODES = ("class", "height", "intensity", "return_number")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._point_data = None
        self._colour_mode = "class"
        self._has_geometry = False

        # Open3D
        self._renderer = None   # OffscreenRenderer
        self._scene = None      # Open3DScene

        # Camera
        self._cam_center = np.array([0.0, 0.0, 0.0])
        self._cam_eye = np.array([0.0, -100.0, 50.0])
        self._cam_up = np.array([0.0, 0.0, 1.0])
        self._cam_fov = 45.0

        # Highlight overlay geometry name → colour
        self._highlight_geom: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = _PreviewView(self, orbit_callback=self._on_orbit)
        layout.addWidget(self._view, 1)

        if HAS_OPEN3D:
            self._init_renderer()
        self.destroyed.connect(self._cleanup_renderer)

    def _cleanup_renderer(self):
        """Release OpenGL resources (called on destroyed signal)."""
        if self._renderer is not None:
            try:
                self._scene = None
                del self._renderer
                self._renderer = None
            except Exception:
                pass

    # ── public API ─────────────────────────────────────────────────

    def load_point_cloud(
        self, xs, ys, zs,
        classifications=None, intensities=None, return_numbers=None,
    ):
        n = len(xs)
        if n == 0:
            return

        # Subsample for GPU budget
        if n > 2_000_000:
            step = max(1, n // 2_000_000)
            idx = np.arange(0, n, step)
            xs, ys, zs = xs[idx], ys[idx], zs[idx]
            if classifications is not None:
                classifications = classifications[idx]
            if intensities is not None:
                intensities = intensities[idx]
            if return_numbers is not None:
                return_numbers = return_numbers[idx]

        self._point_data = {
            "x": xs, "y": ys, "z": zs,
            "classification": classifications,
            "intensity": intensities,
            "return_number": return_numbers,
        }
        self._has_geometry = True
        self._fit_camera(xs, ys, zs)
        self._rebuild_scene()

    def load_point_cloud_colored(self, xs, ys, zs, colors):
        n = len(xs)
        if n == 0:
            return
        if n > 2_000_000:
            step = max(1, n // 2_000_000)
            idx = np.arange(0, n, step)
            xs, ys, zs, colors = xs[idx], ys[idx], zs[idx], colors[idx]
        self._point_data = {
            "x": xs, "y": ys, "z": zs,
            "classification": None, "intensity": None, "return_number": None,
        }
        self._colour_mode = "_custom"
        self._has_geometry = True
        self._fit_camera(xs, ys, zs)
        self._build_and_render(xs, ys, zs, np.asarray(colors, dtype=np.float64))

    def set_colour_mode(self, mode: str):
        if mode not in self.COLOUR_MODES and mode != "_custom":
            return
        self._colour_mode = mode
        if self._point_data is not None:
            self._rebuild_scene()

    def highlight_points(self, indices, colour=(1.0, 0.2, 0.2)):
        """Add a highlighted overlay on top of the main point cloud."""
        if self._point_data is None or self._scene is None:
            return
        # Remove previous highlight
        if self._highlight_geom is not None:
            self._scene.remove_geometry(self._highlight_geom)
            self._highlight_geom = None
        if len(indices) == 0:
            self._render()
            return
        d = self._point_data
        mask = np.zeros(len(d["x"]), dtype=bool)
        mask[indices] = True
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack((
            d["x"][mask], d["y"][mask], d["z"][mask]
        )))
        c = np.tile(np.array(colour, dtype=np.float64), (mask.sum(), 1))
        pcd.colors = o3d.utility.Vector3dVector(c)
        mat = o3d_render.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 4.0
        self._highlight_geom = "_highlight"
        self._scene.add_geometry(self._highlight_geom, pcd, mat)
        self._render()

    def clear(self):
        self._point_data = None
        self._has_geometry = False
        if self._scene is not None:
            self._scene.clear_geometry()
        self._highlight_geom = None
        self._view.set_pixmap(QPixmap())
        self._view.update()

    @property
    def has_geometry(self) -> bool:
        return self._has_geometry

    # ── internals ──────────────────────────────────────────────────

    def _init_renderer(self):
        if self._renderer is not None:
            return
        try:
            self._renderer = o3d_render.OffscreenRenderer(800, 600)
            self._scene = self._renderer.scene
            self._scene.set_background([0.10, 0.10, 0.18, 1.0])
        except Exception as exc:
            logger.warning("OffscreenRenderer failed: %s", exc)
            self._renderer = None
            self._scene = None

    def _fit_camera(self, xs, ys, zs):
        self._cam_center = np.array([float(xs.mean()), float(ys.mean()), float(zs.mean())])
        extent = float(np.ptp(zs)) or float(np.ptp(xs)) or 1.0
        self._cam_eye = self._cam_center + np.array([0.0, -extent * 2.5, extent * 0.8])
        self._cam_up = np.array([0.0, 0.0, 1.0])

    def _rebuild_scene(self):
        if self._point_data is None or self._scene is None:
            return
        d = self._point_data
        colors = self._compute_colours()
        self._build_and_render(d["x"], d["y"], d["z"], colors)

    def _build_and_render(self, xs, ys, zs, colors):
        if self._scene is None:
            return
        self._scene.clear_geometry()
        self._highlight_geom = None

        pts = np.column_stack((xs, ys, zs))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        mat = o3d_render.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.5
        self._scene.add_geometry("_points", pcd, mat)
        self._render()

    def _compute_colours(self):
        d = self._point_data
        n = len(d["x"])
        mode = self._colour_mode

        if mode == "_custom":
            return np.full((n, 3), 0.5, dtype=np.float64)

        colors = np.zeros((n, 3), dtype=np.float64)

        if mode == "class":
            cls = d["classification"]
            if cls is not None:
                for code in np.unique(cls):
                    c = ASPRS_CLASS_COLORS.get(int(code), FALLBACK_CLASS_COLOR)
                    colors[cls == code] = c
            else:
                colors[:] = 0.5
        elif mode == "height":
            z = d["z"].astype(np.float64)
            z_min, z_max = float(z.min()), float(z.max())
            if z_max > z_min:
                t = (z - z_min) / (z_max - z_min)
                colors[:, 0] = np.clip((t - 0.5) * 4, 0, 1) + np.clip((t - 0.75) * 4, 0, 1)
                colors[:, 1] = np.clip(t * 4, 0, 1) * (t <= 0.5) + np.clip((1 - t) * 4, 0, 1) * (t > 0.5)
                colors[:, 2] = np.clip((0.5 - t) * 4, 0, 1)
            else:
                colors[:] = 0.5
        elif mode == "intensity":
            intens = d["intensity"]
            if intens is not None and intens.max() > intens.min():
                t = (intens.astype(np.float64) - intens.min()) / (intens.max() - intens.min())
                colors = np.column_stack((t, t, t))
            else:
                colors[:] = 0.5
        elif mode == "return_number":
            rn = d["return_number"]
            palette = {1: (0.2, 0.7, 0.2), 2: (0.7, 0.7, 0.2),
                       3: (0.7, 0.4, 0.2), 4: (0.7, 0.2, 0.2), 5: (0.4, 0.2, 0.7)}
            if rn is not None:
                for r, col in palette.items():
                    colors[rn == r] = col
            else:
                colors[:] = 0.5
        else:
            colors[:] = 0.5

        return colors

    def _render(self):
        if self._renderer is None:
            return
        self._renderer.setup_camera(self._cam_fov, self._cam_center,
                                     self._cam_eye, self._cam_up)
        try:
            img = self._renderer.render_to_image()
            arr = np.asarray(img).copy()
            h, w = arr.shape[:2]
            ch = arr.shape[2] if arr.ndim == 3 else 1
            fmt = {4: QImage.Format_RGBA8888, 3: QImage.Format_RGB888}.get(
                ch, QImage.Format_Grayscale8)
            qimg = QImage(arr.data, w, h, w * ch, fmt)
            self._view.set_pixmap(QPixmap.fromImage(qimg.copy()))
        except Exception as exc:
            logger.warning("Render failed: %s", exc)

    def _on_orbit(self, action, dx, dy):
        if self._renderer is None:
            return
        if action == "orbit":
            direction = self._cam_eye - self._cam_center
            up = self._cam_up / np.linalg.norm(self._cam_up)
            angle_h = -dx * 0.005
            cos_h, sin_h = math.cos(angle_h), math.sin(angle_h)
            direction = (cos_h * direction + sin_h * np.cross(up, direction)
                         + (1 - cos_h) * np.dot(direction, up) * up)
            right = np.cross(direction, up)
            right /= np.linalg.norm(right) + 1e-12
            angle_v = -dy * 0.005
            cos_v, sin_v = math.cos(angle_v), math.sin(angle_v)
            new_dir = cos_v * direction + sin_v * np.cross(right, direction)
            if np.dot(new_dir / (np.linalg.norm(new_dir) + 1e-12), up) < 0.99:
                direction = new_dir
            self._cam_eye = self._cam_center + direction
        elif action == "zoom":
            direction = self._cam_eye - self._cam_center
            dist = float(np.linalg.norm(direction))
            new_dist = dist * (1.0 - dx * 0.1)
            if new_dist > 0.01:
                self._cam_eye = self._cam_center + direction / dist * new_dist
        self._render()
