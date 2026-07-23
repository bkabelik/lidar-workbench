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
    - ``"flightline"`` — coloured by point_source_id / flight line
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
    """QWidget that paints a stored QPixmap, with mouse orbit/pan callbacks."""

    def __init__(self, parent=None, orbit_callback=None, pan_callback=None,
                 click_callback=None, dblclick_callback=None):
        super().__init__(parent)
        self._pixmap = None
        self._orbit_cb = orbit_callback
        self._pan_cb = pan_callback
        self._click_cb = click_callback
        self._dblclick_cb = dblclick_callback
        self._mouse_last = None
        self._mouse_button = None
        self._click_start = None  # for distinguishing click vs drag
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
        self._mouse_last = (event.position().x(), event.position().y())
        self._mouse_button = event.button()
        self._click_start = self._mouse_last

    def mouseMoveEvent(self, event):
        if self._mouse_last is None:
            return
        x, y = event.position().x(), event.position().y()
        dx = x - self._mouse_last[0]
        dy = y - self._mouse_last[1]
        self._mouse_last = (x, y)

        if self._mouse_button == Qt.LeftButton and self._orbit_cb:
            self._orbit_cb("orbit", dx, dy)
        elif self._mouse_button == Qt.MiddleButton and self._pan_cb:
            self._pan_cb("pan", dx, dy)

    def mouseReleaseEvent(self, event):
        if event.button() == self._mouse_button:
            # Distinguish click (small movement) from drag
            if self._click_start is not None and self._click_cb is not None:
                ex, ey = event.position().x(), event.position().y()
                dist = ((ex - self._click_start[0])**2 + (ey - self._click_start[1])**2)**0.5
                if dist < 4:  # < 4 pixels = click, not drag
                    self._click_cb(ex, ey)
            self._mouse_last = None
            self._mouse_button = None
            self._click_start = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._dblclick_cb is not None:
            self._dblclick_cb(event.position().x(), event.position().y())

    def wheelEvent(self, event):
        if self._orbit_cb is None:
            return
        self._orbit_cb("zoom", event.angleDelta().y() / 120.0, 0.0)


# ── Flightline palette (20 distinct colours) ───────────────────────

_FL_PALETTE = np.array([
    [0.90, 0.10, 0.10],  # red
    [0.10, 0.50, 0.90],  # blue
    [0.10, 0.80, 0.10],  # green
    [0.90, 0.60, 0.10],  # orange
    [0.70, 0.10, 0.90],  # purple
    [0.10, 0.80, 0.80],  # cyan
    [0.90, 0.10, 0.70],  # magenta
    [0.60, 0.60, 0.10],  # olive
    [0.90, 0.50, 0.50],  # salmon
    [0.20, 0.60, 0.20],  # forest
    [0.20, 0.20, 0.80],  # navy
    [0.80, 0.30, 0.10],  # rust
    [0.10, 0.70, 0.70],  # teal
    [0.70, 0.70, 0.10],  # gold
    [0.60, 0.10, 0.60],  # plum
    [0.10, 0.50, 0.50],  # dark cyan
    [0.80, 0.80, 0.10],  # yellow
    [0.50, 0.10, 0.50],  # indigo
    [0.10, 0.40, 0.10],  # dark green
    [0.50, 0.50, 0.50],  # grey
], dtype=np.float64)


# ── View3D ─────────────────────────────────────────────────────────

class View3D(QWidget):
    """GPU-accelerated 3D view using Open3D OffscreenRenderer."""

    COLOUR_MODES = ("class", "height", "intensity", "return_number", "flightline")

    # emitted when user clicks a point in the 3D view
    point_picked = Signal(int, float, float, float, int, int)
    # (index, x, y, z, classification, intensity)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._point_data = None
        self._colour_mode = "class"
        self._has_geometry = False

        # World offset: UTMs are large (394000, 5216000) — subtracting the
        # centroid before Open3D rendering avoids float32 precision issues.
        self._world_offset: Optional[np.ndarray] = None  # (3,) xyz offset

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

        # Picked point marker
        self._picked_pt: Optional[np.ndarray] = None  # (3,) xyz
        self._picked_geom: Optional[str] = None

        # Point Info tool state (controlled externally)
        self._point_info_active: bool = False

        # Flightline visibility: None = all visible
        self._hidden_flightlines: set = set()

        # Class visibility: None = all visible (bool array indexed by class code)
        self._class_visibility: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = _PreviewView(
            self,
            orbit_callback=self._on_orbit,
            pan_callback=self._on_pan,
            click_callback=self._on_click,
            dblclick_callback=self._on_dblclick,
        )
        layout.addWidget(self._view, 1)

        if HAS_OPEN3D:
            self._init_renderer()
        # Don't connect destroyed → _cleanup_renderer here.
        # Open3D's C++ destructor can crash during Python shutdown
        # when Filament resources are already freed by other destructors.
        # Cleanup is done explicitly via MultiViewWidget.cleanup().

    def _cleanup_renderer(self):
        """Release OpenGL resources. Called explicitly, not on destroyed."""
        if self._renderer is not None:
            # Null Python references first so the C++ destructor
            # doesn't try to access already-freed Filament resources.
            self._scene = None
            self._picked_geom = None
            self._highlight_geom = None
            try:
                del self._renderer
            except Exception:
                pass
            self._renderer = None

    # ── public API ─────────────────────────────────────────────────

    def load_point_cloud(
        self, xs, ys, zs,
        classifications=None, intensities=None, return_numbers=None,
        point_source_ids=None,
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
            if point_source_ids is not None:
                point_source_ids = point_source_ids[idx]

        self._point_data = {
            "x": xs, "y": ys, "z": zs,
            "classification": classifications,
            "intensity": intensities,
            "return_number": return_numbers,
            "point_source_id": point_source_ids,
        }
        self._has_geometry = True
        self._hidden_flightlines.clear()

        # Center large UTM coordinates for GPU float32 precision
        self._world_offset = np.array([float(xs.mean()), float(ys.mean()), float(zs.mean())])
        xs_local = xs - self._world_offset[0]
        ys_local = ys - self._world_offset[1]
        zs_local = zs - self._world_offset[2]
        self._point_data["x"] = xs_local
        self._point_data["y"] = ys_local
        self._point_data["z"] = zs_local

        self._fit_camera(xs_local, ys_local, zs_local)
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
            "classification": None, "intensity": None,
            "return_number": None, "point_source_id": None,
        }
        self._colour_mode = "_custom"
        self._has_geometry = True
        self._hidden_flightlines.clear()

        # Center for GPU float32 precision
        self._world_offset = np.array([float(xs.mean()), float(ys.mean()), float(zs.mean())])
        xs_local = xs - self._world_offset[0]
        ys_local = ys - self._world_offset[1]
        zs_local = zs - self._world_offset[2]
        self._point_data["x"] = xs_local
        self._point_data["y"] = ys_local
        self._point_data["z"] = zs_local

        self._fit_camera(xs_local, ys_local, zs_local)
        self._build_and_render(xs_local, ys_local, zs_local, np.asarray(colors, dtype=np.float64))

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

    def toggle_flightline(self, flightline: int, visible: bool) -> None:
        """Show or hide a flightline in the 3D view."""
        if not visible:
            self._hidden_flightlines.add(flightline)
        else:
            self._hidden_flightlines.discard(flightline)
        if self._point_data is not None:
            self._rebuild_scene()

    def set_all_flightlines_visible(self) -> None:
        """Show all flightlines."""
        self._hidden_flightlines.clear()
        if self._point_data is not None:
            self._rebuild_scene()

    def set_class_visibility(self, visibility: np.ndarray) -> None:
        """Set which ASPRS classes are visible (bool array indexed by class code).
        Triggers an immediate scene rebuild."""
        self._class_visibility = visibility
        if self._point_data is not None:
            self._rebuild_scene()

    @property
    def flightlines(self) -> list[int]:
        """Return sorted list of flightline numbers in the current data."""
        d = self._point_data
        if d is None or d.get("point_source_id") is None:
            return []
        ids = np.unique(d["point_source_id"])
        return sorted(int(x) for x in ids if x > 0)

    @property
    def point_info_active(self) -> bool:
        """Whether the Point Info tool is active."""
        return self._point_info_active

    @point_info_active.setter
    def point_info_active(self, value: bool) -> None:
        """Enable/disable Point Info tool. When disabled, clears the pick marker."""
        self._point_info_active = value
        if not value:
            # Clear pick marker when tool is deactivated
            self._picked_pt = None
            if self._scene is not None and self._picked_geom is not None:
                self._scene.remove_geometry(self._picked_geom)
                self._picked_geom = None
            self._render()

    def clear(self):
        self._point_data = None
        self._has_geometry = False
        self._hidden_flightlines.clear()
        self._picked_pt = None
        self._picked_geom = None
        self._highlight_geom = None
        self._world_offset = None
        # Don't touch the Open3D scene — clear_geometry can segfault if
        # the renderer is in an inconsistent state. _build_and_render
        # will clear and rebuild the scene when new data arrives.
        self._view.set_pixmap(QPixmap())
        self._view.update()

    @property
    def has_geometry(self) -> bool:
        return self._has_geometry

    # ── internals ──────────────────────────────────────────────────

    def _safe_clear_scene(self) -> None:
        """Clear the Open3D scene, recreating the renderer if it crashes."""
        if self._scene is None:
            return
        try:
            self._scene.clear_geometry()
        except Exception:
            # Open3D C++ may have crashed internally — recreate renderer
            logger.warning("clear_geometry failed — recreating renderer")
            self._renderer = None
            self._scene = None
            self._init_renderer()

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
        n = len(d["x"])

        # Build combined mask from flightline + class visibility filters
        mask = np.ones(n, dtype=bool)

        # Flightline filter
        if self._hidden_flightlines and d.get("point_source_id") is not None:
            for fl in self._hidden_flightlines:
                mask &= (d["point_source_id"] != fl)

        # Class visibility filter
        if self._class_visibility is not None and d.get("classification") is not None:
            cls_mask = self._class_visibility[d["classification"]]
            mask &= cls_mask

        if mask.sum() == 0:
            self._safe_clear_scene()
            self._highlight_geom = None
            self._render()
            return

        if mask.all():
            # No filtering needed — fast path
            colors = self._compute_colours(d)
            self._build_and_render(d["x"], d["y"], d["z"], colors)
        else:
            xs = d["x"][mask]
            ys = d["y"][mask]
            zs = d["z"][mask]
            sub_data = {
                "x": xs, "y": ys, "z": zs,
                "classification": d["classification"][mask] if d["classification"] is not None else None,
                "intensity": d["intensity"][mask] if d["intensity"] is not None else None,
                "return_number": d["return_number"][mask] if d["return_number"] is not None else None,
                "point_source_id": d["point_source_id"][mask] if d["point_source_id"] is not None else None,
            }
            colors = self._compute_colours(sub_data)
            self._build_and_render(xs, ys, zs, colors)

    def _build_and_render(self, xs, ys, zs, colors):
        if self._scene is None:
            return
        self._safe_clear_scene()
        self._highlight_geom = None
        self._picked_geom = None

        pts = np.column_stack((xs, ys, zs))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        mat = o3d_render.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.5

        self._scene.add_geometry("_points", pcd, mat)

        # Re-add pick marker if one exists
        self._show_pick_marker()

        self._render()

    def _compute_colours(self, d=None):
        if d is None:
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
        elif mode == "flightline":
            fl = d["point_source_id"]
            if fl is not None:
                unique_fl = np.unique(fl)
                for i, fl_val in enumerate(unique_fl):
                    colors[fl == fl_val] = _FL_PALETTE[i % len(_FL_PALETTE)]
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

    def _on_pan(self, action, dx, dy):
        """Pan the camera center and eye by dx/dy in screen space."""
        if self._renderer is None or action != "pan":
            return
        direction = self._cam_eye - self._cam_center
        dist = float(np.linalg.norm(direction))
        direction /= dist
        up = self._cam_up / np.linalg.norm(self._cam_up)
        right = np.cross(direction, up)
        right /= np.linalg.norm(right) + 1e-12

        pan_speed = dist * 0.002
        shift = -dx * right * pan_speed + dy * up * pan_speed

        self._cam_center = self._cam_center + shift
        self._cam_eye = self._cam_eye + shift
        self._render()

    def _on_click(self, sx, sy):
        """Handle a click in the 3D view — project all points to screen
        space, find the nearest one to the click, emit point_picked,
        and show a marker sphere.

        Only active when Point Info tool is toggled on.
        """
        if not self._point_info_active:
            return
        if self._point_data is None or self._renderer is None:
            return
        d = self._point_data
        n = len(d["x"])
        if n == 0:
            return

        vp_w = max(self._view.width(), 1)
        vp_h = max(self._view.height(), 1)
        pts = np.column_stack((d["x"], d["y"], d["z"]))

        # Project all points to screen space
        screen_pts = self._project_to_screen(pts, vp_w, vp_h)
        if screen_pts is None:
            return

        # Distance in pixels from click to each projected point
        sx_arr = screen_pts[:, 0]
        sy_arr = screen_pts[:, 1]
        in_front = screen_pts[:, 2].astype(bool)  # depth (column_stack coerces bool→float)

        pixel_dist = np.sqrt((sx_arr - sx)**2 + (sy_arr - sy)**2)
        pixel_dist[~in_front] = np.inf

        # Find nearest within tolerance (20 pixels)
        nearest = int(np.argmin(pixel_dist))
        if pixel_dist[nearest] > 20:
            return  # too far from any point

        px, py, pz = float(d["x"][nearest]), float(d["y"][nearest]), float(d["z"][nearest])
        cls = int(d["classification"][nearest]) if d["classification"] is not None else 0
        intens = int(d["intensity"][nearest]) if d["intensity"] is not None else 0

        # Store picked point (local coords for marker)
        self._picked_pt = np.array([px, py, pz])
        self._show_pick_marker()
        self._render()

        # Emit world coordinates (add back the UTM offset)
        wx, wy, wz = px, py, pz
        if self._world_offset is not None:
            wx += self._world_offset[0]
            wy += self._world_offset[1]
            wz += self._world_offset[2]
        self.point_picked.emit(nearest, wx, wy, wz, cls, intens)

    def _on_dblclick(self, sx, sy):
        """Double-click: re-center the orbit on the nearest point."""
        if self._point_data is None or self._renderer is None:
            return
        d = self._point_data
        n = len(d["x"])
        if n == 0:
            return

        vp_w = max(self._view.width(), 1)
        vp_h = max(self._view.height(), 1)
        pts = np.column_stack((d["x"], d["y"], d["z"]))

        screen_pts = self._project_to_screen(pts, vp_w, vp_h)
        if screen_pts is None:
            return

        sx_arr = screen_pts[:, 0]
        sy_arr = screen_pts[:, 1]
        in_front = screen_pts[:, 2].astype(bool)
        pixel_dist = np.sqrt((sx_arr - sx)**2 + (sy_arr - sy)**2)
        pixel_dist[~in_front] = np.inf

        nearest = int(np.argmin(pixel_dist))
        if pixel_dist[nearest] > 30:
            return  # generous tolerance for double-click

        new_center = np.array([float(d["x"][nearest]), float(d["y"][nearest]), float(d["z"][nearest])])
        offset = self._cam_eye - self._cam_center
        self._cam_center = new_center
        self._cam_eye = new_center + offset
        self._render()

    def _project_to_screen(self, pts, vp_w, vp_h):
        """Project 3D points to screen coordinates.
        Returns (N,3) array of (sx, sy, in_front)."""
        direction = self._cam_eye - self._cam_center
        view_dir = direction / np.linalg.norm(direction)
        up = self._cam_up / np.linalg.norm(self._cam_up)
        right = np.cross(view_dir, up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, view_dir)
        up /= np.linalg.norm(up) + 1e-12

        half_h = math.tan(math.radians(self._cam_fov / 2.0))
        half_w = half_h * vp_w / vp_h

        eye = self._cam_eye
        to_pts = pts - eye
        t = np.dot(to_pts, view_dir)  # depth along view direction
        in_front = t > 0.01

        # Perspective divide
        px = np.divide(np.dot(to_pts, right), t, out=np.zeros_like(t), where=in_front)
        py = np.divide(np.dot(to_pts, up), t, out=np.zeros_like(t), where=in_front)

        # NDC to pixel
        sx = (px / half_w * 0.5 + 0.5) * vp_w
        sy = (0.5 - py / half_h * 0.5) * vp_h

        return np.column_stack((sx, sy, in_front))

    def _show_pick_marker(self):
        """Add a sphere at the picked point location, sized to ~1% of point-cloud extent."""
        if self._scene is None:
            return
        # Remove old marker
        if self._picked_geom is not None:
            self._scene.remove_geometry(self._picked_geom)
            self._picked_geom = None
        if self._picked_pt is None:
            return

        # Compute a visible radius — 1% of point cloud extent, min 0.1 m
        radius = 0.5  # default fallback
        if self._point_data is not None and len(self._point_data["x"]) > 0:
            d = self._point_data
            extent = max(
                float(np.ptp(d["x"])), float(np.ptp(d["y"])), float(np.ptp(d["z"]))
            )
            radius = max(extent * 0.008, 0.1)

        import open3d as o3d
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(self._picked_pt)
        sphere.paint_uniform_color([1.0, 0.15, 0.15])  # bright red
        mat = o3d_render.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0, 0.15, 0.15, 1.0]
        self._picked_geom = "_picked_marker"
        self._scene.add_geometry(self._picked_geom, sphere, mat)
