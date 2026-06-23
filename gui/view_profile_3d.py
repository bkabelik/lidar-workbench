"""
LiDAR Workbench — 3D Profile Slice View.

Shows the points inside the active profile corridor in a rotated 3D
perspective (distance along profile × perpendicular offset × elevation)
using an embedded Open3D ``SceneWidget``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..config import MAX_POINTS_PER_VIEW, get_class_color

logger = logging.getLogger("lidar_workbench.gui.view_profile_3d")

try:
    import open3d as o3d
    import open3d.visualization.gui as o3d_gui
    import open3d.visualization.rendering as o3d_render
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None
    o3d_gui = None
    o3d_render = None


class ViewProfile3D(QWidget):
    """
    3D view of the profile corridor points.

    Coordinate mapping (for display):
        - **X** = distance along the profile line (meters)
        - **Y** = perpendicular offset from the profile line (meters)
        - **Z** = elevation

    Selection highlights are shown in red.

    Signals:
        points_selected(indices: np.ndarray):
            Emitted when the user selects points in this view.
    """

    points_selected = Signal(np.ndarray)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._distances: Optional[np.ndarray] = None
        self._elevations: Optional[np.ndarray] = None
        self._classifications: Optional[np.ndarray] = None
        self._selection_mask: Optional[np.ndarray] = None
        self._scene_widget = None
        self._scene = None
        self._container: Optional[QWidget] = None
        self._has_data = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_OPEN3D:
            lbl = QLabel("3D Profile Slice\n\n(Open3D required)")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #555; background: #f0f0f0; border: 1px dashed #bbb;")
            layout.addWidget(lbl)
            return

        try:
            self._scene_widget = o3d_gui.SceneWidget()
            # Open3D ≥0.19 removed SceneWidget.window; use shared Renderer
            if hasattr(self._scene_widget, 'window'):
                self._scene = o3d_render.Open3DScene(self._scene_widget.window)
            else:
                from ._renderer import get_shared_renderer
                self._scene = o3d_render.Open3DScene(get_shared_renderer())
            self._scene.set_background([0.10, 0.10, 0.15, 1.0])
            self._scene_widget.scene = self._scene
            self._scene_widget.set_view_controls(
                o3d_gui.SceneWidget.Controls.ROTATE_CAMERA
            )

            self._container = QWidget.createWindowContainer(self._scene_widget, self)
            self._container.setMinimumSize(160, 120)
            self._container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(self._container)

        except Exception as exc:
            logger.warning("Failed to create profile 3D SceneWidget: %s", exc)
            lbl = QLabel(f"3D Profile init failed:\n{exc}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #c44;")
            layout.addWidget(lbl)
            self._scene_widget = None
            self._scene = None

    # ── public API ─────────────────────────────────────────────────

    def load_profile_points(
        self,
        distances: np.ndarray,
        elevations: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        classifications: np.ndarray,
    ) -> None:
        """
        Load the profile corridor points and display them in
        profile-space (distance, 0, elevation).
        """
        if self._scene_widget is None:
            return

        n = len(distances)
        if n == 0:
            return

        self._distances = distances
        self._elevations = elevations
        self._classifications = classifications
        self._selection_mask = None
        self._has_data = True

        # Downsample if needed
        if n > MAX_POINTS_PER_VIEW:
            step = max(1, n // MAX_POINTS_PER_VIEW)
            idx = np.arange(0, n, step)
            d = distances[idx]
            e = elevations[idx]
            c = classifications[idx]
        else:
            d = distances
            e = elevations
            c = classifications

        pts = np.column_stack((d, np.zeros(len(d)), e))

        colors = np.zeros((len(d), 3), dtype=np.float64)
        for code in np.unique(c):
            mask = c == code
            colors[mask] = get_class_color(int(code))

        self._build_and_show(pts, colors)

        # Orient camera to look along the profile
        center = pts.mean(axis=0)
        extent = float(np.ptp(pts, axis=0).max()) or 1.0
        self._scene_widget.setup_camera(
            45.0,
            center,
            center + np.array([-extent * 0.3, -extent * 2.0, extent * 0.8]),
            np.array([0.0, 0.0, 1.0]),
        )

    def set_selection(self, mask: np.ndarray) -> None:
        """
        Highlight selected points (red) in the 3D profile view.

        Args:
            mask: Boolean array over the *loaded* profile points.
        """
        if self._scene_widget is None or not self._has_data:
            return

        self._selection_mask = mask

        if self._distances is None:
            return

        n = len(self._distances)
        if n > MAX_POINTS_PER_VIEW:
            step = max(1, n // MAX_POINTS_PER_VIEW)
            idx = np.arange(0, n, step)
            d = self._distances[idx]
            e = self._elevations[idx]
            c = self._classifications[idx]
            mask_ds = mask[idx]
        else:
            d = self._distances
            e = self._elevations
            c = self._classifications
            mask_ds = mask

        pts = np.column_stack((d, np.zeros(len(d)), e))

        colors = np.zeros((len(d), 3), dtype=np.float64)
        for code in np.unique(c):
            cm = (c == code) & (~mask_ds)
            colors[cm] = get_class_color(int(code))
        # Selected = red
        colors[mask_ds] = (1.0, 0.15, 0.15)

        self._build_and_show(pts, colors)

    def clear(self) -> None:
        """Remove all geometry."""
        self._has_data = False
        self._distances = None
        self._elevations = None
        self._classifications = None
        self._selection_mask = None
        if self._scene is not None:
            self._scene.clear_geometry()

    # ── internal ───────────────────────────────────────────────────

    def _build_and_show(
        self, pts: np.ndarray, colors: np.ndarray
    ) -> None:
        """Upload point cloud with colours to the scene."""
        if self._scene is None:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        self._scene.clear_geometry()
        mat = o3d_render.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.5
        self._scene.add_geometry("profile_points", pcd, mat)
