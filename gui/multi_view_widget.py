"""
LiDAR Workbench — Multi-View Widget.

Manages the three synchronised views (3D, DTM, profile) arranged in a
resizable layout:

    ┌───────────┬──────────┐
    │  View3D   │ ViewDTM  │  ← top splitter (2:1 stretch)
    ├───────────┴──────────┤
    │    ViewProfile       │  ← full-width bottom
    └──────────────────────┘

Colour modes are synchronised across the 3D overview.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .view_3d import View3D
from .view_dtm import ViewDTM
from .view_profile import ViewProfile

logger = logging.getLogger("lidar_workbench.gui.multi_view")


class MultiViewWidget(QWidget):
    """
    Container for the three synchronised views, arranged in a resizable
    layout built from nested :class:`QSplitter` widgets.

    Layout:
        ``QVBoxLayout``
        ├── toolbar (colour combo)
        └── vertical ``QSplitter``
            ├── top horizontal ``QSplitter``
            │   ├── :class:`View3D`      (3D point cloud)
            │   └── :class:`ViewDTM`     (2D top-down DTM)
            └── :class:`ViewProfile`      (2D profile side view, full width)

    Signals:
        profile_line_defined(start_xy, end_xy):
            Forwarded from the DTM view when the user draws a profile line.
    """

    profile_line_defined = Signal(tuple, tuple)
    tile_loaded = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_tile_id: Optional[str] = None
        self._point_data: Optional[dict] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # ── toolbar ────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)

        # Selection mode buttons
        self._sel_mode_combo = QComboBox()
        self._sel_mode_combo.addItem("🖌 Brush", "brush")
        self._sel_mode_combo.addItem("↗ Above Line", "line_above")
        self._sel_mode_combo.addItem("↘ Below Line", "line_below")
        self._sel_mode_combo.addItem("▭ Rectangle", "rectangle")
        self._sel_mode_combo.currentIndexChanged.connect(self._on_sel_mode_changed)
        toolbar.addWidget(QLabel("Select:"))
        toolbar.addWidget(self._sel_mode_combo)

        toolbar.addSpacing(12)

        self._colour_combo = QComboBox()
        self._colour_combo.addItem("By Class", "class")
        self._colour_combo.addItem("By Height", "height")
        self._colour_combo.addItem("By Intensity", "intensity")
        self._colour_combo.addItem("By Return Number", "return_number")
        self._colour_combo.currentIndexChanged.connect(self._on_colour_mode_changed)
        toolbar.addWidget(QLabel("Colour:"))
        toolbar.addWidget(self._colour_combo)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # ── views ──────────────────────────────────────────────────
        self._view_3d = View3D()
        self._view_dtm = ViewDTM()
        self._view_dtm.profile_line_defined.connect(self.profile_line_defined)

        self._view_profile = ViewProfile()

        # ── nested splitters for resizable layout ──────────────────
        # Top row: 3D (left, stretch 2) | DTM (right, stretch 1)
        self._top_splitter = QSplitter(Qt.Horizontal)
        self._top_splitter.addWidget(self._view_3d)
        self._top_splitter.addWidget(self._view_dtm)
        self._top_splitter.setStretchFactor(0, 2)
        self._top_splitter.setStretchFactor(1, 1)

        # Vertical: top row (stretch 2) | profile (stretch 1)
        self._vertical_splitter = QSplitter(Qt.Vertical)
        self._vertical_splitter.addWidget(self._top_splitter)
        self._vertical_splitter.addWidget(self._view_profile)
        self._vertical_splitter.setStretchFactor(0, 2)
        self._vertical_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self._vertical_splitter, stretch=1)

    # ── public API ─────────────────────────────────────────────────

    def load_tile(self, tile_id: str, point_data: dict) -> None:
        """
        Load a tile into all three views.

        Args:
            tile_id:    Tile identifier.
            point_data: Dict with ``x, y, z, classification, intensity,
                        return_number``.
        """
        self._current_tile_id = tile_id
        self._point_data = point_data

        # 3D overview
        self._view_3d.load_point_cloud(
            point_data["x"], point_data["y"], point_data["z"],
            point_data.get("classification"),
            point_data.get("intensity"),
            point_data.get("return_number"),
        )

        # DTM top-down
        self._view_dtm.load_points(point_data)

        # Clear profile view (populated when a profile line is drawn)
        self._view_profile.clear()

        self.tile_loaded.emit(tile_id)

    def clear(self) -> None:
        """Clear all views."""
        self._current_tile_id = None
        self._point_data = None
        self._view_3d.clear()
        self._view_dtm.clear()
        self._view_profile.clear()

    # ── slots ──────────────────────────────────────────────────────

    def _on_sel_mode_changed(self, index: int) -> None:
        """Propagate selection mode to the profile view."""
        mode = self._sel_mode_combo.currentData()
        self._view_profile.set_selection_mode(mode)

    def _on_colour_mode_changed(self, index: int) -> None:
        """Propagate colour mode to the 3D overview view."""
        mode = self._colour_combo.currentData()
        self._view_3d.set_colour_mode(mode)
        # Reload with new colour mode if data is loaded
        if self._point_data is not None:
            self._view_3d.load_point_cloud(
                self._point_data["x"],
                self._point_data["y"],
                self._point_data["z"],
                self._point_data.get("classification"),
                self._point_data.get("intensity"),
                self._point_data.get("return_number"),
            )
