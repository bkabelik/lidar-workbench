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
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import ASPRS_CLASS_COLORS, ASPRS_CLASS_NAMES
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

        # Per-viewer ASPRS class visibility (indexed by class code).
        # Each viewer can hide/show classes independently.
        self._class_visibility = {
            "3d": np.ones(256, dtype=bool),
            "dtm": np.ones(256, dtype=bool),
            "profile": np.ones(256, dtype=bool),
        }
        self._class_menus: dict = {}

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
        self._sel_mode_combo.addItem("▬ Rect Brush", "rect_brush")
        self._sel_mode_combo.currentIndexChanged.connect(self._on_sel_mode_changed)
        toolbar.addWidget(QLabel("Select:"))
        toolbar.addWidget(self._sel_mode_combo)

        # Tool size adjustment controls
        self._brush_size_label = QLabel("Radius:")
        self._brush_size_spin = QDoubleSpinBox()
        self._brush_size_spin.setRange(0.05, 25.0)
        self._brush_size_spin.setSingleStep(0.1)
        self._brush_size_spin.setDecimals(2)
        self._brush_size_spin.setValue(0.6)
        self._brush_size_spin.setSuffix(" m")
        self._brush_size_spin.setToolTip("Brush selection radius (m). Shortcuts: '[' / ']' or Shift+Scroll")
        self._brush_size_spin.valueChanged.connect(self._on_brush_spin_changed)

        self._rect_w_label = QLabel("W:")
        self._rect_w_spin = QDoubleSpinBox()
        self._rect_w_spin.setRange(0.1, 50.0)
        self._rect_w_spin.setSingleStep(0.2)
        self._rect_w_spin.setDecimals(1)
        self._rect_w_spin.setValue(1.0)
        self._rect_w_spin.setSuffix(" m")
        self._rect_w_spin.setToolTip("Rectangle brush width (m)")
        self._rect_w_spin.valueChanged.connect(self._on_rect_spin_changed)

        self._rect_h_label = QLabel("H:")
        self._rect_h_spin = QDoubleSpinBox()
        self._rect_h_spin.setRange(0.05, 25.0)
        self._rect_h_spin.setSingleStep(0.1)
        self._rect_h_spin.setDecimals(2)
        self._rect_h_spin.setValue(0.4)
        self._rect_h_spin.setSuffix(" m")
        self._rect_h_spin.setToolTip("Rectangle brush height (m)")
        self._rect_h_spin.valueChanged.connect(self._on_rect_spin_changed)

        toolbar.addWidget(self._brush_size_label)
        toolbar.addWidget(self._brush_size_spin)
        toolbar.addWidget(self._rect_w_label)
        toolbar.addWidget(self._rect_w_spin)
        toolbar.addWidget(self._rect_h_label)
        toolbar.addWidget(self._rect_h_spin)

        toolbar.addSpacing(12)

        # Profile corridor width control
        self._corridor_label = QLabel("Corridor:")
        self._corridor_spin = QDoubleSpinBox()
        self._corridor_spin.setRange(0.2, 50.0)
        self._corridor_spin.setSingleStep(0.5)
        self._corridor_spin.setDecimals(1)
        self._corridor_spin.setValue(2.0)
        self._corridor_spin.setSuffix(" m")
        self._corridor_spin.setToolTip("Profile corridor slice width (m). Shortcut: Ctrl+Scroll in Profile View")
        self._corridor_spin.valueChanged.connect(self._on_corridor_spin_changed)
        toolbar.addWidget(self._corridor_label)
        toolbar.addWidget(self._corridor_spin)

        toolbar.addSpacing(12)

        self._colour_combo = QComboBox()
        self._colour_combo.addItem("By Class", "class")
        self._colour_combo.addItem("By Height", "height")
        self._colour_combo.addItem("By Intensity", "intensity")
        self._colour_combo.addItem("By Return Number", "return_number")
        self._colour_combo.addItem("By Flightline", "flightline")
        self._colour_combo.currentIndexChanged.connect(self._on_colour_mode_changed)
        toolbar.addWidget(QLabel("Colour:"))
        toolbar.addWidget(self._colour_combo)

        toolbar.addSpacing(12)

        # Per-viewer class visibility buttons
        toolbar.addWidget(QLabel("Classes:"))
        self._cls_btn_3d = self._add_class_button("3d", "3D")
        self._cls_btn_dtm = self._add_class_button("dtm", "DTM")
        self._cls_btn_profile = self._add_class_button("profile", "Profile")
        toolbar.addWidget(self._cls_btn_3d)
        toolbar.addWidget(self._cls_btn_dtm)
        toolbar.addWidget(self._cls_btn_profile)

        toolbar.addSpacing(12)

        # Flightline toggle checkboxes (populated on load)
        self._fl_toggle_layout = QHBoxLayout()
        self._fl_toggle_layout.setContentsMargins(0, 0, 0, 0)
        self._fl_toggle_layout.setSpacing(4)
        toolbar.addLayout(self._fl_toggle_layout)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # ── views ──────────────────────────────────────────────────
        self._view_3d = View3D()
        self._view_dtm = ViewDTM()
        self._view_dtm.profile_line_defined.connect(self.profile_line_defined)

        self._view_profile = ViewProfile()
        self._view_profile.brush_size_changed.connect(self._on_profile_brush_size_changed)
        self._view_profile.rect_size_changed.connect(self._on_profile_rect_size_changed)
        self._view_profile.profile_width_changed.connect(self._on_profile_width_changed)
        self._update_tool_size_widgets_visibility("brush")

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

    # ── per-viewer class visibility ───────────────────────────────

    def _add_class_button(self, view_key: str, label: str) -> QPushButton:
        """Create a toolbar button with a checkable ASPRS-class popup menu."""
        btn = QPushButton(f"{label} ▾")
        btn.setFlat(True)
        btn.setToolTip(f"Toggle ASPRS class visibility in the {label} view")
        btn.setStyleSheet(
            "QPushButton { padding: 2px 6px; font-weight: bold; }"
            "QPushButton::menu-indicator { image: none; }"
        )
        menu = QMenu(btn)
        btn.setMenu(menu)
        menu.triggered.connect(
            lambda action, key=view_key: self._on_class_toggled(key, action)
        )
        menu.aboutToShow.connect(
            lambda key=view_key: self._rebuild_class_menu(key)
        )
        self._class_menus[view_key] = menu
        return btn

    def _rebuild_class_menu(self, view_key: str) -> None:
        """(Re)build the popup menu for one viewer from its visibility state."""
        menu = self._class_menus.get(view_key)
        if menu is None:
            return
        menu.clear()

        vis = self._class_visibility[view_key]

        all_action = menu.addAction("▸ Show All")
        all_action.setData(-1)
        none_action = menu.addAction("▸ Hide All")
        none_action.setData(-2)
        menu.addSeparator()

        for code in sorted(ASPRS_CLASS_NAMES.keys()):
            name = ASPRS_CLASS_NAMES[code]
            r, g, b = ASPRS_CLASS_COLORS.get(code, (0.5, 0.5, 0.5))
            pm = QPixmap(14, 14)
            pm.fill(QColor(int(r * 255), int(g * 255), int(b * 255)))
            action = menu.addAction(f"{code:2d}: {name}")
            action.setCheckable(True)
            action.setChecked(bool(vis[code]))
            action.setData(code)
            action.setIcon(pm)

    def _on_class_toggled(self, view_key: str, action) -> None:
        """Handle a class-visibility menu action for one viewer."""
        code = action.data()
        vis = self._class_visibility[view_key]
        if code == -1:
            vis[:] = True
        elif code == -2:
            vis[:] = False
        else:
            vis[code] = action.isChecked()

        self._apply_class_visibility(view_key)

    def _apply_class_visibility(self, view_key: str) -> None:
        """Push the stored visibility for one viewer to its widget."""
        vis = self._class_visibility[view_key]
        if view_key == "3d":
            self._view_3d.set_class_visibility(vis)
        elif view_key == "dtm":
            self._view_dtm.set_class_visibility(vis)
        elif view_key == "profile":
            self._view_profile.set_class_visibility(vis)

    def _apply_all_class_visibility(self) -> None:
        """Push all three viewers' visibility state to their widgets."""
        for key in ("3d", "dtm", "profile"):
            self._apply_class_visibility(key)

    def set_class_visibility(self, view_key: str, visibility: np.ndarray) -> None:
        """Set the class-visibility array for one viewer (``3d``/``dtm``/``profile``)."""
        if view_key not in self._class_visibility:
            raise ValueError(f"Unknown view key: {view_key!r}")
        self._class_visibility[view_key] = np.asarray(visibility, dtype=bool)
        self._apply_class_visibility(view_key)

    def class_visibility(self, view_key: str) -> np.ndarray:
        """Return the current class-visibility array for one viewer."""
        return self._class_visibility[view_key]

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
            point_data.get("point_source_id"),
        )

        # DTM top-down
        self._view_dtm.load_points(point_data)

        # Clear profile view (populated when a profile line is drawn)
        self._view_profile.clear()

        # Re-apply per-viewer class visibility (views may have been reloaded)
        self._apply_all_class_visibility()

        # Rebuild flightline toggle checkboxes
        self._rebuild_flightline_toggles()

        self.tile_loaded.emit(tile_id)

    def clear(self) -> None:
        """Clear all views."""
        self._current_tile_id = None
        self._point_data = None
        self._view_3d.clear()
        self._view_dtm.clear()
        self._view_profile.clear()

    def cleanup(self) -> None:
        """Release Open3D resources held by the views before Qt shutdown."""
        # Force cleanup of the 3D view's renderer
        if hasattr(self, '_view_3d'):
            self._view_3d._cleanup_renderer()

    def _rebuild_flightline_toggles(self) -> None:
        """Create/update flightline visibility checkboxes in the toolbar."""
        # Remove old toggles
        while self._fl_toggle_layout.count():
            item = self._fl_toggle_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        fls = self._view_3d.flightlines
        if len(fls) <= 1:
            return  # only one or zero flightlines — no need for toggles

        # Add "All" checkbox
        all_cb = QCheckBox("FL:All")
        all_cb.setChecked(True)
        all_cb.setToolTip("Show/hide all flightlines")
        all_cb.toggled.connect(self._on_fl_all_toggled)
        self._fl_toggle_layout.addWidget(all_cb)

        # Add per-flightline checkboxes
        for fl in fls:
            cb = QCheckBox(str(fl))
            cb.setChecked(True)
            cb.setToolTip(f"Toggle flightline {fl}")
            cb.toggled.connect(lambda checked, f=fl: self._view_3d.toggle_flightline(f, checked))
            self._fl_toggle_layout.addWidget(cb)

    def _on_fl_all_toggled(self, checked: bool) -> None:
        """Show or hide all flightlines."""
        if checked:
            self._view_3d.set_all_flightlines_visible()
        else:
            for fl in self._view_3d.flightlines:
                self._view_3d.toggle_flightline(fl, False)

    # ── slots ──────────────────────────────────────────────────────
    def _update_tool_size_widgets_visibility(self, mode: str) -> None:
        is_brush = (mode == "brush")
        is_rect = (mode == "rect_brush")
        self._brush_size_label.setVisible(is_brush)
        self._brush_size_spin.setVisible(is_brush)
        self._rect_w_label.setVisible(is_rect)
        self._rect_w_spin.setVisible(is_rect)
        self._rect_h_label.setVisible(is_rect)
        self._rect_h_spin.setVisible(is_rect)

    def _on_brush_spin_changed(self, val: float) -> None:
        self._view_profile.set_brush_radius(val, emit=False)

    def _on_rect_spin_changed(self, _val: float) -> None:
        w = self._rect_w_spin.value() * 0.5  # half-width
        h = self._rect_h_spin.value() * 0.5  # half-height
        self._view_profile.set_rect_size(w, h, emit=False)

    def _on_profile_brush_size_changed(self, radius: float) -> None:
        self._brush_size_spin.blockSignals(True)
        self._brush_size_spin.setValue(radius)
        self._brush_size_spin.blockSignals(False)

    def _on_profile_rect_size_changed(self, half_w: float, half_h: float) -> None:
        self._rect_w_spin.blockSignals(True)
        self._rect_w_spin.setValue(half_w * 2.0)
        self._rect_w_spin.blockSignals(False)
        self._rect_h_spin.blockSignals(True)
        self._rect_h_spin.setValue(half_h * 2.0)
        self._rect_h_spin.blockSignals(False)

    def _on_corridor_spin_changed(self, val: float) -> None:
        self._view_profile.set_profile_width(val)
        self._view_profile.profile_width_changed.emit(val)

    def _on_profile_width_changed(self, val: float) -> None:
        self._corridor_spin.blockSignals(True)
        self._corridor_spin.setValue(val)
        self._corridor_spin.blockSignals(False)

    def _on_sel_mode_changed(self, index: int) -> None:
        """Propagate selection mode to the profile view."""
        mode = self._sel_mode_combo.currentData()
        self._update_tool_size_widgets_visibility(mode)
        self._view_profile.set_selection_mode(mode)

    def _on_colour_mode_changed(self, index: int) -> None:
        """Propagate colour mode to 3D, Profile, and DTM views."""
        mode = self._colour_combo.currentData()
        self._view_3d.set_colour_mode(mode)
        self._view_profile.set_colour_mode(mode)
        self._view_dtm.set_colour_mode(mode)
