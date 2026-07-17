"""
LiDAR Workbench — Properties Panel.

Right-side panel showing point properties and quick-classification
buttons, context-sensitive to the active view and selection.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import ASPRS_CLASS_NAMES, get_class_color
from .settings_dialog import load_shortcuts

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    laspy = None

logger = logging.getLogger("lidar_workbench.gui.properties_panel")

# Quick-access classification buttons: (code, label, colour)
QUICK_CLASSES = [
    (0,  "Created, Never Classified", "#808080"),
    (1,  "Unclassified",             "#CCCCCC"),
    (2,  "Ground",                   "#8B4513"),
    (3,  "Low Vegetation",           "#009900"),
    (4,  "Medium Vegetation",        "#00CC00"),
    (5,  "High Vegetation",          "#33FF33"),
    (6,  "Building",                 "#CC3333"),
    (7,  "Low Point (Noise)",        "#4D4D4D"),
    (8,  "Model Key/Reserved",       "#E6E600"),
    (9,  "Water",                    "#0066CC"),
    (10, "Rail",                     "#999999"),
    (11, "Road Surface",             "#999999"),
    (12, "Overlap/Reserved",         "#B34D00"),
    (13, "Wire – Guard (Shield)",    "#FFFFFF"),
    (14, "Wire – Conductor (Phase)", "#808080"),
    (15, "Transmission Tower",       "#E580E5"),
    (16, "Wire-Structure Connector", "#B3B3B3"),
    (17, "Bridge Deck",              "#4DB3B3"),
    (18, "High Noise",               "#E61A1A"),
]


class PropertiesPanel(QWidget):
    """
    Right-side panel for point properties and classification actions.

    Displays:
        - Selected point count
        - Properties of the hovered/selected point (coords, class, intensity)
        - Quick-classify buttons for common ASPRS classes
        - Undo / Redo buttons

    Signals:
        classify_requested(new_class: int):
            Emitted when the user clicks a quick-classify button.
        undo_requested():
            Emitted for undo.
        redo_requested():
            Emitted for redo.
    """

    classify_requested = Signal(int)
    undo_requested = Signal()
    redo_requested = Signal()
    point_info_requested = Signal()  # user clicked the Info button

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)
        self._tile_data: Optional[dict] = None  # full point arrays for current tile
        self._current_point_idx: int = -1       # index of the hovered/selected point
        self._info_dlg: Optional[QDialog] = None  # reusable non-modal info dialog
        self._info_text: Optional[QTextEdit] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Point properties ───────────────────────────────────────
        prop_group = QGroupBox("Point Properties")
        prop_form = QFormLayout(prop_group)

        self._sel_count_label = QLabel("0 selected")
        prop_form.addRow("Selected:", self._sel_count_label)

        self._coord_label = QLabel("—")
        prop_form.addRow("Coordinates:", self._coord_label)

        self._class_label = QLabel("—")
        prop_form.addRow("Class:", self._class_label)

        self._intensity_label = QLabel("—")
        prop_form.addRow("Intensity:", self._intensity_label)

        self._return_label = QLabel("—")
        prop_form.addRow("Return #:", self._return_label)

        self._height_label = QLabel("—")
        prop_form.addRow("Height:", self._height_label)

        layout.addWidget(prop_group)

        # Info button
        self._info_btn = QPushButton("ℹ Point Info…")
        self._info_btn.setToolTip("Show all attributes of the currently hovered/selected point (non-blocking)")
        self._info_btn.setCheckable(True)
        self._info_btn.clicked.connect(self._toggle_point_info_dialog)
        self._info_btn.setEnabled(False)
        layout.addWidget(self._info_btn)

        # ── File / LAS header metadata ─────────────────────────────
        meta_group = QGroupBox("File Metadata")
        meta_form = QFormLayout(meta_group)

        self._scanner_label = QLabel("—")
        meta_form.addRow("Scanner:", self._scanner_label)

        self._sensor_type_label = QLabel("—")
        meta_form.addRow("Sensor Type:", self._sensor_type_label)

        self._flight_line_label = QLabel("—")
        meta_form.addRow("Flight Line:", self._flight_line_label)

        self._las_version_label = QLabel("—")
        meta_form.addRow("LAS Version:", self._las_version_label)

        self._file_point_count_label = QLabel("—")
        meta_form.addRow("Total Points:", self._file_point_count_label)

        self._crs_label = QLabel("—")
        meta_form.addRow("CRS:", self._crs_label)

        layout.addWidget(meta_group)

        # ── Quick classify ─────────────────────────────────────────
        classify_group = QGroupBox("Quick Classify")
        classify_layout = QVBoxLayout(classify_group)

        # Map class codes to shortcut keys
        _classify_keys = {
            0: "classify_created", 1: "classify_unclass",
            2: "classify_ground", 3: "classify_low_veg",
            4: "classify_med_veg", 5: "classify_high_veg",
            6: "classify_building", 7: "classify_noise",
            8: "classify_model_key", 9: "classify_water",
            10: "classify_rail", 11: "classify_road",
            12: "classify_overlap", 13: "classify_wire_guard",
            14: "classify_wire_conductor", 15: "classify_tower",
            16: "classify_wire_connector", 17: "classify_bridge",
            18: "classify_high_noise",
        }
        _sc = load_shortcuts()

        for code, name, hex_color in QUICK_CLASSES:
            sc_key = _classify_keys.get(code, "")
            sc_text = _sc.get(sc_key, "")
            suffix = f"  [{sc_text}]" if sc_text else ""
            btn = QPushButton(f"  {code}: {name}{suffix}")
            btn.setToolTip(f"Classify selected points as {name} ({sc_text or 'no shortcut'})")
            btn.setStyleSheet(
                f"QPushButton {{"
                f"  text-align: left;"
                f"  padding: 4px 8px;"
                f"  border-left: 4px solid {hex_color};"
                f"  background: #f9f9f9;"
                f"}}"
                f"QPushButton:hover {{ background: #e8e8e8; }}"
            )
            btn.clicked.connect(lambda checked, c=code: self.classify_requested.emit(c))
            classify_layout.addWidget(btn)

        layout.addWidget(classify_group)

        # ── Undo / Redo ────────────────────────────────────────────
        undo_group = QGroupBox("History")
        undo_layout = QHBoxLayout(undo_group)

        self._undo_btn = QPushButton("↩ Undo")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        undo_layout.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("↪ Redo")
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self.redo_requested.emit)
        undo_layout.addWidget(self._redo_btn)

        layout.addWidget(undo_group)

        layout.addStretch()

    # ── public API ─────────────────────────────────────────────────

    def set_tile_data(self, point_data: Optional[dict]) -> None:
        """
        Store the full point arrays for the currently loaded tile,
        enabling the Info button to look up all attributes by index.

        Args:
            point_data: Dict with arrays ``x, y, z, classification,
                        intensity, return_number, point_source_id``,
                        plus any extra LAS fields that are available.
        """
        self._tile_data = point_data
        self._info_btn.setEnabled(point_data is not None and len(point_data.get("x", [])) > 0)

    def set_point_info(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        classification: Optional[int] = None,
        intensity: Optional[int] = None,
        return_number: Optional[int] = None,
        point_index: int = -1,
    ) -> None:
        """
        Update the point properties display.

        Pass ``None`` for any field to show "—".
        """
        self._current_point_idx = point_index

        if x is not None and y is not None and z is not None:
            self._coord_label.setText(f"{x:.2f}, {y:.2f}, {z:.2f}")
        else:
            self._coord_label.setText("—")

        if classification is not None:
            name = ASPRS_CLASS_NAMES.get(classification, f"Unknown")
            self._class_label.setText(f"{classification}: {name}")
        else:
            self._class_label.setText("—")

        if intensity is not None:
            self._intensity_label.setText(str(intensity))
        else:
            self._intensity_label.setText("—")

        if return_number is not None:
            self._return_label.setText(str(return_number))
        else:
            self._return_label.setText("—")

        if z is not None:
            self._height_label.setText(f"{z:.3f} m")
        else:
            self._height_label.setText("—")

        # Auto-refresh non-modal info dialog if open
        if self._info_dlg is not None and self._info_dlg.isVisible():
            self._update_info_dialog()

    def set_selection_count(self, count: int) -> None:
        """Update the selected-point count label."""
        if count > 0:
            self._sel_count_label.setText(f"<b>{count:,} selected</b>")
        else:
            self._sel_count_label.setText("0 selected")

    def set_undo_state(self, can_undo: bool, can_redo: bool) -> None:
        """Enable or disable undo/redo buttons."""
        self._undo_btn.setEnabled(can_undo)
        self._redo_btn.setEnabled(can_redo)

    def set_undo_info(self, undo_count: int, redo_count: int) -> None:
        """Show undo/redo stack sizes."""
        self.set_undo_state(undo_count > 0, redo_count > 0)

    def _toggle_point_info_dialog(self) -> None:
        """Toggle the non-modal point info dialog on/off."""
        if self._info_dlg is not None and self._info_dlg.isVisible():
            self._info_dlg.close()
            self._info_dlg = None
            self._info_text = None
            self._info_btn.setChecked(False)
            return
        if self._tile_data is None:
            return
        self._info_dlg = QDialog(self)
        self._info_dlg.setWindowTitle("Point Attributes")
        self._info_dlg.setMinimumWidth(420)
        self._info_dlg.setAttribute(Qt.WA_DeleteOnClose, False)
        layout = QVBoxLayout(self._info_dlg)
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        layout.addWidget(self._info_text)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._info_dlg.accept)
        layout.addWidget(close_btn)
        self._info_dlg.finished.connect(self._on_info_dlg_closed)
        self._info_btn.setChecked(True)
        self._update_info_dialog()
        self._info_dlg.show()

    def _on_info_dlg_closed(self) -> None:
        self._info_dlg = None
        self._info_text = None
        self._info_btn.setChecked(False)

    def _update_info_dialog(self) -> None:
        """Refresh the contents of the non-modal info dialog."""
        if self._info_text is None or self._tile_data is None:
            return
        if self._current_point_idx < 0:
            self._info_text.setHtml(
                "<i>Hover over a point in the profile view, "
                "or click a point in the 3D view to see its attributes.</i>"
            )
            return
        idx = self._current_point_idx
        data = self._tile_data
        n = len(data.get("x", []))
        if idx >= n:
            return
        lines = [f"<b>Point index:</b> {idx} / {n:,}"]
        fields = [
            ("X", "x", ".3f"), ("Y", "y", ".3f"), ("Z", "z", ".3f"),
            ("Classification", "classification", "d"),
            ("Intensity", "intensity", "d"),
            ("Return Number", "return_number", "d"),
            ("Number of Returns", "num_returns", "d"),
            ("Point Source ID (Flightline)", "point_source_id", "d"),
            ("Sensor Type", "sensor_type", "sensor"),
            ("Scan Direction", "scan_direction_flag", "d"),
            ("Edge of Flight Line", "edge_of_flight_line", "d"),
            ("Scan Angle Rank", "scan_angle_rank", "d"),
            ("User Data", "user_data", "d"),
            ("GPS Time", "gps_time", ".6f"),
            ("Red", "red", "d"), ("Green", "green", "d"), ("Blue", "blue", "d"),
            ("Key Point", "key_point", "d"),
            ("Synthetic", "synthetic", "d"),
            ("Withheld", "withheld", "d"),
            ("Overlap", "overlap", "d"),
        ]
        for label, key, fmt in fields:
            arr = data.get(key)
            if arr is not None and idx < len(arr):
                val = arr[idx]
                if fmt == ".3f":
                    lines.append(f"<b>{label}:</b> {float(val):{fmt}}")
                elif fmt == ".6f":
                    lines.append(f"<b>{label}:</b> {float(val):{fmt}}")
                elif fmt == "sensor":
                    sensor_names = {0: "Unknown", 1: "Topography", 2: "Bathymetry"}
                    lines.append(f"<b>{label}:</b> {sensor_names.get(int(val), 'Unknown')}")
                else:
                    lines.append(f"<b>{label}:</b> {int(val)}")
        cls_arr = data.get("classification")
        if cls_arr is not None and idx < len(cls_arr):
            cls_code = int(cls_arr[idx])
            cls_name = ASPRS_CLASS_NAMES.get(cls_code, "Unknown")
            lines.append(f"<b>Class Name:</b> {cls_code} — {cls_name}")
        self._info_text.setHtml("<br>".join(lines))

    def set_tile_metadata(
        self,
        scanner: str = '',
        all_scanners: Optional[List[str]] = None,
        sensor_type: str = '',
        flight_line: int = 0,
        las_version: str = '',
        point_count: int = 0,
        crs: str = '',
    ) -> None:
        """
        Update the file metadata display with info from the database
        and/or LAS header.

        Args:
            scanner:      Dominant scanner name (e.g. 'vq820g').
            all_scanners: List of all scanner names in this tile.
            sensor_type:  ``'topo'``, ``'bathy'``, or ``''``.
            flight_line:  Flight-line number (0 = unknown).
            las_version:  LAS format version string (e.g. ``'1.4'``).
            point_count:  Number of points in the tile.
            crs:          Coordinate reference system description.
        """
        if all_scanners and len(all_scanners) > 0:
            # Show dominant first, then all
            scanner_text = scanner if scanner else all_scanners[0]
            if len(all_scanners) > 1:
                scanner_text += f" (+{len(all_scanners)-1} more: {', '.join(all_scanners[1:])})"
            self._scanner_label.setText(scanner_text)
        elif scanner:
            self._scanner_label.setText(scanner)
        else:
            self._scanner_label.setText("—")
        st_display = {'topo': 'Topography', 'bathy': 'Bathymetry'}
        self._sensor_type_label.setText(st_display.get(sensor_type, sensor_type) if sensor_type else "—")
        self._flight_line_label.setText(str(flight_line) if flight_line else "—")
        self._las_version_label.setText(las_version if las_version else "—")
        self._file_point_count_label.setText(f"{point_count:,}" if point_count else "—")
        self._crs_label.setText(crs if crs else "—")
