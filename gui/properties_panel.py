"""
LiDAR Workbench — Properties Panel.

Right-side panel showing point properties and quick-classification
buttons, context-sensitive to the active view and selection.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import ASPRS_CLASS_NAMES, get_class_color
from .settings_dialog import load_shortcuts

logger = logging.getLogger("lidar_workbench.gui.properties_panel")

# Quick-access classification buttons: (code, label, colour)
QUICK_CLASSES = [
    (2,  "Ground",           "#8B4513"),
    (3,  "Low Veg",          "#009900"),
    (4,  "Med Veg",          "#00CC00"),
    (5,  "High Veg",         "#33FF33"),
    (6,  "Building",         "#CC3333"),
    (9,  "Water",            "#0066CC"),
    (7,  "Low Pt (Noise)",   "#4D4D4D"),
    (1,  "Unclassified",     "#CCCCCC"),
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)
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

        # ── Quick classify ─────────────────────────────────────────
        classify_group = QGroupBox("Quick Classify")
        classify_layout = QVBoxLayout(classify_group)

        # Map class codes to shortcut keys
        _classify_keys = {
            2: "classify_ground", 3: "classify_low_veg",
            4: "classify_med_veg", 5: "classify_high_veg",
            6: "classify_building", 9: "classify_water",
            7: "classify_noise", 1: "classify_unclass",
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

    def set_point_info(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        classification: Optional[int] = None,
        intensity: Optional[int] = None,
        return_number: Optional[int] = None,
    ) -> None:
        """
        Update the point properties display.

        Pass ``None`` for any field to show "—".
        """
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
