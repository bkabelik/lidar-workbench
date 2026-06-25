"""
LiDAR Workbench — Settings Dialog.

Allows the user to configure keyboard shortcuts for application actions.
Shortcuts are persisted to ``.shortcuts.json``.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("lidar_workbench.gui.settings_dialog")

_SHORTCUTS_FILE = ".shortcuts.json"
_SHORTCUTS_FILE = ".settings.json"


def load_general_settings() -> dict:
    """Load general settings, falling back to defaults."""
    from ..config import DEFAULT_FILTER_WORKERS, DEFAULT_CLASSIFY_WORKERS
    defaults = {"filter_workers": DEFAULT_FILTER_WORKERS,
                "classify_workers": DEFAULT_CLASSIFY_WORKERS}
    try:
        with open(_SHORTCUTS_FILE, "r") as f:
            saved = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    defaults.update({k: v for k, v in saved.items() if k in defaults})
    return defaults


def save_general_settings(settings: dict) -> None:
    """Persist general settings to disk."""
    try:
        with open(_SHORTCUTS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save settings: %s", exc)

# Default shortcuts
DEFAULTS: Dict[str, str] = {
    "new_project": "Ctrl+N",
    "open_project": "Ctrl+O",
    "save_project": "Ctrl+S",
    "import_las": "Ctrl+I",
    "filter": "Ctrl+F",
    "classify": "Ctrl+Shift+C",
    "undo": "Ctrl+Z",
    "redo": "Ctrl+Y",
    "sel_brush": "B",
    "sel_above": "A",
    "sel_below": "L",
    "sel_rectangle": "R",
    "next_tile": "Tab",
    "prev_tile": "Shift+Tab",
    "classify_ground": "Ctrl+2",
    "classify_low_veg": "Ctrl+3",
    "classify_med_veg": "Ctrl+4",
    "classify_high_veg": "Ctrl+5",
    "classify_building": "Ctrl+6",
    "classify_water": "Ctrl+9",
    "classify_noise": "Ctrl+7",
    "classify_unclass": "Ctrl+1",
}

ACTION_LABELS: Dict[str, str] = {
    "new_project": "New Project",
    "open_project": "Open Project",
    "save_project": "Save Project",
    "import_las": "Import LAS/LAZ",
    "filter": "Noise Filter",
    "classify": "Classify (Pointcept)",
    "undo": "Undo",
    "redo": "Redo",
    "sel_brush": "Select: Brush",
    "sel_above": "Select: Above Line",
    "sel_below": "Select: Below Line",
    "sel_rectangle": "Select: Rectangle",
    "next_tile": "Next Tile",
    "prev_tile": "Previous Tile",
    "classify_ground": "Quick Classify: Ground (2)",
    "classify_low_veg": "Quick Classify: Low Veg (3)",
    "classify_med_veg": "Quick Classify: Med Veg (4)",
    "classify_high_veg": "Quick Classify: High Veg (5)",
    "classify_building": "Quick Classify: Building (6)",
    "classify_water": "Quick Classify: Water (9)",
    "classify_noise": "Quick Classify: Low Pt Noise (7)",
    "classify_unclass": "Quick Classify: Unclassified (1)",
}


def load_shortcuts() -> Dict[str, str]:
    """Load saved shortcuts, falling back to defaults."""
    try:
        with open(_SHORTCUTS_FILE, "r") as f:
            saved = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    result = dict(DEFAULTS)
    result.update({k: v for k, v in saved.items() if k in DEFAULTS})
    return result


def save_shortcuts(shortcuts: Dict[str, str]) -> None:
    """Persist shortcuts to disk."""
    try:
        with open(_SHORTCUTS_FILE, "w") as f:
            json.dump(shortcuts, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save shortcuts: %s", exc)


class SettingsDialog(QDialog):
    """
    Dialog for editing keyboard shortcuts.

    Each row shows the action name and its current shortcut.
    The user can click a cell and press the desired key combination.
    """

    shortcuts_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._shortcuts = load_shortcuts()
        self._settings = load_general_settings()
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 450)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Double-click a shortcut to edit. Press Esc to cancel or\n"
            "any key combination to set a new shortcut."
        ))

        self._table = QTableWidget(len(ACTION_LABELS), 2)
        self._table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 220)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        for row, (key, label) in enumerate(ACTION_LABELS.items()):
            self._table.setItem(row, 0, QTableWidgetItem(label))
            ks = self._shortcuts.get(key, DEFAULTS.get(key, ""))
            self._table.setItem(row, 1, QTableWidgetItem(ks))

        layout.addWidget(self._table)

        # ── General settings ───────────────────────────────────────
        layout.addWidget(QLabel("<b>General</b>"))
        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("Filter parallel workers:"))
        from PySide6.QtWidgets import QSpinBox
        self._filter_workers_spin = QSpinBox()
        self._filter_workers_spin.setRange(1, 16)
        self._filter_workers_spin.setValue(self._settings.get("filter_workers", 4))
        self._filter_workers_spin.setToolTip("Number of tiles to filter in parallel")
        gen_layout.addWidget(self._filter_workers_spin)
        gen_layout.addStretch()
        layout.addLayout(gen_layout)

        gen2_layout = QHBoxLayout()
        gen2_layout.addWidget(QLabel("Classify parallel workers:"))
        self._classify_workers_spin = QSpinBox()
        self._classify_workers_spin.setRange(1, 8)
        self._classify_workers_spin.setValue(self._settings.get("classify_workers", 1))
        self._classify_workers_spin.setToolTip("Number of tiles to classify in parallel (GPU memory limited)")
        gen2_layout.addWidget(self._classify_workers_spin)
        gen2_layout.addStretch()
        layout.addLayout(gen2_layout)

        # Buttons
        btn_box = QDialogButtonBox()
        reset_btn = QPushButton("Restore Defaults")
        reset_btn.clicked.connect(self._on_reset)
        btn_box.addButton(reset_btn, QDialogButtonBox.ResetRole)

        ok_btn = btn_box.addButton(QDialogButtonBox.Ok)
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = btn_box.addButton(QDialogButtonBox.Cancel)
        cancel_btn.clicked.connect(self.reject)

        layout.addWidget(btn_box)

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Open an inline shortcut capture dialog."""
        action_key = list(ACTION_LABELS.keys())[row]
        current = self._table.item(row, 1).text()

        dlg = ShortcutCaptureDialog(action_key, current, self)
        if dlg.exec() == QDialog.Accepted:
            self._table.item(row, 1).setText(dlg.shortcut_text)
            self._shortcuts[action_key] = dlg.shortcut_text

    def _on_reset(self) -> None:
        self._shortcuts = dict(DEFAULTS)
        for row, key in enumerate(ACTION_LABELS):
            self._table.item(row, 1).setText(DEFAULTS.get(key, ""))

    def _on_accept(self) -> None:
        save_shortcuts(self._shortcuts)
        self._settings["filter_workers"] = self._filter_workers_spin.value()
        self._settings["classify_workers"] = self._classify_workers_spin.value()
        save_general_settings(self._settings)
        self.shortcuts_changed.emit(self._shortcuts)
        self.accept()


class ShortcutCaptureDialog(QDialog):
    """Small modal dialog that captures a key press as a shortcut."""

    def __init__(
        self, action_key: str, current: str, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.shortcut_text = current
        self.setWindowTitle(f"Set Shortcut — {ACTION_LABELS.get(action_key, action_key)}")
        self.setFixedSize(350, 130)

        layout = QVBoxLayout(self)
        self._label = QLabel(
            f"Press the desired key combination for:\n"
            f"<b>{ACTION_LABELS.get(action_key, action_key)}</b>\n\n"
            f"<i>Press Esc to cancel</i>"
        )
        self._label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._label)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        seq = QKeySequence(event.modifiers() | event.key())
        text = seq.toString()
        if text:
            self.shortcut_text = text
            self.accept()
