"""
LiDAR Workbench — Settings Dialog.

Allows the user to configure general preferences, shared network storage,
AI / Pointcept inference paths, and keyboard shortcuts.
All settings are persisted permanently to %APPDATA%/LiDAR_Workbench/settings.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_BATHY_WORKERS,
    DEFAULT_BRUSH_RADIUS_M,
    DEFAULT_CLASSIFY_WORKERS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_FILTER_WORKERS,
    DEFAULT_GROUND_WORKERS,
    DEFAULT_MODEL_PATH,
    DEFAULT_POINTCEPT_PATH,
    DEFAULT_RECT_HEIGHT_M,
    DEFAULT_RECT_WIDTH_M,
    USER_SETTINGS_FILE,
    load_all_settings,
    save_all_settings,
    load_general_settings,
    save_general_settings,
    load_pointcept_settings,
    save_pointcept_settings,
)

logger = logging.getLogger("lidar_workbench.gui.settings_dialog")


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
    "sel_rect_brush": "Shift+R",
    "next_tile": "Tab",
    "prev_tile": "Shift+Tab",
    "classify_created": "Ctrl+0",
    "classify_unclass": "Ctrl+1",
    "classify_ground": "Ctrl+2",
    "classify_low_veg": "Ctrl+3",
    "classify_med_veg": "Ctrl+4",
    "classify_high_veg": "Ctrl+5",
    "classify_building": "Ctrl+6",
    "classify_noise": "Ctrl+7",
    "classify_model_key": "Ctrl+8",
    "classify_water": "Ctrl+9",
    "classify_rail": "Ctrl+Shift+0",
    "classify_road": "Ctrl+Shift+1",
    "classify_overlap": "Ctrl+Shift+2",
    "classify_wire_guard": "Ctrl+Shift+3",
    "classify_wire_conductor": "Ctrl+Shift+4",
    "classify_tower": "Ctrl+Shift+5",
    "classify_wire_connector": "Ctrl+Shift+6",
    "classify_bridge": "Ctrl+Shift+7",
    "classify_high_noise": "Ctrl+Shift+8",
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
    "sel_rect_brush": "Select: Rect Brush",
    "next_tile": "Next Tile",
    "prev_tile": "Previous Tile",
    "classify_created": "Quick Classify: Created (0)",
    "classify_unclass": "Quick Classify: Unclassified (1)",
    "classify_ground": "Quick Classify: Ground (2)",
    "classify_low_veg": "Quick Classify: Low Veg (3)",
    "classify_med_veg": "Quick Classify: Med Veg (4)",
    "classify_high_veg": "Quick Classify: High Veg (5)",
    "classify_building": "Quick Classify: Building (6)",
    "classify_noise": "Quick Classify: Low Pt Noise (7)",
    "classify_model_key": "Quick Classify: Model Key (8)",
    "classify_water": "Quick Classify: Water (9)",
    "classify_rail": "Quick Classify: Rail (10)",
    "classify_road": "Quick Classify: Road Surface (11)",
    "classify_overlap": "Quick Classify: Overlap (12)",
    "classify_wire_guard": "Quick Classify: Wire Guard (13)",
    "classify_wire_conductor": "Quick Classify: Wire Conductor (14)",
    "classify_tower": "Quick Classify: Tower (15)",
    "classify_wire_connector": "Quick Classify: Wire Connector (16)",
    "classify_bridge": "Quick Classify: Bridge Deck (17)",
    "classify_high_noise": "Quick Classify: High Noise (18)",
}


def load_shortcuts() -> Dict[str, str]:
    """Load saved shortcuts, falling back to defaults."""
    all_settings = load_all_settings()
    saved = all_settings.get("shortcuts", {})
    if not saved and any(k in all_settings for k in DEFAULTS):
        saved = all_settings
    result = dict(DEFAULTS)
    result.update({k: v for k, v in saved.items() if k in DEFAULTS})
    return result


def save_shortcuts(shortcuts: Dict[str, str]) -> None:
    """Persist shortcuts to disk."""
    all_settings = load_all_settings()
    all_settings["shortcuts"] = shortcuts
    save_all_settings(all_settings)


class SettingsDialog(QDialog):
    """
    Dialog for configuring application preferences, AI/Pointcept paths,
    shared cluster storage, and keyboard shortcuts.
    """

    shortcuts_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._shortcuts = load_shortcuts()
        self._general = load_general_settings()
        self._pointcept = load_pointcept_settings()
        self.setWindowTitle("Settings")
        self.setMinimumSize(620, 520)
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        self._tabs = QTabWidget(self)

        # ── Tab 1: General & Shared Storage ───────────────────────────
        gen_tab = QWidget()
        gen_layout = QVBoxLayout(gen_tab)
        gen_layout.setSpacing(14)

        workers_group = QGroupBox("Parallel Processing Workers")
        workers_form = QFormLayout(workers_group)

        self._filter_workers_spin = QSpinBox()
        self._filter_workers_spin.setRange(1, 32)
        self._filter_workers_spin.setValue(self._general.get("filter_workers", 4))
        workers_form.addRow("Noise Filter Workers:", self._filter_workers_spin)

        self._classify_workers_spin = QSpinBox()
        self._classify_workers_spin.setRange(1, 8)
        self._classify_workers_spin.setValue(self._general.get("classify_workers", 1))
        workers_form.addRow("Local Classification Workers:", self._classify_workers_spin)

        self._bathy_workers_spin = QSpinBox()
        self._bathy_workers_spin.setRange(1, 32)
        self._bathy_workers_spin.setValue(self._general.get("bathy_workers", 4))
        workers_form.addRow("Bathymetry Workers:", self._bathy_workers_spin)

        self._ground_workers_spin = QSpinBox()
        self._ground_workers_spin.setRange(1, 32)
        self._ground_workers_spin.setValue(self._general.get("ground_workers", 4))
        workers_form.addRow("Ground Classification Workers:", self._ground_workers_spin)

        gen_layout.addWidget(workers_group)

        tools_group = QGroupBox("Manual Edit Tools")
        tools_form = QFormLayout(tools_group)

        self._brush_radius_spin = QDoubleSpinBox()
        self._brush_radius_spin.setRange(0.1, 50.0)
        self._brush_radius_spin.setSingleStep(0.5)
        self._brush_radius_spin.setDecimals(1)
        self._brush_radius_spin.setValue(self._general.get("brush_radius", 2.0))
        tools_form.addRow("Brush Radius (m):", self._brush_radius_spin)

        self._rect_width_spin = QDoubleSpinBox()
        self._rect_width_spin.setRange(0.1, 100.0)
        self._rect_width_spin.setSingleStep(0.5)
        self._rect_width_spin.setDecimals(1)
        self._rect_width_spin.setValue(self._general.get("rect_width", 4.0))
        tools_form.addRow("Rect Brush Width (m):", self._rect_width_spin)

        self._rect_height_spin = QDoubleSpinBox()
        self._rect_height_spin.setRange(0.1, 100.0)
        self._rect_height_spin.setSingleStep(0.5)
        self._rect_height_spin.setDecimals(1)
        self._rect_height_spin.setValue(self._general.get("rect_height", 2.0))
        tools_form.addRow("Rect Brush Height (m):", self._rect_height_spin)

        gen_layout.addWidget(tools_group)

        storage_group = QGroupBox("Distributed Cluster — Shared Network Storage")
        storage_vbox = QVBoxLayout(storage_group)
        storage_vbox.addWidget(
            QLabel("Root folder or network share where survey projects are stored (e.g. <code>\\\\server\\share\\projects</code> or <code>/mnt/projects</code>):")
        )
        storage_row = QHBoxLayout()
        self._storage_edit = QLineEdit(self._general.get("shared_storage_root", ""))
        self._storage_edit.setPlaceholderText("Optional: e.g. \\\\nas01\\lidar_projects or Z:\\projects")
        storage_browse = QPushButton("Browse…")
        storage_browse.clicked.connect(self._browse_storage_root)
        storage_row.addWidget(self._storage_edit)
        storage_row.addWidget(storage_browse)
        storage_vbox.addLayout(storage_row)

        gen_layout.addWidget(storage_group)
        gen_layout.addStretch()

        self._tabs.addTab(gen_tab, "General & Storage")

        # ── Tab 2: AI & Pointcept ─────────────────────────────────────
        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        ai_layout.setSpacing(14)

        paths_group = QGroupBox("Pointcept Paths")
        paths_form = QFormLayout(paths_group)

        self._pc_root_edit = self._make_browse_row(
            paths_form, "Pointcept Root:", self._pointcept.get("pointcept_path", DEFAULT_POINTCEPT_PATH), is_dir=True
        )
        self._model_edit = self._make_browse_row(
            paths_form, "Model Checkpoint (.pth):", self._pointcept.get("model_path", DEFAULT_MODEL_PATH), is_dir=False, filter_str="PyTorch Model (*.pth *.pt)"
        )
        self._config_edit = self._make_browse_row(
            paths_form, "Model Config (.py):", self._pointcept.get("config_path", DEFAULT_CONFIG_PATH), is_dir=False, filter_str="Python Config (*.py)"
        )
        self._python_edit = self._make_browse_row(
            paths_form, "Python Interpreter:", self._pointcept.get("python_exe", sys.executable), is_dir=False, filter_str="Executables (*.exe python*);;All Files (*)"
        )
        ai_layout.addWidget(paths_group)

        infer_group = QGroupBox("Default Inference Parameters")
        infer_form = QFormLayout(infer_group)

        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setRange(0.05, 5.0)
        self._voxel_spin.setSingleStep(0.05)
        self._voxel_spin.setDecimals(2)
        self._voxel_spin.setValue(self._pointcept.get("voxel_size", 0.15))
        infer_form.addRow("Default Voxel Size (m):", self._voxel_spin)

        self._smoothing_chk = QCheckBox("Enable majority voting spatial smoothing")
        self._smoothing_chk.setChecked(self._pointcept.get("smoothing", True))
        infer_form.addRow("Smoothing:", self._smoothing_chk)

        ai_layout.addWidget(infer_group)
        ai_layout.addStretch()

        self._tabs.addTab(ai_tab, "AI & Pointcept")

        # ── Tab 3: Keyboard Shortcuts ─────────────────────────────────
        shortcuts_tab = QWidget()
        shortcuts_layout = QVBoxLayout(shortcuts_tab)

        shortcuts_layout.addWidget(QLabel(
            "Double-click any row to assign a new key combination. Press Esc to cancel."
        ))

        self._table = QTableWidget(len(ACTION_LABELS), 2)
        self._table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 240)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        for row, (key, label) in enumerate(ACTION_LABELS.items()):
            self._table.setItem(row, 0, QTableWidgetItem(label))
            ks = self._shortcuts.get(key, DEFAULTS.get(key, ""))
            self._table.setItem(row, 1, QTableWidgetItem(ks))

        shortcuts_layout.addWidget(self._table)
        self._tabs.addTab(shortcuts_tab, "Shortcuts")

        main_layout.addWidget(self._tabs)

        # ── Buttons ───────────────────────────────────────────────────
        btn_box = QDialogButtonBox()
        reset_btn = QPushButton("Restore Defaults")
        reset_btn.clicked.connect(self._on_reset)
        btn_box.addButton(reset_btn, QDialogButtonBox.ResetRole)

        ok_btn = btn_box.addButton(QDialogButtonBox.Ok)
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = btn_box.addButton(QDialogButtonBox.Cancel)
        cancel_btn.clicked.connect(self.reject)

        main_layout.addWidget(btn_box)

    def _make_browse_row(self, form: QFormLayout, label: str, init_val: str, is_dir: bool, filter_str: str = "") -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit(init_val)
        btn = QPushButton("Browse…")
        if is_dir:
            btn.clicked.connect(lambda: self._browse_dir(edit, f"Select {label}"))
        else:
            btn.clicked.connect(lambda: self._browse_file(edit, f"Select {label}", filter_str))
        row.addWidget(edit)
        row.addWidget(btn)
        form.addRow(label, row)
        return edit

    def _browse_dir(self, line_edit: QLineEdit, title: str) -> None:
        cur = line_edit.text().strip() or str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, title, cur)
        if chosen:
            line_edit.setText(chosen)

    def _browse_file(self, line_edit: QLineEdit, title: str, filter_str: str) -> None:
        cur = line_edit.text().strip() or str(Path.cwd())
        chosen, _ = QFileDialog.getOpenFileName(self, title, cur, filter_str)
        if chosen:
            line_edit.setText(chosen)

    def _browse_storage_root(self) -> None:
        cur = self._storage_edit.text().strip() or str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "Select Shared Projects Directory / NAS Share", cur)
        if chosen:
            self._storage_edit.setText(chosen)

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Open an inline shortcut capture dialog."""
        action_key = list(ACTION_LABELS.keys())[row]
        current = self._table.item(row, 1).text()

        dlg = ShortcutCaptureDialog(action_key, current, self)
        if dlg.exec() == QDialog.Accepted:
            self._table.item(row, 1).setText(dlg.shortcut_text)
            self._shortcuts[action_key] = dlg.shortcut_text

    def _on_reset(self) -> None:
        current_tab = self._tabs.currentIndex()
        if current_tab == 2:  # Shortcuts
            self._shortcuts = dict(DEFAULTS)
            for row, key in enumerate(ACTION_LABELS):
                self._table.item(row, 1).setText(DEFAULTS.get(key, ""))
        elif current_tab == 1:  # AI
            self._pc_root_edit.setText(DEFAULT_POINTCEPT_PATH)
            self._model_edit.setText(DEFAULT_MODEL_PATH)
            self._config_edit.setText(DEFAULT_CONFIG_PATH)
            self._python_edit.setText(sys.executable)
            self._voxel_spin.setValue(0.5)
            self._smoothing_chk.setChecked(True)
        elif current_tab == 0:  # General
            self._filter_workers_spin.setValue(DEFAULT_FILTER_WORKERS)
            self._classify_workers_spin.setValue(DEFAULT_CLASSIFY_WORKERS)
            self._bathy_workers_spin.setValue(DEFAULT_BATHY_WORKERS)
            self._ground_workers_spin.setValue(DEFAULT_GROUND_WORKERS)
            self._brush_radius_spin.setValue(DEFAULT_BRUSH_RADIUS_M)
            self._rect_width_spin.setValue(DEFAULT_RECT_WIDTH_M)
            self._rect_height_spin.setValue(DEFAULT_RECT_HEIGHT_M)
            self._storage_edit.setText("")

    def _on_accept(self) -> None:
        # Save Shortcuts
        save_shortcuts(self._shortcuts)

        # Save General
        self._general["filter_workers"] = self._filter_workers_spin.value()
        self._general["classify_workers"] = self._classify_workers_spin.value()
        self._general["bathy_workers"] = self._bathy_workers_spin.value()
        self._general["ground_workers"] = self._ground_workers_spin.value()
        self._general["brush_radius"] = self._brush_radius_spin.value()
        self._general["rect_width"] = self._rect_width_spin.value()
        self._general["rect_height"] = self._rect_height_spin.value()
        self._general["shared_storage_root"] = self._storage_edit.text().strip()
        save_general_settings(self._general)

        # Save Pointcept / AI
        self._pointcept["pointcept_path"] = self._pc_root_edit.text().strip()
        self._pointcept["model_path"] = self._model_edit.text().strip()
        self._pointcept["config_path"] = self._config_edit.text().strip()
        self._pointcept["python_exe"] = self._python_edit.text().strip()
        self._pointcept["voxel_size"] = self._voxel_spin.value()
        self._pointcept["smoothing"] = self._smoothing_chk.isChecked()
        save_pointcept_settings(self._pointcept)

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
