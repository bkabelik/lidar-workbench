"""
LiDAR Workbench — Classification Configuration Dialog.

Provides a dialog for configuring Pointcept inference parameters
before launching the background classification worker.  User settings
(paths and parameters) are persisted between sessions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_POINTCEPT_PATH,
)
from ..database import Database
from ..pointcept_worker import PointceptWorker
from ..tile_manager import TileManager

logger = logging.getLogger("lidar_workbench.gui.classification_dialog")

_SETTINGS_FILE = ".pointcept_settings.json"


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict) -> None:
    try:
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save Pointcept settings: %s", exc)


class ClassificationDialog(QDialog):
    """
    Dialog for configuring and launching Pointcept classification.

    The user sets the model path, config file, Pointcept root,
    and inference hyperparameters, then clicks "Start" to launch
    the worker in a background thread.  All settings are persisted
    to ``.pointcept_settings.json``.
    """

    def __init__(
        self,
        tile_manager: TileManager,
        database: Database,
        tile_ids: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tm = tile_manager
        self._db = database
        self._tile_ids = tile_ids
        self._worker: Optional[PointceptWorker] = None

        self.setWindowTitle("Pointcept Classification")
        self.setMinimumSize(550, 400)
        self._setup_ui()
        self._restore_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Paths ──────────────────────────────────────────────────
        paths_group = QGroupBox("Paths")
        paths_form = QFormLayout(paths_group)

        self._pointcept_edit = self._make_path_row(
            paths_form, "Pointcept Root:", DEFAULT_POINTCEPT_PATH, directory=True
        )
        self._model_edit = self._make_path_row(
            paths_form, "Model (.pth):", DEFAULT_MODEL_PATH, directory=False
        )
        self._config_edit = self._make_path_row(
            paths_form, "Config (.py):", DEFAULT_CONFIG_PATH, directory=False
        )
        py_row = QHBoxLayout()
        self._python_edit = QLineEdit()
        self._python_edit.setPlaceholderText("Auto-detect (sys.executable)")
        py_browse = QPushButton("…")
        py_browse.setFixedWidth(30)
        py_browse.clicked.connect(
            lambda: self._browse_file(self._python_edit, "Select Python Interpreter", False)
        )
        py_row.addWidget(self._python_edit)
        py_row.addWidget(py_browse)
        paths_form.addRow("Python:", py_row)

        layout.addWidget(paths_group)

        # ── Inference parameters ───────────────────────────────────
        infer_group = QGroupBox("Inference Parameters")
        infer_form = QFormLayout(infer_group)

        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setRange(0.05, 2.0)
        self._voxel_spin.setSingleStep(0.05)
        self._voxel_spin.setValue(0.15)
        self._voxel_spin.setDecimals(2)
        self._voxel_spin.setSuffix(" m")
        self._voxel_spin.setToolTip(
            "Voxel size for density normalisation (should be > line spacing, e.g. 0.15 m)"
        )
        infer_form.addRow("Voxel Size:", self._voxel_spin)

        self._smoothing_check = QCheckBox("Enable k-NN smoothing (recommended)")
        self._smoothing_check.setChecked(True)
        self._smoothing_check.setToolTip(
            "Apply edge-preserving k-NN majority voting to smooth predictions"
        )
        infer_form.addRow(self._smoothing_check)

        layout.addWidget(infer_group)

        # ── Info ───────────────────────────────────────────────────
        info_label = QLabel(
            f"<b>{len(self._tile_ids)} tile(s)</b> selected for classification.\n"
            "Intensity scale is computed automatically (97th percentile).\n"
            "Noise filter is disabled (headless mode).\n"
            "A backup (.bak) is created before overwriting the LAS file."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # ── Progress ───────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)

        # ── Buttons ────────────────────────────────────────────────
        button_box = QDialogButtonBox()
        self._start_btn = QPushButton("Start Classification")
        self._start_btn.clicked.connect(self._on_start)
        button_box.addButton(self._start_btn, QDialogButtonBox.ActionRole)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        button_box.addButton(self._cancel_btn, QDialogButtonBox.RejectRole)

        close_btn = button_box.addButton(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.close)

        layout.addWidget(button_box)

    # ── settings persistence ───────────────────────────────────────

    def _restore_settings(self) -> None:
        s = _load_settings()
        if s.get("pointcept_path"):
            self._pointcept_edit.setText(s["pointcept_path"])
        if s.get("model_path"):
            self._model_edit.setText(s["model_path"])
        if s.get("config_path"):
            self._config_edit.setText(s["config_path"])
        if s.get("python_exe"):
            self._python_edit.setText(s["python_exe"])
        if "voxel_size" in s:
            self._voxel_spin.setValue(s["voxel_size"])
        if "smoothing" in s:
            self._smoothing_check.setChecked(s["smoothing"])

    def _persist_settings(self) -> None:
        _save_settings({
            "pointcept_path": self._pointcept_edit.text().strip(),
            "model_path": self._model_edit.text().strip(),
            "config_path": self._config_edit.text().strip(),
            "python_exe": self._python_edit.text().strip(),
            "voxel_size": self._voxel_spin.value(),
            "smoothing": self._smoothing_check.isChecked(),
        })

    # ── browse ─────────────────────────────────────────────────────

    def _make_path_row(
        self, form: QFormLayout, label: str, default: str, directory: bool
    ) -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit(default)
        edit.setMinimumWidth(300)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        if directory:
            browse.clicked.connect(
                lambda: self._browse_file(edit, "Select Directory", True)
            )
        else:
            browse.clicked.connect(
                lambda: self._browse_file(edit, "Select File", False)
            )
        row.addWidget(edit)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _browse_file(
        self, edit: QLineEdit, title: str, directory: bool
    ) -> None:
        if directory:
            path = QFileDialog.getExistingDirectory(self, title)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title)
        if path:
            edit.setText(path)

    # ── start / cancel ─────────────────────────────────────────────

    def _on_start(self) -> None:
        pointcept_path = self._pointcept_edit.text().strip()
        model_path = self._model_edit.text().strip()
        config_path = self._config_edit.text().strip()
        python_exe = self._python_edit.text().strip() or None

        if not pointcept_path or not Path(pointcept_path).is_dir():
            QMessageBox.warning(self, "Invalid Path", "Pointcept root directory not found.")
            return
        if not model_path or not Path(model_path).is_file():
            QMessageBox.warning(self, "Invalid Path", "Model checkpoint file not found.")
            return
        if not config_path:
            QMessageBox.warning(self, "Invalid Path", "Config file path is empty.")
            return

        self._persist_settings()

        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Starting Pointcept…")

        smoothing = "yes" if self._smoothing_check.isChecked() else "no"

        from .settings_dialog import load_general_settings
        settings = load_general_settings()
        workers = settings.get("classify_workers", 1)

        self._worker = PointceptWorker(
            self._tm,
            self._db,
            self._tile_ids,
            pointcept_path=pointcept_path,
            model_path=model_path,
            config_path=config_path,
            python_exe=python_exe,
            voxel_size=self._voxel_spin.value(),
            smoothing=smoothing,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.tile_done.connect(self._on_tile_done)
        self._worker.tile_error.connect(self._on_tile_error)
        self._worker.all_done.connect(self._on_all_done)
        self._worker._workers = workers
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._status_label.setText("Cancelling…")

    # ── worker signal handlers ─────────────────────────────────────

    def _on_progress(self, tile_id: str, step: str, pct: float) -> None:
        self._status_label.setText(step)
        self._progress_bar.setValue(int(pct))

    def _on_tile_done(self, tile_id: str) -> None:
        logger.info("Tile %s classified successfully", tile_id)

    def _on_tile_error(self, tile_id: str, error: str) -> None:
        logger.error("Tile %s failed: %s", tile_id, error)
        QMessageBox.warning(
            self,
            "Classification Error",
            f"Tile {tile_id} failed:\n\n{error[:400]}",
        )

    def _on_all_done(self, completed: List[str]) -> None:
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setValue(100)
        self._status_label.setText(
            f"Done — {len(completed)}/{len(self._tile_ids)} tile(s) classified."
        )

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)
