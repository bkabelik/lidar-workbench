"""
LiDAR Workbench — Export Dialog.

A QDialog that lets the user configure DTM / DSM export parameters:
resolution, target classes, hillshade toggle, merged vs. tiled output.
Runs the export on a background QThread so the GUI stays responsive.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..config import ASPRS_CLASS_NAMES
from ..export_manager import ExportConfig, export_dtm, export_dsm, export_merged_raster

logger = logging.getLogger("lidar_workbench.gui.export_dialog")

# ── common ASPRS classes the user is likely to select for DSM ───────────
DSM_CLASS_OPTIONS: List[Tuple[int, str]] = [
    (2, "Ground"),
    (3, "Low Vegetation"),
    (4, "Medium Vegetation"),
    (5, "High Vegetation"),
    (6, "Building"),
    (7, "Low Point (Noise)"),
    (9, "Water"),
    (10, "Rail"),
    (11, "Road Surface"),
    (15, "Transmission Tower"),
    (17, "Bridge Deck"),
]


class _ExportWorker(QThread):
    """Background worker that calls the export engine."""

    progress = Signal(float, str)
    finished_ok = Signal(list)
    finished_err = Signal(str)

    def __init__(
        self,
        tile_points: Dict[str, Dict[str, np.ndarray]],
        tile_bboxes: Dict[str, Tuple[float, float, float, float]],
        config: ExportConfig,
        merged: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._tile_points = tile_points
        self._tile_bboxes = tile_bboxes
        self._config = config
        self._merged = merged

    def run(self) -> None:
        try:
            if self._merged:
                files = export_merged_raster(
                    self._tile_points,
                    self._tile_bboxes,
                    self._config,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                )
            elif self._config.mode == "dtm":
                files = export_dtm(
                    self._tile_points,
                    self._tile_bboxes,
                    self._config,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                )
            else:
                files = export_dsm(
                    self._tile_points,
                    self._tile_bboxes,
                    self._config,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                )
            self.finished_ok.emit(files)
        except Exception as exc:
            logger.exception("Export failed")
            self.finished_err.emit(str(exc))


class ExportDialog(QDialog):
    """
    Modal dialog for configuring raster export.

    Usage::

        dlg = ExportDialog(tile_ids, tile_points, tile_bboxes, project_dtm_dir, parent=self)
        if dlg.exec() == QDialog.Accepted:
            print("Export complete — files:", dlg.written_files)
    """

    def __init__(
        self,
        tile_ids: List[str],
        tile_points: Dict[str, Dict[str, np.ndarray]],
        tile_bboxes: Dict[str, Tuple[float, float, float, float]],
        default_output_dir: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tile_ids = tile_ids
        self._tile_points = tile_points
        self._tile_bboxes = tile_bboxes
        self._default_output_dir = default_output_dir
        self._worker: Optional[_ExportWorker] = None
        self.written_files: List[str] = []

        self.setWindowTitle("Export Raster (DTM / DSM)")
        self.setMinimumWidth(480)
        self._setup_ui()

    # ── UI construction ─────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Mode selection ──────────────────────────────────────────
        mode_group = QGroupBox("Export Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._dtm_radio = QRadioButton("DTM — Digital Terrain Model (ground only)")
        self._dsm_radio = QRadioButton("DSM — Digital Surface Model (highest point)")
        self._dtm_radio.setChecked(True)
        mode_layout.addWidget(self._dtm_radio)
        mode_layout.addWidget(self._dsm_radio)
        layout.addWidget(mode_group)

        self._dtm_radio.toggled.connect(self._on_mode_changed)

        # ── Resolution ──────────────────────────────────────────────
        res_group = QGroupBox("Resolution")
        res_form = QFormLayout(res_group)

        self._res_spin = QDoubleSpinBox()
        self._res_spin.setRange(0.05, 100.0)
        self._res_spin.setValue(1.0)
        self._res_spin.setSingleStep(0.1)
        self._res_spin.setDecimals(2)
        self._res_spin.setSuffix(" m")
        res_form.addRow("Cell size:", self._res_spin)

        self._res_presets = QComboBox()
        self._res_presets.addItems(["1.0 m", "0.5 m", "0.25 m", "2.0 m", "5.0 m", "Custom…"])
        self._res_presets.setCurrentIndex(0)
        self._res_presets.currentTextChanged.connect(self._on_preset_changed)
        res_form.addRow("Preset:", self._res_presets)
        layout.addWidget(res_group)

        # ── DSM class selection ─────────────────────────────────────
        self._dsm_class_group = QGroupBox("DSM Classes (points to include)")
        self._dsm_class_group.setEnabled(False)
        dsm_layout = QVBoxLayout(self._dsm_class_group)

        self._dsm_class_checks: Dict[int, QCheckBox] = {}
        for code, name in DSM_CLASS_OPTIONS:
            cb = QCheckBox(f"Class {code} — {name}")
            # Default: include vegetation & buildings
            cb.setChecked(code in {2, 3, 4, 5, 6})
            self._dsm_class_checks[code] = cb
            dsm_layout.addWidget(cb)

        # Select all / none buttons
        btn_row = QHBoxLayout()
        all_btn = QPushButton("Select All")
        none_btn = QPushButton("Select None")
        all_btn.clicked.connect(lambda: self._set_all_dsm_classes(True))
        none_btn.clicked.connect(lambda: self._set_all_dsm_classes(False))
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        btn_row.addStretch()
        dsm_layout.addLayout(btn_row)

        layout.addWidget(self._dsm_class_group)

        # ── Output options ──────────────────────────────────────────
        out_group = QGroupBox("Output")
        out_form = QFormLayout(out_group)

        # Output directory
        dir_row = QHBoxLayout()
        self._out_dir_label = QLabel(self._default_output_dir)
        self._out_dir_label.setWordWrap(True)
        dir_row.addWidget(self._out_dir_label, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(browse_btn)
        out_form.addRow("Directory:", dir_row)

        # Hillshade
        self._hillshade_check = QCheckBox("Also export hillshade raster")
        self._hillshade_check.setChecked(True)
        out_form.addRow(self._hillshade_check)

        # Merged vs tiled
        self._merged_check = QCheckBox("Export as single merged raster (instead of per-tile)")
        self._merged_check.setChecked(False)
        self._merged_check.setToolTip(
            "When checked, all tiles are combined into one large .asc file. "
            "Unchecked: one .asc per tile (seamless — aligned to common grid)."
        )
        out_form.addRow(self._merged_check)

        layout.addWidget(out_group)

        # ── Progress bar ────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # ── Dialog buttons ──────────────────────────────────────────
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._button_box.accepted.connect(self._on_export)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    # ── slots ───────────────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        dsm_mode = self._dsm_radio.isChecked()
        self._dsm_class_group.setEnabled(dsm_mode)

    def _on_preset_changed(self, text: str) -> None:
        try:
            val = float(text.split()[0])
            self._res_spin.setValue(val)
        except (ValueError, IndexError):
            pass  # "Custom…" — leave current value

    def _set_all_dsm_classes(self, checked: bool) -> None:
        for cb in self._dsm_class_checks.values():
            cb.setChecked(checked)

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output Directory", self._default_output_dir)
        if d:
            self._out_dir_label.setText(d)

    def _on_export(self) -> None:
        """Validate config and start the background export."""
        out_dir = self._out_dir_label.text()
        if not out_dir:
            QMessageBox.warning(self, "Missing Directory", "Please select an output directory.")
            return

        # Build config
        config = ExportConfig(
            mode="dtm" if self._dtm_radio.isChecked() else "dsm",
            resolution=self._res_spin.value(),
            tile_ids=list(self._tile_ids),
            output_dir=out_dir,
            compute_hillshade=self._hillshade_check.isChecked(),
        )

        if config.mode == "dsm":
            config.dsm_classes = {
                code for code, cb in self._dsm_class_checks.items() if cb.isChecked()
            }
            if not config.dsm_classes:
                QMessageBox.warning(
                    self, "No Classes Selected",
                    "Please select at least one class for the DSM export."
                )
                return

        # Disable UI during export
        self._set_ui_enabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Loading tile data…")

        merged = self._merged_check.isChecked()

        # Start worker
        self._worker = _ExportWorker(
            self._tile_points,
            self._tile_bboxes,
            config,
            merged,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.finished_err.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: float, msg: str) -> None:
        self._progress_bar.setValue(int(pct))
        self._status_label.setText(msg)

    def _on_finished(self, files: List[str]) -> None:
        self.written_files = files
        self._progress_bar.setValue(100)
        self._status_label.setText(f"Done — {len(files)} file(s) written.")

        QMessageBox.information(
            self,
            "Export Complete",
            f"Wrote {len(files)} file(s) to:\n{self._out_dir_label.text()}\n\n"
            + ("\n".join(Path(f).name for f in files[:20]))
            + ("\n…" if len(files) > 20 else ""),
        )
        self.accept()

    def _on_error(self, msg: str) -> None:
        self._status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Export Error", msg)
        self._set_ui_enabled(True)
        self._progress_bar.setVisible(False)

    def _set_ui_enabled(self, enabled: bool) -> None:
        self._dtm_radio.setEnabled(enabled)
        self._dsm_radio.setEnabled(enabled)
        self._res_spin.setEnabled(enabled)
        self._dsm_class_group.setEnabled(enabled and self._dsm_radio.isChecked())
        self._hillshade_check.setEnabled(enabled)
        self._merged_check.setEnabled(enabled)
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(enabled)
