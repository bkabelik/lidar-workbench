"""
LiDAR Workbench — Export Dialog.

A QDialog that lets the user configure DTM / DSM export parameters:
resolution, target classes, hillshade toggle, merged vs. tiled output.
Runs the export on a background QThread so the GUI stays responsive.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from ..export_manager import (
    ExportConfig,
    export_dsm_parallel,
    export_dtm_parallel,
    export_merged_raster_parallel,
)

logger = logging.getLogger("lidar_workbench.gui.export_dialog")

# ── common ASPRS classes the user is likely to select for DSM ───────────
DSM_CLASS_OPTIONS: List[Tuple[int, str]] = [
    (1, "Unassigned"),
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
    (20, "Ignored Ground"),
]


class _ExportWorker(QThread):
    """Background worker that calls the parallel raster export engine."""

    progress = Signal(float, str)
    finished_ok = Signal(list)
    finished_err = Signal(str)

    def __init__(
        self,
        jobs: list,
        global_bbox: Tuple[float, float, float, float],
        config: ExportConfig,
        merged: bool,
        workers: int = 4,
        parent=None,
    ):
        super().__init__(parent)
        self._jobs = jobs
        self._global_bbox = global_bbox
        self._config = config
        self._merged = merged
        self._workers = workers

    def run(self) -> None:
        try:
            if self._merged:
                files = export_merged_raster_parallel(
                    self._jobs,
                    self._global_bbox,
                    self._config,
                    workers=self._workers,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                )
            elif self._config.mode == "dtm":
                files = export_dtm_parallel(
                    self._jobs,
                    self._global_bbox,
                    self._config,
                    workers=self._workers,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                )
            else:
                files = export_dsm_parallel(
                    self._jobs,
                    self._global_bbox,
                    self._config,
                    workers=self._workers,
                    progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
                )
            self.finished_ok.emit(files)
        except Exception as exc:
            logger.exception("Export failed")
            self.finished_err.emit(str(exc))


class ExportDialog(QDialog):
    """
    Modal dialog for configuring raster export.

    Tile rasters are computed in parallel worker processes via *jobs* and
    *global_bbox*, so large projects never hold all point clouds in memory
    and use all configured CPU cores.

    Usage::

        dlg = ExportDialog(jobs, global_bbox, project_dtm_dir, workers=4, parent=self)
        if dlg.exec() == QDialog.Accepted:
            print("Export complete — files:", dlg.written_files)
    """

    def __init__(
        self,
        jobs: List[dict],
        global_bbox: Tuple[float, float, float, float],
        default_output_dir: str,
        workers: int = 4,
        crs_epsg: Optional[int] = None,
        crs_wkt: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = list(jobs)
        self._global_bbox = global_bbox
        self._default_output_dir = default_output_dir
        self._workers = max(1, int(workers))
        self._crs_epsg = crs_epsg
        self._crs_wkt = crs_wkt
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
            # Default: include surface points, structures, bridge decks, roads, and water
            cb.setChecked(code in {1, 2, 3, 4, 5, 6, 9, 10, 11, 17, 20})
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

        # Void filling option
        self._fill_voids_check = QCheckBox("Fill voids / holes (interpolate over water, bridges, and data gaps)")
        self._fill_voids_check.setChecked(True)
        self._fill_voids_check.setToolTip(
            "When checked, missing data areas like unreflected water bodies (lakes, rivers) and "
            "data gaps are smoothly interpolated at the water surface or surrounding terrain level."
        )
        dsm_layout.addWidget(self._fill_voids_check)

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
            "Tiles are composited one at a time, so memory stays bounded. "
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
            tile_ids=[job["tile_id"] for job in self._jobs],
            output_dir=out_dir,
            compute_hillshade=self._hillshade_check.isChecked(),
            crs_epsg=self._crs_epsg,
            crs_wkt=self._crs_wkt,
            fill_dsm_voids=self._fill_voids_check.isChecked(),
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
        self._status_label.setText("Rasterising tiles…")

        merged = self._merged_check.isChecked()

        # Start worker
        self._worker = _ExportWorker(
            self._jobs,
            self._global_bbox,
            config,
            merged,
            workers=self._workers,
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
        self._fill_voids_check.setEnabled(enabled and self._dsm_radio.isChecked())
        self._hillshade_check.setEnabled(enabled)
        self._merged_check.setEnabled(enabled)
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(enabled)


# ── LAS Export ──────────────────────────────────────────────────────────

class _ExportLasWorker(QThread):
    """Background worker for exporting LAS tiles."""

    progress = Signal(int, int)          # done, total
    tile_done = Signal(str, str)         # tile_id, written path
    finished_all = Signal(int)           # number of exported tiles
    error = Signal(str)

    def __init__(
        self,
        jobs: list,
        output_dir: str,
        global_bbox: tuple,
        overlap_m: float,
        tile_size_m: float = 200.0,
        classes: Optional[Set[int]] = None,
        clip_overlap: bool = True,
        workers: int = 4,
        parent=None,
    ):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._output_dir = output_dir
        self._global_bbox = tuple(global_bbox)
        self._overlap_m = float(overlap_m)
        self._tile_size_m = float(tile_size_m)
        self._classes = set(classes) if classes is not None else None
        self._clip_overlap = bool(clip_overlap)
        self._workers = max(1, int(workers))

    def run(self) -> None:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from ..export_manager import export_las_tile

        total = len(self._jobs)
        if total == 0:
            self.finished_all.emit(0)
            return

        done = 0
        try:
            with ProcessPoolExecutor(max_workers=min(self._workers, total)) as pool:
                futures = {
                    pool.submit(
                        export_las_tile,
                        job,
                        self._output_dir,
                        self._global_bbox,
                        self._overlap_m,
                        self._tile_size_m,
                        classes=self._classes,
                        clip_overlap=self._clip_overlap,
                    ): job
                    for job in self._jobs
                }
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        path = fut.result()
                        if path:
                            self.tile_done.emit(job["tile_id"], path)
                    except Exception as exc:
                        logger.exception("LAS export failed for %s", job["tile_id"])
                        self.error.emit(f"{job['tile_id']}: {exc}")
                    done += 1
                    self.progress.emit(done, total)
            self.finished_all.emit(done)
        except Exception as exc:
            logger.exception("LAS export failed")
            self.error.emit(str(exc))


class ExportLasDialog(QDialog):
    """
    Dialog for exporting LAS tiles with user-selected classes, clean naming,
    and optional overlap trimming.
    """

    def __init__(
        self,
        jobs: List[dict],
        global_bbox: Tuple[float, float, float, float],
        default_output_dir: str,
        overlap_m: float = 0.0,
        tile_size_m: float = 200.0,
        workers: int = 4,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = list(jobs)
        self._global_bbox = global_bbox
        self._default_output_dir = default_output_dir
        self._overlap_m = float(overlap_m)
        self._tile_size_m = float(tile_size_m)
        self._workers = max(1, int(workers))
        self._worker: Optional[_ExportLasWorker] = None
        self.exported_count: int = 0
        self.written_files: List[str] = []

        self.setWindowTitle("Export LAS Tiles")
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(f"Export <b>{len(self._jobs)}</b> selected tile(s) to LAS format.")
        layout.addWidget(info)

        # ── Classes to export ─────────────────────────────────────────
        class_group = QGroupBox("Points to Export")
        class_layout = QVBoxLayout(class_group)

        self._all_points_radio = QRadioButton("All points (entire point cloud → {tile_id}.las)")
        self._ground_only_radio = QRadioButton("Ground points only (Class 2 → {tile_id}_ground.las)")
        self._all_points_radio.setChecked(True)
        class_layout.addWidget(self._all_points_radio)
        class_layout.addWidget(self._ground_only_radio)
        layout.addWidget(class_group)

        # ── Overlap margins ───────────────────────────────────────────
        ov_group = QGroupBox("Tile Boundaries")
        ov_layout = QVBoxLayout(ov_group)

        self._clip_overlap_check = QCheckBox("Cut tile overlap margins (non-overlapping core boundaries)")
        self._clip_overlap_check.setChecked(True)
        self._clip_overlap_check.setToolTip(
            "When checked, any overlapping buffer margins between tiles are clipped so adjacent "
            "exported tiles fit together seamlessly without duplicate points."
        )
        ov_layout.addWidget(self._clip_overlap_check)
        layout.addWidget(ov_group)

        # ── Output directory ──────────────────────────────────────────
        out_group = QGroupBox("Output")
        out_form = QFormLayout(out_group)

        dir_row = QHBoxLayout()
        self._out_dir_label = QLabel(self._default_output_dir)
        self._out_dir_label.setWordWrap(True)
        dir_row.addWidget(self._out_dir_label, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(browse_btn)
        out_form.addRow("Directory:", dir_row)

        self._workers_spin = QDoubleSpinBox()
        self._workers_spin.setRange(1, 32)
        self._workers_spin.setDecimals(0)
        self._workers_spin.setValue(self._workers)
        out_form.addRow("Workers:", self._workers_spin)

        layout.addWidget(out_group)

        # ── Progress bar ────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # ── Buttons ──────────────────────────────────────────────────
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._button_box.button(QDialogButtonBox.Ok).setText("Export")
        self._button_box.accepted.connect(self._on_export)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output Directory", self._out_dir_label.text())
        if d:
            self._out_dir_label.setText(d)

    def _on_export(self) -> None:
        out_dir = self._out_dir_label.text()
        if not out_dir:
            QMessageBox.warning(self, "Missing Directory", "Please select an output directory.")
            return

        classes = {2} if self._ground_only_radio.isChecked() else None
        clip_overlap = self._clip_overlap_check.isChecked()
        workers = int(self._workers_spin.value())

        self._all_points_radio.setEnabled(False)
        self._ground_only_radio.setEnabled(False)
        self._clip_overlap_check.setEnabled(False)
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, len(self._jobs))
        self._progress_bar.setValue(0)
        self._status_label.setText("Exporting LAS tiles…")

        self._worker = _ExportLasWorker(
            self._jobs,
            out_dir,
            self._global_bbox,
            self._overlap_m,
            tile_size_m=self._tile_size_m,
            classes=classes,
            clip_overlap=clip_overlap,
            workers=workers,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.tile_done.connect(self._on_tile_done)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self._progress_bar.setValue(done)
        self._status_label.setText(f"Exported {done}/{total} tiles…")

    def _on_tile_done(self, tile_id: str, path: str) -> None:
        self.written_files.append(path)

    def _on_finished(self, n: int) -> None:
        self.exported_count = n
        self._progress_bar.setValue(len(self._jobs))
        self._status_label.setText(f"Done — {n} tile(s) exported.")
        QMessageBox.information(
            self,
            "LAS Export Complete",
            f"Successfully exported {n} tile(s) to:\n{self._out_dir_label.text()}",
        )
        self.accept()

    def _on_error(self, msg: str) -> None:
        self._status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "LAS Export Error", msg)
        self._all_points_radio.setEnabled(True)
        self._ground_only_radio.setEnabled(True)
        self._clip_overlap_check.setEnabled(True)
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        self._progress_bar.setVisible(False)
