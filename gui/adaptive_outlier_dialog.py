"""
LiDAR Workbench — Automatic Adaptive Outlier Classification Dialog.

Phase 3 PySide6 UI for the "Evaluate First, Then Adaptively Classify Outliers"
pipeline (`adaptive_outlier.py`):
  - Displays `CloudEvaluationReport` metrics on open (<100 ms).
  - Provides Preset selector (Auto, Drone, Helicopter, Fixed-Wing, Custom) and
    full user parameter overrides with explanatory tooltips.
  - Runs classification in a background QThread (with cancellation) or
    distributed across company nodes, applies ASPRS Class 7 & 18 in-place,
    and records a reversible batch command in SQLite `edit_history`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..adaptive_outlier import (
    CloudEvaluationReport,
    OutlierParams,
    apply_outlier_classification_with_undo,
    classify_outliers_adaptive,
    evaluate_point_cloud,
)
from ..database import Database
from ..tile_manager import TileManager
from .classification_dialog import ClusterExecutionWidget

logger = logging.getLogger("lidar_workbench.gui.adaptive_outlier_dialog")


class _AdaptiveOutlierWorker(QThread):
    """Background worker running `classify_outliers_adaptive` across selected tiles."""

    progress = Signal(str, float)
    tile_done = Signal(str, int, int)  # tile_id, n_class_7, n_class_18
    finished_all = Signal(int, int, int)  # n_tiles, total_c7, total_c18
    error = Signal(str)

    def __init__(
        self,
        tile_manager: TileManager,
        db: Database,
        tile_ids: List[str],
        params: OutlierParams,
        parent=None,
    ):
        super().__init__(parent)
        self._tm = tile_manager
        self._db = db
        self._tile_ids = list(tile_ids)
        self._params = params
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        import laspy

        n_total = len(self._tile_ids)
        total_c7 = 0
        total_c18 = 0
        processed = 0

        try:
            for idx, tid in enumerate(self._tile_ids):
                if self._cancelled:
                    break
                base_pct = (idx / max(1, n_total)) * 100.0
                self.progress.emit(f"Evaluating & classifying {tid} ({idx + 1}/{n_total})…", base_pct)

                data = self._tm.load_tile_points_full(tid)
                if data is None or len(data.get("x", [])) == 0:
                    continue

                old_cls = np.asarray(data["classification"], dtype=np.uint8).copy()
                rep = evaluate_point_cloud(data)
                res = classify_outliers_adaptive(
                    data,
                    params=self._params,
                    report=rep,
                    cancel_check=lambda: self._cancelled,
                )
                if self._cancelled:
                    break

                # Apply Class 7 & Class 18 in-place on disk while preserving all extra bytes
                tile_info = self._db.get_tile(tid)
                if tile_info and self._tm._pm.tiles_dir:
                    las_path = self._tm._pm.tiles_dir / tile_info["filename"]
                    if las_path.is_file():
                        las = laspy.read(str(las_path))
                        las.classification = res.updated_classification
                        las.write(str(las_path))

                # Update in-memory cache if loaded
                if hasattr(self._tm, "_point_cache") and tid in self._tm._point_cache:
                    self._tm._point_cache[tid]["classification"] = res.updated_classification

                # Record single reversible batch command to SQLite undo/redo history
                apply_outlier_classification_with_undo(self._db, tid, old_cls, res)

                c7 = res.stats.get("n_class_7", 0)
                c18 = res.stats.get("n_class_18", 0)
                total_c7 += c7
                total_c18 += c18
                processed += 1
                self.tile_done.emit(tid, c7, c18)
                self.progress.emit(
                    f"{tid}: Class 7={c7:,}, Class 18={c18:,} ({processed}/{n_total})",
                    ((idx + 1) / max(1, n_total)) * 100.0,
                )

            self.finished_all.emit(processed, total_c7, total_c18)
        except Exception as exc:
            logger.exception("Adaptive outlier worker error")
            self.error.emit(str(exc))


class AdaptiveOutlierDialog(QDialog):
    """Dialog for automatic evaluation and adaptive outlier classification."""

    classification_applied = Signal()

    def __init__(
        self,
        tile_manager: TileManager,
        db: Database,
        tile_ids: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self._tm = tile_manager
        self._db = db
        self._tile_ids = list(tile_ids)
        self._worker: Optional[_AdaptiveOutlierWorker] = None
        self._cluster_monitor = None
        self._updating_ui = False

        self.setWindowTitle(f"Auto Adaptive Outlier Filter ({len(self._tile_ids)} tile(s))")
        self.setMinimumSize(640, 720)

        self._report = self._evaluate_first_tile()
        self._build_ui()
        self._apply_preset("auto")

    def _evaluate_first_tile(self) -> CloudEvaluationReport:
        if self._tile_ids:
            data = self._tm.load_tile_points_full(self._tile_ids[0])
            if data is not None and len(data.get("x", [])) > 0:
                return evaluate_point_cloud(data)
        return CloudEvaluationReport()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ── 1. Cloud Evaluation Report Summary ───────────────────────
        rep_box = QGroupBox("Phase 1 — Cloud Evaluation Report (Sample Tile)")
        rep_lay = QFormLayout(rep_box)
        r = self._report
        eb_flags = []
        if r.has_reflectance:
            eb_flags.append("Reflectance")
        if r.has_amplitude:
            eb_flags.append("Amplitude")
        if r.has_deviation:
            eb_flags.append("Deviation")
        eb_str = ", ".join(eb_flags) if eb_flags else "None (Standard LAS intensity)"

        self._lbl_platform = QLabel(
            f"<b>{r.estimated_platform.upper()}</b>  "
            f"(Density: {r.median_density_pts_m2:.1f} pts/m², NPS: {r.nps_m:.3f} m, "
            f"P90/P10: {r.density_heterogeneity:.2f}×)"
        )
        rep_lay.addRow("Inferred Platform:", self._lbl_platform)
        rep_lay.addRow(
            "Geometry & Strips:",
            QLabel(
                f"{r.point_count:,} pts | {r.flightline_count} flightline(s) | "
                f"Scan spread: {r.scan_angle_spread_deg:.1f}° | Relief P50/P90: {r.relief_p50_m:.2f}/{r.relief_p90_m:.2f} m"
            ),
        )
        rep_lay.addRow(
            "Radiometry & Bathy:",
            QLabel(
                f"RIEGL Extra Bytes: {eb_str} | "
                f"Bathymetry: {'Detected' if r.has_bathymetry else 'No'} | "
                f"Eval: {r.eval_time_ms:.1f} ms"
            ),
        )
        root.addWidget(rep_box)

        # ── 2. Preset Selector ───────────────────────────────────────
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("<b>Platform Preset:</b>"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Auto (From Evaluation Report)", "auto")
        self._preset_combo.addItem("Drone / UAV (~300 pts/m²)", "drone")
        self._preset_combo.addItem("Helicopter (~100 pts/m², Multi-Strip)", "helicopter")
        self._preset_combo.addItem("Fixed-Wing (~20 pts/m², Wide Scan)", "fixed_wing")
        self._preset_combo.addItem("Custom (Manual Overrides)", "custom")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, stretch=1)
        root.addLayout(preset_row)

        # ── 3. Scrollable Parameter Overrides ────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        param_host = QWidget()
        p_lay = QVBoxLayout(param_host)

        # Group 1: Radiometric & Pulse Gate
        g1 = QGroupBox("1. Return-Aware RIEGL Radiometric & Pulse-Shape Gate")
        f1 = QFormLayout(g1)
        self._chk_refl_gate = QCheckBox("Enable Low-Reflectance Gate")
        self._chk_refl_gate.setToolTip("Apply reflectance threshold to single/first returns when RIEGL reflectance is available.")
        f1.addRow(self._chk_refl_gate)
        self._spn_min_refl = self._dspin(-60.0, 20.0, 1.0, " dB", "Minimum RIEGL reflectance (dB) for single/first returns.")
        f1.addRow("Min Reflectance (dB):", self._spn_min_refl)
        self._spn_max_dev = self._dspin(0.0, 200.0, 1.0, "", "Maximum RIEGL pulse shape deviation before flagging isolated returns.")
        f1.addRow("Max Pulse Deviation:", self._spn_max_dev)
        self._chk_protect_canopy = QCheckBox("Protect Multi-Return Forest Canopy (return_number > 1)")
        self._chk_protect_canopy.setToolTip("Never apply strict low-reflectance or PCA planar rejection to intermediate/last vegetation returns.")
        f1.addRow(self._chk_protect_canopy)
        p_lay.addWidget(g1)

        # Group 2: 3D Voxel Isolation
        g2 = QGroupBox("2. 3D Voxel Isolation Pre-Filter (O(N))")
        f2 = QFormLayout(g2)
        self._spn_voxel = self._dspin(0.05, 10.0, 0.05, " m", "3D voxel edge length (auto: 3.0 × NPS).")
        f2.addRow("Voxel Size:", self._spn_voxel)
        self._spn_voxel_nbr = self._ispin(1, 20, " pts", "Minimum points in 3×3×3 voxel neighborhood to survive.")
        f2.addRow("Min Voxel Neighbors:", self._spn_voxel_nbr)
        p_lay.addWidget(g2)

        # Group 3: Asymmetric IRLS & Strip Envelope
        g3 = QGroupBox("3. Flightline-Aware Asymmetric IRLS Lower & Upper Envelope")
        f3 = QFormLayout(g3)
        self._spn_cell = self._dspin(0.25, 25.0, 0.25, " m", "IRLS 2D terrain envelope cell size (auto: 8.0 × NPS).")
        f3.addRow("Grid Cell Size:", self._spn_cell)
        self._spn_irls_iter = self._ispin(1, 10, "", "Number of asymmetric Huber/Tukey IRLS refinement iterations.")
        f3.addRow("IRLS Iterations:", self._spn_irls_iter)
        self._spn_lower_mad = self._dspin(1.0, 20.0, 0.5, " × MAD", "Class 7 threshold below IRLS lower terrain envelope.")
        f3.addRow("Lower MAD Multiplier:", self._spn_lower_mad)
        self._spn_upper_mad = self._dspin(1.0, 25.0, 0.5, " × MAD", "Class 18 high-flyer threshold above upper envelope.")
        f3.addRow("Upper MAD Multiplier:", self._spn_upper_mad)
        self._chk_per_strip = QCheckBox("Evaluate Per-Strip First (preserves helicopter ripples)")
        self._chk_per_strip.setToolTip("Evaluate each point_source_id strip independently before flagging high flyers.")
        f3.addRow(self._chk_per_strip)
        self._spn_strip_disc = self._dspin(0.1, 20.0, 0.1, " m", "Maximum allowed cross-strip elevation discrepancy.")
        f3.addRow("Max Strip Discrepancy:", self._spn_strip_disc)
        p_lay.addWidget(g3)

        # Group 4: Multi-Scale PCA + Radiometric SOR
        g4 = QGroupBox("4. Multi-Scale PCA Consistency & Radiometric SOR")
        f4 = QFormLayout(g4)
        self._spn_k_small = self._ispin(4, 32, "", "Small neighborhood k for local plane/linearity PCA.")
        f4.addRow("SOR k (Small):", self._spn_k_small)
        self._spn_k_large = self._ispin(8, 64, "", "Large neighborhood k for single cKDTree query & multi-scale check.")
        f4.addRow("SOR k (Large):", self._spn_k_large)
        self._spn_sor_std = self._dspin(0.5, 10.0, 0.1, " σ", "Radiometric-weighted k-NN distance MAD/std multiplier.")
        f4.addRow("SOR Std Multiplier:", self._spn_sor_std)
        self._spn_plane_res = self._dspin(0.02, 5.0, 0.02, " m", "Max PCA smallest-eigenvalue plane residual (auto: 1.5 × NPS).")
        f4.addRow("Max Plane Residual:", self._spn_plane_res)
        self._spn_norm_ang = self._dspin(5.0, 85.0, 1.0, "°", "Max surface normal angular jump between k_small and k_large.")
        f4.addRow("Max Normal Angle:", self._spn_norm_ang)
        self._spn_rad_wt = self._dspin(0.0, 3.0, 0.1, "", "Weight of reflectance/intensity penalty in k-NN distance SOR.")
        f4.addRow("Radiometric Weight:", self._spn_rad_wt)
        p_lay.addWidget(g4)

        # Group 5: Bathymetry
        g5 = QGroupBox("5. Conditional Bathymetry Sub-Pass")
        f5 = QFormLayout(g5)
        self._cmb_bathy = QComboBox()
        self._cmb_bathy.addItem("Auto (Run if bathymetry detected)", "auto")
        self._cmb_bathy.addItem("On (Force water-column & spray filter)", "on")
        self._cmb_bathy.addItem("Off", "off")
        f5.addRow("Bathymetry Mode:", self._cmb_bathy)
        p_lay.addWidget(g5)

        scroll.setWidget(param_host)
        root.addWidget(scroll, stretch=1)

        # ── 4. Distributed Cluster Execution ─────────────────────────
        self._cluster_exec = ClusterExecutionWidget(self._tm, parent=self)
        root.addWidget(self._cluster_exec)

        # ── 5. Progress & Buttons ────────────────────────────────────
        self._status = QLabel("Ready — non-destructive ASPRS Class 7 & Class 18 classification.")
        root.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        root.addWidget(self._progress)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("✨ Run Adaptive Outlier Classification")
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(self._on_cancel)
        btn_row.addWidget(box)
        root.addLayout(btn_row)

        # Wire manual overrides to switch preset combo to "custom"
        for w in (
            self._spn_min_refl, self._spn_max_dev, self._spn_voxel, self._spn_voxel_nbr,
            self._spn_cell, self._spn_irls_iter, self._spn_lower_mad, self._spn_upper_mad,
            self._spn_strip_disc, self._spn_k_small, self._spn_k_large, self._spn_sor_std,
            self._spn_plane_res, self._spn_norm_ang, self._spn_rad_wt,
        ):
            w.valueChanged.connect(self._mark_custom)
        for c in (self._chk_refl_gate, self._chk_protect_canopy, self._chk_per_strip):
            c.toggled.connect(self._mark_custom)
        self._cmb_bathy.currentIndexChanged.connect(self._mark_custom)

    def _dspin(self, lo: float, hi: float, step: float, suffix: str, tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(3 if step < 0.05 else 2)
        if suffix:
            s.setSuffix(suffix)
        s.setToolTip(tip)
        return s

    def _ispin(self, lo: int, hi: int, suffix: str, tip: str) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        if suffix:
            s.setSuffix(suffix)
        s.setToolTip(tip)
        return s

    def _mark_custom(self) -> None:
        if self._updating_ui:
            return
        idx = self._preset_combo.findData("custom")
        if idx >= 0 and self._preset_combo.currentIndex() != idx:
            self._updating_ui = True
            self._preset_combo.setCurrentIndex(idx)
            self._updating_ui = False

    def _on_preset_changed(self) -> None:
        if self._updating_ui:
            return
        preset = self._preset_combo.currentData()
        if preset and preset != "custom":
            self._apply_preset(preset)

    def _apply_preset(self, preset: str) -> None:
        builders = {
            "auto": OutlierParams.auto,
            "drone": OutlierParams.drone,
            "helicopter": OutlierParams.helicopter,
            "fixed_wing": OutlierParams.fixed_wing,
        }
        p = builders.get(preset, OutlierParams.auto)(self._report)
        self._updating_ui = True
        try:
            self._chk_refl_gate.setChecked(p.use_reflectance_gate)
            self._spn_min_refl.setValue(p.min_reflectance_db)
            self._spn_max_dev.setValue(p.max_pulse_deviation)
            self._chk_protect_canopy.setChecked(p.protect_multi_return_canopy)
            self._spn_voxel.setValue(p.voxel_size_m)
            self._spn_voxel_nbr.setValue(p.min_voxel_neighbors)
            self._spn_cell.setValue(p.grid_cell_size_m)
            self._spn_irls_iter.setValue(p.irls_iterations)
            self._spn_lower_mad.setValue(p.z_lower_mad_multiplier)
            self._spn_upper_mad.setValue(p.z_upper_mad_multiplier)
            self._chk_per_strip.setChecked(p.per_strip_first)
            self._spn_strip_disc.setValue(p.max_strip_discrepancy_m)
            self._spn_k_small.setValue(p.sor_k_small)
            self._spn_k_large.setValue(p.sor_k_large)
            self._spn_sor_std.setValue(p.sor_std_multiplier)
            self._spn_plane_res.setValue(p.max_plane_residual_m)
            self._spn_norm_ang.setValue(p.max_normal_angle_deg)
            self._spn_rad_wt.setValue(p.radiometric_weight)
            b_idx = self._cmb_bathy.findData(p.bathy_mode)
            if b_idx >= 0:
                self._cmb_bathy.setCurrentIndex(b_idx)
        finally:
            self._updating_ui = False

    def collect_params(self) -> OutlierParams:
        return OutlierParams(
            preset_name=str(self._preset_combo.currentData() or "custom"),
            use_reflectance_gate=self._chk_refl_gate.isChecked(),
            min_reflectance_db=self._spn_min_refl.value(),
            max_pulse_deviation=self._spn_max_dev.value(),
            protect_multi_return_canopy=self._chk_protect_canopy.isChecked(),
            voxel_size_m=self._spn_voxel.value(),
            min_voxel_neighbors=self._spn_voxel_nbr.value(),
            grid_cell_size_m=self._spn_cell.value(),
            irls_iterations=self._spn_irls_iter.value(),
            z_lower_mad_multiplier=self._spn_lower_mad.value(),
            z_upper_mad_multiplier=self._spn_upper_mad.value(),
            per_strip_first=self._chk_per_strip.isChecked(),
            max_strip_discrepancy_m=self._spn_strip_disc.value(),
            sor_k_small=self._spn_k_small.value(),
            sor_k_large=self._spn_k_large.value(),
            sor_std_multiplier=self._spn_sor_std.value(),
            max_plane_residual_m=self._spn_plane_res.value(),
            max_normal_angle_deg=self._spn_norm_ang.value(),
            radiometric_weight=self._spn_rad_wt.value(),
            bathy_mode=str(self._cmb_bathy.currentData() or "auto"),
        )

    def _on_run(self) -> None:
        if not self._tile_ids:
            QMessageBox.warning(self, "No Tiles", "No tiles selected.")
            return
        params = self.collect_params()

        if self._cluster_exec.is_distributed():
            from dataclasses import asdict
            self._run_btn.setEnabled(False)
            self._progress.setValue(0)
            self._status.setText(f"Dispatching adaptive outlier classification ({len(self._tile_ids)} tiles)…")
            mon, err = self._cluster_exec.start_distributed_job(
                self._tile_ids,
                job_type="adaptive_outlier",
                job_params={"outlier_params": asdict(params)},
                on_progress=self._on_progress,
                on_completed=lambda _r: self._on_finished(len(self._tile_ids), 0, 0),
                on_error=self._on_error,
                parent=self,
            )
            if err:
                self._run_btn.setEnabled(True)
                QMessageBox.warning(self, "Distributed Outlier Filter", err)
                return
            self._cluster_monitor = mon
            return

        self._run_btn.setEnabled(False)
        self._progress.setValue(0)
        self._worker = _AdaptiveOutlierWorker(self._tm, self._db, self._tile_ids, params, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str, pct: float) -> None:
        self._status.setText(msg)
        self._progress.setValue(int(pct))

    def _on_finished(self, n_tiles: int, total_c7: int, total_c18: int) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setValue(100)
        self._status.setText(
            f"Done: {n_tiles} tile(s) — Class 7: {total_c7:,}, Class 18: {total_c18:,} (undoable in SQLite)"
        )
        self.classification_applied.emit()
        QMessageBox.information(
            self,
            "Adaptive Outlier Classification Complete",
            f"Processed {n_tiles} tile(s).\nClass 7 (Low Noise): {total_c7:,}\nClass 18 (High Noise): {total_c18:,}",
        )

    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Adaptive Outlier Error", msg)

    def _on_cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        if self._cluster_monitor is not None and self._cluster_monitor.isRunning():
            self._cluster_monitor.cancel()
            self._cluster_monitor.wait(2000)
        self.reject()
