"""
LiDAR Workbench — Noise Filter Dialog.

Interactive dialog for configuring and previewing noise filters
(SOR, ROR, DBSCAN) with a live 3D point cloud preview.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer
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
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from ..config import (
    DEFAULT_ROR_MIN_POINTS,
    DEFAULT_ROR_RADIUS,
    DEFAULT_SOR_NB_NEIGHBORS,
    DEFAULT_SOR_STD_RATIO,
    get_class_color,
)
from ..noise_filter import (
    dbscan_outlier_removal,
    radius_outlier_removal,
    statistical_outlier_removal,
)
from ..tile_manager import TileManager
from .view_3d import View3D

logger = logging.getLogger("lidar_workbench.gui.filter_dialog")


class FilterDialog(QDialog):
    """
    Dialog for configuring and previewing noise filters.

    The preview section contains an interactive 3D view where:
        - **Outlier points** are shown in **red**
        - **Kept points** are shown in their class colour

    Mouse controls (native Open3D):
        - Left-drag = orbit    |  Shift+left-drag / right-drag = pan
        - Scroll = zoom

    Signals:
        filter_applied(tile_ids, filter_params):
            Emitted after the user clicks "Apply to All Tiles".
    """

    filter_applied = Signal(list, dict)

    def __init__(
        self,
        tile_manager: TileManager,
        tile_ids: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tm = tile_manager
        self._tile_ids = tile_ids
        self._preview_points: Optional[dict] = None
        self._current_keep_mask: Optional[np.ndarray] = None

        self.setWindowTitle("Noise Filter")
        self.setMinimumSize(700, 550)
        self._setup_ui()
        self._load_preview_sample()

    # ── UI ─────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)

        # ── Left: controls ─────────────────────────────────────────
        left = QVBoxLayout()
        left.setContentsMargins(4, 4, 4, 4)

        # --- Filter type ---
        type_group = QGroupBox("Filter Type")
        type_layout = QVBoxLayout(type_group)
        self._filter_type_combo = QComboBox()
        self._filter_type_combo.addItem("Statistical Outlier Removal (SOR)", "sor")
        self._filter_type_combo.addItem("Radius Outlier Removal (ROR)", "ror")
        self._filter_type_combo.addItem("DBSCAN — Above (aerial noise)", "dbscan_above")
        self._filter_type_combo.addItem("DBSCAN — Below (sub-surface noise)", "dbscan_below")
        self._filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
        type_layout.addWidget(self._filter_type_combo)
        left.addWidget(type_group)

        # --- SOR params ---
        self._sor_group = QGroupBox("SOR Parameters")
        sf = QFormLayout(self._sor_group)
        self._sor_nb_spin = QSpinBox()
        self._sor_nb_spin.setRange(1, 200)
        self._sor_nb_spin.setValue(DEFAULT_SOR_NB_NEIGHBORS)
        self._sor_nb_spin.valueChanged.connect(self._schedule_preview_update)
        sf.addRow("Neighbors:", self._sor_nb_spin)

        self._sor_std_spin = QDoubleSpinBox()
        self._sor_std_spin.setRange(0.1, 10.0)
        self._sor_std_spin.setSingleStep(0.1)
        self._sor_std_spin.setValue(DEFAULT_SOR_STD_RATIO)
        self._sor_std_spin.valueChanged.connect(self._schedule_preview_update)
        sf.addRow("Std Ratio:", self._sor_std_spin)

        self._sor_slider = QSlider(Qt.Horizontal)
        self._sor_slider.setRange(10, 100)
        self._sor_slider.setValue(int(DEFAULT_SOR_STD_RATIO * 10))
        self._sor_slider.valueChanged.connect(self._on_sor_slider_changed)
        sf.addRow("Quick Adjust:", self._sor_slider)
        left.addWidget(self._sor_group)

        # --- ROR params ---
        self._ror_group = QGroupBox("ROR Parameters")
        rf = QFormLayout(self._ror_group)
        self._ror_radius_spin = QDoubleSpinBox()
        self._ror_radius_spin.setRange(0.01, 100.0)
        self._ror_radius_spin.setDecimals(2)
        self._ror_radius_spin.setValue(DEFAULT_ROR_RADIUS)
        self._ror_radius_spin.valueChanged.connect(self._schedule_preview_update)
        rf.addRow("Radius:", self._ror_radius_spin)

        self._ror_min_spin = QSpinBox()
        self._ror_min_spin.setRange(1, 1000)
        self._ror_min_spin.setValue(DEFAULT_ROR_MIN_POINTS)
        self._ror_min_spin.valueChanged.connect(self._schedule_preview_update)
        rf.addRow("Min Points:", self._ror_min_spin)
        self._ror_group.setVisible(False)
        left.addWidget(self._ror_group)

        # --- DBSCAN params ---
        self._dbscan_group = QGroupBox("DBSCAN Parameters")
        df = QFormLayout(self._dbscan_group)
        self._dbscan_eps_spin = QDoubleSpinBox()
        self._dbscan_eps_spin.setRange(0.1, 100.0)
        self._dbscan_eps_spin.setDecimals(2)
        self._dbscan_eps_spin.setValue(2.0)
        self._dbscan_eps_spin.setSuffix(" m")
        self._dbscan_eps_spin.valueChanged.connect(self._schedule_preview_update)
        df.addRow("Epsilon (eps):", self._dbscan_eps_spin)

        self._dbscan_min_samples_spin = QSpinBox()
        self._dbscan_min_samples_spin.setRange(2, 500)
        self._dbscan_min_samples_spin.setValue(10)
        self._dbscan_min_samples_spin.valueChanged.connect(self._schedule_preview_update)
        df.addRow("Min Samples:", self._dbscan_min_samples_spin)

        self._dbscan_min_cluster_spin = QSpinBox()
        self._dbscan_min_cluster_spin.setRange(1, 10000)
        self._dbscan_min_cluster_spin.setValue(50)
        self._dbscan_min_cluster_spin.valueChanged.connect(self._schedule_preview_update)
        df.addRow("Min Cluster Size:", self._dbscan_min_cluster_spin)
        self._dbscan_group.setVisible(False)
        left.addWidget(self._dbscan_group)

        # --- Status ---
        self._preview_status = QLabel("Loading preview…")
        self._preview_status.setWordWrap(True)
        self._preview_status.setStyleSheet("font-weight: bold; color: #333;")
        left.addWidget(self._preview_status)

        # --- Batch apply ---
        self._batch_check = QCheckBox(
            f"Apply to all {len(self._tile_ids)} selected tile(s)"
        )
        self._batch_check.setChecked(True)
        left.addWidget(self._batch_check)

        left.addStretch()

        # --- Buttons ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        btn_box.button(QDialogButtonBox.Apply).setText("Apply Filter")
        btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        btn_box.rejected.connect(self.reject)
        left.addWidget(btn_box)

        layout.addLayout(left, stretch=1)

        # ── Right: interactive 3D preview ──────────────────────────
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)

        self._preview_view = View3D()
        self._preview_view.setMinimumSize(400, 350)
        right.addWidget(self._preview_view, stretch=1)

        layout.addLayout(right, stretch=2)

        # Debounce timer
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._update_preview)

    # ── filter type switching ──────────────────────────────────────

    def _on_filter_type_changed(self, index: int) -> None:
        ft = self._filter_type_combo.currentData()
        self._sor_group.setVisible(ft == "sor")
        self._ror_group.setVisible(ft == "ror")
        self._dbscan_group.setVisible(ft in ("dbscan_above", "dbscan_below"))
        self._schedule_preview_update()

    def _on_sor_slider_changed(self, value: int) -> None:
        self._sor_std_spin.blockSignals(True)
        self._sor_std_spin.setValue(value / 10.0)
        self._sor_std_spin.blockSignals(False)
        self._schedule_preview_update()

    # ── preview ────────────────────────────────────────────────────

    def _schedule_preview_update(self) -> None:
        self._preview_timer.start()

    def _load_preview_sample(self) -> None:
        """Load up to 50k points from the first tile for interactive preview."""
        if not self._tile_ids:
            self._preview_status.setText("No tiles selected.")
            return

        data = self._tm.load_tile_points_full(self._tile_ids[0])
        if data is None:
            self._preview_status.setText("Failed to load preview data.")
            return

        n = len(data["x"])
        if n > 50_000:
            indices = np.random.choice(n, 50_000, replace=False)
            data = {k: v[indices] for k, v in data.items()}

        self._preview_points = data
        self._update_preview()

    def _update_preview(self) -> None:
        """Re-run the filter and update the 3D preview."""
        if self._preview_points is None:
            return

        pts = self._preview_points
        ft = self._filter_type_combo.currentData()

        try:
            if ft == "sor":
                keep, outlier = statistical_outlier_removal(
                    pts["x"], pts["y"], pts["z"],
                    nb_neighbors=self._sor_nb_spin.value(),
                    std_ratio=self._sor_std_spin.value(),
                )
            elif ft == "ror":
                keep, outlier = radius_outlier_removal(
                    pts["x"], pts["y"], pts["z"],
                    radius=self._ror_radius_spin.value(),
                    min_points=self._ror_min_spin.value(),
                )
            else:  # dbscan_above / dbscan_below
                mode = "above" if ft == "dbscan_above" else "below"
                keep, outlier = dbscan_outlier_removal(
                    pts["x"], pts["y"], pts["z"],
                    eps=self._dbscan_eps_spin.value(),
                    min_samples=self._dbscan_min_samples_spin.value(),
                    min_cluster_size=self._dbscan_min_cluster_spin.value(),
                    mode=mode,
                )
        except Exception as exc:
            self._preview_status.setText(f"Preview error: {exc}")
            return

        self._current_keep_mask = keep

        n_out = int(outlier.sum())
        n_tot = len(keep)
        pct = n_out / n_tot * 100 if n_tot > 0 else 0

        self._preview_status.setText(
            f"Outliers: {n_out:,} / {n_tot:,} ({pct:.1f}%)\n"
            f"Kept:     {n_tot - n_out:,} points"
        )

        # Build coloured preview: outliers = red, kept = class colour
        cls_arr = pts.get("classification")
        n = len(pts["x"])
        colors = np.zeros((n, 3), dtype=np.float64)

        if cls_arr is not None:
            for code in np.unique(cls_arr):
                mask = (cls_arr == code) & keep
                colors[mask] = get_class_color(int(code))
        else:
            colors[keep] = (0.6, 0.6, 0.6)

        colors[outlier] = (1.0, 0.15, 0.15)  # red outliers

        self._preview_view.load_point_cloud_colored(
            pts["x"], pts["y"], pts["z"], colors,
        )

    # ── apply ──────────────────────────────────────────────────────

    def _on_apply(self) -> None:
        if self._current_keep_mask is None:
            self._preview_status.setText("No preview computed — adjust parameters first.")
            return

        params = self._get_filter_params()
        if self._batch_check.isChecked():
            self.filter_applied.emit(self._tile_ids, params)
        self.accept()

    def _get_filter_params(self) -> dict:
        ft = self._filter_type_combo.currentData()
        if ft == "sor":
            return {
                "type": "sor",
                "nb_neighbors": self._sor_nb_spin.value(),
                "std_ratio": self._sor_std_spin.value(),
            }
        elif ft == "ror":
            return {
                "type": "ror",
                "radius": self._ror_radius_spin.value(),
                "min_points": self._ror_min_spin.value(),
            }
        else:
            return {
                "type": ft,
                "eps": self._dbscan_eps_spin.value(),
                "min_samples": self._dbscan_min_samples_spin.value(),
                "min_cluster_size": self._dbscan_min_cluster_spin.value(),
            }
