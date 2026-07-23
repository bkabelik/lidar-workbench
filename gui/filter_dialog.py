"""
LiDAR Workbench — Noise Filter Dialog.

Interactive dialog for configuring and previewing noise filters
(SOR, ROR, DBSCAN) with a live 3D point cloud preview.

Supports **multi-step filter pipelines** — add multiple filters that
run in sequence.
"""

from __future__ import annotations

import json
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
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
    DEFAULT_ISOLATED_SEARCH_RADIUS,
    DEFAULT_ISOLATED_MIN_NEIGHBORS,
    DEFAULT_LOW_POINTS_SEARCH_RADIUS,
    DEFAULT_LOW_POINTS_BELOW,
    DEFAULT_LOW_POINTS_ABOVE,
    DEFAULT_SURFACE_NOISE_GRID,
    DEFAULT_SURFACE_NOISE_TOLERANCE,
    get_class_color,
)
from ..noise_filter import (
    dbscan_outlier_removal,
    radius_outlier_removal,
    statistical_outlier_removal,
    isolated_point_removal,
    low_point_removal,
    surface_noise_removal,
    multipath_reflection_removal,
    bilateral_filter,
    thin_points_average,
)
from ..tile_manager import TileManager
from .view_3d import View3D

logger = logging.getLogger("lidar_workbench.gui.filter_dialog")

FILTER_TYPES = [
    ("SOR (Statistical Outlier Removal)", "sor"),
    ("ROR (Radius Outlier Removal)", "ror"),
    ("DBSCAN — Above (aerial noise)", "dbscan_above"),
    ("DBSCAN — Below (sub-surface noise)", "dbscan_below"),
    ("Isolated Points (fliers)", "isolated"),
    ("Low Points — Multipath", "low_points"),
    ("Surface Proximity Noise", "surface_noise"),
    ("Multipath Reflection Removal", "multipath"),
    ("Bilateral Filter (Edge-Preserving)", "bilateral"),
    ("Thin — Grid Average", "thin_average"),
]


class FilterDialog(QDialog):
    """
    Dialog for configuring and previewing noise filter pipelines.

    Signals:
        filter_applied(tile_ids, pipeline_params):
            Emitted after the user clicks "Apply Pipeline".
            *pipeline_params* is a list of filter-step dicts.
    """

    filter_applied = Signal(list, list)  # tile_ids, pipeline

    def __init__(self, tile_manager, tile_ids, parent=None):
        super().__init__(parent)
        self._tm = tile_manager
        self._tile_ids = tile_ids
        self._preview_points = None
        self._current_keep_mask = None
        self._pipeline: list[dict] = []  # list of filter-param dicts
        self._preview_tile_idx: int = 0   # which selected tile is previewed

        self.setWindowTitle("Noise Filter — Pipeline")
        self.setMinimumSize(800, 550)
        self._setup_ui()
        self._load_preview_sample()

    # ── UI ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # ── Left: controls ─────────────────────────────────────────
        left = QVBoxLayout()
        left.setContentsMargins(4, 4, 4, 4)

        # --- Filter type ---
        type_group = QGroupBox("Add Filter Step")
        type_layout = QVBoxLayout(type_group)
        self._filter_type_combo = QComboBox()
        for label, key in FILTER_TYPES:
            self._filter_type_combo.addItem(label, key)
        self._filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
        type_layout.addWidget(self._filter_type_combo)
        left.addWidget(type_group)

        # --- SOR params ---
        self._sor_group = QGroupBox("SOR Parameters")
        sf = QFormLayout(self._sor_group)
        self._sor_nb_spin = QSpinBox()
        self._sor_nb_spin.setRange(1, 200)
        self._sor_nb_spin.setValue(DEFAULT_SOR_NB_NEIGHBORS)
        sf.addRow("Neighbors:", self._sor_nb_spin)
        self._sor_std_spin = QDoubleSpinBox()
        self._sor_std_spin.setRange(0.1, 10.0)
        self._sor_std_spin.setSingleStep(0.1)
        self._sor_std_spin.setValue(DEFAULT_SOR_STD_RATIO)
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
        rf.addRow("Radius:", self._ror_radius_spin)
        self._ror_min_spin = QSpinBox()
        self._ror_min_spin.setRange(1, 1000)
        self._ror_min_spin.setValue(DEFAULT_ROR_MIN_POINTS)
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
        df.addRow("Epsilon (eps):", self._dbscan_eps_spin)
        self._dbscan_min_samples_spin = QSpinBox()
        self._dbscan_min_samples_spin.setRange(2, 500)
        self._dbscan_min_samples_spin.setValue(10)
        df.addRow("Min Samples:", self._dbscan_min_samples_spin)
        self._dbscan_min_cluster_spin = QSpinBox()
        self._dbscan_min_cluster_spin.setRange(1, 10000)
        self._dbscan_min_cluster_spin.setValue(50)
        df.addRow("Min Cluster Size:", self._dbscan_min_cluster_spin)
        self._dbscan_group.setVisible(False)
        left.addWidget(self._dbscan_group)

        # --- Isolated Points params ---
        self._isolated_group = QGroupBox("Isolated Points Parameters")
        iso_f = QFormLayout(self._isolated_group)
        self._isolated_radius_spin = QDoubleSpinBox()
        self._isolated_radius_spin.setRange(0.5, 20.0)
        self._isolated_radius_spin.setDecimals(2)
        self._isolated_radius_spin.setValue(DEFAULT_ISOLATED_SEARCH_RADIUS)
        self._isolated_radius_spin.setSuffix(" m")
        iso_f.addRow("Search Radius:", self._isolated_radius_spin)
        self._isolated_min_nbr_spin = QSpinBox()
        self._isolated_min_nbr_spin.setRange(1, 20)
        self._isolated_min_nbr_spin.setValue(DEFAULT_ISOLATED_MIN_NEIGHBORS)
        self._isolated_min_nbr_spin.setSuffix(" pts")
        iso_f.addRow("Min Neighbours:", self._isolated_min_nbr_spin)
        self._isolated_group.setVisible(False)
        left.addWidget(self._isolated_group)

        # --- Low Points params ---
        self._lowpts_group = QGroupBox("Low Points Parameters")
        lpf = QFormLayout(self._lowpts_group)
        self._lowpts_radius_spin = QDoubleSpinBox()
        self._lowpts_radius_spin.setRange(0.5, 20.0)
        self._lowpts_radius_spin.setDecimals(2)
        self._lowpts_radius_spin.setValue(DEFAULT_LOW_POINTS_SEARCH_RADIUS)
        self._lowpts_radius_spin.setSuffix(" m")
        lpf.addRow("Search Radius:", self._lowpts_radius_spin)
        self._lowpts_below_spin = QDoubleSpinBox()
        self._lowpts_below_spin.setRange(0.1, 50.0)
        self._lowpts_below_spin.setDecimals(2)
        self._lowpts_below_spin.setValue(DEFAULT_LOW_POINTS_BELOW)
        self._lowpts_below_spin.setSuffix(" m")
        lpf.addRow("Max Below Neighbours:", self._lowpts_below_spin)
        self._lowpts_above_spin = QDoubleSpinBox()
        self._lowpts_above_spin.setRange(1.0, 200.0)
        self._lowpts_above_spin.setDecimals(1)
        self._lowpts_above_spin.setValue(DEFAULT_LOW_POINTS_ABOVE)
        self._lowpts_above_spin.setSuffix(" m")
        lpf.addRow("Max Above Neighbours:", self._lowpts_above_spin)
        self._lowpts_group.setVisible(False)
        left.addWidget(self._lowpts_group)

        # --- Surface Noise params ---
        self._surf_group = QGroupBox("Surface Noise Parameters")
        sf = QFormLayout(self._surf_group)
        self._surf_grid_spin = QDoubleSpinBox()
        self._surf_grid_spin.setRange(0.1, 10.0)
        self._surf_grid_spin.setDecimals(2)
        self._surf_grid_spin.setValue(DEFAULT_SURFACE_NOISE_GRID)
        self._surf_grid_spin.setSuffix(" m")
        sf.addRow("Grid Size:", self._surf_grid_spin)
        self._surf_tol_spin = QDoubleSpinBox()
        self._surf_tol_spin.setRange(0.01, 2.0)
        self._surf_tol_spin.setDecimals(3)
        self._surf_tol_spin.setValue(DEFAULT_SURFACE_NOISE_TOLERANCE)
        self._surf_tol_spin.setSuffix(" m")
        sf.addRow("Tolerance (±):", self._surf_tol_spin)
        self._surf_group.setVisible(False)
        left.addWidget(self._surf_group)

        # --- Multipath params ---
        self._multipath_group = QGroupBox("Multipath Parameters")
        mpf = QFormLayout(self._multipath_group)
        self._mp_depth_spin = QDoubleSpinBox()
        self._mp_depth_spin.setRange(0.1, 5.0)
        self._mp_depth_spin.setDecimals(2)
        self._mp_depth_spin.setValue(0.5)
        self._mp_depth_spin.setSuffix(" m")
        mpf.addRow("Depth Threshold:", self._mp_depth_spin)
        self._multipath_group.setVisible(False)
        left.addWidget(self._multipath_group)

        # --- Bilateral params ---
        self._bilateral_group = QGroupBox("Bilateral Filter Parameters")
        bf = QFormLayout(self._bilateral_group)
        self._bilat_spatial_spin = QDoubleSpinBox()
        self._bilat_spatial_spin.setRange(0.01, 10.0)
        self._bilat_spatial_spin.setDecimals(2)
        self._bilat_spatial_spin.setValue(0.5)
        self._bilat_spatial_spin.setSuffix(" m")
        bf.addRow("Spatial Sigma:", self._bilat_spatial_spin)
        self._bilat_range_spin = QDoubleSpinBox()
        self._bilat_range_spin.setRange(0.01, 5.0)
        self._bilat_range_spin.setDecimals(2)
        self._bilat_range_spin.setValue(0.1)
        self._bilat_range_spin.setSuffix(" m")
        bf.addRow("Range Sigma:", self._bilat_range_spin)
        self._bilat_knn_spin = QSpinBox()
        self._bilat_knn_spin.setRange(5, 200)
        self._bilat_knn_spin.setValue(20)
        bf.addRow("K Neighbors:", self._bilat_knn_spin)
        self._bilateral_group.setVisible(False)
        left.addWidget(self._bilateral_group)

        # --- Thin Average params ---
        self._thin_group = QGroupBox("Thin — Grid Average Parameters")
        tf = QFormLayout(self._thin_group)
        self._thin_grid_spin = QDoubleSpinBox()
        self._thin_grid_spin.setRange(0.1, 100.0)
        self._thin_grid_spin.setDecimals(2)
        self._thin_grid_spin.setValue(2.0)
        self._thin_grid_spin.setSuffix(" m")
        tf.addRow("Grid Size:", self._thin_grid_spin)
        self._thin_group.setVisible(False)
        left.addWidget(self._thin_group)

        # --- Add to pipeline button ---
        add_btn = QPushButton("➕ Add to Pipeline & Preview")
        add_btn.clicked.connect(self._on_add_to_pipeline)
        left.addWidget(add_btn)

        left.addStretch()
        layout.addLayout(left, stretch=1)

        # ── Middle: pipeline list ───────────────────────────────────
        mid = QVBoxLayout()
        mid.addWidget(QLabel("<b>Pipeline Steps:</b>"))
        self._pipeline_list = QListWidget()
        self._pipeline_list.setAlternatingRowColors(True)
        mid.addWidget(self._pipeline_list, stretch=1)

        btn_row = QHBoxLayout()
        self._remove_btn = QPushButton("✕ Remove Step")
        self._remove_btn.clicked.connect(self._on_remove_step)
        btn_row.addWidget(self._remove_btn)
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._on_clear_pipeline)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        self._save_btn = QPushButton("💾 Save Preset")
        self._save_btn.setToolTip("Save current pipeline as a preset file (.json)")
        self._save_btn.clicked.connect(self._on_save_preset)
        btn_row.addWidget(self._save_btn)
        self._load_btn = QPushButton("📂 Load Preset")
        self._load_btn.setToolTip("Load a saved pipeline preset")
        self._load_btn.clicked.connect(self._on_load_preset)
        btn_row.addWidget(self._load_btn)
        mid.addLayout(btn_row)

        # --- Tile navigation ---
        tile_nav = QHBoxLayout()
        self._prev_tile_btn = QPushButton("◀ Prev Tile")
        self._prev_tile_btn.setToolTip("Preview previous tile (no pipeline change)")
        self._prev_tile_btn.clicked.connect(self._on_prev_tile)
        tile_nav.addWidget(self._prev_tile_btn)
        self._tile_label = QLabel(f"Tile 1 / {len(self._tile_ids)}")
        self._tile_label.setAlignment(Qt.AlignCenter)
        tile_nav.addWidget(self._tile_label)
        self._next_tile_btn = QPushButton("Next Tile ▶")
        self._next_tile_btn.setToolTip("Preview next tile (no pipeline change)")
        self._next_tile_btn.clicked.connect(self._on_next_tile)
        tile_nav.addWidget(self._next_tile_btn)
        mid.addLayout(tile_nav)
        self._update_tile_nav_buttons()

        # --- Status ---
        self._preview_status = QLabel("Loading preview…")
        self._preview_status.setWordWrap(True)
        self._preview_status.setStyleSheet("font-weight: bold; color: #333;")
        mid.addWidget(self._preview_status)

        # --- Batch apply ---
        self._batch_check = QCheckBox(
            f"Apply to all {len(self._tile_ids)} selected tile(s)"
        )
        self._batch_check.setChecked(True)
        mid.addWidget(self._batch_check)

        # --- Buttons ---
        btn_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Apply).setText("Apply Pipeline")
        btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        btn_box.rejected.connect(self.reject)
        mid.addWidget(btn_box)

        layout.addLayout(mid, stretch=1)

        # ── Right: 3D preview ───────────────────────────────────────
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

        # Connect all parameter spinboxes for live preview updates
        self._connect_live_preview()

    def _connect_live_preview(self):
        """Wire all parameter spinboxes to trigger a debounced preview update."""
        spinboxes = [
            self._sor_nb_spin, self._sor_std_spin,
            self._ror_radius_spin, self._ror_min_spin,
            self._dbscan_eps_spin, self._dbscan_min_samples_spin, self._dbscan_min_cluster_spin,
            self._isolated_radius_spin, self._isolated_min_nbr_spin,
            self._lowpts_radius_spin, self._lowpts_below_spin, self._lowpts_above_spin,
            self._surf_grid_spin, self._surf_tol_spin,
            self._mp_depth_spin,
            self._bilat_spatial_spin, self._bilat_range_spin, self._bilat_knn_spin,
            self._thin_grid_spin,
        ]
        for sb in spinboxes:
            sb.valueChanged.connect(self._schedule_preview_update)

    # ── filter type switching ──────────────────────────────────────

    def _on_filter_type_changed(self, index):
        ft = self._filter_type_combo.currentData()
        self._sor_group.setVisible(ft == "sor")
        self._ror_group.setVisible(ft == "ror")
        self._dbscan_group.setVisible(ft in ("dbscan_above", "dbscan_below"))
        self._isolated_group.setVisible(ft == "isolated")
        self._lowpts_group.setVisible(ft == "low_points")
        self._surf_group.setVisible(ft == "surface_noise")
        self._multipath_group.setVisible(ft == "multipath")
        self._bilateral_group.setVisible(ft == "bilateral")
        self._thin_group.setVisible(ft == "thin_average")

    def _on_sor_slider_changed(self, value):
        self._sor_std_spin.blockSignals(True)
        self._sor_std_spin.setValue(value / 10.0)
        self._sor_std_spin.blockSignals(False)
        self._schedule_preview_update()

    # ── pipeline management ────────────────────────────────────────

    def _on_add_to_pipeline(self):
        params = self._get_current_params()
        self._pipeline.append(params)
        self._rebuild_pipeline_list()
        self._update_preview()

    def _on_remove_step(self):
        row = self._pipeline_list.currentRow()
        if 0 <= row < len(self._pipeline):
            self._pipeline.pop(row)
            self._rebuild_pipeline_list()
            self._update_preview()

    def _on_clear_pipeline(self):
        self._pipeline.clear()
        self._rebuild_pipeline_list()
        self._update_preview()

    def _rebuild_pipeline_list(self):
        self._pipeline_list.clear()
        for i, step in enumerate(self._pipeline):
            label = f"{i+1}. {step['type'].upper()}"
            if step["type"] == "sor":
                label += f"  (k={step['nb_neighbors']}, σ={step['std_ratio']})"
            elif step["type"] == "ror":
                label += f"  (r={step['radius']}, min={step['min_points']})"
            elif step["type"] == "isolated":
                label += f"  (r={step['search_radius']}, n={step['min_neighbors']})"
            elif step["type"] == "low_points":
                label += f"  (r={step['search_radius']}, ↓{step['below_threshold']} ↑{step['above_threshold']})"
            elif step["type"] == "surface_noise":
                label += f"  (grid={step['grid_size']}, tol=±{step['tolerance']}m)"
            elif step["type"] == "multipath":
                label += f"  (depth_thresh={step['depth_threshold']})"
            elif step["type"] == "bilateral":
                label += f"  (σs={step['spatial_sigma']}, σr={step['range_sigma']}, k={step['knn']})"
            elif step["type"] == "thin_average":
                label += f"  (grid={step['grid_size']}m)"
            else:
                label += f"  (ε={step['eps']}, min_s={step['min_samples']}, min_c={step['min_cluster_size']})"
            self._pipeline_list.addItem(QListWidgetItem(label))

    def _get_current_params(self):
        ft = self._filter_type_combo.currentData()
        if ft == "sor":
            return {"type": "sor",
                    "nb_neighbors": self._sor_nb_spin.value(),
                    "std_ratio": self._sor_std_spin.value()}
        elif ft == "ror":
            return {"type": "ror",
                    "radius": self._ror_radius_spin.value(),
                    "min_points": self._ror_min_spin.value()}
        elif ft == "isolated":
            return {"type": "isolated",
                    "search_radius": self._isolated_radius_spin.value(),
                    "min_neighbors": self._isolated_min_nbr_spin.value()}
        elif ft == "low_points":
            return {"type": "low_points",
                    "search_radius": self._lowpts_radius_spin.value(),
                    "below_threshold": self._lowpts_below_spin.value(),
                    "above_threshold": self._lowpts_above_spin.value()}
        elif ft == "surface_noise":
            return {"type": "surface_noise",
                    "grid_size": self._surf_grid_spin.value(),
                    "tolerance": self._surf_tol_spin.value()}
        elif ft == "multipath":
            return {"type": "multipath",
                    "depth_threshold": self._mp_depth_spin.value()}
        elif ft == "bilateral":
            return {"type": "bilateral",
                    "spatial_sigma": self._bilat_spatial_spin.value(),
                    "range_sigma": self._bilat_range_spin.value(),
                    "knn": self._bilat_knn_spin.value()}
        elif ft == "thin_average":
            return {"type": "thin_average",
                    "grid_size": self._thin_grid_spin.value()}
        elif ft == "":
            return {"type": "sor", "nb_neighbors": 20, "std_ratio": 2.0}
        else:
            return {"type": ft,
                    "eps": self._dbscan_eps_spin.value(),
                    "min_samples": self._dbscan_min_samples_spin.value(),
                    "min_cluster_size": self._dbscan_min_cluster_spin.value()}

    # ── preview ────────────────────────────────────────────────────

    def _schedule_preview_update(self):
        self._preview_timer.start()

    def _load_preview_sample(self):
        if not self._tile_ids:
            self._preview_status.setText("No tiles selected.")
            return
        tile_id = self._tile_ids[self._preview_tile_idx]
        data = self._tm.load_tile_points_full(tile_id)
        if data is None:
            self._preview_status.setText("Failed to load preview data.")
            return
        n = len(data["x"])
        TARGET = 500_000
        if n > TARGET:
            # Spatial cluster around a random center, with margin from
            # tile edges so the cluster fits fully inside the tile.
            # Compute expected cluster radius from point density.
            x_range = float(data["x"].max() - data["x"].min())
            y_range = float(data["y"].max() - data["y"].min())
            area = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
            density = n / area
            cluster_radius = np.sqrt(TARGET / (density * np.pi)) if density > 0 else x_range
            margin = max(cluster_radius, 1.0)

            x_min, x_max = float(data["x"].min()), float(data["x"].max())
            y_min, y_max = float(data["y"].min()), float(data["y"].max())
            if x_max - x_min > 2 * margin and y_max - y_min > 2 * margin:
                cx = np.random.uniform(x_min + margin, x_max - margin)
                cy = np.random.uniform(y_min + margin, y_max - margin)
                cz = float(data["z"].mean())  # height doesn't matter for XY
            else:
                # Tile too small for margin — use centroid
                cx, cy, cz = float(data["x"].mean()), float(data["y"].mean()), float(data["z"].mean())

            # XY-only distance → cylinder, not sphere. Includes noise
            # points far above/below ground that would be excluded by a
            # spherical 3D distance.
            dists = (data["x"] - cx)**2 + (data["y"] - cy)**2
            indices = np.argpartition(dists, TARGET)[:TARGET]
            data = {k: v[indices] for k, v in data.items()}
        self._preview_points = data
        self._update_preview()

    def _on_prev_tile(self):
        if self._preview_tile_idx > 0:
            self._preview_tile_idx -= 1
            self._load_preview_sample()
            self._update_tile_nav_buttons()

    def _on_next_tile(self):
        if self._preview_tile_idx < len(self._tile_ids) - 1:
            self._preview_tile_idx += 1
            self._load_preview_sample()
            self._update_tile_nav_buttons()

    def _update_tile_nav_buttons(self):
        self._prev_tile_btn.setEnabled(self._preview_tile_idx > 0)
        self._next_tile_btn.setEnabled(self._preview_tile_idx < len(self._tile_ids) - 1)
        self._tile_label.setText(
            f"Tile {self._preview_tile_idx + 1} / {len(self._tile_ids)}"
        )

    def _update_preview(self):
        if self._preview_points is None:
            return
        pts = self._preview_points
        n = len(pts["x"])

        # Get live UI params for the currently selected filter type
        live_params = self._get_current_params()

        # Start with all points kept, then apply pipeline
        keep = np.ones(n, dtype=bool)
        for i, step in enumerate(self._pipeline):
            # If this is the last step and its type matches the current
            # filter type, use live UI values instead of stored params
            # so parameter changes preview instantly.
            is_last = (i == len(self._pipeline) - 1)
            params = live_params if (is_last and step["type"] == live_params["type"]) else step
            try:
                if params["type"] == "sor":
                    k, _ = statistical_outlier_removal(
                        pts["x"][keep], pts["y"][keep], pts["z"][keep],
                        nb_neighbors=params["nb_neighbors"],
                        std_ratio=params["std_ratio"],
                    )
                elif params["type"] == "ror":
                    k, _ = radius_outlier_removal(
                        pts["x"][keep], pts["y"][keep], pts["z"][keep],
                        radius=params["radius"],
                        min_points=params["min_points"],
                    )
                elif params["type"] == "isolated":
                    k, _ = isolated_point_removal(
                        pts["x"][keep], pts["y"][keep], pts["z"][keep],
                        search_radius=params["search_radius"],
                        min_neighbors=params["min_neighbors"],
                    )
                elif params["type"] == "low_points":
                    k, _ = low_point_removal(
                        pts["x"][keep], pts["y"][keep], pts["z"][keep],
                        search_radius=params["search_radius"],
                        below_threshold=params["below_threshold"],
                        above_threshold=params["above_threshold"],
                    )
                elif params["type"] == "surface_noise":
                    k, _ = surface_noise_removal(
                        pts["x"][keep], pts["y"][keep], pts["z"][keep],
                        grid_size=params["grid_size"],
                        tolerance=params["tolerance"],
                    )
                elif params["type"] == "thin_average":
                    k, _ = thin_points_average(
                        pts["x"][keep], pts["y"][keep], pts["z"][keep],
                        grid_size=params["grid_size"],
                    )
                else:
                    mode = "above" if params["type"] == "dbscan_above" else "below"
                    k, _ = dbscan_outlier_removal(
                        pts["x"][keep], pts["y"][keep], pts["z"][keep],
                        eps=params["eps"], min_samples=params["min_samples"],
                        min_cluster_size=params["min_cluster_size"],
                        mode=mode,
                    )
                # Map back to original indices
                keep_indices = np.where(keep)[0]
                keep[keep_indices[~k]] = False
            except Exception as exc:
                self._preview_status.setText(f"Pipeline error: {exc}")
                return

        self._current_keep_mask = keep
        outlier = ~keep

        n_out = int(outlier.sum())
        pct = n_out / n * 100 if n > 0 else 0
        self._preview_status.setText(
            f"Outliers: {n_out:,} / {n:,} ({pct:.1f}%)\n"
            f"Kept:     {n - n_out:,} points  |  {len(self._pipeline)} step(s)"
        )

        # Coloured preview: outliers = red, kept = class colour
        cls_arr = pts.get("classification")
        colors = np.zeros((n, 3), dtype=np.float64)
        if cls_arr is not None:
            for code in np.unique(cls_arr):
                colors[(cls_arr == code) & keep] = get_class_color(int(code))
        else:
            colors[keep] = (0.6, 0.6, 0.6)
        colors[outlier] = (1.0, 0.15, 0.15)

        self._preview_view.load_point_cloud_colored(pts["x"], pts["y"], pts["z"], colors)

    # ── apply ──────────────────────────────────────────────────────

    def _on_apply(self):
        if not self._pipeline:
            self._preview_status.setText("Add at least one filter step to the pipeline.")
            return
        if self._batch_check.isChecked():
            self.filter_applied.emit(self._tile_ids, self._pipeline)
        self.accept()

    # ── preset save / load ─────────────────────────────────────────

    def _on_save_preset(self):
        """Save the current pipeline as a JSON preset file."""
        if not self._pipeline:
            self._preview_status.setText("No pipeline steps to save.")
            return
        import json
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Filter Preset", "filter_preset.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump({
                    "description": "LiDAR Workbench filter pipeline preset",
                    "pipeline": self._pipeline,
                }, f, indent=2)
            self._preview_status.setText(f"Preset saved: {path}")
        except Exception as exc:
            self._preview_status.setText(f"Save failed: {exc}")

    def _on_load_preset(self):
        """Load a pipeline from a JSON preset file."""
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Filter Preset", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            loaded = data.get("pipeline", [])
            if not isinstance(loaded, list):
                self._preview_status.setText("Invalid preset: 'pipeline' must be a list.")
                return
            # Validate basic structure
            for step in loaded:
                if not isinstance(step, dict) or "type" not in step:
                    self._preview_status.setText("Invalid preset: each step must have 'type'.")
                    return
            self._pipeline = loaded
            self._rebuild_pipeline_list()
            self._update_preview()
            self._preview_status.setText(f"Loaded {len(loaded)} step(s) from: {path}")
        except Exception as exc:
            self._preview_status.setText(f"Load failed: {exc}")
