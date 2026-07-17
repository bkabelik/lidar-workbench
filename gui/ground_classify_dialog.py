"""
LiDAR Workbench — Ground Classification Dialog.

Dedicated dialog for classifying ground points using industry-standard
algorithms: SMRF (Simple Morphological Filter) and Progressive TIN
Densification (Axelsson).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..noise_filter import ground_classify_smrf, ground_classify_tin

logger = logging.getLogger("lidar_workbench.gui.ground_classify_dialog")


class _GroundClassifyWorker(QThread):
    progress = Signal(str, float)
    finished = Signal(np.ndarray)  # ground_mask
    error = Signal(str)

    def __init__(self, xs, ys, zs, classifications, method: str, params: dict, parent=None):
        super().__init__(parent)
        self._xs, self._ys, self._zs = xs, ys, zs
        self._classifications = classifications
        self._method = method
        self._params = params

    def run(self):
        try:
            if self._method == "smrf":
                mask = ground_classify_smrf(
                    self._xs, self._ys, self._zs,
                    cell_size=self._params.get("cell_size"),
                    slope_threshold=self._params["slope_threshold"],
                    max_window=self._params["max_window"],
                    elevation_threshold=self._params["elevation_threshold"],
                    base_window=self._params.get("base_window", 2.0),
                    window_growth=self._params.get("window_growth", 1.5),
                    progress=lambda msg, pct: self.progress.emit(msg, pct),
                )
            else:
                mask = ground_classify_tin(
                    self._xs, self._ys, self._zs,
                    max_distance=self._params["max_distance"],
                    max_angle=self._params["max_angle"],
                    max_distance_above=self._params.get("max_distance_above", 0.15),
                    cell_size=self._params.get("cell_size"),
                    progress=lambda msg, pct: self.progress.emit(msg, pct),
                )
            self.finished.emit(mask)
        except Exception as exc:
            self.error.emit(str(exc))


class GroundClassifyDialog(QDialog):
    """Dialog for ground classification with preview."""

    ground_applied = Signal(np.ndarray, int)  # ground_mask, source_class

    def __init__(self, tile_data: dict, parent=None):
        super().__init__(parent)
        self._data = tile_data
        self._worker: Optional[_GroundClassifyWorker] = None
        self._ground_mask: Optional[np.ndarray] = None
        self._source_class: int = 2  # default: class 2 (ground) from Pointcept

        self.setWindowTitle("Ground Classification")
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Method selector
        method_group = QGroupBox("Algorithm")
        mf = QFormLayout(method_group)
        self._method_combo = QComboBox()
        self._method_combo.addItem("SMRF — Simple Morphological Filter (PDAL)", "smrf")
        self._method_combo.addItem("TIN — Progressive Densification", "tin")
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        mf.addRow("Method:", self._method_combo)

        # Source class filter — only reclassify points matching this class
        self._source_class_combo = QComboBox()
        self._source_class_combo.addItem("2: Ground (Pointcept default)", 2)
        self._source_class_combo.addItem("0: Created, Never Classified", 0)
        self._source_class_combo.addItem("1: Unclassified", 1)
        self._source_class_combo.addItem("1 & 2: Unclassified + Ground", -2)
        self._source_class_combo.addItem("All classes", -1)
        self._source_class_combo.setCurrentIndex(0)
        self._source_class_combo.setToolTip(
            "Only reclassify points matching the selected source class(es). "
            "Other classes are left unchanged."
        )
        mf.addRow("Source Class:", self._source_class_combo)
        layout.addWidget(method_group)

        # SMRF params
        self._smrf_group = QGroupBox("SMRF Parameters")
        sf = QFormLayout(self._smrf_group)
        self._smrf_slope = QDoubleSpinBox()
        self._smrf_slope.setRange(0.01, 1.0)
        self._smrf_slope.setDecimals(3)
        self._smrf_slope.setValue(0.15)
        self._smrf_slope.setToolTip("Lower = more aggressive vegetation removal")
        sf.addRow("Slope Threshold:", self._smrf_slope)
        self._smrf_maxw = QDoubleSpinBox()
        self._smrf_maxw.setRange(5.0, 200.0)
        self._smrf_maxw.setDecimals(1)
        self._smrf_maxw.setValue(20.0)
        self._smrf_maxw.setSuffix(" m")
        self._smrf_maxw.setToolTip("Largest morphological window")
        sf.addRow("Max Window:", self._smrf_maxw)
        self._smrf_elev = QDoubleSpinBox()
        self._smrf_elev.setRange(0.05, 5.0)
        self._smrf_elev.setDecimals(2)
        self._smrf_elev.setValue(0.5)
        self._smrf_elev.setSuffix(" m")
        self._smrf_elev.setToolTip("Max height above filtered surface for ground")
        sf.addRow("Elevation Thresh:", self._smrf_elev)
        layout.addWidget(self._smrf_group)

        # TIN params
        self._tin_group = QGroupBox("TIN Densification Parameters")
        tf = QFormLayout(self._tin_group)
        self._tin_dist = QDoubleSpinBox()
        self._tin_dist.setRange(0.1, 10.0)
        self._tin_dist.setDecimals(2)
        self._tin_dist.setValue(1.4)
        self._tin_dist.setSuffix(" m")
        self._tin_dist.setToolTip("Max distance from point to TIN surface")
        tf.addRow("Max Distance:", self._tin_dist)
        self._tin_angle = QDoubleSpinBox()
        self._tin_angle.setRange(1.0, 45.0)
        self._tin_angle.setDecimals(1)
        self._tin_angle.setValue(6.0)
        self._tin_angle.setSuffix("°")
        self._tin_angle.setToolTip("Max angle between point and TIN vertices")
        tf.addRow("Max Angle:", self._tin_angle)
        self._tin_above = QDoubleSpinBox()
        self._tin_above.setRange(0.01, 2.0)
        self._tin_above.setDecimals(2)
        self._tin_above.setValue(0.15)
        self._tin_above.setSuffix(" m")
        self._tin_above.setToolTip(
            "Max height ABOVE TIN for a point to be ground. "
            "Very tight (0.10–0.20 m) to reject water surface."
        )
        tf.addRow("Max Above TIN:", self._tin_above)
        self._tin_group.setVisible(False)
        layout.addWidget(self._tin_group)

        # Info
        info = QLabel(
            "<b>SMRF</b> (PDAL): fast, good for most terrain. "
            "Uses progressive morphological opening.\n\n"
            "<b>TIN Densification</b>: iterative, "
            "preserves sharp terrain breaks (cliffs, riverbanks). "
            "Slower but more precise on complex terrain."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Progress
        self._status = QLabel("Ready")
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        # Buttons
        btn_layout = QHBoxLayout()
        self._run_btn = QPushButton("▶ Classify Ground")
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self._ok_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout)

    def _on_method_changed(self):
        method = self._method_combo.currentData()
        self._smrf_group.setVisible(method == "smrf")
        self._tin_group.setVisible(method == "tin")

    def _on_run(self):
        method = self._method_combo.currentData()
        if method == "smrf":
            params = {
                "slope_threshold": self._smrf_slope.value(),
                "max_window": self._smrf_maxw.value(),
                "elevation_threshold": self._smrf_elev.value(),
                "base_window": 2.0,
                "window_growth": 1.5,
                "cell_size": None,
            }
        else:
            params = {
                "max_distance": self._tin_dist.value(),
                "max_angle": self._tin_angle.value(),
                "max_distance_above": self._tin_above.value(),
                "cell_size": None,
            }

        self._run_btn.setEnabled(False)
        self._status.setText("Classifying ground...")
        self._progress.setValue(0)

        self._worker = _GroundClassifyWorker(
            self._data["x"], self._data["y"], self._data["z"],
            self._data["classification"],
            method=method, params=params, parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str, pct: float):
        self._status.setText(msg)
        self._progress.setValue(int(pct))

    def _on_finished(self, mask: np.ndarray):
        self._ground_mask = mask
        n_total = len(mask)
        n_ground = mask.sum()
        self._status.setText(
            f"Done: {n_ground:,} ground / {n_total:,} points "
            f"({n_ground/n_total*100:.1f}%)"
        )
        self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self._ok_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg}")
        self._run_btn.setEnabled(True)

    def _on_accept(self):
        if self._ground_mask is not None:
            src_cls = self._source_class_combo.currentData()
            self.ground_applied.emit(self._ground_mask, src_cls)
        self.accept()

    @property
    def ground_mask(self) -> Optional[np.ndarray]:
        return self._ground_mask
