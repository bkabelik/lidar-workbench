"""
LiDAR Workbench — Surface Cleanup Dialog.

Post-classification tool that flags near-surface noise using the
already-classified ground (class 2) as a reference surface.

Shows a live preview of a spatial sample so the user can tune
grid size and tolerance before applying to the full tile.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..noise_filter import surface_noise_removal

logger = logging.getLogger("lidar_workbench.gui.surface_cleanup_dialog")

# ASPRS class used for low-point noise
NOISE_CLASS = 7


class _PreviewCanvas(QWidget):
    """Minimal QPainter canvas showing kept (blue) vs noise (red) points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self._pts_x: Optional[np.ndarray] = None
        self._pts_y: Optional[np.ndarray] = None
        self._keep: Optional[np.ndarray] = None
        self._noise: Optional[np.ndarray] = None

    def set_data(self, xs, ys, keep_mask):
        self._pts_x = np.asarray(xs, dtype=np.float64)
        self._pts_y = np.asarray(ys, dtype=np.float64)
        self._keep = np.asarray(keep_mask, dtype=bool)
        self._noise = ~self._keep
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pts_x is None or len(self._pts_x) == 0:
            return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1a1a2e"))
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            p.end(); return

        # Scale to fit
        pad = 0.05
        x_min, x_max = float(self._pts_x.min()), float(self._pts_x.max())
        y_min, y_max = float(self._pts_y.min()), float(self._pts_y.max())
        rx, ry = x_max - x_min, y_max - y_min
        if rx <= 0: rx = 1.0
        if ry <= 0: ry = 1.0
        sx = w * (1 - 2 * pad) / rx
        sy = h * (1 - 2 * pad) / ry
        s = min(sx, sy)
        ox = (w - rx * s) / 2
        oy = (h - ry * s) / 2

        # Draw kept points (blue)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(60, 120, 255, 180)))
        for i in np.where(self._keep)[0]:
            px = ox + (self._pts_x[i] - x_min) * s
            py = oy + (y_max - self._pts_y[i]) * s
            p.drawEllipse(int(px), int(py), 2, 2)

        # Draw noise points (red)
        p.setBrush(QBrush(QColor(255, 60, 60, 220)))
        for i in np.where(self._noise)[0]:
            px = ox + (self._pts_x[i] - x_min) * s
            py = oy + (y_max - self._pts_y[i]) * s
            p.drawEllipse(int(px), int(py), 3, 3)
        p.end()


class SurfaceCleanupDialog(QDialog):
    """
    Dialog for cleaning near-surface noise after ground classification.

    Builds a surface from ground points (class 2), then flags non-ground
    points within ±tolerance as noise (class 7).
    """

    # (noise_mask, noise_class) emitted when user clicks Apply
    cleanup_applied = Signal(object, int)

    def __init__(self, point_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Surface Cleanup")
        self.setMinimumWidth(500)
        self._data = point_data
        self._n = len(point_data["x"])
        self._sample: Optional[dict] = None
        self._setup_ui()
        self._load_sample()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Parameters ─────────────────────────────────────────
        param_group = QGroupBox("Parameters")
        pf = QFormLayout(param_group)

        self._grid_spin = QDoubleSpinBox()
        self._grid_spin.setRange(0.1, 5.0)
        self._grid_spin.setDecimals(2)
        self._grid_spin.setValue(0.5)
        self._grid_spin.setSuffix(" m")
        self._grid_spin.setToolTip("Surface grid cell size")
        pf.addRow("Grid Size:", self._grid_spin)

        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(0.01, 1.0)
        self._tol_spin.setDecimals(3)
        self._tol_spin.setValue(0.15)
        self._tol_spin.setSuffix(" m")
        self._tol_spin.setToolTip("Band around the ground surface to flag as noise")
        pf.addRow("Tolerance (±):", self._tol_spin)

        layout.addWidget(param_group)

        # ── Preview ────────────────────────────────────────────
        prev_group = QGroupBox("Preview (500k sample)")
        pv = QVBoxLayout(prev_group)
        self._canvas = _PreviewCanvas()
        pv.addWidget(self._canvas, stretch=1)

        self._status_label = QLabel("Loading sample…")
        pv.addWidget(self._status_label)

        layout.addWidget(prev_group, stretch=1)

        # ── Buttons ────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        self._apply_btn = QPushButton("Apply — Flag as Noise (class 7)")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; font-weight: bold; padding: 6px 14px; }"
            "QPushButton:hover { background: #e74c3c; }"
        )
        self._apply_btn.clicked.connect(self._on_apply)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(self._apply_btn)
        layout.addLayout(btn_layout)

        # ── Live preview timer ─────────────────────────────────
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._update_preview)

        self._grid_spin.valueChanged.connect(self._schedule_preview)
        self._tol_spin.valueChanged.connect(self._schedule_preview)

    # ── preview logic ──────────────────────────────────────────

    def _schedule_preview(self):
        self._preview_timer.start()

    def _load_sample(self):
        """Take a 500k spatial cluster for live preview."""
        n = self._n
        TARGET = 500_000
        if n <= TARGET:
            self._sample = self._data
        else:
            data = self._data
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
            else:
                cx, cy = float(data["x"].mean()), float(data["y"].mean())

            dists = (data["x"] - cx)**2 + (data["y"] - cy)**2
            indices = np.argpartition(dists, TARGET)[:TARGET]
            self._sample = {k: v[indices] for k, v in data.items()}
        self._update_preview()

    def _update_preview(self):
        if self._sample is None:
            return
        s = self._sample
        try:
            _, outlier = surface_noise_removal(
                s["x"], s["y"], s["z"],
                grid_size=self._grid_spin.value(),
                tolerance=self._tol_spin.value(),
                classifications=s.get("classification"),
                reference_class=2,  # ground
            )
            n_out = int(outlier.sum())
            n_total = len(s["x"])
            self._status_label.setText(
                f"Noise flagged: {n_out:,} / {n_total:,} points "
                f"({n_out / n_total * 100:.1f}%)"
            )
            self._canvas.set_data(s["x"], s["y"], ~outlier)
        except Exception as exc:
            self._status_label.setText(f"Preview error: {exc}")

    def _on_apply(self):
        """Run on full data and emit the result."""
        data = self._data
        try:
            _, outlier = surface_noise_removal(
                data["x"], data["y"], data["z"],
                grid_size=self._grid_spin.value(),
                tolerance=self._tol_spin.value(),
                classifications=data.get("classification"),
                reference_class=2,
            )
            self.cleanup_applied.emit(outlier, NOISE_CLASS)
            self.accept()
        except Exception as exc:
            logger.error("Surface cleanup failed: %s", exc)
