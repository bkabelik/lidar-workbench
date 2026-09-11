"""
LiDAR Workbench — Ground Control Dialog.

Provides two ground control workflows:
  - **Ground Control Points (GCP)**: Import CSV with name/X/Y/Z, compute
    elevation difference against selected point-cloud classes, apply Z shift.
  - **House Roofs & Areas**: Import polygon surfaces, find the best-fit
    plane in the point cloud, compute normal distance, apply shift.

Both tabs include a *Visual Check* mode to step through each control
element and inspect the point cloud around it in 3-D.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csgraph
import pyqtgraph as pg

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor
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
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import ASPRS_CLASS_NAMES
from ..crs import transform_coordinates

try:
    from pyproj import CRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

logger = logging.getLogger("lidar_workbench.gui.ground_control_dialog")


def _safe_float(val: any) -> float:
    """Parse float value, handling European decimal commas and whitespace."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(" ", "").replace(",", ".")
    return float(s)


class _NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically by float value."""

    def __init__(self, val: Optional[float], text: str = ""):
        display_text = text if text != "" else (f"{val:.3f}" if val is not None else "N/A")
        super().__init__(display_text)
        self._val = val

    def __lt__(self, other):
        if isinstance(other, _NumericTableWidgetItem):
            v1 = self._val
            v2 = other._val
            if v1 is None:
                return False
            if v2 is None:
                return True
            return v1 < v2
        return super().__lt__(other)


class _CheckboxTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem for checkbox that sorts checked before unchecked."""

    def __init__(self, checked: bool = True):
        super().__init__()
        self.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            c1 = 1 if self.checkState() == Qt.Checked else 0
            c2 = 1 if other.checkState() == Qt.Checked else 0
            return c1 < c2
        return super().__lt__(other)


# ═══════════════════════════════════════════════════════════════════════
# Roof & Patch Helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_roof_point_name(name: str) -> Tuple[str, str]:
    """
    Parse a surveyor point name to determine (patch_id, point_id).

    Conventions handled:
      - 6-digit numeric IDs: e.g. '010203' -> GCP '01', House '02', Point '03'
        Returns ('0102', '03') to guarantee houses remain unique across GCPs.
      - Delimited 3-part: '01_02_03', '01-02-03', 'GCP1_H2_P3' -> ('01_02', '03')
      - Delimited 2-part: 'H1_1', 'ROOF2-P1' -> ('H1', '1'), ('ROOF2', 'P1')
      - Fallback: (name, '1')
    """
    s = str(name).strip()
    # 1) 6-digit format GGRRPP (e.g. 010101 -> GCP 01, House 01, Point 01)
    m6 = re.match(r"^(\d{2})(\d{2})(\d{2})$", s)
    if m6:
        return f"{m6.group(1)}{m6.group(2)}", m6.group(3)
    # 2) Delimited 3-part (e.g. 01_01_01, 1-2-3, 01.02.03)
    m3 = re.match(r"^(.*?)[_\-\.\/](\w+)[_\-\.\/]([pP]?\d+|\w+)$", s)
    if m3:
        return f"{m3.group(1)}_{m3.group(2)}", m3.group(3)
    # 3) Delimited 2-part (e.g. H1_1, ROOF1-2, P1_P2)
    m2 = re.match(r"^(.*?)[_\-\.\/]([pP]?\d+)$", s)
    if m2:
        return m2.group(1), m2.group(2)
    return s, "1"


def cluster_roof_points_spatially(coords_xy: np.ndarray, radius: float = 10.0) -> np.ndarray:
    """
    Group 2D coordinates within `radius` (meters) into patches using connected components.
    Returns integer cluster labels for each point.
    """
    coords = np.asarray(coords_xy)
    if len(coords) == 0:
        return np.array([], dtype=int)
    if len(coords) == 1:
        return np.array([0], dtype=int)
    tree = cKDTree(coords)
    adj = tree.sparse_distance_matrix(tree, max_distance=radius)
    _, labels = csgraph.connected_components(adj)
    return labels


def sort_vertices_angular(verts: np.ndarray, clockwise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort polygon vertices angularly around their 2D centroid to eliminate zigzag/hourglass polygons.

    Parameters
    ----------
    verts : np.ndarray
        Array of shape (N, 2) or (N, 3).
    clockwise : bool
        If True, sort clockwise; otherwise counter-clockwise (default).

    Returns
    -------
    sorted_verts : np.ndarray
    sort_indices : np.ndarray
    """
    verts_arr = np.asarray(verts)
    if len(verts_arr) <= 2:
        return verts_arr, np.arange(len(verts_arr))
    cx = float(np.mean(verts_arr[:, 0]))
    cy = float(np.mean(verts_arr[:, 1]))
    angles = np.arctan2(verts_arr[:, 1] - cy, verts_arr[:, 0] - cx)
    order = np.argsort(-angles if clockwise else angles)
    return verts_arr[order], order


# ═══════════════════════════════════════════════════════════════════════
# 2-View Inspector (XY Plan + Z Elevation)
# ═══════════════════════════════════════════════════════════════════════

class _Roof2ViewInspector(QWidget):
    """
    Two-view interactive inspector and manual nudge editor for roof & ground patches.
      - Top View: XY Plan View (top-down) with LiDAR scatter, inliers, and control polygon.
      - Bottom View: Z Elevation Profile View (cross-section along slope/normal).
      - Step size selector and nudge buttons for XY and Z.
      - Live metric status and 'Reset to Auto-Fit' button.
    """

    offset_changed = Signal(str, float, float, float)  # patch_name, dx, dy, dz
    use_toggled = Signal(str, bool)                     # patch_name, used
    delete_requested = Signal(str)                      # patch_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_patch: Optional[dict] = None
        self._dx: float = 0.0
        self._dy: float = 0.0
        self._dz: float = 0.0
        self._base_dx: float = 0.0
        self._base_dy: float = 0.0
        self._base_dz: float = 0.0
        self._step_size: float = 0.05
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header bar
        head_box = QHBoxLayout()
        self._title_label = QLabel("<b>Select a patch from the results to inspect</b>")
        self._title_label.setStyleSheet("font-size: 12px;")
        head_box.addWidget(self._title_label, 1)

        self._use_chk = QCheckBox("☑ Use in Shift")
        self._use_chk.setChecked(True)
        self._use_chk.setToolTip("Include or exclude this patch from the XYZ shift calculation")
        self._use_chk.toggled.connect(self._on_use_toggled)
        self._use_chk.setEnabled(False)
        head_box.addWidget(self._use_chk)

        self._reset_btn = QPushButton("↺ Reset to Auto-Fit")
        self._reset_btn.setToolTip("Reset manual nudges back to RANSAC calculated offset")
        self._reset_btn.clicked.connect(self._reset_to_autofit)
        self._reset_btn.setEnabled(False)
        head_box.addWidget(self._reset_btn)

        self._del_btn = QPushButton("🗑 Delete")
        self._del_btn.setToolTip("Delete this bad or wrong patch from the surface model")
        self._del_btn.setStyleSheet("QPushButton { color: #e74c3c; font-weight: bold; }")
        self._del_btn.clicked.connect(self._on_delete_clicked)
        self._del_btn.setEnabled(False)
        head_box.addWidget(self._del_btn)
        layout.addLayout(head_box)

        # Splitter between XY View and Z View
        splitter = QSplitter(Qt.Vertical)

        # ── View 1: XY Plan View ──
        xy_container = QWidget()
        xy_layout = QVBoxLayout(xy_container)
        xy_layout.setContentsMargins(0, 0, 0, 0)
        xy_layout.setSpacing(2)

        self._plot_xy = pg.PlotWidget(title="XY Plan View (Yellow: GCP Boundary | Cyan: Shifted Cloud)")
        self._plot_xy.showGrid(x=True, y=True, alpha=0.3)
        self._plot_xy.setAspectLocked(True)
        self._plot_xy.setLabel("bottom", "X (Easting)", units="m")
        self._plot_xy.setLabel("left", "Y (Northing)", units="m")

        self._scatter_xy_pts = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(140, 140, 140, 100))
        self._scatter_xy_inl = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(0, 220, 255, 220))
        self._poly_xy_orig = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 200, 0), width=2))
        self._poly_xy_shifted = pg.PlotCurveItem(pen=pg.mkPen(color=(50, 255, 50, 0), width=0))
        self._scatter_xy_verts = pg.ScatterPlotItem(size=9, pen=pg.mkPen('w', width=1), brush=pg.mkBrush(255, 60, 60))

        self._plot_xy.addItem(self._scatter_xy_pts)
        self._plot_xy.addItem(self._scatter_xy_inl)
        self._plot_xy.addItem(self._poly_xy_orig)
        self._plot_xy.addItem(self._poly_xy_shifted)
        self._plot_xy.addItem(self._scatter_xy_verts)
        xy_layout.addWidget(self._plot_xy, 1)

        # XY Legend Bar
        xy_legend = QHBoxLayout()
        xy_legend.setContentsMargins(4, 1, 4, 1)
        xy_legend.setSpacing(10)
        lbl_gcp_poly = QLabel("🟡 <b>GCP Outline</b> (Fixed Truth)")
        lbl_gcp_poly.setStyleSheet("color: #f1c40f; font-size: 11px;")
        lbl_gcp_pts = QLabel("🔴 <b>GCP Corners</b>")
        lbl_gcp_pts.setStyleSheet("color: #e74c3c; font-size: 11px;")
        lbl_cld_inl = QLabel("🔵 <b>Shifted Cloud Inliers</b>")
        lbl_cld_inl.setStyleSheet("color: #00dcf5; font-size: 11px;")
        lbl_cld_all = QLabel("⚪ <b>Surrounding Cloud</b>")
        lbl_cld_all.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        xy_legend.addWidget(lbl_gcp_poly)
        xy_legend.addWidget(lbl_gcp_pts)
        xy_legend.addWidget(lbl_cld_inl)
        xy_legend.addWidget(lbl_cld_all)
        xy_legend.addStretch(1)
        xy_layout.addLayout(xy_legend)

        # XY Controls
        xy_ctrl = QHBoxLayout()
        xy_ctrl.addWidget(QLabel("Step:"))
        self._step_combo = QComboBox()
        for s in ["0.01", "0.02", "0.05", "0.10", "0.20", "0.50", "1.00"]:
            self._step_combo.addItem(f"{s} m", float(s))
        self._step_combo.setCurrentIndex(2)  # 0.05 m default
        self._step_combo.currentIndexChanged.connect(self._on_step_changed)
        xy_ctrl.addWidget(self._step_combo)

        btn_left = QPushButton("← -X")
        btn_left.setFixedWidth(52)
        btn_left.clicked.connect(lambda: self._nudge("x", -1))
        xy_ctrl.addWidget(btn_left)

        btn_right = QPushButton("→ +X")
        btn_right.setFixedWidth(52)
        btn_right.clicked.connect(lambda: self._nudge("x", +1))
        xy_ctrl.addWidget(btn_right)

        xy_ctrl.addWidget(QLabel("ΔX:"))
        self._spin_dx = QDoubleSpinBox()
        self._spin_dx.setRange(-100.0, 100.0)
        self._spin_dx.setDecimals(3)
        self._spin_dx.setSingleStep(0.01)
        self._spin_dx.setSuffix(" m")
        self._spin_dx.valueChanged.connect(self._on_spin_changed)
        xy_ctrl.addWidget(self._spin_dx)

        btn_down = QPushButton("↓ -Y")
        btn_down.setFixedWidth(52)
        btn_down.clicked.connect(lambda: self._nudge("y", -1))
        xy_ctrl.addWidget(btn_down)

        btn_up = QPushButton("↑ +Y")
        btn_up.setFixedWidth(52)
        btn_up.clicked.connect(lambda: self._nudge("y", +1))
        xy_ctrl.addWidget(btn_up)

        xy_ctrl.addWidget(QLabel("ΔY:"))
        self._spin_dy = QDoubleSpinBox()
        self._spin_dy.setRange(-100.0, 100.0)
        self._spin_dy.setDecimals(3)
        self._spin_dy.setSingleStep(0.01)
        self._spin_dy.setSuffix(" m")
        self._spin_dy.valueChanged.connect(self._on_spin_changed)
        xy_ctrl.addWidget(self._spin_dy)
        xy_ctrl.addStretch()

        xy_layout.addLayout(xy_ctrl)
        splitter.addWidget(xy_container)

        # ── View 2: Z Elevation View ──
        z_container = QWidget()
        z_layout = QVBoxLayout(z_container)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.setSpacing(2)

        self._plot_z = pg.PlotWidget(title="Z Profile (Yellow: GCP Plane | Green: Shifted Cloud Plane)")
        self._plot_z.showGrid(x=True, y=True, alpha=0.3)
        self._plot_z.setLabel("bottom", "Profile Distance (Along Slope)", units="m")
        self._plot_z.setLabel("left", "Elevation Z", units="m")

        self._scatter_z_pts = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(140, 140, 140, 100))
        self._scatter_z_inl = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(0, 220, 255, 220))
        self._line_z_cloud = pg.PlotCurveItem(pen=pg.mkPen(color=(0, 180, 220, 120), width=1, style=Qt.DotLine))
        self._line_z_orig = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 200, 0), width=2, style=Qt.DashLine))
        self._line_z_shifted = pg.PlotCurveItem(pen=pg.mkPen(color=(50, 255, 50), width=2))
        self._scatter_z_verts = pg.ScatterPlotItem(size=9, pen=pg.mkPen('w', width=1), brush=pg.mkBrush(255, 60, 60))

        self._plot_z.addItem(self._scatter_z_pts)
        self._plot_z.addItem(self._scatter_z_inl)
        self._plot_z.addItem(self._line_z_cloud)
        self._plot_z.addItem(self._line_z_orig)
        self._plot_z.addItem(self._line_z_shifted)
        self._plot_z.addItem(self._scatter_z_verts)
        z_layout.addWidget(self._plot_z, 1)

        # Z Legend Bar
        z_legend = QHBoxLayout()
        z_legend.setContentsMargins(4, 1, 4, 1)
        z_legend.setSpacing(10)
        lbl_z_gcp = QLabel("🟡 <b>GCP Plane</b> (Fixed Truth)")
        lbl_z_gcp.setStyleSheet("color: #f1c40f; font-size: 11px;")
        lbl_z_shift = QLabel("🟢 <b>Shifted Cloud Plane</b>")
        lbl_z_shift.setStyleSheet("color: #2ecc71; font-size: 11px;")
        lbl_z_inl = QLabel("🔵 <b>Shifted Points</b>")
        lbl_z_inl.setStyleSheet("color: #00dcf5; font-size: 11px;")
        lbl_z_raw = QLabel("┈ <b>Raw Cloud Plane</b>")
        lbl_z_raw.setStyleSheet("color: #00b4dc; font-size: 11px;")
        z_legend.addWidget(lbl_z_gcp)
        z_legend.addWidget(lbl_z_shift)
        z_legend.addWidget(lbl_z_inl)
        z_legend.addWidget(lbl_z_raw)
        z_legend.addStretch(1)
        z_layout.addLayout(z_legend)

        # Z Controls
        z_ctrl = QHBoxLayout()
        btn_z_down = QPushButton("▼ -Z")
        btn_z_down.setFixedWidth(52)
        btn_z_down.clicked.connect(lambda: self._nudge("z", -1))
        z_ctrl.addWidget(btn_z_down)

        btn_z_up = QPushButton("▲ +Z")
        btn_z_up.setFixedWidth(52)
        btn_z_up.clicked.connect(lambda: self._nudge("z", +1))
        z_ctrl.addWidget(btn_z_up)

        z_ctrl.addWidget(QLabel("ΔZ:"))
        self._spin_dz = QDoubleSpinBox()
        self._spin_dz.setRange(-100.0, 100.0)
        self._spin_dz.setDecimals(3)
        self._spin_dz.setSingleStep(0.01)
        self._spin_dz.setSuffix(" m")
        self._spin_dz.valueChanged.connect(self._on_spin_changed)
        z_ctrl.addWidget(self._spin_dz)

        legend_lbl = QLabel(
            "<span style='color:#00d2ff'>― LiDAR Plane</span> &nbsp; "
            "<span style='color:#ffc800'>-- Raw Input</span> &nbsp; "
            "<span style='color:#32ff32'>― Shifted Input</span> &nbsp; "
            "<span style='color:#ff3c3c'>■ Vertices</span>"
        )
        z_ctrl.addSpacing(15)
        z_ctrl.addWidget(legend_lbl)
        z_ctrl.addStretch()

        z_layout.addLayout(z_ctrl)
        splitter.addWidget(z_container)

        layout.addWidget(splitter, 1)

        # Bottom Metrics Bar
        self._metrics_label = QLabel("<i>No patch loaded</i>")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet(
            "background-color: #2b2b2b; color: #f0f0f0; padding: 6px; border-radius: 4px;"
        )
        layout.addWidget(self._metrics_label)

    def _on_step_changed(self):
        val = self._step_combo.currentData()
        if val is not None:
            self._step_size = float(val)

    def _nudge(self, axis: str, direction: int):
        if not self._current_patch:
            return
        delta = direction * self._step_size
        if axis == "x":
            self._spin_dx.setValue(self._spin_dx.value() + delta)
        elif axis == "y":
            self._spin_dy.setValue(self._spin_dy.value() + delta)
        elif axis == "z":
            self._spin_dz.setValue(self._spin_dz.value() + delta)

    def _on_spin_changed(self):
        if not self._current_patch:
            return
        self._dx = float(self._spin_dx.value())
        self._dy = float(self._spin_dy.value())
        self._dz = float(self._spin_dz.value())
        self._update_plots()
        self._update_metrics()
        self.offset_changed.emit(
            str(self._current_patch.get("name", "")),
            self._dx, self._dy, self._dz,
        )

    def _reset_to_autofit(self):
        if not self._current_patch:
            return
        self._spin_dx.blockSignals(True)
        self._spin_dy.blockSignals(True)
        self._spin_dz.blockSignals(True)
        self._spin_dx.setValue(self._base_dx)
        self._spin_dy.setValue(self._base_dy)
        self._spin_dz.setValue(self._base_dz)
        self._spin_dx.blockSignals(False)
        self._spin_dy.blockSignals(False)
        self._spin_dz.blockSignals(False)
    def _on_use_toggled(self, checked: bool):
        if not self._current_patch:
            return
        self._current_patch["used"] = checked
        self._update_title_label()
        self.use_toggled.emit(str(self._current_patch.get("name", "")), checked)

    def _on_delete_clicked(self):
        if not self._current_patch:
            return
        self.delete_requested.emit(str(self._current_patch.get("name", "")))

    def set_used(self, used: bool):
        if self._current_patch:
            self._current_patch["used"] = used
        self._use_chk.blockSignals(True)
        self._use_chk.setChecked(used)
        self._use_chk.blockSignals(False)
        self._update_title_label()

    def _update_title_label(self):
        if not self._current_patch:
            self._title_label.setText("<b>Select a patch from the results to inspect</b>")
            return
        name = self._current_patch.get("name", "Unnamed")
        ptype = self._current_patch.get("type", "roof")
        type_icon = "🏠 Roof" if ptype == "roof" else "🌱 Ground"
        n_verts = self._current_patch.get("n_verts", len(self._current_patch.get("verts", [])))
        n_nearby = self._current_patch.get("n_nearby", 0)
        is_used = self._current_patch.get("used", True)
        status_tag = "" if is_used else " &nbsp;<span style='color:#e74c3c; font-weight:bold;'>[DISMISSED / EXCLUDED]</span>"
        self._title_label.setText(
            f"<b>Patch:</b> {name} &nbsp; ({type_icon}) &nbsp; "
            f"| <b>Vertices:</b> {n_verts} &nbsp; | <b>Nearby Points:</b> {n_nearby}{status_tag}"
        )

    def clear_patch(self):
        self._current_patch = None
        self._title_label.setText("<b>Select a patch from the results to inspect</b>")
        self._use_chk.setEnabled(False)
        self._reset_btn.setEnabled(False)
        self._del_btn.setEnabled(False)
        self._scatter_xy_pts.clear()
        self._scatter_xy_inl.clear()
        self._poly_xy_orig.clear()
        self._poly_xy_shifted.clear()
        self._scatter_xy_verts.clear()
        self._scatter_z_pts.clear()
        self._scatter_z_inl.clear()
        self._line_z_cloud.clear()
        self._line_z_orig.clear()
        self._line_z_shifted.clear()
        self._scatter_z_verts.clear()
        self._metrics_label.setText("<i>No patch loaded</i>")

    def load_patch(self, patch: dict):
        self._current_patch = patch
        is_used = patch.get("used", True)
        self._use_chk.setEnabled(True)
        self._use_chk.blockSignals(True)
        self._use_chk.setChecked(is_used)
        self._use_chk.blockSignals(False)
        self._reset_btn.setEnabled(True)
        self._del_btn.setEnabled(True)
        self._update_title_label()

        self._dx = float(patch.get("dx", 0.0) or 0.0)
        self._dy = float(patch.get("dy", 0.0) or 0.0)
        self._dz = float(patch.get("dz", 0.0) or 0.0)
        self._base_dx = float(patch.get("base_dx", self._dx))
        self._base_dy = float(patch.get("base_dy", self._dy))
        self._base_dz = float(patch.get("base_dz", self._dz))

        self._spin_dx.blockSignals(True)
        self._spin_dy.blockSignals(True)
        self._spin_dz.blockSignals(True)
        self._spin_dx.setValue(self._dx)
        self._spin_dy.setValue(self._dy)
        self._spin_dz.setValue(self._dz)
        self._spin_dx.blockSignals(False)
        self._spin_dy.blockSignals(False)
        self._spin_dz.blockSignals(False)

        self._update_plots()
        self._update_metrics()

    def _update_plots(self):
        if not self._current_patch:
            return

        verts = self._current_patch.get("verts")
        if verts is None:
            return
        verts = np.asarray(verts, dtype=np.float64)
        if len(verts) == 0:
            return

        pts = self._current_patch.get("local_pts")
        inl = self._current_patch.get("inlier_mask")
        norm = self._current_patch.get("cloud_normal")
        cloud_mean = self._current_patch.get("cloud_mean")

        # ── XY Plan View ──
        if pts is not None and len(pts) > 0:
            self._scatter_xy_pts.setData(x=pts[:, 0], y=pts[:, 1])
            if inl is not None and len(inl) == len(pts) and inl.any():
                self._scatter_xy_inl.setData(
                    x=pts[inl, 0] + self._dx,
                    y=pts[inl, 1] + self._dy,
                )
            else:
                self._scatter_xy_inl.clear()
        else:
            self._scatter_xy_pts.clear()
            self._scatter_xy_inl.clear()

        # Closed polygon loop of surveyor GCP (FIXED ground truth reference)
        poly_x = np.append(verts[:, 0], verts[0, 0])
        poly_y = np.append(verts[:, 1], verts[0, 1])
        self._poly_xy_orig.setData(poly_x, poly_y)
        self._poly_xy_shifted.clear()
        self._scatter_xy_verts.setData(x=verts[:, 0], y=verts[:, 1])

        # ── Z Elevation View ──
        cx = float(np.mean(verts[:, 0]))
        cy = float(np.mean(verts[:, 1]))
        cz = float(np.mean(verts[:, 2]))

        if norm is not None:
            nh = float(np.hypot(norm[0], norm[1]))
            if nh > 1e-5:
                ux = float(norm[0] / nh)
                uy = float(norm[1] / nh)
                slope = float(nh / max(float(norm[2]), 0.01))
            else:
                ux, uy = 1.0, 0.0
                slope = 0.0
        else:
            ux, uy = 1.0, 0.0
            slope = 0.0

        d_shift = self._dx * ux + self._dy * uy

        if pts is not None and len(pts) > 0:
            d_pts = (pts[:, 0] - cx) * ux + (pts[:, 1] - cy) * uy
            z_pts = pts[:, 2]
            self._scatter_z_pts.setData(x=d_pts, y=z_pts)
            if inl is not None and len(inl) == len(pts) and inl.any():
                self._scatter_z_inl.setData(
                    x=d_pts[inl] + d_shift,
                    y=z_pts[inl] + self._dz,
                )
            else:
                self._scatter_z_inl.clear()

            d_min = float(np.min(d_pts)) - 1.0
            d_max = float(np.max(d_pts)) + 1.0
        else:
            d_verts_raw = (verts[:, 0] - cx) * ux + (verts[:, 1] - cy) * uy
            d_min = float(np.min(d_verts_raw)) - 2.0
            d_max = float(np.max(d_verts_raw)) + 2.0
            self._scatter_z_pts.clear()
            self._scatter_z_inl.clear()

        d_grid = np.array([d_min, d_max])

        # Raw cloud plane line (faint dotted) & Shifted cloud plane line (bright solid green)
        if cloud_mean is not None and norm is not None:
            c_cloud_z = float(cloud_mean[2])
            d_cloud_center = (float(cloud_mean[0]) - cx) * ux + (float(cloud_mean[1]) - cy) * uy
            z_cloud_line = c_cloud_z - slope * (d_grid - d_cloud_center)
            self._line_z_cloud.setData(d_grid, z_cloud_line)

            z_shifted_cloud = (c_cloud_z + self._dz) - slope * (d_grid - (d_cloud_center + d_shift))
            self._line_z_shifted.setData(d_grid, z_shifted_cloud)
        else:
            self._line_z_cloud.clear()
            self._line_z_shifted.clear()

        # Fixed surveyor GCP target line and corner vertices
        d_verts_raw = (verts[:, 0] - cx) * ux + (verts[:, 1] - cy) * uy
        self._scatter_z_verts.setData(x=d_verts_raw, y=verts[:, 2])
        z_orig_line = cz - slope * d_grid
        self._line_z_orig.setData(d_grid, z_orig_line)

    def _update_metrics(self):
        if not self._current_patch:
            self._metrics_label.setText("<i>No patch loaded</i>")
            return

        mag = float(np.sqrt(self._dx**2 + self._dy**2 + self._dz**2))
        norm = self._current_patch.get("cloud_normal")
        rmse = self._current_patch.get("plane_rmse")
        n_nearby = self._current_patch.get("n_nearby", 0)
        inl = self._current_patch.get("inlier_mask")
        n_inl = int(inl.sum()) if inl is not None else 0

        slope_str = "N/A"
        if norm is not None:
            nh = float(np.hypot(norm[0], norm[1]))
            nz = max(abs(float(norm[2])), 1e-6)
            dip_deg = float(np.degrees(np.arctan(nh / nz)))
            slope_str = f"{dip_deg:.1f}°"

        rmse_str = f"{rmse*100:.1f} cm" if rmse is not None else "N/A"

        self._metrics_label.setText(
            f"<b>Tile Point Cloud Shift:</b> &nbsp; "
            f"ΔX = <b>{self._dx:+.3f}</b> m, &nbsp; "
            f"ΔY = <b>{self._dy:+.3f}</b> m, &nbsp; "
            f"ΔZ = <b>{self._dz:+.3f}</b> m &nbsp; "
            f"(<b>|Shift| = {mag:.3f} m</b>)<br>"
            f"<b>Surface Metrics:</b> &nbsp; "
            f"Slope / Dip = {slope_str} &nbsp; | &nbsp; "
            f"Plane RMSE = {rmse_str} &nbsp; | &nbsp; "
            f"Inliers = {n_inl}/{n_nearby}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════════════

class _GroundControlWorker(QThread):
    """Runs heavy spatial queries off the GUI thread."""

    progress = Signal(str, float)
    finished_gcp = Signal(list)    # list of per-point result dicts
    finished_roofs = Signal(list)  # list of per-surface result dicts
    error = Signal(str)

    def __init__(self, data: dict, task: str, params: dict, parent=None):
        """
        Args:
            data: Full tile point data dict (x, y, z, classification, …).
            task: ``"gcp"`` or ``"roofs"``.
            params: Task-specific parameters.
        """
        super().__init__(parent)
        self._data = data
        self._task = task
        self._params = params

    def run(self):
        try:
            if self._task == "gcp":
                self._run_gcp()
            elif self._task == "roofs":
                self._run_roofs()
        except Exception as exc:
            logger.exception("Ground control worker failed")
            self.error.emit(str(exc))

    # ── GCP ──────────────────────────────────────────────────────────

    def _run_gcp(self):
        points = self._params["points"]        # list of (name, x, y, z)
        classes = self._params["classes"]       # set of int class codes
        radius = self._params.get("radius", 5.0)

        xs = self._data["x"]
        ys = self._data["y"]
        zs = self._data["z"]
        cls = self._data.get("classification")

        n_pts = len(xs)
        n_gcp = len(points)
        results = []

        # Build a mask of selected classes
        if cls is not None and classes:
            class_mask = np.isin(cls, list(classes))
        else:
            class_mask = np.ones(n_pts, dtype=bool)

        if not class_mask.any():
            checked = ", ".join(str(c) for c in sorted(classes))
            for name, gx, gy, gz in points:
                results.append({
                    "name": name, "x": gx, "y": gy, "z_in": gz,
                    "z_cloud": None, "dz": None,
                    "n_nearby": 0,
                    "warning": f"No points of classes {checked} in loaded tiles",
                })
            self.finished_gcp.emit(results)
            return

        # Filter to selected classes
        cx = xs[class_mask]
        cy = ys[class_mask]
        cz = zs[class_mask]

        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack((cx, cy)))

        for i, (name, gx, gy, gz) in enumerate(points):
            pct = (i + 1) / n_gcp * 100
            self.progress.emit(f"Processing GCP {name}…", pct)

            # Find nearby points within radius
            indices = tree.query_ball_point([gx, gy], radius)
            n_nearby = len(indices)

            if n_nearby == 0:
                results.append({
                    "name": name, "x": gx, "y": gy, "z_in": gz,
                    "z_cloud": None, "dz": None,
                    "n_nearby": 0,
                    "warning": f"No points within {radius:.1f} m",
                })
                continue

            nearby_xyz = np.column_stack((cx[indices], cy[indices], cz[indices]))
            z_cloud, dz, z_std = self._compare_gcp_to_surface(gx, gy, gz, nearby_xyz)

            results.append({
                "name": name, "x": gx, "y": gy, "z_in": gz,
                "z_cloud": z_cloud, "dz": dz,
                "n_nearby": n_nearby,
                "z_std": z_std,
            })

        self.finished_gcp.emit(results)

    # ── Roofs ────────────────────────────────────────────────────────

    def _run_roofs(self):
        surfaces = self._params["surfaces"]  # list of (name, verts) or (name, verts, type)
        radius = self._params.get("radius", 10.0)

        xs = self._data["x"]
        ys = self._data["y"]
        zs = self._data["z"]
        cls = self._data.get("classification")

        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack((xs, ys)))

        n_surf = len(surfaces)
        results = []

        for i, surf_item in enumerate(surfaces):
            pct = (i + 1) / n_surf * 100
            if len(surf_item) == 3:
                name, verts, ptype = surf_item
            else:
                name, verts = surf_item
                ptype = "roof"

            self.progress.emit(f"Processing surface {name} ({ptype})…", pct)

            verts_arr = np.array(verts, dtype=np.float64)
            centroid = verts_arr.mean(axis=0)

            # Find nearby points within search radius
            indices = tree.query_ball_point(centroid[:2], radius)
            if len(indices) < 3:
                results.append({
                    "name": name, "type": ptype, "n_verts": len(verts),
                    "verts": verts_arr, "centroid": centroid,
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                    "n_nearby": len(indices),
                    "dx": None, "dy": None, "dz": None, "shift_mag": None,
                    "warning": f"Not enough nearby points ({len(indices)})",
                })
                continue

            nearby_idx = np.asarray(indices, dtype=np.int64)
            nearby_xyz = np.column_stack((xs[nearby_idx], ys[nearby_idx], zs[nearby_idx]))

            # Class-specific candidate filtering
            cand_pts = nearby_xyz
            if cls is not None and len(cls) == len(xs):
                nearby_cls = cls[nearby_idx]
                if ptype == "ground":
                    # Filter for Ground (class 2) or Water bottom (class 9)
                    gnd_mask = np.isin(nearby_cls, [2, 9])
                    if gnd_mask.sum() >= 4:
                        cand_pts = nearby_xyz[gnd_mask]
                else:
                    # Roof: prioritize Building (class 6)
                    bldg_mask = (nearby_cls == 6)
                    if bldg_mask.sum() >= 4:
                        cand_pts = nearby_xyz[bldg_mask]
                    else:
                        # Exclude ground returns and filter to elevations near centroid
                        not_gnd = (nearby_cls != 2)
                        elev_mask = np.abs(nearby_xyz[:, 2] - centroid[2]) <= 2.5
                        if (not_gnd & elev_mask).sum() >= 4:
                            cand_pts = nearby_xyz[not_gnd & elev_mask]
                        elif not_gnd.sum() >= 4:
                            cand_pts = nearby_xyz[not_gnd]
                        elif elev_mask.sum() >= 4:
                            cand_pts = nearby_xyz[elev_mask]
            else:
                # Unclassified: filter by elevation around roof/surface centroid
                elev_mask = np.abs(nearby_xyz[:, 2] - centroid[2]) <= 2.5
                if elev_mask.sum() >= 4:
                    cand_pts = nearby_xyz[elev_mask]

            # Fit plane to candidate points via RANSAC
            cloud_normal, cloud_mean, inlier_mask, rmse = self._fit_ransac_plane_detailed(cand_pts)

            if cloud_normal is not None:
                # Signed distances from cloud inliers to cloud plane
                d_cloud = np.dot(cand_pts[inlier_mask] - cloud_mean, cloud_normal)
                cloud_median = float(np.median(d_cloud))

                # Signed distances from input vertices to cloud plane
                d_input = np.dot(verts_arr - cloud_mean, cloud_normal)
                input_median = float(np.median(d_input))

                # Scalar offset: positive = input (GCP) is above cloud along normal
                # Shift vector to add to point cloud to align point cloud with GCP
                offset = input_median - cloud_median
                shift_vec = cloud_normal * offset
                dx, dy, dz = float(shift_vec[0]), float(shift_vec[1]), float(shift_vec[2])
            else:
                # Fallback: simple Z difference
                cloud_z_median = float(np.median(cand_pts[:, 2])) if len(cand_pts) else float(centroid[2])
                input_z_mean = float(verts_arr[:, 2].mean())
                dx, dy = 0.0, 0.0
                dz = input_z_mean - cloud_z_median
                cloud_mean = cand_pts.mean(axis=0) if len(cand_pts) else centroid
                cloud_normal = np.array([0.0, 0.0, 1.0])
                inlier_mask = np.ones(len(cand_pts), dtype=bool)
                rmse = 0.0

            # Subsample nearby points for responsive UI rendering
            if len(cand_pts) > 1500:
                sub_idx = np.random.choice(len(cand_pts), 1500, replace=False)
                ui_pts = cand_pts[sub_idx]
                ui_inl = inlier_mask[sub_idx] if inlier_mask is not None else None
            else:
                ui_pts = cand_pts
                ui_inl = inlier_mask

            results.append({
                "name": name,
                "type": ptype,
                "n_verts": len(verts),
                "verts": verts_arr,
                "n_nearby": len(cand_pts),
                "dx": dx, "dy": dy, "dz": dz,
                "base_dx": dx, "base_dy": dy, "base_dz": dz,
                "shift_mag": float(np.sqrt(dx*dx + dy*dy + dz*dz)) if dx is not None else None,
                "centroid": centroid,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "cloud_normal": cloud_normal,
                "cloud_mean": cloud_mean,
                "plane_rmse": rmse,
                "local_pts": ui_pts,
                "inlier_mask": ui_inl,
            })

        self.finished_roofs.emit(results)

    @staticmethod
    def _fit_plane_normal(verts: np.ndarray) -> np.ndarray:
        """Fit a plane to *verts* (N×3) via SVD, return unit normal."""
        centroid = verts.mean(axis=0)
        _, _, vh = np.linalg.svd(verts - centroid)
        normal = vh[2]
        if normal[2] < 0:
            normal = -normal
        return normal

    @staticmethod
    def _fit_ransac_plane(xyz: np.ndarray, n_iter: int = 200,
                          threshold: float = 0.3) -> Tuple[Optional[np.ndarray], float]:
        """RANSAC plane fit. Returns (normal, centroid_z) or (None, median_z)."""
        norm, mean, _, _ = _GroundControlWorker._fit_ransac_plane_detailed(xyz, n_iter, threshold)
        if norm is not None:
            return norm, float(mean[2])
        return None, float(np.median(xyz[:, 2])) if len(xyz) else 0.0

    @staticmethod
    def _fit_ransac_plane_detailed(xyz: np.ndarray, n_iter: int = 200,
                                  threshold: float = 0.20) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], float]:
        """
        Robust RANSAC plane fit.
        Returns: (unit_normal with nz >= 0, cloud_mean, inlier_mask, rmse)
        """
        if len(xyz) < 3:
            mean = xyz.mean(axis=0) if len(xyz) > 0 else np.zeros(3)
            return None, mean, None, 0.0

        best_inliers = 0
        best_normal = None
        best_mean = xyz.mean(axis=0)

        for _ in range(n_iter):
            idx = np.random.choice(len(xyz), 3, replace=False)
            sample = xyz[idx]
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            nrm = float(np.linalg.norm(normal))
            if nrm < 1e-10:
                continue
            normal /= nrm
            if normal[2] < 0:
                normal = -normal

            centroid = sample.mean(axis=0)
            dists = np.abs(np.dot(xyz - centroid, normal))
            inliers = int((dists < threshold).sum())

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_mean = centroid

        if best_normal is not None:
            dists = np.abs(np.dot(xyz - best_mean, best_normal))
            inlier_mask = dists < threshold
            if inlier_mask.sum() >= 3:
                pts_inl = xyz[inlier_mask]
                best_mean = pts_inl.mean(axis=0)
                _, _, vh = np.linalg.svd(pts_inl - best_mean)
                best_normal = vh[2]
                if best_normal[2] < 0:
                    best_normal = -best_normal
                d_inl = np.dot(pts_inl - best_mean, best_normal)
                rmse = float(np.sqrt(np.mean(d_inl**2)))
                return best_normal, best_mean, inlier_mask, rmse

        return best_normal, best_mean, np.ones(len(xyz), dtype=bool), 0.0

    @staticmethod
    def _compare_gcp_to_surface(gx: float, gy: float, gz: float,
                                nearby_xyz: np.ndarray) -> Tuple[float, float, float]:
        """
        Compare a GCP against a robust local point-cloud surface.

        Best practice (ASPRS/USGS vertical checkpoints):
        1. Prioritize immediate neighborhood points (radius <= 1.5 m, or closest 20-50 pts).
        2. Fit an inverse-distance-weighted (IDW) local plane centered at (gx, gy):
           z(x, y) = z0 + a*(x - gx) + b*(y - gy)
           where weights w_i = 1 / (d_i + 0.05)^2.
        3. Perform robust outlier rejection on plane residuals (sigma clipping) to reject
           vegetation / noise points while strictly preserving the true terrain slope.
        4. Evaluate elevation at (gx, gy) -> exactly z0, eliminating slope-induced bias.
        5. Return (z_surface, dz, z_std) with dz = gz - z_surface.
        """
        if len(nearby_xyz) < 3:
            z_med = float(np.median(nearby_xyz[:, 2])) if len(nearby_xyz) else float("nan")
            return z_med, gz - z_med, float(np.std(nearby_xyz[:, 2])) if len(nearby_xyz) else 0.0

        dx = nearby_xyz[:, 0] - gx
        dy = nearby_xyz[:, 1] - gy
        dists = np.sqrt(dx * dx + dy * dy)

        # Prioritize points close to the GCP location (radius <= 1.5 m)
        # to capture the immediate target vicinity and avoid macro-topography distortion.
        close_mask = dists <= 1.5
        if close_mask.sum() >= 6:
            pts = nearby_xyz[close_mask]
            p_dx = dx[close_mask]
            p_dy = dy[close_mask]
            p_dists = dists[close_mask]
        else:
            # Fallback for sparse areas: take the closest 20 to 50 points
            sort_idx = np.argsort(dists)
            take = min(len(dists), max(10, min(50, len(dists))))
            idx = sort_idx[:take]
            pts = nearby_xyz[idx]
            p_dx = dx[idx]
            p_dy = dy[idx]
            p_dists = dists[idx]

        # Weights: inverse square distance with a small floor (5 cm)
        weights = 1.0 / (p_dists + 0.05) ** 2
        weights /= weights.sum()

        # Design matrix for local plane centered at (gx, gy):
        # z = z0 + a*dx + b*dy
        A = np.column_stack((np.ones_like(p_dx), p_dx, p_dy))
        Aw = A * np.sqrt(weights[:, None])
        zw = pts[:, 2] * np.sqrt(weights)

        try:
            beta, _, _, _ = np.linalg.lstsq(Aw, zw, rcond=None)
            residuals = pts[:, 2] - (beta[0] + beta[1] * p_dx + beta[2] * p_dy)

            # Robust residual clipping (reject vegetation / noise around the plane)
            res_med = float(np.median(residuals))
            res_mad = float(np.median(np.abs(residuals - res_med)))
            sigma_res = 1.4826 * res_mad
            inlier_mask = np.abs(residuals - res_med) <= max(2.5 * sigma_res, 0.08)

            if inlier_mask.sum() >= 3 and inlier_mask.sum() < len(pts):
                Aw_inl = Aw[inlier_mask]
                zw_inl = zw[inlier_mask]
                beta, _, _, _ = np.linalg.lstsq(Aw_inl, zw_inl, rcond=None)
                residuals = pts[inlier_mask, 2] - (beta[0] + beta[1] * p_dx[inlier_mask] + beta[2] * p_dy[inlier_mask])
                z_surf = float(beta[0])
                std_res = float(np.std(residuals))
            else:
                z_surf = float(beta[0])
                std_res = float(np.std(residuals))

            # Bound fitted surface within observed inlier Z range to avoid wild extrapolation
            z_surf = float(np.clip(z_surf, pts[:, 2].min(), pts[:, 2].max()))
        except Exception:
            z_surf = float(np.average(pts[:, 2], weights=weights))
            std_res = float(np.std(pts[:, 2]))

        return z_surf, gz - z_surf, std_res


# ═══════════════════════════════════════════════════════════════════════
# Tile Selection Dialog
# ═══════════════════════════════════════════════════════════════════════

class _TileSelectionDialog(QDialog):
    """Modal dialog for interactively selecting target tiles for shift application."""

    def __init__(self, all_tile_ids: List[str], selected_tile_ids: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Tiles for Ground Control Shift")
        self.setMinimumWidth(450)
        self.setMinimumHeight(420)
        self.resize(500, 500)

        self._all_tile_ids = list(all_tile_ids)
        self._initial_selected = set(selected_tile_ids)

        layout = QVBoxLayout(self)

        info_lbl = QLabel(
            "<b>Select the tiles to which the ground control shift will be applied:</b>"
        )
        layout.addWidget(info_lbl)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter tile names…")
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_edit)
        layout.addLayout(filter_row)

        self._list_widget = QListWidget()
        for tid in self._all_tile_ids:
            item = QListWidgetItem(tid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if tid in self._initial_selected else Qt.Unchecked)
            self._list_widget.addItem(item)
        self._list_widget.itemChanged.connect(self._update_count)
        layout.addWidget(self._list_widget, 1)

        act_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        act_row.addWidget(btn_all)

        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(self._deselect_all)
        act_row.addWidget(btn_none)

        self._count_lbl = QLabel()
        act_row.addStretch(1)
        act_row.addWidget(self._count_lbl)
        layout.addLayout(act_row)

        self._update_count()

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def _apply_filter(self, text: str):
        query = text.strip().lower()
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            item.setHidden(query not in item.text().lower() if query else False)

    def _select_all(self):
        self._list_widget.blockSignals(True)
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)
        self._list_widget.blockSignals(False)
        self._update_count()

    def _deselect_all(self):
        self._list_widget.blockSignals(True)
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked)
        self._list_widget.blockSignals(False)
        self._update_count()

    def _update_count(self):
        checked = sum(
            1 for i in range(self._list_widget.count())
            if self._list_widget.item(i).checkState() == Qt.Checked
        )
        total = self._list_widget.count()
        self._count_lbl.setText(f"Selected: <b>{checked}</b> / {total} tiles")

    def get_selected_tile_ids(self) -> List[str]:
        chosen = []
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.checkState() == Qt.Checked:
                chosen.append(item.text())
        return chosen


# ═══════════════════════════════════════════════════════════════════════
# Main dialog
# ═══════════════════════════════════════════════════════════════════════

class GroundControlDialog(QDialog):
    """
    Dialog for Ground Control Point (GCP) and Roof/Surface elevation validation.

    Signals
    -------
    shift_applied(float, float, float, list):
        Emitted when the user clicks 'Apply Shift'. Args: (dx, dy, dz, tile_ids).
        Overloaded with (float, float, float) for backward-compatibility.
    visualize_point(float, float, float, object, str):
        Emitted when user navigates to a GCP or surface.
        Args: (x, y, z_gcp, z_cloud, label).
    """

    shift_applied = Signal((float, float, float, list), (float, float, float))
    visualize_point = Signal(float, float, float, object, str)

    SEPARATORS = {
        "Comma (,)": ",",
        "Semicolon (;)": ";",
        "Tab": "\t",
        "Space": " ",
    }

    def __init__(self, tile_data: dict, parent=None, data_epsg: Optional[int] = None,
                 tile_ids: Optional[list] = None,
                 current_tile_id: Optional[str] = None,
                 tile_manager=None, database=None):
        super().__init__(parent)
        self._data = tile_data
        self._data_epsg = data_epsg  # EPSG code of the point cloud data
        self._worker: Optional[_GroundControlWorker] = None

        # Multi-tile support
        self._selected_tile_ids: list = list(tile_ids) if tile_ids else []
        self._tile_ids: list = self._selected_tile_ids  # backward-compatibility
        self._current_tile_id: Optional[str] = current_tile_id
        self._tm = tile_manager       # TileManager for loading tile data
        self._db = database           # Database for bbox queries

        # Discover all project tiles
        self._all_tile_ids: list = []
        if self._tm:
            try:
                self._all_tile_ids = list(self._tm.tile_ids())
            except Exception:
                pass
        if not self._all_tile_ids and self._db:
            try:
                self._all_tile_ids = [t["id"] for t in self._db.get_all_tiles() if t.get("id")]
            except Exception:
                pass
        if not self._all_tile_ids:
            if self._selected_tile_ids:
                self._all_tile_ids = list(self._selected_tile_ids)
            elif self._current_tile_id:
                self._all_tile_ids = [self._current_tile_id]

        self._custom_tile_ids: list = list(self._selected_tile_ids or self._all_tile_ids)

        # GCP state
        self._gcp_points: List[Tuple[str, float, float, float]] = []
        self._gcp_data_lines: List[List[str]] = []
        self._gcp_results: list = []
        self._gcp_shift: Optional[float] = None
        self._gcp_source_epsg: Optional[int] = None  # EPSG of the GCP CSV

        # Roof state
        self._roof_raw_records: list = []
        self._roof_data_lines: list = []
        self._roof_surfaces: List[Tuple[str, List[Tuple[float, float, float]], str]] = []
        self._roof_results: list = []
        self._roof_shift: Optional[Tuple[float, float, float]] = None

        # Visual check state
        self._vis_mode = "gcp"    # "gcp" or "roofs"
        self._vis_index: int = 0

        self.setWindowTitle("Ground Control")
        self.setMinimumWidth(1100)
        self.setMinimumHeight(750)
        self.resize(1200, 800)
        self._setup_ui()

        # Show tile info if multi-tile mode
        if self._tile_ids:
            self._status.setText(f"Selected tiles: {len(self._tile_ids)}")

    # ── UI construction ──────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_gcp_tab(), "Ground Control Points")
        tabs.addTab(self._build_roofs_tab(), "House Roofs & Areas")
        tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(tabs)

        # ── Visual check section ──
        vis_group = QGroupBox("Visual Check")
        vis_layout = QVBoxLayout(vis_group)

        nav_row = QHBoxLayout()
        self._vis_prev_btn = QPushButton("◀ Prev")
        self._vis_prev_btn.clicked.connect(self._on_vis_prev)
        nav_row.addWidget(self._vis_prev_btn)

        self._vis_label = QLabel("No results yet")
        self._vis_label.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self._vis_label, 1)

        self._vis_next_btn = QPushButton("Next ▶")
        self._vis_next_btn.clicked.connect(self._on_vis_next)
        nav_row.addWidget(self._vis_next_btn)

        vis_layout.addLayout(nav_row)

        action_row = QHBoxLayout()
        self._vis_goto_btn = QPushButton("🔍 Go To in 3-D View")
        self._vis_goto_btn.clicked.connect(self._on_vis_goto)
        action_row.addWidget(self._vis_goto_btn, 1)

        self._vis_use_chk = QCheckBox("Use point in shift calculation")
        self._vis_use_chk.setChecked(True)
        self._vis_use_chk.setToolTip("Include or exclude this point from the elevation shift and statistics")
        self._vis_use_chk.toggled.connect(self._on_vis_use_toggled)
        action_row.addWidget(self._vis_use_chk)

        vis_layout.addLayout(action_row)

        layout.addWidget(vis_group)

        # ── Progress ──
        self._status = QLabel("Ready — load a CSV to begin")
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # ── Bottom buttons ──
        btn_layout = QHBoxLayout()
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout)

        self._update_vis_nav()
        self._populate_scope_combos()

    # ═══════════════════════════════════════════════════════════════════
    # Tab 1 — Ground Control Points
    # ═══════════════════════════════════════════════════════════════════

    def _build_gcp_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # ── CSV import ──
        csv_group = QGroupBox("1. Import CSV")
        cf = QFormLayout(csv_group)

        file_row = QHBoxLayout()
        self._gcp_csv_edit = QLineEdit()
        self._gcp_csv_edit.setPlaceholderText("Select CSV file with name, X, Y, Z columns…")
        self._gcp_csv_edit.setReadOnly(True)
        file_row.addWidget(self._gcp_csv_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_gcp_csv)
        file_row.addWidget(browse_btn)
        cf.addRow("CSV File:", file_row)

        sep_row = QHBoxLayout()
        self._gcp_sep_combo = QComboBox()
        for label in self.SEPARATORS:
            self._gcp_sep_combo.addItem(label)
        self._gcp_sep_combo.setCurrentText("Comma (,)")
        sep_row.addWidget(QLabel("Separator:"))
        sep_row.addWidget(self._gcp_sep_combo)
        self._gcp_custom_sep = QLineEdit()
        self._gcp_custom_sep.setPlaceholderText("custom")
        self._gcp_custom_sep.setMaximumWidth(60)
        self._gcp_custom_sep.setVisible(False)
        sep_row.addWidget(self._gcp_custom_sep)
        self._gcp_sep_combo.currentTextChanged.connect(self._on_gcp_sep_changed)
        sep_row.addStretch()
        cf.addRow("", sep_row)

        self._gcp_has_header = QCheckBox("First row contains column names")
        self._gcp_has_header.setChecked(True)
        self._gcp_has_header.toggled.connect(self._on_gcp_has_header_toggled)
        cf.addRow("", self._gcp_has_header)

        self._gcp_preview = QTextEdit()
        self._gcp_preview.setReadOnly(True)
        self._gcp_preview.setMaximumHeight(80)
        self._gcp_preview.setPlaceholderText("CSV preview…")
        cf.addRow("Preview:", self._gcp_preview)

        layout.addWidget(csv_group)

        # ── Coordinate System & Column Mapping ──
        map_group = QGroupBox("2. Coordinate System & Column Mapping")
        mf = QVBoxLayout(map_group)

        crs_form = QFormLayout()
        crs_row = QHBoxLayout()
        crs_row.addWidget(QLabel("LiDAR Data EPSG:"))
        self._data_epsg_spin = QSpinBox()
        self._data_epsg_spin.setRange(0, 99999)
        self._data_epsg_spin.setSpecialValueText("Unknown (0)")
        if self._data_epsg:
            self._data_epsg_spin.setValue(self._data_epsg)
        self._data_epsg_spin.setToolTip(
            "EPSG code of the LiDAR point cloud. If unknown, specify the EPSG here (e.g. 25832, 31256, 32633)."
        )
        self._data_epsg_spin.valueChanged.connect(self._on_data_epsg_changed)
        crs_row.addWidget(self._data_epsg_spin)

        crs_row.addSpacing(15)
        crs_row.addWidget(QLabel("GCP CSV EPSG:"))
        self._gcp_epsg_spin = QSpinBox()
        self._gcp_epsg_spin.setRange(0, 99999)
        self._gcp_epsg_spin.setSpecialValueText("Auto (same as data)")
        self._gcp_epsg_spin.setValue(0)
        self._gcp_epsg_spin.setToolTip(
            "EPSG code of the GCP coordinates. Set to 0 if the CSV already uses the same CRS as the LiDAR data. "
            "If different (e.g. 4326 for WGS84 GPS coords, or 31256 for Austrian GK), set it here to auto-transform."
        )
        self._gcp_epsg_spin.valueChanged.connect(self._on_gcp_epsg_changed)
        crs_row.addWidget(self._gcp_epsg_spin)
        crs_row.addStretch()
        crs_form.addRow("CRS / EPSG:", crs_row)

        self._gcp_crs_info = QLabel("")
        self._gcp_crs_info.setWordWrap(True)
        crs_form.addRow("", self._gcp_crs_info)
        mf.addLayout(crs_form)

        col_row = QHBoxLayout()
        self._gcp_name_col = QComboBox()
        self._gcp_name_col.setMinimumWidth(80)
        col_row.addWidget(QLabel("Name:"))
        col_row.addWidget(self._gcp_name_col)
        self._gcp_x_col = QComboBox()
        col_row.addWidget(QLabel("X:"))
        col_row.addWidget(self._gcp_x_col)
        self._gcp_y_col = QComboBox()
        col_row.addWidget(QLabel("Y:"))
        col_row.addWidget(self._gcp_y_col)
        self._gcp_z_col = QComboBox()
        col_row.addWidget(QLabel("Z:"))
        col_row.addWidget(self._gcp_z_col)

        self._gcp_swap_xy_btn = QPushButton("⇄ Swap X & Y")
        self._gcp_swap_xy_btn.setToolTip("Swap X and Y columns (Easting/Northing)")
        self._gcp_swap_xy_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 3px 8px; }")
        self._gcp_swap_xy_btn.clicked.connect(self._on_gcp_swap_xy)
        col_row.addWidget(self._gcp_swap_xy_btn)
        col_row.addStretch()
        mf.addLayout(col_row)

        # Connect combo signals
        self._gcp_name_col.currentIndexChanged.connect(self._on_gcp_columns_changed)
        self._gcp_x_col.currentIndexChanged.connect(self._on_gcp_columns_changed)
        self._gcp_y_col.currentIndexChanged.connect(self._on_gcp_columns_changed)
        self._gcp_z_col.currentIndexChanged.connect(self._on_gcp_columns_changed)

        # Diagnostics & Extent feedback box
        diag_box = QGroupBox("Coordinate & Extent Diagnostics")
        diag_layout = QVBoxLayout(diag_box)
        self._gcp_diag_label = QLabel("<i>Load a CSV to view coordinate alignment diagnostics.</i>")
        self._gcp_diag_label.setWordWrap(True)
        diag_layout.addWidget(self._gcp_diag_label)
        mf.addWidget(diag_box)

        layout.addWidget(map_group)

        # ── Class selection ──
        cls_group = QGroupBox("3. Target Classes (point cloud)")
        cls_layout = QVBoxLayout(cls_group)
        self._gcp_class_list = QListWidget()
        self._gcp_class_list.setMaximumHeight(150)
        for code in sorted(ASPRS_CLASS_NAMES.keys()):
            name = ASPRS_CLASS_NAMES[code]
            item = QListWidgetItem(f"{code:2d}: {name}")
            item.setData(Qt.UserRole, code)
            item.setCheckState(Qt.Unchecked)
            self._gcp_class_list.addItem(item)
        # Default: check Ground (2), Unclassified (1), Created (0), Building (6)
        for i in range(self._gcp_class_list.count()):
            code = self._gcp_class_list.item(i).data(Qt.UserRole)
            if code in (0, 1, 2, 6):
                self._gcp_class_list.item(i).setCheckState(Qt.Checked)
        cls_layout.addWidget(self._gcp_class_list)
        cls_layout.addWidget(QLabel(
            "<i>Only points matching checked classes are compared against GCPs. "
            "If your data is unclassified, check class 0 or 1.</i>"
        ))
        layout.addWidget(cls_group)

        # ── Parameters ──
        param_group = QGroupBox("4. Parameters")
        pf = QFormLayout(param_group)
        self._gcp_radius_spin = QDoubleSpinBox()
        self._gcp_radius_spin.setRange(0.1, 200.0)
        self._gcp_radius_spin.setDecimals(1)
        self._gcp_radius_spin.setValue(5.0)
        self._gcp_radius_spin.setSuffix(" m")
        self._gcp_radius_spin.setToolTip("Search radius around each GCP")
        pf.addRow("Search Radius:", self._gcp_radius_spin)
        layout.addWidget(param_group)

        # ── Run button ──
        run_row = QHBoxLayout()
        self._gcp_run_btn = QPushButton("▶ Calculate Elevation Differences")
        self._gcp_run_btn.clicked.connect(self._on_run_gcp)
        self._gcp_run_btn.setEnabled(False)
        run_row.addWidget(self._gcp_run_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        # ── Results table ──
        res_group = QGroupBox("5. Results")
        rl = QVBoxLayout(res_group)

        # Quick filter & outlier toolbar
        filter_row = QHBoxLayout()
        self._gcp_select_all_btn = QPushButton("Select All")
        self._gcp_select_all_btn.clicked.connect(self._on_gcp_select_all)
        filter_row.addWidget(self._gcp_select_all_btn)

        self._gcp_deselect_all_btn = QPushButton("Deselect All")
        self._gcp_deselect_all_btn.clicked.connect(self._on_gcp_deselect_all)
        filter_row.addWidget(self._gcp_deselect_all_btn)

        filter_row.addSpacing(15)
        self._gcp_outlier_spin = QDoubleSpinBox()
        self._gcp_outlier_spin.setRange(0.01, 50.0)
        self._gcp_outlier_spin.setDecimals(2)
        self._gcp_outlier_spin.setValue(0.50)
        self._gcp_outlier_spin.setSuffix(" m")
        self._gcp_outlier_spin.setToolTip("Elevation difference threshold for outlier filtering")

        self._gcp_outlier_btn = QPushButton("Disable Outliers (|ΔZ| >)")
        self._gcp_outlier_btn.setToolTip("Uncheck all GCP points whose absolute elevation difference |ΔZ| exceeds threshold")
        self._gcp_outlier_btn.clicked.connect(self._on_gcp_disable_outliers)
        filter_row.addWidget(self._gcp_outlier_btn)
        filter_row.addWidget(self._gcp_outlier_spin)
        filter_row.addStretch()
        rl.addLayout(filter_row)

        self._gcp_table = QTableWidget(0, 8)
        self._gcp_table.setHorizontalHeaderLabels(
            ["Use", "Name", "X", "Y", "Z Input", "Z Cloud", "ΔZ", "Nearby pts"]
        )
        self._gcp_table.horizontalHeader().setStretchLastSection(True)
        self._gcp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._gcp_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._gcp_table.setSortingEnabled(True)
        self._gcp_table.itemChanged.connect(self._on_gcp_table_item_changed)
        rl.addWidget(self._gcp_table)

        self._gcp_stats = QLabel("")
        self._gcp_stats.setWordWrap(True)
        rl.addWidget(self._gcp_stats)

        layout.addWidget(res_group)

        # ── Apply shift ──
        layout.addWidget(self._create_scope_selector_widget("gcp"))
        self._gcp_apply_btn = QPushButton("⬆ Apply Z Shift to Current Tile")
        self._gcp_apply_btn.setEnabled(False)
        self._gcp_apply_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 14px; }"
        )
        self._gcp_apply_btn.clicked.connect(self._on_apply_gcp)
        layout.addWidget(self._gcp_apply_btn)

        layout.addStretch()
        return w

    # ═══════════════════════════════════════════════════════════════════
    # Tab 2 — House Roofs & Areas
    # ═══════════════════════════════════════════════════════════════════

    def _build_roofs_tab(self) -> QWidget:
        w = QWidget()
        main_layout = QVBoxLayout(w)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # ── LEFT PANE: Import, Options, Tables, Run ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(6)

        # 1. CSV import
        csv_group = QGroupBox("1. Import Points / Surfaces CSV")
        cf = QFormLayout(csv_group)

        info = QLabel(
            "Load surveyor CSV points (e.g. 010101, 01_01_01, or delimited). "
            "Patches are automatically detected and grouped for roofs and ground surfaces."
        )
        info.setWordWrap(True)
        cf.addRow(info)

        file_row = QHBoxLayout()
        self._roof_csv_edit = QLineEdit()
        self._roof_csv_edit.setPlaceholderText("Select CSV file with ID, X, Y, Z columns…")
        self._roof_csv_edit.setReadOnly(True)
        file_row.addWidget(self._roof_csv_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_roof_csv)
        file_row.addWidget(browse_btn)
        cf.addRow("CSV File:", file_row)

        sep_row = QHBoxLayout()
        self._roof_sep_combo = QComboBox()
        for label in self.SEPARATORS:
            self._roof_sep_combo.addItem(label)
        self._roof_sep_combo.setCurrentText("Comma (,)")
        sep_row.addWidget(QLabel("Separator:"))
        sep_row.addWidget(self._roof_sep_combo)
        self._roof_custom_sep = QLineEdit()
        self._roof_custom_sep.setPlaceholderText("custom")
        self._roof_custom_sep.setMaximumWidth(60)
        self._roof_custom_sep.setVisible(False)
        sep_row.addWidget(self._roof_custom_sep)
        self._roof_sep_combo.currentTextChanged.connect(self._on_roof_sep_changed)
        sep_row.addStretch()
        cf.addRow("", sep_row)

        self._roof_preview = QTextEdit()
        self._roof_preview.setReadOnly(True)
        self._roof_preview.setMaximumHeight(65)
        self._roof_preview.setPlaceholderText("CSV preview…")
        cf.addRow("Preview:", self._roof_preview)

        # Column mapping row
        col_row = QHBoxLayout()
        self._roof_id_col = QComboBox()
        col_row.addWidget(QLabel("ID:"))
        col_row.addWidget(self._roof_id_col)
        self._roof_x_col = QComboBox()
        col_row.addWidget(QLabel("X:"))
        col_row.addWidget(self._roof_x_col)
        self._roof_y_col = QComboBox()
        col_row.addWidget(QLabel("Y:"))
        col_row.addWidget(self._roof_y_col)
        self._roof_z_col = QComboBox()
        col_row.addWidget(QLabel("Z:"))
        col_row.addWidget(self._roof_z_col)
        col_row.addStretch()
        cf.addRow("Columns:", col_row)

        self._roof_id_col.currentIndexChanged.connect(self._update_roof_records_from_combos)
        self._roof_x_col.currentIndexChanged.connect(self._update_roof_records_from_combos)
        self._roof_y_col.currentIndexChanged.connect(self._update_roof_records_from_combos)
        self._roof_z_col.currentIndexChanged.connect(self._update_roof_records_from_combos)

        left_layout.addWidget(csv_group)

        # 2. Patch Detection & Grouping
        grp_group = QGroupBox("2. Patch Detection & Vertex Sorting")
        gf = QFormLayout(grp_group)

        self._roof_detect_mode = QComboBox()
        self._roof_detect_mode.addItem("Auto (Naming then 10m Proximity)", "auto")
        self._roof_detect_mode.addItem("Naming Convention (010101 / Delimited)", "naming")
        self._roof_detect_mode.addItem("Spatial Proximity (10m Radius)", "spatial")
        self._roof_detect_mode.addItem("CSV Surface ID Column", "csv_id")
        self._roof_detect_mode.currentIndexChanged.connect(self._detect_and_group_roof_points)
        gf.addRow("Group Mode:", self._roof_detect_mode)

        param_row = QHBoxLayout()
        self._roof_radius_spin = QDoubleSpinBox()
        self._roof_radius_spin.setRange(0.5, 200.0)
        self._roof_radius_spin.setDecimals(1)
        self._roof_radius_spin.setValue(10.0)
        self._roof_radius_spin.setSuffix(" m")
        self._roof_radius_spin.setToolTip("Search radius for spatial clustering (10m default) and LiDAR plane fitting")
        self._roof_radius_spin.valueChanged.connect(self._on_roof_radius_changed)
        param_row.addWidget(QLabel("Radius:"))
        param_row.addWidget(self._roof_radius_spin)

        self._roof_sort_chk = QCheckBox("Sort vertices angularly (CW/CCW)")
        self._roof_sort_chk.setChecked(True)
        self._roof_sort_chk.setToolTip("Sorts vertices by polar angle around centroid to eliminate zigzag / crossed hourglass shapes")
        self._roof_sort_chk.toggled.connect(self._rebuild_roof_surfaces_from_table)
        param_row.addWidget(self._roof_sort_chk)
        param_row.addStretch()
        gf.addRow("", param_row)

        left_layout.addWidget(grp_group)

        # 3. Sub-tabs: Points List vs Detected Patches
        subtabs = QTabWidget()
        self._roof_subtabs = subtabs

        # Tab: Points (with editable House / Patch No. and Type)
        pts_tab = QWidget()
        pts_vbox = QVBoxLayout(pts_tab)
        pts_vbox.setContentsMargins(2, 2, 2, 2)
        pts_hint = QLabel("<i>Double-click 'Patch / House No.' to edit grouping, or select Type dropdown.</i>")
        pts_hint.setStyleSheet("color: #888; font-size: 11px;")
        pts_vbox.addWidget(pts_hint)

        self._roof_points_table = QTableWidget(0, 6)
        self._roof_points_table.setHorizontalHeaderLabels(
            ["Point Name", "Patch / House No. ✎", "Type ✎", "X", "Y", "Z"]
        )
        self._roof_points_table.horizontalHeader().setStretchLastSection(True)
        self._roof_points_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._roof_points_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._roof_points_table.itemChanged.connect(self._on_roof_point_item_changed)
        pts_vbox.addWidget(self._roof_points_table)

        pts_btn_row = QHBoxLayout()
        btn_regroup = QPushButton("↻ Re-detect / Reset Groups")
        btn_regroup.clicked.connect(self._detect_and_group_roof_points)
        pts_btn_row.addWidget(btn_regroup)

        self._roof_del_point_btn = QPushButton("🗑 Delete Selected Point(s)")
        self._roof_del_point_btn.setToolTip("Delete selected corner points from the table and regroup patches")
        self._roof_del_point_btn.clicked.connect(self._on_delete_roof_points)
        pts_btn_row.addWidget(self._roof_del_point_btn)
        pts_vbox.addLayout(pts_btn_row)

        subtabs.addTab(pts_tab, "Points (Editable Groups)")

        # Tab: Detected Patches & Offsets
        patches_tab = QWidget()
        pat_vbox = QVBoxLayout(patches_tab)
        pat_vbox.setContentsMargins(2, 2, 2, 2)

        pat_toolbar = QHBoxLayout()
        self._roof_toggle_use_btn = QPushButton("⇄ Toggle Use / Dismiss")
        self._roof_toggle_use_btn.setToolTip("Toggle inclusion of selected patch in mean XYZ shift calculation")
        self._roof_toggle_use_btn.clicked.connect(self._on_toggle_selected_patch_use)
        pat_toolbar.addWidget(self._roof_toggle_use_btn)

        self._roof_del_patch_btn = QPushButton("🗑 Delete Selected Patch")
        self._roof_del_patch_btn.setToolTip("Delete selected patch and its points from the surface model")
        self._roof_del_patch_btn.setStyleSheet("QPushButton { color: #e74c3c; }")
        self._roof_del_patch_btn.clicked.connect(self._on_delete_selected_patch)
        pat_toolbar.addWidget(self._roof_del_patch_btn)
        pat_toolbar.addStretch()
        pat_vbox.addLayout(pat_toolbar)

        self._roof_table = QTableWidget(0, 9)
        self._roof_table.setHorizontalHeaderLabels(
            ["Use", "Patch", "Type", "Vertices", "Nearby", "ΔX (m)", "ΔY (m)", "ΔZ (m)", "|Shift|"]
        )
        self._roof_table.horizontalHeader().setStretchLastSection(True)
        self._roof_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._roof_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._roof_table.itemSelectionChanged.connect(self._on_roof_table_selection_changed)
        self._roof_table.itemChanged.connect(self._on_roof_table_item_changed)
        pat_vbox.addWidget(self._roof_table)

        subtabs.addTab(patches_tab, "Detected Patches & Offsets")
        left_layout.addWidget(subtabs, 1)

        # 4. Run & Apply buttons
        run_box = QVBoxLayout()
        self._roof_run_btn = QPushButton("▶ Calculate Surface Offsets")
        self._roof_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        self._roof_run_btn.clicked.connect(self._on_run_roofs)
        self._roof_run_btn.setEnabled(False)
        run_box.addWidget(self._roof_run_btn)

        self._roof_stats = QLabel("")
        self._roof_stats.setWordWrap(True)
        run_box.addWidget(self._roof_stats)
        run_box.addWidget(self._create_scope_selector_widget("roof"))
        self._roof_apply_btn = QPushButton("⬆ Apply XYZ Shift to Current Tile")
        self._roof_apply_btn.setEnabled(False)
        self._roof_apply_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px 14px; }")
        self._roof_apply_btn.clicked.connect(self._on_apply_roofs)
        run_box.addWidget(self._roof_apply_btn)

        left_layout.addLayout(run_box)
        splitter.addWidget(left_widget)

        # ── RIGHT PANE: 2-View Inspector ──
        self._roof_inspector = _Roof2ViewInspector(self)
        self._roof_inspector.offset_changed.connect(self._on_patch_offset_modified)
        self._roof_inspector.use_toggled.connect(self._on_inspector_use_toggled)
        self._roof_inspector.delete_requested.connect(self.delete_roof_patch)
        splitter.addWidget(self._roof_inspector)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        main_layout.addWidget(splitter)
        return w

    # ── Helper: get active separator ─────────────────────────────────

    def _get_separator(self, combo: QComboBox, custom_edit: QLineEdit) -> str:
        label = combo.currentText()
        if label in self.SEPARATORS:
            return self.SEPARATORS[label]
        return custom_edit.text() or ","

    # ── Parse CSV header → populate column combos ────────────────────

    def _populate_column_combos(self, header: List[str],
                                name_combo: QComboBox,
                                x_combo: QComboBox,
                                y_combo: QComboBox,
                                z_combo: QComboBox):
        for cb in (name_combo, x_combo, y_combo, z_combo):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("(none)", -1)
        for i, col in enumerate(header):
            col_clean = col.strip().strip('"').strip("'")
            name_combo.addItem(col_clean, i)
            x_combo.addItem(col_clean, i)
            y_combo.addItem(col_clean, i)
            z_combo.addItem(col_clean, i)
        # Auto-detect including German/survey aliases
        for i, col in enumerate(header):
            low = col.strip().lower().strip('"').strip("'").replace(" ", "_")
            if low in ("name", "id", "label", "point", "point_name", "pkt", "punkt", "punktnummer", "punkt_nr", "pn", "nr", "station", "target"):
                name_combo.setCurrentIndex(i + 1)
            elif low in ("x", "easting", "east", "lon", "longitude", "rw", "rechts", "rechtswert", "e", "ost", "ostwert"):
                x_combo.setCurrentIndex(i + 1)
            elif low in ("y", "northing", "north", "lat", "latitude", "hw", "hoch", "hochwert", "n", "nord", "nordwert"):
                y_combo.setCurrentIndex(i + 1)
            elif low in ("z", "elev", "elevation", "height", "alt", "altitude", "h", "hoehe", "höhe", "kot", "kote"):
                z_combo.setCurrentIndex(i + 1)
        for cb in (name_combo, x_combo, y_combo, z_combo):
            cb.blockSignals(False)

    # ── Browse GCP CSV ───────────────────────────────────────────────

    def _on_browse_gcp_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GCP CSV", "",
            "CSV Files (*.csv *.txt);;All Files (*)",
        )
        if not path:
            return
        self._gcp_csv_edit.setText(path)
        self._parse_gcp_csv(path)

    def _split_csv_line(self, line: str, sep: str) -> List[str]:
        """Split a CSV line by separator, collapsing multiple spaces."""
        if sep == " ":
            # Treat any whitespace as a single separator
            return line.split()
        # Use csv module for proper quote handling with other separators
        reader = csv.reader([line], delimiter=sep)
        try:
            return next(reader)
        except StopIteration:
            return []

    def _line_looks_like_data(self, fields: List[str]) -> bool:
        """Check if a parsed line looks like data (mostly numeric) rather than a header."""
        if not fields:
            return False
        numeric_count = 0
        for f in fields:
            try:
                _safe_float(f)
                numeric_count += 1
            except ValueError:
                pass
        # If >= 75% of fields are numeric, treat as data row
        return numeric_count >= max(2, len(fields) * 0.75)

    def _parse_gcp_csv(self, path: str):
        sep = self._get_separator(self._gcp_sep_combo, self._gcp_custom_sep)
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception as exc:
            self._gcp_preview.setText(f"Error reading file: {exc}")
            return

        # Preview
        raw_lines = content.splitlines()
        preview = "\n".join(raw_lines[:10])
        if len(raw_lines) > 10:
            preview += f"\n… ({len(raw_lines)} total lines)"
        self._gcp_preview.setText(preview)

        # Parse lines into fields (handle spaces properly)
        parsed_lines = []
        for line in raw_lines:
            if line.strip() == "":
                continue
            fields = self._split_csv_line(line, sep)
            if fields:
                parsed_lines.append(fields)

        if not parsed_lines:
            self._gcp_preview.setText("Empty file")
            return

        # ── Auto-detect header vs data ──
        first_line_is_data = self._line_looks_like_data(parsed_lines[0])
        if first_line_is_data and self._gcp_has_header.isChecked():
            self._gcp_has_header.setChecked(False)
            self._status.setText(
                "Auto-detected: no header row (first line looks like data)"
            )
            has_header = False
        else:
            has_header = self._gcp_has_header.isChecked()

        # ── Header handling ──
        if has_header:
            header = parsed_lines[0]
            data_lines = parsed_lines[1:]
        else:
            n_cols = max(len(fl) for fl in parsed_lines)
            header = [f"Col {i}" for i in range(n_cols)]
            data_lines = parsed_lines

        self._gcp_data_lines = data_lines

        self._populate_column_combos(
            header,
            self._gcp_name_col, self._gcp_x_col,
            self._gcp_y_col, self._gcp_z_col,
        )

        # If auto-detection failed (no X/Y/Z columns matched), use positional defaults
        if self._gcp_x_col.currentData() is None or self._gcp_x_col.currentData() < 0:
            n_cols = len(header)
            self._gcp_name_col.blockSignals(True)
            self._gcp_x_col.blockSignals(True)
            self._gcp_y_col.blockSignals(True)
            self._gcp_z_col.blockSignals(True)
            if n_cols >= 4:
                self._gcp_name_col.setCurrentIndex(1)  # Col 0 → Name
                self._gcp_x_col.setCurrentIndex(2)     # Col 1 → X
                self._gcp_y_col.setCurrentIndex(3)     # Col 2 → Y
                self._gcp_z_col.setCurrentIndex(4)     # Col 3 → Z
            elif n_cols == 3:
                self._gcp_x_col.setCurrentIndex(1)     # Col 0 → X
                self._gcp_y_col.setCurrentIndex(2)     # Col 1 → Y
                self._gcp_z_col.setCurrentIndex(3)     # Col 2 → Z
            self._gcp_name_col.blockSignals(False)
            self._gcp_x_col.blockSignals(False)
            self._gcp_y_col.blockSignals(False)
            self._gcp_z_col.blockSignals(False)

        self._update_gcp_points_from_combos()

    def _on_gcp_swap_xy(self):
        """Swap selected X and Y columns."""
        x_idx = self._gcp_x_col.currentIndex()
        y_idx = self._gcp_y_col.currentIndex()
        self._gcp_x_col.blockSignals(True)
        self._gcp_y_col.blockSignals(True)
        self._gcp_x_col.setCurrentIndex(y_idx)
        self._gcp_y_col.setCurrentIndex(x_idx)
        self._gcp_x_col.blockSignals(False)
        self._gcp_y_col.blockSignals(False)
        self._update_gcp_points_from_combos()

    def _on_gcp_columns_changed(self):
        self._update_gcp_points_from_combos()

    def _on_data_epsg_changed(self, val: int):
        self._data_epsg = val if val > 0 else None
        self._update_coordinate_diagnostics()

    def _on_gcp_epsg_changed(self, val: int):
        self._update_coordinate_diagnostics()

    def _update_gcp_points_from_combos(self):
        """Parse GCP points from stored CSV data lines using current column combo selection."""
        if not hasattr(self, "_gcp_data_lines") or not self._gcp_data_lines:
            return

        name_idx = self._gcp_name_col.currentData()
        x_idx = self._gcp_x_col.currentData()
        y_idx = self._gcp_y_col.currentData()
        z_idx = self._gcp_z_col.currentData()

        self._gcp_points = []
        if x_idx is None or y_idx is None or z_idx is None or x_idx < 0 or y_idx < 0 or z_idx < 0:
            self._gcp_run_btn.setEnabled(False)
            self._update_coordinate_diagnostics()
            return

        auto_name = 0
        for row in self._gcp_data_lines:
            if not row or all(c.strip() == "" for c in row):
                continue
            if max(x_idx, y_idx, z_idx) >= len(row):
                continue
            try:
                gx = _safe_float(row[x_idx])
                gy = _safe_float(row[y_idx])
                gz = _safe_float(row[z_idx])
            except (ValueError, IndexError):
                continue

            if name_idx is not None and 0 <= name_idx < len(row) and row[name_idx].strip():
                name = row[name_idx].strip()
            else:
                auto_name += 1
                name = str(auto_name)

            self._gcp_points.append((name, gx, gy, gz))

        self._gcp_run_btn.setEnabled(len(self._gcp_points) > 0)
        self._status.setText(f"Loaded {len(self._gcp_points)} GCPs from CSV")
        self._update_coordinate_diagnostics()

    def _get_lidar_extent(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return (min_x, max_x, min_y, max_y, min_z, max_z) of available LiDAR data."""
        if self._data and "x" in self._data and len(self._data["x"]) > 0:
            xs = self._data["x"]
            ys = self._data["y"]
            zs = self._data["z"]
            return (float(np.min(xs)), float(np.max(xs)),
                    float(np.min(ys)), float(np.max(ys)),
                    float(np.min(zs)), float(np.max(zs)))
        if self._db:
            try:
                tiles = []
                if self._tile_ids:
                    for tid in self._tile_ids:
                        t = self._db.get_tile(tid)
                        if t:
                            tiles.append(t)
                if not tiles:
                    tiles = self._db.get_all_tiles()
                if tiles:
                    valid_xs = [t["min_x"] for t in tiles if t.get("min_x") is not None]
                    valid_max_xs = [t["max_x"] for t in tiles if t.get("max_x") is not None]
                    valid_ys = [t["min_y"] for t in tiles if t.get("min_y") is not None]
                    valid_max_ys = [t["max_y"] for t in tiles if t.get("max_y") is not None]
                    valid_zs = [t["min_z"] for t in tiles if t.get("min_z") is not None]
                    valid_max_zs = [t["max_z"] for t in tiles if t.get("max_z") is not None]
                    if valid_xs and valid_max_xs and valid_ys and valid_max_ys:
                        min_x = min(valid_xs)
                        max_x = max(valid_max_xs)
                        min_y = min(valid_ys)
                        max_y = max(valid_max_ys)
                        min_z = min(valid_zs) if valid_zs else 0.0
                        max_z = max(valid_max_zs) if valid_max_zs else 0.0
                        return (min_x, max_x, min_y, max_y, min_z, max_z)
            except Exception as e:
                logger.debug("Failed to calculate lidar extent from db: %s", e)
        return None

    def _update_coordinate_diagnostics(self):
        """Update live coordinate status, bounds, and alignment warnings."""
        self._update_gcp_crs_info()

        if not hasattr(self, "_gcp_diag_label"):
            return

        if not self._gcp_points:
            self._gcp_diag_label.setText("<i>Load a CSV and map columns to view coordinate diagnostics.</i>")
            return

        coords = self._get_gcp_coords_to_use()
        if not coords:
            return

        g_xs = [p[1] for p in coords]
        g_ys = [p[2] for p in coords]
        g_zs = [p[3] for p in coords]

        g_min_x, g_max_x = min(g_xs), max(g_xs)
        g_min_y, g_max_y = min(g_ys), max(g_ys)
        g_min_z, g_max_z = min(g_zs), max(g_zs)

        lidar_ext = self._get_lidar_extent()

        lines = []
        s_name, s_x, s_y, s_z = coords[0]
        lines.append(f"<b>Sample Point 1 ('{s_name}'):</b> X = {s_x:.3f}, &nbsp; Y = {s_y:.3f}, &nbsp; Z = {s_z:.3f}")
        lines.append(f"<b>GCP Extent ({len(coords)} pts):</b> X: [{g_min_x:.1f} .. {g_max_x:.1f}], &nbsp; Y: [{g_min_y:.1f} .. {g_max_y:.1f}], &nbsp; Z: [{g_min_z:.1f} .. {g_max_z:.1f}]")

        if lidar_ext:
            l_min_x, l_max_x, l_min_y, l_max_y, l_min_z, l_max_z = lidar_ext
            lines.append(f"<b>LiDAR Extent:</b> X: [{l_min_x:.1f} .. {l_max_x:.1f}], &nbsp; Y: [{l_min_y:.1f} .. {l_max_y:.1f}], &nbsp; Z: [{l_min_z:.1f} .. {l_max_z:.1f}]")

            # Check 1: Degrees?
            if (-180.0 <= g_min_x <= 180.0 and -180.0 <= g_max_x <= 180.0 and
                -90.0 <= g_min_y <= 90.0 and -90.0 <= g_max_y <= 90.0 and
                (abs(l_max_x) > 1000.0 or abs(l_max_y) > 1000.0)):
                lines.append(
                    "<span style='color:#e67e22; font-weight:bold;'>"
                    "⚠ GCP coordinates appear to be Geographic Lat/Lon in degrees! "
                    "Enter 4326 in 'GCP CSV EPSG' above to auto-transform to projected LiDAR coordinates.</span>"
                )
            # Check 2: Swapped X/Y?
            elif (l_min_x - 5000.0 <= g_min_y <= l_max_x + 5000.0 and
                  l_min_y - 5000.0 <= g_min_x <= l_max_y + 5000.0 and
                  not (l_min_x - 5000.0 <= g_min_x <= l_max_x + 5000.0)):
                lines.append(
                    "<span style='color:#e74c3c; font-weight:bold;'>"
                    "⚠ Coordinates appear SWAPPED (Easting in Y column, Northing in X column)! "
                    "Click <b>'⇄ Swap X & Y'</b> above.</span>"
                )
            else:
                in_bounds = sum(
                    1 for _, x, y, _ in coords
                    if (l_min_x - 50.0 <= x <= l_max_x + 50.0 and l_min_y - 50.0 <= y <= l_max_y + 50.0)
                )
                if in_bounds > 0:
                    lines.append(
                        f"<span style='color:#27ae60; font-weight:bold;'>"
                        f"✓ Coordinate overlap verified: {in_bounds}/{len(coords)} GCPs fall inside LiDAR project area.</span>"
                    )
                else:
                    lines.append(
                        "<span style='color:#c0392b; font-weight:bold;'>"
                        "❌ No overlap! GCPs are outside the LiDAR project bounds. Check EPSG codes, column mappings, or use '⇄ Swap X & Y'.</span>"
                    )
        else:
            lines.append("<i>LiDAR extent unavailable for bounding box verification.</i>")

        self._gcp_diag_label.setText("<br>".join(lines))

    def _on_gcp_sep_changed(self):
        self._gcp_custom_sep.setVisible(
            self._gcp_sep_combo.currentText() not in self.SEPARATORS
        )
        if self._gcp_csv_edit.text():
            self._parse_gcp_csv(self._gcp_csv_edit.text())

    def _on_gcp_has_header_toggled(self):
        """Re-parse when the header checkbox changes."""
        if self._gcp_csv_edit.text():
            self._parse_gcp_csv(self._gcp_csv_edit.text())

    def _update_gcp_crs_info(self):
        """Update the CRS info label showing transform status."""
        gcp_epsg = self._gcp_epsg_spin.value()
        data_epsg = self._data_epsg_spin.value() if hasattr(self, "_data_epsg_spin") and self._data_epsg_spin.value() > 0 else self._data_epsg

        if not data_epsg:
            self._gcp_crs_info.setText(
                "⚠ LiDAR Data CRS unknown. If your GCPs need reprojection, enter the LiDAR Data EPSG above."
            )
            self._gcp_crs_info.setStyleSheet("color: #c09853;")
        elif gcp_epsg == 0:
            self._gcp_crs_info.setText(
                f"GCP coordinates assumed in LiDAR CRS (EPSG:{data_epsg}) — no transform."
            )
            self._gcp_crs_info.setStyleSheet("color: #888;")
        elif gcp_epsg == data_epsg:
            self._gcp_crs_info.setText(
                f"GCP EPSG:{gcp_epsg} matches LiDAR EPSG:{data_epsg} — no transform needed."
            )
            self._gcp_crs_info.setStyleSheet("color: #5cb85c;")
        else:
            self._gcp_crs_info.setText(
                f"GCP EPSG:{gcp_epsg} → auto-transforming to LiDAR EPSG:{data_epsg}."
            )
            self._gcp_crs_info.setStyleSheet("color: #27ae60; font-weight: bold;")

    def _get_gcp_coords_to_use(self) -> List[Tuple[str, float, float, float]]:
        """Return GCP points, transformed to data CRS if needed."""
        gcp_epsg = self._gcp_epsg_spin.value()
        data_epsg = self._data_epsg_spin.value() if hasattr(self, "_data_epsg_spin") and self._data_epsg_spin.value() > 0 else self._data_epsg

        if (not data_epsg or gcp_epsg == 0
                or gcp_epsg == data_epsg
                or not HAS_PYPROJ):
            return list(self._gcp_points)

        # Transform from GCP EPSG to data EPSG
        import numpy as np
        gcp_xs = np.array([p[1] for p in self._gcp_points], dtype=np.float64)
        gcp_ys = np.array([p[2] for p in self._gcp_points], dtype=np.float64)
        gcp_zs = np.array([p[3] for p in self._gcp_points], dtype=np.float64)

        try:
            tx, ty, tz = transform_coordinates(
                gcp_xs, gcp_ys, gcp_zs,
                source_crs=f"EPSG:{gcp_epsg}",
                target_crs=f"EPSG:{data_epsg}",
            )
        except Exception as exc:
            logger.warning("GCP CRS transform failed: %s", exc)
            self._status.setText(f"⚠ CRS transform failed: {exc}")
            return list(self._gcp_points)

        transformed = []
        for i, (name, _, _, _) in enumerate(self._gcp_points):
            tz_val = float(tz[i]) if tz is not None else self._gcp_points[i][3]
            transformed.append((name, float(tx[i]), float(ty[i]), tz_val))

        return transformed

    # ── Browse Roofs CSV ─────────────────────────────────────────────

    def _on_browse_roof_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Surfaces CSV", "",
            "CSV Files (*.csv *.txt);;All Files (*)",
        )
        if not path:
            return
        self._roof_csv_edit.setText(path)
        self._parse_roof_csv(path)

    def _parse_roof_csv(self, path: str):
        sep = self._get_separator(self._roof_sep_combo, self._roof_custom_sep)
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception as exc:
            self._roof_preview.setText(f"Error reading file: {exc}")
            return

        raw_lines = content.splitlines()
        preview = "\n".join(raw_lines[:10])
        if len(raw_lines) > 10:
            preview += f"\n… ({len(raw_lines)} total lines)"
        self._roof_preview.setText(preview)

        parsed_lines = []
        for line in raw_lines:
            if line.strip() == "":
                continue
            fields = self._split_csv_line(line, sep)
            if fields:
                parsed_lines.append(fields)

        if not parsed_lines:
            self._roof_preview.setText("Empty file")
            return

        first_is_data = self._line_looks_like_data(parsed_lines[0])
        if first_is_data:
            n_cols = max(len(fl) for fl in parsed_lines)
            header = [f"Col {i}" for i in range(n_cols)]
            data_lines = parsed_lines
        else:
            header = parsed_lines[0]
            data_lines = parsed_lines[1:]

        self._roof_data_lines = data_lines

        self._populate_column_combos(
            header,
            self._roof_id_col, self._roof_x_col,
            self._roof_y_col, self._roof_z_col,
        )

        # Positional defaults if auto-detection failed
        if self._roof_x_col.currentData() is None or self._roof_x_col.currentData() < 0:
            n_cols = len(header)
            self._roof_id_col.blockSignals(True)
            self._roof_x_col.blockSignals(True)
            self._roof_y_col.blockSignals(True)
            self._roof_z_col.blockSignals(True)
            if n_cols >= 4:
                self._roof_id_col.setCurrentIndex(1)  # Col 0 -> Name/ID
                self._roof_x_col.setCurrentIndex(2)   # Col 1 -> X
                self._roof_y_col.setCurrentIndex(3)   # Col 2 -> Y
                self._roof_z_col.setCurrentIndex(4)   # Col 3 -> Z
            elif n_cols == 3:
                self._roof_x_col.setCurrentIndex(1)
                self._roof_y_col.setCurrentIndex(2)
                self._roof_z_col.setCurrentIndex(3)
            self._roof_id_col.blockSignals(False)
            self._roof_x_col.blockSignals(False)
            self._roof_y_col.blockSignals(False)
            self._roof_z_col.blockSignals(False)

        self._update_roof_records_from_combos()

    def _update_roof_records_from_combos(self):
        """Parse raw records from stored CSV lines using current column combo selection."""
        if not hasattr(self, "_roof_data_lines") or not self._roof_data_lines:
            return

        id_idx = self._roof_id_col.currentData()
        x_idx = self._roof_x_col.currentData()
        y_idx = self._roof_y_col.currentData()
        z_idx = self._roof_z_col.currentData()

        if x_idx is None or y_idx is None or z_idx is None or x_idx < 0 or y_idx < 0 or z_idx < 0:
            return

        records = []
        auto_name = 0
        for row in self._roof_data_lines:
            if not row or all(c.strip() == "" for c in row):
                continue
            if max(x_idx, y_idx, z_idx) >= len(row):
                continue
            try:
                vx = _safe_float(row[x_idx])
                vy = _safe_float(row[y_idx])
                vz = _safe_float(row[z_idx])
            except (ValueError, IndexError):
                continue

            if id_idx is not None and 0 <= id_idx < len(row) and row[id_idx].strip():
                name = row[id_idx].strip()
            else:
                auto_name += 1
                name = str(auto_name)

            records.append({
                "name": name,
                "raw_id": name,
                "x": vx,
                "y": vy,
                "z": vz,
            })

        self._roof_raw_records = records
        self._detect_and_group_roof_points()

    def _on_roof_radius_changed(self):
        if self._roof_detect_mode.currentData() in ("spatial", "auto"):
            self._detect_and_group_roof_points()

    def _detect_and_group_roof_points(self):
        """Group raw points into patches based on naming, proximity, or surface ID."""
        if not hasattr(self, "_roof_raw_records") or not self._roof_raw_records:
            return

        mode = self._roof_detect_mode.currentData()
        radius = self._roof_radius_spin.value()
        records = self._roof_raw_records

        groups = []
        if mode == "naming":
            for r in records:
                grp, _ = parse_roof_point_name(r["name"])
                groups.append(grp)
        elif mode == "spatial":
            coords = np.array([[r["x"], r["y"]] for r in records])
            labels = cluster_roof_points_spatially(coords, radius=radius)
            groups = [f"Patch {lbl + 1}" for lbl in labels]
        elif mode == "csv_id":
            for r in records:
                groups.append(r.get("raw_id") or "1")
        else:  # "auto"
            naming_groups = [parse_roof_point_name(r["name"])[0] for r in records]
            unique_grps = set(naming_groups)
            counts = {g: naming_groups.count(g) for g in unique_grps}
            valid_naming_patches = sum(1 for c in counts.values() if c >= 3)

            # If naming convention produces >= 1 patch with >= 3 vertices and multiple groups
            if valid_naming_patches >= 1 and (len(unique_grps) > 1 or len(records) <= 8):
                groups = naming_groups
            else:
                # Spatial clustering fallback
                coords = np.array([[r["x"], r["y"]] for r in records])
                labels = cluster_roof_points_spatially(coords, radius=radius)
                groups = [f"Patch {lbl + 1}" for lbl in labels]

        # Populate _roof_points_table
        self._roof_points_table.blockSignals(True)
        self._roof_points_table.setRowCount(len(records))
        for row, (rec, grp) in enumerate(zip(records, groups)):
            # Col 0: Point Name
            name_item = QTableWidgetItem(str(rec["name"]))
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._roof_points_table.setItem(row, 0, name_item)

            # Col 1: Patch / House No. (EDITABLE by user)
            house_item = QTableWidgetItem(str(grp))
            house_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            self._roof_points_table.setItem(row, 1, house_item)

            # Col 2: Type combobox
            combo = QComboBox()
            combo.addItem("🏠 Roof", "roof")
            combo.addItem("🌱 Ground", "ground")
            low_name = (str(rec["name"]) + " " + str(grp)).lower()
            if any(k in low_name for k in ("ground", "gnd", "boden", "flur", "terrain", "pad", "slab")):
                combo.setCurrentIndex(1)
            else:
                combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(self._rebuild_roof_surfaces_from_table)
            self._roof_points_table.setCellWidget(row, 2, combo)

            # Col 3, 4, 5: X, Y, Z
            for c_idx, val in enumerate((rec["x"], rec["y"], rec["z"]), start=3):
                it = QTableWidgetItem(f"{val:.3f}")
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._roof_points_table.setItem(row, c_idx, it)

        self._roof_points_table.blockSignals(False)
        self._roof_points_table.resizeColumnsToContents()
        self._rebuild_roof_surfaces_from_table()

    def _on_roof_point_item_changed(self, item: QTableWidgetItem):
        if item.column() == 1:  # Patch / House No.
            self._rebuild_roof_surfaces_from_table()

    def _rebuild_roof_surfaces_from_table(self):
        """Collect points grouped by House No. column and build surfaces with optional angular sorting."""
        n_rows = self._roof_points_table.rowCount()
        if n_rows == 0:
            self._roof_surfaces = []
            self._roof_table.setRowCount(0)
            self._roof_run_btn.setEnabled(False)
            return

        groups: dict = {}
        for row in range(n_rows):
            name_it = self._roof_points_table.item(row, 0)
            house_it = self._roof_points_table.item(row, 1)
            type_widget = self._roof_points_table.cellWidget(row, 2)
            x_it = self._roof_points_table.item(row, 3)
            y_it = self._roof_points_table.item(row, 4)
            z_it = self._roof_points_table.item(row, 5)

            if not house_it or not x_it or not y_it or not z_it:
                continue

            house_no = house_it.text().strip()
            if not house_no:
                continue

            try:
                x = _safe_float(x_it.text())
                y = _safe_float(y_it.text())
                z = _safe_float(z_it.text())
            except ValueError:
                continue

            pt_type = type_widget.currentData() if isinstance(type_widget, QComboBox) else "roof"
            if house_no not in groups:
                groups[house_no] = {"verts": [], "names": [], "type": pt_type}
            groups[house_no]["verts"].append((x, y, z))
            groups[house_no]["names"].append(name_it.text() if name_it else "")

        # Build surfaces list
        surfaces = []
        sort_angular = self._roof_sort_chk.isChecked()

        for sid, data in groups.items():
            verts = np.array(data["verts"], dtype=np.float64)
            if len(verts) < 3:
                continue
            if sort_angular:
                verts, _ = sort_vertices_angular(verts)
            surfaces.append((sid, [tuple(v) for v in verts], data["type"]))

        self._roof_surfaces = surfaces

        # Refresh patches summary preview
        self._roof_table.blockSignals(True)
        self._roof_table.setRowCount(len(surfaces))
        for r_idx, (sid, verts, ptype) in enumerate(surfaces):
            type_str = "🏠 Roof" if ptype == "roof" else "🌱 Ground"
            chk_item = _CheckboxTableWidgetItem(checked=True)
            chk_item.setData(Qt.UserRole, r_idx)
            self._roof_table.setItem(r_idx, 0, chk_item)
            self._roof_table.setItem(r_idx, 1, QTableWidgetItem(str(sid)))
            self._roof_table.setItem(r_idx, 2, QTableWidgetItem(type_str))
            self._roof_table.setItem(r_idx, 3, QTableWidgetItem(str(len(verts))))
            self._roof_table.setItem(r_idx, 4, QTableWidgetItem("-"))
            self._roof_table.setItem(r_idx, 5, QTableWidgetItem("-"))
            self._roof_table.setItem(r_idx, 6, QTableWidgetItem("-"))
            self._roof_table.setItem(r_idx, 7, QTableWidgetItem("-"))
            self._roof_table.setItem(r_idx, 8, QTableWidgetItem("-"))

        self._roof_table.blockSignals(False)
        self._roof_table.resizeColumnsToContents()
        self._roof_run_btn.setEnabled(len(surfaces) > 0)
        self._status.setText(f"Detected {len(surfaces)} patch(es) from {n_rows} point(s)")

    def _on_roof_table_selection_changed(self):
        sel = self._roof_table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        if hasattr(self, "_roof_results") and self._roof_results and row < len(self._roof_results):
            res = self._roof_results[row]
            self._roof_inspector.load_patch(res)
        elif self._roof_surfaces and row < len(self._roof_surfaces):
            sid, verts, ptype = self._roof_surfaces[row]
            verts_arr = np.array(verts, dtype=np.float64)
            patch_preview = {
                "name": sid,
                "type": ptype,
                "verts": verts_arr,
                "n_verts": len(verts),
                "centroid": verts_arr.mean(axis=0),
                "dx": 0.0, "dy": 0.0, "dz": 0.0,
                "base_dx": 0.0, "base_dy": 0.0, "base_dz": 0.0,
                "shift_mag": 0.0,
                "used": True,
            }
            self._roof_inspector.load_patch(patch_preview)

    def _on_patch_offset_modified(self, patch_name: str, dx: float, dy: float, dz: float):
        if not hasattr(self, "_roof_results") or not self._roof_results:
            return

        target_res = None
        for r in self._roof_results:
            if str(r.get("name")) == str(patch_name):
                target_res = r
                break
        if not target_res:
            return

        target_res["dx"] = dx
        target_res["dy"] = dy
        target_res["dz"] = dz
        target_res["shift_mag"] = float(np.sqrt(dx*dx + dy*dy + dz*dz))

        # Update table row (Col 1 is Patch Name, Cols 5..8 are ΔX, ΔY, ΔZ, |Shift|)
        self._roof_table.blockSignals(True)
        for row in range(self._roof_table.rowCount()):
            it = self._roof_table.item(row, 1)
            if it and it.text() == str(patch_name):
                self._roof_table.setItem(row, 5, _NumericTableWidgetItem(dx, f"{dx:+.3f}"))
                self._roof_table.setItem(row, 6, _NumericTableWidgetItem(dy, f"{dy:+.3f}"))
                self._roof_table.setItem(row, 7, _NumericTableWidgetItem(dz, f"{dz:+.3f}"))
                self._roof_table.setItem(row, 8, _NumericTableWidgetItem(target_res['shift_mag'], f"{target_res['shift_mag']:.3f}"))
                break
        self._roof_table.blockSignals(False)

        # Recompute overall statistics using active/used patches
        self._recalculate_roof_statistics()

    def _on_roof_table_item_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            orig_idx = item.data(Qt.UserRole)
            is_checked = (item.checkState() == Qt.Checked)
            if hasattr(self, "_roof_results") and self._roof_results:
                if orig_idx is not None and 0 <= orig_idx < len(self._roof_results):
                    self._roof_results[orig_idx]["used"] = is_checked
                    patch_name = str(self._roof_results[orig_idx].get("name", ""))
                    if self._roof_inspector._current_patch and str(self._roof_inspector._current_patch.get("name", "")) == patch_name:
                        self._roof_inspector.set_used(is_checked)
            self._recalculate_roof_statistics()

    def _on_inspector_use_toggled(self, patch_name: str, used: bool):
        if not hasattr(self, "_roof_results") or not self._roof_results:
            return
        for r in self._roof_results:
            if str(r.get("name", "")) == str(patch_name):
                r["used"] = used
                break

        # Sync table checkbox
        self._roof_table.blockSignals(True)
        for row in range(self._roof_table.rowCount()):
            name_item = self._roof_table.item(row, 1)
            if name_item and name_item.text() == str(patch_name):
                use_item = self._roof_table.item(row, 0)
                if use_item:
                    use_item.setCheckState(Qt.Checked if used else Qt.Unchecked)
                break
        self._roof_table.blockSignals(False)

        self._recalculate_roof_statistics()

    def _on_toggle_selected_patch_use(self):
        sel = self._roof_table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        use_item = self._roof_table.item(row, 0)
        if not use_item:
            return
        cur_state = use_item.checkState()
        new_state = Qt.Unchecked if cur_state == Qt.Checked else Qt.Checked
        use_item.setCheckState(new_state)

    def delete_roof_patch(self, patch_name: str, confirm: bool = True):
        """Delete an entire bad/wrong patch, removing its points and surfaces."""
        if confirm:
            reply = QMessageBox.question(
                self, "Delete Patch",
                f"Are you sure you want to delete patch '{patch_name}'?\n\n"
                f"This will remove all associated corner points and exclude it from calculations.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # 1. Remove points from _roof_points_table whose Patch / House No. matches patch_name
        self._roof_points_table.blockSignals(True)
        rows_to_delete = []
        for row in range(self._roof_points_table.rowCount()):
            house_item = self._roof_points_table.item(row, 1)
            if house_item and house_item.text().strip() == str(patch_name).strip():
                rows_to_delete.append(row)
        for row in reversed(rows_to_delete):
            self._roof_points_table.removeRow(row)
        self._roof_points_table.blockSignals(False)

        # 2. Also sync raw records if present
        if hasattr(self, "_roof_raw_records") and self._roof_raw_records:
            new_records = []
            for row in range(self._roof_points_table.rowCount()):
                name_it = self._roof_points_table.item(row, 0)
                x_it = self._roof_points_table.item(row, 3)
                y_it = self._roof_points_table.item(row, 4)
                z_it = self._roof_points_table.item(row, 5)
                if name_it and x_it and y_it and z_it:
                    try:
                        new_records.append({
                            "name": name_it.text(),
                            "raw_id": self._roof_points_table.item(row, 1).text().strip() if self._roof_points_table.item(row, 1) else "",
                            "x": _safe_float(x_it.text()),
                            "y": _safe_float(y_it.text()),
                            "z": _safe_float(z_it.text()),
                        })
                    except ValueError:
                        pass
            self._roof_raw_records = new_records

        # 3. Rebuild surfaces from the points table
        self._rebuild_roof_surfaces_from_table()

        # 4. Remove from _roof_results
        if hasattr(self, "_roof_results") and self._roof_results:
            self._roof_results = [r for r in self._roof_results if str(r.get("name", "")).strip() != str(patch_name).strip()]

        # 5. Remove from _roof_table
        self._roof_table.blockSignals(True)
        for row in range(self._roof_table.rowCount()):
            name_item = self._roof_table.item(row, 1)
            if name_item and name_item.text().strip() == str(patch_name).strip():
                self._roof_table.removeRow(row)
                break
        # Re-index UserRole in column 0
        for row in range(self._roof_table.rowCount()):
            use_item = self._roof_table.item(row, 0)
            if use_item:
                use_item.setData(Qt.UserRole, row)
        self._roof_table.blockSignals(False)

        # 6. Clear or reload inspector
        if self._roof_inspector._current_patch and str(self._roof_inspector._current_patch.get("name", "")).strip() == str(patch_name).strip():
            self._roof_inspector.clear_patch()
            if hasattr(self, "_roof_results") and self._roof_results:
                self._roof_table.selectRow(0)
                self._roof_inspector.load_patch(self._roof_results[0])

        # 7. Recalculate stats
        self._recalculate_roof_statistics()
        self._status.setText(f"Deleted patch '{patch_name}' ({len(rows_to_delete)} points removed)")

    def _on_delete_selected_patch(self):
        sel = self._roof_table.selectedItems()
        patch_name = None
        if sel:
            row = sel[0].row()
            name_item = self._roof_table.item(row, 1)
            if name_item:
                patch_name = name_item.text()
        elif self._roof_inspector._current_patch:
            patch_name = self._roof_inspector._current_patch.get("name")

        if patch_name:
            self.delete_roof_patch(patch_name)
        else:
            QMessageBox.information(self, "No Patch Selected", "Please select a patch in the table to delete.")

    def _on_delete_roof_points(self, confirm: bool = True):
        selected_rows = set()
        for item in self._roof_points_table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            QMessageBox.information(self, "No Points Selected", "Please select one or more rows in the points table to delete.")
            return

        sorted_rows = sorted(selected_rows, reverse=True)
        if confirm:
            reply = QMessageBox.question(
                self, "Delete Selected Points",
                f"Delete {len(sorted_rows)} selected point(s)?\n\n"
                f"Remaining points will be regrouped automatically.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._roof_points_table.blockSignals(True)
        for row in sorted_rows:
            self._roof_points_table.removeRow(row)
        self._roof_points_table.blockSignals(False)

        # Sync raw records
        if hasattr(self, "_roof_raw_records") and self._roof_raw_records:
            new_records = []
            for row in range(self._roof_points_table.rowCount()):
                name_it = self._roof_points_table.item(row, 0)
                x_it = self._roof_points_table.item(row, 3)
                y_it = self._roof_points_table.item(row, 4)
                z_it = self._roof_points_table.item(row, 5)
                if name_it and x_it and y_it and z_it:
                    try:
                        new_records.append({
                            "name": name_it.text(),
                            "raw_id": self._roof_points_table.item(row, 1).text().strip() if self._roof_points_table.item(row, 1) else "",
                            "x": _safe_float(x_it.text()),
                            "y": _safe_float(y_it.text()),
                            "z": _safe_float(z_it.text()),
                        })
                    except ValueError:
                        pass
            self._roof_raw_records = new_records

        self._rebuild_roof_surfaces_from_table()
        self._status.setText(f"Deleted {len(sorted_rows)} point(s) and rebuilt surfaces")

    def _recalculate_roof_statistics(self):
        """Recalculate mean XYZ shift and stats using only active (used) patches with valid fit."""
        if not hasattr(self, "_roof_results") or not self._roof_results:
            self._roof_shift = None
            self._roof_apply_btn.setEnabled(False)
            self._roof_stats.setText("")
            return

        used_patches = [
            r for r in self._roof_results
            if r.get("used", True) and r.get("dx") is not None
        ]

        if used_patches:
            dxs = [r["dx"] for r in used_patches]
            dys = [r["dy"] for r in used_patches]
            dzs = [r["dz"] for r in used_patches]
            m_dx, m_dy, m_dz = float(np.mean(dxs)), float(np.mean(dys)), float(np.mean(dzs))
            mag_arr = np.sqrt(np.array(dxs)**2 + np.array(dys)**2 + np.array(dzs)**2)
            med_mag = float(np.median(mag_arr))
            mean_mag = float(np.mean(mag_arr))
            total_patches = len(self._roof_results)

            self._roof_stats.setText(
                f"<b>Statistics (using {len(used_patches)} of {total_patches} surfaces):</b><br>"
                f"Mean ΔX = <b>{m_dx:+.3f} m</b>  |  "
                f"Mean ΔY = <b>{m_dy:+.3f} m</b>  |  "
                f"Mean ΔZ = <b>{m_dz:+.3f} m</b><br>"
                f"Median |Shift| = {med_mag:.3f} m  |  "
                f"Mean |Shift| = {mean_mag:.3f} m"
            )
            self._roof_shift = (m_dx, m_dy, m_dz)
            self._roof_apply_btn.setEnabled(True)
            self._update_apply_buttons_text()
            self._status.setText(
                f"Roof calculation: using {len(used_patches)}/{total_patches} surfaces (Mean shift: {m_dx:+.3f}, {m_dy:+.3f}, {m_dz:+.3f} m)"
            )
        else:
            self._roof_shift = None
            self._roof_apply_btn.setEnabled(False)
            self._roof_stats.setText(
                "<span style='color:#c0392b; font-weight:bold;'>"
                "No active surfaces in calculation. Check at least one surface to apply shift.</span>"
            )
            self._status.setText("Roof calculation: no active surfaces selected")

        # Synchronize inspector if showing a patch
        if hasattr(self, "_roof_inspector") and self._roof_inspector._current_patch:
            cur_name = str(self._roof_inspector._current_patch.get("name", ""))
            for r in self._roof_results:
                if str(r.get("name", "")) == cur_name:
                    self._roof_inspector.set_used(r.get("used", True))
                    break

        # Synchronize visual check navigation checkbox if in roofs mode
        if self._vis_mode == "roofs" and hasattr(self, "_vis_use_chk"):
            if 0 <= self._vis_index < len(self._roof_results):
                r = self._roof_results[self._vis_index]
                is_valid = (r.get("dx") is not None)
                is_used = r.get("used", True) and is_valid
                self._vis_use_chk.blockSignals(True)
                self._vis_use_chk.setEnabled(is_valid)
                self._vis_use_chk.setChecked(is_used)
                self._vis_use_chk.blockSignals(False)

    def _on_roof_sep_changed(self):
        self._roof_custom_sep.setVisible(
            self._roof_sep_combo.currentText() not in self.SEPARATORS
        )
        if self._roof_csv_edit.text():
            self._parse_roof_csv(self._roof_csv_edit.text())

    # ── Run GCP ──────────────────────────────────────────────────────

    def _on_run_gcp(self):
        # Update points from current combo state (never clear or reset combos!)
        self._update_gcp_points_from_combos()

        if not self._gcp_points:
            QMessageBox.warning(self, "No GCP Points",
                                "No valid GCP points could be parsed from the CSV.\n"
                                "Check the file, separator, and column mapping.")
            return

        # Get GCP points with CRS transform if needed
        gcp_points = self._get_gcp_coords_to_use()

        classes = set()
        for i in range(self._gcp_class_list.count()):
            item = self._gcp_class_list.item(i)
            if item.checkState() == Qt.Checked:
                classes.add(item.data(Qt.UserRole))

        if not classes:
            QMessageBox.warning(self, "No Classes Selected",
                                "Please select at least one ASPRS class to compare against.")
            return

        # ── Load data: multi-tile or single-tile ──
        data = self._load_data_for_gcps(gcp_points)

        # Validate that we actually have point data to work with
        if not data or "x" not in data or len(data.get("x", [])) == 0:
            QMessageBox.warning(
                self, "No Point Data",
                "No point cloud data could be loaded for the selected GCP locations.\n\n"
                "Check that:\n"
                "1. GCP coordinates are in the same CRS as the LiDAR data (or specify EPSG to auto-transform).\n"
                "2. X and Y are not reversed (try '⇄ Swap X & Y').\n"
                "3. The loaded project tiles actually cover this area.",
            )
            return

        self._gcp_run_btn.setEnabled(False)
        self._status.setText("Calculating elevation differences…")
        self._progress.setVisible(True)
        self._progress.setValue(0)

        params = {
            "points": gcp_points,
            "classes": classes,
            "radius": self._gcp_radius_spin.value(),
        }

        self._worker = _GroundControlWorker(data, "gcp", params, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_gcp.connect(self._on_gcp_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _load_data_for_gcps(self, gcp_points: list) -> dict:
        """Load point data from tiles that contain any of the given GCPs.

        Queries the database for ALL tiles whose bbox overlaps any GCP
        point (buffered by search radius) — not limited to the tiles currently
        selected in the tile list. Returns an empty dict when no tile data
        could be loaded.
        """
        if not self._db or not self._tm:
            return self._data  # fallback to whatever was passed at construction

        if not gcp_points:
            return {}

        radius = self._gcp_radius_spin.value() if hasattr(self, "_gcp_radius_spin") else 5.0

        # ── Query DB for all tiles covering any GCP (buffered by radius) ──
        tiles_with_gcps: set = set()
        for _name, gx, gy, _gz in gcp_points:
            matches = self._db.get_tiles_in_bbox(gx - radius, gy - radius, gx + radius, gy + radius)
            for info in matches:
                tid = info.get("id")
                if tid:
                    tiles_with_gcps.add(tid)

        if not tiles_with_gcps:
            # Fallback: if single tile data was passed into dialog, check if it contains points
            if self._data and "x" in self._data and len(self._data.get("x", [])) > 0:
                return self._data
            # Fallback: if tile_ids were passed, try using them
            if self._tile_ids:
                tiles_with_gcps.update(self._tile_ids)

        if not tiles_with_gcps:
            return {}  # caller validates and shows appropriate message

        # ── Load and merge data ──
        self._status.setText(
            f"Loading {len(tiles_with_gcps)} tile(s) containing GCPs…"
        )

        xs_list, ys_list, zs_list, cls_list = [], [], [], []
        for tid in sorted(tiles_with_gcps):
            td = self._tm.load_tile_points_full(tid)
            if td is None:
                continue
            xs_list.append(np.asarray(td["x"], dtype=np.float64))
            ys_list.append(np.asarray(td["y"], dtype=np.float64))
            zs_list.append(np.asarray(td["z"], dtype=np.float64))
            if "classification" in td:
                cls_list.append(np.asarray(td["classification"], dtype=np.int32))

        if not xs_list:
            if self._data and "x" in self._data and len(self._data.get("x", [])) > 0:
                return self._data
            return {}

        return {
            "x": np.concatenate(xs_list),
            "y": np.concatenate(ys_list),
            "z": np.concatenate(zs_list),
            "classification": np.concatenate(cls_list) if cls_list else None,
        }

    def _reparse_current_gcp(self):
        """Re-read points using current column mapping without resetting combos."""
        self._update_gcp_points_from_combos()

    def _on_gcp_select_all(self):
        """Check all GCP points with valid elevations."""
        self._gcp_table.blockSignals(True)
        for row in range(self._gcp_table.rowCount()):
            use_item = self._gcp_table.item(row, 0)
            dz_item = self._gcp_table.item(row, 6)
            if use_item and dz_item and isinstance(dz_item, _NumericTableWidgetItem) and dz_item._val is not None:
                use_item.setCheckState(Qt.Checked)
        self._gcp_table.blockSignals(False)
        self._recalculate_gcp_statistics()

    def _on_gcp_deselect_all(self):
        """Uncheck all GCP points."""
        self._gcp_table.blockSignals(True)
        for row in range(self._gcp_table.rowCount()):
            use_item = self._gcp_table.item(row, 0)
            if use_item:
                use_item.setCheckState(Qt.Unchecked)
        self._gcp_table.blockSignals(False)
        self._recalculate_gcp_statistics()

    def _on_gcp_disable_outliers(self):
        """Uncheck all GCP points whose absolute elevation difference |ΔZ| exceeds threshold."""
        thresh = self._gcp_outlier_spin.value()
        self._gcp_table.blockSignals(True)
        disabled_count = 0
        for row in range(self._gcp_table.rowCount()):
            dz_item = self._gcp_table.item(row, 6)
            use_item = self._gcp_table.item(row, 0)
            if dz_item and isinstance(dz_item, _NumericTableWidgetItem) and dz_item._val is not None:
                if abs(dz_item._val) > thresh:
                    if use_item and use_item.checkState() == Qt.Checked:
                        use_item.setCheckState(Qt.Unchecked)
                        disabled_count += 1
        self._gcp_table.blockSignals(False)
        self._recalculate_gcp_statistics()
        self._status.setText(f"Disabled {disabled_count} outliers with |ΔZ| > {thresh:.2f} m")

    def _on_gcp_table_item_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            self._recalculate_gcp_statistics()

    def _on_vis_use_toggled(self, checked: bool):
        if self._vis_mode == "gcp":
            if not self._gcp_results or not (0 <= self._vis_index < len(self._gcp_results)):
                return

            self._gcp_results[self._vis_index]["used"] = checked

            # Synchronize corresponding checkbox item in table
            self._gcp_table.blockSignals(True)
            for row in range(self._gcp_table.rowCount()):
                use_item = self._gcp_table.item(row, 0)
                if use_item and use_item.data(Qt.UserRole) == self._vis_index:
                    use_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    break
            self._gcp_table.blockSignals(False)

            self._recalculate_gcp_statistics()
        elif self._vis_mode == "roofs":
            if not self._roof_results or not (0 <= self._vis_index < len(self._roof_results)):
                return

            patch_name = str(self._roof_results[self._vis_index].get("name", ""))
            self._on_inspector_use_toggled(patch_name, checked)
            if hasattr(self, "_roof_inspector") and self._roof_inspector._current_patch:
                if str(self._roof_inspector._current_patch.get("name", "")) == patch_name:
                    self._roof_inspector.set_used(checked)

    def _recalculate_gcp_statistics(self):
        """Recalculate GCP statistics using only enabled/checked points."""
        used_dzs = []
        total_pts = self._gcp_table.rowCount()
        enabled_pts = 0

        for row in range(total_pts):
            use_item = self._gcp_table.item(row, 0)
            if not use_item:
                continue
            orig_idx = use_item.data(Qt.UserRole)
            is_checked = (use_item.checkState() == Qt.Checked)

            if orig_idx is not None and 0 <= orig_idx < len(self._gcp_results):
                self._gcp_results[orig_idx]["used"] = is_checked

            if is_checked:
                enabled_pts += 1
                dz_item = self._gcp_table.item(row, 6)
                if dz_item and isinstance(dz_item, _NumericTableWidgetItem) and dz_item._val is not None:
                    used_dzs.append(dz_item._val)

        if used_dzs:
            dz_arr = np.array(used_dzs)
            median_dz = float(np.median(dz_arr))
            mean_dz = float(np.mean(dz_arr))
            std_dz = float(np.std(dz_arr))
            rms_dz = float(np.sqrt(np.mean(dz_arr ** 2)))
            max_abs_dz = float(np.max(np.abs(dz_arr)))

            stats_txt = (
                f"<b>Statistics (using {len(used_dzs)} of {total_pts} points):</b>  "
                f"Median shift = <b>{median_dz:+.3f} m</b>  |  "
                f"Average shift = {mean_dz:+.3f} m  |  "
                f"StdDev = {std_dz:.3f} m  |  "
                f"RMS = {rms_dz:.3f} m"
            )
            if max_abs_dz > 20.0:
                stats_txt += (
                    f"<br><span style='color:#c09853;'>"
                    f"⚠ Max |ΔZ| in active points = {max_abs_dz:.1f} m — check for CRS/datum mismatch.</span>"
                )
            elif max_abs_dz > 5.0:
                stats_txt += (
                    f"<br><span style='color:#c09853;'>"
                    f"⚠ Max |ΔZ| in active points = {max_abs_dz:.2f} m.</span>"
                )
            self._gcp_stats.setText(stats_txt)

            self._gcp_shift = median_dz
            self._gcp_apply_btn.setEnabled(True)
            self._update_apply_buttons_text()
            self._status.setText(f"GCP calculation: using {len(used_dzs)}/{total_pts} points (Median shift: {median_dz:+.3f} m)")
        else:
            self._gcp_shift = None
            self._gcp_apply_btn.setEnabled(False)
            if total_pts > 0:
                self._gcp_stats.setText(
                    f"<span style='color:#c0392b; font-weight:bold;'>"
                    f"No active points ({enabled_pts} enabled, 0 with valid ΔZ). "
                    f"Please check at least one valid GCP point.</span>"
                )
                self._status.setText("GCP calculation: no active points selected")
            else:
                self._gcp_stats.setText("")

        # Synchronize visual check checkbox if showing a GCP
        if self._vis_mode == "gcp" and self._gcp_results and 0 <= self._vis_index < len(self._gcp_results):
            r = self._gcp_results[self._vis_index]
            self._vis_use_chk.blockSignals(True)
            self._vis_use_chk.setEnabled(r.get("z_cloud") is not None)
            self._vis_use_chk.setChecked(r.get("used", True) and r.get("z_cloud") is not None)
            self._vis_use_chk.blockSignals(False)

    def _on_gcp_finished(self, results: list):
        self._gcp_results = results
        for r in self._gcp_results:
            r["used"] = (r.get("z_cloud") is not None)

        self._gcp_run_btn.setEnabled(True)
        self._progress.setVisible(False)

        # Populate table with numeric sorting enabled
        self._gcp_table.blockSignals(True)
        self._gcp_table.setSortingEnabled(False)
        self._gcp_table.setRowCount(0)

        for i, r in enumerate(results):
            row = self._gcp_table.rowCount()
            self._gcp_table.insertRow(row)

            # Col 0: Checkbox "Use"
            is_valid = (r.get("z_cloud") is not None)
            chk_item = _CheckboxTableWidgetItem(checked=is_valid)
            chk_item.setData(Qt.UserRole, i)  # map to results index
            if not is_valid:
                chk_item.setFlags(Qt.ItemIsSelectable)
            self._gcp_table.setItem(row, 0, chk_item)

            # Col 1: Name
            self._gcp_table.setItem(row, 1, QTableWidgetItem(str(r["name"])))

            # Col 2: X
            self._gcp_table.setItem(row, 2, _NumericTableWidgetItem(r['x']))

            # Col 3: Y
            self._gcp_table.setItem(row, 3, _NumericTableWidgetItem(r['y']))

            # Col 4: Z Input
            self._gcp_table.setItem(row, 4, _NumericTableWidgetItem(r['z_in']))

            # Col 5: Z Cloud
            if is_valid:
                self._gcp_table.setItem(row, 5, _NumericTableWidgetItem(r['z_cloud']))
            else:
                self._gcp_table.setItem(row, 5, _NumericTableWidgetItem(None, "N/A"))

            # Col 6: ΔZ
            if is_valid:
                dz = r["dz"]
                dz_item = _NumericTableWidgetItem(dz, f"{dz:+.3f}")
                if abs(dz) > 20.0:
                    dz_item.setForeground(QColor("#c0392b"))  # red
                    dz_item.setText(f"{dz:+.3f} ⚠ CRS/datum?")
                    dz_item.setToolTip(
                        "ΔZ is implausibly large — the GCP's X/Y or Z is "
                        "likely in a different CRS/vertical datum than the "
                        "point cloud. Check the GCP EPSG and the Z column."
                    )
                elif abs(dz) > 5.0:
                    dz_item.setForeground(QColor("#c09853"))  # amber
                    dz_item.setToolTip(
                        "ΔZ is larger than expected — verify the GCP position "
                        "and elevation against the point cloud."
                    )
                self._gcp_table.setItem(row, 6, dz_item)
            else:
                warn_txt = r.get("warning", "N/A")
                self._gcp_table.setItem(row, 6, _NumericTableWidgetItem(None, warn_txt))

            # Col 7: Nearby pts
            self._gcp_table.setItem(row, 7, _NumericTableWidgetItem(float(r.get("n_nearby", 0)), str(r.get("n_nearby", 0))))

        self._gcp_table.resizeColumnsToContents()
        self._gcp_table.setSortingEnabled(True)
        self._gcp_table.blockSignals(False)

        # Recalculate statistics from active points
        self._recalculate_gcp_statistics()

        has_valid = any(r.get("z_cloud") is not None for r in results)
        if not has_valid:
            n_total = len(results)
            self._gcp_stats.setText(
                f"<span style='color:#c0392b; font-weight:bold;'>"
                f"0 / {n_total} GCP points found nearby LiDAR points!</span><br>"
                f"• Check that X and Y columns are not reversed (try <b>'⇄ Swap X & Y'</b> above).<br>"
                f"• Check Coordinate System: ensure LiDAR Data EPSG and GCP EPSG are correctly specified.<br>"
                f"• Check Target Classes: if the LiDAR data is unclassified, ensure Class 0 or 1 is checked.<br>"
                f"• Try increasing the Search Radius (currently {self._gcp_radius_spin.value():.1f} m)."
            )
            self._gcp_apply_btn.setEnabled(False)

        self._vis_mode = "gcp"
        self._vis_index = 0
        self._update_vis_nav()

    # ── Run Roofs ────────────────────────────────────────────────────

    def _on_run_roofs(self):
        self._rebuild_roof_surfaces_from_table()

        if not self._roof_surfaces:
            QMessageBox.warning(self, "No Roof Surfaces",
                                "No valid surfaces could be constructed from the points table.\n"
                                "Ensure at least 3 points belong to each patch/house number.")
            return

        # Load point data — pass centroid of each surface
        centroids = []
        for item in self._roof_surfaces:
            sid, verts = item[0], item[1]
            if not verts:
                continue
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            centroids.append((sid, sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)))

        data = self._load_data_for_gcps(centroids)

        if not data or "x" not in data or len(data.get("x", [])) == 0:
            QMessageBox.warning(
                self, "No Point Data",
                "No point cloud data could be loaded for the selected roof locations.\n"
                "Check that the coordinates are in the same CRS as the tiles,\n"
                "and that the selected tiles actually contain points.",
            )
            return

        self._roof_run_btn.setEnabled(False)
        self._status.setText("Calculating surface offsets…")
        self._progress.setVisible(True)
        self._progress.setValue(0)

        params = {
            "surfaces": self._roof_surfaces,
            "radius": self._roof_radius_spin.value(),
        }
        self._worker = _GroundControlWorker(data, "roofs", params, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_roofs.connect(self._on_roofs_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_roofs_finished(self, results: list):
        self._roof_results = results
        for r in self._roof_results:
            if "used" not in r:
                r["used"] = (r.get("dx") is not None)

        self._roof_run_btn.setEnabled(True)
        self._progress.setVisible(False)

        self._roof_table.blockSignals(True)
        self._roof_table.setSortingEnabled(False)
        self._roof_table.setRowCount(0)

        for i, r in enumerate(results):
            row = self._roof_table.rowCount()
            self._roof_table.insertRow(row)

            is_valid = (r.get("dx") is not None)
            is_used = r.get("used", True) and is_valid
            chk_item = _CheckboxTableWidgetItem(checked=is_used)
            chk_item.setData(Qt.UserRole, i)
            if not is_valid:
                chk_item.setFlags(Qt.ItemIsSelectable)
            self._roof_table.setItem(row, 0, chk_item)

            self._roof_table.setItem(row, 1, QTableWidgetItem(str(r["name"])))
            ptype_str = "🏠 Roof" if r.get("type") == "roof" else "🌱 Ground"
            self._roof_table.setItem(row, 2, QTableWidgetItem(ptype_str))
            self._roof_table.setItem(row, 3, QTableWidgetItem(str(r.get("n_verts", "?"))))
            self._roof_table.setItem(row, 4, QTableWidgetItem(str(r.get("n_nearby", 0))))
            if is_valid:
                mag = r.get("shift_mag", 0.0) or 0.0
                self._roof_table.setItem(row, 5, _NumericTableWidgetItem(r['dx'], f"{r['dx']:+.3f}"))
                self._roof_table.setItem(row, 6, _NumericTableWidgetItem(r['dy'], f"{r['dy']:+.3f}"))
                self._roof_table.setItem(row, 7, _NumericTableWidgetItem(r['dz'], f"{r['dz']:+.3f}"))
                self._roof_table.setItem(row, 8, _NumericTableWidgetItem(mag, f"{mag:.3f}"))
            else:
                warn = r.get("warning", "N/A")
                self._roof_table.setItem(row, 5, _NumericTableWidgetItem(None, warn))
                self._roof_table.setItem(row, 6, _NumericTableWidgetItem(None, ""))
                self._roof_table.setItem(row, 7, _NumericTableWidgetItem(None, ""))
                self._roof_table.setItem(row, 8, _NumericTableWidgetItem(None, ""))

        self._roof_table.resizeColumnsToContents()
        self._roof_table.setSortingEnabled(False)
        self._roof_table.blockSignals(False)
        self._roof_subtabs.setCurrentIndex(1)

        self._recalculate_roof_statistics()

        if self._roof_results:
            self._roof_table.selectRow(0)
            valid_results = [r for r in self._roof_results if r.get("dx") is not None]
            if valid_results:
                self._roof_inspector.load_patch(valid_results[0])
            self._vis_mode = "roofs"
            self._vis_index = 0
            self._update_vis_nav()

        valid_count = sum(1 for r in results if r.get("dx") is not None)
        self._status.setText(f"Roof calculation done: {valid_count} valid surfaces")

    # ── Target Scope Management ──────────────────────────────────────

    def _create_scope_selector_widget(self, tab_name: str) -> QWidget:
        """Create a target scope selector bar for applying shifts to all, selected, or custom tiles."""
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(6)

        lbl = QLabel("<b>Apply Shift Target:</b>")
        lay.addWidget(lbl)

        combo = QComboBox()
        combo.setToolTip("Select which tiles in the project will receive the ground control shift")
        lay.addWidget(combo, 1)

        btn = QPushButton("📂 Choose Tiles…")
        btn.setToolTip("Open dialog to pick specific project tiles")
        lay.addWidget(btn)

        if tab_name == "gcp":
            self._gcp_scope_combo = combo
            self._gcp_scope_btn = btn
            btn.clicked.connect(self._open_tile_selection_dialog)
            combo.currentIndexChanged.connect(lambda: self._on_scope_changed("gcp"))
        else:
            self._roof_scope_combo = combo
            self._roof_scope_btn = btn
            btn.clicked.connect(self._open_tile_selection_dialog)
            combo.currentIndexChanged.connect(lambda: self._on_scope_changed("roof"))

        return w

    def _populate_scope_combos(self):
        combos = []
        if hasattr(self, "_gcp_scope_combo"):
            combos.append(self._gcp_scope_combo)
        if hasattr(self, "_roof_scope_combo"):
            combos.append(self._roof_scope_combo)
        if not combos:
            return

        n_all = len(self._all_tile_ids)
        n_sel = len(self._selected_tile_ids)
        n_custom = len(self._custom_tile_ids)

        for cb in combos:
            cb.blockSignals(True)
            prev_data = cb.currentData() or ("selected" if n_sel > 0 and n_sel < n_all else "all")
            cb.clear()

            cb.addItem(f"All Project Tiles ({n_all} tiles)", "all")
            if n_sel > 0 and n_sel < n_all:
                cb.addItem(f"Selected Tiles ({n_sel} tiles)", "selected")
            if self._current_tile_id:
                cb.addItem(f"Current Tile only ({self._current_tile_id})", "current")
            cb.addItem(f"Custom Selection ({n_custom} tiles)…", "custom")

            idx = cb.findData(prev_data)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            else:
                cb.setCurrentIndex(0)
            cb.blockSignals(False)

        self._update_apply_buttons_text()

    def _on_scope_changed(self, source: str):
        target_cb = self._roof_scope_combo if source == "gcp" else self._gcp_scope_combo
        source_cb = self._gcp_scope_combo if source == "gcp" else self._roof_scope_combo

        if hasattr(self, "_gcp_scope_combo") and hasattr(self, "_roof_scope_combo"):
            chosen_data = source_cb.currentData()
            if target_cb.currentData() != chosen_data:
                target_cb.blockSignals(True)
                idx = target_cb.findData(chosen_data)
                if idx >= 0:
                    target_cb.setCurrentIndex(idx)
                target_cb.blockSignals(False)

        self._update_apply_buttons_text()

    def _open_tile_selection_dialog(self):
        current_chosen = self.get_target_tile_ids()
        dlg = _TileSelectionDialog(self._all_tile_ids, current_chosen, parent=self)
        if dlg.exec():
            selected = dlg.get_selected_tile_ids()
            if selected:
                self._custom_tile_ids = selected
                self._populate_scope_combos()
                for cb in [getattr(self, "_gcp_scope_combo", None), getattr(self, "_roof_scope_combo", None)]:
                    if cb:
                        cb.blockSignals(True)
                        idx = cb.findData("custom")
                        if idx >= 0:
                            cb.setCurrentIndex(idx)
                        cb.blockSignals(False)
                self._update_apply_buttons_text()

    def get_target_tile_ids(self) -> List[str]:
        scope = "all"
        for cb in [getattr(self, "_roof_scope_combo", None), getattr(self, "_gcp_scope_combo", None)]:
            if cb and cb.count() > 0:
                scope = cb.currentData() or "all"
                break

        if scope == "all":
            return list(self._all_tile_ids)
        elif scope == "selected":
            return list(self._selected_tile_ids) if self._selected_tile_ids else list(self._all_tile_ids)
        elif scope == "current":
            if self._current_tile_id:
                return [self._current_tile_id]
            elif self._selected_tile_ids:
                return [self._selected_tile_ids[0]]
            elif self._all_tile_ids:
                return [self._all_tile_ids[0]]
            return []
        elif scope == "custom":
            return list(self._custom_tile_ids)
        return list(self._all_tile_ids)

    def _get_scope_display_label(self) -> str:
        scope = "all"
        for cb in [getattr(self, "_roof_scope_combo", None), getattr(self, "_gcp_scope_combo", None)]:
            if cb and cb.count() > 0:
                scope = cb.currentData() or "all"
                break

        if scope == "all":
            return "All Project Tiles"
        elif scope == "selected":
            return "Selected Tiles"
        elif scope == "current":
            return f"Current Tile ({self._current_tile_id})" if self._current_tile_id else "Current Tile"
        elif scope == "custom":
            return "Custom Tiles"
        return "Tiles"

    def _update_apply_buttons_text(self):
        target_tids = self.get_target_tile_ids()
        count = len(target_tids)
        scope_lbl = self._get_scope_display_label()

        if hasattr(self, "_roof_apply_btn"):
            if self._roof_shift is not None:
                mx, my, mz = self._roof_shift
                self._roof_apply_btn.setText(
                    f"⬆ Apply XYZ Shift ({mx:+.3f}, {my:+.3f}, {mz:+.3f}) m to {scope_lbl} ({count})"
                )
            else:
                self._roof_apply_btn.setText(f"⬆ Apply XYZ Shift to {scope_lbl} ({count})")

        if hasattr(self, "_gcp_apply_btn"):
            if self._gcp_shift is not None:
                self._gcp_apply_btn.setText(
                    f"⬆ Apply Z Shift ({self._gcp_shift:+.3f} m) to {scope_lbl} ({count})"
                )
            else:
                self._gcp_apply_btn.setText(f"⬆ Apply Z Shift to {scope_lbl} ({count})")

    # ── Apply shift ──────────────────────────────────────────────────

    def _on_apply_gcp(self):
        if self._gcp_shift is not None:
            target_tids = self.get_target_tile_ids()
            if not target_tids:
                QMessageBox.warning(
                    self, "No Tiles Selected",
                    "Please select at least one tile to apply the ground control shift."
                )
                return

            n = len(target_tids)
            scope_lbl = self._get_scope_display_label()
            tiles_preview = ", ".join(target_tids[:6]) + (f" … (+{n - 6} more)" if n > 6 else "")

            reply = QMessageBox.question(
                self, "Apply Z Shift to Tiles",
                f"Apply a Z shift of {self._gcp_shift:+.3f} m to {n} tile(s) [{scope_lbl}]?\n\n"
                f"Target Tiles ({n}):\n"
                f"  {tiles_preview}\n\n"
                f"This will add {self._gcp_shift:+.3f} m to every point's Z coordinate in the LAS files on disk.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.shift_applied.emit(0.0, 0.0, self._gcp_shift, target_tids)

    def _on_apply_roofs(self):
        if self._roof_shift is not None:
            target_tids = self.get_target_tile_ids()
            if not target_tids:
                QMessageBox.warning(
                    self, "No Tiles Selected",
                    "Please select at least one tile to apply the ground control shift."
                )
                return

            dx, dy, dz = self._roof_shift
            mag = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            n = len(target_tids)
            scope_lbl = self._get_scope_display_label()
            tiles_preview = ", ".join(target_tids[:6]) + (f" … (+{n - 6} more)" if n > 6 else "")

            reply = QMessageBox.question(
                self, "Apply XYZ Shift to Tiles",
                f"Apply the 3-D ground control shift to {n} tile(s) [{scope_lbl}]?\n\n"
                f"  ΔX = {dx:+.3f} m\n"
                f"  ΔY = {dy:+.3f} m\n"
                f"  ΔZ = {dz:+.3f} m\n"
                f"  |Shift| = {mag:.3f} m\n\n"
                f"Target Tiles ({n}):\n"
                f"  {tiles_preview}\n\n"
                f"This will translate every point by (ΔX, ΔY, ΔZ) in the LAS files on disk.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.shift_applied.emit(dx, dy, dz, target_tids)

    # ── Visual check navigation ──────────────────────────────────────

    def _on_tab_changed(self, idx: int):
        if idx == 0:
            self._vis_mode = "gcp"
        else:
            self._vis_mode = "roofs"
        self._vis_index = 0
        self._update_vis_nav()

    def _current_vis_items(self) -> list:
        if self._vis_mode == "gcp":
            return self._gcp_results
        return self._roof_results

    def _update_vis_nav(self):
        items = self._current_vis_items()
        n = len(items)
        if n == 0:
            self._vis_label.setText("No results yet")
            self._vis_prev_btn.setEnabled(False)
            self._vis_next_btn.setEnabled(False)
            self._vis_goto_btn.setEnabled(False)
            if hasattr(self, "_vis_use_chk"):
                self._vis_use_chk.setVisible(False)
            return

        self._vis_prev_btn.setEnabled(self._vis_index > 0)
        self._vis_next_btn.setEnabled(self._vis_index < n - 1)
        self._vis_goto_btn.setEnabled(True)

        item = items[self._vis_index]
        name = item.get("name", "?")
        if self._vis_mode == "gcp":
            x, y = item.get("x", 0), item.get("y", 0)
            dz = item.get("dz")
            dz_str = f"{dz:+.3f} m" if dz is not None else "N/A"
            is_used = item.get("used", True) if dz is not None else False
            status_tag = " [ACTIVE]" if is_used else " [DISABLED]"
            self._vis_label.setText(
                f"[{self._vis_index + 1}/{n}]  {name}  @ ({x:.2f}, {y:.2f})  ΔZ={dz_str}{status_tag}"
            )
            if hasattr(self, "_vis_use_chk"):
                self._vis_use_chk.setVisible(True)
                self._vis_use_chk.blockSignals(True)
                self._vis_use_chk.setEnabled(dz is not None)
                self._vis_use_chk.setChecked(is_used)
                self._vis_use_chk.blockSignals(False)
        else:
            cx = item.get("centroid_x", 0)
            cy = item.get("centroid_y", 0)
            dx = item.get("dx")
            dy = item.get("dy")
            dz_val = item.get("dz")
            if dx is not None:
                shift_str = f"Δ=({dx:+.2f}, {dy:+.2f}, {dz_val:+.2f}) m"
            else:
                shift_str = "N/A"
            is_valid = (dx is not None)
            is_used = item.get("used", True) and is_valid
            status_tag = " [ACTIVE]" if is_used else " [DISMISSED / EXCLUDED]"
            self._vis_label.setText(
                f"[{self._vis_index + 1}/{n}]  {name}  @ ({cx:.2f}, {cy:.2f})  {shift_str}{status_tag}"
            )
            if hasattr(self, "_vis_use_chk"):
                self._vis_use_chk.setVisible(True)
                self._vis_use_chk.blockSignals(True)
                self._vis_use_chk.setEnabled(is_valid)
                self._vis_use_chk.setChecked(is_used)
                self._vis_use_chk.blockSignals(False)
            if hasattr(self, "_roof_inspector") and item:
                self._roof_inspector.load_patch(item)
            if hasattr(self, "_roof_table") and self._vis_index < self._roof_table.rowCount():
                self._roof_table.blockSignals(True)
                self._roof_table.selectRow(self._vis_index)
                self._roof_table.blockSignals(False)

    def _on_vis_prev(self):
        if self._vis_index > 0:
            self._vis_index -= 1
            self._update_vis_nav()

    def _on_vis_next(self):
        items = self._current_vis_items()
        if self._vis_index < len(items) - 1:
            self._vis_index += 1
            self._update_vis_nav()

    def _on_vis_goto(self):
        items = self._current_vis_items()
        if not items or self._vis_index >= len(items):
            return
        item = items[self._vis_index]
        if self._vis_mode == "gcp":
            x, y = float(item.get("x", 0)), float(item.get("y", 0))
            z = item.get("z_in")
            cloud_z = item.get("z_cloud")
            if z is None:
                z = cloud_z
            self.visualize_point.emit(
                x, y, float(z if z is not None else 0.0),
                float(cloud_z) if cloud_z is not None else None,
                str(item.get("name", "GCP")),
            )
        else:
            x = float(item.get("centroid_x", 0))
            y = float(item.get("centroid_y", 0))
            z = float(item.get("centroid_z", 0))
            self.visualize_point.emit(x, y, z, None, str(item.get("name", "Surface")))

    # ── Progress / error slots ───────────────────────────────────────

    def _on_progress(self, msg: str, pct: float):
        self._status.setText(msg)
        self._progress.setValue(int(pct))

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg}")
        self._progress.setVisible(False)
        self._gcp_run_btn.setEnabled(True)
        self._roof_run_btn.setEnabled(True)

    # ── Accept ───────────────────────────────────────────────────────

    def _on_accept(self):
        self.accept()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def gcp_results(self) -> list:
        return self._gcp_results

    @property
    def roof_results(self) -> list:
        return self._roof_results

    @property
    def current_shift(self) -> Optional[Tuple[float, float, float]]:
        """Return the most recently computed shift (GCP or roof) as (dx, dy, dz)."""
        if self._vis_mode == "gcp" and self._gcp_shift is not None:
            return (0.0, 0.0, self._gcp_shift)
        return self._roof_shift
