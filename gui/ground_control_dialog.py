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
from typing import List, Optional, Tuple

import numpy as np

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
        surfaces = self._params["surfaces"]  # list of (name, [(x,y,z),…])
        radius = self._params.get("radius", 5.0)

        xs = self._data["x"]
        ys = self._data["y"]
        zs = self._data["z"]

        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack((xs, ys)))

        n_surf = len(surfaces)
        results = []

        for i, (name, verts) in enumerate(surfaces):
            pct = (i + 1) / n_surf * 100
            self.progress.emit(f"Processing surface {name}…", pct)

            verts_arr = np.array(verts, dtype=np.float64)
            centroid = verts_arr.mean(axis=0)

            # Find nearby points
            indices = tree.query_ball_point(centroid[:2], radius)
            if len(indices) < 3:
                results.append({
                    "name": name, "n_verts": len(verts),
                    "n_nearby": len(indices),
                    "dx": None, "dy": None, "dz": None, "shift_mag": None,
                    "warning": f"Not enough nearby points ({len(indices)})",
                })
                continue

            nearby_xyz = np.column_stack((xs[indices], ys[indices], zs[indices]))

            # Compute input plane normal from vertices
            input_normal = self._fit_plane_normal(verts_arr)

            # Fit plane to nearby point cloud via RANSAC
            cloud_normal, cloud_centroid_z = self._fit_ransac_plane(nearby_xyz)

            # Compute 3-D shift vector between input surface and cloud
            if cloud_normal is not None:
                cloud_mean = nearby_xyz.mean(axis=0)

                # Signed distances from cloud points to cloud plane
                d_cloud = np.dot(nearby_xyz - cloud_mean, cloud_normal)
                cloud_median = float(np.median(d_cloud))

                # Signed distances from input vertices to cloud plane
                d_input = np.dot(verts_arr - cloud_mean, cloud_normal)
                input_median = float(np.median(d_input))

                # Scalar offset: positive = input is "above" cloud along normal
                offset = input_median - cloud_median

                # 3-D shift vector (what to add to cloud points to align)
                shift_vec = cloud_normal * (-offset)
                dx, dy, dz = float(shift_vec[0]), float(shift_vec[1]), float(shift_vec[2])
            else:
                # Fallback: simple Z difference
                cloud_z_median = float(np.median(nearby_xyz[:, 2]))
                input_z_mean = float(verts_arr[:, 2].mean())
                dx, dy = 0.0, 0.0
                dz = input_z_mean - cloud_z_median
                offset = dz

            results.append({
                "name": name, "n_verts": len(verts),
                "n_nearby": len(indices),
                "dx": dx, "dy": dy, "dz": dz,
                "shift_mag": float(np.sqrt(dx*dx + dy*dy + dz*dz)),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
            })

        self.finished_roofs.emit(results)

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

    @staticmethod
    def _fit_plane_normal(verts: np.ndarray) -> np.ndarray:
        """Fit a plane to *verts* (N×3) via SVD, return unit normal."""
        centroid = verts.mean(axis=0)
        _, _, vh = np.linalg.svd(verts - centroid)
        return vh[2]  # smallest singular value → normal

    @staticmethod
    def _fit_ransac_plane(xyz: np.ndarray, n_iter: int = 200,
                          threshold: float = 0.3) -> Tuple[Optional[np.ndarray], float]:
        """RANSAC plane fit. Returns (normal, centroid_z) or (None, median_z)."""
        if len(xyz) < 3:
            return None, float(np.median(xyz[:, 2]))

        best_inliers = 0
        best_normal = None

        for _ in range(n_iter):
            idx = np.random.choice(len(xyz), 3, replace=False)
            sample = xyz[idx]
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            nrm = np.linalg.norm(normal)
            if nrm < 1e-10:
                continue
            normal /= nrm

            # Distance of all points to this plane
            centroid = sample.mean(axis=0)
            dists = np.abs(np.dot(xyz - centroid, normal))
            inliers = (dists < threshold).sum()

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal

        if best_normal is not None:
            # Refit using all inliers within threshold
            centroid = xyz.mean(axis=0)
            dists = np.abs(np.dot(xyz - centroid, best_normal))
            inlier_mask = dists < threshold
            if inlier_mask.sum() >= 3:
                _, _, vh = np.linalg.svd(xyz[inlier_mask] - xyz[inlier_mask].mean(axis=0))
                best_normal = vh[2]

        return best_normal, float(np.median(xyz[:, 2]))


# ═══════════════════════════════════════════════════════════════════════
# Main dialog
# ═══════════════════════════════════════════════════════════════════════

class GroundControlDialog(QDialog):
    """
    Dialog for Ground Control Point (GCP) and Roof/Surface elevation validation.

    Signals
    -------
    shift_applied(float, float, float):
        Emitted when the user clicks 'Apply Shift'. Args: (dx, dy, dz).
    visualize_point(float, float, float, object, str):
        Emitted when user navigates to a GCP or surface.
        Args: (x, y, z_gcp, z_cloud, label).
    """

    shift_applied = Signal(float, float, float)
    visualize_point = Signal(float, float, float, object, str)

    SEPARATORS = {
        "Comma (,)": ",",
        "Semicolon (;)": ";",
        "Tab": "\t",
        "Space": " ",
    }

    def __init__(self, tile_data: dict, parent=None, data_epsg: Optional[int] = None,
                 tile_ids: Optional[list] = None,
                 tile_manager=None, database=None):
        super().__init__(parent)
        self._data = tile_data
        self._data_epsg = data_epsg  # EPSG code of the point cloud data
        self._worker: Optional[_GroundControlWorker] = None

        # Multi-tile support
        self._tile_ids: list = tile_ids or []
        self._tm = tile_manager       # TileManager for loading tile data
        self._db = database           # Database for bbox queries

        # GCP state
        self._gcp_points: List[Tuple[str, float, float, float]] = []
        self._gcp_data_lines: List[List[str]] = []
        self._gcp_results: list = []
        self._gcp_shift: Optional[float] = None
        self._gcp_source_epsg: Optional[int] = None  # EPSG of the GCP CSV

        # Roof state
        self._roof_surfaces: List[Tuple[str, List[Tuple[float, float, float]]]] = []
        self._roof_results: list = []
        self._roof_shift: Optional[Tuple[float, float, float]] = None

        # Visual check state
        self._vis_mode = "gcp"    # "gcp" or "roofs"
        self._vis_index: int = 0

        self.setWindowTitle("Ground Control")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
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
        layout = QVBoxLayout(w)

        # ── CSV import ──
        csv_group = QGroupBox("1. Import Surfaces CSV")
        cf = QFormLayout(csv_group)

        info = QLabel(
            "CSV must group polygon vertices by a surface ID/name column. "
            "Each row = one vertex (X, Y, Z). At least 3 vertices per surface."
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
        self._roof_preview.setMaximumHeight(80)
        self._roof_preview.setPlaceholderText("CSV preview…")
        cf.addRow("Preview:", self._roof_preview)

        layout.addWidget(csv_group)

        # ── Column mapping ──
        map_group = QGroupBox("2. Column Mapping")
        mf = QFormLayout(map_group)

        col_row = QHBoxLayout()
        self._roof_id_col = QComboBox()
        col_row.addWidget(QLabel("Surface ID:"))
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
        mf.addRow("Columns:", col_row)

        layout.addWidget(map_group)

        # ── Parameters ──
        param_group = QGroupBox("3. Parameters")
        pf = QFormLayout(param_group)
        self._roof_radius_spin = QDoubleSpinBox()
        self._roof_radius_spin.setRange(0.1, 200.0)
        self._roof_radius_spin.setDecimals(1)
        self._roof_radius_spin.setValue(5.0)
        self._roof_radius_spin.setSuffix(" m")
        self._roof_radius_spin.setToolTip("Search radius around surface centroid")
        pf.addRow("Search Radius:", self._roof_radius_spin)
        layout.addWidget(param_group)

        # ── Run button ──
        run_row = QHBoxLayout()
        self._roof_run_btn = QPushButton("▶ Calculate Surface Offsets")
        self._roof_run_btn.clicked.connect(self._on_run_roofs)
        self._roof_run_btn.setEnabled(False)
        run_row.addWidget(self._roof_run_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        # ── Results table ──
        res_group = QGroupBox("4. Results")
        rl = QVBoxLayout(res_group)
        self._roof_table = QTableWidget(0, 7)
        self._roof_table.setHorizontalHeaderLabels(
            ["Surface", "Vertices", "Nearby pts", "ΔX (m)", "ΔY (m)", "ΔZ (m)", "|Shift|"]
        )
        self._roof_table.horizontalHeader().setStretchLastSection(True)
        self._roof_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._roof_table.setSelectionBehavior(QTableWidget.SelectRows)
        rl.addWidget(self._roof_table)

        self._roof_stats = QLabel("")
        self._roof_stats.setWordWrap(True)
        rl.addWidget(self._roof_stats)

        layout.addWidget(res_group)

        # ── Apply shift ──
        self._roof_apply_btn = QPushButton("⬆ Apply XYZ Shift to Current Tile")
        self._roof_apply_btn.setEnabled(False)
        self._roof_apply_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 14px; }"
        )
        self._roof_apply_btn.clicked.connect(self._on_apply_roofs)
        layout.addWidget(self._roof_apply_btn)

        layout.addStretch()
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

        lines = content.splitlines()
        preview = "\n".join(lines[:10])
        if len(lines) > 10:
            preview += f"\n… ({len(lines)} total lines)"
        self._roof_preview.setText(preview)

        reader = csv.reader(io.StringIO(content), delimiter=sep)
        try:
            header = next(reader)
        except StopIteration:
            self._roof_preview.setText("Empty file")
            return

        self._populate_column_combos(
            header,
            self._roof_id_col, self._roof_x_col,
            self._roof_y_col, self._roof_z_col,
        )

        # Read vertices, group by surface ID
        rows = list(reader)
        groups: dict = {}
        auto_name = 0
        for row in rows:
            if not row or all(c.strip() == "" for c in row):
                continue
            id_idx = self._roof_id_col.currentData()
            x_idx = self._roof_x_col.currentData()
            y_idx = self._roof_y_col.currentData()
            z_idx = self._roof_z_col.currentData()

            if x_idx is None or y_idx is None or z_idx is None or x_idx < 0 or y_idx < 0 or z_idx < 0:
                continue

            try:
                vx = _safe_float(row[x_idx])
                vy = _safe_float(row[y_idx])
                vz = _safe_float(row[z_idx])
            except (ValueError, IndexError):
                continue

            if id_idx is not None and id_idx >= 0 and id_idx < len(row):
                sid = row[id_idx].strip()
            else:
                auto_name += 1
                sid = str(auto_name)

            groups.setdefault(sid, []).append((vx, vy, vz))

        self._roof_surfaces = [
            (sid, verts) for sid, verts in groups.items()
            if len(verts) >= 3
        ]
        skipped = len(groups) - len(self._roof_surfaces)
        msg = f"Loaded {len(self._roof_surfaces)} surface(s)"
        if skipped:
            msg += f" ({skipped} skipped — need ≥ 3 vertices)"

        self._roof_run_btn.setEnabled(len(self._roof_surfaces) > 0)
        self._status.setText(msg)

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
        if self._vis_mode != "gcp" or not self._gcp_results:
            return
        if not (0 <= self._vis_index < len(self._gcp_results)):
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
            self._gcp_apply_btn.setText(
                f"⬆ Apply Z Shift ({self._gcp_shift:+.3f} m) from {len(used_dzs)} Active Points to Current Tile"
            )
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
        self._reparse_current_roofs()

        if not self._roof_surfaces:
            QMessageBox.warning(self, "No Roof Surfaces",
                                "No valid roof surfaces could be parsed from the CSV.\n"
                                "Check the file, separator, and column mapping.")
            return

        # Load point data — roofs pass centroid of each surface
        centroids = []
        for sid, verts in self._roof_surfaces:
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

    def _reparse_current_roofs(self):
        if not self._roof_csv_edit.text():
            return
        sep = self._get_separator(self._roof_sep_combo, self._roof_custom_sep)
        try:
            with open(self._roof_csv_edit.text(), "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception:
            return
        reader = csv.reader(io.StringIO(content), delimiter=sep)
        try:
            next(reader)
        except StopIteration:
            return
        groups: dict = {}
        auto_name = 0
        for row in reader:
            id_idx = self._roof_id_col.currentData()
            x_idx = self._roof_x_col.currentData()
            y_idx = self._roof_y_col.currentData()
            z_idx = self._roof_z_col.currentData()
            if x_idx is None or y_idx is None or z_idx is None or x_idx < 0 or y_idx < 0 or z_idx < 0:
                continue
            try:
                vx = _safe_float(row[x_idx])
                vy = _safe_float(row[y_idx])
                vz = _safe_float(row[z_idx])
            except (ValueError, IndexError):
                continue
            if id_idx is not None and id_idx >= 0 and id_idx < len(row):
                sid = row[id_idx].strip()
            else:
                auto_name += 1
                sid = str(auto_name)
            groups.setdefault(sid, []).append((vx, vy, vz))
        self._roof_surfaces = [(sid, v) for sid, v in groups.items() if len(v) >= 3]

    def _on_roofs_finished(self, results: list):
        self._roof_results = results
        self._roof_run_btn.setEnabled(True)
        self._progress.setVisible(False)

        self._roof_table.setRowCount(0)
        dxs, dys, dzs = [], [], []
        for r in results:
            row = self._roof_table.rowCount()
            self._roof_table.insertRow(row)
            self._roof_table.setItem(row, 0, QTableWidgetItem(str(r["name"])))
            self._roof_table.setItem(row, 1, QTableWidgetItem(str(r.get("n_verts", "?"))))
            self._roof_table.setItem(row, 2, QTableWidgetItem(str(r.get("n_nearby", 0))))
            if r["dx"] is not None:
                self._roof_table.setItem(row, 3, QTableWidgetItem(f"{r['dx']:+.3f}"))
                self._roof_table.setItem(row, 4, QTableWidgetItem(f"{r['dy']:+.3f}"))
                self._roof_table.setItem(row, 5, QTableWidgetItem(f"{r['dz']:+.3f}"))
                mag = r.get("shift_mag", 0) or 0
                self._roof_table.setItem(row, 6, QTableWidgetItem(f"{mag:.3f}"))
                dxs.append(r["dx"]); dys.append(r["dy"]); dzs.append(r["dz"])
            else:
                warn = r.get("warning", "N/A")
                self._roof_table.setItem(row, 3, QTableWidgetItem(warn))
                self._roof_table.setItem(row, 4, QTableWidgetItem(""))
                self._roof_table.setItem(row, 5, QTableWidgetItem(""))
                self._roof_table.setItem(row, 6, QTableWidgetItem(""))
        self._roof_table.resizeColumnsToContents()

        if dxs:
            dx_arr = np.array(dxs); dy_arr = np.array(dys); dz_arr = np.array(dzs)
            mag_arr = np.sqrt(dx_arr**2 + dy_arr**2 + dz_arr**2)
            mx, my, mz = float(np.mean(dx_arr)), float(np.mean(dy_arr)), float(np.mean(dz_arr))
            med_mag = float(np.median(mag_arr))
            mean_mag = float(np.mean(mag_arr))
            self._roof_stats.setText(
                f"<b>Statistics ({len(dxs)} surfaces):</b>  "
                f"Mean ΔX = {mx:+.3f} m  |  "
                f"Mean ΔY = {my:+.3f} m  |  "
                f"Mean ΔZ = {mz:+.3f} m\n"
                f"Median |Shift| = {med_mag:.3f} m  |  "
                f"Mean |Shift| = {mean_mag:.3f} m"
            )
            self._roof_shift = (mx, my, mz)
            self._roof_apply_btn.setEnabled(True)
            self._roof_apply_btn.setText(
                f"⬆ Apply XYZ Shift ({mx:+.3f}, {my:+.3f}, {mz:+.3f}) m to Current Tile"
            )
            self._vis_mode = "roofs"
            self._vis_index = 0
            self._update_vis_nav()
        else:
            self._roof_stats.setText("No valid results — check search radius")
            self._roof_apply_btn.setEnabled(False)

        self._status.setText(f"Roof calculation done: {len(dxs)} valid surfaces")

    # ── Apply shift ──────────────────────────────────────────────────

    def _on_apply_gcp(self):
        if self._gcp_shift is not None:
            reply = QMessageBox.question(
                self, "Apply Z Shift",
                f"Apply a Z shift of {self._gcp_shift:+.3f} m to all points "
                f"in the current tile?\n\n"
                f"This will add {self._gcp_shift:+.3f} m to every point's Z coordinate.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.shift_applied.emit(0.0, 0.0, self._gcp_shift)

    def _on_apply_roofs(self):
        if self._roof_shift is not None:
            dx, dy, dz = self._roof_shift
            mag = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            reply = QMessageBox.question(
                self, "Apply XYZ Shift",
                f"Apply the 3-D shift to all points in the current tile?\n\n"
                f"  ΔX = {dx:+.3f} m\n"
                f"  ΔY = {dy:+.3f} m\n"
                f"  ΔZ = {dz:+.3f} m\n"
                f"  |Shift| = {mag:.3f} m\n\n"
                f"This will translate every point by (ΔX, ΔY, ΔZ).",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.shift_applied.emit(dx, dy, dz)

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
            if hasattr(self, "_vis_use_chk"):
                self._vis_use_chk.setVisible(False)
            cx = item.get("centroid_x", 0)
            cy = item.get("centroid_y", 0)
            dx = item.get("dx")
            dy = item.get("dy")
            dz_val = item.get("dz")
            if dx is not None:
                shift_str = f"Δ=({dx:+.2f}, {dy:+.2f}, {dz_val:+.2f}) m"
            else:
                shift_str = "N/A"
            self._vis_label.setText(
                f"[{self._vis_index + 1}/{n}]  {name}  @ ({cx:.2f}, {cy:.2f})  {shift_str}"
            )

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
