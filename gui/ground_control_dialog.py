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
            for name, gx, gy, gz in points:
                results.append({
                    "name": name, "x": gx, "y": gy, "z_in": gz,
                    "z_cloud": None, "dz": None,
                    "n_nearby": 0, "warning": "No points of selected classes",
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

            nearby_z = cz[indices]
            z_median = float(np.median(nearby_z))
            dz = gz - z_median

            results.append({
                "name": name, "x": gx, "y": gy, "z_in": gz,
                "z_cloud": z_median, "dz": dz,
                "n_nearby": n_nearby,
                "z_std": float(np.std(nearby_z)),
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
            })

        self.finished_roofs.emit(results)

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
    Dialog for ground control point and roof-surface adjustment.

    Signals
    -------
    shift_applied(float, float, float):
        Emitted when the user clicks *Apply Shift*, carrying the
        signed (dx, dy, dz) shift in metres to add to the point cloud.
        For GCP (Z-only), dx=0, dy=0.
    visualize_point(float, float, str):
        Emitted when the user clicks *Go To* in the visual-check
        section: *(x, y, label)*.
    """

    shift_applied = Signal(float, float, float)
    visualize_point = Signal(float, float, str)

    SEPARATORS = {
        "Comma (,)": ",",
        "Semicolon (;)": ";",
        "Tab": "\t",
        "Space": " ",
    }

    def __init__(self, tile_data: dict, parent=None, data_epsg: Optional[int] = None):
        super().__init__(parent)
        self._data = tile_data
        self._data_epsg = data_epsg  # EPSG code of the point cloud data
        self._worker: Optional[_GroundControlWorker] = None

        # GCP state
        self._gcp_points: List[Tuple[str, float, float, float]] = []
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

        self._vis_goto_btn = QPushButton("🔍 Go To in 3-D View")
        self._vis_goto_btn.clicked.connect(self._on_vis_goto)
        vis_layout.addWidget(self._vis_goto_btn)

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

        # ── GCP CRS ──
        crs_group = QGroupBox("Coordinate System")
        crs_f = QFormLayout(crs_group)
        crs_row = QHBoxLayout()
        crs_row.addWidget(QLabel("GCP EPSG:"))
        self._gcp_epsg_spin = QSpinBox()
        self._gcp_epsg_spin.setRange(1000, 99999)
        self._gcp_epsg_spin.setSpecialValueText("Auto (same as data)")
        self._gcp_epsg_spin.setValue(0)
        self._gcp_epsg_spin.setToolTip(
            "EPSG code of the GCP coordinates. Set to 0 to use the same as data. "
            "If different from the data CRS, coordinates will be automatically transformed."
        )
        crs_row.addWidget(self._gcp_epsg_spin)
        crs_row.addStretch()
        crs_f.addRow("", crs_row)
        self._gcp_crs_info = QLabel("")
        self._gcp_crs_info.setWordWrap(True)
        crs_f.addRow(self._gcp_crs_info)
        layout.addWidget(crs_group)

        # ── Column mapping ──
        map_group = QGroupBox("2. Column Mapping")
        mf = QFormLayout(map_group)

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
        col_row.addStretch()
        mf.addRow("Columns:", col_row)

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
        # Default: check Ground (2) and Building (6)
        for i in range(self._gcp_class_list.count()):
            code = self._gcp_class_list.item(i).data(Qt.UserRole)
            if code in (2, 6):
                self._gcp_class_list.item(i).setCheckState(Qt.Checked)
        cls_layout.addWidget(self._gcp_class_list)
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
        self._gcp_table = QTableWidget(0, 7)
        self._gcp_table.setHorizontalHeaderLabels(
            ["Name", "X", "Y", "Z Input", "Z Cloud", "ΔZ", "Nearby pts"]
        )
        self._gcp_table.horizontalHeader().setStretchLastSection(True)
        self._gcp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._gcp_table.setSelectionBehavior(QTableWidget.SelectRows)
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
            cb.clear()
            cb.addItem("(none)", -1)
        for i, col in enumerate(header):
            col_clean = col.strip().strip('"').strip("'")
            name_combo.addItem(col_clean, i)
            x_combo.addItem(col_clean, i)
            y_combo.addItem(col_clean, i)
            z_combo.addItem(col_clean, i)
        # Auto-detect
        for i, col in enumerate(header):
            low = col.strip().lower().strip('"').strip("'")
            if low in ("name", "id", "label", "point", "point_name"):
                name_combo.setCurrentIndex(i + 1)
            elif low in ("x", "easting", "east", "lon", "longitude"):
                x_combo.setCurrentIndex(i + 1)
            elif low in ("y", "northing", "north", "lat", "latitude"):
                y_combo.setCurrentIndex(i + 1)
            elif low in ("z", "elev", "elevation", "height", "alt", "altitude"):
                z_combo.setCurrentIndex(i + 1)

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
                float(f.strip().replace(",", "."))
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
        # If the first line looks like numeric data, it's probably NOT a header
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
            # Generate synthetic header: "Col 0", "Col 1", ...
            n_cols = max(len(fl) for fl in parsed_lines)
            header = [f"Col {i}" for i in range(n_cols)]
            data_lines = parsed_lines  # all lines are data

        self._populate_column_combos(
            header,
            self._gcp_name_col, self._gcp_x_col,
            self._gcp_y_col, self._gcp_z_col,
        )

        # If auto-detection failed (no X/Y/Z columns matched), use positional defaults
        if self._gcp_x_col.currentData() is None or self._gcp_x_col.currentData() < 0:
            n_cols = len(header)
            if n_cols >= 4:
                # Standard layout: Name, X, Y, Z
                self._gcp_name_col.setCurrentIndex(1)  # Col 0 → Name
                self._gcp_x_col.setCurrentIndex(2)     # Col 1 → X
                self._gcp_y_col.setCurrentIndex(3)     # Col 2 → Y
                self._gcp_z_col.setCurrentIndex(4)     # Col 3 → Z
            elif n_cols == 3:
                # X, Y, Z only — name will be auto-numbered
                self._gcp_x_col.setCurrentIndex(1)     # Col 0 → X
                self._gcp_y_col.setCurrentIndex(2)     # Col 1 → Y
                self._gcp_z_col.setCurrentIndex(3)     # Col 2 → Z

        # Read points (store in memory for later use)
        self._gcp_points = []
        auto_name = 0
        for row in data_lines:
            if not row or all(c.strip() == "" for c in row):
                continue
            name_idx = self._gcp_name_col.currentData()
            x_idx = self._gcp_x_col.currentData()
            y_idx = self._gcp_y_col.currentData()
            z_idx = self._gcp_z_col.currentData()

            if x_idx is None or y_idx is None or z_idx is None or x_idx < 0 or y_idx < 0 or z_idx < 0:
                continue

            try:
                gx = float(row[x_idx])
                gy = float(row[y_idx])
                gz = float(row[z_idx])
            except (ValueError, IndexError):
                continue

            if name_idx is not None and name_idx >= 0 and name_idx < len(row):
                name = row[name_idx].strip()
            else:
                auto_name += 1
                name = str(auto_name)

            self._gcp_points.append((name, gx, gy, gz))

        self._gcp_run_btn.setEnabled(len(self._gcp_points) > 0)
        self._status.setText(f"Loaded {len(self._gcp_points)} GCPs from CSV")

        # Update CRS info label
        self._update_gcp_crs_info()

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
        data_epsg = self._data_epsg

        if not data_epsg:
            self._gcp_crs_info.setText(
                "⚠ Data CRS unknown — cannot auto-transform. "
                "Assign a CRS to the tile first (Tools → Coordinate Systems)."
            )
            self._gcp_crs_info.setStyleSheet("color: #c09853;")
        elif gcp_epsg == 0:
            self._gcp_crs_info.setText(
                f"GCP coords will be used as-is (same as data EPSG:{data_epsg})."
            )
            self._gcp_crs_info.setStyleSheet("color: #888;")
        elif gcp_epsg == data_epsg:
            self._gcp_crs_info.setText(
                f"GCP EPSG:{gcp_epsg} matches data EPSG:{data_epsg} — no transform needed."
            )
            self._gcp_crs_info.setStyleSheet("color: #5cb85c;")
        else:
            self._gcp_crs_info.setText(
                f"GCP EPSG:{gcp_epsg} → will auto-transform to data EPSG:{data_epsg}."
            )
            self._gcp_crs_info.setStyleSheet("color: #5cb85c; font-weight: bold;")

    def _get_gcp_coords_to_use(self) -> List[Tuple[str, float, float, float]]:
        """Return GCP points, transformed to data CRS if needed."""
        gcp_epsg = self._gcp_epsg_spin.value()
        data_epsg = self._data_epsg

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
                vx = float(row[x_idx])
                vy = float(row[y_idx])
                vz = float(row[z_idx])
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
        if not self._gcp_points:
            return

        # Re-parse column mapping from current combo state
        self._reparse_current_gcp()

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

        self._gcp_run_btn.setEnabled(False)
        self._status.setText("Calculating elevation differences…")
        self._progress.setVisible(True)
        self._progress.setValue(0)

        params = {
            "points": gcp_points,
            "classes": classes,
            "radius": self._gcp_radius_spin.value(),
        }

        self._worker = _GroundControlWorker(self._data, "gcp", params, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_gcp.connect(self._on_gcp_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _reparse_current_gcp(self):
        """Re-read points using current column mapping (in case user changed)."""
        if not self._gcp_csv_edit.text():
            return
        sep = self._get_separator(self._gcp_sep_combo, self._gcp_custom_sep)
        try:
            with open(self._gcp_csv_edit.text(), "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception:
            return

        # Parse lines into fields (handle spaces properly)
        raw_lines = content.splitlines()
        parsed_lines = []
        for line in raw_lines:
            if line.strip() == "":
                continue
            fields = self._split_csv_line(line, sep)
            if fields:
                parsed_lines.append(fields)

        if not parsed_lines:
            return

        # ── Header handling ──
        has_header = self._gcp_has_header.isChecked()
        if has_header:
            data_lines = parsed_lines[1:]
        else:
            data_lines = parsed_lines  # all lines are data

        self._gcp_points = []
        auto_name = 0
        for row in data_lines:
            name_idx = self._gcp_name_col.currentData()
            x_idx = self._gcp_x_col.currentData()
            y_idx = self._gcp_y_col.currentData()
            z_idx = self._gcp_z_col.currentData()
            if x_idx is None or y_idx is None or z_idx is None or x_idx < 0 or y_idx < 0 or z_idx < 0:
                continue
            try:
                gx, gy, gz = float(row[x_idx]), float(row[y_idx]), float(row[z_idx])
            except (ValueError, IndexError):
                continue
            if name_idx is not None and name_idx >= 0 and name_idx < len(row):
                name = row[name_idx].strip()
            else:
                auto_name += 1
                name = str(auto_name)
            self._gcp_points.append((name, gx, gy, gz))

    def _on_gcp_finished(self, results: list):
        self._gcp_results = results
        self._gcp_run_btn.setEnabled(True)
        self._progress.setVisible(False)

        # Populate table
        self._gcp_table.setRowCount(0)
        dzs = []
        for r in results:
            row = self._gcp_table.rowCount()
            self._gcp_table.insertRow(row)
            self._gcp_table.setItem(row, 0, QTableWidgetItem(str(r["name"])))
            self._gcp_table.setItem(row, 1, QTableWidgetItem(f"{r['x']:.3f}"))
            self._gcp_table.setItem(row, 2, QTableWidgetItem(f"{r['y']:.3f}"))
            self._gcp_table.setItem(row, 3, QTableWidgetItem(f"{r['z_in']:.3f}"))
            if r["z_cloud"] is not None:
                self._gcp_table.setItem(row, 4, QTableWidgetItem(f"{r['z_cloud']:.3f}"))
                dz_str = f"{r['dz']:+.3f}"
                self._gcp_table.setItem(row, 5, QTableWidgetItem(dz_str))
                dzs.append(r["dz"])
            else:
                self._gcp_table.setItem(row, 4, QTableWidgetItem("N/A"))
                self._gcp_table.setItem(row, 5, QTableWidgetItem(r.get("warning", "N/A")))
            self._gcp_table.setItem(row, 6, QTableWidgetItem(str(r.get("n_nearby", 0))))
        self._gcp_table.resizeColumnsToContents()

        # Statistics
        if dzs:
            dz_arr = np.array(dzs)
            median_dz = float(np.median(dz_arr))
            mean_dz = float(np.mean(dz_arr))
            std_dz = float(np.std(dz_arr))
            rms_dz = float(np.sqrt(np.mean(dz_arr ** 2)))
            self._gcp_stats.setText(
                f"<b>Statistics ({len(dzs)} points):</b>  "
                f"Median shift = {median_dz:+.3f} m  |  "
                f"Average shift = {mean_dz:+.3f} m  |  "
                f"StdDev = {std_dz:.3f} m  |  "
                f"RMS = {rms_dz:.3f} m"
            )
            self._gcp_shift = -mean_dz  # shift = opposite of mean difference
            self._gcp_apply_btn.setEnabled(True)
            self._gcp_apply_btn.setText(
                f"⬆ Apply Z Shift ({self._gcp_shift:+.3f} m) to Current Tile"
            )
            self._vis_mode = "gcp"
            self._vis_index = 0
            self._update_vis_nav()
        else:
            self._gcp_stats.setText("No valid results — check class selection and search radius")
            self._gcp_apply_btn.setEnabled(False)

        self._status.setText(f"GCP calculation done: {len(dzs)} valid, {len(results) - len(dzs)} failed")

    # ── Run Roofs ────────────────────────────────────────────────────

    def _on_run_roofs(self):
        if not self._roof_surfaces:
            return
        self._reparse_current_roofs()

        self._roof_run_btn.setEnabled(False)
        self._status.setText("Calculating surface offsets…")
        self._progress.setVisible(True)
        self._progress.setValue(0)

        params = {
            "surfaces": self._roof_surfaces,
            "radius": self._roof_radius_spin.value(),
        }
        self._worker = _GroundControlWorker(self._data, "roofs", params, parent=self)
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
                vx, vy, vz = float(row[x_idx]), float(row[y_idx]), float(row[z_idx])
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
            self._vis_label.setText(
                f"[{self._vis_index + 1}/{n}]  {name}  @ ({x:.2f}, {y:.2f})  ΔZ={dz_str}"
            )
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
            x, y = item.get("x", 0), item.get("y", 0)
            self.visualize_point.emit(x, y, item.get("name", "GCP"))
        else:
            x = item.get("centroid_x", 0)
            y = item.get("centroid_y", 0)
            self.visualize_point.emit(x, y, item.get("name", "Surface"))

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
