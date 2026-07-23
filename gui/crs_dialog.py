"""
LiDAR Workbench — CRS / Projection Dialog.

Provides CRS selection, coordinate transformation, and matching-point
parameter estimation in a single dialog.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import crs as crs_module

logger = logging.getLogger("lidar_workbench.gui.crs_dialog")

# Common EPSG codes with descriptions
_COMMON_EPSG = [
    (4326, "WGS 84 (geographic, lat/lon)"),
    (32601, "WGS 84 / UTM zone 1N"),
    (32632, "WGS 84 / UTM zone 32N"),
    (32633, "WGS 84 / UTM zone 33N"),
    (32634, "WGS 84 / UTM zone 34N"),
    (25832, "ETRS89 / UTM zone 32N"),
    (25833, "ETRS89 / UTM zone 33N"),
    (31254, "MGI / Austria GK East"),
    (31255, "MGI / Austria GK Central"),
    (31256, "MGI / Austria GK West"),
    (5514, "S-JTSK / Krovak East North"),
    (21781, "CH1903 / LV03"),
    (2056, "CH1903+ / LV95"),
]


class CrsDialog(QDialog):
    """
    Multi-tab dialog for CRS management.

    Tabs:
      - **Info**: Display current CRS, assign a new one.
      - **Transform**: Transform coordinates from source to target CRS.
      - **Match Points**: Estimate Helmert / affine parameters from
        matching point pairs.
    """

    crs_assigned = Signal(str, int)   # wkt, epsg
    transform_applied = Signal(str, str)  # source_crs, target_crs
    match_transform_applied = Signal(str, dict)  # method ("helmert"/"affine"), params

    def __init__(
        self,
        current_crs_wkt: str = "",
        current_epsg: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._current_wkt = current_crs_wkt
        self._current_epsg = current_epsg
        self._match_points_src: List[Tuple[float, float]] = []
        self._match_points_tgt: List[Tuple[float, float]] = []

        self.setWindowTitle("Coordinate Reference System")
        self.setMinimumWidth(580)
        self.setMinimumHeight(500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Current CRS info banner
        info_text = "No CRS assigned"
        if self._current_wkt:
            info_text = crs_module.get_crs_info(self._current_wkt).get("name", "Unknown")
            if self._current_epsg:
                info_text += f" (EPSG:{self._current_epsg})"
        self._info_banner = QLabel(f"<b>Current:</b> {info_text}")
        layout.addWidget(self._info_banner)

        tabs = QTabWidget()
        tabs.addTab(self._build_info_tab(), "Info")
        tabs.addTab(self._build_transform_tab(), "Transform")
        tabs.addTab(self._build_matchpoints_tab(), "Match Points")
        layout.addWidget(tabs)

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ── Info tab ──────────────────────────────────────────────────

    def _build_info_tab(self) -> QWidget:
        w = QWidget()
        layout = QFormLayout(w)

        self._epsg_combo = QComboBox()
        self._epsg_combo.setEditable(True)
        self._epsg_combo.setInsertPolicy(QComboBox.NoInsert)
        self._epsg_combo.addItem("— Auto-detect / None —", -1)
        for code, desc in _COMMON_EPSG:
            self._epsg_combo.addItem(f"EPSG:{code} — {desc}", code)
        if self._current_epsg:
            idx = self._epsg_combo.findData(self._current_epsg)
            if idx >= 0:
                self._epsg_combo.setCurrentIndex(idx)
        self._epsg_combo.currentIndexChanged.connect(self._on_epsg_selected)
        layout.addRow("Assign CRS:", self._epsg_combo)

        self._crs_detail = QLabel("")
        self._crs_detail.setWordWrap(True)
        layout.addRow("Details:", self._crs_detail)

        self._epsg_search_edit = QLineEdit()
        self._epsg_search_edit.setPlaceholderText("Search EPSG codes (e.g. 'UTM 33')…")
        self._epsg_search_edit.textChanged.connect(self._on_search)
        layout.addRow("Search:", self._epsg_search_edit)

        self._search_results = QListWidget()
        self._search_results.setVisible(False)
        self._search_results.setMaximumHeight(120)
        self._search_results.itemDoubleClicked.connect(self._on_search_result_double_clicked)
        layout.addRow(self._search_results)

        return w

    def _on_epsg_selected(self):
        epsg = self._epsg_combo.currentData()
        if epsg and epsg > 0:
            info = crs_module.get_crs_info(epsg)
            self._crs_detail.setText(
                f"Name: {info['name']}\n"
                f"Type: {info['type']}\n"
                f"Unit: {info['unit']}"
            )
        else:
            self._crs_detail.setText("")

    def _on_search(self, text: str):
        if len(text) < 2:
            self._search_results.setVisible(False)
            return
        results = crs_module.get_epsg_suggestions(text, limit=15)
        self._search_results.clear()
        for r in results:
            item = QListWidgetItem(f"EPSG:{r['epsg']} — {r['name']}  [{r['type']}]")
            item.setData(Qt.UserRole, r['epsg'])
            self._search_results.addItem(item)
        self._search_results.setVisible(len(results) > 0)

    def _on_search_result_double_clicked(self, item: QListWidgetItem):
        epsg = item.data(Qt.UserRole)
        if epsg:
            idx = self._epsg_combo.findData(epsg)
            if idx < 0:
                name = item.text().split(" — ", 1)[1].split("  [")[0] if " — " in item.text() else item.text()
                self._epsg_combo.addItem(item.text(), epsg)
                idx = self._epsg_combo.count() - 1
            self._epsg_combo.setCurrentIndex(idx)
            self._search_results.setVisible(False)

    # ── Transform tab ─────────────────────────────────────────────

    def _build_transform_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Source
        src_group = QGroupBox("Source CRS")
        sf = QFormLayout(src_group)
        self._src_epsg_combo = QComboBox()
        self._src_epsg_combo.setEditable(True)
        self._src_epsg_combo.addItem("(auto from data)", -1)
        for code, desc in _COMMON_EPSG:
            self._src_epsg_combo.addItem(f"EPSG:{code} — {desc}", code)
        sf.addRow("Source:", self._src_epsg_combo)
        layout.addWidget(src_group)

        # Target
        tgt_group = QGroupBox("Target CRS")
        tf = QFormLayout(tgt_group)
        self._tgt_epsg_combo = QComboBox()
        self._tgt_epsg_combo.setEditable(True)
        for code, desc in _COMMON_EPSG:
            self._tgt_epsg_combo.addItem(f"EPSG:{code} — {desc}", code)
        self._tgt_epsg_combo.setCurrentIndex(0)
        tf.addRow("Target:", self._tgt_epsg_combo)
        layout.addWidget(tgt_group)

        self._transform_info = QLabel(
            "Transform will be applied to all selected tiles. "
            "Coordinates (X, Y, Z) are reprojected using pyproj."
        )
        self._transform_info.setWordWrap(True)
        layout.addWidget(self._transform_info)

        layout.addStretch()
        return w

    # ── Match Points tab ──────────────────────────────────────────

    def _build_matchpoints_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Info
        info = QLabel(
            "Enter matching 3-D point pairs to estimate a Helmert "
            "(7-parameter similarity) or affine (12-parameter) "
            "transformation.\n\n"
            "At least 3 pairs for Helmert, 4 for affine.\n"
            "Points are expected in the same CRS units (metres).\n"
            "The Z column can be left at 0 for 2-D-only data."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Table — 6 columns for 3-D
        self._match_table = QTableWidget(0, 6)
        self._match_table.setHorizontalHeaderLabels(
            ["Src X", "Src Y", "Src Z", "Tgt X", "Tgt Y", "Tgt Z"]
        )
        self._match_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._match_table)

        # Buttons
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Row")
        add_btn.clicked.connect(self._on_add_match_row)
        btn_row.addWidget(add_btn)
        del_btn = QPushButton("- Remove Selected")
        del_btn.clicked.connect(self._on_del_match_row)
        btn_row.addWidget(del_btn)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_match_rows)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Estimate buttons
        est_row = QHBoxLayout()
        self._helmert_btn = QPushButton("Estimate Helmert (7-param 3D)")
        self._helmert_btn.clicked.connect(lambda: self._on_estimate("helmert"))
        est_row.addWidget(self._helmert_btn)
        self._affine_btn = QPushButton("Estimate Affine (12-param 3D)")
        self._affine_btn.clicked.connect(lambda: self._on_estimate("affine"))
        est_row.addWidget(self._affine_btn)
        layout.addLayout(est_row)

        # Apply button (enabled after estimation)
        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("▶ Apply to Selected Tiles")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 12px; }"
        )
        self._apply_btn.clicked.connect(self._on_apply_match)
        apply_row.addStretch()
        apply_row.addWidget(self._apply_btn)
        layout.addLayout(apply_row)

        # Results
        self._match_result = QLabel("")
        self._match_result.setWordWrap(True)
        layout.addWidget(self._match_result)

        # Add a few initial rows
        for _ in range(3):
            self._on_add_match_row()

        return w

    def _on_add_match_row(self):
        row = self._match_table.rowCount()
        self._match_table.insertRow(row)
        for col in range(6):
            self._match_table.setItem(row, col, QTableWidgetItem("0.0"))

    def _on_del_match_row(self):
        rows = set()
        for item in self._match_table.selectedItems():
            rows.add(item.row())
        for row in sorted(rows, reverse=True):
            self._match_table.removeRow(row)

    def _on_clear_match_rows(self):
        self._match_table.setRowCount(0)

    def _on_estimate(self, method: str):
        src_pts, tgt_pts = [], []
        for row in range(self._match_table.rowCount()):
            try:
                sx = float(self._match_table.item(row, 0).text())
                sy = float(self._match_table.item(row, 1).text())
                sz = float(self._match_table.item(row, 2).text())
                tx = float(self._match_table.item(row, 3).text())
                ty = float(self._match_table.item(row, 4).text())
                tz = float(self._match_table.item(row, 5).text())
                src_pts.append([sx, sy, sz])
                tgt_pts.append([tx, ty, tz])
            except (ValueError, AttributeError):
                continue

        min_required = 3 if method == "helmert" else 4
        if len(src_pts) < min_required:
            self._match_result.setText(
                f"⚠ Need at least {min_required} valid point pairs "
                f"(got {len(src_pts)})."
            )
            return

        try:
            src_arr = np.array(src_pts, dtype=np.float64)
            tgt_arr = np.array(tgt_pts, dtype=np.float64)

            if method == "helmert":
                result = crs_module.estimate_helmert_3d(src_arr, tgt_arr)
                rx, ry, rz = result["rotation_xyz_deg"]
                text = (
                    f"<b>Helmert 7-Parameter (3-D Similarity)</b>\n"
                    f"Scale:       {result['scale']:.8f}\n"
                    f"Rotation X:  {rx:.6f}°\n"
                    f"Rotation Y:  {ry:.6f}°\n"
                    f"Rotation Z:  {rz:.6f}°\n"
                    f"Shift X:     {result['tx']:.4f} m\n"
                    f"Shift Y:     {result['ty']:.4f} m\n"
                    f"Shift Z:     {result['tz']:.4f} m\n"
                    f"RMSE:        {result['rmse']:.4f} m"
                )
            else:
                result = crs_module.estimate_affine_3d(src_arr, tgt_arr)
                p = result["params"]
                text = (
                    f"<b>Affine 12-Parameter (3-D)</b>\n"
                    f"X' = {p['a1']:.6f}·X + {p['a2']:.6f}·Y + {p['a3']:.6f}·Z + {p['a4']:.4f}\n"
                    f"Y' = {p['b1']:.6f}·X + {p['b2']:.6f}·Y + {p['b3']:.6f}·Z + {p['b4']:.4f}\n"
                    f"Z' = {p['c1']:.6f}·X + {p['c2']:.6f}·Y + {p['c3']:.6f}·Z + {p['c4']:.4f}\n"
                    f"RMSE:  {result['rmse']:.4f} m"
                )

            # Accuracy stats
            res_list = result.get("residuals", [])
            if res_list:
                res_arr = np.array(res_list)
                p68 = float(np.percentile(res_arr, 68))
                p95 = float(np.percentile(res_arr, 95))
                p99 = float(np.percentile(res_arr, 99))
                text += (
                    f"\n<b>Accuracy:</b>"
                    f"\n  RMS = {result['rmse']:.4f} m"
                    f"\n  68% ≤ {p68:.4f} m"
                    f"\n  95% ≤ {p95:.4f} m"
                    f"\n  99% ≤ {p99:.4f} m"
                    f"\n  Max = {res_arr.max():.4f} m  (n={len(res_arr)})"
                )

            self._match_result.setText(text)
            self._last_estimate = result
            self._last_estimate_method = method
            self._apply_btn.setEnabled(True)
        except Exception as exc:
            self._match_result.setText(f"⚠ Estimation failed: {exc}")

    def _on_apply_match(self):
        """Emit the match transform signal so the main window applies it."""
        if self._last_estimate is not None:
            self.match_transform_applied.emit(
                self._last_estimate_method,
                self._last_estimate,
            )

    # ── Accept ────────────────────────────────────────────────────

    def _on_accept(self):
        # Info tab: assign CRS
        epsg = self._epsg_combo.currentData()
        if epsg and epsg > 0:
            wkt = crs_module.epsg_to_wkt(epsg) or ""
            self.crs_assigned.emit(wkt, epsg)

        # Transform tab: signal intent
        src_epsg = self._src_epsg_combo.currentData()
        tgt_epsg = self._tgt_epsg_combo.currentData()
        if src_epsg and src_epsg > 0 and tgt_epsg and tgt_epsg > 0:
            self.transform_applied.emit(str(src_epsg), str(tgt_epsg))

        self.accept()

    @property
    def assigned_epsg(self) -> Optional[int]:
        epsg = self._epsg_combo.currentData()
        return epsg if epsg and epsg > 0 else None

    @property
    def assigned_wkt(self) -> Optional[str]:
        epsg = self._epsg_combo.currentData()
        if epsg and epsg > 0:
            return crs_module.epsg_to_wkt(epsg)
        return None

    @property
    def source_epsg(self) -> Optional[int]:
        v = self._src_epsg_combo.currentData()
        return v if v and v > 0 else None

    @property
    def target_epsg(self) -> Optional[int]:
        v = self._tgt_epsg_combo.currentData()
        return v if v and v > 0 else None

    @property
    def last_estimate(self) -> Optional[dict]:
        return getattr(self, "_last_estimate", None)

    @property
    def last_estimate_method(self) -> str:
        return getattr(self, "_last_estimate_method", "")
