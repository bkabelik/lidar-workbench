"""
LiDAR Strip Adjustment Dialog (StripAdj).

Provides a comprehensive GUI for flightline/strip alignment:
- Auto-detects strips (point_source_id) across project tiles.
- Trajectory matching (auto-matches OPALS / Riegl / ASCII trajectories by timestamp).
- 3-Step adjustment execution:
    * Step 1: Bulk XYZ translation.
    * Step 2: Boresight angles (Roll, Pitch, Yaw).
    * Step 3: Dynamic trajectory drift (smooth spline).
- Before/after RMSE diagnostics and application to project tiles.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..strip_adjustment import (
    StripAdjustmentSolver,
    StripCorrection,
    TieSurface,
    Trajectory,
    apply_strip_corrections,
    extract_tie_planes,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Background Worker Thread
# ═══════════════════════════════════════════════════════════════════════

class _StripAdjustmentWorker(QThread):
    """Worker thread for non-blocking overlap extraction across tiles and 3-step solving."""

    progress = Signal(str, float)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        strip_ids: Sequence[int],
        trajectories: Dict[int, Trajectory],
        step1_enabled: bool,
        step2_enabled: bool,
        step3_enabled: bool,
        ref_strip_id: Optional[int] = None,
        time_knot_interval: float = 20.0,
        tile_manager=None,
        tile_ids: Optional[Sequence[str]] = None,
        strip_data_map: Optional[Dict[int, dict]] = None,
        strip_meta: Optional[Dict[int, dict]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.strip_ids = sorted(list(strip_ids))
        self.trajectories = trajectories
        self.step1_enabled = step1_enabled
        self.step2_enabled = step2_enabled
        self.step3_enabled = step3_enabled
        self.ref_strip_id = ref_strip_id
        self.time_knot_interval = time_knot_interval
        self.tile_manager = tile_manager
        self.tile_ids = list(tile_ids) if tile_ids else []
        self.strip_data_map = strip_data_map
        self.strip_meta = strip_meta or {}

    def run(self):
        try:
            n_strips = len(self.strip_ids)
            if n_strips < 2:
                self.error.emit("At least two strips are required for strip adjustment.")
                return

            # Determine reference strip (if not set, choose strip with most points)
            if self.ref_strip_id is not None and self.ref_strip_id in self.strip_ids:
                ref_id = self.ref_strip_id
            elif self.strip_meta:
                pts_counts = {sid: self.strip_meta.get(sid, {}).get("count", 0) for sid in self.strip_ids}
                ref_id = max(pts_counts, key=pts_counts.get)
            elif self.strip_data_map:
                pts_counts = {sid: len(self.strip_data_map[sid]["x"]) for sid in self.strip_ids}
                ref_id = max(pts_counts, key=pts_counts.get)
            else:
                ref_id = self.strip_ids[0]

            self.progress.emit("Extracting planar tie-surfaces across project tiles…", 5.0)

            all_tie_surfaces: List[TieSurface] = []

            # ── Mode A: Tile-native streaming extraction across disk tiles ──
            if self.tile_manager is not None and len(self.tile_ids) > 0:
                n_tiles = len(self.tile_ids)

                for idx, tid in enumerate(self.tile_ids):
                    pct = 5.0 + (idx / max(1, n_tiles)) * 50.0
                    self.progress.emit(
                        f"Analyzing overlap in tile {idx+1}/{n_tiles} ({tid})…", pct
                    )

                    data = self.tile_manager.load_tile_points_full(tid)
                    if not data or len(data.get("x", [])) == 0:
                        continue

                    psids = data.get("point_source_id")
                    if psids is None or len(psids) == 0:
                        continue

                    unique_in_tile = np.unique(psids)
                    present = [sid for sid in self.strip_ids if sid in unique_in_tile]
                    if len(present) < 2:
                        continue

                    xs = data["x"]
                    ys = data["y"]
                    zs = data["z"]
                    ts = data.get("gps_time", np.zeros_like(zs))

                    # Split points by strip in this tile
                    tile_subsets = {}
                    for sid in present:
                        mask = (psids == sid)
                        tile_subsets[sid] = {
                            "x": xs[mask],
                            "y": ys[mask],
                            "z": zs[mask],
                            "gps_time": ts[mask],
                        }

                    # Extract tie surfaces for all active strip pairs in this tile
                    for i in range(len(present)):
                        for j in range(i + 1, len(present)):
                            s1, s2 = present[i], present[j]
                            ties = extract_tie_planes(
                                tile_subsets[s1],
                                tile_subsets[s2],
                                s1, s2,
                                patch_radius=1.5,
                                min_patch_pts=15,
                                max_roughness=0.04,
                                max_samples=250,
                            )
                            if ties:
                                all_tie_surfaces.extend(ties)

            # ── Mode B: In-memory fallback (used in standalone unit tests) ─
            elif self.strip_data_map is not None:
                pairs = []
                for i in range(n_strips):
                    for j in range(i + 1, n_strips):
                        pairs.append((self.strip_ids[i], self.strip_ids[j]))

                total_pairs = len(pairs)
                for idx, (s1, s2) in enumerate(pairs):
                    pct = 5.0 + (idx / max(1, total_pairs)) * 50.0
                    self.progress.emit(f"Checking overlap between Strip {s1} and Strip {s2}…", pct)

                    ties = extract_tie_planes(
                        self.strip_data_map[s1],
                        self.strip_data_map[s2],
                        s1, s2,
                        patch_radius=1.5,
                        min_patch_pts=15,
                        max_roughness=0.04,
                    )
                    if ties:
                        all_tie_surfaces.extend(ties)

            if not all_tie_surfaces:
                self.error.emit(
                    "No valid planar overlap patches found between the active flightline strips.\n\n"
                    "Ensure that the survey flightlines have spatial overlap and contain stable ground or roofs."
                )
                return

            init_residuals = [t.residual for t in all_tie_surfaces]
            initial_rmse = math.sqrt(float(np.mean(np.square(init_residuals))))
            initial_mean_dz = float(np.mean(init_residuals))

            # Setup Solver
            solver = StripAdjustmentSolver(
                strip_ids=self.strip_ids,
                trajectories=self.trajectories,
                ref_strip_id=ref_id,
            )

            shifts = {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}
            angles = {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}
            drift_nodes = {sid: [] for sid in self.strip_ids}

            # Step 1: Bulk XYZ Shift
            if self.step1_enabled:
                self.progress.emit("Solving Step 1: Global rigid XYZ translation…", 60.0)
                shifts = solver.solve_step1_xyz(all_tie_surfaces)

            # Step 2: Roll, Pitch, Yaw Boresight
            if self.step2_enabled:
                self.progress.emit("Solving Step 2: Boresight angles (Roll, Pitch, Yaw)…", 75.0)
                angles = solver.solve_step2_boresight(all_tie_surfaces, current_shifts=shifts)

            # Step 3: Dynamic Trajectory Drift
            if self.step3_enabled:
                self.progress.emit("Solving Step 3: Trajectory drift spline…", 88.0)
                drift_nodes = solver.solve_step3_drift(
                    all_tie_surfaces,
                    current_shifts=shifts,
                    current_boresight=angles,
                    time_knot_interval=self.time_knot_interval,
                )

            # Assemble corrections and evaluate final post-adjustment residuals
            corrections: Dict[int, StripCorrection] = {}
            for sid in self.strip_ids:
                dx, dy, dz = shifts.get(sid, (0.0, 0.0, 0.0))
                dr, dp, dyaw = angles.get(sid, (0.0, 0.0, 0.0))
                dnodes = drift_nodes.get(sid, [])
                corrections[sid] = StripCorrection(
                    strip_id=sid,
                    dx=dx, dy=dy, dz=dz,
                    droll=dr, dpitch=dp, dyaw=dyaw,
                    drift_nodes=dnodes,
                )

            self.progress.emit("Evaluating post-adjustment overlap residuals…", 95.0)

            final_residuals = []
            for ts in all_tie_surfaces:
                ca = corrections[ts.strip_a]
                cb = corrections[ts.strip_b]

                pa_adj = ts.point_a + np.array([ca.dx, ca.dy, ca.dz])
                cb_adj = ts.centroid_b + np.array([cb.dx, cb.dy, cb.dz])

                res_adj = float(np.dot(pa_adj - cb_adj, ts.normal_b))
                final_residuals.append(res_adj)

            final_rmse = math.sqrt(float(np.mean(np.square(final_residuals))))
            final_mean_dz = float(np.mean(final_residuals))

            self.progress.emit("Strip adjustment complete.", 100.0)

            results = {
                "strip_ids": self.strip_ids,
                "ref_strip_id": ref_id,
                "corrections": corrections,
                "total_ties": len(all_tie_surfaces),
                "initial_rmse": initial_rmse,
                "initial_mean_dz": initial_mean_dz,
                "final_rmse": final_rmse,
                "final_mean_dz": final_mean_dz,
            }
            self.finished.emit(results)

        except Exception as exc:
            logger.exception("Error in strip adjustment worker")
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════
# Main Strip Adjustment Dialog
# ═══════════════════════════════════════════════════════════════════════

class StripAdjustmentDialog(QDialog):
    """Interactive dialog for Strip Adjustment across tiled point clouds."""

    adjustment_applied = Signal(dict)

    def __init__(
        self,
        tile_manager=None,
        database=None,
        project_dir: Optional[str] = None,
        tile_ids: Optional[Sequence[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("LiDAR Strip Adjustment (StripAdj)")
        self.resize(1060, 750)

        self._tm = tile_manager
        self._db = database
        self._project_dir = project_dir or "."
        self._selected_tile_ids: Optional[List[str]] = list(tile_ids) if tile_ids else None
        self._all_project_tile_ids: List[str] = []

        self._strip_meta: Dict[int, dict] = {}
        self._strip_data_map: Dict[int, dict] = {}
        self._trajectories: Dict[int, Trajectory] = {}
        self._worker: Optional[_StripAdjustmentWorker] = None
        self._solved_results: Optional[dict] = None

        self._init_ui()
        self._load_project_strips()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # ── Group 1: Strips & Trajectory Assignment ───────────────────
        strip_grp = QGroupBox("1. Detected Flightlines / Strips & Trajectory Assignment")
        strip_layout = QVBoxLayout(strip_grp)

        # Scope row (All tiles vs Selected tiles)
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Tile Processing Scope:"))
        self._scope_combo = QComboBox()
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        scope_row.addWidget(self._scope_combo)
        scope_row.addStretch()
        strip_layout.addLayout(scope_row)

        self._strip_table = QTableWidget()
        self._strip_table.setColumnCount(8)
        self._strip_table.setHorizontalHeaderLabels([
            "Use", "Strip ID", "Sensor", "Points (approx)", "GPS Time Range", "Spanning Tiles", "Trajectory File", "Trajectory Pts"
        ])
        self._strip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._strip_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._strip_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._strip_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._strip_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._strip_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._strip_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self._strip_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self._strip_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        strip_layout.addWidget(self._strip_table)

        btn_row = QHBoxLayout()
        self._auto_match_btn = QPushButton("Auto-Match Trajectories in Project")
        self._auto_match_btn.setToolTip("Search project folder for trajectory files (.txt, .csv) covering strip GPS times")
        self._auto_match_btn.clicked.connect(self._on_auto_match_trajectories)
        btn_row.addWidget(self._auto_match_btn)

        self._browse_traj_btn = QPushButton("Browse Trajectory for Selected Strip…")
        self._browse_traj_btn.clicked.connect(self._on_browse_trajectory)
        btn_row.addWidget(self._browse_traj_btn)

        self._clear_traj_btn = QPushButton("Clear Trajectory")
        self._clear_traj_btn.clicked.connect(self._on_clear_trajectory)
        btn_row.addWidget(self._clear_traj_btn)

        btn_row.addStretch()
        strip_layout.addLayout(btn_row)
        main_layout.addWidget(strip_grp)

        # ── Group 2: Correction Steps & Parameters ────────────────────
        steps_grp = QGroupBox("2. Adjustment Steps & Datum Constraints")
        steps_layout = QHBoxLayout(steps_grp)

        v_steps = QVBoxLayout()
        self._step1_chk = QCheckBox("Step 1: Bulk XYZ Translation (lever-arm / datum offset)")
        self._step1_chk.setChecked(True)
        v_steps.addWidget(self._step1_chk)

        self._step2_chk = QCheckBox("Step 2: Roll, Pitch, Yaw Boresight (mounting angles around scanner center)")
        self._step2_chk.setChecked(True)
        v_steps.addWidget(self._step2_chk)

        self._step3_chk = QCheckBox("Step 3: Dynamic Trajectory Drift (time-dependent smooth spline)")
        self._step3_chk.setChecked(False)
        v_steps.addWidget(self._step3_chk)
        steps_layout.addLayout(v_steps, 2)

        v_params = QVBoxLayout()
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference Strip (Fixed):"))
        self._ref_strip_combo = QComboBox()
        ref_row.addWidget(self._ref_strip_combo)
        v_params.addLayout(ref_row)

        knot_row = QHBoxLayout()
        knot_row.addWidget(QLabel("Step 3 Drift Knot Interval:"))
        self._knot_spin = QDoubleSpinBox()
        self._knot_spin.setRange(5.0, 120.0)
        self._knot_spin.setValue(20.0)
        self._knot_spin.setSuffix(" s")
        knot_row.addWidget(self._knot_spin)
        v_params.addLayout(knot_row)
        steps_layout.addLayout(v_params, 1)

        main_layout.addWidget(steps_grp)

        # ── Group 3: Overlap Diagnostics & Execution ──────────────────
        diag_grp = QGroupBox("3. Overlap Analysis & Execution")
        diag_layout = QVBoxLayout(diag_grp)

        self._diag_label = QLabel("No adjustment calculated yet. Click 'Calculate Strip Adjustment' to analyze overlaps.")
        self._diag_label.setWordWrap(True)
        diag_layout.addWidget(self._diag_label)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(7)
        self._results_table.setHorizontalHeaderLabels([
            "Strip ID", "ΔX (m)", "ΔY (m)", "ΔZ (m)", "ΔRoll (°)", "ΔPitch (°)", "ΔYaw (°)"
        ])
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        diag_layout.addWidget(self._results_table)

        run_row = QHBoxLayout()
        self._calc_btn = QPushButton("▶ Calculate Strip Adjustment")
        self._calc_btn.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        self._calc_btn.clicked.connect(self._on_calculate)
        run_row.addWidget(self._calc_btn)

        self._apply_btn = QPushButton("✔ Apply Adjustment to Project Tiles")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStyleSheet("font-weight: bold; padding: 6px 14px; background-color: #2e7d32; color: white;")
        self._apply_btn.clicked.connect(self._on_apply_adjustment)
        run_row.addWidget(self._apply_btn)

        run_row.addStretch()
        diag_layout.addLayout(run_row)
        main_layout.addWidget(diag_grp)

        # Progress bar & status
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        main_layout.addWidget(self._progress)

        self._status_label = QLabel("Ready")
        main_layout.addWidget(self._status_label)

        # Bottom buttons
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        bottom_row.addWidget(close_btn)
        main_layout.addLayout(bottom_row)

    # ── Load project data ─────────────────────────────────────────────

    def _get_active_tile_ids(self) -> List[str]:
        """Return tile IDs corresponding to the current scope combo selection."""
        idx = self._scope_combo.currentIndex()
        if idx == 1 and self._selected_tile_ids:
            return self._selected_tile_ids
        return self._all_project_tile_ids

    def _on_scope_changed(self):
        self._scan_tiles_for_strips()
        self._populate_strip_table()
        self._on_auto_match_trajectories(silent=True)

    def _load_project_strips(self):
        """Discover all project tiles and populate the scope selector."""
        # 1. Discover all tile IDs
        if self._tm:
            all_tids = self._tm.tile_ids()
        elif self._db:
            all_tids = [t["id"] for t in self._db.get_all_tiles()]
        else:
            all_tids = []
            if self._project_dir:
                tdir = Path(self._project_dir) / "tiles"
                if tdir.is_dir():
                    all_tids = [p.stem for p in sorted(tdir.glob("*.las")) if not p.name.endswith(".bak")]

        self._all_project_tile_ids = all_tids

        # 2. Setup scope combo
        self._scope_combo.blockSignals(True)
        self._scope_combo.clear()
        self._scope_combo.addItem(f"All Project Tiles ({len(all_tids)} tiles)", "all")
        if self._selected_tile_ids and len(self._selected_tile_ids) < len(all_tids):
            self._scope_combo.addItem(f"Selected Tiles ({len(self._selected_tile_ids)} tiles)", "selected")
            self._scope_combo.setCurrentIndex(1)
        else:
            self._scope_combo.setCurrentIndex(0)
        self._scope_combo.blockSignals(False)

        self._scan_tiles_for_strips()
        self._populate_strip_table()
        self._on_auto_match_trajectories(silent=True)

    def _scan_tiles_for_strips(self):
        """Fast sample scan across target tiles to discover flightlines, sensor types, and GPS bounds."""
        target_tids = self._get_active_tile_ids()
        self._strip_meta.clear()

        if not target_tids:
            return

        # Query database for strip sensor mapping if available
        strip_sensors: Dict[int, str] = {}
        if self._db:
            try:
                with self._db.connect() as conn:
                    rows = conn.execute("SELECT flightline_sensor_types FROM tiles").fetchall()
                    for (fl_json,) in rows:
                        if not fl_json:
                            continue
                        import json
                        mapping = json.loads(fl_json) if isinstance(fl_json, str) else fl_json
                        for sid_str, stype in mapping.items():
                            strip_sensors[int(sid_str)] = stype
            except Exception:
                pass

        import laspy

        for tid in target_tids:
            tile_path = self._tm.tile_las_path(tid) if self._tm else None
            if not tile_path or not os.path.exists(tile_path):
                direct = Path(self._project_dir) / "tiles" / f"{tid}.las"
                if direct.is_file():
                    tile_path = direct
                else:
                    continue

            try:
                with laspy.open(tile_path) as r:
                    n_pts = r.header.point_count
                    if n_pts == 0:
                        continue

                    # Sample points to inspect point_source_id and gps_time quickly
                    sample_n = min(n_pts, 50000)
                    chunk = r.read_points(sample_n)
                    sids = np.unique(chunk.point_source_id)
                    gt = chunk.gps_time if hasattr(chunk, "gps_time") else np.zeros(sample_n)

                    for sid in sids:
                        sid_int = int(sid)
                        if sid_int not in self._strip_meta:
                            # Sensor identification
                            stype = strip_sensors.get(sid_int)
                            if not stype:
                                stype = "bathy" if sid_int >= 7 else "topo"

                            self._strip_meta[sid_int] = {
                                "count": 0,
                                "sensor": stype,
                                "t_min": float("inf"),
                                "t_max": float("-inf"),
                                "tiles": set(),
                            }

                        mask = chunk.point_source_id == sid
                        pts_in_sample = int(mask.sum())
                        est_count = int(round((pts_in_sample / sample_n) * n_pts))
                        self._strip_meta[sid_int]["count"] += est_count
                        self._strip_meta[sid_int]["tiles"].add(tid)

                        t_sub = gt[mask]
                        if len(t_sub) > 0 and t_sub.max() > 0:
                            self._strip_meta[sid_int]["t_min"] = min(
                                self._strip_meta[sid_int]["t_min"], float(t_sub.min())
                            )
                            self._strip_meta[sid_int]["t_max"] = max(
                                self._strip_meta[sid_int]["t_max"], float(t_sub.max())
                            )

            except Exception:
                continue

    def _populate_strip_table(self):
        self._strip_table.setRowCount(0)
        self._ref_strip_combo.clear()

        # If in-memory strip_data_map is present (e.g. unit tests)
        if not self._strip_meta and self._strip_data_map:
            for sid, data in self._strip_data_map.items():
                ts = data.get("gps_time", np.zeros(len(data["x"])))
                self._strip_meta[sid] = {
                    "count": len(data["x"]),
                    "sensor": "topo" if sid <= 6 else "bathy",
                    "t_min": float(ts.min()) if len(ts) and ts.max() > 0 else 0.0,
                    "t_max": float(ts.max()) if len(ts) and ts.max() > 0 else 0.0,
                    "tiles": {"memory"},
                }

        sorted_sids = sorted(self._strip_meta.keys())
        self._strip_table.setRowCount(len(sorted_sids))

        for row, sid in enumerate(sorted_sids):
            meta = self._strip_meta[sid]
            n_pts = meta["count"]
            t_min = meta["t_min"]
            t_max = meta["t_max"]
            t_span_str = f"{t_min:.1f} – {t_max:.1f}" if t_max > 0 and t_min != float("inf") else "N/A"
            n_tiles_str = f"{len(meta['tiles']):,} tiles"

            stype = meta.get("sensor", "").lower()
            if "bathy" in stype:
                sensor_label = "Bathy (VQ-860-G)"
            elif "topo" in stype:
                sensor_label = "Topo (VUX-160)"
            else:
                sensor_label = meta.get("sensor", "Unknown")

            # Checkbox
            chk_item = QTableWidgetItem()
            chk_item.setCheckState(Qt.Checked)
            self._strip_table.setItem(row, 0, chk_item)

            # Strip ID
            self._strip_table.setItem(row, 1, QTableWidgetItem(str(sid)))
            # Sensor
            self._strip_table.setItem(row, 2, QTableWidgetItem(sensor_label))
            # Points
            self._strip_table.setItem(row, 3, QTableWidgetItem(f"{n_pts:,}"))
            # GPS Time Range
            self._strip_table.setItem(row, 4, QTableWidgetItem(t_span_str))
            # Spanning tiles
            self._strip_table.setItem(row, 5, QTableWidgetItem(n_tiles_str))

            # Trajectory status
            traj = self._trajectories.get(sid)
            if traj is not None:
                filename = os.path.basename(traj.filepath) if traj.filepath else "Custom"
                n_rec = f"{len(traj.times):,}"
            else:
                filename = "None (nadir estimated)"
                n_rec = "-"

            self._strip_table.setItem(row, 6, QTableWidgetItem(filename))
            self._strip_table.setItem(row, 7, QTableWidgetItem(n_rec))

            self._ref_strip_combo.addItem(f"Strip {sid} [{sensor_label}] ({n_pts:,} pts)", sid)

    # ── Trajectory matching ───────────────────────────────────────────

    def _on_auto_match_trajectories(self, silent: bool = False):
        """Scan project folder for candidate trajectory files and match per-sensor."""
        candidate_files = []
        search_dirs = [self._project_dir, os.path.join(self._project_dir, "traj_wsm")]
        parent_dir = os.path.dirname(os.path.abspath(self._project_dir))
        search_dirs.extend([
            os.path.join(parent_dir, "00_zederhausbach_demo_snapshot", "traj_wsm"),
            os.path.join(parent_dir, "traj_wsm"),
        ])

        for sdir in search_dirs:
            if not os.path.isdir(sdir):
                continue
            for ext in ("*.txt", "*.csv", "*.pos"):
                candidate_files.extend(glob.glob(os.path.join(sdir, ext)))

        matched_count = 0
        for fpath in candidate_files:
            fname = os.path.basename(fpath).lower()
            is_bathy_traj = any(k in fname for k in ("vq860", "vq-860", "860", "bathy", "hydro", "green"))
            is_topo_traj = any(k in fname for k in ("vux160", "vux-160", "vux", "160", "topo", "nir"))

            try:
                traj = Trajectory.from_file(fpath)
                for sid, meta in self._strip_meta.items():
                    if sid in self._trajectories:
                        continue
                    sensor = str(meta.get("sensor", "")).lower()

                    # Strictly prevent bathy trajectory from being applied to topo scanner (and vice versa)
                    if is_bathy_traj and "bathy" not in sensor and "860" not in sensor and sensor:
                        continue
                    if is_topo_traj and "topo" not in sensor and "vux" not in sensor and sensor:
                        continue

                    t_min = meta["t_min"]
                    t_max = meta["t_max"]
                    if t_max <= 0 or t_min == float("inf"):
                        continue
                    # Check timestamp overlap
                    if not (t_max < traj.t_min or t_min > traj.t_max):
                        self._trajectories[sid] = traj
                        matched_count += 1
            except Exception:
                continue

        self._populate_strip_table()
        if not silent:
            if matched_count > 0:
                QMessageBox.information(
                    self, "Trajectory Auto-Match",
                    f"Successfully matched {matched_count} flightline strip(s) to sensor trajectory files.",
                )
            else:
                QMessageBox.information(
                    self, "Trajectory Auto-Match",
                    "No matching trajectory files found covering the strip GPS times.\n"
                    "You can manually select a trajectory using 'Browse Trajectory for Selected Strip…'.",
                )

    def _on_browse_trajectory(self):
        row = self._strip_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select Strip", "Please select a strip row in the table first.")
            return

        sid = int(self._strip_table.item(row, 1).text())
        fpath, _ = QFileDialog.getOpenFileName(
            self, f"Select Trajectory File for Strip {sid}",
            self._project_dir, "Trajectory Files (*.txt *.csv *.pos);;All Files (*)",
        )
        if not fpath:
            return

        try:
            traj = Trajectory.from_file(fpath)
            self._trajectories[sid] = traj
            self._populate_strip_table()
            self._strip_table.selectRow(row)
            self._status_label.setText(f"Loaded trajectory with {len(traj.times):,} records for Strip {sid}.")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Trajectory", f"Could not parse trajectory file:\n{exc}")

    def _on_clear_trajectory(self):
        row = self._strip_table.currentRow()
        if row < 0:
            return
        sid = int(self._strip_table.item(row, 1).text())
        if sid in self._trajectories:
            del self._trajectories[sid]
            self._populate_strip_table()

    # ── Calculation ───────────────────────────────────────────────────

    def _on_calculate(self):
        active_sids = []
        for row in range(self._strip_table.rowCount()):
            if self._strip_table.item(row, 0).checkState() == Qt.Checked:
                active_sids.append(int(self._strip_table.item(row, 1).text()))

        if len(active_sids) < 2:
            QMessageBox.warning(self, "Insufficient Strips", "Please check at least two active strips for adjustment.")
            return

        ref_id = self._ref_strip_combo.currentData()
        target_tids = self._get_active_tile_ids()

        self._calc_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Starting strip adjustment calculation across tiles…")

        self._worker = _StripAdjustmentWorker(
            strip_ids=active_sids,
            trajectories=self._trajectories,
            step1_enabled=self._step1_chk.isChecked(),
            step2_enabled=self._step2_chk.isChecked(),
            step3_enabled=self._step3_chk.isChecked(),
            ref_strip_id=ref_id,
            time_knot_interval=self._knot_spin.value(),
            tile_manager=self._tm,
            tile_ids=target_tids,
            strip_data_map=self._strip_data_map if self._strip_data_map else None,
            strip_meta=self._strip_meta,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_calculation_finished)
        self._worker.error.connect(self._on_calculation_error)
        self._worker.start()

    def _on_progress(self, msg: str, pct: float):
        self._status_label.setText(msg)
        self._progress.setValue(int(pct))

    def _on_calculation_error(self, err: str):
        self._calc_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Calculation failed: {err}")
        QMessageBox.critical(self, "Strip Adjustment Failed", err)

    def _on_calculation_finished(self, results: dict):
        self._solved_results = results
        self._calc_btn.setEnabled(True)
        self._apply_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText("Calculation complete. Review results below before applying.")

        corrections: Dict[int, StripCorrection] = results["corrections"]
        init_rmse = results["initial_rmse"]
        final_rmse = results["final_rmse"]
        total_ties = results["total_ties"]
        ref_id = results["ref_strip_id"]

        pct_improve = ((init_rmse - final_rmse) / init_rmse) * 100.0 if init_rmse > 1e-6 else 0.0

        diag_text = (
            f"<b>Analysis Summary:</b> Extracted <b>{total_ties:,}</b> planar tie-surfaces across strip overlaps.<br>"
            f"Initial Overlap RMSE: <b>{init_rmse:.3f} m</b> ({init_rmse*100:.1f} cm)  ➜  "
            f"Adjusted Overlap RMSE: <b>{final_rmse:.3f} m</b> ({final_rmse*100:.1f} cm) "
            f"<span style='color: #2e7d32; font-weight: bold;'>({pct_improve:+.1f}% improvement)</span><br>"
            f"Reference Strip (Held Fixed): <b>Strip {ref_id}</b>"
        )
        self._diag_label.setText(diag_text)

        # Populate results table
        sorted_sids = sorted(corrections.keys())
        self._results_table.setRowCount(len(sorted_sids))

        for row, sid in enumerate(sorted_sids):
            c = corrections[sid]
            self._results_table.setItem(row, 0, QTableWidgetItem(f"Strip {sid}" + (" (Ref)" if sid == ref_id else "")))
            self._results_table.setItem(row, 1, QTableWidgetItem(f"{c.dx:+.3f}"))
            self._results_table.setItem(row, 2, QTableWidgetItem(f"{c.dy:+.3f}"))
            self._results_table.setItem(row, 3, QTableWidgetItem(f"{c.dz:+.3f}"))
            self._results_table.setItem(row, 4, QTableWidgetItem(f"{c.droll:+.4f}"))
            self._results_table.setItem(row, 5, QTableWidgetItem(f"{c.dpitch:+.4f}"))
            self._results_table.setItem(row, 6, QTableWidgetItem(f"{c.dyaw:+.4f}"))

    # ── Apply to Project Tiles ────────────────────────────────────────

    def _on_apply_adjustment(self):
        if not self._solved_results or not self._tm:
            return

        corrections: Dict[int, StripCorrection] = self._solved_results["corrections"]
        target_tids = self._get_active_tile_ids()

        reply = QMessageBox.question(
            self, "Apply Strip Adjustment",
            f"This will adjust coordinates for points in {len(target_tids)} project tiles based on the solved parameters.\n\n"
            f"Original tile files will be backed up (.bak).\n\n"
            f"Do you wish to proceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        self._apply_btn.setEnabled(False)
        self._calc_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        n_tiles = len(target_tids)
        modified_count = 0

        for i, tid in enumerate(target_tids):
            pct = (i + 1) / n_tiles * 100
            self._progress.setValue(int(pct))
            self._status_label.setText(f"Applying adjustment to tile {i+1}/{n_tiles} ({tid})…")

            tile_path = self._tm.tile_las_path(tid)
            if not tile_path or not os.path.exists(tile_path):
                direct = Path(self._project_dir) / "tiles" / f"{tid}.las"
                if direct.is_file():
                    tile_path = direct
                else:
                    continue

            import laspy
            try:
                las = laspy.read(tile_path)
            except Exception as exc:
                logger.error("Failed to read tile %s: %s", tid, exc)
                continue

            psids = np.array(las.point_source_id) if hasattr(las, "point_source_id") else None
            if psids is None:
                continue

            xs = np.array(las.x, dtype=np.float64)
            ys = np.array(las.y, dtype=np.float64)
            zs = np.array(las.z, dtype=np.float64)
            ts = np.array(las.gps_time, dtype=np.float64) if hasattr(las, "gps_time") else np.zeros_like(zs)

            needs_update = False
            for sid, corr in corrections.items():
                mask = (psids == sid)
                if not mask.any():
                    continue

                sub_x, sub_y, sub_z = apply_strip_corrections(
                    xs[mask], ys[mask], zs[mask], ts[mask],
                    corr, self._trajectories.get(sid),
                )
                xs[mask] = sub_x
                ys[mask] = sub_y
                zs[mask] = sub_z
                needs_update = True

            if needs_update:
                # Backup before modifying
                bak_path = str(tile_path) + ".bak"
                if not os.path.exists(bak_path):
                    shutil.copy2(tile_path, bak_path)

                las.x = xs
                las.y = ys
                las.z = zs
                las.write(tile_path)
                modified_count += 1

                if hasattr(self._tm, "_point_cache"):
                    self._tm._point_cache.pop(tid, None)

                if self._db:
                    with self._db.connect() as conn:
                        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                        self._db.update_tile_bbox(conn, tid, bbox, point_count=len(xs))

        self._progress.setVisible(False)
        self._calc_btn.setEnabled(True)
        self._status_label.setText(f"Successfully applied strip adjustment to {modified_count} tiles.")
        self.adjustment_applied.emit(self._solved_results)

        QMessageBox.information(
            self, "Strip Adjustment Complete",
            f"Successfully applied strip adjustment to {modified_count} tiles!\n\n"
            f"Tile coordinates have been updated to align all flightlines seamlessly across the project.",
        )
        self.accept()

