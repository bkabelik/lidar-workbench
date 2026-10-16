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
        self._is_cancelled = False

    def request_cancel(self):
        self._is_cancelled = True

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
                    if self._is_cancelled:
                        self.error.emit("Strip adjustment calculation cancelled by user.")
                        return

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

                    # Extract tie surfaces for active strip pairs in this tile
                    # Limit candidates per pair and per tile to ensure fast streaming
                    tile_ties_count = 0
                    for i in range(len(present)):
                        for j in range(i + 1, len(present)):
                            if tile_ties_count >= 50:
                                break
                            s1, s2 = present[i], present[j]
                            ties = extract_tie_planes(
                                tile_subsets[s1],
                                tile_subsets[s2],
                                s1, s2,
                                patch_radius=1.5,
                                min_patch_pts=15,
                                max_roughness=0.04,
                                max_samples=40,
                            )
                            if ties:
                                all_tie_surfaces.extend(ties)
                                tile_ties_count += len(ties)

            # Subsample across the whole dataset if total tie surfaces exceed 8,000
            if len(all_tie_surfaces) > 8000:
                step = len(all_tie_surfaces) // 8000
                all_tie_surfaces = all_tie_surfaces[::step][:8000]

            # ── Mode B: In-memory fallback (used in standalone unit tests) ─
            elif self.strip_data_map is not None:
                pairs = []
                for i in range(n_strips):
                    for j in range(i + 1, n_strips):
                        pairs.append((self.strip_ids[i], self.strip_ids[j]))

                total_pairs = len(pairs)
                for idx, (s1, s2) in enumerate(pairs):
                    if self._is_cancelled:
                        self.error.emit("Strip adjustment calculation cancelled by user.")
                        return

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

                pa_adj = ts.point_a.copy()
                if abs(ca.droll) > 1e-6 or abs(ca.dpitch) > 1e-6 or abs(ca.dyaw) > 1e-6:
                    sensor_a, ar_a, ap_a, ayaw = solver._get_sensor_pose(ts.strip_a, ts.time_a, ts.point_a)
                    ra = pa_adj - sensor_a
                    rot_vec_a = math.radians(ca.droll) * ar_a + math.radians(ca.dpitch) * ap_a + math.radians(ca.dyaw) * ayaw
                    pa_adj += np.cross(rot_vec_a, ra)
                pa_adj += np.array([ca.dx, ca.dy, ca.dz])
                if ca.drift_nodes and len(ca.drift_nodes) > 0:
                    kt_a = np.array([n[0] for n in ca.drift_nodes])
                    kz_a = np.array([n[3] for n in ca.drift_nodes])
                    pa_adj[2] += float(np.interp(ts.time_a, kt_a, kz_a))

                cb_adj = ts.centroid_b.copy()
                if abs(cb.droll) > 1e-6 or abs(cb.dpitch) > 1e-6 or abs(cb.dyaw) > 1e-6:
                    sensor_b, ar_b, ap_b, ayaw = solver._get_sensor_pose(ts.strip_b, ts.time_b, ts.centroid_b)
                    rb = cb_adj - sensor_b
                    rot_vec_b = math.radians(cb.droll) * ar_b + math.radians(cb.dpitch) * ap_b + math.radians(cb.dyaw) * ayaw
                    cb_adj += np.cross(rot_vec_b, rb)
                cb_adj += np.array([cb.dx, cb.dy, cb.dz])
                if cb.drift_nodes and len(cb.drift_nodes) > 0:
                    kt_b = np.array([n[0] for n in cb.drift_nodes])
                    kz_b = np.array([n[3] for n in cb.drift_nodes])
                    cb_adj[2] += float(np.interp(ts.time_b, kt_b, kz_b))

                res_adj = float(np.dot(pa_adj - cb_adj, ts.normal_b))
                final_residuals.append(res_adj)

            final_rmse = math.sqrt(float(np.mean(np.square(final_residuals))))
            final_mean_dz = float(np.mean(final_residuals))

            # Compute strip origins (mean spatial position of tie surfaces for each strip)
            strip_origins: Dict[int, Tuple[float, float, float]] = {}
            for sid in self.strip_ids:
                pts_sid = []
                for ts in all_tie_surfaces:
                    if ts.strip_a == sid:
                        pts_sid.append(ts.point_a)
                    if ts.strip_b == sid:
                        pts_sid.append(ts.centroid_b)
                if pts_sid:
                    m = np.mean(pts_sid, axis=0)
                    strip_origins[sid] = (float(m[0]), float(m[1]), float(m[2]))

            self.progress.emit("Strip adjustment complete.", 100.0)

            results = {
                "strip_ids": self.strip_ids,
                "ref_strip_id": ref_id,
                "corrections": corrections,
                "strip_origins": strip_origins,
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


class _ApplyAdjustmentWorker(QThread):
    """Worker thread for applying strip corrections to project tile LAS files in background."""

    progress = Signal(str, float)
    finished = Signal(int)
    error = Signal(str)

    def __init__(
        self,
        tile_manager,
        database,
        project_dir: str,
        target_tids: Sequence[str],
        corrections: Dict[int, StripCorrection],
        trajectories: Dict[int, Trajectory],
        strip_origins: Optional[Dict[int, Tuple[float, float, float]]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.tile_manager = tile_manager
        self.database = database
        self.project_dir = project_dir
        self.target_tids = list(target_tids)
        self.corrections = corrections
        self.trajectories = trajectories
        self.strip_origins = strip_origins or {}
        self._is_cancelled = False

    def request_cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            import laspy
            n_tiles = len(self.target_tids)
            if n_tiles == 0:
                self.finished.emit(0)
                return

            modified_count = 0
            db_updates = []

            for i, tid in enumerate(self.target_tids):
                if self._is_cancelled:
                    self.error.emit(f"Adjustment cancelled by user after processing {modified_count} tiles.")
                    return

                pct = (i / max(1, n_tiles)) * 100.0
                self.progress.emit(f"Applying adjustment to tile {i+1}/{n_tiles} ({tid})…", pct)

                tile_path = self.tile_manager.tile_las_path(tid) if self.tile_manager else None
                if not tile_path or not os.path.exists(tile_path):
                    direct = Path(self.project_dir) / "tiles" / f"{tid}.las"
                    if direct.is_file():
                        tile_path = direct
                    else:
                        continue

                try:
                    las = laspy.read(tile_path)
                except Exception as exc:
                    logger.error("Failed to read tile %s: %s", tid, exc)
                    continue

                psids = np.array(las.point_source_id) if hasattr(las, "point_source_id") else None
                if psids is None or len(psids) == 0:
                    continue

                xs = np.array(las.x, dtype=np.float64)
                ys = np.array(las.y, dtype=np.float64)
                zs = np.array(las.z, dtype=np.float64)
                ts = np.array(las.gps_time, dtype=np.float64) if hasattr(las, "gps_time") else np.zeros_like(zs)

                needs_update = False
                for sid, corr in self.corrections.items():
                    mask = (psids == sid)
                    if not mask.any():
                        continue

                    strip_origin = self.strip_origins.get(sid)
                    sub_x, sub_y, sub_z = apply_strip_corrections(
                        xs[mask], ys[mask], zs[mask], ts[mask],
                        corr, self.trajectories.get(sid),
                        strip_origin=strip_origin,
                    )
                    xs[mask] = sub_x
                    ys[mask] = sub_y
                    zs[mask] = sub_z
                    needs_update = True

                if needs_update:
                    bak_path = str(tile_path) + ".bak"
                    if not os.path.exists(bak_path):
                        shutil.copy2(tile_path, bak_path)

                    las.x = xs
                    las.y = ys
                    las.z = zs
                    las.write(tile_path)
                    modified_count += 1

                    if hasattr(self.tile_manager, "_point_cache"):
                        self.tile_manager._point_cache.pop(tid, None)

                    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                    db_updates.append((tid, bbox, len(xs)))

            # Batch update database bounding boxes
            if self.database and db_updates:
                try:
                    with self.database.connect() as conn:
                        for tid, bbox, pt_cnt in db_updates:
                            self.database.update_tile_bbox(conn, tid, bbox, point_count=pt_cnt)
                except Exception as exc:
                    logger.warning("Failed to batch update tile bboxes in DB: %s", exc)

            self.progress.emit(f"Successfully applied strip adjustment to {modified_count} tiles.", 100.0)
            self.finished.emit(modified_count)

        except Exception as exc:
            logger.exception("Error applying strip adjustments")
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
        self._apply_worker: Optional[_ApplyAdjustmentWorker] = None
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

        steps_hint = QLabel(
            "<span style='color: #555; font-size: 11px;'>"
            "💡 <b>Pipeline Tip:</b> Steps 1, 2, and 3 work cumulatively. "
            "When enabling Step 3 (Drift), keep Step 1 and Step 2 checked so XYZ lever-arm shifts and boresight angles are solved alongside trajectory drift."
            "</span>"
        )
        steps_hint.setWordWrap(True)
        v_steps.addWidget(steps_hint)
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
        self._results_table.setColumnCount(8)
        self._results_table.setHorizontalHeaderLabels([
            "Strip ID", "ΔX (m) [Step 1]", "ΔY (m) [Step 1]", "ΔZ (m) [Step 1]",
            "ΔRoll (°) [Step 2]", "ΔPitch (°) [Step 2]", "ΔYaw (°) [Step 2]", "Trajectory Drift [Step 3]"
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

        self._cancel_btn = QPushButton("⏹ Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setStyleSheet("font-weight: bold; padding: 6px 14px; background-color: #c62828; color: white;")
        self._cancel_btn.clicked.connect(self._on_cancel_worker)
        run_row.addWidget(self._cancel_btn)

        self._revert_btn = QPushButton("↺ Revert Tiles from Backup (.bak)")
        self._revert_btn.setEnabled(False)
        self._revert_btn.setToolTip("Restore all tiles to their original pre-adjustment coordinates from backup (.bak) files")
        self._revert_btn.clicked.connect(self._on_revert_backups)
        run_row.addWidget(self._revert_btn)

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
            return [tid for tid in self._selected_tile_ids if not tid.endswith("_noise")]
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

        # Exclude noise tiles and backup artifacts from flightline strip adjustment
        all_tids = [
            tid for tid in all_tids
            if not tid.endswith("_noise") and not tid.endswith(".bak")
        ]

        self._all_project_tile_ids = all_tids

        # 2. Setup scope combo (default to All Project Tiles so adjustment runs globally)
        self._scope_combo.blockSignals(True)
        self._scope_combo.clear()
        self._scope_combo.addItem(f"All Project Tiles ({len(all_tids)} tiles)", "all")
        if self._selected_tile_ids and len(self._selected_tile_ids) < len(all_tids):
            self._scope_combo.addItem(f"Selected Tiles ({len(self._selected_tile_ids)} tiles)", "selected")
        self._scope_combo.setCurrentIndex(0)
        self._scope_combo.blockSignals(False)

        self._scan_tiles_for_strips()
        self._populate_strip_table()
        self._on_auto_match_trajectories(silent=True)
        self._check_backup_status()

    def _scan_tiles_for_strips(self):
        """Discover flightlines, sensor types, and spanning tiles directly from database, sampling GPS bounds."""
        target_tids = self._get_active_tile_ids()
        self._strip_meta.clear()

        if not target_tids:
            return

        target_set = set(target_tids)
        import json
        import laspy

        # 1. Primary path: query SQLite database for accurate flightline metadata & spanning tiles
        if self._db:
            try:
                with self._db.connect() as conn:
                    rows = conn.execute("SELECT id, point_count, flightline_sensor_types FROM tiles").fetchall()
                    for tid, pt_cnt, fl_json in rows:
                        if tid not in target_set or not fl_json:
                            continue
                        mapping = json.loads(fl_json) if isinstance(fl_json, str) else fl_json
                        k = max(1, len(mapping))
                        for sid_str, stype in mapping.items():
                            sid_int = int(sid_str)
                            if sid_int not in self._strip_meta:
                                sensor = stype if stype else ("bathy" if sid_int >= 7 else "topo")
                                self._strip_meta[sid_int] = {
                                    "count": 0,
                                    "sensor": sensor,
                                    "t_min": float("inf"),
                                    "t_max": float("-inf"),
                                    "tiles": set(),
                                }
                            self._strip_meta[sid_int]["tiles"].add(tid)
                            self._strip_meta[sid_int]["count"] += int((pt_cnt or 0) / k)
            except Exception as exc:
                logger.warning("Could not query strip info from database: %s", exc)

        # 2. GPS bounds discovery: fast start/end sampling across all active tiles
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
                    n = r.header.point_count
                    if n == 0:
                        continue
                    k = min(30000, n)
                    pts_1 = r.read_points(k)
                    pts_2 = None
                    if n > k:
                        r.seek(n - k)
                        pts_2 = r.read_points(k)

                    for pts in ([pts_1, pts_2] if pts_2 is not None else [pts_1]):
                        sids = np.asarray(pts.point_source_id)
                        gt = np.asarray(pts.gps_time)
                        for sid in np.unique(sids):
                            sid_int = int(sid)
                            if sid_int not in self._strip_meta:
                                continue
                            sub_gt = gt[sids == sid]
                            valid = sub_gt[sub_gt > 0]
                            if len(valid):
                                cur_min = self._strip_meta[sid_int]["t_min"]
                                cur_max = self._strip_meta[sid_int]["t_max"]
                                self._strip_meta[sid_int]["t_min"] = min(cur_min, float(valid.min()))
                                self._strip_meta[sid_int]["t_max"] = max(cur_max, float(valid.max()))
            except Exception:
                pass

        # 3. Fallback path (e.g. if DB is empty or missing flightline metadata)
        if not self._strip_meta:
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

                        for chunk in r.chunk_iterator(500000):
                            sids = np.unique(chunk.point_source_id)
                            gt = chunk.gps_time if hasattr(chunk, "gps_time") else np.zeros(len(chunk))
                            for sid in sids:
                                sid_int = int(sid)
                                if sid_int not in self._strip_meta:
                                    stype = "bathy" if sid_int >= 7 else "topo"
                                    self._strip_meta[sid_int] = {
                                        "count": 0,
                                        "sensor": stype,
                                        "t_min": float("inf"),
                                        "t_max": float("-inf"),
                                        "tiles": set(),
                                    }
                                mask = chunk.point_source_id == sid
                                self._strip_meta[sid_int]["count"] += int(mask.sum())
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
            traj = self._trajectories.get(sid)

            if t_max > 0 and t_min != float("inf"):
                dur = t_max - t_min
                t_span_str = f"{t_min:.1f} – {t_max:.1f} ({dur:.1f}s)"
            elif traj is not None:
                dur = traj.t_max - traj.t_min
                t_span_str = f"{traj.t_min:.1f} – {traj.t_max:.1f} ({dur:.1f}s traj)"
            else:
                t_span_str = "N/A"

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
                    # Check timestamp overlap if available, else match per sensor
                    if t_max > 0 and t_min != float("inf"):
                        if not (t_max < traj.t_min or t_min > traj.t_max):
                            self._trajectories[sid] = traj
                            matched_count += 1
                    else:
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

    def _check_backup_status(self):
        """Check if any backup (.bak) tile files exist in the project and update the Revert button."""
        bak_count = 0
        if self._project_dir:
            tdir = Path(self._project_dir) / "tiles"
            if tdir.is_dir():
                bak_count = len(list(tdir.glob("*.las.bak")))

        if bak_count > 0:
            self._revert_btn.setEnabled(True)
            self._revert_btn.setText(f"↺ Revert {bak_count} Tiles from Backup (.bak)")
        else:
            self._revert_btn.setEnabled(False)
            self._revert_btn.setText("↺ Revert Tiles from Backup (.bak)")

    def _on_cancel_worker(self):
        """Cancel running calculation or application worker."""
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._status_label.setText("Cancelling strip adjustment calculation…")
            self._cancel_btn.setEnabled(False)
        if self._apply_worker and self._apply_worker.isRunning():
            self._apply_worker.request_cancel()
            self._status_label.setText("Cancelling adjustment application…")
            self._cancel_btn.setEnabled(False)

    # ── Calculation ───────────────────────────────────────────────────

    def _on_calculate(self):
        active_sids = []
        for row in range(self._strip_table.rowCount()):
            if self._strip_table.item(row, 0).checkState() == Qt.Checked:
                active_sids.append(int(self._strip_table.item(row, 1).text()))

        if len(active_sids) < 2:
            QMessageBox.warning(self, "Insufficient Strips", "Please check at least two active strips for adjustment.")
            return

        # If user checked Step 3 (Drift) but unchecked Step 1 & Step 2
        if self._step3_chk.isChecked() and not self._step1_chk.isChecked() and not self._step2_chk.isChecked():
            res = QMessageBox.question(
                self,
                "Enable Step 1 & Step 2?",
                "Step 3 (Dynamic Trajectory Drift) solves a time-varying vertical spline (dZ) to absorb IMU drift.\n\n"
                "Because Step 1 (XYZ translation) and Step 2 (RPY boresight) are currently unchecked, "
                "horizontal lever-arm shifts and angular mounting offsets will NOT be adjusted, leading to a higher residual RMSE.\n\n"
                "Would you like to enable Step 1 and Step 2 as well for full 3D alignment?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if res == QMessageBox.Yes:
                self._step1_chk.setChecked(True)
                self._step2_chk.setChecked(True)

        ref_id = self._ref_strip_combo.currentData()
        target_tids = self._get_active_tile_ids()

        self._calc_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._revert_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._cancel_btn.setEnabled(True)
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
        self._cancel_btn.setVisible(False)
        self._progress.setVisible(False)
        self._check_backup_status()
        self._status_label.setText(f"Calculation failed or cancelled: {err}")
        QMessageBox.critical(self, "Strip Adjustment Failed", err)

    def _on_calculation_finished(self, results: dict):
        self._solved_results = results
        self._calc_btn.setEnabled(True)
        self._apply_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._progress.setVisible(False)
        self._check_backup_status()
        self._status_label.setText("Calculation complete. Review results below before applying.")

        corrections: Dict[int, StripCorrection] = results["corrections"]
        init_rmse = results["initial_rmse"]
        final_rmse = results["final_rmse"]
        total_ties = results["total_ties"]
        ref_id = results["ref_strip_id"]

        pct_improve = ((init_rmse - final_rmse) / init_rmse) * 100.0 if init_rmse > 1e-6 else 0.0

        active_steps = []
        if self._step1_chk.isChecked():
            active_steps.append("Step 1 (XYZ)")
        if self._step2_chk.isChecked():
            active_steps.append("Step 2 (RPY)")
        if self._step3_chk.isChecked():
            active_steps.append("Step 3 (Drift)")
        steps_str = " + ".join(active_steps) if active_steps else "None"

        note = ""
        if self._step3_chk.isChecked() and not self._step1_chk.isChecked() and not self._step2_chk.isChecked():
            note = (
                "<br><span style='color: #d32f2f; font-weight: bold;'>"
                "⚠️ Note: Step 1 and Step 2 were disabled. Only 1D vertical drift was adjusted; "
                "horizontal lever-arm offsets and boresight angles remain uncorrected."
                "</span>"
            )

        diag_text = (
            f"<b>Analysis Summary:</b> Extracted <b>{total_ties:,}</b> planar tie-surfaces across strip overlaps.<br>"
            f"Active Steps: <b>{steps_str}</b><br>"
            f"Initial Overlap RMSE: <b>{init_rmse:.3f} m</b> ({init_rmse*100:.1f} cm)  ➜  "
            f"Adjusted Overlap RMSE: <b>{final_rmse:.3f} m</b> ({final_rmse*100:.1f} cm) "
            f"<span style='color: #2e7d32; font-weight: bold;'>({pct_improve:+.1f}% improvement)</span><br>"
            f"Reference Strip (Held Fixed): <b>Strip {ref_id}</b>"
            f"{note}"
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

            if sid == ref_id:
                drift_str = "0.0 mm (Fixed Datum)"
            elif c.drift_nodes and len(c.drift_nodes) > 0:
                dzs = [n[3] * 1000.0 for n in c.drift_nodes]
                drift_str = f"{len(c.drift_nodes)} knots [{min(dzs):+.1f}, {max(dzs):+.1f}] mm"
            elif self._step3_chk.isChecked():
                drift_str = "0.0 mm (No drift detected)"
            else:
                drift_str = "Disabled"
            self._results_table.setItem(row, 7, QTableWidgetItem(drift_str))

    # ── Apply to Project Tiles ────────────────────────────────────────

    def _on_apply_adjustment(self):
        if not self._solved_results:
            return

        corrections: Dict[int, StripCorrection] = self._solved_results["corrections"]
        strip_origins = self._solved_results.get("strip_origins", {})
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
        self._revert_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText(f"Starting adjustment on {len(target_tids)} tiles in background…")

        self._apply_worker = _ApplyAdjustmentWorker(
            tile_manager=self._tm,
            database=self._db,
            project_dir=self._project_dir,
            target_tids=target_tids,
            corrections=corrections,
            trajectories=self._trajectories,
            strip_origins=strip_origins,
            parent=self,
        )
        self._apply_worker.progress.connect(self._on_progress)
        self._apply_worker.finished.connect(self._on_apply_finished)
        self._apply_worker.error.connect(self._on_apply_error)
        self._apply_worker.start()

    def _on_apply_finished(self, modified_count: int):
        self._progress.setVisible(False)
        self._calc_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._check_backup_status()

        self._status_label.setText(f"Successfully applied strip adjustment to {modified_count} tiles.")
        self.adjustment_applied.emit(self._solved_results)

        # Clear point caches in tile manager so subsequent calculations read the newly written files
        if self._tm and hasattr(self._tm, "_point_cache"):
            self._tm._point_cache.clear()

        # Update diagnostic label to show applied status
        init_rmse = self._solved_results.get("initial_rmse", 0.0) if self._solved_results else 0.0
        final_rmse = self._solved_results.get("final_rmse", 0.0) if self._solved_results else 0.0
        pct_improve = ((init_rmse - final_rmse) / init_rmse) * 100.0 if init_rmse > 1e-6 else 0.0

        applied_banner = (
            f"<div style='background-color: #e8f5e9; border: 1px solid #4caf50; border-radius: 4px; padding: 8px;'>"
            f"<b style='color: #2e7d32;'>✔ Strip Adjustment Applied to {modified_count} Tiles!</b><br>"
            f"Initial Overlap RMSE: <b>{init_rmse*100:.1f} cm</b>  ➜  "
            f"Adjusted Overlap RMSE: <b>{final_rmse*100:.1f} cm</b> "
            f"<span style='color: #2e7d32; font-weight: bold;'>({pct_improve:+.1f}% improvement)</span><br>"
            f"<i>Tile coordinates have been updated on disk with .bak backups.<br>"
            f"You can now run another pass (e.g. Step 3 Drift) to adjust from the new baseline, or close this dialog.</i>"
            f"</div>"
        )
        self._diag_label.setText(applied_banner)

        QMessageBox.information(
            self, "Strip Adjustment Applied",
            f"Successfully applied strip adjustment to {modified_count} tiles!\n\n"
            f"Tile coordinates on disk have been updated.\n"
            f"You can now re-run calculation to verify the new baseline or close the dialog.",
        )

    def _on_apply_error(self, err: str):
        self._calc_btn.setEnabled(True)
        self._apply_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._progress.setVisible(False)
        self._check_backup_status()
        self._status_label.setText(f"Adjustment interrupted: {err}")
        QMessageBox.warning(self, "Strip Adjustment Interrupted", f"Adjustment could not be fully applied:\n{err}")

    def _on_revert_backups(self):
        """Restore all modified tiles from their original .las.bak backups."""
        if not self._project_dir:
            return

        tdir = Path(self._project_dir) / "tiles"
        if not tdir.is_dir():
            return

        bak_files = list(tdir.glob("*.las.bak"))
        if not bak_files:
            QMessageBox.information(self, "No Backups Found", "No tile backup (.bak) files were found to revert.")
            return

        reply = QMessageBox.question(
            self,
            "Revert Tiles to Backup",
            f"Are you sure you want to revert {len(bak_files)} tiles to their original pre-adjustment coordinates?\n\n"
            f"All strip adjustment changes made to these tiles will be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._calc_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._revert_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        import laspy
        reverted_count = 0
        db_updates = []
        n_total = len(bak_files)

        for i, bak_path in enumerate(bak_files):
            pct = ((i + 1) / max(1, n_total)) * 100.0
            self._progress.setValue(int(pct))
            tid = bak_path.name[:-8]  # strip ".las.bak"
            self._status_label.setText(f"Reverting {tid} from backup…")

            orig_path = bak_path.with_name(f"{tid}.las")
            try:
                shutil.copy2(bak_path, orig_path)
                os.remove(bak_path)
                reverted_count += 1

                if self._tm and hasattr(self._tm, "_point_cache"):
                    self._tm._point_cache.pop(tid, None)

                if self._db:
                    try:
                        las = laspy.read(orig_path)
                        bbox = (float(las.x.min()), float(las.y.min()), float(las.x.max()), float(las.y.max()))
                        db_updates.append((tid, bbox, len(las.x)))
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to restore backup %s: %s", bak_path, exc)

        if self._db and db_updates:
            try:
                with self._db.connect() as conn:
                    for tid, bbox, pt_cnt in db_updates:
                        self._db.update_tile_bbox(conn, tid, bbox, point_count=pt_cnt)
            except Exception as exc:
                logger.warning("Failed to batch update bboxes on revert: %s", exc)

        if self._tm and hasattr(self._tm, "_point_cache"):
            self._tm._point_cache.clear()

        self._progress.setVisible(False)
        self._calc_btn.setEnabled(True)
        self._check_backup_status()
        self._solved_results = None
        self._diag_label.setText("Tiles reverted to pre-adjustment state. Re-run calculation if needed.")
        self._status_label.setText(f"Successfully reverted {reverted_count} tiles from backup.")

        QMessageBox.information(
            self,
            "Revert Complete",
            f"Successfully reverted {reverted_count} tiles from their .bak backups to original state.",
        )

    def reject(self):
        """Cancel background threads cleanly if dialog is closed."""
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(1000)
        if self._apply_worker and self._apply_worker.isRunning():
            self._apply_worker.request_cancel()
            self._apply_worker.wait(1000)
        super().reject()

