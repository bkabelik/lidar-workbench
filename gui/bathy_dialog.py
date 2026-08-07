"""
LiDAR Workbench — Bathymetry Processing Dialog.

Dedicated dialog for bathymetric LiDAR processing:
  - Snell's law refraction correction (with trajectory file + per-point sensor)
  - Water surface from manual Z, GeoTIFF, or auto-detection
  - Water surface + ghost removal
  - Automatic river corridor cropping
  - Benthic continuity filtering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

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
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from ..noise_filter import (
    water_surface_ghost_removal,
    river_corridor_mask,
    benthic_continuity_filter,
)
from ..refraction import (
    apply_snells_correction,
    apply_snells_correction_nadir,
    crop_to_water_surface_model,
    detect_water_surface_ransac,
    interpolate_sensor_position,
    interpolate_sensor_positions_batch,
    load_water_surface_geotiff,
    read_trajectory_ascii,
    sample_geotiff_at_points,
)

logger = logging.getLogger("lidar_workbench.gui.bathy_dialog")


class _BathyWorker(QThread):
    progress = Signal(str, float)
    tile_done = Signal(str, dict)     # (tile_id, result_dict) — save immediately
    finished_all = Signal(int)        # number of tiles processed
    error = Signal(str)

    def __init__(self, tile_ids: list, steps: list[dict],
                 load_func, max_workers: int = 4, parent=None):
        """
        Args:
            tile_ids: List of tile ID strings to process.
            steps: Pipeline step configurations.
            load_func: Callable ``(tile_id) -> dict | None`` that loads tile data.
            max_workers: Number of tiles to process in parallel.
        """
        super().__init__(parent)
        self._tile_ids = list(tile_ids)
        self._steps = steps
        self._load_func = load_func
        self._max_workers = max_workers

    def run(self):
        """Process tiles in parallel batches: load N, process N in pool,
        save results, load next batch.  Loading is sequential (laspy safety);
        processing is parallel via ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        n_total = len(self._tile_ids)
        if n_total == 0:
            self.finished_all.emit(0)
            return

        t_start = time.time()
        processed = 0
        max_in_flight = max(1, self._max_workers)
        cursor = 0  # index into self._tile_ids

        try:
            with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
                futures: dict = {}

                while processed < n_total:
                    # ── Feed the pool: submit new work while capacity
                    #     exists and tiles remain to load ──────────
                    while len(futures) < max_in_flight and cursor < n_total:
                        tid = self._tile_ids[cursor]
                        cursor += 1
                        data = self._load_func(tid)
                        if data is not None:
                            fut = pool.submit(self._process_one, data,
                                             cursor, n_total)
                            futures[fut] = tid
                        else:
                            processed += 1  # skip failed loads

                    if not futures:
                        break

                    # ── Wait for ONE result, then loop back to
                    #     immediately refill the pool ─────────────
                    completed = as_completed(futures)
                    fut = next(completed)
                    tid = futures.pop(fut)
                    try:
                        result = fut.result()
                        self.tile_done.emit(tid, result)
                        processed += 1
                        self.progress.emit(
                            f"Done {processed}/{n_total} tiles",
                            processed / n_total * 100,
                        )
                    except Exception as exc:
                        logger.error("Bathy error on %s: %s", tid, exc)
                        processed += 1

            elapsed = time.time() - t_start
            summary = f"Done: {processed}/{n_total} tiles in {elapsed:.1f}s"
            self.progress.emit(summary, 100.0)
            self.finished_all.emit(processed)
        except Exception as exc:
            logger.exception("Bathy processing failed")
            self.error.emit(str(exc))

    def _process_one(self, data: dict, tile_idx: int, n_tiles: int) -> dict:
        """Run the pipeline on a single tile's data, return result dict."""
        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in data.items()}
        water_surface_z: Optional[float] = None

        # ── Early skip: tile with zero bathy points ─────────────────
        st = result.get("sensor_type")
        has_bathy = (st is not None and (st == 2).any()) if st is not None else False
        bathy_step_names = {"snells", "water_surface", "river_crop", "benthic", "ws_crop"}
        has_non_bathy_steps = any(s["name"] not in bathy_step_names
                                  for s in self._steps)
        if not has_bathy and not has_non_bathy_steps and len(result["x"]) > 0:
            logger.info("Tile %s: no bathy points, skipping pipeline", tile_idx)
            return result

        for i, step in enumerate(self._steps):
            name = step["name"]
            pct_base = i / len(self._steps) * 100
            # Scale progress into this tile's slice of overall progress
            tile_pct_base = (tile_idx / n_tiles) * 100
            tile_pct_range = 100.0 / n_tiles

            if name == "ws_crop":
                label = f"Crop WSM…" if n_tiles == 1 else f"Tile {tile_idx+1}/{n_tiles}: Crop WSM…"
                self.progress.emit(label, tile_pct_base + pct_base * tile_pct_range / 100)
                surface = step["surface"]
                georef = step["georef"]
                keep_mask, _ = crop_to_water_surface_model(
                    result["x"], result["y"], result["z"],
                    surface=surface, georef=georef,
                    tolerance_above_cm=step.get("tolerance_above_cm", 0.0),
                    extrapolate_m=step.get("extrapolate_m", 0.0),
                    data_epsg=step.get("data_epsg"),
                )
                is_bathy = (result.get("sensor_type") is not None
                            and (result["sensor_type"] == 2).any())
                if is_bathy:
                    k = keep_mask | (result["sensor_type"] != 2)
                else:
                    k = keep_mask
                result["x"], result["y"], result["z"] = result["x"][k], result["y"][k], result["z"][k]
                for key in ("classification", "intensity", "return_number",
                            "point_source_id", "sensor_type", "gps_time"):
                    if key in result and result[key] is not None:
                        result[key] = result[key][k]

            elif name == "snells":
                label = f"Snell's…" if n_tiles == 1 else f"Tile {tile_idx+1}/{n_tiles}: Snell's…"
                self.progress.emit(label, tile_pct_base + pct_base * tile_pct_range / 100)
                n_water = step.get("n_water", 1.3333)

                # Water surface: GeoTIFF (sampled on-demand) or scalar
                ws_input = step["water_surface_z"]
                if ws_input is None and step.get("geotiff_surface") is not None:
                    # Sample GeoTIFF at this tile's points
                    ws_z = sample_geotiff_at_points(
                        step["geotiff_surface"], step["geotiff_georef"],
                        result["x"], result["y"],
                        fill_value=step.get("ws_fallback", 0.0),
                    )
                    water_surface_z = float(np.nanmedian(ws_z))
                elif isinstance(ws_input, np.ndarray):
                    ws_z = ws_input
                    water_surface_z = float(np.median(ws_z))
                else:
                    ws_z = float(ws_input)
                    water_surface_z = ws_z

                orig_x, orig_y, orig_z = result["x"].copy(), result["y"].copy(), result["z"].copy()

                if step.get("use_trajectory"):
                    traj = read_trajectory_ascii(
                        step["trajectory_path"],
                        separator=step.get("separator"),
                        skip_rows=step.get("skip_rows", 0),
                        has_header=step.get("has_header", False),
                        time_col=step.get("time_col", 0),
                        x_col=step.get("x_col", 1),
                        y_col=step.get("y_col", 2),
                        z_col=step.get("z_col", 3),
                    )
                    gps_times = result.get("gps_time")
                    if gps_times is not None and len(gps_times) == len(result["x"]):
                        # Vectorized batch interpolation — O(N log M) instead of
                        # per-point Python loop (was the #1 bottleneck on large tiles)
                        sx_arr, sy_arr, sz_arr = interpolate_sensor_positions_batch(
                            traj, gps_times.astype(np.float64),
                        )
                        xc, yc, zc = apply_snells_correction(
                            result["x"], result["y"], result["z"],
                            water_surface_z=ws_z,
                            sensor_position=(sx_arr, sy_arr, sz_arr),
                            n_water=n_water,
                        )
                    else:
                        sx = float(np.median(traj[:, 1]))
                        sy = float(np.median(traj[:, 2]))
                        sz = float(np.median(traj[:, 3]))
                        xc, yc, zc = apply_snells_correction(
                            result["x"], result["y"], result["z"],
                            water_surface_z=ws_z,
                            sensor_position=(sx, sy, sz),
                            n_water=n_water,
                        )
                else:
                    if isinstance(ws_z, np.ndarray):
                        ws_scalar = float(np.median(ws_z))
                    else:
                        ws_scalar = ws_z
                    xc, yc, zc = apply_snells_correction_nadir(
                        result["x"], result["y"], result["z"],
                        water_surface_z=ws_scalar,
                        n_water=n_water,
                    )

                is_bathy = (result.get("sensor_type") is not None
                            and (result["sensor_type"] == 2).any())
                if is_bathy:
                    bathy_mask = result["sensor_type"] == 2
                    result["x"] = np.where(bathy_mask, xc, orig_x)
                    result["y"] = np.where(bathy_mask, yc, orig_y)
                    result["z"] = np.where(bathy_mask, zc, orig_z)
                else:
                    result["x"], result["y"], result["z"] = xc, yc, zc

            elif name == "water_surface":
                label = f"Ghost removal…" if n_tiles == 1 else f"Tile {tile_idx+1}/{n_tiles}: Ghost removal…"
                self.progress.emit(label, tile_pct_base + pct_base * tile_pct_range / 100)
                k, _ = water_surface_ghost_removal(
                    result["x"], result["y"], result["z"],
                    tile_length=step.get("tile_length", 40.0),
                )
                is_bathy = (result.get("sensor_type") is not None
                            and (result["sensor_type"] == 2).any())
                if is_bathy:
                    k = k | (result["sensor_type"] != 2)
                result["x"], result["y"], result["z"] = result["x"][k], result["y"][k], result["z"][k]
                for key in ("classification", "intensity", "return_number",
                            "point_source_id", "sensor_type", "gps_time"):
                    if key in result and result[key] is not None:
                        result[key] = result[key][k]

            elif name == "river_crop":
                label = f"River crop…" if n_tiles == 1 else f"Tile {tile_idx+1}/{n_tiles}: River crop…"
                self.progress.emit(label, tile_pct_base + pct_base * tile_pct_range / 100)
                k, _ = river_corridor_mask(
                    result["x"], result["y"], result["z"],
                    result.get("intensity", np.zeros(len(result["x"]), dtype=np.uint16)),
                    result.get("return_number", np.ones(len(result["x"]), dtype=np.uint8)),
                    water_surface_z=water_surface_z,
                )
                st = result.get("sensor_type")
                if st is not None and (st == 2).any():
                    k = k | (st != 2)
                result["x"], result["y"], result["z"] = result["x"][k], result["y"][k], result["z"][k]
                for key in ("classification", "intensity", "return_number",
                            "point_source_id", "sensor_type", "gps_time"):
                    if key in result and result[key] is not None:
                        result[key] = result[key][k]

            elif name == "benthic":
                label = f"Benthic filter…" if n_tiles == 1 else f"Tile {tile_idx+1}/{n_tiles}: Benthic filter…"
                self.progress.emit(label, tile_pct_base + pct_base * tile_pct_range / 100)
                certain = np.ones(len(result["x"]), dtype=bool)
                k = benthic_continuity_filter(
                    result["x"], result["y"], result["z"],
                    certain_bed_mask=certain,
                    search_radius=step.get("search_radius", 0.3),
                    max_slope=step.get("max_slope", 0.35),
                )
                is_bathy = (result.get("sensor_type") is not None
                            and (result["sensor_type"] == 2).any())
                if is_bathy:
                    k = k | (result["sensor_type"] != 2)
                result["x"], result["y"], result["z"] = result["x"][k], result["y"][k], result["z"][k]
                for key in ("classification", "intensity", "return_number",
                            "point_source_id", "sensor_type", "gps_time"):
                    if key in result and result[key] is not None:
                        result[key] = result[key][k]

        return result


class BathyDialog(QDialog):
    """Dialog for bathymetric processing pipeline."""

    bathy_applied = Signal()                     # all tiles saved, dialog can close
    tile_save_requested = Signal(str, dict)      # (tile_id, result) — save to disk now

    def __init__(self, tile_ids: list,
                 scanner: str = '', sensor_type: str = '',
                 parent=None, data_epsg: Optional[int] = None):
        """
        Args:
            tile_ids: Tile IDs to process.
        """
        super().__init__(parent)
        self._tile_ids = tile_ids
        self._scanner = scanner
        self._sensor_type = sensor_type
        self._data_epsg = data_epsg
        self._worker: Optional[_BathyWorker] = None
        self._water_surface_z: float = 0.0
        self._geotiff_surface: Optional[np.ndarray] = None
        self._geotiff_georef: Optional[dict] = None
        self._crop_surface: Optional[np.ndarray] = None
        self._crop_georef: Optional[dict] = None

        n_tiles = len(tile_ids)
        title = "Bathymetry Processing"
        if n_tiles > 1:
            title += f" ({n_tiles} tiles selected)"
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ═══════════════════════════════════════════════════════════
        # 0. Water Surface Model Crop (optional, at top of pipeline)
        # ═══════════════════════════════════════════════════════════
        crop_group = QGroupBox("0. Water Surface Crop (remove data outside river)")
        cf = QFormLayout(crop_group)

        self._ws_crop_enabled = QCheckBox(
            "Crop points to water surface model extent"
        )
        self._ws_crop_enabled.setToolTip(
            "Uses a float GeoTIFF water surface model to remove "
            "bathymetry points outside the river corridor. Points "
            "above the water surface + tolerance are also removed."
        )
        cf.addRow(self._ws_crop_enabled)

        # GeoTIFF browse for crop
        crop_geotiff_row = QHBoxLayout()
        self._crop_geotiff_edit = QLineEdit()
        self._crop_geotiff_edit.setPlaceholderText(
            "Select water surface GeoTIFF for cropping…"
        )
        self._crop_geotiff_edit.setReadOnly(True)
        crop_geotiff_row.addWidget(self._crop_geotiff_edit)
        crop_geotiff_browse = QPushButton("Browse…")
        crop_geotiff_browse.clicked.connect(self._on_browse_crop_geotiff)
        crop_geotiff_row.addWidget(crop_geotiff_browse)
        cf.addRow("Water Surface:", crop_geotiff_row)
        self._crop_geotiff_info = QLabel("")
        self._crop_geotiff_info.setWordWrap(True)
        self._crop_geotiff_info.setVisible(False)
        cf.addRow(self._crop_geotiff_info)

        # Tolerance above surface
        self._crop_tol_spin = QDoubleSpinBox()
        self._crop_tol_spin.setRange(0.0, 500.0)
        self._crop_tol_spin.setDecimals(0)
        self._crop_tol_spin.setValue(0.0)
        self._crop_tol_spin.setSuffix(" cm")
        self._crop_tol_spin.setToolTip(
            "Points this far ABOVE the water surface are also kept. "
            "Default 0 = strictly at or below surface."
        )
        cf.addRow("Tolerance above:", self._crop_tol_spin)

        # Extrapolation
        self._crop_extrap_spin = QDoubleSpinBox()
        self._crop_extrap_spin.setRange(0.0, 200.0)
        self._crop_extrap_spin.setDecimals(1)
        self._crop_extrap_spin.setValue(1.0)
        self._crop_extrap_spin.setSuffix(" m")
        self._crop_extrap_spin.setToolTip(
            "Expand the water surface model outward by this distance. "
            "Creates a safety margin so the riverbank edges are not "
            "accidentally cropped."
        )
        cf.addRow("Extrapolate:", self._crop_extrap_spin)

        layout.addWidget(crop_group)

        # ═══════════════════════════════════════════════════════════
        # 1. Snell's Law Refraction Correction
        # ═══════════════════════════════════════════════════════════
        snells_group = QGroupBox("1. Snell's Law Refraction Correction")
        sf = QFormLayout(snells_group)

        self._snells_enabled = QCheckBox("Apply refraction correction")
        self._snells_enabled.setToolTip(
            "Corrects submerged points for light bending/slowing at "
            "the air-water interface. Required for accurate bathymetry."
        )
        sf.addRow(self._snells_enabled)

        # ── Water surface source ──
        ws_group = QGroupBox("Water Surface Source")
        ws_layout = QVBoxLayout(ws_group)

        self._ws_manual_radio = QRadioButton("Manual Z value")
        self._ws_manual_radio.setChecked(True)
        ws_layout.addWidget(self._ws_manual_radio)

        self._ws_z_spin = QDoubleSpinBox()
        self._ws_z_spin.setRange(-1000, 10000)
        self._ws_z_spin.setDecimals(2)
        self._ws_z_spin.setValue(0.0)
        self._ws_z_spin.setSuffix(" m")
        self._ws_z_spin.setToolTip(
            "Water surface elevation (auto-detected if possible)"
        )
        ws_z_row = QHBoxLayout()
        ws_z_row.setContentsMargins(20, 0, 0, 0)
        ws_z_row.addWidget(QLabel("Z:"))
        ws_z_row.addWidget(self._ws_z_spin)
        ws_z_row.addStretch()
        ws_layout.addLayout(ws_z_row)

        self._ws_geotiff_radio = QRadioButton("From GeoTIFF (float raster)")
        ws_layout.addWidget(self._ws_geotiff_radio)

        geotiff_row = QHBoxLayout()
        geotiff_row.setContentsMargins(20, 0, 0, 0)
        self._geotiff_path_edit = QLineEdit()
        self._geotiff_path_edit.setPlaceholderText(
            "Select a float GeoTIFF water surface model…"
        )
        self._geotiff_path_edit.setReadOnly(True)
        geotiff_row.addWidget(self._geotiff_path_edit)
        geotiff_browse = QPushButton("Browse…")
        geotiff_browse.clicked.connect(self._on_browse_geotiff)
        geotiff_row.addWidget(geotiff_browse)
        ws_layout.addLayout(geotiff_row)

        self._geotiff_info = QLabel("")
        self._geotiff_info.setWordWrap(True)
        self._geotiff_info.setVisible(False)
        ws_layout.addWidget(self._geotiff_info)

        self._ws_manual_radio.toggled.connect(
            lambda checked: self._ws_z_spin.setEnabled(checked)
        )
        self._ws_geotiff_radio.toggled.connect(
            lambda checked: self._geotiff_path_edit.setEnabled(checked)
        )

        sf.addRow(ws_group)

        # ── Refractive index ──
        self._n_water_spin = QDoubleSpinBox()
        self._n_water_spin.setRange(1.0, 2.0)
        self._n_water_spin.setDecimals(4)
        self._n_water_spin.setValue(1.3333)
        self._n_water_spin.setToolTip("Refractive index of water")
        sf.addRow("n_water:", self._n_water_spin)

        # ── Trajectory ──
        traj_layout = QHBoxLayout()
        self._traj_path_edit = QLineEdit()
        self._traj_path_edit.setPlaceholderText(
            "Trajectory file (ASCII, for per-point sensor position)"
        )
        traj_layout.addWidget(self._traj_path_edit)
        traj_browse = QPushButton("Browse…")
        traj_browse.clicked.connect(self._on_browse_trajectory)
        traj_layout.addWidget(traj_browse)
        sf.addRow("Trajectory:", traj_layout)

        # Separator
        traj_sep_row = QHBoxLayout()
        self._traj_sep_combo = QComboBox()
        self._traj_sep_combo.addItems(["Auto", "Comma (,)", "Semicolon (;)", "Tab", "Space", "Custom"])
        self._traj_sep_combo.setCurrentText("Auto")
        traj_sep_row.addWidget(QLabel("Separator:"))
        traj_sep_row.addWidget(self._traj_sep_combo)
        self._traj_custom_sep = QLineEdit()
        self._traj_custom_sep.setPlaceholderText("custom")
        self._traj_custom_sep.setMaximumWidth(60)
        self._traj_custom_sep.setVisible(False)
        self._traj_sep_combo.currentTextChanged.connect(self._on_traj_sep_changed)
        traj_sep_row.addWidget(self._traj_custom_sep)
        self._traj_has_header = QCheckBox("Header row")
        traj_sep_row.addWidget(self._traj_has_header)
        self._traj_skip_spin = QSpinBox()
        self._traj_skip_spin.setRange(0, 1000)
        self._traj_skip_spin.setValue(0)
        self._traj_skip_spin.setToolTip("Skip first N rows")
        traj_sep_row.addWidget(QLabel("Skip:"))
        traj_sep_row.addWidget(self._traj_skip_spin)
        traj_sep_row.addStretch()
        sf.addRow("", traj_sep_row)

        # Column mapping
        traj_col_row = QHBoxLayout()
        self._traj_time_col = QComboBox()
        self._traj_time_col.setMinimumWidth(60)
        traj_col_row.addWidget(QLabel("Time:"))
        traj_col_row.addWidget(self._traj_time_col)
        self._traj_x_col = QComboBox()
        traj_col_row.addWidget(QLabel("X:"))
        traj_col_row.addWidget(self._traj_x_col)
        self._traj_y_col = QComboBox()
        traj_col_row.addWidget(QLabel("Y:"))
        traj_col_row.addWidget(self._traj_y_col)
        self._traj_z_col = QComboBox()
        traj_col_row.addWidget(QLabel("Z:"))
        traj_col_row.addWidget(self._traj_z_col)
        traj_col_row.addStretch()
        sf.addRow("Columns:", traj_col_row)

        # Preview
        self._traj_preview = QLabel("")
        self._traj_preview.setWordWrap(True)
        self._traj_preview.setMaximumHeight(50)
        sf.addRow("Preview:", self._traj_preview)

        layout.addWidget(snells_group)

        # ═══════════════════════════════════════════════════════════
        # 2. Water Surface + Ghost Removal
        # ═══════════════════════════════════════════════════════════
        ws_clean_group = QGroupBox("2. Water Surface + Ghost Removal")
        wf = QFormLayout(ws_clean_group)
        self._ws_enabled = QCheckBox(
            "Remove water surface and near-surface ghost returns"
        )
        self._ws_enabled.setChecked(False)
        wf.addRow(self._ws_enabled)
        self._ws_tile_spin = QDoubleSpinBox()
        self._ws_tile_spin.setRange(10.0, 200.0)
        self._ws_tile_spin.setDecimals(1)
        self._ws_tile_spin.setValue(40.0)
        self._ws_tile_spin.setSuffix(" m")
        self._ws_tile_spin.setToolTip(
            "Along-track tile length for local surface fitting"
        )
        wf.addRow("Tile Length:", self._ws_tile_spin)
        layout.addWidget(ws_clean_group)

        # ═══════════════════════════════════════════════════════════
        # 3. River Corridor Crop
        # ═══════════════════════════════════════════════════════════
        river_group = QGroupBox("3. River Corridor Crop")
        rf = QFormLayout(river_group)
        self._river_enabled = QCheckBox("Auto-crop to river corridor only")
        self._river_enabled.setChecked(False)
        self._river_enabled.setToolTip(
            "Uses local Z roughness and intensity to distinguish "
            "water from land."
        )
        rf.addRow(self._river_enabled)
        layout.addWidget(river_group)

        # ═══════════════════════════════════════════════════════════
        # 4. Benthic Continuity Filter
        # ═══════════════════════════════════════════════════════════
        benthic_group = QGroupBox("4. Benthic Continuity Filter")
        bf = QFormLayout(benthic_group)
        self._benthic_enabled = QCheckBox(
            "Remove isolated water-column noise (flood-fill)"
        )
        bf.addRow(self._benthic_enabled)
        self._benthic_radius_spin = QDoubleSpinBox()
        self._benthic_radius_spin.setRange(0.1, 5.0)
        self._benthic_radius_spin.setDecimals(2)
        self._benthic_radius_spin.setValue(0.3)
        self._benthic_radius_spin.setSuffix(" m")
        bf.addRow("Search Radius:", self._benthic_radius_spin)
        self._benthic_slope_spin = QDoubleSpinBox()
        self._benthic_slope_spin.setRange(0.05, 2.0)
        self._benthic_slope_spin.setDecimals(2)
        self._benthic_slope_spin.setValue(0.35)
        bf.addRow("Max Slope:", self._benthic_slope_spin)
        layout.addWidget(benthic_group)

        # ── Progress ──
        self._status = QLabel("Ready")
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        self._run_btn = QPushButton("▶ Run Bathymetry Pipeline")
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self._ok_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout)

    # ── Browse handlers ────────────────────────────────────────────

    def _get_traj_separator(self) -> Optional[str]:
        """Get the active trajectory separator. Returns None for auto-detect."""
        label = self._traj_sep_combo.currentText()
        if label == "Auto":
            return None
        if label == "Comma (,)":
            return ","
        if label == "Semicolon (;)":
            return ";"
        if label == "Tab":
            return "\t"
        if label == "Space":
            return r"\s+"
        return self._traj_custom_sep.text() or None

    def _on_traj_sep_changed(self):
        self._traj_custom_sep.setVisible(
            self._traj_sep_combo.currentText() == "Custom"
        )
        if self._traj_path_edit.text():
            self._parse_traj_preview(self._traj_path_edit.text())

    def _parse_traj_preview(self, path: str):
        """Parse first few lines of trajectory to populate column combos."""
        sep = self._get_traj_separator()
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                lines = [l.rstrip("\n\r") for l in f.readlines()]
        except Exception:
            return

        if not lines:
            return

        # Handle header
        has_header = self._traj_has_header.isChecked()
        first_line = lines[0]
        if has_header and len(lines) > 1:
            data_line = lines[1]
        else:
            data_line = first_line

        # Split
        if sep == " ":
            parts = data_line.split()
        elif sep is not None:
            parts = data_line.split(sep)
        else:
            # Auto-detect
            for s in [",", ";", "\t"]:
                parts = data_line.split(s)
                if len(parts) >= 4:
                    break
            else:
                parts = data_line.split()

        n_cols = len(parts)

        # Populate column combos
        for cb in (self._traj_time_col, self._traj_x_col,
                    self._traj_y_col, self._traj_z_col):
            cb.clear()
        for i in range(n_cols):
            label = f"Col {i}" if has_header else f"Col {i}: {parts[i][:12]}"
            for cb in (self._traj_time_col, self._traj_x_col,
                        self._traj_y_col, self._traj_z_col):
                cb.addItem(label, i)

        # Auto-detect from header names
        if has_header:
            header_parts = first_line.split() if sep in (None, " ") else first_line.split(sep if sep else None)
            if sep == " ":
                header_parts = first_line.split()
            elif sep is not None:
                header_parts = first_line.split(sep)
            else:
                header_parts = first_line.split()
            for i, h in enumerate(header_parts):
                low = h.strip().lower().strip('"').strip("'")
                if low in ("time", "t", "gps_time", "gpstime", "timestamp"):
                    self._traj_time_col.setCurrentIndex(i)
                elif low in ("x", "easting", "east", "lon", "longitude"):
                    self._traj_x_col.setCurrentIndex(i)
                elif low in ("y", "northing", "north", "lat", "latitude"):
                    self._traj_y_col.setCurrentIndex(i)
                elif low in ("z", "elev", "elevation", "height", "alt", "altitude"):
                    self._traj_z_col.setCurrentIndex(i)
        else:
            # Default: Col 0=time, Col 1=X, Col 2=Y, Col 3=Z
            if n_cols >= 4:
                self._traj_time_col.setCurrentIndex(0)
                self._traj_x_col.setCurrentIndex(1)
                self._traj_y_col.setCurrentIndex(2)
                self._traj_z_col.setCurrentIndex(3)

        # Preview
        preview = "\n".join(lines[:3])
        if len(lines) > 3:
            preview += f"\n… ({len(lines)} lines)"
        self._traj_preview.setText(preview)

    def _on_browse_trajectory(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Trajectory File", "",
            "Text Files (*.txt *.csv *.dat);;All Files (*)"
        )
        if path:
            self._traj_path_edit.setText(path)
            self._parse_traj_preview(path)

    def _on_browse_geotiff(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Water Surface GeoTIFF", "",
            "GeoTIFF Files (*.tif *.tiff *.vrt);;All Files (*)"
        )
        if not path:
            return
        try:
            surface, georef = load_water_surface_geotiff(path)
            self._geotiff_surface = surface
            self._geotiff_georef = georef
            self._geotiff_path_edit.setText(path)

            geotiff_epsg = georef.get("epsg")
            info = (
                f"Loaded: {surface.shape[1]}×{surface.shape[0]} px, "
                f"range [{np.nanmin(surface):.2f}, {np.nanmax(surface):.2f}] m"
            )
            if geotiff_epsg:
                info += f", EPSG:{geotiff_epsg}"
            if self._data_epsg and geotiff_epsg and self._data_epsg != geotiff_epsg:
                info += f"\n⚠ CRS mismatch: GeoTIFF EPSG:{geotiff_epsg} vs data EPSG:{self._data_epsg}"
            self._geotiff_info.setText(info)
            self._geotiff_info.setVisible(True)
            self._ws_geotiff_radio.setChecked(True)
        except Exception as exc:
            self._geotiff_info.setText(f"⚠ Error loading GeoTIFF: {exc}")
            self._geotiff_info.setVisible(True)

    # ── Crop GeoTIFF browse ──────────────────────────────────────

    def _on_browse_crop_geotiff(self):
        """Browse for a water surface GeoTIFF for the crop step."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Water Surface GeoTIFF for Cropping", "",
            "GeoTIFF Files (*.tif *.tiff *.vrt);;All Files (*)"
        )
        if not path:
            return
        try:
            surface, georef = load_water_surface_geotiff(path)
            self._crop_surface = surface
            self._crop_georef = georef
            self._crop_geotiff_edit.setText(path)

            # CRS comparison
            geotiff_epsg = georef.get("epsg")
            crs_info = f"Loaded: {surface.shape[1]}×{surface.shape[0]} px, "
            crs_info += f"range [{np.nanmin(surface):.2f}, {np.nanmax(surface):.2f}] m, "
            crs_info += f"CRS: {georef['crs'][:60]}"
            if geotiff_epsg:
                crs_info += f" (EPSG:{geotiff_epsg})"
            if self._data_epsg and geotiff_epsg and self._data_epsg != geotiff_epsg:
                crs_info += (
                    f"\n⚠ WARNING: GeoTIFF EPSG:{geotiff_epsg} differs from "
                    f"data EPSG:{self._data_epsg} — crop will likely fail! "
                    "Reproject the GeoTIFF to match the data CRS."
                )
            elif self._data_epsg and geotiff_epsg:
                crs_info += " ✓ CRS matches data"
            elif not geotiff_epsg:
                crs_info += "\n⚠ Could not determine GeoTIFF EPSG"
            self._crop_geotiff_info.setText(crs_info)
            self._crop_geotiff_info.setVisible(True)
        except Exception as exc:
            self._crop_geotiff_info.setText(f"⚠ Error: {exc}")
            self._crop_geotiff_info.setVisible(True)

    # ── Run ────────────────────────────────────────────────────────

    def _load_tile_for_batch(self, tile_id: str):
        """Load tile data for batch processing. Called from worker thread."""
        # We need access to the tile manager — get it from the parent
        parent = self.parent()
        if parent and hasattr(parent, '_tm'):
            return parent._tm.load_tile_points_full(tile_id)
        return None

    def _on_run(self):
        steps = []
        # Step 0: water surface crop (runs first, before Snell)
        if (self._ws_crop_enabled.isChecked()
                and self._crop_surface is not None
                and self._crop_georef is not None):
            steps.append({
                "name": "ws_crop",
                "surface": self._crop_surface,
                "georef": self._crop_georef,
                "tolerance_above_cm": self._crop_tol_spin.value(),
                "extrapolate_m": self._crop_extrap_spin.value(),
                "data_epsg": self._data_epsg,
            })
        # Step 1: Snell's refraction
        if self._snells_enabled.isChecked():
            # Determine water surface input
            if (self._ws_geotiff_radio.isChecked()
                    and self._geotiff_surface is not None
                    and self._geotiff_georef is not None):
                # Pass GeoTIFF for on-demand per-tile sampling
                ws_z = None  # sampled per-tile during processing
                ws_geotiff_surface = self._geotiff_surface
                ws_geotiff_georef = self._geotiff_georef
                ws_fallback = self._ws_z_spin.value()
            else:
                ws_z = self._ws_z_spin.value()
                ws_geotiff_surface = None
                ws_geotiff_georef = None
                ws_fallback = 0.0

            steps.append({
                "name": "snells",
                "water_surface_z": ws_z,
                "geotiff_surface": ws_geotiff_surface,
                "geotiff_georef": ws_geotiff_georef,
                "ws_fallback": ws_fallback,
                "n_water": self._n_water_spin.value(),
                "use_trajectory": bool(self._traj_path_edit.text().strip()),
                "trajectory_path": self._traj_path_edit.text().strip(),
                "separator": self._get_traj_separator(),
                "skip_rows": self._traj_skip_spin.value(),
                "has_header": self._traj_has_header.isChecked(),
                "time_col": self._traj_time_col.currentData() or 0,
                "x_col": self._traj_x_col.currentData() or 1,
                "y_col": self._traj_y_col.currentData() or 2,
                "z_col": self._traj_z_col.currentData() or 3,
            })
        if self._ws_enabled.isChecked():
            steps.append({
                "name": "water_surface",
                "tile_length": self._ws_tile_spin.value(),
            })
        if self._river_enabled.isChecked():
            steps.append({"name": "river_crop"})
        if self._benthic_enabled.isChecked():
            steps.append({
                "name": "benthic",
                "search_radius": self._benthic_radius_spin.value(),
                "max_slope": self._benthic_slope_spin.value(),
            })

        if not steps:
            self._status.setText("No steps selected")
            return

        self._run_btn.setEnabled(False)
        self._ok_btn.setEnabled(False)

        # ── Get workers setting ────────────────────────────────────
        from ..gui.settings_dialog import load_general_settings
        settings = load_general_settings()
        workers = settings.get("bathy_workers", 4)

        # ── Capture tile manager reference for thread-safe loading ─
        parent = self.parent()
        tm = parent._tm if parent and hasattr(parent, '_tm') else None
        if tm is None:
            self._status.setText("No tile manager available")
            self._run_btn.setEnabled(True)
            self._ok_btn.setEnabled(True)
            return

        def _loader(tile_id: str):
            return tm.load_tile_points_full(tile_id)

        # ── Launch parallel worker ─────────────────────────────────
        self._bathy_worker = _BathyWorker(
            self._tile_ids, steps, _loader, max_workers=workers, parent=self,
        )
        self._bathy_worker.progress.connect(self._on_bathy_progress)
        self._bathy_worker.tile_done.connect(self.tile_save_requested.emit)
        self._bathy_worker.finished_all.connect(self._on_bathy_finished)
        self._bathy_worker.error.connect(self._on_bathy_error)
        self._bathy_worker.start()

    def _on_bathy_progress(self, msg: str, pct: float):
        self._status.setText(msg)
        self._progress.setValue(int(pct))

    def _on_bathy_finished(self, n_processed: int):
        self._status.setText(f"Done: {n_processed}/{len(self._tile_ids)} tiles")
        self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self._ok_btn.setEnabled(True)
        self.bathy_applied.emit()

    def _on_bathy_error(self, msg: str):
        self._status.setText(f"Error: {msg}")
        self._run_btn.setEnabled(True)
        self._ok_btn.setEnabled(True)

    def _on_accept(self):
        self.accept()

    @property
    def results(self) -> list:
        return []
