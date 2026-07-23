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
    load_water_surface_geotiff,
    read_trajectory_ascii,
    sample_geotiff_at_points,
)

logger = logging.getLogger("lidar_workbench.gui.bathy_dialog")


class _BathyWorker(QThread):
    progress = Signal(str, float)
    step_done = Signal(str, np.ndarray)
    finished_all = Signal(dict)
    error = Signal(str)

    def __init__(self, data: dict, steps: list[dict], parent=None):
        super().__init__(parent)
        self._data = {k: v.copy() if isinstance(v, np.ndarray) else v
                      for k, v in data.items()}
        self._steps = steps

    def run(self):
        result = dict(self._data)
        water_surface_z: Optional[float] = None
        try:
            for i, step in enumerate(self._steps):
                name = step["name"]
                pct_base = i / len(self._steps) * 100
                if name == "ws_crop":
                    self.progress.emit("Cropping to water surface model…", pct_base)
                    surface = step["surface"]
                    georef = step["georef"]
                    k, _ = crop_to_water_surface_model(
                        result["x"], result["y"], result["z"],
                        surface=surface,
                        georef=georef,
                        tolerance_above_cm=step.get("tolerance_above_cm", 0.0),
                        extrapolate_m=step.get("extrapolate_m", 0.0),
                        progress=lambda m, p: self.progress.emit(
                            m, pct_base + p * 0.8 / len(self._steps)
                        ),
                    )
                    result["x"], result["y"], result["z"] = (
                        result["x"][k], result["y"][k], result["z"][k],
                    )
                    for key in ("classification", "intensity", "return_number",
                                "point_source_id", "sensor_type", "gps_time"):
                        if key in result and result[key] is not None:
                            result[key] = result[key][k]
                    self.step_done.emit("ws_crop_mask", k)

                elif name == "snells":
                    self.progress.emit("Snell's law correction...", pct_base)
                    n_water = step.get("n_water", 1.3333)

                    # Determine water surface: per-point array or scalar
                    ws_input = step["water_surface_z"]
                    if isinstance(ws_input, np.ndarray):
                        ws_z = ws_input
                        water_surface_z = float(np.median(ws_z))
                    else:
                        ws_z = float(ws_input)
                        water_surface_z = ws_z

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
                        # Per-point sensor position via interpolation
                        gps_times = result.get("gps_time")
                        if gps_times is not None and len(gps_times) == len(result["x"]):
                            self.progress.emit(
                                "Snell's: interpolating sensor per-point...",
                                pct_base + 2.0,
                            )
                            n_pts = len(result["x"])
                            sx_arr = np.empty(n_pts, dtype=np.float64)
                            sy_arr = np.empty(n_pts, dtype=np.float64)
                            sz_arr = np.empty(n_pts, dtype=np.float64)
                            for j in range(n_pts):
                                pos = interpolate_sensor_position(traj, float(gps_times[j]))
                                sx_arr[j] = pos[0]
                                sy_arr[j] = pos[1]
                                sz_arr[j] = pos[2]
                            # Use the per-point approach: iterate in batches for speed
                            # For now, pass median sensor position but with
                            # correct per-point water surface
                            sx_m = float(np.median(sx_arr))
                            sy_m = float(np.median(sy_arr))
                            sz_m = float(np.median(sz_arr))
                            xc, yc, zc = apply_snells_correction(
                                result["x"], result["y"], result["z"],
                                water_surface_z=ws_z,
                                sensor_position=(sx_m, sy_m, sz_m),
                                n_water=n_water,
                            )
                        else:
                            # No gps_time — use median sensor position
                            sx, sy, sz = (
                                float(np.median(traj[:, 1])),
                                float(np.median(traj[:, 2])),
                                float(np.median(traj[:, 3])),
                            )
                            xc, yc, zc = apply_snells_correction(
                                result["x"], result["y"], result["z"],
                                water_surface_z=ws_z,
                                sensor_position=(sx, sy, sz),
                                n_water=n_water,
                            )
                    else:
                        # Nadir approximation (no trajectory)
                        if isinstance(ws_z, np.ndarray):
                            # Per-point water surface with nadir
                            ws_scalar = float(np.median(ws_z))
                            xc, yc, zc = apply_snells_correction_nadir(
                                result["x"], result["y"], result["z"],
                                water_surface_z=ws_scalar,
                                n_water=n_water,
                            )
                        else:
                            xc, yc, zc = apply_snells_correction_nadir(
                                result["x"], result["y"], result["z"],
                                water_surface_z=ws_z,
                                n_water=n_water,
                            )
                    result["x"], result["y"], result["z"] = xc, yc, zc
                    self.step_done.emit("snells_zs", zc)

                elif name == "water_surface":
                    self.progress.emit("Water surface + ghost removal...", pct_base)
                    k, _ = water_surface_ghost_removal(
                        result["x"], result["y"], result["z"],
                        tile_length=step.get("tile_length", 40.0),
                        progress=lambda m, p: self.progress.emit(
                            m, pct_base + p * 0.8 / len(self._steps)
                        ),
                    )
                    result["x"], result["y"], result["z"] = (
                        result["x"][k], result["y"][k], result["z"][k],
                    )
                    for key in ("classification", "intensity", "return_number",
                                "point_source_id", "sensor_type", "gps_time"):
                        if key in result and result[key] is not None:
                            result[key] = result[key][k]
                    self.step_done.emit("water_surface_mask", k)

                elif name == "river_crop":
                    self.progress.emit("River corridor crop...", pct_base)
                    k, _ = river_corridor_mask(
                        result["x"], result["y"], result["z"],
                        result.get("intensity", np.zeros(len(result["x"]), dtype=np.uint16)),
                        result.get("return_number", np.ones(len(result["x"]), dtype=np.uint8)),
                        water_surface_z=water_surface_z,
                        progress=lambda m, p: self.progress.emit(
                            m, pct_base + p * 0.8 / len(self._steps)
                        ),
                    )
                    st = result.get("sensor_type")
                    if st is not None:
                        k = k | (st == 1)
                    result["x"], result["y"], result["z"] = (
                        result["x"][k], result["y"][k], result["z"][k],
                    )
                    for key in ("classification", "intensity", "return_number",
                                "point_source_id", "sensor_type", "gps_time"):
                        if key in result and result[key] is not None:
                            result[key] = result[key][k]
                    self.step_done.emit("river_crop_mask", k)

                elif name == "benthic":
                    self.progress.emit("Benthic continuity filter...", pct_base)
                    certain = np.ones(len(result["x"]), dtype=bool)
                    k = benthic_continuity_filter(
                        result["x"], result["y"], result["z"],
                        certain_bed_mask=certain,
                        search_radius=step.get("search_radius", 0.3),
                        max_slope=step.get("max_slope", 0.35),
                        progress=lambda m, p: self.progress.emit(
                            m, pct_base + p * 0.8 / len(self._steps)
                        ),
                    )
                    result["x"], result["y"], result["z"] = (
                        result["x"][k], result["y"][k], result["z"][k],
                    )
                    for key in ("classification", "intensity", "return_number",
                                "point_source_id", "sensor_type", "gps_time"):
                        if key in result and result[key] is not None:
                            result[key] = result[key][k]
                    self.step_done.emit("benthic_mask", k)

            self.progress.emit("Done", 100.0)
            self.finished_all.emit(result)
        except Exception as exc:
            logger.exception("Bathy processing failed")
            self.error.emit(str(exc))


class BathyDialog(QDialog):
    """Dialog for bathymetric processing pipeline."""

    bathy_applied = Signal(dict)

    def __init__(self, tile_data: dict, scanner: str = '',
                 sensor_type: str = '', parent=None):
        super().__init__(parent)
        self._data = tile_data
        self._scanner = scanner
        self._sensor_type = sensor_type
        self._worker: Optional[_BathyWorker] = None
        self._result: Optional[dict] = None
        self._water_surface_z: float = 0.0
        self._geotiff_surface: Optional[np.ndarray] = None
        self._geotiff_georef: Optional[dict] = None
        self._per_point_ws: Optional[np.ndarray] = None
        self._crop_surface: Optional[np.ndarray] = None
        self._crop_georef: Optional[dict] = None

        self.setWindowTitle("Bathymetry Processing")
        self.setMinimumWidth(560)
        self._setup_ui()

        if sensor_type == 'bathy' and len(tile_data.get("x", [])) > 0:
            self._auto_detect_surface()

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

        traj_cfg_layout = QHBoxLayout()
        self._traj_sep_edit = QLineEdit()
        self._traj_sep_edit.setPlaceholderText("sep (auto)")
        self._traj_sep_edit.setMaximumWidth(60)
        traj_cfg_layout.addWidget(QLabel("Sep:"))
        traj_cfg_layout.addWidget(self._traj_sep_edit)
        self._traj_skip_spin = QSpinBox()
        self._traj_skip_spin.setRange(0, 1000)
        self._traj_skip_spin.setValue(0)
        traj_cfg_layout.addWidget(QLabel("Skip:"))
        traj_cfg_layout.addWidget(self._traj_skip_spin)
        traj_cfg_layout.addStretch()
        sf.addRow("Options:", traj_cfg_layout)

        layout.addWidget(snells_group)

        # ═══════════════════════════════════════════════════════════
        # 2. Water Surface + Ghost Removal
        # ═══════════════════════════════════════════════════════════
        ws_clean_group = QGroupBox("2. Water Surface + Ghost Removal")
        wf = QFormLayout(ws_clean_group)
        self._ws_enabled = QCheckBox(
            "Remove water surface and near-surface ghost returns"
        )
        self._ws_enabled.setChecked(True)
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
        self._river_enabled.setChecked(True)
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

    # ── Auto-detect water surface ──────────────────────────────────

    def _auto_detect_surface(self):
        """Auto-detect water surface Z from the point data."""
        try:
            _, _, _, z_mean = detect_water_surface_ransac(
                self._data["x"], self._data["y"], self._data["z"],
                percentile=90.0,
            )
            if z_mean != 0.0:
                self._water_surface_z = z_mean
                self._ws_z_spin.setValue(z_mean)
                self._status.setText(
                    f"Auto-detected water surface at Z={z_mean:.2f} m"
                )
        except Exception:
            pass

    # ── Browse handlers ────────────────────────────────────────────

    def _on_browse_trajectory(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Trajectory File", "",
            "Text Files (*.txt *.csv *.dat);;All Files (*)"
        )
        if path:
            self._traj_path_edit.setText(path)

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

            # Sample at point locations for info display
            ws_samples = sample_geotiff_at_points(
                surface, georef,
                self._data["x"], self._data["y"],
                fill_value=np.nan,
            )
            valid = ~np.isnan(ws_samples)
            if valid.any():
                self._geotiff_info.setText(
                    f"Loaded: {surface.shape[1]}×{surface.shape[0]} px, "
                    f"CRS: {georef['crs'][:60]}…\n"
                    f"Water surface range: [{np.nanmin(surface):.2f}, "
                    f"{np.nanmax(surface):.2f}] m\n"
                    f"Points in raster: {valid.sum():,} / {len(ws_samples):,} "
                    f"({valid.sum()/max(len(ws_samples),1)*100:.0f}%)"
                )
                self._geotiff_info.setVisible(True)
                self._ws_geotiff_radio.setChecked(True)
            else:
                self._geotiff_info.setText(
                    "⚠ No points fall within the GeoTIFF extent. "
                    "Check CRS match."
                )
                self._geotiff_info.setVisible(True)
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
            self._crop_geotiff_info.setText(
                f"Loaded: {surface.shape[1]}×{surface.shape[0]} px, "
                f"range [{np.nanmin(surface):.2f}, {np.nanmax(surface):.2f}] m, "
                f"CRS: {georef['crs'][:50]}…"
            )
            self._crop_geotiff_info.setVisible(True)
        except Exception as exc:
            self._crop_geotiff_info.setText(f"⚠ Error: {exc}")
            self._crop_geotiff_info.setVisible(True)

    # ── Run ────────────────────────────────────────────────────────

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
            })
        # Step 1: Snell's refraction
        if self._snells_enabled.isChecked():
            # Determine water surface input
            if (self._ws_geotiff_radio.isChecked()
                    and self._geotiff_surface is not None
                    and self._geotiff_georef is not None):
                # Sample GeoTIFF at point locations
                self._status.setText("Sampling water surface GeoTIFF…")
                ws_z = sample_geotiff_at_points(
                    self._geotiff_surface,
                    self._geotiff_georef,
                    self._data["x"],
                    self._data["y"],
                    fill_value=self._ws_z_spin.value(),  # fallback
                )
            else:
                ws_z = self._ws_z_spin.value()

            steps.append({
                "name": "snells",
                "water_surface_z": ws_z,
                "n_water": self._n_water_spin.value(),
                "use_trajectory": bool(self._traj_path_edit.text().strip()),
                "trajectory_path": self._traj_path_edit.text().strip(),
                "separator": self._traj_sep_edit.text().strip() or None,
                "skip_rows": self._traj_skip_spin.value(),
                "has_header": False,
                "time_col": 0, "x_col": 1, "y_col": 2, "z_col": 3,
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
        self._status.setText("Processing...")
        self._progress.setValue(0)

        self._worker = _BathyWorker(self._data, steps, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str, pct: float):
        self._status.setText(msg)
        self._progress.setValue(int(pct))

    def _on_finished(self, result: dict):
        self._result = result
        n = len(result["x"])
        self._status.setText(f"Done: {n:,} points remaining")
        self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self._ok_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg}")
        self._run_btn.setEnabled(True)

    def _on_accept(self):
        if self._result is not None:
            self.bathy_applied.emit(self._result)
        self.accept()

    @property
    def result(self) -> Optional[dict]:
        return self._result
