"""
LiDAR Workbench — Ground Classification Dialog.

Dedicated dialog for classifying ground points using industry-standard
algorithms: SMRF (Simple Morphological Filter) and Progressive TIN
Densification (Axelsson).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal, QThread
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..ground import (
    ground_classify_smrf,
    ground_classify_stepdown,
    ground_classify_epptd,
    ground_classify_epptd_two_pass,
    ground_classify_aptd,
    ground_classify_hptd,
    ground_classify_dl_hybrid_ptd,
    ground_classify_multiscale_alpha_shape,
    ground_classify_egs_csf,
)
from ..tile_manager import _read_las_header_template, _write_las_file

logger = logging.getLogger("lidar_workbench.gui.ground_classify_dialog")


def _load_las_for_ground(las_path: str, flightline_sensor_types: dict) -> dict:
    """Load a tile's points directly from a LAS file (runs in a worker
    process).  Mirrors :meth:`TileManager.load_tile_points_full` so the
    ground classifiers receive identical arrays."""
    import laspy

    with laspy.open(las_path) as reader:
        las_data = reader.read()
        result = {
            "x": np.array(las_data.x, dtype=np.float64),
            "y": np.array(las_data.y, dtype=np.float64),
            "z": np.array(las_data.z, dtype=np.float64),
        }
        _safe = lambda attr, dtype, default: (
            np.array(getattr(las_data, attr), dtype=dtype)
            if hasattr(las_data, attr) else
            np.full(len(result["x"]), default, dtype=dtype)
        )
        result["classification"] = _safe("classification", np.uint8, 0)
        result["intensity"] = _safe("intensity", np.uint16, 0)
        result["return_number"] = _safe("return_number", np.uint8, 1)
        result["num_returns"] = _safe("num_returns", np.uint8, 1)
        result["point_source_id"] = _safe("point_source_id", np.uint16, 0)
        result["scan_direction_flag"] = _safe("scan_direction_flag", np.uint8, 0)
        result["edge_of_flight_line"] = _safe("edge_of_flight_line", np.uint8, 0)
        result["scan_angle_rank"] = _safe("scan_angle_rank", np.int8, 0)
        result["user_data"] = _safe("user_data", np.uint8, 0)
        result["gps_time"] = _safe("gps_time", np.float64, 0.0)
        result["red"] = _safe("red", np.uint16, 0)
        result["green"] = _safe("green", np.uint16, 0)
        result["blue"] = _safe("blue", np.uint16, 0)
        result["key_point"] = _safe("key_point", np.uint8, 0)
        result["synthetic"] = _safe("synthetic", np.uint8, 0)
        result["withheld"] = _safe("withheld", np.uint8, 0)
        result["overlap"] = _safe("overlap", np.uint8, 0)

        extra_dims = {}
        for ed in las_data.point_format.extra_dimensions:
            name = ed.name
            if hasattr(las_data, name):
                extra_dims[name] = np.array(getattr(las_data, name))
        if extra_dims:
            result["extra_dims"] = extra_dims

    # Per-point sensor_type from the flightline→sensor mapping.
    fl_to_st = {}
    for k, v in (flightline_sensor_types or {}).items():
        if v == "topo":
            fl_to_st[int(k)] = 1
        elif v == "bathy":
            fl_to_st[int(k)] = 2
    if fl_to_st:
        src_ids = result["point_source_id"]
        st_arr = np.zeros(len(result["x"]), dtype=np.uint8)
        for fl, st_code in fl_to_st.items():
            st_arr[src_ids == fl] = st_code
        result["sensor_type"] = st_arr

    return result


def _ground_process_tile(tile_id: str, las_path: str, method: str,
                         params: dict, source_class: int,
                         flightline_sensor_types: dict) -> dict:
    """Run ground classification on one tile inside a worker process.

    Loads the LAS, classifies SMRF/TIN, writes the updated classification
    back to disk, and returns a small summary dict (safe to pickle back
    to the GUI process)."""
    import time

    las_path = Path(las_path)

    t0 = time.perf_counter()
    result = {
        "tile_id": tile_id,
        "n_ground": 0,
        "n_affected": 0,
        "n_source": 0,
        "duration": 0.0,
    }

    data = _load_las_for_ground(str(las_path), flightline_sensor_types)
    cls_all = data["classification"]
    n_total = len(cls_all)
    sc = source_class

    if method == "dl_hybrid" and sc == 2:
        source_mask = (cls_all == 0) | (cls_all == 1) | (cls_all == 2)
        sc = -2
    elif sc == -2:
        # All unclassified (0, 1) and existing ground (2)
        source_mask = (cls_all == 0) | (cls_all == 1) | (cls_all == 2)
    elif sc == -3:
        # "1 → 2": densify unclassified (0, 1) onto existing ground (2).
        # Class 2 is the trusted initial TIN and is left untouched.
        source_mask = (cls_all == 0) | (cls_all == 1) | (cls_all == 2)
    elif sc >= 0:
        source_mask = (cls_all == sc)
    else:
        source_mask = np.ones(n_total, dtype=bool)

    n_source = int(source_mask.sum())
    result["n_source"] = n_source
    if n_source == 0:
        logger.warning(
            "Ground classification: 0 points matched source class %s (total points in tile: %d). "
            "Please check 'Source Class' filter in the dialog.",
            sc, n_total
        )
        return result

    xs_sub = data["x"][source_mask]
    ys_sub = data["y"][source_mask]
    zs_sub = data["z"][source_mask]
    rn = data.get("return_number")
    nr = data.get("num_returns")
    st = data.get("sensor_type")
    rn_sub = rn[source_mask] if rn is not None else None
    nr_sub = nr[source_mask] if nr is not None else None
    st_sub = st[source_mask] if st is not None else None
    extra_dims = data.get("extra_dims")
    exd_sub = (
        {k: np.asarray(v)[source_mask] for k, v in extra_dims.items()}
        if extra_dims else None
    )

    if method == "smrf":
        mask = ground_classify_smrf(
            xs_sub, ys_sub, zs_sub,
            cell_size=params.get("cell_size"),
            slope_threshold=params["slope_threshold"],
            max_window=params["max_window"],
            elevation_threshold=params["elevation_threshold"],
            base_window=params.get("base_window", 2.0),
            window_growth=params.get("window_growth", 1.5),
        )
    elif method == "aptd":
        mask = ground_classify_aptd(
            xs_sub, ys_sub, zs_sub,
            k_neighbors=params.get("k_neighbors", 4),
            radius_factor=params.get("radius_factor", 4.0),
            min_neighbors=params.get("min_neighbors", 2),
            primary_grid_size=params.get("primary_grid_size"),
            point_count_threshold=params.get("point_count_threshold", 5),
            slope_threshold=params.get("slope_threshold", 0.5),
            max_angle=params.get("max_angle"),
            max_distance=params.get("max_distance"),
            max_terrain_angle=params.get("max_terrain_angle"),
            exclude_single_returns_in_water=params.get("exclude_single_returns_in_water", False),
            sensor_type=st_sub,
            return_numbers=rn_sub,
            num_returns=nr_sub,
        )
    elif method == "hptd":
        mask = ground_classify_hptd(
            xs_sub, ys_sub, zs_sub,
            window_size=params.get("window_size"),
            step_factor=params.get("step_factor", 0.5),
            max_angle=params.get("max_angle", 6.0),
            max_distance=params.get("max_distance"),
            relative_elevation_factor=params.get("relative_elevation_factor", 1.0),
            signed=params.get("signed", True),
            below_tolerance=params.get("below_tolerance", 0.5),
            max_terrain_angle=params.get("max_terrain_angle", 88.0),
            follow_surface_trend=params.get("follow_surface_trend", True),
            exclude_single_returns_in_water=params.get("exclude_single_returns_in_water", False),
            sensor_type=st_sub,
            return_numbers=rn_sub,
            num_returns=nr_sub,
        )
    elif method == "dl_hybrid":
        # Pointcept's ASPRS class 2 (ground) is the DL prior.
        dl_conf_sub = (
            (cls_all[source_mask] == 2).astype(np.float64)
            if cls_all is not None else None
        )
        mask = ground_classify_dl_hybrid_ptd(
            xs_sub, ys_sub, zs_sub,
            dl_ground_confidence=dl_conf_sub,
            dl_confidence_threshold=params.get("dl_confidence_threshold", 0.5),
            dl_seed_weight=params.get("dl_seed_weight", 1.0),
            max_distance=params.get("max_distance", 1.4),
            max_angle=params.get("max_angle", 6.0),
            max_terrain_angle=params.get("max_terrain_angle", 88.0),
            only_upward=params.get("only_upward", False),
            follow_surface_trend=params.get("follow_surface_trend", True),
            exclude_single_returns_in_water=params.get("exclude_single_returns_in_water", False),
            sensor_type=st_sub,
            return_numbers=rn_sub,
            num_returns=nr_sub,
        )
    elif method == "alpha_shape":
        mask = ground_classify_multiscale_alpha_shape(
            xs_sub, ys_sub, zs_sub,
            coarse_alpha=params.get("coarse_alpha", 20.0),
            medium_alpha=params.get("medium_alpha", 6.0),
            fine_alpha=params.get("fine_alpha", 2.0),
            max_distance=params.get("max_distance", 0.20),
            max_terrain_angle=params.get("max_terrain_angle", 45.0),
            all_returns=params.get("all_returns", False),
            return_numbers=rn_sub,
            num_returns=nr_sub,
        )
    elif method == "egs_csf":
        mask = ground_classify_egs_csf(
            xs_sub, ys_sub, zs_sub,
            cloth_resolution=params.get("cloth_resolution", 1.0),
            rigidness=params.get("rigidness", 2.0),
            time_step=params.get("time_step", 0.65),
            class_threshold=params.get("class_threshold", 0.30),
            gradient_factor=params.get("gradient_factor", 0.50),
            max_iterations=params.get("max_iterations", 150),
            spike_down=params.get("spike_down", 2.5),
            slope_smooth=params.get("slope_smooth", True),
            all_returns=params.get("all_returns", False),
            return_numbers=rn_sub,
            num_returns=nr_sub,
            sensor_type=st_sub,
            exclude_single_returns_in_water=params.get("exclude_single_returns_in_water", False),
        )
    elif method == "stepdown":
        existing_ground = None
        if sc == -3:
            existing_ground = (cls_all[source_mask] == 2)

        mask = ground_classify_stepdown(
            xs_sub, ys_sub, zs_sub,
            step=params.get("step", 5.0),
            sub_steps=params.get("sub_steps", 6),
            bulge=params.get("bulge"),
            offset=params.get("offset", 0.08),
            spike=params.get("spike", 0.50),
            spike_down=params.get("spike_down", 0.50),
            refine_loops=params.get("refine_loops", 2),
            all_returns=params.get("all_returns", False),
            return_numbers=rn_sub,
            num_returns=nr_sub,
            existing_ground_mask=existing_ground,
        )
    else:  # "epptd" (and legacy "tin")
        existing_ground = None
        if sc == -3:
            # Densify unclassified (0, 1) onto existing ground (2): class 2
            # points are the trusted initial TIN and must not be re-tested.
            existing_ground = (cls_all[source_mask] == 2)

        mask = ground_classify_epptd(
            xs_sub, ys_sub, zs_sub,
            seed_resolution_search=params.get("seed_resolution", 10.0),
            max_iteration_angle=params.get("max_angle", 6.0),
            max_iteration_distance=params.get("max_distance", 0.5),
            spacing=params.get("spacing", 0.50),
            buffer_size=params.get("buffer_size", 15.0),
            max_iter=params.get("max_iter", 15),
            dense=params.get("dense", True),
            dense_tolerance=params.get("dense_tolerance", 0.10),
            existing_ground_mask=existing_ground,
        )

    # Expand the source-only mask back to the full tile.
    full_mask = np.zeros(n_total, dtype=bool)
    non_source_ground = ~source_mask & (cls_all == 2)
    full_mask[non_source_ground] = True
    full_mask[source_mask] = mask

    new_cls = cls_all.copy()
    if sc == -2:
        in_source = (cls_all == 0) | (cls_all == 1) | (cls_all == 2)
        new_cls[in_source & full_mask] = 2
        new_cls[in_source & ~full_mask] = 1
        n_affected = int(in_source.sum())
    elif sc == -3:
        in_source = (cls_all == 0) | (cls_all == 1) | (cls_all == 2)
        was_ground = cls_all == 2
        # Existing ground stays untouched; only unclassified points are assigned.
        new_cls[in_source & was_ground] = 2
        new_cls[in_source & ~was_ground & full_mask] = 2
        new_cls[in_source & ~was_ground & ~full_mask] = 1
        n_affected = int(((cls_all == 0) | (cls_all == 1)).sum())
    elif sc >= 0:
        in_source = (cls_all == sc)
        new_cls[in_source & full_mask] = 2
        new_cls[in_source & ~full_mask] = 1
        n_affected = int(in_source.sum())
    else:
        new_cls[full_mask] = 2
        new_cls[~full_mask] = 1
        n_affected = n_total

    data["classification"] = new_cls

    # Preserve the original point format / VLRs / extra dims.
    try:
        header_tmpl = _read_las_header_template(las_path)
    except Exception:
        header_tmpl = None

    # Ground results live in <tiles>/ground — never overwrite the source tile.
    base_dir = las_path.parent
    if base_dir.name in ("ground", "bathy", "noise"):
        base_dir = base_dir.parent
    ground_dir = base_dir / "ground"
    ground_dir.mkdir(parents=True, exist_ok=True)
    clean_stem = las_path.stem
    while clean_stem.endswith("_ground"):
        clean_stem = clean_stem[:-7]
    out_path = ground_dir / f"{clean_stem}_ground.las"

    _write_las_file(
        out_path,
        data["x"], data["y"], data["z"],
        classes=data["classification"],
        intensities=data.get("intensity"),
        return_numbers=data.get("return_number"),
        num_returns=data.get("num_returns"),
        point_source_ids=data.get("point_source_id"),
        gps_times=data.get("gps_time"),
        scan_angle_ranks=data.get("scan_angle_rank"),
        scan_direction_flags=data.get("scan_direction_flag"),
        edge_of_flight_lines=data.get("edge_of_flight_line"),
        user_data_array=data.get("user_data"),
        reds=data.get("red"),
        greens=data.get("green"),
        blues=data.get("blue"),
        key_points=data.get("key_point"),
        synthetics=data.get("synthetic"),
        withhelds=data.get("withheld"),
        overlaps=data.get("overlap"),
        header_template=header_tmpl,
        extra_dims=data.get("extra_dims"),
    )

    result["n_ground"] = int(mask.sum())
    result["n_affected"] = n_affected
    result["out_path"] = str(out_path)
    result["duration"] = time.perf_counter() - t0
    return result


class _GroundBatchWorker(QThread):
    """Background worker that classifies ground points for many tiles in
    parallel using a process pool.

    Each tile is handled by a separate Python process (via
    ProcessPoolExecutor) so the GIL-bound TIN/SMRF densification loop can
    actually use all configured CPU cores.  Results are small summary dicts,
    emitted one per tile for the GUI to update status."""

    progress = Signal(str, float)          # message, percentage
    tile_done = Signal(str, dict)          # (tile_id, summary_dict)
    finished_all = Signal(int)             # number of tiles processed
    error = Signal(str)

    def __init__(self, jobs: list, method: str, params: dict,
                 source_class: int, workers: int = 4, db=None, parent=None):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._method = method
        self._params = params
        self._source_class = source_class
        self._workers = max(1, int(workers))
        self._db = db
        self._cancelled = False
        self._pool = None

    def cancel(self):
        """Signal cancellation and terminate child processes cleanly."""
        self._cancelled = True
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            try:
                for p in list(self._pool._processes.values()):
                    if p.is_alive():
                        p.terminate()
            except Exception:
                pass

    def run(self):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import time

        n_total = len(self._jobs)
        if n_total == 0:
            self.finished_all.emit(0)
            return

        t_start = time.time()
        processed = 0
        max_workers = min(self._workers, n_total)

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                self._pool = pool
                futures = {}
                for job in self._jobs:
                    if self._cancelled:
                        break
                    fut = pool.submit(
                        _ground_process_tile,
                        job["tile_id"], job["las_path"],
                        self._method, self._params, self._source_class,
                        job.get("flightline_sensor_types", {}),
                    )
                    futures[fut] = job["tile_id"]

                for fut in as_completed(futures):
                    if self._cancelled:
                        break
                    tid = futures[fut]
                    try:
                        result = fut.result()
                        processed += 1
                        if not self._cancelled:
                            self._record_time(tid, result)
                            self.tile_done.emit(tid, result)
                            self.progress.emit(
                                f"Done {processed}/{n_total} tiles",
                                processed / n_total * 100,
                            )
                    except Exception as exc:
                        processed += 1
                        logger.error(
                            "Ground classification error on %s: %s", tid, exc
                        )
                        if not self._cancelled:
                            self.error.emit(f"{tid}: {exc}")

            if not self._cancelled:
                elapsed = time.time() - t_start
                self.progress.emit(
                    f"Done: {processed}/{n_total} tiles in {elapsed:.1f}s", 100.0
                )
                self.finished_all.emit(processed)
        except Exception as exc:
            if not self._cancelled:
                logger.exception("Ground classification failed")
                self.error.emit(str(exc))
        finally:
            self._pool = None

    def _record_time(self, tile_id: str, result: dict) -> None:
        """Record per-tile processing time (thread-local SQLite connection)."""
        if self._db is None:
            return
        try:
            with self._db.connect() as conn:
                self._db.record_processing_time(
                    conn, tile_id, "ground",
                    duration_seconds=result.get("duration", 0.0),
                    point_count=result.get("n_source", 0),
                    params={**self._params, "method": self._method},
                    batch_id=None,
                )
        except Exception:
            pass


class GroundClassifyDialog(QDialog):
    """Dialog for ground classification over one or more tiles.

    Configures SMRF or Progressive-TIN ground classification and then runs
    it on every selected tile in parallel worker processes.  Each finished
    tile is emitted via ``tile_processed`` so the main window can update
    tile status and refresh the views."""

    ground_applied = Signal()            # all tiles processed
    tile_processed = Signal(str, dict)   # (tile_id, summary_dict)

    def __init__(self, jobs: list, db=None, parent=None):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._db = db
        self._worker: Optional[_GroundBatchWorker] = None

        n_tiles = len(self._jobs)
        title = "Ground Classification"
        if n_tiles > 1:
            title += f" ({n_tiles} tiles selected)"
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self._setup_ui()

    def reject(self):
        """Cancel: safely terminate background worker and child processes before closing."""
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Cancelling worker processes…")
            self._worker.cancel()
            try:
                self._worker.tile_done.disconnect()
            except Exception:
                pass
            try:
                self._worker.progress.disconnect()
            except Exception:
                pass
            try:
                self._worker.finished_all.disconnect()
            except Exception:
                pass
            try:
                self._worker.error.disconnect()
            except Exception:
                pass
            self._worker.wait(5000)
        super().reject()

    def closeEvent(self, event):
        """Handle window close button (X) cleanly."""
        self.reject()
        event.accept()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Method selector
        method_group = QGroupBox("Algorithm")
        mf = QFormLayout(method_group)
        self._method_combo = QComboBox()
        self._method_combo.addItem("Step-Down PTD — Multi-Scale Bulged TIN", "stepdown")
        self._method_combo.addItem("EP-PTD — Progressive TIN Densification (Axelsson)", "epptd")
        self._method_combo.addItem("SMRF — Simple Morphological Filter (Pingel et al.)", "smrf")
        self._method_combo.addItem("APTD — Adaptive Grid PTD (AGPTD)", "aptd")
        self._method_combo.addItem("H-PTD — Hierarchical/Fast PTD (FPTD)", "hptd")
        self._method_combo.addItem("DL-Hybrid PTD — Pointcept + PTD", "dl_hybrid")
        self._method_combo.addItem("M-AlphaShape — Multiscale 3D Alpha Shape (MDPI 2024)", "alpha_shape")
        self._method_combo.addItem("EGS-CSF — Evolutionary Gradient Cloth Simulation (IJDE 2025)", "egs_csf")
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        mf.addRow("Method:", self._method_combo)

        self._algo_guide_label = QLabel()
        self._algo_guide_label.setWordWrap(True)
        self._algo_guide_label.setStyleSheet(
            "QLabel { background-color: palette(alternate-base); border: 1px solid palette(mid); "
            "border-radius: 4px; padding: 6px 8px; font-size: 11px; }"
        )
        mf.addRow(self._algo_guide_label)

        # Source class filter — only reclassify points matching this class
        self._source_class_combo = QComboBox()
        self._source_class_combo.addItem("0, 1 & 2: All Unclassified + Ground", -2)
        self._source_class_combo.addItem("2: Ground (Pointcept default)", 2)
        self._source_class_combo.addItem("0: Created, Never Classified", 0)
        self._source_class_combo.addItem("1: Unclassified", 1)
        self._source_class_combo.addItem(
            "1 → 2: Densify Unclassified onto Existing Ground", -3
        )
        self._source_class_combo.addItem("All classes", -1)
        self._source_class_combo.setCurrentIndex(0)
        self._source_class_combo.setToolTip(
            "Only reclassify points matching the selected source class(es). "
            "Other classes are left unchanged.\n\n"
            "'0, 1 & 2' uses all unclassified points (classes 0 and 1) plus "
            "existing ground (class 2) as input, re-classifying them.\n\n"
            "'1 → 2' uses existing Class 2 ground as the trusted initial TIN and only "
            "reclassifies unclassified points against it. Existing ground is left untouched."
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
        self._smrf_group.setVisible(False)
        layout.addWidget(self._smrf_group)

        # EP-PTD params
        self._tin_group = QGroupBox("EP-PTD Parameters (Progressive TIN Densification)")
        tf = QFormLayout(self._tin_group)

        self._tin_preset_combo = QComboBox()
        self._tin_preset_combo.addItem("Flat / Gentle Terrain (Default)", "flat")
        self._tin_preset_combo.addItem("River Embankments & Levees", "embankment")
        self._tin_preset_combo.addItem("Hilly / Steep Terrain", "steep")
        self._tin_preset_combo.addItem("Urban / Large Buildings", "urban")
        self._tin_preset_combo.currentIndexChanged.connect(self._on_tin_preset_changed)
        tf.addRow("Terrain Preset:", self._tin_preset_combo)

        self._tin_dist = QDoubleSpinBox()
        self._tin_dist.setRange(0.05, 5.0)
        self._tin_dist.setDecimals(2)
        self._tin_dist.setValue(0.50)
        self._tin_dist.setSuffix(" m")
        self._tin_dist.setToolTip("Max perpendicular distance from candidate point to TIN surface (Axelsson iteration distance; default 0.50m).")
        tf.addRow("Max Distance:", self._tin_dist)

        self._tin_angle = QDoubleSpinBox()
        self._tin_angle.setRange(1.0, 45.0)
        self._tin_angle.setDecimals(1)
        self._tin_angle.setValue(6.0)
        self._tin_angle.setSuffix("°")
        self._tin_angle.setToolTip("Max angle between point and TIN vertices (Axelsson iteration angle; 5°-6° flat, 20°-30° for riverbanks/hills).")
        tf.addRow("Max Angle:", self._tin_angle)

        self._tin_seed_res = QDoubleSpinBox()
        self._tin_seed_res.setRange(1.0, 200.0)
        self._tin_seed_res.setDecimals(1)
        self._tin_seed_res.setValue(10.0)
        self._tin_seed_res.setSuffix(" m")
        self._tin_seed_res.setToolTip(
            "Initial coarse seed grid size (seed resolution). Evaluates two half-step shifted "
            "grids to sample lowest points without cell-edge artifacts."
        )
        tf.addRow("Seed Resolution:", self._tin_seed_res)

        self._tin_spacing = self._make_auto_spin(20.0, 2, "Auto")
        self._tin_spacing.setValue(0.50)
        self._tin_spacing.setToolTip(
            "Point spacing for candidate pre-thinning (spacing). Evaluates lowest point per spacing "
            "cell and freezes triangles smaller than spacing."
        )
        tf.addRow("Candidate Spacing (Auto):", self._tin_spacing)

        self._tin_buffer_size = QDoubleSpinBox()
        self._tin_buffer_size.setRange(0.0, 200.0)
        self._tin_buffer_size.setDecimals(1)
        self._tin_buffer_size.setValue(15.0)
        self._tin_buffer_size.setSuffix(" m")
        self._tin_buffer_size.setToolTip(
            "Boundary buffer width to prevent tile boundary edge distortion."
        )
        tf.addRow("Buffer Size:", self._tin_buffer_size)

        self._tin_max_iter = QSpinBox()
        self._tin_max_iter.setRange(1, 200)
        self._tin_max_iter.setValue(15)
        self._tin_max_iter.setToolTip(
            "Maximum progressive densification iterations."
        )
        tf.addRow("Max Iterations:", self._tin_max_iter)

        self._tin_full_density = QCheckBox("Classify Full Density Ground (all points onto TIN)")
        self._tin_full_density.setChecked(True)
        self._tin_full_density.setToolTip(
            "When unchecked (sparse mode), only the single lowest point per candidate cell "
            "(the TIN vertices) is marked as ground, resulting in a thinned/sparse ground model. "
            "When checked, all points sitting on the ground surface are classified so the ground "
            "cloud retains 100% full density."
        )
        tf.addRow(self._tin_full_density)

        self._tin_dense_tol = QDoubleSpinBox()
        self._tin_dense_tol.setRange(0.02, 0.50)
        self._tin_dense_tol.setDecimals(2)
        self._tin_dense_tol.setValue(0.10)
        self._tin_dense_tol.setSuffix(" m")
        self._tin_dense_tol.setToolTip(
            "Strict vertical tolerance to ground TIN facet for densification. "
            "Vague, noisy, or elevated vegetation points are strictly left unclassified."
        )
        self._tin_full_density.toggled.connect(self._tin_dense_tol.setEnabled)
        tf.addRow("Dense Tolerance:", self._tin_dense_tol)

        self._tin_group.setVisible(False)
        layout.addWidget(self._tin_group)

        # APTD (AGPTD) params
        self._aptd_group = QGroupBox("APTD (Adaptive Grid PTD) Parameters")
        af = QFormLayout(self._aptd_group)
        self._aptd_k = QSpinBox()
        self._aptd_k.setRange(2, 32)
        self._aptd_k.setValue(4)
        self._aptd_k.setToolTip("Neighbours for the Kd-tree elevation-outlier check")
        af.addRow("K Neighbours:", self._aptd_k)
        self._aptd_radius_factor = QDoubleSpinBox()
        self._aptd_radius_factor.setRange(1.0, 20.0)
        self._aptd_radius_factor.setDecimals(1)
        self._aptd_radius_factor.setValue(4.0)
        self._aptd_radius_factor.setToolTip("Radius = factor × max point spacing")
        af.addRow("Radius Factor:", self._aptd_radius_factor)
        self._aptd_min_neighbors = QSpinBox()
        self._aptd_min_neighbors.setRange(1, 20)
        self._aptd_min_neighbors.setValue(2)
        self._aptd_min_neighbors.setToolTip("Min neighbours within radius (else outlier)")
        af.addRow("Min Neighbours:", self._aptd_min_neighbors)
        self._aptd_point_count = QSpinBox()
        self._aptd_point_count.setRange(2, 100)
        self._aptd_point_count.setValue(5)
        self._aptd_point_count.setToolTip("Min points in a cell to consider grid refinement")
        af.addRow("Refine If Points ≥:", self._aptd_point_count)
        self._aptd_slope_threshold = QDoubleSpinBox()
        self._aptd_slope_threshold.setRange(0.05, 5.0)
        self._aptd_slope_threshold.setDecimals(2)
        self._aptd_slope_threshold.setValue(0.5)
        self._aptd_slope_threshold.setToolTip("Relative-slope threshold to refine a cell")
        af.addRow("Slope Threshold:", self._aptd_slope_threshold)
        self._aptd_angle = self._make_auto_spin(90.0, 1, "Auto")
        af.addRow("Max Angle (Auto):", self._aptd_angle)
        self._aptd_distance = self._make_auto_spin(10.0, 2, "Auto")
        af.addRow("Max Distance (Auto):", self._aptd_distance)
        self._aptd_terrain_angle = self._make_auto_spin(90.0, 1, "Auto")
        af.addRow("Max Terrain Angle (Auto):", self._aptd_terrain_angle)
        self._aptd_exclude_single_returns_water = QCheckBox(
            "Exclude single returns in water (bathy bed-only)"
        )
        af.addRow(self._aptd_exclude_single_returns_water)
        self._aptd_group.setVisible(False)
        layout.addWidget(self._aptd_group)

        # H-PTD (FPTD) params
        self._hptd_group = QGroupBox("H-PTD (Fast PTD) Parameters")
        hf = QFormLayout(self._hptd_group)
        self._hptd_window = self._make_auto_spin(200.0, 1, "Auto")
        hf.addRow("Window Size (Auto):", self._hptd_window)
        self._hptd_step_factor = QDoubleSpinBox()
        self._hptd_step_factor.setRange(0.1, 1.0)
        self._hptd_step_factor.setDecimals(2)
        self._hptd_step_factor.setValue(0.5)
        self._hptd_step_factor.setToolTip("Sliding-window step as a fraction of window size")
        hf.addRow("Step Factor:", self._hptd_step_factor)
        self._hptd_angle = QDoubleSpinBox()
        self._hptd_angle.setRange(1.0, 45.0)
        self._hptd_angle.setDecimals(1)
        self._hptd_angle.setValue(6.0)
        self._hptd_angle.setSuffix("°")
        hf.addRow("Max Angle:", self._hptd_angle)
        self._hptd_distance = self._make_auto_spin(10.0, 2, "Auto")
        hf.addRow("Max Distance (Auto):", self._hptd_distance)
        self._hptd_rel_elev = QDoubleSpinBox()
        self._hptd_rel_elev.setRange(0.0, 10.0)
        self._hptd_rel_elev.setDecimals(2)
        self._hptd_rel_elev.setValue(1.0)
        self._hptd_rel_elev.setToolTip("Relative elevation threshold grows with facet slope")
        hf.addRow("Relative Elev. Factor:", self._hptd_rel_elev)
        self._hptd_signed = QCheckBox("Signed computation (reject below-surface points)")
        self._hptd_signed.setChecked(True)
        hf.addRow(self._hptd_signed)
        self._hptd_below = QDoubleSpinBox()
        self._hptd_below.setRange(0.05, 10.0)
        self._hptd_below.setDecimals(2)
        self._hptd_below.setValue(0.5)
        self._hptd_below.setSuffix(" m")
        self._hptd_below.setToolTip("Reject points this far below the local TIN surface")
        hf.addRow("Below Tolerance:", self._hptd_below)
        self._hptd_terrain_angle = QDoubleSpinBox()
        self._hptd_terrain_angle.setRange(10.0, 90.0)
        self._hptd_terrain_angle.setDecimals(1)
        self._hptd_terrain_angle.setValue(88.0)
        self._hptd_terrain_angle.setSuffix("°")
        self._hptd_terrain_angle.setToolTip(
            "Max allowed slope of TIN triangles. "
            "Keep 88-90° to classify steep natural embankments and riverbanks; "
            "lower it only to exclude man-made vertical walls."
        )
        hf.addRow("Max Terrain Angle:", self._hptd_terrain_angle)
        self._hptd_follow_trend = QCheckBox("Follow surface trend (adapt to local slope)")
        self._hptd_follow_trend.setChecked(True)
        self._hptd_follow_trend.setToolTip(
            "Relaxes the iteration angle on steep facets for points above "
            "the TIN, so embankments and cliffs are climbed reliably."
        )
        hf.addRow(self._hptd_follow_trend)
        self._hptd_exclude_single_returns_water = QCheckBox(
            "Exclude single returns in water (bathy bed-only)"
        )
        hf.addRow(self._hptd_exclude_single_returns_water)
        self._hptd_group.setVisible(False)
        layout.addWidget(self._hptd_group)

        # DL-Hybrid PTD params
        self._dlhybrid_group = QGroupBox("DL-Hybrid PTD Parameters")
        df = QFormLayout(self._dlhybrid_group)
        self._dl_conf_threshold = QDoubleSpinBox()
        self._dl_conf_threshold.setRange(0.0, 1.0)
        self._dl_conf_threshold.setDecimals(2)
        self._dl_conf_threshold.setValue(0.5)
        self._dl_conf_threshold.setToolTip("Min DL confidence to treat a point as DL ground")
        df.addRow("DL Confidence Threshold:", self._dl_conf_threshold)
        self._dl_seed_weight = QDoubleSpinBox()
        self._dl_seed_weight.setRange(0.0, 10.0)
        self._dl_seed_weight.setDecimals(2)
        self._dl_seed_weight.setValue(1.0)
        self._dl_seed_weight.setToolTip("How strongly DL confidence relaxes the distance threshold")
        df.addRow("DL Seed Weight:", self._dl_seed_weight)
        self._dl_distance = QDoubleSpinBox()
        self._dl_distance.setRange(0.1, 10.0)
        self._dl_distance.setDecimals(2)
        self._dl_distance.setValue(1.4)
        self._dl_distance.setSuffix(" m")
        df.addRow("Max Distance:", self._dl_distance)
        self._dl_angle = QDoubleSpinBox()
        self._dl_angle.setRange(1.0, 45.0)
        self._dl_angle.setDecimals(1)
        self._dl_angle.setValue(6.0)
        self._dl_angle.setSuffix("°")
        df.addRow("Max Angle:", self._dl_angle)
        self._dl_terrain_angle = QDoubleSpinBox()
        self._dl_terrain_angle.setRange(10.0, 90.0)
        self._dl_terrain_angle.setDecimals(1)
        self._dl_terrain_angle.setValue(88.0)
        self._dl_terrain_angle.setSuffix("°")
        df.addRow("Max Terrain Angle:", self._dl_terrain_angle)
        self._dl_only_upward = QCheckBox("Only add points above initial surface")
        df.addRow(self._dl_only_upward)
        self._dl_follow_trend = QCheckBox("Follow surface trend (adapt to local slope)")
        self._dl_follow_trend.setChecked(True)
        df.addRow(self._dl_follow_trend)
        self._dl_exclude_single_returns_water = QCheckBox(
            "Exclude single returns in water (bathy bed-only)"
        )
        df.addRow(self._dl_exclude_single_returns_water)
        self._dlhybrid_group.setVisible(False)
        layout.addWidget(self._dlhybrid_group)

        # M-AlphaShape params
        self._alpha_shape_group = QGroupBox("M-AlphaShape Parameters (MDPI 2024)")
        asf = QFormLayout(self._alpha_shape_group)
        self._as_coarse_alpha = QDoubleSpinBox()
        self._as_coarse_alpha.setRange(5.0, 100.0)
        self._as_coarse_alpha.setDecimals(1)
        self._as_coarse_alpha.setValue(20.0)
        self._as_coarse_alpha.setSuffix(" m")
        self._as_coarse_alpha.setToolTip("Coarse rolling sphere radius for large-scale terrain seeds (bridges roofs & tree canopies)")
        asf.addRow("Coarse Scale (α₁):", self._as_coarse_alpha)
        self._as_med_alpha = QDoubleSpinBox()
        self._as_med_alpha.setRange(2.0, 50.0)
        self._as_med_alpha.setDecimals(1)
        self._as_med_alpha.setValue(6.0)
        self._as_med_alpha.setSuffix(" m")
        asf.addRow("Medium Scale (α₂):", self._as_med_alpha)
        self._as_fine_alpha = QDoubleSpinBox()
        self._as_fine_alpha.setRange(0.5, 20.0)
        self._as_fine_alpha.setDecimals(1)
        self._as_fine_alpha.setValue(2.0)
        self._as_fine_alpha.setSuffix(" m")
        asf.addRow("Fine Scale (α₃):", self._as_fine_alpha)
        self._as_dist = QDoubleSpinBox()
        self._as_dist.setRange(0.02, 5.0)
        self._as_dist.setDecimals(2)
        self._as_dist.setValue(0.20)
        self._as_dist.setSuffix(" m")
        self._as_dist.setToolTip("Max perpendicular distance to bottom alpha-shape TIN. Lower (e.g. 0.15-0.20m) strips water surface reflections and low vegetation.")
        asf.addRow("Max Distance:", self._as_dist)
        self._as_angle = QDoubleSpinBox()
        self._as_angle.setRange(1.0, 45.0)
        self._as_angle.setDecimals(1)
        self._as_angle.setValue(8.0)
        self._as_angle.setSuffix("°")
        asf.addRow("Max Angle:", self._as_angle)
        self._as_terrain_angle = QDoubleSpinBox()
        self._as_terrain_angle.setRange(10.0, 90.0)
        self._as_terrain_angle.setDecimals(1)
        self._as_terrain_angle.setValue(45.0)
        self._as_terrain_angle.setSuffix("°")
        self._as_terrain_angle.setToolTip("Max allowed slope angle of ground facets. 45° prevents steep bridging triangles across riverbanks and water.")
        asf.addRow("Max Terrain Angle:", self._as_terrain_angle)
        self._alpha_shape_group.setVisible(False)
        layout.addWidget(self._alpha_shape_group)

        # EGS-CSF params
        self._egs_csf_group = QGroupBox("EGS-CSF Parameters (IJDE 2025)")
        csff = QFormLayout(self._egs_csf_group)
        self._egs_cloth_res = QDoubleSpinBox()
        self._egs_cloth_res.setRange(0.2, 10.0)
        self._egs_cloth_res.setDecimals(2)
        self._egs_cloth_res.setValue(1.0)
        self._egs_cloth_res.setSuffix(" m")
        self._egs_cloth_res.setToolTip("Cloth particle grid resolution")
        csff.addRow("Cloth Resolution:", self._egs_cloth_res)
        self._egs_rigidness = QDoubleSpinBox()
        self._egs_rigidness.setRange(0.5, 10.0)
        self._egs_rigidness.setDecimals(1)
        self._egs_rigidness.setValue(2.0)
        self._egs_rigidness.setToolTip("Base cloth rigidness (stiff in flat areas, automatically relaxed on steep slopes)")
        csff.addRow("Base Rigidness:", self._egs_rigidness)
        self._egs_class_thresh = QDoubleSpinBox()
        self._egs_class_thresh.setRange(0.05, 5.0)
        self._egs_class_thresh.setDecimals(2)
        self._egs_class_thresh.setValue(0.30)
        self._egs_class_thresh.setSuffix(" m")
        self._egs_class_thresh.setToolTip("Base distance threshold to cloth surface")
        csff.addRow("Classification Threshold:", self._egs_class_thresh)
        self._egs_grad_factor = QDoubleSpinBox()
        self._egs_grad_factor.setRange(0.0, 5.0)
        self._egs_grad_factor.setDecimals(2)
        self._egs_grad_factor.setValue(0.50)
        self._egs_grad_factor.setToolTip("Gradient adaptation factor (widens threshold on steep riverbanks)")
        csff.addRow("Gradient Factor (β):", self._egs_grad_factor)
        self._egs_max_iter = QSpinBox()
        self._egs_max_iter.setRange(10, 500)
        self._egs_max_iter.setValue(150)
        self._egs_max_iter.setToolTip("Maximum simulation iterations (100–150 recommended for large-relief terrain or riverbanks)")
        csff.addRow("Max Iterations:", self._egs_max_iter)
        self._egs_spike_down = QDoubleSpinBox()
        self._egs_spike_down.setRange(0.0, 50.0)
        self._egs_spike_down.setDecimals(1)
        self._egs_spike_down.setValue(2.5)
        self._egs_spike_down.setSuffix(" m")
        self._egs_spike_down.setToolTip("Down-spike / subterranean pit filter threshold. Suppresses low multipath noise so the cloth does not hang on underground spikes (0 to disable).")
        csff.addRow("Down-Spike Threshold:", self._egs_spike_down)
        self._egs_slope_smooth = QCheckBox("Slope & Embankment Crest Smoothing")
        self._egs_slope_smooth.setChecked(True)
        self._egs_slope_smooth.setToolTip("Snaps suspended cloth into inverted V-troughs over continuous slopes so rounded embankment crests, levees, and mounds are not cut off.")
        csff.addRow(self._egs_slope_smooth)
        self._egs_exclude_single_returns_water = QCheckBox("Exclude single returns in water (bathy)")
        self._egs_exclude_single_returns_water.setChecked(False)
        self._egs_exclude_single_returns_water.setToolTip("Drops single returns in water so shallow water is not classified as ground.")
        csff.addRow(self._egs_exclude_single_returns_water)
        self._egs_csf_group.setVisible(False)
        layout.addWidget(self._egs_csf_group)

        # Step-Down PTD params
        self._stepdown_group = QGroupBox("Step-Down PTD Parameters")
        lgf = QFormLayout(self._stepdown_group)

        self._stepdown_preset_combo = QComboBox()
        self._stepdown_preset_combo.addItem("Embankments & River Corridors (step 5m, sub 6, bulge 1.0m)", "river")
        self._stepdown_preset_combo.addItem("Nature & Undulating (step 5m, sub 5, bulge 1.0m)", "nature")
        self._stepdown_preset_combo.addItem("Steep Hills & Mountains (step 1.5m, sub 7, bulge 1.5m)", "steep")
        self._stepdown_preset_combo.addItem("Town & Low Density (step 10m, sub 4, bulge 1.5m)", "town")
        self._stepdown_preset_combo.addItem("City & Large Warehouses (step 25m, sub 4, bulge 1.5m)", "city")
        self._stepdown_preset_combo.addItem("Custom", "custom")
        self._stepdown_preset_combo.currentIndexChanged.connect(self._on_stepdown_preset_changed)
        lgf.addRow("Terrain Preset:", self._stepdown_preset_combo)

        self._stepdown_step = QDoubleSpinBox()
        self._stepdown_step.setRange(0.5, 100.0)
        self._stepdown_step.setDecimals(1)
        self._stepdown_step.setValue(5.0)
        self._stepdown_step.setSuffix(" m")
        self._stepdown_step.setToolTip("Initial coarse grid resolution. 5m for riverbanks, 5m for nature, 25m for city.")
        lgf.addRow("Initial Step Size:", self._stepdown_step)

        self._stepdown_sub = QSpinBox()
        self._stepdown_sub.setRange(1, 10)
        self._stepdown_sub.setValue(6)
        self._stepdown_sub.setToolTip("Number of multi-scale step-down substeps. Higher = finer detail on steep slopes.")
        lgf.addRow("Substeps:", self._stepdown_sub)

        bulge_box = QHBoxLayout()
        self._stepdown_bulge = QDoubleSpinBox()
        self._stepdown_bulge.setRange(0.0, 10.0)
        self._stepdown_bulge.setDecimals(2)
        self._stepdown_bulge.setValue(1.00)
        self._stepdown_bulge.setSuffix(" m")
        self._stepdown_bulge.setToolTip("Max allowable elevation rise when curving the TIN across ridges/slopes.")
        bulge_box.addWidget(self._stepdown_bulge)

        self._stepdown_bulge_auto = QCheckBox("Auto (step/10)")
        self._stepdown_bulge_auto.setToolTip("When checked, bulge is automatically calculated as step / 10 clamped into [1.0, 2.0]m.")
        self._stepdown_bulge_auto.toggled.connect(lambda checked: self._stepdown_bulge.setEnabled(not checked))
        bulge_box.addWidget(self._stepdown_bulge_auto)
        lgf.addRow("Bulge Allowance:", bulge_box)

        self._stepdown_offset = QDoubleSpinBox()
        self._stepdown_offset.setRange(0.01, 2.0)
        self._stepdown_offset.setDecimals(2)
        self._stepdown_offset.setValue(0.08)
        self._stepdown_offset.setSuffix(" m")
        self._stepdown_offset.setToolTip("Base elevation inclusion tolerance above bulged ground surface. Scales dynamically with slope.")
        lgf.addRow("Offset Tolerance:", self._stepdown_offset)

        self._stepdown_spike = QDoubleSpinBox()
        self._stepdown_spike.setRange(0.0, 10.0)
        self._stepdown_spike.setDecimals(2)
        self._stepdown_spike.setValue(0.50)
        self._stepdown_spike.setSuffix(" m")
        self._stepdown_spike.setToolTip("Up-spike removal threshold. Filters building roofs and tree tops from coarse seeds.")
        lgf.addRow("Up-Spike Threshold:", self._stepdown_spike)

        self._stepdown_spike_down = QDoubleSpinBox()
        self._stepdown_spike_down.setRange(0.0, 10.0)
        self._stepdown_spike_down.setDecimals(2)
        self._stepdown_spike_down.setValue(0.50)
        self._stepdown_spike_down.setSuffix(" m")
        self._stepdown_spike_down.setToolTip("Down-spike removal threshold. Filters low pits and multipath noise.")
        lgf.addRow("Down-Spike Threshold:", self._stepdown_spike_down)

        self._stepdown_refine = QSpinBox()
        self._stepdown_refine.setRange(0, 20)
        self._stepdown_refine.setValue(2)
        self._stepdown_refine.setToolTip("Progressive refinement passes. Automatically early-terminates when surface converges.")
        lgf.addRow("Refinement Loops:", self._stepdown_refine)

        self._stepdown_all_returns = QCheckBox("Consider all returns")
        self._stepdown_all_returns.setChecked(False)
        self._stepdown_all_returns.setToolTip("Default behavior (unchecked) considers only last returns for bare-earth seeding, preventing water surface and tree crown infiltration.")
        lgf.addRow("", self._stepdown_all_returns)

        self._stepdown_group.setVisible(True)
        layout.addWidget(self._stepdown_group)

        # Info / Algorithm Guide
        info = QLabel(
            "<b>Algorithm Guide — When & Where to Use Each Method:</b><br>"
            "• <b>Step-Down PTD:</b> <i>Recommended for river corridors, embankments, levees, dikes, floodplains, and bathymetry.</i> Multi-scale bulged TIN snaps directly to 3D breaklines and bridges water channels without coordinate inversion.<br>"
            "• <b>M-AlphaShape:</b> <i>Best for complex natural terrain, steep rocky cliffs, and ridges.</i> Rolling 3D alpha probes naturally hug convex peaks and steep slopes without raster aliasing.<br>"
            "• <b>EP-PTD (Axelsson):</b> <i>Classic workhorse for standard airborne topographic surveys and rolling countryside.</i> Iterative TIN densification.<br>"
            "• <b>EGS-CSF:</b> <i>Best for flat urban areas and large building complexes.</i> Inverted cloth simulation. <i>Avoid on narrow river embankments or dikes</i> due to cloth tension across inverted crest troughs.<br>"
            "• <b>SMRF (Pingel et al.):</b> <i>Best for flat open plains and gentle agricultural fields where processing speed is the priority.</i><br>"
            "• <b>APTD / H-PTD:</b> <i>Specialized for extreme mountain-valley relief (APTD) or very large point clouds where standard PTD is too slow (H-PTD).</i><br>"
            "• <b>DL-Hybrid PTD:</b> <i>Best when Pointcept AI ground predictions (Class 2) already exist and require geometric TIN refinement.</i>"
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

        self._on_method_changed()

    @staticmethod
    def _make_auto_spin(maximum: float, decimals: int, auto_text: str = "Auto") -> QDoubleSpinBox:
        """A spin box where value 0 means 'Auto' (returns None in params)."""
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(decimals)
        spin.setValue(0.0)
        spin.setSpecialValueText(auto_text)
        return spin

    def _on_stepdown_preset_changed(self):
        preset = self._stepdown_preset_combo.currentData()
        if preset == "river":
            self._stepdown_step.setValue(5.0)
            self._stepdown_sub.setValue(6)
            self._stepdown_bulge_auto.setChecked(False)
            self._stepdown_bulge.setValue(1.00)
            self._stepdown_offset.setValue(0.08)
            self._stepdown_spike.setValue(0.50)
            self._stepdown_spike_down.setValue(0.50)
        elif preset == "nature":
            self._stepdown_step.setValue(5.0)
            self._stepdown_sub.setValue(5)
            self._stepdown_bulge_auto.setChecked(False)
            self._stepdown_bulge.setValue(1.00)
            self._stepdown_offset.setValue(0.10)
            self._stepdown_spike.setValue(1.0)
            self._stepdown_spike_down.setValue(1.0)
        elif preset == "steep":
            self._stepdown_step.setValue(1.5)
            self._stepdown_sub.setValue(7)
            self._stepdown_bulge_auto.setChecked(False)
            self._stepdown_bulge.setValue(1.50)
            self._stepdown_offset.setValue(0.12)
            self._stepdown_spike.setValue(1.0)
            self._stepdown_spike_down.setValue(1.0)
        elif preset == "town":
            self._stepdown_step.setValue(10.0)
            self._stepdown_sub.setValue(4)
            self._stepdown_bulge_auto.setChecked(False)
            self._stepdown_bulge.setValue(1.50)
            self._stepdown_offset.setValue(0.10)
            self._stepdown_spike.setValue(1.0)
            self._stepdown_spike_down.setValue(1.0)
        elif preset == "city":
            self._stepdown_step.setValue(25.0)
            self._stepdown_sub.setValue(4)
            self._stepdown_bulge_auto.setChecked(False)
            self._stepdown_bulge.setValue(1.50)
            self._stepdown_offset.setValue(0.10)
            self._stepdown_spike.setValue(1.5)
            self._stepdown_spike_down.setValue(1.5)

    def _on_tin_preset_changed(self):
        preset = self._tin_preset_combo.currentData()
        if preset == "flat":
            self._tin_dist.setValue(0.50)
            self._tin_angle.setValue(6.0)
            self._tin_seed_res.setValue(10.0)
            self._tin_spacing.setValue(0.50)
            self._tin_dense_tol.setValue(0.10)
        elif preset == "embankment":
            self._tin_dist.setValue(1.50)
            self._tin_angle.setValue(25.0)
            self._tin_seed_res.setValue(5.0)
            self._tin_spacing.setValue(0.50)
            self._tin_dense_tol.setValue(0.20)
        elif preset == "steep":
            self._tin_dist.setValue(2.00)
            self._tin_angle.setValue(30.0)
            self._tin_seed_res.setValue(3.0)
            self._tin_spacing.setValue(0.50)
            self._tin_dense_tol.setValue(0.30)
        elif preset == "urban":
            self._tin_dist.setValue(0.50)
            self._tin_angle.setValue(6.0)
            self._tin_seed_res.setValue(25.0)
            self._tin_spacing.setValue(0.50)
            self._tin_dense_tol.setValue(0.10)

    def _on_method_changed(self):
        method = self._method_combo.currentData()
        self._stepdown_group.setVisible(method == "stepdown")
        self._smrf_group.setVisible(method == "smrf")
        self._tin_group.setVisible(method == "epptd")
        self._aptd_group.setVisible(method == "aptd")
        self._hptd_group.setVisible(method == "hptd")
        self._dlhybrid_group.setVisible(method == "dl_hybrid")
        self._alpha_shape_group.setVisible(method == "alpha_shape")
        self._egs_csf_group.setVisible(method == "egs_csf")

        guides = {
            "stepdown": (
                "<b>★ Recommended For:</b> River corridors, water embankments, levees, dikes, floodplains, and bathymetry.<br>"
                "<b>How It Works:</b> Multi-scale step-down progressive TIN densification with normal bulge. Steps from coarse landscape to fine scale in true 3D.<br>"
                "<b>Key Advantage:</b> Triangle vertices snap directly onto 3D breaklines and levee crests; zero raster aliasing; bridges water channels cleanly."
            ),
            "alpha_shape": (
                "<b>★ Recommended For:</b> Natural complex terrain, steep rocky cliffs, rugged slopes, and sharp ridges.<br>"
                "<b>How It Works:</b> Multiscale spherical 3D alpha probes (fine 2m, medium 6m, coarse 20m) rolling across the terrain from above.<br>"
                "<b>Key Advantage:</b> True 3D geometry without planar leveling or inverted tension traps; preserves convex peaks and mounds."
            ),
            "epptd": (
                "<b>★ Recommended For:</b> Standard airborne topographic LiDAR, countryside, and rolling terrain.<br>"
                "<b>How It Works:</b> Classic Axelsson Progressive TIN Densification. Seeds lowest points in coarse cells and densifies iteratively.<br>"
                "<b>Note:</b> Widen distance/angle tolerances if working on steep slopes or sharp ridges."
            ),
            "egs_csf": (
                "<b>★ Recommended For:</b> Flat or gently rolling urban areas, industrial sites, and large building complexes.<br>"
                "<b>How It Works:</b> Inverts point cloud (Z_inv = -Z) and drops a simulated elastic cloth with evolutionary gradient relaxation.<br>"
                "<b>Caution:</b> <i>Unsuited for river embankments and dikes.</i> In inverted space, convex ridges become narrow trenches where cloth tension suspends the cloth below the real crest."
            ),
            "smrf": (
                "<b>★ Recommended For:</b> Flat agricultural land, open floodplains, and large gentle terrain where speed is essential.<br>"
                "<b>How It Works:</b> Simple Morphological Filter (Pingel et al. 2013). Fast raster morphological opening (erode + dilate) with progressive window growth.<br>"
                "<b>Caution:</b> Tends to shave off steep riverbanks, narrow embankments, and sharp breaklines."
            ),
            "aptd": (
                "<b>★ Recommended For:</b> Variable-density point clouds and mixed terrain containing both steep mountains and flat valleys.<br>"
                "<b>How It Works:</b> Two-level adaptive grid with localized terrain slope adaptation and automatic outlier suppression."
            ),
            "hptd": (
                "<b>★ Recommended For:</b> Very large point clouds where standard PTD is too slow, but TIN accuracy is required.<br>"
                "<b>How It Works:</b> Hierarchical multi-resolution seed pyramid with signed-angle criteria to accelerate processing."
            ),
            "dl_hybrid": (
                "<b>★ Recommended For:</b> Scenes where Pointcept deep learning has already predicted ground (Class 2).<br>"
                "<b>How It Works:</b> Seeds the initial TIN directly from AI ground predictions and geometrically densifies unclassified points."
            ),
        }
        if hasattr(self, "_algo_guide_label"):
            self._algo_guide_label.setText(guides.get(method, ""))

        # DL-Hybrid needs both class 1 and class 2 in the source set:
        # class 2 is the DL ground prior, class 1 are the candidates that
        # are densified against it.  Auto-switch the default class-2-only
        # selection so the hybrid does not silently degenerate to EP-PTD.
        if method == "dl_hybrid" and self._source_class_combo.currentData() == 2:
            idx = self._source_class_combo.findData(-2)
            if idx >= 0:
                self._source_class_combo.setCurrentIndex(idx)

    def _collect_params(self, method: str) -> dict:
        """Build the algorithm parameter dict from the current widgets."""
        if method == "stepdown":
            return {
                "step": self._stepdown_step.value(),
                "sub_steps": self._stepdown_sub.value(),
                "bulge": None if self._stepdown_bulge_auto.isChecked() else self._stepdown_bulge.value(),
                "offset": self._stepdown_offset.value(),
                "spike": self._stepdown_spike.value(),
                "spike_down": self._stepdown_spike_down.value(),
                "refine_loops": self._stepdown_refine.value(),
                "all_returns": self._stepdown_all_returns.isChecked(),
            }
        if method == "smrf":
            return {
                "slope_threshold": self._smrf_slope.value(),
                "max_window": self._smrf_maxw.value(),
                "elevation_threshold": self._smrf_elev.value(),
                "base_window": 2.0,
                "window_growth": 1.5,
                "cell_size": None,
            }
        if method == "aptd":
            return {
                "k_neighbors": self._aptd_k.value(),
                "radius_factor": self._aptd_radius_factor.value(),
                "min_neighbors": self._aptd_min_neighbors.value(),
                "primary_grid_size": None,
                "point_count_threshold": self._aptd_point_count.value(),
                "slope_threshold": self._aptd_slope_threshold.value(),
                "max_angle": self._aptd_angle.value() if self._aptd_angle.value() > 0 else None,
                "max_distance": self._aptd_distance.value() if self._aptd_distance.value() > 0 else None,
                "max_terrain_angle": self._aptd_terrain_angle.value() if self._aptd_terrain_angle.value() > 0 else None,
                "exclude_single_returns_in_water": self._aptd_exclude_single_returns_water.isChecked(),
            }
        if method == "hptd":
            return {
                "window_size": self._hptd_window.value() if self._hptd_window.value() > 0 else None,
                "step_factor": self._hptd_step_factor.value(),
                "max_angle": self._hptd_angle.value(),
                "max_distance": self._hptd_distance.value() if self._hptd_distance.value() > 0 else None,
                "relative_elevation_factor": self._hptd_rel_elev.value(),
                "signed": self._hptd_signed.isChecked(),
                "below_tolerance": self._hptd_below.value(),
                "max_terrain_angle": self._hptd_terrain_angle.value(),
                "follow_surface_trend": self._hptd_follow_trend.isChecked(),
                "exclude_single_returns_in_water": self._hptd_exclude_single_returns_water.isChecked(),
            }
        if method == "dl_hybrid":
            return {
                "dl_confidence_threshold": self._dl_conf_threshold.value(),
                "dl_seed_weight": self._dl_seed_weight.value(),
                "max_distance": self._dl_distance.value(),
                "max_angle": self._dl_angle.value(),
                "max_terrain_angle": self._dl_terrain_angle.value(),
                "only_upward": self._dl_only_upward.isChecked(),
                "follow_surface_trend": self._dl_follow_trend.isChecked(),
                "exclude_single_returns_in_water": self._dl_exclude_single_returns_water.isChecked(),
            }
        if method == "alpha_shape":
            return {
                "coarse_alpha": self._as_coarse_alpha.value(),
                "medium_alpha": self._as_med_alpha.value(),
                "fine_alpha": self._as_fine_alpha.value(),
                "max_distance": self._as_dist.value(),
                "max_terrain_angle": self._as_terrain_angle.value(),
            }
        if method == "egs_csf":
            return {
                "cloth_resolution": self._egs_cloth_res.value(),
                "rigidness": self._egs_rigidness.value(),
                "class_threshold": self._egs_class_thresh.value(),
                "gradient_factor": self._egs_grad_factor.value(),
                "max_iterations": self._egs_max_iter.value(),
                "spike_down": self._egs_spike_down.value(),
                "slope_smooth": self._egs_slope_smooth.isChecked(),
                "exclude_single_returns_in_water": self._egs_exclude_single_returns_water.isChecked(),
            }
        # epptd (and legacy "tin")
        return {
            "max_distance": self._tin_dist.value(),
            "max_angle": self._tin_angle.value(),
            "seed_resolution": self._tin_seed_res.value(),
            "spacing": self._tin_spacing.value() if self._tin_spacing.value() > 0 else 0.50,
            "buffer_size": self._tin_buffer_size.value(),
            "max_iter": self._tin_max_iter.value(),
            "dense": self._tin_full_density.isChecked(),
            "dense_tolerance": self._tin_dense_tol.value(),
        }

    def _on_run(self):
        method = self._method_combo.currentData()
        source_class = self._source_class_combo.currentData()
        params = self._collect_params(method)

        from ..gui.settings_dialog import load_general_settings
        settings = load_general_settings()
        workers = settings.get("ground_workers", 4)

        self._run_btn.setEnabled(False)
        self._status.setText(
            f"Classifying ground on {len(self._jobs)} tile(s) "
            f"with {workers} worker process(es)…"
        )
        self._progress.setValue(0)

        self._worker = _GroundBatchWorker(
            self._jobs,
            method=method,
            params=params,
            source_class=source_class,
            workers=workers,
            db=self._db,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.tile_done.connect(
            self._on_tile_done,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker.finished_all.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str, pct: float):
        self._status.setText(msg)
        self._progress.setValue(int(pct))

    def _on_tile_done(self, tile_id: str, result: dict):
        n_ground = result.get("n_ground", 0)
        n_affected = result.get("n_affected", 0)
        self._status.setText(
            f"{tile_id}: {n_ground:,} ground / {n_affected:,} affected "
            f"({n_ground/max(n_affected,1)*100:.1f}%)"
        )
        self.tile_processed.emit(tile_id, result)

    def _on_finished(self, n_processed: int):
        self._status.setText(f"Done: {n_processed}/{len(self._jobs)} tiles")
        self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self._ok_btn.setEnabled(True)
        self.ground_applied.emit()

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg}")
        self._run_btn.setEnabled(True)

    def accept(self):
        """OK: wait for worker thread before closing."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        super().accept()

    def _on_accept(self):
        self.accept()
