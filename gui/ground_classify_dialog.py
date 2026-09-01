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
    ground_classify_epptd,
    ground_classify_epptd_two_pass,
    ground_classify_aptd,
    ground_classify_hptd,
    ground_classify_dl_hybrid_ptd,
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
        # With source class 2 only, every source point is already a DL
        # ground prediction and the DL prior carries no information —
        # the result degenerates to EP-PTD.  Broaden the source set to
        # classes 1 & 2 so class-2 points act as trusted DL seeds while
        # class-1 points are densified geometrically against them.
        source_mask = (cls_all == 1) | (cls_all == 2)
        sc = -2
    elif sc == -2:
        source_mask = (cls_all == 1) | (cls_all == 2)
    elif sc == -3:
        # "1 → 2": densify unclassified (1) onto existing ground (2).
        # Class 2 is the trusted initial TIN and is left untouched.
        source_mask = (cls_all == 1) | (cls_all == 2)
    elif sc >= 0:
        source_mask = (cls_all == sc)
    else:
        source_mask = np.ones(n_total, dtype=bool)

    n_source = int(source_mask.sum())
    result["n_source"] = n_source
    if n_source == 0:
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
    else:  # "epptd" (and legacy "tin")
        existing_ground = None
        if sc == -3:
            # Densify unclassified (1) onto existing ground (2): class 2
            # points are the trusted initial TIN and must not be re-tested.
            existing_ground = (cls_all[source_mask] == 2)

        extra_dim_filters = params.get("extra_dim_filters")
        if params.get("two_pass") and existing_ground is None:
            mask = ground_classify_epptd_two_pass(
                xs_sub, ys_sub, zs_sub,
                pass1_cell_size=params.get("pass1_cell_size", 60.0),
                pass1_max_distance=params.get("pass1_max_distance", 1.0),
                pass1_max_angle=params.get("pass1_max_angle", 12.0),
                pass2_max_distance=params.get("pass2_max_distance", 1.0),
                pass2_max_angle=params.get("pass2_max_angle", 6.0),
                max_terrain_angle=params.get("max_terrain_angle", 88.0),
                reduce_iter_angle_when_edge=params.get("pass2_reduce_edge"),
                stop_tri_when_edge=params.get("pass2_stop_edge"),
                only_upward=False,
                follow_surface_trend=params.get("pass2_follow_trend", False),
                remove_low_outliers_after_pass1=params.get(
                    "pass2_remove_low_outliers", True
                ),
                low_outlier_threshold=params.get("low_outlier_threshold", 1.0),
                exclude_single_returns_in_water=params.get(
                    "pass2_exclude_single", True
                ),
                sensor_type=st_sub,
                return_numbers=rn_sub,
                num_returns=nr_sub,
                extra_dims=exd_sub,
                extra_dim_filters=extra_dim_filters,
            )
        else:
            mask = ground_classify_epptd(
                xs_sub, ys_sub, zs_sub,
                max_distance=params["max_distance"],
                max_angle=params["max_angle"],
                max_terrain_angle=params.get("max_terrain_angle", 88.0),
                reduce_iter_angle_when_edge=params.get("reduce_iter_angle_when_edge"),
                stop_tri_when_edge=params.get("stop_tri_when_edge"),
                only_upward=params.get("only_upward", False),
                follow_surface_trend=params.get("follow_surface_trend", True),
                remove_low_outliers=params.get("remove_low_outliers", False),
                low_outlier_neighbors=params.get("low_outlier_neighbors", 8),
                low_outlier_threshold=params.get("low_outlier_threshold", 1.0),
                exclude_single_returns_in_water=params.get("exclude_single_returns_in_water", False),
                sensor_type=st_sub,
                cell_size=params.get("cell_size"),
                existing_ground_mask=existing_ground,
                return_numbers=rn_sub,
                num_returns=nr_sub,
                extra_dims=exd_sub,
                extra_dim_filters=extra_dim_filters,
            )

    # Expand the source-only mask back to the full tile.
    full_mask = np.zeros(n_total, dtype=bool)
    non_source_ground = ~source_mask & (cls_all == 2)
    full_mask[non_source_ground] = True
    full_mask[source_mask] = mask

    new_cls = cls_all.copy()
    if sc == -2:
        in_source = (cls_all == 1) | (cls_all == 2)
        new_cls[in_source & full_mask] = 2
        new_cls[in_source & ~full_mask] = 1
        n_affected = int(in_source.sum())
    elif sc == -3:
        in_source = (cls_all == 1) | (cls_all == 2)
        was_ground = cls_all == 2
        # Existing ground stays untouched; only class-1 points are assigned.
        new_cls[in_source & was_ground] = 2
        new_cls[in_source & ~was_ground & full_mask] = 2
        new_cls[in_source & ~was_ground & ~full_mask] = 1
        n_affected = int((cls_all == 1).sum())
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
    _write_las_file(
        las_path,
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
                futures = {}
                for job in self._jobs:
                    fut = pool.submit(
                        _ground_process_tile,
                        job["tile_id"], job["las_path"],
                        self._method, self._params, self._source_class,
                        job.get("flightline_sensor_types", {}),
                    )
                    futures[fut] = job["tile_id"]

                for fut in as_completed(futures):
                    tid = futures[fut]
                    try:
                        result = fut.result()
                        processed += 1
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
                        self.error.emit(f"{tid}: {exc}")

            elapsed = time.time() - t_start
            self.progress.emit(
                f"Done: {processed}/{n_total} tiles in {elapsed:.1f}s", 100.0
            )
            self.finished_all.emit(processed)
        except Exception as exc:
            logger.exception("Ground classification failed")
            self.error.emit(str(exc))

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
        """Cancel: wait for worker thread before closing."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(3000)  # 3 s timeout
        super().reject()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Method selector
        method_group = QGroupBox("Algorithm")
        mf = QFormLayout(method_group)
        self._method_combo = QComboBox()
        self._method_combo.addItem("SMRF — Simple Morphological Filter (PDAL)", "smrf")
        self._method_combo.addItem("EP-PTD — Edge-Preserving PTD (Axelsson)", "epptd")
        self._method_combo.addItem("APTD — Adaptive Grid PTD (AGPTD)", "aptd")
        self._method_combo.addItem("H-PTD — Hierarchical/Fast PTD (FPTD)", "hptd")
        self._method_combo.addItem("DL-Hybrid PTD — Pointcept + PTD", "dl_hybrid")
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        mf.addRow("Method:", self._method_combo)

        # Source class filter — only reclassify points matching this class
        self._source_class_combo = QComboBox()
        self._source_class_combo.addItem("2: Ground (Pointcept default)", 2)
        self._source_class_combo.addItem("0: Created, Never Classified", 0)
        self._source_class_combo.addItem("1: Unclassified", 1)
        self._source_class_combo.addItem("1 & 2: Unclassified + Ground", -2)
        self._source_class_combo.addItem(
            "1 → 2: Densify Unclassified onto Existing Ground", -3
        )
        self._source_class_combo.addItem("All classes", -1)
        self._source_class_combo.setCurrentIndex(0)
        self._source_class_combo.setToolTip(
            "Only reclassify points matching the selected source class(es). "
            "Other classes are left unchanged.\n\n"
            "'1 → 2' uses existing Class 2 ground as the initial TIN and only "
            "reclassifies Class 1 points against it (TerraScan densify). "
            "Existing ground is left untouched.  Most useful with the "
            "EP-PTD method."
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
        layout.addWidget(self._smrf_group)

        # EP-PTD params
        self._tin_group = QGroupBox("EP-PTD Parameters")
        tf = QFormLayout(self._tin_group)
        self._tin_dist = QDoubleSpinBox()
        self._tin_dist.setRange(0.1, 10.0)
        self._tin_dist.setDecimals(2)
        self._tin_dist.setValue(1.4)
        self._tin_dist.setSuffix(" m")
        self._tin_dist.setToolTip("Max distance from point to TIN surface")
        tf.addRow("Max Distance:", self._tin_dist)
        self._tin_angle = QDoubleSpinBox()
        self._tin_angle.setRange(1.0, 45.0)
        self._tin_angle.setDecimals(1)
        self._tin_angle.setValue(6.0)
        self._tin_angle.setSuffix("°")
        self._tin_angle.setToolTip("Max angle between point and TIN vertices")
        tf.addRow("Max Angle:", self._tin_angle)

        # Seed grid / cell size (0 = auto from point spacing)
        self._tin_cell_size = self._make_auto_spin(200.0, 1, "Auto")
        self._tin_cell_size.setToolTip(
            "Seed grid cell size. Auto derives it from point spacing; "
            "a coarse value (e.g. 60 m) guarantees the seed is the true "
            "terrain/riverbed minimum in river valleys."
        )
        tf.addRow("Seed Grid (Auto):", self._tin_cell_size)

        # Terrain angle
        self._tin_terrain_angle = QDoubleSpinBox()
        self._tin_terrain_angle.setRange(10.0, 90.0)
        self._tin_terrain_angle.setDecimals(1)
        self._tin_terrain_angle.setValue(88.0)
        self._tin_terrain_angle.setSuffix("°")
        self._tin_terrain_angle.setToolTip(
            "Max allowed slope of TIN triangles. "
            "Keep 88-90° to classify steep natural embankments and riverbanks; "
            "lower it only to exclude man-made vertical walls."
        )
        tf.addRow("Max Terrain Angle:", self._tin_terrain_angle)

        # Two-pass TerraScan-style workflow (opt-in; single pass unchanged)
        self._tin_two_pass = QCheckBox(
            "Two-Pass TerraScan workflow (12° → 6°, densify)"
        )
        self._tin_two_pass.setToolTip(
            "Pass 1: coarse seed grid + 12° angle + 1.0 m distance to "
            "establish the base terrain/riverbed, then remove low outliers. "
            "Pass 2: densify against pass-1 ground with 6° angle + 1.0 m "
            "distance + edge damping.  Recommended for river/bathy corridors; "
            "leave off for standard topographic LiDAR."
        )
        tf.addRow(self._tin_two_pass)

        self._tin_two_pass_group = QGroupBox("Two-Pass Parameters")
        tpf = QFormLayout(self._tin_two_pass_group)
        self._tp_cell = QDoubleSpinBox()
        self._tp_cell.setRange(5.0, 200.0)
        self._tp_cell.setDecimals(1)
        self._tp_cell.setValue(60.0)
        self._tp_cell.setSuffix(" m")
        self._tp_cell.setToolTip(
            "Pass-1 seed grid cell size. 60 m puts the seed at the true "
            "riverbed minimum in river valleys narrower than the cell."
        )
        tpf.addRow("Pass 1 Seed Grid:", self._tp_cell)
        self._tp_angle1 = QDoubleSpinBox()
        self._tp_angle1.setRange(1.0, 45.0)
        self._tp_angle1.setDecimals(1)
        self._tp_angle1.setValue(12.0)
        self._tp_angle1.setSuffix("°")
        tpf.addRow("Pass 1 Max Angle:", self._tp_angle1)
        self._tp_angle2 = QDoubleSpinBox()
        self._tp_angle2.setRange(1.0, 45.0)
        self._tp_angle2.setDecimals(1)
        self._tp_angle2.setValue(6.0)
        self._tp_angle2.setSuffix("°")
        tpf.addRow("Pass 2 Max Angle:", self._tp_angle2)
        self._tp_dist = QDoubleSpinBox()
        self._tp_dist.setRange(0.1, 10.0)
        self._tp_dist.setDecimals(2)
        self._tp_dist.setValue(1.0)
        self._tp_dist.setSuffix(" m")
        self._tp_dist.setToolTip(
            "Iteration distance for both passes. 1.0 m keeps the water "
            "surface (~3 m above the bed) strictly outside the tolerance "
            "envelope."
        )
        tpf.addRow("Pass 1 & 2 Max Distance:", self._tp_dist)
        self._tp_reduce_edge = QDoubleSpinBox()
        self._tp_reduce_edge.setRange(0.0, 20.0)
        self._tp_reduce_edge.setDecimals(1)
        self._tp_reduce_edge.setValue(5.0)
        self._tp_reduce_edge.setSuffix(" m")
        self._tp_reduce_edge.setSpecialValueText("Off")
        self._tp_reduce_edge.setToolTip(
            "Pass-2 edge damping: reduce iteration angle in triangles with "
            "edges shorter than this, preventing dense water-surface "
            "reflections from expanding upward."
        )
        tpf.addRow("Reduce Iter. Angle < Edge:", self._tp_reduce_edge)
        self._tp_stop_edge = QDoubleSpinBox()
        self._tp_stop_edge.setRange(0.0, 10.0)
        self._tp_stop_edge.setDecimals(2)
        self._tp_stop_edge.setValue(2.0)
        self._tp_stop_edge.setSuffix(" m")
        self._tp_stop_edge.setSpecialValueText("Off")
        self._tp_stop_edge.setToolTip(
            "Pass-2 stop: stop processing in triangles with edges shorter "
            "than this."
        )
        tpf.addRow("Stop Triang. < Edge:", self._tp_stop_edge)
        self._tp_remove_low_outliers = QCheckBox(
            "Remove low outliers after pass 1"
        )
        self._tp_remove_low_outliers.setChecked(True)
        tpf.addRow(self._tp_remove_low_outliers)
        self._tp_follow_trend = QCheckBox(
            "Follow surface trend (adapt to local slope)"
        )
        self._tp_follow_trend.setChecked(False)
        self._tp_follow_trend.setToolTip(
            "Standard Axelsson PTD keeps this off so the TIN cannot ramp "
            "uphill onto the water surface. Enable only if riverbanks are "
            "under-classified."
        )
        tpf.addRow(self._tp_follow_trend)
        self._tp_exclude_single_water = QCheckBox(
            "Exclude single returns in water (bathy bed-only)"
        )
        self._tp_exclude_single_water.setChecked(True)
        tpf.addRow(self._tp_exclude_single_water)
        self._tin_two_pass_group.setVisible(False)
        self._tin_two_pass.toggled.connect(
            self._tin_two_pass_group.setVisible
        )
        tf.addRow(self._tin_two_pass_group)

        # Extra-byte (extra dimension) filter — opt-in, bathy-only
        self._tin_extra_group = QGroupBox(
            "Extra-Byte Filter (bathy points only)"
        )
        ef = QFormLayout(self._tin_extra_group)
        self._tin_extra_enabled = QCheckBox("Enable extra-byte filter")
        self._tin_extra_enabled.setToolTip(
            "Exclude bathymetric points (sensor_type == 2) whose selected "
            "extra dimension falls outside the given bounds.  Typical use: "
            "drop water-column turbidity scatter with high RIEGL deviation "
            "or low amplitude before the TIN is built.  Topo points are "
            "never affected."
        )
        ef.addRow(self._tin_extra_enabled)
        self._tin_extra_dim = QLineEdit()
        self._tin_extra_dim.setPlaceholderText(
            "extra dim name, e.g. deviation or amplitude"
        )
        ef.addRow("Dimension Name:", self._tin_extra_dim)
        self._tin_extra_min = self._make_auto_spin(1e9, 3, "Off")
        self._tin_extra_min.setToolTip(
            "Reject bathy points below this value (0 = no lower bound)."
        )
        ef.addRow("Reject Below:", self._tin_extra_min)
        self._tin_extra_max = self._make_auto_spin(1e9, 3, "Off")
        self._tin_extra_max.setToolTip(
            "Reject bathy points above this value (0 = no upper bound)."
        )
        ef.addRow("Reject Above:", self._tin_extra_max)
        self._tin_extra_dim.setEnabled(False)
        self._tin_extra_min.setEnabled(False)
        self._tin_extra_max.setEnabled(False)
        self._tin_extra_enabled.toggled.connect(self._tin_extra_dim.setEnabled)
        self._tin_extra_enabled.toggled.connect(self._tin_extra_min.setEnabled)
        self._tin_extra_enabled.toggled.connect(self._tin_extra_max.setEnabled)
        tf.addRow(self._tin_extra_group)

        # Edge-based iteration control (optional, off by default)
        self._tin_reduce_edge = QDoubleSpinBox()
        self._tin_reduce_edge.setRange(0.0, 20.0)
        self._tin_reduce_edge.setDecimals(1)
        self._tin_reduce_edge.setValue(0.0)
        self._tin_reduce_edge.setSuffix(" m")
        self._tin_reduce_edge.setSpecialValueText("Off")
        self._tin_reduce_edge.setToolTip(
            "When triangle edges are shorter than this, reduce "
            "iteration angle to avoid over-densification. "
            "0 = disabled."
        )
        tf.addRow("Reduce Iter. Angle < Edge:", self._tin_reduce_edge)

        self._tin_stop_edge = QDoubleSpinBox()
        self._tin_stop_edge.setRange(0.0, 10.0)
        self._tin_stop_edge.setDecimals(2)
        self._tin_stop_edge.setValue(0.0)
        self._tin_stop_edge.setSuffix(" m")
        self._tin_stop_edge.setSpecialValueText("Off")
        self._tin_stop_edge.setToolTip(
            "Stop processing in triangles with edges shorter than "
            "this. 0 = disabled."
        )
        tf.addRow("Stop Triang. < Edge:", self._tin_stop_edge)

        # Only upward
        self._tin_only_upward = QCheckBox("Only add points above initial surface")
        self._tin_only_upward.setToolTip(
            "If checked, only points ABOVE the initial seed surface "
            "are added. Prevents low-error noise from pulling the "
            "TIN downward."
        )
        tf.addRow(self._tin_only_upward)

        # Follow surface trend
        self._tin_follow_trend = QCheckBox("Follow surface trend (adapt to local slope)")
        self._tin_follow_trend.setChecked(True)
        self._tin_follow_trend.setToolTip(
            "If checked, the iteration angle is locally relaxed on "
            "steep natural slopes (riverbanks, cliffs). This helps "
            "the TIN climb slopes where the terrain angle changes "
            "rapidly. Disable for flat urban areas."
        )
        tf.addRow(self._tin_follow_trend)

        # Low-outlier removal (lidR-style)
        low_outlier_row = QHBoxLayout()
        self._tin_remove_low_outliers = QCheckBox("Remove low outliers")
        self._tin_remove_low_outliers.setToolTip(
            "Post-densification cleanup that drops ground points sitting far "
            "below their local neighbourhood (low outliers that become seeds "
            "and spike the DTM). Mirrors lidR's internal outlier removal."
        )
        low_outlier_row.addWidget(self._tin_remove_low_outliers)
        self._tin_low_outlier_threshold = QDoubleSpinBox()
        self._tin_low_outlier_threshold.setRange(0.1, 20.0)
        self._tin_low_outlier_threshold.setDecimals(2)
        self._tin_low_outlier_threshold.setValue(1.0)
        self._tin_low_outlier_threshold.setSuffix(" m")
        self._tin_low_outlier_threshold.setToolTip(
            "A point is a low outlier when it is more than this far below the "
            "median Z of its 8 nearest ground neighbours."
        )
        low_outlier_row.addWidget(self._tin_low_outlier_threshold)
        low_outlier_row.addStretch()
        tf.addRow(low_outlier_row)

        # Exclude single returns in water (bathy bed-only)
        self._tin_exclude_single_returns_water = QCheckBox(
            "Exclude single returns in water (bathy bed-only)"
        )
        self._tin_exclude_single_returns_water.setToolTip(
            "In bathymetry, single returns (num_returns == 1) are usually the "
            "water surface, not the riverbed.  When checked, they are excluded "
            "so the shallow-water surface isn't classified as ground alongside "
            "the bed.  Land points are unaffected."
        )
        tf.addRow(self._tin_exclude_single_returns_water)

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

        # Info
        info = QLabel(
            "<b>SMRF</b> (PDAL): fast, good for most terrain. "
            "Uses progressive morphological opening.\n\n"
            "<b>EP-PTD</b> (Axelsson + edge controls): iterative, "
            "preserves sharp terrain breaks (cliffs, riverbanks).\n\n"
            "<b>APTD</b> (AGPTD): adaptive two-level grid + outlier removal, "
            "good for low points and disconnected/steep terrain. "
            "<i>Slowest method on large tiles — prefer H-PTD or SMRF when "
            "processing speed matters.</i>\n\n"
            "<b>H-PTD</b> (FPTD): sliding-window seeds + signed/relative "
            "criteria — faster, robust on steep slopes.\n\n"
            "<b>DL-Hybrid PTD</b>: seeds the TIN from Pointcept's ground "
            "predictions (class 2) and refines geometrically — use after "
            "Pointcept classification with source class <b>1 &amp; 2</b> "
            "(selected automatically).\n\n"
            "<b>River data:</b> keep <i>Max Terrain Angle</i> at 88-90° to "
            "classify steep embankments and riverbanks. Enable "
            "<i>Exclude single returns in water</i> for bathy bed-only "
            "classification."
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

    @staticmethod
    def _make_auto_spin(maximum: float, decimals: int, auto_text: str = "Auto") -> QDoubleSpinBox:
        """A spin box where value 0 means 'Auto' (returns None in params)."""
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(decimals)
        spin.setValue(0.0)
        spin.setSpecialValueText(auto_text)
        return spin

    def _on_method_changed(self):
        method = self._method_combo.currentData()
        self._smrf_group.setVisible(method == "smrf")
        self._tin_group.setVisible(method == "epptd")
        self._aptd_group.setVisible(method == "aptd")
        self._hptd_group.setVisible(method == "hptd")
        self._dlhybrid_group.setVisible(method == "dl_hybrid")

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
        # epptd (and legacy "tin")
        extra_dim_name = self._tin_extra_dim.text().strip()
        extra_dim_filters = None
        if (self._tin_extra_enabled.isChecked() and extra_dim_name
                and (self._tin_extra_min.value() > 0
                     or self._tin_extra_max.value() > 0)):
            extra_dim_filters = [(
                extra_dim_name,
                self._tin_extra_min.value()
                if self._tin_extra_min.value() > 0 else None,
                self._tin_extra_max.value()
                if self._tin_extra_max.value() > 0 else None,
            )]
        return {
            "max_distance": self._tin_dist.value(),
            "max_angle": self._tin_angle.value(),
            "max_terrain_angle": self._tin_terrain_angle.value(),
            "cell_size": (
                self._tin_cell_size.value()
                if self._tin_cell_size.value() > 0
                else None
            ),
            "reduce_iter_angle_when_edge": (
                self._tin_reduce_edge.value()
                if self._tin_reduce_edge.value() > 0
                else None
            ),
            "stop_tri_when_edge": (
                self._tin_stop_edge.value()
                if self._tin_stop_edge.value() > 0
                else None
            ),
            "only_upward": self._tin_only_upward.isChecked(),
            "follow_surface_trend": self._tin_follow_trend.isChecked(),
            "remove_low_outliers": self._tin_remove_low_outliers.isChecked(),
            "low_outlier_neighbors": 8,
            "low_outlier_threshold": self._tin_low_outlier_threshold.value(),
            "exclude_single_returns_in_water": self._tin_exclude_single_returns_water.isChecked(),
            "two_pass": self._tin_two_pass.isChecked(),
            "pass1_cell_size": self._tp_cell.value(),
            "pass1_max_distance": self._tp_dist.value(),
            "pass1_max_angle": self._tp_angle1.value(),
            "pass2_max_distance": self._tp_dist.value(),
            "pass2_max_angle": self._tp_angle2.value(),
            "pass2_reduce_edge": (
                self._tp_reduce_edge.value()
                if self._tp_reduce_edge.value() > 0
                else None
            ),
            "pass2_stop_edge": (
                self._tp_stop_edge.value()
                if self._tp_stop_edge.value() > 0
                else None
            ),
            "pass2_follow_trend": self._tp_follow_trend.isChecked(),
            "pass2_remove_low_outliers": self._tp_remove_low_outliers.isChecked(),
            "pass2_exclude_single": self._tp_exclude_single_water.isChecked(),
            "extra_dim_filters": extra_dim_filters,
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
        # BlockingQueuedConnection gives back-pressure so tile results are
        # handled (status updates) before the next one is emitted.
        self._worker.tile_done.connect(
            self._on_tile_done,
            Qt.ConnectionType.BlockingQueuedConnection,
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
            self._worker.quit()
            self._worker.wait(3000)
        super().accept()

    def _on_accept(self):
        self.accept()
