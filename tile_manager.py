"""
LiDAR Workbench — Tile Manager.

Handles LAS/LAZ file import, spatial tiling, metadata tracking,
and lazy data loading for the view system.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    laspy = None  # type: ignore[assignment]

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .config import (
    DEFAULT_MIN_POINTS_PER_TILE,
    DEFAULT_TILE_OVERLAP_M,
    DEFAULT_TILE_SIZE_M,
    TARGET_POINTS_PER_TILE,
    TileStatus,
)
from .database import Database
from .project_manager import ProjectManager

logger = logging.getLogger("lidar_workbench.tile_manager")

# Type aliases
PointCloud = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (x, y, z)

# ── Structured dtype for per-file tile chunk flushing ──────────────
# All 21 point attributes in a single numpy structured array so we can
# raw-binary append to disk without per-attribute file explosion.
_TILE_CHUNK_DTYPE = np.dtype([
    ("x", "f8"), ("y", "f8"), ("z", "f8"),
    ("classification", "u1"), ("intensity", "u2"),
    ("return_number", "u1"), ("number_of_returns", "u1"),
    ("point_source_id", "u2"), ("gps_time", "f8"),
    ("scan_angle_rank", "i1"), ("scan_angle", "f8"),
    ("scan_direction_flag", "u1"), ("edge_of_flight_line", "u1"),
    ("user_data", "u1"),
    ("red", "u2"), ("green", "u2"), ("blue", "u2"),
    ("key_point", "u1"), ("synthetic", "u1"),
    ("withheld", "u1"), ("overlap", "u1"),
])

# Mapping from buffer keys to structured-array field names
_TILE_BUF_TO_CHUNK: Dict[str, str] = {
    "x": "x", "y": "y", "z": "z",
    "cl": "classification", "in": "intensity",
    "rn": "return_number", "nr": "number_of_returns",
    "src": "point_source_id", "gt": "gps_time",
    "sar": "scan_angle_rank", "sa": "scan_angle",
    "sdf": "scan_direction_flag", "efl": "edge_of_flight_line",
    "ud": "user_data",
    "red": "red", "grn": "green", "blu": "blue",
    "kp": "key_point", "syn": "synthetic",
    "wh": "withheld", "ov": "overlap",
}
BBox = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


class TileManager:
    """
    Manages tile import, creation, and data access.

    Coordinates between the project manager (directory layout) and the
    database (metadata).  Performs spatial tiling of flight strips into
    regularly-sized tiles.

    Usage::

        tm = TileManager(project_manager, database)
        tile_ids = tm.import_las_directory("/data/flight_strips/")
        points = tm.load_tile_points("tile_001")
    """

    def __init__(self, project_manager: ProjectManager, database: Database) -> None:
        """
        Args:
            project_manager: Initialised :class:`ProjectManager`.
            database:        Initialised :class:`Database`.
        """
        if not HAS_LASPY:
            logger.warning(
                "laspy is not installed. LAS/LAZ import will raise RuntimeError. "
                "Install with: pip install laspy"
            )
        self._pm = project_manager
        self._db = database
        self._point_cache: Dict[str, PointCloud] = {}  # simple in-memory cache

    # ── import ─────────────────────────────────────────────────────

    def import_las_directory(
        self,
        directory: str | Path,
        tile_size_m: Optional[float] = None,
        overlap_m: Optional[float] = None,
        target_points_per_tile: Optional[int] = None,
        min_points_per_tile: Optional[int] = None,
        sensor_type: str = '',
        scanner_override: str = '',
        crs_epsg: Optional[int] = None,
        crs_wkt: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """
        Import all LAS/LAZ files from a directory into the project.

        Streams each file once in chunks, bins points into the tile grid
        using vectorised numpy, and writes tiles incrementally.  Each
        input file is assigned a sequential flight-line number stored in
        ``point_source_id``.

        Args:
            directory:              Path to a directory containing ``.las`` / ``.laz`` files.
            tile_size_m:            Tile edge length in meters.  When ``None``, computed
                                    automatically from point density to hit the target
                                    points per tile.
            overlap_m:              Overlap between adjacent tiles in meters.
            target_points_per_tile: Target point count for auto-detect tile sizing
                                    (default ``TARGET_POINTS_PER_TILE``, ~1.5 M).
            min_points_per_tile:    Skip tiles with fewer than this many points
                                    (default ``DEFAULT_MIN_POINTS_PER_TILE``, 0).
            sensor_type:            ``'topo'`` for topography, ``'bathy'`` for bathymetry,
                                    or ``''`` to leave unset.
            scanner_override:       Manual scanner name override (ignores filename detection
                                    and LAS header).  Use when filenames don't encode the sensor.
            crs_epsg:               EPSG code for the CRS. Overrides auto-detection when set.
            crs_wkt:                OGC WKT string for the CRS. Used when *crs_epsg* is not set.
            progress_callback:      Optional ``callable(step: str, pct: float)`` for progress
                                    reporting.

        Returns:
            List of tile IDs that were imported.

        Raises:
            RuntimeError: If ``laspy`` is not installed.
            FileNotFoundError: If the directory does not exist.
        """
        if not HAS_LASPY:
            raise RuntimeError("laspy is required for LAS/LAZ import")

        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        las_files = sorted(
            list(directory.glob("*.las")) + list(directory.glob("*.laz"))
        )
        if not las_files:
            logger.warning("No .las/.laz files found in %s", directory)
            return []

        logger.info("Found %d LAS/LAZ file(s) in %s", len(las_files), directory)

        # ── Phase 1: read headers ──────────────────────────────────
        if progress_callback:
            progress_callback("Reading LAS headers…", 0.0)

        all_bboxes: List[BBox] = []
        total_points = 0
        header_infos: List[Dict[str, Any]] = []
        flight_line_counter = 0  # auto-increment when file has no flight line
        header_template: Optional[dict] = None  # captured from first file
        # Union extra dimensions across all files (different scanners may differ)
        _union_extra_dims: Dict[str, str] = {}  # name → type
        _template_format_id: Optional[int] = None

        for i, las_path in enumerate(las_files):
            try:
                # Snapshot header template from the first readable file,
                # then union extra dimensions from every file.
                try:
                    file_tmpl = _read_las_header_template(las_path)
                    fmt_id = file_tmpl["point_format_id"]
                    # Warn if point formats differ across files
                    if _template_format_id is not None and fmt_id != _template_format_id:
                        logger.warning(
                            "Point format mismatch: %s is format %d, previous files were format %d. "
                            "Output will use format %d, some attributes may be missing.",
                            las_path.name, fmt_id, _template_format_id, _template_format_id,
                        )
                    # Capture first file's template as base
                    if header_template is None:
                        header_template = file_tmpl
                        _template_format_id = fmt_id
                        logger.info(
                            "Header template from %s: format %d, %d VLR(s), %d extra dim(s)",
                            las_path.name, fmt_id,
                            len(header_template["vlrs"]),
                            len(header_template["extra_dimensions"]),
                        )
                    # Union extra dimensions from all files
                    for ed in file_tmpl["extra_dimensions"]:
                        _union_extra_dims[ed.name] = ed.type
                except Exception as exc:
                    if header_template is None:
                        logger.warning("Could not read header template from %s: %s", las_path, exc)

                with laspy.open(las_path) as reader:
                    hdr = reader.header
                    bbox: BBox = (hdr.x_min, hdr.y_min, hdr.x_max, hdr.y_max)
                    n_pts = hdr.point_count
                    all_bboxes.append(bbox)
                    total_points += n_pts

                # Extended metadata
                meta = _read_las_header_meta(las_path)

                # Flight-line number: use file_source_id if available, else auto-increment
                file_fl = meta["file_source_id"]
                if file_fl and file_fl > 0:
                    flight_line = int(file_fl)
                else:
                    flight_line_counter += 1
                    flight_line = flight_line_counter

                # Scanner detection: explicit override > filename > header system_id
                scanner = scanner_override.strip().lower() if scanner_override else ''
                if not scanner:
                    scanner = _detect_scanner_from_filename(las_path)
                if not scanner:
                    sid = meta["system_id"].strip()
                    if sid and sid.lower() not in ('', 'default', 'none'):
                        scanner = sid.lower()

                # Sensor type: explicit > auto-detect from scanner > ''
                file_sensor_type = sensor_type
                if not file_sensor_type and scanner:
                    file_sensor_type = _detect_sensor_type(scanner, las_path)

                header_infos.append({
                    "path": las_path,
                    "bbox": bbox,
                    "point_count": n_pts,
                    "version": meta["version"],
                    "flight_line": flight_line,
                    "scanner": scanner,
                    "sensor_type": file_sensor_type,
                })
            except Exception as exc:
                logger.error("Failed to read header of %s: %s", las_path, exc)
                continue
            if progress_callback:
                progress_callback("Reading headers…", (i + 1) / len(las_files) * 5.0)

        if not header_infos:
            logger.warning("No readable LAS/LAZ files found")
            return []

        # Fallback template if we couldn't read any header
        if header_template is None:
            header_template = {
                "version": "1.4",
                "point_format_id": 6,
                "extra_dimensions": [],
                "vlrs": [],
                "x_scale": 0.001,
                "y_scale": 0.001,
                "z_scale": 0.001,
            }
        else:
            # Replace file-specific list with the union across all files
            header_template["extra_dimensions"] = list(_union_extra_dims.items())  # [(name, type), ...]

        # ── Phase 2: compute grid ───────────────────────────────────
        global_bbox: BBox = (
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        )
        area_m2 = (global_bbox[2] - global_bbox[0]) * (global_bbox[3] - global_bbox[1])
        point_density = total_points / area_m2 if area_m2 > 0 else 0.0

        # Apply defaults for new optional params
        if target_points_per_tile is None:
            target_points_per_tile = TARGET_POINTS_PER_TILE
        if min_points_per_tile is None:
            min_points_per_tile = DEFAULT_MIN_POINTS_PER_TILE

        if tile_size_m is None:
            if point_density > 0:
                tile_area = target_points_per_tile / point_density
                tile_size_m = np.sqrt(tile_area)
                tile_size_m = max(50.0, round(tile_size_m / 50.0) * 50.0)
            else:
                tile_size_m = DEFAULT_TILE_SIZE_M
        overlap_m = overlap_m if overlap_m is not None else DEFAULT_TILE_OVERLAP_M

        tile_bboxes = _compute_tile_grid(global_bbox, tile_size_m, overlap_m)
        # Grid origin for tile-index math
        grid_x0, grid_y0 = global_bbox[0], global_bbox[1]
        stride = tile_size_m - overlap_m
        if stride <= 0:
            stride = tile_size_m
        grid_cols = int(np.ceil((global_bbox[2] - grid_x0) / stride))

        logger.info(
            "Global bbox: (%.2f, %.2f) – (%.2f, %.2f), %d pts, density %.2f pts/m², "
            "%d tile(s) @ %.0f m",
            *global_bbox, total_points, point_density, len(tile_bboxes), tile_size_m,
        )

        if progress_callback:
            progress_callback(f"Importing {len(las_files)} file(s) → {len(tile_bboxes)} tile(s)…", 5.0)

        # ── Phase 3: stream files, flush per file to disk ─────────
        tiles_dir = self._pm.tiles_dir
        assert tiles_dir is not None

        chunk_dir = tiles_dir / ".tmp_import"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        # Clean leftovers from a previous crashed import
        for stale in chunk_dir.glob("*.bin"):
            stale.unlink()

        # In-memory buffers for the CURRENT file only (flushed after each file)
        tile_buffers: Dict[int, Dict[str, list]] = {}
        # Cumulative metadata trackers (tiny — just counter dicts)
        tile_scanner_map: Dict[int, Dict[str, int]] = {}
        tile_sensor_type_map: Dict[int, Dict[str, int]] = {}
        tile_fl_sensor_types: Dict[int, Dict[int, str]] = {}
        tile_total_points: Dict[int, int] = {}  # running total across flushed chunks

        imported_ids: List[str] = []
        total_processed = 0

        for file_idx, info in enumerate(header_infos):
            flight_line = info["flight_line"]
            las_path = info["path"]
            file_total = info["point_count"]
            file_processed = 0

            # Fresh buffers for this file
            tile_buffers.clear()

            try:
                with laspy.open(las_path) as reader:
                    for chunk in reader.chunk_iterator(1_000_000):
                        n = len(chunk)
                        x = np.array(chunk.x, dtype=np.float64)
                        y = np.array(chunk.y, dtype=np.float64)

                        col = ((x - grid_x0) / stride).astype(np.int64)
                        row = ((y - grid_y0) / stride).astype(np.int64)
                        col = np.clip(col, 0, grid_cols - 1)
                        max_rows = int(np.ceil((global_bbox[3] - grid_y0) / stride))
                        row = np.clip(row, 0, max_rows - 1)
                        tile_idx_arr = row * grid_cols + col

                        z = np.array(chunk.z, dtype=np.float64)
                        cl = (_safe_attr(chunk, "classification", np.uint8, 0))
                        intens = (_safe_attr(chunk, "intensity", np.uint16, 0))
                        rn = (_safe_attr(chunk, "return_number", np.uint8, 1))
                        nr = (_safe_attr(chunk, "number_of_returns", np.uint8, 1))
                        src = np.full(n, flight_line, dtype=np.uint16)
                        gt  = (_safe_attr(chunk, "gps_time", np.float64, 0.0))
                        sar = (_safe_attr(chunk, "scan_angle_rank", np.int8, 0))
                        sa_deg = (_safe_attr(chunk, "scan_angle", np.float64, 0.0))
                        sdf = (_safe_attr(chunk, "scan_direction_flag", np.uint8, 0))
                        efl = (_safe_attr(chunk, "edge_of_flight_line", np.uint8, 0))
                        ud  = (_safe_attr(chunk, "user_data", np.uint8, 0))
                        red = (_safe_attr(chunk, "red", np.uint16, 0))
                        grn = (_safe_attr(chunk, "green", np.uint16, 0))
                        blu = (_safe_attr(chunk, "blue", np.uint16, 0))
                        kp  = (_safe_attr(chunk, "key_point", np.uint8, 0))
                        syn = (_safe_attr(chunk, "synthetic", np.uint8, 0))
                        wh  = (_safe_attr(chunk, "withheld", np.uint8, 0))
                        ov  = (_safe_attr(chunk, "overlap", np.uint8, 0))

                        # Only iterate tiles that actually received points
                        occupied = np.unique(tile_idx_arr)
                        for tidx in occupied:
                            tidx = int(tidx)
                            mask = tile_idx_arr == tidx
                            if tidx not in tile_buffers:
                                tile_buffers[tidx] = {
                                    "x":[], "y":[], "z":[],
                                    "cl":[], "in":[], "rn":[], "nr":[], "src":[],
                                    "gt":[], "sar":[], "sa":[], "sdf":[], "efl":[], "ud":[],
                                    "red":[], "grn":[], "blu":[],
                                    "kp":[], "syn":[], "wh":[], "ov":[],
                                }
                                if tidx not in tile_scanner_map:
                                    tile_scanner_map[tidx] = {}
                                if tidx not in tile_sensor_type_map:
                                    tile_sensor_type_map[tidx] = {}
                            buf = tile_buffers[tidx]
                            buf["x"].append(x[mask])
                            buf["y"].append(y[mask])
                            buf["z"].append(z[mask])
                            buf["cl"].append(cl[mask])
                            buf["in"].append(intens[mask])
                            buf["rn"].append(rn[mask])
                            buf["nr"].append(nr[mask])
                            buf["src"].append(src[mask])
                            buf["gt"].append(gt[mask])
                            buf["sar"].append(sar[mask])
                            buf["sa"].append(sa_deg[mask])
                            buf["sdf"].append(sdf[mask])
                            buf["efl"].append(efl[mask])
                            buf["ud"].append(ud[mask])
                            buf["red"].append(red[mask])
                            buf["grn"].append(grn[mask])
                            buf["blu"].append(blu[mask])
                            buf["kp"].append(kp[mask])
                            buf["syn"].append(syn[mask])
                            buf["wh"].append(wh[mask])
                            buf["ov"].append(ov[mask])
                            sc_name = info.get("scanner", '')
                            tile_scanner_map[tidx][sc_name] = (
                                tile_scanner_map[tidx].get(sc_name, 0) + mask.sum()
                            )
                            st = info.get("sensor_type", '')
                            tile_sensor_type_map[tidx][st] = (
                                tile_sensor_type_map[tidx].get(st, 0) + mask.sum()
                            )
                            if tidx not in tile_fl_sensor_types:
                                tile_fl_sensor_types[tidx] = {}
                            tile_fl_sensor_types[tidx][flight_line] = st if st else ''

                        file_processed += n
                        total_processed += n
                        if progress_callback:
                            pct = 5.0 + (total_processed / total_points) * 85.0
                            progress_callback(
                                f"File {file_idx+1}/{len(header_infos)} — "
                                f"{file_processed/file_total*100:.0f}%", pct
                            )

                # Flush this file's buffers to disk, then free RAM
                if tile_buffers:
                    flushed = _flush_tile_buffers(tile_buffers, chunk_dir, file_idx)
                    for tidx, n_pts in flushed.items():
                        tile_total_points[tidx] = tile_total_points.get(tidx, 0) + n_pts
                    tile_buffers.clear()

            except Exception as exc:
                logger.error("Error reading %s: %s", las_path, exc)
                tile_buffers.clear()
                continue

        # ── Phase 4: merge chunks and write final tiles ────────────
        if progress_callback:
            progress_callback("Merging tile chunks…", 90.0)

        with self._db.connect() as conn:
            tile_idxs = sorted(tile_total_points.keys())
            for i, tidx in enumerate(tile_idxs):
                # Gather all chunks for this tile
                pattern = f"tile_{tidx:04d}_*.bin"
                chunk_paths = sorted(chunk_dir.glob(pattern))
                if not chunk_paths:
                    continue

                if len(chunk_paths) == 1:
                    # Single chunk — read directly
                    data = np.fromfile(str(chunk_paths[0]), dtype=_TILE_CHUNK_DTYPE)
                else:
                    # Byte-level concatenate all chunks into one temp file,
                    # then read once.  Avoids loading N arrays + np.concatenate.
                    merged_path = chunk_dir / f"tile_{tidx:04d}_merged.bin"
                    with open(merged_path, "wb") as out:
                        for cp in chunk_paths:
                            with open(cp, "rb") as inp:
                                while True:
                                    buf = inp.read(16 * 1024 * 1024)  # 16 MiB
                                    if not buf:
                                        break
                                    out.write(buf)
                    data = np.fromfile(str(merged_path), dtype=_TILE_CHUNK_DTYPE)
                    merged_path.unlink()  # clean up the merge temp

                n_pts = len(data)
                if n_pts < min_points_per_tile:
                    for p in chunk_paths:
                        p.unlink()
                    continue

                xs = data["x"]; ys = data["y"]; zs = data["z"]
                cls = data["classification"]; intens = data["intensity"]
                rns = data["return_number"]; nrs = data["number_of_returns"]
                srcs = data["point_source_id"]; gts = data["gps_time"]
                sars = data["scan_angle_rank"]; sa_degs = data["scan_angle"]
                sdfs = data["scan_direction_flag"]; efls = data["edge_of_flight_line"]
                uds = data["user_data"]
                reds = data["red"]; grns = data["green"]; blus = data["blue"]
                kps = data["key_point"]; syns = data["synthetic"]
                whs = data["withheld"]; ovs = data["overlap"]
                del data  # free the structured array

                tile_id = f"tile_{tidx:04d}"
                tile_path = tiles_dir / f"{tile_id}.las"
                _write_las_file(
                    tile_path, xs, ys, zs,
                    classes=cls, intensities=intens,
                    return_numbers=rns, num_returns=nrs,
                    point_source_ids=srcs,
                    gps_times=gts, scan_angle_ranks=sars,
                    scan_angles=sa_degs,
                    scan_direction_flags=sdfs, edge_of_flight_lines=efls,
                    user_data_array=uds,
                    reds=reds, greens=grns, blues=blus,
                    key_points=kps, synthetics=syns,
                    withhelds=whs, overlaps=ovs,
                    header_template=header_template,
                )

                # Determine dominant scanner for this tile
                sc_map = tile_scanner_map.get(tidx, {})
                dominant_scanner = ''
                all_scanner_names: List[str] = []
                if sc_map:
                    sorted_scanners = sorted(sc_map.items(), key=lambda kv: kv[1], reverse=True)
                    dominant_scanner = sorted_scanners[0][0] if sorted_scanners[0][0] else ''
                    all_scanner_names = [s for s, _ in sorted_scanners if s]

                # Representative flight_line (mode of point_source_id)
                if len(srcs) > 0:
                    src_int = srcs.astype(np.int64)
                    flight_line_val = int(np.bincount(src_int).argmax())
                else:
                    flight_line_val = 0

                # Dominant sensor_type
                st_map = tile_sensor_type_map.get(tidx, {})
                dominant_sensor_type = sensor_type
                if not dominant_sensor_type and st_map:
                    best_st = max(st_map.items(), key=lambda kv: kv[1])
                    dominant_sensor_type = best_st[0] if best_st[0] else ''

                self._db.insert_tile(
                    conn, tile_id=tile_id, filename=f"{tile_id}.las",
                    bbox=tile_bboxes[tidx], point_count=n_pts,
                    flight_line=flight_line_val,
                    scanner=dominant_scanner,
                    all_scanners=all_scanner_names,
                    sensor_type=dominant_sensor_type,
                    flightline_sensor_types=tile_fl_sensor_types.get(tidx),
                    crs_epsg=crs_epsg,
                    crs_wkt=crs_wkt,
                    status=TileStatus.IMPORTED,
                )
                imported_ids.append(tile_id)

                # Clean up chunks for this tile
                for p in chunk_paths:
                    p.unlink()

                if progress_callback:
                    pct = 90.0 + ((i + 1) / len(tile_idxs)) * 10.0
                    progress_callback(f"Tile {i+1}/{len(tile_idxs)}…", pct)

        # Remove temp directory if empty
        try:
            chunk_dir.rmdir()
        except OSError:
            pass

        if progress_callback:
            progress_callback("Import complete", 100.0)

        logger.info("Imported %d tile(s) from %d file(s)", len(imported_ids), len(las_files))
        return imported_ids

    # ── tile data access ───────────────────────────────────────────

    def load_tile_points(self, tile_id: str) -> Optional[PointCloud]:
        """
        Load all points for a tile as ``(x, y, z)`` numpy arrays.

        Uses a simple in-memory cache; for large projects the caller
        should manage cache eviction.

        Args:
            tile_id: Tile identifier.

        Returns:
            Tuple of ``(x, y, z)`` arrays or ``None`` if the tile is not found.
        """
        if tile_id in self._point_cache:
            return self._point_cache[tile_id]

        tile_info = self._db.get_tile(tile_id)
        if tile_info is None:
            logger.warning("Tile %s not found in database", tile_id)
            return None

        tiles_dir = self._pm.tiles_dir
        assert tiles_dir is not None
        las_path = tiles_dir / tile_info["filename"]
        if not las_path.is_file():
            logger.warning("Tile file missing: %s", las_path)
            return None

        try:
            with laspy.open(las_path) as reader:
                las_data = reader.read()
                xs = np.array(las_data.x, dtype=np.float64)
                ys = np.array(las_data.y, dtype=np.float64)
                zs = np.array(las_data.z, dtype=np.float64)
        except Exception as exc:
            logger.error("Failed to load tile %s: %s", tile_id, exc)
            return None

        self._point_cache[tile_id] = (xs, ys, zs)
        return (xs, ys, zs)

    def load_tile_points_full(self, tile_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all point attributes for a tile.

        Returns a dict with keys ``x, y, z, classification, intensity,
        return_number, point_source_id, sensor_type`` (and optional LAS fields).
        ``sensor_type`` is encoded as 0=unknown, 1=topo, 2=bathy.  Each value is a 1-D numpy array.
        """
        tile_info = self._db.get_tile(tile_id)
        if tile_info is None:
            return None

        tiles_dir = self._pm.tiles_dir
        assert tiles_dir is not None
        las_path = tiles_dir / tile_info["filename"]
        if not las_path.is_file():
            return None

        try:
            with laspy.open(las_path) as reader:
                las_data = reader.read()
                result = {
                    "x": np.array(las_data.x, dtype=np.float64),
                    "y": np.array(las_data.y, dtype=np.float64),
                    "z": np.array(las_data.z, dtype=np.float64),
                }
                # Optional fields
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

                # Build per-point sensor_type from flightline→sensor_type mapping
                fl_st_raw = tile_info.get("flightline_sensor_types", "{}")
                try:
                    fl_st_json: dict = json.loads(fl_st_raw) if isinstance(fl_st_raw, str) else fl_st_raw
                    # JSON keys are strings; convert to int
                    fl_to_st: Dict[int, int] = {}
                    for k, v in fl_st_json.items():
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
                except Exception:
                    logger.debug("Could not parse flightline_sensor_types for %s", tile_id)

                return result
        except Exception as exc:
            logger.error("Failed to load tile %s: %s", tile_id, exc)
            return None

    def update_tile_classifications(
        self, tile_id: str, indices: np.ndarray, new_class: int
    ) -> bool:
        """
        Update the classification field for a subset of points in a tile.

        Args:
            tile_id:   Tile identifier.
            indices:   0-based indices of points to reclassify.
            new_class: New ASPRS classification code.

        Returns:
            ``True`` on success.
        """
        tile_info = self._db.get_tile(tile_id)
        if tile_info is None:
            return False

        tiles_dir = self._pm.tiles_dir
        assert tiles_dir is not None
        las_path = tiles_dir / tile_info["filename"]
        if not las_path.is_file():
            return False

        # Backup before modifying
        backup_path = las_path.with_suffix(las_path.suffix + ".bak")
        if not backup_path.exists():
            import shutil
            shutil.copy2(las_path, backup_path)
            logger.debug("Backup created: %s", backup_path)

        try:
            las_data = laspy.read(las_path)
            old_classes = las_data.classification[indices].copy()
            las_data.classification[indices] = new_class
            las_data.write(str(las_path))
        except Exception as exc:
            logger.error("Failed to update classifications for %s: %s", tile_id, exc)
            return False

        # Invalidate cache
        self._point_cache.pop(tile_id, None)

        # Record edit
        with self._db.connect() as conn:
            self._db.add_edit_command(
                conn,
                tile_id,
                {
                    "type": "classify",
                    "point_indices": indices.tolist(),
                    "old_class": old_classes.tolist() if len(old_classes) > 0 else [],
                    "new_class": new_class,
                },
            )
            self._db.update_status(conn, tile_id, TileStatus.EDITED)

        logger.info("Updated %d point(s) in %s to class %d", len(indices), tile_id, new_class)
        return True

    def update_tile_status(self, tile_id: str, status: str) -> None:
        """Update a tile's processing status in the database."""
        with self._db.connect() as conn:
            self._db.update_status(conn, tile_id, status)

    def get_tile_bbox(self, tile_id: str) -> Optional[BBox]:
        """Return the bounding box of a tile."""
        info = self._db.get_tile(tile_id)
        if info is None:
            return None
        return (
            info["bbox_min_x"], info["bbox_min_y"],
            info["bbox_max_x"], info["bbox_max_y"],
        )

    def get_tiles_in_viewport(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> List[str]:
        """Return tile IDs whose bbox intersects the given viewport."""
        tiles = self._db.get_tiles_in_bbox(min_x, min_y, max_x, max_y)
        return [t["id"] for t in tiles]

    def clear_cache(self) -> None:
        """Clear the in-memory point cache."""
        self._point_cache.clear()
        logger.debug("Point cache cleared")


# ── Internal helpers ──────────────────────────────────────────────────


def _detect_sensor_type_from_las(las_path: os.PathLike) -> str:
    """
    Inspect LAS file content for bathymetric indicators.

    Checks VLR descriptions and extra dimension names for keywords
    that suggest a green-laser bathymetric sensor:

    - VLR descriptions containing: green, bathy, hydro, water, 532, cwlaser
    - Extra dimension names: amplitude, reflectance, deviation, water_depth,
      water_surface, bottom_return
    - RIEGL VLRs with model name ending in -G / g

    Returns ``'bathy'``, ``'topo'``, or ``''`` when no indicators found.
    """
    import laspy as _laspy
    try:
        with _laspy.open(las_path) as reader:
            hdr = reader.header
            pf = hdr.point_format

            # ── Check VLR descriptions for bathy keywords ──────────
            if hasattr(hdr, 'vlrs'):
                for vlr in hdr.vlrs:
                    desc = (getattr(vlr, 'description', '') or '').lower()
                    if not desc:
                        continue
                    # RIEGL VLRs encode full model name: "RIEGL VQ-880-G Extra Bytes"
                    if re.search(r'vq\d{3,4}-?g\b', desc):
                        return 'bathy'
                    # Generic bathy keywords in VLRs
                    for kw in ('green laser', 'bathy', 'hydrographic',
                               '532 nm', '532nm', 'water surface',
                               'cwlaser'):
                        if kw in desc:
                            return 'bathy'

            # ── Check extra dimension names ────────────────────────
            extra_names = set()
            for ed in pf.extra_dimensions:
                extra_names.add((ed.name or '').lower())

            _BATHY_DIMS = {
                'water_depth', 'water surface', 'bottom_return',
                'bathy_depth', 'sea_floor',
                'subsea_depth', 'subsea_surface',
            }
            if extra_names & _BATHY_DIMS:
                return 'bathy'

            # ── No bathy indicators ────────────────────────────────
            return ''
    except Exception:
        return ''


# Known scanner/sensor name patterns (Riegl, Leica, Optech, etc.)
_SCANNER_PATTERN = re.compile(
    r'(?:^|[^a-zA-Z0-9])('
    r'vq\d{3,4}(?:-[a-z])?[a-z]?'   # Riegl VQ series: vq820g, vq580, vq1560i
    r'|vux-\d+(?:-[a-z])?[a-z]?'    # Riegl VUX series: vux-1, vux-240
    r'|vq-\d+(?:-[a-z])?[a-z]?'     # Riegl VQ hyphenated: vq-860, vq-880-g
    r'|minivux-\d+(?:-[a-z])?[a-z]?' # Riegl miniVUX
    r'|als\d+'                  # Leica ALS series: als50, als70, als80
    r'|pegasus.*?\d+'           # Leica Pegasus
    r'|orion.*?\w\d+'           # Optech Orion
    r'|gemini.*?\w\d+'          # Optech Gemini
    r'|alta\s*x\d+'             # Teledyne Optech ALTA
    r'|galaxy.*?\w\d+'          # Teledyne Optech Galaxy
    r'|ri.*?copl\d+'            # RIEGL RiCOPTER
    r')(?:[^a-zA-Z0-9]|$)',
    re.IGNORECASE,
)


def _detect_sensor_type(scanner_name: str, las_path: Optional[os.PathLike] = None) -> str:
    """
    Determine sensor type from scanner name and (optionally) LAS file content.

    RIEGL convention: models ending in ``-G`` or ``g`` are green-laser
    bathymetric scanners (e.g. VQ-820-G, VQ-880-G, VQ-840-G).
    Also recognizes other known bathymetric systems (Chiroptera, Hawkeye,
    CZMIL, SHOALS, LADS, EAARL).

    When *las_path* is provided and the scanner name alone is ambiguous
    (matches ``_SCANNER_PATTERN`` but no bathy pattern), inspects VLR
    descriptions and extra dimension names for bathymetric indicators.

    Returns ``'bathy'``, ``'topo'``, or ``''`` when unknown.
    """
    if not scanner_name:
        # Try LAS content inspection as last resort
        if las_path is not None:
            return _detect_sensor_type_from_las(las_path)
        return ''
    name = scanner_name.strip().lower()

    # ── Known bathymetric scanners ────────────────────────────────
    # RIEGL -G suffix = bathymetric green laser (e.g. vq820g, vq-880-g)
    if re.search(r'vq-?\d{3,4}-?g\b', name):
        return 'bathy'

    # Hardcoded bathymetric system names
    _BATHY_NAMES = (
        'chiroptera', 'hawkeye', 'dragoneye',   # Leica / Teledyne
        'czmil', 'shoals', 'lads', 'eaarl',     # USACE / Optech / NASA
        'coastal', 'aquarius',                   # Teledyne Optech Aquarius
    )
    for bn in _BATHY_NAMES:
        if bn in name:
            return 'bathy'

    # RIEGL model with 'g' appended directly: vq820g, vq880g
    if re.search(r'vq-?\d{3,4}g\b', name):
        return 'bathy'

    # Any other recognised scanner → topo
    if _SCANNER_PATTERN.search(name):
        return 'topo'
    return ''


def _detect_scanner_from_filename(filepath: os.PathLike) -> str:
    """Extract scanner/sensor name from a LAS filename.

    Scans the filename (stem + parent dir) for known LiDAR sensor patterns.
    Returns the lowercased match or ``''`` if nothing is recognised.

    Examples:
        ``1 - vq820g - 210430_105206_vq820g - originalpoints.las`` → ``'vq820g'``
        ``vq580_flight03.las`` → ``'vq580'``
        ``als70_hd_strip_01.laz`` → ``'als70'``
    """
    fname = str(Path(filepath).stem)
    m = _SCANNER_PATTERN.search(fname)
    if m:
        return m.group(1).lower()
    # Also try the parent directory name
    parent = str(Path(filepath).parent.name)
    m = _SCANNER_PATTERN.search(parent)
    if m:
        return m.group(1).lower()
    return ''


def _read_las_header_meta(las_path: os.PathLike) -> dict:
    """Read extended metadata from a LAS header.

    Returns a dict with keys:
        ``system_id``, ``software``, ``file_source_id``, ``version``.
    """
    import laspy as _laspy
    meta = {"system_id": "", "software": "", "file_source_id": 0, "version": ""}
    try:
        with _laspy.open(las_path) as reader:
            hdr = reader.header
            meta["system_id"] = getattr(hdr, 'system_identifier', '') or ''
            meta["software"] = getattr(hdr, 'generating_software', '') or ''
            meta["file_source_id"] = int(getattr(hdr, 'file_source_id', 0) or 0)
            meta["version"] = f"{hdr.version.major}.{hdr.version.minor}"
    except Exception:
        pass
    return meta


def _read_las_header_template(las_path: Path) -> dict:
    """Snapshot the header of the first input LAS file as a template for output tiles.

    Returns a dict with keys:
        ``version``, ``point_format_id``, ``extra_dimensions``,
        ``vlrs``, ``x_scale``, ``y_scale``, ``z_scale``.
    """
    import laspy as _laspy
    with _laspy.open(las_path) as reader:
        hdr = reader.header
        pf = hdr.point_format
        template = {
            "version": f"{hdr.version.major}.{hdr.version.minor}",
            "point_format_id": pf.id,
            "extra_dimensions": list(pf.extra_dimensions),
            "vlrs": list(hdr.vlrs) if hasattr(hdr, 'vlrs') else [],
            "x_scale": float(hdr.x_scale),
            "y_scale": float(hdr.y_scale),
            "z_scale": float(hdr.z_scale),
        }
    return template


def _safe_attr(las_chunk, attr: str, dtype, default):
    """Return ``las_chunk.<attr>`` as a numpy array, or *default* if missing."""
    if hasattr(las_chunk, attr):
        return np.array(getattr(las_chunk, attr), dtype=dtype)
    return np.full(len(las_chunk), default, dtype=dtype)


def _flush_tile_buffers(
    tile_buffers: Dict[int, Dict[str, list]],
    chunk_dir: Path,
    file_idx: int,
) -> Dict[int, int]:
    """Serialize accumulated in-memory tile buffers to raw binary chunks on disk.

    Returns a dict mapping ``tile_idx → point_count`` for the flushed chunk.
    """
    tile_pts: Dict[int, int] = {}
    for tidx, buf in tile_buffers.items():
        # Concatenate all per-attribute lists into flat arrays
        arrays: Dict[str, np.ndarray] = {}
        n_pts = 0
        for buf_key, field_name in _TILE_BUF_TO_CHUNK.items():
            arr = np.concatenate(buf[buf_key])
            arrays[field_name] = arr
            if n_pts == 0:
                n_pts = len(arr)
        if n_pts == 0:
            continue

        # Build structured array and write raw binary
        data = np.empty(n_pts, dtype=_TILE_CHUNK_DTYPE)
        for field_name, arr in arrays.items():
            data[field_name] = arr

        chunk_path = chunk_dir / f"tile_{tidx:04d}_{file_idx:04d}.bin"
        data.tofile(str(chunk_path))
        tile_pts[tidx] = n_pts

    return tile_pts


def _compute_tile_grid(
    global_bbox: BBox,
    tile_size: float,
    overlap: float,
) -> List[BBox]:
    """
    Generate a regular grid of tile bounding boxes covering *global_bbox*.

    Args:
        global_bbox: ``(min_x, min_y, max_x, max_y)``.
        tile_size:   Tile edge length in CRS units.
        overlap:     Overlap between adjacent tiles.

    Returns:
        List of tile ``(min_x, min_y, max_x, max_y)`` tuples.
    """
    min_x, min_y, max_x, max_y = global_bbox
    stride = tile_size - overlap
    if stride <= 0:
        stride = tile_size

    tiles: List[BBox] = []
    x0 = min_x
    while x0 < max_x:
        y0 = min_y
        while y0 < max_y:
            tx_max = min(x0 + tile_size, max_x)
            ty_max = min(y0 + tile_size, max_y)
            tiles.append((x0, y0, tx_max, ty_max))
            y0 += stride
        x0 += stride

    return tiles


def _write_las_file(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    classes: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    point_source_ids: Optional[np.ndarray] = None,
    gps_times: Optional[np.ndarray] = None,
    scan_angle_ranks: Optional[np.ndarray] = None,
    scan_angles: Optional[np.ndarray] = None,
    scan_direction_flags: Optional[np.ndarray] = None,
    edge_of_flight_lines: Optional[np.ndarray] = None,
    user_data_array: Optional[np.ndarray] = None,
    reds: Optional[np.ndarray] = None,
    greens: Optional[np.ndarray] = None,
    blues: Optional[np.ndarray] = None,
    key_points: Optional[np.ndarray] = None,
    synthetics: Optional[np.ndarray] = None,
    withhelds: Optional[np.ndarray] = None,
    overlaps: Optional[np.ndarray] = None,
    header_template: Optional[dict] = None,
) -> None:
    """
    Write a set of points to a LAS file via laspy.

    Creates a LAS header matching *header_template* (point format, version,
    VLRs, extra dimensions, scales) when provided; falls back to LAS 1.4
    point format 6 otherwise.  Only writes attributes that the chosen point
    format actually supports.
    """
    if not HAS_LASPY:
        raise RuntimeError("laspy required")

    n = len(xs)

    # ── Build header from template (or fallback) ──────────────────
    if header_template is not None:
        header = laspy.LasHeader(
            version=header_template["version"],
            point_format=header_template["point_format_id"],
        )
        # Copy VLRs (carries CRS, etc.)
        for vlr in header_template["vlrs"]:
            header.vlrs.append(vlr)
        # Copy extra-dimension definitions (union across all input files)
        for ed in header_template["extra_dimensions"]:
            # ed is (name, type) tuple from the union, or ExtraBytesParams
            if isinstance(ed, tuple):
                dim_name, dim_type = ed
            else:
                dim_name, dim_type = ed.name, ed.type
            try:
                header.add_extra_dim(name=dim_name, type=dim_type)
            except Exception:
                logger.debug("Skipping extra dim %s (may already exist)", dim_name)
        # Copy scales from source
        header.x_scale = header_template["x_scale"]
        header.y_scale = header_template["y_scale"]
        header.z_scale = header_template["z_scale"]
    else:
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.x_scale = 0.001
        header.y_scale = 0.001
        header.z_scale = 0.001

    # Per-tile offsets
    header.x_offset = xs.min() if n > 0 else 0.0
    header.y_offset = ys.min() if n > 0 else 0.0
    header.z_offset = zs.min() if n > 0 else 0.0

    las_data = laspy.LasData(header)
    las_data.x = xs
    las_data.y = ys
    las_data.z = zs

    def _set(attr_name: str, arr: Optional[np.ndarray], default_dtype, default_val) -> None:
        """Set *attr_name* on *las_data* if the point format supports it."""
        if not hasattr(las_data, attr_name):
            return
        if arr is not None and len(arr) == n:
            setattr(las_data, attr_name, arr)
        else:
            setattr(las_data, attr_name, np.full(n, default_val, dtype=default_dtype))

    _set("classification", classes, np.uint8, 0)
    _set("intensity", intensities, np.uint16, 0)
    _set("return_number", return_numbers, np.uint8, 1)
    _set("number_of_returns", num_returns, np.uint8, 1)
    _set("point_source_id", point_source_ids, np.uint16, 0)
    _set("gps_time", gps_times, np.float64, 0.0)

    # scan_angle vs scan_angle_rank: format 0-5 uses scan_angle_rank (int8),
    # format 6-10 uses scan_angle (int16, stored as scaled integer in file).
    if hasattr(las_data, "scan_angle_rank"):
        _set("scan_angle_rank", scan_angle_ranks, np.int8, 0)
    if hasattr(las_data, "scan_angle"):
        if scan_angles is not None and len(scan_angles) == n:
            las_data.scan_angle = scan_angles
        elif scan_angle_ranks is not None and len(scan_angle_ranks) == n:
            # Convert int8 rank → float degrees (LAS 1.4 spec: rank * 0.006)
            las_data.scan_angle = scan_angle_ranks.astype(np.float64) * 0.006
        else:
            las_data.scan_angle = np.full(n, 0.0, dtype=np.float64)

    _set("scan_direction_flag", scan_direction_flags, np.uint8, 0)
    _set("edge_of_flight_line", edge_of_flight_lines, np.uint8, 0)
    _set("user_data", user_data_array, np.uint8, 0)

    # RGB — only point formats 2, 3, 5, 7, 8, 10
    if hasattr(las_data, "red"):
        _set("red", reds, np.uint16, 0)
    if hasattr(las_data, "green"):
        _set("green", greens, np.uint16, 0)
    if hasattr(las_data, "blue"):
        _set("blue", blues, np.uint16, 0)

    # Classification flags (LAS 1.4 only)
    for attr_name, arr in [
        ("key_point", key_points),
        ("synthetic", synthetics),
        ("withheld", withhelds),
        ("overlap", overlaps),
    ]:
        if arr is not None and len(arr) == n and hasattr(las_data, attr_name):
            try:
                setattr(las_data, attr_name, arr.astype(np.uint8))
            except Exception:
                pass

    path.parent.mkdir(parents=True, exist_ok=True)
    las_data.write(str(path))
    logger.debug("Wrote %d points to %s", n, path)
