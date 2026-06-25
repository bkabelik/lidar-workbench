"""
LiDAR Workbench — Tile Manager.

Handles LAS/LAZ file import, spatial tiling, metadata tracking,
and lazy data loading for the view system.
"""

from __future__ import annotations

import logging
import os
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
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """
        Import all LAS/LAZ files from a directory into the project.

        Streams each file once in chunks, bins points into the tile grid
        using vectorised numpy, and writes tiles incrementally.  Each
        input file is assigned a sequential flight-line number stored in
        ``point_source_id``.

        Args:
            directory:          Path to a directory containing ``.las`` / ``.laz`` files.
            tile_size_m:        Tile edge length in meters.  When ``None``, computed
                                automatically from point density to hit ~1.5 M points/tile.
            overlap_m:          Overlap between adjacent tiles in meters.
            progress_callback:  Optional ``callable(step: str, pct: float)`` for progress
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

        for i, las_path in enumerate(las_files):
            try:
                with laspy.open(las_path) as reader:
                    hdr = reader.header
                    bbox: BBox = (hdr.x_min, hdr.y_min, hdr.x_max, hdr.y_max)
                    n_pts = hdr.point_count
                    all_bboxes.append(bbox)
                    total_points += n_pts
                    header_infos.append({
                        "path": las_path,
                        "bbox": bbox,
                        "point_count": n_pts,
                        "version": f"{hdr.version.major}.{hdr.version.minor}",
                    })
            except Exception as exc:
                logger.error("Failed to read header of %s: %s", las_path, exc)
                continue
            if progress_callback:
                progress_callback("Reading headers…", (i + 1) / len(las_files) * 5.0)

        if not header_infos:
            logger.warning("No readable LAS/LAZ files found")
            return []

        # ── Phase 2: compute grid ───────────────────────────────────
        global_bbox: BBox = (
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        )
        area_m2 = (global_bbox[2] - global_bbox[0]) * (global_bbox[3] - global_bbox[1])
        point_density = total_points / area_m2 if area_m2 > 0 else 0.0

        if tile_size_m is None:
            if point_density > 0:
                tile_area = TARGET_POINTS_PER_TILE / point_density
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

        # ── Phase 3: single-pass streaming import ───────────────────
        tiles_dir = self._pm.tiles_dir
        assert tiles_dir is not None

        # Per-tile point accumulators  (lists of numpy arrays, flushed periodically)
        tile_buffers: Dict[int, Dict[str, list]] = {}  # tile_idx → {x:[], y:[], z:[], cl:[], in:[], rn:[], src:[]}

        imported_ids: List[str] = []
        total_processed = 0

        for file_idx, info in enumerate(header_infos):
            flight_line = file_idx + 1  # 1-based flight strip number
            las_path = info["path"]
            file_total = info["point_count"]
            file_processed = 0

            try:
                with laspy.open(las_path) as reader:
                    for chunk in reader.chunk_iterator(1_000_000):  # 1M pts per chunk
                        n = len(chunk)
                        x = np.array(chunk.x, dtype=np.float64)
                        y = np.array(chunk.y, dtype=np.float64)

                        # Vectorised tile-index computation
                        col = ((x - grid_x0) / stride).astype(np.int64)
                        row = ((y - grid_y0) / stride).astype(np.int64)
                        # Clamp to valid range
                        col = np.clip(col, 0, grid_cols - 1)
                        max_rows = int(np.ceil((global_bbox[3] - grid_y0) / stride))
                        row = np.clip(row, 0, max_rows - 1)
                        tile_idx_arr = row * grid_cols + col

                        # Bin points by tile
                        z = np.array(chunk.z, dtype=np.float64)
                        cl = (_safe_attr(chunk, "classification", np.uint8, 0))
                        intens = (_safe_attr(chunk, "intensity", np.uint16, 0))
                        rn = (_safe_attr(chunk, "return_number", np.uint8, 1))
                        src = np.full(n, flight_line, dtype=np.uint16)

                        for tidx in range(len(tile_bboxes)):
                            mask = tile_idx_arr == tidx
                            if not mask.any():
                                continue
                            if tidx not in tile_buffers:
                                tile_buffers[tidx] = {"x":[], "y":[], "z":[],
                                                       "cl":[], "in":[], "rn":[], "src":[]}
                            buf = tile_buffers[tidx]
                            buf["x"].append(x[mask])
                            buf["y"].append(y[mask])
                            buf["z"].append(z[mask])
                            buf["cl"].append(cl[mask])
                            buf["in"].append(intens[mask])
                            buf["rn"].append(rn[mask])
                            buf["src"].append(src[mask])

                        file_processed += n
                        total_processed += n
                        if progress_callback:
                            pct = 5.0 + (total_processed / total_points) * 90.0
                            progress_callback(
                                f"File {file_idx+1}/{len(header_infos)} — "
                                f"{file_processed/file_total*100:.0f}%", pct
                            )

            except Exception as exc:
                logger.error("Error reading %s: %s", las_path, exc)
                continue

        # ── Phase 4: write tile files ───────────────────────────────
        if progress_callback:
            progress_callback("Writing tile files…", 95.0)

        with self._db.connect() as conn:
            for tidx in sorted(tile_buffers.keys()):
                buf = tile_buffers[tidx]
                xs = np.concatenate(buf["x"])
                ys = np.concatenate(buf["y"])
                zs = np.concatenate(buf["z"])
                cls = np.concatenate(buf["cl"])
                intens = np.concatenate(buf["in"])
                rns = np.concatenate(buf["rn"])
                srcs = np.concatenate(buf["src"])

                if len(xs) == 0:
                    continue

                tile_id = f"tile_{tidx:04d}"
                tile_path = tiles_dir / f"{tile_id}.las"
                _write_las_file(
                    tile_path, xs, ys, zs,
                    classes=cls, intensities=intens,
                    return_numbers=rns, point_source_ids=srcs,
                )

                self._db.insert_tile(
                    conn, tile_id=tile_id, filename=f"{tile_id}.las",
                    bbox=tile_bboxes[tidx], point_count=len(xs),
                    status=TileStatus.IMPORTED,
                )
                imported_ids.append(tile_id)

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
        return_number``.  Each value is a 1-D numpy array.
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
                if hasattr(las_data, "classification"):
                    result["classification"] = np.array(las_data.classification, dtype=np.uint8)
                else:
                    result["classification"] = np.zeros(len(result["x"]), dtype=np.uint8)

                if hasattr(las_data, "intensity"):
                    result["intensity"] = np.array(las_data.intensity, dtype=np.uint16)
                else:
                    result["intensity"] = np.zeros(len(result["x"]), dtype=np.uint16)

                if hasattr(las_data, "return_number"):
                    result["return_number"] = np.array(las_data.return_number, dtype=np.uint8)
                else:
                    result["return_number"] = np.ones(len(result["x"]), dtype=np.uint8)

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


def _safe_attr(las_chunk, attr: str, dtype, default):
    """Return ``las_chunk.<attr>`` as a numpy array, or *default* if missing."""
    if hasattr(las_chunk, attr):
        return np.array(getattr(las_chunk, attr), dtype=dtype)
    return np.full(len(las_chunk), default, dtype=dtype)


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
    point_source_ids: Optional[np.ndarray] = None,
) -> None:
    """
    Write a set of points to a LAS file via laspy.

    Creates LAS 1.4 point format 6 (includes classification, intensity,
    return number, point source ID).
    """
    if not HAS_LASPY:
        raise RuntimeError("laspy required")

    n = len(xs)
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.x_offset = xs.min() if n > 0 else 0.0
    header.y_offset = ys.min() if n > 0 else 0.0
    header.z_offset = zs.min() if n > 0 else 0.0
    header.x_scale = 0.001
    header.y_scale = 0.001
    header.z_scale = 0.001

    las_data = laspy.LasData(header)
    las_data.x = xs
    las_data.y = ys
    las_data.z = zs

    las_data.classification = (
        classes if classes is not None and len(classes) == n
        else np.zeros(n, dtype=np.uint8)
    )
    las_data.intensity = (
        intensities if intensities is not None and len(intensities) == n
        else np.zeros(n, dtype=np.uint16)
    )
    las_data.return_number = (
        return_numbers if return_numbers is not None and len(return_numbers) == n
        else np.ones(n, dtype=np.uint8)
    )
    if point_source_ids is not None and len(point_source_ids) == n:
        las_data.point_source_id = point_source_ids

    path.parent.mkdir(parents=True, exist_ok=True)
    las_data.write(str(path))
    logger.debug("Wrote %d points to %s", n, path)
