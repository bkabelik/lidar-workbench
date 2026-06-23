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

        Automatically detects whether the files are individual tiles or
        flight strips (based on spatial extent) and applies tiling as needed.

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

        # Aggregate all points to determine global extent and point density
        if progress_callback:
            progress_callback("Reading LAS headers…", 0.0)

        all_bboxes: List[BBox] = []
        total_points = 0
        header_infos: List[Dict[str, Any]] = []

        for i, las_path in enumerate(las_files):
            try:
                with laspy.open(las_path) as reader:
                    hdr = reader.header
                    bbox: BBox = (
                        hdr.x_min, hdr.y_min,
                        hdr.x_max, hdr.y_max,
                    )
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
                progress_callback("Reading headers…", (i + 1) / len(las_files) * 10.0)

        if not header_infos:
            logger.warning("No readable LAS/LAZ files found")
            return []

        # Compute global bbox
        global_bbox: BBox = (
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        )
        area_m2 = (global_bbox[2] - global_bbox[0]) * (global_bbox[3] - global_bbox[1])
        point_density = total_points / area_m2 if area_m2 > 0 else 0.0

        logger.info(
            "Global bbox: (%.2f, %.2f) – (%.2f, %.2f), %d points, density %.2f pts/m²",
            *global_bbox, total_points, point_density,
        )

        # Determine tile size
        if tile_size_m is None:
            if point_density > 0:
                # target: TARGET_POINTS_PER_TILE per tile
                tile_area = TARGET_POINTS_PER_TILE / point_density
                tile_size_m = np.sqrt(tile_area)
                # Round to nearest 50 m
                tile_size_m = max(50.0, round(tile_size_m / 50.0) * 50.0)
            else:
                tile_size_m = DEFAULT_TILE_SIZE_M

        overlap_m = overlap_m if overlap_m is not None else DEFAULT_TILE_OVERLAP_M

        logger.info("Tile size: %.0f m, overlap: %.0f m", tile_size_m, overlap_m)

        # Generate tile grid
        if progress_callback:
            progress_callback("Generating tile grid…", 10.0)

        tile_bboxes = _compute_tile_grid(global_bbox, tile_size_m, overlap_m)

        if progress_callback:
            progress_callback(f"Importing {len(tile_bboxes)} tile(s)…", 15.0)

        # Load all points (streaming) and assign to tiles
        imported_ids: List[str] = []
        tiles_dir = self._pm.tiles_dir
        assert tiles_dir is not None

        with self._db.connect() as conn:
            for tile_idx, tbbox in enumerate(tile_bboxes):
                tile_id = f"tile_{tile_idx:04d}"
                tile_filename = f"{tile_id}.las"
                tile_path = tiles_dir / tile_filename

                # Collect points that fall into this tile bbox
                xs, ys, zs, classes, intensities, return_nums = _collect_points_in_bbox(
                    header_infos, tbbox, progress_callback=None
                )

                if len(xs) == 0:
                    logger.debug("Tile %s has no points — skipping", tile_id)
                    continue

                # Write tile LAS file
                _write_las_file(
                    tile_path,
                    xs, ys, zs,
                    classes=classes,
                    intensities=intensities,
                    return_numbers=return_nums,
                )

                # Insert into database
                self._db.insert_tile(
                    conn,
                    tile_id=tile_id,
                    filename=tile_filename,
                    bbox=tbbox,
                    point_count=len(xs),
                    status=TileStatus.IMPORTED,
                )
                imported_ids.append(tile_id)

                pct = 15.0 + (tile_idx + 1) / len(tile_bboxes) * 80.0
                if progress_callback:
                    progress_callback(f"Tile {tile_idx + 1}/{len(tile_bboxes)}", pct)

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


def _collect_points_in_bbox(
    header_infos: List[Dict[str, Any]],
    bbox: BBox,
    progress_callback: Optional[callable] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Stream through LAS files and collect points that fall within *bbox*.

    Returns:
        ``(xs, ys, zs, classes, intensities, return_numbers)`` as float/int arrays.
    """
    min_x, min_y, max_x, max_y = bbox
    xs_all: List[np.ndarray] = []
    ys_all: List[np.ndarray] = []
    zs_all: List[np.ndarray] = []
    cls_all: List[np.ndarray] = []
    int_all: List[np.ndarray] = []
    rn_all: List[np.ndarray] = []

    for info in header_infos:
        # Quick-reject: skip files that don't intersect bbox
        fb = info["bbox"]
        if fb[2] < min_x or fb[0] > max_x or fb[3] < min_y or fb[1] > max_y:
            continue

        try:
            with laspy.open(info["path"]) as reader:
                las_data = reader.read()
                x = np.array(las_data.x, dtype=np.float64)
                y = np.array(las_data.y, dtype=np.float64)

                mask = (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
                if not mask.any():
                    continue

                xs_all.append(x[mask])
                ys_all.append(y[mask])
                zs_all.append(np.array(las_data.z, dtype=np.float64)[mask])

                if hasattr(las_data, "classification"):
                    cls_all.append(np.array(las_data.classification, dtype=np.uint8)[mask])
                else:
                    cls_all.append(np.zeros(mask.sum(), dtype=np.uint8))

                if hasattr(las_data, "intensity"):
                    int_all.append(np.array(las_data.intensity, dtype=np.uint16)[mask])
                else:
                    int_all.append(np.zeros(mask.sum(), dtype=np.uint16))

                if hasattr(las_data, "return_number"):
                    rn_all.append(np.array(las_data.return_number, dtype=np.uint8)[mask])
                else:
                    rn_all.append(np.ones(mask.sum(), dtype=np.uint8))
        except Exception as exc:
            logger.error("Error reading %s: %s", info["path"], exc)
            continue

    if not xs_all:
        empty = np.array([], dtype=np.float64)
        empty_u8 = np.array([], dtype=np.uint8)
        empty_u16 = np.array([], dtype=np.uint16)
        return empty, empty, empty, empty_u8, empty_u16, empty_u8

    return (
        np.concatenate(xs_all),
        np.concatenate(ys_all),
        np.concatenate(zs_all),
        np.concatenate(cls_all),
        np.concatenate(int_all),
        np.concatenate(rn_all),
    )


def _write_las_file(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    classes: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
    return_numbers: Optional[np.ndarray] = None,
) -> None:
    """
    Write a set of points to a LAS file via laspy.

    Creates LAS 1.4 point format 6 (includes classification, intensity,
    return number).
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

    if classes is not None and len(classes) == n:
        las_data.classification = classes
    else:
        las_data.classification = np.zeros(n, dtype=np.uint8)

    if intensities is not None and len(intensities) == n:
        las_data.intensity = intensities
    else:
        las_data.intensity = np.zeros(n, dtype=np.uint16)

    if return_numbers is not None and len(return_numbers) == n:
        las_data.return_number = return_numbers
    else:
        las_data.return_number = np.ones(n, dtype=np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    las_data.write(str(path))

    logger.debug("Wrote %d points to %s", n, path)
