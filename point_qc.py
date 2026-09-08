"""
point_qc.py - LiDAR Quality Control & Raster Generation Engine.

Computes:
1. Point density rasters (pts / m²)
2. Nominal point spacing rasters (m)
3. Strip height difference rasters (dZ = Z_A - Z_B)
4. Vector tile grid generation from database metadata
All rasters are exported as GeoTIFFs with internal pyramid overviews
for high-performance viewing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

logger = logging.getLogger(__name__)


def get_project_crs_string(database: Any) -> str:
    """Retrieve the project CRS as an EPSG string (e.g. 'EPSG:25833')."""
    if database is None:
        return "EPSG:32633"
    try:
        tiles = database.get_all_tiles()
        for t in tiles:
            epsg = t.get("crs_epsg")
            if epsg:
                return f"EPSG:{epsg}"
    except Exception as exc:
        logger.warning("Failed to retrieve CRS from database: %s", exc)
    return "EPSG:32633"


def _get_tile_bbox(tile: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Extract (min_x, min_y, max_x, max_y) supporting both bbox_min_x and min_x schemas."""
    min_x = tile.get("bbox_min_x") if tile.get("bbox_min_x") is not None else tile.get("min_x")
    max_x = tile.get("bbox_max_x") if tile.get("bbox_max_x") is not None else tile.get("max_x")
    min_y = tile.get("bbox_min_y") if tile.get("bbox_min_y") is not None else tile.get("min_y")
    max_y = tile.get("bbox_max_y") if tile.get("bbox_max_y") is not None else tile.get("max_y")
    if None in (min_x, max_x, min_y, max_y):
        return None
    return float(min_x), float(min_y), float(max_x), float(max_y)


def _get_target_bboxes(
    tile_manager: Any, database: Any, target_tids: List[str]
) -> List[Tuple[float, float, float, float]]:
    """Determine real geographic bounding boxes from LAS headers or database metadata."""
    bboxes: List[Tuple[float, float, float, float]] = []
    tile_dict = {}
    if database:
        try:
            for t in database.get_all_tiles():
                tile_dict[str(t["id"])] = t
        except Exception:
            pass

    for tid in target_tids:
        found_las = False
        if tile_manager and hasattr(tile_manager, "tile_las_path"):
            las_p = tile_manager.tile_las_path(tid)
            if las_p and Path(las_p).is_file():
                try:
                    import laspy
                    with laspy.open(las_p) as r:
                        h = r.header
                        if h.point_count > 0:
                            bboxes.append((float(h.x_min), float(h.y_min), float(h.x_max), float(h.y_max)))
                            found_las = True
                except Exception:
                    pass

        if not found_las:
            t_info = tile_dict.get(tid)
            if t_info:
                b = _get_tile_bbox(t_info)
                if b is not None:
                    bboxes.append(b)
    return bboxes


def generate_density_raster(
    tile_manager: Any,
    database: Any,
    tile_ids: Optional[List[str]] = None,
    cell_size: float = 1.0,
    output_path: Optional[Path | str] = None,
    return_filter: str = "all",  # "all", "first", "ground"
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Stream project tiles to generate a high-resolution point density raster (pts/m²).

    Args:
        tile_manager: TileManager instance for streaming tile point data.
        database: Project Database instance.
        tile_ids: Optional subset of tile IDs (if None, all tiles are used).
        cell_size: Grid resolution in coordinate units (meters), default 1.0 m.
        output_path: Target GeoTIFF output path.
        return_filter: "all", "first", or "ground".
        progress_callback: Optional fn(current, total, status_text).

    Returns:
        Dict with keys: output_path, min_density, max_density, mean_density,
                        total_cells, active_cells, crs, bbox.
    """
    if database is None or tile_manager is None:
        raise ValueError("Both tile_manager and database must be provided.")

    all_tiles = database.get_all_tiles()
    tile_dict = {t["id"]: t for t in all_tiles}

    if tile_ids is None or len(tile_ids) == 0:
        target_tids = [t["id"] for t in all_tiles]
    else:
        target_tids = [tid for tid in tile_ids if tid in tile_dict]

    if not target_tids:
        raise ValueError("No matching tiles found to generate density raster.")

    # Determine global bounding box from LAS headers or DB
    bboxes = _get_target_bboxes(tile_manager, database, target_tids)

    if not bboxes:
        raise ValueError("Tile bounding boxes are missing in the database.")

    global_min_x = min(b[0] for b in bboxes)
    global_min_y = min(b[1] for b in bboxes)
    global_max_x = max(b[2] for b in bboxes)
    global_max_y = max(b[3] for b in bboxes)

    # Snap bounds to cell_size multiples
    global_min_x = np.floor(global_min_x / cell_size) * cell_size
    global_max_x = np.ceil(global_max_x / cell_size) * cell_size
    global_min_y = np.floor(global_min_y / cell_size) * cell_size
    global_max_y = np.ceil(global_max_y / cell_size) * cell_size

    width = int(round((global_max_x - global_min_x) / cell_size))
    height = int(round((global_max_y - global_min_y) / cell_size))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid grid dimensions: {width} x {height}")

    logger.info("Allocating QC density grid: %d x %d (cell size: %.2f m)", width, height, cell_size)

    # 32-bit float count grid
    count_grid = np.zeros((height, width), dtype=np.float32)

    total_tiles = len(target_tids)
    for idx, tid in enumerate(target_tids):
        if progress_callback:
            progress_callback(idx, total_tiles, f"Processing tile {tid} ({idx+1}/{total_tiles})…")

        data = tile_manager.load_tile_points_full(tid)
        if data is None or "x" not in data or len(data["x"]) == 0:
            continue

        x = data["x"]
        y = data["y"]

        # Apply return / classification filter if requested
        if return_filter == "first" and "return_number" in data:
            mask = data["return_number"] == 1
            x = x[mask]
            y = y[mask]
        elif return_filter == "ground" and "classification" in data:
            mask = np.isin(data["classification"], [2, 9, 8])  # Ground, Water, Reserved
            x = x[mask]
            y = y[mask]

        if len(x) == 0:
            continue

        # Fast 2D histogram
        col_indices = np.floor((x - global_min_x) / cell_size).astype(np.int64)
        # Note: image row 0 is top (max_y) down to row height-1 (min_y)
        row_indices = np.floor((global_max_y - y) / cell_size).astype(np.int64)

        valid = (col_indices >= 0) & (col_indices < width) & (row_indices >= 0) & (row_indices < height)
        col_indices = col_indices[valid]
        row_indices = row_indices[valid]

        flat_indices = row_indices * width + col_indices
        counts = np.bincount(flat_indices, minlength=width * height)
        count_grid += counts.reshape((height, width)).astype(np.float32)

    # Convert counts to density (pts / m²)
    area_per_cell = cell_size * cell_size
    density_grid = count_grid / area_per_cell

    # Set 0 counts to nodata (-9999.0)
    nodata_val = -9999.0
    active_mask = count_grid > 0
    density_grid[~active_mask] = nodata_val

    # Default output path if not provided
    if output_path is None:
        qc_dir = Path(tile_manager._pm.project_root) / "qc" / "rasters"
        qc_dir.mkdir(parents=True, exist_ok=True)
        output_path = qc_dir / f"point_density_{cell_size:.1f}m.tif"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    crs_str = get_project_crs_string(database)
    transform = from_origin(global_min_x, global_max_y, cell_size, cell_size)

    # Write GeoTIFF with pyramid overviews
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs_str,
        transform=transform,
        nodata=nodata_val,
        compress="lzw",
    ) as dst:
        dst.write(density_grid, 1)

    # Build internal pyramid overviews for high performance
    if max(width, height) >= 256:
        with rasterio.open(output_path, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    active_vals = density_grid[active_mask]
    stats = {
        "output_path": str(output_path),
        "min_density": float(active_vals.min()) if len(active_vals) > 0 else 0.0,
        "max_density": float(active_vals.max()) if len(active_vals) > 0 else 0.0,
        "mean_density": float(active_vals.mean()) if len(active_vals) > 0 else 0.0,
        "median_density": float(np.median(active_vals)) if len(active_vals) > 0 else 0.0,
        "total_cells": int(width * height),
        "active_cells": int(np.count_nonzero(active_mask)),
        "cell_size": float(cell_size),
        "crs": crs_str,
        "bbox": (global_min_x, global_min_y, global_max_x, global_max_y),
    }
    logger.info("Generated point density raster: %s (Mean: %.2f pts/m²)", output_path, stats["mean_density"])
    return stats


def generate_spacing_raster(
    density_raster_path: Path | str,
    output_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Derive nominal point spacing (m) from an existing point density raster.
    Nominal Point Spacing s = 1.0 / sqrt(density) for density > 0.
    """
    density_path = Path(density_raster_path)
    if not density_path.exists():
        raise FileNotFoundError(f"Density raster not found: {density_path}")

    with rasterio.open(density_path) as src:
        density = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    spacing = np.full_like(density, -9999.0, dtype=np.float32)
    valid_mask = (density > 0) & (density != nodata) if nodata is not None else density > 0
    spacing[valid_mask] = 1.0 / np.sqrt(density[valid_mask])

    if output_path is None:
        output_path = density_path.parent / density_path.name.replace("density", "spacing")
        if output_path == density_path:
            output_path = density_path.parent / f"{density_path.stem}_spacing.tif"
    else:
        output_path = Path(output_path)

    profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(spacing, 1)

    if max(src.width, src.height) >= 256:
        with rasterio.open(output_path, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    valid_vals = spacing[valid_mask]
    return {
        "output_path": str(output_path),
        "min_spacing": float(valid_vals.min()) if len(valid_vals) > 0 else 0.0,
        "max_spacing": float(valid_vals.max()) if len(valid_vals) > 0 else 0.0,
        "mean_spacing": float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0,
        "active_cells": int(np.count_nonzero(valid_mask)),
    }


def generate_strip_difference_raster(
    tile_manager: Any,
    database: Any,
    tile_ids: Optional[List[str]] = None,
    strip_a: Optional[int] = None,
    strip_b: Optional[int] = None,
    cell_size: float = 1.0,
    output_path: Optional[Path | str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Generate vertical strip height difference raster dZ = Z_A - Z_B.

    If strip_a and strip_b are None, compares adjacent/overlapping strips
    or computes standard deviation of Z across all strips in overlapping cells.
    """
    if database is None or tile_manager is None:
        raise ValueError("Both tile_manager and database must be provided.")

    all_tiles = database.get_all_tiles()
    tile_dict = {t["id"]: t for t in all_tiles}

    if tile_ids is None or len(tile_ids) == 0:
        target_tids = [t["id"] for t in all_tiles]
    else:
        target_tids = [tid for tid in tile_ids if tid in tile_dict]

    # Global bounding box from LAS headers or DB
    bboxes = _get_target_bboxes(tile_manager, database, target_tids)

    if not bboxes:
        raise ValueError("Tile bounding boxes are missing in the database.")

    global_min_x = np.floor(min(b[0] for b in bboxes) / cell_size) * cell_size
    global_max_x = np.ceil(max(b[2] for b in bboxes) / cell_size) * cell_size
    global_min_y = np.floor(min(b[1] for b in bboxes) / cell_size) * cell_size
    global_max_y = np.ceil(max(b[3] for b in bboxes) / cell_size) * cell_size

    width = int(round((global_max_x - global_min_x) / cell_size))
    height = int(round((global_max_y - global_min_y) / cell_size))

    logger.info("Allocating Strip Difference grid: %d x %d (cell size: %.2f m)", width, height, cell_size)

    # Accumulators for Strip A and Strip B
    sum_z_a = np.zeros((height, width), dtype=np.float64)
    cnt_z_a = np.zeros((height, width), dtype=np.int32)

    sum_z_b = np.zeros((height, width), dtype=np.float64)
    cnt_z_b = np.zeros((height, width), dtype=np.int32)

    total_tiles = len(target_tids)
    for idx, tid in enumerate(target_tids):
        if progress_callback:
            progress_callback(idx, total_tiles, f"Scanning tile {tid} for strip overlap…")

        data = tile_manager.load_tile_points_full(tid)
        if data is None or "x" not in data or "point_source_id" not in data:
            continue

        psid = data["point_source_id"]
        x = data["x"]
        y = data["y"]
        z = data["z"]

        unique_strips = np.unique(psid)
        if len(unique_strips) < 2 and (strip_a is None or strip_b is None):
            # No strip overlap in this tile and no explicit pair
            continue

        # Target strips
        s_a = strip_a if strip_a is not None else int(unique_strips[0])
        s_b = strip_b if strip_b is not None else int(unique_strips[1] if len(unique_strips) > 1 else unique_strips[0])

        mask_a = psid == s_a
        mask_b = psid == s_b

        for mask, sum_arr, cnt_arr in [(mask_a, sum_z_a, cnt_z_a), (mask_b, sum_z_b, cnt_z_b)]:
            if not np.any(mask):
                continue
            sub_x = x[mask]
            sub_y = y[mask]
            sub_z = z[mask]

            cols = np.floor((sub_x - global_min_x) / cell_size).astype(np.int64)
            rows = np.floor((global_max_y - sub_y) / cell_size).astype(np.int64)

            valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            if not np.any(valid):
                continue
            cols = cols[valid]
            rows = rows[valid]
            sub_z = sub_z[valid]

            flat_idx = rows * width + cols
            np.add.at(sum_arr.ravel(), flat_idx, sub_z)
            np.add.at(cnt_arr.ravel(), flat_idx, 1)

    # Compute difference where both strips exist
    overlap_mask = (cnt_z_a > 0) & (cnt_z_b > 0)
    nodata_val = -9999.0
    dz_grid = np.full((height, width), nodata_val, dtype=np.float32)

    if np.any(overlap_mask):
        mean_za = (sum_z_a[overlap_mask] / cnt_z_a[overlap_mask]).astype(np.float32)
        mean_zb = (sum_z_b[overlap_mask] / cnt_z_b[overlap_mask]).astype(np.float32)
        dz_grid[overlap_mask] = mean_za - mean_zb

    if output_path is None:
        qc_dir = Path(tile_manager._pm.project_root) / "qc" / "rasters"
        qc_dir.mkdir(parents=True, exist_ok=True)
        pair_str = f"strip_{strip_a}_vs_{strip_b}" if strip_a and strip_b else "strip_overlap_diff"
        output_path = qc_dir / f"{pair_str}_{cell_size:.1f}m.tif"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    crs_str = get_project_crs_string(database)
    transform = from_origin(global_min_x, global_max_y, cell_size, cell_size)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs_str,
        transform=transform,
        nodata=nodata_val,
        compress="lzw",
    ) as dst:
        dst.write(dz_grid, 1)

    if max(width, height) >= 256:
        with rasterio.open(output_path, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    overlap_vals = dz_grid[overlap_mask]
    stats = {
        "output_path": str(output_path),
        "overlap_cells": int(np.count_nonzero(overlap_mask)),
        "mean_dz": float(overlap_vals.mean()) if len(overlap_vals) > 0 else 0.0,
        "rmse_dz": float(np.sqrt(np.mean(overlap_vals**2))) if len(overlap_vals) > 0 else 0.0,
        "min_dz": float(overlap_vals.min()) if len(overlap_vals) > 0 else 0.0,
        "max_dz": float(overlap_vals.max()) if len(overlap_vals) > 0 else 0.0,
        "cell_size": float(cell_size),
        "crs": crs_str,
        "bbox": (global_min_x, global_min_y, global_max_x, global_max_y),
    }
    logger.info("Generated strip difference raster: %s (RMSE: %.3f m)", output_path, stats["rmse_dz"])
    return stats


def generate_tile_grid_vector(
    database: Any,
    output_path: Optional[Path | str] = None,
    grid_mode: str = "all_together",  # "all_together" (full contiguous grid) or "loaded_only"
    tile_size: Optional[float] = None,
    overlap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a GeoJSON vector polygon layer representing the project tile grid.

    Excludes NOISE pseudo-tiles and ensures all tiles have identical square dimensions
    and the designated overlap (e.g. 200m x 200m with 5m overlap -> 195m stride).

    Args:
        database: Project Database instance.
        output_path: Target GeoJSON file path.
        grid_mode: "all_together" generates the full contiguous survey grid (with loaded
                   and empty cells identified); "loaded_only" generates only the loaded tiles.
        tile_size: Nominal tile edge length in meters (default 200.0 m).
        overlap: Overlap between adjacent tiles in meters (default 5.0 m).
    """
    if database is None:
        raise ValueError("Database must be provided.")

    raw_tiles = database.get_all_tiles()
    if not raw_tiles:
        raise ValueError("No tiles found in project database.")

    # Exclude noise pseudo-tiles
    tiles = [
        t for t in raw_tiles
        if str(t.get("status", "")).upper() != "NOISE" and "_noise" not in str(t.get("id", "")).lower()
    ]
    if not tiles:
        tiles = raw_tiles

    # Collect valid bounding boxes
    valid_tiles_info = []
    for t in tiles:
        bbox = _get_tile_bbox(t)
        if bbox is not None:
            valid_tiles_info.append((t, bbox))

    if not valid_tiles_info:
        raise ValueError("No valid tile bounding boxes found in database.")

    # Determine tile size and overlap
    if tile_size is None or overlap is None:
        db_p = getattr(database, "db_path", getattr(database, "_db_path", None))
        if callable(db_p):
            db_p = db_p()
        proj_json = Path(db_p).parent / "project.json" if db_p else None
        p_size, p_overlap = None, None
        if proj_json and proj_json.exists():
            try:
                with open(proj_json, "r", encoding="utf-8") as f:
                    p_meta = json.load(f)
                    p_size = float(p_meta.get("tile_size_m", 0.0) or 0.0) or None
                    p_overlap = float(p_meta.get("tile_overlap_m", 0.0) or 0.0)
            except Exception:
                pass

        if p_size is None and valid_tiles_info:
            dxs = [b[2] - b[0] for _, b in valid_tiles_info if (b[2] - b[0]) > 0]
            dys = [b[3] - b[1] for _, b in valid_tiles_info if (b[3] - b[1]) > 0]
            if dxs and dys:
                p_size = float(np.median(dxs))
            else:
                p_size = 200.0

        if p_overlap is None:
            p_overlap = 0.0

        tile_size = tile_size or p_size or 200.0
        if overlap is None:
            overlap = p_overlap if p_overlap is not None else 5.0

    stride = tile_size - overlap
    if stride <= 0:
        stride = tile_size

    # Grid origin anchored to min_x, min_y across valid tiles
    min_x0 = min(b[0] for _, b in valid_tiles_info)
    min_y0 = min(b[1] for _, b in valid_tiles_info)

    # Index loaded tiles by integer (col_idx, row_idx)
    db_by_coord: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for t, b in valid_tiles_info:
        c = int(round((b[0] - min_x0) / stride))
        row = int(round((b[1] - min_y0) / stride))
        db_by_coord[(c, row)] = t

    max_c = max(c for c, _ in db_by_coord.keys())
    max_r = max(r for _, r in db_by_coord.keys())

    features = []

    if grid_mode == "all_together":
        # Full contiguous rectangular grid covering the whole project
        for c in range(max_c + 1):
            for r in range(max_r + 1):
                x0 = min_x0 + c * stride
                y0 = min_y0 + r * stride
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                t_info = db_by_coord.get((c, r))
                if t_info is not None:
                    tid = str(t_info.get("id"))
                    pts = int(t_info.get("point_count", 0))
                    status = str(t_info.get("status", "LOADED"))
                    has_data = True
                else:
                    tid = f"grid_c{c:02d}_r{r:02d}"
                    pts = 0
                    status = "EMPTY"
                    has_data = False

                coords = [
                    [
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                        [x0, y0],
                    ]
                ]
                features.append({
                    "type": "Feature",
                    "id": tid,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coords,
                    },
                    "properties": {
                        "tile_id": tid,
                        "point_count": pts,
                        "status": status,
                        "has_data": has_data,
                        "col": c,
                        "row": r,
                        "tile_size": float(tile_size),
                        "overlap": float(overlap),
                        "min_x": float(x0),
                        "max_x": float(x1),
                        "min_y": float(y0),
                        "max_y": float(y1),
                        "centroid_x": float(0.5 * (x0 + x1)),
                        "centroid_y": float(0.5 * (y0 + y1)),
                    },
                })
    else:
        # Loaded tiles only, but standardized to exact nominal tile_size and overlap
        for (c, r), t_info in sorted(db_by_coord.items()):
            x0 = min_x0 + c * stride
            y0 = min_y0 + r * stride
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            tid = str(t_info.get("id"))
            pts = int(t_info.get("point_count", 0))
            status = str(t_info.get("status", "LOADED"))

            coords = [
                [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                    [x0, y0],
                ]
            ]
            features.append({
                "type": "Feature",
                "id": tid,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords,
                },
                "properties": {
                    "tile_id": tid,
                    "point_count": pts,
                    "status": status,
                    "has_data": True,
                    "col": c,
                    "row": r,
                    "tile_size": float(tile_size),
                    "overlap": float(overlap),
                    "min_x": float(x0),
                    "max_x": float(x1),
                    "min_y": float(y0),
                    "max_y": float(y1),
                    "centroid_x": float(0.5 * (x0 + x1)),
                    "centroid_y": float(0.5 * (y0 + y1)),
                },
            })

    geojson_doc = {
        "type": "FeatureCollection",
        "features": features,
    }

    if output_path is None:
        db_p = getattr(database, "db_path", getattr(database, "_db_path", "project.db"))
        qc_dir = Path(db_p).parent / "qc" / "vectors"
        qc_dir.mkdir(parents=True, exist_ok=True)
        output_path = qc_dir / "tile_grid.geojson"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2)

    logger.info("Exported tile grid vector to %s (%d tiles)", output_path, len(features))
    return {
        "output_path": str(output_path),
        "tile_count": len(features),
        "features": features,
    }
