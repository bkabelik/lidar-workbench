"""
LiDAR Workbench — Export Manager.

Produces DTM and DSM rasters as ESRI ASCII Grid (.asc) from classified
LiDAR tiles, plus derived hillshade rasters.  Designed for seamless
multi-tile output: all tiles share a single master-grid origin so
adjacent .asc files align perfectly with no gaps or visible seams.

DTM: Triangulated-Irregular-Network (Delaunay) interpolation of ground
     points (ASPRS class 2 only), rasterised at user resolution.
     Empty cells (outside the convex hull) are filled via IDW fallback
     then nearest-neighbour as a last resort.

DSM: Highest-point-per-cell (max-Z) from a user-selected set of ASPRS
     classes, equivalent to PDAL ``writers.gdal`` binmode + max.

Hillshade: Standard illumination model (azimuth 315°, altitude 45°)
     computed from the DTM or DSM raster, written as GeoTIFF (.tif).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("lidar_workbench.export_manager")

# ── constants ───────────────────────────────────────────────────────────
NODATA_VALUE: float = -9999.0
DEFAULT_HILLSHADE_AZIMUTH: float = 315.0   # degrees (NW light)
DEFAULT_HILLSHADE_ALTITUDE: float = 45.0   # degrees above horizon


# ── data structures ─────────────────────────────────────────────────────

@dataclass
class ExportConfig:
    """Parameters for a DTM or DSM export run."""

    # What to export
    mode: str = "dtm"               # "dtm" | "dsm"
    resolution: float = 1.0         # cell size in CRS units (metres)

    # For DTM
    ground_class: int = 2

    # For DSM — which ASPRS classes to include (empty = all)
    dsm_classes: Set[int] = field(default_factory=lambda: {2, 3, 4, 5, 6})

    # Tiling
    tile_ids: List[str] = field(default_factory=list)

    # Output
    output_dir: str = ""
    compute_hillshade: bool = True

    # Hillshade parameters
    hillshade_azimuth: float = DEFAULT_HILLSHADE_AZIMUTH
    hillshade_altitude: float = DEFAULT_HILLSHADE_ALTITUDE


@dataclass
class RasterGrid:
    """A regular raster grid ready for writing."""

    data: np.ndarray          # 2-D (rows, cols) — row 0 = northernmost
    xllcorner: float          # lower-left X of lower-left cell
    yllcorner: float          # lower-left Y of lower-left cell
    cellsize: float           # cell edge length
    nodata: float = NODATA_VALUE

    @property
    def nrows(self) -> int:
        return self.data.shape[0]

    @property
    def ncols(self) -> int:
        return self.data.shape[1]


# ── public API ──────────────────────────────────────────────────────────

def export_dtm(
    tile_points: Dict[str, Dict[str, np.ndarray]],
    tile_bboxes: Dict[str, Tuple[float, float, float, float]],
    config: ExportConfig,
    progress_callback: Optional[callable] = None,
) -> List[Path]:
    """
    Export a DTM as one ESRI ASCII Grid per tile (+ optional hillshade).

    Uses Delaunay triangulation (TIN) of ground points with IDW fallback
    for cells outside the convex hull.

    Args:
        tile_points: ``{tile_id: {"x":..., "y":..., "z":..., "classification":...}}``.
        tile_bboxes: ``{tile_id: (min_x, min_y, max_x, max_y)}``.
        config:       :class:`ExportConfig` with resolution, output_dir, etc.
        progress_callback: Optional ``callable(pct: float, msg: str)``.

    Returns:
        List of written file paths.
    """
    _validate_config(config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. compute master grid (seamless tiling) ──────────────────────
    global_bbox = _global_bbox(tile_bboxes)
    master = _master_grid(global_bbox, config.resolution)

    written: List[Path] = []
    total = len(config.tile_ids)

    for idx, tile_id in enumerate(config.tile_ids):
        pts = tile_points.get(tile_id)
        bbox = tile_bboxes.get(tile_id)
        if pts is None or bbox is None:
            logger.warning("Skipping tile %s — no data loaded", tile_id)
            continue

        if progress_callback:
            progress_callback((idx / total) * 90.0, f"DTM tile {tile_id}…")

        # ── 2. filter ground points ──────────────────────────────────
        cls = pts["classification"]
        ground_mask = cls == config.ground_class
        if not ground_mask.any():
            logger.warning("Tile %s: no ground points — skipping DTM", tile_id)
            continue

        gx = pts["x"][ground_mask]
        gy = pts["y"][ground_mask]
        gz = pts["z"][ground_mask]

        # ── 3. slice master grid to tile extent ──────────────────────
        sub = _slice_grid(master, bbox)

        # ── 4. TIN interpolation ─────────────────────────────────────
        sub.data = _tin_interpolate(gx, gy, gz, sub)

        # ── 5. write ─────────────────────────────────────────────────
        stem = tile_id if tile_id.startswith("tile_") else f"tile_{tile_id}"
        asc_path = out / f"{stem}_dtm.asc"
        _write_ascii_grid(sub, asc_path)
        written.append(asc_path)

        if config.compute_hillshade:
            hs_path = out / f"{stem}_dtm_hillshade.tif"
            _write_hillshade(sub, hs_path, config.hillshade_azimuth, config.hillshade_altitude)
            written.append(hs_path)

    if progress_callback:
        progress_callback(100.0, f"DTM export complete — {len(written)} file(s)")

    return written


def export_dsm(
    tile_points: Dict[str, Dict[str, np.ndarray]],
    tile_bboxes: Dict[str, Tuple[float, float, float, float]],
    config: ExportConfig,
    progress_callback: Optional[callable] = None,
) -> List[Path]:
    """
    Export a DSM as one ESRI ASCII Grid per tile (+ optional hillshade).

    Each cell gets the *maximum* Z of all points from the selected
    classes that fall inside it (PDAL binmode / max equivalent).

    Args:
        tile_points: ``{tile_id: {"x":..., "y":..., "z":..., "classification":...}}``.
        tile_bboxes: ``{tile_id: (min_x, min_y, max_x, max_y)}``.
        config:       :class:`ExportConfig` with resolution, class set, etc.
        progress_callback: Optional ``callable(pct: float, msg: str)``.

    Returns:
        List of written file paths.
    """
    _validate_config(config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_bbox = _global_bbox(tile_bboxes)
    master = _master_grid(global_bbox, config.resolution)

    written: List[Path] = []
    total = len(config.tile_ids)

    for idx, tile_id in enumerate(config.tile_ids):
        pts = tile_points.get(tile_id)
        bbox = tile_bboxes.get(tile_id)
        if pts is None or bbox is None:
            continue

        if progress_callback:
            progress_callback((idx / total) * 90.0, f"DSM tile {tile_id}…")

        # ── filter to selected classes ───────────────────────────────
        cls = pts["classification"]
        keep = np.isin(cls, list(config.dsm_classes))
        if not keep.any():
            logger.warning("Tile %s: no points in selected classes — skipping DSM", tile_id)
            continue

        px = pts["x"][keep]
        py = pts["y"][keep]
        pz = pts["z"][keep]

        # ── slice master grid ────────────────────────────────────────
        sub = _slice_grid(master, bbox)

        # ── bin max-Z ────────────────────────────────────────────────
        sub.data = _bin_max(px, py, pz, sub)

        # ── write ────────────────────────────────────────────────────
        stem = tile_id if tile_id.startswith("tile_") else f"tile_{tile_id}"
        asc_path = out / f"{stem}_dsm.asc"
        _write_ascii_grid(sub, asc_path)
        written.append(asc_path)

        if config.compute_hillshade:
            hs_path = out / f"{stem}_dsm_hillshade.tif"
            _write_hillshade(sub, hs_path, config.hillshade_azimuth, config.hillshade_altitude)
            written.append(hs_path)

    if progress_callback:
        progress_callback(100.0, f"DSM export complete — {len(written)} file(s)")

    return written


def export_merged_raster(
    tile_points: Dict[str, Dict[str, np.ndarray]],
    tile_bboxes: Dict[str, Tuple[float, float, float, float]],
    config: ExportConfig,
    progress_callback: Optional[callable] = None,
) -> List[Path]:
    """
    Export a single merged raster (not per-tile) covering all tiles.

    Useful when the user wants one continuous .asc for the entire project.
    Otherwise identical to the per-tile exports.
    """
    _validate_config(config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_bbox = _global_bbox(tile_bboxes)
    master = _master_grid(global_bbox, config.resolution)

    # Collect all points from all tiles
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_z: List[np.ndarray] = []
    all_cls: List[np.ndarray] = []

    for tile_id in config.tile_ids:
        pts = tile_points.get(tile_id)
        if pts is None:
            continue
        all_x.append(pts["x"])
        all_y.append(pts["y"])
        all_z.append(pts["z"])
        all_cls.append(pts["classification"])

    if not all_x:
        logger.warning("No point data to export")
        return []

    xs = np.concatenate(all_x)
    ys = np.concatenate(all_y)
    zs = np.concatenate(all_z)
    cls = np.concatenate(all_cls)

    if progress_callback:
        progress_callback(10.0, f"Loaded {len(xs):,} points for merged {config.mode}…")

    written: List[Path] = []

    if config.mode == "dtm":
        ground_mask = cls == config.ground_class
        if not ground_mask.any():
            logger.warning("No ground points for merged DTM")
            return []
        master.data = _tin_interpolate(xs[ground_mask], ys[ground_mask], zs[ground_mask], master)
        asc_path = out / "merged_dtm.asc"
        _write_ascii_grid(master, asc_path)
        written.append(asc_path)

        if config.compute_hillshade:
            hs_path = out / "merged_dtm_hillshade.tif"
            _write_hillshade(master, hs_path, config.hillshade_azimuth, config.hillshade_altitude)
            written.append(hs_path)

    elif config.mode == "dsm":
        keep = np.isin(cls, list(config.dsm_classes))
        if not keep.any():
            logger.warning("No points in selected classes for merged DSM")
            return []
        master.data = _bin_max(xs[keep], ys[keep], zs[keep], master)
        asc_path = out / "merged_dsm.asc"
        _write_ascii_grid(master, asc_path)
        written.append(asc_path)

        if config.compute_hillshade:
            hs_path = out / "merged_dsm_hillshade.tif"
            _write_hillshade(master, hs_path, config.hillshade_azimuth, config.hillshade_altitude)
            written.append(hs_path)

    if progress_callback:
        progress_callback(100.0, f"Merged {config.mode.upper()} export complete")

    return written


# ── interpolation methods ───────────────────────────────────────────────

def _tin_interpolate(
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    grid: RasterGrid,
) -> np.ndarray:
    """
    Delaunay-triangulation (TIN) interpolation of scattered ground points
    onto *grid*.  Cells outside the convex hull are filled with IDW,
    then nearest-neighbour as a last resort.

    Returns a 2-D (rows, cols) array matching *grid* dimensions.
    """
    from scipy.spatial import Delaunay

    nrows, ncols = grid.nrows, grid.ncols
    result = np.full((nrows, ncols), grid.nodata, dtype=np.float64)

    # Cell-centre coordinates
    cx = grid.xllcorner + (np.arange(ncols) + 0.5) * grid.cellsize
    cy = grid.yllcorner + (np.arange(nrows) + 0.5) * grid.cellsize
    ccx, ccy = np.meshgrid(cx, cy)   # (rows, cols)

    # Delaunay triangulation of ground points
    tri = Delaunay(np.column_stack([gx, gy]))

    # Find which triangle each cell centre falls in
    simplex = tri.find_simplex(np.column_stack([ccx.ravel(), ccy.ravel()]))
    simplex = simplex.reshape(nrows, ncols)

    inside = simplex >= 0
    if inside.any():
        # Barycentric interpolation
        # For each cell inside the convex hull, compute barycentric coords
        flat_idx = np.flatnonzero(inside.ravel())
        s = simplex.ravel()[flat_idx]
        # Transform points to barycentric coords
        transform = tri.transform[s]  # (n_pts, 3, 2) — affine to barycentric
        offsets = tri.transform[s, 2]  # (n_pts, 2) — barycentric → cartesian offset
        # Actually transform gives us: [b1, b2] = T @ (x - o)
        # where T is (2,2) and o is origin of the simplex
        pts_2d = np.column_stack([ccx.ravel()[flat_idx], ccy.ravel()[flat_idx]])
        origins = tri.transform[s, 2]
        b = np.einsum("nij,nj->ni", tri.transform[s, :2, :2], pts_2d - origins)
        b1, b2 = b[:, 0], b[:, 1]
        b3 = 1.0 - b1 - b2

        # Vertex indices for each simplex
        verts = tri.simplices[s]  # (n_pts, 3)
        z_verts = gz[verts]       # (n_pts, 3)

        interp_z = b1 * z_verts[:, 0] + b2 * z_verts[:, 1] + b3 * z_verts[:, 2]
        result.ravel()[flat_idx] = interp_z

    # ── IDW fallback for cells outside convex hull ─────────────────
    outside = result == grid.nodata
    if outside.any() and len(gx) > 0:
        result = _idw_fill(result, outside, gx, gy, gz, grid, power=2.0, max_dist=grid.cellsize * 5)

    # ── nearest-neighbour fallback ──────────────────────────────────
    still_nodata = result == grid.nodata
    if still_nodata.any() and len(gx) > 0:
        from scipy.spatial import KDTree
        tree = KDTree(np.column_stack([gx, gy]))
        ox, oy = np.where(still_nodata)
        query_x = grid.xllcorner + (oy + 0.5) * grid.cellsize
        query_y = grid.yllcorner + (ox + 0.5) * grid.cellsize
        _, nn = tree.query(np.column_stack([query_x, query_y]))
        result[still_nodata] = gz[nn]
        logger.debug("Filled %d cells with nearest-neighbour fallback", len(nn))

    return result


def _idw_fill(
    raster: np.ndarray,
    mask: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    grid: RasterGrid,
    power: float = 2.0,
    max_dist: float = 5.0,
) -> np.ndarray:
    """
    Inverse-distance-weighted fill for cells in *mask* using nearby
    ground points within *max_dist*.
    """
    from scipy.spatial import KDTree

    tree = KDTree(np.column_stack([gx, gy]))
    ox, oy = np.where(mask)
    qx = grid.xllcorner + (oy + 0.5) * grid.cellsize
    qy = grid.yllcorner + (ox + 0.5) * grid.cellsize

    # Query up to 12 neighbours
    distances, indices = tree.query(np.column_stack([qx, qy]), k=min(12, len(gx)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Mask far neighbours
    w = np.where(distances < max_dist, 1.0 / (distances ** power + 1e-12), 0.0)
    denom = w.sum(axis=1)
    valid = denom > 0
    num = (w * gz[indices]).sum(axis=1)
    raster[ox[valid], oy[valid]] = num[valid] / denom[valid]

    return raster


def _bin_max(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    grid: RasterGrid,
) -> np.ndarray:
    """
    Bin points into *grid*, assigning each cell the maximum Z of points
    that fall inside it (highest-hit DSM).
    """
    nrows, ncols = grid.nrows, grid.ncols
    result = np.full((nrows, ncols), grid.nodata, dtype=np.float64)

    # Pixel indices for each point
    col = np.floor((px - grid.xllcorner) / grid.cellsize).astype(np.int32)
    row = np.floor((py - grid.yllcorner) / grid.cellsize).astype(np.int32)

    valid = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    col = col[valid]
    row = row[valid]
    z = pz[valid]

    # Use np.maximum.at for efficient binning
    flat = row * ncols + col
    np.maximum.at(result.ravel(), flat, z)

    # Fill empty cells via IDW from nearby points
    empty = result == grid.nodata
    if empty.any() and len(z) > 0:
        # Build a sparse set of representatives (one per non-empty cell)
        unique_flat, unique_idx = np.unique(flat, return_index=True)
        rep_x = px[valid][unique_idx]
        rep_y = py[valid][unique_idx]
        rep_z = z[unique_idx]
        result = _idw_fill(result, empty, rep_x, rep_y, rep_z, grid,
                           power=2.0, max_dist=grid.cellsize * 3)

    return result


# ── hillshade ────────────────────────────────────────────────────────────

def compute_hillshade(
    grid: RasterGrid,
    azimuth: float = DEFAULT_HILLSHADE_AZIMUTH,
    altitude: float = DEFAULT_HILLSHADE_ALTITUDE,
) -> RasterGrid:
    """
    Compute a hillshade raster from *grid* using the standard
    illumination model (Horn, 1981).

    Args:
        grid:      Input elevation raster.
        azimuth:   Light azimuth in degrees (0 = north, clockwise).
        altitude:  Light altitude above horizon in degrees.

    Returns:
        A new :class:`RasterGrid` with hillshade values (0–255).
    """
    az_rad = math.radians(360.0 - azimuth + 90.0)   # convert to math convention
    alt_rad = math.radians(altitude)

    data = grid.data.astype(np.float64)
    nodata_mask = data == grid.nodata
    # Temporarily replace nodata with mean to avoid edge artefacts
    valid = data[~nodata_mask]
    fill_val = float(valid.mean()) if len(valid) > 0 else 0.0
    data_filled = np.where(nodata_mask, fill_val, data)

    # Slope and aspect via central differences
    dzdx = np.zeros_like(data_filled)
    dzdy = np.zeros_like(data_filled)

    # Horn (1981) central-difference slope estimator
    # 3×3 window:  a b c    (row i-1)
    #              d e f    (row i)
    #              g h i    (row i+1)
    # dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8*cellsize)
    # dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8*cellsize)
    a = data_filled[:-2, :-2]
    b = data_filled[:-2, 1:-1]
    c = data_filled[:-2, 2:]
    d = data_filled[1:-1, :-2]
    f = data_filled[1:-1, 2:]
    g = data_filled[2:, :-2]
    h = data_filled[2:, 1:-1]
    i = data_filled[2:, 2:]

    dzdx[1:-1, 1:-1] = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * grid.cellsize)
    dzdy[1:-1, 1:-1] = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * grid.cellsize)

    slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    aspect = np.arctan2(dzdy, -dzdx)
    # aspect = 0 where slope is 0 (flat)
    aspect = np.where(slope > 0, aspect, 0.0)

    hs = (np.cos(alt_rad) * np.cos(slope)
          + np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))

    # Clamp and scale to 0-255
    hs = np.clip(hs, 0.0, 1.0) * 254.0 + 1.0
    hs[nodata_mask] = grid.nodata

    return RasterGrid(
        data=hs,
        xllcorner=grid.xllcorner,
        yllcorner=grid.yllcorner,
        cellsize=grid.cellsize,
        nodata=grid.nodata,
    )


# ── grid helpers ────────────────────────────────────────────────────────

def _master_grid(
    bbox: Tuple[float, float, float, float],
    resolution: float,
) -> RasterGrid:
    """
    Create the master grid for seamless tiling.

    The origin is snapped to *resolution* so that every tile's grid
    cells align exactly.  ``xllcorner`` / ``yllcorner`` mark the lower-left
    corner of the lower-left cell (ESRI convention).
    """
    xmin, xmax, ymin, ymax = bbox

    # Snap origin to resolution
    xll = math.floor(xmin / resolution) * resolution
    yll = math.floor(ymin / resolution) * resolution

    ncols = max(1, int(math.ceil((xmax - xll) / resolution)))
    nrows = max(1, int(math.ceil((ymax - yll) / resolution)))

    return RasterGrid(
        data=np.empty((nrows, ncols), dtype=np.float64),
        xllcorner=xll,
        yllcorner=yll,
        cellsize=resolution,
        nodata=NODATA_VALUE,
    )


def _slice_grid(
    master: RasterGrid,
    bbox: Tuple[float, float, float, float],
) -> RasterGrid:
    """
    Return the subset of *master* that covers *bbox*.

    The returned grid shares the same origin and resolution as the
    master, guaranteeing alignment.  Even a single-pixel-tall grid
    works correctly.
    """
    xmin, _, ymin, ymax = bbox

    # Col / row range (integer indices into the master)
    col_start = int(round((bbox[0] - master.xllcorner) / master.cellsize))
    col_end = int(round((bbox[2] - master.xllcorner) / master.cellsize))
    row_start = int(round((bbox[1] - master.yllcorner) / master.cellsize))
    row_end = int(round((bbox[3] - master.yllcorner) / master.cellsize))

    # Clamp
    col_start = max(0, col_start)
    col_end = min(master.ncols, max(col_start + 1, col_end))
    row_start = max(0, row_start)
    row_end = min(master.nrows, max(row_start + 1, row_end))

    sub_data = master.data[row_start:row_end, col_start:col_end].copy()

    # xllcorner / yllcorner of the slice
    sub_xll = master.xllcorner + col_start * master.cellsize
    sub_yll = master.yllcorner + row_start * master.cellsize

    return RasterGrid(
        data=sub_data,
        xllcorner=sub_xll,
        yllcorner=sub_yll,
        cellsize=master.cellsize,
        nodata=master.nodata,
    )


def _global_bbox(
    tile_bboxes: Dict[str, Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """Compute the overall bounding box of all tiles."""
    xs_min, ys_min, xs_max, ys_max = [], [], [], []
    for b in tile_bboxes.values():
        xs_min.append(b[0])
        ys_min.append(b[1])
        xs_max.append(b[2])
        ys_max.append(b[3])
    return (min(xs_min), max(xs_max), min(ys_min), max(ys_max))


# ── ASCII Grid I/O ──────────────────────────────────────────────────────

def _write_ascii_grid(grid: RasterGrid, path: Path) -> None:
    """
    Write *grid* as an ESRI ASCII Grid file.

    Format example::

        ncols         480
        nrows         450
        xllcorner     378923.0
        yllcorner     4072345.0
        cellsize      0.5
        NODATA_value  -9999.0
        326.5 327.0 327.5 ...
    """
    lines: List[str] = []
    lines.append(f"ncols         {grid.ncols}")
    lines.append(f"nrows         {grid.nrows}")
    lines.append(f"xllcorner     {grid.xllcorner:.6f}")
    lines.append(f"yllcorner     {grid.yllcorner:.6f}")
    lines.append(f"cellsize      {grid.cellsize:.6f}")
    lines.append(f"NODATA_value  {grid.nodata:.1f}")

    # Data: one row per line, space-separated, %.3f precision
    with np.printoptions(threshold=np.inf, linewidth=np.inf, suppress=True,
                         formatter={"float_kind": lambda v: f"{v:.3f}"}):
        for row in range(grid.nrows):
            lines.append(" ".join(f"{v:.3f}" for v in grid.data[row, :]))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d × %d)", path, grid.ncols, grid.nrows)


def _write_hillshade(
    grid: RasterGrid,
    path: Path,
    azimuth: float,
    altitude: float,
) -> None:
    """Compute hillshade and write as GeoTIFF (.tif)."""
    hs = compute_hillshade(grid, azimuth, altitude)
    _write_geotiff(hs, path)
    logger.info("Wrote hillshade %s", path)


def _write_geotiff(grid: RasterGrid, path: Path) -> None:
    """
    Write *grid* as a georeferenced GeoTIFF (.tif).

    Produces a valid GeoTIFF with ModelTiepointTag and ModelPixelScaleTag
    so that QGIS, ArcGIS, GDAL, and Global Mapper can place the raster
    correctly.  Pure Python — no external library required.
    """
    import struct

    # Convert hillshade float → uint8: nodata → 0, valid → 1..255
    data = grid.data.copy()
    nodata_mask = data == grid.nodata
    valid_mask = ~nodata_mask
    data[valid_mask] = np.clip(np.round(data[valid_mask]), 0, 255)
    data[nodata_mask] = 0
    img = data.astype(np.uint8)
    nrows, ncols = img.shape
    img_bytes = img.tobytes()

    # ── Build IFD entry descriptors ─────────────────────────────────
    # Each entry: (tag, type, count, value)
    #   type 1=BYTE, 2=ASCII, 3=SHORT, 4=LONG, 5=RATIONAL, 12=DOUBLE
    #   value is either an int, a list, a tuple, or a string.
    entries = [
        (256,  3, 1, ncols),                                        # ImageWidth
        (257,  3, 1, nrows),                                        # ImageLength
        (258,  3, 1, 8),                                            # BitsPerSample
        (259,  3, 1, 1),                                            # Compression (none)
        (262,  3, 1, 1),                                            # PhotometricInterpretation (BlackIsZero)
        (273,  4, 1, 0),                                            # StripOffsets (patched later)
        (277,  3, 1, 1),                                            # SamplesPerPixel
        (278,  4, 1, nrows),                                        # RowsPerStrip
        (279,  4, 1, len(img_bytes)),                               # StripByteCounts
        (282,  5, 1, [(1, 1)]),                                     # XResolution
        (283,  5, 1, [(1, 1)]),                                     # YResolution
        (296,  3, 1, 1),                                            # ResolutionUnit (none)
        (339,  3, 1, 1),                                            # SampleFormat (uint)
        (33550, 12, 3, [grid.cellsize, -grid.cellsize, 0.0]),        # ModelPixelScaleTag (Y negative: row↓ = Y↓)
        (33922, 12, 6, [0.0, 0.0, 0.0,
                         grid.xllcorner,
                         grid.yllcorner + grid.nrows * grid.cellsize,
                         0.0]),  # ModelTiepointTag (I,J,K → X,Y,Z): pixel (0,0) = top-left corner
        (42113, 2, 2, "0"),                                         # GDAL_NODATA
    ]

    # ── Encode entries → raw bytes + overflow ──────────────────────
    ifd_entries_raw = bytearray()
    overflow = bytearray()      # values > 4 bytes stored here
    over_pos = 0                # current offset within overflow area

    # Layout (known ahead):
    #   header:    8 bytes  (offset 0)
    #   IFD:       2 + N*12 + 4 bytes  (offset 8)
    #   overflow:  starts at 8 + 2 + N*12 + 4
    #   image:     starts after overflow, word-aligned
    ifd_start = 8
    num_entries = len(entries)
    ifd_body_size = 2 + num_entries * 12 + 4
    overflow_start = ifd_start + ifd_body_size

    for tag, typ, count, value in entries:
        # Encode the 4-byte value/offset slot
        if typ == 3:  # SHORT
            if count == 1:
                slot = struct.pack("<HH", value, 0)
            elif count == 2:
                slot = struct.pack("<HH", value[0], value[1])
            else:
                slot = struct.pack("<I", overflow_start + over_pos)
                overflow.extend(struct.pack(f"<{count}H", *value))
                over_pos += count * 2
        elif typ == 4:  # LONG
            if count == 1:
                slot = struct.pack("<I", value)
            else:
                slot = struct.pack("<I", overflow_start + over_pos)
                overflow.extend(struct.pack(f"<{count}I", *value))
                over_pos += count * 4
        elif typ == 5:  # RATIONAL (pairs of LONG)
            slot = struct.pack("<I", overflow_start + over_pos)
            for n, d in value:
                overflow.extend(struct.pack("<II", n, d))
                over_pos += 8
        elif typ == 12:  # DOUBLE
            slot = struct.pack("<I", overflow_start + over_pos)
            for v in value:
                overflow.extend(struct.pack("<d", v))
                over_pos += 8
        elif typ == 2:  # ASCII
            s = value.encode("ascii") + b"\x00"
            if len(s) <= 4:
                slot = s.ljust(4, b"\x00")
            else:
                slot = struct.pack("<I", overflow_start + over_pos)
                overflow.extend(s)
                over_pos += len(s)
        else:
            slot = b"\x00\x00\x00\x00"

        ifd_entries_raw.extend(struct.pack("<HHI", tag, typ, count) + slot)

    # ── Compute image offset (word-aligned) ────────────────────────
    img_offset = overflow_start + len(overflow)
    if img_offset % 2 != 0:
        img_offset += 1

    # ── Patch StripOffsets (tag 273) ───────────────────────────────
    # Find the entry and replace the 4-byte value at offset +8 within the entry
    for i in range(num_entries):
        e_off = i * 12
        raw_tag = struct.unpack_from("<H", ifd_entries_raw, e_off)[0]
        if raw_tag == 273:
            struct.pack_into("<I", ifd_entries_raw, e_off + 8, img_offset)
            break

    # ── Assemble and write ─────────────────────────────────────────
    # Header: "II" (little-endian) + magic 42 + offset to first IFD
    header = struct.pack("<HHI", 0x4949, 42, ifd_start)

    # IFD: count + entries + next_ifd_offset (0 = last)
    ifd = struct.pack("<H", num_entries) + bytes(ifd_entries_raw) + struct.pack("<I", 0)

    # Padding to word-align the image data
    pad_len = img_offset - (overflow_start + len(overflow))
    pad = b"\x00" * pad_len

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(ifd)
        fh.write(bytes(overflow))
        fh.write(pad)
        fh.write(img_bytes)

    logger.info("Wrote %s (%d × %d, %.1f KB)", path, ncols, nrows,
                os.path.getsize(path) / 1024)

def _validate_config(config: ExportConfig) -> None:
    """Raise ``ValueError`` if *config* is unusable."""
    if config.resolution <= 0:
        raise ValueError("resolution must be > 0")
    if not config.output_dir:
        raise ValueError("output_dir is required")
    if not config.tile_ids:
        raise ValueError("at least one tile_id is required")
    if config.mode not in ("dtm", "dsm"):
        raise ValueError(f"Unknown export mode: {config.mode}")
