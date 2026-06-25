"""
LiDAR Workbench — DTM Generator.

Interpolates a Digital Terrain Model (DTM) raster from ground-classified
LiDAR points using scipy griddata or GDAL grid methods.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("lidar_workbench.dtm_generator")

# Default DTM resolution in CRS units (meters)
DEFAULT_DTM_RESOLUTION: float = 1.0


def generate_dtm(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    classifications: np.ndarray,
    resolution: float = DEFAULT_DTM_RESOLUTION,
    ground_class: int = 2,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """
    Generate a DTM raster from ground-classified points.

    Args:
        xs, ys, zs:    Point coordinate arrays.
        classifications: ASPRS classification codes.
        resolution:    DTM cell size in CRS units (meters).
        ground_class:  ASPRS code for Ground (default: 2).
        method:        Interpolation method: ``"linear"``, ``"nearest"``,
                       or ``"cubic"``.

    Returns:
        ``(grid_x, grid_y, grid_z, bbox)`` where:
            - ``grid_x``: 1-D array of X coordinates for grid columns.
            - ``grid_y``: 1-D array of Y coordinates for grid rows.
            - ``grid_z``: 2-D array of interpolated elevation values.
            - ``bbox``:   ``(min_x, max_x, min_y, max_y)``.
    """
    # Filter ground points
    ground_mask = classifications == ground_class
    if not ground_mask.any():
        logger.warning("No ground points (class %d) found — using all points", ground_class)
        ground_mask = np.ones(len(xs), dtype=bool)

    gx = xs[ground_mask]
    gy = ys[ground_mask]
    gz = zs[ground_mask]

    # Subsample for speed — 50K points is plenty for a 1 m DTM
    MAX_GROUND_PTS = 50_000
    n_ground = len(gx)
    if n_ground > MAX_GROUND_PTS:
        idx = np.random.choice(n_ground, MAX_GROUND_PTS, replace=False)
        gx, gy, gz = gx[idx], gy[idx], gz[idx]

    logger.info("DTM: %d ground points for interpolation (from %d)", len(gx), n_ground)

    # Define grid extent
    x_min, x_max = gx.min(), gx.max()
    y_min, y_max = gy.min(), gy.max()

    # Pad slightly to avoid edge effects
    pad = resolution * 2
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    # Create regular grid
    nx = max(2, int((x_max - x_min) / resolution) + 1)
    ny = max(2, int((y_max - y_min) / resolution) + 1)

    grid_x = np.linspace(x_min, x_max, nx)
    grid_y = np.linspace(y_min, y_max, ny)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # Interpolate
    try:
        from scipy.interpolate import griddata
        grid_z = griddata(
            (gx, gy), gz, (grid_xx, grid_yy),
            method=method,
        )
        # Fill NaN values (outside convex hull) with nearest-neighbour
        if np.any(np.isnan(grid_z)):
            nan_mask = np.isnan(grid_z)
            grid_z_nn = griddata(
                (gx, gy), gz, (grid_xx, grid_yy),
                method="nearest",
            )
            grid_z[nan_mask] = grid_z_nn[nan_mask]
            logger.debug(
                "Filled %d NaN DTM cells with nearest-neighbour interpolation",
                nan_mask.sum(),
            )
    except ImportError:
        logger.warning("scipy not available — DTM will be empty")
        grid_z = np.full((ny, nx), np.nan)

    bbox = (x_min, x_max, y_min, y_max)

    return grid_x, grid_y, grid_z, bbox


def get_dtm_elevation_at(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    px: float,
    py: float,
) -> float:
    """
    Bilinear interpolation of DTM elevation at a point.

    Args:
        grid_x, grid_y, grid_z: As returned by :func:`generate_dtm`.
        px, py:                Query point coordinates.

    Returns:
        Interpolated elevation, or ``NaN`` if the point is outside the grid.
    """
    if px < grid_x[0] or px > grid_x[-1] or py < grid_y[0] or py > grid_y[-1]:
        return float("nan")

    # Find grid cell
    ix = np.searchsorted(grid_x, px) - 1
    iy = np.searchsorted(grid_y, py) - 1
    ix = max(0, min(ix, len(grid_x) - 2))
    iy = max(0, min(iy, len(grid_y) - 2))

    # Bilinear weights
    x0, x1 = grid_x[ix], grid_x[ix + 1]
    y0, y1 = grid_y[iy], grid_y[iy + 1]

    wx = (px - x0) / (x1 - x0) if x1 != x0 else 0.0
    wy = (py - y0) / (y1 - y0) if y1 != y0 else 0.0

    z00 = grid_z[iy, ix]
    z10 = grid_z[iy, ix + 1]
    z01 = grid_z[iy + 1, ix]
    z11 = grid_z[iy + 1, ix + 1]

    return (
        (1 - wx) * (1 - wy) * z00
        + wx * (1 - wy) * z10
        + (1 - wx) * wy * z01
        + wx * wy * z11
    )


def extract_dtm_profile(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    num_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract an elevation profile along a 2D line from the DTM.

    Args:
        grid_x, grid_y, grid_z: DTM grid data.
        start_xy:               Profile start ``(x, y)``.
        end_xy:                 Profile end ``(x, y)``.
        num_samples:            Number of sample points along the line.

    Returns:
        ``(distances, elevations)`` — 1-D arrays of the same length.
        *distances* are measured from the start point in CRS units.
    """
    sx, sy = start_xy
    ex, ey = end_xy

    t = np.linspace(0, 1, num_samples)
    px = sx + t * (ex - sx)
    py = sy + t * (ey - sy)

    elevations = np.array([
        get_dtm_elevation_at(grid_x, grid_y, grid_z, xi, yi)
        for xi, yi in zip(px, py)
    ])

    # Distances along the profile
    dx = px - sx
    dy = py - sy
    distances = np.sqrt(dx * dx + dy * dy)

    return distances, elevations
