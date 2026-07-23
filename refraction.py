"""
LiDAR Workbench — Refraction Correction Module.

Implements Snell's law refraction correction for bathymetric LiDAR and
a configurable ASCII trajectory file reader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import rowcol
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None

logger = logging.getLogger("lidar_workbench.refraction")

# Refractive index of water relative to air
N_WATER_DEFAULT: float = 1.3333


# ── Trajectory file reader ──────────────────────────────────────────

def read_trajectory_ascii(
    path: str | Path,
    columns: Optional[list[int]] = None,
    separator: Optional[str] = None,
    skip_rows: int = 0,
    has_header: bool = False,
    time_col: int = 0,
    x_col: int = 1,
    y_col: int = 2,
    z_col: int = 3,
) -> np.ndarray:
    """
    Read a sensor trajectory from an ASCII file with configurable columns.

    The file is expected to contain at least time, X, Y, Z columns.
    Returns an (N, 4) array of ``[gps_time, x, y, z]`` sorted by time.

    Args:
        path:        Path to the trajectory file.
        columns:     Explicit column indices to keep ``[time, x, y, z]``.
                     When provided, overrides *time_col*, *x_col*, etc.
        separator:   Column delimiter.  Auto-detected (comma, whitespace,
                     tab, semicolon) when ``None``.
        skip_rows:   Rows to skip at the start of the file.
        has_header:  If ``True``, the first non-skipped row is treated as
                     a header and discarded after column detection.
        time_col:    Column index for GPS time (default 0).
        x_col:       Column index for X / easting (default 1).
        y_col:       Column index for Y / northing (default 2).
        z_col:       Column index for Z / altitude (default 3).

    Returns:
        ``(N, 4)`` float64 array ``[time, x, y, z]`` sorted by time.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the file is empty or columns are invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    # Read raw text
    with open(path, "r") as f:
        lines = f.readlines()

    # Skip rows
    if skip_rows > 0:
        lines = lines[skip_rows:]

    if has_header and lines:
        lines = lines[1:]

    if not lines:
        raise ValueError(f"Trajectory file is empty after skipping: {path}")

    # Auto-detect separator from first data line
    if separator is None:
        first = lines[0].strip()
        for sep in [",", ";", "\t", r"\s+"]:
            if sep == r"\s+":
                parts = first.split()
            else:
                parts = first.split(sep)
            if len(parts) >= 4:
                separator = sep
                break
        if separator is None:
            separator = ","  # fallback

    # Parse
    if columns is not None:
        tc, xc, yc, zc = columns
    else:
        tc, xc, yc, zc = time_col, x_col, y_col, z_col

    max_col = max(tc, xc, yc, zc)
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if separator == r"\s+":
            parts = line.split()
        else:
            parts = line.split(separator)
        if len(parts) <= max_col:
            continue
        try:
            t = float(parts[tc])
            x = float(parts[xc])
            y = float(parts[yc])
            z = float(parts[zc])
            data.append([t, x, y, z])
        except (ValueError, IndexError):
            continue

    if not data:
        raise ValueError(f"No valid trajectory records found in: {path}")

    arr = np.array(data, dtype=np.float64)
    # Sort by time
    arr = arr[arr[:, 0].argsort()]
    logger.info("Loaded %d trajectory records from %s", len(arr), path)
    return arr


def interpolate_sensor_position(
    trajectory: np.ndarray,
    gps_time: float,
) -> np.ndarray:
    """
    Interpolate the sensor position at a given GPS time from a trajectory.

    Args:
        trajectory: ``(N, 4)`` array ``[time, x, y, z]`` sorted by time.
        gps_time:   GPS time at which to query the position.

    Returns:
        ``(3,)`` array ``[x, y, z]``.
    """
    times = trajectory[:, 0]
    if gps_time <= times[0]:
        return trajectory[0, 1:4].copy()
    if gps_time >= times[-1]:
        return trajectory[-1, 1:4].copy()

    idx = np.searchsorted(times, gps_time)
    t0, t1 = times[idx - 1], times[idx]
    frac = (gps_time - t0) / (t1 - t0) if t1 > t0 else 0.0
    pos0 = trajectory[idx - 1, 1:4]
    pos1 = trajectory[idx, 1:4]
    return pos0 + frac * (pos1 - pos0)


# ── Snell's law refraction correction ───────────────────────────────

def apply_snells_correction(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    water_surface_z: float | np.ndarray,
    sensor_position: Tuple[float, float, float],
    n_water: float = N_WATER_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Snell's law refraction correction to submerged bathymetric points.

    The green laser beam bends at the air-water interface and slows down.
    This function corrects the apparent 3-D positions to their true
    geometric locations using a flat or spatially-varying water surface.

    For each submerged point the correction:
      1. Finds the ray intersection with the water surface plane.
      2. Computes the incident angle in air.
      3. Applies Snell's law to get the refracted angle in water.
      4. Adjusts the underwater path length by the refractive index ratio.

    Args:
        xs, ys, zs:       Point coordinates (apparent, uncorrected).
        water_surface_z:  Water surface elevation — either a scalar float
                          for a flat surface, or a per-point array for a
                          spatially-varying surface (e.g. from a GeoTIFF).
        sensor_position:  ``(X_s, Y_s, Z_s)`` of the scanner at pulse time.
        n_water:          Refractive index of water (default 1.3333).

    Returns:
        ``(corrected_xs, corrected_ys, corrected_zs)`` — arrays of the
        same length as the input.  Points above the water surface are
        returned unchanged.
    """
    n = len(xs)
    out_x = xs.copy()
    out_y = ys.copy()
    out_z = zs.copy()

    # Get per-point water surface Z
    if isinstance(water_surface_z, np.ndarray):
        ws_z = water_surface_z
    else:
        ws_z = np.full(n, water_surface_z, dtype=np.float64)

    # Only correct points below the water surface
    submerged = zs < ws_z
    if not submerged.any():
        return out_x, out_y, out_z

    sx, sy, sz = sensor_position

    sub_x = xs[submerged]
    sub_y = ys[submerged]
    sub_z = zs[submerged]
    sub_ws = ws_z[submerged]

    # Ray vector from sensor to apparent point
    dx = sub_x - sx
    dy = sub_y - sy
    dz = sub_z - sz

    # Intersection with water surface plane at Z = water_surface_z (per-point)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (sub_ws - sz) / dz
    t = np.clip(t, 0.0, 1.0)

    inter_x = sx + t * dx
    inter_y = sy + t * dy
    inter_z = sub_ws

    # Air path: sensor → surface intersection
    air_dx = inter_x - sx
    air_dy = inter_y - sy
    air_dz = inter_z - sz
    air_dist = np.sqrt(air_dx**2 + air_dy**2 + air_dz**2)

    # Apparent water path: surface → apparent bottom
    water_app_dx = sub_x - inter_x
    water_app_dy = sub_y - inter_y
    water_app_dz = sub_z - sub_ws  # negative (downward)
    water_app_dist = np.sqrt(water_app_dx**2 + water_app_dy**2 + water_app_dz**2)

    # Incident angle in air (relative to vertical/nadir)
    cos_theta_air = np.abs(air_dz) / np.maximum(air_dist, 1e-12)
    cos_theta_air = np.clip(cos_theta_air, 0.0, 1.0)
    sin_theta_air = np.sqrt(np.maximum(1.0 - cos_theta_air**2, 0.0))

    # Snell's law: sin(theta_water) = sin(theta_air) / n_water
    sin_theta_water = sin_theta_air / n_water
    sin_theta_water = np.clip(sin_theta_water, 0.0, 1.0)
    cos_theta_water = np.sqrt(np.maximum(1.0 - sin_theta_water**2, 0.0))

    # True underwater distance (speed slowdown correction)
    true_water_dist = water_app_dist / n_water

    # Horizontal components are scaled by sin(theta_water)/sin(theta_air)
    h_scale = np.divide(
        sin_theta_water, np.maximum(sin_theta_air, 1e-12),
        out=np.zeros_like(sin_theta_water),
        where=sin_theta_air > 1e-12,
    )

    # True underwater displacement
    true_dx = water_app_dx * h_scale * (true_water_dist / np.maximum(water_app_dist, 1e-12))
    true_dy = water_app_dy * h_scale * (true_water_dist / np.maximum(water_app_dist, 1e-12))

    # Downward component
    true_dz = -true_water_dist * cos_theta_water

    out_x[submerged] = inter_x + true_dx
    out_y[submerged] = inter_y + true_dy
    out_z[submerged] = sub_ws + true_dz

    n_corrected = submerged.sum()
    logger.info(
        "Snell's correction: %d/%d points corrected (n=%.4f)",
        n_corrected, n, n_water,
    )
    return out_x, out_y, out_z


def apply_snells_correction_nadir(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    water_surface_z: float,
    n_water: float = N_WATER_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified refraction correction assuming near-nadir scan angles.

    Only corrects the Z coordinate by scaling the apparent depth by the
    refractive index.  Horizontal coordinates are unchanged.  This is a
    good approximation for scan angles < 15°.

    Args:
        xs, ys, zs:       Point coordinates.
        water_surface_z:  Water surface elevation.
        n_water:          Refractive index of water.

    Returns:
        ``(xs, ys, corrected_zs)``.
    """
    out_z = zs.copy()
    submerged = zs < water_surface_z
    apparent_depth = water_surface_z - zs[submerged]
    true_depth = apparent_depth * n_water
    out_z[submerged] = water_surface_z - true_depth
    n_corrected = submerged.sum()
    logger.info(
        "Snell's correction (nadir): %d/%d points corrected",
        n_corrected, len(xs),
    )
    return xs.copy(), ys.copy(), out_z


def detect_water_surface_ransac(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    percentile: float = 90.0,
    residual_threshold: float = 0.3,
) -> Tuple[float, float, float, float]:
    """
    Detect a planar water surface using RANSAC on the highest points.

    Args:
        xs, ys, zs:          Point coordinates.
        percentile:          Percentile of Z used to select candidate
                             surface points (default 90 = top 10%).
        residual_threshold:  RANSAC inlier threshold (metres).

    Returns:
        ``(a, b, c, z_mean)`` — plane coefficients ``z = a*x + b*y + c``
        and the mean Z of the fitted surface.  Returns ``(0, 0, 0, 0)``
        if detection fails.
    """
    n = len(xs)
    if n < 50:
        return 0.0, 0.0, 0.0, 0.0

    z_cutoff = float(np.percentile(zs, percentile))
    mask = zs >= z_cutoff
    if mask.sum() < 20:
        return 0.0, 0.0, 0.0, 0.0

    from sklearn.linear_model import RANSACRegressor

    ransac = RANSACRegressor(
        residual_threshold=residual_threshold,
        max_trials=200,
        random_state=0,
    )
    try:
        ransac.fit(
            np.column_stack((xs[mask], ys[mask])),
            zs[mask],
        )
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    z_mean = float(np.median(zs[mask][ransac.inlier_mask_]))

    logger.info(
        "Water surface: z = %.6f*x + %.6f*y + %.6f  (mean_z=%.2f)",
        a, b, c, z_mean,
    )
    return float(a), float(b), float(c), z_mean


# ── GeoTIFF water surface loading ────────────────────────────────


def load_water_surface_geotiff(
    path: str | Path,
) -> Tuple[np.ndarray, dict]:
    """
    Load a float GeoTIFF representing a water surface model.

    The GeoTIFF should be a single-band float32 or float64 raster
    where pixel values represent water surface elevation in CRS units
    (typically metres).

    Args:
        path: Path to the GeoTIFF file.

    Returns:
        ``(surface_array, georef)`` where *surface_array* is a 2-D
        ``(rows, cols)`` numpy array of water surface elevations,
        and *georef* is a dict with keys:
        ``transform`` (rasterio Affine), ``crs``, ``shape``, ``bounds``,
        ``nodata``.

    Raises:
        ImportError: If rasterio is not installed.
        FileNotFoundError: If the file does not exist.
    """
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for GeoTIFF water surface loading. "
            "Install with: pip install rasterio"
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(str(path)) as ds:
        if ds.count < 1:
            raise ValueError(f"GeoTIFF has no bands: {path}")
        surface = ds.read(1).astype(np.float64)
        # Handle nodata
        nodata = ds.nodata
        if nodata is not None:
            surface = np.where(surface == nodata, np.nan, surface)
        georef = {
            "transform": ds.transform,
            "crs": str(ds.crs) if ds.crs else "",
            "shape": ds.shape,
            "bounds": ds.bounds,
            "nodata": nodata,
        }

    logger.info(
        "Loaded water surface GeoTIFF: %s, shape=%s, range=[%.2f, %.2f]",
        path.name, surface.shape,
        float(np.nanmin(surface)), float(np.nanmax(surface)),
    )
    return surface, georef


def sample_geotiff_at_points(
    surface: np.ndarray,
    georef: dict,
    xs: np.ndarray,
    ys: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Bilinear-sample a GeoTIFF raster at point locations.

    Args:
        surface:    2-D array of water surface elevations (rows, cols).
        georef:     Dict from :func:`load_water_surface_geotiff`.
        xs, ys:     Point coordinates in the GeoTIFF's CRS.
        fill_value: Value to use for points outside the raster.

    Returns:
        1-D array of sampled water surface elevations, same length as *xs*.
    """
    transform = georef["transform"]
    rows, cols = surface.shape
    n = len(xs)

    # Pixel coordinates
    inv_transform = ~transform
    col_f = inv_transform[0] * xs + inv_transform[1] * ys + inv_transform[2]
    row_f = inv_transform[3] * xs + inv_transform[4] * ys + inv_transform[5]

    result = np.full(n, fill_value, dtype=np.float64)

    for i in range(n):
        r, c = row_f[i], col_f[i]
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))

        if 0 <= r0 < rows - 1 and 0 <= c0 < cols - 1:
            wr = r - r0
            wc = c - c0
            v00 = surface[r0, c0]
            v10 = surface[r0 + 1, c0]
            v01 = surface[r0, c0 + 1]
            v11 = surface[r0 + 1, c0 + 1]

            # Only interpolate if all four corners are valid
            if not (np.isnan(v00) or np.isnan(v10) or np.isnan(v01) or np.isnan(v11)):
                result[i] = (
                    v00 * (1 - wr) * (1 - wc)
                    + v10 * wr * (1 - wc)
                    + v01 * (1 - wr) * wc
                    + v11 * wr * wc
                )
            elif not np.isnan(surface[r0, c0]):
                # Nearest-neighbour fallback
                result[i] = surface[r0, c0]

    n_valid = (~np.isnan(result) & (result != fill_value)).sum()
    logger.info(
        "Sampled water surface at %d points: %d in-raster (%d fell outside)",
        n, n_valid, n - n_valid,
    )
    return result


# ── Water surface model crop ──────────────────────────────────────


def crop_to_water_surface_model(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    surface: np.ndarray,
    georef: dict,
    tolerance_above_cm: float = 0.0,
    extrapolate_m: float = 0.0,
    progress: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop points to the water surface model extent.

    Points are kept if they:
      1. Fall within the water surface GeoTIFF valid-data region
         (optionally extrapolated outward by *extrapolate_m* metres), AND
      2. Are at or below the water surface elevation plus
         *tolerance_above_cm* (centimetres).

    This is used to remove bathymetric data points that lie outside
    the river corridor as defined by the water surface model.

    Args:
        xs, ys, zs:        Point coordinates.
        surface:           2-D array of water surface elevations (from
                           :func:`load_water_surface_geotiff`).
        georef:            Georeference dict from the same function.
        tolerance_above_cm: Allow points this many cm above the water
                           surface.  Default 0 = strictly at or below.
        extrapolate_m:     Expand the valid-data mask outward by this
                           many metres.  Useful as a safety margin.
        progress:          Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)`` — boolean arrays.
    """
    n = len(xs)
    if n == 0:
        empty = np.array([], dtype=bool)
        return empty, empty

    transform = georef["transform"]
    rows, cols = surface.shape

    if progress:
        progress("Building water-surface mask…", 0.0)

    # ── Build valid-data binary mask ──────────────────────────────
    valid_mask = ~np.isnan(surface)  # (rows, cols)

    # ── Optional extrapolation (dilate the valid mask) ────────────
    if extrapolate_m > 0:
        # Pixel size in CRS units
        pixel_size_x = abs(transform[0])
        pixel_size_y = abs(transform[4])
        pixel_size = max(pixel_size_x, pixel_size_y, 0.01)
        dilate_px = max(1, int(np.ceil(extrapolate_m / pixel_size)))

        # Simple morphological dilation via scipy or manual
        try:
            from scipy.ndimage import binary_dilation
            from scipy.ndimage import generate_binary_structure
            struct = generate_binary_structure(2, 1)  # 4-connected
            for _ in range(dilate_px):
                valid_mask = binary_dilation(valid_mask, structure=struct)
        except ImportError:
            # Fallback: manual iterative dilation
            for _ in range(dilate_px):
                dilated = valid_mask.copy()
                dilated[1:, :] |= valid_mask[:-1, :]
                dilated[:-1, :] |= valid_mask[1:, :]
                dilated[:, 1:] |= valid_mask[:, :-1]
                dilated[:, :-1] |= valid_mask[:, 1:]
                valid_mask = dilated

    if progress:
        progress("Sampling water-surface at points…", 30.0)

    # ── Determine valid extend for each point ─────────────────────
    inv_transform = ~transform
    col_f = inv_transform[0] * xs + inv_transform[1] * ys + inv_transform[2]
    row_f = inv_transform[3] * xs + inv_transform[4] * ys + inv_transform[5]

    col_i = np.clip(col_f.astype(np.int32), 0, cols - 1)
    row_i = np.clip(row_f.astype(np.int32), 0, rows - 1)

    # Points whose pixel coords are inside the raster bounds
    in_bounds = (col_f >= 0) & (col_f < cols) & (row_f >= 0) & (row_f < rows)

    if progress:
        progress("Checking against water surface…", 60.0)

    # Per-point: is the pixel valid?
    point_in_model = np.zeros(n, dtype=bool)
    if in_bounds.any():
        point_in_model[in_bounds] = valid_mask[row_i[in_bounds], col_i[in_bounds]]

    # Sample water surface Z at each point that's in the model
    ws_z = sample_geotiff_at_points(surface, georef, xs, ys, fill_value=np.nan)

    # ── Z check: point must be at or below water surface + tolerance ──
    tolerance_m = tolerance_above_cm / 100.0
    below_surface = zs <= (ws_z + tolerance_m)
    # Points outside the raster (NaN ws_z) pass the Z check if they
    # are also outside the model (they'll be caught by point_in_model)
    below_surface = np.where(np.isnan(ws_z), point_in_model, below_surface)

    keep_mask = point_in_model & below_surface
    outlier_mask = ~keep_mask

    if progress:
        progress(
            f"Water-surface crop: {outlier_mask.sum()}/{n} removed "
            f"(extrapolate={extrapolate_m:.1f}m, tol={tolerance_above_cm:.0f}cm)",
            100.0,
        )

    logger.info(
        "Water-surface crop: %d/%d removed "
        "(extrapolate=%.1fm, tol_above=%.0fcm, raster=%dx%d)",
        outlier_mask.sum(), n, extrapolate_m, tolerance_above_cm, cols, rows,
    )
    return keep_mask, outlier_mask
