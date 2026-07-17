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
    water_surface_z: float,
    sensor_position: Tuple[float, float, float],
    n_water: float = N_WATER_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Snell's law refraction correction to submerged bathymetric points.

    The green laser beam bends at the air-water interface and slows down.
    This function corrects the apparent 3-D positions to their true
    geometric locations using a flat water surface assumption.

    For each submerged point the correction:
      1. Finds the ray intersection with the water surface plane.
      2. Computes the incident angle in air.
      3. Applies Snell's law to get the refracted angle in water.
      4. Adjusts the underwater path length by the refractive index ratio.

    Args:
        xs, ys, zs:       Point coordinates (apparent, uncorrected).
        water_surface_z:  Elevation of the flat water surface.
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

    # Only correct points below the water surface
    submerged = zs < water_surface_z
    if not submerged.any():
        return out_x, out_y, out_z

    sx, sy, sz = sensor_position

    sub_x = xs[submerged]
    sub_y = ys[submerged]
    sub_z = zs[submerged]

    # Ray vector from sensor to apparent point
    dx = sub_x - sx
    dy = sub_y - sy
    dz = sub_z - sz

    # Intersection with horizontal water surface plane at Z = water_surface_z
    # Parametric: P(t) = sensor + t * (point - sensor), find t where Z=water_surface_z
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (water_surface_z - sz) / dz
    t = np.clip(t, 0.0, 1.0)

    inter_x = sx + t * dx
    inter_y = sy + t * dy
    inter_z = np.full_like(sub_z, water_surface_z)

    # Air path: sensor → surface intersection
    air_dx = inter_x - sx
    air_dy = inter_y - sy
    air_dz = inter_z - sz
    air_dist = np.sqrt(air_dx**2 + air_dy**2 + air_dz**2)

    # Apparent water path: surface → apparent bottom
    water_app_dx = sub_x - inter_x
    water_app_dy = sub_y - inter_y
    water_app_dz = sub_z - inter_z  # negative (downward)
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
    out_z[submerged] = water_surface_z + true_dz

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
