"""
LiDAR Workbench — Noise Filter Module.

Implements statistical and radius-based outlier removal filters
operating on numpy point clouds.  Designed to be called from both
interactive (GUI preview) and batch (background) contexts.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

from .config import (
    DEFAULT_ROR_MIN_POINTS,
    DEFAULT_ROR_RADIUS,
    DEFAULT_SOR_NB_NEIGHBORS,
    DEFAULT_SOR_STD_RATIO,
)

logger = logging.getLogger("lidar_workbench.noise_filter")

# Type alias: filter callback for progress
ProgressCB = Optional[Callable[[str, float], None]]


def statistical_outlier_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    nb_neighbors: int = DEFAULT_SOR_NB_NEIGHBORS,
    std_ratio: float = DEFAULT_SOR_STD_RATIO,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Statistical Outlier Removal (SOR).

    For each point, computes the mean distance to its *nb_neighbors*
    nearest neighbours.  Points whose mean distance exceeds
    ``global_mean + std_ratio * global_std`` are flagged as outliers.

    Args:
        xs, ys, zs:  Point coordinates as 1-D float arrays.
        nb_neighbors: Number of neighbours for the KNN query.
        std_ratio:    Standard-deviation multiplier threshold.
        progress:     Optional callback ``(step, pct)``.

    Returns:
        ``(keep_mask, outlier_mask)`` — boolean arrays of the same length
        as the input.  ``keep_mask[i]`` is ``True`` for inliers.
    """
    n = len(xs)
    if n == 0:
        empty = np.array([], dtype=bool)
        return empty, empty

    if progress:
        progress("Building KDTree…", 10.0)

    points = np.column_stack((xs, ys, zs))

    # Use scipy KDTree if available, otherwise brute-force
    try:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        # +1 because the first neighbour is the point itself
        k = min(nb_neighbors + 1, n)
        distances, _ = tree.query(points, k=k)
        if k > 1:
            # Drop self-distance (index 0)
            mean_dists = distances[:, 1:].mean(axis=1)
        else:
            mean_dists = distances[:, 0]
    except ImportError:
        logger.debug("scipy not available — using brute-force KNN for SOR")
        mean_dists = _brute_force_mean_knn(points, nb_neighbors)

    if progress:
        progress("Computing threshold…", 60.0)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std

    keep_mask = mean_dists <= threshold
    outlier_mask = ~keep_mask

    if progress:
        progress(
            f"SOR: {outlier_mask.sum()} outliers / {n} points", 100.0
        )

    logger.info(
        "SOR (k=%d, std=%.2f): %d outliers removed out of %d",
        nb_neighbors, std_ratio, outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask


def radius_outlier_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    radius: float = DEFAULT_ROR_RADIUS,
    min_points: int = DEFAULT_ROR_MIN_POINTS,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radius Outlier Removal (ROR).

    Points that have fewer than *min_points* neighbours within *radius*
    are flagged as outliers.

    Args:
        xs, ys, zs: Point coordinates.
        radius:      Search radius in CRS units.
        min_points:  Minimum number of neighbours to be considered an inlier.
        progress:    Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)``.
    """
    n = len(xs)
    if n == 0:
        empty = np.array([], dtype=bool)
        return empty, empty

    if progress:
        progress("Building KDTree…", 10.0)

    points = np.column_stack((xs, ys, zs))

    try:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        # Count neighbours within radius
        indices_list = tree.query_ball_point(points, r=radius, return_sorted=False)
        neighbour_counts = np.array([len(idx) for idx in indices_list], dtype=np.int32)
    except ImportError:
        logger.debug("scipy not available — using brute-force radius search")
        neighbour_counts = _brute_force_radius_count(points, radius)

    if progress:
        progress("Computing mask…", 70.0)

    keep_mask = neighbour_counts >= min_points
    outlier_mask = ~keep_mask

    if progress:
        progress(
            f"ROR: {outlier_mask.sum()} outliers / {n} points", 100.0
        )

    logger.info(
        "ROR (r=%.2f, min=%d): %d outliers removed out of %d",
        radius, min_points, outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask


def apply_filter_to_tile(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    classifications: np.ndarray,
    intensities: np.ndarray,
    return_numbers: np.ndarray,
    keep_mask: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Apply a boolean keep-mask to all point attributes.

    Returns filtered copies of all arrays.
    """
    if not keep_mask.any():
        empty_f = np.array([], dtype=np.float64)
        empty_u8 = np.array([], dtype=np.uint8)
        empty_u16 = np.array([], dtype=np.uint16)
        return empty_f, empty_f, empty_f, empty_u8, empty_u16, empty_u8

    return (
        xs[keep_mask].copy(),
        ys[keep_mask].copy(),
        zs[keep_mask].copy(),
        classifications[keep_mask].copy(),
        intensities[keep_mask].copy(),
        return_numbers[keep_mask].copy(),
    )


def dbscan_outlier_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    eps: float = 2.0,
    min_samples: int = 10,
    min_cluster_size: int = 50,
    mode: str = "above",
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DBSCAN-based outlier removal.

    Clusters points in 3-D space using scikit-learn's DBSCAN, then flags
    points belonging to clusters smaller than *min_cluster_size* as outliers.

    The *mode* parameter restricts which small clusters are flagged:

    - ``"above"`` — only flag small clusters whose mean Z is **above**
      the global median Z (aerial noise: birds, dust, sensor artifacts).
    - ``"below"`` — only flag small clusters whose mean Z is **below**
      the global median Z (sub-surface noise: multipath errors).
    - ``"both"`` — flag all small clusters regardless of elevation.

    Args:
        xs, ys, zs:      Point coordinates.
        eps:             DBSCAN neighbourhood radius (CRS units).
        min_samples:     Minimum points to form a core point in DBSCAN.
        min_cluster_size: Clusters smaller than this are noise candidates.
        mode:            ``"above"``, ``"below"``, or ``"both"``.
        progress:        Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)``.
    """
    n = len(xs)
    if n == 0:
        empty = np.array([], dtype=bool)
        return empty, empty

    if progress:
        progress("DBSCAN clustering…", 5.0)

    from sklearn.cluster import DBSCAN

    points = np.column_stack((xs, ys, zs))

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(points)

    if progress:
        progress("Computing cluster sizes…", 60.0)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("DBSCAN: %d clusters found (eps=%.2f, min_samples=%d)", n_clusters, eps, min_samples)

    # Points with label == -1 are already noise (DBSCAN's own classification)
    # PLUS points in clusters smaller than min_cluster_size
    global_median_z = float(np.median(zs))

    outlier_mask = np.zeros(n, dtype=bool)

    for label_val in set(labels):
        cluster_mask = labels == label_val
        cluster_size = cluster_mask.sum()

        if label_val == -1:
            # DBSCAN noise points — always out if mode allows
            if mode == "both":
                outlier_mask[cluster_mask] = True
            elif mode == "above":
                # Noise points above median
                outlier_mask[cluster_mask] = zs[cluster_mask] > global_median_z
            else:  # "below"
                outlier_mask[cluster_mask] = zs[cluster_mask] < global_median_z
        elif cluster_size < min_cluster_size:
            cluster_mean_z = float(zs[cluster_mask].mean())
            if mode == "both":
                outlier_mask[cluster_mask] = True
            elif mode == "above" and cluster_mean_z > global_median_z:
                outlier_mask[cluster_mask] = True
            elif mode == "below" and cluster_mean_z < global_median_z:
                outlier_mask[cluster_mask] = True

    keep_mask = ~outlier_mask

    if progress:
        progress(
            f"DBSCAN: {outlier_mask.sum()} outliers / {n} points", 100.0,
        )

    logger.info(
        "DBSCAN (%s, eps=%.2f, min_samples=%d, min_cluster=%d): "
        "%d outliers removed out of %d",
        mode, eps, min_samples, min_cluster_size, outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask



def isolated_point_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    search_radius: float = 3.0,
    min_distance: float = 0.5,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolated-point filter (TerraSolid: FnScanClassifyIsolated).

    Points whose nearest neighbour is farther than *min_distance*
    within *search_radius* are flagged as noise — catches flying
    artifacts, birds, sensor errors.

    Args:
        xs, ys, zs:     Point coordinates.
        search_radius:  Search radius in CRS units (metres).
        min_distance:   Minimum required distance to the closest neighbour.
        progress:       Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)``.
    """
    n = len(xs)
    if n < 2:
        empty = np.array([], dtype=bool)
        return np.ones(n, dtype=bool), empty

    if progress:
        progress("Isolated filter: building KDTree…", 10.0)

    points = np.column_stack((xs, ys, zs))
    try:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        k = min(2, n)
        dists, _ = tree.query(points, k=k)
        if dists.ndim == 1:
            nn_dist = dists
        else:
            nn_dist = dists[:, 1]
    except ImportError:
        # fallback: compute pairwise and take min non-self distance
        logger.debug("scipy not available — using brute-force for isolated filter")
        nn_dist = np.full(n, np.inf)
        for i in range(n):
            diff = points - points[i]
            d = np.sqrt((diff * diff).sum(axis=1))
            d[i] = np.inf
            nn_dist[i] = d.min()

    if progress:
        progress("Isolated filter: computing mask…", 60.0)

    keep_mask = nn_dist <= min_distance
    outlier_mask = ~keep_mask

    if progress:
        progress(f"Isolated: {outlier_mask.sum()} outliers / {n} points", 100.0)

    logger.info(
        "Isolated (r=%.2f, min_dist=%.2f): %d outliers removed out of %d",
        search_radius, min_distance, outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask


def low_point_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    search_radius: float = 2.0,
    below_threshold: float = 1.0,
    above_threshold: float = 10.0,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Low-point filter (TerraSolid: FnScanClassifyLow).

    For each point the lowest and highest Z among neighbours within
    *search_radius* is found.  If the point is more than
    *below_threshold* below the lowest neighbour, or more than
    *above_threshold* above the highest neighbour, it is flagged.

    Classic multipath (underground) and extreme aerial outlier removal.

    Args:
        xs, ys, zs:        Point coordinates.
        search_radius:     Neighbourhood radius (metres).
        below_threshold:   Max Z below the local minimum allowed.
        above_threshold:   Max Z above the local maximum allowed.
        progress:          Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)``.
    """
    n = len(xs)
    if n == 0:
        empty = np.array([], dtype=bool)
        return np.ones(0, dtype=bool), empty

    if progress:
        progress("Low-point filter: building KDTree…", 10.0)

    points = np.column_stack((xs, ys, zs))
    try:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        indices_list = tree.query_ball_point(points, r=search_radius)
    except ImportError:
        logger.debug("scipy not available — using brute-force for low-point filter")
        indices_list = _brute_force_radius_indices(points, search_radius)

    if progress:
        progress("Low-point filter: computing mask…", 60.0)

    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        neighbours = indices_list[i]
        if len(neighbours) < 2:
            continue
        nbr_z = zs[neighbours]
        if zs[i] < nbr_z.min() - below_threshold or \
           zs[i] > nbr_z.max() + above_threshold:
            keep_mask[i] = False

    outlier_mask = ~keep_mask

    if progress:
        progress(f"Low-point: {outlier_mask.sum()} outliers / {n} points", 100.0)

    logger.info(
        "Low-point (r=%.2f, below=%.2f, above=%.2f): %d outliers removed out of %d",
        search_radius, below_threshold, above_threshold, outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask


def surface_noise_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    grid_size: float = 0.5,
    surface_tolerance: float = 0.05,
    proximity_threshold: float = 0.25,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Surface-proximity noise filter (TerraSolid: FnScanClassifySurface
    + FnScanClassifyCloseby).

    Builds a smooth minimum-Z surface grid, then flags points that
    sit in a narrow band just above the surface — near-ground noise
    that is too high to be ground but too low to be real features.

    Algorithm:
        1.  Grid points at *grid_size*, keep minimum Z per cell.
        2.  Hole-fill and median-smooth the grid.
        3.  Bilinear-interpolate surface height for every point.
        4.  Points with height-above-surface in
            ``(surface_tolerance, proximity_threshold]`` are flagged.

    Args:
        xs, ys, zs:           Point coordinates.
        grid_size:            Cell size for the surface raster (metres).
        surface_tolerance:    Max height to be "on" the surface (metres).
        proximity_threshold:  Upper bound of the noise band (metres).
        progress:             Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)``.
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    if progress:
        progress("Surface-noise filter: gridding…", 10.0)

    points = np.column_stack((xs, ys, zs))
    import scipy.ndimage as ndimage

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    nx = int(np.ceil((max_x - min_x) / grid_size)) + 1
    ny = int(np.ceil((max_y - min_y) / grid_size)) + 1

    # --- minimum Z per cell ---
    grid = np.full((nx, ny), np.inf, dtype=np.float32)
    gx = np.clip(((xs - min_x) / grid_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys - min_y) / grid_size).astype(np.int32), 0, ny - 1)

    flat_idx = gx * ny + gy
    sort_idx = np.argsort(flat_idx)
    sorted_flat = flat_idx[sort_idx]
    sorted_z = zs[sort_idx]
    unique_bins, first_idx = np.unique(sorted_flat, return_index=True)
    last_idx = np.append(first_idx[1:], len(sort_idx))
    for j, bin_id in enumerate(unique_bins):
        bin_z = sorted_z[first_idx[j]:last_idx[j]]
        xi, yi = divmod(bin_id, ny)
        grid[xi, yi] = bin_z.min()
    grid[grid == np.inf] = np.nan

    # --- hole-fill ---
    grid = _simple_grid_fill(grid)

    # --- smooth ---
    grid = ndimage.median_filter(grid, size=3)

    if progress:
        progress("Surface-noise filter: interpolating…", 60.0)

    # --- bilinear surface height for every point ---
    fx = (xs - min_x) / grid_size
    fy = (ys - min_y) / grid_size
    x0 = np.clip(np.floor(fx).astype(np.int32), 0, nx - 1)
    y0 = np.clip(np.floor(fy).astype(np.int32), 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)

    wx = fx - x0
    wy = fy - y0

    fill_val = float(np.nanmedian(grid))
    if np.isnan(fill_val):
        fill_val = float(np.median(zs))

    def _safe(arr):
        out = arr.copy()
        out[np.isnan(out)] = fill_val
        return out

    z_surface = (_safe(grid[x0, y0]) * (1 - wx) * (1 - wy) +
                 _safe(grid[x1, y0]) * wx * (1 - wy) +
                 _safe(grid[x0, y1]) * (1 - wx) * wy +
                 _safe(grid[x1, y1]) * wx * wy)

    h_above = zs - z_surface
    is_noise = (h_above > surface_tolerance) & (h_above <= proximity_threshold)

    keep_mask = ~is_noise
    outlier_mask = is_noise

    if progress:
        progress(f"Surface-noise: {outlier_mask.sum()} outliers / {n} points", 100.0)

    logger.info(
        "Surface-noise (grid=%.2f, band=(%.2f, %.2f]): "
        "%d outliers removed out of %d",
        grid_size, surface_tolerance, proximity_threshold, outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask


# ── brute-force fallbacks ─────────────────────────────────────────────


def _brute_force_mean_knn(
    points: np.ndarray, k: int
) -> np.ndarray:
    """
    Compute mean distance to *k* nearest neighbours via brute force.

    **Note:** O(n²) — only used when scipy is unavailable and for small
    preview samples.
    """
    n = len(points)
    # For datasets > 5000 points this gets slow; warn once
    if n > 5000:
        logger.warning(
            "Brute-force KNN on %d points will be slow. "
            "Install scipy for KDTree acceleration.", n
        )

    mean_dists = np.zeros(n, dtype=np.float64)
    for i in range(n):
        diff = points - points[i]
        dists = np.sqrt((diff * diff).sum(axis=1))
        dists.sort()
        k_eff = min(k + 1, n)
        if k_eff > 1:
            mean_dists[i] = dists[1:k_eff].mean()
        else:
            mean_dists[i] = dists[0]
    return mean_dists


def _brute_force_radius_count(
    points: np.ndarray, radius: float
) -> np.ndarray:
    """Count neighbours within *radius* via brute force."""
    n = len(points)
    if n > 5000:
        logger.warning(
            "Brute-force radius search on %d points will be slow. "
            "Install scipy for KDTree acceleration.", n
        )
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        diff = points - points[i]
        dists_sq = (diff * diff).sum(axis=1)
        counts[i] = (dists_sq <= radius * radius).sum()
    return counts


def _brute_force_radius_indices(
    points: np.ndarray, radius: float
) -> list:
    """Return list of neighbour-index arrays within *radius* via brute force."""
    n = len(points)
    if n > 5000:
        logger.warning(
            "Brute-force radius search on %d points will be slow. "
            "Install scipy for KDTree acceleration.", n
        )
    indices_list = []
    for i in range(n):
        diff = points - points[i]
        dists_sq = (diff * diff).sum(axis=1)
        indices_list.append(np.where(dists_sq <= radius * radius)[0])
    return indices_list


def _simple_grid_fill(grid: np.ndarray, passes: int = 20) -> np.ndarray:
    """
    Fill NaN cells in a 2-D grid by averaging valid 4-connected neighbours.
    Iterative — repeated *passes* times (or until no NaNs remain).
    """
    res = grid.copy()
    for _ in range(passes):
        valid = ~np.isnan(res)
        if np.all(valid):
            break
        shifted_sum = np.zeros_like(res)
        count = np.zeros_like(res)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            s = np.full_like(res, np.nan)
            txs, txe = max(0, dx), res.shape[0] + min(0, dx)
            sxs, sxe = max(0, -dx), res.shape[0] + min(0, -dx)
            tys, tye = max(0, dy), res.shape[1] + min(0, dy)
            sys, sye = max(0, -dy), res.shape[1] + min(0, -dy)
            if txe > txs and tye > tys:
                s[txs:txe, tys:tye] = res[sxs:sxe, sys:sye]
            m = ~np.isnan(s)
            shifted_sum[m] += s[m]
            count[m] += 1
        upd = (~valid) & (count > 0)
        res[upd] = shifted_sum[upd] / count[upd]
    return res


# ── Ground classification algorithms ─────────────────────────────────


def ground_classify_smrf(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    cell_size: Optional[float] = None,
    slope_threshold: float = 0.15,
    max_window: float = 20.0,
    elevation_threshold: float = 0.5,
    base_window: float = 2.0,
    window_growth: float = 1.5,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    Simple Morphological Filter (SMRF) for ground classification.

    PDAL's classic ground-filter algorithm.  Applies morphological
    opening (erode + dilate) with progressively larger window sizes,
    then classifies points as ground if they fall within an
    elevation threshold of the filtered surface.

    The slope_threshold controls how aggressively the filter removes
    vegetation — higher values preserve more terrain features; lower
    values strip more non-ground points.

    Args:
        xs, ys, zs:          Point coordinates.
        cell_size:           Grid cell size (metres). Auto from spacing.
        slope_threshold:     Max slope for terrain (rise/run).
        max_window:          Largest morphological window (metres).
        elevation_threshold: Max height above filtered surface for ground.
        base_window:         Starting window size (metres).
        window_growth:       Multiplier for each window step.
        progress:            Optional callback.

    Returns:
        ``ground_mask`` — boolean array, ``True`` for ground points.
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    try:
        import scipy.ndimage as ndimage
    except ImportError:
        logger.warning("scipy not available — skipping SMRF")
        return np.ones(n, dtype=bool)

    if progress:
        progress("SMRF: computing spacing…", 5.0)

    if cell_size is None:
        cell_size = compute_adaptive_spacing(xs, ys)
        cell_size = max(cell_size, 0.5)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)

    # Build minimum-Z grid
    grid = np.full((nx, ny), np.inf, dtype=np.float32)
    gx = np.clip(((xs - min_x) / cell_size).astype(int), 0, nx - 1)
    gy = np.clip(((ys - min_y) / cell_size).astype(int), 0, ny - 1)

    flat_idx = gx * ny + gy
    sort_idx = np.argsort(flat_idx)
    sorted_flat = flat_idx[sort_idx]
    sorted_z = zs[sort_idx]
    unique_bins, first_idx = np.unique(sorted_flat, return_index=True)
    last_idx = np.append(first_idx[1:], n)
    for j, bin_id in enumerate(unique_bins):
        xi, yi = divmod(bin_id, ny)
        grid[xi, yi] = sorted_z[first_idx[j]:last_idx[j]].min()

    # Fill no-data cells
    nan_mask = np.isinf(grid)
    if nan_mask.any():
        fill_val = float(np.median(zs))
        for _ in range(20):
            valid = ~nan_mask
            if valid.all():
                break
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                s = np.roll(grid, (dx, dy), axis=(0, 1))
                nan_mask[nan_mask & ~np.isinf(s)] = False
                grid[nan_mask & ~np.isinf(s)] = s[nan_mask & ~np.isinf(s)]

    if progress:
        progress("SMRF: morphological filtering…", 20.0)

    # Compute slope-based elevation thresholds per window step
    window = base_window
    filtered = grid.copy()
    step_total = int(np.ceil(np.log(max_window / base_window) / np.log(window_growth))) + 1
    step_idx = 0

    while window <= max_window:
        if progress:
            progress(f"SMRF: window {window:.1f}m…", 20 + 60 * step_idx / step_total)

        radius_pixels = max(1, int(np.ceil(window / cell_size / 2.0)))
        thr = slope_threshold * window

        # Erode
        eroded = ndimage.grey_erosion(filtered, size=(2 * radius_pixels + 1))
        # Dilate
        dilated = ndimage.grey_dilation(eroded, size=(2 * radius_pixels + 1))

        # Accept only Z values that are not too far from the original
        diff = grid - dilated
        filtered = np.where(diff < thr, dilated, filtered)

        window *= window_growth
        step_idx += 1

    if progress:
        progress("SMRF: classifying points…", 90.0)

    # Bilinear-interpolate filtered surface for each point
    fx = (xs - min_x) / cell_size
    fy = (ys - min_y) / cell_size
    ix0, iy0 = np.clip(np.floor(fx).astype(int), 0, nx - 1), np.clip(np.floor(fy).astype(int), 0, ny - 1)
    ix1, iy1 = np.clip(ix0 + 1, 0, nx - 1), np.clip(iy0 + 1, 0, ny - 1)
    wx, wy = fx - ix0, fy - iy0

    z_surf = (
        filtered[ix0, iy0] * (1 - wx) * (1 - wy)
        + filtered[ix1, iy0] * wx * (1 - wy)
        + filtered[ix0, iy1] * (1 - wx) * wy
        + filtered[ix1, iy1] * wx * wy
    )

    ground_mask = (zs - z_surf) <= elevation_threshold

    logger.info(
        "SMRF (cell=%.1f, slope=%.2f): %d ground / %d points",
        cell_size, slope_threshold, ground_mask.sum(), n,
    )
    return ground_mask


def ground_probability_refinement(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    classifications: np.ndarray,
    intensities: np.ndarray,
    return_numbers: np.ndarray,
    ground_class_code: int = 2,
    low_noise_code: int = 7,
    knn: int = 20,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-classification ground probability refinement.

    After Pointcept (or other) classification, scores each point's
    ground likelihood using geometric consistency features:
      - Local planarity (PCA eigenvalue ratio)
      - Vertical distance to a local minimum-Z surface
      - Return-number heuristics (single returns more likely ground)
      - Intensity relative to local median
      - Neighborhood classification consensus

    Points with high ground probability but currently classified as
    non-ground (or vice versa) are flagged for reclassification.

    Args:
        xs, ys, zs:        Point coordinates.
        classifications:   Current ASPRS class codes.
        intensities:       LiDAR intensity values.
        return_numbers:    Return numbers.
        ground_class_code: ASPRS code for ground (default 2).
        low_noise_code:    ASPRS code for low-point noise (default 7).
        knn:               Neighbors for local analysis.
        progress:          Optional callback.

    Returns:
        ``(ground_probability, suggested_class)`` —
        probability in [0, 1] and suggested ASPRS class code per point.
    """
    n = len(xs)
    ground_prob = np.zeros(n, dtype=np.float64)
    suggested_class = classifications.copy()

    if n < knn:
        ground_prob[:] = 0.5
        return ground_prob, suggested_class

    try:
        from scipy.spatial import KDTree
    except ImportError:
        ground_prob[:] = 0.5
        return ground_prob, suggested_class

    if progress:
        progress("Ground prob: building KDTree…", 5.0)

    points = np.column_stack((xs, ys, zs))
    tree = KDTree(points)
    _, all_indices = tree.query(points, k=min(knn + 1, n))

    if progress:
        progress("Ground prob: computing features…", 15.0)

    # Feature 1: local minimum surface estimate
    spacing = compute_adaptive_spacing(xs, ys)
    cell_size = max(spacing * 5, 1.0)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)

    gx = np.clip(((xs - min_x) / cell_size).astype(int), 0, nx - 1)
    gy = np.clip(((ys - min_y) / cell_size).astype(int), 0, ny - 1)
    min_grid = np.full((nx, ny), np.nan)
    for i in range(n):
        xi, yi = gx[i], gy[i]
        if np.isnan(min_grid[xi, yi]) or zs[i] < min_grid[xi, yi]:
            min_grid[xi, yi] = zs[i]

    # Fill NaN cells with median of neighbors
    for _ in range(10):
        nan_mask = np.isnan(min_grid)
        if not nan_mask.any():
            break
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            rolled = np.roll(min_grid, (dx, dy), axis=(0, 1))
            fill = nan_mask & ~np.isnan(rolled)
            min_grid[fill] = rolled[fill]

    if progress:
        progress("Ground prob: scoring points…", 40.0)

    is_currently_ground = (classifications == ground_class_code)
    med_intensity = float(np.median(intensities)) if len(intensities) > 0 else 0.0

    for i in range(n):
        # Feature 1: planarity from local PCA
        nbr_idx = all_indices[i]
        if all_indices.ndim > 1 and all_indices.shape[1] > 1:
            nbr_idx = nbr_idx[1:]
        nbr_pts = points[nbr_idx]

        if len(nbr_pts) < 5:
            ground_prob[i] = 0.5
            continue

        try:
            cov = np.cov(nbr_pts.T)
            eigvals = np.linalg.eigvalsh(cov)
            l_sum = eigvals.sum()
            if l_sum > 0:
                planarity = (eigvals[1] - eigvals[0]) / eigvals[2] if eigvals[2] > 0 else 0.0
            else:
                planarity = 0.0
        except np.linalg.LinAlgError:
            planarity = 0.0

        # Feature 2: height above local minimum
        xi, yi = gx[i], gy[i]
        h_above_min = zs[i] - min_grid[xi, yi]

        # Feature 3: return-number heuristic
        is_single_return = (return_numbers[i] == 1)

        # Feature 4: intensity relative to local
        local_intens = intensities[nbr_idx] if len(intensities) > 0 else np.array([0])
        if med_intensity > 0:
            int_ratio = intensities[i] / max(med_intensity, 1)
        else:
            int_ratio = 1.0

        # Feature 5: neighborhood ground ratio
        nbr_class = classifications[nbr_idx[1:]] if len(nbr_idx) > 1 else classifications[nbr_idx]
        nbr_ground_ratio = (nbr_class == ground_class_code).mean()

        # --- Score computation ---
        score = 0.0

        # Planar = likely ground (weight 0.35)
        if planarity > 0.4:
            score += 0.35
        elif planarity > 0.2:
            score += 0.20
        else:
            score += 0.05

        # Close to local minimum = likely ground (weight 0.25)
        if h_above_min < 0.5:
            score += 0.25
        elif h_above_min < 1.5:
            score += 0.15
        else:
            score += 0.0

        # Single return = likely ground (weight 0.15)
        if is_single_return:
            score += 0.15

        # Intensity not anomalous (weight 0.10)
        if 0.3 < int_ratio < 2.5:
            score += 0.10
        elif 0.1 < int_ratio < 5.0:
            score += 0.05

        # Neighborhood consensus (weight 0.15)
        score += 0.15 * nbr_ground_ratio

        ground_prob[i] = np.clip(score, 0.0, 1.0)

    if progress:
        progress("Ground prob: refining classifications…", 80.0)

    # Reclassify based on probability
    high_confidence = ground_prob > 0.65
    low_confidence = ground_prob < 0.25

    # Non-ground with high ground probability → reclassify to ground
    reclass_to_ground = high_confidence & ~is_currently_ground
    suggested_class[reclass_to_ground] = ground_class_code

    # Currently ground with very low probability → flag as low noise
    reclass_from_ground = low_confidence & is_currently_ground
    suggested_class[reclass_from_ground] = low_noise_code

    logger.info(
        "Ground prob refinement: %.1f%% avg confidence, "
        "%d → ground, %d → noise (of %d points)",
        ground_prob.mean() * 100,
        reclass_to_ground.sum(), reclass_from_ground.sum(), n,
    )

    return ground_prob, suggested_class


def ground_classify_tin(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    max_distance: float = 1.4,
    max_angle: float = 6.0,
    max_distance_above: float = 0.15,
    cell_size: Optional[float] = None,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    Progressive TIN Densification ground classification.

    Based on Axelsson's algorithm — the industry standard for
    progressive TIN densification for ground classification.

    Algorithm:
      1. Select seed points from lowest points in a coarse grid.
      2. Build an initial TIN from the seeds.
      3. Iteratively add unclassified points that are close to the TIN
         and whose angle to the TIN vertices is below the threshold.
      4. Rebuild TIN and repeat until no more points are added.

    This naturally preserves sharp terrain breaks (cliffs, banks) while
    correctly classifying large buildings and vegetation as non-ground.

    Water surface points are rejected by the asymmetric distance check:
    points above the TIN must be within ``max_distance_above`` (tight),
    while points on/below the TIN use the standard ``max_distance``.

    Args:
        xs, ys, zs:          Point coordinates.
        max_distance:        Max allowed distance to TIN for points on/below
                             the TIN surface (metres).
        max_angle:           Max allowed angle between candidate and TIN
                             vertices (degrees). Lower = more conservative.
        max_distance_above:  Max allowed distance for points ABOVE the TIN
                             (metres). Very tight to reject water surface.
        cell_size:           Seed grid cell size. Auto when ``None``.
        progress:            Optional callback.

    Returns:
        ``ground_mask`` — boolean array, ``True`` for ground.
    """
    import math

    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    if progress:
        progress("TIN ground: computing spacing…", 5.0)

    if cell_size is None:
        cell_size = compute_adaptive_spacing(xs, ys)
        cell_size = max(cell_size * 10, 5.0)  # seed grid ~10x point spacing

    # --- Step 1: select seed points (lowest in each grid cell) ---
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)

    gx = np.clip(((xs - min_x) / cell_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys - min_y) / cell_size).astype(np.int32), 0, ny - 1)
    flat_idx = gx * ny + gy

    order = np.argsort(flat_idx)
    sorted_flat = flat_idx[order]
    sorted_z = zs[order]
    unique_cells, starts = np.unique(sorted_flat, return_index=True)

    seed_mask = np.zeros(n, dtype=bool)
    for i in range(len(unique_cells)):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else n
        lowest_local = int(np.argmin(sorted_z[start:end]))
        seed_mask[order[start + lowest_local]] = True

    if seed_mask.sum() < 3:
        logger.warning("TIN: too few seed points")
        return np.ones(n, dtype=bool)

    ground_mask = seed_mask.copy()
    n_ground = seed_mask.sum()

    if progress:
        progress(f"TIN ground: {n_ground} seeds…", 15.0)

    max_angle_rad = math.radians(max_angle)
    sin_max_angle = math.sin(max_angle_rad)

    try:
        from scipy.spatial import Delaunay, KDTree
    except ImportError:
        logger.warning("scipy not available — TIN densification limited")
        return ground_mask

    # Pre-compute useful arrays
    xs_f64 = np.asarray(xs, dtype=np.float64)
    ys_f64 = np.asarray(ys, dtype=np.float64)
    zs_f64 = np.asarray(zs, dtype=np.float64)

    max_iterations = 15
    for iteration in range(max_iterations):
        if progress:
            progress(
                f"TIN ground: iter {iteration+1}/{max_iterations} ({ground_mask.sum()} pts)…",
                20 + 60 * iteration / max_iterations,
            )

        # Build Delaunay triangulation on current ground points
        g_xy = np.column_stack((xs_f64[ground_mask], ys_f64[ground_mask]))
        g_z = zs_f64[ground_mask]
        if len(g_xy) < 3:
            break
        try:
            tri = Delaunay(g_xy)
        except Exception:
            break

        # --- Process non-ground candidates ---
        cand_indices = np.where(~ground_mask)[0]
        n_cand = len(cand_indices)
        if n_cand == 0:
            break

        # Vectorised: find simplex for ALL candidates at once
        cand_xy = np.column_stack((xs_f64[cand_indices], ys_f64[cand_indices]))
        cand_z = zs_f64[cand_indices]
        simplex_ids = tri.find_simplex(cand_xy)

        # Mask: candidates that fall inside the TIN convex hull
        inside = simplex_ids >= 0
        outside = ~inside

        new_ground = np.zeros(n_cand, dtype=bool)

        # --- Handle points inside convex hull ---
        if inside.any():
            s_ids = simplex_ids[inside]           # simplex index per inside-point
            s_local = cand_indices[inside]         # original point indices
            s_xy = cand_xy[inside]
            s_z = cand_z[inside]

            simplices = tri.simplices[s_ids]       # (M, 3) indices into ground_pts
            v0 = np.column_stack((g_xy[simplices[:, 0]], g_z[simplices[:, 0]]))
            v1 = np.column_stack((g_xy[simplices[:, 1]], g_z[simplices[:, 1]]))
            v2 = np.column_stack((g_xy[simplices[:, 2]], g_z[simplices[:, 2]]))

            # Barycentric coordinates
            denom = (
                (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0])
                + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
            )
            valid_denom = np.abs(denom) >= 1e-12
            if valid_denom.any():
                inv_d = np.zeros_like(denom)
                inv_d[valid_denom] = 1.0 / denom[valid_denom]

                w0 = (
                    (v1[:, 1] - v2[:, 1]) * (s_xy[:, 0] - v2[:, 0])
                    + (v2[:, 0] - v1[:, 0]) * (s_xy[:, 1] - v2[:, 1])
                ) * inv_d
                w1 = (
                    (v2[:, 1] - v0[:, 1]) * (s_xy[:, 0] - v2[:, 0])
                    + (v0[:, 0] - v2[:, 0]) * (s_xy[:, 1] - v2[:, 1])
                ) * inv_d
                w2 = 1.0 - w0 - w1

                # Point is inside triangle + within Z tolerance
                inside_tri = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                if inside_tri.any():
                    tin_z = w0 * v0[:, 2] + w1 * v1[:, 2] + w2 * v2[:, 2]

                    # Asymmetric distance: tight for points above TIN (water surface,
                    # vegetation), standard for points on/below (actual ground).
                    dz = s_z - tin_z                     # positive = above TIN
                    d_above = np.maximum(dz, 0.0)
                    d_below = np.maximum(-dz, 0.0)
                    d_ok = (d_above <= max_distance_above) & (d_below <= max_distance)

                    # Axelsson angle check: for each vertex V of the TIN triangle,
                    # the angle between P and the TIN plane is arcsin(d / |P-V|).
                    # d is the perpendicular distance (|dz|).
                    d_abs = np.abs(dz)
                    p_xyz = np.column_stack((s_xy, s_z))
                    d_v0 = np.linalg.norm(p_xyz - v0, axis=1)
                    d_v1 = np.linalg.norm(p_xyz - v1, axis=1)
                    d_v2 = np.linalg.norm(p_xyz - v2, axis=1)
                    sin_max = np.maximum(np.maximum(
                        np.divide(d_abs, d_v0, out=np.zeros_like(d_abs), where=d_v0 > 1e-9),
                        np.divide(d_abs, d_v1, out=np.zeros_like(d_abs), where=d_v1 > 1e-9)),
                        np.divide(d_abs, d_v2, out=np.zeros_like(d_abs), where=d_v2 > 1e-9))
                    angle_ok = sin_max <= sin_max_angle

                    ok = inside_tri & valid_denom & d_ok & angle_ok
                    new_ground[inside] = ok

        # --- Handle points outside convex hull ---
        if outside.any():
            o_local = cand_indices[outside]
            o_xy = cand_xy[outside]
            o_z = cand_z[outside]

            tree = KDTree(g_xy)
            dists, nbrs = tree.query(o_xy, k=1)
            nbr_z = g_z[nbrs]
            dz_out = o_z - nbr_z
            ok_outside = (dz_out <= max_distance_above) & (-dz_out <= max_distance)
            new_ground[outside] = ok_outside

        # Apply new ground points
        added = new_ground.sum()
        if added == 0:
            break
        ground_mask[cand_indices[new_ground]] = True

    logger.info(
        "TIN ground (max_d=%.2f, max_a=%.1f°): %d ground / %d points",
        max_distance, max_angle, ground_mask.sum(), n,
    )
    return ground_mask


def compute_adaptive_spacing(
    xs: np.ndarray,
    ys: np.ndarray,
    sample_size: int = 5000,
) -> float:
    """
    Estimate mean point spacing from a random sample.

    Returns the median nearest-neighbour distance in the XY plane.
    """
    n = len(xs)
    if n < 2:
        return 1.0
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return 1.0
    idx = np.random.choice(n, min(sample_size, n), replace=False)
    tree = KDTree(np.column_stack((xs[idx], ys[idx])))
    dist, _ = tree.query(tree.data, k=2)
    return float(np.median(dist[:, 1])) if dist.ndim > 1 else 1.0


def multipath_reflection_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    depth_threshold: float = 0.5,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove ghost surfaces caused by multi-path reflections.

    Detects planar surfaces via RANSAC, projects points onto the plane
    normal, and keeps only the main cluster — discarding offset "ghost"
    planes parallel to the real surface.

    Adapted for ALS/bathy: works on locally flat regions like water
    surfaces, building facades, and riverbeds.

    Args:
        xs, ys, zs:       Point coordinates.
        depth_threshold:  Maximum offset along normal to keep (metres).
        progress:         Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)``.
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    if progress:
        progress("Multipath: detecting planes…", 10.0)

    points = np.column_stack((xs, ys, zs))

    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import RANSACRegressor
    except ImportError:
        logger.warning("sklearn not available — skipping multipath filter")
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    keep_mask = np.ones(n, dtype=bool)

    # Work in tiles for large datasets
    tile_n = 50000
    for start in range(0, n, tile_n):
        end = min(start + tile_n, n)
        tile_pts = points[start:end]
        if len(tile_pts) < 100:
            continue

        # Fit a dominant plane via RANSAC
        try:
            ransac = RANSACRegressor(residual_threshold=depth_threshold * 2)
            ransac.fit(tile_pts[:, :2], tile_pts[:, 2])
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
        except ValueError:
            continue

        normal = np.array([a, b, -1.0])
        normal /= np.linalg.norm(normal)
        center = tile_pts.mean(axis=0)
        proj = np.dot(tile_pts - center, normal)

        # Histogram of projections → keep the densest cluster
        hist, bins = np.histogram(proj, bins=min(50, len(tile_pts) // 20))
        if len(hist) == 0:
            continue
        main_bin = np.argmax(hist)
        main_center = (bins[main_bin] + bins[main_bin + 1]) / 2.0

        tile_keep = np.abs(proj - main_center) < depth_threshold
        keep_mask[start:end] = tile_keep

    outlier_mask = ~keep_mask
    logger.info(
        "Multipath (depth_thresh=%.2f): %d outliers removed out of %d",
        depth_threshold, outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask


def bilateral_filter(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    spatial_sigma: float = 0.5,
    range_sigma: float = 0.1,
    knn: int = 20,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Edge-preserving bilateral filter for point cloud denoising.

    Smooths points while preserving sharp edges by weighting neighbours
    by both spatial distance and depth (Z) similarity.

    Args:
        xs, ys, zs:      Point coordinates.
        spatial_sigma:   Gaussian sigma for spatial distance weighting.
        range_sigma:     Gaussian sigma for depth (Z) difference weighting.
        knn:             Number of nearest neighbours.
        progress:        Optional callback.

    Returns:
        ``(smoothed_xs, smoothed_ys, smoothed_zs)``.
    """
    n = len(xs)
    if n < knn:
        return xs.copy(), ys.copy(), zs.copy()

    try:
        from scipy.spatial import KDTree
    except ImportError:
        logger.warning("scipy not available — skipping bilateral filter")
        return xs.copy(), ys.copy(), zs.copy()

    if progress:
        progress("Bilateral: building KDTree…", 10.0)

    points = np.column_stack((xs, ys, zs))
    tree = KDTree(points)
    _, all_indices = tree.query(points, k=min(knn + 1, n))

    out_x, out_y, out_z = xs.copy(), ys.copy(), zs.copy()

    for i in range(n):
        nbr_idx = all_indices[i]
        if all_indices.ndim > 1 and all_indices.shape[1] > 1:
            nbr_idx = nbr_idx[1:]  # skip self
        if len(nbr_idx) < 3:
            continue

        nbrs = points[nbr_idx]
        dist = np.linalg.norm(nbrs - points[i], axis=1)

        spatial_w = np.exp(-dist**2 / (2 * spatial_sigma**2))
        z_diff = np.abs(nbrs[:, 2] - points[i, 2])
        range_w = np.exp(-z_diff**2 / (2 * range_sigma**2))
        weights = spatial_w * range_w
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
            out_x[i] = np.average(nbrs[:, 0], weights=weights)
            out_y[i] = np.average(nbrs[:, 1], weights=weights)
            out_z[i] = np.average(nbrs[:, 2], weights=weights)

    return out_x, out_y, out_z


def topo_discriminator(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    intensities: np.ndarray,
    return_numbers: np.ndarray,
    k_neighbors: int = 15,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate topographic points into terrain (ground/rock) vs canopy
    (vegetation) using PCA eigenvalue analysis and multi-echo logic.

    The local covariance structure reveals:
      - Ground / rock:  high planarity, low scattering, near-vertical normal
      - Vegetation:     low planarity, high scattering (volumetric)
      - Rock outcrops:  high planarity, tilted normal, high intensity

    Args:
        xs, ys, zs:        Point coordinates.
        intensities:       LiDAR intensity values.
        return_numbers:    Return numbers (1 = single/first).
        k_neighbors:       Number of neighbours for PCA.
        progress:          Optional callback.

    Returns:
        ``(is_terrain, is_rock)`` — boolean arrays.
    """
    n = len(xs)
    is_terrain = np.zeros(n, dtype=bool)
    is_rock = np.zeros(n, dtype=bool)

    if n < k_neighbors:
        return is_terrain, is_rock

    try:
        from scipy.spatial import KDTree
    except ImportError:
        return is_terrain, is_rock

    if progress:
        progress("Topo discriminator: building KDTree…", 10.0)

    points = np.column_stack((xs, ys, zs))
    tree = KDTree(points)
    _, all_indices = tree.query(points, k=min(k_neighbors, n))

    med_intensity = np.median(intensities) if len(intensities) > 0 else 0

    for i in range(n):
        nbr_idx = all_indices[i]
        nbr_pts = points[nbr_idx]
        if len(nbr_pts) < 5:
            continue

        cov = np.cov(nbr_pts.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        idx_sort = np.argsort(eigenvalues)[::-1]
        l1, l2, l3 = eigenvalues[idx_sort]
        l_sum = l1 + l2 + l3
        if l_sum == 0:
            continue

        planarity = (l2 - l3) / l1 if l1 > 0 else 0.0
        scattering = l3 / l1 if l1 > 0 else 1.0
        normal = eigenvectors[:, idx_sort[2]]
        vertical_alignment = abs(normal[2])

        # High scattering → vegetation
        if scattering > 0.25:
            continue

        # High planarity → solid surface (terrain or rock)
        if planarity > 0.4:
            if vertical_alignment < 0.7 or (med_intensity > 0 and intensities[i] > med_intensity * 1.5):
                is_rock[i] = True
            is_terrain[i] = True
        else:
            # Flat, horizontal → soil/ground if single return
            if return_numbers[i] == 1:
                is_terrain[i] = True

    if progress:
        progress(
            f"Topo: {is_terrain.sum()} terrain / {is_rock.sum()} rock / {n} points",
            100.0,
        )

    logger.info(
        "Topo discriminator: %d terrain, %d rock out of %d points",
        is_terrain.sum(), is_rock.sum(), n,
    )
    return is_terrain, is_rock


def water_surface_ghost_removal(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    tile_length: float = 40.0,
    roughness_threshold: float = 0.02,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove water surface and parallel ghost layer from bathymetric data.

    Splits the point cloud into short along-track tiles, detects the
    water surface plane using RANSAC on smooth (low-roughness) points,
    and removes points within a narrow adaptive band around the surface.

    Designed for river bathymetry — handles sloping water surfaces and
    preserves shallow riverbeds.

    Args:
        xs, ys, zs:            Point coordinates.
        tile_length:           Along-track tile length in metres.
        roughness_threshold:   Surface-variation threshold for water points.
        progress:              Optional callback.

    Returns:
        ``(keep_mask, outlier_mask)``.
    """
    n = len(xs)
    if n < 200:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    try:
        from scipy.spatial import KDTree
        from sklearn.linear_model import RANSACRegressor
    except ImportError:
        logger.warning("scipy/sklearn not available — skipping water surface filter")
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    spacing = compute_adaptive_spacing(xs, ys)
    points = np.column_stack((xs, ys, zs))

    if progress:
        progress("Bathy water surface: tiling…", 5.0)

    # Tile along X axis
    xmin, xmax = xs.min(), xs.max()
    keep_mask = np.ones(n, dtype=bool)

    x = xmin
    while x < xmax:
        tile_mask = (xs >= x) & (xs < x + tile_length) & keep_mask
        if tile_mask.sum() < 100:
            x += tile_length * 0.5
            continue

        tile_idx = np.where(tile_mask)[0]
        tile_pts = points[tile_idx]

        # 1. Compute local roughness
        radius = max(5 * spacing, 0.5)
        tree = KDTree(tile_pts)
        roughness = np.ones(len(tile_pts))
        for i in range(len(tile_pts)):
            nbrs = tree.query_ball_point(tile_pts[i], radius)
            if len(nbrs) < 5:
                continue
            cov = np.cov(tile_pts[nbrs].T)
            eigvals = np.linalg.eigvalsh(cov)
            roughness[i] = eigvals[0] / eigvals.sum() if eigvals.sum() > 0 else 1.0

        smooth = roughness < roughness_threshold
        if smooth.sum() < 50:
            x += tile_length * 0.5
            continue

        # 2. Fit plane to smooth points
        smooth_pts = tile_pts[smooth]
        try:
            ransac = RANSACRegressor(residual_threshold=0.15, random_state=0)
            ransac.fit(smooth_pts[:, :2], smooth_pts[:, 2])
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
        except ValueError:
            x += tile_length * 0.5
            continue

        normal = np.array([a, b, -1.0])
        normal /= np.linalg.norm(normal)

        # 3. Project onto normal, find main cluster, remove band
        proj = np.dot(tile_pts, normal) - (-c)
        hist, bins = np.histogram(proj, bins=min(100, len(tile_pts) // 10))
        if len(hist) < 2:
            x += tile_length * 0.5
            continue

        main_bin = np.argmax(hist)
        cluster_center = (bins[main_bin] + bins[main_bin + 1]) / 2.0
        cluster_mask = (proj > bins[main_bin]) & (proj < bins[main_bin + 1])
        cluster_std = float(np.std(proj[cluster_mask])) if cluster_mask.sum() > 5 else 0.1
        band_width = min(2.0 * cluster_std, 0.3)

        # Keep points DEEPER than the surface band
        tile_keep = proj < (cluster_center - band_width)
        keep_mask[tile_idx] = tile_keep

        x += tile_length * 0.5

    outlier_mask = ~keep_mask
    logger.info(
        "Bathy water-surface/ghost: %d points removed / %d total",
        outlier_mask.sum(), n,
    )
    return keep_mask, outlier_mask


def river_corridor_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    intensities: np.ndarray,
    return_numbers: np.ndarray,
    grid_res: float = 2.0,
    water_surface_z: Optional[float] = None,
    progress: ProgressCB = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically crop bathymetric data to the river corridor.

    Uses local geometric roughness and intensity characteristics to
    distinguish water from land — water has higher vertical jitter and
    characteristic green-laser intensity profiles.

    Args:
        xs, ys, zs:        Point coordinates.
        intensities:       LiDAR intensity values.
        return_numbers:    Return numbers.
        grid_res:          Grid resolution in metres for the mask.
        water_surface_z:   Optional known water surface elevation.
                           Points far from this Z are excluded.
        progress:          Optional callback.

    Returns:
        ``(river_mask, land_mask)`` — boolean arrays.
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    if progress:
        progress("River crop: computing mask…", 10.0)

    x_bins = (xs / grid_res).astype(int)
    y_bins = (ys / grid_res).astype(int)
    combined = np.column_stack((x_bins, y_bins))
    unique_bins, inverse = np.unique(combined, axis=0, return_inverse=True)

    median_intensity = np.median(intensities) if len(intensities) > 0 else 0
    river_mask = np.zeros(n, dtype=bool)

    for i in range(len(unique_bins)):
        idx = np.where(inverse == i)[0]
        if len(idx) < 15:
            continue

        col_zs = zs[idx]
        col_int = intensities[idx] if len(intensities) > 0 else np.zeros(len(idx))
        z_std = float(np.std(col_zs))

        # Water: moderate vertical jitter (not perfectly flat ground, not tall vegetation)
        # Typical river water surface has subtle waves but is mostly flat
        has_water_jitter = 0.03 < z_std < 1.0
        # Water returns have lower intensity than land for green laser
        has_aquatic_intensity = (
            np.mean(col_int) < median_intensity * 0.6
            if median_intensity > 0
            else True
        )

        if has_water_jitter and has_aquatic_intensity:
            # Optional: restrict to points near water surface
            if water_surface_z is not None:
                col_z_mean = float(np.mean(col_zs))
                if abs(col_z_mean - water_surface_z) > 5.0:  # more than 5m away
                    continue
            river_mask[idx] = True

    # Morphological closing to fill gaps
    if progress:
        progress("River crop: filling gaps…", 60.0)

    try:
        from scipy.ndimage import binary_closing, binary_fill_holes, label

        x_range = x_bins.max() - x_bins.min() + 1
        y_range = y_bins.max() - y_bins.min() + 1
        grid = np.zeros((x_range, y_range), dtype=bool)
        valid = (x_bins >= x_bins.min()) & (y_bins >= y_bins.min())
        grid[x_bins[valid] - x_bins.min(), y_bins[valid] - y_bins.min()] = river_mask[valid]

        grid = binary_closing(grid, iterations=2)
        grid = binary_fill_holes(grid)
        labeled, ncomp = label(grid)
        if ncomp > 0:
            largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            grid = labeled == largest

        valid = (x_bins >= x_bins.min()) & (y_bins >= y_bins.min())
        river_mask[valid] = grid[x_bins[valid] - x_bins.min(), y_bins[valid] - y_bins.min()]
    except ImportError:
        pass  # no scipy — keep raw mask

    land_mask = ~river_mask
    logger.info(
        "River crop: %d river / %d land points",
        river_mask.sum(), land_mask.sum(),
    )
    return river_mask, land_mask


def benthic_continuity_filter(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    certain_bed_mask: np.ndarray,
    search_radius: float = 0.3,
    max_slope: float = 0.35,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    Discriminate true riverbed from floating water-column noise using
    topological continuity (region growing / flood fill).

    Starting from high-confidence bed points, "walks" outward and
    accepts neighbouring points that connect with a gentle slope,
    rejecting isolated noise clusters.

    Args:
        xs, ys, zs:         Point coordinates.
        certain_bed_mask:   Boolean mask of high-confidence bed points.
        search_radius:      Neighbour search radius (metres).
        max_slope:          Maximum allowable slope (rise/run) for continuity.
        progress:           Optional callback.

    Returns:
        ``bed_mask`` — boolean array.
    """
    n = len(xs)
    if n < 100:
        return certain_bed_mask.copy()

    try:
        from scipy.spatial import KDTree
    except ImportError:
        return certain_bed_mask.copy()

    if progress:
        progress("Benthic continuity: growing…", 10.0)

    points = np.column_stack((xs, ys, zs))
    tree = KDTree(points)

    final_bed = certain_bed_mask.copy()
    queue = list(np.where(certain_bed_mask)[0])
    visited = set(queue)

    while queue:
        curr = queue.pop(0)
        curr_pt = points[curr]
        nbrs = tree.query_ball_point(curr_pt, r=search_radius)
        for nb in nbrs:
            if nb in visited:
                continue
            visited.add(nb)
            nb_pt = points[nb]
            delta_z = abs(nb_pt[2] - curr_pt[2])
            delta_xy = np.linalg.norm(nb_pt[:2] - curr_pt[:2])
            if delta_xy > 0 and (delta_z / delta_xy) <= max_slope:
                final_bed[nb] = True
                queue.append(nb)

    logger.info(
        "Benthic continuity: %d → %d bed points (search_r=%.2f, max_slope=%.2f)",
        certain_bed_mask.sum(), final_bed.sum(), search_radius, max_slope,
    )
    return final_bed


# ── parallel filter worker ──────────────────────────────────────────

try:
    from PySide6.QtCore import QThread, Signal
    _HAS_QT = True
except ImportError:
    _HAS_QT = False


if _HAS_QT:

    class FilterWorker(QThread):
        """Background worker that applies a filter pipeline to tiles in
        parallel using a thread pool, reporting progress."""

        progress = Signal(str, float)         # message, percentage
        tile_done = Signal(str)               # tile_id that finished
        finished_all = Signal(list, list)     # tile_ids, keep_masks
        error_occurred = Signal(str)

        def __init__(self, tile_data: list, pipeline: list,
                     workers: int = 4, parent=None):
            super().__init__(parent)
            self._tile_data = tile_data  # list of (tile_id, data_dict)
            self._pipeline = pipeline
            self._workers = workers
            self._results: list = []

        def run(self):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            total = len(self._tile_data)
            done = 0
            try:
                with ThreadPoolExecutor(max_workers=self._workers) as pool:
                    futures = {
                        pool.submit(_apply_pipeline, data, self._pipeline): tid
                        for tid, data in self._tile_data
                    }
                    for fut in as_completed(futures):
                        tid = futures[fut]
                        try:
                            keep = fut.result()
                            self._results.append((tid, keep))
                            done += 1
                            self.progress.emit(
                                f"Filtered {done}/{total} tile(s)…",
                                done / total * 100.0,
                            )
                            self.tile_done.emit(tid)
                        except Exception as exc:
                            self.error_occurred.emit(f"{tid}: {exc}")
            except Exception as exc:
                self.error_occurred.emit(str(exc))
            self.finished_all.emit(
                [r[0] for r in self._results],
                [r[1] for r in self._results],
            )


def _apply_pipeline(data: dict, pipeline: list) -> np.ndarray:
    """Apply a filter pipeline to a single tile's data, return keep mask."""
    n = len(data["x"])
    keep = np.ones(n, dtype=bool)
    for step in pipeline:
        if step["type"] == "sor":
            k, _ = statistical_outlier_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                nb_neighbors=step["nb_neighbors"],
                std_ratio=step["std_ratio"],
            )
        elif step["type"] == "ror":
            k, _ = radius_outlier_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                radius=step["radius"], min_points=step["min_points"],
            )
        elif step["type"] == "isolated":
            k, _ = isolated_point_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                search_radius=step["search_radius"],
                min_distance=step["min_distance"],
            )
        elif step["type"] == "low_points":
            k, _ = low_point_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                search_radius=step["search_radius"],
                below_threshold=step["below_threshold"],
                above_threshold=step["above_threshold"],
            )
        elif step["type"] == "surface_noise":
            k, _ = surface_noise_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                grid_size=step["grid_size"],
                surface_tolerance=step["surface_tolerance"],
                proximity_threshold=step["proximity_threshold"],
            )
        elif step["type"] == "multipath":
            k, _ = multipath_reflection_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                depth_threshold=step.get("depth_threshold", 0.5),
            )
        elif step["type"] == "bilateral":
            # bilateral_filter returns smoothed coordinates, not a mask
            sx, sy, sz = bilateral_filter(
                data["x"][keep], data["y"][keep], data["z"][keep],
                spatial_sigma=step.get("spatial_sigma", 0.5),
                range_sigma=step.get("range_sigma", 0.1),
                knn=step.get("knn", 20),
            )
            # Update data in-place for subsequent pipeline steps
            keep_indices = np.where(keep)[0]
            data["x"][keep_indices] = sx
            data["y"][keep_indices] = sy
            data["z"][keep_indices] = sz
            continue  # no mask to update
        elif step["type"] == "topo_discriminator":
            target = step.get("target", "terrain_only")
            is_terrain, is_rock = topo_discriminator(
                data["x"][keep], data["y"][keep], data["z"][keep],
                data.get("intensity", np.zeros(keep.sum(), dtype=np.uint16))[keep],
                data.get("return_number", np.ones(keep.sum(), dtype=np.uint8))[keep],
            )
            if target == "terrain_only":
                k = is_terrain
            elif target == "rock_only":
                k = is_rock
            else:
                k = is_terrain | is_rock
        elif step["type"] == "bathy_water_surface":
            k, _ = water_surface_ghost_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                tile_length=step.get("tile_length", 40.0),
            )
        elif step["type"] == "bathy_river_crop":
            k, _ = river_corridor_mask(
                data["x"][keep], data["y"][keep], data["z"][keep],
                data.get("intensity", np.zeros(keep.sum(), dtype=np.uint16))[keep],
                data.get("return_number", np.ones(keep.sum(), dtype=np.uint8))[keep],
            )
            # Preserve topographic points — river crop should only affect bathy/unknown
            st = data.get("sensor_type")
            if st is not None:
                k = k | (st[keep] == 1)  # keep all topo points regardless of river mask
        elif step["type"] == "benthic_continuity":
            certain_mask = np.ones(keep.sum(), dtype=bool)
            k = benthic_continuity_filter(
                data["x"][keep], data["y"][keep], data["z"][keep],
                certain_bed_mask=certain_mask,
                search_radius=step.get("search_radius", 0.3),
                max_slope=step.get("max_slope", 0.35),
            )
        elif step["type"] == "ground_smrf":
            k = ground_classify_smrf(
                data["x"][keep], data["y"][keep], data["z"][keep],
                cell_size=step.get("cell_size"),
                slope_threshold=step.get("slope_threshold", 0.15),
                max_window=step.get("max_window", 20.0),
                elevation_threshold=step.get("elevation_threshold", 0.5),
                base_window=step.get("base_window", 2.0),
                window_growth=step.get("window_growth", 1.5),
            )
        elif step["type"] == "ground_tin":
            k = ground_classify_tin(
                data["x"][keep], data["y"][keep], data["z"][keep],
                max_distance=step.get("max_distance", 1.4),
                max_angle=step.get("max_angle", 6.0),
                cell_size=step.get("cell_size"),
            )
        else:
            mode = "above" if step["type"] == "dbscan_above" else "below"
            k, _ = dbscan_outlier_removal(
                data["x"][keep], data["y"][keep], data["z"][keep],
                eps=step["eps"], min_samples=step["min_samples"],
                min_cluster_size=step["min_cluster_size"], mode=mode,
            )
        keep_indices = np.where(keep)[0]
        keep[keep_indices[~k]] = False
    return keep
