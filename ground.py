"""
LiDAR Workbench — Ground classification.

Ground filtering algorithms and their helpers, moved out of ``noise_filter``
so that noise removal and terrain classification stay separate:

  - :func:`ground_classify_smrf`  — Simple Morphological Filter (Pingel 2013).
  - :func:`ground_classify_tin`   — Progressive TIN Densification (Axelsson 2000).
  - :func:`ground_probability_refinement` — post-classification scoring.
  - :func:`compute_adaptive_spacing` — point-spacing helper shared with bathy.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger("lidar_workbench.ground")

# Type alias: progress callback
ProgressCB = Optional[Callable[[str, float], None]]

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
    max_terrain_angle: float = 88.0,
    max_building_size: Optional[float] = None,
    reduce_iter_angle_when_edge: Optional[float] = None,
    stop_tri_when_edge: Optional[float] = None,
    only_upward: bool = False,
    follow_surface_trend: bool = True,
    remove_low_outliers: bool = False,
    low_outlier_neighbors: int = 8,
    low_outlier_threshold: float = 1.0,
    exclude_single_returns_in_water: bool = False,
    sensor_type: Optional[np.ndarray] = None,
    cell_size: Optional[float] = None,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    Progressive TIN Densification ground classification (Axelsson 2000).

    Terrasolid-compatible implementation.  No pre-filters, no post-filters,
    no water-rejection heuristics — just the canonical algorithm:

      1. Select seed points (lowest Z in each cell of a coarse grid).
      2. Build an initial Delaunay TIN from the seeds.
      3. Iteratively add points that are close to the TIN surface (within
         ``max_distance``) and whose angle to the TIN vertices is below
         ``max_angle``.
      4. Rebuild TIN and repeat until no more points are added.

    Args:
        xs, ys, zs:          Point coordinates.
        max_distance:        Iteration distance — max allowed orthogonal
                             distance from a point to the TIN plane (metres).
                             Symmetric (same for above and below).
                             Typical: 0.5–1.5 m.
        max_angle:           Iteration angle — max angle between the
                             candidate point, its projection on the TIN
                             plane, and the closest triangle vertex
                             (degrees).  Typical: 4–10°.
        max_terrain_angle:   Steepest allowed slope of a TIN triangle
                             (degrees).  Use 88–90° for man-made terrain;
                             lower (e.g. 45–60°) for purely natural terrain.
        max_building_size:   Largest expected building dimension in the
                             project area (metres).  Determines the seed
                             grid cell size so every area has at least one
                             seed point.
        reduce_iter_angle_when_edge:
                             When every edge of a TIN triangle is shorter
                             than this (metres), linearly reduce the
                             iteration angle to avoid over-densification.
                             ``None`` = disabled.
        stop_tri_when_edge:  When every edge of a TIN triangle is shorter
                             than this (metres), stop processing inside
                             that triangle entirely.  ``None`` = disabled.
        only_upward:         If ``True``, only accept points whose Z is
                             ≥ the seed-cell minimum Z.  Prevents low
                             outliers from pulling the TIN downward.
        follow_surface_trend: If ``True``, locally relax the iteration
                             angle on steep triangles (slope > 10°) so
                             the TIN can climb riverbanks and cliffs.
        remove_low_outliers: If ``True``, run a post-densification pass
                             that removes ground points sitting well below
                             their local ground neighbourhood (low outliers
                             that become seed points and spike the DTM).
                             Mirrors lidR's internal outlier removal.
        low_outlier_neighbors: Number of nearest ground neighbours used to
                             estimate the local ground surface (default 8).
        low_outlier_threshold: A ground point is flagged as a low outlier
                             when it is more than this many metres below the
                             median Z of its neighbours (default 1.0 m).
        exclude_single_returns_in_water:
                             If ``True``, drop single returns
                             (``num_returns == 1``) for bathymetric points
                             (``sensor_type == 2``) before densification, so
                             the shallow-water surface isn't swept into the
                             ground set along with the riverbed.
        sensor_type:         Optional per-point sensor code (0=unknown,
                             1=topo, 2=bathy).  Used only when
                             ``exclude_single_returns_in_water`` is set.
        cell_size:           Explicit seed-grid cell size.  Auto-computed
                             from point spacing when ``None``.
        return_numbers:      Optional array of return numbers (1-based).
                             When provided together with ``num_returns``,
                             only last returns and single returns (where
                             ``return_number == num_returns``) are used
                             to build the TIN.  First/intermediate returns
                             are excluded from ground.
        num_returns:         Optional array of number-of-returns per pulse.
                             Used with ``return_numbers`` to identify
                             last/single returns.
        progress:            Optional ``(message, percent)`` callback.

    Returns:
        ``ground_mask`` — boolean array, ``True`` for ground.
    """
    import math

    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    # ── Last-return pre-filter ────────────────────────────────────
    # Ground classification works best on last returns and single
    # returns (only echoes), because first/intermediate returns are
    # often vegetation or structures above ground.  Building the TIN
    # from last/single returns gives a cleaner ground surface.
    have_returns = (return_numbers is not None and num_returns is not None
                    and len(return_numbers) == n and len(num_returns) == n)
    if have_returns:
        # The last-return filter is only meaningful when the data actually
        # records multiple returns per pulse.  Some exports leave
        # ``num_returns`` (number_of_returns) at a constant 1 while
        # ``return_number`` still counts 1..N.  Filtering on
        # ``return_number == 1`` there would keep FIRST/single returns
        # (e.g. the water surface in bathymetry) and throw away the true
        # last returns (the riverbed).  When the field is not populated,
        # fall through and classify on all points instead.
        max_num_returns = int(np.max(num_returns))
        if max_num_returns <= 1:
            logger.info(
                "TIN: num_returns not populated (max=%d) — classifying on all points",
                max_num_returns,
            )
            have_returns = False

    if have_returns:
        last_mask = (return_numbers == num_returns)
        if (exclude_single_returns_in_water and sensor_type is not None
                and len(sensor_type) == n):
            # In water, "ground" is the last return of a MULTI-return pulse
            # (the riverbed).  Single returns (num_returns == 1) in water are
            # usually the water surface, which otherwise gets swept into the
            # ground set when the water is shallow.  Keep them for land.
            bathy = np.asarray(sensor_type) == 2
            last_mask = last_mask & (~bathy | (num_returns > 1))
        n_last = last_mask.sum()
        if n_last < 3:
            logger.warning(
                "TIN: too few last/single returns (%d), using all points",
                n_last,
            )
        elif n_last < n:
            logger.info(
                "TIN: using %d last/single returns out of %d points (%.1f%%)",
                n_last, n, 100.0 * n_last / n,
            )
            sub_map = np.where(last_mask)[0]
            sub_ground = ground_classify_tin(
                xs[last_mask], ys[last_mask], zs[last_mask],
                max_distance=max_distance,
                max_angle=max_angle,
                max_terrain_angle=max_terrain_angle,
                max_building_size=max_building_size,
                reduce_iter_angle_when_edge=reduce_iter_angle_when_edge,
                stop_tri_when_edge=stop_tri_when_edge,
                only_upward=only_upward,
                follow_surface_trend=follow_surface_trend,
                remove_low_outliers=remove_low_outliers,
                low_outlier_neighbors=low_outlier_neighbors,
                low_outlier_threshold=low_outlier_threshold,
                cell_size=cell_size,
                progress=progress,
            )
            full_mask = np.zeros(n, dtype=bool)
            full_mask[sub_map[sub_ground]] = True
            return full_mask

    if progress:
        progress("TIN ground: computing spacing…", 5.0)

    # --- Determine seed grid cell size ---
    if max_building_size is not None:
        seed_cell_size = max_building_size
    elif cell_size is not None:
        seed_cell_size = max(cell_size, 5.0)
    else:
        seed_cell_size = compute_adaptive_spacing(xs, ys)
        seed_cell_size = max(seed_cell_size * 10, 5.0)

    # --- Step 1: select seed points (lowest Z in each grid cell) ---
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    nx = max(1, int(np.ceil((max_x - min_x) / seed_cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / seed_cell_size)) + 1)

    gx = np.clip(((xs - min_x) / seed_cell_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys - min_y) / seed_cell_size).astype(np.int32), 0, ny - 1)
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
        idx = order[start + lowest_local]
        seed_mask[idx] = True

    if seed_mask.sum() < 3:
        logger.warning("TIN: too few seed points")
        return np.ones(n, dtype=bool)

    ground_mask = seed_mask.copy()
    n_ground = seed_mask.sum()

    if progress:
        progress(f"TIN ground: {n_ground} seeds (cell={seed_cell_size:.1f}m)…", 15.0)

    max_angle_rad = math.radians(max_angle)
    sin_max_angle = math.sin(max_angle_rad)
    max_terrain_angle_rad = math.radians(max_terrain_angle)
    cos_terrain_angle = math.cos(max_terrain_angle_rad)

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        logger.warning("scipy not available — TIN densification limited")
        return ground_mask

    xs_f64 = np.asarray(xs, dtype=np.float64)
    ys_f64 = np.asarray(ys, dtype=np.float64)
    zs_f64 = np.asarray(zs, dtype=np.float64)

    # --- Only-upward: record seed-cell minimum Z to prevent downward pulls ---
    if only_upward:
        seed_z_surface = np.full(n, np.inf, dtype=np.float64)
        for i in range(len(unique_cells)):
            start = starts[i]
            end = starts[i + 1] if i + 1 < len(starts) else n
            cell_min_z = sorted_z[start:end].min()
            cell_mask = flat_idx == unique_cells[i]
            seed_z_surface[cell_mask] = cell_min_z

    max_iterations = 15
    for iteration in range(max_iterations):
        if progress:
            progress(
                f"TIN ground: iter {iteration+1}/{max_iterations} "
                f"({ground_mask.sum()} pts)…",
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

        # --- Per-triangle properties ---
        simplices = tri.simplices
        tri_normals = np.zeros((len(simplices), 3), dtype=np.float64)
        tri_slope_ok = np.ones(len(simplices), dtype=bool)
        tri_max_edge = np.zeros(len(simplices), dtype=np.float64)
        tri_active = np.ones(len(simplices), dtype=bool)

        for ti, s in enumerate(simplices):
            v0_3d = np.array([g_xy[s[0], 0], g_xy[s[0], 1], g_z[s[0]]])
            v1_3d = np.array([g_xy[s[1], 0], g_xy[s[1], 1], g_z[s[1]]])
            v2_3d = np.array([g_xy[s[2], 0], g_xy[s[2], 1], g_z[s[2]]])
            normal = np.cross(v1_3d - v0_3d, v2_3d - v0_3d)
            n_len = np.linalg.norm(normal)
            if n_len > 1e-15:
                normal /= n_len
            tri_normals[ti] = normal

            cos_slope = abs(normal[2])
            tri_slope_ok[ti] = cos_slope >= cos_terrain_angle

            e01_xy = g_xy[s[1]] - g_xy[s[0]]
            e02_xy = g_xy[s[2]] - g_xy[s[0]]
            e12_xy = g_xy[s[2]] - g_xy[s[1]]
            tri_max_edge[ti] = max(
                np.linalg.norm(e01_xy),
                np.linalg.norm(e02_xy),
                np.linalg.norm(e12_xy),
            )

        # --- Edge-based controls ---
        if stop_tri_when_edge is not None:
            tri_active[tri_max_edge < stop_tri_when_edge] = False

        # --- Candidates: all non-ground points ---
        cand_indices = np.where(~ground_mask)[0]
        n_cand = len(cand_indices)
        if n_cand == 0:
            break

        cand_xy = np.column_stack((xs_f64[cand_indices], ys_f64[cand_indices]))
        cand_z = zs_f64[cand_indices]
        simplex_ids = tri.find_simplex(cand_xy)

        inside = simplex_ids >= 0
        new_ground = np.zeros(n_cand, dtype=bool)

        if inside.any():
            s_ids = simplex_ids[inside]
            s_local = cand_indices[inside]
            s_xy = cand_xy[inside]
            s_z = cand_z[inside]

            tri_s = tri.simplices[s_ids]
            v0 = np.column_stack((g_xy[tri_s[:, 0]], g_z[tri_s[:, 0]]))
            v1 = np.column_stack((g_xy[tri_s[:, 1]], g_z[tri_s[:, 1]]))
            v2 = np.column_stack((g_xy[tri_s[:, 2]], g_z[tri_s[:, 2]]))

            # Barycentric coordinates (2-D)
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

                inside_tri = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                if inside_tri.any():
                    tin_z = w0 * v0[:, 2] + w1 * v1[:, 2] + w2 * v2[:, 2]
                    dz = s_z - tin_z  # vertical offset (kept for trend logic)

                    # Perpendicular (orthogonal) distance from the candidate
                    # point to the TIN plane — the quantity Terrasolid's
                    # "iteration distance" and Axelsson's angle both use.
                    # Using the vertical |dz| instead over-rejects points on
                    # steep slopes (embankments, riverbanks).
                    p_xyz = np.column_stack((s_xy, s_z))
                    d_perp = np.abs(np.sum(
                        (p_xyz - v0) * tri_normals[s_ids], axis=1
                    ))

                    # --- Symmetric distance check (Terrasolid "Iteration distance") ---
                    d_ok = d_perp <= max_distance

                    # --- Terrain angle check ---
                    slope_ok = tri_slope_ok[s_ids]

                    # --- Edge-based stop ---
                    edge_ok = tri_active[s_ids]

                    # --- Edge-based angle reduction ---
                    if reduce_iter_angle_when_edge is not None:
                        edge_frac = np.clip(
                            tri_max_edge[s_ids] / reduce_iter_angle_when_edge,
                            0.1, 1.0,
                        )
                        local_sin_max = sin_max_angle * edge_frac
                    else:
                        local_sin_max = np.full(
                            len(s_ids) if isinstance(s_ids, np.ndarray)
                            else sum(inside),
                            sin_max_angle,
                        )

                    # --- Axelsson angle check ---
                    d_v0 = np.linalg.norm(p_xyz - v0, axis=1)
                    d_v1 = np.linalg.norm(p_xyz - v1, axis=1)
                    d_v2 = np.linalg.norm(p_xyz - v2, axis=1)
                    sin_max = np.maximum(np.maximum(
                        np.divide(d_perp, d_v0, out=np.zeros_like(d_perp),
                                  where=d_v0 > 1e-9),
                        np.divide(d_perp, d_v1, out=np.zeros_like(d_perp),
                                  where=d_v1 > 1e-9)),
                        np.divide(d_perp, d_v2, out=np.zeros_like(d_perp),
                                  where=d_v2 > 1e-9))
                    angle_ok = sin_max <= local_sin_max

                    # --- Follow surface trend: relax angle on steep slopes ---
                    if follow_surface_trend:
                        tri_slope_cos = np.abs(tri_normals[:, 2])
                        tri_slope_deg = np.degrees(
                            np.arccos(np.clip(tri_slope_cos, 0.0, 1.0))
                        )
                        slope_deg = tri_slope_deg[s_ids]
                        steep = slope_deg > 10.0
                        if steep.any():
                            st_idx = np.where(
                                steep & inside_tri & valid_denom
                            )[0]
                            if len(st_idx) > 0:
                                # Relax iteration angle proportionally to slope
                                adapted_angle = (
                                    max_angle + slope_deg[st_idx] * 0.5
                                )
                                adapted_sin = np.sin(
                                    np.radians(adapted_angle)
                                )
                                # Only relax for points above the TIN on
                                # steep slopes (uphill direction)
                                uphill_st = dz[st_idx] > 0
                                if uphill_st.any():
                                    u_idx = st_idx[uphill_st]
                                    relaxed_ok = (
                                        sin_max[u_idx]
                                        <= adapted_sin[uphill_st]
                                    )
                                    angle_ok[u_idx] = (
                                        angle_ok[u_idx] | relaxed_ok
                                    )

                    # --- Only-upward check ---
                    if only_upward:
                        upward_ok = s_z >= seed_z_surface[s_local]
                    else:
                        upward_ok = np.ones(sum(inside), dtype=bool)

                    ok = (
                        inside_tri & valid_denom & d_ok & angle_ok
                        & slope_ok & edge_ok & upward_ok
                    )
                    new_ground[inside] = ok

        # --- Apply new ground points ---
        added = new_ground.sum()
        if added == 0:
            break
        ground_mask[cand_indices[new_ground]] = True

    # --- Optional low-outlier removal (lidR-style post-cleanup) ---
    if remove_low_outliers:
        if progress:
            progress("TIN ground: removing low outliers…", 92.0)
        before = ground_mask.sum()
        ground_mask = _remove_low_outliers(
            xs, ys, zs, ground_mask,
            neighbors=low_outlier_neighbors,
            threshold=low_outlier_threshold,
        )
        logger.info(
            "TIN: removed %d low outliers (%d → %d ground)",
            before - ground_mask.sum(), before, ground_mask.sum(),
        )

    logger.info(
        "TIN ground (max_d=%.2f, max_a=%.1f, terr_a=%.1f, build=%.1f): "
        "%d ground / %d points",
        max_distance, max_angle, max_terrain_angle,
        max_building_size if max_building_size is not None else seed_cell_size,
        ground_mask.sum(), n,
    )
    return ground_mask


def _remove_low_outliers(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    ground_mask: np.ndarray,
    neighbors: int = 8,
    threshold: float = 1.0,
    passes: int = 2,
) -> np.ndarray:
    """
    Remove low outliers from a ground mask (lidR-style post-cleanup).

    A point is flagged as a low outlier when it sits more than *threshold*
    metres below the median Z of its *neighbors* nearest ground neighbours
    (in XY).  Iterates a few passes so clustered outliers are all removed.

    Note: this is a purely local test, so in strongly concave terrain
    (narrow valley floors) genuine local minima can occasionally be flagged.
    Keep *threshold* large enough for the local relief, or disable.
    """
    if not ground_mask.any():
        return ground_mask
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return ground_mask

    mask = ground_mask.copy()
    for _ in range(max(1, passes)):
        idx = np.where(mask)[0]
        if len(idx) < neighbors + 1:
            break
        xy = np.column_stack((xs[idx], ys[idx]))
        tree = KDTree(xy)
        k = min(neighbors + 1, len(idx))
        _, nbr = tree.query(xy, k=k)
        if nbr.ndim == 1:
            nbr = nbr[:, None]
        nbr = nbr[:, 1:]  # drop self (column 0)
        nbr_z = zs[idx[nbr]]  # (m, k-1)
        median_z = np.median(nbr_z, axis=1)
        low = zs[idx] < (median_z - threshold)
        if not low.any():
            break
        mask[idx[low]] = False
    return mask


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
