"""
LiDAR Workbench — Ground classification.

Ground filtering algorithms and their helpers, moved out of ``noise_filter``
so that noise removal and terrain classification stay separate:

  - :func:`ground_classify_smrf`    — Simple Morphological Filter (Pingel 2013).
  - :func:`ground_classify_epptd`   — Edge-Preserving PTD (Axelsson 2000 + edge controls).
  - :func:`ground_classify_epptd_two_pass` — coarse-to-fine two-pass EP-PTD (12° → 6°).
  - :func:`ground_classify_aptd`    — Adaptive Grid PTD, AGPTD (Zheng et al. 2024).
  - :func:`ground_classify_hptd`    — Hierarchical/Fast PTD, FPTD (Li et al. 2021).
  - :func:`ground_classify_dl_hybrid_ptd` — Pointcept DL prior + PTD densification.
  - :func:`ground_probability_refinement` — post-classification scoring.
  - :func:`compute_adaptive_spacing` — point-spacing helper shared with bathy.

``ground_classify_tin`` is kept as a backward-compatible alias of
``ground_classify_epptd``.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence, Tuple

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


def _lowest_per_grid_cells(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    cell_size: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Return a mask with the lowest-Z point in each grid cell.

    The grid origin is shifted by ``offset_x``/``offset_y``.  Used both for
    classic Axelsson seed selection and for shifted dual-grid seeding
    (two grids offset by half a cell, then unioned).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return np.zeros(0, dtype=bool)

    origin_x = min_x + offset_x
    origin_y = min_y + offset_y
    nx = max(1, int(np.ceil((max_x - origin_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - origin_y) / cell_size)) + 1)
    gx = np.clip(((xs - origin_x) / cell_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys - origin_y) / cell_size).astype(np.int32), 0, ny - 1)
    flat_idx = gx * ny + gy

    cell_min = np.full(nx * ny, np.inf, dtype=np.float64)
    np.minimum.at(cell_min, flat_idx, zs)

    is_cell_min = zs == cell_min[flat_idx]
    min_pos = np.flatnonzero(is_cell_min)
    mask = np.zeros(n, dtype=bool)
    if len(min_pos):
        # Keep one point per cell when several tie for the minimum.
        _, first = np.unique(flat_idx[min_pos], return_index=True)
        mask[min_pos[first]] = True
    return mask


def ground_classify_epptd(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    max_distance: float = 1.6,
    max_angle: float = 25.0,
    seed_resolution: float = 2.0,
    spacing: Optional[float] = None,
    follow_surface_trend: bool = True,
    dense: bool = True,
    dense_tolerance: float = 0.40,
    remove_low_outliers: bool = True,
    low_outlier_threshold: float = 1.0,
    existing_ground_mask: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
    **kwargs,
) -> np.ndarray:
    """Progressive TIN Densification (Axelsson 2000).

    Algorithm:
      1. Seeds: Lowest point per ``seed_resolution`` grid cell from two grids offset
         by half a cell, unioned. In river corridors, this naturally selects
         the lowest points (submerged riverbed).
      2. Candidates: Lowest point per ``spacing`` grid cell, sorted ascending by Z.
         Higher vegetation and water surface reflections are automatically excluded
         at this pre-thinning stage.
      3. Iterative Triangulation: Rebuilds Delaunay TIN per iteration; freezes
         candidates in triangles smaller than ``spacing``; inserts candidates
         meeting perpendicular distance and Axelsson vertex angle criteria.
      4. Dense pass (optional): Labels all unclassified points against the frozen
         final TIN within tolerance.
      5. Low outlier cleanup (optional): Removes points sitting far below their
         nearest ground neighbors.

    No sensor type or water surface assumptions required.
    """
    import math
    from scipy.spatial import Delaunay

    # Backwards compatibility with previous parameter names
    if "cell_size" in kwargs and kwargs["cell_size"] is not None and kwargs["cell_size"] > 0:
        seed_resolution = float(kwargs["cell_size"])
    if "max_building_size" in kwargs and kwargs["max_building_size"] is not None:
        seed_resolution = float(kwargs["max_building_size"])
    if "dense_pass" in kwargs and kwargs["dense_pass"] is not None:
        dense = bool(kwargs["dense_pass"])
    if "dense_above_tolerance" in kwargs and kwargs["dense_above_tolerance"] is not None:
        dense_tolerance = float(kwargs["dense_above_tolerance"])

    n = len(xs)
    if n < 3:
        return np.ones(n, dtype=bool)

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    min_x, max_x = float(xs_f.min()), float(xs_f.max())
    min_y, max_y = float(ys_f.min()), float(ys_f.max())

    # 1. Point spacing (candidate resolution)
    if spacing is None or spacing <= 0.0:
        spacing = max(compute_adaptive_spacing(xs_f, ys_f), 0.5)

    if seed_resolution <= 0.0:
        seed_resolution = max(spacing * 15.0, 15.0)

    if progress:
        progress("PTD: selecting seeds and candidates…", 5.0)

    # 2. Virtual bounding box corner anchors
    corner_xy = np.empty((4, 2), dtype=np.float64)
    corner_z = np.empty(4, dtype=np.float64)
    for i, (cx, cy) in enumerate(
        ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))
    ):
        d2 = (xs_f - cx) ** 2 + (ys_f - cy) ** 2
        k = min(5, n)
        nearest = np.argpartition(d2, k - 1)[:k]
        corner_xy[i] = (cx, cy)
        corner_z[i] = zs_f[nearest[int(np.argmin(zs_f[nearest]))]]

    # 3. Initial seeds
    if existing_ground_mask is not None and np.asarray(existing_ground_mask).sum() >= 3:
        seed_mask = np.asarray(existing_ground_mask, dtype=bool).copy()
    else:
        # Two shifted grids, unioned
        grid1 = _lowest_per_grid_cells(
            xs_f, ys_f, zs_f,
            min_x, min_y, max_x, max_y,
            seed_resolution, 0.0, 0.0,
        )
        grid2 = _lowest_per_grid_cells(
            xs_f, ys_f, zs_f,
            min_x, min_y, max_x, max_y,
            seed_resolution, seed_resolution * 0.5, seed_resolution * 0.5,
        )
        seed_mask = grid1 | grid2

    if int(seed_mask.sum()) < 3:
        logger.warning("PTD: too few seeds (%d), using lowest points", int(seed_mask.sum()))
        seed_mask = _lowest_per_grid_cells(
            xs_f, ys_f, zs_f,
            min_x, min_y, max_x, max_y,
            max(spacing * 3.0, 2.0), 0.0, 0.0,
        )
        if int(seed_mask.sum()) < 3:
            return np.ones(n, dtype=bool)

    ground_mask = seed_mask.copy()

    # 4. Candidates: lowest point per `spacing` cell
    cand_mask = _lowest_per_grid_cells(
        xs_f, ys_f, zs_f,
        min_x, min_y, max_x, max_y,
        spacing, 0.0, 0.0,
    )
    # Exclude points that are already seeds
    cand_indices = np.flatnonzero(cand_mask & ~ground_mask)

    if len(cand_indices) == 0:
        logger.info("PTD: all candidates are seeds (%d points)", int(ground_mask.sum()))
        return ground_mask

    # Sort candidates by Z ascending
    cand_indices = cand_indices[np.argsort(zs_f[cand_indices])]
    done = np.zeros(len(cand_indices), dtype=bool)

    sin_max_angle = math.sin(math.radians(max_angle))
    max_iterations = 50

    logger.info(
        "PTD: starting densification (%d seeds, %d candidates, spacing=%.2fm, max_d=%.2fm, max_a=%.1f°)",
        int(ground_mask.sum()), len(cand_indices), spacing, max_distance, max_angle,
    )

    # 5. Iterative TIN Densification
    for iteration in range(max_iterations):
        remaining = np.flatnonzero(~done)
        if len(remaining) == 0:
            break

        g_idx = np.flatnonzero(ground_mask)
        g_xy = np.vstack((
            np.column_stack((xs_f[g_idx], ys_f[g_idx])),
            corner_xy,
        ))
        g_z = np.concatenate((zs_f[g_idx], corner_z))

        try:
            tri = Delaunay(g_xy)
        except Exception as e:
            logger.warning("PTD: triangulation failed at iter %d: %s", iteration, e)
            break

        normals, slope_deg, max_edge = _tri_props(tri, g_xy, g_z)

        if progress:
            pct = 10.0 + 75.0 * (iteration / max_iterations)
            progress(f"PTD: iteration {iteration + 1} ({len(g_idx)} vertices)…", pct)

        rem_cand = cand_indices[remaining]
        q_xy = np.column_stack((xs_f[rem_cand], ys_f[rem_cand]))
        q_z = zs_f[rem_cand]

        s_ids = tri.find_simplex(q_xy)
        inside = s_ids >= 0

        # Freeze candidates whose containing triangle is smaller than spacing
        small = np.zeros(len(remaining), dtype=bool)
        if inside.any():
            small = inside & (max_edge[s_ids] < spacing)
            if small.any():
                done[remaining[small]] = True

        test = inside & ~small
        added = 0
        if test.any():
            t_rem = remaining[test]
            geo = _geo_query(
                tri, g_xy, g_z,
                q_xy[test], q_z[test], s_ids[test],
                normals, slope_deg, max_edge,
            )
            # Axelsson criteria with optional terrain slope adaptation for embankments:
            slopes = geo["slope_deg"]
            if follow_surface_trend:
                # Relax distance & depth thresholds on steep facets
                rel = 1.0 + 0.8 * np.tan(np.radians(np.minimum(slopes, 75.0)))
                d_ok = geo["d_perp"] <= (max_distance * rel)
                dz_ok = geo["dz"] >= (-0.50 * rel)

                # Relax angle for candidates on steep facets or short facet edges bridging convex crests
                short_crest = (geo["max_edge"] <= 8.0) & (geo["dz"] > 0.0)
                steep = (slopes > 5.0) | short_crest
                adapted_angle = np.where(short_crest, np.maximum(max_angle + slopes * 0.5, 35.0), max_angle + slopes * 0.5)
                adapted_sin = np.sin(np.radians(np.minimum(adapted_angle, 80.0)))
                eff_sin = np.where(steep, adapted_sin, sin_max_angle)
                a_ok = geo["sin_angle"] <= eff_sin
            else:
                d_ok = geo["d_perp"] <= max_distance
                dz_ok = geo["dz"] >= -0.50
                a_ok = geo["sin_angle"] <= sin_max_angle

            ok = (
                geo["inside_tri"]
                & geo["valid_denom"]
                & d_ok
                & a_ok
                & dz_ok
            )
            accept_rem = t_rem[ok]
            if len(accept_rem):
                ground_mask[cand_indices[accept_rem]] = True
                done[accept_rem] = True
                added = len(accept_rem)

        if added == 0:
            break
        # Early stop: added < 0.05% of vertex count
        if added < max(1, int(0.0005 * len(g_xy))):
            break

    # 6. Dense pass: project unclassified points against final frozen TIN
    if dense:
        if progress:
            progress("PTD: dense pass…", 88.0)
        cand_dense = np.flatnonzero(~ground_mask)
        if len(cand_dense) > 0:
            g_idx = np.flatnonzero(ground_mask)
            g_xy = np.vstack((
                np.column_stack((xs_f[g_idx], ys_f[g_idx])),
                corner_xy,
            ))
            g_z = np.concatenate((zs_f[g_idx], corner_z))
            try:
                tri = Delaunay(g_xy)
                normals, slope_deg, max_edge = _tri_props(tri, g_xy, g_z)
                q_xy = np.column_stack((xs_f[cand_dense], ys_f[cand_dense]))
                q_z = zs_f[cand_dense]
                s_ids = tri.find_simplex(q_xy)
                inside = s_ids >= 0
                if inside.any():
                    ii = np.flatnonzero(inside)
                    geo = _geo_query(
                        tri, g_xy, g_z,
                        q_xy[ii], q_z[ii], s_ids[ii],
                        normals, slope_deg, max_edge,
                    )
                    valid = geo["inside_tri"] & geo["valid_denom"]
                    slopes = geo["slope_deg"]
                    if follow_surface_trend:
                        short_crest = (geo["max_edge"] <= 8.0) & (geo["dz"] > 0.0)
                        slope_scale = 1.0 + np.tan(np.radians(np.minimum(slopes, 75.0)))
                        dense_tol = np.where(short_crest, np.maximum(dense_tolerance * slope_scale, 0.45), dense_tolerance * slope_scale)
                        below_tol = 0.50 * slope_scale
                    else:
                        dense_tol = dense_tolerance
                        below_tol = 0.50
                    below = valid & (geo["dz"] <= 0.0) & (geo["d_perp"] <= below_tol)
                    above = valid & (geo["dz"] > 0.0) & (geo["d_perp"] <= dense_tol)
                    accept_dense = below | above
                    if accept_dense.any():
                        ground_mask[cand_dense[ii[accept_dense]]] = True
            except Exception as e:
                logger.warning("PTD: dense pass failed: %s", e)

    # 7. Low outlier removal
    if remove_low_outliers:
        if progress:
            progress("PTD: removing low outliers…", 95.0)
        ground_mask = _remove_low_outliers(
            xs_f, ys_f, zs_f, ground_mask,
            threshold=low_outlier_threshold,
        )

    logger.info(
        "PTD complete: %d ground points (%.1f%%)",
        int(ground_mask.sum()), 100.0 * ground_mask.sum() / n,
    )
    return ground_mask


# Backward-compatible aliases
ground_classify_tin = ground_classify_epptd
ground_classify_epptd_two_pass = ground_classify_epptd


def _remove_low_outliers(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    ground_mask: np.ndarray,
    neighbors: int = 8,
    threshold: float = 1.0,
    passes: int = 2,
    sensor_type: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Remove low outliers from a ground mask (post-cleanup).

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
        dist_xy, nbr = tree.query(xy, k=k)
        if nbr.ndim == 1:
            nbr = nbr[:, None]
            dist_xy = dist_xy[:, None]
        nbr = nbr[:, 1:]  # drop self (column 0)
        dist_xy = dist_xy[:, 1:]
        nbr_z = zs[idx[nbr]]  # (m, k-1)
        median_z = np.median(nbr_z, axis=1)

        # Adapt threshold for local slope on embankments and steep banks
        dz = np.abs(nbr_z - zs[idx, None])
        slopes = np.divide(dz, dist_xy, out=np.zeros_like(dz), where=dist_xy > 1e-6)
        mean_slope = np.mean(slopes, axis=1)
        adaptive_thr = threshold * (1.0 + mean_slope)

        low = zs[idx] < (median_z - adaptive_thr)
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


# ─────────────────────────────────────────────────────────────────────
# Shared helpers for the PTD family (APTD / H-PTD / DL-Hybrid PTD).
# ─────────────────────────────────────────────────────────────────────

def _estimate_max_spacing(
    xs: np.ndarray,
    ys: np.ndarray,
    sample_size: int = 5000,
) -> float:
    """Estimate a robust upper bound of the point spacing (95th pctl of
    nearest-neighbour distances)."""
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
    d = dist[:, 1] if dist.ndim > 1 else dist
    return float(np.percentile(d, 95.0)) if len(d) else 1.0


def _last_return_working_mask(
    return_numbers: Optional[np.ndarray],
    num_returns: Optional[np.ndarray],
    sensor_type: Optional[np.ndarray],
    n: int,
    exclude_single_returns_in_water: bool,
) -> Optional[np.ndarray]:
    """Return the subset mask of points suitable for TIN ground extraction.

    Uses last/single returns when ``num_returns`` is meaningfully populated
    (max > 1).  Optionally drops single returns in water (bathy) so the
    shallow-water surface is not swept into the ground set.  Returns ``None``
    when no subsetting should be applied.
    """
    if return_numbers is None or num_returns is None:
        return None
    if len(return_numbers) != n or len(num_returns) != n:
        return None
    if int(np.max(num_returns)) <= 1:
        return None
    mask = return_numbers == num_returns
    if exclude_single_returns_in_water and sensor_type is not None and len(sensor_type) == n:
        bathy = np.asarray(sensor_type) == 2
        mask = mask & (~bathy | (num_returns > 1))
    if mask.sum() < 3:
        return None
    return mask


def _tri_props(tri, g_xy: np.ndarray, g_z: np.ndarray):
    """Per-triangle normals, slope (deg) and max XY edge length."""
    simplices = tri.simplices
    a = np.column_stack((g_xy[simplices[:, 0], 0], g_xy[simplices[:, 0], 1], g_z[simplices[:, 0]]))
    b = np.column_stack((g_xy[simplices[:, 1], 0], g_xy[simplices[:, 1], 1], g_z[simplices[:, 1]]))
    c = np.column_stack((g_xy[simplices[:, 2], 0], g_xy[simplices[:, 2], 1], g_z[simplices[:, 2]]))
    normal = np.cross(b - a, c - a)
    n_len = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, n_len, out=np.zeros_like(normal), where=n_len > 1e-15)
    slope_deg = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0.0, 1.0)))

    x0 = g_xy[simplices[:, 0], 0]
    y0 = g_xy[simplices[:, 0], 1]
    x1 = g_xy[simplices[:, 1], 0]
    y1 = g_xy[simplices[:, 1], 1]
    x2 = g_xy[simplices[:, 2], 0]
    y2 = g_xy[simplices[:, 2], 1]
    e01 = np.hypot(x1 - x0, y1 - y0)
    e02 = np.hypot(x2 - x0, y2 - y0)
    e12 = np.hypot(x2 - x1, y2 - y1)
    max_edge = np.maximum(np.maximum(e01, e02), e12)
    return normal, slope_deg, max_edge


def _geo_query(tri, g_xy, g_z, q_xy, q_z, s_ids, normals, slope_deg, max_edge):
    """Geometric quantities for query points relative to their TIN facet."""
    simplices = tri.simplices
    tri_s = simplices[s_ids]
    v0 = np.column_stack((g_xy[tri_s[:, 0], 0], g_xy[tri_s[:, 0], 1], g_z[tri_s[:, 0]]))
    v1 = np.column_stack((g_xy[tri_s[:, 1], 0], g_xy[tri_s[:, 1], 1], g_z[tri_s[:, 1]]))
    v2 = np.column_stack((g_xy[tri_s[:, 2], 0], g_xy[tri_s[:, 2], 1], g_z[tri_s[:, 2]]))

    denom = (
        (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0])
        + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
    )
    valid_denom = np.abs(denom) >= 1e-12
    inv_d = np.zeros_like(denom)
    inv_d[valid_denom] = 1.0 / denom[valid_denom]

    w0 = (
        (v1[:, 1] - v2[:, 1]) * (q_xy[:, 0] - v2[:, 0])
        + (v2[:, 0] - v1[:, 0]) * (q_xy[:, 1] - v2[:, 1])
    ) * inv_d
    w1 = (
        (v2[:, 1] - v0[:, 1]) * (q_xy[:, 0] - v2[:, 0])
        + (v0[:, 0] - v2[:, 0]) * (q_xy[:, 1] - v2[:, 1])
    ) * inv_d
    w2 = 1.0 - w0 - w1

    inside_tri = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    tin_z = w0 * v0[:, 2] + w1 * v1[:, 2] + w2 * v2[:, 2]
    dz = q_z - tin_z

    p = np.column_stack((q_xy, q_z))
    d_perp = np.abs(np.sum((p - v0) * normals[s_ids], axis=1))

    d_v0 = np.maximum(np.linalg.norm(p - v0, axis=1), 0.5)
    d_v1 = np.maximum(np.linalg.norm(p - v1, axis=1), 0.5)
    d_v2 = np.maximum(np.linalg.norm(p - v2, axis=1), 0.5)
    sin_angle = np.maximum(np.maximum(d_perp / d_v0, d_perp / d_v1), d_perp / d_v2)

    return {
        "inside_tri": inside_tri,
        "valid_denom": valid_denom,
        "tin_z": tin_z,
        "dz": dz,
        "d_perp": d_perp,
        "sin_angle": sin_angle,
        "slope_deg": slope_deg[s_ids],
        "max_edge": max_edge[s_ids],
    }


def _densify_core(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    seed_mask: np.ndarray,
    criterion,
    max_iterations: int = 15,
    min_added: Optional[int] = None,
    progress: ProgressCB = None,
    progress_label: str = "PTD",
) -> np.ndarray:
    """Shared iterative TIN densification loop.

    ``criterion(geo, ctx)`` receives the per-candidate geometry dict plus a
    context dict (``tri``, ``g_xy``, ``g_z``, ``q_xy``, ``q_z``, ``s_ids``,
    ``normals``, ``slope_deg``, ``max_edge``) and returns a boolean accept
    array for the inside candidates.

    ``min_added`` optionally stops the loop early: when an iteration adds
    fewer than this many points the accepted points are still applied, but
    the TIN is not rebuilt again.  PTD tails typically spend many expensive
    Delaunay rebuilds on a handful of points each; a small relative threshold
    (e.g. 0.1 % of the point count) is nearly lossless for ground results.
    """
    from scipy.spatial import Delaunay

    ground_mask = seed_mask.copy()
    n = len(xs)
    for iteration in range(max_iterations):
        if progress:
            progress(
                f"{progress_label}: iter {iteration+1}/{max_iterations} "
                f"({ground_mask.sum()} pts)…",
                20 + 60 * iteration / max_iterations,
            )
        g_xy = np.column_stack((xs[ground_mask], ys[ground_mask]))
        g_z = zs[ground_mask]
        if len(g_xy) < 3:
            break
        try:
            tri = Delaunay(g_xy)
        except Exception:
            break

        normals, slope_deg, max_edge = _tri_props(tri, g_xy, g_z)

        cand = np.where(~ground_mask)[0]
        n_cand = len(cand)
        if n_cand == 0:
            break
        q_xy = np.column_stack((xs[cand], ys[cand]))
        q_z = zs[cand]
        s_ids = tri.find_simplex(q_xy)
        inside = s_ids >= 0

        new_ground = np.zeros(n_cand, dtype=bool)
        if inside.any():
            ii = np.where(inside)[0]
            geo = _geo_query(
                tri, g_xy, g_z, q_xy[ii], q_z[ii], s_ids[ii],
                normals, slope_deg, max_edge,
            )
            ctx = {
                "tri": tri,
                "g_xy": g_xy,
                "g_z": g_z,
                "q_xy": q_xy[ii],
                "q_z": q_z[ii],
                "s_ids": s_ids[ii],
                "normals": normals,
                "slope_deg": slope_deg,
                "max_edge": max_edge,
            }
            new_ground[ii] = criterion(geo, ctx)

        added = new_ground.sum()
        if added == 0:
            break
        ground_mask[cand[new_ground]] = True
        if min_added is not None and added < min_added:
            break

    return ground_mask


def _grid_lowest_seeds(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    cell_size: float,
) -> Tuple[np.ndarray, float]:
    """Regular-grid seed selection: lowest Z per cell.

    Returns ``(seed_indices, max_grid_slope)`` where ``max_grid_slope`` is
    the steepest per-cell slope encountered (used by AGPTD for its terrain
    slope threshold).
    """
    n = len(xs)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)

    gx = np.clip(((xs - min_x) / cell_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys - min_y) / cell_size).astype(np.int32), 0, ny - 1)
    flat = gx * ny + gy

    order = np.argsort(flat)
    sorted_flat = flat[order]
    sorted_z = zs[order]
    unique_cells, starts = np.unique(sorted_flat, return_index=True)
    ends = np.append(starts[1:], n)

    seed_indices = []
    max_slope = 0.0
    for i, cell in enumerate(unique_cells):
        s, e = starts[i], ends[i]
        local = order[s:e]
        lowest_local = int(np.argmin(sorted_z[s:e]))
        seed_idx = local[lowest_local]
        seed_indices.append(seed_idx)

        # Max per-cell slope from the lowest point (for AGPTD terrain angle).
        if len(local) > 1:
            dz = np.abs(zs[local] - zs[seed_idx])
            dist = np.hypot(xs[local] - xs[seed_idx], ys[local] - ys[seed_idx])
            slopes = np.divide(dz, dist, out=np.zeros_like(dz), where=dist > 1e-9)
            if len(slopes):
                max_slope = max(max_slope, float(slopes.max()))

    return np.asarray(seed_indices, dtype=np.int64), max_slope


# ─────────────────────────────────────────────────────────────────────
# APTD — Adaptive Grid Progressive TIN Densification (AGPTD).
# Zheng, Xiang, Zhang & Zhou, Remote Sensing 2024, 16, 3846.
# ─────────────────────────────────────────────────────────────────────

def ground_classify_aptd(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    k_neighbors: int = 4,
    radius_factor: float = 4.0,
    min_neighbors: int = 2,
    primary_grid_size: Optional[float] = None,
    point_count_threshold: int = 5,
    slope_threshold: float = 0.5,
    max_angle: Optional[float] = None,
    max_distance: Optional[float] = None,
    max_terrain_angle: Optional[float] = None,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    sensor_type: Optional[np.ndarray] = None,
    exclude_single_returns_in_water: bool = False,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    Adaptive Grid-based PTD (AGPTD) ground classification (Zheng et al. 2024).

    Improvements over the classical PTD:

      1. **Outlier removal** — a radius-outlier pass (``radius_factor`` ×
         point spacing, ``min_neighbors``) plus a Kd-tree elevation-statistics
         pass remove low/high/isolated outliers before seeding, so low points
         cannot become seeds.
      2. **Adaptive two-level grid** — a primary grid is refined into a
         secondary grid (half size) where the cell holds enough points
         (``point_count_threshold``) and the relative slope exceeds
         ``slope_threshold``, yielding denser seeds on steep/disconnected
         terrain.
      3. **Adaptive thresholds + mirroring** — the terrain-slope threshold is
         taken from the steepest grid slope; ``max_angle``/``max_distance``
         default from point spacing; and steep TIN facets use the mirroring
         technique to preserve disconnected terrain.

    Args:
        k_neighbors:            Neighbours for the Kd-tree elevation check.
        radius_factor:          Radius = factor × max point spacing.
        min_neighbors:          Min neighbours within radius (else outlier).
        primary_grid_size:      Primary grid cell size (auto from spacing).
        point_count_threshold:  Min points in a cell to consider refinement.
        slope_threshold:        Relative-slope threshold to refine a cell.
        max_angle, max_distance, max_terrain_angle:
                                Densification thresholds (auto when ``None``).
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    work = _last_return_working_mask(
        return_numbers, num_returns, sensor_type, n, exclude_single_returns_in_water
    )
    if work is not None and work.sum() < n:
        sub_map = np.where(work)[0]
        sub = ground_classify_aptd(
            xs[work], ys[work], zs[work],
            k_neighbors=k_neighbors, radius_factor=radius_factor,
            min_neighbors=min_neighbors, primary_grid_size=primary_grid_size,
            point_count_threshold=point_count_threshold,
            slope_threshold=slope_threshold,
            max_angle=max_angle, max_distance=max_distance,
            max_terrain_angle=max_terrain_angle, progress=progress,
        )
        full = np.zeros(n, dtype=bool)
        full[sub_map[sub]] = True
        return full

    return _aptd_impl(
        xs, ys, zs,
        k_neighbors=k_neighbors, radius_factor=radius_factor,
        min_neighbors=min_neighbors, primary_grid_size=primary_grid_size,
        point_count_threshold=point_count_threshold,
        slope_threshold=slope_threshold,
        max_angle=max_angle, max_distance=max_distance,
        max_terrain_angle=max_terrain_angle, progress=progress,
    )


def _aptd_impl(
    xs, ys, zs,
    k_neighbors, radius_factor, min_neighbors, primary_grid_size,
    point_count_threshold, slope_threshold,
    max_angle, max_distance, max_terrain_angle, progress,
) -> np.ndarray:
    import math
    import os

    n = len(xs)
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return np.ones(n, dtype=bool)

    if progress:
        progress("APTD: estimating spacing…", 5.0)

    spacing = compute_adaptive_spacing(xs, ys)
    max_spacing = _estimate_max_spacing(xs, ys)

    # ── Step 1a: radius outlier removal ────────────────────────────
    radius = radius_factor * max_spacing
    xy = np.column_stack((xs, ys))
    tree = KDTree(xy)
    n_workers = max(1, min((os.cpu_count() or 1), 8))
    counts = tree.query_ball_point(xy, radius, return_length=True, workers=n_workers)
    keep = counts >= max(1, min_neighbors)

    if progress:
        progress(f"APTD: radius outliers removed ({int((~keep).sum())})…", 12.0)

    xs_k = xs[keep]
    ys_k = ys[keep]
    zs_k = zs[keep]
    keep_map = np.where(keep)[0]

    # ── Step 1b: Kd-tree elevation-statistics outlier removal ─────
    if len(xs_k) >= 3:
        tree_k = KDTree(np.column_stack((xs_k, ys_k)))
        kk = min(k_neighbors + 1, len(xs_k))
        _, nbr = tree_k.query(np.column_stack((xs_k, ys_k)), k=kk, workers=n_workers)
        if nbr.ndim == 1:
            nbr = nbr[:, None]
        nbr = nbr[:, 1:]
        nbr_z = zs_k[nbr]
        med = np.median(nbr_z, axis=1)
        mad = np.median(np.abs(nbr_z - med[:, None]), axis=1) * 1.4826
        floor = max(0.5, spacing * 0.5)
        thr = np.maximum(mad * 3.0, floor)
        z_ok = np.abs(zs_k - med) <= thr
        if progress:
            progress(f"APTD: elevation outliers removed ({int((~z_ok).sum())})…", 18.0)
    else:
        z_ok = np.ones(len(xs_k), dtype=bool)

    keep2 = keep_map[z_ok]
    xs2 = xs[keep2]
    ys2 = ys[keep2]
    zs2 = zs[keep2]

    # ── Step 2: adaptive two-level grid seeding ────────────────────
    s_pri = primary_grid_size or max(spacing * 4.0, 5.0)
    min_x, max_x = xs2.min(), xs2.max()
    min_y, max_y = ys2.min(), ys2.max()
    nx = max(1, int(np.ceil((max_x - min_x) / s_pri)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / s_pri)) + 1)

    gx = np.clip(((xs2 - min_x) / s_pri).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys2 - min_y) / s_pri).astype(np.int32), 0, ny - 1)
    flat = gx * ny + gy

    order = np.argsort(flat)
    sorted_flat = flat[order]
    sorted_z = zs2[order]
    unique_cells, starts = np.unique(sorted_flat, return_index=True)
    ends = np.append(starts[1:], len(xs2))

    primary_seeds = []
    secondary_seeds = []
    max_grid_slope = 0.0

    for i in range(len(unique_cells)):
        s, e = starts[i], ends[i]
        local = order[s:e]
        lowest_local = int(np.argmin(sorted_z[s:e]))
        seed_idx = local[lowest_local]
        primary_seeds.append(seed_idx)

        n_pp = len(local)
        if n_pp < point_count_threshold:
            continue

        dz = zs2[local] - zs2[seed_idx]
        dist = np.hypot(xs2[local] - xs2[seed_idx], ys2[local] - ys2[seed_idx])
        slopes = np.divide(dz, dist, out=np.zeros_like(dz), where=dist > 1e-9)
        s_a = float(slopes.mean()) if len(slopes) else 0.0
        s_min = float(slopes.min()) if len(slopes) else 0.0
        max_grid_slope = max(max_grid_slope, s_a)
        rel_slope = s_a - s_min

        if rel_slope <= slope_threshold:
            continue

        # Refine into a secondary grid (half size).
        s_sec = s_pri / 2.0
        sgx = np.clip(((xs2[local] - min_x) / s_sec).astype(np.int32), 0, 2 * nx)
        sgy = np.clip(((ys2[local] - min_y) / s_sec).astype(np.int32), 0, 2 * ny)
        sflat = sgx * (2 * ny + 1) + sgy
        sorder = np.argsort(sflat)
        ssorted_flat = sflat[sorder]
        suniq, sstarts = np.unique(ssorted_flat, return_index=True)
        sends = np.append(sstarts[1:], len(local))
        for j in range(len(suniq)):
            sl = sorder[sstarts[j]:sends[j]]
            s_low = int(np.argmin(zs2[local[sl]]))
            secondary_seeds.append(local[sl[s_low]])

    all_seeds = np.unique(np.concatenate([primary_seeds, secondary_seeds]).astype(np.int64))
    if len(all_seeds) < 3:
        logger.warning("APTD: too few seed points")
        return np.ones(n, dtype=bool)

    if progress:
        progress(f"APTD: {len(all_seeds)} adaptive seeds (cell={s_pri:.1f}m)…", 25.0)

    # ── Step 3: densification with mirroring ───────────────────────
    if max_angle is None:
        max_angle = 12.0
    if max_distance is None:
        max_distance = max(spacing, 0.6)
    if max_terrain_angle is None:
        max_terrain_angle = math.degrees(math.atan(max_grid_slope)) if max_grid_slope > 0 else 88.0
    max_terrain_angle = min(max_terrain_angle, 89.5)

    sin_max_angle = math.sin(math.radians(max_angle))

    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[all_seeds] = True

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    def _mirror_check(ctx, s_idx, d_max, sin_a):
        """Mirroring technique (AGPTD): for steep facets, judge the mirror
        point reflected across the facet's highest vertex (XY only, Z kept)."""
        out = np.zeros(len(s_idx), dtype=bool)
        tri = ctx["tri"]
        g_xy = ctx["g_xy"]
        g_z = ctx["g_z"]
        tri_s = tri.simplices[ctx["s_ids"][s_idx]]
        q_xy = ctx["q_xy"][s_idx]
        q_z = ctx["q_z"][s_idx]

        z0 = g_z[tri_s[:, 0]]
        z1 = g_z[tri_s[:, 1]]
        z2 = g_z[tri_s[:, 2]]
        hv = np.argmax(np.stack([z0, z1, z2], axis=1), axis=1)
        vx = np.where(hv == 0, g_xy[tri_s[:, 0], 0],
             np.where(hv == 1, g_xy[tri_s[:, 1], 0], g_xy[tri_s[:, 2], 0]))
        vy = np.where(hv == 0, g_xy[tri_s[:, 0], 1],
             np.where(hv == 1, g_xy[tri_s[:, 1], 1], g_xy[tri_s[:, 2], 1]))
        mx = 2.0 * vx - q_xy[:, 0]
        my = 2.0 * vy - q_xy[:, 1]
        m_xy = np.column_stack((mx, my))

        m_sids = tri.find_simplex(m_xy)
        m_inside = m_sids >= 0
        if not m_inside.any():
            return out
        mi = np.where(m_inside)[0]
        geo_m = _geo_query(
            tri, g_xy, g_z, m_xy[mi], q_z[mi], m_sids[mi],
            ctx["normals"], ctx["slope_deg"], ctx["max_edge"],
        )
        ok = (
            geo_m["inside_tri"] & geo_m["valid_denom"]
            & (geo_m["d_perp"] <= d_max) & (geo_m["sin_angle"] <= sin_a)
        )
        out[mi[ok]] = True
        return out

    def criterion(geo, ctx):
        base = geo["inside_tri"] & geo["valid_denom"]
        d_ok = geo["d_perp"] <= max_distance
        a_ok = geo["sin_angle"] <= sin_max_angle
        ok = base & d_ok & a_ok
        steep = base & (geo["slope_deg"] > max_terrain_angle)
        if steep.any():
            s_idx = np.where(steep & ~ok)[0]
            if len(s_idx):
                mir_ok = _mirror_check(ctx, s_idx, max_distance, sin_max_angle)
                ok[s_idx] = ok[s_idx] | mir_ok
        return ok

    ground_mask = _densify_core(
        xs_f, ys_f, zs_f, seed_mask, criterion,
        progress=progress, progress_label="APTD",
        min_added=max(20, int(n * 0.001)),
    )

    # Removed outliers are never ground (AGPTD removes them up front).
    removed = np.ones(n, dtype=bool)
    removed[keep2] = False
    ground_mask[removed] = False

    logger.info(
        "APTD ground (cell=%.1f, d=%.2f, a=%.1f, terr_a=%.1f): %d ground / %d points",
        s_pri, max_distance, max_angle, max_terrain_angle,
        ground_mask.sum(), n,
    )
    return ground_mask


# ─────────────────────────────────────────────────────────────────────
# H-PTD — Hierarchical/Fast PTD using adjacent surface information (FPTD).
# Li, Ye, Guo, Wei, Wang & Li, IEEE JSTARS 2021, DOI 10.1109/JSTARS.2021.3131586.
# ─────────────────────────────────────────────────────────────────────

def ground_classify_hptd(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    window_size: Optional[float] = None,
    step_factor: float = 0.5,
    max_angle: float = 6.0,
    max_distance: Optional[float] = None,
    relative_elevation_factor: float = 1.0,
    signed: bool = True,
    below_tolerance: float = 0.5,
    max_terrain_angle: float = 88.0,
    follow_surface_trend: bool = True,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    sensor_type: Optional[np.ndarray] = None,
    exclude_single_returns_in_water: bool = False,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    Hierarchical / Fast PTD (FPTD) ground classification (Li et al. 2021).

    Faithful implementation of the paper's three improvements:

      1. **Sliding-window seeds** — a window (``window_size``) slides in
         ``step_factor`` increments, taking the lowest point of each window
         position.  This yields far denser, better-distributed seeds than a
         fixed regular grid, especially on steep slopes.
      2. **Signed computation** — points sitting more than ``below_tolerance``
         below the local TIN surface are rejected (low outliers / avoidable
         non-ground points).  On steep facets the tolerance is scaled by the
         same relative-elevation factor so embankment points are not rejected
         while the TIN is still climbing the slope.
      3. **Relative elevation threshold** — the allowed elevation offset is
         not absolute but relaxes with the local facet slope via
         ``relative_elevation_factor``.
      4. **Surface-trend relaxation** — when ``follow_surface_trend`` is set,
         the iteration angle is relaxed on steep facets (slope > 10°) for
         points *above* the TIN, which is essential for river embankments,
         cliffs and road ramps.  ``max_terrain_angle`` still bounds the
         steepest acceptable TIN facet (88° by default, like EP-PTD).

    The overlapping sliding windows span cell boundaries, which is how
    adjacent-surface information is incorporated into each facet.  True
    per-block multithreading is handled by the GUI's process pool (one
    process per tile).
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    work = _last_return_working_mask(
        return_numbers, num_returns, sensor_type, n, exclude_single_returns_in_water
    )
    if work is not None and work.sum() < n:
        sub_map = np.where(work)[0]
        sub = ground_classify_hptd(
            xs[work], ys[work], zs[work],
            window_size=window_size, step_factor=step_factor,
            max_angle=max_angle, max_distance=max_distance,
            relative_elevation_factor=relative_elevation_factor,
            signed=signed, below_tolerance=below_tolerance,
            max_terrain_angle=max_terrain_angle,
            follow_surface_trend=follow_surface_trend,
            progress=progress,
        )
        full = np.zeros(n, dtype=bool)
        full[sub_map[sub]] = True
        return full

    return _hptd_impl(
        xs, ys, zs,
        window_size=window_size, step_factor=step_factor,
        max_angle=max_angle, max_distance=max_distance,
        relative_elevation_factor=relative_elevation_factor,
        signed=signed, below_tolerance=below_tolerance,
        max_terrain_angle=max_terrain_angle,
        follow_surface_trend=follow_surface_trend,
        progress=progress,
    )


def _hptd_impl(
    xs, ys, zs, window_size, step_factor, max_angle, max_distance,
    relative_elevation_factor, signed, below_tolerance,
    max_terrain_angle, follow_surface_trend, progress,
) -> np.ndarray:
    import math

    n = len(xs)
    try:
        from scipy.spatial import Delaunay as _Delaunay  # noqa: F401
    except ImportError:
        return np.ones(n, dtype=bool)

    if progress:
        progress("H-PTD: estimating spacing…", 5.0)

    spacing = compute_adaptive_spacing(xs, ys)
    w = window_size or max(spacing * 6.0, 10.0)
    step = max(w * step_factor, spacing)

    if max_distance is None:
        max_distance = max(spacing * 1.5, 0.6)

    sin_max_angle = math.sin(math.radians(max_angle))

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    # ── Sliding-window seed selection ──────────────────────────────
    # Bin points to a fine grid of ``step`` size; each sliding window covers
    # a block of fine cells, and the lowest point of every window position
    # becomes a seed.  This gives dense, overlapping seeds.
    fine = step
    fnx = max(1, int(np.ceil((max_x - min_x) / fine)) + 1)
    fny = max(1, int(np.ceil((max_y - min_y) / fine)) + 1)
    gx = np.clip(((xs - min_x) / fine).astype(np.int32), 0, fnx - 1)
    gy = np.clip(((ys - min_y) / fine).astype(np.int32), 0, fny - 1)
    flat = gx * fny + gy

    order = np.argsort(flat)
    sorted_flat = flat[order]
    sorted_z = zs[order]
    unique_cells, starts = np.unique(sorted_flat, return_index=True)
    ends = np.append(starts[1:], n)

    # Lowest point per fine cell.
    fine_min_idx = np.full(len(unique_cells), -1, dtype=np.int64)
    fine_min_z = np.full(len(unique_cells), np.inf)
    for i in range(len(unique_cells)):
        s, e = starts[i], ends[i]
        local = order[s:e]
        j = int(np.argmin(sorted_z[s:e]))
        fine_min_idx[i] = local[j]
        fine_min_z[i] = sorted_z[s:e][j]

    # Map each fine cell to its (cx, cy).
    cell_x = np.asarray([int(c) // fny for c in unique_cells], dtype=np.int32)
    cell_y = np.asarray([int(c) % fny for c in unique_cells], dtype=np.int32)

    # Sliding window = ``win_cells`` fine cells per side.
    win_cells = max(2, int(round(w / fine)))
    seeds = []
    seen = set()
    max_cx = cell_x.max() if len(cell_x) else 0
    max_cy = cell_y.max() if len(cell_y) else 0
    for ox in range(0, max_cx + 1):
        for oy in range(0, max_cy + 1):
            x0, x1 = ox, ox + win_cells
            y0, y1 = oy, oy + win_cells
            sel = (cell_x >= x0) & (cell_x < x1) & (cell_y >= y0) & (cell_y < y1)
            if not sel.any():
                continue
            j = int(np.argmin(fine_min_z[sel]))
            idx = fine_min_idx[sel][j]
            if int(idx) not in seen:
                seen.add(int(idx))
                seeds.append(int(idx))

    if len(seeds) < 3:
        # Fall back to a plain regular-grid seed selection.
        seed_arr, _ = _grid_lowest_seeds(xs, ys, zs, w)
        seeds = seed_arr.tolist()

    if len(seeds) < 3:
        logger.warning("H-PTD: too few seed points")
        return np.ones(n, dtype=bool)

    if progress:
        progress(f"H-PTD: {len(seeds)} sliding-window seeds (w={w:.1f}m)…", 20.0)

    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[np.asarray(seeds, dtype=np.int64)] = True

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    def criterion(geo, ctx):
        base = geo["inside_tri"] & geo["valid_denom"]
        slope_deg = geo["slope_deg"]
        slope_ok = slope_deg <= max_terrain_angle

        # Relative elevation threshold (relaxes with facet slope).
        rel = 1.0 + relative_elevation_factor * np.tan(
            np.radians(np.minimum(slope_deg, 80.0))
        )
        dz_ok = np.abs(geo["dz"]) <= max_distance * rel
        if signed:
            # Scale the below-surface tolerance with the same relative factor
            # so embankment points are not rejected while the TIN is still
            # climbing a steep slope.
            dz_ok = dz_ok & (geo["dz"] >= -below_tolerance * rel)

        a_ok = geo["sin_angle"] <= sin_max_angle

        ok = base & dz_ok & a_ok & slope_ok

        # Follow surface trend: on steep facets, relax the iteration angle
        # for points above the TIN (uphill) so the TIN can climb embankments.
        if follow_surface_trend:
            steep = base & (slope_deg > 10.0)
            if steep.any():
                st = np.where(steep)[0]
                if len(st):
                    adapted_angle = max_angle + slope_deg[st] * 0.5
                    adapted_sin = np.sin(np.radians(adapted_angle))
                    uphill = geo["dz"][st] > 0
                    u = st[uphill]
                    if len(u):
                        ok[u] = ok[u] | (
                            geo["sin_angle"][u] <= adapted_sin[uphill]
                        )

        return ok

    ground_mask = _densify_core(
        xs_f, ys_f, zs_f, seed_mask, criterion,
        progress=progress, progress_label="H-PTD",
    )

    logger.info(
        "H-PTD ground (w=%.1f, d=%.2f, a=%.1f): %d ground / %d points",
        w, max_distance, max_angle, ground_mask.sum(), n,
    )
    return ground_mask


# ─────────────────────────────────────────────────────────────────────
# DL-Hybrid PTD — Pointcept deep-learning prior + PTD densification.
# ─────────────────────────────────────────────────────────────────────

def ground_classify_dl_hybrid_ptd(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    dl_ground_confidence: Optional[np.ndarray] = None,
    dl_ground_mask: Optional[np.ndarray] = None,
    dl_confidence_threshold: float = 0.5,
    dl_seed_weight: float = 1.0,
    max_distance: float = 1.4,
    max_angle: float = 6.0,
    max_terrain_angle: float = 88.0,
    max_building_size: Optional[float] = None,
    only_upward: bool = False,
    follow_surface_trend: bool = True,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    sensor_type: Optional[np.ndarray] = None,
    exclude_single_returns_in_water: bool = False,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    DL-Hybrid PTD ground classification.

    Combines a deep-learning prior (e.g. Pointcept's ASPRS class-2 "ground"
    predictions) with geometric PTD densification:

      1. Seeds come from BOTH the lowest-Z-per-cell (geometric) and the
         lowest-Z-per-cell among DL-trusted ground points — so the DL model's
         ground calls help anchor the TIN in vegetated/steep areas.
      2. The densification distance threshold is relaxed proportionally to the
         per-point DL confidence (``dl_seed_weight``), trusting the model's
         ground calls while geometry still rejects clear non-ground points.

    Args:
        dl_ground_confidence:  Per-point DL ground probability in [0, 1]
                               (e.g. Pointcept softmax ground channel).
        dl_ground_mask:        Boolean per-point DL ground mask.  Used when
                               ``dl_ground_confidence`` is ``None``.
        dl_confidence_threshold: Minimum confidence to treat a point as
                               DL ground (for seeding and threshold relax).
        dl_seed_weight:        How strongly the DL confidence relaxes the
                               densification distance (0 = pure geometric).
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    # Build a per-point confidence array (0..1).
    if dl_ground_confidence is not None and len(dl_ground_confidence) == n:
        dl_conf = np.clip(np.asarray(dl_ground_confidence, dtype=np.float64), 0.0, 1.0)
    elif dl_ground_mask is not None and len(dl_ground_mask) == n:
        dl_conf = np.asarray(dl_ground_mask, dtype=np.float64)
    else:
        dl_conf = np.zeros(n, dtype=np.float64)

    work = _last_return_working_mask(
        return_numbers, num_returns, sensor_type, n, exclude_single_returns_in_water
    )
    if work is not None and work.sum() < n:
        sub_map = np.where(work)[0]
        sub = ground_classify_dl_hybrid_ptd(
            xs[work], ys[work], zs[work],
            dl_ground_confidence=dl_conf[work],
            dl_confidence_threshold=dl_confidence_threshold,
            dl_seed_weight=dl_seed_weight,
            max_distance=max_distance, max_angle=max_angle,
            max_terrain_angle=max_terrain_angle,
            max_building_size=max_building_size,
            only_upward=only_upward,
            follow_surface_trend=follow_surface_trend,
            progress=progress,
        )
        full = np.zeros(n, dtype=bool)
        full[sub_map[sub]] = True
        return full

    return _dl_hybrid_impl(
        xs, ys, zs, dl_conf,
        dl_confidence_threshold=dl_confidence_threshold,
        dl_seed_weight=dl_seed_weight,
        max_distance=max_distance, max_angle=max_angle,
        max_terrain_angle=max_terrain_angle,
        max_building_size=max_building_size,
        only_upward=only_upward,
        follow_surface_trend=follow_surface_trend,
        progress=progress,
    )


def _dl_hybrid_impl(
    xs, ys, zs, dl_conf,
    dl_confidence_threshold, dl_seed_weight,
    max_distance, max_angle, max_terrain_angle, max_building_size,
    only_upward, follow_surface_trend, progress,
) -> np.ndarray:
    import math

    n = len(xs)
    try:
        from scipy.spatial import Delaunay as _Delaunay  # noqa: F401
    except ImportError:
        return np.ones(n, dtype=bool)

    spacing = compute_adaptive_spacing(xs, ys)

    if max_building_size is not None:
        seed_cell = max_building_size
    else:
        seed_cell = max(spacing * 10, 5.0)

    # Geometric seeds: lowest Z per cell.
    geo_seeds, _ = _grid_lowest_seeds(xs, ys, zs, seed_cell)

    # DL seeds: lowest Z per cell among DL-trusted ground points.
    dl_trusted = dl_conf >= dl_confidence_threshold
    dl_seeds = np.array([], dtype=np.int64)
    if dl_trusted.any():
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        nx = max(1, int(np.ceil((max_x - min_x) / seed_cell)) + 1)
        ny = max(1, int(np.ceil((max_y - min_y) / seed_cell)) + 1)
        gx = np.clip(((xs[dl_trusted] - min_x) / seed_cell).astype(np.int32), 0, nx - 1)
        gy = np.clip(((ys[dl_trusted] - min_y) / seed_cell).astype(np.int32), 0, ny - 1)
        flat = gx * ny + gy
        order = np.argsort(flat)
        sorted_flat = flat[order]
        dl_idx = np.where(dl_trusted)[0][order]
        sorted_z = zs[dl_idx]
        uniq, starts = np.unique(sorted_flat, return_index=True)
        ends = np.append(starts[1:], len(dl_idx))
        dl_seeds = np.array([
            dl_idx[s + int(np.argmin(sorted_z[s:e]))]
            for s, e in zip(starts, ends)
        ], dtype=np.int64)

    seeds = np.unique(np.concatenate([geo_seeds, dl_seeds]))
    if len(seeds) < 3:
        logger.warning("DL-Hybrid PTD: too few seed points")
        return np.ones(n, dtype=bool)

    if progress:
        progress(f"DL-Hybrid: {len(seeds)} seeds (DL={len(dl_seeds)}, geo={len(geo_seeds)})…", 20.0)

    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[seeds] = True

    max_angle_rad = math.radians(max_angle)
    sin_max_angle = math.sin(max_angle_rad)

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)
    dl_conf_f = np.asarray(dl_conf, dtype=np.float64)

    # Optional only-upward seed surface (per-cell min Z).
    if only_upward:
        seed_z_surface = _seed_cell_min_surface(xs, ys, zs, seed_cell)
    else:
        seed_z_surface = None

    def criterion(geo, ctx):
        base = geo["inside_tri"] & geo["valid_denom"]
        slope_ok = geo["slope_deg"] <= max_terrain_angle
        # Relax distance for DL-trusted points.
        conf = dl_conf_f[ctx["cand_idx"]]
        d_allow = max_distance * (1.0 + dl_seed_weight * conf)
        d_ok = geo["d_perp"] <= d_allow
        a_ok = geo["sin_angle"] <= sin_max_angle

        ok = base & d_ok & a_ok & slope_ok

        if follow_surface_trend:
            steep = geo["slope_deg"] > 10.0
            if steep.any():
                st = np.where(base & steep)[0]
                if len(st):
                    adapted = max_angle + geo["slope_deg"][st] * 0.5
                    adapted_sin = np.sin(np.radians(adapted))
                    uphill = geo["dz"][st] > 0
                    u = st[uphill]
                    if len(u):
                        ok[u] = ok[u] | (geo["sin_angle"][u] <= adapted_sin[uphill])

        if only_upward and seed_z_surface is not None:
            ok = ok & (ctx["q_z"] >= seed_z_surface[ctx["cand_idx"]])

        return ok

    ground_mask = _densify_core_with_conf(
        xs_f, ys_f, zs_f, seed_mask, criterion,
        cand_conf=dl_conf_f, progress=progress, progress_label="DL-Hybrid PTD",
    )

    logger.info(
        "DL-Hybrid PTD ground (d=%.2f, a=%.1f, dl_weight=%.1f): %d ground / %d points",
        max_distance, max_angle, dl_seed_weight, ground_mask.sum(), n,
    )
    return ground_mask


def _seed_cell_min_surface(xs, ys, zs, cell_size):
    n = len(xs)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)
    gx = np.clip(((xs - min_x) / cell_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys - min_y) / cell_size).astype(np.int32), 0, ny - 1)
    flat = gx * ny + gy
    surface = np.full(n, np.inf)
    order = np.argsort(flat)
    sorted_flat = flat[order]
    sorted_z = zs[order]
    uniq, starts = np.unique(sorted_flat, return_index=True)
    ends = np.append(starts[1:], n)
    for s, e in zip(starts, ends):
        cell_min = sorted_z[s:e].min()
        surface[order[s:e]] = cell_min
    return surface


def _densify_core_with_conf(
    xs, ys, zs, seed_mask, criterion, cand_conf, max_iterations=15,
    progress=None, progress_label="PTD",
) -> np.ndarray:
    """Like :func:`_densify_core` but threads per-point confidence through
    ``ctx["cand_idx"]`` / ``ctx["conf"]`` for DL-hybrid criteria."""
    from scipy.spatial import Delaunay as _D

    ground_mask = seed_mask.copy()
    n = len(xs)
    for iteration in range(max_iterations):
        if progress:
            progress(
                f"{progress_label}: iter {iteration+1}/{max_iterations} "
                f"({ground_mask.sum()} pts)…",
                20 + 60 * iteration / max_iterations,
            )
        g_xy = np.column_stack((xs[ground_mask], ys[ground_mask]))
        g_z = zs[ground_mask]
        if len(g_xy) < 3:
            break
        try:
            tri = _D(g_xy)
        except Exception:
            break
        normals, slope_deg, max_edge = _tri_props(tri, g_xy, g_z)

        cand = np.where(~ground_mask)[0]
        n_cand = len(cand)
        if n_cand == 0:
            break
        q_xy = np.column_stack((xs[cand], ys[cand]))
        q_z = zs[cand]
        s_ids = tri.find_simplex(q_xy)
        inside = s_ids >= 0
        new_ground = np.zeros(n_cand, dtype=bool)
        if inside.any():
            ii = np.where(inside)[0]
            geo = _geo_query(
                tri, g_xy, g_z, q_xy[ii], q_z[ii], s_ids[ii],
                normals, slope_deg, max_edge,
            )
            ctx = {
                "tri": tri, "g_xy": g_xy, "g_z": g_z,
                "q_xy": q_xy[ii], "q_z": q_z[ii], "s_ids": s_ids[ii],
                "normals": normals, "slope_deg": slope_deg, "max_edge": max_edge,
                "cand_idx": cand[ii],
            }
            new_ground[ii] = criterion(geo, ctx)
        added = new_ground.sum()
        if added == 0:
            break
        ground_mask[cand[new_ground]] = True
    return ground_mask


# ─────────────────────────────────────────────────────────────────────
# Multiscale 3D Alpha Shape Filtering (MDPI Remote Sensing 2024, 16(8), 1443)
# ─────────────────────────────────────────────────────────────────────

def _extract_alpha_shape_bottom_layer(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    alpha: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> np.ndarray:
    """Extract lower boundary points using a bottom rolling sphere of radius alpha.

    Implements the modified 3D alpha shape lower envelope test:
    A sphere of radius alpha placed beneath point P tests if any neighbor
    point penetrates the sphere from below. Points where the sphere is clear
    belong to the lower alpha-shape boundary.
    """
    n = len(xs)
    cell_size = max(alpha * 0.4, 0.5)
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)

    gx = np.clip(((xs - min_x) / cell_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys - min_y) / cell_size).astype(np.int32), 0, ny - 1)
    flat_idx = gx * ny + gy

    # Find minimum Z in each cell
    cell_min = np.full(nx * ny, np.inf, dtype=np.float64)
    np.minimum.at(cell_min, flat_idx, zs)
    cand_mask = (zs <= cell_min[flat_idx] + 0.05)
    cand_idx = np.flatnonzero(cand_mask)
    if len(cand_idx) == 0:
        return np.zeros(n, dtype=bool)

    # KDTree query to test rolling sphere
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack((xs, ys)))
        # Search radius = min(alpha * 1.5, 30.0)
        search_r = min(alpha * 1.4, 35.0)
        neighbors_list = tree.query_ball_point(
            np.column_stack((xs[cand_idx], ys[cand_idx])), r=search_r
        )
        is_bottom = np.ones(len(cand_idx), dtype=bool)
        alpha2 = alpha * alpha
        two_alpha = 2.0 * alpha

        for i, (ci, nbrs) in enumerate(zip(cand_idx, neighbors_list)):
            if len(nbrs) <= 1:
                continue
            nbrs_arr = np.asarray(nbrs)
            dx = xs[nbrs_arr] - xs[ci]
            dy = ys[nbrs_arr] - ys[ci]
            dz = zs[ci] - zs[nbrs_arr]  # positive when neighbor is below ci
            d_xy2 = dx * dx + dy * dy

            # Point penetrates bottom rolling sphere if:
            # (z - (z0 - alpha))^2 + d_xy^2 < alpha^2 and neighbor is below
            below = dz > 1e-4
            if below.any():
                dz_b = dz[below]
                d_xy2_b = d_xy2[below]
                # Sphere condition: d_xy2 + (alpha - dz)^2 < alpha^2 <=> d_xy2 + dz^2 < 2*alpha*dz
                violates = (d_xy2_b + dz_b * dz_b) < (two_alpha * dz_b)
                if violates.any():
                    is_bottom[i] = False

        bottom_mask = np.zeros(n, dtype=bool)
        bottom_mask[cand_idx[is_bottom]] = True
        return bottom_mask
    except Exception:
        # Fallback to local grid minimums
        return cand_mask


def ground_classify_multiscale_alpha_shape(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    coarse_alpha: float = 20.0,
    medium_alpha: float = 6.0,
    fine_alpha: float = 2.0,
    max_distance: float = 1.0,
    max_angle: float = 8.0,
    max_terrain_angle: float = 88.0,
    exclude_single_returns_in_water: bool = True,
    sensor_type: Optional[np.ndarray] = None,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    Multiscale 3D Alpha Shape ground filtering for airborne LiDAR data.

    Reference:
        "A Multiscale Filtering Method for Airborne LiDAR Data Using Modified 3D Alpha Shape",
        Remote Sensing 2024, 16(8), 1443.

    Methodology:
      1. Preprocessing: Fast outlier removal and dual-sensor water surface/column exclusion.
      2. Multiscale 3D Alpha Shape extraction:
         - Coarse scale (alpha1 = 20 m): bridges across large buildings, vegetation canopies,
           and water surface gaps, isolating large-scale terrain seeds.
         - Medium scale (alpha2 = 6 m): penetrates into valleys, gullies, and breaklines.
         - Fine scale (alpha3 = 2 m): captures micro-topography, steep riverbanks, and riverbeds.
      3. Multiscale Progressive TIN Densification:
         - Initializes TIN from coarse alpha-shape seeds.
         - Progressively densifies medium and fine alpha-shape layers with slope trend adaptation
           and FPTD relative elevation caps (z_u - z_max <= max_distance).
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    min_x, max_x = float(xs_f.min()), float(xs_f.max())
    min_y, max_y = float(ys_f.min()), float(ys_f.max())

    # Preprocessing: water surface and water-column exclusion
    water_mask = None
    keep_mask = np.ones(n, dtype=bool)
    if exclude_single_returns_in_water:
        water_mask = _detect_water_surface_and_column(
            xs_f, ys_f, zs_f,
            sensor_type=sensor_type,
            return_numbers=return_numbers,
            num_returns=num_returns,
        )
        if water_mask.any():
            keep_mask &= ~water_mask
            logger.info("M-AlphaShape: water filter excluded %d points", int(water_mask.sum()))

    # Stage 2: Multiscale Alpha-Shape lower layers
    if progress:
        progress("M-AlphaShape: extracting coarse alpha layer (20m)…", 10.0)

    # Coarse alpha layer (large scale seeds)
    coarse_mask = _extract_alpha_shape_bottom_layer(
        xs_f, ys_f, zs_f, coarse_alpha, min_x, min_y, max_x, max_y
    ) & keep_mask

    if progress:
        progress("M-AlphaShape: extracting medium alpha layer (6m)…", 25.0)
    # Medium alpha layer
    med_mask = _extract_alpha_shape_bottom_layer(
        xs_f, ys_f, zs_f, medium_alpha, min_x, min_y, max_x, max_y
    ) & keep_mask

    if progress:
        progress("M-AlphaShape: extracting fine alpha layer (2m)…", 40.0)
    # Fine alpha layer
    fine_mask = _extract_alpha_shape_bottom_layer(
        xs_f, ys_f, zs_f, fine_alpha, min_x, min_y, max_x, max_y
    ) & keep_mask

    if int(coarse_mask.sum()) < 3:
        coarse_mask = med_mask.copy()
    if int(coarse_mask.sum()) < 3:
        coarse_mask = _lowest_per_grid_cells(
            xs_f, ys_f, zs_f, min_x, min_y, max_x, max_y, coarse_alpha, 0.0, 0.0
        ) & keep_mask

    # Stage 3: Multiscale progressive TIN densification
    if progress:
        progress("M-AlphaShape: multiscale TIN densification…", 55.0)

    # Pass 1: densify medium layer against coarse seeds
    pass1_ground = ground_classify_epptd(
        xs_f, ys_f, zs_f,
        max_distance=max_distance * 1.5,
        max_angle=max_angle * 1.25,
        max_terrain_angle=max_terrain_angle,
        follow_surface_trend=True,
        existing_ground_mask=coarse_mask,
        exclude_single_returns_in_water=False,
        progress=lambda m, p: progress and progress(f"M-AlphaShape: pass 1…", 55.0 + p * 0.20),
    ) & keep_mask

    # Pass 2: densify fine layer and remaining points against pass 1 ground
    if progress:
        progress("M-AlphaShape: fine layer densification…", 75.0)

    pass2_ground = ground_classify_epptd(
        xs_f, ys_f, zs_f,
        max_distance=max_distance,
        max_angle=max_angle,
        max_terrain_angle=max_terrain_angle,
        follow_surface_trend=True,
        existing_ground_mask=pass1_ground,
        exclude_single_returns_in_water=False,
        progress=lambda m, p: progress and progress(f"M-AlphaShape: pass 2…", 75.0 + p * 0.20),
    ) & keep_mask

    if progress:
        progress("M-AlphaShape: done", 100.0)
    logger.info("M-AlphaShape: classified %d ground / %d points", int(pass2_ground.sum()), n)
    return pass2_ground


# ─────────────────────────────────────────────────────────────────────
# EGS-CSF: Evolutionary Gradient Search Cloth Simulation (IJDE 2025)
# ─────────────────────────────────────────────────────────────────────

def ground_classify_egs_csf(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    cloth_resolution: float = 1.0,
    rigidness: float = 2.0,
    time_step: float = 0.65,
    class_threshold: float = 0.30,
    gradient_factor: float = 0.50,
    max_iterations: int = 50,
    exclude_single_returns_in_water: bool = True,
    sensor_type: Optional[np.ndarray] = None,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
) -> np.ndarray:
    """
    EGS-CSF: Evolutionary Gradient Search Cloth Simulation Filter.

    Reference:
        "EGS-CSF: an adaptive ground point cloud filtering framework based on evolutionary gradient search",
        International Journal of Digital Earth 2025, DOI: 10.1080/17538947.2025.2531843.

    Methodology:
      1. Preprocessing: Exclude water surface and water-column reflections.
      2. Point cloud inversion: Z_inv = max(Z) - Z.
      3. 2D Particle Cloth Grid:
         - Discretize space into cloth particles with spacing = cloth_resolution.
         - Obstacle surface Z_obs = min(Z_inv) in each particle cell.
      4. Evolutionary Gradient Adaptation:
         - Computes terrain gradient magnitude G(x, y) = ||grad(Z_obs)||.
         - In flat areas: high rigidness prevents cloth from sagging into buildings/trees.
         - In steep areas (embankments, cliffs, riverbanks): dynamically relaxes rigidness
           and accelerates displacement along the slope gradient, conforming to steep terrain.
      5. Spring-mass particle simulation:
         - Gravity + collision with obstacle surface + gradient-guided internal displacement.
         - Self-terminates when max particle movement < 5 mm.
      6. Gradient-Adaptive Classification:
         - True ground height Z_cloth = max(Z) - Z_c.
         - Distance tolerance scales with local slope: h(G) = h0 * (1 + beta * G).
         - Classifies all points within h(G) of the cloth as ground.
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    min_x, max_x = float(xs_f.min()), float(xs_f.max())
    min_y, max_y = float(ys_f.min()), float(ys_f.max())

    # Preprocessing: water surface and water column exclusion
    water_mask = None
    keep_mask = np.ones(n, dtype=bool)
    if exclude_single_returns_in_water:
        water_mask = _detect_water_surface_and_column(
            xs_f, ys_f, zs_f,
            sensor_type=sensor_type,
            return_numbers=return_numbers,
            num_returns=num_returns,
        )
        if water_mask.any():
            keep_mask &= ~water_mask
            logger.info("EGS-CSF: water filter excluded %d points", int(water_mask.sum()))

    if progress:
        progress("EGS-CSF: initializing cloth particle grid…", 10.0)

    # Step 1: Invert point cloud (considering only non-water points)
    max_z_val = float(zs_f[keep_mask].max()) if keep_mask.any() else float(zs_f.max())
    z_inv = max_z_val - zs_f

    # Step 2: Build 2D Cloth Particle Grid
    cr = max(0.5, float(cloth_resolution))
    nx = max(3, int(np.ceil((max_x - min_x) / cr)) + 1)
    ny = max(3, int(np.ceil((max_y - min_y) / cr)) + 1)

    gx = np.clip(((xs_f - min_x) / cr).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys_f - min_y) / cr).astype(np.int32), 0, ny - 1)
    flat_idx = gx * ny + gy

    # Obstacle height in inverted space: minimum Z_inv of points in cell
    z_obs = np.full(nx * ny, np.inf, dtype=np.float64)
    valid_pts = keep_mask
    np.minimum.at(z_obs, flat_idx[valid_pts], z_inv[valid_pts])
    z_obs_2d = z_obs.reshape((nx, ny))

    # Fill empty cells (voids) with nearest neighbor interpolation
    unoccupied = np.isinf(z_obs_2d)
    if unoccupied.all():
        return np.ones(n, dtype=bool)
    if unoccupied.any():
        try:
            from scipy.ndimage import distance_transform_edt
            _, indices = distance_transform_edt(unoccupied, return_indices=True)
            z_obs_2d = z_obs_2d[indices[0], indices[1]]
        except Exception:
            med_val = float(np.median(z_obs_2d[~unoccupied]))
            z_obs_2d[unoccupied] = med_val

    # Step 3: Evolutionary Gradient Calculation
    # Gradient magnitude: G = sqrt((dZ/dx)^2 + (dZ/dy)^2)
    gz, gx_grad = np.gradient(z_obs_2d, cr)
    grad_mag = np.sqrt(gz * gz + gx_grad * gx_grad)
    grad_max = float(np.percentile(grad_mag, 98.0)) if grad_mag.size else 1.0
    grad_norm = np.clip(grad_mag / max(1e-3, grad_max), 0.0, 1.0)

    # Adaptive rigidness: flat areas RI ~ rigidness; steep areas RI ~ 0.5 (flexible)
    base_ri = max(1.0, float(rigidness))
    ri_2d = np.maximum(0.5, base_ri * (1.0 - 0.70 * grad_norm))

    # Step 4: Iterative Cloth Simulation
    if progress:
        progress("EGS-CSF: running evolutionary cloth simulation…", 25.0)

    # Initial cloth elevation at top of inverted space
    z_cloth = np.full((nx, ny), float(np.max(z_obs_2d)) + 2.0, dtype=np.float64)
    g = 9.8 * 0.1
    dt = float(time_step)

    for it in range(max_iterations):
        if progress and it % 10 == 0:
            progress(f"EGS-CSF: cloth iteration {it+1}/{max_iterations}…", 25.0 + 50.0 * it / max_iterations)

        z_prev = z_cloth.copy()

        # 1. Gravity step
        z_cloth -= g * dt

        # 2. Collision constraint with inverted surface
        np.maximum(z_cloth, z_obs_2d, out=z_cloth)

        # 3. Internal spring displacement (4-neighbor smoothing)
        # Shift neighbors
        north = np.roll(z_cloth, -1, axis=0); north[-1, :] = z_cloth[-1, :]
        south = np.roll(z_cloth, 1, axis=0);  south[0, :] = z_cloth[0, :]
        east  = np.roll(z_cloth, -1, axis=1); east[:, -1] = z_cloth[:, -1]
        west  = np.roll(z_cloth, 1, axis=1);  west[:, 0] = z_cloth[:, 0]

        neighbor_avg = 0.25 * (north + south + east + west)
        delta = (neighbor_avg - z_cloth) / ri_2d

        # Apply displacement where particles are not resting on rigid collision
        not_collision = z_cloth > (z_obs_2d + 1e-4)
        z_cloth[not_collision] += delta[not_collision]
        np.maximum(z_cloth, z_obs_2d, out=z_cloth)

        max_mov = float(np.max(np.abs(z_cloth - z_prev)))
        if max_mov < 0.005:
            break

    # Step 5: Re-inversion and Gradient-Adaptive Classification
    if progress:
        progress("EGS-CSF: classifying points with adaptive gradient threshold…", 85.0)

    # True cloth elevation in original point cloud space
    z_cloth_orig = max_z_val - z_cloth

    # Point-wise cloth elevation via bilinear or nearest sampling
    point_cloth_z = z_cloth_orig[gx, gy]
    point_grad = grad_mag[gx, gy]

    # Adaptive distance threshold: h(G) = h0 * (1 + beta * G)
    h_dynamic = float(class_threshold) * (1.0 + float(gradient_factor) * np.clip(point_grad, 0.0, 3.0))

    # Point is ground if distance to cloth is within h_dynamic
    ground_mask = (np.abs(zs_f - point_cloth_z) <= h_dynamic) & keep_mask

    if progress:
        progress("EGS-CSF: done", 100.0)
    logger.info("EGS-CSF: classified %d ground / %d points (%.1f%%)", int(ground_mask.sum()), n, 100.0 * ground_mask.sum() / n)
    return ground_mask

