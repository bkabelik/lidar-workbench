"""
LiDAR Workbench — Ground classification.

Ground filtering algorithms and their helpers, moved out of ``noise_filter``
so that noise removal and terrain classification stay separate:

  - :func:`ground_classify_smrf`    — Simple Morphological Filter (Pingel 2013).
  - :func:`ground_classify_stepdown` — Adaptive Step-Down PTD (Multi-Scale Bulged TIN).
  - :func:`ground_classify_epptd`   — Edge-Preserving PTD (Axelsson 2000 + edge controls).
  - :func:`ground_classify_epptd_two_pass` — coarse-to-fine two-pass EP-PTD (12° → 6°).
  - :func:`ground_classify_aptd`    — Adaptive Grid PTD, AGPTD (Zheng et al. 2024).
  - :func:`ground_classify_hptd`    — Hierarchical/Fast PTD, FPTD (Li et al. 2021).
  - :func:`ground_classify_dl_hybrid_ptd` — Pointcept DL prior + PTD densification.
  - :func:`ground_classify_multiscale_alpha_shape` — Multiscale 3D Alpha Shape (MDPI 2024).
  - :func:`ground_classify_egs_csf` — Evolutionary Gradient Cloth Simulation (IJDE 2025).
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

    Pingel, Clarke and McBride's (2013) classic ground-filter algorithm.
    Applies morphological opening (erode + dilate) with progressively larger
    window sizes, then classifies points as ground if they fall within an
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


def _axelsson_metrics_vec(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Vectorized calculation of Axelsson metrics (plane normal, perpendicular distance, max angle).
    P, A, B, C are (M, 3) arrays.
    Returns (inside, dist_d, max_angle).
    """
    e1 = B - A
    e2 = C - A
    n = np.cross(e1, e2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    valid_norm = (norm_n[:, 0] > 1e-9)
    norm_n = np.where(valid_norm[:, None], norm_n, 1.0)
    unit_n = n / norm_n

    v = P - A
    v_dot_n = np.sum(v * unit_n, axis=1, keepdims=True)
    dist_d = np.abs(v_dot_n[:, 0])

    # Project P onto plane
    P_proj = P - unit_n * v_dot_n

    # Barycentric coordinates check
    v2 = P_proj - A
    d00 = np.sum(e1 * e1, axis=1)
    d01 = np.sum(e1 * e2, axis=1)
    d11 = np.sum(e2 * e2, axis=1)
    d20 = np.sum(v2 * e1, axis=1)
    d21 = np.sum(v2 * e2, axis=1)

    denom = d00 * d11 - d01 * d01
    valid_denom = np.abs(denom) > 1e-9
    denom_safe = np.where(valid_denom, denom, 1.0)
    v_coord = (d11 * d20 - d01 * d21) / denom_safe
    w_coord = (d00 * d21 - d01 * d20) / denom_safe
    u_coord = 1.0 - v_coord - w_coord

    inside = valid_norm & valid_denom & (u_coord >= -1e-5) & (v_coord >= -1e-5) & (w_coord >= -1e-5)

    h0 = np.linalg.norm(P_proj - A, axis=1)
    h1 = np.linalg.norm(P_proj - B, axis=1)
    h2 = np.linalg.norm(P_proj - C, axis=1)

    alpha = np.degrees(np.arctan2(dist_d, np.maximum(h0, 1e-9)))
    beta = np.degrees(np.arctan2(dist_d, np.maximum(h1, 1e-9)))
    gamma = np.degrees(np.arctan2(dist_d, np.maximum(h2, 1e-9)))
    max_angle = np.maximum(np.maximum(alpha, beta), gamma)

    return inside, dist_d, max_angle


def _detect_spikes_ptd(vertices_xyz: np.ndarray, k: int = 12, threshold: float = 0.75, spacing: float = 0.25) -> np.ndarray:
    """Local neighborhood RANSAC plane fitting spike detector."""
    from scipy.spatial import KDTree
    n_v = len(vertices_xyz)
    if n_v < k + 1 or k <= 0:
        return np.zeros(n_v, dtype=bool)

    tree = KDTree(vertices_xyz[:, :2])
    dists, indices = tree.query(vertices_xyz[:, :2], k=min(k * 2 + 1, n_v))
    spacing_limit_sq = (3.0 * spacing) ** 2

    is_spike = np.zeros(n_v, dtype=bool)
    rng = np.random.default_rng(42)

    for i in range(n_v):
        nbr_mask = dists[i] ** 2 > spacing_limit_sq
        nbr_idx = indices[i][nbr_mask]
        nbr_idx = nbr_idx[~is_spike[nbr_idx]]
        if len(nbr_idx) > k:
            nbr_idx = nbr_idx[:k]
        if len(nbr_idx) < 3:
            continue

        nb_pos = vertices_xyz[nbr_idx]
        best_inliers = -1
        best_plane = None
        sufficient = int(len(nbr_idx) * 0.75)
        consensus_thresh = threshold * 0.75

        for _ in range(25):
            idx3 = rng.choice(len(nb_pos), size=3, replace=False)
            p1, p2, p3 = nb_pos[idx3]
            v1 = p2 - p1
            v2 = p3 - p1
            cp = np.cross(v1, v2)
            n_len = np.linalg.norm(cp)
            if n_len < 1e-9:
                continue
            normal = cp / n_len
            d_val = -float(np.dot(normal, p1))
            residuals = np.abs(np.dot(nb_pos, normal) + d_val)
            inliers = int(np.sum(residuals < consensus_thresh))
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal, d_val)
                if inliers >= sufficient:
                    break

        if best_plane is not None:
            norm, d_val = best_plane
            q_dist = abs(float(np.dot(vertices_xyz[i], norm)) + d_val)
            if q_dist > threshold:
                is_spike[i] = True

    return is_spike


def _ground_classify_ptd_py(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    seed_resolution_search: float = 10.0,
    max_iteration_angle: float = 6.0,
    max_iteration_distance: float = 0.5,
    spacing: float = 0.50,
    buffer_size: float = 15.0,
    max_iter: int = 15,
    k_spikes: int = 12,
    spike_threshold: float = 0.75,
    seed_indices: Optional[np.ndarray] = None,
    reject_mask: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
) -> np.ndarray:
    """Vectorized pure Python implementation of Progressive TIN Densification."""
    from scipy.spatial import Delaunay, KDTree
    n = len(xs)
    valid = np.ones(n, dtype=bool)
    if reject_mask is not None:
        valid &= ~reject_mask
    val_idx = np.flatnonzero(valid)
    if len(val_idx) < 3:
        return np.zeros(n, dtype=bool)

    min_x, max_x = float(xs[val_idx].min()), float(xs[val_idx].max())
    min_y, max_y = float(ys[val_idx].min()), float(ys[val_idx].max())

    cand_mask = _lowest_per_grid_cells(
        xs, ys, zs,
        min_x, min_y, max_x, max_y,
        spacing, 0.0, 0.0,
    )
    if reject_mask is not None:
        cand_mask &= ~reject_mask

    cand_indices = np.flatnonzero(cand_mask)
    if len(cand_indices) == 0:
        return np.zeros(n, dtype=bool)

    order = np.argsort(zs[cand_indices])
    cand_indices = cand_indices[order]

    if seed_indices is not None and len(seed_indices) > 0:
        seed_mask = np.zeros(n, dtype=bool)
        seed_mask[seed_indices] = True
        if reject_mask is not None:
            seed_mask &= ~reject_mask
    else:
        g0 = _lowest_per_grid_cells(
            xs[cand_indices], ys[cand_indices], zs[cand_indices],
            min_x, min_y, max_x, max_y,
            seed_resolution_search, 0.0, 0.0,
        )
        g1 = _lowest_per_grid_cells(
            xs[cand_indices], ys[cand_indices], zs[cand_indices],
            min_x, min_y, max_x, max_y,
            seed_resolution_search, seed_resolution_search * 0.5, seed_resolution_search * 0.5,
        )
        seed_mask = np.zeros(n, dtype=bool)
        seed_mask[cand_indices[g0 | g1]] = True

    seed_idx = np.flatnonzero(seed_mask)
    if len(seed_idx) < 3:
        return np.zeros(n, dtype=bool)

    vbuff_xy = np.empty((0, 2), dtype=np.float64)
    vbuff_z = np.empty(0, dtype=np.float64)
    if buffer_size > 0:
        rng = np.random.default_rng(42)
        b_xmin = min_x - buffer_size
        b_ymin = min_y - buffer_size
        b_xmax = max_x + buffer_size
        b_ymax = max_y + buffer_size
        dx = b_xmax - b_xmin
        dy = b_ymax - b_ymin
        nx = max(1, int(round(dx / seed_resolution_search)))
        ny = max(1, int(round(dy / seed_resolution_search)))
        sx = dx / nx
        sy = dy / ny

        buf_pts = []
        for i in range(nx + 1):
            x = b_xmin + i * sx + rng.uniform(-0.5, 0.5)
            buf_pts.append((x, b_ymin + rng.uniform(-0.5, 0.5)))
            buf_pts.append((x, b_ymax + rng.uniform(-0.5, 0.5)))
        for j in range(1, ny):
            y = b_ymin + j * sy + rng.uniform(-0.5, 0.5)
            buf_pts.append((b_xmin + rng.uniform(-0.5, 0.5), y))
            buf_pts.append((b_xmax + rng.uniform(-0.5, 0.5), y))

        buf_pts = np.array(buf_pts)
        tree_cand = KDTree(np.column_stack((xs[cand_indices], ys[cand_indices])))
        k_near = min(5, len(cand_indices))
        _, near_idx = tree_cand.query(buf_pts, k=k_near)
        if k_near == 1:
            near_z = zs[cand_indices[near_idx]]
        else:
            near_z = np.min(zs[cand_indices[near_idx]], axis=1)
        vbuff_xy = buf_pts
        vbuff_z = near_z

    is_inserted = np.zeros(len(cand_indices), dtype=bool)
    seed_set = set(seed_idx)
    for ci_idx, ci in enumerate(cand_indices):
        if ci in seed_set:
            is_inserted[ci_idx] = True

    active_indices = list(seed_idx)

    for it in range(max_iter):
        if len(vbuff_xy) > 0:
            cur_xy = np.vstack((vbuff_xy, np.column_stack((xs[active_indices], ys[active_indices]))))
            cur_z = np.concatenate((vbuff_z, zs[active_indices]))
        else:
            cur_xy = np.column_stack((xs[active_indices], ys[active_indices]))
            cur_z = zs[active_indices]

        try:
            tri = Delaunay(cur_xy)
        except Exception:
            break

        remaining = np.flatnonzero(~is_inserted)
        if len(remaining) == 0:
            break

        q_xy = np.column_stack((xs[cand_indices[remaining]], ys[cand_indices[remaining]]))
        q_z = zs[cand_indices[remaining]]

        s_ids = tri.find_simplex(q_xy)
        inside = s_ids >= 0
        if not inside.any():
            break

        in_rem = remaining[inside]
        in_s = s_ids[inside]
        tris = tri.simplices[in_s]

        A = np.column_stack((cur_xy[tris[:, 0]], cur_z[tris[:, 0]]))
        B = np.column_stack((cur_xy[tris[:, 1]], cur_z[tris[:, 1]]))
        C = np.column_stack((cur_xy[tris[:, 2]], cur_z[tris[:, 2]]))
        P = np.column_stack((q_xy[inside], q_z[inside]))

        eAB = np.sum((A - B) ** 2, axis=1)
        eAC = np.sum((A - C) ** 2, axis=1)
        eBC = np.sum((B - C) ** 2, axis=1)
        max_edge_sq = np.maximum(np.maximum(eAB, eAC), eBC)
        small_tri = max_edge_sq < (spacing * spacing)
        is_inserted[in_rem[small_tri]] = True

        eval_mask = ~small_tri
        if not eval_mask.any():
            break

        ev_rem = in_rem[eval_mask]
        ev_s = in_s[eval_mask]
        ins, dist_d, max_ang = _axelsson_metrics_vec(P[eval_mask], A[eval_mask], B[eval_mask], C[eval_mask])

        ok = ins & (dist_d <= max_iteration_distance) & (max_ang <= max_iteration_angle)
        accept_rem = ev_rem[ok]
        accept_s = ev_s[ok]

        if len(accept_rem) == 0:
            break

        chosen = accept_rem

        is_inserted[chosen] = True
        active_indices.extend(cand_indices[chosen])

        if len(chosen) < max(1, int(0.0005 * len(cur_xy))):
            break

    if k_spikes > 0:
        g_pts = np.column_stack((xs[active_indices], ys[active_indices], zs[active_indices]))
        spikes = _detect_spikes_ptd(g_pts, k=k_spikes, threshold=spike_threshold, spacing=spacing)
        final_ground = np.array(active_indices)[~spikes]
    else:
        final_ground = np.array(active_indices)

    out_mask = np.zeros(n, dtype=bool)
    out_mask[final_ground] = True
    return out_mask


def ground_classify_epptd(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    seed_resolution_search: float = 10.0,
    max_iteration_angle: float = 6.0,
    max_iteration_distance: float = 0.5,
    spacing: float = 0.50,
    buffer_size: float = 15.0,
    max_iter: int = 15,
    k_spikes: int = 12,
    spike_threshold: float = 0.75,
    dense: bool = True,
    dense_tolerance: float = 0.10,
    existing_ground_mask: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
    **kwargs,
) -> np.ndarray:
    """Progressive TIN Densification (Axelsson 2000).

    Progressive TIN Densification bare-earth classification:

    Algorithm:
      1. Point insertion & candidate pre-thinning to grid of size ``spacing``
         (keeps lowest point per cell, preserving original FID).
      2. Candidates sorted ascending by Z.
      3. Dual-grid shifted seed selection (two shifted grids with cell size
         ``seed_resolution_search``, picking lowest per cell and unioning).
      4. Virtual perimeter buffer points at ``buffer_size`` with elevation from
         nearest seed in 2D to eliminate boundary edge artifacts.
      5. Incremental/iterative Delaunay densification up to ``max_iter``:
         - Evaluates exact Axelsson metrics (plane normal, perpendicular distance,
           in-triangle projection check, max vertex angle).
         - Freezes candidates in triangles with max edge < ``spacing``.
         - Inserts candidates meeting distance and angle thresholds.
      6. Spike outlier filtering via local neighborhood RANSAC plane fitting.
    """
    from scipy.spatial import Delaunay

    # Backwards compatibility with previous parameter names
    if "cell_size" in kwargs and kwargs["cell_size"] is not None and kwargs["cell_size"] > 0:
        seed_resolution_search = float(kwargs["cell_size"])
    if "seed_resolution" in kwargs and kwargs["seed_resolution"] is not None and kwargs["seed_resolution"] > 0:
        seed_resolution_search = float(kwargs["seed_resolution"])
    if "max_building_size" in kwargs and kwargs["max_building_size"] is not None:
        seed_resolution_search = float(kwargs["max_building_size"])
    if "max_angle" in kwargs and kwargs["max_angle"] is not None:
        max_iteration_angle = float(kwargs["max_angle"])
    if "max_distance" in kwargs and kwargs["max_distance"] is not None:
        max_iteration_distance = float(kwargs["max_distance"])
    if "dense_pass" in kwargs and kwargs["dense_pass"] is not None:
        dense = bool(kwargs["dense_pass"])
    if "dense_above_tolerance" in kwargs and kwargs["dense_above_tolerance"] is not None:
        dense_tolerance = float(kwargs["dense_above_tolerance"])
    if "low_outlier_threshold" in kwargs and kwargs["low_outlier_threshold"] is not None:
        spike_threshold = float(kwargs["low_outlier_threshold"])
    if "remove_low_outliers" in kwargs and not kwargs["remove_low_outliers"]:
        k_spikes = 0

    n = len(xs)
    if n < 3:
        return np.ones(n, dtype=bool)

    extra_dims = kwargs.get("extra_dims")
    extra_dim_filters = kwargs.get("extra_dim_filters")
    sensor_type = kwargs.get("sensor_type")
    extra_reject = np.zeros(n, dtype=bool)
    if extra_dim_filters and extra_dims and sensor_type is not None and len(sensor_type) == n:
        bathy_pts = (np.asarray(sensor_type) == 2)
        if bathy_pts.any():
            for name, lo, hi in extra_dim_filters:
                key = next((k for k in extra_dims if str(k).lower() == str(name).lower()), None)
                if key is not None:
                    vals = np.asarray(extra_dims[key], dtype=np.float64)
                    if lo is not None:
                        extra_reject |= bathy_pts & (vals < lo)
                    if hi is not None:
                        extra_reject |= bathy_pts & (vals > hi)

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    seed_idx = None
    if existing_ground_mask is not None:
        eg = np.asarray(existing_ground_mask, dtype=bool)
        if eg.any():
            seed_idx = np.flatnonzero(eg & ~extra_reject)
            # When explicit seeds are provided across the tile, disable artificial buffer
            buffer_size = 0.0

    if progress:
        progress("PTD: running Progressive TIN Densification…", 10.0)

    ground_mask = _ground_classify_ptd_py(
        xs_f, ys_f, zs_f,
        seed_resolution_search=seed_resolution_search,
        max_iteration_angle=max_iteration_angle,
        max_iteration_distance=max_iteration_distance,
            spacing=spacing,
            buffer_size=buffer_size,
            max_iter=max_iter,
            k_spikes=k_spikes,
            spike_threshold=spike_threshold,
            seed_indices=seed_idx,
            reject_mask=extra_reject if extra_reject.any() else None,
            progress=progress,
        )

    if "dense_tolerance" in kwargs and kwargs["dense_tolerance"] is not None:
        dense_tolerance = float(kwargs["dense_tolerance"])
    elif "dense_above_tolerance" in kwargs and kwargs["dense_above_tolerance"] is not None:
        dense_tolerance = float(kwargs["dense_above_tolerance"])
    elif dense_tolerance is None:
        dense_tolerance = 0.10

    # Strict densification pass (evaluates unclassified points against final ground TIN surface)
    if dense:
        if progress:
            progress("PTD: strict densification pass…", 80.0)
        g_idx = np.flatnonzero(ground_mask)
        if len(g_idx) >= 3:
            unclass = np.flatnonzero(~ground_mask & ~extra_reject)
            if len(unclass) > 0:
                try:
                    tri = Delaunay(np.column_stack((xs_f[g_idx], ys_f[g_idx])))
                    q_xy = np.column_stack((xs_f[unclass], ys_f[unclass]))
                    s_ids = tri.find_simplex(q_xy)
                    inside = s_ids >= 0
                    if inside.any():
                        in_u = unclass[inside]
                        in_s = s_ids[inside]
                        tris = tri.simplices[in_s]
                        A = np.column_stack((xs_f[g_idx[tris[:, 0]]], ys_f[g_idx[tris[:, 0]]], zs_f[g_idx[tris[:, 0]]]))
                        B = np.column_stack((xs_f[g_idx[tris[:, 1]]], ys_f[g_idx[tris[:, 1]]], zs_f[g_idx[tris[:, 1]]]))
                        C = np.column_stack((xs_f[g_idx[tris[:, 2]]], ys_f[g_idx[tris[:, 2]]], zs_f[g_idx[tris[:, 2]]]))
                        P = np.column_stack((xs_f[in_u], ys_f[in_u], zs_f[in_u]))

                        ins, dist_d, max_ang = _axelsson_metrics_vec(P, A, B, C)

                        n_vec = np.cross(B - A, C - A)
                        n_len = np.linalg.norm(n_vec, axis=1, keepdims=True)
                        n_unit = n_vec / np.maximum(n_len, 1e-9)
                        flip = n_unit[:, 2] < 0
                        n_unit[flip] = -n_unit[flip]
                        n_unit[:, 2] = np.maximum(1e-4, n_unit[:, 2])
                        v_vec = P - A
                        ortho_dist = np.sum(v_vec * n_unit, axis=1)

                        strict_tol = min(float(dense_tolerance), 0.15)
                        strict_ok = (
                            ins
                            & (dist_d <= strict_tol)
                            & (max_ang <= max_iteration_angle)
                            & (ortho_dist <= strict_tol)
                            & (ortho_dist >= -0.5)
                        )
                        ground_mask[in_u[strict_ok]] = True
                except Exception as e:
                    logger.warning("PTD strict densification pass failed: %s", e)

    # Preserve all trusted existing ground points
    if existing_ground_mask is not None:
        ground_mask[np.asarray(existing_ground_mask, dtype=bool) & ~extra_reject] = True

    if extra_reject.any():
        ground_mask &= ~extra_reject

    if progress:
        progress("PTD: complete", 100.0)

    logger.info(
        "PTD complete: %d ground points (%.1f%%)",
        int(ground_mask.sum()), 100.0 * ground_mask.sum() / n,
    )
    return ground_mask


# Backward-compatible aliases
ground_classify_tin = ground_classify_epptd


def ground_classify_epptd_two_pass(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    pass1_cell_size: float = 60.0,
    pass1_max_distance: float = 1.0,
    pass1_max_angle: float = 12.0,
    pass2_max_distance: float = 1.0,
    pass2_max_angle: float = 6.0,
    max_terrain_angle: float = 88.0,
    only_upward: bool = False,
    follow_surface_trend: bool = False,
    remove_low_outliers_after_pass1: bool = True,
    low_outlier_neighbors: int = 8,
    low_outlier_threshold: float = 1.0,
    dense: bool = True,
    dense_tolerance: float = 0.20,
    sensor_type: Optional[np.ndarray] = None,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
    **kwargs,
) -> np.ndarray:
    """
    Two-pass TerraScan-style EP-PTD ground classification.

    Pass 1 seeds a coarse grid (default 60 m) so the initial TIN is built
    at the true local terrain/riverbed minimum, uses a permissive 12° angle
    and 1.0 m iteration distance, and optionally removes low outliers.
    Pass 2 then densifies against the pass-1 ground (no grid re-seeding)
    with a tighter 6° angle, 1.0 m distance, which rejects
    the water surface while refining fine bed/terrain detail.
    """
    n = len(xs)
    if n < 100:
        return np.ones(n, dtype=bool)

    def _report(msg: str, pct: float, base: float) -> None:
        if progress:
            progress(msg, base + pct)

    _report("Two-pass TIN: pass 1 (coarse seeds)…", 2.0, 0.0)
    mask1 = ground_classify_epptd(
        xs, ys, zs,
        max_distance=pass1_max_distance,
        max_angle=pass1_max_angle,
        seed_resolution=pass1_cell_size,
        max_terrain_angle=max_terrain_angle,
        only_upward=only_upward,
        follow_surface_trend=follow_surface_trend,
        remove_low_outliers=remove_low_outliers_after_pass1,
        low_outlier_neighbors=low_outlier_neighbors,
        low_outlier_threshold=low_outlier_threshold,
        sensor_type=sensor_type,
        return_numbers=return_numbers,
        num_returns=num_returns,
        progress=lambda m, p: _report(m, p * 0.45, 5.0),
    )

    if int(mask1.sum()) < 3:
        logger.warning("Two-pass TIN: pass 1 produced < 3 ground points — returning pass 1")
        return mask1

    _report("Two-pass TIN: pass 2 (densify existing ground)…", 52.0, 0.0)
    mask2 = ground_classify_epptd(
        xs, ys, zs,
        max_distance=pass2_max_distance,
        max_angle=pass2_max_angle,
        max_terrain_angle=max_terrain_angle,
        only_upward=False,
        follow_surface_trend=follow_surface_trend,
        remove_low_outliers=False,
        sensor_type=sensor_type,
        existing_ground_mask=mask1,
        dense=dense,
        dense_tolerance=dense_tolerance,
        return_numbers=return_numbers,
        num_returns=num_returns,
        progress=lambda m, p: _report(m, 50.0 + p * 0.45, 5.0),
    )
    _report("Two-pass TIN: done", 100.0, 0.0)
    return mask2


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
    cand_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract lower boundary points using a bottom rolling sphere of radius alpha.

    Implements the modified 3D alpha shape lower envelope test (Cao et al. 2024):
    A sphere of radius alpha placed beneath point P tests if any neighbor
    point penetrates the sphere from below. Points where the sphere is clear
    belong to the lower alpha-shape boundary.
    """
    n_total = len(xs)
    if cand_indices is not None and len(cand_indices) > 0:
        sub_xs = xs[cand_indices]
        sub_ys = ys[cand_indices]
        sub_zs = zs[cand_indices]
        orig_indices = cand_indices
    else:
        sub_xs, sub_ys, sub_zs = xs, ys, zs
        orig_indices = np.arange(n_total)

    cell_size = max(alpha * 0.4, 0.5)
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)) + 1)

    gx = np.clip(((sub_xs - min_x) / cell_size).astype(np.int32), 0, nx - 1)
    gy = np.clip(((sub_ys - min_y) / cell_size).astype(np.int32), 0, ny - 1)
    flat_idx = gx * ny + gy

    # Find minimum Z in each cell
    cell_min = np.full(nx * ny, np.inf, dtype=np.float64)
    np.minimum.at(cell_min, flat_idx, sub_zs)
    cand_mask = (sub_zs <= cell_min[flat_idx] + 0.05)
    local_cands = np.flatnonzero(cand_mask)
    if len(local_cands) == 0:
        return np.zeros(n_total, dtype=bool)

    # KDTree query on candidate lowest points only (10-100x speedup over querying all points)
    try:
        from scipy.spatial import cKDTree
        c_x = sub_xs[local_cands]
        c_y = sub_ys[local_cands]
        c_z = sub_zs[local_cands]
        c_orig = orig_indices[local_cands]

        cand_tree = cKDTree(np.column_stack((c_x, c_y)))
        search_r = min(alpha * 1.2, 35.0)
        neighbors_list = cand_tree.query_ball_point(
            np.column_stack((c_x, c_y)), r=search_r
        )
        is_bottom = np.ones(len(local_cands), dtype=bool)
        two_alpha = 2.0 * alpha

        for i, nbrs in enumerate(neighbors_list):
            if len(nbrs) < 4:
                continue
            nbrs_arr = np.asarray(nbrs)
            dx = c_x[nbrs_arr] - c_x[i]
            dy = c_y[nbrs_arr] - c_y[i]
            dz = c_z[nbrs_arr] - c_z[i]

            # Fast 2x2 analytic normal (avoiding expensive np.linalg.lstsq in Python loop)
            s_xx = np.dot(dx, dx)
            s_yy = np.dot(dy, dy)
            s_xy = np.dot(dx, dy)
            s_xz = np.dot(dx, dz)
            s_yz = np.dot(dy, dz)
            det = s_xx * s_yy - s_xy * s_xy

            if det > 1e-6:
                a = (s_yy * s_xz - s_xy * s_yz) / det
                b = (s_xx * s_yz - s_xy * s_xz) / det
                norm_factor = 1.0 / np.sqrt(a * a + b * b + 1.0)
                nx_val = -a * norm_factor
                ny_val = -b * norm_factor
                nz_val = norm_factor
            else:
                nx_val, ny_val, nz_val = 0.0, 0.0, 1.0

            d_perp = dx * nx_val + dy * ny_val + dz * nz_val
            below = d_perp < -0.05
            if below.any():
                d_sq = dx[below] ** 2 + dy[below] ** 2 + dz[below] ** 2
                if (d_sq < (-two_alpha * d_perp[below])).any():
                    is_bottom[i] = False

        bottom_mask = np.zeros(n_total, dtype=bool)
        bottom_mask[c_orig[is_bottom]] = True
        return bottom_mask
    except Exception:
        fallback_mask = np.zeros(n_total, dtype=bool)
        fallback_mask[orig_indices[local_cands]] = True
        return fallback_mask


def ground_classify_multiscale_alpha_shape(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    coarse_alpha: float = 20.0,
    medium_alpha: float = 6.0,
    fine_alpha: float = 2.0,
    max_distance: float = 0.20,
    max_terrain_angle: float = 45.0,
    all_returns: bool = False,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
    **kwargs,
) -> np.ndarray:
    """
    Multiscale Modified 3D Alpha Shape ground filtering for airborne LiDAR data.

    Reference:
        Cao, D., Wang, C., Du, M., & Xi, X. (2024).
        "A Multiscale Filtering Method for Airborne LiDAR Data Using Modified 3D Alpha Shape",
        Remote Sensing, 16(8), 1443. https://doi.org/10.3390/rs16081443

    Methodology:
      1. Modified 3D Alpha Shape bottom boundary extraction:
         - A probe sphere of radius alpha rolls along the bottom surface of the point cloud.
         - A point belongs to the bottom alpha-shape if an empty sphere of radius alpha
           is tangent to it from below with no other points penetrating the sphere.
      2. Multiscale layer extraction:
         - Coarse scale (alpha1 = 20 m): bridges large buildings and dense vegetation canopy,
           isolating macro-scale terrain seeds.
         - Medium scale (alpha2 = 6 m): captures valleys, gullies, and breaklines.
         - Fine scale (alpha3 = 2 m): captures micro-topography and steep terrain.
      3. Ground surface classification:
         - The union of the multiscale bottom points forms the lower terrain boundary.
         - Construct TIN on the bottom boundary points.
         - Classify points within max_distance of the bottom alpha-shape TIN as ground.
    """
    import math
    from scipy.spatial import Delaunay

    n = len(xs)
    if n < 3:
        return np.ones(n, dtype=bool)

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    # Return filtering: default prefers last returns
    if not all_returns and return_numbers is not None and num_returns is not None:
        last_mask = np.asarray(return_numbers) == np.asarray(num_returns)
        cand_indices = np.flatnonzero(last_mask) if last_mask.sum() >= 3 else np.arange(n)
    else:
        cand_indices = np.arange(n)

    min_x, max_x = float(xs_f.min()), float(xs_f.max())
    min_y, max_y = float(ys_f.min()), float(ys_f.max())

    if progress:
        progress("M-AlphaShape: extracting coarse alpha layer (α₁)…", 15.0)

    # 1. Coarse alpha layer (large scale seeds)
    coarse_mask = _extract_alpha_shape_bottom_layer(
        xs_f, ys_f, zs_f, coarse_alpha, min_x, min_y, max_x, max_y, cand_indices=cand_indices
    )

    if progress:
        progress("M-AlphaShape: extracting medium alpha layer (α₂)…", 35.0)

    # 2. Medium alpha layer
    med_mask = _extract_alpha_shape_bottom_layer(
        xs_f, ys_f, zs_f, medium_alpha, min_x, min_y, max_x, max_y, cand_indices=cand_indices
    )

    if progress:
        progress("M-AlphaShape: extracting fine alpha layer (α₃)…", 55.0)

    # 3. Fine alpha layer
    fine_mask = _extract_alpha_shape_bottom_layer(
        xs_f, ys_f, zs_f, fine_alpha, min_x, min_y, max_x, max_y, cand_indices=cand_indices
    )

    # Hierarchical Multiscale Constraint (Cao et al. 2024):
    # Coarse bottom layer defines macro-terrain; finer layers are constrained by it to reject roofs & canopy.
    bottom_seeds = coarse_mask.copy()
    c_idx = np.flatnonzero(coarse_mask)
    if len(c_idx) >= 3:
        try:
            tri_c = Delaunay(np.column_stack((xs_f[c_idx], ys_f[c_idx])))

            # Constrain medium layer
            m_idx = np.flatnonzero(med_mask)
            if len(m_idx):
                s_m = tri_c.find_simplex(np.column_stack((xs_f[m_idx], ys_f[m_idx])))
                ins_m = s_m >= 0
                if ins_m.any():
                    b_z = zs_f[c_idx][tri_c.simplices[s_m[ins_m]]]
                    min_tri_z = np.min(b_z, axis=1)
                    dz_m = zs_f[m_idx[ins_m]] - min_tri_z
                    valid_m = dz_m <= 1.5
                    drop_m = m_idx[ins_m][~valid_m]
                    med_mask[drop_m] = False
            bottom_seeds |= med_mask

            # Constrain fine layer
            f_idx = np.flatnonzero(fine_mask)
            if len(f_idx):
                s_f = tri_c.find_simplex(np.column_stack((xs_f[f_idx], ys_f[f_idx])))
                ins_f = s_f >= 0
                if ins_f.any():
                    b_z = zs_f[c_idx][tri_c.simplices[s_f[ins_f]]]
                    min_tri_z = np.min(b_z, axis=1)
                    dz_f = zs_f[f_idx[ins_f]] - min_tri_z
                    valid_f = dz_f <= 1.0
                    drop_f = f_idx[ins_f][~valid_f]
                    fine_mask[drop_f] = False
            bottom_seeds |= fine_mask
        except Exception:
            bottom_seeds = coarse_mask | med_mask | fine_mask
    else:
        bottom_seeds = coarse_mask | med_mask | fine_mask

    if int(bottom_seeds.sum()) < 3:
        bottom_seeds = _lowest_per_grid_cells(
            xs_f, ys_f, zs_f, min_x, min_y, max_x, max_y, 5.0, 0.0, 0.0
        )
        if int(bottom_seeds.sum()) < 3:
            return np.ones(n, dtype=bool)

    if progress:
        progress("M-AlphaShape: building lower boundary surface…", 75.0)

    # 4. Construct lower boundary terrain surface TIN
    b_idx = np.flatnonzero(bottom_seeds)
    # Add virtual bounding corners to ensure complete spatial coverage
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

    b_xy = np.vstack((
        np.column_stack((xs_f[b_idx], ys_f[b_idx])),
        corner_xy,
    ))
    b_z = np.concatenate((zs_f[b_idx], corner_z))

    ground_mask = bottom_seeds.copy()
    try:
        tri = Delaunay(b_xy)
        normals, slope_deg, max_edge = _tri_props(tri, b_xy, b_z)

        # 5. Classify all points within max_distance of the bottom alpha-shape TIN
        cand_indices = np.flatnonzero(~ground_mask)
        q_xy = np.column_stack((xs_f[cand_indices], ys_f[cand_indices]))
        q_z = zs_f[cand_indices]

        s_ids = tri.find_simplex(q_xy)
        inside = s_ids >= 0
        if inside.any():
            ii = np.flatnonzero(inside)
            geo = _geo_query(
                tri, b_xy, b_z,
                q_xy[ii], q_z[ii], s_ids[ii],
                normals, slope_deg, max_edge,
            )
            valid = geo["inside_tri"] & geo["valid_denom"]
            slopes = geo["slope_deg"]
            cos_terrain = math.cos(math.radians(max_terrain_angle))
            slope_ok = normals[s_ids[ii], 2] >= cos_terrain

            # Ground points sit on or close to the lower alpha-shape surface
            dz = geo["dz"]
            d_perp = geo["d_perp"]
            is_ground = valid & slope_ok & (dz >= -0.25) & (d_perp <= max_distance)
            ground_mask[cand_indices[ii[is_ground]]] = True

    except Exception as e:
        logger.warning("M-AlphaShape: surface projection failed: %s", e)

    if progress:
        progress("M-AlphaShape: done", 100.0)
    logger.info("M-AlphaShape: classified %d ground / %d points", int(ground_mask.sum()), n)
    return ground_mask


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
    max_iterations: int = 150,
    spike_down: float = 2.5,
    slope_smooth: bool = True,
    all_returns: bool = False,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    sensor_type: Optional[np.ndarray] = None,
    exclude_single_returns_in_water: bool = False,
    progress: ProgressCB = None,
    **kwargs,
) -> np.ndarray:
    """
    EGS-CSF: Evolutionary Gradient Search Cloth Simulation Filter.

    Reference:
        Gao, C., & Liu, D. (2025).
        "EGS-CSF: an adaptive ground point cloud filtering framework based on evolutionary gradient search",
        International Journal of Digital Earth, DOI: 10.1080/17538947.2025.2531843.

    Methodology:
      1. Robust Macro-Relief Leveling:
         - Fits a robust plane trend to level steep slopes so the cloth simulation
           operates without gravitational starvation across large elevation relief.
      2. Invert Point Cloud: Z_inv = -Z_norm (ground at top, buildings/trees at bottom).
      3. 2D Particle Cloth Grid:
         - Discretize 2D plane into cloth particles with spacing = cloth_resolution.
         - Obstacle surface Z_obs = max(Z_inv) in each grid cell (lowest real Z).
      4. Subterranean Pit / Low-Point Outlier Suppression:
         - A multi-cell median filter with boundary reflection suppresses subterranean
           multipath and pit spikes so the cloth never snags on underground noise.
      5. Evolutionary Gradient Adaptation:
         - Computes terrain gradient magnitude G(x, y) = ||grad(Z_obs)||.
         - In flat areas: high rigidity pins cloth taut across roofs and tree canopies.
         - In steep areas (slopes, riverbanks): dynamically relaxes rigidity so the cloth
           conforms closely to the terrain.
      6. Particle Cloth Simulation & Slope Post-Processing (bSlopeSmooth):
         - Particles that touch ground become immovable anchor pins.
         - Spring tension pulls suspended roof particles up towards ground pins.
         - Slope post-processing pulls suspended particles into inverted V-troughs
           so rounded embankment crests, levees, and ridges are preserved.
      7. Bilinear Continuous Classification:
         - Bilinearly interpolates cloth height to exact point coordinates.
         - Distance tolerance scales with local slope & curvature: h(G, K) = h0 * (1 + beta * (G + K)).
         - Classifies all candidate points within h(G, K) of the cloth surface as ground.
    """
    n = len(xs)
    if n < 3:
        return np.ones(n, dtype=bool)

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    min_x, max_x = float(xs_f.min()), float(xs_f.max())
    min_y, max_y = float(ys_f.min()), float(ys_f.max())

    # Return and water surface filtering
    working_mask = None
    if not all_returns:
        working_mask = _last_return_working_mask(
            return_numbers, num_returns, sensor_type, n, exclude_single_returns_in_water
        )
    if working_mask is not None:
        grid_idx = np.flatnonzero(working_mask)
    elif not all_returns and return_numbers is not None and num_returns is not None:
        last_mask = np.asarray(return_numbers) == np.asarray(num_returns)
        grid_idx = np.flatnonzero(last_mask) if last_mask.sum() >= 3 else np.arange(n)
    else:
        grid_idx = np.arange(n)

    if progress:
        progress("EGS-CSF: leveling macro-relief and initializing cloth…", 10.0)

    # Step 1: Robust Macro-Relief Leveling (prevents gravity starvation on steep slopes)
    p10, p90 = float(np.percentile(zs_f, 10.0)), float(np.percentile(zs_f, 90.0))
    valid_trend = (zs_f >= p10) & (zs_f <= p90)
    if valid_trend.sum() < 10:
        valid_trend = np.ones(n, dtype=bool)

    x_mid = (min_x + max_x) * 0.5
    y_mid = (min_y + max_y) * 0.5
    x_c = xs_f - x_mid
    y_c = ys_f - y_mid
    A_sub = np.column_stack([x_c[valid_trend], y_c[valid_trend], np.ones(valid_trend.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A_sub, zs_f[valid_trend], rcond=None)
    z_plane = x_c * coeffs[0] + y_c * coeffs[1] + coeffs[2]

    # Normalized elevation: ground sits around ~0m across any terrain relief
    zs_norm = zs_f - z_plane

    # Step 2: Invert normalized point cloud (ground becomes upper boundary)
    z_inv = -zs_norm

    # Step 3: Build 2D Cloth Particle Grid
    cr = max(0.2, float(cloth_resolution))
    nx = max(3, int(np.ceil((max_x - min_x) / cr)) + 1)
    ny = max(3, int(np.ceil((max_y - min_y) / cr)) + 1)

    gx = np.clip(((xs_f - min_x) / cr).astype(np.int32), 0, nx - 1)
    gy = np.clip(((ys_f - min_y) / cr).astype(np.int32), 0, ny - 1)
    flat_idx = gx * ny + gy

    # Obstacle height in inverted space: maximum Z_inv in each cell (lowest real Z)
    z_obs = np.full(nx * ny, -np.inf, dtype=np.float64)
    np.maximum.at(z_obs, flat_idx[grid_idx], z_inv[grid_idx])
    z_obs_2d = z_obs.reshape((nx, ny))

    # Fill empty cells (voids) with nearest neighbor interpolation
    unoccupied = np.isneginf(z_obs_2d)
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

    # Step 4: Low-Point / Subterranean Pit Outlier Filter
    # In inverted space, subterranean noise pits (multipath / sensor glitches) manifest
    # as isolated tall spikes that snag the falling cloth and prevent it from reaching ground.
    # An adaptive 15m median filter with reflection cleanly suppresses single and cluster pits.
    if spike_down > 0:
        from scipy.ndimage import median_filter
        win_size = max(5, min(21, int(round(15.0 / cr))))
        if win_size % 2 == 0:
            win_size += 1
        med_grid = median_filter(z_obs_2d, size=win_size, mode="reflect")
        spike = z_obs_2d - med_grid
        z_obs_2d = np.where(spike > float(spike_down), med_grid, z_obs_2d)

    # Step 5: Evolutionary Gradient Calculation
    # Computes true world terrain slope (re-incorporating macro plane slope)
    # so embankments, riverbanks, and levees are detected accurately
    gz, gx_grad = np.gradient(z_obs_2d, cr)
    gx_world = -gx_grad + coeffs[0]
    gz_world = -gz + coeffs[1]
    grad_mag = np.sqrt(gx_world * gx_world + gz_world * gz_world)
    grad_max = float(np.percentile(grad_mag, 98.0)) if grad_mag.size else 1.0
    grad_norm = np.clip(grad_mag / max(1e-3, grad_max), 0.0, 1.0)

    # Step 6: Particle Cloth Simulation (Verlet Physics with Immovable Ground Pins)
    if progress:
        progress("EGS-CSF: running evolutionary cloth simulation…", 25.0)

    elev_range = float(np.max(z_obs_2d) - np.min(z_obs_2d))
    z_cloth = np.full((nx, ny), float(np.max(z_obs_2d)) + 0.10, dtype=np.float64)
    old_z = z_cloth.copy()
    movable = np.ones((nx, ny), dtype=bool)

    time_step_f = float(time_step)
    time_step2 = time_step_f * time_step_f
    target_fall_steps = max(20, int(max_iterations * 0.70))
    disp_needed = (elev_range + 2.0) / target_fall_steps
    gravity = max(0.2, disp_needed / time_step2)
    acc = -gravity

    base_r = max(1, min(6, int(round(rigidness))))
    singleMove1 = [0.0, 0.3, 0.51, 0.657, 0.7599, 0.83193, 0.88235]
    doubleMove1 = [0.0, 0.3, 0.42, 0.468, 0.4872, 0.4949, 0.498]

    for it in range(max_iterations):
        if progress and it % 25 == 0:
            progress(f"EGS-CSF: cloth iteration {it+1}/{max_iterations}…", 25.0 + 55.0 * it / max_iterations)

        temp = z_cloth.copy()
        # Verlet integration for movable particles
        z_cloth[movable] = z_cloth[movable] + (z_cloth[movable] - old_z[movable]) * 0.99 + acc * time_step2
        old_z = temp

        # Constraint relaxation: immovable ground pins pull movable roof/canopy particles up
        for c_iter in range(base_r):
            sm = singleMove1[min(c_iter + 1, 6)]
            dm = doubleMove1[min(c_iter + 1, 6)]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n_cloth = np.roll(z_cloth, shift=(di, dj), axis=(0, 1))
                n_movable = np.roll(movable, shift=(di, dj), axis=(0, 1))
                diff = n_cloth - z_cloth

                both = movable & n_movable
                z_cloth[both] += diff[both] * dm

                p1_only = movable & (~n_movable)
                z_cloth[p1_only] += diff[p1_only] * sm

        # Collision with inverted obstacle surface
        hit = z_cloth <= z_obs_2d
        z_cloth[hit] = z_obs_2d[hit]
        movable[hit] = False

        if it > target_fall_steps and float(np.max(np.abs(z_cloth - temp))) < 0.002:
            break

    # Step 6b: Slope & Embankment Crest Post-Processing (bSlopeSmooth)
    # In inverted space (Z_inv = -Z), convex rounded ridges, levees, and mounds become
    # deep V-shaped troughs. Internal cloth tension (rigidness) can suspend particles
    # across the trough like a hammock, leaving the cloth hanging below the true crest in real space.
    # Snaps suspended particles that neighbor ground pins over continuous terrain slopes
    # back down to the obstacle surface, perfectly recovering rounded embankment crests.
    if slope_smooth and movable.any():
        smooth_threshold = max(0.5, cr * 1.0)
        height_threshold = 2.5
        for _ in range(40):
            changed = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n_obs = np.roll(z_obs_2d, shift=(di, dj), axis=(0, 1))
                n_unmovable = np.roll(~movable, shift=(di, dj), axis=(0, 1))
                can_snap = (
                    movable
                    & n_unmovable
                    & (np.abs(z_obs_2d - n_obs) <= smooth_threshold)
                    & (np.abs(z_cloth - z_obs_2d) <= height_threshold)
                )
                if can_snap.any():
                    z_cloth[can_snap] = z_obs_2d[can_snap]
                    movable[can_snap] = False
                    changed = True
            if not changed:
                break

    # Step 7: Bilinear Interpolation of Cloth Elevation and Gradient-Adaptive Classification
    if progress:
        progress("EGS-CSF: classifying points with adaptive gradient threshold…", 85.0)

    # True cloth elevation in normalized original space
    z_cloth_orig = -z_cloth

    # Vectorized continuous bilinear interpolation of cloth surface
    gx_f = np.clip((xs_f - min_x) / cr, 0.0, nx - 1.0)
    gy_f = np.clip((ys_f - min_y) / cr, 0.0, ny - 1.0)
    c0 = np.clip(np.floor(gx_f).astype(np.int32), 0, nx - 2)
    r0 = np.clip(np.floor(gy_f).astype(np.int32), 0, ny - 2)
    fx = gx_f - c0
    fy = gy_f - r0

    point_cloth_norm = (
        z_cloth_orig[c0, r0] * (1.0 - fx) * (1.0 - fy) +
        z_cloth_orig[c0 + 1, r0] * fx * (1.0 - fy) +
        z_cloth_orig[c0, r0 + 1] * (1.0 - fx) * fy +
        z_cloth_orig[c0 + 1, r0 + 1] * fx * fy
    )
    # Restore full world elevation
    point_cloth_world = point_cloth_norm + z_plane
    point_grad = grad_mag[gx, gy]

    # Curvature / convex breakline detection: detects rounded crests and mounds where slope flattens at the peak
    curv = np.abs(
        np.roll(z_obs_2d, 1, 0) + np.roll(z_obs_2d, -1, 0) +
        np.roll(z_obs_2d, 1, 1) + np.roll(z_obs_2d, -1, 1) - 4.0 * z_obs_2d
    ) / (cr * cr)
    point_curv = curv[gx, gy]

    # Adaptive distance threshold: h(G, K) = h0 * (1 + beta * (G + K * cr))
    h_dynamic = float(class_threshold) * (1.0 + float(gradient_factor) * np.clip(point_grad + point_curv * cr, 0.0, 3.0))

    # Point is ground if distance to cloth surface is within h_dynamic
    ground_mask = np.abs(zs_f - point_cloth_world) <= h_dynamic
    if working_mask is not None:
        ground_mask = ground_mask & working_mask

    if progress:
        progress("EGS-CSF: done", 100.0)
    logger.info("EGS-CSF: classified %d ground / %d points (%.1f%%)", int(ground_mask.sum()), n, 100.0 * ground_mask.sum() / n)
    return ground_mask


# ─────────────────────────────────────────────────────────────────────
# Step-Down PTD: Adaptive Multi-Scale Bulged TIN
# ─────────────────────────────────────────────────────────────────────

def ground_classify_stepdown(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    step: float = 5.0,
    sub_steps: int = 5,
    bulge: Optional[float] = None,
    offset: float = 0.10,
    spike: float = 1.0,
    spike_down: float = 1.0,
    refine_loops: int = 2,
    all_returns: bool = False,
    return_numbers: Optional[np.ndarray] = None,
    num_returns: Optional[np.ndarray] = None,
    existing_ground_mask: Optional[np.ndarray] = None,
    progress: ProgressCB = None,
    **kwargs,
) -> np.ndarray:
    """
    Adaptive Step-Down Progressive TIN Densification (Multi-Scale Bulged TIN).

    Algorithm:
      1. Coarse lowest-Z grid seeding at resolution `step` with 4-corner boundary anchors.
      2. Local neighborhood outlier cleaning (up-spikes and down-spikes/pits).
      3. Multi-scale geometric step-down refinement across `sub_steps` levels (step -> fine).
      4. Bulged TIN interpolation using vertex normal blending to allow the TIN to
         curve upward into embankments, ridges, and mounds without planar chord clipping.
      5. Fast progressive refinement passes (`refine_loops`) for micro-topography convergence.
      6. Final slope-adaptive classification: dz <= offset * sqrt(1 + slope^2).

    Args:
        xs, ys, zs:           Point cloud coordinates.
        step:                 Initial search grid resolution in meters (e.g. 5m nature, 3m riverbanks, 25m city).
        sub_steps:            Number of multi-scale step-down levels (default 5).
        bulge:                Max upward/downward TIN bending allowance (meters). Default: step/10 clamped to [1.0, 2.0].
        offset:               Max vertical distance above bulged TIN to be classified as ground (default 0.10m).
        spike:                Up-spike removal threshold (meters, default 1.0m).
        spike_down:           Down-spike / pit removal threshold (meters, default 1.0m).
        refine_loops:         Max progressive refinement iterations (default 2).
        all_returns:          If False (default), only last returns are considered for initial seeds.
        return_numbers:       Array of return numbers (optional).
        num_returns:          Array of total number of returns (optional).
        existing_ground_mask: Boolean mask of trusted ground seeds (for '1 -> 2' densification workflows).
        progress:             Optional progress callback (message, percentage).

    Returns:
        Boolean mask where True indicates Class 2 Ground points.
    """
    n = len(xs)
    if n < 3:
        return np.ones(n, dtype=bool)

    from scipy.spatial import Delaunay, cKDTree

    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    zs_f = np.asarray(zs, dtype=np.float64)

    # Return filtering: default prefers last returns
    if not all_returns and return_numbers is not None and num_returns is not None:
        last_mask = np.asarray(return_numbers) == np.asarray(num_returns)
        cand_indices = np.flatnonzero(last_mask) if last_mask.sum() >= 3 else np.arange(n)
    else:
        cand_indices = np.arange(n)

    min_x, max_x = float(xs_f.min()), float(xs_f.max())
    min_y, max_y = float(ys_f.min()), float(ys_f.max())

    if bulge is None:
        bulge = float(np.clip(step / 10.0 if step > 5.0 else step / 5.0, 1.0, 2.0))
    else:
        bulge = float(max(0.0, bulge))

    s0 = max(0.5, float(step))
    offset = float(max(0.01, offset))
    spike = float(max(0.0, spike))
    spike_down = float(max(0.0, spike_down))

    if progress:
        progress("Step-Down PTD: seeding initial coarse grid…", 5.0)

    # Step 1: Initial Coarse Grid Seeds (vectorized)
    c_mask = _lowest_per_grid_cells(
        xs_f[cand_indices], ys_f[cand_indices], zs_f[cand_indices],
        min_x, min_y, max_x, max_y,
        s0, 0.0, 0.0
    )
    coarse_seeds = list(cand_indices[c_mask])

    # Add 4 bounding box corner anchors to prevent margin collapse
    for cx, cy in ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)):
        d2 = (xs_f[cand_indices] - cx)**2 + (ys_f[cand_indices] - cy)**2
        k_near = min(5, len(cand_indices))
        n_part = np.argpartition(d2, k_near - 1)[:k_near]
        coarse_seeds.append(cand_indices[n_part[np.argmin(zs_f[cand_indices[n_part]])]])
    coarse_seeds = np.array(list(set(coarse_seeds)), dtype=np.int64)

    # Spike cleaning on coarse seeds (removes tree tops, roofs, and underground pits)
    if len(coarse_seeds) > 4 and (spike > 0 or spike_down > 0):
        seed_xy = np.column_stack([xs_f[coarse_seeds], ys_f[coarse_seeds]])
        seed_z = zs_f[coarse_seeds]
        kdt = cKDTree(seed_xy)
        clean_mask = np.ones(len(coarse_seeds), dtype=bool)
        dists, idxs = kdt.query(seed_xy, k=min(7, len(coarse_seeds)), distance_upper_bound=2.0 * s0)
        for i in range(len(coarse_seeds)):
            valid = [idx for d, idx in zip(dists[i], idxs[i]) if d > 1e-4 and idx < len(coarse_seeds)]
            if valid:
                med_z = float(np.median(seed_z[valid]))
                if spike > 0 and (seed_z[i] - med_z) > spike:
                    clean_mask[i] = False
                elif spike_down > 0 and (med_z - seed_z[i]) > spike_down:
                    clean_mask[i] = False
        coarse_seeds = coarse_seeds[clean_mask]

    current_seeds = set(coarse_seeds.tolist())

    # Include existing trusted ground if provided
    if existing_ground_mask is not None:
        trusted_indices = np.flatnonzero(existing_ground_mask)
        if len(trusted_indices) > 0:
            current_seeds.update(trusted_indices.tolist())

    # Helper: compute facet normals, vertex normals, and slopes
    def _compute_normals(s_xy, s_z, tri_obj):
        t_pts = s_xy[tri_obj.simplices]
        t_z = s_z[tri_obj.simplices]
        v0 = np.column_stack([t_pts[:, 1] - t_pts[:, 0], t_z[:, 1] - t_z[:, 0]])
        v1 = np.column_stack([t_pts[:, 2] - t_pts[:, 0], t_z[:, 2] - t_z[:, 0]])
        fn = np.cross(v0, v1)
        flen = np.linalg.norm(fn, axis=1, keepdims=True)
        flen[flen == 0] = 1.0
        fn /= flen
        flip = fn[:, 2] < 0
        fn[flip] = -fn[flip]

        vn = np.zeros((len(s_xy), 3), dtype=np.float64)
        for k in range(3):
            np.add.at(vn, tri_obj.simplices[:, k], fn)
        vlen = np.linalg.norm(vn, axis=1, keepdims=True)
        vlen[vlen == 0] = 1.0
        vn /= vlen
        flip_v = vn[:, 2] < 0
        vn[flip_v] = -vn[flip_v]
        vn[:, 2] = np.maximum(1e-4, vn[:, 2])

        nz = np.maximum(1e-4, fn[:, 2])
        slopes = np.sqrt(fn[:, 0]**2 + fn[:, 1]**2) / nz
        return fn, vn, slopes

    # Helper: bulged TIN elevation query with fast direct matrix multiplication
    def _query_bulged(q_xy, s_xy, s_z, tri_obj, fn, vn, slopes, b_val):
        simps = tri_obj.find_simplex(q_xy)
        inside = simps >= 0
        z_out = np.full(len(q_xy), np.nan, dtype=np.float64)
        sl_out = np.zeros(len(q_xy), dtype=np.float64)

        idx_in = np.flatnonzero(inside)
        if len(idx_in) > 0:
            valid_simps = simps[inside]
            T = tri_obj.transform[valid_simps]
            r3 = s_xy[tri_obj.simplices[valid_simps, 2]]
            dxy = q_xy[idx_in] - r3
            u = T[:, 0, 0] * dxy[:, 0] + T[:, 0, 1] * dxy[:, 1]
            v = T[:, 1, 0] * dxy[:, 0] + T[:, 1, 1] * dxy[:, 1]
            w = 1.0 - u - v

            i0 = tri_obj.simplices[valid_simps, 0]
            i1 = tri_obj.simplices[valid_simps, 1]
            i2 = tri_obj.simplices[valid_simps, 2]

            z_lin = u * s_z[i0] + v * s_z[i1] + w * s_z[i2]

            if b_val > 0:
                p0, p1, p2 = s_xy[i0], s_xy[i1], s_xy[i2]
                vn0, vn1, vn2 = vn[i0], vn[i1], vn[i2]
                p_q = q_xy[idx_in]
                inv_vn0 = 1.0 / vn0[:, 2]
                inv_vn1 = 1.0 / vn1[:, 2]
                inv_vn2 = 1.0 / vn2[:, 2]
                zt0 = s_z[i0] - ((p_q[:, 0] - p0[:, 0]) * vn0[:, 0] + (p_q[:, 1] - p0[:, 1]) * vn0[:, 1]) * inv_vn0
                zt1 = s_z[i1] - ((p_q[:, 0] - p1[:, 0]) * vn1[:, 0] + (p_q[:, 1] - p1[:, 1]) * vn1[:, 1]) * inv_vn1
                zt2 = s_z[i2] - ((p_q[:, 0] - p2[:, 0]) * vn2[:, 0] + (p_q[:, 1] - p2[:, 1]) * vn2[:, 1]) * inv_vn2
                z_curved = u * zt0 + v * zt1 + w * zt2
                dz_bulge = np.clip(z_curved - z_lin, -b_val, b_val)
                z_res = z_lin + dz_bulge
            else:
                z_res = z_lin

            z_out[idx_in] = z_res
            sl_out[idx_in] = slopes[valid_simps]

        # Margin points outside convex hull: extrapolate using tangent plane of nearest seed
        idx_out = np.flatnonzero(~inside)
        if len(idx_out) > 0 and len(s_xy) > 0:
            kdt_seeds = cKDTree(s_xy)
            _, n_idx = kdt_seeds.query(q_xy[idx_out])
            p0 = s_xy[n_idx]
            vn0 = vn[n_idx]
            z0 = s_z[n_idx]
            p_q = q_xy[idx_out]
            z_tan = z0 - ((p_q[:, 0] - p0[:, 0]) * vn0[:, 0] + (p_q[:, 1] - p0[:, 1]) * vn0[:, 1]) / vn0[:, 2]
            z_out[idx_out] = z_tan
            sl_out[idx_out] = np.sqrt(vn0[:, 0]**2 + vn0[:, 1]**2) / vn0[:, 2]

        return z_out, sl_out, np.ones(len(q_xy), dtype=bool)

    # Step 2: Multi-Scale Substep Densification
    sub_steps = max(1, int(sub_steps))
    target_res = max(0.5, s0 / (2.0 ** sub_steps))
    cell_steps = np.geomspace(s0, target_res, sub_steps + 1)[1:]

    for step_idx, cur_step in enumerate(cell_steps):
        if len(current_seeds) < 3:
            break

        if progress:
            progress(f"Step-Down PTD: multi-scale step {step_idx+1}/{len(cell_steps)} ({cur_step:.1f}m)…",
                     10.0 + 50.0 * (step_idx / len(cell_steps)))

        s_arr = np.array(list(current_seeds), dtype=np.int64)
        s_xy = np.column_stack([xs_f[s_arr], ys_f[s_arr]])
        s_z = zs_f[s_arr]

        try:
            tri = Delaunay(s_xy)
        except Exception:
            break

        fn, vn, slopes = _compute_normals(s_xy, s_z, tri)
        cur_bulge = bulge * (cur_step / s0) ** 0.5

        # Fast vectorized candidate selection per cell
        step_mask = _lowest_per_grid_cells(
            xs_f[cand_indices], ys_f[cand_indices], zs_f[cand_indices],
            min_x, min_y, max_x, max_y,
            cur_step, 0.0, 0.0
        )
        cell_min_cands = cand_indices[step_mask]
        unseeded_mask = np.fromiter((c not in current_seeds for c in cell_min_cands), dtype=bool, count=len(cell_min_cands))
        cell_min_cands = cell_min_cands[unseeded_mask]
        if len(cell_min_cands) == 0:
            continue

        cand_xy = np.column_stack([xs_f[cell_min_cands], ys_f[cell_min_cands]])
        z_tin, sl, inside = _query_bulged(cand_xy, s_xy, s_z, tri, fn, vn, slopes, cur_bulge)
        if not inside.any():
            continue

        valid_cands = cell_min_cands[inside]
        dz = zs_f[valid_cands] - z_tin[inside]
        sl_val = sl[inside]

        tol = offset * np.sqrt(1.0 + sl_val**2) + cur_bulge
        accept = (dz >= -spike_down) & (dz <= tol)

        # Retain topographically meaningful points (crests, ditches, relief changes)
        detail_needed = (dz < -0.01) | (dz > 0.04) | (cur_step >= 2.0)
        to_insert = valid_cands[accept & detail_needed]
        current_seeds.update(to_insert.tolist())

    # Step 3: Progressive Refinement Passes (fast convergence)
    refine_loops = min(20, max(0, int(refine_loops)))
    for it in range(refine_loops):
        if len(current_seeds) < 3:
            break

        if progress and (it == 0 or it % 2 == 0):
            progress(f"Step-Down PTD: refinement pass {it+1}/{refine_loops}…", 60.0 + 25.0 * (it / max(1, refine_loops)))

        s_arr = np.array(list(current_seeds), dtype=np.int64)
        s_xy = np.column_stack([xs_f[s_arr], ys_f[s_arr]])
        s_z = zs_f[s_arr]

        try:
            tri = Delaunay(s_xy)
        except Exception:
            break

        fn, vn, slopes = _compute_normals(s_xy, s_z, tri)

        unseeded = np.array([idx for idx in cand_indices if idx not in current_seeds], dtype=np.int64)
        if len(unseeded) == 0:
            break

        q_unseeded = unseeded if len(unseeded) <= 10000 else np.random.choice(unseeded, 10000, replace=False)
        cand_xy = np.column_stack([xs_f[q_unseeded], ys_f[q_unseeded]])
        z_tin, sl, inside = _query_bulged(cand_xy, s_xy, s_z, tri, fn, vn, slopes, bulge * 0.5)

        if not inside.any():
            break

        valid = q_unseeded[inside]
        dz = zs_f[valid] - z_tin[inside]
        sl_val = sl[inside]
        tol = offset * np.sqrt(1.0 + sl_val**2) + (bulge * 0.5)

        acceptable = (dz >= -spike_down) & (dz <= tol)
        accepted = valid[acceptable]

        if len(accepted) < max(5, int(0.005 * len(s_arr))):
            break

        # Thin accepted candidates before adding
        thin_res = max(1.0, target_res)
        thin_m = _lowest_per_grid_cells(
            xs_f[accepted], ys_f[accepted], zs_f[accepted],
            min_x, min_y, max_x, max_y,
            thin_res, 0.0, 0.0
        )
        new_seeds = accepted[thin_m]
        current_seeds.update(new_seeds.tolist())

    # Step 4: Final Surface Classification Pass
    if progress:
        progress("Step-Down PTD: final surface classification…", 90.0)

    s_arr = np.array(list(current_seeds), dtype=np.int64)
    s_xy = np.column_stack([xs_f[s_arr], ys_f[s_arr]])
    s_z = zs_f[s_arr]

    tri = Delaunay(s_xy)
    fn, vn, slopes = _compute_normals(s_xy, s_z, tri)

    all_xy = np.column_stack([xs_f, ys_f])
    z_tin, sl, inside = _query_bulged(all_xy, s_xy, s_z, tri, fn, vn, slopes, bulge)

    ground_mask = np.zeros(n, dtype=bool)
    if inside.any():
        idx_in = np.flatnonzero(inside)
        dz = zs_f[idx_in] - z_tin[idx_in]
        sl_val = sl[idx_in]
        tol = offset * np.sqrt(1.0 + sl_val**2)
        is_ground = (dz >= -spike_down) & (dz <= tol)
        ground_mask[idx_in[is_ground]] = True

    ground_mask[s_arr] = True
    if existing_ground_mask is not None:
        ground_mask[existing_ground_mask] = True

    if progress:
        progress("Step-Down PTD: done", 100.0)

    logger.info("Step-Down PTD: classified %d ground / %d points (%.1f%%)",
                int(ground_mask.sum()), n, 100.0 * ground_mask.sum() / max(1, n))
    return ground_mask



