"""
LiDAR Workbench — Multi-Sensor Auto-Alignment Module.

Provides ICP-based rigid registration to align a bathymetric scanner
(VQ-870-G) to a topographic reference (VUX-160) using overlapping
ground surfaces.  Fully automatic — no manual tie points required.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("lidar_workbench.alignment")


def extract_ground_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    cell_size: Optional[float] = None,
    percentile: float = 5.0,
) -> np.ndarray:
    """
    Extract a rough ground surface using a gridded minimum filter.

    Args:
        xs, ys, zs:    Point coordinates.
        cell_size:     Grid cell size in CRS units. Auto-computed from
                       point spacing when ``None``.
        percentile:    Percentile of Z to take per cell (5 = lowest 5%).

    Returns:
        ``(N, 3)`` array of ground-point candidates.
    """
    n = len(xs)
    if n < 100:
        return np.column_stack((xs, ys, zs))

    if cell_size is None:
        # Auto-compute from point spacing
        from scipy.spatial import KDTree
        idx = np.random.choice(n, min(5000, n), replace=False)
        tree = KDTree(np.column_stack((xs[idx], ys[idx])))
        dist, _ = tree.query(tree.data, k=2)
        spacing = float(np.median(dist[:, 1])) if dist.ndim > 1 else 1.0
        cell_size = max(5.0 * spacing, 0.5)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    nx = max(1, int(np.ceil((max_x - min_x) / cell_size)))
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size)))

    # Per-cell lowest points
    gx = np.clip(((xs - min_x) / cell_size).astype(int), 0, nx - 1)
    gy = np.clip(((ys - min_y) / cell_size).astype(int), 0, ny - 1)

    keep = np.zeros(n, dtype=bool)
    for xi in range(nx):
        for yi in range(ny):
            cell_mask = (gx == xi) & (gy == yi)
            if cell_mask.sum() < 3:
                continue
            cell_zs = zs[cell_mask]
            z_cutoff = float(np.percentile(cell_zs, percentile))
            keep[cell_mask & (zs <= z_cutoff)] = True

    result = np.column_stack((xs[keep], ys[keep], zs[keep]))
    logger.info(
        "Ground extraction: %d / %d points (cell=%.1f m)",
        len(result), n, cell_size,
    )
    return result


def align_point_clouds(
    source_xs: np.ndarray,
    source_ys: np.ndarray,
    source_zs: np.ndarray,
    target_xs: np.ndarray,
    target_ys: np.ndarray,
    target_zs: np.ndarray,
    max_correspondence: float = 2.0,
    use_ground_only: bool = True,
) -> Optional[np.ndarray]:
    """
    Align a source point cloud (VQ-870-G bathy) to a target reference
    (VUX-160 topo) using ICP with point-to-plane metric.

    Args:
        source_xs/ys/zs:   Source point cloud to be transformed.
        target_xs/ys/zs:   Target reference point cloud.
        max_correspondence: Maximum correspondence distance for ICP (metres).
        use_ground_only:   If ``True``, extract ground points first for a
                           more robust alignment.

    Returns:
        ``(4, 4)`` rigid transformation matrix, or ``None`` if alignment
        fails.  Apply with ``source = (R @ source.T + t).T``.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("Open3D not available — cannot run ICP alignment")
        return None

    source_pts = np.column_stack((source_xs, source_ys, source_zs))
    target_pts = np.column_stack((target_xs, target_ys, target_zs))

    if use_ground_only:
        source_pts = extract_ground_points(source_xs, source_ys, source_zs)
        target_pts = extract_ground_points(target_xs, target_ys, target_zs)

    if len(source_pts) < 100 or len(target_pts) < 100:
        logger.warning("Not enough ground points for alignment")
        return None

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_pts)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_pts)

    # Estimate normals for point-to-plane ICP
    try:
        src_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
    except Exception:
        pass  # fall back to point-to-point

    logger.info(
        "ICP aligning %d → %d points (max_corr=%.1f m)",
        len(source_pts), len(target_pts), max_correspondence,
    )

    try:
        reg = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd,
            max_correspondence_distance=max_correspondence,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100,
            ),
        )
    except Exception as exc:
        logger.warning("ICP failed: %s", exc)
        return None

    logger.info(
        "ICP result: fitness=%.3f, rmse=%.4f",
        reg.fitness, reg.inlier_rmse,
    )
    return np.asarray(reg.transformation)


def compute_vertical_shift(
    source_xs: np.ndarray,
    source_ys: np.ndarray,
    source_zs: np.ndarray,
    target_xs: np.ndarray,
    target_ys: np.ndarray,
    target_zs: np.ndarray,
    grid_res: float = 2.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute a robust median vertical shift between two overlapping
    point clouds using grid-based Z comparison.

    Faster and simpler than full ICP — works well when the primary
    error is a constant vertical offset.

    Args:
        source_xs/ys/zs:  Source point cloud.
        target_xs/ys/zs:  Target reference.
        grid_res:         Grid resolution for Z comparison.

    Returns:
        ``(shift_z, dz_per_point)`` — median Z shift to add to source,
        and the per-point Z differences (for diagnostics).
    """
    n_src = len(source_xs)
    if n_src < 100:
        return 0.0, np.array([])

    # Grid the target's Z
    min_x = min(source_xs.min(), target_xs.min())
    max_x = max(source_xs.max(), target_xs.max())
    min_y = min(source_ys.min(), target_ys.min())
    max_y = max(source_ys.max(), target_ys.max())

    nx = max(1, int(np.ceil((max_x - min_x) / grid_res)))
    ny = max(1, int(np.ceil((max_y - min_y) / grid_res)))

    grid_z = np.full((nx, ny), np.nan, dtype=np.float64)
    grid_count = np.zeros((nx, ny), dtype=int)

    t_gx = np.clip(((target_xs - min_x) / grid_res).astype(int), 0, nx - 1)
    t_gy = np.clip(((target_ys - min_y) / grid_res).astype(int), 0, ny - 1)
    for i in range(len(target_xs)):
        gxi, gyi = t_gx[i], t_gy[i]
        if np.isnan(grid_z[gxi, gyi]):
            grid_z[gxi, gyi] = target_zs[i]
        else:
            grid_z[gxi, gyi] += target_zs[i]
        grid_count[gxi, gyi] += 1

    mask = grid_count > 0
    grid_z[mask] /= grid_count[mask]

    # Compare each source point to its grid cell
    s_gx = np.clip(((source_xs - min_x) / grid_res).astype(int), 0, nx - 1)
    s_gy = np.clip(((source_ys - min_y) / grid_res).astype(int), 0, ny - 1)

    dz = np.full(n_src, np.nan)
    for i in range(n_src):
        gxi, gyi = s_gx[i], s_gy[i]
        if not np.isnan(grid_z[gxi, gyi]):
            dz[i] = grid_z[gxi, gyi] - source_zs[i]

    valid = ~np.isnan(dz)
    if valid.sum() == 0:
        return 0.0, dz

    shift_z = float(np.median(dz[valid]))
    logger.info(
        "Vertical shift: %.3f m (from %d matched cells)",
        shift_z, valid.sum(),
    )
    return shift_z, dz


def apply_transformation(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a 4×4 rigid transformation matrix to point coordinates.

    Args:
        xs, ys, zs:  Point coordinates.
        transform:   ``(4, 4)`` homogeneous transformation matrix.

    Returns:
        ``(transformed_xs, transformed_ys, transformed_zs)``.
    """
    n = len(xs)
    pts = np.column_stack((xs, ys, zs, np.ones(n)))
    transformed = (transform @ pts.T).T
    return transformed[:, 0], transformed[:, 1], transformed[:, 2]
