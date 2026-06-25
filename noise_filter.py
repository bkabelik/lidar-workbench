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
