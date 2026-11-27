"""
LiDAR Workbench — Multi-Platform Adaptive Outlier Classifier.

Implements a two-phase "Evaluate First, Then Adaptively Classify Outliers" pipeline:
  Phase 1: Fast Point Cloud Evaluation (`evaluate_point_cloud` -> `CloudEvaluationReport`)
  Phase 2: Scale-Normalized SOTA Adaptive Outlier Classifier (`OutlierParams` +
           `classify_outliers_adaptive`) outputting non-destructive ASPRS Class 7
           (Low Noise) and Class 18 (High Noise) masks/labels.
  Phase 3 helpers: Decoupled SQLite undo/redo batch application.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger("lidar_workbench.adaptive_outlier")

ASPRS_LOW_NOISE = 7
ASPRS_WATER = 9
ASPRS_HIGH_NOISE = 18
ASPRS_BATHY_BOTTOM = 40
ASPRS_BATHY_WATER_SURFACE = 41


def _get_extra_dim(data: Dict[str, Any], name: str) -> Optional[np.ndarray]:
    """Retrieve a RIEGL/LAS extra dimension case-insensitively."""
    if name in data and isinstance(data[name], np.ndarray):
        return data[name]
    ed = data.get("extra_dims")
    if isinstance(ed, dict):
        if name in ed and isinstance(ed[name], np.ndarray):
            return ed[name]
        low = name.lower()
        for k, v in ed.items():
            if str(k).lower() == low and isinstance(v, np.ndarray):
                return v
    return None


@dataclass
class CloudEvaluationReport:
    """Fast statistical & geometric summary of a point cloud tile/dataset."""

    point_count: int = 0
    median_density_pts_m2: float = 20.0
    nps_m: float = 0.22
    density_heterogeneity: float = 1.0  # P90 / P10
    flightline_count: int = 1
    scan_angle_spread_deg: float = 0.0
    has_reflectance: bool = False
    has_amplitude: bool = False
    has_deviation: bool = False
    estimated_platform: str = "fixed_wing"  # "fixed_wing" | "helicopter" | "drone"
    relief_p50_m: float = 0.5
    relief_p90_m: float = 2.0
    has_bathymetry: bool = False
    eval_time_ms: float = 0.0


def evaluate_point_cloud(data: Dict[str, Any]) -> CloudEvaluationReport:
    """
    Evaluate point cloud geometry, density, flightlines, RIEGL extra bytes,
    platform type, terrain relief, and bathymetry in <100 ms on 5M points.
    """
    t0 = time.perf_counter()
    xs = np.asarray(data["x"])
    ys = np.asarray(data["y"])
    zs = np.asarray(data["z"])
    n = int(len(xs))
    if n == 0:
        return CloudEvaluationReport(point_count=0, eval_time_ms=0.0)

    refl = _get_extra_dim(data, "reflectance")
    ampl = _get_extra_dim(data, "amplitude")
    dev = _get_extra_dim(data, "deviation")

    has_reflectance = refl is not None and len(refl) == n
    has_amplitude = ampl is not None and len(ampl) == n
    has_deviation = dev is not None and len(dev) == n

    # Deterministic strided subsample (<= 200k pts) keeps 5M-pt eval well under 100 ms
    max_sub = 200_000
    step = max(1, n // max_sub)
    sx = xs[::step]
    sy = ys[::step]
    sz = zs[::step]
    scale_factor = float(n) / float(len(sx))

    # 1. 2D cell density & heterogeneity (adaptive cell size from XY span)
    x_min, x_max = float(sx.min()), float(sx.max())
    y_min, y_max = float(sy.min()), float(sy.max())
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    area_approx = span_x * span_y
    rough_density = max(0.1, float(n) / area_approx)
    # Aim for ~40-200 pts/cell on full cloud, bounded between 1m and 10m
    cell_sz = float(np.clip(math.sqrt(60.0 / rough_density), 1.0, 10.0))
    cell_area = cell_sz * cell_sz

    ix = np.floor((sx - x_min) / cell_sz).astype(np.int64)
    iy = np.floor((sy - y_min) / cell_sz).astype(np.int64)
    stride_y = int(iy.max() - iy.min() + 2) if len(iy) else 1
    cell_keys = ix * stride_y + iy

    order = np.argsort(cell_keys, kind="mergesort")
    sorted_keys = cell_keys[order]
    sorted_z = sz[order]
    uniq_mask = np.empty(len(sorted_keys), dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = np.flatnonzero(uniq_mask)
    counts = np.diff(np.append(starts, len(sorted_keys)))

    # Exclude edge/sparse 1-pt cells when computing density percentiles if enough cells exist
    valid_counts = counts[counts >= 2] if np.sum(counts >= 2) >= 10 else counts
    densities = (valid_counts.astype(np.float64) * scale_factor) / cell_area
    p10_d, p50_d, p90_d = np.percentile(densities, [10.0, 50.0, 90.0])
    median_density = max(0.1, float(p50_d))
    nps_m = 1.0 / math.sqrt(median_density)
    heterogeneity = float(p90_d / max(p10_d, 1e-6))

    # 2. Terrain relief (2D cell Z-range percentiles using P95-P5 per cell to ignore single outliers)
    # Fast vectorized min/max per cell on cells with >= 3 points
    z_max_cell = np.maximum.reduceat(sorted_z, starts)
    z_min_cell = np.minimum.reduceat(sorted_z, starts)
    z_ranges = (z_max_cell - z_min_cell)[counts >= 3] if np.any(counts >= 3) else (z_max_cell - z_min_cell)
    if len(z_ranges) > 0:
        relief_p50, relief_p90 = np.percentile(z_ranges, [50.0, 90.0])
    else:
        relief_p50, relief_p90 = 0.5, 2.0

    # 3. Flightline count & scan-angle spread
    psid = data.get("point_source_id")
    if isinstance(psid, np.ndarray) and len(psid) == n:
        flightline_count = int(len(np.unique(psid[::step])))
    else:
        flightline_count = 1

    sa = data.get("scan_angle")
    if not (isinstance(sa, np.ndarray) and len(sa) == n):
        sa = data.get("scan_angle_rank")
    if isinstance(sa, np.ndarray) and len(sa) == n:
        sa_sub = sa[::step].astype(np.float64)
        p5_a, p95_a = np.percentile(sa_sub, [5.0, 95.0])
        scan_angle_spread = float(max(0.0, p95_a - p5_a))
    else:
        scan_angle_spread = 0.0

    # 4. Platform inference from density / NPS / scan-angle geometry
    if median_density >= 150.0:
        estimated_platform = "drone"
    elif median_density >= 45.0 or (flightline_count >= 2 and nps_m <= 0.18 and scan_angle_spread <= 45.0):
        estimated_platform = "helicopter"
    else:
        estimated_platform = "fixed_wing"

    # 5. Bathymetry detection
    has_bathymetry = False
    cls_arr = data.get("classification")
    if isinstance(cls_arr, np.ndarray) and len(cls_arr) == n:
        cls_sub = cls_arr[::step]
        if np.any((cls_sub == ASPRS_WATER) | (cls_sub == ASPRS_BATHY_BOTTOM) | (cls_sub == ASPRS_BATHY_WATER_SURFACE)):
            has_bathymetry = True

    if not has_bathymetry and has_reflectance:
        nret = data.get("num_returns")
        if not (isinstance(nret, np.ndarray) and len(nret) == n):
            nret = data.get("number_of_returns")
        if isinstance(nret, np.ndarray) and len(nret) == n:
            refl_sub = refl[::step]
            nret_sub = nret[::step]
            water_like = (refl_sub < -18.0) & (nret_sub > 1)
            if np.sum(water_like) >= max(20, int(0.005 * len(sx))):
                has_bathymetry = True

    eval_ms = (time.perf_counter() - t0) * 1000.0
    return CloudEvaluationReport(
        point_count=n,
        median_density_pts_m2=round(median_density, 2),
        nps_m=round(nps_m, 4),
        density_heterogeneity=round(heterogeneity, 2),
        flightline_count=max(1, flightline_count),
        scan_angle_spread_deg=round(scan_angle_spread, 2),
        has_reflectance=has_reflectance,
        has_amplitude=has_amplitude,
        has_deviation=has_deviation,
        estimated_platform=estimated_platform,
        relief_p50_m=round(float(relief_p50), 3),
        relief_p90_m=round(float(relief_p90), 3),
        has_bathymetry=has_bathymetry,
        eval_time_ms=round(eval_ms, 2),
    )


@dataclass
class OutlierParams:
    """Scale-normalized parameters for `classify_outliers_adaptive`."""

    preset_name: str = "auto"

    # 1. Return-aware RIEGL Radiometric & Pulse-Shape Gate
    use_reflectance_gate: bool = True
    min_reflectance_db: float = -22.0
    max_pulse_deviation: float = 25.0
    protect_multi_return_canopy: bool = True

    # 2. 3D Voxel Isolation Pre-Filter
    voxel_size_m: float = 0.65
    min_voxel_neighbors: int = 2

    # 3. Flightline-Aware Asymmetric IRLS Lower Surface & Upper Envelope
    grid_cell_size_m: float = 2.0
    irls_iterations: int = 3
    z_lower_mad_multiplier: float = 5.0
    z_upper_mad_multiplier: float = 6.0
    per_strip_first: bool = False
    max_strip_discrepancy_m: float = 1.5

    # 4. Multi-Scale Consistency + Return-Aware Radiometric SOR
    sor_k_small: int = 8
    sor_k_large: int = 24
    sor_std_multiplier: float = 2.5
    max_plane_residual_m: float = 0.35
    max_normal_angle_deg: float = 25.0
    radiometric_weight: float = 0.5

    # 5. Conditional Bathymetry Sub-Pass
    bathy_mode: str = "auto"  # "auto" | "on" | "off"

    @classmethod
    def _from_platform(
        cls,
        platform: str,
        report: Optional[CloudEvaluationReport] = None,
        preset_name: Optional[str] = None,
    ) -> "OutlierParams":
        nps = report.nps_m if report else (
            0.06 if platform == "drone" else (0.10 if platform == "helicopter" else 0.22)
        )
        nps = max(0.03, min(1.5, float(nps)))
        has_refl = report.has_reflectance if report else True
        multi_strip = (report.flightline_count > 1) if report else (platform == "helicopter")
        steep_relief = (report.relief_p90_m > 3.0) if report else False

        voxel_mult = 3.0 if platform != "fixed_wing" else 3.5
        cell_mult = 8.0 if platform != "drone" else 10.0
        upper_mad = 7.5 if steep_relief else (6.5 if platform == "helicopter" else 6.0)
        strip_disc = 1.8 if platform == "helicopter" else (2.5 if platform == "fixed_wing" else 1.2)

        return cls(
            preset_name=preset_name or platform,
            use_reflectance_gate=has_refl,
            min_reflectance_db=-22.0,
            max_pulse_deviation=25.0 if platform != "drone" else 30.0,
            protect_multi_return_canopy=True,
            voxel_size_m=round(max(0.90, 10.0 * nps), 3),
            min_voxel_neighbors=2,
            grid_cell_size_m=round(max(0.5, cell_mult * nps), 3),
            irls_iterations=3,
            z_lower_mad_multiplier=5.0,
            z_upper_mad_multiplier=upper_mad,
            per_strip_first=(platform == "helicopter" and multi_strip),
            max_strip_discrepancy_m=strip_disc,
            sor_k_small=8,
            sor_k_large=24,
            sor_std_multiplier=2.5 if platform != "drone" else 2.8,
            max_plane_residual_m=round(max(0.08, 1.5 * nps), 3),
            max_normal_angle_deg=25.0 if not steep_relief else 32.0,
            radiometric_weight=0.5,
            bathy_mode="auto",
        )

    @classmethod
    def auto(cls, report: CloudEvaluationReport) -> "OutlierParams":
        return cls._from_platform(report.estimated_platform, report=report, preset_name="auto")

    @classmethod
    def drone(cls, report: Optional[CloudEvaluationReport] = None) -> "OutlierParams":
        return cls._from_platform("drone", report=report, preset_name="drone")

    @classmethod
    def helicopter(cls, report: Optional[CloudEvaluationReport] = None) -> "OutlierParams":
        return cls._from_platform("helicopter", report=report, preset_name="helicopter")

    @classmethod
    def fixed_wing(cls, report: Optional[CloudEvaluationReport] = None) -> "OutlierParams":
        return cls._from_platform("fixed_wing", report=report, preset_name="fixed_wing")

    @classmethod
    def custom(cls, report: Optional[CloudEvaluationReport] = None, **overrides: Any) -> "OutlierParams":
        base = cls.auto(report) if report is not None else cls()
        base.preset_name = "custom"
        for k, v in overrides.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base


@dataclass
class OutlierResult:
    """Output of `classify_outliers_adaptive` (non-destructive)."""

    class_7_mask: np.ndarray
    class_18_mask: np.ndarray
    updated_classification: np.ndarray
    report: CloudEvaluationReport
    params: OutlierParams
    stats: Dict[str, Any] = field(default_factory=dict)


def _voxel_3d_neighbor_counts(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Compute 3x3x3 3D voxel neighborhood point count for every point in O(N)."""
    v = max(1e-3, float(voxel_size))
    ix = np.floor((xs - xs.min()) / v).astype(np.int64)
    iy = np.floor((ys - ys.min()) / v).astype(np.int64)
    iz = np.floor((zs - zs.min()) / v).astype(np.int64)

    # Pack 3D voxel coordinates into unique 64-bit keys (21 bits per axis)
    bias = 1 << 20
    kx = np.clip(ix, 0, (1 << 21) - 1)
    ky = np.clip(iy, 0, (1 << 21) - 1)
    kz = np.clip(iz, 0, (1 << 21) - 1)
    keys = (kx << 42) | (ky << 21) | kz

    uniq_keys, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    # Fast path: if a voxel itself already has >= 3 points, no need to check 26 neighbors
    nbr_counts = counts[inv].copy()

    # For sparse voxels (count <= 4), accumulate counts from the 26 adjacent voxels
    sparse_pts = np.flatnonzero(nbr_counts <= 4)
    if len(sparse_pts) > 0:
        skx = kx[sparse_pts]
        sky = ky[sparse_pts]
        skz = kz[sparse_pts]
        extra = np.zeros(len(sparse_pts), dtype=np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx = skx + dx
                    ny = sky + dy
                    nz = skz + dz
                    valid = (nx >= 0) & (ny >= 0) & (nz >= 0)
                    q_keys = np.where(valid, (nx << 42) | (ny << 21) | nz, -1)
                    pos = np.searchsorted(uniq_keys, q_keys)
                    pos_clipped = np.minimum(pos, len(uniq_keys) - 1)
                    hit = valid & (uniq_keys[pos_clipped] == q_keys)
                    if np.any(hit):
                        extra[hit] += counts[pos_clipped[hit]]
        nbr_counts[sparse_pts] += extra

    return nbr_counts


def _voxel_3d_connected_components_outliers(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    voxel_size: float,
    min_component_pts: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    O(N log N) 3D 26-connected voxel component filter.
    Returns (tiny_mask [< 12 pts, unconditional], small_mask [< min_component_pts]).
    """
    n = len(xs)
    if n == 0 or min_component_pts <= 1:
        z = np.zeros(n, dtype=bool)
        return z, z

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    vs = max(0.5, float(voxel_size))
    x0, y0, z0 = float(xs.min()), float(ys.min()), float(zs.min())
    kx = np.floor((xs - x0) / vs).astype(np.int64)
    ky = np.floor((ys - y0) / vs).astype(np.int64)
    kz = np.floor((zs - z0) / vs).astype(np.int64)

    max_coord = (1 << 20) - 1
    kx = np.clip(kx, 0, max_coord)
    ky = np.clip(ky, 0, max_coord)
    kz = np.clip(kz, 0, max_coord)
    keys = (kx << 42) | (ky << 21) | kz

    uniq_keys, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    nv = len(uniq_keys)
    ukx = uniq_keys >> 42
    uky = (uniq_keys >> 21) & ((1 << 21) - 1)
    ukz = uniq_keys & ((1 << 21) - 1)

    rows = []
    cols = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                nx, ny, nz = ukx + dx, uky + dy, ukz + dz
                valid = (nx >= 0) & (ny >= 0) & (nz >= 0)
                q = np.where(valid, (nx << 42) | (ny << 21) | nz, -1)
                pos = np.searchsorted(uniq_keys, q)
                pos_c = np.minimum(pos, nv - 1)
                hit = valid & (uniq_keys[pos_c] == q)
                if np.any(hit):
                    rows.append(np.flatnonzero(hit))
                    cols.append(pos_c[hit])

    if rows:
        r_all = np.concatenate(rows)
        c_all = np.concatenate(cols)
        adj = csr_matrix((np.ones(len(r_all), dtype=np.int8), (r_all, c_all)), shape=(nv, nv))
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(nv, dtype=np.int64)

    comp_pts = np.bincount(labels, weights=counts)
    tiny_vox = comp_pts[labels] < min(12, int(min_component_pts))
    small_vox = comp_pts[labels] < int(min_component_pts)
    return tiny_vox[inv], small_vox[inv]


def _asymmetric_irls_lower_surface(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    cell_size: float,
    iterations: int = 3,
) -> Tuple[np.ndarray, float]:
    """
    Coarse-to-fine 2D bilinear lower terrain envelope initialized on per-cell P10
    and refined with asymmetric Tukey/Huber IRLS.
    Returns (predicted_surface_z_per_point, robust_lower_mad).
    """
    n = len(xs)
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.5

    cs = max(0.25, float(cell_size))
    x0 = float(xs.min())
    y0 = float(ys.min())
    nx = max(2, int(math.ceil((float(xs.max()) - x0) / cs)) + 2)
    ny = max(2, int(math.ceil((float(ys.max()) - y0) / cs)) + 2)

    # Cap grid resolution for huge extents to stay O(N) and memory-safe
    if nx * ny > 4_000_000:
        scale = math.sqrt((nx * ny) / 4_000_000.0)
        cs *= scale
        nx = max(2, int(math.ceil((float(xs.max()) - x0) / cs)) + 2)
        ny = max(2, int(math.ceil((float(ys.max()) - y0) / cs)) + 2)

    fx = np.clip((xs - x0) / cs, 0.0, nx - 1.0001)
    fy = np.clip((ys - y0) / cs, 0.0, ny - 1.0001)
    ix = fx.astype(np.int64)
    iy = fy.astype(np.int64)
    cell_idx = ix * ny + iy

    # Initialize grid with per-cell P10 (robust against subterranean pits & high trees)
    order = np.argsort(cell_idx, kind="mergesort")
    s_cells = cell_idx[order]
    s_z = zs[order]
    uniq_mask = np.empty(len(s_cells), dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = s_cells[1:] != s_cells[:-1]
    starts = np.flatnonzero(uniq_mask)
    ends = np.append(starts[1:], len(s_cells))
    uniq_cells = s_cells[starts]

    # Coarse-to-fine level 1: coarse grid (~20m+) + 3x3 spatial median filter
    # prevents dense subterranean pit clusters (>10m below surface) from pulling
    # the fine lower envelope down into the pit.
    coarse_cs = max(20.0, cs * 8.0)
    cnx = max(2, int(math.ceil((float(xs.max()) - x0) / coarse_cs)) + 2)
    cny = max(2, int(math.ceil((float(ys.max()) - y0) / coarse_cs)) + 2)
    cix = np.clip(((xs - x0) / coarse_cs).astype(np.int64), 0, cnx - 1)
    ciy = np.clip(((ys - y0) / coarse_cs).astype(np.int64), 0, cny - 1)
    c_cell = cix * cny + ciy
    c_ord = np.argsort(c_cell, kind="mergesort")
    sc_cells = c_cell[c_ord]
    sc_z = zs[c_ord]
    cu_mask = np.empty(len(sc_cells), dtype=bool)
    cu_mask[0] = True
    cu_mask[1:] = sc_cells[1:] != sc_cells[:-1]
    c_starts = np.flatnonzero(cu_mask)
    c_ends = np.append(c_starts[1:], len(sc_cells))
    c_uniq = sc_cells[c_starts]
    c_p10 = np.empty(len(c_starts), dtype=np.float64)
    for ic in range(len(c_starts)):
        seg = np.sort(sc_z[c_starts[ic] : c_ends[ic]])
        k10 = max(0, min(len(seg) - 1, int(0.10 * (len(seg) - 1))))
        c_p10[ic] = seg[k10]
    c_grid = np.full(cnx * cny, np.nan, dtype=np.float64)
    c_grid[c_uniq] = c_p10
    c_grid = c_grid.reshape(cnx, cny)
    global_p10 = float(np.percentile(zs, 10.0))
    # Fill & 3x3 spatial median across coarse cells
    c_pad = np.pad(c_grid, 1, mode="edge")
    c_stack = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            c_stack.append(c_pad[1 + dx : 1 + dx + cnx, 1 + dy : 1 + dy + cny])
    c_stack_arr = np.stack(c_stack, axis=0)
    all_nan = np.all(np.isnan(c_stack_arr), axis=0)
    c_med = np.full((cnx, cny), global_p10, dtype=np.float64)
    if np.any(~all_nan):
        c_med[~all_nan] = np.nanmedian(c_stack_arr[:, ~all_nan], axis=0)

    # Project coarse envelope onto fine grid centers to reject subterranean pit seeds
    fine_cx = np.clip((((np.arange(nx) + 0.5) * cs) / coarse_cs).astype(np.int64), 0, cnx - 1)
    fine_cy = np.clip((((np.arange(ny) + 0.5) * cs) / coarse_cs).astype(np.int64), 0, cny - 1)
    coarse_on_fine = c_med[np.ix_(fine_cx, fine_cy)].ravel()

    grid_flat = np.full(nx * ny, np.nan, dtype=np.float64)
    # Vectorized P5/P15: initialize on P5, but reject seeds > 3.0m below coarse envelope
    cell_p5 = np.empty(len(starts), dtype=np.float64)
    cell_p15 = np.empty(len(starts), dtype=np.float64)
    for idx_c in range(len(starts)):
        a, b = starts[idx_c], ends[idx_c]
        seg = np.sort(s_z[a:b])
        k5 = max(0, min(len(seg) - 1, int(0.05 * (len(seg) - 1))))
        k15 = max(0, min(len(seg) - 1, int(0.15 * (len(seg) - 1))))
        c_floor = coarse_on_fine[uniq_cells[idx_c]] - 8.0
        cell_p5[idx_c] = max(seg[k5], c_floor)
        cell_p15[idx_c] = max(seg[k15], c_floor + 1.0)
    grid_flat[uniq_cells] = cell_p5
    grid = grid_flat.reshape(nx, ny)
    grid_cap = np.full(nx * ny, np.inf, dtype=np.float64)
    grid_cap[uniq_cells] = cell_p15
    grid_cap = grid_cap.reshape(nx, ny)

    # Fill empty cells via iterative 3x3 neighbor mean
    for _ in range(4):
        nan_mask = np.isnan(grid)
        if not np.any(nan_mask):
            break
        padded = np.pad(grid, 1, mode="edge")
        acc = np.zeros_like(grid)
        cnt = np.zeros_like(grid)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                sub = padded[1 + dx : 1 + dx + nx, 1 + dy : 1 + dy + ny]
                ok = ~np.isnan(sub)
                acc[ok] += sub[ok]
                cnt[ok] += 1.0
        fill_ok = nan_mask & (cnt > 0)
        grid[fill_ok] = acc[fill_ok] / cnt[fill_ok]
    grid[np.isnan(grid)] = global_p10

    # Bilinear interpolation helper
    wx = fx - ix
    wy = fy - iy
    ix1 = np.minimum(ix + 1, nx - 1)
    iy1 = np.minimum(iy + 1, ny - 1)

    def _interp(g: np.ndarray) -> np.ndarray:
        z00 = g[ix, iy]
        z10 = g[ix1, iy]
        z01 = g[ix, iy1]
        z11 = g[ix1, iy1]
        return (
            z00 * (1.0 - wx) * (1.0 - wy)
            + z10 * wx * (1.0 - wy)
            + z01 * (1.0 - wx) * wy
            + z11 * wx * wy
        )

    # Asymmetric Huber/Tukey IRLS refinement
    mad = 0.5
    z_pred = _interp(grid)
    for _ in range(max(1, int(iterations))):
        res = zs - z_pred
        # Robust scale from near-ground band
        near = res[(res > -5.0) & (res < 10.0)]
        if len(near) < 20:
            near = res
        med = float(np.median(near))
        mad = max(0.15, 1.4826 * float(np.median(np.abs(near - med))))

        # Asymmetric weights:
        #   - Points far below surface (res < -2.5*mad): Tukey 0 weight (pit clusters ignored)
        #   - Points on/near ground (-2.5*mad <= res <= 1.0*mad): Huber weight
        #   - Points above surface (res > 1.0*mad): 0 weight (buildings/canopy ignored)
        w = np.zeros(n, dtype=np.float64)
        in_band = (res >= -2.5 * mad) & (res <= 1.0 * mad)
        if np.any(in_band):
            r_b = res[in_band]
            w_b = np.where(np.abs(r_b) <= mad, 1.0, mad / np.maximum(np.abs(r_b), 1e-6))
            w_b = np.where(r_b < -mad, w_b * 0.35, w_b)
            w[in_band] = w_b

        w_sum = np.bincount(cell_idx, weights=w, minlength=nx * ny)
        wz_sum = np.bincount(cell_idx, weights=w * zs, minlength=nx * ny)
        upd = w_sum > 1e-6
        new_flat = grid.ravel().copy()
        new_flat[upd] = wz_sum[upd] / w_sum[upd]
        grid = np.minimum(new_flat.reshape(nx, ny), grid_cap)
        z_pred = _interp(grid)

    return z_pred, mad


def _detect_detached_high_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    z_surf: np.ndarray,
    cell_size: float = 4.0,
    min_gap_m: float = 8.0,
    min_height_m: float = 12.0,
    hard_max_height_m: float = 65.0,
) -> np.ndarray:
    """
    Detect points floating detached in mid-air above an empty vertical gap
    (e.g., atmospheric/cloud noise layers) without slicing continuous tall forest.
    """
    n = len(xs)
    detached = np.zeros(n, dtype=bool)
    if n == 0:
        return detached

    h = zs - z_surf
    # Unconditionally flag extreme heights far above any terrestrial structure
    detached |= (h > hard_max_height_m)

    cand_idx = np.flatnonzero((h > min_height_m) & (~detached))
    if len(cand_idx) == 0:
        return detached

    # Group all points by 2D column (cell_size x cell_size) to check vertical continuity
    cs = max(2.0, float(cell_size))
    x0 = float(xs.min())
    y0 = float(ys.min())
    ny = max(2, int(math.ceil((float(ys.max()) - y0) / cs)) + 2)
    cx = ((xs - x0) / cs).astype(np.int64)
    cy = ((ys - y0) / cs).astype(np.int64)
    col_id = cx * ny + cy

    cand_cols = np.unique(col_id[cand_idx])
    in_cand_col = np.isin(col_id, cand_cols)
    sub_idx = np.flatnonzero(in_cand_col)
    sub_col = col_id[sub_idx]
    sub_h = h[sub_idx]

    # Sort by (col_id, h) so each column's heights are contiguous & ascending
    order = np.lexsort((sub_h, sub_col))
    s_idx = sub_idx[order]
    s_col = sub_col[order]
    s_h = sub_h[order]

    uniq = np.empty(len(s_col), dtype=bool)
    uniq[0] = True
    uniq[1:] = s_col[1:] != s_col[:-1]
    starts = np.flatnonzero(uniq)
    ends = np.append(starts[1:], len(s_col))

    for ic in range(len(starts)):
        a, b = starts[ic], ends[ic]
        col_h = s_h[a:b]
        if col_h[-1] <= min_height_m:
            continue
        # Check gap from terrain (0.0) to first point, and between consecutive points
        if col_h[0] >= min_gap_m:
            detached[s_idx[a:b]] = True
            continue
        diffs = np.diff(col_h)
        gap_pos = np.flatnonzero(diffs >= min_gap_m)
        if len(gap_pos) > 0:
            # First gap >= min_gap_m above terrain detaches everything above it
            cut = int(gap_pos[0]) + 1
            if col_h[cut] > min_height_m:
                detached[s_idx[a + cut : b]] = True

    return detached


def classify_outliers_adaptive(
    data: Dict[str, Any],
    params: Optional[OutlierParams] = None,
    report: Optional[CloudEvaluationReport] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> OutlierResult:
    """
    Pure, non-destructive adaptive outlier classifier producing ASPRS Class 7
    (Low Noise) and Class 18 (High Noise) masks and updated classification array.
    """
    t_start = time.perf_counter()
    if report is None:
        report = evaluate_point_cloud(data)
    if params is None:
        params = OutlierParams.auto(report)

    xs = np.asarray(data["x"], dtype=np.float64)
    ys = np.asarray(data["y"], dtype=np.float64)
    zs = np.asarray(data["z"], dtype=np.float64)
    n = len(xs)

    cls_orig = np.asarray(data.get("classification", np.zeros(n, dtype=np.uint8)), dtype=np.uint8)
    cls_out = cls_orig.copy()
    class_7 = np.zeros(n, dtype=bool)
    class_18 = np.zeros(n, dtype=bool)

    if n == 0:
        return OutlierResult(class_7, class_18, cls_out, report, params, {"total_ms": 0.0})

    def _prog(msg: str, pct: float) -> None:
        if progress_cb:
            progress_cb(msg, pct)

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    refl = _get_extra_dim(data, "reflectance")
    dev = _get_extra_dim(data, "deviation")
    intensity = data.get("intensity")
    ret_num = np.asarray(data.get("return_number", np.ones(n, dtype=np.uint8)), dtype=np.uint8)
    num_ret = data.get("num_returns")
    if not (isinstance(num_ret, np.ndarray) and len(num_ret) == n):
        num_ret = np.asarray(data.get("number_of_returns", np.ones(n, dtype=np.uint8)), dtype=np.uint8)
    psid = np.asarray(data.get("point_source_id", np.zeros(n, dtype=np.uint16)), dtype=np.uint16)

    is_single_or_first = (num_ret <= 1) | (ret_num <= 1)
    canopy_protected = (~is_single_or_first) if params.protect_multi_return_canopy else np.zeros(n, dtype=bool)

    # ── Step 1: Return-Aware RIEGL Radiometric & Pulse-Shape Gate ───────
    _prog("Step 1/5: Radiometric & pulse-shape gate…", 10.0)
    radiometric_suspect = np.zeros(n, dtype=bool)
    if params.use_reflectance_gate and refl is not None and len(refl) == n:
        low_refl = refl < params.min_reflectance_db
        radiometric_suspect |= (low_refl & ~canopy_protected)
    if dev is not None and len(dev) == n and params.max_pulse_deviation > 0:
        high_dev = dev > params.max_pulse_deviation
        radiometric_suspect |= (high_dev & ~canopy_protected)
        # Severe pulse distortion + isolation is handled cleanly with Step 2/4

    if _cancelled():
        return OutlierResult(class_7, class_18, cls_out, report, params, {"cancelled": True})

    # ── Step 2: 3D Voxel Isolation + Connected-Component Pre-Filter (O(N)) ─
    _prog("Step 2/5: 3D voxel isolation & connected components…", 28.0)
    nbr_counts = _voxel_3d_neighbor_counts(xs, ys, zs, params.voxel_size_m)
    isolated_mask = nbr_counts < max(1, int(params.min_voxel_neighbors))
    # Only flag radiometric suspects if truly isolated (1 point in 3x3x3 voxel)
    # so sparse bathymetry / shadowed ground beside houses is never shredded.
    isolated_mask |= (radiometric_suspect & (nbr_counts <= 1))

    # 3D Voxel Connected Components catches disconnected over/under blobs (1–50 pts)
    # at any height (-30m, -3m, +6m, +80m) without touching connected forest or ground.
    cc_voxel = max(1.1, min(1.8, params.voxel_size_m * 1.25))
    cc_tiny_mask, cc_small_mask = _voxel_3d_connected_components_outliers(
        xs, ys, zs, voxel_size=cc_voxel, min_component_pts=50
    )
    isolated_mask |= cc_small_mask

    z_med_global = float(np.median(zs))
    class_7 |= (isolated_mask & (zs < z_med_global))
    # Tiny disconnected blobs (<12 pts) above z_med are always noise; larger ones (12..49 pts)
    # are deferred to Step 4 so linear powerline corner spans (is_linear) are preserved.
    class_18 |= (cc_tiny_mask & (zs >= z_med_global))
    step2_iso_high = isolated_mask & (~cc_tiny_mask) & (zs >= z_med_global)

    if _cancelled():
        return OutlierResult(class_7, class_18, cls_out, report, params, {"cancelled": True})

    # ── Step 3: Flightline-Aware Asymmetric IRLS Lower & Detached-High Envelope ─
    _prog("Step 3/5: Asymmetric IRLS terrain & strip envelope…", 48.0)
    alive_idx = np.flatnonzero(~(class_7 | class_18))
    step3_detached_high = np.zeros(n, dtype=bool)
    if len(alive_idx) > 20:
        z_surf, mad_lower = _asymmetric_irls_lower_surface(
            xs[alive_idx],
            ys[alive_idx],
            zs[alive_idx],
            cell_size=params.grid_cell_size_m,
            iterations=params.irls_iterations,
        )
        res_alive = zs[alive_idx] - z_surf
        # Safe minimum floor (-1.5 m) prevents ground/curbs/ditches from being flagged as Class 7.
        # Points in the moderate band [-6.0m, low_thresh) are only flagged if locally sparse
        # (protects steep slopes, road cuts, and embankments with dense ground neighbors).
        low_thresh = min(-1.5, -abs(params.z_lower_mad_multiplier) * mad_lower)
        hard_low_thresh = min(-6.0, low_thresh * 2.5)
        low_hits = (res_alive < hard_low_thresh) | (
            (res_alive < low_thresh) & (nbr_counts[alive_idx] <= 4)
        )
        class_7[alive_idx[low_hits]] = True

        # Detached high-noise detection (replaces blind p95_r slicing that clipped tall
        # forest/powerlines on flat tiles and missed >5% high-noise clouds).
        # Deferred to Step 4 so linear powerlines (is_linear) are protected.
        still_alive = alive_idx[~low_hits]
        if len(still_alive) > 20:
            hard_max_h = max(75.0, params.z_upper_mad_multiplier * 12.0)
            res_still = zs[still_alive] - z_surf[~low_hits]
            # Unconditionally flag extreme sky flyers > 75m above terrain (even linear scan-line streaks)
            extreme_sky = res_still > hard_max_h
            class_18[still_alive[extreme_sky]] = True

            det_mask = _detect_detached_high_mask(
                xs[still_alive],
                ys[still_alive],
                zs[still_alive],
                z_surf[~low_hits],
                cell_size=max(params.grid_cell_size_m * 3.0, 4.0),
                min_gap_m=max(6.0, params.z_upper_mad_multiplier * 1.25),
                min_height_m=max(10.0, params.z_upper_mad_multiplier * 1.6),
                hard_max_height_m=hard_max_h,
            )
            step3_detached_high[still_alive[det_mask & (~extreme_sky)]] = True

    if _cancelled():
        return OutlierResult(class_7, class_18, cls_out, report, params, {"cancelled": True})

    # ── Step 4: Single cKDTree Multi-Scale Consistency + Radiometric SOR ─
    _prog("Step 4/5: Multi-scale consistency & radiometric SOR…", 72.0)
    surv_idx = np.flatnonzero(~(class_7 | class_18))
    k_large = max(6, int(params.sor_k_large))
    k_small = max(4, min(k_large, int(params.sor_k_small)))

    if len(surv_idx) > k_large + 1:
        pts_surv = np.column_stack((xs[surv_idx], ys[surv_idx], zs[surv_idx]))
        tree = cKDTree(pts_surv)
        dists, nbrs = tree.query(pts_surv, k=k_large + 1, workers=-1)
        # Drop self (index 0)
        d_large = dists[:, 1:]
        n_large = nbrs[:, 1:]
        d_small = d_large[:, :k_small]
        n_small = n_large[:, :k_small]

        mean_d_large = d_large.mean(axis=1)
        mean_d_small = d_small.mean(axis=1)

        # 4a. Return-aware radiometric weighting on k-NN distance
        rad_factor = np.ones(len(surv_idx), dtype=np.float64)
        if params.radiometric_weight > 0:
            surv_protected = canopy_protected[surv_idx]
            if refl is not None and len(refl) == n:
                r_s = refl[surv_idx]
                penalty = np.clip((-15.0 - r_s) / 15.0, 0.0, 1.0)
                rad_factor += np.where(surv_protected, 0.0, params.radiometric_weight * penalty)
            elif isinstance(intensity, np.ndarray) and len(intensity) == n:
                i_s = intensity[surv_idx].astype(np.float64)
                p10_i = max(1.0, float(np.percentile(i_s, 10.0)))
                penalty = np.clip((p10_i - i_s) / p10_i, 0.0, 1.0)
                rad_factor += np.where(surv_protected, 0.0, params.radiometric_weight * penalty)

        weighted_d = mean_d_large * rad_factor
        med_wd = float(np.median(weighted_d))
        upper_diff = weighted_d[weighted_d >= med_wd] - med_wd
        mad_wd = max(1e-4, 1.4826 * float(np.median(upper_diff)) if len(upper_diff) > 0 else 1e-4)
        # Physical spacing floor protects sparse bathymetry & shadowed ground beside houses
        min_sparse_floor = max(1.2, 6.0 * max(0.05, float(report.nps_m)))
        sor_thresh = max(med_wd + (params.sor_std_multiplier * 2.0) * mad_wd, 3.0 * med_wd, min_sparse_floor)
        sor_reject = (weighted_d > sor_thresh) & (mean_d_small > min_sparse_floor * 0.75)

        # 4b. Multi-scale PCA plane residual & linearity/normal consistency
        # Compute covariance for k_small and k_large on all survivors in vectorized batches
        def _pca_stats(nbr_idx_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # shape: (M, k, 3)
            M = len(nbr_idx_arr)
            lam1 = np.empty(M, dtype=np.float64)
            lam2 = np.empty(M, dtype=np.float64)
            lam3 = np.empty(M, dtype=np.float64)
            normals = np.empty((M, 3), dtype=np.float64)
            batch = 50_000
            for s0 in range(0, M, batch):
                s1 = min(M, s0 + batch)
                nb = pts_surv[nbr_idx_arr[s0:s1]]  # (B, k, 3)
                centered = nb - nb.mean(axis=1, keepdims=True)
                cov = np.einsum("bki,bkj->bij", centered, centered) / float(nb.shape[1])
                evals, evecs = np.linalg.eigh(cov)  # ascending: 0=smallest, 2=largest
                lam3[s0:s1] = np.maximum(evals[:, 0], 0.0)
                lam2[s0:s1] = np.maximum(evals[:, 1], 0.0)
                lam1[s0:s1] = np.maximum(evals[:, 2], 1e-12)
                normals[s0:s1] = evecs[:, :, 0]
            return lam1, lam2, np.sqrt(lam3), normals

        l1_s, l2_s, res_s, norm_s = _pca_stats(n_small)
        l1_l, l2_l, res_l, norm_l = _pca_stats(n_large)

        # Linearity preserves powerlines & poles (high (l1 - l2)/l1)
        linearity_s = (l1_s - l2_s) / np.maximum(l1_s, 1e-12)
        linearity_l = (l1_l - l2_l) / np.maximum(l1_l, 1e-12)
        is_linear = (linearity_s > 0.70) | (linearity_l > 0.70)

        # Planar surfaces have small plane residual sqrt(lambda_3)
        is_planar = (res_s <= params.max_plane_residual_m) | (res_l <= params.max_plane_residual_m)

        # Multi-scale normal angular jump
        dot = np.clip(np.abs(np.sum(norm_s * norm_l, axis=1)), 0.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        unstable_normal = angle_deg > params.max_normal_angle_deg

        # Protect multi-return forest canopy, vertical walls/fences/nets, linear features,
        # and wall-ground corners (which are dense, mean_d_small <= min_sparse_floor).
        surv_canopy = canopy_protected[surv_idx]
        is_vertical_or_linear = (is_planar & (np.abs(norm_l[:, 2]) < 0.55)) | is_linear
        sparse_for_pca = (weighted_d > (med_wd + 2.5 * mad_wd)) & (mean_d_small > min_sparse_floor * 0.85)
        pca_reject = (~is_linear) & (~is_planar) & unstable_normal & (~surv_canopy) & sparse_for_pca & (res_s > params.max_plane_residual_m)

        # Extreme SOR catches truly isolated outliers even on canopy/vertical structures
        # (powerlines/wires remain protected via ~is_linear)
        extreme_sor = (weighted_d > (sor_thresh * 1.65)) & (mean_d_small > min_sparse_floor) & (~is_linear)
        deferred_high = (step2_iso_high[surv_idx] | step3_detached_high[surv_idx]) & (~is_linear)
        final_reject = (sor_reject & (~surv_canopy) & (~is_vertical_or_linear)) | extreme_sor | pca_reject | deferred_high

        rej_pts = surv_idx[final_reject]
        if len(rej_pts) > 0:
            # Only assign Class 7 if clearly below local neighbor median Z; otherwise Class 18
            local_med_z = np.median(zs[surv_idx[n_large[final_reject]]], axis=1)
            below_local = zs[rej_pts] < (local_med_z - 0.5)
            class_7[rej_pts[below_local]] = True
            class_18[rej_pts[~below_local]] = True

    # ── Step 5: Conditional Bathymetry Sub-Pass ────────────────────────
    run_bathy = (params.bathy_mode == "on") or (params.bathy_mode == "auto" and report.has_bathymetry)
    bathy_flagged = 0
    if run_bathy and not _cancelled():
        _prog("Step 5/5: Bathymetry water-column & spray sub-pass…", 90.0)
        b_surv = np.flatnonzero(~(class_7 | class_18))
        water_mask = np.zeros(n, dtype=bool)
        if isinstance(cls_orig, np.ndarray) and len(cls_orig) == n:
            water_mask |= (cls_orig == ASPRS_WATER) | (cls_orig == ASPRS_BATHY_WATER_SURFACE)
        if refl is not None and len(refl) == n:
            water_mask |= (refl < -18.0) & is_single_or_first

        water_pts = b_surv[water_mask[b_surv]]
        if len(water_pts) >= 20:
            water_z = float(np.median(zs[water_pts]))
            # Flag sparse above-water spray & sparse water-column backscatter
            b_cand = b_surv[(zs[b_surv] > water_z + 0.35) & (zs[b_surv] < water_z + 3.0) & water_mask[b_surv]]
            if len(b_cand) > 0:
                b_counts = _voxel_3d_neighbor_counts(xs[b_cand], ys[b_cand], zs[b_cand], max(0.3, params.voxel_size_m))
                spray = b_cand[b_counts <= 3]
                class_18[spray] = True
                bathy_flagged += len(spray)

            sub_cand = b_surv[(zs[b_surv] < water_z - 0.35)]
            if len(sub_cand) > 10:
                sub_counts = _voxel_3d_neighbor_counts(xs[sub_cand], ys[sub_cand], zs[sub_cand], max(0.4, params.voxel_size_m * 1.2))
                backscatter = sub_cand[sub_counts <= 2]
                class_7[backscatter] = True
                bathy_flagged += len(backscatter)

    # Ensure disjoint masks (Class 18 takes precedence if somehow both)
    class_7 &= ~class_18
    cls_out[class_7] = ASPRS_LOW_NOISE
    cls_out[class_18] = ASPRS_HIGH_NOISE

    total_ms = (time.perf_counter() - t_start) * 1000.0
    _prog("Completed adaptive outlier classification.", 100.0)

    stats = {
        "n_total": n,
        "n_class_7": int(class_7.sum()),
        "n_class_18": int(class_18.sum()),
        "n_bathy_flagged": int(bathy_flagged),
        "total_ms": round(total_ms, 2),
        "eval_ms": report.eval_time_ms,
    }
    return OutlierResult(
        class_7_mask=class_7,
        class_18_mask=class_18,
        updated_classification=cls_out,
        report=report,
        params=params,
        stats=stats,
    )


# ── Phase 3: Decoupled SQLite Undo/Redo Integration ────────────────────

def apply_outlier_classification_with_undo(
    db: Any,
    tile_id: str,
    old_classes: np.ndarray,
    result: OutlierResult,
    description: str = "Adaptive Outlier Classification (Class 7 & 18)",
) -> Optional[int]:
    """
    Record a single reversible batch command in SQLite `edit_history`
    storing only the changed point indices and their previous ASPRS classes.
    """
    changed_mask = result.class_7_mask | result.class_18_mask
    changed_idx = np.flatnonzero(changed_mask)
    if len(changed_idx) == 0:
        return None

    cmd = {
        "type": "batch_outlier_classify",
        "description": description,
        "tile_id": tile_id,
        "preset": result.params.preset_name,
        "indices": changed_idx.tolist(),
        "old_classes": np.asarray(old_classes[changed_idx], dtype=np.uint8).tolist(),
        "new_classes": np.asarray(result.updated_classification[changed_idx], dtype=np.uint8).tolist(),
        "undone": False,
    }
    with db.connect() as conn:
        cmd_id = db.add_edit_command(conn, tile_id, cmd)
        conn.commit()
    return cmd_id


def undo_outlier_batch(db: Any, tile_id: str, cls_array: np.ndarray, command_id: Optional[int] = None) -> Tuple[np.ndarray, bool]:
    """Revert a batch outlier command from SQLite `edit_history` on `cls_array`."""
    rows = db.get_edit_history(tile_id, limit=50)
    target_row = None
    for r in rows:
        if command_id is not None and r["id"] != command_id:
            continue
        cmd = json.loads(r["command"])
        if cmd.get("type") == "batch_outlier_classify" and not cmd.get("undone", False):
            target_row = (r["id"], cmd)
            break
    if target_row is None:
        return cls_array, False

    row_id, cmd = target_row
    out = cls_array.copy()
    idx = np.asarray(cmd["indices"], dtype=np.int64)
    old_c = np.asarray(cmd["old_classes"], dtype=np.uint8)
    out[idx] = old_c
    cmd["undone"] = True
    with db.connect() as conn:
        conn.execute("UPDATE edit_history SET command = ? WHERE id = ?", (json.dumps(cmd), row_id))
        conn.commit()
    return out, True


def redo_outlier_batch(db: Any, tile_id: str, cls_array: np.ndarray, command_id: Optional[int] = None) -> Tuple[np.ndarray, bool]:
    """Re-apply a previously undone batch outlier command from SQLite `edit_history`."""
    rows = db.get_edit_history(tile_id, limit=50)
    target_row = None
    for r in rows:
        if command_id is not None and r["id"] != command_id:
            continue
        cmd = json.loads(r["command"])
        if cmd.get("type") == "batch_outlier_classify" and cmd.get("undone", False):
            target_row = (r["id"], cmd)
            break
    if target_row is None:
        return cls_array, False

    row_id, cmd = target_row
    out = cls_array.copy()
    idx = np.asarray(cmd["indices"], dtype=np.int64)
    new_c = np.asarray(cmd["new_classes"], dtype=np.uint8)
    out[idx] = new_c
    cmd["undone"] = False
    with db.connect() as conn:
        conn.execute("UPDATE edit_history SET command = ? WHERE id = ?", (json.dumps(cmd), row_id))
        conn.commit()
    return out, True
