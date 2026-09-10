"""
LiDAR Workbench — River Centerline Water Surface Model (WSM) Engine.

Provides algorithms for:
  - Parametrization of river centerlines (arc length, tangents, orthogonal normals)
  - Transverse cross-section point slicing along the centerline
  - Automated shoreline void (NIR cutoff) and water surface detection
  - Monotonic downhill elevation enforcement with user anchor preservation
  - Rasterization of 3D horizontal water surface models into GeoTIFF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None

logger = logging.getLogger("lidar_workbench.centerline_wsm")


class RiverCenterline:
    """
    Parametrized 2D river centerline polyline.

    Allows evaluation of position P(s), unit tangent T(s), and unit normal
    N(s) at any station distance s along the river channel.
    """

    def __init__(self, vertices: np.ndarray):
        """
        Args:
            vertices: (K, 2) array of [X, Y] coordinates along the river.
        """
        verts = np.asarray(vertices, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] < 2:
            raise ValueError("Vertices must be a 2D array with at least 2 coordinate columns [X, Y].")
        # Elevation (Z) is safely discarded — centerline uses purely 2D horizontal alignment
        verts = verts[:, :2]
        if len(verts) < 2:
            raise ValueError("Centerline requires at least 2 vertices.")

        # Remove duplicate consecutive points
        dists = np.hypot(np.diff(verts[:, 0]), np.diff(verts[:, 1]))
        keep = np.ones(len(verts), dtype=bool)
        keep[1:] = dists > 1e-6
        self._verts = verts[keep]
        if len(self._verts) < 2:
            raise ValueError("Centerline has zero length after removing duplicate vertices.")

        # Compute cumulative distance along vertices
        diffs = np.diff(self._verts, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self._cum_dist = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        self._total_length = float(self._cum_dist[-1])

    @property
    def total_length(self) -> float:
        """Total length of the centerline in coordinate units (metres)."""
        return self._total_length

    @property
    def vertices(self) -> np.ndarray:
        return self._verts

    def sample_stations(self, spacing: float) -> np.ndarray:
        """Return station distances s sampled uniformly every ``spacing`` metres."""
        spacing = max(0.1, float(spacing))
        n_samples = max(2, int(np.ceil(self._total_length / spacing)) + 1)
        stations = np.linspace(0.0, self._total_length, n_samples)
        return stations

    def evaluate(self, s: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate position, unit tangent, and unit normal at station distance(s) s.

        Normal points to the left of the flow direction: N = (-T_y, T_x).

        Returns:
            (pos, tangent, normal) where:
                pos: (N, 2) or (2,) [X, Y]
                tangent: (N, 2) or (2,) unit tangent vector
                normal: (N, 2) or (2,) unit normal vector
        """
        scalar = np.isscalar(s)
        s_arr = np.atleast_1d(np.clip(s, 0.0, self._total_length))

        # Find segment indices
        idx = np.searchsorted(self._cum_dist, s_arr, side="right") - 1
        idx = np.clip(idx, 0, len(self._verts) - 2)

        s0 = self._cum_dist[idx]
        s1 = self._cum_dist[idx + 1]
        ds = s1 - s0
        frac = np.divide(s_arr - s0, ds, out=np.zeros_like(s_arr), where=ds > 1e-12)

        # Interpolate position
        p0 = self._verts[idx]
        p1 = self._verts[idx + 1]
        pos = p0 + frac[:, None] * (p1 - p0)

        # Unit tangent vector
        diff = p1 - p0
        seg_len = np.hypot(diff[:, 0], diff[:, 1])
        t_x = np.divide(diff[:, 0], seg_len, out=np.zeros_like(seg_len), where=seg_len > 1e-12)
        t_y = np.divide(diff[:, 1], seg_len, out=np.zeros_like(seg_len), where=seg_len > 1e-12)
        tangent = np.column_stack((t_x, t_y))

        # Unit normal vector: pointing left of flow
        normal = np.column_stack((-t_y, t_x))

        if scalar:
            return pos[0], tangent[0], normal[0]
        return pos, tangent, normal

    def project_point_to_station(self, x: float, y: float) -> float:
        """
        Project any 2D point (x, y) onto the centerline polyline and return the station distance s.
        """
        pt = np.array([float(x), float(y)], dtype=np.float64)
        n_segs = len(self._verts) - 1
        best_dist_sq = float("inf")
        best_s = 0.0

        for i in range(n_segs):
            p0 = self._verts[i]
            p1 = self._verts[i + 1]
            seg_vec = p1 - p0
            seg_len = np.hypot(seg_vec[0], seg_vec[1])
            if seg_len < 1e-9:
                continue

            u = seg_vec / seg_len
            w = pt - p0
            t = float(np.dot(w, u))
            t_clamped = max(0.0, min(seg_len, t))
            proj = p0 + t_clamped * u
            d_sq = float(np.sum((pt - proj) ** 2))

            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_s = float(self._cum_dist[i] + t_clamped)

        return best_s

    @property
    def cum_dist(self) -> np.ndarray:
        """Cumulative distance array along the vertices."""
        return self._cum_dist

    def trim(self, s_start: float, s_end: float) -> "RiverCenterline":
        """
        Trim the centerline to the station interval [s_start, s_end].
        Returns a new RiverCenterline object whose station 0.0 corresponds to s_start.
        """
        s_a = max(0.0, min(float(s_start), float(s_end)))
        s_b = min(self._total_length, max(float(s_start), float(s_end)))
        if s_b - s_a < 1.0:
            raise ValueError(f"Trim range [{s_a:.1f}, {s_b:.1f}] is too short (must be >= 1.0 m).")

        p_start, _, _ = self.evaluate(s_a)
        p_end, _, _ = self.evaluate(s_b)

        # Vertices strictly between s_a and s_b
        mask = (self._cum_dist > s_a + 1e-4) & (self._cum_dist < s_b - 1e-4)
        mid_verts = self._verts[mask]

        new_verts = [p_start]
        if len(mid_verts) > 0:
            new_verts.extend(mid_verts)
        new_verts.append(p_end)

        return RiverCenterline(np.array(new_verts))


def project_points_to_centerline(
    centerline: RiverCenterline,
    xs: np.ndarray,
    ys: np.ndarray,
    chunk_size: int = 25000,
) -> np.ndarray:
    """
    Project an array of 2D points (xs, ys) onto the centerline polyline,
    returning station distance s along the centerline for each point.
    Vectorized over points and polyline segments.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n_pts = len(xs)
    if n_pts == 0:
        return np.array([], dtype=np.float64)

    verts = centerline.vertices
    cum_dist = centerline.cum_dist

    p0 = verts[:-1]  # (S, 2)
    p1 = verts[1:]   # (S, 2)
    seg_vec = p1 - p0
    seg_len_sq = np.sum(seg_vec ** 2, axis=1)  # (S,)
    seg_len = np.sqrt(seg_len_sq)

    valid_segs = seg_len > 1e-9
    p0 = p0[valid_segs]
    seg_vec = seg_vec[valid_segs]
    seg_len_sq = seg_len_sq[valid_segs]
    cum_dist_valid = cum_dist[:-1][valid_segs]

    out_s = np.empty(n_pts, dtype=np.float64)

    # Process in chunks to prevent memory spikes if n_pts is large
    for start_idx in range(0, n_pts, chunk_size):
        end_idx = min(start_idx + chunk_size, n_pts)
        pts_chunk = np.column_stack((xs[start_idx:end_idx], ys[start_idx:end_idx]))  # (C, 2)

        # w: (C, S, 2)
        w = pts_chunk[:, None, :] - p0[None, :, :]
        # dot product w * seg_vec: (C, S)
        t = np.sum(w * seg_vec[None, :, :], axis=2) / seg_len_sq[None, :]
        t_clamped = np.clip(t, 0.0, 1.0)

        # proj: p0 + t_clamped * seg_vec -> (C, S, 2)
        proj = p0[None, :, :] + t_clamped[:, :, None] * seg_vec[None, :, :]
        # dist_sq: (C, S)
        d_sq = np.sum((pts_chunk[:, None, :] - proj) ** 2, axis=2)

        # Best segment for each point in chunk
        best_seg = np.argmin(d_sq, axis=1)  # (C,)
        best_t = t_clamped[np.arange(len(pts_chunk)), best_seg]  # (C,)
        best_s = cum_dist_valid[best_seg] + best_t * np.sqrt(seg_len_sq[best_seg])
        out_s[start_idx:end_idx] = best_s

    return out_s


def slice_cross_section_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    center_pos: Tuple[float, float],
    normal: Tuple[float, float],
    tangent: Tuple[float, float],
    corridor_width: float,
    slice_thickness: float = 2.0,
    sensor_types: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    sorted_x: bool = False,
) -> dict:
    """
    Slice 3D points in a transverse corridor perpendicular to the river centerline.

    Args:
        xs, ys, zs: Point cloud coordinates.
        center_pos: [X, Y] position on the centerline.
        normal: (Nx, Ny) unit vector pointing across the river.
        tangent: (Tx, Ty) unit vector pointing downstream.
        corridor_width: Total cross-section width across the river (metres).
        slice_thickness: Corridor thickness along the stream (metres, default 2.0).
        sensor_types: Optional array of sensor types (1=NIR/topo, 2=Green/bathy).
        classes: Optional array of point classifications.
        sorted_x: Whether xs is sorted in ascending order for O(log N) binary search.

    Returns:
        Dictionary with sliced points in cross-section local coordinates:
            - "offset": Transverse distance across channel [-W/2, +W/2]
            - "z": Elevation
            - "stream_dist": Distance along flow from the slice plane [-thick/2, +thick/2]
            - "sensor_type": Subsetted sensor types (or None)
            - "classification": Subsetted classes (or None)
            - "indices": Original point cloud indices
    """
    cx, cy = center_pos
    nx, ny = normal
    tx, ty = tangent

    max_r = corridor_width * 0.5 + slice_thickness

    if sorted_x:
        i0 = int(np.searchsorted(xs, cx - max_r, side="left"))
        i1 = int(np.searchsorted(xs, cx + max_r, side="right"))
        if i0 >= i1:
            return {
                "offset": np.array([], dtype=np.float64),
                "z": np.array([], dtype=np.float64),
                "stream_dist": np.array([], dtype=np.float64),
                "sensor_type": None,
                "classification": None,
                "indices": np.array([], dtype=np.int64),
            }
        dx_cand = xs[i0:i1] - cx
        dy_cand = ys[i0:i1] - cy
        sub_mask = np.abs(dy_cand) <= max_r
        if not sub_mask.any():
            return {
                "offset": np.array([], dtype=np.float64),
                "z": np.array([], dtype=np.float64),
                "stream_dist": np.array([], dtype=np.float64),
                "sensor_type": None,
                "classification": None,
                "indices": np.array([], dtype=np.int64),
            }
        rel_idx = np.flatnonzero(sub_mask)
        sub_idx = i0 + rel_idx
        dx_sub = dx_cand[rel_idx]
        dy_sub = dy_cand[rel_idx]
        z_sub = zs[sub_idx]
    else:
        # Vector from center to points
        dx = xs - cx
        dy = ys - cy

        # Fast bounding box pre-filter
        bbox_mask = (np.abs(dx) <= max_r) & (np.abs(dy) <= max_r)
        if not bbox_mask.any():
            return {
                "offset": np.array([], dtype=np.float64),
                "z": np.array([], dtype=np.float64),
                "stream_dist": np.array([], dtype=np.float64),
                "sensor_type": None,
                "classification": None,
                "indices": np.array([], dtype=np.int64),
            }

        sub_idx = np.flatnonzero(bbox_mask)
        dx_sub = dx[sub_idx]
        dy_sub = dy[sub_idx]
        z_sub = zs[sub_idx]

    # Transverse offset across stream (along normal)
    offset = dx_sub * nx + dy_sub * ny
    # Longitudinal distance along stream (along tangent)
    stream_dist = dx_sub * tx + dy_sub * ty

    half_w = corridor_width * 0.5
    half_t = slice_thickness * 0.5

    in_slice = (np.abs(offset) <= half_w) & (np.abs(stream_dist) <= half_t)
    final_sub = in_slice

    res_idx = sub_idx[final_sub]
    res = {
        "offset": offset[final_sub],
        "z": z_sub[final_sub],
        "stream_dist": stream_dist[final_sub],
        "sensor_type": sensor_types[res_idx] if sensor_types is not None else None,
        "classification": classes[res_idx] if classes is not None else None,
        "indices": res_idx,
    }
    return res


def detect_section_water_level(
    offsets: np.ndarray,
    elevations: np.ndarray,
    sensor_types: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    channel_search_width: Optional[float] = None,
    prior_z: Optional[float] = None,
    z_window: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Automatically detect the horizontal water surface level at a cross-section.

    Combines:
      1. NIR Shoreline Cutoff: Find where NIR/topo returns stop on left & right banks.
      2. Green Water Surface Scatter: Detects the upper envelope of returns in channel.
      3. Prior Guidance: If prior_z is provided (e.g. from neighboring user anchors),
         focuses on the plausible vertical window [prior_z - z_window, prior_z + z_window]
         to accurately follow flat reservoir pools, weirs, or power plants.

    Returns:
        (water_z, left_bank_z, right_bank_z)
    """
    n_pts = len(offsets)
    if n_pts < 5:
        return (prior_z if prior_z is not None else np.nan), np.nan, np.nan

    # If prior_z is provided, filter out distant noise (e.g. high trees or deep trenches)
    if prior_z is not None:
        win_mask = np.abs(elevations - prior_z) <= z_window
        if win_mask.sum() >= 5:
            offsets = offsets[win_mask]
            elevations = elevations[win_mask]
            if sensor_types is not None:
                sensor_types = sensor_types[win_mask]
            if classes is not None:
                classes = classes[win_mask]
            n_pts = len(offsets)

    # Identify NIR/topo vs Green/bathy points
    is_nir = np.ones(n_pts, dtype=bool)
    is_bathy = np.zeros(n_pts, dtype=bool)
    if sensor_types is not None:
        is_bathy = (sensor_types == 2)
        is_nir = (sensor_types != 2)

    # Sort points from left bank (negative offset) to right bank (positive offset)
    order = np.argsort(offsets)
    off_s = offsets[order]
    z_s = elevations[order]
    nir_s = is_nir[order]
    bathy_s = is_bathy[order]

    # Method 1: Dual-Bank NIR Shoreline Cutoff
    left_mask = (off_s < 0.0) & nir_s
    right_mask = (off_s > 0.0) & nir_s

    z_left = np.nan
    z_right = np.nan

    if left_mask.sum() >= 3:
        l_off = off_s[left_mask]
        l_z = z_s[left_mask]
        edge_thr = np.percentile(l_off, 70.0)  # closest to center (offset near 0)
        edge_pts = l_z[l_off >= edge_thr]
        if len(edge_pts):
            z_left = float(np.percentile(edge_pts, 15.0))  # lowest bank edge

    if right_mask.sum() >= 3:
        r_off = off_s[right_mask]
        r_z = z_s[right_mask]
        edge_thr = np.percentile(r_off, 30.0)  # closest to center
        edge_pts = r_z[r_off <= edge_thr]
        if len(edge_pts):
            z_right = float(np.percentile(edge_pts, 15.0))  # lowest bank edge

    # Method 2: Channel Core NIR / Topo surface returns (NIR laser reflects on air-water interface)
    chan_nir_mask = (np.abs(off_s) <= 3.0) & nir_s
    z_chan_nir = np.nan
    if chan_nir_mask.sum() >= 2:
        chan_pts = z_s[chan_nir_mask]
        z_chan_nir = float(np.percentile(chan_pts, 75.0)) if len(chan_pts) >= 4 else float(np.median(chan_pts))

    # Method 3: Bathymetric Green Surface Echo / Channel Upper Envelope (fallback if no topo)
    z_bathy_surf = np.nan
    if bathy_s.sum() >= 5:
        b_z = z_s[bathy_s]
        z_bathy_surf = float(np.percentile(b_z, 92.0))

    # Reconcile water levels:
    # 1. Prioritize Topo LiDAR returns in the channel core (direct physical water surface returns)
    if not np.isnan(z_chan_nir):
        water_z = z_chan_nir
    # 2. Reconcile bank water levels if available
    elif not np.isnan(z_left) and not np.isnan(z_right):
        if abs(z_left - z_right) <= 0.80:
            water_z = 0.5 * (z_left + z_right)
        else:
            water_z = min(z_left, z_right)
    elif not np.isnan(z_left):
        water_z = z_left
    elif not np.isnan(z_right):
        water_z = z_right
    # 3. Fall back to top of bathymetric returns if no topo returns available
    elif not np.isnan(z_bathy_surf):
        water_z = z_bathy_surf
    elif prior_z is not None:
        water_z = prior_z
    else:
        water_z = float(np.median(elevations))

    # Sanity check against bathymetric returns: water level cannot be deeper than bathy returns
    if not np.isnan(z_bathy_surf) and water_z < (z_bathy_surf - 0.15):
        water_z = z_bathy_surf

    return water_z, z_left, z_right


def enforce_downstream_monotonicity(
    water_levels: np.ndarray,
    locked_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Enforce downstream monotonicity so water elevation never increases downhill.

    Preserves all user-approved / locked anchor elevations exactly, smoothly
    relaxing intermediate auto-estimated sections.
    """
    levels = np.array(water_levels, dtype=np.float64)
    n = len(levels)
    if n <= 1:
        return levels

    if locked_mask is None:
        locked_mask = np.zeros(n, dtype=bool)

    # Replace any NaNs via linear interpolation
    nans = np.isnan(levels)
    if nans.all():
        return np.zeros(n, dtype=np.float64)
    if nans.any():
        valid_idx = np.flatnonzero(~nans)
        levels[nans] = np.interp(np.flatnonzero(nans), valid_idx, levels[valid_idx])

    # Clamp intermediate levels between adjacent locked anchors
    locked_idx = np.flatnonzero(locked_mask)
    if len(locked_idx) >= 2:
        for k in range(len(locked_idx) - 1):
            i0, i1 = locked_idx[k], locked_idx[k + 1]
            z_up = levels[i0]
            z_down = levels[i1]
            levels[i0 : i1 + 1] = np.clip(levels[i0 : i1 + 1], z_down, z_up)

        # Before first anchor: non-increasing backwards
        if locked_idx[0] > 0:
            levels[:locked_idx[0]] = np.maximum(levels[:locked_idx[0]], levels[locked_idx[0]])
        # After last anchor: non-increasing forwards
        if locked_idx[-1] < n - 1:
            levels[locked_idx[-1] + 1 :] = np.minimum(levels[locked_idx[-1] + 1 :], levels[locked_idx[-1]])

    # Standard downstream cumulative minimum to guarantee non-increasing
    for i in range(1, n):
        if not locked_mask[i]:
            if levels[i] > levels[i - 1]:
                levels[i] = levels[i - 1]

    return levels


def drape_centerline_water_surface(
    centerline: RiverCenterline,
    stations: np.ndarray,
    xs: Optional[np.ndarray] = None,
    ys: Optional[np.ndarray] = None,
    zs: Optional[np.ndarray] = None,
    sensor_types: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    corridor_width: float = 40.0,
    center_strip_width: float = 1.5,
    max_water_depth: float = 3.5,
    outlier_threshold_m: float = 0.60,
    median_window: int = 5,
    cached_sections: Optional[List[Optional[dict]]] = None,
) -> np.ndarray:
    """
    Estimate clean water surface elevations along a river centerline by draping a narrow corridor
    directly over the water channel, filtering out high riverbank ground and tree canopies.

    Parameters:
        centerline: The RiverCenterline object.
        stations: Sampled 1D array of station distances along the centerline.
        xs, ys, zs: Global corridor point cloud coordinates (if cached_sections is None).
        sensor_types: Sensor type array (1=NIR, 2=Bathy green).
        classes: ASPRS point classification array.
        corridor_width: Full cross-section corridor width in meters.
        center_strip_width: Half-width of central channel strip to inspect (default 1.5 m -> 3.0 m total).
        max_water_depth: Maximum plausible water depth above riverbed to isolate channel from canopy (default 3.5 m).
        outlier_threshold_m: Max deviation from local running median before outlier replacement (default 0.60 m).
        median_window: Window size (number of stations) for running median filtering (default 5).
        cached_sections: Optional pre-sliced cross-sections list from slice_cross_section_points.

    Returns:
        np.ndarray of smooth, monotonic water surface elevations for each station.
    """
    n_sec = len(stations)
    if n_sec == 0:
        return np.array([], dtype=np.float64)

    raw_levels = np.full(n_sec, np.nan, dtype=np.float64)
    pos, tangent, normal = (None, None, None)

    for i in range(n_sec):
        if cached_sections is not None and i < len(cached_sections) and cached_sections[i] is not None:
            sec = cached_sections[i]
        elif xs is not None and ys is not None and zs is not None:
            if pos is None:
                pos, tangent, normal = centerline.evaluate(stations)
            sec = slice_cross_section_points(
                xs, ys, zs,
                center_pos=pos[i],
                normal=normal[i],
                tangent=tangent[i],
                corridor_width=corridor_width,
                slice_thickness=2.0,
                sensor_types=sensor_types,
                classes=classes,
            )
        else:
            sec = None

        if sec is None or len(sec["offset"]) == 0:
            continue

        off = sec["offset"]
        z = sec["z"]
        st = sec.get("sensor_type")

        # 1. Isolate the central river channel strip (e.g. +/- 1.5m)
        strip = np.abs(off) <= center_strip_width
        if strip.sum() < 3:
            # Expand slightly if data is sparse along the centerline
            strip = np.abs(off) <= max(2.5, center_strip_width * 1.6)

        if strip.sum() < 3:
            continue

        z_strip = z[strip]
        st_strip = st[strip] if st is not None else None

        # 2. Estimate riverbed elevation Z_bed (lowest 5th percentile)
        z_bed = float(np.percentile(z_strip, 5.0))

        # 3. Canopy / bridge rejection: only consider points in the plausible water channel column
        # Points higher than z_bed + max_water_depth are overhanging tree branches, leaves, or bridges
        chan_mask = (z_strip >= (z_bed - 0.5)) & (z_strip <= (z_bed + max_water_depth))
        if chan_mask.sum() < 3:
            continue

        chan_z = z_strip[chan_mask]
        chan_st = st_strip[chan_mask] if st_strip is not None else None

        # 4. Determine water surface elevation:
        # Prioritize Topo LiDAR (NIR, sensor_type != 2) because NIR light reflects directly
        # off the air-water boundary. Bathy green returns (sensor_type == 2) penetrate to the bed.
        if chan_st is not None:
            topo_mask = (chan_st != 2)
            if topo_mask.sum() >= 2:
                topo_z = chan_z[topo_mask]
                raw_levels[i] = float(np.percentile(topo_z, 75.0)) if len(topo_z) >= 4 else float(np.median(topo_z))
                continue

            bathy_mask = (chan_st == 2)
            if bathy_mask.sum() >= 3:
                # Top of bathymetric returns as fallback when topo returns are absent
                raw_levels[i] = float(np.percentile(chan_z[bathy_mask], 85.0))
                continue

        # Otherwise use upper envelope (85th percentile) of channel points
        raw_levels[i] = float(np.percentile(chan_z, 85.0))

    # 4. Longitudinal running median filter to eliminate bridge / dense canopy outliers
    filtered = np.copy(raw_levels)
    half_w = max(1, median_window // 2)

    for i in range(n_sec):
        win = raw_levels[max(0, i - half_w) : min(n_sec, i + half_w + 1)]
        valid = win[~np.isnan(win)]
        if len(valid) > 0:
            med = float(np.median(valid))
            if np.isnan(filtered[i]) or abs(filtered[i] - med) > outlier_threshold_m:
                filtered[i] = med

    # 5. Linearly interpolate across any remaining NaNs
    nans = np.isnan(filtered)
    if nans.all():
        return np.zeros(n_sec, dtype=np.float64)
    if nans.any():
        val_idx = np.flatnonzero(~nans)
        filtered[nans] = np.interp(np.flatnonzero(nans), val_idx, filtered[val_idx])

    # 6. Downstream monotonicity enforcement
    return enforce_downstream_monotonicity(filtered)


def interpolate_anchor_sections(
    stations: np.ndarray,
    water_levels: np.ndarray,
    locked_mask: np.ndarray,
    cached_sections: Optional[List[Optional[dict]]] = None,
    z_search_window: float = 1.5,
) -> np.ndarray:
    """
    Interpolate intermediate cross-sections between user-approved locked anchors.

    Instead of blindly linearly averaging across reservoirs, weirs, or power plants,
    the algorithm uses the user anchors as prior guidance to re-examine the real
    in-between point cloud data (NIR shorelines / green surface) within [prior_z - tol, prior_z + tol].
    """
    levels = np.array(water_levels, dtype=np.float64)
    locked_idx = np.flatnonzero(locked_mask)
    if len(locked_idx) == 0:
        return enforce_downstream_monotonicity(levels)
    if len(locked_idx) == 1:
        anchor_i = locked_idx[0]
        anchor_z = levels[anchor_i]
        diff = anchor_z - levels[anchor_i]
        levels += diff
        return enforce_downstream_monotonicity(levels, locked_mask)

    anc_s = stations[locked_idx]
    anc_z = levels[locked_idx]
    priors = np.interp(stations, anc_s, anc_z)

    # Re-evaluate intermediate unlocked sections between consecutive locked anchors
    for k in range(len(locked_idx) - 1):
        i0, i1 = locked_idx[k], locked_idx[k + 1]
        z_up = levels[i0]
        z_down = levels[i1]
        mid_z = 0.5 * (z_up + z_down)
        win = max(z_search_window, 0.5 * abs(z_up - z_down) + 0.5)

        for i in range(i0 + 1, i1):
            if cached_sections is not None and i < len(cached_sections) and cached_sections[i] is not None:
                sec = cached_sections[i]
                w_z, _, _ = detect_section_water_level(
                    sec["offset"], sec["z"], sec["sensor_type"], sec["classification"],
                    prior_z=mid_z, z_window=win,
                )
                if not np.isnan(w_z):
                    levels[i] = min(z_up, max(z_down, w_z))
                else:
                    levels[i] = priors[i]
            else:
                levels[i] = priors[i]

    # Handle sections before first anchor and after last anchor
    for i in range(0, locked_idx[0]):
        levels[i] = max(levels[i], levels[locked_idx[0]])
    for i in range(locked_idx[-1] + 1, len(stations)):
        levels[i] = min(levels[i], levels[locked_idx[-1]])

    return enforce_downstream_monotonicity(levels, locked_mask)


def insert_custom_station(
    stations: np.ndarray,
    water_levels: np.ndarray,
    locked_mask: np.ndarray,
    cached_sections: List[Optional[dict]],
    new_s: float,
    new_z: float,
    new_sec_data: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Optional[dict]], int]:
    """
    Insert a manual user-defined cross-section station into the river sequence.
    Returns:
        (updated_stations, updated_water_levels, updated_locked_mask, updated_cached_sections, insert_index)
    """
    new_s = float(new_s)
    new_z = float(new_z)
    idx = int(np.searchsorted(stations, new_s))

    # If an exact station already exists within 0.1m, just overwrite and lock it
    if idx < len(stations) and abs(stations[idx] - new_s) < 0.1:
        water_levels[idx] = new_z
        locked_mask[idx] = True
        if new_sec_data is not None:
            cached_sections[idx] = new_sec_data
        return stations, water_levels, locked_mask, cached_sections, idx

    up_stations = np.insert(stations, idx, new_s)
    up_levels = np.insert(water_levels, idx, new_z)
    up_locked = np.insert(locked_mask, idx, True)
    up_cached = list(cached_sections)
    up_cached.insert(idx, new_sec_data)

    return up_stations, up_levels, up_locked, up_cached, idx


def remove_station(
    stations: np.ndarray,
    water_levels: np.ndarray,
    locked_mask: np.ndarray,
    cached_sections: List[Optional[dict]],
    idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Optional[dict]], int]:
    """
    Remove a cross-section station at index `idx`.

    Parameters:
        stations: Array of station distances.
        water_levels: Array of water surface elevations.
        locked_mask: Boolean array of anchor lock status.
        cached_sections: List of sliced cross-section point dictionaries.
        idx: 0-based index of the station to remove.

    Returns:
        (updated_stations, updated_water_levels, updated_locked_mask, updated_cached_sections, new_curr_idx)
    """
    n = len(stations)
    if n <= 1:
        raise ValueError("Cannot remove the only remaining cross-section.")
    if idx < 0 or idx >= n:
        raise IndexError(f"Index {idx} out of range for stations length {n}.")

    up_stations = np.delete(stations, idx)
    up_levels = np.delete(water_levels, idx)
    up_locked = np.delete(locked_mask, idx)
    up_cached = list(cached_sections)
    if idx < len(up_cached):
        up_cached.pop(idx)

    new_idx = min(idx, len(up_stations) - 1)
    return up_stations, up_levels, up_locked, up_cached, new_idx


def detect_embankment_extents(
    offsets: np.ndarray,
    elevations: np.ndarray,
    water_z: float,
    corridor_width: float,
    margin: float = 2.5,
) -> Tuple[float, float]:
    """
    Detect the transverse offsets where the water surface touches the left and right embankments,
    plus a margin into the bank terrain.

    Returns:
        (left_offset, right_offset) in metres where left_offset < 0 and right_offset > 0.
    """
    half_w = float(corridor_width * 0.5)
    default_left = -half_w
    default_right = half_w

    if np.isnan(water_z) or len(offsets) == 0:
        return default_left, default_right

    # Select all points below or at the calculated water surface
    # Exclude extreme underground noise (> 20m depth)
    below_mask = (elevations <= (water_z + 0.05)) & (elevations >= (water_z - 20.0))
    if below_mask.sum() == 0:
        below_mask = elevations <= (water_z + 0.05)

    if below_mask.sum() >= 1:
        sub_offsets = offsets[below_mask]
        min_off = float(np.min(sub_offsets))
        max_off = float(np.max(sub_offsets))

        # Left extent: leftmost water point minus bank margin
        left_ext = max(-half_w, min(0.0, min_off) - margin)

        # Right extent: rightmost water point plus bank margin
        right_ext = min(half_w, max(0.0, max_off) + margin)
    else:
        left_ext = default_left
        right_ext = default_right

    # Guarantee minimum width of 2m across channel so it never degenerates
    if right_ext - left_ext < 2.0:
        left_ext = min(left_ext, -1.0)
        right_ext = max(right_ext, 1.0)

    return float(left_ext), float(right_ext)


def rasterize_water_surface_model(
    centerline: RiverCenterline,
    stations: np.ndarray,
    water_levels: np.ndarray,
    corridor_width: float,
    output_path: str | Path,
    resolution: float = 1.0,
    data_epsg: Optional[int] = None,
    left_offsets: Optional[np.ndarray] = None,
    right_offsets: Optional[np.ndarray] = None,
    water_levels_left: Optional[np.ndarray] = None,
    water_levels_right: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Rasterize the 3D water surface cross-sections into a GeoTIFF.
    Supports per-section embankment widths (left_offsets, right_offsets)
    and cross-stream tilt (water_levels_left, water_levels_right).
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to rasterize water surface models.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos, tangent, normal = centerline.evaluate(stations)

    if left_offsets is not None:
        l_off = np.asarray(left_offsets, dtype=np.float64)
    else:
        l_off = np.full(len(stations), -corridor_width * 0.5)

    if right_offsets is not None:
        r_off = np.asarray(right_offsets, dtype=np.float64)
    else:
        r_off = np.full(len(stations), corridor_width * 0.5)

    if water_levels_left is not None:
        z_left = np.asarray(water_levels_left, dtype=np.float64)
    else:
        z_left = np.asarray(water_levels, dtype=np.float64)

    if water_levels_right is not None:
        z_right = np.asarray(water_levels_right, dtype=np.float64)
    else:
        z_right = np.asarray(water_levels, dtype=np.float64)

    # Subdivide station intervals along curved centerline so the quad mesh
    # accurately tracks river bends, especially where sections were removed or spaced apart.
    max_step = max(resolution, min(corridor_width * 0.25, 5.0))

    sub_stations = []
    sub_l_off = []
    sub_r_off = []
    sub_z_left = []
    sub_z_right = []

    for i in range(len(stations) - 1):
        s0, s1 = stations[i], stations[i + 1]
        ds = s1 - s0
        n_sub = max(1, int(np.ceil(ds / max_step)))
        for k in range(n_sub):
            t = k / n_sub
            sub_stations.append(s0 + t * ds)
            sub_l_off.append((1.0 - t) * l_off[i] + t * l_off[i + 1])
            sub_r_off.append((1.0 - t) * r_off[i] + t * r_off[i + 1])
            sub_z_left.append((1.0 - t) * z_left[i] + t * z_left[i + 1])
            sub_z_right.append((1.0 - t) * z_right[i] + t * z_right[i + 1])

    # Append the final station
    sub_stations.append(stations[-1])
    sub_l_off.append(l_off[-1])
    sub_r_off.append(r_off[-1])
    sub_z_left.append(z_left[-1])
    sub_z_right.append(z_right[-1])

    sub_stations = np.array(sub_stations, dtype=np.float64)
    sub_l_off = np.array(sub_l_off, dtype=np.float64)
    sub_r_off = np.array(sub_r_off, dtype=np.float64)
    sub_z_left = np.array(sub_z_left, dtype=np.float64)
    sub_z_right = np.array(sub_z_right, dtype=np.float64)

    pos, tangent, normal = centerline.evaluate(sub_stations)
    left_x = pos[:, 0] + sub_l_off * normal[:, 0]
    left_y = pos[:, 1] + sub_l_off * normal[:, 1]
    right_x = pos[:, 0] + sub_r_off * normal[:, 0]
    right_y = pos[:, 1] + sub_r_off * normal[:, 1]

    all_x = np.concatenate([left_x, right_x])
    all_y = np.concatenate([left_y, right_y])
    min_x = float(np.min(all_x)) - resolution * 2.0
    max_x = float(np.max(all_x)) + resolution * 2.0
    min_y = float(np.min(all_y)) - resolution * 2.0
    max_y = float(np.max(all_y)) + resolution * 2.0

    cols = int(np.ceil((max_x - min_x) / resolution))
    rows = int(np.ceil((max_y - min_y) / resolution))

    transform = from_origin(min_x, max_y, resolution, resolution)

    tri_vertices = []
    tri_z = []

    for i in range(len(sub_stations) - 1):
        p_li = (left_x[i], left_y[i])
        p_ri = (right_x[i], right_y[i])
        p_li1 = (left_x[i + 1], left_y[i + 1])
        p_ri1 = (right_x[i + 1], right_y[i + 1])

        z_li = sub_z_left[i]
        z_ri = sub_z_right[i]
        z_li1 = sub_z_left[i + 1]
        z_ri1 = sub_z_right[i + 1]

        tri_vertices.append([p_li, p_ri, p_li1])
        tri_z.append((z_li, z_ri, z_li1))

        tri_vertices.append([p_ri, p_ri1, p_li1])
        tri_z.append((z_ri, z_ri1, z_li1))

    surface = np.full((rows, cols), np.nan, dtype=np.float32)

    gx = min_x + (np.arange(cols) + 0.5) * resolution
    gy = max_y - (np.arange(rows) + 0.5) * resolution

    for tri, z_vals in zip(tri_vertices, tri_z):
        (x0, y0), (x1, y1), (x2, y2) = tri
        z0, z1, z2 = z_vals

        t_min_x = min(x0, x1, x2)
        t_max_x = max(x0, x1, x2)
        t_min_y = min(y0, y1, y2)
        t_max_y = max(y0, y1, y2)

        c0 = max(0, int(np.floor((t_min_x - min_x) / resolution)))
        c1 = min(cols, int(np.ceil((t_max_x - min_x) / resolution)) + 1)
        r0 = max(0, int(np.floor((max_y - t_max_y) / resolution)))
        r1 = min(rows, int(np.ceil((max_y - t_min_y) / resolution)) + 1)

        if c0 >= c1 or r0 >= r1:
            continue

        sub_gx = gx[c0:c1]
        sub_gy = gy[r0:r1]
        mesh_x, mesh_y = np.meshgrid(sub_gx, sub_gy)

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue

        w0 = ((y1 - y2) * (mesh_x - x2) + (x2 - x1) * (mesh_y - y2)) / denom
        w1 = ((y2 - y0) * (mesh_x - x2) + (x0 - x2) * (mesh_y - y2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if inside.any():
            interp_z = w0 * z0 + w1 * z1 + w2 * z2
            target_slice = surface[r0:r1, c0:c1]
            target_slice[inside] = interp_z[inside]

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": np.nan,
        "width": cols,
        "height": rows,
        "count": 1,
        "crs": f"EPSG:{data_epsg}" if data_epsg else None,
        "transform": transform,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(surface, 1)

    georef = {
        "transform": transform,
        "crs": f"EPSG:{data_epsg}" if data_epsg else None,
        "epsg": data_epsg,
        "nodata": np.nan,
        "width": cols,
        "height": rows,
    }

    logger.info("Saved Water Surface Model GeoTIFF to %s (%dx%d)", output_path, cols, rows)
    return surface, georef


def load_centerline_from_file(path: str | Path) -> np.ndarray:
    """
    Load a 2D polyline centerline from Shapefile, GeoJSON, CSV, or DXF.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Centerline file not found: {path}")

    ext = path.suffix.lower()

    # 1. GeoJSON (pure python via json)
    if ext in (".geojson", ".json"):
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            features = gj.get("features", [gj]) if isinstance(gj, dict) else [gj]
            coords = []
            for feat in features:
                geom = feat.get("geometry", feat)
                gtype = geom.get("type", "")
                gcoords = geom.get("coordinates", [])
                if gtype == "LineString":
                    coords.extend([(float(c[0]), float(c[1])) for c in gcoords])
                elif gtype == "MultiLineString":
                    for line in gcoords:
                        coords.extend([(float(c[0]), float(c[1])) for c in line])
            if coords:
                return np.array(coords, dtype=np.float64)[:, :2]
        except Exception:
            pass

    # 2. Shapefile (pure python binary reader for .shp, works for 2D PolyLine (3) and 3D PolyLineZ (13) and PolyLineM (23))
    if ext == ".shp":
        import struct
        try:
            with open(path, "rb") as f:
                header = f.read(100)
                if len(header) == 100:
                    coords = []
                    while True:
                        rec_hdr = f.read(8)
                        if len(rec_hdr) < 8:
                            break
                        rec_num, content_len = struct.unpack(">2i", rec_hdr)
                        rec_bytes = f.read(content_len * 2)
                        if len(rec_bytes) < 4:
                            break
                        stype = struct.unpack("<i", rec_bytes[:4])[0]
                        # 3: PolyLine, 13: PolyLineZ, 23: PolyLineM, 5: Polygon, 15: PolygonZ
                        if stype in (3, 13, 23, 5, 15):
                            # Box: 4 doubles = 32 bytes (offset 4 to 36)
                            num_parts, num_points = struct.unpack("<2i", rec_bytes[36:44])
                            pts_offset = 44 + num_parts * 4
                            pts_bytes = rec_bytes[pts_offset : pts_offset + num_points * 16]
                            pts = struct.unpack(f"<{num_points * 2}d", pts_bytes)
                            for k in range(0, len(pts), 2):
                                coords.append((pts[k], pts[k + 1]))
                    if coords:
                        return np.array(coords, dtype=np.float64)[:, :2]
        except Exception:
            pass

    # 3. DXF (lightweight ASCII parser for LWPOLYLINE and LINE entities)
    if ext == ".dxf":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f.readlines()]
            coords = []
            cur_x, cur_y = None, None
            in_lwpoly = False
            i = 0
            while i < len(lines) - 1:
                code = lines[i]
                val = lines[i + 1]
                i += 2
                if code == "0" and val == "LWPOLYLINE":
                    in_lwpoly = True
                elif code == "0" and val in ("SEQEND", "ENDSEC", "EOF"):
                    in_lwpoly = False
                elif in_lwpoly:
                    if code == "10":
                        cur_x = float(val)
                    elif code == "20":
                        cur_y = float(val)
                        if cur_x is not None:
                            coords.append((cur_x, cur_y))
                            cur_x, cur_y = None, None
            if coords:
                return np.array(coords, dtype=np.float64)[:, :2]
        except Exception:
            pass

    # 4. CSV / Text (comma, semicolon, tab, or space-separated; ignores elevation Z)
    for delim in [",", ";", "\t", None]:
        try:
            data = np.loadtxt(str(path), delimiter=delim, comments="#")
            if data.ndim == 2 and data.shape[1] >= 2:
                # Discard elevation or any 3rd/4th columns
                return data[:, :2].astype(np.float64)
        except Exception:
            pass

    raise ValueError(f"Could not parse centerline polyline coordinates from {path}")


def clip_segment_to_bbox(
    p0: Sequence[float],
    p1: Sequence[float],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Clip 2D line segment (p0 -> p1) against rectangular bounding box using Liang-Barsky algorithm.
    Returns (clipped_p0, clipped_p1) or None if completely outside.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    p = [-dx, dx, -dy, dy]
    q = [p0[0] - xmin, xmax - p0[0], p0[1] - ymin, ymax - p0[1]]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    return (
        np.array([p0[0] + u1 * dx, p0[1] + u1 * dy], dtype=np.float64),
        np.array([p0[0] + u2 * dx, p0[1] + u2 * dy], dtype=np.float64),
    )


def clip_polyline_to_bbox(
    verts: np.ndarray,
    bbox: Tuple[float, float, float, float],
    buffer_m: float = 20.0,
) -> List[np.ndarray]:
    """
    Clip polyline against a bounding box (min_x, min_y, max_x, max_y) with expansion buffer.
    Returns list of contiguous polyline vertex arrays (each shape (K, 2)) inside the box.
    """
    if len(verts) < 2:
        return []
    xmin, ymin, xmax, ymax = bbox
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m

    chains: List[List[np.ndarray]] = []
    curr_chain: List[np.ndarray] = []

    for i in range(len(verts) - 1):
        clipped = clip_segment_to_bbox(verts[i], verts[i + 1], xmin, ymin, xmax, ymax)
        if clipped is not None:
            c0, c1 = clipped
            if not curr_chain:
                curr_chain = [c0, c1]
            else:
                if np.hypot(curr_chain[-1][0] - c0[0], curr_chain[-1][1] - c0[1]) < 1e-3:
                    curr_chain.append(c1)
                else:
                    chains.append(curr_chain)
                    curr_chain = [c0, c1]
        else:
            if curr_chain:
                chains.append(curr_chain)
                curr_chain = []
    if curr_chain:
        chains.append(curr_chain)

    return [np.array(ch, dtype=np.float64) for ch in chains if len(ch) >= 2]


def polyline_length_in_bbox(
    verts: np.ndarray,
    bbox: Tuple[float, float, float, float],
    buffer_m: float = 0.0,
) -> float:
    """Calculate the total length in metres of a polyline lying inside a bounding box."""
    if len(verts) < 2:
        return 0.0
    xmin, ymin, xmax, ymax = bbox
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m
    tot_len = 0.0
    for i in range(len(verts) - 1):
        c = clip_segment_to_bbox(verts[i], verts[i + 1], xmin, ymin, xmax, ymax)
        if c is not None:
            tot_len += float(np.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]))
    return tot_len


def stitch_contiguous_waterways(ways: List[dict], max_gap: float = 15.0) -> List[dict]:
    """
    Stitch contiguous waterway segments (with matching names) into continuous polylines.
    """
    if not ways:
        return []

    # Group by name / waterway
    unmerged = list(ways)
    merged = []

    while unmerged:
        curr = unmerged.pop(0)
        curr_name = curr.get("name", "")
        curr_pts = list(curr["vertices"])

        changed = True
        while changed:
            changed = False
            for i, other in enumerate(unmerged):
                if curr_name and other.get("name") != curr_name:
                    continue
                o_pts = other["vertices"]

                # Check 4 connectivity cases:
                # 1. curr end to other start
                d_end_start = np.hypot(curr_pts[-1][0] - o_pts[0][0], curr_pts[-1][1] - o_pts[0][1])
                if d_end_start <= max_gap:
                    curr_pts.extend(o_pts[1:] if d_end_start < 1e-4 else o_pts)
                    unmerged.pop(i)
                    changed = True
                    break

                # 2. other end to curr start
                d_start_end = np.hypot(curr_pts[0][0] - o_pts[-1][0], curr_pts[0][1] - o_pts[-1][1])
                if d_start_end <= max_gap:
                    curr_pts = list(o_pts[:-1] if d_start_end < 1e-4 else o_pts) + curr_pts
                    unmerged.pop(i)
                    changed = True
                    break

                # 3. curr end to other end (reverse other)
                d_end_end = np.hypot(curr_pts[-1][0] - o_pts[-1][0], curr_pts[-1][1] - o_pts[-1][1])
                if d_end_end <= max_gap:
                    rev_o = o_pts[::-1]
                    curr_pts.extend(rev_o[1:] if d_end_end < 1e-4 else rev_o)
                    unmerged.pop(i)
                    changed = True
                    break

                # 4. curr start to other start (reverse other)
                d_start_start = np.hypot(curr_pts[0][0] - o_pts[0][0], curr_pts[0][1] - o_pts[0][1])
                if d_start_start <= max_gap:
                    rev_o = o_pts[::-1]
                    curr_pts = list(rev_o[:-1] if d_start_start < 1e-4 else rev_o) + curr_pts
                    unmerged.pop(i)
                    changed = True
                    break

        v_arr = np.array(curr_pts, dtype=np.float64)
        diffs = np.diff(v_arr, axis=0)
        tot_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

        merged.append({
            "id": curr.get("id"),
            "name": curr.get("name", "Unnamed Waterway"),
            "waterway": curr.get("waterway", "river"),
            "vertices": v_arr,
            "length_m": tot_len,
        })

    # Sort descending by length
    merged.sort(key=lambda x: x["length_m"], reverse=True)
    return merged


def fetch_osm_waterways(
    bbox_local: Tuple[float, float, float, float],
    data_epsg: int,
    buffer_m: float = 100.0,
    timeout: int = 25,
) -> List[dict]:
    """
    Fetch river, stream, and canal centerlines from OpenStreetMap via Overpass API.

    Args:
        bbox_local: (min_x, min_y, max_x, max_y) in local project CRS.
        data_epsg: EPSG code of local project CRS (e.g. 25833).
        buffer_m: Bounding box expansion buffer in metres.
        timeout: Network timeout in seconds per endpoint attempt.

    Returns:
        List of candidate river dicts sorted by length inside bbox descending:
            {
                "id": osm_id,
                "name": river_name,
                "waterway": "river" | "stream" | "canal",
                "vertices": np.ndarray of shape (K, 2) in local CRS [X, Y],
                "length_m": float (total OSM polyline length),
                "length_inside_m": float (length within query bbox_local),
            }
    """
    import urllib.parse
    import urllib.request
    import json
    import pyproj

    min_x, min_y, max_x, max_y = bbox_local
    min_x -= buffer_m
    min_y -= buffer_m
    max_x += buffer_m
    max_y += buffer_m

    to_wgs84 = pyproj.Transformer.from_crs(f"EPSG:{data_epsg}", "EPSG:4326", always_xy=True)
    to_local = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{data_epsg}", always_xy=True)

    # 4 corners to WGS84
    lons, lats = to_wgs84.transform(
        [min_x, max_x, max_x, min_x],
        [min_y, min_y, max_y, max_y]
    )
    south = float(np.min(lats))
    north = float(np.max(lats))
    west = float(np.min(lons))
    east = float(np.max(lons))

    # Overpass QL query
    ep_timeout = min(timeout, 8)
    query = f"""[out:json][timeout:{ep_timeout}];
(
  way["waterway"~"river|stream|canal"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
);
out body geom;
"""
    endpoints = [
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]

    import hashlib
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "lidar_workbench" / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    box_round = (round(min_x, -1), round(min_y, -1), round(max_x, -1), round(max_y, -1))
    cache_key = hashlib.md5(f"{data_epsg}_{box_round}".encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"osm_{data_epsg}_{cache_key}.json"

    last_err = None
    res = None
    post_data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    for url in endpoints:
        try:
            req = urllib.request.Request(
                url, data=post_data,
                headers={"User-Agent": "LiDARWorkbench/1.0 (RiverWSM)"}
            )
            with urllib.request.urlopen(req, timeout=ep_timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                break
        except Exception as e:
            last_err = e
            continue

    if res is None:
        # Check local disk cache as fallback
        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_ways = []
                for item in cached_data:
                    item["vertices"] = np.array(item["vertices"], dtype=np.float64)
                    cached_ways.append(item)
                logger.info("Loaded %d OSM waterways from local disk cache (%s)", len(cached_ways), cache_file)
                return cached_ways
            except Exception as ce:
                logger.warning("Failed to read OSM cache %s: %s", cache_file, ce)
        raise RuntimeError(f"Failed to fetch waterways from OpenStreetMap: {last_err}")

    elements = res.get("elements", [])
    raw_ways = []
    for elem in elements:
        geom = elem.get("geometry", [])
        if len(geom) < 2:
            continue
        g_lats = [pt["lat"] for pt in geom]
        g_lons = [pt["lon"] for pt in geom]
        loc_x, loc_y = to_local.transform(g_lons, g_lats)
        verts = np.column_stack((loc_x, loc_y))

        diffs = np.diff(verts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        length_m = float(np.sum(seg_lens))

        tags = elem.get("tags", {})
        raw_ways.append({
            "id": elem.get("id"),
            "name": tags.get("name", "Unnamed Waterway"),
            "waterway": tags.get("waterway", "river"),
            "vertices": verts,
            "length_m": length_m,
        })

    # Stitch contiguous ways
    stitched = stitch_contiguous_waterways(raw_ways)

    # Compute length inside the local project bounding box (unbuffered)
    valid_ways = []
    for w in stitched:
        l_in = polyline_length_in_bbox(w["vertices"], bbox_local, buffer_m=0.0)
        w["length_inside_m"] = l_in
        # Filter out waterways that have negligible overlap with the project box
        if l_in >= 1.0 or w["length_m"] >= 1.0:
            valid_ways.append(w)

    # Sort primarily by length inside project bbox, then by total length
    valid_ways.sort(key=lambda x: (x["length_inside_m"], x["length_m"]), reverse=True)

    # Persist to disk cache
    try:
        to_cache = []
        for w in valid_ways:
            to_cache.append({
                "id": w.get("id"),
                "name": w.get("name"),
                "waterway": w.get("waterway"),
                "vertices": w["vertices"].tolist(),
                "length_m": float(w["length_m"]),
                "length_inside_m": float(w.get("length_inside_m", 0.0)),
            })
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(to_cache, f)
    except Exception as ce:
        logger.debug("Failed to write OSM waterways cache: %s", ce)

    return valid_ways


def save_water_surface_sections(
    file_path: str | Path,
    stations: np.ndarray,
    water_levels: np.ndarray,
    locked_mask: np.ndarray,
    water_levels_left: Optional[np.ndarray] = None,
    water_levels_right: Optional[np.ndarray] = None,
    left_offsets: Optional[np.ndarray] = None,
    right_offsets: Optional[np.ndarray] = None,
    corridor_width: float = 40.0,
    section_spacing: float = 10.0,
    bank_margin: float = 2.5,
    data_epsg: Optional[int] = None,
    centerline: Optional[RiverCenterline] = None,
) -> Path:
    """
    Save the current cross-section profile, elevations, anchors, offsets, and centerline to JSON.
    """
    import json
    from datetime import datetime, timezone

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(stations)
    z_l = water_levels_left if water_levels_left is not None else water_levels
    z_r = water_levels_right if water_levels_right is not None else water_levels
    half_w = corridor_width * 0.5
    l_off = left_offsets if left_offsets is not None else np.full(n, -half_w)
    r_off = right_offsets if right_offsets is not None else np.full(n, half_w)

    sections_list = []
    for i in range(n):
        sections_list.append({
            "station": round(float(stations[i]), 3),
            "water_z": round(float(water_levels[i]), 3),
            "water_z_left": round(float(z_l[i]), 3),
            "water_z_right": round(float(z_r[i]), 3),
            "left_offset": round(float(l_off[i]), 3),
            "right_offset": round(float(r_off[i]), 3),
            "is_locked": bool(locked_mask[i]),
        })

    payload = {
        "file_type": "lidar_workbench_water_surface_profile",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_epsg": data_epsg,
        "corridor_width": float(corridor_width),
        "section_spacing": float(section_spacing),
        "bank_margin": float(bank_margin),
        "centerline": {
            "total_length": round(float(centerline.total_length), 2) if centerline else 0.0,
            "vertices": centerline.vertices.tolist() if centerline else [],
        },
        "sections": sections_list,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return file_path


def load_water_surface_sections(file_path: str | Path) -> dict:
    """
    Load cross-section profile and metadata from a saved JSON file.

    Returns dict with keys:
        'stations': np.ndarray,
        'water_levels': np.ndarray,
        'water_levels_left': np.ndarray,
        'water_levels_right': np.ndarray,
        'left_offsets': np.ndarray,
        'right_offsets': np.ndarray,
        'locked_mask': np.ndarray,
        'corridor_width': float,
        'section_spacing': float,
        'bank_margin': float,
        'data_epsg': Optional[int],
        'centerline_vertices': Optional[np.ndarray],
    """
    import json

    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "sections" not in data:
        raise ValueError(f"Invalid water surface profile file format: {file_path.name}")

    sections = data.get("sections", [])
    n = len(sections)

    stations = np.zeros(n, dtype=np.float64)
    water_levels = np.zeros(n, dtype=np.float64)
    water_levels_left = np.zeros(n, dtype=np.float64)
    water_levels_right = np.zeros(n, dtype=np.float64)
    left_offsets = np.zeros(n, dtype=np.float64)
    right_offsets = np.zeros(n, dtype=np.float64)
    locked_mask = np.zeros(n, dtype=bool)

    for i, s in enumerate(sections):
        stations[i] = float(s.get("station", 0.0))
        water_levels[i] = float(s.get("water_z", np.nan))
        water_levels_left[i] = float(s.get("water_z_left", water_levels[i]))
        water_levels_right[i] = float(s.get("water_z_right", water_levels[i]))
        left_offsets[i] = float(s.get("left_offset", -20.0))
        right_offsets[i] = float(s.get("right_offset", 20.0))
        locked_mask[i] = bool(s.get("is_locked", False))

    cl_verts = None
    if "centerline" in data and "vertices" in data["centerline"]:
        verts_list = data["centerline"]["vertices"]
        if verts_list and len(verts_list) >= 2:
            cl_verts = np.array(verts_list, dtype=np.float64)

    return {
        "stations": stations,
        "water_levels": water_levels,
        "water_levels_left": water_levels_left,
        "water_levels_right": water_levels_right,
        "left_offsets": left_offsets,
        "right_offsets": right_offsets,
        "locked_mask": locked_mask,
        "corridor_width": float(data.get("corridor_width", 40.0)),
        "section_spacing": float(data.get("section_spacing", 10.0)),
        "bank_margin": float(data.get("bank_margin", 2.5)),
        "data_epsg": data.get("data_epsg"),
        "centerline_vertices": cl_verts,
    }



