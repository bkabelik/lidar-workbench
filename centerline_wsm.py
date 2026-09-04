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

    # Vector from center to points
    dx = xs - cx
    dy = ys - cy

    # Fast bounding box pre-filter
    max_r = corridor_width * 0.5 + slice_thickness
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

    # Method 2: Bathymetric Green Surface Echo / Channel Upper Envelope
    z_bathy_surf = np.nan
    if bathy_s.sum() >= 5:
        b_z = z_s[bathy_s]
        z_bathy_surf = float(np.percentile(b_z, 92.0))

    # Reconcile bank water levels
    if not np.isnan(z_left) and not np.isnan(z_right):
        if abs(z_left - z_right) <= 0.80:
            water_z = 0.5 * (z_left + z_right)
        else:
            water_z = min(z_left, z_right)
    elif not np.isnan(z_left):
        water_z = z_left
    elif not np.isnan(z_right):
        water_z = z_right
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


def rasterize_water_surface_model(
    centerline: RiverCenterline,
    stations: np.ndarray,
    water_levels: np.ndarray,
    corridor_width: float,
    output_path: str | Path,
    resolution: float = 1.0,
    data_epsg: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Rasterize the 3D horizontal water surface cross-sections into a GeoTIFF.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to rasterize water surface models.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos, tangent, normal = centerline.evaluate(stations)
    half_w = corridor_width * 0.5

    left_x = pos[:, 0] - half_w * normal[:, 0]
    left_y = pos[:, 1] - half_w * normal[:, 1]
    right_x = pos[:, 0] + half_w * normal[:, 0]
    right_y = pos[:, 1] + half_w * normal[:, 1]

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

    for i in range(len(stations) - 1):
        p_li = (left_x[i], left_y[i])
        p_ri = (right_x[i], right_y[i])
        p_li1 = (left_x[i + 1], left_y[i + 1])
        p_ri1 = (right_x[i + 1], right_y[i + 1])
        zi = water_levels[i]
        zi1 = water_levels[i + 1]

        tri_vertices.append([p_li, p_ri, p_li1])
        tri_z.append((zi, zi, zi1))

        tri_vertices.append([p_ri, p_ri1, p_li1])
        tri_z.append((zi, zi1, zi1))

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


def stitch_contiguous_waterways(ways: List[dict], max_gap: float = 30.0) -> List[dict]:
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
        data_epsg: EPSG code of local project CRS (e.g. 25832).
        buffer_m: Bounding box expansion buffer in metres.
        timeout: Network timeout in seconds.

    Returns:
        List of candidate river dicts sorted by length descending:
            {
                "id": osm_id,
                "name": river_name,
                "waterway": "river" | "stream" | "canal",
                "vertices": np.ndarray of shape (K, 2) in local CRS [X, Y],
                "length_m": float,
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
    query = f"""[out:json][timeout:{timeout}];
(
  way["waterway"~"river|stream|canal"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
);
out body geom;
"""
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    last_err = None
    res = None
    post_data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    for url in endpoints:
        try:
            req = urllib.request.Request(
                url, data=post_data,
                headers={"User-Agent": "LiDARWorkbench/1.0 (RiverWSM)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                break
        except Exception as e:
            last_err = e
            continue

    if res is None:
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
    return stitch_contiguous_waterways(raw_ways)


