"""
LiDAR Strip Adjustment Engine (StripAdj).

Provides rigorous alignment of overlapping LiDAR flightlines/strips:
- Trajectory parsing and time-indexed pose interpolation (scanner center X, Y, Z, Roll, Pitch, Yaw).
- Automated extraction of stable planar tie-patches (roofs, roads, flat ground) in overlap zones.
- 3-Step adjustment solver:
    * Step 1: Bulk XYZ translations (ΔX, ΔY, ΔZ per strip).
    * Step 2: Boresight misalignments (ΔRoll, ΔPitch, ΔYaw per strip around scanner center).
    * Step 3: Dynamic trajectory drift (time-dependent spline/piecewise drift correction).
- Application of solved corrections to point clouds and project tiles.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Trajectory Representation & Parsing
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryPose:
    time: float
    x: float
    y: float
    z: float
    roll: float = 0.0   # degrees (positive = right wing down)
    pitch: float = 0.0  # degrees (positive = nose up)
    yaw: float = 0.0    # degrees (0 = North, 90 = East, azimuth / heading)


class Trajectory:
    """
    Time-indexed sensor trajectory for a LiDAR scanner.

    Supports interpolation of scanner center position (x, y, z) and attitude
    (roll, pitch, yaw) at any laser pulse `gps_time`.
    """

    def __init__(self, times: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                 zs: np.ndarray, rolls: Optional[np.ndarray] = None,
                 pitches: Optional[np.ndarray] = None,
                 yaws: Optional[np.ndarray] = None,
                 filepath: Optional[str] = None):
        sort_idx = np.argsort(times)
        self.times = np.asarray(times[sort_idx], dtype=np.float64)
        self.xs = np.asarray(xs[sort_idx], dtype=np.float64)
        self.ys = np.asarray(ys[sort_idx], dtype=np.float64)
        self.zs = np.asarray(zs[sort_idx], dtype=np.float64)

        n = len(self.times)
        self.rolls = np.asarray(rolls[sort_idx] if rolls is not None else np.zeros(n), dtype=np.float64)
        self.pitches = np.asarray(pitches[sort_idx] if pitches is not None else np.zeros(n), dtype=np.float64)
        self.yaws = np.asarray(yaws[sort_idx] if yaws is not None else np.zeros(n), dtype=np.float64)
        self.filepath = filepath

    @property
    def t_min(self) -> float:
        return float(self.times[0]) if len(self.times) else 0.0

    @property
    def t_max(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    def interpolate_position(self, query_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate scanner center XYZ at query_times."""
        qx = np.interp(query_times, self.times, self.xs)
        qy = np.interp(query_times, self.times, self.ys)
        qz = np.interp(query_times, self.times, self.zs)
        return qx, qy, qz

    def interpolate_attitude(self, query_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate roll, pitch, yaw in degrees at query_times."""
        q_roll = np.interp(query_times, self.times, self.rolls)
        q_pitch = np.interp(query_times, self.times, self.pitches)

        # Yaw has circular wrap-around at 360 deg
        rad = np.deg2rad(self.yaws)
        sin_y = np.interp(query_times, self.times, np.sin(rad))
        cos_y = np.interp(query_times, self.times, np.cos(rad))
        q_yaw = np.rad2deg(np.arctan2(sin_y, cos_y)) % 360.0
        return q_roll, q_pitch, q_yaw

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "Trajectory":
        """
        Parse a trajectory file.

        Supports:
        - OPALS ASCII / Riegl trajectory format: [GPSTime, x, y, z, roll, pitch, yaw]
        - Standard space-, comma-, or tab-delimited trajectory files.
        """
        filepath = str(filepath)
        times, xs, ys, zs = [], [], [], []
        rolls, pitches, yaws = [], [], []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "%", "//", "<")):
                    continue
                # Replace comma decimals if necessary, or split by comma/whitespace
                if "," in line and "\t" not in line and " " not in line:
                    parts = line.split(",")
                else:
                    parts = line.split()

                if len(parts) < 4:
                    continue

                try:
                    # Typical columns: time, x, y, z, [roll, pitch, yaw]
                    t = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    r = float(parts[4]) if len(parts) > 4 else 0.0
                    p = float(parts[5]) if len(parts) > 5 else 0.0
                    yw = float(parts[6]) if len(parts) > 6 else 0.0

                    times.append(t)
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    rolls.append(r)
                    pitches.append(p)
                    yaws.append(yw)
                except ValueError:
                    continue

        if not times:
            raise ValueError(f"No valid trajectory records found in {filepath}")

        return cls(
            np.array(times), np.array(xs), np.array(ys), np.array(zs),
            np.array(rolls), np.array(pitches), np.array(yaws),
            filepath=filepath,
        )


# ═══════════════════════════════════════════════════════════════════════
# Tie-Plane & Overlap Extraction
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TieSurface:
    """
    Planar tie-surface patch between Strip A and Strip B.

    Point `p_a` in Strip A corresponds to planar facet with centroid `centroid_b`
    and unit normal `normal_b` in Strip B.
    """
    strip_a: int
    strip_b: int
    point_a: np.ndarray      # (3,) xyz in Strip A
    centroid_b: np.ndarray   # (3,) xyz centroid of facet in Strip B
    normal_b: np.ndarray     # (3,) unit normal of facet in Strip B
    time_a: float            # gps_time of point A
    time_b: float            # gps_time of facet B
    roughness: float = 0.0   # residual standard deviation around facet B

    @property
    def residual(self) -> float:
        """Signed point-to-plane distance: (p_a - c_b) · n_b."""
        return float(np.dot(self.point_a - self.centroid_b, self.normal_b))


def extract_tie_planes(
    strip_a_data: dict,
    strip_b_data: dict,
    strip_a_id: int,
    strip_b_id: int,
    patch_radius: float = 1.5,
    min_patch_pts: int = 15,
    max_roughness: float = 0.04,
    max_samples: int = 2000,
) -> List[TieSurface]:
    """
    Extract robust planar tie-patches between two overlapping strips.

    Candidate points from Strip A in the overlap zone are tested against local
    neighborhoods in Strip B. If Strip B forms a clean plane (low roughness, high planarity),
    a Point-to-Plane tie observation is created.
    """
    xa, ya, za = strip_a_data["x"], strip_a_data["y"], strip_a_data["z"]
    ta = strip_a_data.get("gps_time", np.zeros_like(za))

    xb, yb, zb = strip_b_data["x"], strip_b_data["y"], strip_b_data["z"]
    tb = strip_b_data.get("gps_time", np.zeros_like(zb))

    if len(xa) == 0 or len(xb) == 0:
        return []

    # 1. Bounding box intersection
    min_x = max(xa.min(), xb.min())
    max_x = min(xa.max(), xb.max())
    min_y = max(ya.min(), yb.min())
    max_y = min(ya.max(), yb.max())

    if min_x >= max_x or min_y >= max_y:
        return []  # no horizontal overlap

    # Margin filter for overlap
    mask_a = (xa >= min_x) & (xa <= max_x) & (ya >= min_y) & (ya <= max_y)
    mask_b = (xb >= min_x) & (xb <= max_x) & (yb >= min_y) & (yb <= max_y)

    if mask_a.sum() < min_patch_pts or mask_b.sum() < min_patch_pts:
        return []

    sub_xa, sub_ya, sub_za, sub_ta = xa[mask_a], ya[mask_a], za[mask_a], ta[mask_a]
    sub_xb, sub_yb, sub_zb, sub_tb = xb[mask_b], yb[mask_b], zb[mask_b], tb[mask_b]

    # Build 2D KD-Tree on Strip B in overlap area
    tree_b = cKDTree(np.column_stack((sub_xb, sub_yb)))

    # Subsample candidates in Strip A evenly across the overlap area
    n_cand = len(sub_xa)
    if n_cand > max_samples:
        step = max(1, n_cand // max_samples)
        cand_indices = np.arange(0, n_cand, step)
    else:
        cand_indices = np.arange(n_cand)

    tie_surfaces: List[TieSurface] = []

    for idx in cand_indices:
        pa = np.array([sub_xa[idx], sub_ya[idx], sub_za[idx]])
        t_a = float(sub_ta[idx])

        # Find points in Strip B within patch_radius
        b_indices = tree_b.query_ball_point([pa[0], pa[1]], patch_radius)
        if len(b_indices) < min_patch_pts:
            continue

        b_pts = np.column_stack((sub_xb[b_indices], sub_yb[b_indices], sub_zb[b_indices]))

        # Check vertical range (reject steep vertical objects/walls or multi-story points)
        if (b_pts[:, 2].max() - b_pts[:, 2].min()) > patch_radius * 1.5:
            continue

        # Fit plane to B points via SVD
        centroid_b = b_pts.mean(axis=0)
        centered = b_pts - centroid_b
        cov = np.dot(centered.T, centered) / len(b_pts)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Eigenvalues sorted ascending: l0 <= l1 <= l2
        l0, l1, l2 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
        tot = l0 + l1 + l2
        if tot < 1e-9:
            continue

        # Roughness = standard deviation along normal
        roughness = math.sqrt(max(0.0, l0))
        if roughness > max_roughness:
            continue

        normal_b = eigenvectors[:, 0]  # vector corresponding to smallest eigenvalue
        if np.linalg.norm(normal_b) < 1e-6:
            continue
        normal_b /= np.linalg.norm(normal_b)

        # Ensure normal points upwards (positive Z)
        if normal_b[2] < 0:
            normal_b = -normal_b

        # Filter out near-vertical surfaces (we want ground, road, roof surfaces with slope < 60 deg)
        if normal_b[2] < 0.5:
            continue

        # Check initial point-to-plane residual
        res = float(np.dot(pa - centroid_b, normal_b))
        if abs(res) > 1.5:  # discard wild gross outliers (> 1.5m)
            continue

        t_b = float(np.median(sub_tb[b_indices]))

        tie_surfaces.append(TieSurface(
            strip_a=strip_a_id,
            strip_b=strip_b_id,
            point_a=pa,
            centroid_b=centroid_b,
            normal_b=normal_b,
            time_a=t_a,
            time_b=t_b,
            roughness=roughness,
        ))

    return tie_surfaces


# ═══════════════════════════════════════════════════════════════════════
# 3-Step Strip Adjustment Solver
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StripCorrection:
    """Correction parameters for a single flightline strip."""
    strip_id: int
    dx: float = 0.0          # metres (add to point X)
    dy: float = 0.0          # metres (add to point Y)
    dz: float = 0.0          # metres (add to point Z)
    droll: float = 0.0       # degrees
    dpitch: float = 0.0      # degrees
    dyaw: float = 0.0        # degrees
    # Step 3 drift nodes: list of (time, dx, dy, dz)
    drift_nodes: List[Tuple[float, float, float, float]] = field(default_factory=list)


class StripAdjustmentSolver:
    """
    Multi-strip least-squares optimization solver.

    Step 1: Bulk XYZ shift (ΔX, ΔY, ΔZ per strip)
    Step 2: Boresight angles (ΔRoll, ΔPitch, ΔYaw per strip around scanner center)
    Step 3: Trajectory drift (time-varying smooth spline drift)
    """

    def __init__(self, strip_ids: Sequence[int],
                 trajectories: Optional[Dict[int, Trajectory]] = None,
                 ref_strip_id: Optional[int] = None):
        self.strip_ids = list(sorted(strip_ids))
        self.trajectories = trajectories or {}
        # Reference strip held fixed as datum (defaults to first strip)
        self.ref_strip_id = ref_strip_id if ref_strip_id is not None else self.strip_ids[0]
        self.strip_idx_map = {sid: i for i, sid in enumerate(self.strip_ids)}

    def solve_step1_xyz(self, tie_surfaces: Sequence[TieSurface]) -> Dict[int, Tuple[float, float, float]]:
        """
        Solve Step 1: Global rigid XYZ translation for each strip.

        Residual for tie observation between Strip A and Strip B:
            r = ((p_A + d_A) - (c_B + d_B)) · n_B = 0
            n_B · d_A - n_B · d_B = -(p_A - c_B) · n_B
        """
        n_strips = len(self.strip_ids)
        if n_strips <= 1 or not tie_surfaces:
            return {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}

        # 3 unknowns per strip (dx, dy, dz) -> total 3 * n_strips
        n_params = 3 * n_strips
        A_rows = []
        b_vals = []

        for ts in tie_surfaces:
            if ts.strip_a not in self.strip_idx_map or ts.strip_b not in self.strip_idx_map:
                continue
            ia = self.strip_idx_map[ts.strip_a]
            ib = self.strip_idx_map[ts.strip_b]

            row = np.zeros(n_params, dtype=np.float64)
            # Derivative w.r.t d_A: +n_B
            row[ia * 3 + 0] = ts.normal_b[0]
            row[ia * 3 + 1] = ts.normal_b[1]
            row[ia * 3 + 2] = ts.normal_b[2]

            # Derivative w.r.t d_B: -n_B
            row[ib * 3 + 0] = -ts.normal_b[0]
            row[ib * 3 + 1] = -ts.normal_b[1]
            row[ib * 3 + 2] = -ts.normal_b[2]

            # RHS: -( (p_A - c_B) · n_B )
            rhs = -ts.residual
            A_rows.append(row)
            b_vals.append(rhs)

        if not A_rows:
            return {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}

        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_vals, dtype=np.float64)

        # Fix the reference strip to 0 with high weight
        ref_idx = self.strip_idx_map[self.ref_strip_id]
        weight_ref = 1e4
        for k in range(3):
            fix_row = np.zeros(n_params, dtype=np.float64)
            fix_row[ref_idx * 3 + k] = weight_ref
            A = np.vstack((A, fix_row))
            b = np.append(b, 0.0)

        # Regularization (small ridge penalty to prevent singular horizontal drift)
        ridge = 1e-2
        reg_A = np.eye(n_params, dtype=np.float64) * ridge
        reg_b = np.zeros(n_params, dtype=np.float64)
        A_full = np.vstack((A, reg_A))
        b_full = np.append(b, reg_b)

        solution, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=None)

        results = {}
        for sid, idx in self.strip_idx_map.items():
            dx = float(solution[idx * 3 + 0])
            dy = float(solution[idx * 3 + 1])
            dz = float(solution[idx * 3 + 2])
            results[sid] = (dx, dy, dz)

        return results

    def _get_sensor_pose(self, strip_id: int, time_val: float,
                          point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (sensor_pos, a_roll, a_pitch, a_yaw) for a point in strip_id.
        a_roll: along-track axis (flight direction)
        a_pitch: cross-track axis (wing axis)
        a_yaw: vertical axis (up)
        """
        traj = self.trajectories.get(strip_id)
        if traj is not None and len(traj.times) > 0:
            xs, ys, zs = traj.interpolate_position(np.array([time_val]))
            sensor_pos = np.array([xs[0], ys[0], zs[0]])
            _, _, q_yaw = traj.interpolate_attitude(np.array([time_val]))
            yaw_deg = float(q_yaw[0])
            yaw_rad = math.radians(yaw_deg)
            # Along track vector (azimuth clockwise from North / Y axis)
            # sin(yaw) is East (X), cos(yaw) is North (Y)
            a_roll = np.array([math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
            a_pitch = np.array([math.cos(yaw_rad), -math.sin(yaw_rad), 0.0])
        else:
            # Fallback when no trajectory: assume default flight along Y (or estimated from centroid)
            sensor_pos = point + np.array([0.0, 0.0, 400.0])
            a_roll = np.array([0.0, 1.0, 0.0])
            a_pitch = np.array([1.0, 0.0, 0.0])

        a_yaw = np.array([0.0, 0.0, 1.0])
        return sensor_pos, a_roll, a_pitch, a_yaw

    def solve_step2_boresight(
        self,
        tie_surfaces: Sequence[TieSurface],
        current_shifts: Optional[Dict[int, Tuple[float, float, float]]] = None,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Solve Step 2: Roll, Pitch, and Yaw boresight angles per strip.

        Displacement under small rotation vector around sensor center:
            δp = (θ_roll * a_roll + θ_pitch * a_pitch + θ_yaw * a_yaw) × r
        """
        n_strips = len(self.strip_ids)
        if n_strips <= 1 or not tie_surfaces:
            return {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}

        n_params = 3 * n_strips
        A_rows = []
        b_vals = []

        shifts = current_shifts or {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}

        for ts in tie_surfaces:
            if ts.strip_a not in self.strip_idx_map or ts.strip_b not in self.strip_idx_map:
                continue

            ia = self.strip_idx_map[ts.strip_a]
            ib = self.strip_idx_map[ts.strip_b]

            # Shifted points from Step 1
            sa = np.array(shifts.get(ts.strip_a, (0.0, 0.0, 0.0)))
            sb = np.array(shifts.get(ts.strip_b, (0.0, 0.0, 0.0)))
            pa = ts.point_a + sa
            cb = ts.centroid_b + sb

            # Get sensor position and attitude axes for Strip A and Strip B
            sensor_a, a_roll_a, a_pitch_a, a_yaw_a = self._get_sensor_pose(ts.strip_a, ts.time_a, pa)
            sensor_b, a_roll_b, a_pitch_b, a_yaw_b = self._get_sensor_pose(ts.strip_b, ts.time_b, cb)

            ra = pa - sensor_a
            rb = cb - sensor_b

            # Gradients along normal: (a_axis × r) · n
            ga_roll = float(np.dot(np.cross(a_roll_a, ra), ts.normal_b))
            ga_pitch = float(np.dot(np.cross(a_pitch_a, ra), ts.normal_b))
            ga_yaw = float(np.dot(np.cross(a_yaw_a, ra), ts.normal_b))

            gb_roll = float(np.dot(np.cross(a_roll_b, rb), ts.normal_b))
            gb_pitch = float(np.dot(np.cross(a_pitch_b, rb), ts.normal_b))
            gb_yaw = float(np.dot(np.cross(a_yaw_b, rb), ts.normal_b))

            row = np.zeros(n_params, dtype=np.float64)
            # Strip A unknowns (radians)
            row[ia * 3 + 0] = ga_roll
            row[ia * 3 + 1] = ga_pitch
            row[ia * 3 + 2] = ga_yaw

            # Strip B unknowns (radians)
            row[ib * 3 + 0] = -gb_roll
            row[ib * 3 + 1] = -gb_pitch
            row[ib * 3 + 2] = -gb_yaw

            rhs = -float(np.dot(pa - cb, ts.normal_b))
            A_rows.append(row)
            b_vals.append(rhs)

        if not A_rows:
            return {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}

        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_vals, dtype=np.float64)

        # Fix reference strip rotation to 0
        ref_idx = self.strip_idx_map[self.ref_strip_id]
        weight_ref = 1e5
        for k in range(3):
            fix_row = np.zeros(n_params, dtype=np.float64)
            fix_row[ref_idx * 3 + k] = weight_ref
            A = np.vstack((A, fix_row))
            b = np.append(b, 0.0)

        # Damping regularization on angles
        ridge = 1e-1
        reg_A = np.eye(n_params, dtype=np.float64) * ridge
        reg_b = np.zeros(n_params, dtype=np.float64)
        A_full = np.vstack((A, reg_A))
        b_full = np.append(b, reg_b)

        solution, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=None)

        results = {}
        for sid, idx in self.strip_idx_map.items():
            # Convert solved radians to degrees
            droll = float(np.rad2deg(solution[idx * 3 + 0]))
            dpitch = float(np.rad2deg(solution[idx * 3 + 1]))
            dyaw = float(np.rad2deg(solution[idx * 3 + 2]))
            results[sid] = (droll, dpitch, dyaw)

        return results

    def solve_step3_drift(
        self,
        tie_surfaces: Sequence[TieSurface],
        current_shifts: Optional[Dict[int, Tuple[float, float, float]]] = None,
        current_boresight: Optional[Dict[int, Tuple[float, float, float]]] = None,
        time_knot_interval: float = 20.0,
    ) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """
        Solve Step 3: Time-varying trajectory drift correction along flight timeline.

        Divides each strip timeline into knot intervals and solves a smooth
        piecewise linear / spline correction (time, dx, dy, dz) to absorb
        small GPS/IMU drift.
        """
        results: Dict[int, List[Tuple[float, float, float, float]]] = {}
        shifts = current_shifts or {sid: (0.0, 0.0, 0.0) for sid in self.strip_ids}

        # Group tie surfaces by strip to find timeline span
        for sid in self.strip_ids:
            if sid == self.ref_strip_id:
                # Reference strip has 0 drift
                results[sid] = []
                continue

            # Find all tie observations involving this strip
            obs = [ts for ts in tie_surfaces if ts.strip_a == sid or ts.strip_b == sid]
            if not obs:
                results[sid] = []
                continue

            times = [ts.time_a if ts.strip_a == sid else ts.time_b for ts in obs]
            t_min, t_max = min(times), max(times)
            duration = t_max - t_min

            if duration < time_knot_interval * 0.5:
                # Too short for drift segmentation
                results[sid] = []
                continue

            n_knots = max(2, int(math.ceil(duration / time_knot_interval)) + 1)
            knot_times = np.linspace(t_min, t_max, n_knots)

            # Build linear system for vertical drift at knots
            # (dz(t) = w_k * dz_k + w_{k+1} * dz_{k+1})
            A_list = []
            b_list = []

            for ts in obs:
                is_a = (ts.strip_a == sid)
                t_obs = ts.time_a if is_a else ts.time_b

                # Find knot segment
                pos = (t_obs - t_min) / (t_max - t_min + 1e-9) * (n_knots - 1)
                k0 = int(math.floor(pos))
                k0 = min(max(0, k0), n_knots - 2)
                alpha = pos - k0
                w0 = 1.0 - alpha
                w1 = alpha

                row = np.zeros(n_knots, dtype=np.float64)
                sign = +1.0 if is_a else -1.0
                row[k0] = sign * w0 * ts.normal_b[2]
                row[k0 + 1] = sign * w1 * ts.normal_b[2]

                sa = np.array(shifts.get(ts.strip_a, (0.0, 0.0, 0.0)))
                sb = np.array(shifts.get(ts.strip_b, (0.0, 0.0, 0.0)))
                pa = ts.point_a + sa
                cb = ts.centroid_b + sb
                res = float(np.dot(pa - cb, ts.normal_b))

                A_list.append(row)
                b_list.append(-res)

            if not A_list:
                results[sid] = []
                continue

            A = np.array(A_list, dtype=np.float64)
            b = np.array(b_list, dtype=np.float64)

            # Laplacian smoothness regularizer: (dz_k - 2*dz_{k+1} + dz_{k+2} ~ 0)
            smooth_weight = 5.0
            for k in range(n_knots - 2):
                srow = np.zeros(n_knots, dtype=np.float64)
                srow[k] = 1.0 * smooth_weight
                srow[k + 1] = -2.0 * smooth_weight
                srow[k + 2] = 1.0 * smooth_weight
                A = np.vstack((A, srow))
                b = np.append(b, 0.0)

            # Damping on knot values
            ridge = 0.5
            A = np.vstack((A, np.eye(n_knots, dtype=np.float64) * ridge))
            b = np.append(b, np.zeros(n_knots, dtype=np.float64))

            sol_dz, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            nodes = []
            for k in range(n_knots):
                nodes.append((float(knot_times[k]), 0.0, 0.0, float(sol_dz[k])))
            results[sid] = nodes

        return results


# ═══════════════════════════════════════════════════════════════════════
# Point Cloud Transformation
# ═══════════════════════════════════════════════════════════════════════

def rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """3D rotation matrix for a rotation of angle_rad around unit axis."""
    nrm = np.linalg.norm(axis)
    if nrm < 1e-9 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = axis / nrm
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def apply_strip_corrections(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    times: np.ndarray,
    correction: StripCorrection,
    trajectory: Optional[Trajectory] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply solved Step 1 (XYZ), Step 2 (Boresight), and Step 3 (Drift) corrections
    to a set of points belonging to a specific strip.
    """
    out_x = xs.copy()
    out_y = ys.copy()
    out_z = zs.copy()

    # Step 2: Boresight rotations around scanner center
    if abs(correction.droll) > 1e-6 or abs(correction.dpitch) > 1e-6 or abs(correction.dyaw) > 1e-6:
        roll_rad = math.radians(correction.droll)
        pitch_rad = math.radians(correction.dpitch)
        yaw_rad = math.radians(correction.dyaw)

        if trajectory is not None and len(times) > 0:
            tx, ty, tz = trajectory.interpolate_position(times)
            _, _, q_yaw = trajectory.interpolate_attitude(times)
            # Compute average heading for strip rotation
            mean_yaw_rad = math.radians(float(q_yaw.mean()))
            a_roll = np.array([math.sin(mean_yaw_rad), math.cos(mean_yaw_rad), 0.0])
            a_pitch = np.array([math.cos(mean_yaw_rad), -math.sin(mean_yaw_rad), 0.0])
        else:
            tx = np.full_like(out_x, out_x.mean())
            ty = np.full_like(out_y, out_y.mean())
            tz = np.full_like(out_z, out_z.max() + 400.0)
            a_roll = np.array([0.0, 1.0, 0.0])
            a_pitch = np.array([1.0, 0.0, 0.0])

        a_yaw = np.array([0.0, 0.0, 1.0])

        R_roll = rodrigues_rotation(a_roll, roll_rad)
        R_pitch = rodrigues_rotation(a_pitch, pitch_rad)
        R_yaw = rodrigues_rotation(a_yaw, yaw_rad)
        R_total = R_yaw @ R_pitch @ R_roll

        dx_rel = out_x - tx
        dy_rel = out_y - ty
        dz_rel = out_z - tz

        pts_rel = np.column_stack((dx_rel, dy_rel, dz_rel))
        pts_rot = np.dot(pts_rel, R_total.T)

        out_x = tx + pts_rot[:, 0]
        out_y = ty + pts_rot[:, 1]
        out_z = tz + pts_rot[:, 2]

    # Step 1: Bulk XYZ shift
    out_x += correction.dx
    out_y += correction.dy
    out_z += correction.dz

    # Step 3: Trajectory drift spline
    if correction.drift_nodes and len(times) > 0:
        knot_times = np.array([node[0] for node in correction.drift_nodes])
        knot_dz = np.array([node[3] for node in correction.drift_nodes])
        interp_dz = np.interp(times, knot_times, knot_dz)
        out_z += interp_dz

    return out_x, out_y, out_z
