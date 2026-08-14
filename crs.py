"""
LiDAR Workbench — CRS (Coordinate Reference System) Module.

Handles CRS detection from LAS files, EPSG/WKT lookups via pyproj,
coordinate transformations, and Helmert/affine parameter estimation
from matching point pairs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import pyproj
    from pyproj import CRS, Transformer
    from pyproj.enums import WktVersion
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    pyproj = None
    CRS = None
    Transformer = None
    WktVersion = None

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    laspy = None

logger = logging.getLogger("lidar_workbench.crs")


# ── CRS detection from LAS ─────────────────────────────────────────


def read_crs_from_las(path: str | Path) -> Optional[str]:
    """
    Extract the CRS WKT string from a LAS/LAZ file header.

    Reads VLR record_id=2112 (OGC WKT Coordinate System) per the
    LAS 1.4 specification.

    Args:
        path: Path to a LAS or LAZ file.

    Returns:
        WKT string if found, ``None`` otherwise.
    """
    if not HAS_LASPY:
        logger.warning("laspy not available — cannot read CRS from LAS")
        return None

    path = Path(path)
    if not path.is_file():
        return None

    try:
        with laspy.open(str(path)) as reader:
            for vlr in reader.header.vlrs:
                if hasattr(vlr, 'record_id') and vlr.record_id == 2112:
                    # VLR body contains the WKT string as ASCII
                    if hasattr(vlr, 'record_data'):
                        data = vlr.record_data
                        if isinstance(data, bytes):
                            return data.decode('ascii', errors='replace')
                        return str(data)
                    # Fallback: some laspy versions expose as VLR body
                    body = getattr(vlr, 'body', None) or getattr(vlr, 'record_data', None)
                    if body is not None:
                        if isinstance(body, bytes):
                            return body.decode('ascii', errors='replace')
                        return str(body)
    except Exception:
        logger.debug("Could not read CRS from LAS VLR", exc_info=True)
    return None


def read_crs_from_las_directory(
    directory: str | Path,
) -> dict[str, Optional[str]]:
    """
    Extract CRS from every LAS/LAZ file in a directory.

    Args:
        directory: Path containing LAS/LAZ files.

    Returns:
        ``{filename: crs_wkt}`` mapping.
    """
    directory = Path(directory)
    results: dict[str, Optional[str]] = {}
    for f in sorted(directory.glob("*.las")) + sorted(directory.glob("*.laz")):
        crs = read_crs_from_las(f)
        results[f.name] = crs
    return results


def wkt_to_epsg(wkt: str) -> Optional[int]:
    """
    Try to extract an EPSG code from a WKT string using pyproj.

    Args:
        wkt: OGC WKT coordinate system string.

    Returns:
        EPSG code (e.g. 32633), or ``None`` if not determinable.
    """
    if not HAS_PYPROJ:
        logger.warning("pyproj not available — cannot resolve EPSG from WKT")
        return None
    try:
        crs = CRS.from_wkt(wkt)
        code = crs.to_epsg()
        return int(code) if code is not None else None
    except Exception:
        return None


def epsg_to_wkt(epsg: int, version: str = "WKT2_2019") -> Optional[str]:
    """
    Convert an EPSG code to a WKT string.

    Args:
        epsg:    EPSG code (e.g. 4326, 32633).
        version: ``"WKT1_GDAL"`` or ``"WKT2_2019"``.

    Returns:
        WKT string, or ``None`` on failure.
    """
    if not HAS_PYPROJ:
        return None
    try:
        crs = CRS.from_epsg(epsg)
        wkt_version = getattr(WktVersion, version, WktVersion.WKT2_2019)
        return crs.to_wkt(wkt_version)
    except Exception:
        return None


def get_crs_info(wkt_or_epsg: str | int) -> dict:
    """
    Return a human-readable information dict about a CRS.

    Args:
        wkt_or_epsg: WKT string or EPSG code.

    Returns:
        Dict with keys: ``name``, ``epsg``, ``wkt``, ``type``, ``unit``.
    """
    info: dict = {"name": "Unknown", "epsg": None, "wkt": "", "type": "", "unit": ""}
    if not HAS_PYPROJ:
        return info

    try:
        if isinstance(wkt_or_epsg, int):
            crs = CRS.from_epsg(wkt_or_epsg)
        else:
            crs = CRS.from_wkt(wkt_or_epsg)
        info["name"] = crs.name or "Unknown"
        info["epsg"] = crs.to_epsg()
        info["wkt"] = crs.to_wkt(WktVersion.WKT2_2019) if WktVersion else ""
        info["type"] = "Projected" if crs.is_projected else "Geographic" if crs.is_geographic else "Other"
        info["unit"] = crs.axis_info[0].unit_name if crs.axis_info else ""
    except Exception:
        pass
    return info


def get_epsg_suggestions(search: str, limit: int = 20) -> list[dict]:
    """
    Search pyproj's database for EPSG codes matching *search*.

    Args:
        search: Substring to match against CRS name (e.g. "UTM zone 33").
        limit:  Max results.

    Returns:
        List of ``{"epsg": int, "name": str, "type": str}`` dicts.
    """
    results: list[dict] = []
    search_lower = search.lower()

    if HAS_PYPROJ:
        try:
            # Broad search via pyproj's database
            from pyproj.database import query_crs_info
            for crs in query_crs_info(search_lower, allow_deprecated=False):
                if len(results) >= limit:
                    break
                if search_lower in crs.name.lower():
                    results.append({
                        "epsg": int(crs.code),
                        "name": crs.name,
                        "type": "Projected" if crs.is_projected else "Geographic" if crs.is_geographic else "Other",
                    })
        except Exception:
            pass

        # Also query UTM codes specifically
        if len(results) < limit:
            try:
                for datum in ["WGS 84", "ETRS89", "NAD83"]:
                    for crs in CRS.query_utm_crs_info(datum_name=datum):
                        if len(results) >= limit:
                            break
                        if search_lower in crs.name.lower():
                            results.append({
                                "epsg": int(crs.code),
                                "name": crs.name,
                                "type": "Projected",
                            })
                    if len(results) >= limit:
                        break
            except Exception:
                pass

    # Extended fallback: common EPSG codes
    _COMMON_EPSG = {
        4326: "WGS 84",
        4269: "NAD83",
        4277: "OSGB36",
        4612: "JGD2000",
        3035: "ETRS89-extended / LAEA Europe",
        32601: "WGS 84 / UTM zone 1N", 32602: "WGS 84 / UTM zone 2N",
        32610: "WGS 84 / UTM zone 10N", 32611: "WGS 84 / UTM zone 11N",
        32612: "WGS 84 / UTM zone 12N", 32613: "WGS 84 / UTM zone 13N",
        32614: "WGS 84 / UTM zone 14N", 32615: "WGS 84 / UTM zone 15N",
        32616: "WGS 84 / UTM zone 16N", 32617: "WGS 84 / UTM zone 17N",
        32618: "WGS 84 / UTM zone 18N", 32619: "WGS 84 / UTM zone 19N",
        32620: "WGS 84 / UTM zone 20N", 32621: "WGS 84 / UTM zone 21N",
        32622: "WGS 84 / UTM zone 22N", 32623: "WGS 84 / UTM zone 23N",
        32624: "WGS 84 / UTM zone 24N", 32625: "WGS 84 / UTM zone 25N",
        32626: "WGS 84 / UTM zone 26N", 32627: "WGS 84 / UTM zone 27N",
        32628: "WGS 84 / UTM zone 28N", 32629: "WGS 84 / UTM zone 29N",
        32630: "WGS 84 / UTM zone 30N", 32631: "WGS 84 / UTM zone 31N",
        32632: "WGS 84 / UTM zone 32N", 32633: "WGS 84 / UTM zone 33N",
        32634: "WGS 84 / UTM zone 34N", 32635: "WGS 84 / UTM zone 35N",
        32636: "WGS 84 / UTM zone 36N", 32637: "WGS 84 / UTM zone 37N",
        32638: "WGS 84 / UTM zone 38N", 32639: "WGS 84 / UTM zone 39N",
        32701: "WGS 84 / UTM zone 1S", 32733: "WGS 84 / UTM zone 33S",
        25828: "ETRS89 / UTM zone 28N", 25829: "ETRS89 / UTM zone 29N",
        25830: "ETRS89 / UTM zone 30N", 25831: "ETRS89 / UTM zone 31N",
        25832: "ETRS89 / UTM zone 32N", 25833: "ETRS89 / UTM zone 33N",
        25834: "ETRS89 / UTM zone 34N", 25835: "ETRS89 / UTM zone 35N",
        25836: "ETRS89 / UTM zone 36N", 25837: "ETRS89 / UTM zone 37N",
        26910: "NAD83 / UTM zone 10N", 26911: "NAD83 / UTM zone 11N",
        26912: "NAD83 / UTM zone 12N", 26913: "NAD83 / UTM zone 13N",
        26914: "NAD83 / UTM zone 14N", 26915: "NAD83 / UTM zone 15N",
        26916: "NAD83 / UTM zone 16N", 26917: "NAD83 / UTM zone 17N",
        26918: "NAD83 / UTM zone 18N", 26919: "NAD83 / UTM zone 19N",
        # Europe regional
        31254: "MGI / Austria GK East", 31255: "MGI / Austria GK Central",
        31256: "MGI / Austria GK West", 31257: "MGI / Austria GK M28",
        31258: "MGI / Austria GK M31", 31259: "MGI / Austria GK M34",
        5514: "S-JTSK / Krovak East North", 5513: "S-JTSK / Krovak",
        5221: "S-JTSK / Krovak East North (Greenwich)",
        21781: "CH1903 / LV03", 2056: "CH1903+ / LV95",
        3034: "ETRS89-extended / LCC Europe",
        3044: "ETRS89 / UTM zone 32N (N-zE)",
        3857: "WGS 84 / Pseudo-Mercator",
        27700: "OSGB36 / British National Grid",
        28992: "Amersfoort / RD New",
        2154: "RGF93 v1 / Lambert-93",
        3003: "Monte Mario / Italy zone 1", 3004: "Monte Mario / Italy zone 2",
    }
    for code, name in _COMMON_EPSG.items():
        if len(results) >= limit:
            break
        if search_lower in name.lower() or search_lower in str(code):
            results.append({"epsg": code, "name": name, "type": "Projected"})
    return results


# ── Coordinate transformation ──────────────────────────────────────


def transform_coordinates(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: Optional[np.ndarray],
    source_crs: str | int,
    target_crs: str | int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Transform point coordinates between two CRS.

    Uses pyproj's Transformer which automatically selects the best
    available transformation pipeline.  For Austria this includes
    NTV2 grid-based transforms (e.g. MGI ↔ ETRS89/WGS84 via the
    ``AT_GIS_GRID`` or similar PROJ grid), yielding centimetre-level
    accuracy when the grid files are installed with PROJ.

    For the best results install the PROJ data package:
        ``conda install proj-data``  or  ``pip install pyproj``
    and ensure the grid files are in ``PROJ_DATA`` or ``PROJ_LIB``.

    Args:
        xs, ys, zs: Source coordinates in *source_crs*.
        source_crs: Source CRS (WKT string or EPSG code).
        target_crs: Target CRS (WKT string or EPSG code).

    Returns:
        ``(transformed_xs, transformed_ys, transformed_zs)``.
    """
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj is required for coordinate transformation")

    def _resolve_crs(crs_arg):
        if isinstance(crs_arg, int):
            return CRS.from_epsg(crs_arg)
        # Handle "EPSG:XXXXX" authority strings
        if isinstance(crs_arg, str) and crs_arg.upper().startswith("EPSG:"):
            return CRS.from_epsg(int(crs_arg.split(":")[1]))
        return CRS.from_wkt(crs_arg)

    src = _resolve_crs(source_crs)
    dst = _resolve_crs(target_crs)

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    if zs is not None and len(zs) > 0:
        out_x, out_y, out_z = transformer.transform(xs, ys, zs)
        return (
            np.asarray(out_x, dtype=np.float64),
            np.asarray(out_y, dtype=np.float64),
            np.asarray(out_z, dtype=np.float64),
        )
    else:
        out_x, out_y = transformer.transform(xs, ys)
        return (
            np.asarray(out_x, dtype=np.float64),
            np.asarray(out_y, dtype=np.float64),
            None,
        )


# ── Transformation parameter estimation from matching points ───────


def estimate_helmert_2d(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> dict:
    """
    Estimate a 2-D Helmert (similarity) transformation from matching
    point pairs.

    ``target = scale * R(rotation) * source + translation``

    Uses least-squares with SVD for stable rotation estimation.

    Args:
        source_xy: ``(N, 2)`` source coordinates.
        target_xy: ``(N, 2)`` target coordinates.

    Returns:
        Dict with keys: ``scale``, ``rotation_deg``, ``tx``, ``ty``,
        ``rmse`` (root-mean-square error in target units).
    """
    n = len(source_xy)
    if n < 2:
        raise ValueError("At least 2 point pairs required")

    # Centroids
    src_mean = source_xy.mean(axis=0)
    tgt_mean = target_xy.mean(axis=0)

    # Demeaned
    src_dm = source_xy - src_mean
    tgt_dm = target_xy - tgt_mean

    # Scale
    src_rms = np.sqrt(np.mean(np.sum(src_dm ** 2, axis=1)))
    tgt_rms = np.sqrt(np.mean(np.sum(tgt_dm ** 2, axis=1)))
    if src_rms < 1e-15:
        raise ValueError("Source points are coincident")
    scale = tgt_rms / src_rms

    # Rotation via SVD of covariance matrix
    H = src_dm.T @ tgt_dm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (det = +1)
    det = np.linalg.det(R)
    if det < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    rotation_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    # Translation
    tx = tgt_mean[0] - scale * (R[0, 0] * src_mean[0] + R[0, 1] * src_mean[1])
    ty = tgt_mean[1] - scale * (R[1, 0] * src_mean[0] + R[1, 1] * src_mean[1])

    # Residuals
    transformed = scale * (source_xy @ R.T) + np.array([tx, ty])
    residuals = np.sqrt(np.sum((transformed - target_xy) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "scale": float(scale),
        "rotation_deg": float(rotation_deg),
        "tx": float(tx),
        "ty": float(ty),
        "rotation_matrix": R.tolist(),
        "rmse": rmse,
        "residuals": residuals.tolist(),
    }


def estimate_affine_2d(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> dict:
    """
    Estimate a 6-parameter 2-D affine transformation from matching
    point pairs.

    ``target[0] = a*source[0] + b*source[1] + c``
    ``target[1] = d*source[0] + e*source[1] + f``

    Uses ordinary least squares.

    Args:
        source_xy: ``(N, 2)`` source coordinates.
        target_xy: ``(N, 2)`` target coordinates.

    Returns:
        Dict with keys: ``a``, ``b``, ``c``, ``d``, ``e``, ``f``,
        ``rmse``.
    """
    n = len(source_xy)
    if n < 3:
        raise ValueError("At least 3 point pairs required for affine")

    # Design matrix: [x, y, 1] per row
    A = np.column_stack((source_xy, np.ones(n)))
    # Solve for X: A @ [a, b, c]^T = target_x
    params_x, _, _, _ = np.linalg.lstsq(A, target_xy[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, target_xy[:, 1], rcond=None)

    a, b, c = params_x[0], params_x[1], params_x[2]
    d, e, f = params_y[0], params_y[1], params_y[2]

    # Residuals
    pred_x = a * source_xy[:, 0] + b * source_xy[:, 1] + c
    pred_y = d * source_xy[:, 0] + e * source_xy[:, 1] + f
    residuals = np.sqrt((pred_x - target_xy[:, 0]) ** 2 + (pred_y - target_xy[:, 1]) ** 2)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "a": float(a), "b": float(b), "c": float(c),
        "d": float(d), "e": float(e), "f": float(f),
        "rmse": rmse,
        "residuals": residuals.tolist(),
    }


def apply_helmert_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a 2-D Helmert transformation to coordinates.

    Args:
        xs, ys: Point coordinates.
        params: Dict from :func:`estimate_helmert_2d`.

    Returns:
        ``(transformed_xs, transformed_ys)``.
    """
    R = np.array(params["rotation_matrix"])
    scale = params["scale"]
    tx, ty = params["tx"], params["ty"]
    xy = np.column_stack((xs, ys))
    transformed = scale * (xy @ R.T) + np.array([tx, ty])
    return transformed[:, 0], transformed[:, 1]


def apply_affine_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a 2-D affine transformation to coordinates.

    Args:
        xs, ys: Point coordinates.
        params: Dict from :func:`estimate_affine_2d`.

    Returns:
        ``(transformed_xs, transformed_ys)``.
    """
    a, b, c = params["a"], params["b"], params["c"]
    d, e, f = params["d"], params["e"], params["f"]
    tx = a * xs + b * ys + c
    ty = d * xs + e * ys + f
    return tx, ty


# ── 3-D Helmert (7-parameter similarity) ──────────────────────────


def estimate_helmert_3d(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> dict:
    """
    Estimate a 7-parameter 3-D Helmert (similarity) transformation.

    ``target = scale * R * source + translation``

    where *R* is a 3×3 rotation matrix built from three Euler angles
    (applied in order Z → Y → X).  Uses SVD for robust rotation
    estimation (Kabsch algorithm).

    Args:
        source_xyz: ``(N, 3)`` source coordinates (X, Y, Z).
        target_xyz: ``(N, 3)`` target coordinates (X, Y, Z).

    Returns:
        Dict with keys: ``scale``, ``rotation_matrix`` (3×3),
        ``rotation_xyz_deg`` (Euler angles in degrees around X, Y, Z),
        ``tx``, ``ty``, ``tz``, ``rmse``, ``residuals``.
    """
    n = len(source_xyz)
    if n < 3:
        raise ValueError("At least 3 point pairs required for 3-D Helmert")

    src_mean = source_xyz.mean(axis=0)
    tgt_mean = target_xyz.mean(axis=0)

    src_dm = source_xyz - src_mean
    tgt_dm = target_xyz - tgt_mean

    # Scale
    src_rms = np.sqrt(np.mean(np.sum(src_dm ** 2, axis=1)))
    tgt_rms = np.sqrt(np.mean(np.sum(tgt_dm ** 2, axis=1)))
    if src_rms < 1e-15:
        raise ValueError("Source points are coincident")
    scale = tgt_rms / src_rms

    # Rotation via SVD of 3×3 covariance matrix
    H = src_dm.T @ tgt_dm  # 3×3
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Extract Euler angles (Z-Y-X convention, intrinsic)
    # R = Rz(rz) @ Ry(ry) @ Rx(rx)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-10:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0

    # Translation: T = tgt_mean - scale * R * src_mean
    src_rotated = (src_mean.reshape(1, 3) @ R.T).ravel()
    tx = tgt_mean[0] - scale * src_rotated[0]
    ty = tgt_mean[1] - scale * src_rotated[1]
    tz = tgt_mean[2] - scale * src_rotated[2]

    # Residuals
    transformed = scale * (source_xyz @ R.T) + np.array([tx, ty, tz])
    residuals = np.sqrt(np.sum((transformed - target_xyz) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "scale": float(scale),
        "rotation_matrix": R.tolist(),
        "rotation_xyz_deg": [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))],
        "tx": float(tx), "ty": float(ty), "tz": float(tz),
        "rmse": rmse,
        "residuals": residuals.tolist(),
    }


def apply_helmert_3d(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a 7-parameter 3-D Helmert transformation.

    Args:
        xs, ys, zs: Point coordinates.
        params:     Dict from :func:`estimate_helmert_3d`.

    Returns:
        ``(transformed_xs, transformed_ys, transformed_zs)``.
    """
    R = np.array(params["rotation_matrix"])
    scale = params["scale"]
    tx, ty, tz = params["tx"], params["ty"], params["tz"]
    xyz = np.column_stack((xs, ys, zs))
    transformed = scale * (xyz @ R.T) + np.array([tx, ty, tz])
    return transformed[:, 0], transformed[:, 1], transformed[:, 2]


# ── 3-D affine (12-parameter) ─────────────────────────────────────


def estimate_affine_3d(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> dict:
    """
    Estimate a 12-parameter 3-D affine transformation from matching
    point pairs.

    ``target[0] = a1*x + a2*y + a3*z + a4``
    ``target[1] = b1*x + b2*y + b3*z + b4``
    ``target[2] = c1*x + c2*y + c3*z + c4``

    Uses ordinary least squares.

    Args:
        source_xyz: ``(N, 3)`` source coordinates.
        target_xyz: ``(N, 3)`` target coordinates.

    Returns:
        Dict with keys: ``matrix`` (3×3), ``translation`` (3,),
        ``rmse``, ``residuals``.
    """
    n = len(source_xyz)
    if n < 4:
        raise ValueError("At least 4 point pairs required for 3-D affine")

    A = np.column_stack((source_xyz, np.ones(n)))  # (N, 4)
    params_x, _, _, _ = np.linalg.lstsq(A, target_xyz[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, target_xyz[:, 1], rcond=None)
    params_z, _, _, _ = np.linalg.lstsq(A, target_xyz[:, 2], rcond=None)

    a1, a2, a3, a4 = params_x[0], params_x[1], params_x[2], params_x[3]
    b1, b2, b3, b4 = params_y[0], params_y[1], params_y[2], params_y[3]
    c1, c2, c3, c4 = params_z[0], params_z[1], params_z[2], params_z[3]

    pred_x = a1 * source_xyz[:, 0] + a2 * source_xyz[:, 1] + a3 * source_xyz[:, 2] + a4
    pred_y = b1 * source_xyz[:, 0] + b2 * source_xyz[:, 1] + b3 * source_xyz[:, 2] + b4
    pred_z = c1 * source_xyz[:, 0] + c2 * source_xyz[:, 1] + c3 * source_xyz[:, 2] + c4
    residuals = np.sqrt(
        (pred_x - target_xyz[:, 0]) ** 2
        + (pred_y - target_xyz[:, 1]) ** 2
        + (pred_z - target_xyz[:, 2]) ** 2
    )
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "matrix": [[float(a1), float(a2), float(a3)],
                    [float(b1), float(b2), float(b3)],
                    [float(c1), float(c2), float(c3)]],
        "translation": [float(a4), float(b4), float(c4)],
        "params": {
            "a1": float(a1), "a2": float(a2), "a3": float(a3), "a4": float(a4),
            "b1": float(b1), "b2": float(b2), "b3": float(b3), "b4": float(b4),
            "c1": float(c1), "c2": float(c2), "c3": float(c3), "c4": float(c4),
        },
        "rmse": rmse,
        "residuals": residuals.tolist(),
    }


def apply_affine_3d(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a 12-parameter 3-D affine transformation.

    Args:
        xs, ys, zs: Point coordinates.
        params:     Dict from :func:`estimate_affine_3d`.

    Returns:
        ``(transformed_xs, transformed_ys, transformed_zs)``.
    """
    M = np.array(params["matrix"])
    t = np.array(params["translation"])
    xyz = np.column_stack((xs, ys, zs))
    transformed = xyz @ M.T + t
    return transformed[:, 0], transformed[:, 1], transformed[:, 2]


# ── CRS round-trip helpers ─────────────────────────────────────────


def crs_to_las_vlr_wkt(crs_input: str | int) -> Optional[bytes]:
    """
    Convert a CRS input to a WKT string suitable for LAS VLR record 2112.

    Args:
        crs_input: WKT string or EPSG code.

    Returns:
        UTF-8 encoded WKT bytes, or ``None``.
    """
    if isinstance(crs_input, int):
        wkt = epsg_to_wkt(crs_input, version="WKT1_GDAL")
    else:
        wkt = crs_input
    return wkt.encode("utf-8") if wkt else None
