"""
osm_basemap.py - OpenStreetMap Slippy Map Tile Downloader, Cache & Reprojection Engine.

Provides asynchronous fetching and reprojecting of OSM tiles to arbitrary
project coordinate reference systems (e.g. UTM EPSG:25833, EPSG:3857, etc.)
for display in PyQtGraph 2D canvases.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Union
import urllib.request

import numpy as np
from PIL import Image
import pyproj
from PySide6.QtCore import QObject, QThread, Signal
import rasterio.warp
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)

# Web Mercator constants
ORIGIN_SHIFT = 20037508.342789244
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lidar_workbench" / "osm_tiles"
USER_AGENT = "LiDARWorkbench/1.0 (https://github.com/bkabelik/lidar-workbench)"


def _get_cache_path(z: int, x: int, y: int, cache_dir: Path) -> Path:
    return cache_dir / str(z) / str(x) / f"{y}.png"


def fetch_osm_tile(z: int, x: int, y: int, cache_dir: Optional[Path] = None) -> Optional[Image.Image]:
    """Fetch a single OSM tile from local disk cache or download via HTTP."""
    cdir = cache_dir or DEFAULT_CACHE_DIR
    tile_file = _get_cache_path(z, x, y, cdir)
    if tile_file.is_file():
        try:
            return Image.open(tile_file).convert("RGBA")
        except Exception:
            pass

    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGBA")
            tile_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(tile_file, "wb") as f:
                    f.write(data)
            except Exception:
                pass
            return img
    except Exception as exc:
        logger.debug("Failed to fetch OSM tile %d/%d/%d: %s", z, x, y, exc)
        return None


def fetch_osm_mosaic(
    viewport_bounds: Tuple[float, float, float, float],
    project_crs: Union[str, int],
    target_size: Tuple[int, int] = (800, 800),
    cache_dir: Optional[Path] = None,
    max_tiles: int = 49,
) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    """
    Fetch, stitch, and reproject OpenStreetMap tiles covering the viewport.

    Args:
        viewport_bounds: (min_x, min_y, max_x, max_y) in project_crs units.
        project_crs: EPSG code (e.g. 25833) or CRS string (e.g. "EPSG:25833").
        target_size: (width, height) desired output pixel resolution.
        cache_dir: Optional path for tile disk caching.
        max_tiles: Maximum number of tiles to download to prevent bandwidth overload.

    Returns:
        (rgba_hwc_uint8, (min_x, min_y, max_x, max_y)) or None if out of bounds.
    """
    if isinstance(project_crs, int):
        crs_str = f"EPSG:{project_crs}"
    else:
        crs_str = str(project_crs)
        if not crs_str.upper().startswith("EPSG:"):
            crs_str = f"EPSG:{crs_str}"

    x0, y0, x1, y1 = viewport_bounds
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)

    if (max_x - min_x) <= 1e-3 or (max_y - min_y) <= 1e-3:
        return None

    try:
        to_3857 = pyproj.Transformer.from_crs(crs_str, "EPSG:3857", always_xy=True)
    except Exception as exc:
        logger.warning("Failed to create CRS transformer to EPSG:3857 from %s: %s", crs_str, exc)
        return None

    # Sample corners + midpoint to get conservative 3857 bounding box
    pts_x = [min_x, max_x, min_x, max_x, (min_x + max_x) * 0.5]
    pts_y = [min_y, min_y, max_y, max_y, (min_y + max_y) * 0.5]
    m_xs, m_ys = to_3857.transform(pts_x, pts_y)
    m_xs = np.array(m_xs)
    m_ys = np.array(m_ys)
    valid = np.isfinite(m_xs) & np.isfinite(m_ys)
    if not np.any(valid):
        return None

    m_minx, m_maxx = float(np.min(m_xs[valid])), float(np.max(m_xs[valid]))
    m_miny, m_maxy = float(np.min(m_ys[valid])), float(np.max(m_ys[valid]))

    # Clamp to EPSG:3857 extent
    m_minx = max(-ORIGIN_SHIFT, min(ORIGIN_SHIFT, m_minx))
    m_maxx = max(-ORIGIN_SHIFT, min(ORIGIN_SHIFT, m_maxx))
    m_miny = max(-ORIGIN_SHIFT, min(ORIGIN_SHIFT, m_miny))
    m_maxy = max(-ORIGIN_SHIFT, min(ORIGIN_SHIFT, m_maxy))

    if (m_maxx - m_minx) <= 1.0 or (m_maxy - m_miny) <= 1.0:
        return None

    # Determine zoom level based on target resolution
    target_w, target_h = max(200, target_size[0]), max(200, target_size[1])
    res_m = (m_maxx - m_minx) / float(target_w)

    # Initial zoom estimate
    # res = (2 * ORIGIN_SHIFT) / (256 * 2^z)  =>  2^z = (2 * ORIGIN_SHIFT) / (256 * res)
    z_ideal = np.log2((2.0 * ORIGIN_SHIFT) / (256.0 * max(1.0, res_m)))
    z = int(np.clip(np.round(z_ideal), 1, 19))

    # Compute tile bounds and adapt zoom if tile count exceeds max_tiles
    while z >= 1:
        n = 2 ** z
        tile_size_m = (2.0 * ORIGIN_SHIFT) / n
        tx0 = int(np.floor((m_minx + ORIGIN_SHIFT) / tile_size_m))
        tx1 = int(np.floor((m_maxx + ORIGIN_SHIFT) / tile_size_m))
        ty0 = int(np.floor((ORIGIN_SHIFT - m_maxy) / tile_size_m))
        ty1 = int(np.floor((ORIGIN_SHIFT - m_miny) / tile_size_m))

        tx0 = max(0, min(n - 1, tx0))
        tx1 = max(0, min(n - 1, tx1))
        ty0 = max(0, min(n - 1, ty0))
        ty1 = max(0, min(n - 1, ty1))

        n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        if n_tiles <= max_tiles:
            break
        z -= 1

    tile_span_x = tx1 - tx0 + 1
    tile_span_y = ty1 - ty0 + 1

    # Fetch tiles in parallel
    tile_coords = [(z, tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    cdir = cache_dir or DEFAULT_CACHE_DIR

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_coord = {
            pool.submit(fetch_osm_tile, cz, cx, cy, cdir): (cx, cy)
            for (cz, cx, cy) in tile_coords
        }
        for fut in future_to_coord:
            cx, cy = future_to_coord[fut]
            try:
                img = fut.result()
                if img is not None:
                    results[(cx, cy)] = img
            except Exception:
                pass

    if not results:
        return None

    # Stitch tiles into composite Web Mercator RGBA image
    mosaic_w = tile_span_x * 256
    mosaic_h = tile_span_y * 256
    mosaic = Image.new("RGBA", (mosaic_w, mosaic_h), (240, 240, 240, 0))

    for (cx, cy), tile_img in results.items():
        px = (cx - tx0) * 256
        py = (cy - ty0) * 256
        mosaic.paste(tile_img, (px, py))

    # Stitched bounds in EPSG:3857
    st_minx = -ORIGIN_SHIFT + tx0 * tile_size_m
    st_maxx = -ORIGIN_SHIFT + (tx1 + 1) * tile_size_m
    st_maxy = ORIGIN_SHIFT - ty0 * tile_size_m
    st_miny = ORIGIN_SHIFT - (ty1 + 1) * tile_size_m

    src_rgba = np.array(mosaic).transpose(2, 0, 1)  # (4, H, W)
    src_tf = from_bounds(st_minx, st_miny, st_maxx, st_maxy, mosaic_w, mosaic_h)

    # Destination bounds in project CRS
    dst_w = min(1200, target_w)
    dst_h = min(1200, target_h)
    dst_tf = from_bounds(min_x, min_y, max_x, max_y, dst_w, dst_h)
    dst_rgba = np.zeros((4, dst_h, dst_w), dtype=np.uint8)

    try:
        rasterio.warp.reproject(
            source=src_rgba,
            destination=dst_rgba,
            src_transform=src_tf,
            src_crs="EPSG:3857",
            dst_transform=dst_tf,
            dst_crs=crs_str,
            resampling=rasterio.warp.Resampling.bilinear,
        )
    except Exception as exc:
        logger.warning("Reprojection from EPSG:3857 to %s failed: %s", crs_str, exc)
        return None

    # Transpose to (H, W, 4)
    dst_hwc = np.ascontiguousarray(dst_rgba.transpose(1, 2, 0))
    return dst_hwc, (min_x, min_y, max_x, max_y)


class OSMBasemapWorker(QThread):
    """
    Background worker that fetches and reprojects OSM basemap tiles
    without stalling the UI.
    """
    mosaic_ready = Signal(object, tuple)  # (rgba_hwc: np.ndarray, (minx, miny, maxx, maxy))

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._pending_request = None
        self._is_running = True

    def request_mosaic(
        self,
        bounds: Tuple[float, float, float, float],
        crs: Union[str, int],
        target_size: Tuple[int, int] = (800, 800),
    ) -> None:
        """Post a new viewport bounding box to fetch."""
        self._pending_request = (bounds, crs, target_size)
        if not self.isRunning():
            self.start()

    def run(self) -> None:
        while self._is_running and self._pending_request is not None:
            req = self._pending_request
            self._pending_request = None
            bounds, crs, target_size = req
            res = fetch_osm_mosaic(bounds, crs, target_size=target_size)
            if res is not None:
                rgba, r_bounds = res
                self.mosaic_ready.emit(rgba, r_bounds)

    def stop(self) -> None:
        self._is_running = False
        self._pending_request = None
        if self.isRunning():
            self.quit()
            self.wait(1000)

