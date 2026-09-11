"""
point_qc_layers.py - High-Performance Raster and Vector Layer Engine for LiDAR QC.

Provides:
- RasterLayer: Decimated & windowed GeoTIFF streaming via rasterio, overviews,
  memory release on close, custom colormaps, opacity, coordinate value query.
- VectorLayer: Ingests GeoJSON and ESRI Shapefiles (via pyshp), polygon/polyline
  rendering, tile label font size scaling, spatial queries.
- LayerGroup & Workspace: Hierarchical folder grouping and persistent project state on SSD.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, shape

try:
    import shapefile
except ImportError:
    shapefile = None

logger = logging.getLogger(__name__)


class RasterLayer:
    """High-performance raster layer backed by rasterio on SSD with pyramid overviews."""

    def __init__(self, file_path: Path | str, name: Optional[str] = None):
        self.file_path = Path(file_path).resolve()
        self.name = name or self.file_path.stem
        self.visible: bool = True
        self.opacity: float = 1.0
        self.colormap_name: str = "viridis"
        self.vmin: Optional[float] = None
        self.vmax: Optional[float] = None

        # Raster metadata
        self._dataset: Optional[rasterio.DatasetReader] = None
        self.bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
        self.width: int = 0
        self.height: int = 0
        self.crs: str = ""
        fname = self.name.lower()
        self.is_diverging: bool = (
            ("diff" in fname or "dz" in fname or "_vs_" in fname)
            and ("max" not in fname)
        )
        if self.is_diverging:
            self.colormap_name = "coolwarm"
        elif "max" in fname and ("diff" in fname or "strip" in fname):
            self.colormap_name = "turbo"

        self.open()

    def open(self) -> bool:
        """Open dataset handle and read metadata."""
        if not self.file_path.exists():
            logger.error("Raster file not found: %s", self.file_path)
            return False
        try:
            self._dataset = rasterio.open(self.file_path)
            b = self._dataset.bounds
            self.bounds = (b.left, b.bottom, b.right, b.top)
            self.width = self._dataset.width
            self.height = self._dataset.height
            self.crs = str(self._dataset.crs)
            self.nodata = self._dataset.nodata

            # If vmin/vmax not already specified, initialize with auto range
            if self.vmin is None or self.vmax is None:
                self.reset_auto_range()
            return True
        except Exception as exc:
            logger.error("Failed to open raster %s: %s", self.file_path, exc)
            self._dataset = None
            return False

    def compute_auto_range(self) -> Tuple[float, float]:
        """Compute robust auto-stretch limits from thumbnail statistics."""
        if self._dataset is None:
            return (0.0, 1.0)
        try:
            thumb = self._dataset.read(1, out_shape=(100, 100))
            valid = (thumb != self.nodata) & np.isfinite(thumb) if self.nodata is not None else np.isfinite(thumb)
            if np.any(valid):
                v = thumb[valid]
                p1, p99 = np.percentile(v, [1.0, 99.0])
                if self.is_diverging:
                    m = max(abs(float(p1)), abs(float(p99)))
                    return (-m, m)
                elif "max" in self.name.lower() and ("diff" in self.name.lower() or "strip" in self.name.lower()):
                    return (0.0, max(0.2, float(p99)))
                else:
                    return (float(p1), float(p99))
        except Exception as exc:
            logger.warning("Failed to compute auto range for %s: %s", self.name, exc)
        return (0.0, 1.0)

    def reset_auto_range(self) -> None:
        """Reset vmin and vmax to automatic thumbnail statistics."""
        self.vmin, self.vmax = self.compute_auto_range()

    def close(self) -> None:
        """Close dataset handle and free resources."""
        if self._dataset is not None:
            try:
                self._dataset.close()
            except Exception:
                pass
            self._dataset = None

    @property
    def is_open(self) -> bool:
        return self._dataset is not None and not self._dataset.closed

    def get_value_at(self, x: float, y: float) -> Optional[float]:
        """Query raster pixel value at coordinate (x, y)."""
        if not self.is_open and not self.open():
            return None
        if self.bounds is None:
            return None
        minx, miny, maxx, maxy = self.bounds
        if not (minx <= x <= maxx and miny <= y <= maxy):
            return None
        try:
            row, col = self._dataset.index(x, y)
            if 0 <= row < self.height and 0 <= col < self.width:
                # 1x1 window read
                window = rasterio.windows.Window(col, row, 1, 1)
                val = self._dataset.read(1, window=window)[0, 0]
                if self.nodata is not None and np.isclose(val, self.nodata):
                    return None
                return float(val)
        except Exception:
            return None
        return None

    def render_viewport(
        self,
        viewport_bounds: Tuple[float, float, float, float],
        target_size: Tuple[int, int] = (800, 800),
    ) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """
        Read decimated raster window intersecting the current viewport.

        Args:
            viewport_bounds: (min_x, min_y, max_x, max_y)
            target_size: (w, h) in screen pixels for decimation.

        Returns:
            Tuple of (rgba_uint8_array, (render_min_x, render_min_y, render_max_x, render_max_y))
            or None if outside bounds.
        """
        if not self.visible or not self.is_open:
            if self.visible and not self.open():
                return None
            if not self.visible:
                return None

        minx, miny, maxx, maxy = self.bounds
        vx0, vy0, vx1, vy1 = viewport_bounds
        vx0, vx1 = min(vx0, vx1), max(vx0, vx1)
        vy0, vy1 = min(vy0, vy1), max(vy0, vy1)

        # Intersect bounds
        ix0 = max(minx, vx0)
        iy0 = max(miny, vy0)
        ix1 = min(maxx, vx1)
        iy1 = min(maxy, vy1)

        if ix0 >= ix1 or iy0 >= iy1:
            return None

        try:
            window = from_bounds(ix0, iy0, ix1, iy1, transform=self._dataset.transform)
            # Constrain window to dataset dimensions
            window = window.intersection(
                rasterio.windows.Window(0, 0, self.width, self.height)
            )
            if window.width <= 0 or window.height <= 0:
                return None

            # Calculate decimation target size proportional to window aspect ratio
            aspect = float(window.width) / float(window.height)
            tw = min(int(target_size[0]), int(window.width))
            th = max(1, int(round(tw / aspect)))
            th = min(th, int(target_size[1]), int(window.height))
            tw = max(1, tw)

            arr = self._dataset.read(1, window=window, out_shape=(th, tw))
            # Get actual bounds of this window
            actual_bounds = rasterio.windows.bounds(window, transform=self._dataset.transform)

            # Apply colormap to RGBA
            if self.vmin is not None and self.vmax is not None:
                vmin = self.vmin
                vmax = self.vmax
            else:
                valid_mask = (arr != self.nodata) & np.isfinite(arr) if self.nodata is not None else np.isfinite(arr)
                if np.any(valid_mask):
                    vmin = float(np.min(arr[valid_mask]))
                    vmax = float(np.max(arr[valid_mask]))
                else:
                    vmin, vmax = 0.0, 1.0
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0

            # Normalize [0, 1]
            norm_arr = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

            import matplotlib
            cmap = matplotlib.colormaps.get(self.colormap_name, matplotlib.colormaps["viridis"])
            rgba = (cmap(norm_arr) * 255).astype(np.uint8)

            # Mask nodata or non-finite values to alpha 0
            if self.nodata is not None:
                mask = np.isclose(arr, self.nodata) | ~np.isfinite(arr)
            else:
                mask = ~np.isfinite(arr)

            rgba[mask, 3] = 0
            # Apply layer opacity
            if self.opacity < 1.0:
                rgba[~mask, 3] = (rgba[~mask, 3] * self.opacity).astype(np.uint8)

            # Note: rasterio image rows are top-down (0 = top = actual_bounds[3])
            # For PyQtGraph ImageItem, standard convention is (x, y) or flipped rows.
            # We return (rgba, (actual_bounds[0], actual_bounds[1], actual_bounds[2], actual_bounds[3]))
            return rgba, (actual_bounds[0], actual_bounds[1], actual_bounds[2], actual_bounds[3])
        except Exception as exc:
            logger.warning("Error rendering raster viewport %s: %s", self.name, exc)
            return None


class VectorFeature:
    """Represents a vector feature with geometry and properties."""

    def __init__(
        self,
        geom_type: str,
        coordinates: Any,
        properties: Dict[str, Any],
        fid: Optional[str] = None,
    ):
        self.geom_type = geom_type  # "Polygon", "LineString", "Point", etc.
        self.coordinates = coordinates
        self.properties = properties
        self.fid = fid or properties.get("id", properties.get("tile_id", ""))
        self._shapely_geom = None

        # Precompute bounding box
        self.bbox: Tuple[float, float, float, float] = self._calc_bbox()
        self.centroid: Tuple[float, float] = (
            0.5 * (self.bbox[0] + self.bbox[2]),
            0.5 * (self.bbox[1] + self.bbox[3]),
        )

    def _calc_bbox(self) -> Tuple[float, float, float, float]:
        if "min_x" in self.properties and "max_x" in self.properties:
            return (
                float(self.properties["min_x"]),
                float(self.properties["min_y"]),
                float(self.properties["max_x"]),
                float(self.properties["max_y"]),
            )
        pts = self._extract_points(self.coordinates)
        if not pts:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def _extract_points(self, coords: Any) -> List[Tuple[float, float]]:
        flat: List[Tuple[float, float]] = []
        if isinstance(coords, (list, tuple)):
            if len(coords) >= 2 and isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
                flat.append((float(coords[0]), float(coords[1])))
            else:
                for sub in coords:
                    flat.extend(self._extract_points(sub))
        return flat

    @property
    def shapely_geom(self) -> Any:
        if self._shapely_geom is None:
            try:
                self._shapely_geom = shape({
                    "type": self.geom_type,
                    "coordinates": self.coordinates,
                })
            except Exception:
                pass
        return self._shapely_geom


class VectorLayer:
    """High-performance vector layer for GeoJSON and ESRI Shapefiles."""

    def __init__(self, file_path: Path | str, name: Optional[str] = None):
        self.file_path = Path(file_path).resolve()
        self.name = name or self.file_path.stem
        self.visible: bool = True
        self.opacity: float = 1.0
        self.line_color: Tuple[int, int, int] = (255, 255, 0)  # Yellow default
        self.line_width: float = 1.5
        self.show_labels: bool = True
        self.label_font_size: int = 10
        self.label_color: Tuple[int, int, int] = (255, 255, 255)
        self.label_field: str = "tile_id"

        self.features: List[VectorFeature] = []
        self.bounds: Optional[Tuple[float, float, float, float]] = None

        self.open()

    def open(self) -> bool:
        """Parse vector file (GeoJSON or Shapefile) into features."""
        if not self.file_path.exists():
            logger.error("Vector file not found: %s", self.file_path)
            return False

        ext = self.file_path.suffix.lower()
        try:
            if ext in (".geojson", ".json"):
                self._load_geojson()
            elif ext == ".shp":
                self._load_shapefile()
            else:
                logger.warning("Unsupported vector format: %s", ext)
                return False

            if self.features:
                min_x = min(f.bbox[0] for f in self.features)
                min_y = min(f.bbox[1] for f in self.features)
                max_x = max(f.bbox[2] for f in self.features)
                max_y = max(f.bbox[3] for f in self.features)
                self.bounds = (min_x, min_y, max_x, max_y)
            return True
        except Exception as exc:
            logger.error("Failed to load vector layer %s: %s", self.file_path, exc)
            return False

    def _load_geojson(self) -> None:
        with open(self.file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        features_data = doc.get("features", [])
        self.features = []
        for feat in features_data:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            fid = feat.get("id")
            vf = VectorFeature(
                geom_type=geom.get("type", "Polygon"),
                coordinates=geom.get("coordinates", []),
                properties=props,
                fid=str(fid) if fid else None,
            )
            self.features.append(vf)

    def _load_shapefile(self) -> None:
        if shapefile is None:
            raise ImportError("pyshp library is required to read ESRI shapefiles.")
        sf = shapefile.Reader(str(self.file_path))
        self.features = []
        fields = [f[0] for f in sf.fields[1:]]

        for sr in sf.shapeRecords():
            props = dict(zip(fields, sr.record))
            geom = sr.shape.__geo_interface__
            vf = VectorFeature(
                geom_type=geom.get("type", "Polygon"),
                coordinates=geom.get("coordinates", []),
                properties=props,
            )
            self.features.append(vf)

    def close(self) -> None:
        """Clear loaded features to release memory."""
        self.features.clear()
        self.bounds = None

    def find_feature_at(self, x: float, y: float) -> Optional[VectorFeature]:
        """Spatial query: find first feature containing (x, y)."""
        pt = Point(x, y)
        for f in self.features:
            b0, b1, b2, b3 = f.bbox
            if b0 <= x <= b2 and b1 <= y <= b3:
                geom = f.shapely_geom
                if geom and geom.contains(pt):
                    return f
                elif not geom:
                    # Fallback to bbox hit if shapely unavailable
                    return f
        return None

    def find_features_in_bbox(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> List[VectorFeature]:
        """Spatial query: find all features whose bounding box intersects query box."""
        hits = []
        for f in self.features:
            b0, b1, b2, b3 = f.bbox
            if not (b2 < min_x or b0 > max_x or b3 < min_y or b1 > max_y):
                hits.append(f)
        return hits


class LayerGroup:
    """Hierarchical folder group containing child layers or nested groups."""

    def __init__(self, name: str):
        self.name = name
        self.visible: bool = True
        self.children: List[Union[RasterLayer, VectorLayer, LayerGroup]] = []

    @property
    def layers(self) -> List[Union[RasterLayer, VectorLayer, LayerGroup]]:
        return self.children

    def add_layer(self, layer: Union[RasterLayer, VectorLayer, LayerGroup]) -> None:
        self.children.append(layer)

    def remove_layer(self, layer: Union[RasterLayer, VectorLayer, LayerGroup]) -> bool:
        if layer in self.children:
            self.children.remove(layer)
            if hasattr(layer, "close"):
                layer.close()
            return True
        for child in self.children:
            if isinstance(child, LayerGroup):
                if child.remove_layer(layer):
                    return True
        return False

    def get_all_layers(self) -> List[Union[RasterLayer, VectorLayer]]:
        layers = []
        for c in self.children:
            if isinstance(c, LayerGroup):
                layers.extend(c.get_all_layers())
            else:
                layers.append(c)
        return layers

    def to_dict(self) -> Dict[str, Any]:
        """Serialize group state for project persistence."""
        items = []
        for c in self.children:
            if isinstance(c, LayerGroup):
                items.append(c.to_dict())
            elif isinstance(c, RasterLayer):
                items.append({
                    "type": "raster",
                    "path": str(c.file_path),
                    "name": c.name,
                    "visible": c.visible,
                    "opacity": c.opacity,
                    "colormap": c.colormap_name,
                    "vmin": c.vmin,
                    "vmax": c.vmax,
                })
            elif isinstance(c, VectorLayer):
                items.append({
                    "type": "vector",
                    "path": str(c.file_path),
                    "name": c.name,
                    "visible": c.visible,
                    "opacity": c.opacity,
                    "line_color": c.line_color,
                    "line_width": c.line_width,
                    "show_labels": c.show_labels,
                    "label_font_size": c.label_font_size,
                    "label_color": c.label_color,
                })
        return {
            "type": "group",
            "name": self.name,
            "visible": self.visible,
            "children": items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LayerGroup:
        """Reconstruct layer group hierarchy from dictionary."""
        group = cls(name=data.get("name", "Group"))
        group.visible = data.get("visible", True)
        for item in data.get("children", []):
            itype = item.get("type")
            if itype == "group":
                group.add_layer(cls.from_dict(item))
            elif itype == "raster":
                path = item.get("path", "")
                if Path(path).exists():
                    rl = RasterLayer(path, name=item.get("name"))
                    rl.visible = item.get("visible", True)
                    rl.opacity = item.get("opacity", 1.0)
                    rl.colormap_name = item.get("colormap", "viridis")
                    rl.vmin = item.get("vmin")
                    rl.vmax = item.get("vmax")
                    group.add_layer(rl)
            elif itype == "vector":
                path = item.get("path", "")
                if Path(path).exists():
                    vl = VectorLayer(path, name=item.get("name"))
                    vl.visible = item.get("visible", True)
                    vl.opacity = item.get("opacity", 1.0)
                    vl.line_color = tuple(item.get("line_color", (255, 255, 0)))
                    vl.line_width = item.get("line_width", 1.5)
                    vl.show_labels = item.get("show_labels", True)
                    vl.label_font_size = item.get("label_font_size", 10)
                    vl.label_color = tuple(item.get("label_color", (255, 255, 255)))
                    group.add_layer(vl)
        return group


def save_layer_workspace(groups: List[LayerGroup], file_path: Path | str) -> None:
    """Save all layer groups and layers to a JSON project workspace file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": "1.0",
        "groups": [g.to_dict() for g in groups],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    logger.info("Saved QC layer workspace to %s", file_path)


def load_layer_workspace(file_path: Path | str) -> List[LayerGroup]:
    """Load layer groups and layers from a JSON project workspace file."""
    file_path = Path(file_path)
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    groups = []
    for g_data in state.get("groups", []):
        groups.append(LayerGroup.from_dict(g_data))
    return groups
