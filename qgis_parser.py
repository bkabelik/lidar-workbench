"""
LiDAR Workbench — QGIS Project (.qgs / .qgz) & GIS Directory Parser.
Extracts Orthophoto Rasters and Vector Layers with exact QGIS symbology (colors, stroke widths, fill opacity).
"""

from __future__ import annotations

import os
import json
import zipfile
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    import shapefile  # PyShp
except ImportError:
    shapefile = None

try:
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:
    rasterio = None

logger = logging.getLogger("lidar_workbench.qgis_parser")


def rgba_str_to_hex_opacity(rgba_str: str) -> Tuple[str, float]:
    """
    Convert QGIS RGBA string (e.g. '255,0,0,255' or '0,128,255,100') to ('#ff0000', 1.0).
    """
    if not rgba_str:
        return "#38bdf8", 1.0
    parts = [p.strip() for p in rgba_str.split(",")]
    if len(parts) >= 3:
        try:
            r = max(0, min(255, int(parts[0])))
            g = max(0, min(255, int(parts[1])))
            b = max(0, min(255, int(parts[2])))
            a = int(parts[3]) if len(parts) >= 4 else 255
            hex_col = f"#{r:02x}{g:02x}{b:02x}"
            opacity = round(a / 255.0, 3)
            return hex_col, opacity
        except ValueError:
            pass
    return "#38bdf8", 1.0


def parse_qgis_symbology(renderer_elem: Optional[ET.Element]) -> Dict[str, Any]:
    """
    Extract line color, stroke width, fill color, and opacity from QGIS <renderer-v2>.
    """
    style = {
        "stroke": "#ef4444",
        "stroke-width": 2.0,
        "stroke-opacity": 1.0,
        "fill": "#38bdf8",
        "fill-opacity": 0.45,
    }
    if renderer_elem is None:
        return style

    # Scan symbols in renderer
    for symbol in renderer_elem.findall(".//symbol"):
        for prop in symbol.findall(".//prop"):
            k = prop.attrib.get("k", "")
            v = prop.attrib.get("v", "")

            # Line / outline color
            if k in ("line_color", "outline_color", "color"):
                hex_c, op = rgba_str_to_hex_opacity(v)
                style["stroke"] = hex_c
                style["stroke-opacity"] = op

            # Fill color for polygons
            if k in ("fill_color", "color") and symbol.attrib.get("type") == "fill":
                hex_c, op = rgba_str_to_hex_opacity(v)
                style["fill"] = hex_c
                style["fill-opacity"] = op

            # Line / stroke width
            if k in ("line_width", "outline_width", "width"):
                try:
                    w = float(v)
                    # Convert mm/units to pixels roughly
                    style["stroke-width"] = max(1.0, min(8.0, round(w * 3.78, 1)))
                except ValueError:
                    pass

    return style


class QgisProjectParser:
    """Parses a .qgz or .qgs file and resolves layers with their styling."""

    def __init__(self, project_path: str | Path):
        self.project_path = Path(project_path)
        self.project_dir = self.project_path.parent
        self.raster_layers: List[Dict[str, Any]] = []
        self.vector_layers: List[Dict[str, Any]] = []
        self.crs: str = "EPSG:4326"

    def parse(self) -> None:
        xml_content: str = ""
        if self.project_path.suffix.lower() == ".qgz":
            with zipfile.ZipFile(self.project_path, "r") as z:
                # Find the main .qgs file inside zip
                qgs_names = [n for n in z.namelist() if n.endswith(".qgs")]
                if not qgs_names:
                    raise ValueError(f"No .qgs file found inside {self.project_path.name}")
                with z.open(qgs_names[0]) as f:
                    xml_content = f.read().decode("utf-8", errors="ignore")
        elif self.project_path.suffix.lower() == ".qgs":
            with open(self.project_path, "r", encoding="utf-8", errors="ignore") as f:
                xml_content = f.read()
        else:
            raise ValueError(f"Unsupported QGIS format: {self.project_path.suffix}")

        root = ET.fromstring(xml_content)

        # Extract project CRS
        proj_crs = root.find(".//projectCrs/spatialrefsys/authid")
        if proj_crs is not None and proj_crs.text:
            self.crs = proj_crs.text

        # Extract layers
        for layer in root.findall(".//maplayer"):
            layer_type = layer.attrib.get("type", "")
            layer_name = (layer.findtext("layername") or "Layer").strip()
            datasource = (layer.findtext("datasource") or "").strip()

            # Clean datasource path (QGIS sometimes appends '|layername=...' or sqlite options)
            clean_path_str = datasource.split("|")[0].split("?")[0].strip()
            fpath = Path(clean_path_str)
            if not fpath.is_absolute():
                fpath = self.project_dir / fpath

            # Layer CRS
            layer_crs_el = layer.find(".//srs/spatialrefsys/authid")
            layer_crs = layer_crs_el.text if layer_crs_el is not None else self.crs

            if layer_type == "raster":
                if fpath.exists():
                    self.raster_layers.append({
                        "name": layer_name,
                        "path": str(fpath),
                        "crs": layer_crs,
                        "exists": True,
                    })
                else:
                    logger.warning("QGIS raster layer not found on disk: %s", fpath)

            elif layer_type == "vector":
                renderer = layer.find("renderer-v2")
                style = parse_qgis_symbology(renderer)

                if fpath.exists():
                    self.vector_layers.append({
                        "name": layer_name,
                        "path": str(fpath),
                        "crs": layer_crs,
                        "style": style,
                        "exists": True,
                    })
                else:
                    logger.warning("QGIS vector layer not found on disk: %s", fpath)


def read_vector_to_wgs84_geojson(
    file_path: str | Path,
    source_crs: str = "EPSG:4326",
    style: Optional[Dict[str, Any]] = None,
    layer_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reads a Shapefile (.shp) or GeoJSON (.geojson), transforms to WGS84,
    and attaches QGIS style properties to every feature.
    """
    fpath = Path(file_path)
    style = style or {
        "stroke": "#ef4444",
        "stroke-width": 2.0,
        "stroke-opacity": 1.0,
        "fill": "#38bdf8",
        "fill-opacity": 0.4,
    }
    layer_name = layer_name or fpath.stem

    features = []

    # Coordinate transformer to WGS84
    transformer = None
    if pyproj and source_crs and source_crs.upper() != "EPSG:4326":
        try:
            transformer = pyproj.Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        except Exception as e:
            logger.warning("Could not build CRS transformer from %s to EPSG:4326: %s", source_crs, e)

    # 1. Shapefile
    if fpath.suffix.lower() == ".shp" and shapefile:
        try:
            sf = shapefile.Reader(str(fpath))
            fields = [field[0] for field in sf.fields[1:]]

            for shape_rec in sf.shapeRecords():
                geom = shape_rec.shape.__geo_interface__
                props = dict(zip(fields, shape_rec.record))

                # Inject QGIS styling properties
                props["_layer"] = layer_name
                props["stroke"] = style["stroke"]
                props["stroke-width"] = style["stroke-width"]
                props["stroke-opacity"] = style["stroke-opacity"]
                props["fill"] = style["fill"]
                props["fill-opacity"] = style["fill-opacity"]

                # Transform coordinates to WGS84 if needed
                if transformer:
                    geom = _transform_geom_coords(geom, transformer)

                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props
                })
        except Exception as e:
            logger.error("Error reading shapefile %s: %s", fpath, e)

    # 2. GeoJSON
    elif fpath.suffix.lower() in (".json", ".geojson"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_feats = data.get("features", [])
            for rf in raw_feats:
                geom = rf.get("geometry", {})
                props = rf.get("properties", {}) or {}
                props["_layer"] = layer_name
                props["stroke"] = style["stroke"]
                props["stroke-width"] = style["stroke-width"]
                props["stroke-opacity"] = style["stroke-opacity"]
                props["fill"] = style["fill"]
                props["fill-opacity"] = style["fill-opacity"]

                if transformer:
                    geom = _transform_geom_coords(geom, transformer)

                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props
                })
        except Exception as e:
            logger.error("Error reading geojson %s: %s", fpath, e)

    return {
        "type": "FeatureCollection",
        "name": layer_name,
        "features": features
    }


def _transform_geom_coords(geom: dict, transformer: Any) -> dict:
    """Recursively transform geometry coordinates using pyproj transformer."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    def tx_point(pt):
        if len(pt) >= 2:
            lon, lat = transformer.transform(pt[0], pt[1])
            return [round(lon, 7), round(lat, 7)] + list(pt[2:])
        return pt

    if gtype == "Point":
        coords = tx_point(coords)
    elif gtype in ("MultiPoint", "LineString"):
        coords = [tx_point(pt) for pt in coords]
    elif gtype in ("MultiLineString", "Polygon"):
        coords = [[tx_point(pt) for pt in ring] for ring in coords]
    elif gtype == "MultiPolygon":
        coords = [[[tx_point(pt) for pt in ring] for ring in poly] for poly in coords]

    return {"type": gtype, "coordinates": coords}


def get_raster_wgs84_bounds(file_path: str | Path) -> Optional[List[float]]:
    """Get raster bounding box in WGS84: [min_lon, min_lat, max_lon, max_lat]."""
    if not rasterio:
        return None
    try:
        with rasterio.open(str(file_path)) as src:
            bounds = src.bounds
            crs = src.crs
            if not crs:
                return [bounds.left, bounds.bottom, bounds.right, bounds.top]
            wgs84_bounds = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
            return [round(b, 7) for b in wgs84_bounds]
    except Exception as e:
        logger.warning("Could not extract raster bounds for %s: %s", file_path, e)
        return None


def export_ortho_to_web_image(file_path: str | Path, max_dim: int = 4096) -> Tuple[bytes, Optional[List[float]]]:
    """
    Reads a GeoTIFF or image raster, transforms bounds to WGS84,
    and returns (webp_bytes, [min_lon, min_lat, max_lon, max_lat]) ready for MapLibre image layer.
    """
    if not rasterio:
        return b"", None
    try:
        import io
        import numpy as np
        from PIL import Image

        with rasterio.open(str(file_path)) as src:
            bounds = src.bounds
            crs = src.crs
            wgs84_bounds = None
            if crs:
                wgs84_bounds = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
                wgs84_bounds = [round(b, 7) for b in wgs84_bounds]
            else:
                wgs84_bounds = [round(bounds.left, 7), round(bounds.bottom, 7), round(bounds.right, 7), round(bounds.top, 7)]

            out_shape = None
            orig_h, orig_w = src.height, src.width
            if max(orig_h, orig_w) > max_dim:
                scale = max_dim / float(max(orig_h, orig_w))
                out_shape = (int(orig_h * scale), int(orig_w * scale))

            count = src.count
            if count >= 3:
                data = src.read([1, 2, 3], out_shape=out_shape)
                data = np.moveaxis(data, 0, -1)
            elif count == 1:
                single = src.read(1, out_shape=out_shape)
                if np.issubdtype(single.dtype, np.floating):
                    valid = np.isfinite(single)
                    vmin = float(np.nanmin(single[valid])) if np.any(valid) else 0.0
                    vmax = float(np.nanmax(single[valid])) if np.any(valid) else 1.0
                    if vmax > vmin:
                        single = ((single - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                    else:
                        single = np.zeros_like(single, dtype=np.uint8)
                data = np.stack([single, single, single], axis=-1)
            else:
                data = src.read(out_shape=out_shape)
                data = np.moveaxis(data, 0, -1)

            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)

            img = Image.fromarray(data)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=85)
            return buf.getvalue(), wgs84_bounds
    except Exception as e:
        logger.error("Failed to export ortho to web image %s: %s", file_path, e)
        return b"", None
