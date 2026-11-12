"""
Tests for QGIS Project Parser, Vector / Orthophoto Styling, and Cloud Publishing.
"""

import json
import zipfile
import tempfile
from pathlib import Path
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import requests

from qgis_parser import (
    QgisProjectParser,
    parse_qgis_symbology,
    read_vector_to_wgs84_geojson,
    export_ortho_to_web_image,
    get_raster_wgs84_bounds,
    rgba_str_to_hex_opacity,
)


def test_rgba_to_hex():
    hex_col, op = rgba_str_to_hex_opacity("255,128,0,204")
    assert hex_col.lower() == "#ff8000"
    assert 0.79 <= op <= 0.81


def test_qgis_project_parsing_and_symbology(tmp_path):
    # Create sample QGS XML
    qgs_xml = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
    <qgis projectname="Test GIS Survey" version="3.28.0">
      <projectCrs>
        <spatialrefsys>
          <authid>EPSG:4326</authid>
        </spatialrefsys>
      </projectCrs>
      <projectlayers>
        <maplayer type="vector">
          <id>survey_boundary_1</id>
          <datasource>./vectors/boundary.geojson</datasource>
          <layername>Survey Boundary</layername>
          <srs><spatialrefsys><authid>EPSG:4326</authid></spatialrefsys></srs>
          <renderer-v2 type="singleSymbol">
            <symbols>
              <symbol type="line" name="0">
                <layer class="SimpleLine">
                  <prop k="line_color" v="255,50,50,255"/>
                  <prop k="line_width" v="0.8"/>
                </layer>
              </symbol>
            </symbols>
          </renderer-v2>
        </maplayer>
      </projectlayers>
    </qgis>
    """
    
    # Write GeoJSON referenced by QGIS
    vec_dir = tmp_path / "vectors"
    vec_dir.mkdir()
    geojson_file = vec_dir / "boundary.geojson"
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[13.5, 47.1], [13.6, 47.1], [13.6, 47.2], [13.5, 47.2], [13.5, 47.1]]]
                },
                "properties": {"name": "Zone A"}
            }
        ]
    }
    with open(geojson_file, "w") as f:
        json.dump(sample_geojson, f)

    # Write .qgs
    qgs_file = tmp_path / "survey.qgs"
    with open(qgs_file, "w") as f:
        f.write(qgs_xml)

    # Test .qgs parsing
    parser = QgisProjectParser(qgs_file)
    parser.parse()
    assert len(parser.vector_layers) == 1
    assert parser.vector_layers[0]["name"] == "Survey Boundary"
    assert parser.vector_layers[0]["style"]["stroke"].lower() == "#ff3232"

    # Test .qgz (zipped) parsing
    qgz_file = tmp_path / "survey.qgz"
    with zipfile.ZipFile(qgz_file, "w") as z:
        z.writestr("survey.qgs", qgs_xml)

    parser_z = QgisProjectParser(qgz_file)
    parser_z.parse()
    assert len(parser_z.vector_layers) == 1
    assert parser_z.vector_layers[0]["name"] == "Survey Boundary"

    # Test read_vector_to_wgs84_geojson with styling injection
    fc = read_vector_to_wgs84_geojson(
        geojson_file,
        source_crs="EPSG:4326",
        style=parser.vector_layers[0]["style"],
        layer_name=parser.vector_layers[0]["name"],
    )
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["properties"]["_layer"] == "Survey Boundary"
    assert feat["properties"]["stroke"].lower() == "#ff3232"
    assert feat["properties"]["name"] == "Zone A"


def test_export_ortho_to_web_image(tmp_path):
    # Create sample RGB GeoTIFF in EPSG:4326
    tif_path = tmp_path / "orthophoto.tif"
    transform = from_origin(13.50, 47.20, 0.001, 0.001)
    
    # 3 bands 100x100
    arr = np.full((3, 100, 100), 180, dtype=np.uint8)
    with rasterio.open(
        str(tif_path),
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=3,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(arr)

    bounds = get_raster_wgs84_bounds(tif_path)
    assert bounds is not None
    assert round(bounds[0], 2) == 13.50
    assert round(bounds[3], 2) == 47.20

    # Test web export
    webp_bytes, web_bounds = export_ortho_to_web_image(tif_path)
    assert len(webp_bytes) > 0
    assert webp_bytes[:4] == b"RIFF"  # WebP container magic
    assert web_bounds == bounds


def test_cloud_publish_and_viewer_integration(tmp_path):
    server_url = "http://127.0.0.1:8000"
    
    # 1. Publish project metadata
    pub_url = f"{server_url}/api/v1/projects/publish"
    qc_summary = {
        "center_lon": 13.55,
        "center_lat": 47.15,
        "min_alt": 950.0,
        "max_alt": 1250.0,
        "ortho_bounds": [13.50, 47.10, 13.60, 47.20],
        "total_points": 50000,
    }
    resp = requests.post(
        pub_url,
        data={
            "project_name": "QGIS Integration Survey",
            "description": "Survey with QGIS orthophoto and vectors",
            "total_tiles": 4,
            "processed_tiles": 4,
            "status": "completed",
            "qc_summary_json": json.dumps(qc_summary),
            "make_public": True,
        },
        timeout=10,
    )
    assert resp.status_code == 200
    res_body = resp.json()
    project_id = res_body["project_id"]
    share_token = res_body["share_token"]

    # 2. Upload web orthophoto
    upload_url = f"{server_url}/api/v1/projects/{project_id}/upload-asset"
    fake_webp = b"RIFFfake_webp_bytes"
    resp_ortho = requests.post(
        upload_url,
        data={"asset_type": "orthophoto"},
        files={"file": ("orthophoto.webp", fake_webp, "image/webp")},
        timeout=10,
    )
    assert resp_ortho.status_code == 200

    # 3. Upload styled vectors
    fake_vectors = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.55, 47.15]},
                "properties": {"_layer": "Control Points", "stroke": "#10b981", "point_id": "CP-1"}
            }
        ]
    }
    resp_vec = requests.post(
        upload_url,
        data={"asset_type": "vector"},
        files={"file": ("vectors.geojson", json.dumps(fake_vectors).encode("utf-8"), "application/geo+json")},
        timeout=10,
    )
    assert resp_vec.status_code == 200

    # 4. Request the 3D GIS Globe Viewer page and verify links are present
    viewer_resp = requests.get(f"{server_url}/viewer/{share_token}", timeout=10)
    assert viewer_resp.status_code == 200
    html = viewer_resp.text
    assert "QGIS Integration Survey" in html
    assert "orthophoto.webp" in html
    assert "vectors.geojson" in html
    assert "13.5" in html  # Bounds / center present
