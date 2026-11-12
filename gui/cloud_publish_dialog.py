"""
LiDAR Workbench — Cloud Sync & Publish Dialog.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import numpy as np

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    import laspy
except ImportError:
    laspy = None

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QMessageBox,
    QApplication,
    QFileDialog,
)

from ..licensing import DEFAULT_SERVER_URL
from ..project_manager import ProjectManager

try:
    from ..qgis_parser import (
        QgisProjectParser,
        read_vector_to_wgs84_geojson,
        export_ortho_to_web_image,
        get_raster_wgs84_bounds,
    )
except ImportError:
    try:
        from qgis_parser import (
            QgisProjectParser,
            read_vector_to_wgs84_geojson,
            export_ortho_to_web_image,
            get_raster_wgs84_bounds,
        )
    except ImportError:
        QgisProjectParser = None
        read_vector_to_wgs84_geojson = None
        export_ortho_to_web_image = None
        get_raster_wgs84_bounds = None

logger = logging.getLogger("lidar_workbench.gui.cloud_publish")


class CloudPublishDialog(QDialog):
    """Dialog allowing users to publish project deliverables and generate client share links."""

    def __init__(self, pm: ProjectManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publish Deliverables to Cloud")
        self.setMinimumWidth(580)
        self._pm = pm
        self._share_url: Optional[str] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        if not self._pm.is_open:
            lbl = QLabel("No project currently open. Please open a project first.")
            layout.addWidget(lbl)
            return

        proj_meta = self._pm.metadata
        proj_name = proj_meta.get("name", "Untitled Project")

        # ── Project Info Group ─────────────────────────────────────────
        info_group = QGroupBox("Project Information")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(QLabel(f"<b>Project Name:</b> {proj_name}"))

        tiles = self._pm.db.get_all_tiles() if self._pm.db else []
        tiles_count = len(tiles)
        classified_count = sum(1 for t in tiles if t.get("classified"))
        info_layout.addWidget(QLabel(f"<b>Tiles:</b> {classified_count} / {tiles_count} classified"))
        layout.addWidget(info_group)

        # ── Publish Settings Group ────────────────────────────────────
        opts_group = QGroupBox("Cloud Publish Settings")
        opts_layout = QVBoxLayout(opts_group)

        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Server URL:"))
        self._server_input = QLineEdit(getattr(self._lic, "server_url", DEFAULT_SERVER_URL))
        server_layout.addWidget(self._server_input)
        opts_layout.addLayout(server_layout)

        self._chk_globe = QCheckBox("Generate & Upload 3D GIS Globe Point Cloud (Open-Source Streaming)")
        self._chk_globe.setChecked(True)
        opts_layout.addWidget(self._chk_globe)

        self._chk_dtm = QCheckBox("Upload Exported DTM Rasters (from dtm/)")
        self._chk_dtm.setChecked(True)
        opts_layout.addWidget(self._chk_dtm)

        self._chk_dsm = QCheckBox("Upload Exported DSM Rasters (from dsm/)")
        self._chk_dsm.setChecked(True)
        opts_layout.addWidget(self._chk_dsm)

        self._chk_qc = QCheckBox("Upload Quality Control (QC) Summary & Accuracy Report")
        self._chk_qc.setChecked(True)
        opts_layout.addWidget(self._chk_qc)

        self._lic = get_license_manager()
        is_ent = (self._lic.plan in ("enterprise", "pro")) or getattr(self._lic, "is_admin", False)

        if is_ent:
            self._chk_las = QCheckBox("Upload Classified LAS / LAZ Deliverable (Full Resolution)")
            quota_text = f"Company Cloud Storage: {self._lic.plan.upper()} Tier ({self._lic.company_name or 'Enterprise'}) — Multi-VPS Storage Active."
            quota_color = "#10b981"
        else:
            self._chk_las = QCheckBox("Upload Classified LAS / LAZ Deliverable (Free Tier: < 500 MB)")
            quota_text = "Cloud Limit: Maximum 1.0 GB total per project deliverable for Free tier."
            quota_color = "#94a3b8"

        self._chk_las.setChecked(True)
        opts_layout.addWidget(self._chk_las)

        self._chk_public = QCheckBox("Enable Public Stakeholder Link (viewable by end clients)")
        self._chk_public.setChecked(True)
        opts_layout.addWidget(self._chk_public)

        quota_notice = QLabel(quota_text)
        quota_notice.setStyleSheet(f"color: {quota_color}; font-size: 11px; font-weight: 500;")
        opts_layout.addWidget(quota_notice)

        layout.addWidget(opts_group)

        # ── QGIS & External GIS Assets Group ─────────────────────────
        gis_group = QGroupBox("QGIS Project & External GIS Assets (Orthophoto & Vectors)")
        gis_layout = QVBoxLayout(gis_group)

        gis_desc = QLabel("Publish georeferenced orthophotos and vector files from disk or a QGIS project (.qgz / .qgs). QGIS styling (stroke colors, line widths, and fill opacities) is automatically converted for 3D web viewing.")
        gis_desc.setWordWrap(True)
        gis_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        gis_layout.addWidget(gis_desc)

        # 1. QGIS Project file
        qgis_row = QHBoxLayout()
        qgis_row.addWidget(QLabel("QGIS Project:"))
        self._qgis_input = QLineEdit()
        self._qgis_input.setPlaceholderText("Select .qgz or .qgs project file (optional)...")
        qgis_row.addWidget(self._qgis_input)
        btn_qgis_browse = QPushButton("Browse...")
        btn_qgis_browse.clicked.connect(self._browse_qgis_project)
        btn_qgis_clear = QPushButton("Clear")
        btn_qgis_clear.clicked.connect(lambda: self._qgis_input.clear())
        qgis_row.addWidget(btn_qgis_browse)
        qgis_row.addWidget(btn_qgis_clear)
        gis_layout.addLayout(qgis_row)

        self._qgis_status_lbl = QLabel("")
        self._qgis_status_lbl.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 500;")
        gis_layout.addWidget(self._qgis_status_lbl)

        # 2. Orthophoto Raster file/dir
        ortho_row = QHBoxLayout()
        ortho_row.addWidget(QLabel("Orthophoto:"))
        self._ortho_input = QLineEdit()
        self._ortho_input.setPlaceholderText("GeoTIFF .tif file or directory with orthophotos...")
        ortho_row.addWidget(self._ortho_input)
        btn_ortho_file = QPushButton("Browse File...")
        btn_ortho_file.clicked.connect(self._browse_ortho_file)
        btn_ortho_dir = QPushButton("Browse Dir...")
        btn_ortho_dir.clicked.connect(self._browse_ortho_dir)
        btn_ortho_clear = QPushButton("Clear")
        btn_ortho_clear.clicked.connect(lambda: self._ortho_input.clear())
        ortho_row.addWidget(btn_ortho_file)
        ortho_row.addWidget(btn_ortho_dir)
        ortho_row.addWidget(btn_ortho_clear)
        gis_layout.addLayout(ortho_row)

        # 3. Vector Layers file/dir
        vector_row = QHBoxLayout()
        vector_row.addWidget(QLabel("Vector Layers:"))
        self._vector_input = QLineEdit()
        self._vector_input.setPlaceholderText("Shapefile (.shp), GeoJSON (.geojson), or directory...")
        vector_row.addWidget(self._vector_input)
        btn_vector_file = QPushButton("Browse File...")
        btn_vector_file.clicked.connect(self._browse_vector_file)
        btn_vector_dir = QPushButton("Browse Dir...")
        btn_vector_dir.clicked.connect(self._browse_vector_dir)
        btn_vector_clear = QPushButton("Clear")
        btn_vector_clear.clicked.connect(lambda: self._vector_input.clear())
        vector_row.addWidget(btn_vector_file)
        vector_row.addWidget(btn_vector_dir)
        vector_row.addWidget(btn_vector_clear)
        gis_layout.addLayout(vector_row)

        layout.addWidget(gis_group)

        # ── Progress & Status ─────────────────────────────────────────
        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setVisible(False)
        layout.addWidget(self._prog_bar)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #38bdf8; font-weight: 500;")
        layout.addWidget(self._status_lbl)

        # ── Result Share Link Box ─────────────────────────────────────
        self._result_group = QGroupBox("Client Share Link")
        self._result_group.setVisible(False)
        res_layout = QVBoxLayout(self._result_group)
        
        self._link_input = QLineEdit()
        self._link_input.setReadOnly(True)
        res_layout.addWidget(self._link_input)

        link_btns = QHBoxLayout()
        copy_btn = QPushButton("Copy Link")
        copy_btn.clicked.connect(self._copy_link)
        open_btn = QPushButton("Launch 3D GIS Globe")
        open_btn.clicked.connect(self._open_in_browser)
        link_btns.addWidget(copy_btn)
        link_btns.addWidget(open_btn)
        res_layout.addLayout(link_btns)
        layout.addWidget(self._result_group)

        # ── Bottom Action Buttons ─────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._publish_btn = QPushButton("Publish Deliverables to Cloud")
        self._publish_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self._publish_btn.clicked.connect(self._on_publish)
        btn_layout.addWidget(self._publish_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _browse_qgis_project(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select QGIS Project File",
            str(self._pm.project_dir or Path.home()),
            "QGIS Projects (*.qgz *.qgs);;All Files (*)",
        )
        if f:
            self._qgis_input.setText(f)
            self._inspect_qgis_project(f)

    def _inspect_qgis_project(self, project_path: str):
        if not QgisProjectParser:
            return
        try:
            parser = QgisProjectParser(project_path)
            parser.parse()
            raster_count = len(parser.raster_layers)
            vector_count = len(parser.vector_layers)
            self._qgis_status_lbl.setText(
                f"Found {raster_count} raster orthophoto(s) and {vector_count} vector layer(s) with QGIS symbology."
            )
            if raster_count > 0 and not self._ortho_input.text():
                self._ortho_input.setText(parser.raster_layers[0]["path"])
            if vector_count > 0 and not self._vector_input.text():
                self._vector_input.setText(parser.vector_layers[0]["path"])
        except Exception as e:
            logger.warning("Could not parse QGIS project: %s", e)
            self._qgis_status_lbl.setText(f"Notice: {e}")

    def _browse_ortho_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select Orthophoto GeoTIFF",
            str(self._pm.project_dir or Path.home()),
            "GeoTIFF Rasters (*.tif *.tiff);;All Files (*)",
        )
        if f:
            self._ortho_input.setText(f)

    def _browse_ortho_dir(self):
        d = QFileDialog.getExistingDirectory(
            self,
            "Select Orthophoto Directory",
            str(self._pm.project_dir or Path.home()),
        )
        if d:
            self._ortho_input.setText(d)

    def _browse_vector_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vector File (Shapefile / GeoJSON)",
            str(self._pm.project_dir or Path.home()),
            "Vector Files (*.shp *.geojson *.json);;All Files (*)",
        )
        if f:
            self._vector_input.setText(f)

    def _browse_vector_dir(self):
        d = QFileDialog.getExistingDirectory(
            self,
            "Select Vector Directory",
            str(self._pm.project_dir or Path.home()),
        )
        if d:
            self._vector_input.setText(d)

    def _copy_link(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self._link_input.text())
        QMessageBox.information(self, "Copied", "Client share link copied to clipboard.")

    def _open_in_browser(self):
        if self._share_url:
            QDesktopServices.openUrl(QUrl(self._share_url))

    def _generate_web_pointcloud_preview(self) -> tuple[bytes, dict]:
        """Sample points across project tiles, convert coordinates to WGS84, and pack into binary float32."""
        tiles = self._pm.db.get_all_tiles() if self._pm.db else []
        if not tiles or laspy is None:
            return b"", {}

        all_lons = []
        all_lats = []
        all_alts = []
        all_clss = []
        all_ints = []
        
        crs_str = self._pm.metadata.get("crs", "EPSG:32633")
        transformer = None
        if pyproj:
            try:
                transformer = pyproj.Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
            except Exception as e:
                logger.warning("Could not build CRS transformer for %s: %s", crs_str, e)

        points_per_tile = max(5000, 200000 // max(1, len(tiles)))
        for t in tiles[:15]:
            fpath = Path(t.get("file_path", ""))
            if not fpath.is_absolute() and self._pm.project_dir:
                fpath = self._pm.project_dir / fpath
            if not fpath.exists():
                continue
            try:
                with laspy.open(str(fpath)) as f:
                    las = f.read()
                    n = len(las.x)
                    if n == 0:
                        continue
                    step = max(1, n // points_per_tile)
                    xs = np.array(las.x[::step], dtype=np.float64)
                    ys = np.array(las.y[::step], dtype=np.float64)
                    zs = np.array(las.z[::step], dtype=np.float32)
                    clss = np.array(las.classification[::step], dtype=np.float32)
                    ints = np.array(las.intensity[::step], dtype=np.float32) if hasattr(las, "intensity") else np.zeros_like(zs)
                    
                    if transformer:
                        lons, lats = transformer.transform(xs, ys)
                    else:
                        lons, lats = xs, ys

                    all_lons.append(lons)
                    all_lats.append(lats)
                    all_alts.append(zs)
                    all_clss.append(clss)
                    all_ints.append(ints)
            except Exception as e:
                logger.warning("Error reading tile for web preview %s: %s", fpath, e)

        if not all_lons:
            return b"", {}

        lons = np.concatenate(all_lons).astype(np.float32)
        lats = np.concatenate(all_lats).astype(np.float32)
        alts = np.concatenate(all_alts).astype(np.float32)
        clss = np.concatenate(all_clss).astype(np.float32)
        ints = np.concatenate(all_ints).astype(np.float32)

        stats = {
            "center_lon": float(np.mean(lons)),
            "center_lat": float(np.mean(lats)),
            "min_alt": float(np.min(alts)),
            "max_alt": float(np.max(alts)),
            "point_count": int(len(lons)),
        }

        # Pack: [lon, lat, alt, cls, intensity] as 5 x Float32 (20 bytes per point)
        packed = np.column_stack([lons, lats, alts, clss, ints]).tobytes()
        return packed, stats

    def _on_publish(self):
        server_url = self._server_input.text().strip().rstrip("/")
        if not server_url:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid server URL.")
            return

        self._publish_btn.setEnabled(False)
        self._prog_bar.setVisible(True)
        self._prog_bar.setValue(10)
        self._status_lbl.setText("Generating 3D GIS Globe point cloud preview...")
        QApplication.processEvents()

        proj_meta = self._pm.metadata
        proj_name = proj_meta.get("name", "Untitled Project")
        tiles = self._pm.db.get_all_tiles() if self._pm.db else []
        total_tiles = len(tiles)
        processed_tiles = sum(1 for t in tiles if t.get("classified"))

        # 1. Generate Web Point Cloud Preview
        preview_bytes, geo_stats = b"", {}
        if self._chk_globe.isChecked():
            preview_bytes, geo_stats = self._generate_web_pointcloud_preview()

        # 2. Inspect QGIS Project & External GIS Layers
        qgis_parser = None
        qgis_path_str = self._qgis_input.text().strip()
        if qgis_path_str and os.path.isfile(qgis_path_str) and QgisProjectParser:
            try:
                qgis_parser = QgisProjectParser(qgis_path_str)
                qgis_parser.parse()
            except Exception as e:
                logger.warning("Error parsing QGIS project: %s", e)

        # 2a. Resolve Orthophoto Raster
        ortho_file = None
        ortho_path_str = self._ortho_input.text().strip()
        if ortho_path_str:
            op = Path(ortho_path_str)
            if op.is_file():
                ortho_file = op
            elif op.is_dir():
                for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.webp"):
                    matches = list(op.glob(ext))
                    if matches:
                        ortho_file = matches[0]
                        break
        elif qgis_parser and qgis_parser.raster_layers:
            ortho_file = Path(qgis_parser.raster_layers[0]["path"])

        ortho_webp_bytes = b""
        if ortho_file and ortho_file.exists():
            self._status_lbl.setText(f"Processing orthophoto raster: {ortho_file.name}...")
            QApplication.processEvents()
            if export_ortho_to_web_image:
                ortho_webp_bytes, ortho_bounds = export_ortho_to_web_image(ortho_file)
                if ortho_bounds:
                    geo_stats["ortho_bounds"] = ortho_bounds
                    if not geo_stats.get("center_lon") or geo_stats.get("center_lon") == 0:
                        geo_stats["center_lon"] = round((ortho_bounds[0] + ortho_bounds[2]) / 2.0, 7)
                        geo_stats["center_lat"] = round((ortho_bounds[1] + ortho_bounds[3]) / 2.0, 7)
            elif get_raster_wgs84_bounds:
                ortho_bounds = get_raster_wgs84_bounds(ortho_file)
                if ortho_bounds:
                    geo_stats["ortho_bounds"] = ortho_bounds

        # 2b. Resolve Vector Layers (preserving exact QGIS symbology)
        all_vector_features = []
        if qgis_parser and qgis_parser.vector_layers and read_vector_to_wgs84_geojson:
            for vl in qgis_parser.vector_layers:
                try:
                    fc = read_vector_to_wgs84_geojson(
                        vl["path"],
                        source_crs=vl.get("crs", qgis_parser.crs),
                        style=vl.get("style"),
                        layer_name=vl.get("name"),
                    )
                    all_vector_features.extend(fc.get("features", []))
                except Exception as e:
                    logger.warning("Error reading QGIS vector %s: %s", vl.get("name"), e)

        vec_path_str = self._vector_input.text().strip()
        if vec_path_str and read_vector_to_wgs84_geojson:
            vp = Path(vec_path_str)
            vfiles = []
            if vp.is_file():
                vfiles.append(vp)
            elif vp.is_dir():
                for ext in ("*.shp", "*.geojson", "*.json"):
                    vfiles.extend(vp.glob(ext))
            for vf in vfiles:
                try:
                    fc = read_vector_to_wgs84_geojson(vf, source_crs=proj_meta.get("crs", "EPSG:4326"))
                    all_vector_features.extend(fc.get("features", []))
                except Exception as e:
                    logger.warning("Error reading standalone vector file %s: %s", vf, e)

        vectors_geojson_bytes = b""
        if all_vector_features:
            combined_geojson = {
                "type": "FeatureCollection",
                "features": all_vector_features,
            }
            vectors_geojson_bytes = json.dumps(combined_geojson).encode("utf-8")

        # 3. Build QC & metadata summary
        total_pts = sum(t.get("point_count", 0) for t in tiles)
        qc_summary = {
            "total_points": total_pts,
            "tiles_count": total_tiles,
            "classified_tiles": processed_tiles,
            "coordinate_system": proj_meta.get("crs", "Local / Unspecified"),
        }
        qc_summary.update(geo_stats)

        # 4. Register metadata with server
        self._prog_bar.setValue(25)
        self._status_lbl.setText("Registering project with cloud server...")
        QApplication.processEvents()

        publish_url = f"{server_url}/api/v1/projects/publish"
        form_data = {
            "project_name": proj_name,
            "description": f"Survey deliverable with {processed_tiles}/{total_tiles} tiles processed.",
            "total_tiles": total_tiles,
            "processed_tiles": processed_tiles,
            "status": "completed" if processed_tiles >= total_tiles and total_tiles > 0 else "processing",
            "qc_summary_json": json.dumps(qc_summary),
            "make_public": self._chk_public.isChecked(),
            "user_email": self._lic.user_email or "",
        }

        headers = {}
        if self._lic.access_token:
            headers["Authorization"] = f"Bearer {self._lic.access_token}"

        try:
            resp = requests.post(publish_url, data=form_data, headers=headers, timeout=20)
            resp.raise_for_status()
            res_body = resp.json()

            project_id = res_body.get("project_id")
            share_url_path = res_body.get("share_url", "")
            full_share_url = f"{server_url}{share_url_path}"
            self._share_url = full_share_url

            # 5. Upload Assets & Deliverables (supports Multi-VPS storage allocation)
            upload_url = res_body.get("upload_url") or f"/api/v1/projects/{project_id}/upload-asset"
            if not upload_url.startswith("http"):
                upload_url = f"{server_url}{upload_url}"

            # 5a. Upload Web Point Cloud
            if preview_bytes:
                self._prog_bar.setValue(40)
                self._status_lbl.setText("Uploading 3D GIS Globe point cloud...")
                QApplication.processEvents()
                requests.post(
                    upload_url,
                    data={"asset_type": "pointcloud"},
                    files={"file": ("web_preview.bin", preview_bytes, "application/octet-stream")},
                    timeout=30,
                )

            # 5b. Upload Orthophoto (Web-Optimized Image or Raw GeoTIFF)
            if ortho_webp_bytes:
                self._prog_bar.setValue(55)
                self._status_lbl.setText("Uploading web-optimized orthophoto...")
                QApplication.processEvents()
                requests.post(
                    upload_url,
                    data={"asset_type": "orthophoto"},
                    files={"file": ("orthophoto.webp", ortho_webp_bytes, "image/webp")},
                    timeout=60,
                )
            elif ortho_file and ortho_file.exists():
                self._prog_bar.setValue(55)
                self._status_lbl.setText(f"Uploading orthophoto: {ortho_file.name}...")
                QApplication.processEvents()
                with open(ortho_file, "rb") as f:
                    requests.post(
                        upload_url,
                        data={"asset_type": "orthophoto"},
                        files={"file": (ortho_file.name, f, "image/tiff")},
                        timeout=120,
                    )

            # 5c. Upload Vector Layers (with QGIS styling embedded)
            if vectors_geojson_bytes:
                self._prog_bar.setValue(65)
                self._status_lbl.setText("Uploading styled vector layers...")
                QApplication.processEvents()
                requests.post(
                    upload_url,
                    data={"asset_type": "vector"},
                    files={"file": ("vectors.geojson", vectors_geojson_bytes, "application/geo+json")},
                    timeout=60,
                )

            # 5d. Upload QGIS Project if present
            if qgis_path_str and os.path.isfile(qgis_path_str):
                qp = Path(qgis_path_str)
                self._prog_bar.setValue(72)
                self._status_lbl.setText(f"Uploading QGIS project file: {qp.name}...")
                QApplication.processEvents()
                with open(qp, "rb") as f:
                    requests.post(
                        upload_url,
                        data={"asset_type": "qgis_project"},
                        files={"file": (qp.name, f, "application/octet-stream")},
                        timeout=60,
                    )

            # 5e. Upload DTM Rasters if available
            if self._chk_dtm.isChecked() and self._pm.project_dir:
                dtm_dir = self._pm.project_dir / "dtm"
                if dtm_dir.is_dir():
                    for dtmf in dtm_dir.glob("*.tif"):
                        self._prog_bar.setValue(80)
                        self._status_lbl.setText(f"Uploading DTM: {dtmf.name}...")
                        QApplication.processEvents()
                        with open(dtmf, "rb") as f:
                            requests.post(
                                upload_url,
                                data={"asset_type": "dtm"},
                                files={"file": (dtmf.name, f, "image/tiff")},
                                timeout=60,
                            )
                        break

            # 5f. Upload DSM Rasters if available
            if self._chk_dsm.isChecked() and self._pm.project_dir:
                dsm_dir = self._pm.project_dir / "dsm"
                if dsm_dir.is_dir():
                    for dsmf in dsm_dir.glob("*.tif"):
                        self._prog_bar.setValue(88)
                        self._status_lbl.setText(f"Uploading DSM: {dsmf.name}...")
                        QApplication.processEvents()
                        with open(dsmf, "rb") as f:
                            requests.post(
                                upload_url,
                                data={"asset_type": "dsm"},
                                files={"file": (dsmf.name, f, "image/tiff")},
                                timeout=60,
                            )
                        break

            # 5g. Upload QC Report if available
            if self._chk_qc.isChecked() and self._pm.project_dir:
                qc_dir = self._pm.project_dir / "qc"
                if qc_dir.is_dir():
                    for qcf in qc_dir.glob("*.json"):
                        self._prog_bar.setValue(93)
                        self._status_lbl.setText(f"Uploading QC Report: {qcf.name}...")
                        QApplication.processEvents()
                        with open(qcf, "rb") as f:
                            requests.post(
                                upload_url,
                                data={"asset_type": "qc_report"},
                                files={"file": (qcf.name, f, "application/json")},
                                timeout=20,
                            )
                        break

            # 5h. Upload primary LAS / LAZ file if checked
            if self._chk_las.isChecked() and tiles:
                first_tile = tiles[0]
                las_path = Path(first_tile.get("file_path", ""))
                if not las_path.is_absolute() and self._pm.project_dir:
                    las_path = self._pm.project_dir / las_path
                is_ent = (self._lic.plan in ("enterprise", "pro")) or getattr(self._lic, "is_admin", False)
                max_allowed_bytes = (50 * 1024 * 1024 * 1024) if is_ent else (500 * 1024 * 1024)
                if las_path.exists() and las_path.stat().st_size <= max_allowed_bytes:
                    self._prog_bar.setValue(97)
                    self._status_lbl.setText(f"Uploading Deliverable LAS: {las_path.name}...")
                    QApplication.processEvents()
                    with open(las_path, "rb") as f:
                        requests.post(
                            upload_url,
                            data={"asset_type": "pointcloud"},
                            files={"file": (las_path.name, f, "application/octet-stream")},
                            timeout=600,
                        )

            self._prog_bar.setValue(100)
            self._status_lbl.setText("Successfully published to Cloud!")
            self._link_input.setText(full_share_url)
            self._result_group.setVisible(True)

        except Exception as e:
            logger.error("Cloud publish failed: %s", e)
            self._status_lbl.setText("Failed to publish")
            QMessageBox.critical(self, "Publish Error", f"Could not publish to cloud server:\n{e}")
        finally:
            self._publish_btn.setEnabled(True)
