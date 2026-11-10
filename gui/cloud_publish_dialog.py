"""
LiDAR Workbench — Cloud Sync & Publish Dialog.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.parse
import urllib.error

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
)

from ..licensing import DEFAULT_SERVER_URL
from ..project_manager import ProjectManager

logger = logging.getLogger("lidar_workbench.gui.cloud_publish")


class CloudPublishDialog(QDialog):
    """Dialog allowing users to publish project deliverables and generate client share links."""

    def __init__(self, pm: ProjectManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publish Deliverables to Cloud")
        self.setMinimumWidth(560)
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
        self._server_input = QLineEdit(DEFAULT_SERVER_URL)
        server_layout.addWidget(self._server_input)
        opts_layout.addLayout(server_layout)

        self._chk_qc = QCheckBox("Upload Quality Control (QC) Summary Statistics")
        self._chk_qc.setChecked(True)
        opts_layout.addWidget(self._chk_qc)

        self._chk_dtm = QCheckBox("Upload Exported DTM Rasters (if available in dtm/)")
        self._chk_dtm.setChecked(True)
        opts_layout.addWidget(self._chk_dtm)

        self._chk_public = QCheckBox("Enable Client Share Link (viewable by stakeholders)")
        self._chk_public.setChecked(True)
        opts_layout.addWidget(self._chk_public)

        quota_notice = QLabel("💡 <i>Cloud Limit: Maximum 1.0 GB per project deliverable to preserve server bandwidth.</i>")
        quota_notice.setStyleSheet("color: #94a3b8; font-size: 11px;")
        opts_layout.addWidget(quota_notice)

        layout.addWidget(opts_group)

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
        open_btn = QPushButton("Open in Browser ↗")
        open_btn.clicked.connect(self._open_in_browser)
        link_btns.addWidget(copy_btn)
        link_btns.addWidget(open_btn)
        res_layout.addLayout(link_btns)
        layout.addWidget(self._result_group)

        # ── Bottom Action Buttons ─────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._publish_btn = QPushButton("Publish to Cloud 🚀")
        self._publish_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self._publish_btn.clicked.connect(self._on_publish)
        btn_layout.addWidget(self._publish_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _copy_link(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self._link_input.text())
        QMessageBox.information(self, "Copied", "Client share link copied to clipboard.")

    def _open_in_browser(self):
        if self._share_url:
            QDesktopServices.openUrl(QUrl(self._share_url))

    def _on_publish(self):
        server_url = self._server_input.text().strip().rstrip("/")
        if not server_url:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid server URL.")
            return

        self._publish_btn.setEnabled(False)
        self._prog_bar.setVisible(True)
        self._prog_bar.setValue(20)
        self._status_lbl.setText("Preparing project metadata...")
        QApplication.processEvents()

        proj_meta = self._pm.metadata
        proj_name = proj_meta.get("name", "Untitled Project")
        tiles = self._pm.db.get_all_tiles() if self._pm.db else []
        total_tiles = len(tiles)
        processed_tiles = sum(1 for t in tiles if t.get("classified"))

        # Build QC summary
        qc_summary = {}
        if self._chk_qc.isChecked() and self._pm.db:
            try:
                total_pts = sum(t.get("point_count", 0) for t in tiles)
                qc_summary = {
                    "total_points": total_pts,
                    "tiles_count": total_tiles,
                    "classified_tiles": processed_tiles,
                    "coordinate_system": proj_meta.get("crs", "Local / Unspecified"),
                }
            except Exception:
                pass

        # 1. Publish metadata to server
        self._prog_bar.setValue(50)
        self._status_lbl.setText("Connecting to cloud server...")
        QApplication.processEvents()

        publish_url = f"{server_url}/api/v1/projects/publish"
        form_data = {
            "project_name": proj_name,
            "description": f"Survey deliverable with {processed_tiles}/{total_tiles} tiles processed.",
            "total_tiles": str(total_tiles),
            "processed_tiles": str(processed_tiles),
            "status": "completed" if processed_tiles >= total_tiles and total_tiles > 0 else "processing",
            "qc_summary_json": json.dumps(qc_summary),
            "make_public": "true" if self._chk_public.isChecked() else "false",
        }

        try:
            encoded_data = urllib.parse.urlencode(form_data).encode("utf-8")
            req = urllib.request.Request(
                publish_url,
                data=encoded_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                res_body = json.loads(resp.read().decode("utf-8"))

            share_url_path = res_body.get("share_url", "")
            full_share_url = f"{server_url}{share_url_path}"
            self._share_url = full_share_url

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
