"""
LiDAR Workbench — Company Compute Nodes & Distributed P2P Pool Dialog.

Displays active company nodes (e.g. Alice's PC, Bob's Workstation, Office GPU Server),
their GPU hardware, current load status, and allows configuring the local node worker.
"""

from __future__ import annotations

import datetime
import logging
import platform
from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QMessageBox,
)

from ..licensing import get_license_manager, get_local_gpu_name

logger = logging.getLogger("lidar_workbench.gui.company_nodes")


class CompanyNodesDialog(QDialog):
    """Dialog displaying company cluster nodes and hardware compute status."""

    def __init__(self, parent: Optional[QDialog] = None) -> None:
        super().__init__(parent)
        self._lic = get_license_manager()
        self.setWindowTitle("🏢 Company Compute Nodes — Distributed Pool")
        self.setMinimumSize(780, 520)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_nodes_silent)

        self._setup_ui()
        self._refresh_nodes()
        self._timer.start(8000)  # Auto-refresh every 8 seconds

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Header / Company Info ──────────────────────────────────
        company_group = QGroupBox("Company Pool Overview")
        comp_layout = QHBoxLayout(company_group)

        self._comp_name_label = QLabel(f"<b>Company:</b> {self._lic.company_name}")
        self._user_label = QLabel(f"<b>User:</b> {self._lic.user_email or 'Not logged in'}")
        self._plan_label = QLabel(f"<b>Plan:</b> {self._lic.plan.upper()}")

        comp_layout.addWidget(self._comp_name_label)
        comp_layout.addWidget(self._user_label)
        comp_layout.addWidget(self._plan_label)
        layout.addWidget(company_group)

        # ── This Machine Card ──────────────────────────────────────
        this_pc_group = QGroupBox("This Machine (Local Node)")
        pc_layout = QHBoxLayout(this_pc_group)

        self._local_node_label = QLabel(
            f"<b>Node:</b> {platform.node()}  |  "
            f"<b>GPU:</b> {get_local_gpu_name()}  |  "
            f"<b>HWID:</b> <code>{self._lic.hwid[:12]}…</code>"
        )
        self._share_gpu_check = QCheckBox("Share GPU in Company Pool (Accept batch tasks)")
        self._share_gpu_check.setChecked(True)
        self._share_gpu_check.toggled.connect(self._on_share_toggled)

        pc_layout.addWidget(self._local_node_label)
        pc_layout.addStretch()
        pc_layout.addWidget(self._share_gpu_check)
        layout.addWidget(this_pc_group)

        # ── Company Nodes Table ────────────────────────────────────
        table_group = QGroupBox("Active Company Nodes")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "Status", "Node Name", "Team Member", "GPU Model", "OS", "Last Ping"
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)

        table_layout.addWidget(self._table)

        # ── Action bar under table ─────────────────────────────────
        action_row = QHBoxLayout()
        self._node_summary_label = QLabel("Loading company cluster…")
        self._refresh_btn = QPushButton("🔄 Refresh Now")
        self._refresh_btn.clicked.connect(self._refresh_nodes)
        
        action_row.addWidget(self._node_summary_label)
        action_row.addStretch()
        action_row.addWidget(self._refresh_btn)
        table_layout.addLayout(action_row)

        layout.addWidget(table_group)

        # ── Bottom Buttons ─────────────────────────────────────────
        bottom_row = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        bottom_row.addStretch()
        bottom_row.addWidget(close_btn)
        layout.addLayout(bottom_row)

    def _on_share_toggled(self, checked: bool) -> None:
        status = "idle" if checked else "offline"
        self._lic.send_node_heartbeat(status=status)
        self._refresh_nodes()

    def _refresh_nodes_silent(self) -> None:
        if self.isVisible():
            self._refresh_nodes()

    def _refresh_nodes(self) -> None:
        if not self._lic.is_logged_in:
            self._node_summary_label.setText(
                "⚠️ Not signed in to Company SSO. Sign in under Help → License & Plan to join company pool."
            )
            self._table.setRowCount(0)
            return

        # Send heartbeat first to ensure this node is current
        self._lic.send_node_heartbeat(status="idle" if self._share_gpu_check.isChecked() else "offline")

        nodes = self._lic.fetch_company_nodes()
        self._table.setRowCount(len(nodes))

        online_count = 0
        total_gpus = []

        for row, node in enumerate(nodes):
            is_online = node.get("is_online", False)
            status_text = node.get("status", "idle")

            # Status Badge
            if is_online:
                online_count += 1
                if status_text == "busy":
                    status_item = QTableWidgetItem("🟡 Busy")
                else:
                    status_item = QTableWidgetItem("🟢 Online")
            else:
                status_item = QTableWidgetItem("⚪ Offline")
            status_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, status_item)

            # Node Name
            is_this_pc = (node.get("node_name") == platform.node())
            name_display = f"{node.get('node_name')} (This PC)" if is_this_pc else node.get("node_name", "")
            self._table.setItem(row, 1, QTableWidgetItem(name_display))

            # Team member
            member = node.get("user_email", "")
            if node.get("user_name"):
                member = f"{node.get('user_name')} ({member})"
            self._table.setItem(row, 2, QTableWidgetItem(member))

            # GPU
            gpu_str = node.get("gpu_name", "NVIDIA RTX GPU")
            self._table.setItem(row, 3, QTableWidgetItem(gpu_str))
            if is_online:
                total_gpus.append(gpu_str)

            # OS
            self._table.setItem(row, 4, QTableWidgetItem(node.get("os_name") or "Windows"))

            # Last Ping
            last_seen = node.get("last_seen", "")
            if last_seen:
                try:
                    dt = datetime.datetime.fromisoformat(last_seen)
                    ago = int((datetime.datetime.utcnow() - dt).total_seconds())
                    ping_str = f"{ago}s ago" if ago < 60 else f"{ago//60}m ago"
                except Exception:
                    ping_str = "Just now"
            else:
                ping_str = "-"
            self._table.setItem(row, 5, QTableWidgetItem(ping_str))

        self._node_summary_label.setText(
            f"<b>{online_count}/{len(nodes)} Node(s) Online</b> across company network  |  "
            f"Active GPUs: {', '.join(set(total_gpus)) if total_gpus else 'None'}"
        )

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
