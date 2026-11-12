"""
LiDAR Workbench — Company Compute Nodes & Distributed P2P Pool Dialog.

Displays active company nodes (e.g. Alice's PC, Bob's Workstation, Office GPU Server),
their GPU hardware, current load status, and allows controlling the local worker.
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
    QListWidget,
    QWidget,
)

from ..licensing import get_license_manager, get_local_gpu_name

logger = logging.getLogger("lidar_workbench.gui.company_nodes")


class CompanyNodesDialog(QDialog):
    """Dialog displaying company cluster nodes and hardware compute status."""

    def __init__(self, parent: Optional[QWidget] = None, worker: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._lic = get_license_manager()
        self._worker = worker
        self.setWindowTitle("Company Compute Nodes — Distributed Pool")
        self.setMinimumSize(820, 580)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_nodes_silent)

        self._setup_ui()
        self._connect_worker_signals()
        self._refresh_nodes()
        self._timer.start(6000)  # Auto-refresh every 6 seconds

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

        if not self._lic.is_logged_in:
            self._login_btn = QPushButton("Sign In to Company Cloud…")
            self._login_btn.clicked.connect(self._open_login)
            comp_layout.addWidget(self._login_btn)

        layout.addWidget(company_group)

        # ── This Machine Card ──────────────────────────────────────
        this_pc_group = QGroupBox("This Machine (Local Node & Cluster Worker)")
        pc_vbox = QVBoxLayout(this_pc_group)

        # Top row: Hardware info
        hw_row = QHBoxLayout()
        self._local_node_label = QLabel(
            f"<b>Node Name:</b> {platform.node()}  |  "
            f"<b>GPU:</b> {get_local_gpu_name()}  |  "
            f"<b>HWID:</b> <code>{self._lic.hwid[:12]}…</code>"
        )
        hw_row.addWidget(self._local_node_label)
        hw_row.addStretch()
        pc_vbox.addLayout(hw_row)

        # Middle row: Worker status & Control buttons
        ctrl_row = QHBoxLayout()
        self._worker_status_badge = QLabel("<b>Worker Status:</b> Initializing…")
        self._worker_status_badge.setStyleSheet("font-size: 13px;")

        self._worker_toggle_btn = QPushButton("Pause Worker")
        self._worker_toggle_btn.setFixedWidth(130)
        self._worker_toggle_btn.clicked.connect(self._on_toggle_worker)

        self._stats_label = QLabel("Tiles Completed: 0")
        self._stats_label.setStyleSheet("color: #64748b; font-size: 12px;")

        ctrl_row.addWidget(self._worker_status_badge)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(self._stats_label)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self._worker_toggle_btn)
        pc_vbox.addLayout(ctrl_row)

        # Activity list
        self._activity_list = QListWidget()
        self._activity_list.setMaximumHeight(85)
        self._activity_list.setStyleSheet(
            "background: #0f172a; color: #94a3b8; font-family: monospace; font-size: 11px; border-radius: 4px;"
        )
        self._add_log_entry(f"Worker node initialized for {platform.node()}")
        pc_vbox.addWidget(self._activity_list)

        layout.addWidget(this_pc_group)

        # ── Company Nodes Table ────────────────────────────────────
        table_group = QGroupBox("Active Company Nodes (Peer-to-Peer Cluster)")
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
        self._refresh_btn = QPushButton("Refresh Now")
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

        self._update_worker_ui()

    def _connect_worker_signals(self) -> None:
        if self._worker:
            try:
                self._worker.worker_status_changed.connect(self._on_worker_status_changed)
                self._worker.task_started.connect(self._on_worker_task_started)
                self._worker.task_finished.connect(self._on_worker_task_finished)
            except Exception as e:
                logger.debug("Failed connecting worker signals: %s", e)

    def _add_log_entry(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._activity_list.addItem(f"[{ts}] {msg}")
        self._activity_list.scrollToBottom()

    def _on_worker_status_changed(self, status: str) -> None:
        self._update_worker_ui()
        self._add_log_entry(f"Status update: {status}")

    def _on_worker_task_started(self, tile_id: str, job_id: int) -> None:
        self._worker_status_badge.setText(
            f"<b>Worker Status:</b> <span style='color: #eab308;'>🟡 Processing Tile '{tile_id}' (Job #{job_id})</span>"
        )
        self._add_log_entry(f"⚡ Claimed cluster task: Tile '{tile_id}' (Job #{job_id})")

    def _on_worker_task_finished(self, tile_id: str, job_id: int) -> None:
        self._add_log_entry(f"✅ Finished cluster task: Tile '{tile_id}' (Job #{job_id})")
        if self._worker and hasattr(self._worker, "processed_count"):
            self._stats_label.setText(f"Tiles Completed: {self._worker.processed_count}")
        self._update_worker_ui()

    def _update_worker_ui(self) -> None:
        if not self._lic.is_logged_in:
            self._worker_status_badge.setText(
                "<b>Worker Status:</b> <span style='color: #94a3b8;'>⚪ Offline (Sign in required)</span>"
            )
            self._worker_toggle_btn.setEnabled(False)
            return

        self._worker_toggle_btn.setEnabled(True)
        is_paused = getattr(self._worker, "is_paused", False) if self._worker else False

        if is_paused:
            self._worker_status_badge.setText(
                "<b>Worker Status:</b> <span style='color: #94a3b8;'>⚪ Paused (Not accepting cluster tasks)</span>"
            )
            self._worker_toggle_btn.setText("Resume Worker")
        else:
            self._worker_status_badge.setText(
                "<b>Worker Status:</b> <span style='color: #22c55e;'>🟢 Active (Idle — Listening for tasks)</span>"
            )
            self._worker_toggle_btn.setText("Pause Worker")

        if self._worker and hasattr(self._worker, "processed_count"):
            self._stats_label.setText(f"Tiles Completed: {self._worker.processed_count}")

    def _on_toggle_worker(self) -> None:
        if not self._worker:
            return
        if getattr(self._worker, "is_paused", False):
            self._worker.resume()
            self._add_log_entry("Worker resumed — listening for company cluster tasks")
        else:
            self._worker.pause()
            self._add_log_entry("Worker paused by user")
        self._update_worker_ui()
        self._refresh_nodes()

    def _open_login(self) -> None:
        from .license_dialog import LicenseDialog
        dlg = LicenseDialog(self)
        dlg.exec()
        self._refresh_nodes()
        self._update_worker_ui()

    def _refresh_nodes_silent(self) -> None:
        if self.isVisible():
            self._refresh_nodes()

    def _refresh_nodes(self) -> None:
        if not self._lic.is_logged_in:
            self._node_summary_label.setText(
                "Not signed in to Company SSO. Sign in to automatically participate in the compute pool."
            )
            self._table.setRowCount(0)
            return

        # Send heartbeat to keep this node alive in the pool
        status = "offline" if (self._worker and getattr(self._worker, "is_paused", False)) else "idle"
        self._lic.send_node_heartbeat(status=status)

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
