"""
LiDAR Workbench — Classification Configuration Dialog.

Provides a dialog for configuring Pointcept inference parameters
before launching the background classification worker.  User settings
(paths and parameters) are persisted between sessions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_POINTCEPT_PATH,
)
from ..database import Database
from ..pointcept_worker import PointceptWorker
from ..tile_manager import TileManager
from ..licensing import get_license_manager, get_local_gpu_name

logger = logging.getLogger("lidar_workbench.gui.classification_dialog")

_SETTINGS_FILE = ".pointcept_settings.json"


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict) -> None:
    try:
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save Pointcept settings: %s", exc)


class DistributedJobMonitor(QThread):
    progress = Signal(str, float)      # status_text, percent
    completed = Signal(dict)           # final_result
    error = Signal(str)

    def __init__(self, job_id: int, total_tiles: int, licensing_mgr, parent=None):
        super().__init__(parent)
        self.job_id = job_id
        self.total_tiles = total_tiles
        self.lic = licensing_mgr
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import time
        while not self._cancelled:
            res = self.lic.get_job_status(self.job_id)
            if res.get("status") == "error":
                self.error.emit(res.get("message", "Network error while polling job status"))
                return
            
            status = res.get("status", "processing")
            done = res.get("processed_tiles", 0)
            total = res.get("total_tiles", self.total_tiles)
            pct = (done / total * 100.0) if total > 0 else 0.0
            workers = res.get("active_workers", [])
            worker_str = f" across {len(workers)} node(s)" if workers else ""

            self.progress.emit(f"Processing distributed batch: {done}/{total} tiles completed{worker_str}…", pct)

            if status == "completed" or (total > 0 and done >= total):
                self.progress.emit(f"✅ All {done} tiles processed successfully across company nodes!", 100.0)
                self.completed.emit(res)
                return

            if status == "failed":
                self.error.emit("Distributed compute job failed on remote nodes.")
                return

            time.sleep(1.5)


class ClassificationDialog(QDialog):
    """
    Dialog for configuring and launching Pointcept classification.

    The user sets the model path, config file, Pointcept root,
    and inference hyperparameters, then clicks "Start" to launch
    the worker in a background thread.  All settings are persisted
    to ``.pointcept_settings.json``.
    """

    def __init__(
        self,
        tile_manager: TileManager,
        database: Database,
        tile_ids: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tm = tile_manager
        self._db = database
        self._tile_ids = tile_ids
        self._worker: Optional[PointceptWorker] = None
        self._lic = get_license_manager()
        self._cluster_monitor: Optional[DistributedJobMonitor] = None
        self._node_rows: List[tuple[int, QCheckBox]] = []

        self.setWindowTitle("Pointcept Classification")
        self.setMinimumSize(600, 520)
        self._setup_ui()
        self._restore_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Paths ──────────────────────────────────────────────────
        paths_group = QGroupBox("Paths")
        paths_form = QFormLayout(paths_group)

        self._pointcept_edit = self._make_path_row(
            paths_form, "Pointcept Root:", DEFAULT_POINTCEPT_PATH, directory=True
        )
        self._model_edit = self._make_path_row(
            paths_form, "Model (.pth):", DEFAULT_MODEL_PATH, directory=False
        )
        self._config_edit = self._make_path_row(
            paths_form, "Config (.py):", DEFAULT_CONFIG_PATH, directory=False
        )
        py_row = QHBoxLayout()
        self._python_edit = QLineEdit()
        self._python_edit.setPlaceholderText("Auto-detect (sys.executable)")
        py_browse = QPushButton("…")
        py_browse.setFixedWidth(30)
        py_browse.clicked.connect(
            lambda: self._browse_file(self._python_edit, "Select Python Interpreter", False)
        )
        py_row.addWidget(self._python_edit)
        py_row.addWidget(py_browse)
        paths_form.addRow("Python:", py_row)

        layout.addWidget(paths_group)

        # ── Inference parameters ───────────────────────────────────
        infer_group = QGroupBox("Inference Parameters")
        infer_form = QFormLayout(infer_group)

        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setRange(0.05, 2.0)
        self._voxel_spin.setSingleStep(0.05)
        self._voxel_spin.setValue(0.15)
        self._voxel_spin.setDecimals(2)
        self._voxel_spin.setSuffix(" m")
        self._voxel_spin.setToolTip(
            "Voxel size for density normalisation (should be > line spacing, e.g. 0.15 m)"
        )
        infer_form.addRow("Voxel Size:", self._voxel_spin)

        self._smoothing_check = QCheckBox("Enable k-NN smoothing (recommended)")
        self._smoothing_check.setChecked(True)
        self._smoothing_check.setToolTip(
            "Apply edge-preserving k-NN majority voting to smooth predictions"
        )
        infer_form.addRow(self._smoothing_check)

        layout.addWidget(infer_group)

        # ── Compute Execution (Local vs Company Cluster) ───────────
        compute_group = QGroupBox("Compute Execution")
        compute_layout = QVBoxLayout(compute_group)

        local_gpu_label = f"Local GPU Only ({get_local_gpu_name()})"
        self._radio_local = QRadioButton(local_gpu_label)
        self._radio_local.setChecked(True)
        self._radio_local.toggled.connect(self._on_compute_mode_changed)

        cluster_label = "Distribute Across Company Nodes (Peer-to-Peer Cluster)"
        self._radio_cluster = QRadioButton(cluster_label)
        self._radio_cluster.toggled.connect(self._on_compute_mode_changed)

        compute_layout.addWidget(self._radio_local)
        compute_layout.addWidget(self._radio_cluster)

        # Cluster selection container
        self._cluster_widget = QWidget()
        cluster_vbox = QVBoxLayout(self._cluster_widget)
        cluster_vbox.setContentsMargins(0, 4, 0, 0)

        self._nodes_table = QTableWidget()
        self._nodes_table.setColumnCount(5)
        self._nodes_table.setHorizontalHeaderLabels(["Select", "Node Name", "Team Member", "GPU Model", "Status"])
        h = self._nodes_table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._nodes_table.setMaximumHeight(140)
        self._nodes_table.setAlternatingRowColors(True)
        cluster_vbox.addWidget(self._nodes_table)

        btn_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all_nodes)
        self._refresh_nodes_btn = QPushButton("Refresh Nodes")
        self._refresh_nodes_btn.clicked.connect(self._load_cluster_nodes)
        self._cluster_summary_label = QLabel("0 nodes selected")
        btn_row.addWidget(self._select_all_btn)
        btn_row.addWidget(self._refresh_nodes_btn)
        btn_row.addWidget(self._cluster_summary_label)
        btn_row.addStretch()
        cluster_vbox.addLayout(btn_row)

        compute_layout.addWidget(self._cluster_widget)
        self._cluster_widget.setVisible(False)

        layout.addWidget(compute_group)

        # ── Info ───────────────────────────────────────────────────
        info_label = QLabel(
            f"<b>{len(self._tile_ids)} tile(s)</b> selected for classification.\n"
            "Intensity scale is computed automatically (97th percentile).\n"
            "Noise filter is disabled (headless mode).\n"
            "A backup (.bak) is created before overwriting the LAS file."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # ── Progress ───────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)

        # ── Buttons ────────────────────────────────────────────────
        button_box = QDialogButtonBox()
        self._start_btn = QPushButton("Start Classification")
        self._start_btn.clicked.connect(self._on_start)
        button_box.addButton(self._start_btn, QDialogButtonBox.ActionRole)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        button_box.addButton(self._cancel_btn, QDialogButtonBox.RejectRole)

        close_btn = button_box.addButton(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.close)

        layout.addWidget(button_box)

    # ── settings persistence ───────────────────────────────────────

    def _restore_settings(self) -> None:
        s = _load_settings()
        if s.get("pointcept_path"):
            self._pointcept_edit.setText(s["pointcept_path"])
        if s.get("model_path"):
            self._model_edit.setText(s["model_path"])
        if s.get("config_path"):
            self._config_edit.setText(s["config_path"])
        if s.get("python_exe"):
            self._python_edit.setText(s["python_exe"])
        if "voxel_size" in s:
            self._voxel_spin.setValue(s["voxel_size"])
        if "smoothing" in s:
            self._smoothing_check.setChecked(s["smoothing"])

    def _persist_settings(self) -> None:
        _save_settings({
            "pointcept_path": self._pointcept_edit.text().strip(),
            "model_path": self._model_edit.text().strip(),
            "config_path": self._config_edit.text().strip(),
            "python_exe": self._python_edit.text().strip(),
            "voxel_size": self._voxel_spin.value(),
            "smoothing": self._smoothing_check.isChecked(),
        })

    # ── compute mode & cluster handling ────────────────────────────

    def _on_compute_mode_changed(self) -> None:
        is_cluster = self._radio_cluster.isChecked()
        self._cluster_widget.setVisible(is_cluster)
        if is_cluster:
            self._load_cluster_nodes()

    def _select_all_nodes(self) -> None:
        for _, chk in self._node_rows:
            chk.setChecked(True)
        self._update_selected_summary()

    def _update_selected_summary(self) -> None:
        count = sum(1 for _, chk in self._node_rows if chk.isChecked())
        self._cluster_summary_label.setText(f"{count}/{len(self._node_rows)} nodes selected")

    def _load_cluster_nodes(self) -> None:
        if not self._lic.is_logged_in:
            QMessageBox.information(
                self,
                "Sign In Required",
                "You must be signed into your company account to distribute jobs across company nodes.\n\n"
                "Please go to Help → License & Plan to sign in.",
            )
            self._radio_local.setChecked(True)
            return

        # Send heartbeat so our own workstation is also listed
        self._lic.send_node_heartbeat()
        nodes = self._lic.fetch_company_nodes()

        self._node_rows.clear()
        self._nodes_table.setRowCount(len(nodes))

        import platform
        this_pc = platform.node()

        for row, n in enumerate(nodes):
            nid = n.get("id")
            name = n.get("node_name", "")
            is_this = (name == this_pc)
            is_online = n.get("is_online", False)

            # Checkbox
            chk = QCheckBox()
            # Default check all online nodes
            chk.setChecked(is_online)
            chk.setEnabled(is_online)
            chk.toggled.connect(self._update_selected_summary)
            self._node_rows.append((nid, chk))

            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(8, 0, 0, 0)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(Qt.AlignCenter)
            self._nodes_table.setCellWidget(row, 0, chk_widget)

            # Node Name
            disp_name = f"{name} (This PC)" if is_this else name
            self._nodes_table.setItem(row, 1, QTableWidgetItem(disp_name))

            # Member
            member = n.get("user_email", "")
            if n.get("user_name"):
                member = f"{n['user_name']} ({member})"
            self._nodes_table.setItem(row, 2, QTableWidgetItem(member))

            # GPU
            self._nodes_table.setItem(row, 3, QTableWidgetItem(n.get("gpu_name", "NVIDIA GPU")))

            # Status
            stat = "🟢 Online" if is_online else "⚪ Offline"
            if is_online and n.get("status") == "busy":
                stat = "🟡 Busy"
            self._nodes_table.setItem(row, 4, QTableWidgetItem(stat))

        self._update_selected_summary()

    # ── browse ─────────────────────────────────────────────────────

    def _make_path_row(
        self, form: QFormLayout, label: str, default: str, directory: bool
    ) -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit(default)
        edit.setMinimumWidth(300)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        if directory:
            browse.clicked.connect(
                lambda: self._browse_file(edit, "Select Directory", True)
            )
        else:
            browse.clicked.connect(
                lambda: self._browse_file(edit, "Select File", False)
            )
        row.addWidget(edit)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _browse_file(
        self, edit: QLineEdit, title: str, directory: bool
    ) -> None:
        if directory:
            path = QFileDialog.getExistingDirectory(self, title)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title)
        if path:
            edit.setText(path)

    # ── start / cancel ─────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._radio_cluster.isChecked():
            # Distributed Mode across selected Company Nodes
            selected_node_ids = [nid for nid, chk in self._node_rows if chk.isChecked()]
            if not selected_node_ids:
                QMessageBox.warning(
                    self,
                    "No Nodes Selected",
                    "Please select at least one online company node from the table to process the batch.",
                )
                return

            self._start_btn.setEnabled(False)
            self._cancel_btn.setEnabled(True)
            self._progress_bar.setValue(0)
            self._status_label.setText(
                f"Dispatching {len(self._tile_ids)} tiles across {len(selected_node_ids)} company nodes…"
            )

            res = self._lic.dispatch_job(self._tile_ids, selected_node_ids)
            if not res.get("success"):
                self._start_btn.setEnabled(True)
                self._cancel_btn.setEnabled(False)
                QMessageBox.critical(
                    self,
                    "Dispatch Failed",
                    res.get("error", "Failed to dispatch distributed job to company nodes."),
                )
                return

            job_id = res["job_id"]
            self._cluster_monitor = DistributedJobMonitor(job_id, len(self._tile_ids), self._lic, self)
            self._cluster_monitor.progress.connect(self._on_cluster_progress)
            self._cluster_monitor.completed.connect(self._on_cluster_completed)
            self._cluster_monitor.error.connect(self._on_cluster_error)
            self._cluster_monitor.start()
            return

        # Local Mode
        pointcept_path = self._pointcept_edit.text().strip()
        model_path = self._model_edit.text().strip()
        config_path = self._config_edit.text().strip()
        python_exe = self._python_edit.text().strip() or None

        if not pointcept_path or not Path(pointcept_path).is_dir():
            QMessageBox.warning(self, "Invalid Path", "Pointcept root directory not found.")
            return
        if not model_path or not Path(model_path).is_file():
            QMessageBox.warning(self, "Invalid Path", "Model checkpoint file not found.")
            return
        if not config_path:
            QMessageBox.warning(self, "Invalid Path", "Config file path is empty.")
            return

        self._persist_settings()

        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Starting Pointcept…")

        smoothing = "yes" if self._smoothing_check.isChecked() else "no"

        from .settings_dialog import load_general_settings
        settings = load_general_settings()
        workers = settings.get("classify_workers", 1)

        self._worker = PointceptWorker(
            self._tm,
            self._db,
            self._tile_ids,
            pointcept_path=pointcept_path,
            model_path=model_path,
            config_path=config_path,
            python_exe=python_exe,
            voxel_size=self._voxel_spin.value(),
            smoothing=smoothing,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.tile_done.connect(self._on_tile_done)
        self._worker.tile_error.connect(self._on_tile_error)
        self._worker.all_done.connect(self._on_all_done)
        self._worker._workers = workers
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._cluster_monitor is not None and self._cluster_monitor.isRunning():
            self._cluster_monitor.cancel()
            self._status_label.setText("Cancelling distributed job…")
            self._start_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)

        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._status_label.setText("Cancelling…")

    # ── cluster monitor handlers ───────────────────────────────────

    def _on_cluster_progress(self, msg: str, pct: float) -> None:
        self._status_label.setText(msg)
        self._progress_bar.setValue(int(pct))

    def _on_cluster_completed(self, data: dict) -> None:
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setValue(100)
        processed = data.get("processed_tiles", len(self._tile_ids))
        self._status_label.setText(f"Done — {processed}/{len(self._tile_ids)} tiles classified across company nodes.")
        QMessageBox.information(
            self,
            "Distributed Classification Complete",
            f"Successfully processed {processed} tiles using the company compute pool!",
        )

    def _on_cluster_error(self, error: str) -> None:
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Error in distributed execution.")
        QMessageBox.critical(self, "Cluster Error", error)

    # ── worker signal handlers ─────────────────────────────────────

    def _on_progress(self, tile_id: str, step: str, pct: float) -> None:
        self._status_label.setText(step)
        self._progress_bar.setValue(int(pct))

    def _on_tile_done(self, tile_id: str) -> None:
        logger.info("Tile %s classified successfully", tile_id)

    def _on_tile_error(self, tile_id: str, error: str) -> None:
        logger.error("Tile %s failed: %s", tile_id, error)
        QMessageBox.warning(
            self,
            "Classification Error",
            f"Tile {tile_id} failed:\n\n{error[:400]}",
        )

    def _on_all_done(self, completed: List[str]) -> None:
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setValue(100)
        self._status_label.setText(
            f"Done — {len(completed)}/{len(self._tile_ids)} tile(s) classified."
        )

    def closeEvent(self, event) -> None:
        if self._cluster_monitor is not None and self._cluster_monitor.isRunning():
            self._cluster_monitor.cancel()
            self._cluster_monitor.wait(2000)
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)
