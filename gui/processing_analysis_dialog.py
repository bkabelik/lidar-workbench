"""
LiDAR Workbench — Processing Time Analysis Dialog.

Provides filtering, aggregation, and statistics for per-tile per-step
processing times recorded in the ``processing_times`` database table.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..database import Database

logger = logging.getLogger("lidar_workbench.gui.processing_analysis_dialog")

# ── helpers ────────────────────────────────────────────────────────────

def _fmt_seconds(sec: float) -> str:
    """Format seconds as human-readable string."""
    if sec < 0.001:
        return "0.00 s"
    if sec < 1.0:
        return f"{sec * 1000:.0f} ms"
    if sec < 60:
        return f"{sec:.2f} s"
    if sec < 3600:
        m, s = divmod(sec, 60)
        return f"{int(m)}m {s:.1f}s"
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def _fmt_number(n: float) -> str:
    """Format a number with thousands separator."""
    if isinstance(n, int) or n == int(n):
        return f"{int(n):,}"
    return f"{n:,.2f}"


# ── colours ────────────────────────────────────────────────────────────

_STEP_COLORS = {
    "filter":    QColor("#4CAF50"),  # green
    "pointcept": QColor("#2196F3"),  # blue
    "ground":    QColor("#FF9800"),  # orange
}


# ── main dialog ────────────────────────────────────────────────────────

class ProcessingAnalysisDialog(QDialog):
    """
    Dialog for analysing processing times across tiles.

    Features:
        - Filter by step type, batch, tile status, and date range.
        - Summary statistics (mean, median, min, max, std, percentiles).
        - Per-record table with sorting.
        - Group-by: none / by step / by batch.
        - Export to CSV.
    """

    def __init__(self, db: Database, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db = db
        self._all_records: List[Dict[str, Any]] = []
        self._group_by: str = "none"  # none | step | batch

        self.setWindowTitle("Processing Time Analysis")
        self.setMinimumSize(960, 640)
        self.resize(1100, 750)

        self._setup_ui()
        self._load_batches()
        self._refresh()

    # ── UI construction ─────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── filter bar ──────────────────────────────────────────
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)

        # Step filter
        step_layout = QVBoxLayout()
        step_layout.addWidget(QLabel("Step:"))
        self._step_combo = QComboBox()
        self._step_combo.addItem("All Steps", None)
        self._step_combo.addItem("Filter", "filter")
        self._step_combo.addItem("Pointcept", "pointcept")
        self._step_combo.addItem("Ground", "ground")
        self._step_combo.currentIndexChanged.connect(self._on_filter_changed)
        step_layout.addWidget(self._step_combo)
        filter_layout.addLayout(step_layout)

        # Batch filter
        batch_layout = QVBoxLayout()
        batch_layout.addWidget(QLabel("Batch:"))
        self._batch_combo = QComboBox()
        self._batch_combo.addItem("All Batches", None)
        self._batch_combo.currentIndexChanged.connect(self._on_filter_changed)
        batch_layout.addWidget(self._batch_combo)
        filter_layout.addLayout(batch_layout)

        # Date range quick-select
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Period:"))
        self._date_combo = QComboBox()
        self._date_combo.addItem("All Time", "all")
        self._date_combo.addItem("Last Hour", "1h")
        self._date_combo.addItem("Today", "today")
        self._date_combo.addItem("Last 7 Days", "7d")
        self._date_combo.addItem("Last 30 Days", "30d")
        self._date_combo.currentIndexChanged.connect(self._on_filter_changed)
        date_layout.addWidget(self._date_combo)
        filter_layout.addLayout(date_layout)

        # Group-by
        group_layout = QVBoxLayout()
        group_layout.addWidget(QLabel("Group By:"))
        self._group_combo = QComboBox()
        self._group_combo.addItem("None (Flat)", "none")
        self._group_combo.addItem("By Step", "step")
        self._group_combo.addItem("By Batch", "batch")
        self._group_combo.currentIndexChanged.connect(self._on_group_changed)
        group_layout.addWidget(self._group_combo)
        filter_layout.addLayout(group_layout)

        # Export button
        export_layout = QVBoxLayout()
        export_layout.addWidget(QLabel(""))
        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.clicked.connect(self._on_export_csv)
        export_layout.addWidget(self._export_btn)
        filter_layout.addLayout(export_layout)

        filter_layout.addStretch()
        layout.addWidget(filter_group)

        # ── statistics summary ───────────────────────────────────
        stats_group = QGroupBox("Summary Statistics")
        stats_layout = QHBoxLayout(stats_group)
        self._stat_labels: Dict[str, QLabel] = {}
        for key, label_text in [
            ("count", "Count"), ("total", "Total Time"),
            ("mean", "Mean"), ("median", "Median"),
            ("min", "Min"), ("max", "Max"),
            ("std", "Std Dev"), ("p25", "P25"), ("p75", "P75"),
        ]:
            card = QVBoxLayout()
            val_lbl = QLabel("—")
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
            key_lbl = QLabel(label_text)
            key_lbl.setAlignment(Qt.AlignCenter)
            key_lbl.setStyleSheet("color: #888;")
            card.addWidget(val_lbl)
            card.addWidget(key_lbl)
            self._stat_labels[key] = val_lbl
            stats_layout.addLayout(card)
        layout.addWidget(stats_group)

        # ── table ────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, stretch=1)

        # ── close button ─────────────────────────────────────────
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ── data loading ────────────────────────────────────────────────

    def _load_batches(self) -> None:
        """Populate the batch combo box."""
        batches = self._db.get_distinct_batches()
        # Keep "All Batches" at index 0
        for b in batches:
            bid = b.get("batch_id", "")
            if not bid:
                continue
            step_count = b.get("tile_count", 0)
            steps = b.get("steps", "")
            ts = b.get("first_ts", "")[:16] if b.get("first_ts") else ""
            label = f"{bid} ({step_count} tiles, {steps}) [{ts}]"
            self._batch_combo.addItem(label, bid)

    def _refresh(self) -> None:
        """Reload data from DB and update the display."""
        step = self._step_combo.currentData()
        batch_id = self._batch_combo.currentData()
        since_ts = self._date_since()

        self._all_records = self._db.get_processing_times(
            step=step,
            batch_id=batch_id,
            since_timestamp=since_ts,
        )

        stats = self._db.get_processing_stats(
            step=step,
            batch_ids=[batch_id] if batch_id else None,
        )

        self._update_stats(stats)
        self._update_table()

    def _date_since(self) -> Optional[str]:
        """Convert the date combo selection into an ISO timestamp."""
        period = self._date_combo.currentData()
        now = datetime.now()
        if period == "1h":
            return (now - timedelta(hours=1)).isoformat(sep=" ")
        elif period == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(sep=" ")
        elif period == "7d":
            return (now - timedelta(days=7)).isoformat(sep=" ")
        elif period == "30d":
            return (now - timedelta(days=30)).isoformat(sep=" ")
        return None  # "all"

    # ── stats display ───────────────────────────────────────────────

    def _update_stats(self, stats: Dict[str, Any]) -> None:
        """Update the summary statistics card labels."""
        self._stat_labels["count"].setText(str(stats["count"]))
        self._stat_labels["total"].setText(_fmt_seconds(stats["total_seconds"]))
        self._stat_labels["mean"].setText(_fmt_seconds(stats["mean"]))
        self._stat_labels["median"].setText(_fmt_seconds(stats["median"]))
        self._stat_labels["min"].setText(_fmt_seconds(stats["min"]))
        self._stat_labels["max"].setText(_fmt_seconds(stats["max"]))
        self._stat_labels["std"].setText(_fmt_seconds(stats["std"]))
        self._stat_labels["p25"].setText(_fmt_seconds(stats["p25"]))
        self._stat_labels["p75"].setText(_fmt_seconds(stats["p75"]))

    # ── table display ───────────────────────────────────────────────

    def _update_table(self) -> None:
        """Rebuild the table based on current group_by and records."""
        group = self._group_combo.currentData() or "none"

        if group == "none":
            self._build_flat_table()
        elif group == "step":
            self._build_grouped_table("step")
        elif group == "batch":
            self._build_grouped_table("batch")

    def _build_flat_table(self) -> None:
        """Show every record as a row."""
        columns = ["Tile ID", "Step", "Duration", "Points", "Batch", "Timestamp"]
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(self._all_records))

        for row_idx, rec in enumerate(self._all_records):
            self._set_cell(row_idx, 0, rec.get("tile_id", ""))
            self._set_cell(row_idx, 1, rec.get("step", ""),
                          color=_STEP_COLORS.get(rec.get("step", "")))
            self._set_cell(row_idx, 2, _fmt_seconds(rec.get("duration_seconds", 0)),
                          align_right=True)
            self._set_cell(row_idx, 3, _fmt_number(rec.get("point_count") or 0),
                          align_right=True)
            self._set_cell(row_idx, 4, (rec.get("batch_id") or "")[:30])
            ts = rec.get("timestamp", "")
            self._set_cell(row_idx, 5, ts[:19] if ts else "")

        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _build_grouped_table(self, group_key: str) -> None:
        """Group records by *group_key* and show aggregate rows."""
        import statistics
        from collections import defaultdict

        groups: Dict[str, list] = defaultdict(list)
        for rec in self._all_records:
            key = rec.get(group_key) or "(unknown)"
            groups[key].append(rec.get("duration_seconds", 0))

        columns = ["Group", "Count", "Total", "Mean", "Median", "Min", "Max", "Std Dev"]
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(groups))

        for row_idx, (key, durations) in enumerate(sorted(groups.items())):
            d = durations
            n = len(d)
            self._set_cell(row_idx, 0, str(key),
                          color=_STEP_COLORS.get(key) if group_key == "step" else None,
                          bold=True)
            self._set_cell(row_idx, 1, str(n), align_right=True)
            self._set_cell(row_idx, 2, _fmt_seconds(sum(d)), align_right=True)
            self._set_cell(row_idx, 3, _fmt_seconds(statistics.mean(d)) if n else "—",
                          align_right=True)
            self._set_cell(row_idx, 4, _fmt_seconds(statistics.median(d)) if n else "—",
                          align_right=True)
            self._set_cell(row_idx, 5, _fmt_seconds(min(d)) if n else "—",
                          align_right=True)
            self._set_cell(row_idx, 6, _fmt_seconds(max(d)) if n else "—",
                          align_right=True)
            self._set_cell(row_idx, 7,
                          _fmt_seconds(statistics.stdev(d)) if n >= 2 else "—",
                          align_right=True)

        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _set_cell(
        self, row: int, col: int, text: str,
        color: Optional[QColor] = None,
        bold: bool = False,
        align_right: bool = False,
    ) -> None:
        """Helper: set a table cell with optional formatting."""
        item = QTableWidgetItem(text)
        if color:
            item.setForeground(color)
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        if align_right:
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._table.setItem(row, col, item)

    # ── slots ───────────────────────────────────────────────────────

    def _on_filter_changed(self) -> None:
        self._refresh()

    def _on_group_changed(self) -> None:
        self._update_table()

    def _on_export_csv(self) -> None:
        """Export current table contents to a CSV file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Processing Times", "processing_times.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                headers = [
                    self._table.horizontalHeaderItem(c).text()
                    for c in range(self._table.columnCount())
                ]
                writer.writerow(headers)
                # Rows
                for r in range(self._table.rowCount()):
                    row_data = [
                        self._table.item(r, c).text() if self._table.item(r, c) else ""
                        for c in range(self._table.columnCount())
                    ]
                    writer.writerow(row_data)

            QMessageBox.information(
                self, "Export Complete",
                f"Exported {self._table.rowCount()} rows to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Failed", f"Could not write CSV:\n{exc}",
            )
