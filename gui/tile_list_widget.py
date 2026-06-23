"""
LiDAR Workbench — Tile List Widget.

Displays all tiles in the project grouped by processing status, with
coloured status indicators, point counts, and context-menu actions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import TileStatus, get_class_name

logger = logging.getLogger("lidar_workbench.gui.tile_list")

# Status → colour mapping for the status dot icons
_STATUS_COLORS: Dict[str, QColor] = {
    TileStatus.IMPORTED:   QColor("#3498db"),  # blue
    TileStatus.FILTERED:   QColor("#f39c12"),  # orange
    TileStatus.CLASSIFIED: QColor("#2ecc71"),  # green
    TileStatus.EDITED:     QColor("#9b59b6"),  # purple
    TileStatus.ERROR:      QColor("#e74c3c"),  # red
}


def _make_status_icon(color: QColor, size: int = 12) -> QIcon:
    """Generate a filled-circle icon of *size* pixels in *color*."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(color)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return QIcon(pixmap)


class TileListWidget(QWidget):
    """
    Tree widget showing tiles grouped by processing status.

    Columns:
        - Status (icon)
        - Tile ID
        - Points
        - Last Modified

    Signals:
        tile_selected(tile_id: str):
            Emitted when the user clicks a tile row.
        tile_visibility_changed(tile_id: str, visible: bool):
            Emitted when the checkbox for a tile is toggled.
        open_requested(tile_id: str):
            Emitted for the "Open in Multi-View" context action.
        filter_requested(tile_ids: list[str]):
            Emitted for the "Noise Filter" context action.
        classify_requested(tile_ids: list[str]):
            Emitted for the "Classify" context action.
        delete_requested(tile_ids: list[str]):
            Emitted for the "Delete" context action.
    """

    tile_selected = Signal(str)
    tile_visibility_changed = Signal(str, bool)
    open_requested = Signal(str)
    filter_requested = Signal(list)
    classify_requested = Signal(list)
    export_requested = Signal(list)
    delete_requested = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tiles: Dict[str, Dict[str, Any]] = {}  # tile_id → db row
        self._status_groups: Dict[str, QTreeWidgetItem] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["", "Tile", "Points", "Modified"])
        self._tree.setColumnCount(4)
        self._tree.setRootIsDecorated(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree.setAlternatingRowColors(True)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemChanged.connect(self._on_item_changed)

        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        layout.addWidget(self._tree)

    # ── public API ─────────────────────────────────────────────────

    def set_tiles(self, tiles: List[Dict[str, Any]]) -> None:
        """
        Replace the entire tile list.

        Args:
            tiles: List of tile dicts as returned by :meth:`Database.get_all_tiles`.
        """
        self._tree.clear()
        self._status_groups.clear()
        self._tiles = {t["id"]: t for t in tiles}

        # Create group items for each status, respecting the order in TileStatus.ALL
        for status in TileStatus.ALL:
            group = QTreeWidgetItem(self._tree)
            group.setText(0, "")
            group.setText(1, status)
            group.setIcon(0, _make_status_icon(_STATUS_COLORS.get(status, QColor("#999"))))
            group.setFlags(group.flags() | Qt.ItemIsAutoTristate)
            font = group.font(1)
            font.setBold(True)
            group.setFont(1, font)
            self._status_groups[status] = group

        # Populate tiles under their status groups
        for tile in tiles:
            status = tile.get("status", TileStatus.IMPORTED)
            group = self._status_groups.get(status)
            if group is None:
                continue

            item = QTreeWidgetItem(group)
            item.setData(0, Qt.UserRole, tile["id"])
            item.setIcon(0, _make_status_icon(_STATUS_COLORS.get(status, QColor("#999")), 10))
            item.setText(1, tile["id"])
            item.setText(2, f"{tile.get('point_count', 0):,}")
            item.setText(3, tile.get("last_modified", ""))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(1, Qt.Checked)

        self._tree.expandAll()

    def get_selected_tile_ids(self) -> List[str]:
        """Return tile IDs of all currently selected rows."""
        ids: List[str] = []
        for item in self._tree.selectedItems():
            tid = item.data(0, Qt.UserRole)
            if tid:
                ids.append(tid)
        return ids

    def update_tile_status(self, tile_id: str, new_status: str) -> None:
        """Move a tile to a different status group in the tree."""
        old_item = self._find_tile_item(tile_id)
        if old_item is None:
            return

        # Remove from old group
        old_parent = old_item.parent()
        if old_parent:
            old_parent.removeChild(old_item)

        # Add to new group
        new_group = self._status_groups.get(new_status)
        if new_group is None:
            return

        item = QTreeWidgetItem(new_group)
        item.setData(0, Qt.UserRole, tile_id)
        item.setIcon(0, _make_status_icon(_STATUS_COLORS.get(new_status, QColor("#999")), 10))
        item.setText(1, tile_id)
        if tile_id in self._tiles:
            item.setText(2, f"{self._tiles[tile_id].get('point_count', 0):,}")
            item.setText(3, self._tiles[tile_id].get("last_modified", ""))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(1, Qt.Checked)

        # Update internal dict
        if tile_id in self._tiles:
            self._tiles[tile_id]["status"] = new_status

    # ── signal handlers ────────────────────────────────────────────

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        tid = item.data(0, Qt.UserRole)
        if tid:
            self.tile_selected.emit(tid)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        tid = item.data(0, Qt.UserRole)
        if tid:
            self.open_requested.emit(tid)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 1:
            return
        tid = item.data(0, Qt.UserRole)
        if tid:
            visible = item.checkState(1) == Qt.Checked
            self.tile_visibility_changed.emit(tid, visible)

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return

        tid = item.data(0, Qt.UserRole)
        selected = self.get_selected_tile_ids()

        menu = QMenu(self)
        if tid:
            open_action = menu.addAction("Open in Multi-View")
            open_action.triggered.connect(lambda: self.open_requested.emit(tid))

        menu.addSeparator()

        if selected:
            filter_action = menu.addAction("Noise Filter…")
            filter_action.triggered.connect(lambda: self.filter_requested.emit(selected))

            classify_action = menu.addAction("Classify (Pointcept)…")
            classify_action.triggered.connect(lambda: self.classify_requested.emit(selected))

            menu.addSeparator()

            export_action = menu.addAction("Export Raster…")
            export_action.triggered.connect(lambda: self.export_requested.emit(selected))

            menu.addSeparator()

            delete_action = menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self.delete_requested.emit(selected))

        menu.exec(self._tree.viewport().mapToGlobal(pos))

    # ── helpers ────────────────────────────────────────────────────

    def select_next_tile(self) -> None:
        """Select and open the next tile in the tree."""
        all_items = self._all_tile_items()
        current = self._current_tile_index()
        if all_items and current >= 0:
            next_idx = (current + 1) % len(all_items)
            self._select_and_open(all_items[next_idx])

    def select_previous_tile(self) -> None:
        """Select and open the previous tile in the tree."""
        all_items = self._all_tile_items()
        current = self._current_tile_index()
        if all_items and current >= 0:
            prev_idx = (current - 1) % len(all_items)
            self._select_and_open(all_items[prev_idx])

    def _all_tile_items(self) -> list[QTreeWidgetItem]:
        """Return all tile items in tree order."""
        items = []
        for i in range(self._tree.topLevelItemCount()):
            group = self._tree.topLevelItem(i)
            for j in range(group.childCount()):
                items.append(group.child(j))
        return items

    def _current_tile_index(self) -> int:
        selected = self._tree.selectedItems()
        if not selected:
            return -1
        tid = selected[0].data(0, Qt.UserRole)
        for i, item in enumerate(self._all_tile_items()):
            if item.data(0, Qt.UserRole) == tid:
                return i
        return -1

    def _select_and_open(self, item: QTreeWidgetItem) -> None:
        tid = item.data(0, Qt.UserRole)
        if tid:
            self._tree.setCurrentItem(item)
            self.tile_selected.emit(tid)
            self.open_requested.emit(tid)

    def _find_tile_item(self, tile_id: str) -> Optional[QTreeWidgetItem]:
        """Walk the tree to find the item with *tile_id*."""
        for i in range(self._tree.topLevelItemCount()):
            group = self._tree.topLevelItem(i)
            for j in range(group.childCount()):
                child = group.child(j)
                if child.data(0, Qt.UserRole) == tile_id:
                    return child
        return None
