"""
Cross-platform UI layout & state persistence (Windows & Linux).

Automatically saves and restores:
- Window geometry & maximized state (`QMainWindow`, `QDialog`)
- Main window toolbar/dock state (`QMainWindow.saveState`)
- All `QSplitter` pane proportions
- All `QTableView` / `QTreeView` header column widths & sort order

Stored in `~/.lidar_workbench/ui_state.ini` via `QSettings(..., IniFormat)`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QByteArray, QEvent, QObject, QSettings, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QTableView,
    QTreeView,
    QWidget,
)

logger = logging.getLogger("lidar_workbench.gui.ui_state")

_STATE_PATH = Path.home() / ".lidar_workbench" / "ui_state.ini"
_SKIP_CLASSES = (QMessageBox, QProgressDialog)


def _get_settings(path: Optional[Path] = None) -> QSettings:
    target = path or _STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return QSettings(str(target), QSettings.IniFormat)


def _widget_key(widget: QWidget, key: Optional[str] = None) -> str:
    if key:
        return key
    obj_name = widget.objectName().strip()
    cls_name = widget.__class__.__name__
    return f"{cls_name}_{obj_name}" if obj_name else cls_name


def _as_bytearray(val) -> Optional[QByteArray]:
    if isinstance(val, QByteArray) and not val.isEmpty():
        return val
    if isinstance(val, (bytes, bytearray)) and len(val) > 0:
        return QByteArray(val)
    return None


def save_widget_state(root: QWidget, key: Optional[str] = None, settings_path: Optional[Path] = None) -> None:
    """Persist geometry, splitters, and table/tree header states under `key`."""
    if isinstance(root, _SKIP_CLASSES):
        return
    wkey = _widget_key(root, key)
    qs = _get_settings(settings_path)
    qs.beginGroup(wkey)
    try:
        qs.setValue("geometry", root.saveGeometry())
        if isinstance(root, QMainWindow):
            qs.setValue("window_state", root.saveState())

        for idx, sp in enumerate(root.findChildren(QSplitter)):
            skey = sp.objectName().strip() or f"splitter_{idx}"
            qs.setValue(f"splitters/{skey}", sp.saveState())

        for idx, tv in enumerate(root.findChildren(QTableView)):
            hdr = tv.horizontalHeader()
            if hdr is not None and hdr.count() > 0:
                hkey = tv.objectName().strip() or f"table_{idx}"
                qs.setValue(f"headers/{hkey}", hdr.saveState())

        for idx, trv in enumerate(root.findChildren(QTreeView)):
            hdr = trv.header()
            if hdr is not None and hdr.count() > 0:
                hkey = trv.objectName().strip() or f"tree_{idx}"
                qs.setValue(f"headers/{hkey}", hdr.saveState())
    finally:
        qs.endGroup()
        qs.sync()


def _restore_children(qs: QSettings, root: QWidget) -> None:
    """Restore splitter and table/tree header states from an open QSettings group."""
    for idx, sp in enumerate(root.findChildren(QSplitter)):
        skey = sp.objectName().strip() or f"splitter_{idx}"
        sdata = _as_bytearray(qs.value(f"splitters/{skey}"))
        if sdata is not None:
            sp.restoreState(sdata)

    for idx, tv in enumerate(root.findChildren(QTableView)):
        hdr = tv.horizontalHeader()
        if hdr is not None and hdr.count() > 0:
            hkey = tv.objectName().strip() or f"table_{idx}"
            hdata = _as_bytearray(qs.value(f"headers/{hkey}"))
            if hdata is not None:
                hdr.restoreState(hdata)

    for idx, trv in enumerate(root.findChildren(QTreeView)):
        hdr = trv.header()
        if hdr is not None and hdr.count() > 0:
            hkey = trv.objectName().strip() or f"tree_{idx}"
            hdata = _as_bytearray(qs.value(f"headers/{hkey}"))
            if hdata is not None:
                hdr.restoreState(hdata)


def restore_widget_state(root: QWidget, key: Optional[str] = None, settings_path: Optional[Path] = None) -> None:
    """Restore geometry, splitters, and table/tree header states from disk."""
    if isinstance(root, _SKIP_CLASSES):
        return
    wkey = _widget_key(root, key)
    qs = _get_settings(settings_path)
    qs.beginGroup(wkey)
    try:
        geom = _as_bytearray(qs.value("geometry"))
        if geom is not None:
            root.restoreGeometry(geom)

        if isinstance(root, QMainWindow):
            wstate = _as_bytearray(qs.value("window_state"))
            if wstate is not None:
                root.restoreState(wstate)

        _restore_children(qs, root)
    finally:
        qs.endGroup()


class _GlobalUIStateFilter(QObject):
    """App-wide event filter that restores state on first Show and saves on Close/Hide."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if isinstance(obj, (QMainWindow, QDialog)) and not isinstance(obj, _SKIP_CLASSES):
            etype = event.type()
            if etype == QEvent.Show and not obj.property("_ui_state_restored"):
                obj.setProperty("_ui_state_restored", True)
                try:
                    restore_widget_state(obj)
                    # Re-apply splitters/headers after initial layout pass
                    QTimer.singleShot(0, lambda w=obj: _restore_children_only(w))
                except Exception as exc:
                    logger.debug("UI state restore failed for %s: %s", obj.__class__.__name__, exc)
            elif etype in (QEvent.Close, QEvent.Hide) and obj.property("_ui_state_restored"):
                # Avoid saving minimized/transient hide if closing app
                try:
                    save_widget_state(obj)
                except Exception as exc:
                    logger.debug("UI state save failed for %s: %s", obj.__class__.__name__, exc)
        return super().eventFilter(obj, event)


def _restore_children_only(root: QWidget) -> None:
    if root is None or isinstance(root, _SKIP_CLASSES):
        return
    try:
        wkey = _widget_key(root)
        qs = _get_settings()
        qs.beginGroup(wkey)
        try:
            _restore_children(qs, root)
        finally:
            qs.endGroup()
    except Exception:
        pass


_FILTER_INSTANCE: Optional[_GlobalUIStateFilter] = None


def install_global_ui_state_persistence(app: Optional[QApplication]) -> None:
    """Install app-wide UI state persistence for all windows, splitters, and table headers."""
    global _FILTER_INSTANCE
    if app is None or _FILTER_INSTANCE is not None:
        return
    _FILTER_INSTANCE = _GlobalUIStateFilter(app)
    app.installEventFilter(_FILTER_INSTANCE)
