"""
LiDAR Workbench — Main Window.

The central QMainWindow that houses the three-panel layout (tile list,
multi-view container, properties panel), menu bar, and toolbar.
"""

from __future__ import annotations

import json
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QDragEnterEvent, QDropEvent, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    APP_DESCRIPTION, APP_NAME, APP_ORG, APP_VERSION, APP_WEBSITE,
    ASPRS_CLASS_COLORS, ASPRS_CLASS_NAMES,
    DEFAULT_PROFILE_WIDTH_M, QCStatus, TileStatus,
)
from ..database import Database
from ..import_wizard import ImportWizard
from ..manual_edit import ManualEditor
from ..project_manager import ProjectManager
from ..tile_manager import TileManager
from .classification_dialog import ClassificationDialog
from .export_dialog import ExportDialog
from .filter_dialog import FilterDialog
from .ground_control_dialog import GroundControlDialog
from .multi_view_widget import MultiViewWidget
from .preview_dialog import PreviewDialog
from .processing_analysis_dialog import ProcessingAnalysisDialog
from .properties_panel import PropertiesPanel
from .settings_dialog import SettingsDialog, load_shortcuts
from .tile_list_widget import TileListWidget

logger = logging.getLogger("lidar_workbench.gui.main_window")


class MainWindow(QMainWindow):
    """
    The application main window.

    Layout (left → right):
        - **Left panel**: :class:`TileListWidget` — layer / tile manager.
        - **Center panel**: :class:`QStackedWidget` — placeholder for
          :class:`MultiViewWidget` (Phase 2+).
        - **Right panel**: :class:`QWidget` — placeholder for
          :class:`PropertiesPanel` (Phase 4+).

    The window accepts drag-and-drop of folders containing LAS/LAZ files.
    """

    def __init__(
        self,
        project_manager: ProjectManager,
        tile_manager: TileManager,
        database: Database,
    ) -> None:
        """
        Args:
            project_manager: Initialised :class:`ProjectManager`.
            tile_manager:    Initialised :class:`TileManager`.
            database:        Initialised :class:`Database`.
        """
        super().__init__()
        self._pm = project_manager
        self._tm = tile_manager
        self._db = database
        self._editor = ManualEditor(tile_manager)
        self._registered_shortcuts: list = []

        # Cached DTM reference line (preserved across classification edits)
        self._dtm_ref_distances: Optional[np.ndarray] = None
        self._dtm_ref_elevations: Optional[np.ndarray] = None

        # Class visibility filter (indexed by ASPRS class code, all visible by default)
        self.class_visibility = np.ones(256, dtype=bool)

        # Cached profile line for width changes
        self._profile_start: Optional[tuple] = None
        self._profile_end: Optional[tuple] = None
        self._profile_width: float = DEFAULT_PROFILE_WIDTH_M

        # Keep preview dialog alive (prevent GC of Python wrapper)
        self._preview_dlg: Optional[PreviewDialog] = None

        # Keep ground-control dialog alive (non-modal, prevent GC of wrapper)
        self._ground_control_dlg: Optional[GroundControlDialog] = None

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 700)
        self.setAcceptDrops(True)

        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()

        # Initialise tile list with data (empty until a project is opened)
        self._refresh_tile_list()

        # Apply user-configured shortcuts (menu actions only)
        self._apply_shortcuts(load_shortcuts())
        # Register global shortcuts (selection modes, tile nav)
        self._register_shortcuts()

        logger.info("MainWindow initialised")

    def closeEvent(self, event) -> None:
        """Handle window close — prompt to save if tiles have been edited."""
        try:
            edited = self._db.get_tiles_by_status(TileStatus.EDITED)
            if edited:
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    f"You have {len(edited)} tile(s) with unsaved edits.\n"
                    "Close anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    event.ignore()
                    return
        except Exception:
            pass
        super().closeEvent(event)

    # ── menu bar ───────────────────────────────────────────────────

    def _setup_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # ----- File menu -----
        file_menu = menu_bar.addMenu("&File")

        new_action = QAction("&New Project…", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.setObjectName("new_project")
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setObjectName("open_project")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setObjectName("save_project")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # Recent projects submenu
        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        preview_action = QAction("Preview &LAS/LAZ…", self)
        preview_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        preview_action.setObjectName("preview_las")
        preview_action.triggered.connect(self._on_preview)
        file_menu.addAction(preview_action)

        import_action = QAction("&Import LAS/LAZ…", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.setObjectName("import_las")
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ----- Tools menu -----
        tools_menu = menu_bar.addMenu("&Tools")

        filter_action = QAction("&Noise Filter…", self)
        filter_action.setObjectName("filter")
        filter_action.triggered.connect(self._on_filter)
        tools_menu.addAction(filter_action)

        classify_action = QAction("&Classify (Pointcept)…", self)
        classify_action.setObjectName("classify")
        classify_action.triggered.connect(self._on_classify)
        tools_menu.addAction(classify_action)

        tools_menu.addSeparator()

        ground_action = QAction("&Ground Classification…", self)
        ground_action.setObjectName("ground_classify")
        ground_action.triggered.connect(self._on_ground_classify)
        tools_menu.addAction(ground_action)

        bathy_action = QAction("&Bathymetry Processing…", self)
        bathy_action.setObjectName("bathy_process")
        bathy_action.triggered.connect(self._on_bathy_process)
        tools_menu.addAction(bathy_action)

        tools_menu.addSeparator()

        ground_control_action = QAction("&Ground Control…", self)
        ground_control_action.setObjectName("ground_control")
        ground_control_action.triggered.connect(self._on_ground_control)
        tools_menu.addAction(ground_control_action)

        tools_menu.addSeparator()

        export_action = QAction("&Export Raster (DTM / DSM)…", self)
        export_action.setObjectName("export_raster")
        export_action.triggered.connect(self._on_export_raster)
        tools_menu.addAction(export_action)

        tools_menu.addSeparator()

        crs_action = QAction("&CRS / Projection…", self)
        crs_action.setObjectName("crs_projection")
        crs_action.triggered.connect(self._on_crs)
        tools_menu.addAction(crs_action)

        tools_menu.addSeparator()

        analysis_action = QAction("&Processing Time Analysis…", self)
        analysis_action.setObjectName("processing_analysis")
        analysis_action.triggered.connect(self._on_processing_analysis)
        tools_menu.addAction(analysis_action)

        tools_menu.addSeparator()

        settings_action = QAction("&Settings…", self)
        settings_action.triggered.connect(self._on_settings)
        tools_menu.addAction(settings_action)

        # ----- Help menu -----
        help_menu = menu_bar.addMenu("&Help")

        docs_action = QAction("&Documentation…", self)
        docs_action.setShortcut(QKeySequence("F1"))
        docs_action.triggered.connect(self._on_documentation)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()

        website_action = QAction("Kabelik &Website", self)
        website_action.triggered.connect(self._on_website)
        help_menu.addAction(website_action)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ── toolbar ────────────────────────────────────────────────────

    def _setup_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        preview_btn = toolbar.addAction("Preview")
        preview_btn.setToolTip("Preview LAS/LAZ files before import (Ctrl+Shift+P)")
        preview_btn.triggered.connect(self._on_preview)

        import_btn = toolbar.addAction("Import")
        import_btn.setToolTip("Import LAS/LAZ files (Ctrl+I)")
        import_btn.triggered.connect(self._on_import)

        toolbar.addSeparator()

        filter_btn = toolbar.addAction("Filter")
        filter_btn.setToolTip("Apply noise filter to selected tiles")
        filter_btn.triggered.connect(self._on_filter)

        classify_btn = toolbar.addAction("Classify")
        classify_btn.setToolTip("Run Pointcept classification on selected tiles")
        classify_btn.triggered.connect(self._on_classify)

        ground_btn = toolbar.addAction("Ground")
        ground_btn.setToolTip("Ground classification (SMRF or TIN)")
        ground_btn.triggered.connect(self._on_ground_classify)

        bathy_btn = toolbar.addAction("Bathy")
        bathy_btn.setToolTip("Bathymetry processing (refraction, river crop, benthic filter)")
        bathy_btn.triggered.connect(self._on_bathy_process)

        gc_btn = toolbar.addAction("GndCtrl")
        gc_btn.setToolTip("Ground control points and roof surfaces adjustment")
        gc_btn.triggered.connect(self._on_ground_control)

        toolbar.addSeparator()

        crs_btn = toolbar.addAction("CRS")
        crs_btn.setToolTip("CRS / Projection (assign, transform, match points)")
        crs_btn.triggered.connect(self._on_crs)

        toolbar.addSeparator()

        self._class_vis_btn = QPushButton("Classes ▾")
        self._class_vis_btn.setToolTip("Toggle visibility of ASPRS classes in all views")
        self._class_vis_btn.setFlat(True)
        self._class_vis_btn.setStyleSheet(
            "QPushButton { padding: 2px 8px; font-weight: bold; }"
            "QPushButton::menu-indicator { image: none; }"
        )
        self._class_vis_menu = QMenu(self._class_vis_btn)
        self._class_vis_btn.setMenu(self._class_vis_menu)
        toolbar.addWidget(self._class_vis_btn)
        self._build_class_visibility_menu()

    def _build_class_visibility_menu(self) -> None:
        """Build the popup menu with checkable ASPRS class items."""
        menu = self._class_vis_menu
        menu.clear()
        menu.triggered.disconnect()
        menu.triggered.connect(self._on_class_visibility_toggled)

        all_action = menu.addAction("▸ Show All")
        all_action.setData(-1)
        none_action = menu.addAction("▸ Hide All")
        none_action.setData(-2)
        menu.addSeparator()

        for code in sorted(ASPRS_CLASS_NAMES.keys()):
            name = ASPRS_CLASS_NAMES[code]
            r, g, b = ASPRS_CLASS_COLORS.get(code, (0.5, 0.5, 0.5))
            pm = QPixmap(14, 14)
            pm.fill(QColor(int(r * 255), int(g * 255), int(b * 255)))
            action = menu.addAction(f"{code:2d}: {name}")
            action.setCheckable(True)
            action.setChecked(bool(self.class_visibility[code]))
            action.setData(code)
            action.setIcon(pm)

    def _on_class_visibility_toggled(self, action: QAction) -> None:
        code = action.data()
        if code == -1:
            self.class_visibility[:] = True
        elif code == -2:
            self.class_visibility[:] = False
        else:
            self.class_visibility[code] = action.isChecked()

        # Rebuild menu to sync all check states
        self._build_class_visibility_menu()

        # Propagate to views instantly (no disk reload needed)
        self._multi_view._view_3d.set_class_visibility(self.class_visibility)
        self._multi_view._view_profile.set_class_visibility(self.class_visibility)

        # Also do a full data reload for the profile view (class changes
        # may affect profile selection state) if a tile is open
        if self._editor.tile_id is not None:
            data = self._tm.load_tile_points_full(self._editor.tile_id)
            if data is not None:
                self._multi_load_for_edit(data)

    def _setup_central_widget(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: tile list
        self._tile_list_widget = TileListWidget()
        self._tile_list_widget.tile_selected.connect(self._on_tile_selected)
        self._tile_list_widget.open_requested.connect(self._on_tile_open)
        self._tile_list_widget.filter_requested.connect(self._on_tiles_filter)
        self._tile_list_widget.classify_requested.connect(self._on_tiles_classify)
        self._tile_list_widget.export_requested.connect(self._on_tiles_export)
        self._tile_list_widget.delete_requested.connect(self._on_tiles_delete)
        self._tile_list_widget.qc_status_changed.connect(self._on_qc_status_changed_batch)
        splitter.addWidget(self._tile_list_widget)

        # Center panel: multi-view widget
        self._multi_view = MultiViewWidget()
        self._multi_view.profile_line_defined.connect(self._on_profile_line_defined)
        # Wire profile view selection → editor
        self._multi_view._view_profile.selection_changed.connect(self._on_profile_selection)
        # Wire profile view hover → properties panel
        self._multi_view._view_profile.point_hovered.connect(self._on_point_hovered)
        # Wire profile view click → properties panel (Point Info tool)
        self._multi_view._view_profile.point_picked.connect(self._on_point_hovered)
        # Wire 3D view point pick → properties panel
        self._multi_view._view_3d.point_picked.connect(self._on_point_hovered)
        # Wire profile view width change → re-extract
        self._multi_view._view_profile.profile_width_changed.connect(self._on_profile_width_changed)

        # Apply saved tool sizes to profile view
        from .settings_dialog import load_general_settings
        settings = load_general_settings()
        self._multi_view._view_profile.set_brush_radius(settings.get("brush_radius", 2.0))
        self._multi_view._view_profile.set_rect_size(
            settings.get("rect_width", 4.0), settings.get("rect_height", 2.0))

        splitter.addWidget(self._multi_view)

        # Right panel: properties panel
        self._properties_panel = PropertiesPanel()
        self._properties_panel.classify_requested.connect(self._on_classify_selected)
        self._properties_panel.undo_requested.connect(self._on_undo)
        self._properties_panel.redo_requested.connect(self._on_redo)
        # Wire Point Info toggle → views
        self._properties_panel.point_info_toggled.connect(self._on_point_info_toggled)
        self._properties_panel.qc_status_changed.connect(self._on_properties_qc_changed)
        splitter.addWidget(self._properties_panel)

        # Proportions: 1 : 3 : 1
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        self.setCentralWidget(splitter)

    # ── status bar ─────────────────────────────────────────────────

    def _setup_status_bar(self) -> None:
        self._status_bar = self.statusBar()
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

    def set_status(self, message: str, timeout: int = 0) -> None:
        """
        Update the status bar message.

        Args:
            message: Text to display.
            timeout: Milliseconds before the message reverts to "Ready"
                     (0 = permanent).
        """
        self._status_label.setText(message)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self._status_label.setText("Ready"))

    # ── drag-and-drop ──────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            path_obj = Path(path)
            if path_obj.is_dir():
                self._start_import(path)
            elif path_obj.suffix.lower() in (".las", ".laz"):
                # Single file — import its parent directory
                self._start_import(str(path_obj.parent))
            else:
                QMessageBox.warning(
                    self, "Unsupported File",
                    f"Cannot import '{path_obj.name}'.  Please drop a directory "
                    f"containing .las/.laz files, or a single LAS/LAZ file."
                )

    # ── keyboard shortcuts ─────────────────────────────────────

    def _register_shortcuts(self) -> None:
        """Register configurable keyboard shortcuts using QShortcut."""
        from PySide6.QtGui import QShortcut

        # Remove previously registered shortcuts
        if hasattr(self, '_registered_shortcuts'):
            for s in self._registered_shortcuts:
                s.setEnabled(False)
                s.deleteLater()
        self._registered_shortcuts = []

        scs = load_shortcuts()

        def _make(shortcut_key: str, callback):
            seq = QKeySequence(scs.get(shortcut_key, ""))
            if seq.isEmpty():
                return
            sh = QShortcut(seq, self, activated=callback)
            self._registered_shortcuts.append(sh)

        _make("sel_brush", lambda: self._multi_view._sel_mode_combo.setCurrentIndex(0))
        _make("sel_above", lambda: self._multi_view._sel_mode_combo.setCurrentIndex(1))
        _make("sel_below", lambda: self._multi_view._sel_mode_combo.setCurrentIndex(2))
        _make("sel_rectangle", lambda: self._multi_view._sel_mode_combo.setCurrentIndex(3))
        _make("sel_rect_brush", lambda: self._multi_view._sel_mode_combo.setCurrentIndex(4))
        _make("next_tile", self._tile_list_widget.select_next_tile)
        _make("prev_tile", self._tile_list_widget.select_previous_tile)

        # Quick-classify shortcuts (emit directly to properties signal handler)
        _make("classify_created", lambda: self._properties_panel.classify_requested.emit(0))
        _make("classify_unclass", lambda: self._properties_panel.classify_requested.emit(1))
        _make("classify_ground", lambda: self._properties_panel.classify_requested.emit(2))
        _make("classify_low_veg", lambda: self._properties_panel.classify_requested.emit(3))
        _make("classify_med_veg", lambda: self._properties_panel.classify_requested.emit(4))
        _make("classify_high_veg", lambda: self._properties_panel.classify_requested.emit(5))
        _make("classify_building", lambda: self._properties_panel.classify_requested.emit(6))
        _make("classify_noise", lambda: self._properties_panel.classify_requested.emit(7))
        _make("classify_model_key", lambda: self._properties_panel.classify_requested.emit(8))
        _make("classify_water", lambda: self._properties_panel.classify_requested.emit(9))
        _make("classify_rail", lambda: self._properties_panel.classify_requested.emit(10))
        _make("classify_road", lambda: self._properties_panel.classify_requested.emit(11))
        _make("classify_overlap", lambda: self._properties_panel.classify_requested.emit(12))
        _make("classify_wire_guard", lambda: self._properties_panel.classify_requested.emit(13))
        _make("classify_wire_conductor", lambda: self._properties_panel.classify_requested.emit(14))
        _make("classify_tower", lambda: self._properties_panel.classify_requested.emit(15))
        _make("classify_wire_connector", lambda: self._properties_panel.classify_requested.emit(16))
        _make("classify_bridge", lambda: self._properties_panel.classify_requested.emit(17))
        _make("classify_high_noise", lambda: self._properties_panel.classify_requested.emit(18))

    def _on_new_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Project Location"
        )
        if not directory:
            return

        # Ask for project name
        name, ok = QInputDialog.getText(
            self, "Project Name",
            "Enter a name for the project:",
            text="New Project",
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Sanitise: replace path separators with dashes
        safe_name = name.replace("/", "-").replace("\\", "-")
        try:
            proj_dir = Path(directory) / safe_name
            self._pm.create(proj_dir, name=name)
            self._sync_db()
            self._refresh_tile_list()
            self._add_recent_project(str(proj_dir))
            self.set_status(f"Created project '{name}' in {directory}")
        except Exception as exc:
            logger.error("Failed to create project: %s", exc, exc_info=True)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{exc}")

    def _on_open_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Open Project Directory"
        )
        if not directory:
            return
        try:
            self._pm.open(directory)
            self._sync_db()
            self._refresh_tile_list()
            self._add_recent_project(directory)
            self.set_status(f"Opened project: {self._pm.metadata.get('name', directory)}")
        except Exception as exc:
            logger.error("Failed to open project: %s", exc, exc_info=True)
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{exc}")

    def _on_save_project(self) -> None:
        if self._pm.is_open:
            self._pm.save()
            self.set_status("Project saved", timeout=3000)

    def _on_import(self) -> None:
        """Launch the import wizard."""
        if not self._pm.is_open:
            QMessageBox.information(
                self, "No Project Open",
                "Please create or open a project before importing data."
            )
            return

        wizard = ImportWizard(self._tm, parent=self)
        if wizard.exec() == ImportWizard.Accepted:
            tile_ids = wizard.imported_tile_ids
            self._refresh_tile_list()
            self.set_status(f"Imported {len(tile_ids)} tile(s)", timeout=5000)

    def _on_preview(self) -> None:
        """Open the standalone LAS/LAZ preview dialog."""
        if self._preview_dlg is not None:
            self._preview_dlg.close()
            self._preview_dlg = None
        self._preview_dlg = PreviewDialog(parent=self)
        self._preview_dlg.import_requested.connect(self._on_preview_import)
        self._preview_dlg.destroyed.connect(lambda: setattr(self, '_preview_dlg', None))
        self._preview_dlg.show()  # non-modal — user can keep interacting with main window

    def _on_preview_import(self, file_paths: list) -> None:
        """Handle import request from the preview dialog.

        Groups files by parent directory and launches the ImportWizard
        for each unique directory.
        """
        if not self._pm.is_open:
            QMessageBox.information(
                self, "No Project Open",
                "Please create or open a project before importing data."
            )
            return

        # Group files by parent directory
        dirs: dict[str, list[Path]] = defaultdict(list)
        for p in file_paths:
            dirs[str(p.parent)].append(p)

        # For now, import each unique directory via the wizard
        for directory in dirs:
            wizard = ImportWizard(self._tm, parent=self, preselected_dir=directory)
            if wizard.exec() == ImportWizard.Accepted:
                self._refresh_tile_list()
                self.set_status(
                    f"Imported {len(wizard.imported_tile_ids)} tile(s) from "
                    f"{Path(directory).name}", timeout=5000
                )

    def _on_filter(self) -> None:
        """Open the noise filter dialog for selected tiles."""
        selected = self._tile_list_widget.get_selected_tile_ids()
        if not selected:
            QMessageBox.information(
                self, "No Tiles Selected",
                "Select one or more tiles in the tile list first."
            )
            return

        # Clear the 3D view before filtering — Open3D background render
        # threads crash if they're still running during LAS file writes.
        self._multi_view.clear()

        dialog = FilterDialog(self._tm, selected, parent=self)
        dialog.filter_applied.connect(self._on_filter_applied)
        dialog.exec()

    def _on_filter_applied(self, tile_ids: list, pipeline: list) -> None:
        """Apply a filter pipeline to the selected tiles in parallel
        with batch loading to avoid exhausting RAM on large projects."""
        from ..noise_filter import FilterWorker
        from ..gui.settings_dialog import load_general_settings
        import time as _time

        # Generate a batch ID for this processing run
        self._filter_batch_id = f"filter_{_time.strftime('%Y%m%d_%H%M%S')}"

        # Capture tile manager for thread-safe loading in worker
        tm = self._tm

        def _loader(tile_id: str):
            return tm.load_tile_points_full(tile_id)

        settings = load_general_settings()
        workers = settings.get("filter_workers", 4)

        # Progress dialog
        from PySide6.QtWidgets import QProgressDialog
        self._filter_progress = QProgressDialog(
            f"Filtering {len(tile_ids)} tile(s) with {workers} workers…",
            "Cancel", 0, len(tile_ids), self,
        )
        self._filter_progress.setWindowModality(Qt.WindowModal)
        self._filter_progress.setMinimumDuration(500)
        self._filter_progress.canceled.connect(self._on_filter_canceled)

        self._filter_worker = FilterWorker(tile_ids, pipeline, _loader, workers, self)
        self._filter_worker.progress.connect(self._on_filter_progress)
        self._filter_worker.tile_done.connect(self._on_filter_tile_done)
        self._filter_worker.finished_all.connect(self._on_filter_finished)
        self._filter_worker.error_occurred.connect(self._on_filter_error)
        self._filter_worker.start()

    def _on_filter_canceled(self):
        if hasattr(self, '_filter_worker') and self._filter_worker.isRunning():
            # Graceful: stop loading new tiles, finish in-flight ones.
            # _on_filter_finished will close the progress dialog.
            self._filter_worker.cancel()

    def _on_filter_progress(self, msg: str, pct: float):
        if hasattr(self, '_filter_progress'):
            self._filter_progress.setLabelText(msg)
            self._filter_progress.setValue(int(pct))

    def _on_filter_tile_done(self, tile_id: str, duration: float = 0.0, point_count: int = 0):
        """Write filtered tile: extract noise to tiles/noise/, keep clean points in tile."""
        if not hasattr(self, '_filter_worker'):
            return
        for tid, keep in self._filter_worker._results:
            if tid == tile_id:
                break
        else:
            return

        data = self._tm.load_tile_points_full(tile_id)
        if data is None:
            return
        tiles_dir = self._pm.tiles_dir
        if tiles_dir is None:
            return
        tile_info = self._db.get_tile(tile_id)
        if tile_info is None:
            return

        from ..tile_manager import _write_las_file, _read_las_header_template

        n_total = len(data["x"])
        n_noise = int((~keep).sum())
        n_kept = int(keep.sum())

        if n_noise == 0:
            logger.info("Filter on %s: no noise points found", tile_id)
            self._tm.update_tile_status(tile_id, TileStatus.FILTERED)
            self._tile_list_widget.update_tile_status(tile_id, TileStatus.FILTERED)
            # Record processing time even for zero-noise results
            if duration > 0:
                try:
                    with self._db.connect() as conn:
                        self._db.record_processing_time(
                            conn, tile_id, "filter",
                            duration_seconds=duration,
                            point_count=n_total,
                            params={"kept": n_kept, "noise": 0, "pipeline": self._filter_worker._pipeline},
                            batch_id=getattr(self, '_filter_batch_id', None),
                        )
                except Exception as exc:
                    logger.warning("Failed to record filter timing for %s: %s", tile_id, exc)
            return

        # Snapshot header template from original file
        las_path = tiles_dir / tile_info["filename"]
        try:
            header_template = _read_las_header_template(las_path)
        except Exception:
            header_template = None
            logger.warning("Could not read header template from %s — using fallback", las_path)

        # Helper: subset a data dict by boolean mask
        def _subset(d: dict, mask: np.ndarray) -> dict:
            return {k: v[mask] for k, v in d.items() if isinstance(v, np.ndarray) and len(v) == n_total}

        def _subset_extra_dims(d: dict, mask: np.ndarray) -> Optional[dict]:
            ed = d.get("extra_dims")
            if not ed:
                return None
            return {
                k: v[mask]
                for k, v in ed.items()
                if isinstance(v, np.ndarray) and len(v) == n_total
            }

        # ── Save noise points to tiles/noise/{tile_id}_noise.las ─────
        noise_dir = tiles_dir / "noise"
        noise_dir.mkdir(parents=True, exist_ok=True)
        noise_name = Path(tile_info["filename"]).stem + "_noise.las"
        noise_path = noise_dir / noise_name

        noise_data = _subset(data, ~keep)
        _write_las_file(
            noise_path,
            noise_data["x"], noise_data["y"], noise_data["z"],
            classes=noise_data.get("classification"),
            intensities=noise_data.get("intensity"),
            return_numbers=noise_data.get("return_number"),
            num_returns=noise_data.get("num_returns"),
            point_source_ids=noise_data.get("point_source_id"),
            gps_times=noise_data.get("gps_time"),
            scan_angle_ranks=noise_data.get("scan_angle_rank"),
            scan_direction_flags=noise_data.get("scan_direction_flag"),
            edge_of_flight_lines=noise_data.get("edge_of_flight_line"),
            user_data_array=noise_data.get("user_data"),
            reds=noise_data.get("red"),
            greens=noise_data.get("green"),
            blues=noise_data.get("blue"),
            key_points=noise_data.get("key_point"),
            synthetics=noise_data.get("synthetic"),
            withhelds=noise_data.get("withheld"),
            overlaps=noise_data.get("overlap"),
            header_template=header_template,
            extra_dims=_subset_extra_dims(data, ~keep),
        )
        logger.info("Noise file saved: %s (%d points)", noise_path, n_noise)

        # ── Register noise tile in database ──────────────────────────
        noise_tile_id = f"{tile_id}_noise"
        noise_filename = f"noise/{noise_name}"
        try:
            with self._db.connect() as conn:
                self._db.insert_tile(
                    conn,
                    tile_id=noise_tile_id,
                    filename=noise_filename,
                    bbox=(float(noise_data["x"].min()), float(noise_data["y"].min()),
                          float(noise_data["x"].max()), float(noise_data["y"].max())),
                    point_count=n_noise,
                    flight_line=tile_info.get("flight_line", 0),
                    scanner=tile_info.get("scanner", ''),
                    all_scanners=json.loads(tile_info.get("all_scanners", "[]"))
                        if isinstance(tile_info.get("all_scanners"), str) else tile_info.get("all_scanners", []),
                    sensor_type=tile_info.get("sensor_type", ''),
                    flightline_sensor_types=json.loads(tile_info.get("flightline_sensor_types", "{}"))
                        if isinstance(tile_info.get("flightline_sensor_types"), str) else tile_info.get("flightline_sensor_types", {}),
                    crs_epsg=tile_info.get("crs_epsg"),
                    crs_wkt=tile_info.get("crs_wkt"),
                    status=TileStatus.NOISE,
                    filter_params={"source_tile": tile_id, "filter": "isolated"},
                )
            logger.info("Noise tile registered: %s", noise_tile_id)
        except Exception as exc:
            logger.warning("Failed to register noise tile in DB: %s", exc)

        # ── Write kept (clean) points back to the original tile ──────
        keep_data = _subset(data, keep)
        _write_las_file(
            las_path,
            keep_data["x"], keep_data["y"], keep_data["z"],
            classes=keep_data.get("classification"),
            intensities=keep_data.get("intensity"),
            return_numbers=keep_data.get("return_number"),
            num_returns=keep_data.get("num_returns"),
            point_source_ids=keep_data.get("point_source_id"),
            gps_times=keep_data.get("gps_time"),
            scan_angle_ranks=keep_data.get("scan_angle_rank"),
            scan_direction_flags=keep_data.get("scan_direction_flag"),
            edge_of_flight_lines=keep_data.get("edge_of_flight_line"),
            user_data_array=keep_data.get("user_data"),
            reds=keep_data.get("red"),
            greens=keep_data.get("green"),
            blues=keep_data.get("blue"),
            key_points=keep_data.get("key_point"),
            synthetics=keep_data.get("synthetic"),
            withhelds=keep_data.get("withheld"),
            overlaps=keep_data.get("overlap"),
            header_template=header_template,
            extra_dims=_subset_extra_dims(data, keep),
        )

        # Update tile status
        self._tm.update_tile_status(tile_id, TileStatus.FILTERED)
        self._tile_list_widget.update_tile_status(tile_id, TileStatus.FILTERED)
        logger.info("Filtered %s: %d kept, %d noise → %s", tile_id, n_kept, n_noise, noise_name)

        # Record processing time
        if duration > 0:
            try:
                with self._db.connect() as conn:
                    self._db.record_processing_time(
                        conn, tile_id, "filter",
                        duration_seconds=duration,
                        point_count=n_total,
                        params={"kept": n_kept, "noise": n_noise, "pipeline": self._filter_worker._pipeline},
                        batch_id=getattr(self, '_filter_batch_id', None),
                    )
            except Exception as exc:
                logger.warning("Failed to record filter timing for %s: %s", tile_id, exc)

    def _on_filter_finished(self, tile_ids: list, keep_masks: list):
        self._refresh_tile_list()
        n = len(tile_ids)
        self.set_status(f"Filter applied to {n} tile(s)", timeout=5000)
        if hasattr(self, '_filter_progress'):
            self._filter_progress.close()

        # If a filtered tile is currently open in the editor, close it
        # so the user re-opens the freshly-written file.
        # Defer the view clear to avoid Open3D segfaults from pending
        # background render operations.
        open_tid = self._editor.tile_id
        if open_tid is not None and open_tid in tile_ids:
            self._editor.close_tile()
            self._profile_start = None
            self._profile_end = None
            self._dtm_ref_distances = None
            self._dtm_ref_elevations = None
            # Defer clear to next event-loop iteration so any in-flight
            # Open3D render operations complete first.
            QTimer.singleShot(0, self._multi_view.clear)
            self.set_status(
                f"Filter applied — tile '{open_tid}' was updated. "
                f"Re-open it to continue editing.",
                timeout=8000,
            )

    def _on_filter_error(self, msg: str):
        logger.error("Filter error: %s", msg)
        self.set_status(f"Filter error: {msg}", timeout=5000)

    def _on_classify(self) -> None:
        """Open the Pointcept classification dialog."""
        selected = self._tile_list_widget.get_selected_tile_ids()
        if not selected:
            QMessageBox.information(
                self, "No Tiles Selected",
                "Select one or more tiles in the tile list first."
            )
            return

        dialog = ClassificationDialog(self._tm, self._db, selected, parent=self)
        dialog.finished.connect(lambda: self._refresh_tile_list())
        dialog.exec()

    def _on_ground_classify(self) -> None:
        """Open the ground classification dialog for the currently open tile."""
        if self._editor.tile_id is None:
            QMessageBox.information(self, "No Tile Open",
                                    "Please open a tile first (double-click in tile list).")
            return
        data = self._tm.load_tile_points_full(self._editor.tile_id)
        if data is None:
            return

        from .ground_classify_dialog import GroundClassifyDialog
        dlg = GroundClassifyDialog(data, tile_id=self._editor.tile_id, db=self._db, parent=self)
        dlg.ground_applied.connect(lambda mask, sc: self._apply_ground_mask(mask, sc))
        dlg.exec()

    def _apply_ground_mask(self, mask: np.ndarray, source_class: int = -1) -> None:
        """Apply ground classification mask to the current tile.

        Args:
            mask: Boolean array, True = ground.
            source_class: Only reclassify points whose current class matches
                          this value.  -1 means all points, -2 means
                          classes 1 & 2 (unclassified + ground).
        """
        if self._editor.tile_id is None:
            return
        full_data = self._tm.load_tile_points_full(self._editor.tile_id)
        if full_data is None:
            return
        n_total = len(full_data["x"])
        if len(mask) != n_total:
            self.set_status("Ground mask size mismatch", timeout=5000)
            return

        cls = full_data["classification"]
        if source_class == -2:
            # Classes 1 & 2 — allows re-running ground classification
            in_source = (cls == 1) | (cls == 2)
            cls[in_source & mask] = 2     # ground
            cls[in_source & ~mask] = 1    # unclassified
            n_affected = in_source.sum()
        elif source_class >= 0:
            # Only modify points matching the source class
            in_source = (cls == source_class)
            cls[in_source & mask] = 2     # ground
            cls[in_source & ~mask] = 1    # unclassified
            n_affected = in_source.sum()
        else:
            cls[mask] = 2     # ground
            cls[~mask] = 1    # unclassified
            n_affected = n_total

        # ── Write updated classifications back to the LAS file ──────
        tile_info = self._db.get_tile(self._editor.tile_id)
        tiles_dir = self._pm.tiles_dir
        if tile_info is not None and tiles_dir is not None:
            from ..tile_manager import _read_las_header_template, _write_las_file
            las_path = tiles_dir / tile_info["filename"]
            # Preserve the original point format / VLRs / extra dims
            try:
                header_tmpl = _read_las_header_template(las_path)
            except Exception:
                header_tmpl = None
            _write_las_file(
                las_path,
                full_data["x"], full_data["y"], full_data["z"],
                classes=full_data["classification"],
                intensities=full_data.get("intensity"),
                return_numbers=full_data.get("return_number"),
                num_returns=full_data.get("num_returns"),
                point_source_ids=full_data.get("point_source_id"),
                gps_times=full_data.get("gps_time"),
                scan_angle_ranks=full_data.get("scan_angle_rank"),
                scan_direction_flags=full_data.get("scan_direction_flag"),
                edge_of_flight_lines=full_data.get("edge_of_flight_line"),
                user_data_array=full_data.get("user_data"),
                reds=full_data.get("red"),
                greens=full_data.get("green"),
                blues=full_data.get("blue"),
                key_points=full_data.get("key_point"),
                synthetics=full_data.get("synthetic"),
                withhelds=full_data.get("withheld"),
                overlaps=full_data.get("overlap"),
                header_template=header_tmpl,
                extra_dims=full_data.get("extra_dims"),
            )
            self._tm.update_tile_status(self._editor.tile_id, TileStatus.EDITED)

        self._editor.open_tile(self._editor.tile_id)
        self._multi_load_for_edit(full_data)
        self._tile_list_widget.update_tile_status(self._editor.tile_id, TileStatus.EDITED)
        self._regenerate_dtm()
        n_ground = mask.sum()
        self.set_status(
            f"Ground: {n_ground:,} / {n_affected:,} points ({n_ground/max(n_affected,1)*100:.1f}%)",
            timeout=8000,
        )

    def _on_bathy_process(self) -> None:
        """Open the bathymetry processing dialog for selected tiles."""
        selected = self._tile_list_widget.get_selected_tile_ids()
        if not selected:
            QMessageBox.information(
                self, "No Tiles Selected",
                "Select one or more tiles in the tile list first."
            )
            return

        tile_info = self._db.get_tile(selected[0])
        scanner = tile_info.get("scanner", '') if tile_info else ''
        sensor_type = tile_info.get("sensor_type", '') if tile_info else ''
        data_epsg = tile_info.get("crs_epsg") if tile_info else None

        from .bathy_dialog import BathyDialog
        dlg = BathyDialog(tile_ids=selected,
                          scanner=scanner, sensor_type=sensor_type,
                          parent=self, data_epsg=data_epsg)
        dlg.tile_save_requested.connect(self._save_bathy_tile)
        dlg.bathy_applied.connect(self._on_bathy_all_saved)
        dlg.exec()

    def _on_bathy_all_saved(self) -> None:
        """Called when all bathy tiles have been saved."""
        if self._editor.tile_id:
            reload_data = self._tm.load_tile_points_full(self._editor.tile_id)
            if reload_data:
                self._multi_load_for_edit(reload_data)
        self._refresh_tile_list()
        self._regenerate_dtm()

    def _save_bathy_tile(self, tile_id: str, result: dict) -> None:
        """Write bathy-processed data back to a tile's LAS file."""
        tile_info = self._db.get_tile(tile_id)
        if tile_info is None:
            return

        tiles_dir = self._pm.tiles_dir
        if tiles_dir is None:
            return

        las_path = tiles_dir / tile_info["filename"]
        if not las_path.is_file():
            return

        import laspy
        import shutil

        # Backup original
        backup_path = las_path.with_suffix(las_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(las_path, backup_path)

        # Read original file to get header template
        with laspy.open(las_path) as reader:
            extra_dims = []
            try:
                extra_dims = list(reader.header.point_format.extra_dimensions)
            except AttributeError:
                pass
            header_template = {
                "version": reader.header.version,
                "point_format_id": reader.header.point_format.id,
                "vlrs": list(reader.header.vlrs),
                "extra_dimensions": extra_dims,
                "x_scale": reader.header.x_scale,
                "y_scale": reader.header.y_scale,
                "z_scale": reader.header.z_scale,
            }

        from ..tile_manager import _write_las_file
        _write_las_file(
            las_path,
            result["x"], result["y"], result["z"],
            classes=result.get("classification"),
            intensities=result.get("intensity"),
            return_numbers=result.get("return_number"),
            num_returns=result.get("num_returns"),
            point_source_ids=result.get("point_source_id"),
            gps_times=result.get("gps_time"),
            scan_angle_ranks=result.get("scan_angle_rank"),
            scan_direction_flags=result.get("scan_direction_flag"),
            edge_of_flight_lines=result.get("edge_of_flight_line"),
            user_data_array=result.get("user_data"),
            reds=result.get("red"),
            greens=result.get("green"),
            blues=result.get("blue"),
            key_points=result.get("key_point"),
            synthetics=result.get("synthetic"),
            withhelds=result.get("withheld"),
            overlaps=result.get("overlap"),
            header_template=header_template,
            extra_dims=result.get("extra_dims"),
        )

        new_count = len(result["x"])
        with self._db.connect() as conn:
            self._db.update_point_count(conn, tile_id, new_count)

        self._tile_list_widget.update_tile_status(tile_id, TileStatus.EDITED)

    def _on_ground_control(self) -> None:
        """Open the Ground Control dialog for selected tiles."""
        tile_ids = self._tile_list_widget.get_selected_tile_ids()
        if not tile_ids:
            tile_ids = [self._editor.tile_id] if self._editor.tile_id else []

        if not tile_ids:
            QMessageBox.information(self, "No Tiles",
                                    "Please select tiles in the tile list or open a tile first.")
            return

        # Determine a representative EPSG from the first tile with CRS info
        data_epsg = None
        for tid in tile_ids:
            info = self._db.get_tile(tid)
            if info and info.get("crs_epsg"):
                data_epsg = info.get("crs_epsg")
                break

        dlg = GroundControlDialog(
            {}, parent=self, data_epsg=data_epsg,
            tile_ids=tile_ids,
            tile_manager=self._tm,
            database=self._db,
        )
        dlg.shift_applied.connect(self._apply_ground_control_shift)
        dlg.visualize_point.connect(self._on_visualize_control_point)

        # Non-modal so the user can interact with the 3-D view while the
        # dialog is open (essential for the "Go To in 3-D View" check).
        if self._ground_control_dlg is not None:
            self._ground_control_dlg.close()
            self._ground_control_dlg = None
        self._ground_control_dlg = dlg
        dlg.destroyed.connect(
            lambda: setattr(self, "_ground_control_dlg", None)
        )
        dlg.show()

    def _apply_ground_control_shift(self, dx: float, dy: float, dz: float) -> None:
        """Apply an XYZ shift to the current tile based on ground control results."""
        if self._editor.tile_id is None:
            return

        data = self._tm.load_tile_points_full(self._editor.tile_id)
        if data is None:
            return

        # Apply XYZ shift
        data["x"] = data["x"] + dx
        data["y"] = data["y"] + dy
        data["z"] = data["z"] + dz

        # Write back to LAS file
        tile_info = self._db.get_tile(self._editor.tile_id)
        if tile_info is None:
            return

        tiles_dir = self._pm.tiles_dir
        if tiles_dir is None:
            return

        las_path = tiles_dir / tile_info["filename"]
        if not las_path.is_file():
            return

        try:
            import laspy
            import shutil

            # Backup
            backup_path = las_path.with_suffix(las_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(las_path, backup_path)

            with laspy.open(las_path, mode="rw") as writer:
                writer.header = writer.header
                las_data = writer.read()
                las_data.x = data["x"]
                las_data.y = data["y"]
                las_data.z = data["z"]
                writer.write(las_data)

            self._editor.open_tile(self._editor.tile_id)
            self._multi_load_for_edit(data)
            self._tile_list_widget.update_tile_status(
                self._editor.tile_id, TileStatus.EDITED
            )
            self._regenerate_dtm()
            if dx == 0.0 and dy == 0.0:
                self.set_status(
                    f"Ground control Z shift applied: {dz:+.3f} m",
                    timeout=8000,
                )
            else:
                mag = float(np.sqrt(dx*dx + dy*dy + dz*dz))
                self.set_status(
                    f"Ground control XYZ shift applied: "
                    f"({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m, |Shift|={mag:.3f} m",
                    timeout=8000,
                )
        except Exception as exc:
            logger.exception("Failed to apply ground control shift")
            QMessageBox.critical(self, "Shift Failed", str(exc))

    def _on_visualize_control_point(self, x: float, y: float, z: float,
                                    label: str) -> None:
        """
        Navigate the 3-D view to a control point and mark it.

        Non-destructive: if the point is already inside the loaded point
        cloud, the camera simply pans/zooms to it and a marker sphere is
        dropped at ``z``.  Only when the point lies outside the current
        view is the containing tile loaded into the 3-D view first.
        """
        if self._db is None or self._tm is None:
            return

        view_3d = self._multi_view._view_3d

        # Fast path — the point is already visible; just move the camera.
        if view_3d.has_geometry and view_3d.contains_world_xy(x, y):
            view_3d.focus_on_point(x, y, z)
            self.set_status(
                f"Visual check: \"{label}\" @ ({x:.2f}, {y:.2f})",
                timeout=5000,
            )
            return

        # The point is outside the current view — load the tile that covers it.
        tile_ids = [
            info.get("id") for info in self._db.get_tiles_in_bbox(x, y, x, y)
            if info.get("id")
        ]
        if not tile_ids and self._editor.tile_id:
            tile_ids = [self._editor.tile_id]
        if not tile_ids:
            self.set_status(
                f"Visual check: no tile covers ({x:.2f}, {y:.2f})",
                timeout=3000,
            )
            return

        data = None
        loaded_tid = None
        for tid in tile_ids:
            td = self._tm.load_tile_points_full(tid)
            if td is not None and len(td.get("x", [])) > 0:
                data, loaded_tid = td, tid
                break

        if data is None:
            self.set_status(
                f"Visual check: no points loaded for ({x:.2f}, {y:.2f})",
                timeout=3000,
            )
            return

        view_3d.load_point_cloud(
            data["x"], data["y"], data["z"],
            data.get("classification"), data.get("intensity"),
            data.get("return_number"), data.get("point_source_id"),
        )
        view_3d.focus_on_point(x, y, z)

        self.set_status(
            f"Visual check: \"{label}\" — opened {loaded_tid}",
            timeout=5000,
        )

    def _on_crs(self) -> None:
        """Open the CRS / Projection dialog."""
        # Get current CRS from the project database
        crs_info = self._db.get_project_crs() if self._db else None
        current_epsg = crs_info.get("epsg") if crs_info else None
        current_wkt = crs_info.get("wkt", "") if crs_info else ""

        from .crs_dialog import CrsDialog
        dlg = CrsDialog(
            current_crs_wkt=current_wkt or "",
            current_epsg=current_epsg,
            parent=self,
        )
        dlg.crs_assigned.connect(self._on_crs_assigned)
        dlg.transform_applied.connect(self._on_crs_transform)
        dlg.match_transform_applied.connect(self._on_match_transform_applied)
        dlg.exec()

    def _on_processing_analysis(self) -> None:
        """Open the processing time analysis dialog."""
        dlg = ProcessingAnalysisDialog(self._db, parent=self)
        dlg.exec()

    # ── QC status handlers ──────────────────────────────────────

    def _on_qc_status_changed_batch(self, tile_ids: list, qc_status: str,
                                     qc_comment: str) -> None:
        """Handle QC status change from tile list context menu."""
        # If NEEDS_REWORK with __PROMPT__, ask for comment
        if qc_status == QCStatus.NEEDS_REWORK and qc_comment == "__PROMPT__":
            qc_comment, ok = self._prompt_qc_comment(tile_ids)
            if not ok:
                return  # user cancelled
        elif qc_status is None:
            qc_comment = ""

        self._apply_qc_status(tile_ids, qc_status, qc_comment)

    def _on_properties_qc_changed(self, qc_status: str, qc_comment: str) -> None:
        """Handle QC status change from properties panel (single tile)."""
        tile_id = self._editor.tile_id
        if tile_id is None:
            return
        qc_status_val = qc_status if qc_status else None
        self._apply_qc_status([tile_id], qc_status_val, qc_comment)

    def _apply_qc_status(self, tile_ids: list, qc_status, qc_comment: str) -> None:
        """Persist QC status to DB and update the tree."""
        for tid in tile_ids:
            try:
                with self._db.connect() as conn:
                    self._db.set_qc_status(conn, tid, qc_status, qc_comment)
            except Exception as exc:
                logger.warning("Failed to set QC status for %s: %s", tid, exc)
                continue
            self._tile_list_widget.update_tile_qc_status(tid, qc_status, qc_comment)

        # Refresh QC info in properties panel if the open tile was affected
        if self._editor.tile_id and self._editor.tile_id in tile_ids:
            self._properties_panel.set_tile_qc_info(qc_status, qc_comment)

        label = QCStatus.LABELS.get(qc_status, "Cleared") if qc_status else "Cleared"
        self.set_status(f"QC: {label} → {len(tile_ids)} tile(s)", timeout=5000)

    def _prompt_qc_comment(self, tile_ids: list) -> tuple:
        """Show a dialog asking for rework comment. Returns (comment, ok)."""
        tiles_str = ", ".join(tile_ids[:3])
        if len(tile_ids) > 3:
            tiles_str += f" (+{len(tile_ids) - 3} more)"

        comment, ok = QInputDialog.getMultiLineText(
            self, "Rework Required",
            f"What needs to be reworked for:\n{tiles_str}",
            "",
        )
        return comment.strip(), ok

    def _on_crs_assigned(self, wkt: str, epsg: int) -> None:
        """Handle CRS assignment from the dialog."""
        tile_ids = self._tile_list_widget.get_selected_tile_ids()
        if not tile_ids:
            tile_ids = [t["id"] for t in self._db.list_tiles()] if self._db else []
        if tile_ids and self._db:
            for tid in tile_ids:
                self._db.set_tile_crs(tid, epsg, wkt)
            self.set_status(
                f"CRS assigned: EPSG:{epsg} to {len(tile_ids)} tile(s)",
                timeout=5000,
            )
            # Refresh metadata panel if currently open tile is among them
            if self._editor.tile_id and self._editor.tile_id in tile_ids:
                self._populate_las_header(self._editor.tile_id)

    def _on_crs_transform(self, source_epsg: str, target_epsg: str) -> None:
        """Handle CRS transformation request from the dialog."""
        from ..crs import transform_coordinates, HAS_PYPROJ
        if not HAS_PYPROJ:
            QMessageBox.warning(self, "pyproj Required",
                                "Coordinate transformation requires pyproj.\n"
                                "Install with: pip install pyproj")
            return

        tile_ids = self._tile_list_widget.get_selected_tile_ids()
        if not tile_ids:
            QMessageBox.information(self, "No Tiles Selected",
                                    "Select tiles in the tile list to transform.")
            return

        reply = QMessageBox.question(
            self, "Confirm CRS Transform",
            f"Transform {len(tile_ids)} tile(s) from EPSG:{source_epsg} to "
            f"EPSG:{target_epsg}?\n\nThis will modify the X, Y, Z coordinates "
            "of all points. A backup is recommended.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.set_status(f"Transforming {len(tile_ids)} tile(s)…", timeout=0)
        try:
            for tid in tile_ids:
                data = self._tm.load_tile_points_full(tid)
                if data is None:
                    continue
                new_x, new_y, new_z = transform_coordinates(
                    data["x"], data["y"], data["z"],
                    int(source_epsg), int(target_epsg),
                )
                data["x"], data["y"] = new_x, new_y
                if new_z is not None:
                    data["z"] = new_z
                # Write back
                self._editor.open_tile(tid)
                self._multi_load_for_edit(data)
                self._tile_list_widget.update_tile_status(tid, TileStatus.EDITED)
                if self._db:
                    self._db.set_tile_crs(tid, int(target_epsg), None)
            self._regenerate_dtm()
            self.set_status(
                f"CRS transformed {len(tile_ids)} tile(s): EPSG:{source_epsg} → EPSG:{target_epsg}",
                timeout=8000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Transform Failed", str(exc))
            self.set_status("CRS transform failed", timeout=5000)

    def _on_match_transform_applied(self, method: str, params: dict) -> None:
        """Apply a match-points estimated transformation to selected tiles."""
        from ..crs import apply_helmert_3d, apply_affine_3d

        tile_ids = self._tile_list_widget.get_selected_tile_ids()
        if not tile_ids:
            QMessageBox.information(self, "No Tiles Selected",
                                    "Select tiles in the tile list to transform.")
            return

        label = "Helmert 7-param" if method == "helmert" else "Affine 12-param"
        reply = QMessageBox.question(
            self, f"Apply {label} Transform",
            f"Apply the estimated {label} transformation to "
            f"{len(tile_ids)} tile(s)?\n\n"
            f"RMSE: {params.get('rmse', '?'):.4f} m\n\n"
            "This will modify X, Y, Z of all points.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.set_status(f"Applying {label} to {len(tile_ids)} tile(s)…", timeout=0)
        try:
            apply_fn = apply_helmert_3d if method == "helmert" else apply_affine_3d
            for tid in tile_ids:
                data = self._tm.load_tile_points_full(tid)
                if data is None:
                    continue
                new_x, new_y, new_z = apply_fn(
                    data["x"], data["y"], data["z"], params,
                )
                data["x"], data["y"], data["z"] = new_x, new_y, new_z
                self._editor.open_tile(tid)
                self._multi_load_for_edit(data)
                self._tile_list_widget.update_tile_status(tid, TileStatus.EDITED)
            self._regenerate_dtm()
            self.set_status(
                f"Applied {label} to {len(tile_ids)} tile(s) "
                f"(RMSE={params.get('rmse', 0):.4f} m)",
                timeout=8000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Transform Failed", str(exc))
            self.set_status("Match transform failed", timeout=5000)

    def _on_export_raster(self, tile_ids: Optional[List[str]] = None) -> None:
        """Open the DTM / DSM export dialog."""
        if tile_ids is None:
            tile_ids = self._tile_list_widget.get_selected_tile_ids()
        if not tile_ids:
            QMessageBox.information(
                self, "No Tiles Selected",
                "Select one or more tiles in the tile list first."
            )
            return

        selected = tile_ids

        # Load point data and bboxes for selected tiles
        self.set_status(f"Loading {len(selected)} tile(s) for export…", timeout=0)

        tile_points: dict = {}
        tile_bboxes: dict = {}
        for tile_id in selected:
            data = self._tm.load_tile_points_full(tile_id)
            bbox = self._tm.get_tile_bbox(tile_id)
            if data is not None and bbox is not None:
                tile_points[tile_id] = data
                tile_bboxes[tile_id] = bbox

        if not tile_points:
            QMessageBox.warning(self, "Load Error", "Failed to load tile data.")
            self.set_status("Export cancelled — failed to load tiles", timeout=5000)
            return

        output_dir = str(self._pm.dtm_dir) if self._pm.dtm_dir else "."

        dialog = ExportDialog(
            list(tile_points.keys()),
            tile_points,
            tile_bboxes,
            output_dir,
            parent=self,
        )
        if dialog.exec() == QDialog.Accepted:
            self.set_status(
                f"Exported {len(dialog.written_files)} file(s) to {output_dir}",
                timeout=8000,
            )
        else:
            self.set_status("Export cancelled", timeout=3000)

    def _on_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.shortcuts_changed.connect(self._apply_shortcuts)
        dialog.exec()
        # Re-apply tool sizes from saved settings
        from .settings_dialog import load_general_settings
        settings = load_general_settings()
        self._multi_view._view_profile.set_brush_radius(settings.get("brush_radius", 2.0))
        self._multi_view._view_profile.set_rect_size(
            settings.get("rect_width", 4.0), settings.get("rect_height", 2.0))

    def _apply_shortcuts(self, shortcuts: dict) -> None:
        """Update menu shortcuts and re-register global shortcuts."""
        # Update menu actions
        menu_bar = self.menuBar()
        for action in menu_bar.findChildren(QAction):
            name = action.objectName()
            if name in shortcuts:
                action.setShortcut(QKeySequence(shortcuts[name]))
        # Re-register global shortcuts
        self._register_shortcuts()

    def _on_website(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl(APP_WEBSITE))

    def _on_about(self) -> None:
        from pathlib import Path
        logo_path = Path(__file__).parent / "assets" / "logo.png"

        dlg = QDialog(self)
        dlg.setWindowTitle(f"About {APP_NAME}")
        dlg.setMinimumWidth(480)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)

        # Logo
        if logo_path.is_file():
            logo_lbl = QLabel()
            pix = QPixmap(str(logo_path))
            pix = pix.scaledToWidth(360, Qt.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            logo_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_lbl)

        # Title
        title = QLabel(f"<h2>{APP_NAME} v{APP_VERSION}</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Description
        desc = QLabel(f"<p>{APP_DESCRIPTION}</p>")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        # Organisation + website
        org = QLabel(
            f"<p><b>{APP_ORG}</b><br>"
            f"<a href='{APP_WEBSITE}'>{APP_WEBSITE}</a></p>"
        )
        org.setAlignment(Qt.AlignCenter)
        org.setOpenExternalLinks(True)
        layout.addWidget(org)

        # Tech stack
        tech = QLabel(
            "<p><small>Built with PySide6, Open3D, laspy, "
            "Pointcept, NumPy, SciPy<br>"
            "Licensed under GPLv3</small></p>"
        )
        tech.setAlignment(Qt.AlignCenter)
        layout.addWidget(tech)

        # Close button
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        dlg.exec()

    def _on_documentation(self) -> None:
        self._show_help_dialog()

    def _show_help_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} — Documentation")
        dlg.resize(700, 550)
        dlg.setMinimumWidth(500)

        layout = QVBoxLayout(dlg)

        from PySide6.QtWidgets import QTabWidget, QTextEdit, QVBoxLayout as VBox
        tabs = QTabWidget()

        def _add_tab(title: str, text: str) -> None:
            te = QTextEdit()
            te.setReadOnly(True)
            te.setHtml(text)
            tabs.addTab(te, title)

        # ── Overview ────────────────────────────────────────────────
        _add_tab("Overview", f"""
<h2>{APP_NAME} v{APP_VERSION}</h2>
<p>{APP_DESCRIPTION}</p>
<p>Developed by <a href='{APP_WEBSITE}'>{APP_ORG}</a> — Remote Sensing,
Geomatics &amp; IT Services.</p>

<h3>Quick Start</h3>
<ol>
<li><b>New Project</b> (Ctrl+N) — create or open a project folder.</li>
<li><b>Import LAS/LAZ</b> (Ctrl+I) — add point cloud tiles to the project.
   Files are copied into <code>tiles/</code>, indexed in the database,
   and auto-classified by sensor type (topo / bathy).</li>
<li><b>Noise Filter</b> (Ctrl+F) — apply SOR, isolated-point removal,
   DBSCAN, bilateral smoothing, and more.  Build multi-step filter
   pipelines and preview results in 3D before batch-applying.</li>
<li><b>Manual Edit</b> — use selection tools (brush, rectangle, line)
   and quick-classify buttons to manually correct points.</li>
<li><b>Classify (Pointcept)</b> — run the Pointcept deep-learning model
   for automatic ASPRS classification on GPU.</li>
<li><b>Bathymetry Processing</b> — Snell's-law refraction correction,
   water-surface ghost removal, river-corridor cropping, and benthic
   continuity filtering for bathymetric LiDAR.</li>
<li><b>Export Raster</b> — generate DTM and DSM GeoTIFF rasters.</li>
</ol>

<h3>Keyboard Shortcuts</h3>
<p>See <b>Settings → Keyboard Shortcuts</b> for the full list (F1).</p>
""")

        # ── Filter Pipeline ─────────────────────────────────────────
        _add_tab("Noise Filters", """
<h2>Noise Filter Pipeline</h2>
<p>Open via <b>Tools → Noise Filter…</b> (Ctrl+F).  Build a multi-step
pipeline by adding filter stages.  Preview results in 3D on a single tile
before batch-applying to many tiles.</p>

<h3>Filter Types</h3>
<table border='1' cellpadding='4' cellspacing='0'>
<tr><th>Filter</th><th>Description</th><th>Key Parameters</th></tr>
<tr><td><b>SOR</b></td><td>Statistical Outlier Removal — removes points
whose mean KNN distance exceeds the global mean + std_ratio × std.</td>
<td>Neighbors (6–100), Std Ratio (0.5–5.0)</td></tr>
<tr><td><b>ROR</b></td><td>Radius Outlier Removal — removes points with
too few neighbours within a search radius.</td>
<td>Radius (m), Min Points</td></tr>
<tr><td><b>Isolated</b></td><td>Isolated-point filter — same as Terrascan
"Isolated Points". Counts neighbours within search_radius; removes those
below min_neighbors.</td>
<td>Search Radius (m), Min Neighbors (incl. self)</td></tr>
<tr><td><b>DBSCAN Above</b></td><td>Clusters points above the local surface
(aerial noise: birds, dust). Removes small clusters.</td>
<td>Epsilon (m), Min Samples, Min Cluster Size</td></tr>
<tr><td><b>DBSCAN Below</b></td><td>Clusters points below the local surface
(sub-surface noise: multi-path, sensor artefacts).</td>
<td>Epsilon (m), Min Samples, Min Cluster Size</td></tr>
<tr><td><b>Low Points</b></td><td>Finds points that are much lower than
nearby points — classic low-point / multipath filter.</td>
<td>Search Radius (m), Max Below/Above Neighbours</td></tr>
<tr><td><b>Surface Noise</b></td><td>Removes points far from a coarse
local surface model (grid-based).</td>
<td>Grid Size (m), Tolerance (m)</td></tr>
<tr><td><b>Multipath</b></td><td>Removes ghost points caused by laser
multi-path reflections below the ground surface.</td>
<td>Depth Threshold (m)</td></tr>
<tr><td><b>Bilateral</b></td><td>Edge-preserving bilateral smoothing —
denoises point positions without blurring edges.</td>
<td>Spatial Sigma, Range Sigma, KNN</td></tr>
<tr><td><b>Thin Average</b></td><td>Grid-based point thinning by averaging
— reduces point density uniformly.</td>
<td>Grid Size (m)</td></tr>
</table>

<h3>Pipeline Features</h3>
<ul>
<li><b>Add / Remove / Reorder</b> steps with drag-and-drop.</li>
<li><b>Save / Load Presets</b> — reuse filter configurations (JSON).</li>
<li><b>Live 3D Preview</b> — toggle noise visibility in the viewer.</li>
<li><b>Tile navigation</b> — browse tiles with the arrow buttons.</li>
<li><b>Batch Apply</b> — run the pipeline on all selected tiles using
configurable parallel workers (Settings → General).</li>
</ul>
""")

        # ── Bathymetry ──────────────────────────────────────────────
        _add_tab("Bathymetry", """
<h2>Bathymetry Processing Pipeline</h2>
<p>Open via <b>Tools → Bathymetry Processing…</b>.  A dedicated pipeline
for bathymetric (green-wavelength) LiDAR data.</p>

<h3>Pipeline Steps (in order)</h3>
<ol>
<li><b>Water Surface Crop (WSM)</b> — crop points to a water-surface
model GeoTIFF.  Points outside the model or above the water surface
plus tolerance are removed.
<br><i>Parameters:</i> GeoTIFF path, tolerance above surface (cm),
extrapolation distance (m).</li>

<li><b>Snell's Law Refraction Correction</b> — corrects the apparent
3-D position of submerged points for the bending and slowing of the
laser beam at the air-water interface.
<br><i>Water surface:</i> scalar Z, per-point array, or GeoTIFF.
<br><i>Refractive index:</i> n_water (default 1.3333).
<br><i>Trajectory:</i> optional ASCII file with per-pulse sensor
positions (GPS time, X, Y, Z).</li>

<li><b>Water Surface Ghost Removal</b> — detects the water surface
plane via RANSAC and removes points in a narrow band around it
(mirror ghosts / surface reflections).
<br><i>Parameters:</i> along-track tile length (m).</li>

<li><b>River Corridor Crop</b> — uses local roughness and intensity
characteristics to distinguish river-bed points from land points.</li>

<li><b>Benthic Continuity Filter</b> — region-growing flood-fill from
high-confidence bed points to classify connected benthic surfaces.
<br><i>Parameters:</i> search radius (m), max slope.</li>
</ol>

<h3>Parallelism</h3>
<p>Configure workers in <b>Settings → General → Bathy parallel workers</b>.
Each worker processes one tile at a time from disk.</p>
""")

        # ── Classification ──────────────────────────────────────────
        _add_tab("Classification", """
<h2>Pointcept Deep-Learning Classification</h2>
<p>Open via <b>Tools → Classify (Pointcept)…</b>.  Runs the Pointcept
semantic segmentation model on GPU to assign ASPRS classes.</p>

<h3>Configuration</h3>
<ul>
<li><b>Pointcept Root</b> — path to the bundled Pointcept directory.</li>
<li><b>Model Path</b> — trained checkpoint (.pth).</li>
<li><b>Config File</b> — model configuration (.py).</li>
<li><b>Voxel Size</b> — density normalisation voxel size (default 0.15 m).</li>
<li><b>Smoothing</b> — k-NN label smoothing (yes/no).</li>
</ul>

<h3>How It Works</h3>
<p>Each tile is processed in a separate subprocess.  The model is
loaded from disk to GPU, the LAS file is voxelised, inference runs on
overlapping 50 m blocks with 25 m stride, and output labels are remapped
to ASPRS classes.  Results replace the original LAS file.</p>

<h3>Parallelism</h3>
<p>Configure workers in <b>Settings → General → Classify parallel workers</b>.
<em>Note:</em> on a single GPU, set workers to 1 — each subprocess loads
its own model copy, so parallel workers compete for GPU memory.</p>
""")

        # ── Selection & Edit Tools ──────────────────────────────────
        _add_tab("Selection &amp; Edit", """
<h2>Selection &amp; Manual Edit Tools</h2>

<h3>Selection Modes</h3>
<table border='1' cellpadding='4' cellspacing='0'>
<tr><th>Tool</th><th>Shortcut</th><th>How to Use</th></tr>
<tr><td><b>Select Above</b></td><td>A</td><td>Draw a line; points above
the line are selected.</td></tr>
<tr><td><b>Select Below</b></td><td>L</td><td>Draw a line; points below
the line are selected.</td></tr>
<tr><td><b>Select Rectangle</b></td><td>R</td><td>Drag a rectangle;
points inside are selected.</td></tr>
<tr><td><b>Brush</b></td><td>B</td><td>Circular brush; drag to select
points within the brush radius.</td></tr>
<tr><td><b>Rect Brush</b></td><td>Shift+R</td><td>Rectangular brush with
configurable width/height.</td></tr>
<tr><td><b>Point Info</b></td><td>—</td><td>Click a point to see its
coordinates, classification, and attributes in the properties panel.</td></tr>
</table>

<h3>Quick-Classify Buttons</h3>
<p>Assign the selected points to an ASPRS class using the buttons
in the <b>Properties Panel</b> (right sidebar).  Hotkeys Ctrl+0 through
Ctrl+9, Ctrl+Shift+0–5.  All edits are undoable (Ctrl+Z) via the
SQLite-backed command history.</p>

<h3>Tile Navigation</h3>
<ul>
<li><b>Next Tile:</b> Tab</li>
<li><b>Previous Tile:</b> Shift+Tab</li>
</ul>
""")

        # ── Settings ────────────────────────────────────────────────
        _add_tab("Settings", """
<h2>Settings Dialog</h2>
<p>Open via <b>Tools → Settings…</b></p>

<h3>Keyboard Shortcuts</h3>
<p>27 configurable shortcuts for file operations, tool activation,
selection modes, and quick-classify actions.  Double-click a shortcut
to reassign it.</p>

<h3>General</h3>
<table border='1' cellpadding='4' cellspacing='0'>
<tr><th>Setting</th><th>Range</th><th>Default</th><th>Description</th></tr>
<tr><td><b>Filter parallel workers</b></td><td>1–16</td><td>4</td>
<td>Number of tiles to filter in parallel via ThreadPoolExecutor.</td></tr>
<tr><td><b>Classify parallel workers</b></td><td>1–8</td><td>1</td>
<td>Number of Pointcept subprocesses.  Set to 1 for single GPU.</td></tr>
<tr><td><b>Bathy parallel workers</b></td><td>1–16</td><td>4</td>
<td>Number of tiles to process in parallel for bathymetry.</td></tr>
</table>

<h3>Manual Edit Tools</h3>
<table border='1' cellpadding='4' cellspacing='0'>
<tr><th>Setting</th><th>Range</th><th>Default</th></tr>
<tr><td><b>Brush Radius</b></td><td>0.1–50 m</td><td>2.0 m</td></tr>
<tr><td><b>Rectangle Width</b></td><td>0.1–100 m</td><td>5.0 m</td></tr>
<tr><td><b>Rectangle Height</b></td><td>0.1–100 m</td><td>3.0 m</td></tr>
</table>
""")

        # ── Other Tools ─────────────────────────────────────────────
        _add_tab("Other Tools", """
<h2>Other Tools &amp; Dialogs</h2>

<h3>Ground Classification</h3>
<p><b>Tools → Ground Classification…</b> — classify ground points using
SMRF (Simple Morphological Filter) or TIN (Triangulated Irregular Network)
densification.  SMRF builds a minimum surface with progressive window
sizes; TIN iteratively densifies a ground surface from seed points.</p>

<h3>Ground Control</h3>
<p><b>Tools → Ground Control…</b> — import ground control points (GCPs)
for accuracy assessment and georeferencing validation.</p>

<h3>Export Raster (DTM / DSM)</h3>
<p><b>Tools → Export Raster…</b> — generate Digital Terrain Model and
Digital Surface Model GeoTIFF rasters from the classified point cloud.
Choose resolution, interpolation method, and output CRS.</p>

<h3>CRS / Projection</h3>
<p><b>Tools → CRS / Projection…</b> — view and change the coordinate
reference system of the project.</p>

<h3>Processing Time Analysis</h3>
<p><b>Tools → Processing Time Analysis…</b> — review per-tile processing
durations for filter, classify, and bathy operations, grouped by batch.</p>

<h3>Preview LAS/LAZ</h3>
<p><b>File → Preview LAS/LAZ</b> — quick-look at LAS/LAZ files before
importing.  Shows point count, extent, CRS, and attribute summary.</p>
""")

        layout.addWidget(tabs)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dlg.exec()

    # ── slot: tile list signals ────────────────────────────────────

    def _on_tile_selected(self, tile_id: str) -> None:
        self.set_status(f"Selected: {tile_id}", timeout=0)
        # Show tile info in properties panel
        tile_info = self._db.get_tile(tile_id)
        if tile_info:
            all_scanners_raw = tile_info.get("all_scanners", "[]")
            try:
                all_scanners = json.loads(all_scanners_raw) if isinstance(all_scanners_raw, str) else all_scanners_raw
            except Exception:
                all_scanners = []
            self._properties_panel.set_selection_count(tile_info.get("point_count", 0))
            # Fetch CRS from DB
            crs_db = ""
            crs_epsg = tile_info.get("crs_epsg")
            if crs_epsg:
                crs_db = f"EPSG:{crs_epsg}"
            fl_sensor_raw = tile_info.get("flightline_sensor_types", "{}")
            try:
                fl_sensor_types = json.loads(fl_sensor_raw) if isinstance(fl_sensor_raw, str) else fl_sensor_raw
            except Exception:
                fl_sensor_types = {}
            self._properties_panel.set_tile_metadata(
                scanner=tile_info.get("scanner", ''),
                all_scanners=all_scanners,
                sensor_type=tile_info.get("sensor_type", ''),
                flight_line=tile_info.get("flight_line", 0),
                flightline_sensor_types=fl_sensor_types,
                point_count=tile_info.get("point_count", 0),
                crs=crs_db,
                filename=tile_info.get("filename", ""),
                modified=self._file_modified(tile_info),
            )
            # Read LAS header for version
            self._populate_las_header(tile_id)

    def _file_modified(self, tile_info: dict) -> str:
        """Return a human-readable modification time for a tile's LAS file."""
        try:
            tiles_dir = self._pm.tiles_dir
            if tiles_dir is None:
                return ""
            las_path = tiles_dir / tile_info["filename"]
            if las_path.is_file():
                from datetime import datetime
                ts = las_path.stat().st_mtime
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        return ""

    def _on_tile_open(self, tile_id: str) -> None:
        """Open a tile in the multi-view for inspection and editing."""
        self.set_status(f"Opening {tile_id}…", timeout=0)
        logger.info("Open requested for tile: %s", tile_id)

        # Load full point data
        data = self._tm.load_tile_points_full(tile_id)
        if data is None:
            QMessageBox.warning(self, "Load Error", f"Failed to load tile {tile_id}.")
            return

        # Apply class visibility filter for views
        view_data = self._apply_class_filter(data)

        # Open in editor
        if not self._editor.open_tile(tile_id):
            QMessageBox.warning(self, "Edit Error", f"Failed to open tile {tile_id} for editing.")
            return

        # Load into multi-view
        self._multi_view.load_tile(tile_id, view_data)
        self._properties_panel.set_tile_data(view_data)
        self._properties_panel.set_undo_info(*self._editor.undo_stack_info)

        # Auto-generate DTM if tile is classified
        tile_info = self._db.get_tile(tile_id)
        if tile_info and tile_info.get("status") in (TileStatus.CLASSIFIED, TileStatus.EDITED, TileStatus.FILTERED):
            self._multi_view._view_dtm.generate_dtm()

        # Populate metadata panel
        if tile_info:
            all_scanners_raw = tile_info.get("all_scanners", "[]")
            try:
                all_scanners = json.loads(all_scanners_raw) if isinstance(all_scanners_raw, str) else all_scanners_raw
            except Exception:
                all_scanners = []
            fl_sensor_raw = tile_info.get("flightline_sensor_types", "{}")
            try:
                fl_sensor_types = json.loads(fl_sensor_raw) if isinstance(fl_sensor_raw, str) else fl_sensor_raw
            except Exception:
                fl_sensor_types = {}
            # Fetch CRS from DB (may not be in LAS file VLR)
            crs_db = ""
            crs_epsg = tile_info.get("crs_epsg")
            if crs_epsg:
                crs_db = f"EPSG:{crs_epsg}"
            self._properties_panel.set_tile_metadata(
                scanner=tile_info.get("scanner", ''),
                all_scanners=all_scanners,
                sensor_type=tile_info.get("sensor_type", ''),
                flight_line=tile_info.get("flight_line", 0),
                flightline_sensor_types=fl_sensor_types,
                point_count=tile_info.get("point_count", 0),
                crs=crs_db,
                filename=tile_info.get("filename", ""),
                modified=self._file_modified(tile_info),
            )
            self._properties_panel.set_tile_qc_info(
                tile_info.get("qc_status"),
                tile_info.get("qc_comment"),
            )
            self._populate_las_header(tile_id)

        self.set_status(f"Opened: {tile_id} ({data['x'].size:,} points)", timeout=5000)

    # ── slot: profile line ────────────────────────────────────

    def _on_profile_line_defined(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
    ) -> None:
        """Handle a newly-drawn profile line from the DTM view."""
        logger.info("Profile line: %s → %s", start_xy, end_xy)

        if self._editor.tile_id is None:
            return

        self._profile_start = start_xy
        self._profile_end = end_xy

        # Set corridor display in DTM view
        self._multi_view._view_dtm.set_profile_corridor(
            start_xy, end_xy, self._profile_width,
        )

        # Tell profile view the current width
        self._multi_view._view_profile.set_profile_width(self._profile_width)

        profile = self._editor.extract_profile(start_xy, end_xy, self._profile_width)
        if profile is None:
            return

        # Update profile view
        self._multi_view._view_profile.set_profile_data(
            profile.distances,
            profile.elevations,
            profile.classifications,
            intensities=profile.intensities,
            indices=profile.indices,
            xs=profile.xs,
            ys=profile.ys,
            zs=profile.elevations,  # elevations ARE the Z values
        )

        # Extract DTM profile for reference line
        if self._multi_view._view_dtm._dtm_grid_x is not None:
            from ..dtm_generator import extract_dtm_profile
            view_dtm = self._multi_view._view_dtm
            dtm_d, dtm_z = extract_dtm_profile(
                view_dtm._dtm_grid_x,
                view_dtm._dtm_grid_y,
                view_dtm._dtm_grid_z,
                start_xy,
                end_xy,
            )
            self._dtm_ref_distances = dtm_d
            self._dtm_ref_elevations = dtm_z
            self._multi_view._view_profile.set_dtm_reference(dtm_d, dtm_z)

        if len(profile.distances) > 0:
            self.set_status(
                f"Profile: {len(profile.distances)} pts / {profile.distances[-1]:.1f} m "
                f"(width={self._profile_width:.1f} m, scroll to adjust, click to confirm)",
                timeout=8000,
            )
        else:
            self.set_status(
                f"Profile: 0 points in corridor (width={self._profile_width:.1f} m)",
                timeout=5000,
            )

    def _on_profile_selection(self, mask: np.ndarray) -> None:
        """Called when the user makes a selection in the profile view."""
        # Store selection in the editor so classify buttons can use it
        self._editor.set_selection(mask)
        count = self._editor.selected_count
        self._properties_panel.set_selection_count(count)

    def _on_point_hovered(self, idx: int, x: float, y: float, z: float,
                          classification: int, intensity: int) -> None:
        """Update the properties panel when hovering over or clicking a point.
        Only updates when the Point Info tool is active."""
        if not self._properties_panel.point_info_active:
            return
        self._properties_panel.set_point_info(
            x=x, y=y, z=z,
            classification=classification,
            intensity=intensity,
            point_index=idx,
        )

    def _on_point_info_toggled(self, active: bool) -> None:
        """Propagate Point Info toggle state to both views."""
        self._multi_view._view_3d.point_info_active = active
        self._multi_view._view_profile.point_info_active = active

    def _on_profile_width_changed(self, new_width: float) -> None:
        """Called when the user scrolls to change the profile corridor width."""
        if self._editor.tile_id is None or self._profile_start is None:
            return
        self._profile_width = new_width
        # Update DTM corridor display
        self._multi_view._view_dtm.set_profile_corridor(
            self._profile_start, self._profile_end, new_width,
        )
        # Re-extract profile with new width
        profile = self._editor.extract_profile(
            self._profile_start, self._profile_end, self._profile_width
        )
        if profile is None:
            return
        self._multi_view._view_profile.set_profile_data(
            profile.distances, profile.elevations, profile.classifications,
            intensities=profile.intensities,
            indices=profile.indices,
            xs=profile.xs,
            ys=profile.ys,
            zs=profile.elevations,
        )
        # Restore DTM reference line
        if self._dtm_ref_distances is not None:
            self._multi_view._view_profile.set_dtm_reference(
                self._dtm_ref_distances, self._dtm_ref_elevations,
            )
        self.set_status(
            f"Profile width: {self._profile_width:.1f} m — "
            f"{len(profile.distances)} pts (scroll to adjust, click to confirm)",
            timeout=3000,
        )

    # ── slot: classification from properties panel ─────────────

    def _on_classify_selected(self, new_class: int) -> None:
        """Assign a new class to currently selected profile points."""
        if self._editor.selected_count == 0:
            self.set_status("No points selected — select points in the profile view first", timeout=3000)
            return

        ok = self._editor.assign_class(new_class)
        if ok:
            self._properties_panel.set_selection_count(0)
            self._properties_panel.set_undo_info(*self._editor.undo_stack_info)

            # Refresh 3D and DTM views (preserve profile view)
            if self._editor.tile_id:
                data = self._tm.load_tile_points_full(self._editor.tile_id)
                if data:
                    self._multi_load_for_edit(data)
                # Update tile status in the list widget
                self._tile_list_widget.update_tile_status(
                    self._editor.tile_id, TileStatus.EDITED
                )
                self._regenerate_dtm()

            self.set_status(
                f"Reclassified points to class {new_class}",
                timeout=3000,
            )

    # ── slot: undo / redo ─────────────────────────────────────

    def _on_undo(self) -> None:
        desc = self._editor.undo()
        if desc:
            self._properties_panel.set_undo_info(*self._editor.undo_stack_info)
            if self._editor.tile_id:
                data = self._tm.load_tile_points_full(self._editor.tile_id)
                if data:
                    self._multi_load_for_edit(data)
                self._tile_list_widget.update_tile_status(
                    self._editor.tile_id, TileStatus.EDITED
                )
                self._regenerate_dtm()
            self.set_status(f"Undo: {desc}", timeout=3000)

    def _on_redo(self) -> None:
        desc = self._editor.redo()
        if desc:
            self._properties_panel.set_undo_info(*self._editor.undo_stack_info)
            if self._editor.tile_id:
                data = self._tm.load_tile_points_full(self._editor.tile_id)
                if data:
                    self._multi_load_for_edit(data)
                self._tile_list_widget.update_tile_status(
                    self._editor.tile_id, TileStatus.EDITED
                )
                self._regenerate_dtm()
            self.set_status(f"Redo: {desc}", timeout=3000)

    def _regenerate_dtm(self) -> None:
        """Regenerate DTM from the current editor tile data after an edit."""
        if self._editor.tile_id:
            data = self._tm.load_tile_points_full(self._editor.tile_id)
            if data:
                self._multi_view._view_dtm.load_points(data)
                self._multi_view._view_dtm.generate_dtm()

    def _on_tiles_filter(self, tile_ids: List[str]) -> None:
        self._on_filter()

    def _on_tiles_classify(self, tile_ids: List[str]) -> None:
        self._on_classify()

    def _on_tiles_export(self, tile_ids: List[str]) -> None:
        self._on_export_raster(tile_ids)

    def _on_tiles_delete(self, tile_ids: List[str]) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Tiles",
            f"Are you sure you want to delete {len(tile_ids)} tile(s)?\n\n"
            f"This will remove the tile files from disk and cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        tiles_dir = self._pm.tiles_dir
        if tiles_dir is None:
            return

        for tid in tile_ids:
            tile_info = self._db.get_tile(tid)
            if tile_info is None:
                continue
            las_path = tiles_dir / tile_info["filename"]
            if las_path.exists():
                las_path.unlink()
            # Also remove backup if present
            backup = las_path.with_suffix(las_path.suffix + ".bak")
            if backup.exists():
                backup.unlink()
            with self._db.connect() as conn:
                self._db.delete_tile(conn, tid)

        self._refresh_tile_list()
        self.set_status(f"Deleted {len(tile_ids)} tile(s)", timeout=5000)

    # ── helpers ────────────────────────────────────────────────────

    def _apply_class_filter(self, point_data: dict) -> dict:
        """Return a filtered copy of *point_data* with only visible classes."""
        cls = point_data.get("classification")
        if cls is None or self.class_visibility.all():
            return point_data
        visible = self.class_visibility[cls]
        if visible.all():
            return point_data
        filtered = {"x": point_data["x"][visible],
                    "y": point_data["y"][visible],
                    "z": point_data["z"][visible]}
        for key in ("classification", "intensity", "return_number",
                    "num_returns", "point_source_id", "scan_direction_flag",
                    "edge_of_flight_line", "scan_angle_rank", "user_data",
                    "gps_time", "red", "green", "blue",
                    "key_point", "synthetic", "withheld", "overlap",
                    "sensor_type"):
            if key in point_data:
                filtered[key] = point_data[key][visible]
        return filtered

    def _multi_load_for_edit(self, point_data: dict) -> None:
        """
        Refresh 3D + DTM views after an edit, preserving the profile view
        if a profile is already loaded in the editor.
        """
        # Keep tile data in sync for the info button
        self._properties_panel.set_tile_data(point_data)

        # Apply class visibility filter
        filtered = self._apply_class_filter(point_data)

        # Update 3D
        self._multi_view._view_3d.load_point_cloud(
            filtered["x"], filtered["y"], filtered["z"],
            filtered.get("classification"),
            filtered.get("intensity"),
            filtered.get("return_number"),
            filtered.get("point_source_id"),
        )
        # Update DTM
        self._multi_view._view_dtm.load_points(filtered)

        # Refresh profile view if a profile exists in the editor
        profile = self._editor.profile
        if profile is not None and len(profile.distances) > 0:
            new_cls = point_data["classification"][profile.indices]
            self._multi_view._view_profile.set_profile_data(
                profile.distances,
                profile.elevations,
                new_cls,
                intensities=profile.intensities,
                indices=profile.indices,
                xs=profile.xs,
                ys=profile.ys,
                zs=profile.elevations,
            )
            self._multi_view._view_profile.set_class_visibility(self.class_visibility)
            if self._dtm_ref_distances is not None:
                self._multi_view._view_profile.set_dtm_reference(
                    self._dtm_ref_distances, self._dtm_ref_elevations
                )

    def _start_import(self, directory: str) -> None:
        """Launch the import wizard with a pre-selected directory."""
        if not self._pm.is_open:
            QMessageBox.information(
                self, "No Project Open",
                "Please create or open a project before importing data."
            )
            return
        wizard = ImportWizard(self._tm, parent=self, preselected_dir=directory)
        if wizard.exec() == ImportWizard.Accepted:
            self._refresh_tile_list()
            self.set_status(
                f"Imported {len(wizard.imported_tile_ids)} tile(s)", timeout=5000
            )

    def _sync_db(self) -> None:
        """Sync MainWindow and TileManager DB to match ProjectManager's DB.

        After :meth:`ProjectManager.create` or :meth:`ProjectManager.open`,
        the project manager holds a new on-disk database — but MainWindow
        and TileManager still reference the old (often ``:memory:``) one.
        This method rebinds both to the live project database.
        """
        if self._pm.db is not None:
            self._db = self._pm.db
            self._tm._db = self._pm.db
            logger.debug("DB synced to project database")

    def _refresh_tile_list(self) -> None:
        """Reload tiles from the database into the tile list widget."""
        if self._pm.is_open and self._db is not None:
            tiles = self._db.get_all_tiles()
            self._tile_list_widget.set_tiles(tiles)
            self.set_status(f"Loaded {len(tiles)} tile(s)", timeout=3000)
        else:
            self._tile_list_widget.set_tiles([])

    def _populate_las_header(self, tile_id: str) -> None:
        """Read the LAS file header for version/CRS info and update metadata panel."""
        tile_info = self._db.get_tile(tile_id)
        if tile_info is None:
            return
        tiles_dir = self._pm.tiles_dir
        if tiles_dir is None:
            return
        las_path = tiles_dir / tile_info["filename"]
        if not las_path.is_file():
            return
        try:
            import laspy

            # Parse all_scanners and flightline_sensor_types from DB
            all_scanners_raw = tile_info.get("all_scanners", "[]")
            try:
                all_scanners = json.loads(all_scanners_raw) if isinstance(all_scanners_raw, str) else all_scanners_raw
            except Exception:
                all_scanners = []
            fl_sensor_raw = tile_info.get("flightline_sensor_types", "{}")
            try:
                fl_sensor_types = json.loads(fl_sensor_raw) if isinstance(fl_sensor_raw, str) else fl_sensor_raw
            except Exception:
                fl_sensor_types = {}

            with laspy.open(las_path) as reader:
                hdr = reader.header
                las_ver = f"{hdr.version.major}.{hdr.version.minor}"
                # Try to extract CRS from VLRs (WKT or GeoTIFF)
                crs_str = ""
                try:
                    for vlr in getattr(hdr, 'vlrs', []):
                        rec_id = getattr(vlr, 'record_id', None)
                        if rec_id == 2112:  # WKT OGC CS
                            crs_str = str(vlr.record_data.decode('utf-8', errors='replace'))[:80]
                            break
                except Exception:
                    pass
                # Fall back to DB CRS if LAS file has no CRS VLR
                if not crs_str:
                    crs_epsg = tile_info.get("crs_epsg")
                    if crs_epsg:
                        crs_str = f"EPSG:{crs_epsg} (assigned)"
                self._properties_panel.set_tile_metadata(
                    scanner=tile_info.get("scanner", ''),
                    all_scanners=all_scanners,
                    sensor_type=tile_info.get("sensor_type", ''),
                    flight_line=tile_info.get("flight_line", 0),
                    flightline_sensor_types=fl_sensor_types,
                    las_version=las_ver,
                    point_count=tile_info.get("point_count", 0),
                    crs=crs_str,
                    filename=tile_info.get("filename", ""),
                    modified=self._file_modified(tile_info),
                )
        except Exception:
            pass

    # ── recent projects ─────────────────────────────────────────

    _RECENT_FILE = ".recent_projects.json"
    _MAX_RECENT = 8

    @classmethod
    def _load_recent_projects(cls) -> list[str]:
        try:
            with open(cls._RECENT_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [p for p in data if Path(p).is_dir()]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return []

    @classmethod
    def _save_recent_projects(cls, paths: list[str]) -> None:
        try:
            with open(cls._RECENT_FILE, "w") as f:
                json.dump(paths[: cls._MAX_RECENT], f)
        except Exception:
            pass

    def _add_recent_project(self, path: str) -> None:
        recent = self._load_recent_projects()
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._save_recent_projects(recent)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._load_recent_projects()
        if not recent:
            noop = QAction("(No recent projects)", self)
            noop.setEnabled(False)
            self._recent_menu.addAction(noop)
            return
        for p in recent:
            action = QAction(Path(p).name, self)
            action.setToolTip(p)
            action.triggered.connect(lambda checked, path=p: self._open_recent(path))
            self._recent_menu.addAction(action)

    def _open_recent(self, path: str) -> None:
        try:
            self._pm.open(path)
            self._sync_db()
            self._refresh_tile_list()
            self._add_recent_project(path)
            self.set_status(f"Opened: {self._pm.metadata.get('name', path)}")
        except Exception as exc:
            logger.error("Failed to open recent project: %s", exc)
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{exc}")
