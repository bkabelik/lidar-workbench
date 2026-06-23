"""
LiDAR Workbench — Main Window.

The central QMainWindow that houses the three-panel layout (tile list,
multi-view container, properties panel), menu bar, and toolbar.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..config import APP_NAME, APP_VERSION, DEFAULT_PROFILE_WIDTH_M, TileStatus
from ..database import Database
from ..import_wizard import ImportWizard
from ..manual_edit import ManualEditor
from ..project_manager import ProjectManager
from ..tile_manager import TileManager
from .classification_dialog import ClassificationDialog
from .export_dialog import ExportDialog
from .filter_dialog import FilterDialog
from .multi_view_widget import MultiViewWidget
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

        # Cached profile line for width changes
        self._profile_start: Optional[tuple] = None
        self._profile_end: Optional[tuple] = None
        self._profile_width: float = DEFAULT_PROFILE_WIDTH_M

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

        export_action = QAction("&Export Raster (DTM / DSM)…", self)
        export_action.setObjectName("export_raster")
        export_action.triggered.connect(self._on_export_raster)
        tools_menu.addAction(export_action)

        tools_menu.addSeparator()

        settings_action = QAction("&Settings…", self)
        settings_action.triggered.connect(self._on_settings)
        tools_menu.addAction(settings_action)

        # ----- Help menu -----
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ── toolbar ────────────────────────────────────────────────────

    def _setup_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

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

    # ── central widget ─────────────────────────────────────────────

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
        splitter.addWidget(self._tile_list_widget)

        # Center panel: multi-view widget
        self._multi_view = MultiViewWidget()
        self._multi_view.profile_line_defined.connect(self._on_profile_line_defined)
        # Wire profile view selection → editor
        self._multi_view._view_profile.selection_changed.connect(self._on_profile_selection)
        # Wire profile view width change → re-extract
        self._multi_view._view_profile.profile_width_changed.connect(self._on_profile_width_changed)
        splitter.addWidget(self._multi_view)

        # Right panel: properties panel
        self._properties_panel = PropertiesPanel()
        self._properties_panel.classify_requested.connect(self._on_classify_selected)
        self._properties_panel.undo_requested.connect(self._on_undo)
        self._properties_panel.redo_requested.connect(self._on_redo)
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
        _make("next_tile", self._tile_list_widget.select_next_tile)
        _make("prev_tile", self._tile_list_widget.select_previous_tile)

        # Quick-classify shortcuts (emit directly to properties signal handler)
        _make("classify_ground", lambda: self._properties_panel.classify_requested.emit(2))
        _make("classify_low_veg", lambda: self._properties_panel.classify_requested.emit(3))
        _make("classify_med_veg", lambda: self._properties_panel.classify_requested.emit(4))
        _make("classify_high_veg", lambda: self._properties_panel.classify_requested.emit(5))
        _make("classify_building", lambda: self._properties_panel.classify_requested.emit(6))
        _make("classify_water", lambda: self._properties_panel.classify_requested.emit(9))
        _make("classify_noise", lambda: self._properties_panel.classify_requested.emit(7))
        _make("classify_unclass", lambda: self._properties_panel.classify_requested.emit(1))

    def _on_new_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Project Location"
        )
        if not directory:
            return
        try:
            proj_dir = Path(directory) / "lidar_project"
            self._pm.create(proj_dir, name="New Project")
            self._sync_db()
            self._refresh_tile_list()
            self._add_recent_project(str(proj_dir))
            self.set_status(f"Created project in {directory}")
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

    def _on_filter(self) -> None:
        """Open the noise filter dialog for selected tiles."""
        selected = self._tile_list_widget.get_selected_tile_ids()
        if not selected:
            QMessageBox.information(
                self, "No Tiles Selected",
                "Select one or more tiles in the tile list first."
            )
            return

        dialog = FilterDialog(self._tm, selected, parent=self)
        dialog.filter_applied.connect(self._on_filter_applied)
        dialog.exec()

    def _on_filter_applied(self, tile_ids: list, params: dict) -> None:
        """Apply the chosen filter to the selected tiles."""
        self.set_status(f"Applying {params.get('type', 'filter')} to {len(tile_ids)} tile(s)…", timeout=0)

        # Apply filter to each tile
        for tile_id in tile_ids:
            data = self._tm.load_tile_points_full(tile_id)
            if data is None:
                continue

            if params["type"] == "sor":
                from ..noise_filter import statistical_outlier_removal
                keep, _ = statistical_outlier_removal(
                    data["x"], data["y"], data["z"],
                    nb_neighbors=params.get("nb_neighbors", 20),
                    std_ratio=params.get("std_ratio", 2.0),
                )
            elif params["type"] == "ror":
                from ..noise_filter import radius_outlier_removal
                keep, _ = radius_outlier_removal(
                    data["x"], data["y"], data["z"],
                    radius=params.get("radius", 1.0),
                    min_points=params.get("min_points", 5),
                )
            else:
                from ..noise_filter import dbscan_outlier_removal
                mode = "above" if params["type"] == "dbscan_above" else "below"
                keep, _ = dbscan_outlier_removal(
                    data["x"], data["y"], data["z"],
                    eps=params.get("eps", 2.0),
                    min_samples=params.get("min_samples", 10),
                    min_cluster_size=params.get("min_cluster_size", 50),
                    mode=mode,
                )

            # Write filtered data back
            from ..noise_filter import apply_filter_to_tile
            filtered = apply_filter_to_tile(
                data["x"], data["y"], data["z"],
                data["classification"], data["intensity"], data["return_number"],
                keep,
            )
            # Re-write tile via tile_manager internal helper
            tiles_dir = self._pm.tiles_dir
            if tiles_dir is None:
                continue
            tile_info = self._db.get_tile(tile_id)
            if tile_info is None:
                continue
            from ..tile_manager import _write_las_file
            _write_las_file(
                tiles_dir / tile_info["filename"],
                *filtered[:3],
                classes=filtered[3],
                intensities=filtered[4],
                return_numbers=filtered[5],
            )
            self._tm.update_tile_status(tile_id, TileStatus.FILTERED)
            self._tile_list_widget.update_tile_status(tile_id, TileStatus.FILTERED)

        self._refresh_tile_list()
        self.set_status(f"Filter applied to {len(tile_ids)} tile(s)", timeout=5000)

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

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            f"<p>Interactive airborne LiDAR point cloud analysis and "
            f"classification tool.</p>"
            f"<p>Built with PySide6, Open3D, laspy, and Pointcept.</p>",
        )

    # ── slot: tile list signals ────────────────────────────────────

    def _on_tile_selected(self, tile_id: str) -> None:
        self.set_status(f"Selected: {tile_id}", timeout=0)
        # Show tile info in properties panel
        tile_info = self._db.get_tile(tile_id)
        if tile_info:
            self._properties_panel.set_selection_count(tile_info.get("point_count", 0))

    def _on_tile_open(self, tile_id: str) -> None:
        """Open a tile in the multi-view for inspection and editing."""
        self.set_status(f"Opening {tile_id}…", timeout=0)
        logger.info("Open requested for tile: %s", tile_id)

        # Load full point data
        data = self._tm.load_tile_points_full(tile_id)
        if data is None:
            QMessageBox.warning(self, "Load Error", f"Failed to load tile {tile_id}.")
            return

        # Open in editor
        if not self._editor.open_tile(tile_id):
            QMessageBox.warning(self, "Edit Error", f"Failed to open tile {tile_id} for editing.")
            return

        # Load into multi-view
        self._multi_view.load_tile(tile_id, data)
        self._properties_panel.set_undo_info(*self._editor.undo_stack_info)
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

        self.set_status(
            f"Profile: {len(profile.distances)} pts / {profile.distances[-1]:.1f} m "
            f"(width={self._profile_width:.1f} m, scroll to adjust, click to confirm)",
            timeout=8000,
        )

    def _on_profile_selection(self, mask: np.ndarray) -> None:
        """Called when the user makes a selection in the profile view."""
        # Store selection in the editor so classify buttons can use it
        self._editor.set_selection(mask)
        count = self._editor.selected_count
        self._properties_panel.set_selection_count(count)

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
            self.set_status(f"Redo: {desc}", timeout=3000)

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

    def _multi_load_for_edit(self, point_data: dict) -> None:
        """
        Refresh 3D + DTM views after an edit, preserving the profile view
        if a profile is already loaded in the editor.
        """
        # Update 3D
        self._multi_view._view_3d.load_point_cloud(
            point_data["x"], point_data["y"], point_data["z"],
            point_data.get("classification"),
            point_data.get("intensity"),
            point_data.get("return_number"),
        )
        # Update DTM
        self._multi_view._view_dtm.load_points(point_data)

        # Refresh profile view if a profile exists in the editor
        profile = self._editor.profile
        if profile is not None and len(profile.distances) > 0:
            # Get updated classifications for the profile points
            new_cls = point_data["classification"][profile.indices]
            self._multi_view._view_profile.set_profile_data(
                profile.distances,
                profile.elevations,
                new_cls,
            )
            # Restore DTM reference line
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
