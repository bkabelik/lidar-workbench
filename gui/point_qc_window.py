"""
point_qc_window.py - LiDAR Quality Control & High-Performance Raster/Vector Map Viewer.

Features:
- High-performance 2D GIS visualization using PyQtGraph with 1:1 spatial aspect ratio.
- Decimated & windowed streaming of massive project GeoTIFFs (60 FPS pan/zoom).
- Native LiDAR QC generation: Point Density (pts/m²), Point Spacing (m), and Strip Differences (dZ).
- Automatic vector tile grid generation with dynamic text label scaling.
- Layer management with hierarchical folder grouping, open/close file lifecycle, and SSD workspace persistence.
- Interactive crosshairs displaying coordinate (X, Y) and pixel value at cursor.
- Click-to-jump to 3D LiDAR view and box-selection for multi-tile bulk loading.
"""

from __future__ import annotations

import os
# Ensure PyQtGraph binds to PySide6
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QFont, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..point_qc import (
    _get_tile_bbox,
    generate_density_raster,
    generate_spacing_raster,
    generate_strip_difference_raster,
    generate_tile_grid_vector,
)
from .point_qc_layers import (
    LayerGroup,
    RasterLayer,
    VectorFeature,
    VectorLayer,
    load_layer_workspace,
    save_layer_workspace,
)

logger = logging.getLogger(__name__)


# ── QC Worker Thread ────────────────────────────────────────────────────────

class _QCWorker(QThread):
    """Background worker for computing QC rasters without blocking GUI."""
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        task_type: str,  # "density", "spacing", "strip_diff", "tile_grid"
        tile_manager: Any,
        database: Any,
        params: dict,
    ):
        super().__init__()
        self.task_type = task_type
        self.tile_manager = tile_manager
        self.database = database
        self.params = params

    def run(self) -> None:
        try:
            if self.task_type == "density":
                res = generate_density_raster(
                    tile_manager=self.tile_manager,
                    database=self.database,
                    tile_ids=self.params.get("tile_ids"),
                    cell_size=self.params.get("cell_size", 1.0),
                    output_path=self.params.get("output_path"),
                    return_filter=self.params.get("return_filter", "all"),
                    progress_callback=self._on_progress,
                )
                self.finished.emit(res)
            elif self.task_type == "spacing":
                res = generate_spacing_raster(
                    density_raster_path=self.params.get("density_path"),
                    output_path=self.params.get("output_path"),
                )
                self.finished.emit(res)
            elif self.task_type == "strip_diff":
                res = generate_strip_difference_raster(
                    tile_manager=self.tile_manager,
                    database=self.database,
                    tile_ids=self.params.get("tile_ids"),
                    strip_a=self.params.get("strip_a"),
                    strip_b=self.params.get("strip_b"),
                    cell_size=self.params.get("cell_size", 1.0),
                    output_path=self.params.get("output_path"),
                    progress_callback=self._on_progress,
                )
                self.finished.emit(res)
            elif self.task_type == "tile_grid":
                res = generate_tile_grid_vector(
                    database=self.database,
                    output_path=self.params.get("output_path"),
                )
                self.finished.emit(res)
        except Exception as exc:
            logger.exception("Error running QC task %s", self.task_type)
            self.error.emit(str(exc))

    def _on_progress(self, curr: int, total: int, msg: str) -> None:
        self.progress.emit(curr, total, msg)


# ── Custom ViewBox for Navigation & Selection ───────────────────────────────

class QCViewBox(pg.ViewBox):
    """Custom ViewBox with modes: Pan, Click-Jump, and Rubberband Box Select."""
    tile_clicked = Signal(float, float)
    box_selected = Signal(float, float, float, float)
    mouse_moved = Signal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "pan"  # "pan", "click_jump", "box_select"
        self._selection_rect_item = None
        self._drag_start_pos = None

    def mousePressEvent(self, ev):
        if self.mode == "click_jump" and ev.button() == Qt.LeftButton:
            pt = self.mapSceneToView(ev.scenePos())
            self.tile_clicked.emit(pt.x(), pt.y())
            ev.accept()
            return
        elif self.mode == "box_select" and ev.button() == Qt.LeftButton:
            self._drag_start_pos = self.mapSceneToView(ev.scenePos())
            if self._selection_rect_item is None:
                self._selection_rect_item = QGraphicsRectItem()
                pen = QPen(QColor(0, 220, 255, 230), 1.5, Qt.DashLine)
                self._selection_rect_item.setPen(pen)
                self._selection_rect_item.setBrush(QColor(0, 220, 255, 40))
                self.addItem(self._selection_rect_item)
            self._selection_rect_item.setRect(
                QRectF(self._drag_start_pos.x(), self._drag_start_pos.y(), 0, 0)
            )
            self._selection_rect_item.show()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pt = self.mapSceneToView(ev.scenePos())
        self.mouse_moved.emit(pt.x(), pt.y())
        if self.mode == "box_select" and self._drag_start_pos is not None:
            x0 = min(self._drag_start_pos.x(), pt.x())
            y0 = min(self._drag_start_pos.y(), pt.y())
            w = abs(pt.x() - self._drag_start_pos.x())
            h = abs(pt.y() - self._drag_start_pos.y())
            if self._selection_rect_item:
                self._selection_rect_item.setRect(QRectF(x0, y0, w, h))
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.mode == "box_select" and self._drag_start_pos is not None and ev.button() == Qt.LeftButton:
            pt = self.mapSceneToView(ev.scenePos())
            x0 = min(self._drag_start_pos.x(), pt.x())
            x1 = max(self._drag_start_pos.x(), pt.x())
            y0 = min(self._drag_start_pos.y(), pt.y())
            y1 = max(self._drag_start_pos.y(), pt.y())
            self._drag_start_pos = None
            self.box_selected.emit(x0, y0, x1, y1)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


# ── Main Point QC Window ───────────────────────────────────────────────────

class PointQCWindow(QMainWindow):
    """
    High-performance 2D GIS raster and vector viewer with built-in LiDAR QC tools.
    """
    jump_to_tile = Signal(str)
    bulk_load_tiles = Signal(list)

    def __init__(
        self,
        tile_manager: Any = None,
        database: Any = None,
        project_dir: Optional[str] = None,
        selected_tile_ids: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.tile_manager = tile_manager
        self.database = database
        self.project_dir = Path(project_dir) if project_dir else Path(".")
        self.selected_tile_ids = selected_tile_ids or []

        self.setWindowTitle("Point QC & Map Viewer — LiDAR Workbench")
        self.resize(1350, 850)

        # Layer storage
        self.groups: List[LayerGroup] = [
            LayerGroup("QC Rasters"),
            LayerGroup("Vector Grids"),
            LayerGroup("Basemaps"),
        ]
        self._tile_grid_layer: Optional[VectorLayer] = None
        self._active_raster_layer: Optional[RasterLayer] = None

        # Graphics item caches
        self._raster_items: Dict[str, pg.ImageItem] = {}
        self._vector_path_items: Dict[str, QGraphicsPathItem] = {}
        self._vector_text_items: Dict[str, List[pg.TextItem]] = {}
        self._highlight_items: List[QGraphicsRectItem] = []

        self._active_selected_tids: List[str] = []
        self._qc_worker: Optional[_QCWorker] = None

        # Build UI
        self._setup_toolbar()
        self._setup_central_canvas()
        self._setup_left_dock()
        self._setup_status_bar()

        # Check for existing tile grid or QC files in project
        self._scan_and_load_default_layers()

        # Drag and Drop support (handles dropped rasters, vectors, workspaces)
        self.setAcceptDrops(True)
        self._plot_widget.setAcceptDrops(True)
        self._plot_widget.viewport().setAcceptDrops(True)
        self._plot_widget.installEventFilter(self)
        self._plot_widget.viewport().installEventFilter(self)
        self._tree.setAcceptDrops(True)
        self._tree.installEventFilter(self)

    # ── UI Construction ─────────────────────────────────────────────────────

    def _setup_toolbar(self) -> None:
        tb = self.addToolBar("Navigation")
        tb.setMovable(False)

        # Mode group
        self._pan_action = QAction("✋ Pan / Zoom", self)
        self._pan_action.setCheckable(True)
        self._pan_action.setChecked(True)
        self._pan_action.triggered.connect(lambda: self._set_mode("pan"))
        tb.addAction(self._pan_action)

        self._jump_action = QAction("🎯 Click to Jump", self)
        self._jump_action.setCheckable(True)
        self._jump_action.setToolTip("Click a tile to identify and inspect in 3D")
        self._jump_action.triggered.connect(lambda: self._set_mode("click_jump"))
        tb.addAction(self._jump_action)

        self._box_action = QAction("⛶ Box Select", self)
        self._box_action.setCheckable(True)
        self._box_action.setToolTip("Drag a box across multiple tiles for bulk 3D loading")
        self._box_action.triggered.connect(lambda: self._set_mode("box_select"))
        tb.addAction(self._box_action)

        tb.addSeparator()

        zoom_all_action = QAction("🔍 Zoom to Extent", self)
        zoom_all_action.setToolTip("Zoom canvas to full project bounds")
        zoom_all_action.triggered.connect(self._on_zoom_to_extent)
        tb.addAction(zoom_all_action)

        refresh_action = QAction("🔄 Refresh View", self)
        refresh_action.setToolTip("Reload and refresh visible layers")
        refresh_action.triggered.connect(self._render_all_layers)
        tb.addAction(refresh_action)

        tb.addSeparator()

        grid_action = QAction("▦ Generate Tile Grid", self)
        grid_action.setToolTip("Create a vector grid of all project tiles from the database")
        grid_action.triggered.connect(self._on_generate_tile_grid)
        tb.addAction(grid_action)

    def _setup_central_canvas(self) -> None:
        self._view_box = QCViewBox()
        self._view_box.setAspectLocked(True)

        self._plot_widget = pg.PlotWidget(viewBox=self._view_box)
        self._plot_widget.setBackground("#181818")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._plot_widget.setLabel("bottom", "Easting (m)")
        self._plot_widget.setLabel("left", "Northing (m)")

        self._view_box.tile_clicked.connect(self._on_canvas_tile_clicked)
        self._view_box.box_selected.connect(self._on_canvas_box_selected)
        self._view_box.mouse_moved.connect(self._on_canvas_mouse_moved)
        self._view_box.sigRangeChanged.connect(self._on_view_range_changed)

        self.setCentralWidget(self._plot_widget)

    def _setup_left_dock(self) -> None:
        dock = QDockWidget("Point QC & Layers", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self._tab_widget = QTabWidget()

        # Tab 1: Layers
        layer_tab = QWidget()
        layer_layout = QVBoxLayout(layer_tab)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Layer / Group", "Type"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        self._tree.itemSelectionChanged.connect(self._on_layer_selection_changed)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        layer_layout.addWidget(self._tree)

        # Layer action buttons
        btn_box1 = QHBoxLayout()
        add_raster_btn = QPushButton("+ Raster…")
        add_raster_btn.clicked.connect(self._on_add_raster_dialog)
        add_vec_btn = QPushButton("+ Vector…")
        add_vec_btn.clicked.connect(self._on_add_vector_dialog)
        new_grp_btn = QPushButton("+ Folder")
        new_grp_btn.clicked.connect(self._on_add_group_dialog)
        btn_box1.addWidget(add_raster_btn)
        btn_box1.addWidget(add_vec_btn)
        btn_box1.addWidget(new_grp_btn)
        layer_layout.addLayout(btn_box1)

        btn_box2 = QHBoxLayout()
        del_layer_btn = QPushButton("✕ Close / Remove")
        del_layer_btn.clicked.connect(self._on_remove_selected_layer)
        save_ws_btn = QPushButton("💾 Save State")
        save_ws_btn.clicked.connect(self._on_save_workspace)
        open_ws_btn = QPushButton("📂 Open State")
        open_ws_btn.clicked.connect(self._on_open_workspace)
        btn_box2.addWidget(del_layer_btn)
        btn_box2.addWidget(save_ws_btn)
        btn_box2.addWidget(open_ws_btn)
        layer_layout.addLayout(btn_box2)

        # Layer Properties Group
        prop_group = QGroupBox("Selected Layer Style")
        prop_layout = QFormLayout(prop_group)

        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        prop_layout.addRow("Opacity:", self._opacity_slider)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["viridis", "turbo", "coolwarm", "plasma", "inferno", "gray", "terrain"])
        self._cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        prop_layout.addRow("Colormap:", self._cmap_combo)

        self._val_range_lbl = QLabel("Range: N/A")
        prop_layout.addRow("Values:", self._val_range_lbl)
        layer_layout.addWidget(prop_group)

        self._tab_widget.addTab(layer_tab, "Layers")

        # Tab 2: QC Generator
        qc_tab = QWidget()
        qc_layout = QVBoxLayout(qc_tab)

        qc_form = QFormLayout()
        self._qc_type_combo = QComboBox()
        self._qc_type_combo.addItems([
            "Point Density (pts/m²)",
            "Nominal Point Spacing (m)",
            "Strip Differences (dZ raster)",
        ])
        self._qc_type_combo.currentIndexChanged.connect(self._on_qc_type_changed)
        qc_form.addRow("QC Product:", self._qc_type_combo)

        self._cell_size_spin = QDoubleSpinBox()
        self._cell_size_spin.setRange(0.1, 10.0)
        self._cell_size_spin.setSingleStep(0.25)
        self._cell_size_spin.setValue(1.0)
        self._cell_size_spin.setSuffix(" m")
        qc_form.addRow("Grid Resolution:", self._cell_size_spin)

        # Processing Scope
        scope_box = QHBoxLayout()
        self._scope_all_radio = QRadioButton("All Project Tiles")
        self._scope_sel_radio = QRadioButton("Selected Tiles")
        self._scope_all_radio.setChecked(True)
        scope_box.addWidget(self._scope_all_radio)
        scope_box.addWidget(self._scope_sel_radio)
        qc_form.addRow("Scope:", scope_box)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All Returns", "First Returns Only", "Ground Returns Only"])
        qc_form.addRow("Return Filter:", self._filter_combo)

        # Strip diff options
        self._strip_a_combo = QComboBox()
        self._strip_b_combo = QComboBox()
        self._strip_a_lbl = QLabel("Strip A (Ref):")
        self._strip_b_lbl = QLabel("Strip B (Compare):")
        qc_form.addRow(self._strip_a_lbl, self._strip_a_combo)
        qc_form.addRow(self._strip_b_lbl, self._strip_b_combo)
        self._strip_a_lbl.hide()
        self._strip_a_combo.hide()
        self._strip_b_lbl.hide()
        self._strip_b_combo.hide()

        qc_layout.addLayout(qc_form)

        # Progress bar & Generate button
        self._qc_prog = QProgressBar()
        self._qc_prog.setTextVisible(True)
        self._qc_prog.hide()
        qc_layout.addWidget(self._qc_prog)

        self._qc_status_lbl = QLabel("")
        self._qc_status_lbl.setWordWrap(True)
        qc_layout.addWidget(self._qc_status_lbl)

        self._generate_qc_btn = QPushButton("⚡ Generate QC Raster")
        self._generate_qc_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        self._generate_qc_btn.clicked.connect(self._on_start_qc_generation)
        qc_layout.addWidget(self._generate_qc_btn)

        qc_layout.addStretch()
        self._tab_widget.addTab(qc_tab, "QC Tools")

        # Tab 3: Tile Grid & 3D Jump
        jump_tab = QWidget()
        jump_layout = QVBoxLayout(jump_tab)

        grid_group = QGroupBox("Tile Grid Vector & Overlap")
        grid_form = QFormLayout(grid_group)

        grid_mode_box = QHBoxLayout()
        self._grid_full_radio = QRadioButton("Full Grid (All together)")
        self._grid_loaded_radio = QRadioButton("Loaded Tiles Only")
        self._grid_full_radio.setChecked(True)
        self._grid_full_radio.toggled.connect(lambda: self._on_generate_tile_grid())
        grid_mode_box.addWidget(self._grid_full_radio)
        grid_mode_box.addWidget(self._grid_loaded_radio)
        grid_form.addRow("Grid Mode:", grid_mode_box)

        self._show_labels_chk = QCheckBox("Show Tile ID Labels")
        self._show_labels_chk.setChecked(True)
        self._show_labels_chk.toggled.connect(self._on_toggle_tile_labels)
        grid_form.addRow(self._show_labels_chk)

        self._font_size_spin = QSpinBox()
        self._font_size_spin.setRange(6, 48)
        self._font_size_spin.setValue(10)
        self._font_size_spin.setSuffix(" pt")
        self._font_size_spin.valueChanged.connect(self._on_label_font_size_changed)
        grid_form.addRow("Label Font Size:", self._font_size_spin)

        regen_grid_btn = QPushButton("▦ Generate / Update Tile Grid")
        regen_grid_btn.clicked.connect(self._on_generate_tile_grid)
        grid_form.addRow(regen_grid_btn)
        jump_layout.addWidget(grid_group)

        # Selection info
        sel_group = QGroupBox("Active Tile Selection")
        sel_form = QFormLayout(sel_group)

        self._sel_summary_lbl = QLabel("No tiles selected.\n(Click a tile or use Box Select)")
        self._sel_summary_lbl.setWordWrap(True)
        sel_form.addRow(self._sel_summary_lbl)

        self._jump_btn = QPushButton("🚀 Open Tile in 3D View")
        self._jump_btn.setEnabled(False)
        self._jump_btn.clicked.connect(self._on_jump_to_first_selected)
        sel_form.addRow(self._jump_btn)

        self._bulk_btn = QPushButton("📦 Bulk Load Selected Tiles in 3D")
        self._bulk_btn.setEnabled(False)
        self._bulk_btn.clicked.connect(self._on_bulk_load_selected)
        sel_form.addRow(self._bulk_btn)

        jump_layout.addWidget(sel_group)
        jump_layout.addStretch()

        self._tab_widget.addTab(jump_tab, "Tile & Jump")

        dock.setWidget(self._tab_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _setup_status_bar(self) -> None:
        sb = QStatusBar(self)
        self._coord_lbl = QLabel("E: --- | N: ---")
        self._val_lbl = QLabel("Value: ---")
        sb.addPermanentWidget(self._coord_lbl)
        sb.addPermanentWidget(self._val_lbl)
        self.setStatusBar(sb)

    # ── Modes & Tool Handling ────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._view_box.mode = mode
        self._pan_action.setChecked(mode == "pan")
        self._jump_action.setChecked(mode == "click_jump")
        self._box_action.setChecked(mode == "box_select")

        if mode == "pan":
            self.statusBar().showMessage("Mode: Pan / Zoom (drag to pan, scroll to zoom)", 3000)
        elif mode == "click_jump":
            self.statusBar().showMessage("Mode: Click to Jump (click any tile to inspect in 3D)", 4000)
        elif mode == "box_select":
            self.statusBar().showMessage("Mode: Box Select (drag a box over multiple tiles)", 4000)

    def _on_canvas_mouse_moved(self, x: float, y: float) -> None:
        self._coord_lbl.setText(f"E: {x:.2f} m | N: {y:.2f} m")
        # Query active raster value
        if self._active_raster_layer and self._active_raster_layer.visible:
            val = self._active_raster_layer.get_value_at(x, y)
            if val is not None:
                self._val_lbl.setText(f"Value: {val:.3f}")
            else:
                self._val_lbl.setText("Value: NoData")
        else:
            self._val_lbl.setText("Value: ---")

    def _on_canvas_tile_clicked(self, x: float, y: float) -> None:
        """Find tile at (x, y) and select it."""
        if not self._tile_grid_layer:
            # Fall back to database query
            if self.database:
                tiles = self.database.get_tiles_in_bbox(x, y, x, y)
                if tiles:
                    tid = str(tiles[0]["id"])
                    self._set_selected_tiles([tid])
                    return
            return

        feat = self._tile_grid_layer.find_feature_at(x, y)
        if feat:
            tid = feat.fid or feat.properties.get("tile_id", "")
            self._set_selected_tiles([tid])
            self.statusBar().showMessage(f"Selected tile: {tid}. Click 'Open Tile in 3D' to inspect.", 5000)

    def _on_canvas_box_selected(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Find all tiles intersecting the selection bounding box."""
        if self._tile_grid_layer:
            feats = self._tile_grid_layer.find_features_in_bbox(x0, y0, x1, y1)
            tids = [f.fid or f.properties.get("tile_id", "") for f in feats if f.fid or f.properties.get("tile_id")]
        elif self.database:
            tiles = self.database.get_tiles_in_bbox(x0, y0, x1, y1)
            tids = [str(t["id"]) for t in tiles]
        else:
            tids = []

        self._set_selected_tiles(tids)
        self.statusBar().showMessage(f"Box selected {len(tids)} tile(s).", 5000)

    def _set_selected_tiles(self, tids: List[str]) -> None:
        self._active_selected_tids = tids
        # Clear previous highlight rects
        for item in self._highlight_items:
            self._view_box.removeItem(item)
        self._highlight_items.clear()

        if not tids:
            self._sel_summary_lbl.setText("No tiles selected.\n(Click a tile or use Box Select)")
            self._jump_btn.setEnabled(False)
            self._bulk_btn.setEnabled(False)
            return

        total_pts = 0
        tile_dict = {}
        if self.database:
            all_t = self.database.get_all_tiles()
            tile_dict = {str(t["id"]): t for t in all_t}

        for tid in tids:
            t_info = tile_dict.get(tid)
            if t_info:
                pts = t_info.get("point_count", 0)
                total_pts += pts
                bbox = _get_tile_bbox(t_info)
                if bbox is not None:
                    min_x, min_y, max_x, max_y = bbox
                    rect_item = QGraphicsRectItem(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))
                    pen = QPen(QColor(0, 255, 255, 240), 2.5)
                    rect_item.setPen(pen)
                    rect_item.setBrush(QColor(0, 255, 255, 50))
                    self._view_box.addItem(rect_item)
                    self._highlight_items.append(rect_item)

        if len(tids) == 1:
            self._sel_summary_lbl.setText(f"Tile: {tids[0]}\nPoints: {total_pts:,}")
            self._jump_btn.setEnabled(True)
            self._bulk_btn.setEnabled(False)
        else:
            self._sel_summary_lbl.setText(f"{len(tids)} Tiles Selected\nTotal Points: {total_pts:,}")
            self._jump_btn.setEnabled(True)
            self._bulk_btn.setEnabled(True)

    def _on_jump_to_first_selected(self) -> None:
        if self._active_selected_tids:
            tid = self._active_selected_tids[0]
            self.jump_to_tile.emit(tid)
            self.statusBar().showMessage(f"Opened {tid} in 3D editor.", 4000)

    def _on_bulk_load_selected(self) -> None:
        if self._active_selected_tids:
            self.bulk_load_tiles.emit(self._active_selected_tids)
            self.statusBar().showMessage(f"Bulk loading {len(self._active_selected_tids)} tiles in 3D.", 4000)

    # ── Layer Management & Rendering ────────────────────────────────────────

    def _populate_tree(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        for grp in self.groups:
            grp_item = QTreeWidgetItem([grp.name, "Folder"])
            grp_item.setCheckState(0, Qt.Checked if grp.visible else Qt.Unchecked)
            grp_item.setData(0, Qt.UserRole, grp)

            for child in grp.children:
                if isinstance(child, RasterLayer):
                    c_item = QTreeWidgetItem([child.name, "Raster (GeoTIFF)"])
                    c_item.setCheckState(0, Qt.Checked if child.visible else Qt.Unchecked)
                    c_item.setData(0, Qt.UserRole, child)
                    grp_item.addChild(c_item)
                elif isinstance(child, VectorLayer):
                    c_item = QTreeWidgetItem([child.name, "Vector"])
                    c_item.setCheckState(0, Qt.Checked if child.visible else Qt.Unchecked)
                    c_item.setData(0, Qt.UserRole, child)
                    grp_item.addChild(c_item)

            self._tree.addTopLevelItem(grp_item)
            grp_item.setExpanded(True)
        self._tree.blockSignals(False)

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        data = item.data(0, Qt.UserRole)
        is_checked = item.checkState(0) == Qt.Checked

        if isinstance(data, (RasterLayer, VectorLayer)):
            data.visible = is_checked
            self._render_all_layers()
        elif isinstance(data, LayerGroup):
            data.visible = is_checked
            for i in range(item.childCount()):
                child_item = item.child(i)
                child_item.setCheckState(0, Qt.Checked if is_checked else Qt.Unchecked)
            self._render_all_layers()

    def _on_layer_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        obj = items[0].data(0, Qt.UserRole)
        if isinstance(obj, RasterLayer):
            self._active_raster_layer = obj
            self._opacity_slider.setValue(int(obj.opacity * 100))
            idx = self._cmap_combo.findText(obj.colormap_name)
            if idx >= 0:
                self._cmap_combo.setCurrentIndex(idx)
            if obj.vmin is not None and obj.vmax is not None:
                self._val_range_lbl.setText(f"Min: {obj.vmin:.2f} | Max: {obj.vmax:.2f}")
        elif isinstance(obj, VectorLayer):
            self._opacity_slider.setValue(int(obj.opacity * 100))
            self._val_range_lbl.setText(f"Features: {len(obj.features)}")

    def _on_opacity_slider_changed(self, val: int) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        obj = items[0].data(0, Qt.UserRole)
        if isinstance(obj, (RasterLayer, VectorLayer)):
            obj.opacity = val / 100.0
            self._render_all_layers()

    def _on_colormap_changed(self, cmap_name: str) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        obj = items[0].data(0, Qt.UserRole)
        if isinstance(obj, RasterLayer):
            obj.colormap_name = cmap_name
            self._render_all_layers()

    def _on_tree_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if not item:
            return
        obj = item.data(0, Qt.UserRole)
        menu = QMenu(self)

        zoom_act = menu.addAction("Zoom to Layer")
        zoom_act.triggered.connect(lambda: self._zoom_to_layer(obj))

        menu.addSeparator()
        remove_act = menu.addAction("Close / Remove Layer")
        remove_act.triggered.connect(self._on_remove_selected_layer)

        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def _zoom_to_layer(self, layer: Any) -> None:
        if hasattr(layer, "bounds") and layer.bounds:
            x0, y0, x1, y1 = layer.bounds
            self._view_box.setRange(QRectF(x0, y0, x1 - x0, y1 - y0), padding=0.05)

    def _on_zoom_to_extent(self) -> None:
        # Determine global bounding box across all loaded layers
        all_bounds = []
        for grp in self.groups:
            for l in grp.get_all_layers():
                if l.bounds:
                    all_bounds.append(l.bounds)

        if not all_bounds and self.database:
            tiles = self.database.get_all_tiles()
            t_bboxes = [_get_tile_bbox(t) for t in tiles]
            t_bboxes = [b for b in t_bboxes if b is not None]
            if t_bboxes:
                minx = min(b[0] for b in t_bboxes)
                miny = min(b[1] for b in t_bboxes)
                maxx = max(b[2] for b in t_bboxes)
                maxy = max(b[3] for b in t_bboxes)
                all_bounds.append((minx, miny, maxx, maxy))

        if all_bounds:
            minx = min(b[0] for b in all_bounds)
            miny = min(b[1] for b in all_bounds)
            maxx = max(b[2] for b in all_bounds)
            maxy = max(b[3] for b in all_bounds)
            self._view_box.setRange(QRectF(minx, miny, maxx - minx, maxy - miny), padding=0.05)

    def _on_view_range_changed(self) -> None:
        """Called dynamically on pan/zoom to fetch decimated raster viewports."""
        self._render_raster_layers()

    def _render_all_layers(self) -> None:
        self._render_raster_layers()
        self._render_vector_layers()

    def _render_raster_layers(self) -> None:
        if not hasattr(self, "_plot_widget") or self._plot_widget is None:
            return
        # Get current viewport bounds
        view_rect = self._view_box.viewRect()
        vx0, vy0, vx1, vy1 = view_rect.left(), view_rect.bottom(), view_rect.right(), view_rect.top()
        target_size = (max(200, int(self._plot_widget.width())), max(200, int(self._plot_widget.height())))

        for grp in self.groups:
            for l in grp.get_all_layers():
                if isinstance(l, RasterLayer):
                    img_item = self._raster_items.get(l.file_path.as_posix())
                    if not l.visible or not grp.visible:
                        if img_item:
                            img_item.hide()
                        continue

                    # Render decimated viewport
                    res = l.render_viewport((vx0, vy0, vx1, vy1), target_size=target_size)
                    if res is None:
                        if img_item:
                            img_item.hide()
                        continue

                    rgba, bounds = res
                    rx0, ry0, rx1, ry1 = bounds

                    if img_item is None:
                        img_item = pg.ImageItem()
                        self._view_box.addItem(img_item)
                        self._raster_items[l.file_path.as_posix()] = img_item

                    # Row 0 of rgba is top (ry1). To map to Y upward in Cartesian coords:
                    rgba_flipped = np.flipud(rgba)
                    # PyQtGraph ImageItem with shape (H, W, 4) requires transpose for standard (X, Y) display
                    # or axisOrder='row-major'
                    img_item.setImage(rgba_flipped.transpose(1, 0, 2), autoLevels=False)
                    img_item.setRect(QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0))
                    img_item.show()

    def _render_vector_layers(self) -> None:
        for grp in self.groups:
            for l in grp.get_all_layers():
                if isinstance(l, VectorLayer):
                    path_item = self._vector_path_items.get(l.file_path.as_posix())
                    text_items = self._vector_text_items.get(l.file_path.as_posix(), [])

                    if not l.visible or not grp.visible:
                        if path_item:
                            path_item.hide()
                        for t in text_items:
                            t.hide()
                        continue

                    # Create path item if not existing
                    empty_key = l.file_path.as_posix() + "_empty"
                    if path_item is None:
                        painter_path_loaded = QPainterPath()
                        painter_path_empty = QPainterPath()
                        has_empty = False

                        for f in l.features:
                            pts = f._extract_points(f.coordinates)
                            if len(pts) >= 2:
                                is_empty = f.properties.get("has_data") is False
                                target_p = painter_path_empty if is_empty else painter_path_loaded
                                if is_empty:
                                    has_empty = True
                                target_p.moveTo(pts[0][0], pts[0][1])
                                for pt in pts[1:]:
                                    target_p.lineTo(pt[0], pt[1])
                                target_p.closeSubpath()

                        # Loaded tiles item
                        path_item = QGraphicsPathItem(painter_path_loaded)
                        pen = QPen(QColor(*l.line_color), l.line_width)
                        path_item.setPen(pen)
                        self._view_box.addItem(path_item)
                        self._vector_path_items[l.file_path.as_posix()] = path_item

                        # Empty tiles item (if present)
                        if has_empty:
                            empty_item = QGraphicsPathItem(painter_path_empty)
                            empty_pen = QPen(QColor(130, 130, 130, 120), 1.0, Qt.DashLine)
                            empty_item.setPen(empty_pen)
                            self._view_box.addItem(empty_item)
                            self._vector_path_items[empty_key] = empty_item

                    path_item.show()
                    if empty_key in self._vector_path_items:
                        self._vector_path_items[empty_key].show()

                    # Render or update text labels
                    if not text_items:
                        font = QFont("SansSerif", l.label_font_size)
                        for f in l.features:
                            # Only render text label for loaded tiles
                            if f.properties.get("has_data") is False:
                                continue
                            tid = f.fid or f.properties.get(l.label_field, "")
                            if tid:
                                t_item = pg.TextItem(str(tid), color=l.label_color, anchor=(0.5, 0.5))
                                t_item.setFont(font)
                                t_item.setPos(f.centroid[0], f.centroid[1])
                                self._view_box.addItem(t_item)
                                text_items.append(t_item)
                        self._vector_text_items[l.file_path.as_posix()] = text_items

                    for t in text_items:
                        t.setVisible(l.show_labels)

    def _on_toggle_tile_labels(self, show: bool) -> None:
        if self._tile_grid_layer:
            self._tile_grid_layer.show_labels = show
            text_items = self._vector_text_items.get(self._tile_grid_layer.file_path.as_posix(), [])
            for t in text_items:
                t.setVisible(show)

    def _on_label_font_size_changed(self, size: int) -> None:
        if self._tile_grid_layer:
            self._tile_grid_layer.label_font_size = size
            font = QFont("SansSerif", size)
            text_items = self._vector_text_items.get(self._tile_grid_layer.file_path.as_posix(), [])
            for t in text_items:
                t.setFont(font)

    # ── File I/O & Layer Actions ─────────────────────────────────────────────

    def _on_add_raster_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GeoTIFF Raster", str(self.project_dir), "GeoTIFF (*.tif *.tiff)"
        )
        if path:
            self.add_raster_layer(path)

    def add_raster_layer(
        self, path: Path | str, group_name: str = "QC Rasters", render: bool = True, zoom: bool = True
    ) -> Optional[RasterLayer]:
        rl = RasterLayer(path)
        if not rl.is_open:
            QMessageBox.warning(self, "Failed to Open Raster", f"Could not load raster from {path}")
            return None

        # Find target group
        grp = next((g for g in self.groups if g.name == group_name), self.groups[0])
        grp.add_layer(rl)
        self._active_raster_layer = rl
        if render:
            self._populate_tree()
            self._render_all_layers()
            if zoom:
                self._zoom_to_layer(rl)
        return rl

    def _on_add_vector_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Vector File", str(self.project_dir), "Geospatial Vector (*.geojson *.json *.shp)"
        )
        if path:
            self.add_vector_layer(path)

    def add_vector_layer(
        self, path: Path | str, group_name: str = "Vector Grids", render: bool = True, zoom: bool = False
    ) -> Optional[VectorLayer]:
        vl = VectorLayer(path)
        grp = next((g for g in self.groups if g.name == group_name), self.groups[1])
        grp.add_layer(vl)
        if "tile_grid" in vl.name.lower():
            self._tile_grid_layer = vl
        if render:
            self._populate_tree()
            self._render_all_layers()
            if zoom:
                self._zoom_to_layer(vl)
        return vl

    # ── Drag and Drop Event Handlers ────────────────────────────────────────

    def _is_valid_drop(self, mime_data) -> bool:
        """Check if drag event contains supported geospatial raster or vector files."""
        if not mime_data or not mime_data.hasUrls():
            return False
        valid_exts = {
            ".tif", ".tiff", ".geotiff", ".vrt", ".asc", ".img",
            ".geojson", ".shp", ".json",
        }
        for url in mime_data.urls():
            if url.isLocalFile():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in valid_exts:
                    return True
        return False

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._is_valid_drop(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if self._is_valid_drop(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        mime = event.mimeData()
        if not mime.hasUrls():
            return

        paths: List[Path] = []
        for url in mime.urls():
            if url.isLocalFile():
                p = Path(url.toLocalFile())
                if p.is_file():
                    paths.append(p)

        if not paths:
            return

        event.acceptProposedAction()
        self.load_dropped_files(paths)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        etype = event.type()
        if etype == QEvent.DragEnter:
            if self._is_valid_drop(event.mimeData()):
                event.acceptProposedAction()
                return True
        elif etype == QEvent.DragMove:
            if self._is_valid_drop(event.mimeData()):
                event.acceptProposedAction()
                return True
        elif etype == QEvent.Drop:
            self.dropEvent(event)
            return True
        return super().eventFilter(watched, event)

    def load_dropped_files(self, paths: List[Path]) -> int:
        """
        Load dropped raster and vector files efficiently in a single batch pass,
        avoiding multiple redundant redraws to preserve high viewport FPS.
        """
        raster_exts = {".tif", ".tiff", ".geotiff", ".vrt", ".asc", ".img"}
        vector_exts = {".geojson", ".shp"}

        loaded = 0
        for p in paths:
            ext = p.suffix.lower()
            if ext in raster_exts:
                rl = self.add_raster_layer(p, group_name="QC Rasters", render=False)
                if rl:
                    loaded += 1
            elif ext in vector_exts:
                vl = self.add_vector_layer(p, group_name="Vector Grids", render=False)
                if vl:
                    loaded += 1
            elif ext == ".json":
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        snippet = f.read(500)
                    if "FeatureCollection" in snippet or "Feature" in snippet:
                        vl = self.add_vector_layer(p, group_name="Vector Grids", render=False)
                        if vl:
                            loaded += 1
                    elif "layers" in snippet:
                        grps = load_layer_workspace(p)
                        if grps:
                            self.groups = grps
                            loaded += 1
                except Exception as exc:
                    logger.warning("Failed to inspect dropped JSON file %s: %s", p, exc)

        if loaded > 0:
            self._populate_tree()
            self._render_all_layers()
            self._on_zoom_to_extent()
            msg = f"✓ Loaded {loaded} dropped layer(s)."
            self.statusBar().showMessage(msg, 5000)
            self._qc_status_lbl.setText(msg)

        return loaded

    def _on_add_group_dialog(self) -> None:
        grp = LayerGroup("New Folder")
        self.groups.append(grp)
        self._populate_tree()

    def _on_remove_selected_layer(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        obj = items[0].data(0, Qt.UserRole)
        for grp in self.groups:
            if grp.remove_layer(obj):
                break

        # Clean graphics items
        if isinstance(obj, (RasterLayer, VectorLayer)):
            key = obj.file_path.as_posix()
            if key in self._raster_items:
                self._view_box.removeItem(self._raster_items[key])
                del self._raster_items[key]
            if key in self._vector_path_items:
                self._view_box.removeItem(self._vector_path_items[key])
                del self._vector_path_items[key]
            if key in self._vector_text_items:
                for t in self._vector_text_items[key]:
                    self._view_box.removeItem(t)
                del self._vector_text_items[key]
            if obj == self._active_raster_layer:
                self._active_raster_layer = None
            if obj == self._tile_grid_layer:
                self._tile_grid_layer = None

        self._populate_tree()
        self._render_all_layers()

    def _on_save_workspace(self) -> None:
        ws_dir = self.project_dir / "qc"
        ws_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save QC Workspace", str(ws_dir / "workspace.json"), "JSON (*.json)"
        )
        if path:
            save_layer_workspace(self.groups, path)
            QMessageBox.information(self, "Workspace Saved", f"Layer workspace saved to:\n{path}")

    def _on_open_workspace(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open QC Workspace", str(self.project_dir / "qc"), "JSON (*.json)"
        )
        if path:
            grps = load_layer_workspace(path)
            if grps:
                self.groups = grps
                self._populate_tree()
                self._render_all_layers()
                self._on_zoom_to_extent()

    def _scan_and_load_default_layers(self) -> None:
        """Scan project for pre-existing tile_grid or QC rasters."""
        qc_dir = self.project_dir / "qc"
        grid_file = qc_dir / "vectors" / "tile_grid.geojson"
        if grid_file.exists():
            self.add_vector_layer(grid_file, group_name="Vector Grids")

        # Discover strips for strip diff combo
        if self.database:
            try:
                tiles = self.database.get_all_tiles()
                sids = set()
                for t in tiles:
                    fl = t.get("flight_line")
                    if fl:
                        sids.add(int(fl))
                if sids:
                    for sid in sorted(sids):
                        self._strip_a_combo.addItem(f"Strip {sid}", sid)
                        self._strip_b_combo.addItem(f"Strip {sid}", sid)
                    if len(sids) >= 2:
                        self._strip_b_combo.setCurrentIndex(1)
            except Exception:
                pass

        self._populate_tree()
        self._on_zoom_to_extent()

    # ── QC Generation Workflows ──────────────────────────────────────────────

    def _on_qc_type_changed(self, idx: int) -> None:
        is_diff = idx == 2
        self._strip_a_lbl.setVisible(is_diff)
        self._strip_a_combo.setVisible(is_diff)
        self._strip_b_lbl.setVisible(is_diff)
        self._strip_b_combo.setVisible(is_diff)

    def _on_generate_tile_grid(self) -> None:
        if not self.database:
            QMessageBox.warning(self, "No Database", "Project database is required to build tile grid.")
            return

        grid_mode = "all_together" if (hasattr(self, "_grid_full_radio") and self._grid_full_radio.isChecked()) else "loaded_only"
        mode_label = "all together" if grid_mode == "all_together" else "loaded tiles"
        self._qc_status_lbl.setText(f"Generating vector tile grid ({mode_label}) from project database…")
        try:
            res = generate_tile_grid_vector(self.database, grid_mode=grid_mode)
            path = Path(res["output_path"])

            # Clean existing tile grid graphics items before re-adding
            key = path.as_posix()
            empty_key = key + "_empty"
            if key in self._vector_path_items:
                self._view_box.removeItem(self._vector_path_items[key])
                del self._vector_path_items[key]
            if empty_key in self._vector_path_items:
                self._view_box.removeItem(self._vector_path_items[empty_key])
                del self._vector_path_items[empty_key]
            if key in self._vector_text_items:
                for t in self._vector_text_items[key]:
                    self._view_box.removeItem(t)
                del self._vector_text_items[key]

            # Remove existing layer object with matching path or name from groups
            for grp in self.groups:
                for lay in list(grp.layers):
                    if isinstance(lay, VectorLayer) and (lay.file_path.resolve() == path.resolve() or "tile_grid" in lay.name.lower()):
                        grp.remove_layer(lay)

            self.add_vector_layer(path, group_name="Vector Grids")
            self._on_zoom_to_extent()
            self._qc_status_lbl.setText(f"✓ Tile grid created: {res['tile_count']} tiles ({mode_label}).")
            self.statusBar().showMessage(f"Tile grid generated with {res['tile_count']} tiles ({mode_label}).", 4000)
        except Exception as exc:
            logger.exception("Failed to generate tile grid")
            QMessageBox.critical(self, "Tile Grid Error", str(exc))

    def _on_start_qc_generation(self) -> None:
        if not self.database or not self.tile_manager:
            QMessageBox.warning(self, "No Project", "Project database and tile manager are required.")
            return

        qc_idx = self._qc_type_combo.currentIndex()
        cell_size = self._cell_size_spin.value()
        target_tids = self.selected_tile_ids if self._scope_sel_radio.isChecked() and self.selected_tile_ids else None

        params: Dict[str, Any] = {
            "cell_size": cell_size,
            "tile_ids": target_tids,
        }

        ret_filter = "all"
        if self._filter_combo.currentIndex() == 1:
            ret_filter = "first"
        elif self._filter_combo.currentIndex() == 2:
            ret_filter = "ground"
        params["return_filter"] = ret_filter

        if qc_idx == 0:
            task_type = "density"
        elif qc_idx == 1:
            # Spacing requires density first; we generate density if not present
            task_type = "density"
            params["generate_spacing_after"] = True
        elif qc_idx == 2:
            task_type = "strip_diff"
            params["strip_a"] = self._strip_a_combo.currentData()
            params["strip_b"] = self._strip_b_combo.currentData()

        self._generate_qc_btn.setEnabled(False)
        self._qc_prog.setValue(0)
        self._qc_prog.show()
        self._qc_status_lbl.setText("Starting QC calculation…")

        self._qc_worker = _QCWorker(task_type, self.tile_manager, self.database, params)
        self._qc_worker.progress.connect(self._on_qc_progress)
        self._qc_worker.finished.connect(self._on_qc_finished)
        self._qc_worker.error.connect(self._on_qc_error)
        self._qc_worker.start()

    def _on_qc_progress(self, curr: int, total: int, msg: str) -> None:
        if total > 0:
            pct = int((curr / total) * 100)
            self._qc_prog.setValue(pct)
        self._qc_status_lbl.setText(msg)

    def _on_qc_finished(self, results: dict) -> None:
        self._generate_qc_btn.setEnabled(True)
        self._qc_prog.hide()
        out_path = results.get("output_path")

        if self._qc_worker and self._qc_worker.params.get("generate_spacing_after"):
            # Second step: derive spacing raster
            self._qc_status_lbl.setText("Deriving nominal point spacing…")
            try:
                sp_res = generate_spacing_raster(out_path)
                sp_path = sp_res["output_path"]
                self.add_raster_layer(sp_path, group_name="QC Rasters")
                self._qc_status_lbl.setText(f"✓ Point spacing raster complete: {Path(sp_path).name}")
                self.statusBar().showMessage(f"Generated spacing raster (Mean: {sp_res['mean_spacing']:.2f} m)", 5000)
            except Exception as exc:
                self._qc_status_lbl.setText(f"Spacing derivation error: {exc}")
            return

        if out_path:
            rl = self.add_raster_layer(out_path, group_name="QC Rasters")
            if "rmse_dz" in results:
                self._qc_status_lbl.setText(
                    f"✓ Strip Difference complete:\nRMSE dZ: {results['rmse_dz']:.3f} m\nMean dZ: {results['mean_dz']:.3f} m"
                )
                self.statusBar().showMessage(f"Generated Strip Difference raster: RMSE {results['rmse_dz']:.3f} m", 6000)
            elif "mean_density" in results:
                self._qc_status_lbl.setText(
                    f"✓ Point Density complete:\nMean: {results['mean_density']:.1f} pts/m²\nMax: {results['max_density']:.1f} pts/m²"
                )
                self.statusBar().showMessage(f"Generated Density raster: {results['mean_density']:.1f} pts/m²", 6000)

    def _on_qc_error(self, err_msg: str) -> None:
        self._generate_qc_btn.setEnabled(True)
        self._qc_prog.hide()
        self._qc_status_lbl.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "QC Generation Error", err_msg)
