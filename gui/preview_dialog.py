"""
LiDAR Workbench — LAS/LAZ Preview Dialog.

A standalone, non-modal dialog that lets users inspect LAS/LAZ files
*before* importing them into a project.  Supports four LOD levels:

    - **Bounding Box**: wireframe boxes from header metadata (instant).
    - **Subsampled (~1M pts)**: uniform-stride downsampled point cloud.
    - **Subsampled (~10M pts)**: finer subsampled point cloud.
    - **Full Resolution**: every point (warns above 50M pts).

Uses Open3D ``OffscreenRenderer`` for Qt-embeddable rendering (compatible
with Open3D ≥0.19 where ``SceneWidget`` is no longer a Qt widget).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QRectF, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    ASPRS_CLASS_COLORS,
    FALLBACK_CLASS_COLOR,
    PREVIEW_CHUNK_SIZE,
    PREVIEW_FULL_WARN_THRESHOLD,
    PREVIEW_LOD_BBOX,
    PREVIEW_LOD_FULL,
    PREVIEW_LOD_OPTIONS,
    PREVIEW_LOD_SUBSAMPLED_10M,
    PREVIEW_LOD_SUBSAMPLED_1M,
)

logger = logging.getLogger("lidar_workbench.gui.preview_dialog")

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    laspy = None

try:
    import open3d as o3d
    import open3d.visualization.rendering as o3d_render
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None
    o3d_render = None


# ── colormap helpers ───────────────────────────────────────────────────

def _height_colormap(z: np.ndarray) -> np.ndarray:
    """
    Map normalised Z (0–1) to blue→cyan→green→yellow→red ramp.
    Returns ``(N, 3)`` float64.
    """
    z = np.clip(z, 0.0, 1.0)
    r = np.where(z < 0.5, 0.0, np.where(z < 0.75, (z - 0.5) * 4.0, 1.0))
    g = np.where(z < 0.25, z * 4.0, np.where(z < 0.75, 1.0, (1.0 - z) * 4.0))
    b = np.where(z < 0.25, 1.0, np.where(z < 0.5, (0.5 - z) * 4.0, 0.0))
    return np.column_stack((r, g, b))


def _intensity_colormap(intensity: np.ndarray) -> np.ndarray:
    """Map normalised intensity (0–1) → grayscale.  Returns (N, 3)."""
    v = np.clip(intensity, 0.0, 1.0).reshape(-1, 1)
    return np.tile(v, (1, 3))


_RETURN_COLORS = np.array([
    [0.8, 0.2, 0.2],   # 1st → red
    [0.2, 0.8, 0.2],   # 2nd → green
    [0.2, 0.2, 0.8],   # 3rd → blue
    [0.8, 0.8, 0.2],   # 4th → yellow
    [0.8, 0.2, 0.8],   # 5th → magenta
    [0.2, 0.8, 0.8],   # 6th → cyan
    [0.6, 0.6, 0.6],   # 7th+ → grey
], dtype=np.float64)


def _return_colormap(rn: np.ndarray) -> np.ndarray:
    """Map integer return numbers (1-based) → discrete colours.  Returns (N, 3)."""
    idx = np.clip(rn.astype(np.int64) - 1, 0, len(_RETURN_COLORS) - 1)
    return _RETURN_COLORS[idx]


# ── preview paint widget ──────────────────────────────────────────────

class _PreviewView(QWidget):
    """Tiny QWidget that paints a stored QPixmap in its paintEvent.

    This avoids QLabel ownership issues — the pixmap lives in Python,
    not inside a Qt object that can be prematurely destroyed.

    Mouse events are forwarded to an ``orbit_callback`` for camera control.
    """

    def __init__(self, parent: Optional[QWidget] = None,
                 orbit_callback: Optional[callable] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._orbit_cb = orbit_callback
        self._mouse_last = None
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1e1e1e; border: 1px solid #333;")
        self.setMouseTracking(True)

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        """Set the pixmap to display and trigger a repaint."""
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the stored pixmap, scaled to fill the widget."""
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        src = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        dst = QRectF(0, 0, self.width(), self.height())
        p.drawPixmap(dst, self._pixmap, src)
        p.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._mouse_last = (event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:
        if self._mouse_last is None or self._orbit_cb is None:
            return
        x, y = event.position().x(), event.position().y()
        dx = x - self._mouse_last[0]
        dy = y - self._mouse_last[1]
        self._mouse_last = (x, y)
        self._orbit_cb("orbit", dx, dy)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._mouse_last = None

    def wheelEvent(self, event) -> None:
        if self._orbit_cb is None:
            return
        delta = event.angleDelta().y() / 120.0
        self._orbit_cb("zoom", delta, 0.0)


# ── background point loader ────────────────────────────────────────────

class _PointLoader(QThread):
    """
    Load points from a single LAS/LAZ file in a background thread.

    Reads x, y, z + the attribute needed for the active colour mode,
    applies uniform striding, maps to RGB, and emits the result.

    Signals:
        finished(int, ndarray[N,3], ndarray[N,3]): file_index, points, colors
        error_occurred(int, str): file_index, error message
    """
    finished = Signal(int, np.ndarray, np.ndarray)
    error_occurred = Signal(int, str)

    def __init__(
        self,
        file_index: int,
        las_path: Path,
        target_points: Optional[int],
        color_mode: str = "height",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._file_index = file_index
        self._las_path = Path(las_path)
        self._target_points = target_points   # None → full resolution
        self._color_mode = color_mode

    def run(self) -> None:
        try:
            with laspy.open(self._las_path) as reader:
                hdr = reader.header
                total = hdr.point_count
                if total == 0:
                    self.finished.emit(self._file_index,
                                       np.empty((0, 3), dtype=np.float64),
                                       np.empty((0, 3), dtype=np.float64))
                    return

                # Determine stride
                if self._target_points is not None and self._target_points < total:
                    step = max(1, total // self._target_points)
                else:
                    step = 1

                # Use lists — pre-allocating is unsafe because
                # sum(ceil(chunk_n/step)) >= ceil(total/step)
                pts_list = []
                attr_list = []  # for colour attribute

                # Which extra attribute do we need for colour?
                dim_names = {d.name.lower() for d in hdr.point_format.dimensions}
                need_class = (self._color_mode == "classification"
                              and "classification" in dim_names)
                need_intensity = (self._color_mode == "intensity"
                                  and "intensity" in dim_names)
                need_return = (self._color_mode == "return_number"
                               and "return_number" in dim_names)

                for chunk in reader.chunk_iterator(PREVIEW_CHUNK_SIZE):
                    idx = np.arange(0, len(chunk), step)

                    x = np.array(chunk.x[idx], dtype=np.float64)
                    y = np.array(chunk.y[idx], dtype=np.float64)
                    z = np.array(chunk.z[idx], dtype=np.float64)
                    pts_list.append(np.column_stack((x, y, z)))

                    # Collect colour attribute
                    if need_class:
                        attr_list.append(np.array(chunk.classification[idx], dtype=np.int64))
                    elif need_intensity:
                        attr_list.append(np.array(chunk.intensity[idx], dtype=np.float64))
                    elif need_return:
                        attr_list.append(np.array(chunk.return_number[idx], dtype=np.int64))
                    else:
                        # Height — will be computed later
                        attr_list.append(None)

                pts = np.concatenate(pts_list, axis=0)
                n_out = len(pts)
                colors_out = np.empty((n_out, 3), dtype=np.float64)

                # Apply colour mapping
                if need_class:
                    raw = np.concatenate(attr_list)
                    self._map_classification(raw, colors_out, 0, n_out)
                elif need_intensity:
                    raw = np.concatenate(attr_list)
                    self._map_intensity(raw, colors_out, 0, n_out)
                elif need_return:
                    raw = np.concatenate(attr_list)
                    self._map_return(raw, colors_out, 0, n_out)
                else:
                    self._map_height(pts, colors_out)

            self.finished.emit(self._file_index, pts, colors_out)

        except Exception as exc:
            logger.exception("Error loading %s", self._las_path)
            self.error_occurred.emit(self._file_index, str(exc))

    def _map_height(self, pts: np.ndarray, out: np.ndarray) -> None:
        z = pts[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        if z_max - z_min < 1e-8:
            out[:] = (0.5, 0.5, 0.5)
        else:
            out[:] = _height_colormap((z - z_min) / (z_max - z_min))

    @staticmethod
    def _map_classification(raw: np.ndarray, out: np.ndarray,
                            start: int, end: int) -> None:
        for code in np.unique(raw):
            mask = raw == code
            color = ASPRS_CLASS_COLORS.get(int(code), FALLBACK_CLASS_COLOR)
            out[start:end][mask] = color

    @staticmethod
    def _map_intensity(raw: np.ndarray, out: np.ndarray,
                       start: int, end: int) -> None:
        v_min, v_max = float(raw.min()), float(raw.max())
        if v_max - v_min < 1e-8:
            out[start:end] = (0.5, 0.5, 0.5)
        else:
            out[start:end] = _intensity_colormap((raw - v_min) / (v_max - v_min))

    @staticmethod
    def _map_return(raw: np.ndarray, out: np.ndarray,
                    start: int, end: int) -> None:
        out[start:end] = _return_colormap(raw)


# ── file metadata ──────────────────────────────────────────────────────

class _FileMeta:
    """Lightweight holder for per-file header metadata."""
    __slots__ = ("path", "point_count", "bbox", "version", "visible", "z_min", "z_max")

    def __init__(self, path: Path, point_count: int,
                 bbox: Tuple[float, float, float, float],
                 version: str,
                 z_min: float = 0.0, z_max: float = 0.0,
                 visible: bool = True) -> None:
        self.path = path
        self.point_count = point_count
        self.bbox = bbox           # (x_min, y_min, x_max, y_max)
        self.version = version
        self.visible = visible
        self.z_min = z_min
        self.z_max = z_max


# ── the dialog ─────────────────────────────────────────────────────────

class PreviewDialog(QDialog):
    """
    Standalone LAS/LAZ preview dialog.

    Usage::

        dlg = PreviewDialog(parent=main_window)
        dlg.import_requested.connect(main_window._on_preview_import)
        dlg.show()   # non-modal — user can interact with main window

    Signals:
        import_requested(list[Path]):
            Emitted when the user clicks "Import…", carrying the list of
            :class:`pathlib.Path` objects for all *visible* files in the
            current preview set.
    """

    import_requested = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LAS/LAZ Preview")
        self.setMinimumSize(900, 600)
        self.resize(1200, 750)

        # State
        self._files: List[_FileMeta] = []           # loaded file metadata
        self._point_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # idx→(pts,colors)
        self._bbox_geometries: Dict[int, str] = {}   # idx→geometry_name
        self._active_lod: str = PREVIEW_LOD_BBOX
        self._color_mode: str = "height"
        self._loaders: Dict[int, _PointLoader] = {}   # idx→active loader
        self._pending_loads: int = 0
        self._generation: int = 0                     # incremented on reload

        # Open3D — offscreen renderer with custom QWidget display
        self._renderer: Optional[object] = None      # OffscreenRenderer
        self._scene: Optional[object] = None          # Open3DScene
        self._view: Optional[_PreviewView] = None     # custom paint widget
        self._pixmap: Optional[QPixmap] = None         # last rendered frame

        # Camera state for mouse orbit
        self._cam_center = np.array([0.0, 0.0, 0.0])
        self._cam_eye = np.array([0.0, -100.0, 50.0])
        self._cam_up = np.array([0.0, 0.0, 1.0])
        self._cam_fov: float = 45.0
        self._mouse_last = None  # (x, y) of last drag position

        # Point-based density (computed from loaded points, more accurate than bbox)
        self._point_density: Optional[float] = None

        self._setup_ui()

    def closeEvent(self, event):
        """Cancel loaders and clean up OffscreenRenderer."""
        self._cancel_all_loaders()
        if self._renderer is not None:
            try:
                self._scene = None
                del self._renderer
                self._renderer = None
            except Exception:
                pass
        super().closeEvent(event)

    # ── UI construction ────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # --- left panel: file list ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._file_list.itemChanged.connect(self._on_item_toggled)
        left_layout.addWidget(self._file_list, 1)

        btn_add = QPushButton("Add Files…")
        btn_add.clicked.connect(self._on_add_files)
        left_layout.addWidget(btn_add)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._on_remove_files)
        left_layout.addWidget(btn_remove)

        splitter.addWidget(left)

        # --- right panel: custom paint widget ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._view = _PreviewView(right, orbit_callback=self._on_orbit)
        right_layout.addWidget(self._view, 1)

        if not HAS_OPEN3D:
            self._view.set_preview_pixmap(
                self._placeholder_pixmap("3D preview unavailable\n(Open3D required)")
            )

        splitter.addWidget(right)

        # --- bottom bar ---
        bar = QHBoxLayout()
        bar.addWidget(QLabel("LOD:"))

        self._lod_combo = QComboBox()
        for key, label in PREVIEW_LOD_OPTIONS:
            self._lod_combo.addItem(label, key)
        self._lod_combo.currentIndexChanged.connect(self._on_lod_changed)
        bar.addWidget(self._lod_combo)

        bar.addSpacing(16)
        bar.addWidget(QLabel("Color:"))

        self._color_combo = QComboBox()
        self._color_combo.addItem("Height (Z)", "height")
        self._color_combo.addItem("Classification", "classification")
        self._color_combo.addItem("Intensity", "intensity")
        self._color_combo.addItem("Return Number", "return_number")
        self._color_combo.addItem("File (Flight Line)", "file")
        self._color_combo.currentIndexChanged.connect(self._on_color_changed)
        bar.addWidget(self._color_combo)

        bar.addStretch()

        self._import_btn = QPushButton("Import…")
        self._import_btn.setToolTip("Import currently visible files into the project")
        self._import_btn.clicked.connect(self._on_import_clicked)
        self._import_btn.setEnabled(False)
        bar.addWidget(self._import_btn)

        bar.addSpacing(8)

        self._status_label = QLabel("No files loaded")
        bar.addWidget(self._status_label)

        root.addLayout(bar)

    # ── offscreen renderer ─────────────────────────────────────────

    def _init_renderer(self) -> None:
        """Create the OffscreenRenderer (idempotent)."""
        if self._renderer is not None or not HAS_OPEN3D:
            return
        try:
            self._renderer = o3d_render.OffscreenRenderer(800, 600)
            self._scene = self._renderer.scene
            self._scene.set_background([0.12, 0.12, 0.16, 1.0])
            logger.info("OffscreenRenderer created")
        except Exception as exc:
            logger.warning("Failed to create OffscreenRenderer: %s", exc)
            self._renderer = None
            self._scene = None

    def _placeholder_pixmap(self, text: str) -> QPixmap:
        """Create a placeholder pixmap with the given text."""
        pm = QPixmap(400, 300)
        pm.fill(Qt.black)
        from PySide6.QtGui import QPainter, QColor, QFont
        p = QPainter(pm)
        p.setPen(QColor("#888888"))
        p.setFont(QFont("sans-serif", 12))
        p.drawText(pm.rect(), Qt.AlignCenter, text)
        p.end()
        return pm

    # ── file management ────────────────────────────────────────────

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select LAS/LAZ Files",
            "",
            "LiDAR Files (*.las *.laz);;All Files (*)",
        )
        if not paths:
            return

        for p in paths:
            self._add_file(Path(p))
        logger.info("Added %d file(s), %d visible — reloading scene",
                     len(self._files),
                     sum(1 for m in self._files if m.visible))
        self._init_renderer()
        self._update_status()
        self._reload_scene()

    def _add_file(self, las_path: Path, visible: bool = True) -> Optional[int]:
        """Read header and register a file.  Returns the file index or None."""
        if not HAS_LASPY:
            return None

        # Deduplicate
        for meta in self._files:
            if meta.path.resolve() == las_path.resolve():
                return None

        try:
            with laspy.open(las_path) as reader:
                hdr = reader.header
                bbox: Tuple[float, float, float, float] = (
                    float(hdr.x_min), float(hdr.y_min),
                    float(hdr.x_max), float(hdr.y_max),
                )
                z_min = float(getattr(hdr, 'z_min', 0.0) or 0.0)
                z_max = float(getattr(hdr, 'z_max', 0.0) or 0.0)
                meta = _FileMeta(
                    path=las_path.resolve(),
                    point_count=hdr.point_count,
                    bbox=bbox,
                    version=f"{hdr.version.major}.{hdr.version.minor}",
                    visible=visible,
                    z_min=z_min,
                    z_max=z_max,
                )
        except Exception as exc:
            logger.error("Failed to read header of %s: %s", las_path, exc)
            QMessageBox.warning(
                self, "Read Error",
                f"Could not read header of:\n{las_path.name}\n\n{exc}",
            )
            return None

        self._files.append(meta)

        # Compute density (pts/m²)
        x0, y0, x1, y1 = meta.bbox
        area = (x1 - x0) * (y1 - y0)
        density = meta.point_count / area if area > 0 else 0.0

        # List item with area for verification
        item = QListWidgetItem()
        item.setText(
            f"{meta.path.name}  —  {meta.point_count:,} pts  "
            f"|  {area:,.0f} m²  |  ~{density:.1f} pts/m²  "
            f"[{x0:.0f}, {x1:.0f}] × [{y0:.0f}, {y1:.0f}]"
        )
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        item.setData(Qt.UserRole, len(self._files) - 1)  # store index
        self._file_list.addItem(item)

        return len(self._files) - 1

    def _on_remove_files(self) -> None:
        # Remove selected items (work backwards from indices)
        selected = self._file_list.selectedItems()
        if not selected:
            return
        indices = sorted([it.data(Qt.UserRole) for it in selected], reverse=True)
        for idx in indices:
            self._remove_file(idx)

    def _on_import_clicked(self) -> None:
        """Collect paths of all visible files and emit ``import_requested``."""
        visible = [m.path for m in self._files if m.visible]
        if not visible:
            return
        self.import_requested.emit(visible)

    def _remove_file(self, idx: int) -> None:
        """Remove file at the given index, adjusting all subsequent indices."""
        # Remove point data
        self._point_data.pop(idx, None)

        # Remove bbox geometry from scene
        name = self._bbox_geometries.pop(idx, None)
        if name and self._scene:
            self._scene.remove_geometry(name)

        # Shift indices of files/point_data/bbox_geometries after idx
        del self._files[idx]
        new_point_data: Dict[int, Tuple] = {}
        for k, v in self._point_data.items():
            new_point_data[k - 1 if k > idx else k] = v
        self._point_data = new_point_data

        new_bbox: Dict[int, str] = {}
        for k, v in self._bbox_geometries.items():
            new_bbox[k - 1 if k > idx else k] = v
        self._bbox_geometries = new_bbox

        # Rebuild list widget items (simplest)
        self._file_list.blockSignals(True)
        self._file_list.clear()
        for i, meta in enumerate(self._files):
            item = QListWidgetItem()
            x0, y0, x1, y1 = meta.bbox
            area = (x1 - x0) * (y1 - y0)
            density = meta.point_count / area if area > 0 else 0.0
            item.setText(
                f"{meta.path.name}  —  {meta.point_count:,} pts  "
                f"|  {area:,.0f} m²  |  ~{density:.1f} pts/m²  "
                f"[{x0:.0f}, {x1:.0f}] × [{y0:.0f}, {y1:.0f}]"
            )
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if meta.visible else Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            self._file_list.addItem(item)
        self._file_list.blockSignals(False)

        self._update_status()

    def _on_item_toggled(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self._files):
            return
        self._files[idx].visible = (item.checkState() == Qt.Checked)
        self._rebuild_scene_from_cache()

    # ── LOD / colour ───────────────────────────────────────────────

    def _on_lod_changed(self) -> None:
        self._active_lod = self._lod_combo.currentData()
        # Discard cached point data — reload at new LOD
        self._point_data.clear()
        self._reload_scene()

    def _on_color_changed(self) -> None:
        self._color_mode = self._color_combo.currentData()
        if self._active_lod == PREVIEW_LOD_BBOX:
            return  # bbox has no colour
        # "file" mode reuses cached point data — just rebuild scene
        if self._color_mode == "file":
            self._rebuild_scene_from_cache()
            self._update_status()
            return
        # Other modes need point reload for new colour mapping
        self._point_data.clear()
        self._reload_scene()

    # ── scene management ───────────────────────────────────────────

    def _reload_scene(self) -> None:
        """Trigger reload of all visible files at the current LOD."""
        if self._scene is None:
            return

        # Cancel any in-flight loaders
        self._cancel_all_loaders()
        self._generation += 1
        self._pending_loads = 0

        if self._active_lod == PREVIEW_LOD_BBOX:
            self._scene.clear_geometry()
            self._bbox_geometries.clear()
            self._render_bboxes()
            self._fit_camera_to_bboxes()
            self._render_to_label()
            self._update_status()
            return

        # For point-based LODs: launch background loaders
        target = self._target_points_for_lod()

        # Warn if full-res and huge
        if self._active_lod == PREVIEW_LOD_FULL:
            total = sum(m.point_count for m in self._files if m.visible)
            if total > PREVIEW_FULL_WARN_THRESHOLD:
                ans = QMessageBox.question(
                    self, "Large Point Cloud",
                    f"Loading {total:,} points at full resolution may be "
                    f"slow and use significant memory.\n\nContinue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if ans != QMessageBox.Yes:
                    self._lod_combo.setCurrentIndex(
                        self._lod_combo.findData(PREVIEW_LOD_SUBSAMPLED_1M)
                    )
                    return

        self._scene.clear_geometry()
        self._bbox_geometries.clear()

        for i, meta in enumerate(self._files):
            if not meta.visible:
                continue
            self._start_loader(i, meta.path, target)

        if self._pending_loads == 0:
            self._update_status()

    def _target_points_for_lod(self) -> Optional[int]:
        if self._active_lod == PREVIEW_LOD_SUBSAMPLED_1M:
            return 1_000_000
        elif self._active_lod == PREVIEW_LOD_SUBSAMPLED_10M:
            return 10_000_000
        return None  # full

    def _cancel_all_loaders(self) -> None:
        """Disconnect and quit all active _PointLoader threads."""
        for loader in self._loaders.values():
            try:
                loader.finished.disconnect()
                loader.error_occurred.disconnect()
            except (TypeError, RuntimeError):
                pass  # already disconnected or destroyed
            if loader.isRunning():
                loader.quit()
                if not loader.wait(3000):
                    logger.warning("Loader for index %d did not stop in time",
                                   loader._file_index)
        self._loaders.clear()
        self._pending_loads = 0

    def _start_loader(self, file_index: int, las_path: Path,
                      target: Optional[int]) -> None:
        self._pending_loads += 1
        gen = self._generation
        loader = _PointLoader(
            file_index, las_path, target, self._color_mode, parent=self
        )
        loader.finished.connect(lambda fi, pts, cols, g=gen:
                                self._on_points_loaded(fi, pts, cols, g))
        loader.error_occurred.connect(lambda fi, msg, g=gen:
                                      self._on_load_error(fi, msg, g))
        self._loaders[file_index] = loader
        loader.start()

    def _on_points_loaded(self, file_index: int,
                          pts: np.ndarray, colors: np.ndarray,
                          generation: int = 0) -> None:
        if generation != self._generation:
            return  # stale — a newer reload was triggered
        self._point_data[file_index] = (pts, colors)
        self._loaders.pop(file_index, None)
        self._pending_loads -= 1
        if self._pending_loads <= 0:
            self._pending_loads = 0
            self._compute_effective_density()
            self._rebuild_scene_from_cache()
            self._update_status()

    def _on_load_error(self, file_index: int, message: str,
                       generation: int = 0) -> None:
        if generation != self._generation:
            return  # stale
        self._loaders.pop(file_index, None)
        self._pending_loads -= 1
        logger.error("Preview load error [file %d]: %s", file_index, message)
        if self._pending_loads <= 0:
            self._pending_loads = 0
            self._rebuild_scene_from_cache()
            self._update_status()

    def _rebuild_scene_from_cache(self) -> None:
        """Recreate all geometries from cached point data / bboxes."""
        if self._scene is None:
            return

        self._scene.clear_geometry()
        self._bbox_geometries.clear()

        geo_idx = 0
        for i, meta in enumerate(self._files):
            if not meta.visible:
                continue
            data = self._point_data.get(i)
            if data is not None:
                pts, colors = data
                tint = _file_colour(i) if self._color_mode == "file" else None
                self._add_pcd_to_scene(geo_idx, pts, colors, tint)
                geo_idx += 1
            else:
                # Still loading or bbox-only
                if self._active_lod == PREVIEW_LOD_BBOX:
                    self._render_one_bbox(i, meta)

        if self._active_lod == PREVIEW_LOD_BBOX:
            self._fit_camera_to_bboxes()
        else:
            self._fit_camera_to_points()
        self._render_to_label()

    def _add_pcd_to_scene(self, idx: int, pts: np.ndarray,
                          colors: np.ndarray,
                          tint: Optional[Tuple[float, float, float]] = None) -> None:
        if len(pts) == 0 or self._scene is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if tint is not None:
            # Blend tint (30%) with original colors (70%)
            t = np.array(tint, dtype=np.float64).reshape(1, 3)
            blended = colors * 0.7 + t * 0.3
            pcd.colors = o3d.utility.Vector3dVector(np.clip(blended, 0, 1))
        else:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        mat = o3d_render.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.5
        name = f"preview_pcd_{idx}"
        self._scene.add_geometry(name, pcd, mat)

    def _render_bboxes(self) -> None:
        for i, meta in enumerate(self._files):
            if meta.visible:
                self._render_one_bbox(i, meta)

    def _render_one_bbox(self, idx: int, meta: _FileMeta) -> None:
        if self._scene is None:
            return
        ls = _bbox_to_lineset(meta.bbox, meta.z_min, meta.z_max)
        mat = o3d_render.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 2.0
        # Assign a colour per file from a fixed palette
        colour = _file_colour(idx)
        mat.base_color = (*colour, 1.0)
        name = f"preview_bbox_{idx}"
        self._scene.add_geometry(name, ls, mat)
        self._bbox_geometries[idx] = name

    def _fit_camera_to_bboxes(self) -> None:
        if self._renderer is None or not self._files:
            return
        visible = [m.bbox for m in self._files if m.visible]
        if not visible:
            return
        xs = [b[0] for b in visible] + [b[2] for b in visible]
        ys = [b[1] for b in visible] + [b[3] for b in visible]
        center = np.array([(min(xs) + max(xs)) / 2,
                           (min(ys) + max(ys)) / 2,
                           0.0])
        extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        self._cam_center = center
        self._cam_eye = center + np.array([0.0, -extent * 2.0, extent * 1.2])
        self._cam_up = np.array([0.0, 0.0, 1.0])
        self._renderer.setup_camera(
            45.0,
            center,
            center + np.array([0.0, -extent * 2.0, extent * 1.2]),
            np.array([0.0, 0.0, 1.0]),
        )

    def _fit_camera_to_points(self) -> None:
        if self._renderer is None:
            return
        all_pts = []
        for i, meta in enumerate(self._files):
            if not meta.visible:
                continue
            data = self._point_data.get(i)
            if data is not None:
                all_pts.append(data[0])
        if not all_pts:
            return
        pts = np.concatenate(all_pts, axis=0)
        center = pts.mean(axis=0)
        extent = float(np.ptp(pts, axis=0).max()) or 1.0
        self._cam_center = center
        self._cam_eye = center + np.array([0.0, -extent * 2.0, extent * 0.8])
        self._cam_up = np.array([0.0, 0.0, 1.0])
        self._renderer.setup_camera(
            45.0,
            center,
            center + np.array([0.0, -extent * 2.0, extent * 0.8]),
            np.array([0.0, 0.0, 1.0]),
        )

    # ── render to widget ───────────────────────────────────────────

    def _compute_effective_density(self) -> None:
        """Compute point density from loaded point data (grid-based).

        Uses a 1m grid — counts occupied cells, then density =
        total_points / (occupied_cells * cell_area).  Much more
        accurate than the header bounding box for irregular flight
        strips where the axis-aligned bbox includes empty corners.
        """
        self._point_density = None
        all_pts = []
        for i, meta in enumerate(self._files):
            if not meta.visible:
                continue
            data = self._point_data.get(i)
            if data is not None:
                all_pts.append(data[0])
        if not all_pts:
            return
        pts = np.concatenate(all_pts, axis=0)
        n = len(pts)
        if n < 100:
            return
        # Bin into 1 m² cells
        cell_size = 1.0
        x_bins = np.arange(pts[:, 0].min(), pts[:, 0].max() + cell_size, cell_size)
        y_bins = np.arange(pts[:, 1].min(), pts[:, 1].max() + cell_size, cell_size)
        if len(x_bins) < 2 or len(y_bins) < 2:
            return
        h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_bins, y_bins])
        occupied = (h > 0).sum()
        effective_area = occupied * cell_size * cell_size
        if effective_area > 0:
            self._point_density = n / effective_area

    def _on_orbit(self, action: str, dx: float, dy: float) -> None:
        """Handle mouse orbit/zoom from _PreviewView and re-render."""
        if self._renderer is None:
            return
        import math
        if action == "orbit":
            direction = self._cam_eye - self._cam_center
            up = self._cam_up / np.linalg.norm(self._cam_up)
            # Horizontal rotation
            angle_h = -dx * 0.005
            cos_h, sin_h = math.cos(angle_h), math.sin(angle_h)
            direction = (cos_h * direction + sin_h * np.cross(up, direction)
                         + (1 - cos_h) * np.dot(direction, up) * up)
            # Vertical rotation
            right = np.cross(direction, up)
            right /= np.linalg.norm(right) + 1e-12
            angle_v = -dy * 0.005
            cos_v, sin_v = math.cos(angle_v), math.sin(angle_v)
            new_dir = cos_v * direction + sin_v * np.cross(right, direction)
            if np.dot(new_dir / (np.linalg.norm(new_dir) + 1e-12), up) < 0.99:
                direction = new_dir
            self._cam_eye = self._cam_center + direction
        elif action == "zoom":
            direction = self._cam_eye - self._cam_center
            dist = float(np.linalg.norm(direction))
            new_dist = dist * (1.0 - dx * 0.1)
            if new_dist > 0.01:
                self._cam_eye = self._cam_center + direction / dist * new_dist
        self._renderer.setup_camera(self._cam_fov, self._cam_center,
                                    self._cam_eye, self._cam_up)
        self._render_to_label()

    def _render_to_label(self) -> None:
        """Render scene via OffscreenRenderer → QPixmap → _PreviewView."""
        if self._renderer is None or self._view is None:
            return
        try:
            img = self._renderer.render_to_image()
            arr = np.asarray(img).copy()
            h, w = arr.shape[:2]
            channels = arr.shape[2] if arr.ndim == 3 else 1
            if channels == 4:
                fmt = QImage.Format_RGBA8888
            elif channels == 3:
                fmt = QImage.Format_RGB888
            else:
                fmt = QImage.Format_Grayscale8
            # QImage from data, then deep-copy into QPixmap for safety
            qimg = QImage(arr.data, w, h, w * channels, fmt)
            self._view.set_preview_pixmap(QPixmap.fromImage(qimg.copy()))
            logger.info("Rendered %dx%d preview (channels=%d)", w, h, channels)
        except Exception as exc:
            logger.warning("Offscreen render failed: %s", exc)
            import traceback
            traceback.print_exc()

    def _update_status(self) -> None:
        n_files = len(self._files)
        if n_files == 0:
            self._status_label.setText("No files loaded")
            self._import_btn.setEnabled(False)
            return
        total_pts = sum(m.point_count for m in self._files)
        visible_pts = sum(m.point_count for m in self._files if m.visible)

        # Enable / disable Import button
        n_visible = sum(1 for m in self._files if m.visible)
        self._import_btn.setEnabled(n_visible > 0)

        # Warn if requested colour dimension may be missing
        colour_note = ""
        if self._active_lod != PREVIEW_LOD_BBOX:
            dim_wanted = {
                "classification": "Classification",
                "intensity": "Intensity",
                "return_number": "Return Number",
            }.get(self._color_mode, "")
            if dim_wanted:
                # Check if at least one visible file has the dimension
                any_has = False
                for i, m in enumerate(self._files):
                    if not m.visible:
                        continue
                    try:
                        with laspy.open(m.path) as r:
                            dims = {d.name.lower()
                                    for d in r.header.point_format.dimensions}
                            if self._color_mode in dims:
                                any_has = True
                                break
                    except Exception:
                        pass
                if not any_has:
                    colour_note = f"  ({dim_wanted} not available — using height)"

        # Compute overall density for visible files
        density_note = ""
        visible_files = [m for m in self._files if m.visible]
        if visible_files:
            total_vis_pts = sum(m.point_count for m in visible_files)
            total_area = sum(
                (m.bbox[2] - m.bbox[0]) * (m.bbox[3] - m.bbox[1])
                for m in visible_files
            )
            if total_area > 0:
                avg_density = total_vis_pts / total_area
                density_note = f"  |  bbox: ~{avg_density:.0f} pts/m²"
                if self._point_density is not None:
                    density_note += f"  |  effective: ~{self._point_density:.0f} pts/m²"

        if self._pending_loads > 0:
            self._status_label.setText(
                f"{n_files} file(s), {total_pts:,} total pts  —  "
                f"loading {self._pending_loads} file(s)…{density_note}{colour_note}"
            )
        else:
            self._status_label.setText(
                f"{n_files} file(s), {total_pts:,} total pts  "
                f"({visible_pts:,} visible)  |  LOD: {self._active_lod}"
                f"{density_note}{colour_note}"
            )


# ── geometry helpers ───────────────────────────────────────────────────

def _bbox_to_lineset(bbox: Tuple[float, float, float, float],
                     z_min: float = 0.0, z_max: float = 0.0):
    """
    Convert a 2D bbox ``(x_min, y_min, x_max, y_max)`` and optional
    z range to an Open3D ``LineSet`` wireframe.
    """
    x0, y0, x1, y1 = bbox
    z0, z1 = z_min, z_max
    # If no z info, give it a thin slab so the wireframe is still visible
    if abs(z1 - z0) < 1e-8:
        z0 = -1.0
        z1 = 1.0
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],   # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],   # top
        [0, 4], [1, 5], [2, 6], [3, 7],   # verticals
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    return ls


_FILE_PALETTE = np.array([
    [1.0, 0.3, 0.3],   # red
    [0.3, 0.7, 1.0],   # light blue
    [0.3, 1.0, 0.3],   # green
    [1.0, 1.0, 0.3],   # yellow
    [1.0, 0.3, 1.0],   # magenta
    [0.3, 1.0, 1.0],   # cyan
    [1.0, 0.6, 0.2],   # orange
    [0.7, 0.3, 1.0],   # purple
    [0.5, 1.0, 0.5],   # light green
    [1.0, 0.5, 0.7],   # pink
], dtype=np.float64)


def _file_colour(idx: int) -> Tuple[float, float, float]:
    c = _FILE_PALETTE[idx % len(_FILE_PALETTE)]
    return (float(c[0]), float(c[1]), float(c[2]))
