"""
LiDAR Workbench — Import Wizard.

A multi-page QWizard that guides the user through LAS/LAZ file import
and tiling configuration, then runs the import in a background thread.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from .project_manager import ProjectManager
from .tile_manager import TileManager

logger = logging.getLogger("lidar_workbench.import_wizard")


class _ImportWorker(QThread):
    """
    Background thread that runs the tile import without blocking the GUI.

    Signals:
        progress: ``(step_description: str, percentage: float)``
        finished: ``(tile_ids: list[str])``
        error:    ``(error_message: str)``
    """

    progress = Signal(str, float)
    finished_import = Signal(list)
    error_occurred = Signal(str)

    def __init__(
        self,
        tile_manager: TileManager,
        directory: str,
        tile_size_m: Optional[float],
        overlap_m: float,
        target_points_per_tile: Optional[int] = None,
        min_points_per_tile: Optional[int] = None,
        sensor_type: str = '',
        scanner_override: str = '',
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._tm = tile_manager
        self._dir = directory
        self._tile_size = tile_size_m
        self._overlap = overlap_m
        self._target_points = target_points_per_tile
        self._min_points = min_points_per_tile
        self._sensor_type = sensor_type
        self._scanner_override = scanner_override

    def run(self) -> None:
        """Execute the import (runs in the worker thread)."""
        try:
            tile_ids = self._tm.import_las_directory(
                self._dir,
                tile_size_m=self._tile_size,
                overlap_m=self._overlap,
                target_points_per_tile=self._target_points,
                min_points_per_tile=self._min_points,
                sensor_type=self._sensor_type,
                scanner_override=self._scanner_override,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct),
            )
            self.finished_import.emit(tile_ids)
        except Exception as exc:
            logger.exception("Import failed")
            self.error_occurred.emit(str(exc))


# ── Page 1: file selection ─────────────────────────────────────────────


class _ImportFilePage(QWizardPage):
    """First wizard page — select LAS/LAZ files or directory."""

    def __init__(self, parent: Optional[QWizard] = None) -> None:
        super().__init__(parent)
        self.setTitle("Select LiDAR Data Source")
        self.setSubTitle("Choose a directory containing .las or .laz files to import.")

        layout = QVBoxLayout(self)

        # Instruction label
        self._label = QLabel(
            "Drag and drop a folder here, or use the button below to browse."
        )
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumHeight(80)
        self._label.setStyleSheet(
            "QLabel {"
            "  border: 2px dashed #888;"
            "  border-radius: 8px;"
            "  padding: 20px;"
            "  background: #f5f5f5;"
            "}"
        )
        layout.addWidget(self._label)

        # Browse button
        browse_btn = QPushButton("Browse for Directory…")
        browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(browse_btn)

        # File list
        self._file_list = QListWidget()
        self._file_list.setVisible(False)
        layout.addWidget(self._file_list)

        # Summary label
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._directory: Optional[str] = None

        # Accept drops on the page itself
        self.setAcceptDrops(True)

    @property
    def selected_directory(self) -> Optional[str]:
        """The currently selected directory path."""
        return self._directory

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._set_directory(path)

    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory with LAS/LAZ Files"
        )
        if directory:
            self._set_directory(directory)

    def _set_directory(self, directory: str) -> None:
        """Scan the directory for LAS/LAZ files and update the list."""
        self._directory = directory
        dir_path = Path(directory)
        las_files = sorted(
            list(dir_path.glob("*.las")) + list(dir_path.glob("*.laz"))
        )
        self._file_list.clear()
        self._file_list.setVisible(True)
        for f in las_files:
            item = QListWidgetItem(f.name)
            item.setToolTip(str(f))
            self._file_list.addItem(item)

        self._summary.setText(
            f"Found {len(las_files)} LAS/LAZ file(s) in:\n{directory}"
        )
        self._label.setStyleSheet(
            "QLabel {"
            "  border: 2px solid #4a4;"
            "  border-radius: 8px;"
            "  padding: 20px;"
            "  background: #e8f5e9;"
            "}"
        )
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._directory is not None and Path(self._directory).is_dir()


# ── Page 2: tiling parameters ──────────────────────────────────────────


class _TilingParamsPage(QWizardPage):
    """Second wizard page — configure tile size and overlap."""

    def __init__(self, parent: Optional[QWizard] = None) -> None:
        super().__init__(parent)
        self.setTitle("Tiling Parameters")
        self.setSubTitle("Configure how the point cloud is divided into tiles.")

        layout = QVBoxLayout(self)

        # Auto / manual toggle
        self._auto_check = QCheckBox("Auto-detect tile size (recommended)")
        self._auto_check.setChecked(True)
        self._auto_check.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self._auto_check)

        # Tile size group
        size_group = QGroupBox("Tile Size")
        size_form = QFormLayout(size_group)

        self._tile_size_spin = QDoubleSpinBox()
        self._tile_size_spin.setRange(10.0, 5000.0)
        self._tile_size_spin.setValue(200.0)
        self._tile_size_spin.setSuffix(" m")
        self._tile_size_spin.setEnabled(False)
        size_form.addRow("Edge Length:", self._tile_size_spin)

        self._overlap_spin = QDoubleSpinBox()
        self._overlap_spin.setRange(0.0, 500.0)
        self._overlap_spin.setValue(10.0)
        self._overlap_spin.setSuffix(" m")
        size_form.addRow("Overlap:", self._overlap_spin)

        layout.addWidget(size_group)

        # Auto-detect target group
        auto_group = QGroupBox("Auto-Detect Target")
        auto_form = QFormLayout(auto_group)

        self._target_points_spin = QDoubleSpinBox()
        self._target_points_spin.setRange(0.1, 50.0)
        self._target_points_spin.setValue(1.5)
        self._target_points_spin.setDecimals(1)
        self._target_points_spin.setSingleStep(0.5)
        self._target_points_spin.setSuffix(" M pts")
        self._target_points_spin.setToolTip(
            "Target point count per tile when auto-detect is enabled"
        )
        auto_form.addRow("Target Points per Tile:", self._target_points_spin)

        layout.addWidget(auto_group)

        # Min points filter group
        filter_group = QGroupBox("Tile Filtering")
        filter_form = QFormLayout(filter_group)

        self._min_points_spin = QDoubleSpinBox()
        self._min_points_spin.setRange(0.0, 1000.0)
        self._min_points_spin.setValue(0.0)
        self._min_points_spin.setDecimals(0)
        self._min_points_spin.setSingleStep(100.0)
        self._min_points_spin.setSuffix(" K pts")
        self._min_points_spin.setToolTip(
            "Skip tiles with fewer than this many points (0 = keep all)"
        )
        filter_form.addRow("Minimum Points per Tile:", self._min_points_spin)

        layout.addWidget(filter_group)

        # Info label
        self._info_label = QLabel(
            "Auto-detect computes tile size from point density to target "
            "~1.5 million points per tile.  Manual override is useful for "
            "very sparse or very dense datasets.\n\n"
            "Set a minimum point threshold to discard near-empty fringe tiles."
        )
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        layout.addStretch()

    def _on_auto_toggled(self, checked: bool) -> None:
        self._tile_size_spin.setEnabled(not checked)
        self._target_points_spin.setEnabled(checked)

    @property
    def tile_size_m(self) -> Optional[float]:
        """Return ``None`` for auto-detect, or the manual size in meters."""
        if self._auto_check.isChecked():
            return None
        return self._tile_size_spin.value()

    @property
    def overlap_m(self) -> float:
        return self._overlap_spin.value()

    @property
    def target_points_per_tile(self) -> Optional[int]:
        """Target point count for auto-detect, or None to use default."""
        return int(self._target_points_spin.value() * 1_000_000)

    @property
    def min_points_per_tile(self) -> Optional[int]:
        """Minimum points per tile; tiles below this are skipped."""
        return int(self._min_points_spin.value() * 1_000)


# ── Page 3: sensor / scanner settings ──────────────────────────────────


class _SensorParamsPage(QWizardPage):
    """Third wizard page — configure sensor type and scanner name."""

    def __init__(self, parent: Optional[QWizard] = None) -> None:
        super().__init__(parent)
        self.setTitle("Sensor & Scanner Settings")
        self.setSubTitle(
            "Select the sensor type (topography / bathymetry) and "
            "optionally override the scanner name."
        )

        layout = QVBoxLayout(self)

        # Sensor type
        type_group = QGroupBox("Sensor Type")
        type_layout = QVBoxLayout(type_group)

        self._auto_radio = QRadioButton("Auto-detect from filename (recommended)")
        self._auto_radio.setChecked(True)
        self._topo_radio = QRadioButton("Topography (topo)")
        self._bathy_radio = QRadioButton("Bathymetry (bathy)")
        self._none_radio = QRadioButton("Unspecified")

        type_layout.addWidget(self._auto_radio)
        type_layout.addWidget(self._topo_radio)
        type_layout.addWidget(self._bathy_radio)
        type_layout.addWidget(self._none_radio)
        layout.addWidget(type_group)

        # Scanner override
        scanner_group = QGroupBox("Scanner Name Override")
        scanner_layout = QFormLayout(scanner_group)

        self._scanner_edit = QLineEdit()
        self._scanner_edit.setPlaceholderText(
            "e.g. vq820g, vq580, als70 — leave empty for auto-detect"
        )
        scanner_layout.addRow("Scanner:", self._scanner_edit)
        layout.addWidget(scanner_group)

        # Info
        self._info_label = QLabel(
            "The scanner name is auto-detected from the filename "
            "(e.g. 'vq820g' from '1 - vq820g - ...').  "
            "Set an override here only when the filename does not "
            "encode the sensor or you need a custom label.\n\n"
            "<b>Auto-detect:</b> recognizes bathymetric scanners by the "
            "RIEGL -G suffix (VQ-820-G, VQ-870-G, VQ-880-G, VQ-840-G) "
            "and other known systems (Chiroptera, Hawkeye).  All other "
            "recognized scanners are treated as topographic.  Topo and "
            "bathy files can be imported together in a single run.\n\n"
            "Sensor type helps separate topography and bathymetry "
            "LiDAR for later analysis (e.g. automatic bathymetry "
            "cutting to river boundaries)."
        )
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        layout.addStretch()

    @property
    def sensor_type(self) -> str:
        if self._auto_radio.isChecked():
            return ''  # auto-detect per file from scanner name
        elif self._topo_radio.isChecked():
            return 'topo'
        elif self._bathy_radio.isChecked():
            return 'bathy'
        return ''

    @property
    def scanner_override(self) -> str:
        return self._scanner_edit.text().strip()


# ── Page 4: progress ───────────────────────────────────────────────────


class _ProgressPage(QWizardPage):
    """Third wizard page — display import progress."""

    def __init__(self, parent: Optional[QWizard] = None) -> None:
        super().__init__(parent)
        self.setTitle("Importing…")
        self.setSubTitle("Please wait while the data is processed.")

        layout = QVBoxLayout(self)

        self._status_label = QLabel("Preparing…")
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        layout.addWidget(self._detail_label)

        layout.addStretch()

        self._worker: Optional[_ImportWorker] = None
        self._finished = False
        self._tile_ids: List[str] = []

    @property
    def imported_tile_ids(self) -> List[str]:
        return self._tile_ids

    def initializePage(self) -> None:
        """Kick off the import worker when this page is shown."""
        # Prevent re-triggering if going back/forward
        if self._worker is not None:
            return

        wizard = self.wizard()
        if wizard is None:
            return

        # Access pages — wizard stores data via registerField or we
        # can access the pages directly through the wizard.
        pages = wizard.pageIds()
        file_page: _ImportFilePage = wizard.page(pages[0])  # type: ignore[assignment]
        params_page: _TilingParamsPage = wizard.page(pages[1])  # type: ignore[assignment]
        sensor_page: _SensorParamsPage = wizard.page(pages[2])  # type: ignore[assignment]

        directory = file_page.selected_directory
        if directory is None:
            self._status_label.setText("Error: no directory selected.")
            return

        # Obtain tile_manager from wizard property
        tile_manager: Optional[TileManager] = wizard.property("tile_manager")
        if tile_manager is None:
            self._status_label.setText("Error: TileManager not available.")
            return

        tile_size = params_page.tile_size_m
        overlap = params_page.overlap_m
        target_points = params_page.target_points_per_tile
        min_points = params_page.min_points_per_tile
        sensor_type = sensor_page.sensor_type
        scanner_override = sensor_page.scanner_override

        self._worker = _ImportWorker(
            tile_manager, directory, tile_size, overlap,
            target_points_per_tile=target_points,
            min_points_per_tile=min_points,
            sensor_type=sensor_type,
            scanner_override=scanner_override,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_import.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, step: str, pct: float) -> None:
        self._status_label.setText(step)
        self._progress_bar.setValue(int(pct))

    def _on_finished(self, tile_ids: List[str]) -> None:
        self._tile_ids = tile_ids
        self._finished = True
        self._status_label.setText(f"Import complete — {len(tile_ids)} tile(s) created.")
        self._progress_bar.setValue(100)
        self._detail_label.setText(
            f"Tile IDs: {', '.join(tile_ids[:10])}"
            + ("…" if len(tile_ids) > 10 else "")
        )
        self.completeChanged.emit()

    def _on_error(self, msg: str) -> None:
        self._status_label.setText(f"Import failed: {msg}")
        self._detail_label.setText("Check the log for details.")
        self._finished = True
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._finished


# ── Wizard ─────────────────────────────────────────────────────────────


class ImportWizard(QWizard):
    """
    Multi-page wizard for importing LAS/LAZ files into a project.

    Usage::

        wizard = ImportWizard(tile_manager, parent=self)
        if wizard.exec() == QWizard.Accepted:
            tile_ids = wizard.imported_tile_ids

        # Or with a pre-selected directory (skips file page):
        wizard = ImportWizard(tile_manager, parent=self,
                              preselected_dir="/data/flight_strips")
    """

    def __init__(
        self,
        tile_manager: TileManager,
        parent: Optional[QWizard] = None,
        preselected_dir: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import LiDAR Data")
        self.setMinimumSize(560, 600)

        # Store tile_manager as a property so pages can access it
        self.setProperty("tile_manager", tile_manager)

        self._file_page = _ImportFilePage(self)
        self._params_page = _TilingParamsPage(self)
        self._sensor_page = _SensorParamsPage(self)
        self._progress_page = _ProgressPage(self)

        self.addPage(self._file_page)
        self.addPage(self._params_page)
        self.addPage(self._sensor_page)
        self.addPage(self._progress_page)

        # If a directory is pre-selected, fill the file page and jump ahead
        if preselected_dir is not None:
            self._file_page._set_directory(preselected_dir)
            self.setStartId(1)  # skip file page, start at tiling params

    @property
    def imported_tile_ids(self) -> List[str]:
        """Return the tile IDs created during import."""
        return self._progress_page.imported_tile_ids
