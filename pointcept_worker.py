"""
LiDAR Workbench — Pointcept Integration Worker.

Runs Pointcept inference in a background QThread via subprocess,
communicating progress and results back to the GUI through Qt signals.

Calls the bundled ``Pointcept/prediction.py`` entry point which handles
LAS file loading, density normalisation, block-wise prediction, and
ASPRS class remapping.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_POINTCEPT_PATH,
    TileStatus,
)
from .database import Database
from .tile_manager import TileManager

logger = logging.getLogger("lidar_workbench.pointcept_worker")


class PointceptWorker(QThread):
    """
    Background worker that invokes Pointcept inference on selected tiles.

    The worker calls the bundled ``Pointcept/prediction.py`` script,
    which handles:

        - LAS loading & intensity normalisation
        - Density normalisation (voxel downsampling)
        - Block-wise model inference
        - ASPRS class remapping
        - Saving predictions to a ``predictions/`` subdirectory

    Intensity scale is computed as the 97th percentile of all
    intensities across the selected tiles for robust normalisation.

    Signals:
        progress(tile_id, step_msg, pct):
            Emitted during processing.
        tile_done(tile_id):
            Emitted when a single tile finishes successfully.
        tile_error(tile_id, error_msg):
            Emitted when a tile fails.
        all_done(tile_ids):
            Emitted when all tiles have been processed.
    """

    progress = Signal(str, str, float)
    tile_done = Signal(str)
    tile_error = Signal(str, str)
    all_done = Signal(list)

    def __init__(
        self,
        tile_manager: TileManager,
        database: Database,
        tile_ids: List[str],
        pointcept_path: str = DEFAULT_POINTCEPT_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        python_exe: Optional[str] = None,
        voxel_size: float = 0.15,
        smoothing: str = "yes",
        parent: Optional[QThread] = None,
    ) -> None:
        """
        Args:
            tile_manager:   :class:`TileManager` for accessing tile data.
            database:       :class:`Database` for status updates.
            tile_ids:       List of tile IDs to classify.
            pointcept_path: Path to the bundled Pointcept directory.
            model_path:     Path to the trained model checkpoint (.pth).
            config_path:    Path to the model config (.py).
            python_exe:     Python interpreter (default: ``sys.executable``).
            voxel_size:     Voxel size for density normalisation (meters).
            smoothing:      Enable k-NN smoothing (``"yes"`` / ``"no"``).
        """
        super().__init__(parent)
        self._tm = tile_manager
        self._db = database
        self._tile_ids = tile_ids
        self._pointcept_path = Path(pointcept_path).resolve()
        self._model_path = Path(model_path).resolve()
        self._config_path_rel = config_path  # relative to pointcept_path
        self._python_exe = python_exe or sys.executable
        self._voxel_size = voxel_size
        self._smoothing = smoothing
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation.  The worker stops after the current tile."""
        self._cancelled = True
        logger.info("Cancellation requested for PointceptWorker")

    # ── main loop ─────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the classification pipeline (runs in the worker thread)."""
        total = len(self._tile_ids)
        completed: List[str] = []

        # Validate prerequisites
        if not self._model_path.is_file():
            self.tile_error.emit("__all__", f"Model not found: {self._model_path}")
            self.all_done.emit([])
            return

        tiles_dir = self._tm._pm.tiles_dir  # type: ignore[attr-defined]
        if tiles_dir is None:
            self.tile_error.emit("__all__", "Project tiles directory not available")
            self.all_done.emit([])
            return

        # Compute 97th percentile intensity across all selected tiles
        intensity_scale = self._compute_intensity_scale(tiles_dir)

        for i, tile_id in enumerate(self._tile_ids):
            if self._cancelled:
                logger.info("Worker cancelled after %d/%d tiles", i, total)
                break

            pct_base = i / total * 100.0
            self.progress.emit(
                tile_id,
                f"Classifying {tile_id} ({i + 1}/{total})…",
                pct_base,
            )

            try:
                self._classify_tile(tile_id, tiles_dir, intensity_scale)
                with self._db.connect() as conn:
                    self._db.update_status(conn, tile_id, TileStatus.CLASSIFIED)
                self.tile_done.emit(tile_id)
                completed.append(tile_id)
                self.progress.emit(
                    tile_id,
                    f"Done: {tile_id}",
                    (i + 1) / total * 100.0,
                )
            except Exception as exc:
                logger.exception("Classification failed for %s", tile_id)
                self.tile_error.emit(tile_id, str(exc))
                with self._db.connect() as conn:
                    self._db.update_status(conn, tile_id, TileStatus.ERROR)

        self.all_done.emit(completed)

    # ── intensity scale ────────────────────────────────────────────

    def _compute_intensity_scale(self, tiles_dir: Path) -> float:
        """
        Compute the 97th percentile intensity across all selected tiles.

        This is used as ``--intensity_scale`` for Pointcept prediction
        to normalise intensities to [0, 1] range.
        """
        logger.info("Computing 97th-percentile intensity across %d tile(s)…",
                     len(self._tile_ids))
        all_intensities: List[np.ndarray] = []

        for tile_id in self._tile_ids:
            tile_info = self._db.get_tile(tile_id)
            if tile_info is None:
                continue
            las_path = tiles_dir / tile_info["filename"]
            if not las_path.is_file():
                continue
            try:
                import laspy
                las = laspy.read(las_path)
                all_intensities.append(np.array(las.intensity, dtype=np.float64))
            except Exception as exc:
                logger.warning("Could not read intensities from %s: %s", las_path, exc)

        if not all_intensities:
            logger.warning("No intensity data found — using default 65535.0")
            return 65535.0

        combined = np.concatenate(all_intensities)
        p97 = float(np.percentile(combined, 97.0))
        logger.info(
            "Intensity scale (97th pctl): %.1f  (min=%.0f, max=%.0f, n=%d tiles)",
            p97, combined.min(), combined.max(), len(all_intensities),
        )
        return max(p97, 1.0)  # never zero

    # ── single-tile classification ─────────────────────────────────

    def _classify_tile(
        self, tile_id: str, tiles_dir: Path, intensity_scale: float
    ) -> None:
        """
        Run Pointcept prediction on a single tile.

        Because ``prediction.py`` processes **all** LAS files in the
        given folder, we create a temporary directory containing just
        the target tile, run inference there, and copy the result back.

        Calls ``Pointcept/prediction.py`` with:

            --folder         → temporary directory (single tile)
            --model_path     → model checkpoint
            --config_file    → model config
            --noise_filter   → "no" (headless, no GUI pop-up)
            --intensity_scale→ 97th-percentile intensity
            --voxel_size     → voxel size for density normalisation
            --smoothing      → "yes" (k-NN smoothing)
        """
        import tempfile

        tile_info = self._db.get_tile(tile_id)
        if tile_info is None:
            raise ValueError(f"Tile {tile_id} not found in database")

        las_path = tiles_dir / tile_info["filename"]
        if not las_path.is_file():
            raise FileNotFoundError(f"LAS file not found: {las_path}")

        # Resolve config path (may be relative to Pointcept root)
        config_full = self._pointcept_path / self._config_path_rel
        if not config_full.is_file():
            raise FileNotFoundError(f"Config file not found: {config_full}")

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(self._pointcept_path)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        # Create a temp directory with just this tile
        with tempfile.TemporaryDirectory(prefix="pointcept_work_") as work_dir:
            work_path = Path(work_dir)
            # Symlink the LAS file into the work dir
            dest = work_path / las_path.name
            dest.symlink_to(las_path.resolve())

            cmd = [
                self._python_exe,
                str(self._pointcept_path / "prediction.py"),
                "--folder", str(work_path),
                "--model_path", str(self._model_path),
                "--config_file", str(self._config_path_rel),
                "--noise_filter", "no",
                "--smoothing", self._smoothing,
                "--voxel_size", str(self._voxel_size),
                "--intensity_scale", str(intensity_scale),
            ]

            logger.debug("Pointcept command: %s", " ".join(str(x) for x in cmd))

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self._pointcept_path),
                    env=env,
                    timeout=7200,  # 2 hours max per tile
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Pointcept inference timed out for tile {tile_id} (>2 hours)"
                )

            if result.returncode != 0:
                stderr_tail = (
                    result.stderr[-600:]
                    if len(result.stderr) > 600
                    else result.stderr
                )
                raise RuntimeError(
                    f"Pointcept exited with code {result.returncode}:\n{stderr_tail}"
                )

            # Log stdout tail for debugging
            stdout_tail = (
                result.stdout[-400:]
                if len(result.stdout) > 400
                else result.stdout
            )
            logger.debug("Pointcept stdout (tail): %s", stdout_tail)

            # prediction.py writes output to <folder>/predictions/<filename>
            predicted_file = work_path / "predictions" / las_path.name
            if predicted_file.is_file():
                # Replace the original LAS with the classified version
                backup = las_path.with_suffix(las_path.suffix + ".bak")
                try:
                    shutil.move(str(las_path), str(backup))
                except OSError:
                    pass  # may already exist
                shutil.move(str(predicted_file), str(las_path))
                logger.debug("Replaced %s with classified version", las_path)

        # Refresh tile cache
        self._tm.clear_cache()
