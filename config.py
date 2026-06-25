"""
LiDAR Workbench — Central Configuration Module.

Defines application-wide constants: ASPRS class colors, default tiling
parameters, project directory layout, and logging setup helpers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Final, Tuple

# ── Application metadata ──────────────────────────────────────────────
APP_NAME: Final[str] = "LiDAR Workbench"
APP_VERSION: Final[str] = "0.1.0"
APP_ORG: Final[str] = "LiDAR-Workbench"

# ── Default paths ─────────────────────────────────────────────────────
# Pointcept is bundled inside the project (see Pointcept/prediction.py)
DEFAULT_POINTCEPT_PATH: Final[str] = "./Pointcept"
DEFAULT_MODEL_PATH: Final[str] = "./models/model_best.pth"
DEFAULT_CONFIG_PATH: Final[str] = "./configs/dales/ptv3_dales.py"

# ── Tiling defaults ───────────────────────────────────────────────────
DEFAULT_TILE_SIZE_M: Final[float] = 200.0       # meters
DEFAULT_TILE_OVERLAP_M: Final[float] = 10.0     # meters
TARGET_POINTS_PER_TILE: Final[int] = 1_500_000  # ~1.5 million
MAX_POINTS_PER_VIEW: Final[int] = 500_000     # software-renderer budget

# ── Preview defaults ──────────────────────────────────────────────────
PREVIEW_LOD_BBOX: Final[str] = "bbox"
PREVIEW_LOD_SUBSAMPLED_1M: Final[str] = "sub_1m"
PREVIEW_LOD_SUBSAMPLED_10M: Final[str] = "sub_10m"
PREVIEW_LOD_FULL: Final[str] = "full"
PREVIEW_LOD_OPTIONS: Final[tuple] = (
    (PREVIEW_LOD_BBOX, "Bounding Box Only"),
    (PREVIEW_LOD_SUBSAMPLED_1M, "Subsampled (~1M points)"),
    (PREVIEW_LOD_SUBSAMPLED_10M, "Subsampled (~10M points)"),
    (PREVIEW_LOD_FULL, "Full Resolution"),
)
PREVIEW_CHUNK_SIZE: Final[int] = 1_000_000        # points per read chunk
PREVIEW_FULL_WARN_THRESHOLD: Final[int] = 50_000_000  # warn above this

# ── Filter defaults ───────────────────────────────────────────────────
DEFAULT_SOR_NB_NEIGHBORS: Final[int] = 20
DEFAULT_SOR_STD_RATIO: Final[float] = 2.0
DEFAULT_ROR_RADIUS: Final[float] = 1.0
DEFAULT_ROR_MIN_POINTS: Final[int] = 5
DEFAULT_FILTER_WORKERS: Final[int] = 4     # parallel filter threads
DEFAULT_CLASSIFY_WORKERS: Final[int] = 1  # parallel classify processes (GPU-heavy)

# ── Profile defaults ──────────────────────────────────────────────────
DEFAULT_PROFILE_WIDTH_M: Final[float] = 5.0     # meters
DEFAULT_BRUSH_RADIUS_M: Final[float] = 2.0      # meters

# ── Pointcept defaults ────────────────────────────────────────────────
DEFAULT_BATCH_SIZE: Final[int] = 1
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.5

# ── ASPRS class colours (RGB, 0-1 range for Open3D) ───────────────────
# Key: ASPRS classification code
# https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf
ASPRS_CLASS_COLORS: Final[Dict[int, Tuple[float, float, float]]] = {
    0:  (0.5, 0.5, 0.5),      # Created, Never Classified  → grey
    1:  (0.8, 0.8, 0.8),      # Unclassified               → light grey
    2:  (0.55, 0.27, 0.07),   # Ground                      → brown
    3:  (0.0, 0.6, 0.0),      # Low Vegetation              → dark green
    4:  (0.0, 0.8, 0.0),      # Medium Vegetation           → green
    5:  (0.2, 1.0, 0.2),      # High Vegetation             → bright green
    6:  (0.8, 0.2, 0.2),      # Building                    → red
    7:  (0.3, 0.3, 0.3),      # Low Point (noise)           → dark grey
    8:  (0.9, 0.9, 0.0),      # Model Key/Reserved          → yellow
    9:  (0.0, 0.4, 0.8),      # Water                       → blue
    10: (0.6, 0.6, 0.6),      # Rail                        → mid grey
    11: (0.6, 0.6, 0.6),      # Road Surface                → mid grey
    12: (0.7, 0.3, 0.0),      # Overlap/Reserved            → orange
    13: (1.0, 1.0, 1.0),      # Wire – Guard (Shield)       → white
    14: (0.5, 0.5, 0.5),      # Wire – Conductor (Phase)    → grey
    15: (0.9, 0.5, 0.9),      # Transmission Tower          → pink
    16: (0.7, 0.7, 0.7),      # Wire-Structure Connector    → silver
    17: (0.3, 0.7, 0.7),      # Bridge Deck                 → teal
    18: (0.9, 0.1, 0.1),      # High Noise                  → strong red
}

# Default colour for unknown classes
FALLBACK_CLASS_COLOR: Final[Tuple[float, float, float]] = (0.0, 0.0, 0.0)

# ── ASPRS class labels ────────────────────────────────────────────────
ASPRS_CLASS_NAMES: Final[Dict[int, str]] = {
    0:  "Created, Never Classified",
    1:  "Unclassified",
    2:  "Ground",
    3:  "Low Vegetation",
    4:  "Medium Vegetation",
    5:  "High Vegetation",
    6:  "Building",
    7:  "Low Point (Noise)",
    8:  "Model Key/Reserved",
    9:  "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Overlap/Reserved",
    13: "Wire – Guard (Shield)",
    14: "Wire – Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-Structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
}

# ── Tile status constants ─────────────────────────────────────────────
class TileStatus:
    """Enumeration of tile processing statuses."""
    IMPORTED: Final[str] = "IMPORTED"
    FILTERED: Final[str] = "FILTERED"
    CLASSIFIED: Final[str] = "CLASSIFIED"
    EDITED: Final[str] = "EDITED"
    ERROR: Final[str] = "ERROR"

    ALL: Final[Tuple[str, ...]] = (IMPORTED, FILTERED, CLASSIFIED, EDITED, ERROR)

# ── Project directory layout ──────────────────────────────────────────
PROJECT_SUBDIRS: Final[Tuple[str, ...]] = ("tiles", "dtm")
PROJECT_FILES: Final[Tuple[str, ...]] = ("project.json", "filter_settings.json", "tile_database.sqlite")


def setup_logging(log_path: str | Path | None = None, level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure the root logger for LiDAR Workbench.

    Args:
        log_path: Optional file path for log output. If ``None``, logs
                  are written to stderr only.
        level:   Logging level (default: ``logging.DEBUG``).

    Returns:
        The configured root logger.
    """
    logger = logging.getLogger("lidar_workbench")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_class_color(classification: int) -> Tuple[float, float, float]:
    """
    Return the RGB colour for an ASPRS classification code.

    Args:
        classification: Integer classification code (0–18, 1-based per LAS spec).

    Returns:
        RGB tuple with values in [0.0, 1.0].
    """
    return ASPRS_CLASS_COLORS.get(classification, FALLBACK_CLASS_COLOR)


def get_class_name(classification: int) -> str:
    """
    Return the human-readable name for an ASPRS classification code.

    Args:
        classification: Integer classification code.

    Returns:
        Class name string, or ``"Unknown (<code>)"`` for unrecognised codes.
    """
    return ASPRS_CLASS_NAMES.get(classification, f"Unknown ({classification})")
