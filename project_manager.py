"""
LiDAR Workbench — Project Manager.

Handles project lifecycle: creation, opening, saving, and validation
of the on-disk project structure.  Project metadata is persisted as
JSON alongside a SQLite database and LAS tile files.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import PROJECT_SUBDIRS, PROJECT_FILES
from .database import Database

logger = logging.getLogger("lidar_workbench.project_manager")


class ProjectManager:
    """
    Manages a LiDAR Workbench project on disk.

    A project consists of::

        project_root/
        ├── project.json
        ├── filter_settings.json
        ├── tile_database.sqlite
        ├── tiles/        (LAS/LAZ tile files)
        └── dtm/          (GeoTIFF DTM rasters)

    Usage::

        pm = ProjectManager()
        pm.create("/home/user/my_project", name="Highway Survey")
        # ... later ...
        pm2 = ProjectManager()
        pm2.open("/home/user/my_project")
        print(pm2.metadata["name"])  # "Highway Survey"
    """

    def __init__(self) -> None:
        self._project_root: Optional[Path] = None
        self._metadata: Dict[str, Any] = {}
        self._db: Optional[Database] = None

    # ── properties ─────────────────────────────────────────────────

    @property
    def project_root(self) -> Optional[Path]:
        """The project root directory, or ``None`` if no project is open."""
        return self._project_root

    @property
    def metadata(self) -> Dict[str, Any]:
        """In-memory project metadata dictionary (mutable)."""
        return self._metadata

    @property
    def db(self) -> Optional[Database]:
        """The :class:`Database` instance, or ``None`` if no project is open."""
        return self._db

    @property
    def tiles_dir(self) -> Optional[Path]:
        """Path to the ``tiles/`` subdirectory."""
        if self._project_root is None:
            return None
        return self._project_root / "tiles"

    @property
    def dtm_dir(self) -> Optional[Path]:
        """Path to the ``dtm/`` subdirectory."""
        if self._project_root is None:
            return None
        return self._project_root / "dtm"

    @property
    def is_open(self) -> bool:
        """``True`` when a project is loaded and ready."""
        return self._project_root is not None

    # ── project lifecycle ──────────────────────────────────────────

    def create(
        self,
        root_path: str | Path,
        name: str = "Untitled Project",
        overwrite: bool = False,
    ) -> Path:
        """
        Create a new project directory structure.

        Args:
            root_path:  Directory to create the project in.
            name:       Human-readable project name.
            overwrite:  If ``True``, remove an existing project at ``root_path``
                        before creation.

        Returns:
            The project root :class:`Path`.

        Raises:
            FileExistsError: If ``root_path`` already exists and ``overwrite``
                             is ``False``.
        """
        root_path = Path(root_path).resolve()

        if root_path.exists():
            if overwrite:
                logger.info("Removing existing project at %s", root_path)
                shutil.rmtree(root_path)
            else:
                raise FileExistsError(f"Project already exists at {root_path}")

        # Create directory scaffolding
        root_path.mkdir(parents=True, exist_ok=True)
        for sub in PROJECT_SUBDIRS:
            (root_path / sub).mkdir(exist_ok=True)

        # Initial metadata
        now = datetime.now(timezone.utc).isoformat()
        self._project_root = root_path

        self._metadata = {
            "name": name,
            "version": 1,
            "created": now,
            "last_opened": now,
            "description": "",
            "tile_size_m": 200.0,
            "tile_overlap_m": 10.0,
            "pointcept_path": "./PointceptALS",
            "model_path": "./models/model_best.pth",
            "config_path": "./configs/dales/ptv3_dales.py",
            "coordinate_system": "",
        }
        self._save_metadata()

        # Empty filter settings
        filter_settings_path = root_path / "filter_settings.json"
        filter_settings_path.write_text(
            json.dumps({"filters": []}, indent=2), encoding="utf-8"
        )

        # Initialise database
        self._db = Database(root_path / "tile_database.sqlite")
        self._db.initialize()

        logger.info("Created project '%s' at %s", name, root_path)
        return root_path

    def open(self, root_path: str | Path) -> Path:
        """
        Open an existing project.

        Args:
            root_path: Path to the project root directory.

        Returns:
            The project root :class:`Path`.

        Raises:
            FileNotFoundError: If ``root_path`` does not exist.
            ValueError:        If ``project.json`` is missing or corrupted.
        """
        root_path = Path(root_path).resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Project directory not found: {root_path}")

        meta_path = root_path / "project.json"
        if not meta_path.is_file():
            raise ValueError(f"Not a valid project — missing project.json in {root_path}")

        self._project_root = root_path
        self._load_metadata()

        # Touch last_opened
        self._metadata["last_opened"] = datetime.now(timezone.utc).isoformat()
        self._save_metadata()

        # Open database
        self._db = Database(root_path / "tile_database.sqlite")
        self._db.initialize()

        logger.info("Opened project '%s' from %s", self._metadata.get("name"), root_path)
        return root_path

    def save(self) -> None:
        """Persist in-memory metadata to ``project.json``."""
        if self._project_root is None:
            raise RuntimeError("No project is open")
        self._save_metadata()
        logger.debug("Project metadata saved to %s", self._project_root)

    def close(self) -> None:
        """Close the current project, saving metadata first."""
        if self._project_root is not None:
            self.save()
            logger.info("Closed project '%s'", self._metadata.get("name"))
        self._project_root = None
        self._metadata = {}
        self._db = None

    def validate(self) -> bool:
        """
        Check that the project directory structure is intact.

        Returns:
            ``True`` if all expected subdirs and files exist.
        """
        if self._project_root is None:
            return False
        for sub in PROJECT_SUBDIRS:
            if not (self._project_root / sub).is_dir():
                return False
        for fname in PROJECT_FILES:
            if not (self._project_root / fname).is_file():
                return False
        return True

    # ── helpers ────────────────────────────────────────────────────

    def _metadata_path(self) -> Path:
        assert self._project_root is not None
        return self._project_root / "project.json"

    def _save_metadata(self) -> None:
        """Write metadata to disk."""
        path = self._metadata_path()
        path.write_text(json.dumps(self._metadata, indent=2, default=str), encoding="utf-8")

    def _load_metadata(self) -> None:
        """Read metadata from disk."""
        path = self._metadata_path()
        self._metadata = json.loads(path.read_text(encoding="utf-8"))
