#!/usr/bin/env python3
"""
LiDAR Workbench — Application Entry Point.

Usage::

    python -m lidar_workbench.main
    # or
    python lidar_workbench/main.py

If a project path is given as the first argument, it will be opened
automatically on startup.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# Allow running directly (python main.py) as well as via -m
if __name__ == "__main__" and __package__ is None:
    import os as _os

    _pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
    _parent_dir = _os.path.dirname(_pkg_dir)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    __package__ = "lidar_workbench"

from .config import APP_NAME, APP_ORG, setup_logging
from .database import Database
from .project_manager import ProjectManager
from .tile_manager import TileManager
from .gui.main_window import MainWindow


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} — Interactive LiDAR Point Cloud Workbench"
    )
    parser.add_argument(
        "project",
        nargs="?",
        help="Path to an existing project directory to open on startup.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write log output to this file (default: stderr only).",
    )
    return parser.parse_args(argv[1:])


def main() -> int:
    """
    Initialise the application, wire components, and start the event loop.

    Returns:
        Exit code (0 on success).
    """
    args = _parse_args(sys.argv)

    # Logging
    logger = setup_logging(args.log_file, level=logging.DEBUG)
    logger.info("LiDAR Workbench %s starting", "0.1.0")

    # Qt application (high-DPI is automatic in Qt 6)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)

    # ── Initialize Open3D GUI (must happen once after QApplication exists) ──
    try:
        import open3d.visualization.gui as o3d_gui
        o3d_gui.Application.instance.initialize()
        logger.info("Open3D GUI initialised")
    except Exception as exc:
        logger.warning("Open3D GUI not available: %s", exc)

    # Core components (project manager and DB are created, but a project
    # may not be open yet).
    pm = ProjectManager()

    # Attempt to open project from command-line argument
    if args.project:
        try:
            pm.open(args.project)
            logger.info("Opened project from CLI: %s", args.project)
        except Exception as exc:
            logger.error("Failed to open project '%s': %s", args.project, exc)
            print(f"Error: Could not open project '{args.project}': {exc}", file=sys.stderr)
            # Continue without a project — user can create/open via GUI

    # Database and tile manager
    if pm.is_open and pm.db is not None:
        db = pm.db
    else:
        # Use an in-memory database when no project is open
        db = Database(":memory:")
        db.initialize()

    tm = TileManager(pm, db)

    # Main window
    window = MainWindow(pm, tm, db)
    window.show()

    # Run
    exit_code = app.exec()

    # Cleanup
    if pm.is_open:
        pm.close()
    logger.info("LiDAR Workbench shutting down (code %d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
