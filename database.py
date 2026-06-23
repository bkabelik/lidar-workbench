"""
LiDAR Workbench — SQLite Database Layer.

Provides a thin ORM-style wrapper around SQLite for tile metadata and
edit history.  All public methods accept a connection or create one
internally via the context manager.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .config import TileStatus

# ── SQL schema ─────────────────────────────────────────────────────────
SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS tiles (
    id              TEXT PRIMARY KEY,
    filename        TEXT    NOT NULL,
    bbox_min_x      REAL,
    bbox_min_y      REAL,
    bbox_max_x      REAL,
    bbox_max_y      REAL,
    point_count     INTEGER,
    status          TEXT    NOT NULL DEFAULT 'IMPORTED'
                    CHECK(status IN ('IMPORTED','FILTERED','CLASSIFIED','EDITED','ERROR')),
    filter_params   TEXT,   -- JSON
    classification_model TEXT,
    last_modified   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edit_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id     TEXT    NOT NULL,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    command     TEXT,   -- JSON
    FOREIGN KEY (tile_id) REFERENCES tiles(id)
);

CREATE INDEX IF NOT EXISTS idx_tiles_status ON tiles(status);
CREATE INDEX IF NOT EXISTS idx_tiles_bbox  ON tiles(bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y);
CREATE INDEX IF NOT EXISTS idx_edit_tile   ON edit_history(tile_id);
"""


class Database:
    """
    SQLite database handler for tile metadata and edit history.

    Thread-safe: each thread should use its own connection via
    :meth:`connect` or the context manager.  SQLite writes are
    serialised via a module-level lock.

    Usage::

        db = Database("project/tile_database.sqlite")
        db.initialize()

        with db.connect() as conn:
            db.insert_tile(conn, tile_id="tile_001", filename="tile_001.las", ...)

        tiles = db.get_all_tiles()
    """

    _write_lock = threading.Lock()

    def __init__(self, db_path: str | Path) -> None:
        """
        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)

    # ── connection management ──────────────────────────────────────

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager yielding a thread-local SQLite connection.

        The connection has :attr:`row_factory` set to ``sqlite3.Row``
        for dict-like access and WAL journaling enabled automatically.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Convenience: open a connection without context manager."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── schema ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ── tile CRUD ──────────────────────────────────────────────────

    def insert_tile(
        self,
        conn: sqlite3.Connection,
        tile_id: str,
        filename: str,
        bbox: Optional[tuple[float, float, float, float]] = None,
        point_count: int = 0,
        status: str = TileStatus.IMPORTED,
        filter_params: Optional[Dict[str, Any]] = None,
        classification_model: Optional[str] = None,
    ) -> None:
        """
        Insert a new tile row (or replace if the id already exists).

        Args:
            conn:             An open SQLite connection (from :meth:`connect`).
            tile_id:          Unique tile identifier.
            filename:         LAS/LAZ file name (relative to project tiles dir).
            bbox:             ``(min_x, min_y, max_x, max_y)`` in CRS units.
            point_count:      Number of points in the tile.
            status:           One of :class:`TileStatus`.
            filter_params:    Optional dict serialised to JSON.
            classification_model: Model name/path used for classification.
        """
        vals = (
            tile_id,
            filename,
            bbox[0] if bbox else None,
            bbox[1] if bbox else None,
            bbox[2] if bbox else None,
            bbox[3] if bbox else None,
            point_count,
            status,
            json.dumps(filter_params) if filter_params else None,
            classification_model,
        )
        with Database._write_lock:
            conn.execute(
                """INSERT OR REPLACE INTO tiles
                   (id, filename, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y,
                    point_count, status, filter_params, classification_model, last_modified)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                vals,
            )

    def update_status(self, conn: sqlite3.Connection, tile_id: str, status: str) -> None:
        """
        Update the processing status of a tile.

        Args:
            conn:    Open SQLite connection.
            tile_id: Tile identifier.
            status:  New status (must be a valid :class:`TileStatus` value).
        """
        with Database._write_lock:
            conn.execute(
                "UPDATE tiles SET status = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                (status, tile_id),
            )

    def update_point_count(self, conn: sqlite3.Connection, tile_id: str, count: int) -> None:
        """Update the point count for a tile."""
        with Database._write_lock:
            conn.execute(
                "UPDATE tiles SET point_count = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                (count, tile_id),
            )

    def update_filter_params(
        self, conn: sqlite3.Connection, tile_id: str, params: Dict[str, Any]
    ) -> None:
        """Persist filter parameters as JSON."""
        with Database._write_lock:
            conn.execute(
                "UPDATE tiles SET filter_params = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(params), tile_id),
            )

    def get_tile(self, tile_id: str) -> Optional[Dict[str, Any]]:
        """Return a single tile row as a dict, or ``None``."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,)).fetchone()
        return dict(row) if row else None

    def get_all_tiles(self) -> List[Dict[str, Any]]:
        """Return all tile rows ordered by status then id."""
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM tiles ORDER BY status, id").fetchall()
        return [dict(r) for r in rows]

    def get_tiles_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Return tiles filtered by processing status."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tiles WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_tiles_in_bbox(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> List[Dict[str, Any]]:
        """
        Return tiles whose bounding box intersects the query bbox.

        Uses a simple overlap test; for production workloads with many
        tiles an R-tree index would be more appropriate.
        """
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM tiles
                   WHERE bbox_max_x >= ? AND bbox_min_x <= ?
                     AND bbox_max_y >= ? AND bbox_min_y <= ?
                   ORDER BY id""",
                (min_x, max_x, min_y, max_y),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_tile(self, conn: sqlite3.Connection, tile_id: str) -> None:
        """Remove a tile and its edit history (cascading)."""
        with Database._write_lock:
            conn.execute("DELETE FROM edit_history WHERE tile_id = ?", (tile_id,))
            conn.execute("DELETE FROM tiles WHERE id = ?", (tile_id,))

    # ── edit history ───────────────────────────────────────────────

    def add_edit_command(
        self,
        conn: sqlite3.Connection,
        tile_id: str,
        command: Dict[str, Any],
    ) -> int:
        """
        Record an edit operation for undo/redo purposes.

        Args:
            conn:    Open SQLite connection.
            tile_id: Tile the edit applies to.
            command: JSON-serialisable dict describing the operation,
                     e.g. ``{"type":"classify","point_indices":[...],"old_class":1,"new_class":2}``.

        Returns:
            The auto-generated row id.
        """
        with Database._write_lock:
            cur = conn.execute(
                "INSERT INTO edit_history (tile_id, command, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (tile_id, json.dumps(command)),
            )
        return cur.lastrowid

    def get_edit_history(
        self, tile_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return the most recent edit commands for a tile, newest first."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edit_history WHERE tile_id = ? ORDER BY id DESC LIMIT ?",
                (tile_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_edit_history(self, conn: sqlite3.Connection, tile_id: str) -> None:
        """Remove all edit history for a tile."""
        with Database._write_lock:
            conn.execute("DELETE FROM edit_history WHERE tile_id = ?", (tile_id,))
