"""
LiDAR Workbench — SQLite Database Layer.

Provides a thin ORM-style wrapper around SQLite for tile metadata and
edit history.  All public methods accept a connection or create one
internally via the context manager.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .config import TileStatus

logger = logging.getLogger("lidar_workbench.database")

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
    flight_line     INTEGER DEFAULT 0,
    scanner         TEXT    DEFAULT '',
    all_scanners    TEXT    DEFAULT '[]',  -- JSON list of all scanner names in this tile
    sensor_type     TEXT    DEFAULT '' CHECK(sensor_type IN ('', 'topo', 'bathy')),
    flightline_sensor_types TEXT DEFAULT '{}',  -- JSON: {"1": "topo", "2": "bathy"}
    crs_epsg        INTEGER,               -- EPSG code (e.g. 32633)
    crs_wkt         TEXT,                  -- OGC WKT string for the CRS
    status          TEXT    NOT NULL DEFAULT 'IMPORTED'
                    CHECK(status IN ('IMPORTED','FILTERED','CLASSIFIED','EDITED','NOISE','ERROR')),
    qc_status       TEXT    DEFAULT NULL,   -- QC review status: NULL, 'QC_PASSED', 'IN_REVIEW', 'NEEDS_REWORK'
    qc_comment      TEXT    DEFAULT NULL,   -- optional comment (e.g. rework instructions)
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

CREATE TABLE IF NOT EXISTS processing_times (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id         TEXT    NOT NULL,
    step            TEXT    NOT NULL CHECK(step IN ('filter','pointcept','ground')),
    duration_seconds REAL   NOT NULL,
    point_count     INTEGER,
    params          TEXT,   -- JSON with step-specific parameters
    batch_id        TEXT,   -- groups tiles processed together in one run
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tile_id) REFERENCES tiles(id)
);

CREATE INDEX IF NOT EXISTS idx_pt_tile ON processing_times(tile_id);
CREATE INDEX IF NOT EXISTS idx_pt_step ON processing_times(step);
CREATE INDEX IF NOT EXISTS idx_pt_batch ON processing_times(batch_id);
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
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add new columns that may be missing from older databases."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(tiles)")}
        migrations = [
            ("flight_line", "ALTER TABLE tiles ADD COLUMN flight_line INTEGER DEFAULT 0"),
            ("scanner",     "ALTER TABLE tiles ADD COLUMN scanner TEXT DEFAULT ''"),
            ("sensor_type", "ALTER TABLE tiles ADD COLUMN sensor_type TEXT DEFAULT ''"),
            ("flightline_sensor_types", "ALTER TABLE tiles ADD COLUMN flightline_sensor_types TEXT DEFAULT '{}'"),
            ("all_scanners", "ALTER TABLE tiles ADD COLUMN all_scanners TEXT DEFAULT '[]'"),
            ("qc_status",   "ALTER TABLE tiles ADD COLUMN qc_status TEXT DEFAULT NULL"),
            ("qc_comment",  "ALTER TABLE tiles ADD COLUMN qc_comment TEXT DEFAULT NULL"),
        ]
        for col, sql in migrations:
            if col not in existing:
                conn.execute(sql)
                logger.debug("Migrated tiles table: added column %s", col)

        # Migrate: add NOISE to status CHECK constraint (v0.2)
        self._migrate_status_constraint(conn)

        # Ensure processing_times table exists (v0.3+)
        self._ensure_table(
            conn, "processing_times",
            """CREATE TABLE IF NOT EXISTS processing_times (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tile_id         TEXT    NOT NULL,
                step            TEXT    NOT NULL CHECK(step IN ('filter','pointcept','ground')),
                duration_seconds REAL   NOT NULL,
                point_count     INTEGER,
                params          TEXT,
                batch_id        TEXT,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tile_id) REFERENCES tiles(id)
            )""",
        )
        self._ensure_table(
            conn, "idx_pt_tile", "CREATE INDEX IF NOT EXISTS idx_pt_tile ON processing_times(tile_id)",
        )
        self._ensure_table(
            conn, "idx_pt_step", "CREATE INDEX IF NOT EXISTS idx_pt_step ON processing_times(step)",
        )
        self._ensure_table(
            conn, "idx_pt_batch", "CREATE INDEX IF NOT EXISTS idx_pt_batch ON processing_times(batch_id)",
        )

    def _ensure_table(self, conn: sqlite3.Connection, name: str, sql: str) -> None:
        """Create a table/index if it does not already exist."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index') AND name = ?",
            (name,),
        ).fetchone()
        if row is None:
            conn.execute(sql)
            logger.debug("Migrated: created %s", name)

    def _migrate_status_constraint(self, conn: sqlite3.Connection) -> None:
        """Recreate the tiles table if the status CHECK constraint lacks 'NOISE'."""
        # Check current constraint definition
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tiles'"
        ).fetchone()
        if row and 'NOISE' in row[0]:
            return  # already migrated

        logger.info("Migrating tiles.status constraint to include NOISE…")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        try:
            conn.executescript("""
                CREATE TABLE tiles_new (
                    id              TEXT PRIMARY KEY,
                    filename        TEXT    NOT NULL,
                    bbox_min_x      REAL,
                    bbox_min_y      REAL,
                    bbox_max_x      REAL,
                    bbox_max_y      REAL,
                    point_count     INTEGER,
                    flight_line     INTEGER DEFAULT 0,
                    scanner         TEXT    DEFAULT '',
                    all_scanners    TEXT    DEFAULT '[]',
                    sensor_type     TEXT    DEFAULT '' CHECK(sensor_type IN ('', 'topo', 'bathy')),
                    flightline_sensor_types TEXT DEFAULT '{}',
                    crs_epsg        INTEGER,
                    crs_wkt         TEXT,
                    status          TEXT    NOT NULL DEFAULT 'IMPORTED'
                                    CHECK(status IN ('IMPORTED','FILTERED','CLASSIFIED','EDITED','NOISE','ERROR')),
                    filter_params   TEXT,
                    classification_model TEXT,
                    last_modified   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                INSERT INTO tiles_new SELECT * FROM tiles;
                DROP TABLE tiles;
                ALTER TABLE tiles_new RENAME TO tiles;
                CREATE INDEX IF NOT EXISTS idx_tiles_status ON tiles(status);
                CREATE INDEX IF NOT EXISTS idx_tiles_bbox  ON tiles(bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y);
            """)
            conn.execute("PRAGMA foreign_keys = ON")
            logger.info("Status constraint migration complete.")
        except Exception:
            conn.execute("ROLLBACK")
            conn.execute("PRAGMA foreign_keys = ON")
            raise

    # ── tile CRUD ──────────────────────────────────────────────────

    def insert_tile(
        self,
        conn: sqlite3.Connection,
        tile_id: str,
        filename: str,
        bbox: Optional[tuple[float, float, float, float]] = None,
        point_count: int = 0,
        flight_line: int = 0,
        scanner: str = '',
        all_scanners: Optional[List[str]] = None,
        sensor_type: str = '',
        flightline_sensor_types: Optional[Dict[int, str]] = None,
        crs_epsg: Optional[int] = None,
        crs_wkt: Optional[str] = None,
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
            flight_line:      Flight line number (0 = unknown).
            scanner:          Dominant scanner/sensor name (e.g. 'vq820g').
            all_scanners:     List of all unique scanner names found in this tile.
            sensor_type:      'topo', 'bathy', or ''.
            flightline_sensor_types: Optional dict mapping flight_line -> sensor_type.
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
            flight_line,
            scanner,
            json.dumps(all_scanners) if all_scanners else '[]',
            sensor_type,
            json.dumps(flightline_sensor_types) if flightline_sensor_types else '{}',
            crs_epsg,
            crs_wkt,
            status,
            json.dumps(filter_params) if filter_params else None,
            classification_model,
        )
        with Database._write_lock:
            conn.execute(
                """INSERT OR REPLACE INTO tiles
                   (id, filename, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y,
                    point_count, flight_line, scanner, all_scanners, sensor_type,
                    flightline_sensor_types, crs_epsg, crs_wkt,
                    status, filter_params, classification_model, last_modified)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                vals,
            )

    def set_tile_crs(
        self, conn: sqlite3.Connection, tile_id: str,
        crs_epsg: Optional[int], crs_wkt: Optional[str],
    ) -> None:
        """Set the CRS for a tile."""
        with Database._write_lock:
            conn.execute(
                "UPDATE tiles SET crs_epsg = ?, crs_wkt = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                (crs_epsg, crs_wkt, tile_id),
            )

    def get_project_crs(self) -> Optional[dict]:
        """Return the first valid CRS from any tile, or None."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT crs_epsg, crs_wkt FROM tiles WHERE crs_epsg IS NOT NULL OR crs_wkt IS NOT NULL LIMIT 1"
            ).fetchone()
        if row:
            return {"epsg": row["crs_epsg"], "wkt": row["crs_wkt"]}
        return None

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

    def set_qc_status(
        self, conn: sqlite3.Connection, tile_id: str,
        qc_status: Optional[str], qc_comment: Optional[str] = None,
    ) -> None:
        """
        Set the QC review status and optional comment for a tile.

        Args:
            conn:       Open SQLite connection.
            tile_id:    Tile identifier.
            qc_status:  One of 'QC_PASSED', 'IN_REVIEW', 'NEEDS_REWORK', or None to clear.
            qc_comment: Optional free-text comment (e.g. rework instructions).
        """
        with Database._write_lock:
            conn.execute(
                "UPDATE tiles SET qc_status = ?, qc_comment = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                (qc_status, qc_comment, tile_id),
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

    # ── processing times ───────────────────────────────────────────

    def record_processing_time(
        self,
        conn: sqlite3.Connection,
        tile_id: str,
        step: str,
        duration_seconds: float,
        point_count: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        batch_id: Optional[str] = None,
    ) -> int:
        """
        Record a processing time measurement for a tile.

        Args:
            conn:             Open SQLite connection.
            tile_id:          Tile identifier.
            step:             Processing step ('filter', 'pointcept', 'ground').
            duration_seconds: Elapsed wall-clock time in seconds.
            point_count:      Number of points processed (optional).
            params:           Optional step-specific parameters dict.
            batch_id:         Optional batch identifier (groups tiles from one run).

        Returns:
            The auto-generated row id.
        """
        with Database._write_lock:
            cur = conn.execute(
                """INSERT INTO processing_times
                   (tile_id, step, duration_seconds, point_count, params, batch_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (
                    tile_id,
                    step,
                    duration_seconds,
                    point_count,
                    json.dumps(params) if params else None,
                    batch_id,
                ),
            )
        return cur.lastrowid

    def get_processing_times(
        self,
        tile_ids: Optional[List[str]] = None,
        step: Optional[str] = None,
        batch_id: Optional[str] = None,
        since_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query processing time records with optional filters.

        Args:
            tile_ids:        Optional list of tile IDs to filter by.
            step:            Optional step filter ('filter', 'pointcept', 'ground').
            batch_id:        Optional batch filter.
            since_timestamp: Optional ISO timestamp to filter records after.

        Returns:
            List of processing time records as dicts.
        """
        with self.connect() as conn:
            query = "SELECT * FROM processing_times WHERE 1=1"
            params: list = []

            if tile_ids:
                placeholders = ",".join("?" for _ in tile_ids)
                query += f" AND tile_id IN ({placeholders})"
                params.extend(tile_ids)

            if step:
                query += " AND step = ?"
                params.append(step)

            if batch_id:
                query += " AND batch_id = ?"
                params.append(batch_id)

            if since_timestamp:
                query += " AND timestamp >= ?"
                params.append(since_timestamp)

            query += " ORDER BY timestamp DESC"
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_processing_stats(
        self,
        step: Optional[str] = None,
        batch_ids: Optional[List[str]] = None,
        tile_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics for processing times.

        Args:
            step:      Optional step filter.
            batch_ids: Optional list of batch IDs.
            tile_ids:  Optional list of tile IDs.

        Returns:
            Dict with keys: count, total_seconds, mean, median, min, max, std,
            p25, p75, by_step (breakdown), by_batch (breakdown).
        """
        with self.connect() as conn:
            query = "SELECT * FROM processing_times WHERE 1=1"
            params: list = []

            if step:
                query += " AND step = ?"
                params.append(step)

            if batch_ids:
                placeholders = ",".join("?" for _ in batch_ids)
                query += f" AND batch_id IN ({placeholders})"
                params.extend(batch_ids)

            if tile_ids:
                placeholders = ",".join("?" for _ in tile_ids)
                query += f" AND tile_id IN ({placeholders})"
                params.extend(tile_ids)

            rows = conn.execute(query, params).fetchall()
            durations = [r["duration_seconds"] for r in rows]

        if not durations:
            return {
                "count": 0, "total_seconds": 0.0, "mean": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0,
                "by_step": {}, "by_batch": {},
            }

        # Use plain Python for statistics (no numpy)
        sorted_d = sorted(durations)
        n = len(sorted_d)

        def _percentile(data: list, p: float) -> float:
            """Linear-interpolation percentile (no numpy needed)."""
            k = (len(data) - 1) * p / 100.0
            f = int(k)
            c = k - f
            if f + 1 < len(data):
                return data[f] + c * (data[f + 1] - data[f])
            return data[f]

        stats = {
            "count": n,
            "total_seconds": sum(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
            "std": statistics.stdev(durations) if n >= 2 else 0.0,
            "p25": _percentile(sorted_d, 25),
            "p75": _percentile(sorted_d, 75),
        }

        # Breakdown by step
        by_step: Dict[str, Dict[str, float]] = {}
        for r in rows:
            s = r["step"]
            if s not in by_step:
                by_step[s] = {"count": 0, "total_seconds": 0.0, "durations": []}
            by_step[s]["count"] += 1
            by_step[s]["total_seconds"] += r["duration_seconds"]
            by_step[s]["durations"].append(r["duration_seconds"])
        for s, v in by_step.items():
            d = v.pop("durations")
            v["mean"] = statistics.mean(d) if d else 0.0
            v["median"] = statistics.median(d) if d else 0.0
            v["min"] = min(d) if d else 0.0
            v["max"] = max(d) if d else 0.0
        stats["by_step"] = by_step

        # Breakdown by batch
        by_batch: Dict[str, Dict[str, float]] = {}
        for r in rows:
            b = r["batch_id"] or "(no batch)"
            if b not in by_batch:
                by_batch[b] = {"count": 0, "total_seconds": 0.0, "durations": [],
                               "timestamp": r["timestamp"] or ""}
            by_batch[b]["count"] += 1
            by_batch[b]["total_seconds"] += r["duration_seconds"]
            by_batch[b]["durations"].append(r["duration_seconds"])
        for b, v in by_batch.items():
            d = v.pop("durations")
            v["mean"] = statistics.mean(d) if d else 0.0
            v["median"] = statistics.median(d) if d else 0.0
            v["min"] = min(d) if d else 0.0
            v["max"] = max(d) if d else 0.0
        stats["by_batch"] = by_batch

        return stats

    def get_distinct_batches(self) -> List[Dict[str, Any]]:
        """Return a list of distinct batch IDs with their timestamps and step counts."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT batch_id, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts,
                          COUNT(*) as tile_count, GROUP_CONCAT(DISTINCT step) as steps
                   FROM processing_times
                   WHERE batch_id IS NOT NULL
                   GROUP BY batch_id
                   ORDER BY first_ts DESC"""
            ).fetchall()
        return [dict(r) for r in rows]
