"""
LiDAR Workbench — Windows Auto-Updater.

Checks the HTTP `Last-Modified` date of the hosted Windows installer at
`https://lidar.kabelik.at/static/downloads/LiDAR_Workbench_Setup_v1.0_RTX30_40_50.exe`
against the local executable's build/install timestamp, downloads the new
installer with live progress, and launches Inno Setup to upgrade in-place.
"""

from __future__ import annotations

import datetime
import email.utils
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger("lidar_workbench.updater")

UPDATE_EXE_URL = "https://lidar.kabelik.at/static/downloads/LiDAR_Workbench_Setup_v1.0_RTX30_40_50.exe"
_STATE_FILE = Path.home() / ".lidar_workbench" / "update_state.json"


def _load_state() -> Dict[str, Any]:
    try:
        if _STATE_FILE.is_file():
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Could not save update state: %s", exc)


def get_local_build_timestamp() -> float:
    """Return local executable mtime (or recorded remote mtime if newer)."""
    local_ts = 0.0
    try:
        if getattr(sys, "frozen", False):
            local_ts = Path(sys.executable).stat().st_mtime
        else:
            local_ts = Path(__file__).stat().st_mtime
    except Exception:
        local_ts = 0.0

    state = _load_state()
    rec_ts = float(state.get("installed_remote_mtime") or 0.0)
    # Only trust recorded remote_ts if local exe hasn't changed since
    if rec_ts > local_ts and float(state.get("local_exe_mtime") or 0.0) == local_ts:
        return rec_ts
    return local_ts


def _fmt_ts(ts: float) -> str:
    if not ts:
        return "Unknown"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def check_for_update(url: str = UPDATE_EXE_URL, timeout: float = 8.0) -> Dict[str, Any]:
    """Send an HTTP HEAD request and compare `Last-Modified` against local build date."""
    local_ts = get_local_build_timestamp()
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "LiDAR-Workbench-Updater/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        lm_hdr = resp.headers.get("Last-Modified")
        cl_hdr = resp.headers.get("Content-Length")

    if not lm_hdr:
        return {
            "available": False,
            "error": "Server did not return a Last-Modified header.",
            "local_ts": local_ts,
            "local_date": _fmt_ts(local_ts),
        }

    dt = email.utils.parsedate_to_datetime(lm_hdr)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    remote_ts = dt.timestamp()
    size_bytes = int(cl_hdr) if cl_hdr and cl_hdr.isdigit() else 0

    # 5-minute tolerance to avoid false positives from clock/copy skew
    available = remote_ts > (local_ts + 300.0)
    return {
        "available": available,
        "remote_ts": remote_ts,
        "local_ts": local_ts,
        "remote_date": _fmt_ts(remote_ts),
        "local_date": _fmt_ts(local_ts),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 1) if size_bytes else 0.0,
        "url": url,
    }


class UpdateCheckWorker(QThread):
    """Non-blocking background worker to check for a newer installer."""

    finished_check = Signal(dict)

    def __init__(self, url: str = UPDATE_EXE_URL, parent=None):
        super().__init__(parent)
        self._url = url

    def run(self) -> None:
        try:
            res = check_for_update(self._url)
            self.finished_check.emit(res)
        except Exception as exc:
            self.finished_check.emit({"available": False, "error": str(exc)})


class UpdateDownloadWorker(QThread):
    """Streams the Windows installer .exe in chunks with progress & cancel support."""

    progress = Signal(int, int, float)  # received_bytes, total_bytes, mbps
    completed = Signal(str)             # path to downloaded installer
    error = Signal(str)

    def __init__(self, url: str = UPDATE_EXE_URL, parent=None):
        super().__init__(parent)
        self._url = url
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        dest = Path(tempfile.gettempdir()) / "LiDAR_Workbench_Setup_Update.exe"
        try:
            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": "LiDAR-Workbench-Updater/1.0"},
            )
            t0 = time.perf_counter()
            received = 0
            chunk_size = 2 * 1024 * 1024  # 2 MB
            with urllib.request.urlopen(req, timeout=30) as resp:
                cl = resp.headers.get("Content-Length")
                total = int(cl) if cl and cl.isdigit() else 0
                with open(dest, "wb") as f:
                    while True:
                        if self._cancelled:
                            break
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        received += len(chunk)
                        elapsed = max(0.1, time.perf_counter() - t0)
                        mbps = (received * 8.0) / elapsed / 1_000_000.0
                        self.progress.emit(received, total, mbps)

            if self._cancelled:
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
                return

            self.completed.emit(str(dest))
        except Exception as exc:
            self.error.emit(str(exc))


def launch_installer_and_exit(installer_path: str, remote_ts: float = 0.0) -> None:
    """Record installed timestamp and launch the Inno Setup installer."""
    local_exe_mtime = 0.0
    try:
        if getattr(sys, "frozen", False):
            local_exe_mtime = Path(sys.executable).stat().st_mtime
    except Exception:
        pass

    if remote_ts:
        _save_state({
            "installed_remote_mtime": remote_ts,
            "local_exe_mtime": local_exe_mtime,
        })

    args = [installer_path, "/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"]
    subprocess.Popen(args, close_fds=True)
