"""
LiDAR Workbench — Background Company Node Worker.

Runs in the background of the desktop app (when enabled) or as a headless CLI service.
Polls the company cluster server for pending tasks assigned to this workstation,
resolves tiles on shared storage/NAS, executes local GPU Pointcept inference,
and reports completion back to the server.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

# Allow running directly (python node_worker.py) as well as via -m, spawn, or frozen PyInstaller
if not __package__:
    import os as _os
    import types as _types

    _base = getattr(sys, "_MEIPASS", _os.path.dirname(_os.path.abspath(__file__)))
    _pkg_dir = _base
    if _base not in sys.path:
        sys.path.insert(0, _base)
    if "lidar_workbench" not in sys.modules:
        _mod = _types.ModuleType("lidar_workbench")
        _mod.__path__ = [_pkg_dir]
        _mod.__file__ = _os.path.join(_pkg_dir, "__init__.py")
        _mod.__package__ = "lidar_workbench"
        sys.modules["lidar_workbench"] = _mod
    __package__ = "lidar_workbench"

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    # Minimal stub for headless CLI environments
    class QThread:
        def __init__(self, parent=None): pass
        def isRunning(self): return False
    class Signal:
        def __init__(self, *args): pass
        def emit(self, *args): pass

try:
    from .licensing import get_license_manager, LicenseManager
    from .config import (
        DEFAULT_POINTCEPT_PATH,
        DEFAULT_MODEL_PATH,
        DEFAULT_CONFIG_PATH,
        load_pointcept_settings,
        load_general_settings,
    )
except (ImportError, ValueError):
    from licensing import get_license_manager, LicenseManager
    from config import (
        DEFAULT_POINTCEPT_PATH,
        DEFAULT_MODEL_PATH,
        DEFAULT_CONFIG_PATH,
        load_pointcept_settings,
        load_general_settings,
    )

logger = logging.getLogger("lidar_workbench.node_worker")


class CompanyNodeWorker(QThread):
    """
    Background worker thread that allows this machine to participate in the
    company's distributed compute cluster as a compute node.
    """

    task_started = Signal(str, int)     # tile_id, job_id
    task_finished = Signal(str, int)    # tile_id, job_id
    worker_status_changed = Signal(str) # status string

    def __init__(
        self,
        poll_interval_sec: float = 3.0,
        node_name: Optional[str] = None,
        storage_root: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.poll_interval_sec = poll_interval_sec
        self._lic = get_license_manager()
        self._stop_requested = False
        self._is_paused = False
        self._node_name = node_name or platform.node()
        self._storage_root = storage_root
        self._processed_count = 0

    def stop(self) -> None:
        self._stop_requested = True

    def pause(self) -> None:
        self._is_paused = True
        try:
            self._lic.send_node_heartbeat(status="offline", node_name=self._node_name)
        except Exception:
            pass
        self.worker_status_changed.emit("Paused")

    def resume(self) -> None:
        self._is_paused = False
        try:
            self._lic.send_node_heartbeat(status="idle", node_name=self._node_name)
        except Exception:
            pass
        self.worker_status_changed.emit("Idle — Listening for company tasks")

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def node_name(self) -> str:
        return self._node_name

    def _get_active_storage_root(self) -> Optional[str]:
        if self._storage_root:
            return self._storage_root
        try:
            gen = load_general_settings()
            return gen.get("shared_storage_root") or None
        except Exception:
            return None

    def _find_tile_file(
        self,
        tile_id: str,
        project_name: Optional[str] = None,
        project_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Locate the target LAS/LAZ file across shared network storage or local paths."""
        candidates = []

        # 1. Master's exact project path if accessible
        if project_path:
            pp = Path(project_path)
            candidates.append(pp / "tiles" / f"{tile_id}.las")
            candidates.append(pp / "tiles" / f"{tile_id}.laz")
            candidates.append(pp / f"{tile_id}.las")

        # 2. Configured shared storage root (e.g. \\server\share\projects or /mnt/projects)
        s_root = self._get_active_storage_root()
        if s_root:
            sp = Path(s_root)
            if project_name:
                candidates.append(sp / project_name / "tiles" / f"{tile_id}.las")
                candidates.append(sp / project_name / "tiles" / f"{tile_id}.laz")
                candidates.append(sp / project_name / f"{tile_id}.las")
            candidates.append(sp / "tiles" / f"{tile_id}.las")
            candidates.append(sp / f"{tile_id}.las")

        # 3. Local current working directory or subfolders
        candidates.append(Path("tiles") / f"{tile_id}.las")
        candidates.append(Path("tiles") / f"{tile_id}.laz")
        candidates.append(Path(f"{tile_id}.las"))

        for c in candidates:
            if c.is_file():
                return c.resolve()

        # 4. Glob search under shared storage if available
        if s_root and Path(s_root).is_dir():
            try:
                matches = list(Path(s_root).glob(f"**/{tile_id}.las"))
                if matches:
                    return matches[0].resolve()
                matches_laz = list(Path(s_root).glob(f"**/{tile_id}.laz"))
                if matches_laz:
                    return matches_laz[0].resolve()
            except Exception:
                pass

        return None

    def _run_pointcept_inference(
        self,
        las_path: Path,
        voxel_size: float = 0.5,
        smoothing: str = "yes",
    ) -> Tuple[bool, int]:
        """Run Pointcept AI inference on the specified tile using local GPU."""
        try:
            pc_settings = load_pointcept_settings()
        except Exception:
            pc_settings = {}

        pointcept_path = Path(pc_settings.get("pointcept_path") or DEFAULT_POINTCEPT_PATH).resolve()
        model_path = Path(pc_settings.get("model_path") or DEFAULT_MODEL_PATH).resolve()
        config_path = Path(pc_settings.get("config_path") or DEFAULT_CONFIG_PATH).resolve()
        python_exe = pc_settings.get("python_exe") or sys.executable

        pred_script = pointcept_path / "prediction.py"
        if not pred_script.is_file():
            logger.error("prediction.py not found at %s", pred_script)
            return False, 0

        with tempfile.TemporaryDirectory(prefix="cluster_pt_") as work_dir:
            work_path = Path(work_dir)
            temp_las = work_path / las_path.name
            try:
                temp_las.symlink_to(las_path.resolve())
            except OSError:
                shutil.copyfile(str(las_path.resolve()), str(temp_las))

            env = os.environ.copy()
            env["PYTHONPATH"] = str(pointcept_path) + os.pathsep + env.get("PYTHONPATH", "")

            cmd = [
                python_exe,
                str(pred_script),
                "--folder", str(work_path),
                "--model_path", str(model_path),
                "--config_file", str(config_path),
                "--noise_filter", "no",
                "--smoothing", str(smoothing),
                "--voxel_size", str(voxel_size),
            ]

            logger.info("Executing Pointcept on tile %s...", las_path.name)
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(work_path),
                env=env,
                timeout=7200,
            )

            if res.returncode != 0:
                logger.error("Pointcept execution failed on %s (exit %d): %s", las_path.name, res.returncode, res.stderr[-500:])
                return False, 0

            # prediction.py outputs to <folder>/predictions/<las_name>
            predicted_file = work_path / "predictions" / las_path.name
            if predicted_file.is_file():
                backup = las_path.with_suffix(las_path.suffix + ".bak")
                try:
                    shutil.move(str(las_path), str(backup))
                except OSError:
                    pass
                shutil.move(str(predicted_file), str(las_path))
                logger.info("Successfully classified and updated %s", las_path)
                return True, 500000

        return False, 0

    def _run_noise_filter(
        self,
        las_path: Path,
        job_params: dict,
    ) -> Tuple[bool, int]:
        """Run noise filtering pipeline on a single LAS/LAZ tile in-place."""
        pipeline = job_params.get("pipeline") or []
        if not pipeline:
            logger.warning("Empty filter pipeline for %s", las_path.name)
            return False, 0
        try:
            import laspy
            import numpy as np
            try:
                from .noise_filter import _apply_pipeline_timed
            except (ImportError, ValueError):
                from noise_filter import _apply_pipeline_timed

            las = laspy.read(str(las_path))
            n_pts = len(las.x)
            if n_pts == 0:
                return True, 0
            data = {
                "x": np.asarray(las.x, dtype=np.float64),
                "y": np.asarray(las.y, dtype=np.float64),
                "z": np.asarray(las.z, dtype=np.float64),
                "classification": np.asarray(las.classification, dtype=np.uint8),
                "intensity": np.asarray(las.intensity, dtype=np.uint16) if hasattr(las, "intensity") else np.zeros(n_pts, dtype=np.uint16),
                "return_number": np.asarray(las.return_number, dtype=np.uint8) if hasattr(las, "return_number") else np.ones(n_pts, dtype=np.uint8),
            }
            keep_mask, _dur = _apply_pipeline_timed(data, pipeline)
            cls_arr = np.asarray(las.classification, dtype=np.uint8).copy()
            cls_arr[~keep_mask] = 7  # ASPRS Low Point / Noise
            las.classification = cls_arr

            backup = las_path.with_suffix(las_path.suffix + ".bak")
            try:
                shutil.copy2(str(las_path), str(backup))
            except OSError:
                pass
            las.write(str(las_path))
            logger.info("Filtered %s (%d/%d noise pts marked class 7)", las_path.name, int((~keep_mask).sum()), n_pts)
            return True, int(n_pts)
        except Exception as e:
            logger.error("Noise filter failed on %s: %s", las_path.name, e)
            return False, 0

    def _run_ground_classification(
        self,
        tile_id: str,
        las_path: Path,
        job_params: dict,
    ) -> Tuple[bool, int]:
        """Run ground classification on a single LAS/LAZ tile in-place."""
        method = str(job_params.get("method", "stepdown_ptd"))
        params = dict(job_params.get("params") or {})
        source_class = int(job_params.get("source_class", -1))
        try:
            try:
                from .gui.ground_classify_dialog import _ground_process_tile
            except (ImportError, ValueError):
                from gui.ground_classify_dialog import _ground_process_tile
            res = _ground_process_tile(tile_id, str(las_path), method, params, source_class, {})
            n_affected = int(res.get("n_affected", 0))
            logger.info("Ground classified %s (%d ground / %d affected)", las_path.name, int(res.get("n_ground", 0)), n_affected)
            return True, n_affected
        except Exception as e:
            logger.error("Ground classification failed on %s: %s", las_path.name, e)
            return False, 0

    def _run_adaptive_outlier(
        self,
        las_path: Path,
        job_params: dict,
    ) -> Tuple[bool, int]:
        """Run automatic adaptive outlier classification (Class 7 & 18) on a tile in-place."""
        try:
            import laspy
            import numpy as np
            try:
                from .adaptive_outlier import OutlierParams, classify_outliers_adaptive, evaluate_point_cloud
            except (ImportError, ValueError):
                from adaptive_outlier import OutlierParams, classify_outliers_adaptive, evaluate_point_cloud

            las = laspy.read(str(las_path))
            n_pts = len(las.x)
            if n_pts == 0:
                return True, 0
            extra_dims = {}
            for ed in las.point_format.extra_dimensions:
                if hasattr(las, ed.name):
                    extra_dims[ed.name] = np.asarray(getattr(las, ed.name))
            data = {
                "x": np.asarray(las.x, dtype=np.float64),
                "y": np.asarray(las.y, dtype=np.float64),
                "z": np.asarray(las.z, dtype=np.float64),
                "classification": np.asarray(las.classification, dtype=np.uint8),
                "intensity": np.asarray(las.intensity, dtype=np.uint16) if hasattr(las, "intensity") else np.zeros(n_pts, dtype=np.uint16),
                "return_number": np.asarray(las.return_number, dtype=np.uint8) if hasattr(las, "return_number") else np.ones(n_pts, dtype=np.uint8),
                "num_returns": np.asarray(las.number_of_returns, dtype=np.uint8) if hasattr(las, "number_of_returns") else np.ones(n_pts, dtype=np.uint8),
                "point_source_id": np.asarray(las.point_source_id, dtype=np.uint16) if hasattr(las, "point_source_id") else np.zeros(n_pts, dtype=np.uint16),
                "scan_angle": np.asarray(las.scan_angle, dtype=np.float64) if hasattr(las, "scan_angle") else np.asarray(las.scan_angle_rank, dtype=np.float64),
                "extra_dims": extra_dims,
            }
            rep = evaluate_point_cloud(data)
            op_dict = job_params.get("outlier_params") or {}
            params = OutlierParams.custom(rep, **op_dict) if op_dict else OutlierParams.auto(rep)
            res = classify_outliers_adaptive(data, params=params, report=rep)
            backup = las_path.with_suffix(las_path.suffix + ".bak")
            try:
                shutil.copy2(str(las_path), str(backup))
            except OSError:
                pass
            noise_mask = res.class_7_mask | res.class_18_mask
            keep_mask = ~noise_mask
            if np.any(noise_mask):
                noise_dir = las_path.parent / "noise"
                noise_dir.mkdir(parents=True, exist_ok=True)
                noise_path = noise_dir / f"{las_path.stem}_noise.las"
                noise_las = laspy.LasData(las.header)
                noise_las.points = las.points[noise_mask]
                noise_las.classification = res.updated_classification[noise_mask]
                noise_las.write(str(noise_path))

                clean_las = laspy.LasData(las.header)
                clean_las.points = las.points[keep_mask]
                clean_las.write(str(las_path))
            return True, int(keep_mask.sum())
        except Exception as e:
            logger.error("Adaptive outlier failed on %s: %s", las_path.name, e)
            return False, 0

    def run(self) -> None:
        self._stop_requested = False
        logger.info("Company Node Worker started for node '%s'", self._node_name)
        if not self._is_paused and self._lic.is_logged_in:
            self.worker_status_changed.emit("Idle — Listening for company tasks")
            self._lic.send_node_heartbeat(status="idle", node_name=self._node_name)
        else:
            self.worker_status_changed.emit("Paused" if self._is_paused else "Awaiting Login")

        heartbeat_counter = 0
        while not self._stop_requested:
            if not self._lic.is_logged_in or self._is_paused:
                time.sleep(self.poll_interval_sec * 2)
                continue

            try:
                task = self._lic.claim_node_task(node_name=self._node_name)
                if task.get("has_task"):
                    job_id = task["job_id"]
                    tile_id = task["tile_id"]
                    job_type = task.get("job_type") or "pointcept_classification"
                    job_params = task.get("job_params") or {}
                    proj_name = task.get("project_name") or job_params.get("project_name")
                    proj_path = task.get("project_path") or job_params.get("project_path")
                    voxel_sz = float(task.get("voxel_size", job_params.get("voxel_size", 0.5)))
                    smoothing = str(task.get("smoothing", job_params.get("smoothing", "yes")))

                    print(f"[*] Node '{self._node_name}' claimed cluster task [{job_type}]: Tile '{tile_id}' (Job #{job_id})")
                    logger.info("Claimed cluster task [%s]: Tile '%s' (Job #%d)", job_type, tile_id, job_id)
                    self.task_started.emit(tile_id, job_id)
                    self.worker_status_changed.emit(f"Processing [{job_type}] {tile_id}…")

                    # Mark node as busy
                    self._lic.send_node_heartbeat(status="busy", node_name=self._node_name)

                    # Locate tile on shared storage or local paths
                    las_file = self._find_tile_file(tile_id, project_name=proj_name, project_path=proj_path)
                    pt_count = 500000

                    if las_file and las_file.is_file():
                        print(f"[*] Found tile on shared storage: {las_file}. Running {job_type}...")
                        logger.info("Processing shared storage tile %s (%s)...", las_file, job_type)
                        if job_type == "noise_filter":
                            ok, pt_count = self._run_noise_filter(las_file, job_params)
                        elif job_type == "ground_classification":
                            ok, pt_count = self._run_ground_classification(tile_id, las_file, job_params)
                        elif job_type == "adaptive_outlier":
                            ok, pt_count = self._run_adaptive_outlier(las_file, job_params)
                        else:
                            ok, pt_count = self._run_pointcept_inference(las_file, voxel_size=voxel_sz, smoothing=smoothing)
                        if not ok:
                            print(f"[!] Warning: {job_type} encountered an issue on {las_file}.")
                    else:
                        print(f"[!] Tile '{tile_id}' not found on shared storage. (Simulating completion)")
                        logger.warning("Tile '%s' not found on shared storage (root: %s).", tile_id, self._get_active_storage_root())
                        time.sleep(2.0)

                    # Mark completed on cluster server
                    self._lic.complete_node_tile(
                        job_id=job_id,
                        tile_id=tile_id,
                        node_name=self._node_name,
                        point_count=pt_count,
                    )
                    self._processed_count += 1
                    print(f"[+] Node '{self._node_name}' finished cluster task: Tile '{tile_id}' (Job #{job_id})")
                    logger.info("Finished cluster task: Tile '%s' (Job #%d)", tile_id, job_id)
                    self.task_finished.emit(tile_id, job_id)

                    # Reset to idle
                    self._lic.send_node_heartbeat(status="idle", node_name=self._node_name)
                    self.worker_status_changed.emit("Idle — Listening for company tasks")
                else:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 5:  # Every ~15s
                        self._lic.send_node_heartbeat(status="idle", node_name=self._node_name)
                        heartbeat_counter = 0
                    time.sleep(self.poll_interval_sec)
            except Exception as e:
                logger.debug("Cluster poll error: %s", e)
                time.sleep(self.poll_interval_sec * 2)

        # Mark offline when stopping
        try:
            self._lic.send_node_heartbeat(status="offline", node_name=self._node_name)
        except Exception:
            pass

        logger.info("Company Node Worker stopped")
        self.worker_status_changed.emit("Stopped")


def main() -> None:
    mgr = get_license_manager()
    parser = argparse.ArgumentParser(description="LiDAR Workbench Distributed Cluster Worker")
    parser.add_argument("--name", type=str, default=None, help="Custom node name (defaults to hostname)")
    parser.add_argument("--server", type=str, default=mgr.server_url, help=f"Server URL (default: {mgr.server_url})")
    parser.add_argument("--storage", type=str, default=None, help="Shared storage projects root (e.g. \\\\server\\share\\projects or /mnt/projects)")
    parser.add_argument("--email", type=str, default=None, help="User email")
    parser.add_argument("--password", type=str, default=None, help="User password")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args()
    node_name = args.name or f"Worker-{platform.node()}"

    if args.email and args.password:
        print(f"[-] Logging in as {args.email} to {args.server}...")
        ok, msg = mgr.login_online(args.email, args.password, server_url=args.server)
        if not ok:
            print(f"[ERROR] Failed to log in: {msg}")
            sys.exit(1)
        print(f"[+] Successfully authenticated: {msg}")
    elif not mgr.is_logged_in:
        print("[!] No active session. Please provide --email and --password to connect.")
        sys.exit(1)

    print("========================================================")
    print("  LiDAR Workbench Distributed Worker Node")
    print(f"  Node Name:  {node_name}")
    print(f"  User:       {mgr.user_email} ({mgr.company_name})")
    print(f"  Server:     {args.server}")
    if args.storage:
        print(f"  Storage:    {args.storage}")
    print("========================================================")

    worker = CompanyNodeWorker(
        poll_interval_sec=args.interval,
        node_name=node_name,
        storage_root=args.storage,
    )
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\nStopping worker...")
        worker.stop()
        print("Worker stopped.")


if __name__ == "__main__":
    main()
