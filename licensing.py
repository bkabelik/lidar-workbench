"""
LiDAR Workbench — Licensing & Project Quota Manager.

Manages local machine hardware fingerprinting, offline token caching,
and quota enforcement (1 project for Free tier, 5 for Starter, 10 for Pro, etc.).
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import hashlib
import logging
import platform
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import urllib.request
import urllib.error
import urllib.parse

logger = logging.getLogger("lidar_workbench.licensing")

# Global license cache in user's home directory so it persists across updates & reinstalls
if sys.platform == "win32":
    _APP_DATA = Path(os.environ.get("APPDATA", Path.home())) / "LiDAR_Workbench"
else:
    _APP_DATA = Path.home() / ".lidar_workbench"

_APP_DATA.mkdir(parents=True, exist_ok=True)
_LICENSE_CACHE_FILE = _APP_DATA / "license.json"

DEFAULT_SERVER_URL = "https://lidar.kabelik.at"


def get_machine_hwid() -> str:
    """Generate a stable, unique hardware fingerprint for this machine."""
    raw_id = ""
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                raw_id = str(guid).strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                if os.path.exists(p):
                    with open(p, "r") as f:
                        raw_id = f.read().strip()
                        if raw_id:
                            break
            except Exception:
                pass

    if not raw_id:
        # Fallback to node name + MAC address
        raw_id = f"{platform.node()}-{uuid.getnode()}"

    return hashlib.sha256(f"lw-hwid-salt-{raw_id}".encode("utf-8")).hexdigest()[:32]


def get_local_gpu_name() -> str:
    """Detect available NVIDIA RTX GPU or return fallback."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"{name} ({mem_gb:.0f}GB)"
    except Exception:
        pass
    return "NVIDIA RTX GPU"


class LicenseManager:
    """Manages machine activation state and project quotas."""

    def __init__(self, cache_file: Optional[Path] = None):
        self._cache_file = cache_file or _LICENSE_CACHE_FILE
        self._hwid = get_machine_hwid()
        self._data: Dict[str, Any] = self._load()

    @property
    def hwid(self) -> str:
        return self._hwid

    @property
    def plan(self) -> str:
        if sys.platform.startswith("linux"):
            return "open_source_linux"
        return self._data.get("plan", "free")

    @property
    def max_projects(self) -> int:
        """Maximum projects allowed: 1 for Free, 5 for Starter, 10 for Pro, -1 for Unlimited."""
        return self._data.get("max_projects", 1)

    @property
    def license_key(self) -> Optional[str]:
        return self._data.get("license_key")

    @property
    def is_free_tier(self) -> bool:
        return self.plan == "free"

    def _load(self) -> Dict[str, Any]:
        if self._cache_file.is_file():
            try:
                with open(self._cache_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning("Failed to read license cache: %s", e)
        return {
            "license_key": None,
            "plan": "free",
            "max_projects": 1,
            "signed_token": None,
            "created_projects": [],
        }

    def _save(self) -> None:
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save license cache: %s", e)

    def get_registered_projects(self) -> list[str]:
        """List of project paths/IDs created on this installation."""
        return list(self._data.get("created_projects", []))

    def register_created_project(self, project_path: str) -> None:
        """Record that a new project was created on this machine."""
        proj_str = str(Path(project_path).resolve())
        projects = self._data.setdefault("created_projects", [])
        if proj_str not in projects:
            projects.append(proj_str)
            self._save()

    @property
    def user_email(self) -> Optional[str]:
        return self._data.get("user_email")

    @property
    def company_name(self) -> str:
        return self._data.get("company_name", "My Team")

    @property
    def company_id(self) -> Optional[int]:
        return self._data.get("company_id")

    @property
    def access_token(self) -> Optional[str]:
        return self._data.get("access_token")

    @property
    def server_url(self) -> str:
        return self._data.get("server_url") or DEFAULT_SERVER_URL

    @server_url.setter
    def server_url(self, value: str) -> None:
        self._data["server_url"] = value.strip().rstrip("/")
        self._save()

    @property
    def is_logged_in(self) -> bool:
        return bool(self.user_email and self.access_token)

    def logout(self) -> None:
        """Sign out from user account and return to standard Free tier."""
        self._data["user_email"] = None
        self._data["company_name"] = None
        self._data["company_id"] = None
        self._data["access_token"] = None
        self._data["license_key"] = None
        self._data["plan"] = "free"
        self._data["max_projects"] = 1
        self._data["signed_token"] = None
        self._save()

    def login_online(self, email: str, password: str, server_url: Optional[str] = None) -> Tuple[bool, str]:
        """
        Authenticate with company user account and automatically bind machine seat.
        """
        target_url = (server_url or self.server_url).rstrip("/")
        clean_url = target_url + "/api/v1/auth/login-machine"
        payload = {
            "email": email.strip(),
            "password": password,
            "hwid": self._hwid,
            "machine_name": platform.node(),
            "os_name": f"{platform.system()} {platform.release()}",
        }

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                clean_url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                if res.get("success"):
                    self._data["server_url"] = target_url
                    self._data["user_email"] = res.get("user_email")
                    self._data["company_name"] = res.get("company_name", "Company Team")
                    self._data["company_id"] = res.get("company_id")
                    self._data["access_token"] = res.get("access_token")
                    self._data["plan"] = res.get("plan", "free")
                    self._data["max_projects"] = res.get("max_projects", 1)
                    self._data["signed_token"] = res.get("signed_token")
                    self._save()
                    return True, res.get("message", f"Signed in as {res.get('user_email')}")
                return False, res.get("message", "Sign in failed")
        except urllib.error.HTTPError as e:
            try:
                err_json = json.loads(e.read().decode("utf-8"))
                msg = err_json.get("detail", str(e))
            except Exception:
                msg = str(e)
            return False, f"Login failed: {msg}"
        except Exception as e:
            return False, f"Server connection failed: {e}"

    def send_node_heartbeat(
        self,
        status: str = "idle",
        node_name: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send machine heartbeat to server so it shows up in company cluster."""
        if not self.is_logged_in:
            return {"success": False, "message": "Not logged in"}

        target_url = (server_url or self.server_url).rstrip("/")
        url = target_url + "/api/v1/nodes/heartbeat"
        payload = {
            "user_email": self.user_email,
            "hwid": self._hwid,
            "node_name": node_name or platform.node(),
            "gpu_name": get_local_gpu_name(),
            "os_name": f"{platform.system()} {platform.release()}",
            "status": status,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fetch_company_nodes(
        self,
        server_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch list of all active online nodes belonging to user's company."""
        if not self.is_logged_in:
            return []

        target_url = (server_url or self.server_url).rstrip("/")
        url = f"{target_url}/api/v1/nodes/company?user_email={urllib.parse.quote(self.user_email or '')}"
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("Failed to fetch company nodes: %s", e)
            return []

    def dispatch_job(
        self,
        tile_ids: List[str],
        target_node_ids: List[int],
        project_id: Optional[int] = None,
        project_name: Optional[str] = None,
        project_path: Optional[str] = None,
        voxel_size: float = 0.5,
        smoothing: str = "yes",
        job_type: str = "pointcept_classification",
        job_params: Optional[Dict[str, Any]] = None,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch a batch of tiles across the selected company nodes."""
        if not self.is_logged_in:
            return {"success": False, "error": "User not authenticated"}

        target_url = (server_url or self.server_url).rstrip("/")
        url = target_url + "/api/v1/nodes/dispatch-job"
        payload = {
            "user_email": self.user_email,
            "project_id": project_id,
            "project_name": project_name,
            "project_path": project_path,
            "voxel_size": voxel_size,
            "smoothing": smoothing,
            "tile_ids": tile_ids,
            "target_node_ids": target_node_ids,
            "job_type": job_type,
            "job_params": job_params or {},
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_job_status(
        self,
        job_id: int,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Poll distributed job progress."""
        target_url = (server_url or self.server_url).rstrip("/")
        url = f"{target_url}/api/v1/compute/jobs/{job_id}/status"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def claim_node_task(
        self,
        node_name: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Claim next pending tile assigned to this node."""
        name = node_name or platform.node()
        target_url = (server_url or self.server_url).rstrip("/")
        url = f"{target_url}/api/v1/nodes/claim-task/{urllib.parse.quote(name)}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"has_task": False, "message": str(e)}

    def complete_node_tile(
        self,
        job_id: int,
        tile_id: str,
        node_name: Optional[str] = None,
        point_count: int = 500000,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Report completion of a tile."""
        name = node_name or platform.node()
        target_url = (server_url or self.server_url).rstrip("/")
        url = f"{target_url}/api/v1/compute/workers/complete-tile"
        payload = {
            "job_id": job_id,
            "tile_id": tile_id,
            "worker_id": name,
            "status": "completed",
            "point_count": point_count,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sync_project(
        self,
        project_name: str,
        tiles_data: List[Dict[str, Any]],
        activity_action: Optional[str] = None,
        activity_details: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sync local tile states, processing steps, and QC reviews with company cloud."""
        if not self.is_logged_in:
            return {"success": False, "error": "Not signed in"}

        target_url = (server_url or self.server_url).rstrip("/")
        url = target_url + "/api/v1/projects/sync"
        payload = {
            "project_name": project_name,
            "user_email": self.user_email,
            "node_name": platform.node(),
            "tiles": tiles_data,
            "activity_action": activity_action,
            "activity_details": activity_details,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def lock_tile(
        self,
        project_name: str,
        tile_id: str,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Acquire collaborative editing lock on a tile."""
        if not self.is_logged_in:
            return {"acquired": True, "locked": False}

        target_url = (server_url or self.server_url).rstrip("/")
        url = f"{target_url}/api/v1/projects/{urllib.parse.quote(project_name)}/lock-tile"
        payload = {
            "tile_id": tile_id,
            "user_email": self.user_email,
            "node_name": platform.node(),
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"acquired": True, "error": str(e)}

    def unlock_tile(
        self,
        project_name: str,
        tile_id: str,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Release collaborative editing lock on a tile."""
        if not self.is_logged_in:
            return {"unlocked": True}

        target_url = (server_url or self.server_url).rstrip("/")
        url = f"{target_url}/api/v1/projects/{urllib.parse.quote(project_name)}/unlock-tile"
        payload = {
            "tile_id": tile_id,
            "user_email": self.user_email,
            "node_name": platform.node(),
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"unlocked": True, "error": str(e)}

    def update_qc_status_online(
        self,
        project_name: str,
        tile_ids: List[str],
        qc_status: Optional[str],
        qc_comment: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Broadcast QC review changes to company cloud."""
        if not self.is_logged_in:
            return {"success": False}

        target_url = (server_url or self.server_url).rstrip("/")
        url = f"{target_url}/api/v1/projects/{urllib.parse.quote(project_name)}/qc-status"
        payload = {
            "tile_ids": tile_ids,
            "user_email": self.user_email,
            "qc_status": qc_status,
            "qc_comment": qc_comment,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def can_create_project(self) -> Tuple[bool, int, int]:
        """
        Check if the user is allowed to create another project under their plan.

        Returns:
            (allowed: bool, current_count: int, max_allowed: int)
        """
        # Clean up missing project paths if user deleted them
        existing = [p for p in self.get_registered_projects() if Path(p).is_dir()]
        if len(existing) != len(self._data.get("created_projects", [])):
            self._data["created_projects"] = existing
            self._save()

        count = len(existing)

        # Linux open-source build is 100% free and unlimited
        if sys.platform.startswith("linux"):
            return True, count, -1

        limit = self.max_projects
        if limit == -1:  # Unlimited
            return True, count, limit

        return (count < limit), count, limit

    def activate_online(self, license_key: str, server_url: Optional[str] = None) -> Tuple[bool, str]:
        """
        Contact the server to bind this machine HWID and activate the license.
        """
        clean_key = license_key.strip().upper()
        target_url = (server_url or self.server_url).rstrip("/")
        clean_url = target_url + "/api/v1/licenses/activate"

        payload = {
            "license_key": clean_key,
            "hwid": self._hwid,
            "machine_name": platform.node(),
            "os_name": f"{platform.system()} {platform.release()}",
        }

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                clean_url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_body = json.loads(resp.read().decode("utf-8"))
                if res_body.get("success"):
                    self._data["server_url"] = target_url
                    self._data["license_key"] = clean_key
                    self._data["plan"] = res_body.get("plan", "pro")
                    self._data["max_projects"] = res_body.get("max_projects", 5)
                    self._data["signed_token"] = res_body.get("signed_token")
                    self._save()
                    return True, res_body.get("message", "License activated!")
                return False, res_body.get("message", "Activation failed")
        except urllib.error.HTTPError as e:
            try:
                err_json = json.loads(e.read().decode("utf-8"))
                msg = err_json.get("detail", str(e))
            except Exception:
                msg = str(e)
            return False, f"Server error: {msg}"
        except Exception as e:
            return False, f"Network connection failed: {e}"


# Global singleton
_manager_instance: Optional[LicenseManager] = None


def get_license_manager() -> LicenseManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LicenseManager()
    return _manager_instance
