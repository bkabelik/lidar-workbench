"""
Unit tests for Company Nodes, Peer-to-Peer Cluster, and Distributed Compute.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QApplication

from lidar_workbench.licensing import LicenseManager
from lidar_workbench.gui.company_nodes_dialog import CompanyNodesDialog
from lidar_workbench.node_worker import CompanyNodeWorker


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_license_manager_cluster_methods(tmp_path):
    cache = tmp_path / "lic.json"
    mgr = LicenseManager(cache_file=cache)
    mgr._data["user_email"] = "analyst@geoteam.com"
    mgr._data["access_token"] = "fake-jwt-token"
    mgr._data["company_name"] = "GeoTeam Surveying"
    mgr._data["company_id"] = 42

    assert mgr.is_logged_in is True
    assert mgr.company_name == "GeoTeam Surveying"
    assert mgr.company_id == 42


def test_company_nodes_dialog(qapp, tmp_path):
    cache = tmp_path / "lic.json"
    mgr = LicenseManager(cache_file=cache)
    mgr._data["user_email"] = "lead@geoteam.com"
    mgr._data["access_token"] = "fake-jwt-token"
    mgr._data["company_name"] = "GeoTeam Surveying"

    with patch("lidar_workbench.gui.company_nodes_dialog.get_license_manager", return_value=mgr):
        with patch.object(mgr, "send_node_heartbeat", return_value={"success": True}):
            with patch.object(mgr, "fetch_company_nodes", return_value=[
                {
                    "id": 1,
                    "node_name": "Workstation-Alpha",
                    "user_email": "lead@geoteam.com",
                    "gpu_name": "NVIDIA GeForce RTX 4090 (24GB)",
                    "status": "idle",
                    "is_online": True,
                    "last_seen": "2026-11-10T14:00:00",
                },
                {
                    "id": 2,
                    "node_name": "Laptop-Beta",
                    "user_email": "field@geoteam.com",
                    "gpu_name": "NVIDIA GeForce RTX 3070 (8GB)",
                    "status": "busy",
                    "is_online": True,
                    "last_seen": "2026-11-10T14:00:02",
                },
            ]):
                dlg = CompanyNodesDialog()
                assert dlg._table.rowCount() == 2
                assert "Workstation-Alpha" in dlg._table.item(0, 1).text()
                assert "Laptop-Beta" in dlg._table.item(1, 1).text()
                dlg.close()


def test_company_node_worker_lifecycle():
    worker = CompanyNodeWorker(poll_interval_sec=0.1)
    assert worker._stop_requested is False
    worker.stop()
    assert worker._stop_requested is True
