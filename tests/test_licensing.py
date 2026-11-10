import tempfile
from pathlib import Path
from lidar_workbench.licensing import LicenseManager, get_machine_hwid


def test_hwid_generation():
    hwid = get_machine_hwid()
    assert isinstance(hwid, str)
    assert len(hwid) == 32


def test_license_manager_project_quota():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_license.json"
        lm = LicenseManager(cache_file=cache_path)

        # 1. Default is Free (1 project limit)
        assert lm.plan == "free"
        assert lm.max_projects == 1

        allowed, current, limit = lm.can_create_project()
        assert allowed is True
        assert current == 0
        assert limit == 1

        # 2. Register first project
        p1 = Path(tmpdir) / "project_01"
        p1.mkdir()
        lm.register_created_project(str(p1))

        # 3. Quota should now be reached
        allowed, current, limit = lm.can_create_project()
        assert allowed is False
        assert current == 1
        assert limit == 1

        # 4. Simulate activating Starter (5 projects)
        lm._data["plan"] = "starter_5"
        lm._data["max_projects"] = 5
        lm._save()

        allowed, current, limit = lm.can_create_project()
        assert allowed is True
        assert current == 1
        assert limit == 5
