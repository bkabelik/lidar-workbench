import sys
from pathlib import Path

# Make ``lidar_workbench`` importable when running pytest from this directory
# or from the repository root (even if folder is named lidar-workbench with hyphen).
_LIDAR_WORKBENCH = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_LIDAR_WORKBENCH) not in sys.path:
    sys.path.insert(0, str(_LIDAR_WORKBENCH))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if "lidar_workbench" not in sys.modules:
    import types as _types
    _mod = _types.ModuleType("lidar_workbench")
    _mod.__path__ = [str(_LIDAR_WORKBENCH)]
    _mod.__file__ = str(_LIDAR_WORKBENCH / "__init__.py")
    _mod.__package__ = "lidar_workbench"
    sys.modules["lidar_workbench"] = _mod
