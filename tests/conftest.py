import sys
from pathlib import Path

# Make ``lidar_workbench`` importable when running pytest from this directory
# or from the repository root.
_LIDAR_WORKBENCH = Path(__file__).resolve().parents[1]
if str(_LIDAR_WORKBENCH) not in sys.path:
    sys.path.insert(0, str(_LIDAR_WORKBENCH))
