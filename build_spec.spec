# -*- mode: python ; coding: utf-8 -*-
import os
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

try:
    repo_dir = Path(SPECPATH).resolve()
except NameError:
    repo_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path(".").resolve()

# Root workbench modules
root_modules = [
    "alignment",
    "centerline_wsm",
    "config",
    "crs",
    "database",
    "dtm_generator",
    "export_manager",
    "ground",
    "import_wizard",
    "manual_edit",
    "noise_filter",
    "pointcept_worker",
    "point_qc",
    "project_manager",
    "refraction",
    "strip_adjustment",
    "tile_manager",
    "licensing",
    "node_worker",
    "qgis_parser",
    "adaptive_outlier",
    "updater",
]

hidden_imports = [
    "scipy.spatial.transform._rotation_groups",
    "torch",
    "torchvision",
    "torch_scatter",
    "torch_cluster",
    "torch_geometric",
    "spconv",
    "cumm",
    "pccm",
    "timm",
    "peft",
    "open3d",
    "laspy",
    "lazrs",
    "rasterio",
    "requests",
    "pyproj",
    "rasterio.sample",
    "rasterio._features",
    "rasterio.control",
    "rasterio.crs",
    "rasterio.vrt",
    "pyproj",
    "shapely",
    "pyqtgraph",
    "shapefile",
    "Pointcept",
    "Pointcept.prediction",
    "Pointcept.noisefilter",
    "Pointcept.postclassification",
]

for m in root_modules:
    hidden_imports.append(m)
    hidden_imports.append(f"lidar_workbench.{m}")

hidden_imports += ["matplotlib", "matplotlib.pyplot", "matplotlib.cm"]
hidden_imports += collect_submodules("torch")
hidden_imports += collect_submodules("torchvision")
hidden_imports += collect_submodules("pointcept")
hidden_imports += collect_submodules("rasterio")
hidden_imports += collect_submodules("pyproj")
hidden_imports += collect_submodules("shapely")
hidden_imports += collect_submodules("laspy")

import torch_scatter
import torch_cluster
import torch_geometric
import rasterio
import pyproj

# Collect datas
datas = [
    (str(repo_dir / "gui"), "gui"),
    (str(repo_dir / "Pointcept"), "Pointcept"),
    (str(repo_dir / "Pointcept" / "configs"), "configs"),
    (str(repo_dir / "models"), "models"),
    (str(Path(torch_scatter.__file__).parent), "torch_scatter"),
    (str(Path(torch_cluster.__file__).parent), "torch_cluster"),
    (str(Path(torch_geometric.__file__).parent), "torch_geometric"),
    (str(Path(rasterio.__file__).parent), "rasterio"),
    (str(Path(pyproj.__file__).parent), "pyproj"),
    (str(repo_dir / "*.py"), "."),
]

# Collect package data
datas += collect_data_files("open3d")
datas += collect_data_files("rasterio")
datas += collect_data_files("pyproj")
datas += collect_data_files("torch")

# Collect dynamic libraries
binaries = collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("open3d")
binaries += collect_dynamic_libs("rasterio")
binaries += collect_dynamic_libs("pyproj")
binaries += collect_dynamic_libs("torch_scatter")
binaries += collect_dynamic_libs("torch_cluster")
binaries += collect_dynamic_libs("spconv")

a = Analysis(
    [str(repo_dir / "main.py")],
    pathex=[str(repo_dir), str(repo_dir / "Pointcept"), str(repo_dir.parent)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "jupyter", "tornado"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiDAR-Workbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(repo_dir / "gui" / "assets" / "app_icon.ico") if (repo_dir / "gui" / "assets" / "app_icon.ico").is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LiDAR-Workbench",
)
