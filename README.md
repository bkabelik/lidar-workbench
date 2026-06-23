# LiDAR Workbench

**Interactive airborne LiDAR point cloud analysis, classification, and raster export tool.**

Built with PySide6, Open3D, laspy, NumPy/SciPy, and [Pointcept](https://github.com/bkabelik/PointceptALS.git).

---

## Features

### Point Cloud Processing
- **Drag-and-drop import** of `.las`/`.laz` flight strips with automatic spatial tiling
- **Interactive noise filtering** — SOR, ROR, and DBSCAN with real-time 3D preview
- **Pointcept integration** — deep-learning classification via Point Transformer V3
- **Manual editing** — profile-based inspection, line/rectangle/brush selections, class reassignment
- **Full undo/redo** stack for all classification edits

### Multi-View Workspace
- **3D point cloud** — class-coloured rendering with orbit/pan/zoom controls
- **DTM top-down** — hillshade DTM with interactive profile-line drawing
- **2D profile side** — corridor cross-section with selection tools and DTM reference overlay
- **3D profile slice** — perspective view of the profile corridor

### DTM / DSM Export (NEW)
- **DTM** — Delaunay-triangulation (TIN) interpolation of ground points (class 2) to a regular grid
- **DSM** — highest-point-per-cell (max-Z) from user-selected ASPRS classes
- **Hillshade** — Horn's method illumination raster exported as GeoTIFF (.tif) alongside every DTM/DSM
- **Seamless tiling** — all tiles share a master grid with snapped origin; adjacent `.asc` files align perfectly with no gaps
- **ESRI ASCII Grid** (`.asc`) output — readable by QGIS, ArcGIS, Global Mapper, and GDAL
- **Configurable resolution** — from 0.05 m to 100 m, with common presets (0.25, 0.5, 1.0, 2.0, 5.0 m)
- **Merged or tiled** — single large raster or one `.asc` per tile

---

## Installation

### Prerequisites
- **Python** ≥ 3.10
- **pip** (or conda)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/<your-org>/lidar-workbench.git
cd lidar-workbench

# Install core dependencies (PyPI)
pip install PySide6 numpy scipy laspy open3d

# For DTM/DSM export, scipy is already included — no extra deps needed.
# For GeoTIFF output (optional), install GDAL:
#   pip install gdal
#   or: conda install -c conda-forge gdal
```

### Pointcept (Optional — for AI Classification)

```bash
# Clone PointceptALS alongside the workbench
git clone https://github.com/bkabelik/PointceptALS.git
cd PointceptALS

# Follow its README to set up the conda environment:
conda env create -f environment.yml
conda activate pointcept

# You'll also need a trained model checkpoint (.pth) and config (.py).
# Place them in models/ and configs/ respectively, or point the
# workbench at them via Settings.
download [Pointcept Model](https://drive.google.com/file/d/15MlZ6cwed0jFsd7WKOdkDjQIQTiCy5nJ/view?usp=sharing)

```

---

## Quick Start

```bash
# Launch the application
python -m lidar_workbench.main

# Or open a specific project
python -m lidar_workbench.main /path/to/project
```

### Typical Workflow

1. **Create or open a project** — *File → New Project* (Ctrl+N) or *File → Open Project* (Ctrl+O)
2. **Import LAS/LAZ data** — *File → Import LAS/LAZ* (Ctrl+I) or drag-and-drop a folder onto the window
3. **Apply noise filter** — Select tiles in the tile list → *Tools → Noise Filter* → choose SOR/ROR/DBSCAN → *Apply*
4. **Classify with Pointcept** (optional) — Select filtered tiles → *Tools → Classify (Pointcept)* → configure → *Start*
5. **Manual editing** — Double-click a classified tile to open the multi-view:
   - Draw a profile line in the DTM view (right-click + drag)
   - Select misclassified points in the profile view (brush, line, rectangle)
   - Click a class button in the properties panel to reclassify
   - *Undo*/*Redo* as needed (Ctrl+Z / Ctrl+Y)
6. **Export raster** — *Tools → Export Raster (DTM / DSM)*:
   - Choose **DTM** (ground-only TIN interpolation) or **DSM** (max-Z from selected classes)
   - Set resolution (e.g. 0.5 m) and output directory
   - Toggle hillshade, merged vs tiled output
   - Click *OK*

### Export Output Files

```
project/dtm/
├── tile_0013_dtm.asc            # DTM per tile (ESRI ASCII Grid)
├── tile_0013_dtm_hillshade.tif  # Hillshade per tile (GeoTIFF)
├── tile_0014_dtm.asc
├── …
├── merged_dtm.asc               # or one merged raster
└── merged_dtm_hillshade.tif
```

DTM/DSM rasters are standard ESRI ASCII Grid format (.asc); hillshades are
GeoTIFF (.tif) with full georeferencing.  Drag them into QGIS or ArcGIS, or
process with GDAL.

---

## Architecture

```
lidar_workbench/
├── main.py                      # Entry point
├── config.py                    # Constants, ASPRS class colours, logging
├── database.py                  # SQLite ORM (tiles + edit_history)
├── project_manager.py           # Project lifecycle (create/open/save)
├── tile_manager.py              # LAS import, spatial tiling, I/O
├── import_wizard.py             # Guided import dialog (QWizard)
├── noise_filter.py              # SOR / ROR / DBSCAN filter algorithms
├── pointcept_worker.py          # Background subprocess inference runner
├── dtm_generator.py             # In-memory DTM interpolation (griddata)
├── export_manager.py            # DTM/DSM/Hillshade export engine (NEW)
├── manual_edit.py               # Profile extraction, selections, undo/redo
├── gui/
│   ├── main_window.py           # QMainWindow (menus, toolbar, 3-panel splitter)
│   ├── tile_list_widget.py      # Tile browser with status groups
│   ├── multi_view_widget.py     # 2×2 / 1×3 view layout manager
│   ├── view_3d.py               # Open3D 3D point cloud widget
│   ├── view_dtm.py              # 2D DTM view with profile drawing
│   ├── view_profile.py          # 2D profile side view with selection tools
│   ├── view_profile_3d.py       # 3D profile corridor view
│   ├── filter_dialog.py         # Noise filter parameter dialog
│   ├── classification_dialog.py # Pointcept configuration dialog
│   ├── export_dialog.py         # DTM/DSM export configuration dialog (NEW)
│   ├── properties_panel.py      # Point properties + quick-classify
│   └── settings_dialog.py       # Keyboard shortcut editor
└── Pointcept/                   # Bundled deep-learning library
    ├── prediction.py
    ├── postclassification.py
    └── pointcept/               # Core library
```

---

## Export Technical Details

### DTM (Digital Terrain Model)
- **Input**: Ground points only (ASPRS class 2)
- **Method**: Delaunay triangulation (TIN) with barycentric interpolation per raster cell
- **Fallback 1**: Inverse Distance Weighting (IDW, power=2) for cells outside the convex hull, using up to 12 nearest neighbours within 5× resolution radius
- **Fallback 2**: Nearest-neighbour as last resort for isolated empty cells
- **Equivalent to**: PDAL `writers.gdal` with `output_type=idw` + Delaunay pre-filtering

### DSM (Digital Surface Model)
- **Input**: User-selected ASPRS classes (default: ground, vegetation, buildings)
- **Method**: Per-cell maximum Z (binmode), i.e. the highest LiDAR return in each pixel
- **Fallback**: IDW fill for cells with zero points, using representative points from populated neighbouring cells
- **Equivalent to**: PDAL `writers.gdal` with `binmode=true`, `output_type=max`

### Hillshade
- **Algorithm**: Horn (1981) 8-neighbour central-difference slope estimator
- **Defaults**: Azimuth 315° (NW), altitude 45° — industry-standard values
- **Output**: GeoTIFF (.tif) with uint8 1–255 greyscale, 0 = nodata. Full
  georeferencing via ModelTiepointTag and ModelPixelScaleTag TIFF tags.

### Seamless Tiling
All tiles are rasterised against a single **master grid** whose origin is snapped to the resolution grid. This means:
- Adjacent `.asc` files share exactly the same cell boundaries
- No gaps, overlaps, or visible seams when loaded together in GIS software
- Each tile's `xllcorner`/`yllcorner` are exact multiples of `cellsize` from the master origin

---

## Project Structure

Each project is a directory containing:

```
my_project/
├── project.json           # Project metadata (name, paths, settings)
├── filter_settings.json   # Persisted filter parameters
├── tile_database.sqlite   # Tile metadata + edit history
├── tiles/                 # LAS tile files
│   ├── tile_0000.las
│   └── …
└── dtm/                   # Exported DTM/DSM rasters (ESRI ASCII Grid)
    ├── tile_0000_dtm.asc
    ├── tile_0000_dtm_hillshade.tif
    └── …
```

---

## Dependencies

| Package   | Minimum | Purpose                              |
|-----------|---------|--------------------------------------|
| Python    | 3.10    | Runtime                              |
| PySide6   | 6.5     | GUI framework (Qt 6)                 |
| numpy     | 1.24    | Numerical operations                 |
| scipy     | 1.10    | KDTree, Delaunay, spatial algorithms |
| laspy     | 2.5     | LAS/LAZ read/write                   |
| open3d    | 0.18    | 3D point cloud rendering             |

**Optional**: GDAL (for GeoTIFF output instead of ASCII Grid), matplotlib (height colour ramps).

For Pointcept classification, a separate conda environment with PyTorch and Point Transformer V3 dependencies is required — see the [PointceptALS](https://github.com/bkabelik/PointceptALS.git) README.

---

## Key Design Decisions

- **TIN interpolation for DTM** — preserves terrain discontinuities better than IDW-only methods; Delaunay triangulation is the same algorithm used by TerraScan/TerraModeler
- **Max-Z for DSM** — industry standard (PDAL, LASTools) for surface models from LiDAR; avoids the "averaging" artefacts that smooth out building edges and vegetation
- **ESRI ASCII Grid by default** — universal interchange format; every GIS package reads it; no GDAL dependency required
- **Seamless tiling via master grid** — avoids the common pitfall of per-tile floating-point origin drift causing 1-pixel gaps
- **QThread for all long operations** — import, filtering, Pointcept inference, and raster export never block the GUI
- **Command pattern for undo/redo** — every classification change is recorded as a reversible command object in SQLite
- **Lazy loading** — tiles are loaded on demand; the in-memory cache can be cleared when memory is tight
- **ASPRS-compliant** — classification codes follow LAS 1.4 standard (classes 0–18) with correct colour mapping

---

## Configuration

Edit `lidar_workbench/config.py` to adjust:

- Default tile size and overlap (`DEFAULT_TILE_SIZE_M`, `DEFAULT_TILE_OVERLAP_M`)
- ASPRS class colour map (`ASPRS_CLASS_COLORS`)
- Filter parameters (SOR neighbours, ROR radius)
- Point budget for 3D rendering (`MAX_POINTS_PER_VIEW`)

Keyboard shortcuts can be customised via *Tools → Settings*.

---

## License

*To be determined.*
