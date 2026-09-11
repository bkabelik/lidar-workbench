# <img src="gui/assets/logo.png" alt="Kabelik" height="60"> LiDAR Workbench

**Interactive airborne LiDAR point cloud analysis, classification, and raster export tool.**

Developed by **[Kabelik GmbH](https://kabelik.at)** — Remote Sensing, Geomatics &amp; IT Services  
Merbotogasse 58D, 2700 Wiener Neustadt, Austria · [office@kabelik.at](mailto:office@kabelik.at)

Built with PySide6, Open3D, laspy, NumPy/SciPy, and [Pointcept](https://github.com/bkabelik/PointceptALS.git).

---

## Features

### Point Cloud Processing & AI
- **LAS/LAZ Preview** — standalone file inspector with bounding box, subsampled, and full-resolution LODs before import
- **Drag-and-drop import** of `.las`/`.laz` flight strips with automatic spatial tiling
- **Interactive noise filtering** — SOR, ROR, DBSCAN, isolated-point, low-point, and surface-proximity filters with real-time 3D preview
- **Pointcept integration** — deep-learning classification via Point Transformer V3
- **Manual editing & QC** — ultra-responsive profile-based inspection with brush, rect brush, line-above/below, and rectangle selection tools; class reassignment with full undo/redo stack

### Multi-View Workspace & Synchronized Inspection
- **3D point cloud** — class/height/intensity/return/flightline coloured rendering with orbit/pan/zoom controls
- **DTM top-down** — hillshade DTM with interactive profile-line drawing and class/flightline point scatter
- **2D profile side view** — high-performance offscreen-rendered cross-section corridor (120+ FPS) with live HUD overlay, mouse-wheel shortcuts, and DTM reference overlay
- **Multi-Attribute LiDAR Colouring** — synchronized across 3D, DTM, and Profile views (By Class, By Height, By Intensity, By Return Number, By Flightline)
- **Dynamic corridor width & tool sizing** — adjust slice width and tool dimensions directly from the toolbar spinboxes or via hotkeys

### Water Surface Modeler (WSM)
- **Ergonomic vertical layout** — 2D Plan View map placed directly above the 1D cross-section elevation editor
- **2D Channel Intensity Raster** — binned 0.25 m / 0.5 m intensity maps highlighting NIR water absorption voids and bank boundaries to guarantee full channel coverage
- **Interactive cross-section editing** — color-coded normal profiles (cyan), locked anchors (green), and active profile (orange) with drag handles and click-to-jump navigation
- **Complete channel persistence** — "Save All Profiles" and "Load All Profiles" in JSON/GeoJSON format for seamless multi-session editing
- **Downstream monotonicity enforcement** — ensures physically realistic descending water surface elevations along the river reach
- **GeoTIFF Export** — exports high-resolution 3D raster water surface models

### Point QC & Flight Strip Alignment
- **2D Difference Map** — pairwise and max strip difference rasters highlighting vertical misalignments across overlapping flightlines
- **Dynamic Color Stretch** — custom min/max elevation difference clipping (e.g. 0.0 m to 0.3 m) with quick presets for sub-decimeter QA
- **OpenStreetMap Basemap** — asynchronous multithreaded tile downloading, disk caching, and on-the-fly reprojection to project CRS (e.g. UTM EPSG:25833)

### DTM / DSM Export
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
- **Conda** (Miniconda or Anaconda)
- **NVIDIA GPU** with CUDA 12.4+ driver support
- **Git**

### Environment Setup

Create and activate the unified conda environment containing PyTorch 2.5.0 (CUDA 12.4), Pointcept dependencies, and LiDAR Workbench:

```bash
# Clone the repository
git clone https://github.com/bkabelik/lidar-workbench.git
cd lidar-workbench

# Create the conda environment
conda env create -f environment.yml --verbose

# Activate the environment
conda activate pointcept-torch2.5.0-cu12.4
```

### C++/CUDA Operators (Optional)

The C++/CUDA operators (`flash-attn`, `pointops`, `pointgroup_ops`, `pointrope`) are included in `environment.yml`. If they were not compiled during the initial environment creation, install them manually with:

```bash
pip install flash-attn --no-build-isolation
pip install -e ./Pointcept/libs/pointops --no-build-isolation
pip install -e ./Pointcept/libs/pointgroup_ops --no-build-isolation
pip install -e ./Pointcept/libs/pointrope --no-build-isolation
```

### Pointcept AI Model Checkpoints

For deep-learning classification, place your trained model checkpoint (`.pth`) and configuration (`.py`) into `models/` and `configs/`:

**Download Model Weights:** [Pointcept Model](https://drive.google.com/file/d/15MlZ6cwed0jFsd7WKOdkDjQIQTiCy5nJ/view?usp=sharing)

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
2. **Preview LAS/LAZ data** — *File → Preview LAS/LAZ* (Ctrl+Shift+P) to inspect files before import:
   - See bounding boxes, 1M/10M subsampled, or full-resolution point clouds
   - Orbit/zoom with mouse, colour by height/classification/intensity/return/file
   - Check point density (bbox + grid-based effective density)
   - Click **Import…** from the preview to proceed to the import wizard
3. **Import LAS/LAZ data** — *File → Import LAS/LAZ* (Ctrl+I) or drag-and-drop a folder onto the window
4. **Apply noise filter** — Select tiles in the tile list → *Tools → Noise Filter* → choose filter type → *Apply*
5. **Classify with Pointcept** (optional) — Select filtered tiles → *Tools → Classify (Pointcept)* → configure → *Start*
6. **Manual editing & QC** — Double-click a classified tile to open the multi-view:
   - Draw a cross-section line in the DTM view (right-click + drag)
   - Inspect cross-section points in the 2D Profile View with instant 120+ FPS cached rendering
   - Check the **Upper-Left HUD Overlay** for the active corridor width and shortcut hints
   - Dynamically adjust profile parameters on the fly:
     - `Ctrl + Mouse Scroll`: Expand or shrink corridor slice width (e.g. 0.5 m to 20 m)
     - `Shift + Mouse Scroll` or `[` / `]`: Scale circular brush radius or rectangular brush dimensions
     - Mouse Scroll: Zoom smoothly centered at cursor
     - Middle-click Drag: Pan view
   - Direct toolbar spinbox controls: adjust **Radius**, **W**, **H**, and **Corridor** with live bidirectional sync
   - Switch colour modes on the toolbar: **By Class**, **By Height**, **By Intensity**, **By Return Number**, or **By Flightline**
   - Select misclassified points using any selection tool:
     - **Brush** (B) — click/drag to paint a circular region
     - **Rect Brush** (Shift+R) — click/drag to paint a rectangular region
     - **Above Line** (A) / **Below Line** (L) — draw a line to classify points above/below it
     - **Rectangle** (R) — drag to select points in a rectangular area
   - Click a class button in the properties panel to reclassify selected points
   - *Undo*/*Redo* as needed (Ctrl+Z / Ctrl+Y)
7. **Water Surface Modeling (WSM)** (optional) — *Tools → Water Surface Model (WSM)*:
   - Ergonomic vertical view: 2D Plan Map positioned above the 1D cross-section viewer
   - Select **Intensity 0.5 m** or **0.25 m** raster to clearly reveal water voids, shorelines, and banks
   - Step through sections (`A` / `D`), adjust water levels (`W` / `S`), and lock anchor sections (`Space`)
   - Click **Save All Profiles…** to persist all cross-section parameters across project sessions
   - Enforce downstream monotonicity and rasterize a seamless 3D water surface GeoTIFF
8. **Point QC & Strip Alignment** (optional) — *Tools → Point QC 2D Map*:
   - Stream OpenStreetMap (OSM) satellite/cartographic basemap tiles reprojected to your project CRS
   - Review pairwise and Max Strip Difference rasters to identify vertical calibration offsets
   - Adjust Min/Max color stretch (e.g. 0.0 m to 0.3 m) with quick presets for sub-decimeter elevation QC
9. **Export raster** — *Tools → Export Raster (DTM / DSM)*:
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

## Manual QC Profile View Controls & HUD

The 2D Profile View incorporates an ultra-fast offscreen bitmap caching engine (rendering at 120+ FPS) and an upper-left HUD card displaying live feedback:

| Shortcut / Control | Function | Description |
|---|---|---|
| **Ctrl + Mouse Scroll** | Adjust Corridor Width | Dynamically widens or narrows the profile cross-section slice (0.5 m increments). Live width is displayed in the HUD and synchronised with the toolbar spinbox. |
| **Shift + Mouse Scroll** | Adjust Brush / Rect Size | Dynamically scales the circular brush radius or rectangular brush dimensions. Synchronised with the toolbar spinboxes. |
| **`[` / `]` Keys** | Step Tool Size | Alternate keybind to decrease / increase active brush or rect brush size by 15%. |
| **Mouse Scroll** | Zoom View | Smooth zoom centered at the current mouse cursor position without affecting corridor width. |
| **Middle-Click Drag** | Pan View | Pan the 2D cross-section elevation and distance axes. |
| **Left-Click / Drag** | Apply Tool | Paint or draw using the active selection tool (Brush, Rect Brush, Above Line, Below Line, Rectangle). |
| **Upper-Left HUD Card** | Live Status Card | Semi-transparent rounded overlay displaying: `Corridor Width: X.X m`, `Ctrl + Scroll : Change corridor width`, and `Shift + Scroll or [ / ] : Adjust brush / rect size`. |

---

## Water Surface Modeler (WSM)

The Water Surface Modeler generates physically consistent water surface elevation models (3D GeoTIFF) along river corridors:

- **Vertical Split Layout**: The 2D Plan View map is positioned directly above the 1D cross-section editor, providing intuitive geographic alignment while refining individual profiles.
- **2D Channel Intensity Raster**: High-contrast 0.25 m / 0.5 m binned intensity raster (`np.bincount` with percentile stretch) reveals NIR shoreline voids and riverbanks even when water point returns are sparse, ensuring the operator has complete channel coverage.
- **Overlaid Cross-Sections**: 2D cut lines display status at a glance:
  - Cyan lines: Auto-generated cross-sections.
  - Lime green lines: Locked anchor sections confirmed by the user.
  - Glowing orange line: Active cross-section currently loaded in the 1D editor.
- **Batch Profile Management**:
  - **Save All Profiles…**: Exports the entire sequence of cross-sections, locked anchors, embankment offsets, and elevations into a single JSON/GeoJSON project file.
  - **Load All Profiles…**: Restores full channel state in one click for uninterrupted multi-session editing.
- **Monotonicity & Geometry**: Enforces strictly descending downstream water surface profiles and interpolates smoothly between user-locked anchors.

---

## Point QC & Strip Alignment 2D Map

- **OpenStreetMap Basemap**: Streamed slippy map tiles cached locally on disk (`~/.cache/lidar_workbench/osm_tiles`) and reprojected asynchronously via `rasterio.warp` to the project's native CRS.
- **Difference Rasters**:
  - Pairwise strip difference rasters: Compute vertical deviations between specific overlapping flight strips.
  - Max Strip Difference raster: Computes the maximum elevation discrepancy across all flightlines per raster cell.
- **Dynamic Color Stretch**:
  - Interactive Min and Max value spinboxes (e.g. 0.0 m to 0.3 m) allow isolating sub-decimeter vertical calibration offsets.
  - Instant preset buttons for standard elevation error ranges.

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
├── preview_dialog.py            # LAS/LAZ preview inspector
├── noise_filter.py              # SOR / ROR / DBSCAN / advanced filter algorithms
├── pointcept_worker.py          # Background subprocess inference runner
├── dtm_generator.py             # In-memory DTM interpolation
├── export_manager.py            # DTM/DSM/Hillshade export engine
├── manual_edit.py               # Profile extraction, selections, undo/redo
├── centerline_wsm.py            # River centerline extraction, interpolation & GeoTIFF export
├── gui/
│   ├── main_window.py           # QMainWindow (menus, toolbar, 3-panel splitter)
│   ├── tile_list_widget.py      # Tile browser with status groups
│   ├── multi_view_widget.py     # Synchronized 3D, DTM, and Profile view layout
│   ├── view_3d.py               # Open3D 3D point cloud widget
│   ├── view_dtm.py              # 2D DTM view with profile drawing
│   ├── view_profile.py          # 2D profile side view with 120+ FPS cache & HUD
│   ├── view_profile_3d.py       # 3D profile corridor view
│   ├── water_surface_dialog.py  # WSM generator (vertical 2D intensity map + 1D editor)
│   ├── point_qc_layers.py       # Strip difference raster layers & min/max stretch
│   ├── osm_basemap.py           # OSM tile downloader, disk cache & reprojection worker
│   ├── filter_dialog.py         # Noise filter parameter dialog
│   ├── classification_dialog.py # Pointcept configuration dialog
│   ├── export_dialog.py         # DTM/DSM export configuration dialog
│   ├── preview_dialog.py        # LAS/LAZ preview with density analysis
│   ├── properties_panel.py      # Point properties + quick-classify
│   └── settings_dialog.py       # Keyboard shortcuts + tool size settings
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
│   ├── ground/            # Ground-classification results (per tile)
│   ├── bathy/             # Bathymetry-processing results (per tile)
│   ├── noise/             # Noise-filtered points (per tile)
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

**Optional**: GDAL (for GeoTIFF output), matplotlib (height colour ramps).

For Pointcept classification, a separate conda environment with PyTorch and Point Transformer V3 dependencies is required — see the [PointceptALS](https://github.com/bkabelik/PointceptALS.git) README.

---

## Key Design Decisions

- **TIN interpolation for DTM** — preserves terrain discontinuities better than IDW-only methods, using Delaunay triangulation
- **Max-Z for DSM** — industry standard (PDAL) for surface models from LiDAR; avoids the "averaging" artefacts that smooth out building edges and vegetation
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
- Default tool sizes (`DEFAULT_BRUSH_RADIUS_M`, `DEFAULT_RECT_WIDTH_M`, `DEFAULT_RECT_HEIGHT_M`)

Keyboard shortcuts and tool sizes can be customised via *Tools → Settings*.

---

## License

This workbench integrates and builds upon several incredible open-source projects:

- [Pointcept](https://github.com/Pointcept/Pointcept) (MIT License)
- [Open3D](https://www.open3d.org) (MIT License)
- [laspy](https://github.com/laspy/laspy) (BSD 2-Clause)
- [PySide6](https://wiki.qt.io/Qt_for_Python) (LGPLv3)
- [NumPy](https://numpy.org) (BSD 3-Clause)
- [SciPy](https://scipy.org) (BSD 3-Clause)

The custom workbench code in this repository is licensed under the **GNU General Public License v3.0 (GPLv3)**.
See the [`LICENSE`](LICENSE) file for details.
