import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
import json
from scipy.spatial import cKDTree

def load_filter_settings(path=".filter_settings.json"):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def save_filter_settings(settings, path=".filter_settings.json"):
    try:
        with open(path, 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save filter settings: {e}")

def get_interactive_sample(points, intensities, return_num, total_returns, max_points=4000000):
    """
    Returns a central spatial crop of the points, capped at max_points.
    Preserves density for accurate filter tuning.
    """
    if len(points) <= max_points:
        return points, intensities, return_num, total_returns, False
    
    print(f"Large File detected ({len(points):,} points). Reducing to {max_points:,} for interactive tuning...")
    
    # Estimate crop size to hit target point count (2D assumption)
    ratio = (max_points * 1.1) / len(points) # 10% buffer
    spatial_ratio = np.sqrt(ratio)
    
    min_xyz = points.min(0)
    max_xyz = points.max(0)
    center = (min_xyz + max_xyz) / 2.0
    half_size = (max_xyz - min_xyz) * spatial_ratio / 2.0
    
    mask = (np.abs(points[:, 0] - center[0]) < half_size[0]) & \
           (np.abs(points[:, 1] - center[1]) < half_size[1])
    
    idx = np.where(mask)[0]
    if len(idx) > max_points:
        # Use random choice instead of slicing to avoid scan-line artifacts
        idx = np.random.choice(idx, max_points, replace=False)
    
    print(f"  -> Using uniform spatial sample of {len(idx):,} points for GUI.")
    return points[idx], intensities[idx], return_num[idx], total_returns[idx], True

def build_ground_model(points, return_num, total_returns, grid_size=0.5):
    """
    Build a Twin-Pass Robust DTM.
    Uses a coarse Anchor Pass to eliminate tree clusters and a Refined Pass for detail.
    """
    # 1. Base Ground Candidates
    mask = (return_num == total_returns)
    base_pts = points[mask]
    # Established a safety floor to prevent DTM from jumping into buildings/trees
    # print("  -> Fitting Global Plane Baseline...")
    subset_idx = np.random.choice(len(base_pts), min(len(base_pts), 5000), replace=False)
    sub = base_pts[subset_idx]
    
    # robust least squares: [x y 1] * [a b c]^T = z
    A_plane = np.column_stack([sub[:, 0], sub[:, 1], np.ones(len(sub))])
    B_plane = sub[:, 2]
    plane_params, _, _, _ = np.linalg.lstsq(A_plane, B_plane, rcond=None)
    
    # Calculate residuals and adjust 'c' to a robust floor (10th percentile of residuals)
    residuals = base_pts[:, 2] - (base_pts[:, 0] * plane_params[0] + base_pts[:, 1] * plane_params[1] + plane_params[2])
    floor_offset = np.percentile(residuals, 10.0)
    plane_params[2] += floor_offset
    
    # Prune ONLY extreme outliers (e.g. sensor errors), allowing real terrain dips
    res_pruned = base_pts[:, 2] - (base_pts[:, 0] * plane_params[0] + base_pts[:, 1] * plane_params[1] + plane_params[2])
    plane_mask = (res_pruned < 100.0) & (res_pruned > -100.0)
    base_pts = base_pts[plane_mask]
    # print(f"  -> Plane Baseline pruned {np.sum(~plane_mask):,} outliers.")

    min_x, max_x = np.min(base_pts[:, 0]), np.max(base_pts[:, 0])
    min_y, max_y = np.min(base_pts[:, 1]), np.max(base_pts[:, 1])
    
    # --- PASS 1: ANCHOR DTM (Coarse 8m Grid) ---
    anchor_size = 8.0
    anx = int(np.ceil((max_x - min_x) / anchor_size)) + 1
    any = int(np.ceil((max_y - min_y) / anchor_size)) + 1
    
    anchor_grid = np.full((anx, any), np.nan, dtype=np.float32)
    agx = np.clip(((base_pts[:, 0] - min_x) / anchor_size).astype(int), 0, anx - 1)
    agy = np.clip(((base_pts[:, 1] - min_y) / anchor_size).astype(int), 0, any - 1)
    
    # Binning for Anchor
    a_bins = agx * any + agy
    a_sort = np.argsort(a_bins)
    a_sorted_bins, a_first = np.unique(a_bins[a_sort], return_index=True)
    a_last = np.append(a_first[1:], len(a_sort))
    
    for i, b_idx in enumerate(a_sorted_bins):
        # 5th percentile for aggressive tree removal at coarse scale
        v = base_pts[a_sort[a_first[i]:a_last[i]], 2]
        anchor_grid[b_idx // any, b_idx % any] = np.percentile(v, 5.0)

    # Gap fill and smooth the anchor grid
    def simple_fill(g, passes=20):
        res = g.copy()
        for _ in range(passes):
            valid = ~np.isnan(res)
            if np.all(valid): break
            shifted_sum, count = np.zeros_like(res), np.zeros_like(res)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                s = np.full_like(res, np.nan)
                txs, txe = max(0, dx), res.shape[0] + min(0, dx)
                sxs, sxe = max(0, -dx), res.shape[0] + min(0, -dx)
                tys, tye = max(0, dy), res.shape[1] + min(0, dy)
                sys, sye = max(0, -dy), res.shape[1] + min(0, -dy)
                if txe > txs and tye > tys:
                    s[txs:txe, tys:tye] = res[sxs:sxe, sys:sye]
                m = ~np.isnan(s)
                shifted_sum[m] += s[m]; count[m] += 1
            upd = (~valid) & (count > 0)
            res[upd] = shifted_sum[upd] / count[upd]
        return res

    anchor_grid = simple_fill(anchor_grid)
    
    # Robust Morphological Processing for the Anchor
    import scipy.ndimage as ndimage
    
    # 0. Morphological Closing (Max then Min) to ERASE PITS (underground noise)
    # A max filter pulls the grid up over deep spikes, then min filter settles it back
    anchor_grid = ndimage.maximum_filter(anchor_grid, size=3)
    anchor_grid = ndimage.minimum_filter(anchor_grid, size=3)
    
    # 1. Morphological Opening (Min then Max) to ERASE BUILDINGS (warehouses)
    anchor_grid = ndimage.minimum_filter(anchor_grid, size=5)
    anchor_grid = ndimage.maximum_filter(anchor_grid, size=5)
    
    # 2. Final slight smoothing
    anchor_grid = ndimage.median_filter(anchor_grid, size=3)
    
    # --- PASS 2: REFINED DTM (2.0m Grid) ---
    # Only accept ground candidates that are within 2.5m of the Anchor Pass
    agx_full = np.clip(((base_pts[:, 0] - min_x) / anchor_size).astype(int), 0, anx - 1)
    agy_full = np.clip(((base_pts[:, 1] - min_y) / anchor_size).astype(int), 0, any - 1)
    anchor_h = anchor_grid[agx_full, agy_full]
    
    # Anchor Filter: Discard points significantly above OR BELOW the coarse terrain baseline
    # The lower bound (-1.5m) is critical to prevent the refined DTM from dipping into multipath noise
    refined_mask = (base_pts[:, 2] < anchor_h + 2.5) & (base_pts[:, 2] > anchor_h - 1.5)
    refined_pts = base_pts[refined_mask]
    
    nx = int(np.ceil((max_x - min_x) / grid_size)) + 1
    ny = int(np.ceil((max_y - min_y) / grid_size)) + 1
    grid = np.full((nx, ny), np.nan, dtype=np.float32)
    
    gx = np.clip(((refined_pts[:, 0] - min_x) / grid_size).astype(int), 0, nx - 1)
    gy = np.clip(((refined_pts[:, 1] - min_y) / grid_size).astype(int), 0, ny - 1)
    
    r_bins = gx * ny + gy
    r_sort = np.argsort(r_bins)
    r_sorted_bins, r_first = np.unique(r_bins[r_sort], return_index=True)
    r_last = np.append(r_first[1:], len(r_sort))

    for i, b_idx in enumerate(r_sorted_bins):
        # 30th percentile for stability on roads
        v = refined_pts[r_sort[r_first[i]:r_last[i]], 2]
        grid[b_idx // ny, b_idx % ny] = np.percentile(v, 30.0)

    # Removed global_floor to allow DTM to follow deep terrain dips
    
    def morph_op(input_grid, op_func, passes=8):
        curr = input_grid.copy()
        for _ in range(passes):
            snapshot = curr.copy()
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                s = np.full_like(snapshot, np.nan)
                txs, txe = max(0, dx), nx + min(0, dx)
                sxs, sxe = max(0, -dx), nx + min(0, -dx)
                tys, tye = max(0, dy), ny + min(0, dy)
                sys, sye = max(0, -dy), ny + min(0, -dy)
                if txe > txs and tye > tys:
                    s[txs:txe, tys:tye] = snapshot[sxs:sxe, sys:sye]
                curr = op_func(curr, s)
        return curr

    # Final Morphological Polish
    eroded = morph_op(grid, np.fmin, passes=8)
    dilated = morph_op(eroded, np.fmax, passes=8)
    grid = simple_fill(dilated, passes=50)
    
    # Removed safety floor clipping
    
    return grid, (min_x, min_y), grid_size

def get_ground_elevation(points, ground_model):
    """
    Get interpolated ground elevation for a set of points using Bilinear Interpolation.
    Eliminates staircase artifacts on sloped terrain.
    """
    grid, meta, grid_size = ground_model
    min_x, min_y = meta
    nx, ny = grid.shape
    
    # Calculate fractional coordinates
    fx = (points[:, 0] - min_x) / grid_size
    fy = (points[:, 1] - min_y) / grid_size
    
    # Get the four surrounding cell indices
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp to grid bounds
    x0 = np.clip(x0, 0, nx - 1); x1 = np.clip(x1, 0, nx - 1)
    y0 = np.clip(y0, 0, ny - 1); y1 = np.clip(y1, 0, ny - 1)
    
    # Calculate fractional weights [0, 1]
    wx = fx - x0
    wy = fy - y0
    
    # Fetch heights at corners
    z00 = grid[x0, y0]
    z10 = grid[x1, y0]
    z01 = grid[x0, y1]
    z11 = grid[x1, y1]
    
    # Robust NaN handling: use the median of the whole grid for missing data
    # This prevents 'sea level' spikes on GPS-height datasets
    fill_val = np.nanmedian(grid)
    if np.isnan(fill_val): fill_val = 0 # Final fallback
    z00[np.isnan(z00)] = fill_val
    z10[np.isnan(z10)] = fill_val
    z01[np.isnan(z01)] = fill_val
    z11[np.isnan(z11)] = fill_val

    # Bilinear Interpolation Formula
    z_interp = (z00 * (1 - wx) * (1 - wy) +
                z10 * wx * (1 - wy) +
                z01 * (1 - wx) * wy +
                z11 * wx * wy)
    
    return z_interp

def get_heights_above_ground(points, ground_model):
    """
    Get relative heights (Height above ground).
    Useful for DTM-based noise filtering.
    """
    z_ground = get_ground_elevation(points, ground_model)
    return points[:, 2] - z_ground

def calculate_cluster_linearity(points):
    """
    Calculate linearity for a set of points using eigenvalues of the covariance matrix.
    L = (e1 - e2) / e1. Near 1.0 for lines (wires), near 0.0 for blobs.
    """
    if len(points) < 5: # Too few points to reliably determine linearity
        return 0
    try:
        cov = np.cov(points, rowvar=False)
        evals = np.linalg.eigvalsh(cov) # Sorted ascending: e3, e2, e1
        e1, e2 = evals[2], evals[1]
        if e1 < 1e-9: return 0
        return (e1 - e2) / e1
    except:
        return 0
class NoiseFilterApp:
    def __init__(self, points, intensities=None, initial_params=None, initial_active=None, 
                 return_num=None, total_returns=None):
        self._is_loading = True
        self.points = points
        self.intensities = intensities
        self.return_num = return_num
        self.total_returns = total_returns
        self.mask = np.ones(len(points), dtype=bool)
        
        # Initial stats - Robust
        self.points_mean = np.mean(points, axis=0)
        self.points_local = points - self.points_mean # Local coords for rendering precision
        self.heights = None
        self.ground_pcd = None
        
        self.z_min = np.min(self.points_local[:, 2])
        self.z_max = np.max(self.points_local[:, 2])
        self.full_min = np.min(self.points_local, axis=0) # Lock DTM extents
        self.full_max = np.max(self.points_local, axis=0)
        
        # Default Filter Params
        self.params = {
            "sor_neighbors": 20,
            "sor_std_ratio": 4.0,
            "ror_radius": 0.5,           # Reduced default for speed
            "ror_min_points": 2,
            "dbscan_eps": 0.5,
            "dbscan_min_points": 10,
            "dbscan_base_alt": 5.0,
            "dbscan_linearity": 0.85,
            "floor": True,
            "floor_buffer": 2.0, # Increased for safety (no more road erasure)
            "dbscan": True,
            "intensity_min": 0.0,
            "dtm_grid_size": 2.0,
            "max_gui_points": 25000000
        }
        
        # Override with initial params if provided
        if initial_params:
            self.params.update(initial_params)
        
        self.active_filters = {
            "sor": False,
            "ror": False, # OFF by default for instant launch
            "dbscan": False,
            "floor": False, # OFF by default for instant launch
            "intensity": False
        }
        
        if initial_active:
            self.active_filters.update(initial_active)
        self.apply_to_all = False

        # Setup GUI
        self.window = gui.Application.instance.create_window("Advanced Point Cloud Noise Filter", 1280, 800)
        
        # Theme
        em = self.window.theme.font_size
        
        # Scene Widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.05, 0.05, 0.05, 1.0]) # Dark background
        
        # Point Cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points_local[:, :3])
        self.original_colors = self._get_initial_colors()
        self.pcd.colors = o3d.utility.Vector3dVector(self.original_colors)
        
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit" # FLAT unlit points like CloudCompare
        self.mat.point_size = 3.0 # Thicker by default
        
        self.scene.scene.add_geometry("pcd", self.pcd, self.mat)
        
        if self.ground_pcd:
            self.scene.scene.add_geometry("ground", self.ground_pcd, rendering.MaterialRecord())
            self.scene.scene.show_geometry("ground", True) # ON BY DEFAULT
            
        self._reset_camera()
        
        # UI Panels
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        
        # Section: Stats
        self.label_stats = gui.Label(f"Points in GUI: {len(points):,}")
        self.panel.add_child(self.label_stats)
        
        self.label_outliers = gui.Label("Outliers: 0 (0.00%)")
        self.panel.add_child(self.label_outliers)
        
        # Section: Tools
        btn_reset = gui.Button("Reset Camera (Robust)")
        btn_reset.set_on_clicked(self._reset_camera)
        self.panel.add_child(btn_reset)
        
        # Point Size Slider
        hbox_ps = gui.Horiz(0.5 * em)
        hbox_ps.add_child(gui.Label("Point Size: "))
        slider_ps = gui.Slider(gui.Slider.INT)
        slider_ps.set_limits(1, 10)
        slider_ps.double_value = 3.0
        def on_ps_change(val):
            # Safe way to update point size without crashing: 
            # Modify and trigger a redraw/geometry update
            self.mat.point_size = int(val)
            self.scene.scene.remove_geometry("pcd")
            self.scene.scene.add_geometry("pcd", self.pcd, self.mat)
        slider_ps.set_on_value_changed(on_ps_change)
        hbox_ps.add_child(slider_ps)
        self.panel.add_child(hbox_ps)
        
        # Ground Visualization Toggle
        cb_ground = gui.Checkbox("Show Ground Terrain (Green)")
        cb_ground.checked = True # CHECKED BY DEFAULT
        def on_ground_toggle(val):
            if self.ground_pcd:
                self.scene.scene.show_geometry("ground", val)
        cb_ground.set_on_checked(on_ground_toggle)
        self.panel.add_child(cb_ground)
        
        # Section: Ground Filter
        self.panel.add_child(self._create_filter_section("Ground Elevation Filter", "floor", [
            ("Allowance below ground (m)", "floor_buffer", 0.0, 10.0, 0.5)
        ]))
        
        # Section: ROR
        self.panel.add_child(self._create_filter_section("Radius Outlier (ROR)", "ror", [
            ("Radius (m)", "ror_radius", 0.1, 10.0, 2.0),
            ("Min Points", "ror_min_points", 1, 50, 2)
        ]))
        
        # Section: SOR
        self.panel.add_child(self._create_filter_section("Statistical Outlier (SOR)", "sor", [
            ("Neighbors", "sor_neighbors", 5, 100, 20),
            ("Std Ratio", "sor_std_ratio", 0.1, 10.0, 4.0)
        ]))
        
        # Section: DBSCAN
        self.panel.add_child(self._create_filter_section("DBSCAN (Air Clusters)", "dbscan", [
            ("Eps distance", "dbscan_eps", 0.1, 5.0, 0.5),
            ("Min Cluster Size", "dbscan_min_points", 2, 500, 10),
            ("Base Altitude (m)", "dbscan_base_alt", 0.0, 50.0, 5.0),
            ("Linearity Protect", "dbscan_linearity", 0.0, 1.0, 0.85)
        ]))

        # Section: Sampling (Info only)
        section_samp = gui.CollapsableVert("Sampling Settings", 0.25 * em, gui.Margins(em, 0, 0, 0))
        hbox_samp = gui.Horiz(0.5 * em)
        hbox_samp.add_child(gui.Label("GUI Limit: "))
        slider_samp = gui.Slider(gui.Slider.INT)
        slider_samp.set_limits(1000000, 50000000) # Increased to 50M
        slider_samp.double_value = float(self.params["max_gui_points"])
        def on_samp_change(val):
            self.params["max_gui_points"] = int(val)
        slider_samp.set_on_value_changed(on_samp_change)
        hbox_samp.add_child(slider_samp)
        section_samp.add_child(hbox_samp)
        section_samp.add_child(gui.Label("(Apply on next file restart)"))
        self.panel.add_child(section_samp)

        # Section: Intensity
        self.panel.add_child(self._create_filter_section("Intensity Filter", "intensity", [
            ("Min Intensity", "intensity_min", 0.0, 1.0, 0.0)
        ]))

        # Section: Terrain Settings
        self.panel.add_child(self._create_filter_section("Terrain Settings", None, [
            ("DTM Resolution (m)", "dtm_grid_size", 0.2, 10.0, 2.0)
        ]))

        # --- TERRAIN BUTTON ---
        self.panel.add_fixed(em)
        btn_terrain = gui.Button("Build Terrain Model (From Clean Data)")
        btn_terrain.set_on_clicked(self._on_build_dtm)
        self.panel.add_child(btn_terrain)
        self.panel.add_fixed(5)

        # Confirm Buttons
        hbox_btns = gui.Horiz(0.5 * em)
        btn_tile = gui.Button("Accept this tile")
        btn_tile.set_on_clicked(self._on_confirm_tile)
        hbox_btns.add_child(btn_tile)
        
        btn_all = gui.Button("Use for ALL tiles")
        btn_all.set_on_clicked(self._on_confirm_all)
        hbox_btns.add_child(btn_all)
        
        self.panel.add_child(hbox_btns)
        
        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        
        # Set layout
        self.window.set_on_layout(self._on_layout)
        
        # Initial Update
        self._is_loading = False
        self._update_all()

    def _reset_camera(self):
        # Focus on the 'Real' data by calculating a robust bounding box (1st to 99th percentile)
        # This ignores extreme air/ground noise that zooms the camera out too far
        try:
            q_min = np.percentile(self.points_local, 1, axis=0)
            q_max = np.percentile(self.points_local, 99, axis=0)
            
            center = (q_min + q_max) / 2.0
            size = (q_max - q_min)
            # Create a robust box for camera focus
            bbox = o3d.geometry.AxisAlignedBoundingBox(q_min, q_max)
            self.scene.setup_camera(30, bbox, center) # 30 degree FOV for technical look
        except:
            self.scene.setup_camera(30, self.scene.scene.bounding_box, [0, 0, 0])

    def _get_initial_colors(self):
        # 1. Vibrant Turquoise/Blue Height Ramp (User Choice for Noise Filtering)
        z = self.points_local[:, 2]
        z_min = np.percentile(z, 1)
        z_max = np.percentile(z, 99)
        z_norm = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1)
        
        # Original-style vibrant Turquoise scheme
        colors = np.zeros((len(z), 3))
        colors[:, 0] = z_norm * 0.2 + 0.1 # Darker start
        colors[:, 1] = z_norm * 0.7 + 0.3 # Turquoise green-blue
        colors[:, 2] = 0.8               # Strong blue base
        return colors

    def _create_filter_section(self, title, key, sliders):
        em = self.window.theme.font_size
        section = gui.CollapsableVert(title, 0.25 * em, gui.Margins(em, 0, 0, 0))
        
        if key is not None:
            cb = gui.Checkbox("Enabled")
            cb.checked = self.active_filters[key]
            def on_toggle(checked):
                self.active_filters[key] = checked
                self._update_all()
            cb.set_on_checked(on_toggle)
            section.add_child(cb)
        
        for name, p_key, v_min, v_max, v_init in sliders:
            hbox = gui.Horiz(0.5 * em)
            hbox.add_child(gui.Label(f"{name}: "))
            
            # Use the currently loaded param if it exists, otherwise use v_init
            current_val = self.params.get(p_key, v_init)
            
            is_double = isinstance(v_init, float)
            slider = gui.Slider(gui.Slider.DOUBLE if is_double else gui.Slider.INT)
            slider.set_limits(v_min, v_max)
            slider.double_value = float(current_val)
            
            num_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE if is_double else gui.NumberEdit.INT)
            num_edit.set_limits(v_min, v_max)
            num_edit.double_value = float(current_val)
            
            # Use a closure to capture UI elements for synchronization
            def make_on_change(k, s, n):
                def on_change(val):
                    self.params[k] = val
                    s.double_value = val
                    n.double_value = val
                    self._update_all()
                return on_change
            
            change_callback = make_on_change(p_key, slider, num_edit)
            slider.set_on_value_changed(change_callback)
            num_edit.set_on_value_changed(change_callback)
            
            hbox.add_child(slider)
            hbox.add_child(num_edit)
            section.add_child(hbox)
            
        return section

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 450 # Standard sidebar width
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        self.scene.frame = gui.Rect(r.x + panel_width, r.y, r.width - panel_width, r.height)

    def _on_build_dtm(self):
        print("Manual Terrain Build Triggered...")
        valid_indices = np.where(self.mask)[0]
        if len(valid_indices) > 100 and self.return_num is not None:
            dtm_grid, dtm_meta, dtm_size = build_ground_model(
                self.points_local[valid_indices], 
                self.return_num[valid_indices], 
                self.total_returns[valid_indices],
                grid_size=self.params["dtm_grid_size"]
            )
            self.ground_model = (dtm_grid, dtm_meta, dtm_size)
            
            # Calculate heights for visualization
            gx = ((self.points_local[:, 0] - dtm_meta[0]) / dtm_size).astype(int)
            gy = ((self.points_local[:, 1] - dtm_meta[1]) / dtm_size).astype(int)
            gx = np.clip(gx, 0, dtm_grid.shape[0] - 1)
            gy = np.clip(gy, 0, dtm_grid.shape[1] - 1)
            self.heights = self.points_local[:, 2] - dtm_grid[gx, gy]
            
            # Update visualization mesh
            res_x, res_y = dtm_grid.shape
            grid_pts = []
            
            # Use stride=1 to show the actual resolution, with a safety cap for huge grids
            stride = 1
            if res_x * res_y > 250000:
                stride = max(1, int(np.ceil(np.sqrt(res_x * res_y / 250000))))
                print(f"  -> Grid too dense for markers, using stride {stride}")

            for i in range(0, res_x, stride):
                for j in range(0, res_y, stride):
                    val = dtm_grid[i, j]
                    if not np.isnan(val): # Skip holes (prevent graphics crash)
                        # Lift by 0.1m to prevent Z-fighting with the ground points
                        grid_pts.append([dtm_meta[0] + i*dtm_size, dtm_meta[1] + j*dtm_size, val + 0.1])
            
            if len(grid_pts) > 0:
                pcd_array = np.array(grid_pts)
                print(f"  -> Visualizing DTM with {len(grid_pts)} markers.")
                print(f"     Ground BBox: Min {np.min(pcd_array, axis=0)}, Max {np.max(pcd_array, axis=0)}")
                
                if self.ground_pcd is None:
                    self.ground_pcd = o3d.geometry.PointCloud()
                self.ground_pcd.points = o3d.utility.Vector3dVector(pcd_array)
                self.ground_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Vibrant Green
                
                # Use a visible Material for the ground dots
                mat = rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                mat.base_color = [0.0, 1.0, 0.0, 1.0]
                mat.point_size = 12.0 * self.window.scaling # Very High visibility
                
                self.scene.scene.remove_geometry("ground")
                self.scene.scene.add_geometry("ground", self.ground_pcd, mat)
            else:
                print("  ! Warning: DTM calculation resulted in 0 points.")
            
            self._update_all() # Re-run to apply the floor filter with new heights
            self.window.post_redraw()

    def _update_all(self):
        if getattr(self, "_is_loading", False):
            return
        print("Re-calculating Noise Filters...")
        self.mask = np.ones(len(self.points), dtype=bool)
        
        # --- PASS 1: Noise Removal ---
        if self.active_filters["intensity"] and self.intensities is not None:
            max_i = np.max(self.intensities) + 1e-6
            self.mask &= (self.intensities / max_i >= self.params["intensity_min"])

        if self.active_filters["ror"]:
            _, ind = self.pcd.remove_radius_outlier(nb_points=int(self.params["ror_min_points"]), 
                                                   radius=self.params["ror_radius"])
            m = np.zeros(len(self.points), dtype=bool); m[ind] = True
            self.mask &= m
            
        if self.active_filters["sor"]:
            _, ind = self.pcd.remove_statistical_outlier(nb_neighbors=int(self.params["sor_neighbors"]), 
                                                        std_ratio=self.params["sor_std_ratio"])
            m = np.zeros(len(self.points), dtype=bool); m[ind] = True
            self.mask &= m

        # --- PASS 2: Ground-Dependent Filters (Only runs if Ground Model was manually built) ---
        if self.heights is not None:
            if self.active_filters["floor"]:
                self.mask &= (self.heights >= -self.params["floor_buffer"])
                
            if self.active_filters["dbscan"]:
                # Detect noise clusters above the ground
                noise_cand_idx = np.where(self.mask & (self.heights > self.params["dbscan_base_alt"]))[0]
                if len(noise_cand_idx) > 0:
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(self.points_local[noise_cand_idx])
                    labels = np.array(pcd_temp.cluster_dbscan(eps=self.params["dbscan_eps"], 
                                                            min_points=int(self.params["dbscan_min_points"])))
                    
                    subset_noise = (labels < 0)
                    for i in range(labels.max() + 1):
                        cluster_indices = np.where(labels == i)[0]
                        # Power Line Protection (Linearity)
                        lin = calculate_cluster_linearity(self.points_local[noise_cand_idx[cluster_indices]])
                        if lin < self.params["dbscan_linearity"]:
                            subset_noise[cluster_indices] = True
                    
                    m = np.ones(len(self.points), dtype=bool)
                    m[noise_cand_idx[subset_noise]] = False
                    self.mask &= m

        # Update colors in viewer
        new_colors = self.original_colors.copy()
        new_colors[~self.mask] = [1.0, 0.0, 1.0] # Highlight outliers in Vibrant MAGENTA
        self.pcd.colors = o3d.utility.Vector3dVector(new_colors)
        
        # Robust update: remove and re-add to ensure the scene sees the new colors
        self.scene.scene.remove_geometry("pcd")
        self.scene.scene.add_geometry("pcd", self.pcd, self.mat)
        self.window.post_redraw()
        
        # Update stats
        outlier_count = np.sum(~self.mask)
        percent = (outlier_count / len(self.points)) * 100
        self.label_outliers.text = f"Outliers: {outlier_count:,} ({percent:.2f}%)"

    def _on_confirm_tile(self):
        self.apply_to_all = False
        self.is_done = True
        self.window.close()

    def _on_confirm_all(self):
        self.apply_to_all = True
        self.is_done = True
        self.window.close()

    def run(self):
        self.is_done = False
        import time
        while not self.is_done:
            gui.Application.instance.run_one_tick()
            time.sleep(0.01)
        return self.mask, self.params, self.active_filters, self.apply_to_all

_gui_initialized = False
_dummy_window = None

def run_interactive_filter(points, intensities=None, initial_params=None, initial_active=None, return_num=None, total_returns=None):
    global _gui_initialized, _dummy_window
    if not _gui_initialized:
        gui.Application.instance.initialize()
        _dummy_window = gui.Application.instance.create_window("Dummy", 1, 1)
        _dummy_window.show(False)
        _gui_initialized = True
        
    app = NoiseFilterApp(points, intensities, initial_params, initial_active, 
                         return_num=return_num, total_returns=total_returns)
    return app.run()

def apply_headless_filter(points, intensities, params, active_filters, return_num=None, total_returns=None):
    """
    Run the noise filters with Dynamic DTM Re-calculation.
    Sequence: Noise Removal -> Refine DTM -> Elevation Filter
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    combined_mask = np.ones(len(points), dtype=bool)
    
    # 1. Pre-DTM Noise Filtering
    if active_filters.get("intensity", False) and intensities is not None:
        max_i = np.max(intensities) + 1e-6
        combined_mask &= (intensities / max_i >= params["intensity_min"])

    if active_filters.get("ror", False):
        _, ind = pcd.remove_radius_outlier(nb_points=int(params["ror_min_points"]), 
                                            radius=params["ror_radius"])
        m = np.zeros(len(points), dtype=bool); m[ind] = True
        combined_mask &= m
        
    if active_filters.get("sor", False):
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=int(params["sor_neighbors"]), 
                                                std_ratio=params["sor_std_ratio"])
        m = np.zeros(len(points), dtype=bool); m[ind] = True
        combined_mask &= m
    
    # 2. Refined DTM Build from partially cleaned data (ROR/SOR/Intensity only)
    valid_idx = np.where(combined_mask)[0]
    ground_model = build_ground_model(points[valid_idx], return_num[valid_idx], total_returns[valid_idx], 
                                     grid_size=params.get("dtm_grid_size", 2.0))
    heights = get_heights_above_ground(points, ground_model)
    
    # 3. Post-DTM Filtering
    if active_filters.get("floor", False):
        combined_mask &= (heights >= -params["floor_buffer"])
        
    if active_filters.get("dbscan", False):
        noise_cand_idx = np.where(combined_mask & (heights > params["dbscan_base_alt"]))[0]
        if len(noise_cand_idx) > 0:
            pcd_temp = o3d.geometry.PointCloud()
            # Use original positions for clustering
            pcd_temp.points = o3d.utility.Vector3dVector(points[noise_cand_idx])
            labels = np.array(pcd_temp.cluster_dbscan(eps=params["dbscan_eps"], 
                                                    min_points=int(params["dbscan_min_points"])))
            subset_noise = (labels < 0)
            for i in range(labels.max() + 1):
                c_idx = np.where(labels == i)[0]
                lin = calculate_cluster_linearity(points[noise_cand_idx[c_idx]])
                if lin < params["dbscan_linearity"]:
                    subset_noise[c_idx] = True
            combined_mask[noise_cand_idx[subset_noise]] = False
        
    return combined_mask, ground_model
