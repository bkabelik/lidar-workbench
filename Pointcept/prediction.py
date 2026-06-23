import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import copy
from collections import OrderedDict
import laspy

# Pointcept imports
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.builder import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.datasets.transform import Compose
from pointcept.datasets import collate_fn

import open3d as o3d
from scipy.spatial import cKDTree
from scipy import stats
from noisefilter import (build_ground_model, get_heights_above_ground, get_ground_elevation,
                        apply_headless_filter, run_interactive_filter, get_interactive_sample, 
                        load_filter_settings, save_filter_settings)
from postclassification import refine_classifications

def parse_args():
    parser = argparse.ArgumentParser("Pointcept Predictor for LAS files")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .las files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (e.g., model_best.pth)")
    parser.add_argument("--config_file", type=str, default="configs/dales/ptv3_dales.py", help="Model config file")
    parser.add_argument("--noise_filter", choices=["yes", "no", "interactive"], default="interactive", help="Enable noise filtering")
    parser.add_argument("--smoothing", choices=["yes", "no"], default="yes", help="Enable majority voting smoothing")
    parser.add_argument("--post_process", choices=["yes", "no"], default="no", help="Enable post-classification geometric refinement")
    parser.add_argument("--show_thinned", choices=["yes", "no"], default="no", help="Show the thinned point cloud before prediction")
    parser.add_argument("--show_predicted", choices=["yes", "no"], default="no", help="Show the thinned cloud with predictions before saving")
    parser.add_argument("--voxel_size", type=float, default=0.15, help="Voxel size for density normalization (use > line spacing)")
    parser.add_argument("--sample_mode", choices=["voxel", "random", "fps", "poisson"], default="voxel", help="Sampling strategy for density normalization")
    parser.add_argument("--target_pts", type=int, default=1000000, help="Target number of points for random/fps sampling")
    parser.add_argument("--intensity_scale", type=float, default=61430.0, help="Physical max intensity of your scanner to normalize to 0-1.")
    parser.add_argument("--options", nargs="+", action="append", help="override some settings in the used config")
    return parser.parse_args()

def _show_pcd_worker(points, colors, window_name):
    import open3d as o3d
    pcd_show = o3d.geometry.PointCloud()
    pcd_show.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd_show.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_show], window_name=window_name)

def show_pcd_isolated(points, colors, window_name):
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=_show_pcd_worker, args=(points, colors, window_name))
    p.start()
    p.join()

def smooth_predictions(points, preds, k=30, z_threshold=2.0):
    """
    Apply k-NN majority voting smoothing. 
    Improved: Edge-preserving by ignoring neighbors with large height differences.
    """
    print(f"Applying edge-preserving k-NN (k={k}, dZ={z_threshold}m) smoothing...")
    tree = cKDTree(points)
    dists, indices = tree.query(points, k=k)
    
    smoothed_preds = np.zeros_like(preds)
    for i in range(len(preds)):
        neighbor_indices = indices[i]
        z_diff = np.abs(points[neighbor_indices, 2] - points[i, 2])
        valid_neighbors = neighbor_indices[z_diff < z_threshold]
        
        if len(valid_neighbors) > 0:
            neighbors_preds = preds[valid_neighbors]
            counts = np.bincount(neighbors_preds)
            smoothed_preds[i] = np.argmax(counts)
        else:
            smoothed_preds[i] = preds[i]
    return smoothed_preds

# Global state to store persistent filter parameters across tiles
persistent_filter_config = {
    "params": None,
    "active": None,
    "apply_to_all": False,
    "max_gui_points": 25000000
}

def predict_las(las_file_path, model, transform, cfg, args):
    print(f"Loading {las_file_path}...")
    las = laspy.read(las_file_path)
    
    points = np.array(las.xyz)
    intensities = np.array(las.intensity).astype(np.float32)
    
    try:
        return_num = np.array(las.return_number.array if hasattr(las.return_number, 'array') else las.return_number)
        total_returns = np.array(las.number_of_returns.array if hasattr(las.number_of_returns, 'array') else las.number_of_returns)
    except Exception:
        print("  ! Warning: Return number information missing.")
        return_num = np.ones(len(points), dtype=np.uint8)
        total_returns = np.ones(len(points), dtype=np.uint8)

    global persistent_filter_config
    if persistent_filter_config["params"] is None:
        saved = load_filter_settings()
        if saved:
            persistent_filter_config["params"] = saved.get("params")
            persistent_filter_config["active"] = saved.get("active")
            persistent_filter_config["max_gui_points"] = max(saved.get("max_gui_points", 25000000), 25000000)

    if args.noise_filter == "yes":
        keep_mask, ground_model = apply_headless_filter(points, intensities, 
                                         persistent_filter_config["params"], 
                                         persistent_filter_config["active"],
                                         return_num=return_num,
                                         total_returns=total_returns)
    elif args.noise_filter == "interactive":
        if persistent_filter_config["apply_to_all"]:
            keep_mask, ground_model = apply_headless_filter(points, intensities, 
                                             persistent_filter_config["params"], 
                                             persistent_filter_config["active"],
                                             return_num=return_num,
                                             total_returns=total_returns)
        else:
            gui_pts, gui_int, gui_ret, gui_tot, is_sampled = get_interactive_sample(
                points, intensities, return_num, total_returns,
                max_points=persistent_filter_config["max_gui_points"]
            )
            keep_mask_gui, params, active, apply_to_all = run_interactive_filter(
                gui_pts, gui_int,
                initial_params=persistent_filter_config["params"],
                initial_active=persistent_filter_config["active"],
                return_num=gui_ret,
                total_returns=gui_tot
            )
            if params:
                persistent_filter_config["params"] = params
                persistent_filter_config["active"] = active
                save_filter_settings({
                    "params": params, 
                    "active": active,
                    "max_gui_points": persistent_filter_config["max_gui_points"]
                }) 
            if apply_to_all:
                persistent_filter_config["apply_to_all"] = True
            
            keep_mask, ground_model = apply_headless_filter(points, intensities, 
                                             persistent_filter_config["params"], 
                                             persistent_filter_config["active"],
                                             return_num=return_num,
                                             total_returns=total_returns)
    else:
        keep_mask = np.ones(len(points), dtype=bool)
        ground_model = build_ground_model(points, return_num, total_returns)
        
    valid_points_full = points[keep_mask]
    valid_intensities_full = intensities[keep_mask]
    valid_indices_full = np.where(keep_mask)[0]
    
    # Adaptive Density Normalization
    print(f"Normalizing density (mode: {args.sample_mode})...")
    min_x_full, min_y_full = np.min(valid_points_full[:, 0]), np.min(valid_points_full[:, 1])
    
    if args.sample_mode == "voxel":
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(valid_points_full)
        pcd_down = pcd_temp.voxel_down_sample(voxel_size=args.voxel_size)
        valid_points = np.asarray(pcd_down.points) # MUST remain float64 to prevent UTM 0.5m quantization
        subsampled = True
    elif args.sample_mode == "random":
        target_n = min(args.target_pts, len(valid_points_full))
        idx = np.random.choice(len(valid_points_full), target_n, replace=False)
        valid_points = valid_points_full[idx] # Keep float64
        valid_intensities = valid_intensities_full[idx].astype(np.float32)
        subsampled = True
    elif args.sample_mode == "poisson":
        res = args.voxel_size
        min_c = np.min(valid_points_full, axis=0)
        grid_coords = np.floor((valid_points_full - min_c) / res).astype(int)
        keys = grid_coords[:, 0] * 1e12 + grid_coords[:, 1] * 1e6 + grid_coords[:, 2]
        idx_sort = np.argsort(keys)
        keys_sorted = keys[idx_sort]
        _, first_indices, inverse = np.unique(keys_sorted, return_index=True, return_inverse=True)
        counts = np.diff(np.append(first_indices, len(idx_sort)))
        valid_points = np.zeros((len(first_indices), 3), dtype=np.float64) # Keep float64
        for i in range(3):
            valid_points[:, i] = np.bincount(inverse, weights=valid_points_full[idx_sort, i]) / counts
        valid_intensities = np.bincount(inverse, weights=valid_intensities_full[idx_sort]) / counts
        subsampled = True
    else:
        valid_points = valid_points_full.copy() # Keep float64
        valid_intensities = valid_intensities_full.astype(np.float32)
        subsampled = False

    if subsampled and args.sample_mode in ["voxel"]:
        full_tree = cKDTree(valid_points_full)
        _, ii = full_tree.query(valid_points, k=1)
        valid_intensities = valid_intensities_full[ii].astype(np.float32)

    if args.show_thinned == "yes":
        print("Displaying thinned point cloud (Close window to continue)...")
        colors = np.tile([0.5, 0.5, 0.5], (len(valid_points), 1))
        show_pcd_isolated(valid_points, colors, "Thinned Point Cloud")

    # Prediction
    global_pred_logits = np.zeros((len(valid_points), cfg.data.num_classes), dtype=np.float32)
    model.eval()
    
    min_x, max_x = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
    min_y, max_y = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
    block_size = 50.0
    stride = 25.0 
    
    blocks_x = int(np.ceil((max_x - min_x) / stride))
    blocks_y = int(np.ceil((max_y - min_y) / stride))
    print(f"Processing {blocks_x * blocks_y} blocks (50m size, 25m stride)...")
    
    for bx in range(blocks_x):
        for by in range(blocks_y):
            x_start = min_x + bx * stride
            y_start = min_y + by * stride
            x_end = x_start + block_size
            y_end = y_start + block_size
            
            x_mask = (valid_points[:, 0] >= x_start) & (valid_points[:, 0] < x_end)
            y_mask = (valid_points[:, 1] >= y_start) & (valid_points[:, 1] < y_end)
            block_mask = x_mask & y_mask
            
            block_idx = np.where(block_mask)[0]
            if len(block_idx) < 10: continue
                
            block_points = valid_points[block_idx].copy()
            block_intensities = valid_intensities[block_idx]
            
            local_center_x = (x_start + x_end) / 2.0
            local_center_y = (y_start + y_end) / 2.0
            z_floors = get_ground_elevation(block_points, ground_model)
            
            # Normalization to match training data observed in coord.npy (-25 to 25 range)
            block_points[:, 0] -= (x_start + 25.0)
            block_points[:, 1] -= (y_start + 25.0)
            
            # Z Normalization: Height Above Ground (HAG)
            block_points[:, 2] -= z_floors
            
            # Clip Z to match preprocessing (prevents negative Z features)
            # We clip the upper bound safely to 100 as well
            block_points[:, 2] = np.clip(block_points[:, 2], 0, 100.0)
            
            dist_x = np.abs(block_points[:, 0]) / (block_size / 2)
            dist_y = np.abs(block_points[:, 1]) / (block_size / 2)
            weights = (1.0 - np.maximum(dist_x, dist_y)) ** 2
            weights = np.clip(weights, 0.01, 1.0)
            
            # Normalization for Intensity MUST match training metadata
            norm_intensities = np.clip(block_intensities / args.intensity_scale, 0, 1)
            
            data_dict = {
                "coord": block_points.astype(np.float32), # Safely cast to float32 now that coordinates are strictly local [-25, 25]
                "strength": norm_intensities.reshape([-1, 1]),
                "segment": np.zeros(len(block_points), dtype=np.int32),
            }
            
            data_dict = transform(data_dict)
            fragment_list = data_dict["fragment_list"]
            segment = data_dict["segment"]
            
            pred = torch.zeros((segment.shape[0], cfg.data.num_classes)).cuda()
            for i in range(len(fragment_list)):
                input_dict = fragment_list[i]
                input_dict = collate_fn([input_dict])
                for key in list(input_dict.keys()):
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                input_dict.pop("segment", None)
                with torch.no_grad():
                    pred_part = model(input_dict)["seg_logits"]
                    pred_part = F.softmax(pred_part, -1)
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[input_dict["index"][bs:be], :] += pred_part[bs:be]
                        bs = be

            pred_part = pred.cpu().numpy()
            if "inverse" in data_dict.keys():
                inv = data_dict["inverse"]
                if isinstance(inv, torch.Tensor):
                    inv = inv.cpu().numpy()
                pred_part = pred_part[inv]
            global_pred_logits[block_idx] += (pred_part * weights[:, np.newaxis])

    pred_classes = np.argmax(global_pred_logits, axis=1)
    
    if args.show_predicted == "yes":
        print("Displaying raw predictions (Close window to continue)...")
        cmap = np.array([[0.5, 0.5, 0.5], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        colors = cmap[pred_classes % len(cmap)]
        show_pcd_isolated(valid_points, colors, "Raw Predictions")

    if args.smoothing == "yes":
        pred_classes = smooth_predictions(valid_points, pred_classes)
        
    # Mapping back to full cloud
    if subsampled:
        print("Mapping predictions back to full density...")
        final_tree = cKDTree(valid_points)
        _, ii = final_tree.query(valid_points_full, k=5)
        neighbor_labels = pred_classes[ii]
        pred_classes_full, _ = stats.mode(neighbor_labels, axis=1, keepdims=False)
        pred_classes_full = pred_classes_full.squeeze()
    else:
        pred_classes_full = pred_classes

    # Post-classification
    h_above = get_heights_above_ground(valid_points_full, ground_model)
    if args.post_process == "yes":
        pred_classes_full = refine_classifications(valid_points_full, pred_classes_full, h_above)

    # Remap to ASPRS
    id2asprs = np.array([2, 5, 20, 20, 14, 19, 15, 6], dtype=np.uint8)
    asprs_preds = id2asprs[pred_classes_full]
    final_classes = np.full(len(points), 7, dtype=np.uint8) # 7: Low Noise
    final_classes[valid_indices_full] = asprs_preds
    
    las.classification = final_classes
    out_dir = os.path.join(args.folder, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(las_file_path))
    las.write(out_path)
    print(f"Saved: {out_path}")

def main():
    args = parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    cfg = default_setup(cfg)
    
    print("=> Building model ...")
    model = build_model(cfg.model).cuda()
    checkpoint = torch.load(args.model_path)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module."): key = key[7:]
        weight[key] = value
    model.load_state_dict(weight, strict=True)
    
    # Setup transform
    # Setup transform - Dynamically inject return_inverse for GridSample and remove CenterShift
    val_transform_cfg = []
    for t in copy.deepcopy(cfg.data.val.transform):
        if t["type"] == "CenterShift":
            continue # We already do absolute grid centering in the inference loop
        if t["type"] == "GridSample":
            t["return_inverse"] = True
        if t["type"] == "Collect":
            if "keys" in t:
                t["keys"] = list(t["keys"])
                if "inverse" not in t["keys"]:
                    t["keys"].append("inverse")
        val_transform_cfg.append(t)
    
    transform = Compose(val_transform_cfg)
    class TestWrapper:
        def __init__(self, transform, test_cfg):
            self.transform = transform
            self.test_cfg = test_cfg
        def __call__(self, data_dict):
            data_dict = self.transform(data_dict)
            if self.test_cfg is not None:
                voxelize_op = Compose([self.test_cfg.voxelize])
                post_transform = Compose(self.test_cfg.post_transform)
                fragment_list = []
                data_part_list = voxelize_op(copy.deepcopy(data_dict))
                if isinstance(data_part_list, list):
                    for data_part in data_part_list: fragment_list.append(post_transform(data_part))
                else: fragment_list.append(post_transform(data_part_list))
                data_dict["fragment_list"] = fragment_list
            else:
                # If no test_cfg, use a clean copy of the data_dict as the only fragment
                fragment = data_dict.copy()
                fragment.pop("fragment_list", None)
                fragment["segment"] = np.zeros(len(fragment["coord"]), dtype=np.int32)
                fragment["index"] = torch.arange(len(fragment["coord"]))
                data_dict["fragment_list"] = [fragment]
            return data_dict
            
    test_cfg = getattr(cfg.data.test, "test_cfg", None)
    test_transform = TestWrapper(transform, test_cfg)
    
    las_files = glob.glob(os.path.join(args.folder, "*.las")) + glob.glob(os.path.join(args.folder, "*.laz"))
    for las_file in las_files:
        predict_las(las_file, model, test_transform, cfg, args)
        
if __name__ == "__main__":
    main()
