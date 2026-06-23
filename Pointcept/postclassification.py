import numpy as np
from scipy.spatial import cKDTree

def fix_ground_building_bleed(points, pred_classes, h_above, 
                               ground_id=0, building_id=7, 
                               min_building_height=1.5, 
                               max_ground_height=0.15):
    """
    Existing logic to restore ground/building based on height above DTM.
    """
    # 1. High-Altitude Ground -> Building (Fixes roofs predicted as ground)
    misclassified_ground = (pred_classes == ground_id) & (h_above > min_building_height)
    if np.any(misclassified_ground):
        print(f"  -> Reclassified {np.sum(misclassified_ground):,} high-altitude 'Ground' points as Building.")
        pred_classes[misclassified_ground] = building_id
        
    # 2. Low-Altitude Building -> Ground (Restores road patches predicted as building)
    road_restore = (pred_classes == building_id) & (h_above < max_ground_height)
    if np.any(road_restore):
        print(f"  -> Restored {np.sum(road_restore):,} low-altitude 'Building' points as Ground.")
        pred_classes[road_restore] = ground_id
        
    return pred_classes


def fix_vegetation_building_confusion(points, pred_classes, h_above, 
                                       veg_id=1, building_id=7, 
                                       k_neighbors=20, 
                                       spatial_radius=2.0):
    """
    New logic to handle vegetation on buildings and buildings in vegetation.
    Uses spatial neighborhood consistency.
    """
    print(f"  -> Refining Vegetation/Building boundaries (k={k_neighbors})...")
    
    # We only care about points that are either Veg or Building
    target_mask = (pred_classes == veg_id) | (pred_classes == building_id)
    if not np.any(target_mask):
        return pred_classes
        
    target_indices = np.where(target_mask)[0]
    target_points = points[target_indices]
    
    # Build tree for spatial lookup
    tree = cKDTree(target_points)
    
    # Query neighbors for all target points
    # Cap k to the number of available target points
    k_actual = min(k_neighbors, len(target_indices))
    dist, indices = tree.query(target_points, k=k_actual)
    
    # Handle single-neighbor case (query returns 1D instead of 2D)
    if k_actual == 1:
        indices = indices[:, np.newaxis]
        
    # Get labels of neighbors (within the target set)
    neighbor_labels = pred_classes[target_indices[indices]]
    
    # Calculate majorities
    veg_counts = np.sum(neighbor_labels == veg_id, axis=1)
    building_counts = np.sum(neighbor_labels == building_id, axis=1)
    
    refined_preds = pred_classes.copy()
    
    # 1. Vegetation on Buildings: 
    # If point is Veg, but it's high (>2m) and surrounded by Buildings (>80%)
    veg_on_building_mask = (pred_classes[target_indices] == veg_id) & \
                           (h_above[target_indices] > 2.0) & \
                           (building_counts > (k_neighbors * 0.8))
    
    if np.any(veg_on_building_mask):
        count = np.sum(veg_on_building_mask)
        print(f"     * Reclassified {count:,} 'Vegetation' points on rooftops as Building.")
        refined_preds[target_indices[veg_on_building_mask]] = building_id
        
    # 2. Buildings in Vegetation:
    # If point is Building, but surrounded by Vegetation (>80%)
    building_in_veg_mask = (pred_classes[target_indices] == building_id) & \
                            (veg_counts > (k_neighbors * 0.8))
                            
    if np.any(building_in_veg_mask):
        count = np.sum(building_in_veg_mask)
        print(f"     * Reclassified {count:,} isolated 'Building' points in trees as Vegetation.")
        refined_preds[target_indices[building_in_veg_mask]] = veg_id
        
    return refined_preds


def refine_classifications(points, pred_classes, h_above, params=None):
    """
    Consolidated entry point for all post-prediction adjustments.
    """
    if params is None:
        params = {}
        
    # Thresholds (defaulting to meters)
    min_b_h = params.get("min_building_height", 1.5)
    max_g_h = params.get("max_ground_height", 0.15)
    
    # Step 1: Ground/Building height consistency
    pred_classes = fix_ground_building_bleed(points, pred_classes, h_above, 
                                             min_building_height=min_b_h, 
                                             max_ground_height=max_g_h)
    
    # Step 2: Vegetation/Building spatial consistency
    pred_classes = fix_vegetation_building_confusion(points, pred_classes, h_above)
    
    return pred_classes
