from .defaults import (
    DefaultDataset,
    DefaultImagePointDataset,
    DefaultMultiViewImagePointDataset,
    ConcatDataset,
)
from .builder import build_dataset
from .utils import point_collate_fn, collate_fn

def _safe_import(module_name, symbols):
    import importlib
    try:
        mod = importlib.import_module(f".{module_name}", package=__name__)
        for s in symbols:
            globals()[s] = getattr(mod, s, None)
    except Exception:
        pass

# indoor scene
_safe_import("s3dis", ["S3DISDataset"])
_safe_import("scannet", ["ScanNetDataset", "ScanNet200Dataset"])
_safe_import("scannetpp", ["ScanNetPPDataset"])
_safe_import("scannet_pair", ["ScanNetPairDataset"])
_safe_import("hm3d", ["HM3DDataset"])
_safe_import("structure3d", ["Structured3DDataset"])
_safe_import("aeo", ["AEODataset"])

# outdoor scene
_safe_import("semantic_kitti", ["SemanticKITTIDataset", "SemanticKITTIImagePointDataset"])
_safe_import("nuscenes", ["NuScenesDataset", "NuScenesImagePointDataset"])
_safe_import("waymo", ["WaymoDataset"])
_safe_import("hk", ["HKDataset"])

# object
_safe_import("modelnet", ["ModelNetDataset"])
_safe_import("shapenet_part", ["ShapeNetPartDataset"])
_safe_import("cap3d", ["Cap3DDataset", "Cap3DImagePointDataset"])
_safe_import("scanobjectnn", ["ScanObjectNNDataset", "ScanObjectNNHardestDataset", "ScanObjectNNRawDataset"])
_safe_import("partnet", ["PartNetDataDataset"])
_safe_import("partnete", ["PartNetEDataset"])

# dataloader
_safe_import("dataloader", ["MultiDatasetDataloader"])
