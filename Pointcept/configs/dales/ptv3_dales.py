_base_ = ["../_base_/default_runtime.py"]

# --- General Config ---
batch_size = 2  # Total BS
epoch = 150
eval_epoch = 150
empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
clip_grad = 0.5

# --- Model Config (PTv3) ---
model = dict(
    type="DefaultSegmentorV2",
    num_classes=8,
    backbone_out_channels=64,
    backbone=dict(
        type="PTv3", # Using the standard PTv3
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# --- Optimization Config ---
optimizer = dict(type="AdamW", lr=0.004, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.004, 0.0004],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0004)]

# --- Dataset Config ---
dataset_type = "DefaultDataset"
data_root = "/home/fractal01/PointceptALS/data/DALESObjects_training_data"

data = dict(
    num_classes=8,
    ignore_index=-1,
    names=["Ground", "Vegetation", "Cars", "Trucks", "Power lines", "Fences", "Poles", "Buildings"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-3.1415926, 3.1415926], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=0.16, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength")),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=0.16, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength")),
        ],
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=0.16, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength")),
        ],
    ),
)
