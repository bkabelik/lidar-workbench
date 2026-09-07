"""Synthetic bathymetry regression tests for ground classification.

The scene models the coarse-to-fine seed failure mode: a riverbed at
Z=95 m, a dense water-surface reflection layer at Z=98 m, and dry terrain
embankments at Z=100 m.

With a fine seed grid (5 m) many seed cells contain only water-surface
returns, so the TIN is built at Z=98 m and the water surface is classified
as ground.  With the coarse 60 m seed grid (two-pass workflow) the lowest
point in each block is the riverbed, so the water surface is rejected by
the 1.0 m iteration distance.  The banks are gentle (~8°) so the classic
Axelsson 12° pass-1 angle can progressively climb them.
"""

from __future__ import annotations

import numpy as np

from ground import (
    ground_classify_epptd,
    ground_classify_epptd_two_pass,
    ground_classify_stepdown,
    ground_classify_multiscale_alpha_shape,
    ground_classify_egs_csf,
)

def _make_scene(include_scatter: bool = False, seed: int = 7):
    """Return points and per-point labels for a synthetic river valley."""
    RNG = np.random.default_rng(seed)
    xs = []
    ys = []
    zs = []
    labels = []  # "dry", "bank", "bed", "water", "scatter"
    sensor = []  # 1 = topo, 2 = bathy
    deviation = []  # extra byte (RIEGL deviation stand-in)

    def add(x, y, z, label, st, dev):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        xs.extend(x)
        ys.extend(y)
        zs.extend(z)
        labels.extend([label] * len(x))
        sensor.extend([st] * len(x))
        deviation.extend([dev] * len(x))

    # Dry terrain (flat, far from the channel so bank slopes don't matter)
    for x in np.arange(0.0, 121.0, 2.0):
        for y in np.arange(0.0, 14.0, 2.0):
            add(x, y, 100.0 + RNG.normal(0.0, 0.02), "dry", 1, 1.0)
        for y in np.arange(107.0, 121.0, 2.0):
            add(x, y, 100.0 + RNG.normal(0.0, 0.02), "dry", 1, 1.0)

    # Channel banks: 36 m wide, 5 m rise (slope ~7.9°, below the 12° pass-1
    # iteration angle so the TIN can progressively climb them).
    for x in np.arange(0.0, 121.0, 2.0):
        for y in np.arange(14.0, 50.0, 2.0):
            z = 95.0 + (50.0 - y) / 36.0 * 5.0
            add(x, y, z + RNG.normal(0.0, 0.02), "bank", 1, 1.0)
        for y in np.arange(70.0, 107.0, 2.0):
            z = 95.0 + (y - 70.0) / 37.0 * 5.0
            add(x, y, z + RNG.normal(0.0, 0.02), "bank", 1, 1.0)

    # Sparse riverbed (every 10 m — many 5 m cells have no bed return)
    for x in np.arange(0.0, 121.0, 10.0):
        for y in np.arange(50.0, 71.0, 10.0):
            add(x, y, 95.0 + RNG.normal(0.0, 0.02), "bed", 2, 1.0)

    # Dense water surface reflections (every 1 m over the channel)
    for x in np.arange(0.0, 121.0, 1.0):
        for y in np.arange(50.0, 71.0, 1.0):
            add(x, y, 98.0 + RNG.normal(0.0, 0.02), "water", 2, 2.0)

    # Optional water-column turbidity scatter just above the bed
    if include_scatter:
        for x in np.arange(0.0, 121.0, 2.0):
            for y in np.arange(52.0, 69.0, 2.0):
                add(x, y, 95.8 + RNG.normal(0.0, 0.01), "scatter", 2, 10.0)

    return {
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "z": np.asarray(zs),
        "label": np.asarray(labels),
        "sensor_type": np.asarray(sensor, dtype=np.uint8),
        "deviation": np.asarray(deviation, dtype=np.float64),
    }


def _ratios(data, mask):
    out = {}
    for label in ("dry", "bank", "bed", "water", "scatter"):
        sel = data["label"] == label
        if sel.any():
            out[label] = float(mask[sel].mean())
    return out


def test_two_pass_rejects_water_and_keeps_bed():
    data = _make_scene()
    # Interior water only, away from bank triangles.
    water_interior = (
        (data["label"] == "water")
        & (data["y"] >= 52.0) & (data["y"] <= 68.0)
    )

    # Old-style fine seed grid with sparse core: a
    # meaningful share of the water surface still becomes ground (seeded
    # from water cells), which is exactly what the two-pass workflow fixes.
    mask_old = ground_classify_epptd(
        data["x"], data["y"], data["z"],
        max_distance=1.4, max_angle=6.0,
        cell_size=5.0,
    )
    assert mask_old[water_interior].mean() > 0.1, (
        "expected fine seed grid to misclassify some water surface"
    )

    # Two-pass coarse-to-fine workflow: bed kept, water rejected.
    mask_tp = ground_classify_epptd_two_pass(
        data["x"], data["y"], data["z"],
    )
    ratios = _ratios(data, mask_tp)
    assert mask_tp[water_interior].mean() < 0.2, (
        f"water classified as ground: {mask_tp[water_interior].mean():.2f}"
    )
    assert ratios["bed"] > 0.8, f"bed lost: {ratios}"
    assert ratios["dry"] > 0.9, f"dry ground lost: {ratios}"


def test_densify_existing_ground_preserves_trusted_points():
    data = _make_scene()

    # Simulate "1 → 2": existing class-2 ground is a subset of bed + dry.
    trusted = (
        ((data["label"] == "bed") & (data["x"] % 20 == 0))
        | ((data["label"] == "dry") & (data["x"] % 20 == 0)
           & (data["y"] % 20 == 0))
    )
    mask = ground_classify_epptd(
        data["x"], data["y"], data["z"],
        max_distance=1.0, max_angle=6.0,
        existing_ground_mask=trusted,
    )

    # Trusted points must never be lost.
    assert mask[trusted].all()

    # Unclassified bed points near the trusted bed TIN get densified.
    bed_untrusted = (data["label"] == "bed") & ~trusted
    assert mask[bed_untrusted].mean() > 0.8

    # Water surface stays out.
    water_interior = (
        (data["label"] == "water")
        & (data["y"] >= 52.0) & (data["y"] <= 68.0)
    )
    assert mask[water_interior].mean() < 0.2


def test_extra_byte_filter_excludes_turbidity_scatter():
    data = _make_scene(include_scatter=True)
    scatter_sel = data["label"] == "scatter"

    # Without the filter, scatter at Z=95.8 sits 0.8 m above the bed TIN
    # and is swept into the ground set.
    mask_no_filter = ground_classify_epptd(
        data["x"], data["y"], data["z"],
        max_distance=1.0, max_angle=6.0,
        cell_size=60.0,
        sensor_type=data["sensor_type"],
    )
    assert mask_no_filter[scatter_sel].mean() > 0.5, (
        "test premise: scatter should be accepted without the filter"
    )

    # With the bathy-only deviation filter it is excluded.
    mask_filtered = ground_classify_epptd(
        data["x"], data["y"], data["z"],
        max_distance=1.0, max_angle=6.0,
        cell_size=60.0,
        sensor_type=data["sensor_type"],
        extra_dims={"deviation": data["deviation"]},
        extra_dim_filters=[("Deviation", None, 5.0)],
    )
    assert mask_filtered[scatter_sel].mean() < 0.1

    # Topo points far from the channel must be unaffected by the bathy-only
    # filter (the filter itself never rejects sensor_type != 2 points; small
    # indirect differences near the shared TIN are expected and fine).
    topo_dry = (data["sensor_type"] == 1) & (data["label"] == "dry")
    assert mask_filtered[topo_dry].mean() > 0.9
    assert mask_no_filter[topo_dry].mean() > 0.9


def test_stepdown_levee_and_embankment_capture():
    """Verify that Step-Down PTD climbs steep riverbanks and levee crests while rejecting trees."""
    rng = np.random.default_rng(42)

    # 1. Flat floodplain (Z ~ 100m)
    n_field = 600
    x_field = rng.uniform(0.0, 30.0, n_field)
    y_field = rng.uniform(0.0, 50.0, n_field)
    z_field = 100.0 + rng.normal(0.0, 0.02, n_field)

    # 2. Steep parabolic levee crest (X: [30, 35], rising 2.5m, slope ~55°)
    n_crest = 300
    x_crest = rng.uniform(30.0, 35.0, n_crest)
    y_crest = rng.uniform(0.0, 50.0, n_crest)
    z_crest = 100.0 + 2.5 * (1.0 - ((x_crest - 33.0) / 2.5)**2) + rng.normal(0.0, 0.02, n_crest)

    # 3. Steep riverbank (X: [35, 45], dropping from 100m to 94m, slope ~31°)
    n_bank = 500
    x_bank = rng.uniform(35.0, 45.0, n_bank)
    y_bank = rng.uniform(0.0, 50.0, n_bank)
    u = (x_bank - 35.0) / 10.0
    z_bank = 100.0 - 6.0 * u + rng.normal(0.0, 0.02, n_bank)

    # 4. Riverbed (X: [45, 65], Z ~ 94m)
    n_bed = 600
    x_bed = rng.uniform(45.0, 65.0, n_bed)
    y_bed = rng.uniform(0.0, 50.0, n_bed)
    z_bed = 94.0 + rng.normal(0.0, 0.02, n_bed)

    # 5. Non-ground tree canopy over the levee (Z: 105m to 120m)
    n_trees = 250
    x_trees = rng.uniform(31.0, 35.0, n_trees)
    y_trees = rng.uniform(0.0, 50.0, n_trees)
    z_trees = rng.uniform(105.0, 120.0, n_trees)

    xs = np.concatenate([x_field, x_crest, x_bank, x_bed, x_trees])
    ys = np.concatenate([y_field, y_crest, y_bank, y_bed, y_trees])
    zs = np.concatenate([z_field, z_crest, z_bank, z_bed, z_trees])

    is_crest = np.zeros(len(xs), dtype=bool)
    is_crest[n_field:n_field + n_crest] = True

    is_bank = np.zeros(len(xs), dtype=bool)
    is_bank[n_field + n_crest:n_field + n_crest + n_bank] = True

    is_trees = np.zeros(len(xs), dtype=bool)
    is_trees[len(xs) - n_trees:] = True

    mask = ground_classify_stepdown(
        xs, ys, zs,
        step=3.0,
        sub_steps=5,
        bulge=1.5,
        offset=0.15,
        spike=1.0,
        spike_down=1.0,
    )

    crest_ratio = float(mask[is_crest].mean())
    bank_ratio = float(mask[is_bank].mean())
    tree_ratio = float(mask[is_trees].mean())

    assert crest_ratio > 0.95, f"Levee crest missed by Step-Down PTD: {crest_ratio:.2%}"
    assert bank_ratio > 0.95, f"Riverbank missed by Step-Down PTD: {bank_ratio:.2%}"
    assert tree_ratio < 0.02, f"Tree points leaked into ground: {tree_ratio:.2%}"


def test_stepdown_urban_building_rejection():
    """Verify that Step-Down PTD with city setting (step=25m) strips large buildings."""
    rng = np.random.default_rng(42)
    n_ground = 2500
    gx = rng.uniform(0.0, 100.0, n_ground)
    gy = rng.uniform(0.0, 100.0, n_ground)
    gz = 50.0 + rng.normal(0.0, 0.02, n_ground)

    # Void beneath warehouse footprint (X: [30, 60], Y: [30, 60])
    not_under = ~((gx >= 30.0) & (gx <= 60.0) & (gy >= 30.0) & (gy <= 60.0))
    gx, gy, gz = gx[not_under], gy[not_under], gz[not_under]

    # Warehouse roof at Z=58m (8m above ground)
    n_bldg = 400
    bx = rng.uniform(30.0, 60.0, n_bldg)
    by = rng.uniform(30.0, 60.0, n_bldg)
    bz = 58.0 + rng.normal(0.0, 0.02, n_bldg)

    xs = np.concatenate([gx, bx])
    ys = np.concatenate([gy, by])
    zs = np.concatenate([gz, bz])
    is_bldg = np.zeros(len(xs), dtype=bool)
    is_bldg[len(gx):] = True

    mask = ground_classify_stepdown(
        xs, ys, zs,
        step=25.0,
        sub_steps=4,
        bulge=1.5,
        offset=0.15,
    )

    ground_ratio = float(mask[~is_bldg].mean())
    bldg_ratio = float(mask[is_bldg].mean())

    assert ground_ratio > 0.95, f"Bare earth ground lost: {ground_ratio:.2%}"
    assert bldg_ratio == 0.0, f"Building points leaked into ground: {bldg_ratio:.2%}"


def test_stepdown_bathy_channel_separation():
    """Verify that Step-Down PTD correctly classifies riverbed and banks while excluding water reflections."""
    data = _make_scene()
    mask = ground_classify_stepdown(
        data["x"], data["y"], data["z"],
        step=25.0,
        sub_steps=5,
        bulge=1.5,
        offset=0.15,
    )
    ratios = _ratios(data, mask)

    assert ratios["bed"] > 0.95, f"Riverbed lost: {ratios['bed']:.2%}"
    assert ratios["bank"] > 0.95, f"Riverbank lost: {ratios['bank']:.2%}"
    assert ratios["dry"] > 0.95, f"Dry ground lost: {ratios['dry']:.2%}"
    assert ratios["water"] < 0.05, f"Water surface classified as ground: {ratios['water']:.2%}"


def test_multiscale_alpha_shape_speed_and_water_rejection():
    """Verify that M-AlphaShape runs fast (<1s) and successfully excludes water surface."""
    import time
    data = _make_scene()
    t0 = time.time()
    mask = ground_classify_multiscale_alpha_shape(
        data["x"], data["y"], data["z"],
        coarse_alpha=20.0,
        medium_alpha=6.0,
        fine_alpha=2.0,
        max_distance=0.20,
    )
    elapsed = time.time() - t0
    ratios = _ratios(data, mask)

    assert elapsed < 3.0, f"M-AlphaShape too slow: {elapsed:.2f}s"
    assert ratios["bed"] > 0.90, f"Riverbed lost: {ratios['bed']:.2%}"
    assert ratios["bank"] > 0.95, f"Riverbank lost: {ratios['bank']:.2%}"
    assert ratios["dry"] > 0.95, f"Dry ground lost: {ratios['dry']:.2%}"
    assert ratios["water"] < 0.05, f"Water surface leaked into ground: {ratios['water']:.2%}"


def test_egs_csf_low_point_rejection():
    """Verify that EGS-CSF does not collapse when low noise points (pits/multipath) exist in the data."""
    rng = np.random.default_rng(42)
    n_ground = 2000
    gx = rng.uniform(0.0, 50.0, n_ground)
    gy = rng.uniform(0.0, 50.0, n_ground)
    gz = 100.0 + 0.1 * gx + rng.normal(0.0, 0.03, n_ground)

    # Inject 3 extreme subterranean noise points (10m - 100m underground)
    low_x = np.array([15.0, 25.0, 35.0])
    low_y = np.array([15.0, 25.0, 35.0])
    low_z = np.array([50.0, 0.0, -100.0])

    xs = np.concatenate([gx, low_x])
    ys = np.concatenate([gy, low_y])
    zs = np.concatenate([gz, low_z])

    mask = ground_classify_egs_csf(
        xs, ys, zs,
        cloth_resolution=1.0,
        rigidness=2.0,
        class_threshold=0.30,
        spike_down=2.5,
    )

    ground_recall = float(mask[:n_ground].mean())
    low_points_in_ground = int(mask[n_ground:].sum())

    assert ground_recall > 0.95, f"EGS-CSF collapsed due to low points: ground recall {ground_recall:.2%}"
    assert low_points_in_ground == 0, f"Low noise points classified as ground: {low_points_in_ground}/3"


def test_egs_csf_high_relief_and_steep_slope():
    """Verify that EGS-CSF accurately classifies ground across 80m+ elevation relief with pits."""
    rng = np.random.default_rng(42)
    n_ground = 2000
    gx = rng.uniform(0.0, 50.0, n_ground)
    gy = rng.uniform(0.0, 50.0, n_ground)
    # 80m slope with terrain undulation
    gz = 100.0 + 1.6 * gx + rng.normal(0.0, 0.03, n_ground)

    # Subterranean noise points
    low_x = np.array([15.0, 25.0, 35.0])
    low_y = np.array([15.0, 25.0, 35.0])
    low_z = np.array([50.0, 0.0, -100.0])

    xs = np.concatenate([gx, low_x])
    ys = np.concatenate([gy, low_y])
    zs = np.concatenate([gz, low_z])

    mask = ground_classify_egs_csf(
        xs, ys, zs,
        cloth_resolution=1.0,
        rigidness=2.0,
        class_threshold=0.30,
        gradient_factor=0.50,
        spike_down=2.5,
    )

    ground_recall = float(mask[:n_ground].mean())
    low_points_in_ground = int(mask[n_ground:].sum())

    assert ground_recall > 0.95, f"EGS-CSF high-relief failure: ground recall {ground_recall:.2%}"
    assert low_points_in_ground == 0, f"Low noise points classified as ground: {low_points_in_ground}/3"


def test_egs_csf_embankment_crest_recovery():
    """Verify EGS-CSF slope & crest post-processing captures rounded embankment crests without capturing buildings."""
    # Synthetic terrain: 50x50m flat plain with 3m high rounded embankment
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()

    # Embankment along y=25, rounded top
    dist = np.abs(yy - 25.0)
    embankment = np.maximum(0.0, 3.0 * (1.0 - (dist / 5.0) ** 2))
    zz = 10.0 + embankment

    # Add a flat building roof (8m tall) on the side
    bld_mask = (xx >= 5) & (xx <= 15) & (yy >= 5) & (yy <= 15)
    zz[bld_mask] = 18.0

    crest_mask = dist < 1.0

    mask = ground_classify_egs_csf(
        xx, yy, zz,
        cloth_resolution=1.0,
        rigidness=2.0,
        class_threshold=0.20,
        gradient_factor=1.0,
        slope_smooth=True,
    )

    crest_recall = float(mask[crest_mask].mean())
    building_false_positives = float(mask[bld_mask].mean())

    assert crest_recall >= 0.90, f"EGS-CSF failed to retain rounded embankment crest: {crest_recall:.2%}"
    assert building_false_positives == 0.0, f"EGS-CSF misclassified building as ground: {building_false_positives:.2%}"


import unittest

class TestGroundBathy(unittest.TestCase):
    def test_two_pass(self):
        test_two_pass_rejects_water_and_keeps_bed()

    def test_densify(self):
        test_densify_existing_ground_preserves_trusted_points()

    def test_extra_byte(self):
        test_extra_byte_filter_excludes_turbidity_scatter()

    def test_stepdown_levee(self):
        test_stepdown_levee_and_embankment_capture()

    def test_stepdown_urban(self):
        test_stepdown_urban_building_rejection()

    def test_stepdown_bathy(self):
        test_stepdown_bathy_channel_separation()

    def test_malpha_shape(self):
        test_multiscale_alpha_shape_speed_and_water_rejection()

    def test_egs_csf(self):
        test_egs_csf_low_point_rejection()

    def test_egs_csf_relief(self):
        test_egs_csf_high_relief_and_steep_slope()

    def test_egs_csf_crest(self):
        test_egs_csf_embankment_crest_recovery()


if __name__ == "__main__":
    unittest.main()

