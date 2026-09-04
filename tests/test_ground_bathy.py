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

from ground import ground_classify_epptd, ground_classify_epptd_two_pass

RNG = np.random.default_rng(7)


def _make_scene(include_scatter: bool = False):
    """Return points and per-point labels for a synthetic river valley."""
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

    # Old-style fine seed grid with the lidR-faithful sparse core: a
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
