"""
Unit tests & benchmarks for the Multi-Platform Adaptive Outlier Classifier
(`adaptive_outlier.py`) and SQLite undo/redo integration.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import types

REPO_ROOT = Path(__file__).resolve().parent.parent
if "lidar_workbench" not in sys.modules:
    _pkg = types.ModuleType("lidar_workbench")
    _pkg.__path__ = [str(REPO_ROOT)]
    _pkg.__file__ = str(REPO_ROOT / "__init__.py")
    sys.modules["lidar_workbench"] = _pkg

from lidar_workbench.adaptive_outlier import (  # noqa: E402
    ASPRS_HIGH_NOISE,
    ASPRS_LOW_NOISE,
    ASPRS_WATER,
    OutlierParams,
    apply_outlier_classification_with_undo,
    classify_outliers_adaptive,
    evaluate_point_cloud,
    redo_outlier_batch,
    undo_outlier_batch,
)
from lidar_workbench.database import Database  # noqa: E402


def _metrics(pred_noise: np.ndarray, true_noise: np.ndarray):
    tp = int(np.sum(pred_noise & true_noise))
    fp = int(np.sum(pred_noise & ~true_noise))
    fn = int(np.sum(~pred_noise & true_noise))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return prec, rec, tp, fp, fn


def test_fixed_wing_synthetic():
    rng = np.random.default_rng(42)
    # ~20 pts/m2 over 40x40m = 32,000 terrain points with wide scan angle
    n_ground = 32_000
    gx = rng.uniform(0.0, 40.0, n_ground)
    gy = rng.uniform(0.0, 40.0, n_ground)
    gz = 100.0 + 0.15 * gx + 2.0 * np.sin(gy / 6.0) + rng.normal(0, 0.03, n_ground)
    sa = rng.uniform(-28.0, 28.0, n_ground)

    # Inject 200 sky flyers & 200 subterranean pits
    n_out = 400
    ox = rng.uniform(2.0, 38.0, n_out)
    oy = rng.uniform(2.0, 38.0, n_out)
    oz = np.concatenate([
        100.0 + 0.15 * ox[:200] + 45.0 + rng.uniform(0, 20, 200),
        100.0 + 0.15 * ox[200:] - 25.0 - rng.uniform(0, 15, 200),
    ])
    osa = rng.uniform(-28.0, 28.0, n_out)

    data = {
        "x": np.concatenate([gx, ox]),
        "y": np.concatenate([gy, oy]),
        "z": np.concatenate([gz, oz]),
        "classification": np.ones(n_ground + n_out, dtype=np.uint8) * 2,
        "scan_angle": np.concatenate([sa, osa]),
        "point_source_id": np.ones(n_ground + n_out, dtype=np.uint16),
    }
    true_noise = np.zeros(n_ground + n_out, dtype=bool)
    true_noise[n_ground:] = True

    rep = evaluate_point_cloud(data)
    assert rep.estimated_platform == "fixed_wing"
    assert rep.scan_angle_spread_deg > 40.0

    res = classify_outliers_adaptive(data, OutlierParams.auto(rep), report=rep)
    pred_noise = res.class_7_mask | res.class_18_mask
    prec, rec, *_ = _metrics(pred_noise, true_noise)
    assert prec >= 0.90, f"Fixed-wing precision too low: {prec:.3f}"
    assert rec >= 0.95, f"Fixed-wing recall too low: {rec:.3f}"


def test_helicopter_multistrip_ripples():
    rng = np.random.default_rng(7)
    # ~100 pts/m2 over 20x20m = 40,000 points across 2 strips with 0.25m ripple
    n_strip = 20_000
    x1, y1 = rng.uniform(0, 20, n_strip), rng.uniform(0, 20, n_strip)
    z1 = 50.0 + 0.20 * np.sin(x1 * 0.8) + rng.normal(0, 0.02, n_strip)
    x2, y2 = rng.uniform(0, 20, n_strip), rng.uniform(0, 20, n_strip)
    z2 = 50.0 + 0.20 * np.cos(y2 * 0.8) + rng.normal(0, 0.02, n_strip)

    # 150 high flyers + 150 low pits
    ox = rng.uniform(1, 19, 300)
    oy = rng.uniform(1, 19, 300)
    oz = np.concatenate([
        50.0 + 25.0 + rng.uniform(0, 10, 150),
        50.0 - 15.0 - rng.uniform(0, 10, 150),
    ])

    n_clean = 2 * n_strip
    data = {
        "x": np.concatenate([x1, x2, ox]),
        "y": np.concatenate([y1, y2, oy]),
        "z": np.concatenate([z1, z2, oz]),
        "point_source_id": np.concatenate([
            np.ones(n_strip, dtype=np.uint16),
            np.ones(n_strip, dtype=np.uint16) * 2,
            np.ones(300, dtype=np.uint16),
        ]),
        "scan_angle": rng.uniform(-15.0, 15.0, n_clean + 300),
    }
    true_noise = np.zeros(n_clean + 300, dtype=bool)
    true_noise[n_clean:] = True

    rep = evaluate_point_cloud(data)
    assert rep.estimated_platform == "helicopter"
    assert rep.flightline_count == 2

    res = classify_outliers_adaptive(data, OutlierParams.auto(rep), report=rep)
    pred = res.class_7_mask | res.class_18_mask
    prec, rec, *_ = _metrics(pred, true_noise)
    assert prec >= 0.90, f"Helicopter precision too low: {prec:.3f}"
    assert rec >= 0.95, f"Helicopter recall too low: {rec:.3f}"


def test_drone_swath_edge_and_clumped_outliers():
    rng = np.random.default_rng(11)
    # ~300 pts/m2 over 10x10m = 30,000 points
    n_clean = 30_000
    gx = rng.uniform(0, 10, n_clean)
    gy = rng.uniform(0, 10, n_clean)
    gz = 10.0 + 0.05 * gx + rng.normal(0, 0.015, n_clean)

    # Clumped sky noise cluster (12 points) + isolated pits
    cx = 5.0 + rng.normal(0, 0.4, 25)
    cy = 5.0 + rng.normal(0, 0.4, 25)
    cz = 35.0 + rng.normal(0, 0.4, 25)
    px = rng.uniform(1, 9, 100)
    py = rng.uniform(1, 9, 100)
    pz = 10.0 - 8.0 - rng.uniform(0, 5, 100)

    ox = np.concatenate([cx, px])
    oy = np.concatenate([cy, py])
    oz = np.concatenate([cz, pz])

    data = {
        "x": np.concatenate([gx, ox]),
        "y": np.concatenate([gy, oy]),
        "z": np.concatenate([gz, oz]),
        "scan_angle": rng.uniform(-12.0, 12.0, n_clean + len(ox)),
    }
    true_noise = np.zeros(n_clean + len(ox), dtype=bool)
    true_noise[n_clean:] = True

    rep = evaluate_point_cloud(data)
    assert rep.estimated_platform == "drone"

    res = classify_outliers_adaptive(data, OutlierParams.auto(rep), report=rep)
    pred = res.class_7_mask | res.class_18_mask
    prec, rec, *_ = _metrics(pred, true_noise)
    assert prec >= 0.90, f"Drone precision too low: {prec:.3f}"
    assert rec >= 0.85, f"Drone recall too low: {rec:.3f}"


def test_canopy_multireturn_and_powerlines_preserved():
    rng = np.random.default_rng(99)
    # Ground (5,000 pts)
    gx = rng.uniform(0, 20, 5000)
    gy = rng.uniform(0, 20, 5000)
    gz = np.zeros(5000) + rng.normal(0, 0.02, 5000)

    # Multi-return forest canopy (2,000 pts, return_number=2, num_returns=3, low reflectance -25 dB)
    fx = rng.uniform(2, 10, 2000)
    fy = rng.uniform(2, 10, 2000)
    fz = rng.uniform(3.0, 14.0, 2000)

    # Thin powerline (1,000 pts along catenary at z=12m)
    px = np.linspace(0.5, 19.5, 1000)
    py = np.full(1000, 15.0) + rng.normal(0, 0.01, 1000)
    pz = 12.0 + 0.01 * (px - 10.0) ** 2 + rng.normal(0, 0.01, 1000)

    # Actual isolated sky/pit outliers (100 pts)
    ox = rng.uniform(1, 19, 100)
    oy = rng.uniform(1, 19, 100)
    oz = np.concatenate([np.full(50, 45.0), np.full(50, -15.0)])

    n_g, n_f, n_p, n_o = 5000, 2000, 1000, 100
    n_tot = n_g + n_f + n_p + n_o

    ret_num = np.ones(n_tot, dtype=np.uint8)
    num_ret = np.ones(n_tot, dtype=np.uint8)
    ret_num[n_g : n_g + n_f] = 2
    num_ret[n_g : n_g + n_f] = 3

    refl = np.full(n_tot, -5.0, dtype=np.float32)
    refl[n_g : n_g + n_f] = -25.0  # Split-energy canopy return
    refl[-n_o:] = -28.0

    data = {
        "x": np.concatenate([gx, fx, px, ox]),
        "y": np.concatenate([gy, fy, py, oy]),
        "z": np.concatenate([gz, fz, pz, oz]),
        "return_number": ret_num,
        "num_returns": num_ret,
        "extra_dims": {"reflectance": refl},
    }
    res = classify_outliers_adaptive(data, OutlierParams.auto(evaluate_point_cloud(data)))
    pred = res.class_7_mask | res.class_18_mask

    # Canopy & powerline preservation checks
    canopy_shred_rate = float(np.mean(pred[n_g : n_g + n_f]))
    wire_shred_rate = float(np.mean(pred[n_g + n_f : n_g + n_f + n_p]))
    outlier_recall = float(np.mean(pred[-n_o:]))

    assert canopy_shred_rate < 0.05, f"Canopy shredded: {canopy_shred_rate:.3f}"
    assert wire_shred_rate < 0.05, f"Powerline shredded: {wire_shred_rate:.3f}"
    assert outlier_recall >= 0.90, f"Outlier recall too low: {outlier_recall:.3f}"


def test_topo_bathy_subpass():
    rng = np.random.default_rng(123)
    # Flat water surface at z=10m (5,000 pts, Class 9) + seabed at z=6m (4,000 pts)
    wx, wy = rng.uniform(0, 20, 5000), rng.uniform(0, 20, 5000)
    wz = 10.0 + rng.normal(0, 0.02, 5000)
    bx, by = rng.uniform(0, 20, 4000), rng.uniform(0, 20, 4000)
    bz = 6.0 + rng.normal(0, 0.05, 4000)

    # Water spray (30 pts above water) + sparse volume backscatter (30 pts between 6.5 and 9.5)
    sx, sy = rng.uniform(2, 18, 30), rng.uniform(2, 18, 30)
    sz = 11.2 + rng.uniform(0, 1.0, 30)
    vx, vy = rng.uniform(2, 18, 30), rng.uniform(2, 18, 30)
    vz = rng.uniform(7.2, 9.2, 30)

    cls = np.concatenate([
        np.full(5000, ASPRS_WATER, dtype=np.uint8),
        np.full(4000, 40, dtype=np.uint8),
        np.full(30, ASPRS_WATER, dtype=np.uint8),
        np.full(30, 40, dtype=np.uint8),
    ])
    data = {
        "x": np.concatenate([wx, bx, sx, vx]),
        "y": np.concatenate([wy, by, sy, vy]),
        "z": np.concatenate([wz, bz, sz, vz]),
        "classification": cls,
    }
    rep = evaluate_point_cloud(data)
    assert rep.has_bathymetry is True

    res = classify_outliers_adaptive(data, OutlierParams.auto(rep), report=rep)
    pred = res.class_7_mask | res.class_18_mask
    assert float(np.mean(pred[-60:])) >= 0.70


def test_sqlite_undo_redo_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        db.initialize()
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO tiles (id, filename, status) VALUES ('tile_001', 'tile_001.las', 'IMPORTED')"
            )
            conn.commit()

        old_cls = np.array([2, 2, 2, 1, 1], dtype=np.uint8)
        data = {
            "x": np.array([0.0, 0.1, 0.2, 0.1, 50.0]),
            "y": np.array([0.0, 0.0, 0.1, 0.1, 50.0]),
            "z": np.array([0.0, 0.0, 0.0, 0.0, 100.0]),
            "classification": old_cls.copy(),
        }
        res = classify_outliers_adaptive(data, OutlierParams.drone())
        # Force at least one outlier if tiny sample
        if not np.any(res.class_7_mask | res.class_18_mask):
            res.class_18_mask[4] = True
            res.updated_classification[4] = ASPRS_HIGH_NOISE

        cmd_id = apply_outlier_classification_with_undo(db, "tile_001", old_cls, res)
        assert cmd_id is not None

        reverted, ok_undo = undo_outlier_batch(db, "tile_001", res.updated_classification, command_id=cmd_id)
        assert ok_undo is True
        assert np.array_equal(reverted, old_cls)

        reapplied, ok_redo = redo_outlier_batch(db, "tile_001", reverted, command_id=cmd_id)
        assert ok_redo is True
        assert np.array_equal(reapplied, res.updated_classification)


def test_evaluate_5m_points_under_100ms():
    rng = np.random.default_rng(1)
    n = 5_000_000
    data = {
        "x": rng.uniform(0.0, 500.0, n),
        "y": rng.uniform(0.0, 500.0, n),
        "z": rng.uniform(100.0, 120.0, n),
        "point_source_id": np.ones(n, dtype=np.uint16),
        "scan_angle": rng.uniform(-20.0, 20.0, n),
    }
    # Warm-up JIT/numpy caches once
    _ = evaluate_point_cloud({k: v[:1000] for k, v in data.items()})
    rep = evaluate_point_cloud(data)
    assert rep.point_count == n
    assert rep.eval_time_ms < 100.0, f"5M-point evaluation took {rep.eval_time_ms:.1f} ms (>= 100 ms)"
