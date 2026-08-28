"""
Tests for ml/preprocessing.py.
Covers D8 flow accumulation on synthetic DEMs, LineString densification,
nearest drain distance, coordinate conversion properties, and feature matrix formatting.
"""

import numpy as np
from scipy.spatial import cKDTree

from ml.preprocessing import (
    compute_flow_accumulation,
    densify_linestring_segment_points,
    FEATURE_NAMES,
)


def test_d8_flow_accumulation_1d_slope():
    """1D monotone slope: row 2 of a 5x5 DEM decreasing from 50 to 10."""
    dem = np.array([[50, 40, 30, 20, 10]] * 5, dtype=np.float32)
    acc, area = compute_flow_accumulation(dem, pixel_size_m=10.0)
    expected_acc_row = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.allclose(acc[2, :], expected_acc_row)
    assert np.allclose(area[2, :], np.array(expected_acc_row) * 100.0)


def test_d8_flow_accumulation_3x3_central_sink():
    """3x3 DEM where center is strictly lower than all surrounding 8 neighbors."""
    dem = np.array([
        [10.0, 10.0, 10.0],
        [10.0,  2.0, 10.0],
        [10.0, 10.0, 10.0]
    ], dtype=np.float32)
    acc, area = compute_flow_accumulation(dem, pixel_size_m=10.0)
    # Center cell accumulates flow from all 8 surrounding cells + itself = 9
    assert acc[1, 1] == 9.0


test_dem_flat = np.ones((5, 5), dtype=np.float32) * 10.0

def test_d8_flow_accumulation_flat_dem():
    """Flat DEM: no cell has a strictly lower neighbor, so all accumulation counts equal 1."""
    acc, area = compute_flow_accumulation(test_dem_flat, pixel_size_m=10.0)
    assert np.all(acc == 1.0)


def test_densify_linestring_segment_points():
    """Verify segment densification interpolates points at ~10m intervals."""
    p1 = [80.000, 13.000]
    p2 = [80.000, 13.001]
    dense = densify_linestring_segment_points([p1, p2], step_m=10.0)
    assert len(dense) >= 10
    assert dense[0] == p1
    assert dense[-1] == p2


test_segment_p1 = [80.0, 13.0]
test_segment_p2 = [80.0, 13.001]

def test_drain_distance_synthetic_geometry():
    """Test point-to-densified-segment distance accuracy using cKDTree."""
    dense_pts = densify_linestring_segment_points([test_segment_p1, test_segment_p2], step_m=10.0)
    
    # Convert to metric local coordinates
    ref_lon, ref_lat = 80.0, 13.0
    cos_lat = np.cos(np.radians(13.0))
    pts_m = np.array([
        [(pt[0] - ref_lon) * 111000.0 * cos_lat, (ref_lat - pt[1]) * 111000.0]
        for pt in dense_pts
    ])
    tree = cKDTree(pts_m)

    # Query a point 20m orthogonally offset from the midpoint of the line
    mid_lat = 13.0005
    offset_lon = 80.0 + (20.0 / (111000.0 * cos_lat))
    query_m = np.array([(offset_lon - ref_lon) * 111000.0 * cos_lat, (ref_lat - mid_lat) * 111000.0])
    
    dist, _ = tree.query(query_m)
    # Distance should be ~20m with bounded error <= 5m
    assert abs(dist - 20.0) <= 5.0


def test_feature_names_order():
    """Verify standard 7-feature order contract."""
    expected = ["elevation", "slope", "tpi", "dist_to_drain", "q_cap", "q_demand", "r_overload"]
    assert FEATURE_NAMES == expected
