"""
Tests for ml/predict.py.
Covers new rainfall feature recomputation, intensity validation, batch prediction array shapes,
and GeoJSON property contract compliance.
"""

import os
import json
import numpy as np
from ml.flood_model import ChennaiFloodModel
from ml.predict import (
    compute_features_for_rainfall,
    run_batch_inference,
    export_predicted_flood_geojson,
    run_stage5_validations
)
from ml.pipeline_contract import validate_prediction_fields, CANONICAL_FEATURE_NAMES


def assert_raises_value_error(fn, *args, **kwargs):
    """Helper to verify that a function call raises ValueError."""
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    except Exception as e:
        raise AssertionError(f"Expected ValueError, but raised {type(e).__name__}: {e}")
    raise AssertionError("Expected ValueError, but no exception was raised.")


def test_rainfall_intensity_validation():
    """1. Test that negative or non-finite rainfall intensity raises ValueError."""
    X_dummy = np.ones((10, 7), dtype=np.float32)
    assert_raises_value_error(compute_features_for_rainfall, X_dummy, rainfall_intensity_mm_hr=-10.0)
    assert_raises_value_error(compute_features_for_rainfall, X_dummy, rainfall_intensity_mm_hr=float("nan"))


def test_changing_rainfall_changes_q_demand_and_r_overload():
    """2-4. Test that changing rainfall intensity recomputes q_demand and r_overload monotonically."""
    X_base = np.array([
        [10.0, 1.0, 0.0, 50.0, 0.5, 0.2, 0.4] # baseline 80 mm/hr -> q_demand=0.2, r_overload=0.4
    ], dtype=np.float32)

    X_50, q_50, r_50 = compute_features_for_rainfall(X_base, rainfall_intensity_mm_hr=50.0, baseline_rainfall_mm_hr=80.0)
    X_120, q_120, r_120 = compute_features_for_rainfall(X_base, rainfall_intensity_mm_hr=120.0, baseline_rainfall_mm_hr=80.0)

    # 50 mm/hr -> demand = 0.2 * (50/80) = 0.125
    assert np.isclose(q_50[0], 0.125)
    # 120 mm/hr -> demand = 0.2 * (120/80) = 0.3
    assert np.isclose(q_120[0], 0.300)

    assert q_120[0] > q_50[0]
    assert r_120[0] > r_50[0]

    # Verify 7-column feature ordering preserved
    assert X_50.shape == (1, 7)
    assert X_120.shape == (1, 7)


def test_batch_inference_output_shapes_and_probability_range():
    """5-7. Test new rainfall prediction returns one prediction per cell in range [0, 1] with valid risk classes."""
    np.random.seed(42)
    X = np.random.randn(50, 7).astype(np.float32)
    y = np.random.choice([0, 1], size=50).astype(np.uint8)

    model = ChennaiFloodModel(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    raw, adj, classes = run_batch_inference(model, X)
    assert raw.shape == (50,)
    assert adj.shape == (50,)
    assert classes.shape == (50,)
    assert (adj >= 0.0).all() and (adj <= 1.0).all()
    assert set(np.unique(classes)).issubset({"LOW", "MODERATE", "HIGH"})


def test_geojson_exact_contract_properties():
    """8-15. Test GeoJSON export contains required property fields, closed rings, correct counts, and CRS."""
    np.random.seed(42)
    num_cells = 12 # 3 rows x 4 cols
    X = np.random.randn(num_cells, 7).astype(np.float32)
    X[:, 4] = 0.5 # q_cap
    X[:, 5] = 0.2 # q_demand
    X[:, 6] = 0.4 # r_overload

    raw = np.full(num_cells, 0.4, dtype=np.float32)
    adj = np.full(num_cells, 0.45, dtype=np.float32)
    risk_cls = np.full(num_cells, "MODERATE", dtype=object)

    metadata = {
        "height": 3,
        "width": 4,
        "bounds": {"left": 80.0, "bottom": 13.0, "right": 80.04, "top": 13.03}
    }

    out_file = "data/processed/test_new_rainfall_flood.geojson"
    export_predicted_flood_geojson(
        X=X,
        raw_scores=raw,
        adjusted_scores=adj,
        risk_classes=risk_cls,
        metadata=metadata,
        rainfall_intensity_mm_hr=120.0,
        output_path=out_file
    )

    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    assert "CRS84" in data.get("crs", {}).get("properties", {}).get("name", "")
    assert len(data["features"]) == num_cells

    for feat in data["features"]:
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Polygon"
        ring = feat["geometry"]["coordinates"][0]
        assert len(ring) == 5
        assert ring[0] == ring[-1] # Closed ring

        props = feat["properties"]
        # Task 4 Exact Required Property Contracts
        assert "risk_class" in props
        assert "risk_score" in props
        assert "rainfall_mm_hr" in props
        assert props["rainfall_mm_hr"] == 120.0

        # Physical Expose Fields
        assert "runoff_m3_s" in props
        assert "drainage_capacity_m3_s" in props
        assert "overload_ratio" in props

        # Verify pipeline contract helper
        validate_prediction_fields(props)

    if os.path.exists(out_file):
        os.remove(out_file)
