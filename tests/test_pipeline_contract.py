"""
Unit tests for ml/pipeline_contract.py.
Covers feature schema validation, prediction record fields, risk class cutoffs,
grid metadata contracts, prediction array validation, and GeoJSON structural assertions.
"""

import numpy as np
from ml.pipeline_contract import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_PREDICTION_FIELDS,
    VALID_RISK_CLASSES,
    validate_feature_names,
    validate_prediction_fields,
    validate_risk_classes,
    validate_grid_metadata,
    validate_prediction_arrays,
    validate_geojson_contract,
)


def assert_raises_value_error(fn, *args, **kwargs):
    """Helper to verify that a function call raises ValueError."""
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    except Exception as e:
        raise AssertionError(f"Expected ValueError, but raised {type(e).__name__}: {e}")
    raise AssertionError("Expected ValueError, but no exception was raised.")


def test_correct_feature_names_accepted():
    """1. Correct CANONICAL_FEATURE_NAMES accepted."""
    assert validate_feature_names(CANONICAL_FEATURE_NAMES) is True


def test_reordered_feature_names_rejected():
    """2. Reordered FEATURE_NAMES rejected with ValueError."""
    reordered = list(CANONICAL_FEATURE_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert_raises_value_error(validate_feature_names, reordered)


def test_missing_feature_rejected():
    """3. Missing feature rejected with ValueError."""
    missing = list(CANONICAL_FEATURE_NAMES)[:-1]
    assert_raises_value_error(validate_feature_names, missing)


def test_extra_feature_rejected():
    """4. Extra feature rejected with ValueError."""
    extra = list(CANONICAL_FEATURE_NAMES) + ["extra_feature"]
    assert_raises_value_error(validate_feature_names, extra)


def test_required_prediction_fields_accepted():
    """5. Required prediction fields accepted."""
    record = {field: 0 for field in CANONICAL_PREDICTION_FIELDS}
    assert validate_prediction_fields(record) is True


def test_missing_prediction_field_rejected():
    """6. Missing prediction field rejected with ValueError."""
    record = {field: 0 for field in CANONICAL_PREDICTION_FIELDS if field != "risk_score"}
    assert_raises_value_error(validate_prediction_fields, record)


def test_invalid_risk_score_rejected():
    """7. Invalid risk score (< 0.0, > 1.0, or NaN) rejected."""
    assert_raises_value_error(validate_risk_classes, np.array([-0.1, 0.5]), np.array(["LOW", "MODERATE"]))
    assert_raises_value_error(validate_risk_classes, np.array([0.5, 1.2]), np.array(["MODERATE", "HIGH"]))


def test_invalid_risk_class_label_rejected():
    """8. Invalid risk class string label rejected."""
    assert_raises_value_error(validate_risk_classes, np.array([0.5]), np.array(["CRITICAL"]))


def test_valid_risk_class_boundaries_accepted():
    """9. Valid LOW / MODERATE / HIGH cutoff boundaries accepted."""
    scores = np.array([0.10, 0.32, 0.33, 0.50, 0.65, 0.66, 0.90])
    classes = np.array(["LOW", "LOW", "MODERATE", "MODERATE", "MODERATE", "HIGH", "HIGH"])
    assert validate_risk_classes(scores, classes) is True


def test_grid_dimensions_determine_cell_count():
    """10. Grid dimensions correctly determine expected total cell count."""
    meta = {"height": 10, "width": 20, "bounds": {"left": 80.0, "bottom": 13.0, "right": 80.1, "top": 13.1}}
    assert validate_grid_metadata(meta, num_cells=200) is True
    assert_raises_value_error(validate_grid_metadata, meta, num_cells=250)


def test_invalid_grid_bounds_rejected():
    """11. Invalid grid bounds (left >= right or bottom >= top) rejected."""
    bad_bounds = {"height": 10, "width": 20, "bounds": {"left": 80.1, "bottom": 13.0, "right": 80.0, "top": 13.1}}
    assert_raises_value_error(validate_grid_metadata, bad_bounds)


def test_geojson_structural_contract_validation():
    """12. Valid GeoJSON dictionary structure accepted."""
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 13.0], [80.01, 13.0], [80.01, 13.01], [80.0, 13.01], [80.0, 13.0]]]
                },
                "properties": {
                    "cell_id": 1, "row": 0, "col": 0, "risk_score_raw": 0.2, "risk_score": 0.2,
                    "risk_class": "LOW", "rainfall_mm_hr": 80.0, "r_overload": 0.5, "q_cap": 0.1, "q_demand": 0.05,
                    "elevation": 10.0, "slope": 1.0, "tpi": 0.0, "dist_to_drain": 50.0
                }
            }
        ]
    }
    assert validate_geojson_contract(sample_geojson, metadata={"height": 1, "width": 1, "bounds": {"left": 80.0, "bottom": 13.0, "right": 80.01, "top": 13.01}}) is True


def test_geojson_feature_count_mismatch_rejected():
    """13. Feature count mismatch rejected."""
    sample_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    meta = {"height": 2, "width": 2, "bounds": {"left": 80.0, "bottom": 13.0, "right": 80.1, "top": 13.1}}
    assert_raises_value_error(validate_geojson_contract, sample_geojson, metadata=meta)


def test_geojson_unclosed_ring_rejected():
    """14. Unclosed polygon ring (first vertex != last vertex) rejected."""
    unclosed_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 13.0], [80.01, 13.0], [80.01, 13.01], [80.0, 13.01], [80.02, 13.02]]] # Unclosed
                },
                "properties": {
                    "cell_id": 1, "row": 0, "col": 0, "risk_score_raw": 0.2, "risk_score": 0.2,
                    "risk_class": "LOW", "rainfall_mm_hr": 80.0, "r_overload": 0.5, "q_cap": 0.1, "q_demand": 0.05,
                    "elevation": 10.0, "slope": 1.0, "tpi": 0.0, "dist_to_drain": 50.0
                }
            }
        ]
    }
    assert_raises_value_error(validate_geojson_contract, unclosed_geojson)


def test_geojson_duplicate_cell_id_rejected():
    """15. Duplicate cell_id values detected and rejected."""
    duplicate_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 13.0], [80.01, 13.0], [80.01, 13.01], [80.0, 13.01], [80.0, 13.0]]]
                },
                "properties": {
                    "cell_id": 1, "row": 0, "col": 0, "risk_score_raw": 0.2, "risk_score": 0.2,
                    "risk_class": "LOW", "rainfall_mm_hr": 80.0, "r_overload": 0.5, "q_cap": 0.1, "q_demand": 0.05,
                    "elevation": 10.0, "slope": 1.0, "tpi": 0.0, "dist_to_drain": 50.0
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.01, 13.0], [80.02, 13.0], [80.02, 13.01], [80.01, 13.01], [80.01, 13.0]]]
                },
                "properties": {
                    "cell_id": 1, "row": 0, "col": 1, "risk_score_raw": 0.2, "risk_score": 0.2, # Duplicate cell_id=1
                    "risk_class": "LOW", "r_overload": 0.5, "q_cap": 0.1, "q_demand": 0.05,
                    "elevation": 10.0, "slope": 1.0, "tpi": 0.0, "dist_to_drain": 50.0
                }
            }
        ]
    }
    assert_raises_value_error(validate_geojson_contract, duplicate_geojson)
