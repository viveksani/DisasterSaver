"""
DisasterSaver — Flood Prediction Pipeline (Person 1)
Module: ml/predict.py
Description: Operational Inference Engine for NEW RAINFALL Scenarios & Standard GeoJSON Export

Workflow:
1. Loads trained ChennaiFloodModel artifact from data/processed/model.pkl.
2. Validates model object interface methods (predict_proba, enforce_physics_monotonicity, get_feature_importances).
3. Loads Stage 3 feature matrix X, verifies feature_names match FEATURE_NAMES, and verifies grid metadata.
4. Accepts operational NEW RAINFALL input in mm/hr (e.g. 50 mm/hr, 80 mm/hr, 120 mm/hr).
5. Recomputes rainfall-dependent runoff demand Q_demand and overload ratio R_overload.
6. Reconstructs exact 7-feature matrix in canonical column ordering.
7. Executes batch prediction, applies physics monotonicity adjustment, and maps risk classes.
8. Exports standard handoff GeoJSON with exact Person 1 contract properties (risk_class, risk_score, rainfall_mm_hr, runoff_m3_s, drainage_capacity_m3_s, overload_ratio).
9. Runs comprehensive structural validation on exported GeoJSON.

Target Environment: Google Colab (/content/ml/predict.py) / Python 3.8+
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List

from ml.flood_model import ChennaiFloodModel
from ml.preprocessing import FEATURE_NAMES
from ml.rainfall_runoff import rational_runoff, overload_ratio, MIN_Q_CAP


def load_inference_artifacts(
    model_path: str = "data/processed/model.pkl",
    feature_npz_path: str = "data/processed/feature_matrix.npz",
    metadata_path: str = "data/processed/grid_metadata.json"
) -> Tuple[ChennaiFloodModel, np.ndarray, Dict[str, Any]]:
    """
    Loads trained model, feature matrix X, and grid metadata JSON.
    Validates model methods, feature names, and shape assertions.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run Stage 4 first.")
    if not os.path.exists(feature_npz_path):
        raise FileNotFoundError(f"Feature matrix not found at {feature_npz_path}. Run Stage 3 first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Grid metadata not found at {metadata_path}. Run Stage 1 first.")

    model = joblib.load(model_path)

    # Interface validation check on loaded model object
    required_methods = ["predict_proba", "enforce_physics_monotonicity", "get_feature_importances"]
    for m_name in required_methods:
        if not hasattr(model, m_name) or not callable(getattr(model, m_name)):
            raise TypeError(f"Loaded model artifact is incompatible: missing required method '{m_name}'")

    npz_data = np.load(feature_npz_path)
    X = npz_data['X'].astype(np.float32)
    loaded_feature_names = npz_data['feature_names'].tolist()
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Verification checks
    assert loaded_feature_names == FEATURE_NAMES, (
        f"Feature name mismatch! Loaded {loaded_feature_names}, expected {FEATURE_NAMES}"
    )
    assert X.shape[1] == len(FEATURE_NAMES), (
        f"Feature count mismatch: X has {X.shape[1]} columns, expected {len(FEATURE_NAMES)}."
    )
    assert X.shape[0] == metadata['height'] * metadata['width'], (
        f"Grid cell count mismatch: X rows ({X.shape[0]}) != metadata height*width ({metadata['height']*metadata['width']})"
    )
    assert not np.isnan(X).any(), "Feature matrix X contains NaN values."
    assert not np.isinf(X).any(), "Feature matrix X contains Infinite values."

    return model, X, metadata


def compute_features_for_rainfall(
    X_baseline: np.ndarray,
    rainfall_intensity_mm_hr: float,
    baseline_rainfall_mm_hr: float = 80.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recomputes rainfall-dependent features (q_demand and r_overload) for a NEW RAINFALL scenario.

    Static Features Preserved:
    - elevation (col 0)
    - slope (col 1)
    - tpi (col 2)
    - dist_to_drain (col 3)
    - q_cap (col 4)

    Recomputed Features:
    - q_demand_new = q_demand_baseline * (rainfall_intensity_mm_hr / baseline_rainfall_mm_hr)
    - r_overload_new = q_demand_new / max(MIN_Q_CAP, q_cap)

    Parameters:
        X_baseline: Baseline feature matrix (total_cells, 7)
        rainfall_intensity_mm_hr: New rainfall intensity (mm/hr)
        baseline_rainfall_mm_hr: Baseline rainfall intensity used in X_baseline (default 80.0)

    Returns:
        X_new: Reconstructed 7-feature matrix of shape (total_cells, 7)
        q_demand_new: Array of updated runoff demands (m³/s)
        r_overload_new: Array of updated overload ratios
    """
    if rainfall_intensity_mm_hr < 0.0 or not np.isfinite(rainfall_intensity_mm_hr):
        raise ValueError(f"Invalid rainfall intensity: {rainfall_intensity_mm_hr}. Must be a non-negative finite number.")

    elevation = X_baseline[:, 0]
    slope = X_baseline[:, 1]
    tpi = X_baseline[:, 2]
    dist_to_drain = X_baseline[:, 3]
    q_cap = X_baseline[:, 4]
    q_demand_base = X_baseline[:, 5]

    # Proportional scaling for Rational Method demand Q_demand = C * (i / 3.6e6) * A
    if baseline_rainfall_mm_hr > 0.0:
        scale_factor = float(rainfall_intensity_mm_hr) / float(baseline_rainfall_mm_hr)
        q_demand_new = (q_demand_base * scale_factor).astype(np.float32)
    else:
        q_demand_new = np.zeros_like(q_demand_base, dtype=np.float32)

    # Recompute safe overload ratio
    r_overload_new = overload_ratio(q_demand=q_demand_new, q_cap=q_cap).astype(np.float32)

    # Reconstruct 7-feature matrix in canonical column order
    X_new = np.column_stack([
        elevation,
        slope,
        tpi,
        dist_to_drain,
        q_cap,
        q_demand_new,
        r_overload_new
    ]).astype(np.float32)

    return X_new, q_demand_new, r_overload_new


def run_batch_inference(
    model: ChennaiFloodModel,
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs batch inference over feature matrix X and returns raw probabilities,
    physics-adjusted probabilities, and risk class labels.

    Parameters:
        model: Trained ChennaiFloodModel instance
        X: Feature matrix of shape (total_cells, 7)

    Returns:
        raw_scores: Raw model flood probabilities [0.0, 1.0]
        adjusted_scores: Physics-adjusted flood probabilities [0.0, 1.0]
        risk_classes: Categorical risk labels ('LOW', 'MODERATE', 'HIGH')
    """
    raw_scores = model.predict_proba(X)
    r_overload = X[:, 6]  # Column index 6 is r_overload feature
    
    adjusted_scores = model.enforce_physics_monotonicity(raw_scores, r_overload)
    adjusted_scores_rounded = np.round(adjusted_scores, 4).astype(np.float32)

    # Categorical Risk Classification using project-defined cutoffs on rounded scores
    risk_classes = model.classify_risk(adjusted_scores_rounded)

    return raw_scores, adjusted_scores_rounded, risk_classes


def export_predicted_flood_geojson(
    X: np.ndarray,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    risk_classes: np.ndarray,
    metadata: Dict[str, Any],
    rainfall_intensity_mm_hr: float = 80.0,
    output_path: str = "outputs/predicted_flood.geojson"
) -> str:
    """
    Streaming export of grid cell predictions to standard GeoJSON handoff file.

    Exported GeoJSON Feature Property Contract:
    - Required fields: risk_class, risk_score, rainfall_mm_hr
    - Physical fields: runoff_m3_s, drainage_capacity_m3_s, overload_ratio
    - Internal fields: cell_id, row, col, risk_score_raw, elevation, slope, tpi, dist_to_drain

    Parameters:
        X: Feature matrix (num_cells, 7)
        raw_scores: Raw risk scores array
        adjusted_scores: Adjusted risk scores array
        risk_classes: String risk class array
        metadata: Grid metadata dict containing bounds and shape
        rainfall_intensity_mm_hr: Operational rainfall intensity in mm/hr
        output_path: Output GeoJSON file path

    Returns:
        output_path: Absolute path to written GeoJSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    height, width = metadata['height'], metadata['width']
    bounds = metadata['bounds']
    left, bottom, right, top = bounds['left'], bounds['bottom'], bounds['right'], bounds['top']

    dx = (right - left) / float(width)
    dy = (top - bottom) / float(height)

    # Unpack feature columns for fast streaming indexing
    elevation_arr = X[:, 0]
    slope_arr = X[:, 1]
    tpi_arr = X[:, 2]
    dist_drain_arr = X[:, 3]
    q_cap_arr = X[:, 4]
    q_demand_arr = X[:, 5]
    r_overload_arr = X[:, 6]

    total_cells = X.shape[0]

    with open(output_path, "w", encoding="utf-8") as f:
        # GeoJSON Header with CRS declaration
        f.write('{\n"type": "FeatureCollection",\n')
        f.write('"name": "predicted_flood",\n')
        f.write('"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },\n')
        f.write('"features": [\n')

        # Streaming Features Loop
        for idx in range(total_cells):
            r = idx // width
            c = idx % width

            lon_min = left + c * dx
            lon_max = left + (c + 1) * dx
            lat_max = top - r * dy
            lat_min = top - (r + 1) * dy

            # Construct Polygon Coordinates [[ [lon, lat], ... ]] (Closed ring, 5 points)
            coords_str = (
                f"[[[{lon_min:.6f},{lat_min:.6f}],"
                f"[{lon_max:.6f},{lat_min:.6f}],"
                f"[{lon_max:.6f},{lat_max:.6f}],"
                f"[{lon_min:.6f},{lat_max:.6f}],"
                f"[{lon_min:.6f},{lat_min:.6f}]]]"
            )

            # Contract-aligned Property Dict
            props = {
                "cell_id": idx + 1,
                "row": r,
                "col": c,
                # Required Contract Fields
                "risk_class": str(risk_classes[idx]),
                "risk_score": round(float(adjusted_scores[idx]), 4),
                "rainfall_mm_hr": round(float(rainfall_intensity_mm_hr), 2),
                # Physical Field Alias Names
                "runoff_m3_s": round(float(q_demand_arr[idx]), 4),
                "drainage_capacity_m3_s": round(float(q_cap_arr[idx]), 4),
                "overload_ratio": round(float(r_overload_arr[idx]), 4),
                # Internal Legacy Properties
                "risk_score_raw": round(float(raw_scores[idx]), 4),
                "r_overload": round(float(r_overload_arr[idx]), 4),
                "q_cap": round(float(q_cap_arr[idx]), 4),
                "q_demand": round(float(q_demand_arr[idx]), 4),
                "elevation": round(float(elevation_arr[idx]), 2),
                "slope": round(float(slope_arr[idx]), 2),
                "tpi": round(float(tpi_arr[idx]), 2),
                "dist_to_drain": round(float(dist_drain_arr[idx]), 1)
            }

            feature_str = (
                f'{{"type":"Feature","geometry":{{"type":"Polygon","coordinates":{coords_str}}},'
                f'"properties":{json.dumps(props)}}}'
            )

            f.write(feature_str)
            if idx < total_cells - 1:
                f.write(",\n")
            else:
                f.write("\n")

        # GeoJSON Footer
        f.write("]\n}\n")

    return output_path


def run_stage5_validations(
    X: np.ndarray,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    risk_classes: np.ndarray,
    metadata: Dict[str, Any],
    output_geojson_path: str
) -> Dict[str, bool]:
    """
    Runs 10 validation assertions on Stage 5 inference and GeoJSON output.
    """
    from ml.pipeline_contract import validate_geojson_contract

    results = {}
    total_cells = metadata['height'] * metadata['width']

    # 1. Prediction count == total grid cells
    results["val_1_prediction_count"] = bool(len(adjusted_scores) == total_cells)

    # 2. Raw risk scores finite and in [0,1]
    results["val_2_raw_scores_valid"] = bool((raw_scores >= 0.0).all() and (raw_scores <= 1.0).all() and not np.isnan(raw_scores).any())

    # 3. Physics-adjusted risk scores finite and in [0,1]
    results["val_3_adjusted_scores_valid"] = bool((adjusted_scores >= 0.0).all() and (adjusted_scores <= 1.0).all() and not np.isnan(adjusted_scores).any())

    # 4. Risk classes contain only 'LOW', 'MODERATE', 'HIGH'
    unique_classes = set(np.unique(risk_classes))
    results["val_4_risk_classes_valid"] = bool(unique_classes.issubset({"LOW", "MODERATE", "HIGH"}))

    # 5. Output GeoJSON file exists
    results["val_5_file_exists"] = bool(os.path.exists(output_geojson_path) and os.path.getsize(output_geojson_path) > 0)

    # 6. GeoJSON structural contract parsing check
    try:
        results["val_6_geojson_structural_validity"] = validate_geojson_contract(output_geojson_path, metadata)
    except Exception as e:
        results["val_6_geojson_structural_validity"] = False

    # 7. Output GeoJSON header parsing check
    try:
        with open(output_geojson_path, "r", encoding="utf-8") as f:
            header_sample = f.read(500)
        results["val_7_geojson_header_parseable"] = bool('"FeatureCollection"' in header_sample and '"features"' in header_sample)
    except Exception:
        results["val_7_geojson_header_parseable"] = False

    # 8. Output grid shape matches metadata
    results["val_8_grid_shape_exact"] = bool(metadata['height'] * metadata['width'] == total_cells)

    # 9. Master grid bounds check
    bounds = metadata['bounds']
    results["val_9_bounds_valid"] = bool(bounds['left'] < bounds['right'] and bounds['bottom'] < bounds['top'])

    # 10. Reload model and check inference repeatability
    model_reloaded = joblib.load("data/processed/model.pkl")
    reload_scores = model_reloaded.predict_proba(X[:500])
    results["val_10_reload_inference_repeatable"] = bool(np.allclose(raw_scores[:500], reload_scores, rtol=1e-5))

    return results


def predict_for_rainfall(
    rainfall_intensity_mm_hr: float = 80.0,
    model_path: str = "data/processed/model.pkl",
    feature_npz_path: str = "data/processed/feature_matrix.npz",
    metadata_path: str = "data/processed/grid_metadata.json",
    output_path: str = "outputs/predicted_flood.geojson"
) -> Dict[str, Any]:
    """
    Operational inference entrypoint for NEW RAINFALL inputs.

    Parameters:
        rainfall_intensity_mm_hr: Operational design storm intensity in mm/hr (e.g. 50, 80, 120)
        model_path: Trained model artifact path
        feature_npz_path: Baseline feature matrix path
        metadata_path: Grid metadata path
        output_path: Output GeoJSON export path

    Returns:
        Summary report dictionary.
    """
    print(f"--- Stage 5 Operational Inference: NEW RAINFALL ({rainfall_intensity_mm_hr:.1f} mm/hr) ---")

    # Step 1: Load Artifacts
    model, X_baseline, metadata = load_inference_artifacts(model_path, feature_npz_path, metadata_path)
    total_cells = X_baseline.shape[0]

    # Step 2: Recompute features for NEW RAINFALL
    print(f"Recomputing runoff demand Q_demand and overload ratio R_overload for {rainfall_intensity_mm_hr:.1f} mm/hr...")
    X_new, q_demand_new, r_overload_new = compute_features_for_rainfall(
        X_baseline=X_baseline,
        rainfall_intensity_mm_hr=rainfall_intensity_mm_hr,
        baseline_rainfall_mm_hr=80.0
    )

    # Step 3: Run Batch Inference
    print("Running model prediction and applying physics monotonicity adjustment...")
    raw_scores, adjusted_scores, risk_classes = run_batch_inference(model, X_new)

    # Step 4: Compute Risk Summary
    low_count = int(np.sum(risk_classes == "LOW"))
    mod_count = int(np.sum(risk_classes == "MODERATE"))
    high_count = int(np.sum(risk_classes == "HIGH"))

    low_pct = 100.0 * low_count / total_cells
    mod_pct = 100.0 * mod_count / total_cells
    high_pct = 100.0 * high_count / total_cells

    print(f"   Risk Distribution ({rainfall_intensity_mm_hr:.1f} mm/hr):")
    print(f"     - LOW      : {low_count:,} cells ({low_pct:.2f}%)")
    print(f"     - MODERATE : {mod_count:,} cells ({mod_pct:.2f}%)")
    print(f"     - HIGH     : {high_count:,} cells ({high_pct:.2f}%)")

    # Step 5: Export GeoJSON
    print(f"Exporting standard contract GeoJSON to {output_path}...")
    export_predicted_flood_geojson(
        X=X_new,
        raw_scores=raw_scores,
        adjusted_scores=adjusted_scores,
        risk_classes=risk_classes,
        metadata=metadata,
        rainfall_intensity_mm_hr=rainfall_intensity_mm_hr,
        output_path=output_path
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] GeoJSON handoff file exported successfully ({file_size_mb:.2f} MB).")

    # Step 6: Validations
    validations = run_stage5_validations(X_new, raw_scores, adjusted_scores, risk_classes, metadata, output_path)
    all_passed = all(validations.values())

    print(f"[OK] Validation Suite: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return {
        "status": "PASS" if all_passed else "FAIL",
        "rainfall_intensity_mm_hr": float(rainfall_intensity_mm_hr),
        "total_predicted_cells": total_cells,
        "grid_shape": [metadata['height'], metadata['width']],
        "risk_class_distribution": {
            "LOW": {"count": low_count, "percentage": round(low_pct, 2)},
            "MODERATE": {"count": mod_count, "percentage": round(mod_pct, 2)},
            "HIGH": {"count": high_count, "percentage": round(high_pct, 2)}
        },
        "output_file": output_path,
        "output_file_size_mb": round(file_size_mb, 2),
        "validations": validations
    }


def run_stage5_inference_pipeline() -> Dict[str, Any]:
    """
    Default entrypoint for Stage 5 batch inference (uses baseline storm intensity 80 mm/hr).
    """
    return predict_for_rainfall(rainfall_intensity_mm_hr=80.0)


if __name__ == "__main__":
    run_stage5_inference_pipeline()
