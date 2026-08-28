"""
DisasterSaver — Flood Prediction Pipeline (Person 1)
Module: ml/pipeline_contract.py
Description: Pipeline Contract & Downstream Validation Interface Layer.

Purpose:
Establishes and enforces the canonical interface contract between Person 1's flood prediction engine
and downstream modules (Person 2: Road & Priority Zone Risk Analysis; Person 3: Backend API Service).

Contracts Enforced:
1. Feature Contract: Canonical 7-feature schema and exact column ordering.
2. Prediction Record Contract: Exact required GeoJSON property fields (risk_class, risk_score, rainfall_mm_hr, runoff_m3_s, drainage_capacity_m3_s, overload_ratio).
3. Risk Classification Contract: Project-defined risk thresholds (LOW < 0.33, MODERATE 0.33-0.66, HIGH >= 0.66).
4. Grid Metadata Contract: Grid height, width, bounding box (left, bottom, right, top), and CRS84 declaration.
5. GeoJSON Contract: Structural FeatureCollection, ring closure, property types, and cell index integrity.

Target Environment: Python 3.8+ / Google Colab
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from ml.preprocessing import FEATURE_NAMES

# ──────────────────────────────────────────────────────────────────────────────
# CANONICAL CONTRACT CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# 1. Canonical Feature Schema (7 features in strict column order)
CANONICAL_FEATURE_NAMES: List[str] = list(FEATURE_NAMES)

# 2. Canonical Required & Optional GeoJSON Property Fields
REQUIRED_PREDICTION_FIELDS: List[str] = [
    "risk_class",
    "risk_score",
    "rainfall_mm_hr"
]

EXPOSED_PHYSICAL_FIELDS: List[str] = [
    "runoff_m3_s",
    "drainage_capacity_m3_s",
    "overload_ratio"
]

# Complete GeoJSON Property Field List
CANONICAL_PREDICTION_FIELDS: List[str] = [
    "cell_id",
    "row",
    "col",
    "risk_class",
    "risk_score",
    "risk_score_raw",
    "rainfall_mm_hr",
    "runoff_m3_s",
    "drainage_capacity_m3_s",
    "overload_ratio",
    "r_overload",
    "q_cap",
    "q_demand",
    "elevation",
    "slope",
    "tpi",
    "dist_to_drain"
]

# 3. Canonical Risk Class Thresholds (Project-defined engineering cutoffs)
RISK_LOW_CUTOFF: float = 0.33
RISK_HIGH_CUTOFF: float = 0.66
VALID_RISK_CLASSES: List[str] = ["LOW", "MODERATE", "HIGH"]

# 4. Canonical CRS Identifier
CANONICAL_CRS: str = "urn:ogc:def:crs:OGC:1.3:CRS84"


# ──────────────────────────────────────────────────────────────────────────────
# 1. FEATURE CONTRACT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_feature_names(feature_names: List[str]) -> bool:
    """
    Validates that a supplied list of feature names matches CANONICAL_FEATURE_NAMES exactly.

    Rejects:
    - Missing features
    - Extra features
    - Reordered features
    - Renamed features

    Parameters:
        feature_names: List of string feature names

    Returns:
        True if valid; raises ValueError on contract violation.
    """
    if not isinstance(feature_names, (list, tuple, np.ndarray)):
        raise ValueError(f"Feature names must be a list or tuple, got {type(feature_names).__name__}")
    
    names_list = list(feature_names)
    if names_list != CANONICAL_FEATURE_NAMES:
        raise ValueError(
            f"Feature names contract violation!\n"
            f"Expected exact canonical schema: {CANONICAL_FEATURE_NAMES}\n"
            f"Got: {names_list}"
        )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREDICTION RECORD CONTRACT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_prediction_fields(record_or_properties: Dict[str, Any]) -> bool:
    """
    Validates that a prediction dictionary or GeoJSON feature property set contains
    all required contract property fields: risk_class, risk_score, rainfall_mm_hr.

    Parameters:
        record_or_properties: Dictionary containing key-value property pairs.

    Returns:
        True if valid; raises ValueError on contract violation.
    """
    if not isinstance(record_or_properties, dict):
        raise ValueError(f"Prediction record must be a dict, got {type(record_or_properties).__name__}")

    missing_required = [field for field in REQUIRED_PREDICTION_FIELDS if field not in record_or_properties]
    if missing_required:
        raise ValueError(f"Prediction record contract violation! Missing required property fields: {missing_required}")

    return True


# ──────────────────────────────────────────────────────────────────────────────
# 3. RISK CLASS CONTRACT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_risk_classes(
    scores: Union[float, np.ndarray, List[float]],
    risk_classes: Union[str, np.ndarray, List[str]]
) -> bool:
    """
    Validates that continuous risk scores are in [0.0, 1.0] and that categorical risk_classes
    are strictly consistent with project-defined cutoffs:
        LOW       : score < 0.33
        MODERATE  : 0.33 <= score < 0.66
        HIGH      : score >= 0.66

    Parameters:
        scores: Float scalar or array of risk scores
        risk_classes: String scalar or array of risk classes

    Returns:
        True if valid; raises ValueError on contract violation.
    """
    s_arr = np.asarray(scores, dtype=np.float64)
    c_arr = np.asarray(risk_classes, dtype=object)

    if s_arr.shape != c_arr.shape:
        raise ValueError(f"Shape mismatch: scores shape {s_arr.shape} != risk_classes shape {c_arr.shape}")

    # Check score range [0.0, 1.0]
    if np.isnan(s_arr).any() or np.isinf(s_arr).any():
        raise ValueError("Risk scores contain NaN or Inf values.")
    if (s_arr < 0.0).any() or (s_arr > 1.0).any():
        raise ValueError(f"Risk scores must be within [0.0, 1.0]. Range: min={s_arr.min()}, max={s_arr.max()}")

    # Check valid string values
    invalid_labels = set(c_arr.ravel()) - set(VALID_RISK_CLASSES)
    if invalid_labels:
        raise ValueError(f"Invalid risk class labels detected: {invalid_labels}. Allowed: {VALID_RISK_CLASSES}")

    # Verify cutoff consistency
    expected_classes = np.full(s_arr.shape, "LOW", dtype=object)
    expected_classes[(s_arr >= RISK_LOW_CUTOFF) & (s_arr < RISK_HIGH_CUTOFF)] = "MODERATE"
    expected_classes[s_arr >= RISK_HIGH_CUTOFF] = "HIGH"

    mismatches = c_arr != expected_classes
    if mismatches.any():
        idx = np.where(np.atleast_1d(mismatches))[0][0]
        s_flat = np.atleast_1d(s_arr)
        c_flat = np.atleast_1d(c_arr)
        e_flat = np.atleast_1d(expected_classes)
        raise ValueError(
            f"Risk class cutoff contract violation at index {idx}!\n"
            f"Risk score {s_flat[idx]} assigned class '{c_flat[idx]}', expected '{e_flat[idx]}'."
        )

    return True


# ──────────────────────────────────────────────────────────────────────────────
# 4. GRID METADATA CONTRACT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_grid_metadata(
    metadata: Dict[str, Any],
    num_cells: Optional[int] = None
) -> bool:
    """
    Validates that grid metadata dictionary conforms to project contract.

    Required Fields:
        - height: int (> 0)
        - width: int (> 0)
        - bounds: dict with 'left', 'bottom', 'right', 'top' (left < right, bottom < top)

    Parameters:
        metadata: Grid metadata dictionary.
        num_cells: Optional total cell count to verify height * width == num_cells.

    Returns:
        True if valid; raises ValueError on contract violation.
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"Grid metadata must be a dict, got {type(metadata).__name__}")

    for field in ["height", "width", "bounds"]:
        if field not in metadata:
            raise ValueError(f"Grid metadata contract violation! Missing field: '{field}'")

    height = metadata["height"]
    width = metadata["width"]
    if not isinstance(height, int) or height <= 0 or not isinstance(width, int) or width <= 0:
        raise ValueError(f"Grid height and width must be positive integers, got height={height}, width={width}")

    total_cells = height * width
    if num_cells is not None and total_cells != num_cells:
        raise ValueError(
            f"Grid shape contract violation! height ({height}) * width ({width}) = {total_cells} "
            f"does not match expected total cells ({num_cells})."
        )

    bounds = metadata["bounds"]
    for b_key in ["left", "bottom", "right", "top"]:
        if b_key not in bounds:
            raise ValueError(f"Grid bounds contract violation! Missing boundary key: '{b_key}'")

    if bounds["left"] >= bounds["right"]:
        raise ValueError(f"Grid bounds error: left ({bounds['left']}) >= right ({bounds['right']})")
    if bounds["bottom"] >= bounds["top"]:
        raise ValueError(f"Grid bounds error: bottom ({bounds['bottom']}) >= top ({bounds['top']})")

    return True


# ──────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION ARRAYS CONTRACT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_prediction_arrays(
    X: np.ndarray,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    risk_classes: np.ndarray,
    metadata: Dict[str, Any],
    feature_names: Optional[List[str]] = None
) -> bool:
    """
    Validates batch prediction matrix, risk score arrays, and metadata consistency.

    Parameters:
        X: Feature matrix of shape (total_cells, 7)
        raw_scores: Raw model risk probabilities array of shape (total_cells,)
        adjusted_scores: Physics-adjusted probabilities array of shape (total_cells,)
        risk_classes: Categorical risk labels array of shape (total_cells,)
        metadata: Grid metadata dictionary
        feature_names: Optional feature name list to verify ordering

    Returns:
        True if valid; raises ValueError on contract violation.
    """
    if feature_names is not None:
        validate_feature_names(feature_names)

    total_cells = X.shape[0]
    validate_grid_metadata(metadata, num_cells=total_cells)

    if X.shape[1] != len(CANONICAL_FEATURE_NAMES):
        raise ValueError(f"Feature matrix column count mismatch: X has {X.shape[1]} columns, expected {len(CANONICAL_FEATURE_NAMES)}")

    for name, arr in [("raw_scores", raw_scores), ("adjusted_scores", adjusted_scores), ("risk_classes", risk_classes)]:
        if arr.shape[0] != total_cells:
            raise ValueError(f"Array length mismatch: {name} length ({arr.shape[0]}) != total cells ({total_cells})")

    validate_risk_classes(adjusted_scores, risk_classes)
    validate_risk_classes(raw_scores, np.full_like(risk_classes, "LOW"))

    return True


# ──────────────────────────────────────────────────────────────────────────────
# 6. GEOJSON HANDOFF CONTRACT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_geojson_contract(
    geojson_data_or_path: Union[str, Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Validates standard GeoJSON output contract (structural parsing, FeatureCollection, polygon ring closure,
    required property fields, cell index uniqueness, and CRS).

    Parameters:
        geojson_data_or_path: File path (str) or parsed dictionary representation of GeoJSON.
        metadata: Optional grid metadata dictionary for dimension and bound verification.

    Returns:
        True if valid; raises ValueError on contract violation.
    """
    expected_total_cells = (metadata["height"] * metadata["width"]) if metadata is not None else None

    if isinstance(geojson_data_or_path, str):
        if not os.path.exists(geojson_data_or_path):
            raise ValueError(f"GeoJSON file not found at path: {geojson_data_or_path}")

        # Stream parsing for large files
        feat_count = 0
        unique_cell_ids = set()
        has_crs_header = False

        with open(geojson_data_or_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                ls = line.strip()
                if line_no < 5 and ("CRS84" in ls or "CRS" in ls or "FeatureCollection" in ls):
                    has_crs_header = True

                if ls.startswith('{"type":"Feature"'):
                    feat_count += 1
                    feat_obj = json.loads(ls.rstrip(','))
                    
                    if feat_obj.get("type") != "Feature":
                        raise ValueError(f"Feature line {line_no} type is not 'Feature'")

                    geom = feat_obj.get("geometry", {})
                    if geom.get("type") != "Polygon":
                        raise ValueError(f"Feature line {line_no} geometry is not 'Polygon'")

                    ring = geom.get("coordinates", [[]])[0]
                    if len(ring) != 5 or ring[0] != ring[-1]:
                        raise ValueError(f"Feature line {line_no} polygon ring is not closed (first vertex != last vertex)")

                    props = feat_obj.get("properties", {})
                    validate_prediction_fields(props)
                    validate_risk_classes(props["risk_score"], props["risk_class"])

                    cell_id = props.get("cell_id", feat_count)
                    if cell_id in unique_cell_ids:
                        raise ValueError(f"Duplicate cell_id detected: {cell_id}")
                    unique_cell_ids.add(cell_id)

                    if metadata is not None:
                        w = metadata["width"]
                        expected_row = (cell_id - 1) // w
                        expected_col = (cell_id - 1) % w
                        if props["row"] != expected_row or props["col"] != expected_col:
                            raise ValueError(
                                f"Row/Col grid index mismatch for cell_id {cell_id}! "
                                f"Got ({props['row']}, {props['col']}), expected ({expected_row}, {expected_col})"
                            )

        if expected_total_cells is not None and feat_count != expected_total_cells:
            raise ValueError(f"GeoJSON feature count mismatch: found {feat_count}, expected {expected_total_cells}")

        return True

    elif isinstance(geojson_data_or_path, dict):
        g_data = geojson_data_or_path
        if g_data.get("type") != "FeatureCollection":
            raise ValueError(f"GeoJSON top-level type must be 'FeatureCollection', got '{g_data.get('type')}'")

        features = g_data.get("features", [])
        if expected_total_cells is not None and len(features) != expected_total_cells:
            raise ValueError(f"GeoJSON feature count mismatch: found {len(features)}, expected {expected_total_cells}")

        unique_cell_ids = set()
        for idx, feat in enumerate(features):
            if feat.get("type") != "Feature":
                raise ValueError(f"Feature index {idx} type is not 'Feature'")

            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                raise ValueError(f"Feature index {idx} geometry is not 'Polygon'")

            ring = geom.get("coordinates", [[]])[0]
            if len(ring) != 5 or ring[0] != ring[-1]:
                raise ValueError(f"Feature index {idx} polygon ring is not closed")

            props = feat.get("properties", {})
            validate_prediction_fields(props)
            validate_risk_classes(props["risk_score"], props["risk_class"])

            cell_id = props.get("cell_id", idx + 1)
            if cell_id in unique_cell_ids:
                raise ValueError(f"Duplicate cell_id detected: {cell_id}")
            unique_cell_ids.add(cell_id)

        return True
    else:
        raise ValueError(f"Invalid input type for GeoJSON validation: {type(geojson_data_or_path).__name__}")
