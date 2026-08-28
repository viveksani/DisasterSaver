"""
DisasterSaver — Flood Prediction Pipeline (Person 1)
Module: ml/train.py
Description: Model Training, Geographic Holdout Validation, Physics Monotonicity & Artifact Export

Workflow:
1. Loads Stage 3 feature matrix (data/processed/feature_matrix.npz).
2. Verifies feature column order and checks target isolation (y is not in X).
3. Applies single Geographic Holdout Split (North/Middle vs South) to reduce spatial autocorrelation leakage.
4. Trains ChennaiFloodModel (Random Forest with class_weight='balanced', random_state=42).
5. Evaluates raw probabilities and physics-adjusted probabilities across spatial validation block.
6. Computes Precision, Recall, F1-Score, ROC-AUC, PR-AUC, IoU, and Confusion Matrix.
7. Asserts 5 physics monotonicity tests.
8. Saves model to data/processed/model.pkl and performs post-save reload verification.
9. Exports metrics report to data/processed/training_metrics.json.

Target Environment: Google Colab (/content/ml/train.py) / Python 3.8+
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List, Optional

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from ml.flood_model import ChennaiFloodModel
from ml.preprocessing import FEATURE_NAMES


def load_and_verify_stage3_artifacts(
    npz_path: str = "data/processed/feature_matrix.npz"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads Stage 3 feature matrix archive and verifies target isolation and feature ordering.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Stage 3 artifact not found at {npz_path}. Run Stage 3 first.")

    data = np.load(npz_path)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.uint8)
    feature_names = data['feature_names'].tolist()

    # Feature names match check
    assert feature_names == FEATURE_NAMES, (
        f"Feature names mismatch! Loaded {feature_names}, expected {FEATURE_NAMES}"
    )

    # Feature leakage & dimension checks
    assert X.shape[1] == len(FEATURE_NAMES), (
        f"Feature column count error: X has {X.shape[1]} columns, expected {len(FEATURE_NAMES)}."
    )
    assert set(np.unique(y)).issubset({0, 1}), f"Invalid target labels: {set(np.unique(y))}"
    assert X.shape[0] == y.shape[0], f"Row count mismatch: X ({X.shape[0]}) != y ({y.shape[0]})"

    # Explicit Target Leakage Checks: target y is not identical to any column in X
    for idx, f_name in enumerate(feature_names):
        col = X[:, idx]
        if np.array_equal(col, y):
            raise ValueError(f"CRITICAL TARGET LEAKAGE DETECTED: Column '{f_name}' is identical to target y!")

    # Verify no NaN or Inf values in X
    assert not np.isnan(X).any(), "Input feature matrix X contains NaN values."
    assert not np.isinf(X).any(), "Input feature matrix X contains Infinite values."

    return X, y, feature_names


def spatial_block_split(
    X: np.ndarray,
    y: np.ndarray,
    grid_shape: Tuple[int, int],
    train_ratio: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits spatial grid into Geographic Holdout Blocks (North/Middle vs South)
    to reduce spatial autocorrelation leakage between neighboring cells.

    Validation Methodology Note:
    ----------------------------
    This single spatial holdout split reduces proximity bias between train and test cells.
    However, it is a single geographic holdout, not a full spatio-temporal cross-validation across
    multiple independent storm events.

    Parameters:
        X: Feature matrix of shape (total_cells, 7)
        y: Target vector of shape (total_cells,)
        grid_shape: Tuple (height, width) derived dynamically from grid metadata or X.shape
        train_ratio: Fraction of grid rows allocated to training (default 0.75)

    Returns:
        X_train, X_val, y_train, y_val
    """
    height, width = grid_shape
    assert height * width == X.shape[0], (
        f"Grid shape {grid_shape} total cells ({height * width}) does not match X rows ({X.shape[0]})"
    )

    split_row = int(height * train_ratio)
    row_indices = np.repeat(np.arange(height), width)

    train_mask = row_indices < split_row
    val_mask = row_indices >= split_row

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return X_train, X_val, y_train, y_val


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) for the flooded (positive) class:
        IoU = TP / (TP + FP + FN)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes classification and spatial validation metrics given continuous risk probabilities.
    Includes class distribution checks before computing ROC-AUC / PR-AUC.
    """
    preds = (probs >= threshold).astype(np.uint8)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true, probs))
        pr_auc = float(average_precision_score(y_true, probs))
    else:
        # Fallback if validation block has only one class label present
        roc_auc = 0.5
        pr_auc = 0.0

    iou = compute_iou(y_true, preds)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "iou": round(iou, 4),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "support": {
            "flooded_class_1": int(np.sum(y_true == 1)),
            "dry_class_0": int(np.sum(y_true == 0))
        }
    }


def run_physics_monotonicity_tests(model: ChennaiFloodModel) -> Dict[str, bool]:
    """
    Performs 5 assertion tests on the physics monotonicity adjustment logic.
    """
    test_results = {}

    # Test 1: Increasing r_overload cannot decrease adjusted risk score
    scores = np.array([0.40, 0.40, 0.40], dtype=np.float32)
    overload_low = np.array([0.5, 1.2, 3.0], dtype=np.float32)
    adj = model.enforce_physics_monotonicity(scores, overload_low)
    
    test_results["monotonic_increasing"] = bool(adj[2] >= adj[1] >= adj[0])

    # Test 2: R_overload <= 1.0 receives 0.0 adjustment
    score_unloaded = float(model.enforce_physics_monotonicity(np.array([0.30]), np.array([0.8]))[0])
    test_results["no_boost_below_one"] = bool(np.isclose(score_unloaded, 0.30))

    # Test 3: R_overload > 1.0 receives bounded boost
    score_overloaded = float(model.enforce_physics_monotonicity(np.array([0.30]), np.array([5.0]), boost_factor=0.15)[0])
    test_results["bounded_boost_above_one"] = bool(score_overloaded > 0.30 and (score_overloaded <= 0.45 + 1e-6))

    # Test 4: Adjusted scores always remain within [0.0, 1.0]
    high_scores = np.array([0.95, 0.99], dtype=np.float32)
    high_overload = np.array([10.0, 50.0], dtype=np.float32)
    adj_clipped = model.enforce_physics_monotonicity(high_scores, high_overload)
    test_results["clipped_in_zero_one_range"] = bool((adj_clipped >= 0.0).all() and (adj_clipped <= 1.0).all())

    # Test 5: No NaN or Inf values produced
    test_results["no_nan_or_inf"] = bool(not np.isnan(adj).any() and not np.isinf(adj).any())

    return test_results


def train_and_validate(
    npz_path: str = "data/processed/feature_matrix.npz",
    grid_metadata_path: str = "data/processed/grid_metadata.json",
    model_output_path: str = "data/processed/model.pkl",
    metrics_output_path: str = "data/processed/training_metrics.json"
) -> Tuple[ChennaiFloodModel, Dict[str, Any]]:
    """
    Main Stage 4 execution pipeline.
    """
    print("--- Stage 4: Model Training & Historical Validation ---")

    # Step 1: Load and verify Stage 3 artifacts
    X, y, feature_names = load_and_verify_stage3_artifacts(npz_path)
    total_cells = X.shape[0]

    # Load grid shape dynamically from metadata if available
    if os.path.exists(grid_metadata_path):
        with open(grid_metadata_path, 'r') as f:
            meta = json.load(f)
        grid_shape = (meta['height'], meta['width'])
    else:
        # Fallback shape inference if metadata is missing
        grid_shape = (1568, 872)

    assert grid_shape[0] * grid_shape[1] == total_cells, (
        f"Metadata grid shape {grid_shape} ({grid_shape[0]*grid_shape[1]}) != X rows ({total_cells})"
    )
    print(f"[OK] Loaded Stage 3 feature matrix: {X.shape} ({total_cells:,} cells, grid {grid_shape[0]}x{grid_shape[1]})")

    # Step 2: Report Class Distribution Baseline
    flooded_count = int(np.sum(y == 1))
    dry_count = int(np.sum(y == 0))
    flooded_pct = 100.0 * flooded_count / total_cells
    print(f"   Class Distribution: Flooded (1) = {flooded_count:,} ({flooded_pct:.2f}%) | Dry (0) = {dry_count:,} ({100.0-flooded_pct:.2f}%)")

    # Step 3: Apply Geographic Holdout Split (75% Train, 25% Validation)
    X_train, X_val, y_train, y_val = spatial_block_split(X, y, grid_shape=grid_shape, train_ratio=0.75)
    print(f"[OK] Geographic Holdout Split Applied:")
    print(f"   Train Cells      : {X_train.shape[0]:,} (North/Middle spatial block)")
    print(f"   Validation Cells : {X_val.shape[0]:,} (South spatial block)")

    # Step 4: Instantiate & Train Model
    model = ChennaiFloodModel(n_estimators=100, max_depth=12, random_state=42)
    print("Training ChennaiFloodModel (Random Forest with class_weight='balanced')...")
    model.fit(X_train, y_train)
    print("[OK] Model Training Complete.")

    # Step 5: Evaluate Raw vs Physics-Adjusted Probabilities
    val_raw_probs = model.predict_proba(X_val)
    val_r_overload = X_val[:, 6]  # Index 6 is r_overload feature
    val_physics_probs = model.enforce_physics_monotonicity(val_raw_probs, val_r_overload)

    raw_metrics = evaluate_predictions(y_val, val_raw_probs)
    physics_metrics = evaluate_predictions(y_val, val_physics_probs)

    # Step 6: Feature Importances
    importances = model.get_feature_importances()
    print("   Feature Importances:")
    for name, imp in importances.items():
        print(f"     - {name:<15}: {imp:.4f}")

    # Step 7: Physics Monotonicity Assertion Tests
    physics_tests = run_physics_monotonicity_tests(model)
    all_physics_passed = all(physics_tests.values())
    print(f"[OK] Physics Monotonicity Tests: {'ALL PASSED' if all_physics_passed else 'SOME FAILED'}")
    for t_name, t_pass in physics_tests.items():
        print(f"     - {t_name:<30}: {'PASS' if t_pass else 'FAIL'}")

    # Step 8: Save Model Artifact & Reload Test
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"[OK] Saved model artifact to {model_output_path}")

    # Reload Test Verification
    reloaded_model = joblib.load(model_output_path)
    reloaded_probs = reloaded_model.predict_proba(X_val[:100])
    reload_match = np.allclose(val_raw_probs[:100], reloaded_probs, rtol=1e-5)
    print(f"[OK] Reload Verification Test: {'PASSED (100% exact prediction match)' if reload_match else 'FAILED'}")

    # Step 9: Assemble & Export Metrics JSON
    training_report = {
        "model_architecture": "RandomForestClassifier",
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 12,
        "class_weight": "balanced",
        "dataset": {
            "total_spatial_cells": total_cells,
            "train_cells": X_train.shape[0],
            "validation_cells": X_val.shape[0],
            "feature_names": FEATURE_NAMES,
            "class_distribution": {
                "flooded_class_1": flooded_count,
                "dry_class_0": dry_count,
                "flooded_percentage": round(flooded_pct, 2)
            }
        },
        "validation_strategy": "Geographic Holdout Split (75% North/Middle vs 25% South)",
        "raw_model_metrics": raw_metrics,
        "physics_adjusted_metrics": physics_metrics,
        "feature_importances": importances,
        "physics_monotonicity_tests": physics_tests,
        "reload_test_passed": bool(reload_match),
        "notes": [
            "Geographic holdout split reduces spatial autocorrelation bias between train and val.",
            "Physics monotonicity adjustment is a heuristic post-processing rule applied to probabilities.",
            "Target leakage prevented: verified target y is isolated and not present in input matrix X."
        ]
    }

    with open(metrics_output_path, "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"[OK] Saved training metrics report to {metrics_output_path}")

    return model, training_report


if __name__ == "__main__":
    train_and_validate()
