# Technical Correction Pass — Person 1 DisasterSaver Pipeline Changelog

## Overview
This technical correction pass audited and refactored Person 1's machine learning and hydrological pipeline modules (`ml/flood_model.py`, `ml/rainfall_runoff.py`, `ml/preprocessing.py`, `ml/train.py`, `ml/predict.py`) and established a unit test suite under `tests/`.

---

## 1. Newly Created Modules

### `ml/flood_model.py`
- **Class `ChennaiFloodModel`**: Wraps `sklearn.ensemble.RandomForestClassifier` with `n_estimators=100`, `max_depth=12`, `random_state=42`, `class_weight="balanced"`.
- **`fit(X, y)`**: Fits the Random Forest classifier on feature matrix $X$ and binary labels $y$.
- **`predict_proba(X)`**: Returns class 1 (flooded) probability as a 1D float32 numpy array.
- **`get_feature_importances()`**: Returns a dictionary mapping exact feature names (`elevation`, `slope`, `tpi`, `dist_to_drain`, `q_cap`, `q_demand`, `r_overload`) to impurity importances.
- **`enforce_physics_monotonicity(scores, r_overload, boost_factor=0.15)`**: Implements a deterministic, bounded post-processing adjustment:
  $$\text{adjustment} = \min(\text{boost\_factor}, \max(0, R_{\text{overload}} - 1) \times 0.05)$$
  $$\text{adjusted\_score} = \text{clip}(\text{score} + \text{adjustment}, 0.0, 1.0)$$
- **`classify_risk(adjusted_scores)`**: Classifies continuous scores into project-defined engineering categories (`LOW` < 0.33, `MODERATE` 0.33–0.66, `HIGH` ≥ 0.66).

---

## 2. Updated & Corrected Modules

### `ml/rainfall_runoff.py`
- **NaN/Inf Input Protection**: Created `_sanitise()` helper to clamp invalid, infinite, or negative physical inputs to non-negative defaults before arithmetic evaluation.
- **Documentation Precision**: Clarified that `DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S = 0.05` m³/s is an engineering fallback capacity floor for undrained terrain cells, NOT an empirically calibrated soil infiltration rate.
- **Expanded Physical Tests**: Extended internal validation suite (`run_validation_tests()`) from 9 to 11 tests, adding tests for NaN and Inf input sanitisation.

### `ml/preprocessing.py`
- **D8 Flow Accumulation Accuracy**: Corrected docstrings and descriptions to accurately state that `compute_flow_accumulation()` routes flow down steepest descent gradients and treats depressions/flats as local sinks, rather than claiming sink-filled hydrological DEM correction.
- **Drain Distance Approximation**: Clarified that densifying LineStrings every 10m and using `cKDTree` nearest-point lookup is a spatial approximation with a maximum distance error bound $\le 5.0\text{ m}$, rather than claiming exact point-to-segment analytical distance.
- **Coordinate Conversion**: Corrected terminology from "UTM Zone 44N" to "local equirectangular / metric approximation".
- **Dynamic Resolution**: Updated resolution handling to read `pixel_size_m` dynamically from metadata instead of hard-coding 30.83 m.

### `ml/train.py`
- **Validation Terminology**: Replaced generic "spatial cross-validation" claims with "single geographic holdout split" (75% North/Middle vs 25% South).
- **Dynamic Grid Shapes**: Removed hardcoded grid shape `(1568, 872)` inside training logic; grid shape is now loaded dynamically from `grid_metadata.json` or validated against `X.shape[0]`.
- **Target Leakage Assertion**: Added explicit verification checking that target $y$ is not present as a column inside $X$, and verified feature names match `FEATURE_NAMES`.
- **Single-Class Evaluation Safety**: Added checks before `roc_auc_score()` and `average_precision_score()` to handle single-class validation splits gracefully without raising unhandled sklearn exceptions.

### `ml/predict.py`
- **Model Object Interface Validation**: Added explicit attribute/method inspection on `model.pkl` to confirm required methods (`predict_proba`, `enforce_physics_monotonicity`, `get_feature_importances`) exist before running batch inference.
- **Feature Name & Shape Verification**: Added checks ensuring `feature_names` from NPZ match `FEATURE_NAMES` and row count matches `metadata['height'] * metadata['width']`.
- **GeoJSON Structural Parsing**: Replaced arbitrary file size threshold (`file_size_mb > 1.0`) with structural stream parsing validation checking feature count, polygon ring closure, and header validity.

---

## 3. Unit Test Suite (`tests/`)

Created 24 comprehensive unit tests:
- `tests/test_rainfall_runoff.py` (11 tests: Manning equation, Rational Method, zero rainfall, unit conversion, monotonicity, NaN/Inf sanitisation).
- `tests/test_preprocessing.py` (6 tests: 1D slope flow accumulation, 3x3 central sink, flat DEM, segment densification, 20m synthetic drain distance accuracy, feature name ordering).
- `tests/test_flood_model.py` (5 tests: constructor hyperparameters, fit/predict_proba 1D array shapes, feature importances dictionary keys, physics monotonicity rules, risk classification thresholds).
- `tests/test_predict.py` (2 tests: batch inference output array shapes, GeoJSON export structural validity and polygon ring closure).

---

## 4. Intentionally Unchanged Architecture & Contracts

- **Feature Column Order**: The exact 7-feature order (`elevation`, `slope`, `tpi`, `dist_to_drain`, `q_cap`, `q_demand`, `r_overload`) was strictly preserved.
- **File Outputs & Contracts**: `data/processed/feature_matrix.npz`, `data/processed/model.pkl`, `data/processed/training_metrics.json`, and `outputs/predicted_flood.geojson` contracts remain 100% compatible with downstream specifications.
- **Hyperparameters**: `n_estimators=100`, `max_depth=12`, `random_state=42`, `class_weight="balanced"` remain untouched.

---

## 5. Verification Commands

To verify the test suite and pipeline in Google Colab or local terminal:

```bash
# 1. Run full unit test suite
python -m unittest discover tests/

# 2. Alternatively run pytest if installed
pytest tests/

# 3. Execute training & operational batch inference
python -m ml.train
python -m ml.predict
```
