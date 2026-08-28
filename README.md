# DisasterSaver — Urban Flood Prediction Pipeline (Person 1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GeoJSON Contract: CRS84](https://img.shields.io/badge/GeoJSON%20CRS-OGC%3ACRS84-green.svg)](docs/DOWNSTREAM_INTERFACE.md)

This repository contains **Person 1's** hydrological feature extraction engine, machine learning model (`ChennaiFloodModel`), physics post-processing engine, operational batch inference pipeline, and unit validation test suite for urban flood risk forecasting in Chennai, India.

---

## 1. System Overview & Architecture

Person 1 processes raw terrain elevation (Copernicus COP30 DEM), open-channel drainage geometries (`drains.kml`), and rainfall intensity data into a spatial flood risk layer:

```text
DEM / Terrain / Drainage / Rainfall
                 │
                 ▼
  Feature Assembly (7 Canonical Features)
                 │
                 ▼
  Hydrological Physics (Rational Method Q_demand & Manning Q_cap)
                 │
                 ▼
  Random Forest Model Inference (ChennaiFloodModel)
                 │
                 ▼
  Physics Monotonicity Adjustment (Heuristic Boost for R_overload > 1)
                 │
                 ▼
  Risk Classification (LOW / MODERATE / HIGH)
                 │
                 ▼
     outputs/predicted_flood.geojson
```

---

## 2. Mathematical & Hydrological Formulas

Person 1 integrates classical hydrological equations with ensemble machine learning:

### A. Topographic Position Index (TPI)
$$\text{TPI} = Z_0 - \bar{Z}_{\text{neighborhood}}$$
Measures the elevation difference between central grid cell $Z_0$ and surrounding terrain $\bar{Z}$. Negative values indicate topographic hollows/valleys vulnerable to pooling water.

### B. Rational Method Peak Discharge ($Q_{\text{demand}}$)
$$Q_{\text{demand}} = C \cdot i_{\text{m/s}} \cdot A_{\text{catchment}}$$
- $C$: Runoff coefficient ($0.65$ urban weighted).
- $i_{\text{m/s}}$: Rainfall intensity converted from $\text{mm/hr}$ ($i_{\text{m/s}} = \frac{i_{\text{mm/hr}}}{3,600,000}$).
- $A_{\text{catchment}}$: Upstream D8 flow accumulation area ($\text{m}^2$).

### C. Manning's Equation for Channel Capacity ($Q_{\text{cap}}$)
$$Q_{\text{cap}} = \frac{1}{n} \cdot A_{\text{flow}} \cdot R_h^{2/3} \cdot S^{1/2}$$
- $n$: Manning's roughness coefficient ($0.035$).
- $A_{\text{flow}} = w \cdot d$: Cross-sectional area ($\text{m}^2$).
- $P_{\text{wetted}} = w + 2d$: Wetted perimeter ($\text{m}$).
- $R_h = \frac{A_{\text{flow}}}{P_{\text{wetted}}}$: Hydraulic radius ($\text{m}$).
- $S$: Channel bed slope ($\tan(\text{slope\_deg})$).
- *Overland Fallback*: Terrain $>100\text{m}$ from mapped channels defaults to $0.05\text{ m}^3\text{/s}$ sheet-flow capacity floor.

### D. Drainage Overload Ratio ($R_{\text{overload}}$)
$$R_{\text{overload}} = \frac{Q_{\text{demand}}}{\max(Q_{\text{cap}}, 10^{-6})}$$
Dimensionless ratio of hydrological demand to channel capacity. $R_{\text{overload}} > 1.0$ signifies capacity deficit.

### E. Random Forest Class Probability ($P_{\text{raw}}$)
$$P_{\text{raw}}(\text{Flood} \mid X) = \frac{1}{K} \sum_{k=1}^{K} T_k(X)$$
Ensemble tree average across $K=100$ decision trees trained with `max_depth=12` and `class_weight='balanced'`.

### F. Heuristic Physics Monotonicity Adjustment
$$\text{risk\_score} = \min\left(1.0, P_{\text{raw}} + \min\left(0.15, \max\left(0, R_{\text{overload}} - 1.0\right) \times 0.05\right)\right)$$
Deterministic post-processing boost applied when $R_{\text{overload}} > 1.0$ to guarantee monotonic physical response.

---

## 3. Canonical 7-Feature Schema (`FEATURE_NAMES`)

The model accepts exactly 7 features in the following strict column order:

```python
FEATURE_NAMES = [
    "elevation",        # Index 0: Terrain elevation Z (m)
    "slope",            # Index 1: Terrain slope (degrees)
    "tpi",              # Index 2: Topographic Position Index (m)
    "dist_to_drain",    # Index 3: Distance to nearest mapped drain (m, 10m point densification)
    "q_cap",            # Index 4: Manning channel capacity Q_cap (m³/s)
    "q_demand",         # Index 5: Rational Method runoff demand Q_demand (m³/s)
    "r_overload"        # Index 6: Overload ratio R_overload = Q_demand / Q_cap (dimensionless)
]
```

---

## 4. Installation & Operational Commands

### Dependencies
Install standard Python numerical and geospatial libraries:
```bash
pip install numpy scipy pandas scikit-learn joblib
```

### A. Run Unit Test Suite
To verify physical formulas, schema bounds, and contract integrity:
```bash
python scratch/run_all_tests.py
# OR
python -m unittest discover tests/
```

### B. Train the Flood Prediction Model
Trains `ChennaiFloodModel` on North/Middle spatial block ($1,025,472$ cells), evaluates on South geographic holdout ($341,824$ cells), and generates `data/processed/model.pkl`:
```bash
python -m ml.train
```

### C. Run Operational Prediction for NEW RAINFALL
Generate predictions for any design storm intensity without retraining:

#### Command Line Execution (Default 80 mm/hr storm):
```bash
python -m ml.predict
```

#### Python API Execution (Custom NEW RAINFALL storm e.g. 120 mm/hr):
```python
from ml.predict import predict_for_rainfall

# Run operational prediction for 120 mm/hr extreme storm
predict_for_rainfall(
    rainfall_intensity_mm_hr=120.0,
    output_path="outputs/predicted_flood_120mm.geojson"
)
```

---

## 5. Downstream Handoff Contracts

### Person 2 (Road Risk & Priority Routing)
Person 2 consumes `outputs/predicted_flood.geojson` to perform spatial overlay analysis with `roads.geojson` and identify flooded road segments and safe evacuation corridors.

### Person 3 (Backend API & Middleware)
Person 3 consumes the prediction output schema to serve real-time risk scores and GeoJSON layers to the frontend dashboard.

### GeoJSON Specifications (`outputs/predicted_flood.geojson`)
- **CRS**: `urn:ogc:def:crs:OGC:1.3:CRS84` (WGS84 decimal degrees $[\text{lon}, \text{lat}]$).
- **Feature Count**: Exactly 1,367,296 closed 4-corner `Polygon` features.
- **Exact Contract Properties**:
```json
{
  "cell_id": 1,
  "row": 0,
  "col": 0,
  "risk_class": "LOW",
  "risk_score": 0.0932,
  "rainfall_mm_hr": 80.0,
  "runoff_m3_s": 0.0475,
  "drainage_capacity_m3_s": 0.05,
  "overload_ratio": 0.9505
}
```

### Risk Classification Thresholds
- **`LOW`**: $\text{risk\_score} < 0.33$
- **`MODERATE`**: $0.33 \le \text{risk\_score} < 0.66$
- **`HIGH`**: $\text{risk\_score} \ge 0.66$

---

## 6. Key Scientific Limitations

1. **Single-Event Calibration**: Model target $y$ is calibrated on the Dec 2015 Chennai storm event. Multi-event temporal cross-validation across independent storm years is not validated.
2. **Single Geographic Holdout**: Validation split evaluates generalization on the South spatial block ($341,824$ cells).
3. **Rational Method Peak Flow**: $Q_{\text{demand}} = C \cdot i \cdot A$ models instantaneous peak discharge for small urban basins. It does not model dynamic hydrograph routing or water depth in meters.
4. **Heuristic Monotonicity Boost**: Post-processing boost for $R_{\text{overload}} > 1$ is a deterministic heuristic rule, not a frequentist probability.
