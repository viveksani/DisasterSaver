# DisasterSaver — Downstream Pipeline Interface Specification

**Document Version:** 1.0 (Person 1 Hand-off Specification)  
**Target Audience:** Person 2 (Road Analysis, Priority Places, Evacuation Routing) & Person 3 (Backend API / Middleware Service)

---

## 1. Primary Handoff Artifact

Downstream modules (Person 2 & Person 3) MUST consume the standard GeoJSON prediction handoff file:

```text
outputs/predicted_flood.geojson
```

This artifact is exported directly by Person 1's batch inference module (`ml/predict.py`) and validated by `ml/pipeline_contract.py`.

---

## 2. Coordinate Reference System (CRS) & Grid Geometry

- **Declared CRS:** `urn:ogc:def:crs:OGC:1.3:CRS84` (WGS 84 geographic longitude/latitude in decimal degrees).
- **Geometry Type:** Every feature in the `FeatureCollection` is a 4-corner `Polygon` with a closed 5-vertex ring:
  ```json
  [
    [lon_min, lat_min],
    [lon_max, lat_min],
    [lon_max, lat_max],
    [lon_min, lat_max],
    [lon_min, lat_min]
  ]
  ```
- **Grid Dimensions:** Dynamic grid matching master terrain bounds. For the current Chennai study area:
  - Height ($H$) = 1,568 rows
  - Width ($W$) = 872 columns
  - Total Features = $1,568 \times 872 = 1,367,296$ polygon features.
- **Row-Major Feature Order:** Features are streamed sequentially in row-major order ($r \times W + c$ where $r \in [0, H-1]$, $c \in [0, W-1]$).

---

## 3. GeoJSON Feature Property Schema

Every polygon feature contains exactly 13 property fields:

| Property Field | Data Type | Units / Format | Description |
|:---|:---:|:---:|:---|
| `cell_id` | `int` | $1 \dots N$ | 1-based unique spatial cell index ($r \times W + c + 1$). |
| `row` | `int` | $0 \dots H-1$ | 0-based grid row index (North-to-South). |
| `col` | `int` | $0 \dots W-1$ | 0-based grid column index (West-to-East). |
| `risk_score_raw` | `float` | $[0.0, 1.0]$ | Raw unadjusted Random Forest class-1 flood probability. |
| `risk_score` | `float` | $[0.0, 1.0]$ | **Primary Score**: Physics-adjusted flood risk score. |
| `risk_class` | `str` | `"LOW" \| "MODERATE" \| "HIGH"` | Categorical engineering risk classification label. |
| `r_overload` | `float` | Ratio ($\ge 0.0$) | Drainage overload ratio ($Q_{\text{demand}} / Q_{\text{cap}}$). |
| `q_cap` | `float` | $\text{m}^3/\text{s}$ | Local channel conveyance capacity via Manning's equation or $0.05\text{ m}^3\text{/s}$ sheet-flow fallback. |
| `q_demand` | `float` | $\text{m}^3/\text{s}$ | Surface runoff demand via Rational Method ($C \cdot i \cdot A_{\text{D8}}$). |
| `elevation` | `float` | Meters ($\text{m}$) | Terrain elevation $Z$ from Copernicus DEM COP30. |
| `slope` | `float` | Degrees ($^\circ$) | Surface terrain slope $S$. |
| `tpi` | `float` | Meters ($\text{m}$) | Topographic Position Index (relative elevation vs 300m neighborhood). |
| `dist_to_drain` | `float` | Meters ($\text{m}$) | Distance to nearest mapped drain LineString (10m densification approximation). |

---

## 4. Hydrological Fields Meaning & Formulas

1. **`q_cap` (Conveyance Capacity, $\text{m}^3/\text{s}$)**:
   - For cells $\le 100\text{m}$ from a mapped drain: Calculated using Manning's equation for rectangular open channels:
     $$Q_{\text{cap}} = \frac{1}{n} A R^{2/3} \sqrt{S}$$
     where $A = w \cdot d$, $P = w + 2d$, $R = A/P$, $S = \text{bed slope}$, $n = \text{Manning roughness}$.
   - For cells $> 100\text{m}$ from mapped drains: Assigned $Q_{\text{overland}} = 0.05\text{ m}^3/\text{s}$ as an engineering fallback sheet-flow capacity floor.

2. **`q_demand` (Runoff Demand, $\text{m}^3/\text{s}$)**:
   - Calculated using the Rational Method:
     $$Q_{\text{demand}} = C \cdot \left(\frac{i_{\text{mm/hr}}}{3,600,000}\right) \cdot A_{\text{catchment\_m2}}$$
     where $i = 80\text{ mm/hr}$ (Dec 2, 2015 storm peak intensity), $C = 0.75$ (urban land cover coefficient), and $A_{\text{catchment\_m2}}$ is derived from recursive D8 terrain flow accumulation.

3. **`r_overload` (Drainage Overload Ratio)**:
   - Calculated as:
     $$R_{\text{overload}} = \frac{Q_{\text{demand}}}{Q_{\text{cap}}}$$
   - $R_{\text{overload}} \le 1.0$: Drainage capacity is adequate; runoff contained within channel.
   - $R_{\text{overload}} > 1.0$: Runoff demand exceeds conveyance capacity; surcharge / surface flooding expected.

---

## 5. Risk Class Thresholds

The continuous `risk_score` is categorized using **project-defined engineering cutoffs**:

- **`LOW`**: $\text{risk\_score} < 0.33$
- **`MODERATE`**: $0.33 \le \text{risk\_score} < 0.66$
- **`HIGH`**: $\text{risk\_score} \ge 0.66$

---

## 6. Canonical Feature Contract (Person 1 ML Engine)

Person 1's machine learning model accepts exactly 7 input features in the following strict column order:

```python
FEATURE_NAMES = [
    "elevation",        # Index 0: Terrain elevation (m)
    "slope",            # Index 1: Terrain slope (degrees)
    "tpi",              # Index 2: Topographic Position Index (m)
    "dist_to_drain",    # Index 3: Distance to nearest drain geometry (m)
    "q_cap",            # Index 4: Conveyance capacity (m³/s)
    "q_demand",         # Index 5: Runoff demand (m³/s)
    "r_overload"        # Index 6: Drainage overload ratio (dimensionless)
]
```

---

## 7. Critical Methodological Caveats (What Downstream MUST NOT Assume)

1. **Not a Calibrated Probability**:
   `risk_score` is a post-processed model risk score. It is **NOT** a statistically calibrated posterior probability of flooding (e.g., $P(\text{Flood}) = 0.85$ does not mean an 85% frequentist probability).
2. **Physics Monotonicity is a Heuristic Rule**:
   The post-processing adjustment applied when $R_{\text{overload}} > 1$ is a deterministic heuristic boost ($\min(0.15, \max(0, R-1) \times 0.05)$) to enforce monotonic physical behavior. It is **NOT** part of the Random Forest loss function.
3. **Single Geographic Validation Holdout**:
   Model validation was performed on a single North/South spatial holdout (South block rows 1176–1567). It does not validate temporal cross-event generalization across multiple storm years.
4. **Local Distance Approximation**:
   `dist_to_drain` is computed via 10m LineString vertex densification and `cKDTree` nearest-point lookup in local planar metric space. It is a spatial approximation ($\le 5\text{m}$ error bound), not an exact analytical point-to-segment distance.

---

## 8. Python Contract Validation API

Person 2 and Person 3 can validate any prediction record or GeoJSON handoff file using `ml/pipeline_contract.py`:

```python
from ml.pipeline_contract import (
    validate_feature_names,
    validate_prediction_fields,
    validate_risk_classes,
    validate_grid_metadata,
    validate_geojson_contract,
)

# Validate GeoJSON file
validate_geojson_contract("outputs/predicted_flood.geojson")

# Validate individual feature property dict
validate_prediction_fields(feature_properties)
```
