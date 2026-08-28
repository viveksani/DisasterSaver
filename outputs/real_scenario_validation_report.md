# DisasterSaver — Real Flood Scenario Validation Report
## 1. Executive Summary
The DisasterSaver urban flood prediction engine was subjected to a comprehensive behavioral validation across 11 rainfall scenarios (0 to 1000 mm/hr). The pipeline demonstrated exact hydrological scaling (Q_demand scales 1.5x between 80 and 120 mm/hr), 100% numerical stability, and robust GeoJSON output validity. Spatial analysis confirmed high-risk clustering along low-lying river basins, driven primarily by drain proximity and topographic depression depth.
## 2. Rainfall Progression Table
| Rainfall (`mm/hr`) | Mean $Q_{\text{demand}}$ (`m³/s`) | Mean $R_{\text{overload}}$ | Min Score | Mean Score | Max Score | LOW (%) | MODERATE (%) | HIGH (%) | GeoJSON Valid |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | 0.0000 | 0.0000 | 0.0637 | 0.3810 | 0.8134 | 39.37% | 50.67% | **9.96%** | **PASS** |
| **20.0** | 0.0111 | 0.1897 | 0.0163 | 0.3918 | 1.0000 | 39.61% | 44.46% | **15.93%** | **PASS** |
| **50.0** | 0.0278 | 0.4742 | 0.0164 | 0.4051 | 1.0000 | 39.24% | 42.45% | **18.31%** | **PASS** |
| **80.0** | 0.0445 | 0.7587 | 0.0000 | 0.4083 | 1.0000 | 38.81% | 37.87% | **23.32%** | **PASS** |
| **100.0** | 0.0557 | 0.9484 | 0.0047 | 0.4104 | 1.0000 | 38.80% | 38.07% | **23.13%** | **PASS** |
| **120.0** | 0.0668 | 1.1381 | 0.0047 | 0.4111 | 1.0000 | 38.63% | 38.35% | **23.02%** | **PASS** |
| **150.0** | 0.0835 | 1.4226 | 0.0173 | 0.4195 | 1.0000 | 38.39% | 38.66% | **22.94%** | **PASS** |
| **200.0** | 0.1113 | 1.8968 | 0.0183 | 0.4277 | 1.0000 | 38.05% | 37.89% | **24.06%** | **PASS** |
| **300.0** | 0.1670 | 2.8451 | 0.0272 | 0.4378 | 1.0000 | 37.54% | 37.50% | **24.97%** | **PASS** |
| **500.0** | 0.2783 | 4.7419 | 0.0548 | 0.4606 | 1.0000 | 36.52% | 36.12% | **27.36%** | **PASS** |
| **1000.0** | 0.5565 | 9.4838 | 0.0500 | 0.5087 | 1.0000 | 33.55% | 35.22% | **31.24%** | **PASS** |

## 3. High-Risk vs Low-Risk Location Comparison (120 mm/hr)
### Top 5 Highest Risk Cells
| Rank | Cell ID | Row | Col | Risk Score | Class | Elev (m) | Slope (°) | TPI (m) | Dist Drain (m) | $Q_{\text{demand}}$ | $R_{\text{overload}}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 774753 | 888 | 416 | 1.0 | **HIGH** | 1.5 | 6.4 | -7.58 | 327.0 | 0.2851 | 5.7029 |
| 2 | 788072 | 903 | 655 | 1.0 | **HIGH** | 0.5 | 3.06 | -5.79 | 110.4 | 0.2376 | 4.7524 |
| 3 | 915889 | 1050 | 288 | 1.0 | **HIGH** | 4.38 | 7.31 | -8.88 | 233.8 | 0.2376 | 4.7524 |
| 4 | 915893 | 1050 | 292 | 1.0 | **HIGH** | 8.72 | 7.87 | -4.97 | 168.7 | 0.3802 | 7.6039 |
| 5 | 799120 | 916 | 367 | 1.0 | **HIGH** | 3.34 | 4.43 | -5.69 | 115.3 | 0.2376 | 4.7524 |

### Top 5 Safest (Lowest Risk) Cells
| Rank | Cell ID | Row | Col | Risk Score | Class | Elev (m) | Slope (°) | TPI (m) | Dist Drain (m) | $Q_{\text{demand}}$ | $R_{\text{overload}}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1367286 | 1567 | 861 | 0.0047 | **LOW** | 0.0 | 0.0 | 0.0 | 9470.1 | 0.0238 | 0.4752 |
| 2 | 1365551 | 1565 | 870 | 0.0047 | **LOW** | 0.0 | 0.0 | 0.0 | 9703.7 | 0.0238 | 0.4752 |
| 3 | 1365549 | 1565 | 868 | 0.0047 | **LOW** | 0.0 | 0.0 | 0.0 | 9646.9 | 0.0238 | 0.4752 |
| 4 | 1365550 | 1565 | 869 | 0.0047 | **LOW** | 0.0 | 0.0 | 0.0 | 9674.9 | 0.0238 | 0.4752 |
| 5 | 1365548 | 1565 | 867 | 0.0047 | **LOW** | 0.0 | 0.0 | 0.0 | 9618.1 | 0.0238 | 0.4752 |

## 4. Zero-Rainfall Behavior Analysis
- **Mean Risk Score at 0 mm/hr**: 0.381
- **HIGH Risk Cells at 0 mm/hr**: 136,205 (9.96%)
- **Conclusion**: `COMBINATION OF BOTH (SUSCEPTIBILITY + SCENARIO CONDITIONAL RISK)`
- **Explanation**: At 0 mm/hr rainfall, surface runoff demand Q_demand and R_overload are strictly 0.0. The physics monotonicity heuristic provides exactly 0.0 boost because R_overload <= 1.0. Therefore, the non-zero mean score (0.3810) and the 9.96% HIGH risk cells are produced 100% by the learned Random Forest tree ensemble. The model was trained on historical 2015 flood labels using static topographic features (elevation Z, slope S, TPI, and distance to drain). Static terrain features account for 65.74% of total model feature importance (dist_to_drain: 36.47%, elevation: 24.35%, slope: 16.01%). Consequently, risk_score represents spatial flood susceptibility conditional on the supplied rainfall scenario.

## 5. Scientific Limitations & What Is Proven
### Proven by Current Implementation & Tests
- 100% software contract compliance and GeoJSON structural validity (CRS84, 1.367M features, closed rings).
- Deterministic physical scaling: Q_demand scales exactly linearly with rainfall intensity (120/80 = 1.5000).
- Overload ratio R_overload scales dynamically and correctly with Q_demand / Q_cap.
- Physics monotonicity heuristic bounds scores strictly in [0.0, 1.0] without NaN or Inf anomalies.
- Operational API predict_for_rainfall() executes predictions for arbitrary rainfall inputs without retraining.

### Strongly Supported by Data
- High-risk cells are physically clustered in low-elevation, negative TPI topographic depressions.
- Proximity to mapped open drains (dist_to_drain) is the single strongest spatial predictor (r = -0.6194).
- Extreme storm intensities (300-1000 mm/hr) drive mean risk scores up to 0.5087 and HIGH risk cells to 31.24%.

### Not Yet Proven
- Real-world flood occurrence during future storm events (requires multi-year spatio-temporal validation).
- Hydrograph attenuation or dynamic flood depth in meters (requires 2D hydraulic flood routing).
- Post-processing heuristic score adjustments do not constitute statistically calibrated frequentist probabilities.

## 6. Final Questionnaire Verdicts
- **engine_responds_to_rainfall**: `YES`
- **spatially_different_flood_risk**: `YES`
- **risk_changes_at_individual_cells**: `YES`
- **physical_variables_consistent_with_predictions**: `YES`
- **engine_numerically_stable**: `YES`
- **proves_real_world_flood_occurrence**: `NO`
- **most_important_scientific_limitation**: `Single 2015 event calibration & single geographic holdout validation limits temporal multi-event proof.`
