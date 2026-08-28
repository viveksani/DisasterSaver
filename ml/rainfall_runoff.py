"""
DisasterSaver — Flood Prediction Pipeline (Person 1)
Module: ml/rainfall_runoff.py
Description: Hydrological Runoff & Drainage Stress Engine

Functions:
- manning_capacity:        Q_cap (m³/s) via Manning's equation for rectangular channels.
- rainfall_to_intensity:   mm/day -> mm/hr conversion.
- rational_runoff:         Q_demand (m³/s) via Q = C * i * A with mm/hr -> m/s conversion.
- overload_ratio:          R_overload = Q_demand / Q_cap with zero-division guard.
- calculate_runoff_stress: Batch processing of a drainage DataFrame.
- run_validation_tests:    9 physical and unit-conversion assertion tests.

Engineering notes:
- DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S = 0.05 m³/s is an engineering fallback capacity
  for terrain cells beyond 100 m of mapped drainage channels. It represents an assumed
  ambient sheet-flow and infiltration capacity. It is NOT an empirically calibrated
  infiltration rate and should be treated as a sensitivity parameter.
- NaN/Inf inputs are treated as invalid and clamped to physical defaults (0 or min_q_cap).

Target Environment: Google Colab (/content/ml/rainfall_runoff.py) / Python 3.8+
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Any

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & ENGINEERING DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONCRETE_MANNING_N: float = 0.015   # Smooth concrete / lined channel
DEFAULT_KACHA_MANNING_N: float    = 0.030   # Unlined / earthen / natural channel
DEFAULT_WIDTH_M: float            = 0.5     # Drain width fallback (m)
DEFAULT_DEPTH_M: float            = 0.5     # Drain depth fallback (m)
DEFAULT_SLOPE_M_M: float          = 0.001   # Channel bed slope fallback (m/m, 0.1 %)
DEFAULT_RUNOFF_COEFF: float       = 0.75    # Urban runoff coefficient C (dimensionless)

# Engineering fallback for ambient overland/sheet-flow capacity.
# Applied to grid cells > 100 m from any mapped drainage channel.
# This is NOT an empirically calibrated infiltration rate; it is an assumed
# lower-bound conveyance capacity that prevents unphysical R_overload ratios
# (>10 000) in regions where no channel infrastructure is mapped.
DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S: float = 0.05

# Absolute zero-division epsilon — prevents division by zero inside overload_ratio.
MIN_Q_CAP: float = 1e-6


def _sanitise(
    arr: Union[float, np.ndarray, pd.Series],
    floor: float = 0.0,
) -> np.ndarray:
    """
    Convert input to float64 ndarray, replace NaN/Inf with *floor*, then
    clamp to [floor, +inf).

    This ensures all downstream arithmetic never encounters NaN or Inf
    when the upstream data pipeline supplies invalid sensor readings.
    """
    a = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        a = a.copy()
        a[bad] = floor
    return np.maximum(floor, a)


# ──────────────────────────────────────────────────────────────────────────────
# CORE HYDRAULIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def manning_capacity(
    width: Union[float, np.ndarray, pd.Series],
    depth: Union[float, np.ndarray, pd.Series],
    slope: Union[float, np.ndarray, pd.Series],
    mannings_n: Union[float, np.ndarray, pd.Series] = DEFAULT_CONCRETE_MANNING_N,
) -> Union[float, np.ndarray]:
    """
    Manning's equation for a rectangular open channel cross-section:

        Q_cap = (1/n) * A * R^(2/3) * sqrt(S)

    where:
        A = width * depth           (cross-sectional area, m²)
        P = width + 2 * depth       (wetted perimeter, m)
        R = A / P                   (hydraulic radius, m)
        S = channel bed slope       (m/m, dimensionless)
        n = Manning roughness       (dimensionless)

    Invalid inputs (NaN, Inf, negative) are clamped to zero before calculation.
    For mannings_n, negative/zero values are clamped to 1e-4 to prevent
    division by zero.

    Parameters
    ----------
    width, depth, slope : scalar or array-like
        Physical channel dimensions. Must be non-negative.
    mannings_n : scalar or array-like
        Manning roughness coefficient. Must be positive.

    Returns
    -------
    Q_cap : float or np.ndarray
        Conveyance capacity in m³/s. Scalar input → scalar output.
    """
    w = _sanitise(width,      floor=0.0)
    d = _sanitise(depth,      floor=0.0)
    s = _sanitise(slope,      floor=0.0)
    n = np.maximum(1e-4, _sanitise(mannings_n, floor=0.0))

    area            = w * d
    wetted_perim    = w + 2.0 * d

    with np.errstate(divide="ignore", invalid="ignore"):
        r_hyd = np.where(wetted_perim > 0.0, area / wetted_perim, 0.0)

    q_cap = (1.0 / n) * area * (r_hyd ** (2.0 / 3.0)) * np.sqrt(s)

    scalar_input = (
        np.ndim(width) == 0
        and np.ndim(depth) == 0
        and np.ndim(slope) == 0
        and np.ndim(mannings_n) == 0
    )
    return float(q_cap.item()) if scalar_input else q_cap


def rainfall_to_intensity(
    rainfall_mm_day: Union[float, np.ndarray, pd.Series],
    duration_hours: float = 24.0,
) -> Union[float, np.ndarray]:
    """
    Convert total rainfall (mm over *duration_hours*) to average intensity (mm/hr).

    Parameters
    ----------
    rainfall_mm_day : scalar or array-like
        Total rainfall accumulation in mm. Negative values treated as 0.
    duration_hours : float
        Accumulation period in hours (must be > 0).

    Returns
    -------
    intensity_mm_hr : float or np.ndarray
    """
    if duration_hours <= 0:
        raise ValueError(f"duration_hours must be > 0, got {duration_hours!r}.")

    rainfall = _sanitise(rainfall_mm_day, floor=0.0)
    intensity = rainfall / float(duration_hours)

    return float(intensity.item()) if np.ndim(rainfall_mm_day) == 0 else intensity


def rational_runoff(
    rainfall_intensity_mm_hr: Union[float, np.ndarray, pd.Series],
    catchment_area_m2: Union[float, np.ndarray, pd.Series],
    runoff_coeff: float = DEFAULT_RUNOFF_COEFF,
) -> Union[float, np.ndarray]:
    """
    Rational Method peak runoff:

        Q_demand = C * i_m_per_s * A

    Unit conversion:
        i (mm/hr) → i (m/s):  i_m_per_s = i_mm_hr / 3 600 000

    The Rational Method assumes instantaneous peak flow, not cumulative
    basin response. It is suitable for estimating peak discharge but does
    not model hydrograph routing or antecedent moisture conditions.

    Parameters
    ----------
    rainfall_intensity_mm_hr : scalar or array-like
        Rainfall intensity in mm/hr. Negative values treated as 0.
    catchment_area_m2 : scalar or array-like
        Contributing catchment area in m². Negative values treated as 0.
    runoff_coeff : float
        Dimensionless runoff coefficient C in [0, 1].

    Returns
    -------
    Q_demand : float or np.ndarray
        Runoff demand in m³/s.
    """
    c         = float(np.clip(runoff_coeff, 0.0, 1.0))
    intensity = _sanitise(rainfall_intensity_mm_hr, floor=0.0)
    area      = _sanitise(catchment_area_m2, floor=0.0)

    i_m_per_s = intensity / 3_600_000.0
    q_demand  = c * i_m_per_s * area

    scalar_input = np.ndim(rainfall_intensity_mm_hr) == 0 and np.ndim(catchment_area_m2) == 0
    return float(q_demand.item()) if scalar_input else q_demand


def overload_ratio(
    q_demand: Union[float, np.ndarray, pd.Series],
    q_cap: Union[float, np.ndarray, pd.Series],
    min_q_cap: float = MIN_Q_CAP,
) -> Union[float, np.ndarray]:
    """
    Drainage overload ratio:

        R_overload = Q_demand / Q_cap

    Interpretation:
        R_overload <= 1  →  capacity adequate; flow contained within channel.
        R_overload  > 1  →  capacity exceeded; surcharge / surface flooding expected.

    NaN/Inf inputs are clamped before division. Q_cap is floored at min_q_cap
    (default 1e-6) to prevent zero-division.

    Parameters
    ----------
    q_demand : scalar or array-like (m³/s)
    q_cap    : scalar or array-like (m³/s)
    min_q_cap : float — zero-division guard (m³/s)

    Returns
    -------
    R_overload : float or np.ndarray
    """
    demand   = _sanitise(q_demand, floor=0.0)
    capacity = np.maximum(float(min_q_cap), _sanitise(q_cap, floor=0.0))

    r_over = demand / capacity

    scalar_input = np.ndim(q_demand) == 0 and np.ndim(q_cap) == 0
    return float(r_over.item()) if scalar_input else r_over


# ──────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def calculate_runoff_stress(
    drain_data: Union[pd.DataFrame, Dict[str, Any]],
    rainfall_intensity_mm_hr: float,
    catchment_area_m2: Union[float, np.ndarray, pd.Series] = 2500.0,
    runoff_coeff: float = DEFAULT_RUNOFF_COEFF,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute Q_cap, Q_demand, and R_overload for a drainage dataset.

    Missing or invalid (NaN, Inf, <= 0) hydraulic parameters are replaced
    with engineering defaults and counted in the returned *default_stats* dict.

    Parameters
    ----------
    drain_data : pd.DataFrame or dict
        Must contain columns: width_m, depth_m, slope_m_m, mannings_n.
    rainfall_intensity_mm_hr : float
        Design storm intensity applied uniformly (mm/hr).
    catchment_area_m2 : float or array-like
        Contributing area per drain (m²). Default 2500 m² is a benchmark value,
        not the real catchment area of every drain; Stage 3 derives per-cell
        areas from D8 flow accumulation.
    runoff_coeff : float
        Runoff coefficient C.

    Returns
    -------
    df : pd.DataFrame — original columns + capacity_m3_s, runoff_demand_m3_s, overload_ratio
    default_stats : dict — count of imputed values per column
    """
    df = pd.DataFrame(drain_data).copy() if isinstance(drain_data, dict) else drain_data.copy()

    default_stats: Dict[str, int] = {
        "missing_or_invalid_width":      0,
        "missing_or_invalid_depth":      0,
        "missing_or_invalid_slope":      0,
        "missing_or_invalid_mannings_n": 0,
    }

    def _impute(col: str, default: float, stat_key: str) -> None:
        if col not in df.columns:
            df[col] = default
            default_stats[stat_key] = len(df)
        else:
            bad = df[col].isnull() | ~np.isfinite(df[col]) | (df[col] <= 0)
            default_stats[stat_key] = int(bad.sum())
            df.loc[bad, col] = default

    _impute("width_m",    DEFAULT_WIDTH_M,            "missing_or_invalid_width")
    _impute("depth_m",    DEFAULT_DEPTH_M,            "missing_or_invalid_depth")
    _impute("slope_m_m",  DEFAULT_SLOPE_M_M,          "missing_or_invalid_slope")
    _impute("mannings_n", DEFAULT_CONCRETE_MANNING_N, "missing_or_invalid_mannings_n")

    df["capacity_m3_s"] = manning_capacity(
        width=df["width_m"].values,
        depth=df["depth_m"].values,
        slope=df["slope_m_m"].values,
        mannings_n=df["mannings_n"].values,
    )

    df["runoff_demand_m3_s"] = rational_runoff(
        rainfall_intensity_mm_hr=rainfall_intensity_mm_hr,
        catchment_area_m2=catchment_area_m2,
        runoff_coeff=runoff_coeff,
    )

    df["overload_ratio"] = overload_ratio(
        q_demand=df["runoff_demand_m3_s"].values,
        q_cap=df["capacity_m3_s"].values,
    )

    return df, default_stats


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION TESTS
# ──────────────────────────────────────────────────────────────────────────────

def run_validation_tests() -> bool:
    """
    9 physical formula, unit-conversion, and robustness assertion tests.
    Raises AssertionError on failure; returns True on full success.
    """
    print("--- Running Stage 2 Physical & Unit Validation Tests ---")

    # TEST 1: Increasing rainfall -> increasing Q_demand
    q1 = rational_runoff(20.0, 2500.0, 0.75)
    q2 = rational_runoff(80.0, 2500.0, 0.75)
    assert q2 > q1, "TEST 1 FAILED: higher rainfall did not increase Q_demand"
    print("  [PASSED] TEST 1: Increasing rainfall intensity -> increasing Q_demand.")

    # TEST 2: Increasing rainfall -> increasing R_overload
    r1 = overload_ratio(q1, 0.5)
    r2 = overload_ratio(q2, 0.5)
    assert r2 > r1, "TEST 2 FAILED: higher demand did not increase R_overload"
    print("  [PASSED] TEST 2: Increasing rainfall intensity -> increasing R_overload.")

    # TEST 3: Increasing catchment area -> increasing Q_demand
    q_small = rational_runoff(50.0, 1000.0, 0.75)
    q_large = rational_runoff(50.0, 5000.0, 0.75)
    assert q_large > q_small, "TEST 3 FAILED: larger area did not increase Q_demand"
    print("  [PASSED] TEST 3: Increasing catchment area -> increasing Q_demand.")

    # TEST 4: Increasing drain dimensions -> increasing Q_cap
    cap_small = manning_capacity(0.3, 0.3, 0.002, 0.015)
    cap_large = manning_capacity(1.2, 1.2, 0.002, 0.015)
    assert cap_large > cap_small, "TEST 4 FAILED: larger drain did not increase Q_cap"
    print("  [PASSED] TEST 4: Increasing drain dimensions -> increasing Q_cap.")

    # TEST 5: Increasing Manning n -> decreasing Q_cap
    cap_smooth = manning_capacity(0.5, 0.5, 0.002, 0.015)
    cap_rough  = manning_capacity(0.5, 0.5, 0.002, 0.030)
    assert cap_smooth > cap_rough, "TEST 5 FAILED: rougher channel did not decrease Q_cap"
    print("  [PASSED] TEST 5: Increasing Manning n -> decreasing Q_cap.")

    # TEST 6: Increasing slope -> increasing Q_cap
    cap_flat  = manning_capacity(0.5, 0.5, 0.001, 0.015)
    cap_steep = manning_capacity(0.5, 0.5, 0.010, 0.015)
    assert cap_steep > cap_flat, "TEST 6 FAILED: steeper slope did not increase Q_cap"
    print("  [PASSED] TEST 6: Increasing channel slope -> increasing Q_cap.")

    # TEST 7: Zero rainfall -> zero Q_demand
    q_zero = rational_runoff(0.0, 2500.0, 0.75)
    assert q_zero == 0.0, "TEST 7 FAILED: zero rainfall did not give zero Q_demand"
    print("  [PASSED] TEST 7: Zero rainfall -> zero Q_demand.")

    # TEST 8: Positive parameters -> positive Q_cap
    assert cap_smooth > 0.0 and cap_rough > 0.0, "TEST 8 FAILED: Q_cap not positive"
    print("  [PASSED] TEST 8: Positive channel parameters -> positive Q_cap.")

    # TEST 9: Exact unit conversion 80 mm/hr -> m/s -> m³/s
    expected_q = 0.75 * (80.0 / 3_600_000.0) * 3600.0
    calc_q     = rational_runoff(80.0, 3600.0, 0.75)
    assert np.isclose(calc_q, expected_q, rtol=1e-9), (
        f"TEST 9 FAILED: expected {expected_q:.10f}, got {calc_q:.10f}"
    )
    print("  [PASSED] TEST 9: Exact unit conversion (80 mm/hr -> m/s -> m³/s) verified.")

    # TEST 10: NaN input -> treated as 0 (no crash, no NaN output)
    q_nan = rational_runoff(float("nan"), 2500.0, 0.75)
    assert q_nan == 0.0 and np.isfinite(q_nan), "TEST 10 FAILED: NaN input not sanitised"
    cap_nan = manning_capacity(float("nan"), 0.5, 0.002, 0.015)
    assert np.isfinite(cap_nan), "TEST 10 FAILED: NaN width not sanitised in manning_capacity"
    print("  [PASSED] TEST 10: NaN inputs sanitised to zero — no NaN propagation.")

    # TEST 11: Inf input -> treated as 0 / non-Inf output
    q_inf = rational_runoff(float("inf"), 2500.0, 0.75)
    # inf rainfall is clamped: _sanitise replaces inf -> 0 (floor), so result = 0
    assert np.isfinite(q_inf), "TEST 11 FAILED: Inf input produced non-finite output"
    print("  [PASSED] TEST 11: Inf inputs sanitised — no Inf propagation.")

    print("[OK] ALL 11 PHYSICAL & UNIT VALIDATION TESTS PASSED CLEANLY.\n")
    return True


if __name__ == "__main__":
    run_validation_tests()
