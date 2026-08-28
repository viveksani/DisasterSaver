"""
Tests for ml/rainfall_runoff.py.
Covers Manning equation, Rational Method, zero rainfall, unit conversion,
monotonicity, NaN/Inf handling, and DataFrame processing.
"""

import numpy as np
import pandas as pd
from ml.rainfall_runoff import (
    manning_capacity,
    rainfall_to_intensity,
    rational_runoff,
    overload_ratio,
    calculate_runoff_stress,
    run_validation_tests,
    DEFAULT_CONCRETE_MANNING_N,
    DEFAULT_OVERLAND_FLOW_CAPACITY_M3_S,
)


def test_manning_channel_size_monotonicity():
    """Increasing channel size (width and depth) increases capacity."""
    cap_small = manning_capacity(width=0.3, depth=0.3, slope=0.002, mannings_n=0.015)
    cap_large = manning_capacity(width=1.2, depth=1.2, slope=0.002, mannings_n=0.015)
    assert cap_large > cap_small


def test_manning_roughness_monotonicity():
    """Increasing Manning's n decreases capacity."""
    cap_smooth = manning_capacity(width=0.5, depth=0.5, slope=0.002, mannings_n=0.015)
    cap_rough = manning_capacity(width=0.5, depth=0.5, slope=0.002, mannings_n=0.030)
    assert cap_smooth > cap_rough


def test_manning_slope_monotonicity():
    """Increasing channel slope increases capacity."""
    cap_flat = manning_capacity(width=0.5, depth=0.5, slope=0.001, mannings_n=0.015)
    cap_steep = manning_capacity(width=0.5, depth=0.5, slope=0.010, mannings_n=0.015)
    assert cap_steep > cap_flat


def test_rational_runoff_zero_rainfall():
    """Zero rainfall intensity yields zero runoff demand."""
    q_zero = rational_runoff(rainfall_intensity_mm_hr=0.0, catchment_area_m2=2500.0, runoff_coeff=0.75)
    assert q_zero == 0.0


def test_rational_runoff_rainfall_monotonicity():
    """Increasing rainfall intensity increases runoff demand."""
    q_low = rational_runoff(rainfall_intensity_mm_hr=20.0, catchment_area_m2=2500.0, runoff_coeff=0.75)
    q_high = rational_runoff(rainfall_intensity_mm_hr=80.0, catchment_area_m2=2500.0, runoff_coeff=0.75)
    assert q_high > q_low


def test_rational_runoff_area_monotonicity():
    """Increasing catchment area increases runoff demand."""
    q_small = rational_runoff(rainfall_intensity_mm_hr=50.0, catchment_area_m2=1000.0, runoff_coeff=0.75)
    q_large = rational_runoff(rainfall_intensity_mm_hr=50.0, catchment_area_m2=5000.0, runoff_coeff=0.75)
    assert q_large > q_small


def test_rational_runoff_unit_conversion():
    """Exact unit conversion test: Q = C * (i / 3.6e6) * A."""
    expected_q = 0.75 * (80.0 / 3600000.0) * 3600.0
    calc_q = rational_runoff(rainfall_intensity_mm_hr=80.0, catchment_area_m2=3600.0, runoff_coeff=0.75)
    assert np.isclose(calc_q, expected_q, rtol=1e-9)


def test_overload_ratio_safety():
    """Overload ratio formula Q_demand / Q_cap handling zero capacity gracefully."""
    r_normal = overload_ratio(q_demand=0.5, q_cap=0.25)
    assert np.isclose(r_normal, 2.0)

    r_zero_cap = overload_ratio(q_demand=0.5, q_cap=0.0)
    assert np.isfinite(r_zero_cap)
    assert r_zero_cap > 1000.0


def test_nan_inf_sanitisation():
    """Verify NaN and Inf inputs are sanitised cleanly."""
    q_nan = rational_runoff(float("nan"), 2500.0, 0.75)
    assert q_nan == 0.0

    q_inf = rational_runoff(float("inf"), 2500.0, 0.75)
    assert np.isfinite(q_inf)

    cap_nan = manning_capacity(float("nan"), 0.5, 0.001, 0.015)
    assert np.isfinite(cap_nan)


def test_calculate_runoff_stress_dataframe():
    """Test batch DataFrame calculation and missing parameter imputation."""
    df_raw = pd.DataFrame([
        {"width_m": 0.5, "depth_m": 0.5, "slope_m_m": 0.002, "mannings_n": 0.015},
        {"width_m": np.nan, "depth_m": 0.4, "slope_m_m": 0.001, "mannings_n": np.nan}
    ])
    df_out, stats = calculate_runoff_stress(df_raw, rainfall_intensity_mm_hr=50.0, catchment_area_m2=2500.0)
    assert "capacity_m3_s" in df_out.columns
    assert "runoff_demand_m3_s" in df_out.columns
    assert "overload_ratio" in df_out.columns
    assert stats["missing_or_invalid_width"] == 1
    assert stats["missing_or_invalid_mannings_n"] == 1


def test_internal_validation_suite():
    """Run built-in module validation suite."""
    assert run_validation_tests() is True
