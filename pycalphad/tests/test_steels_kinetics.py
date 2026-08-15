"""
Unit tests for Steel Transformation Kinetics and Tempering models.
"""

import numpy as np
import pytest
from pycalphad.steels import (
    calculate_martensite_start_temperature,
    calculate_bainite_start_temperature,
    calculate_martensite_fraction_koistinen_marburger,
    calculate_jmak_isothermal_kinetics,
    simulate_cct_transformation,
    calculate_hollomon_jaffe_parameter,
    calculate_as_quenched_hardness,
    calculate_tempered_hardness,
)


def test_martensite_and_bainite_start():
    """Verify Ms and Bs calculations for AISI 4140 low-alloy steel."""
    # AISI 4140: 0.40% C, 0.85% Mn, 0.25% Si, 1.0% Cr, 0.20% Mo
    wt_4140 = {"C": 0.40, "MN": 0.85, "SI": 0.25, "CR": 1.0, "MO": 0.20}
    ms_res = calculate_martensite_start_temperature(wt_4140)
    bs_res = calculate_bainite_start_temperature(wt_4140)

    assert 300.0 < ms_res["Ms_celsius"] < 360.0
    assert 480.0 < bs_res["Bs_celsius"] < 580.0
    assert bs_res["Bs_celsius"] > ms_res["Ms_celsius"]


def test_koistinen_marburger_and_jmak():
    """Verify progressive martensite formation below Ms and JMAK kinetics."""
    ms = 330.0
    f_330 = calculate_martensite_fraction_koistinen_marburger(330.0, ms)
    f_250 = calculate_martensite_fraction_koistinen_marburger(250.0, ms)
    f_100 = calculate_martensite_fraction_koistinen_marburger(100.0, ms)

    assert f_330 == 0.0
    assert 0.40 < f_250 < 0.70
    assert f_100 > 0.90

    # JMAK kinetics
    times = np.array([0.0, 10.0, 100.0, 1000.0])
    x_t = calculate_jmak_isothermal_kinetics(times, rate_constant_k=1e-4, avrami_exponent_n=2.0)
    assert x_t[0] == 0.0
    assert 0.0 < x_t[1] < x_t[2] < x_t[3] <= 1.0


def test_tempered_hardness_and_secondary_hardening():
    """Verify Hollomon-Jaffe tempering and secondary hardening in tool/alloy steels."""
    wt_4140 = {"C": 0.40, "MN": 0.85, "CR": 1.0, "MO": 0.20, "V": 0.15}
    res_400 = calculate_tempered_hardness(wt_4140, tempering_temperature_celsius=400.0, holding_time_hours=2.0)
    res_540 = calculate_tempered_hardness(wt_4140, tempering_temperature_celsius=540.0, holding_time_hours=2.0)

    assert res_400["tempered_HRC"] > 35.0
    assert res_540["secondary_hardening_increment_HRC"] > 0.0
    assert res_400["estimated_yield_strength_MPa"] > 1000.0
