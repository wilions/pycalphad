"""
Unit tests for Process Metallurgy & Inclusion Engineering modules.
"""

import numpy as np
import pytest
from pycalphad.process import (
    InclusionPopulationBalance,
    check_liquid_calcium_aluminate_window,
    calculate_optical_basicity,
    calculate_slag_sulfide_capacity,
    calculate_phosphorus_partition_ratio,
    estimate_refractory_corrosion_rate,
)


def test_stokes_flotation_and_inclusion_growth():
    """Verify Stokes flotation velocity and radial inclusion growth."""
    pbe = InclusionPopulationBalance()

    # Floatation velocity of a 10-micron Al2O3 inclusion
    v_10um = pbe.calculate_stokes_flotation_velocity(inclusion_radius=5.0e-6, inclusion_density=3980.0)
    v_20um = pbe.calculate_stokes_flotation_velocity(inclusion_radius=10.0e-6, inclusion_density=3980.0)

    assert v_10um > 0.0
    assert v_20um > v_10um
    assert np.isclose(v_20um / v_10um, 4.0, atol=0.1)  # Velocity scales quadratically with radius

    # Population balance simulation over 1200 seconds (20 min refining)
    res = pbe.evaluate_inclusion_evolution(
        initial_mean_radius=1.0e-6,
        total_time_seconds=1200.0,
        inclusion_type="AL2O3"
    )
    assert res["final_radius_microns"] > 1.0
    assert 0.0 <= res["final_removal_percentage"] <= 100.0


def test_calcium_aluminate_liquid_window():
    """Verify liquid calcium aluminate window for SEN clogging prevention."""
    # Optimal Ca treatment: Al = 0.03 wt%, Ca = 30 ppm (0.003 wt%), T.O. = 25 ppm
    opt_res = check_liquid_calcium_aluminate_window(al_wt_pct=0.03, ca_ppm=30.0, total_oxygen_ppm=25.0)
    assert opt_res["is_liquid_window"] is True
    assert "Optimal casting window" in opt_res["clogging_risk_assessment"]

    # Under-treated: Ca = 5 ppm (Al2O3 / CA6 solid formation)
    under_res = check_liquid_calcium_aluminate_window(al_wt_pct=0.03, ca_ppm=5.0, total_oxygen_ppm=25.0)
    assert under_res["is_liquid_window"] is False
    assert "Solid Al2O3" in under_res["clogging_risk_assessment"]


def test_slag_sulfide_capacity_and_refractory_corrosion():
    """Verify optical basicity, sulfide capacity, and refractory wear."""
    basic_slag = {"CAO": 0.50, "SIO2": 0.35, "AL2O3": 0.10, "MGO": 0.05}
    lam = calculate_optical_basicity(basic_slag)
    assert 0.60 < lam < 0.80

    cs_data = calculate_slag_sulfide_capacity(basic_slag, temperature_K=1873.15)
    assert cs_data["sulfide_capacity_Cs"] > 0.0

    dephosphorization_slag = {"CAO": 0.50, "FEO": 0.20, "SIO2": 0.15, "MGO": 0.08, "AL2O3": 0.07}
    lp = calculate_phosphorus_partition_ratio(dephosphorization_slag, temperature_K=1873.15)
    assert lp > 10.0

    wear = estimate_refractory_corrosion_rate(basic_slag, temperature_K=1873.15, refractory_type="MGO_C")
    assert "estimated_wear_rate_mm_per_heat" in wear
    assert wear["estimated_wear_rate_mm_per_heat"] > 0.0
