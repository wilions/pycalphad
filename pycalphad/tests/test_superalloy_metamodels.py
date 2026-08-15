"""
Unit tests for Superalloy Metamodels, APB Energy, and TCP Risk.
"""

import numpy as np
import pytest
from pycalphad.superalloys import (
    calculate_gamma_prime_apb_energy,
    calculate_order_strengthening_increment,
    calculate_tcp_phase_risk_index,
    estimate_larson_miller_creep_life,
)


def test_gamma_prime_apb_energy_and_order_strengthening():
    """Verify APB energy and order strengthening for Ni-base superalloys."""
    in718_gamma_prime = {"NI": 0.70, "AL": 0.10, "TI": 0.12, "NB": 0.08}
    apb_res = calculate_gamma_prime_apb_energy(in718_gamma_prime, temperature_K=300.0)

    assert 140.0 < apb_res["APB_energy_mJ_m2"] < 320.0

    # Order strengthening increment
    strengthening = calculate_order_strengthening_increment(
        apb_energy_mJ_m2=apb_res["APB_energy_mJ_m2"],
        precipitate_radius_nm=15.0,
        volume_fraction=0.20
    )
    assert strengthening["yield_strength_increment_MPa"] > 100.0


def test_tcp_phase_risk_index():
    """Verify TCP phase precipitation risk based on d-orbital energy levels."""
    # Stable superalloy (e.g. IN718 matrix: moderate Cr, low Re/W)
    safe_matrix = {"NI": 0.60, "CR": 0.20, "FE": 0.15, "MO": 0.03, "AL": 0.02}
    safe_res = calculate_tcp_phase_risk_index(safe_matrix)
    assert safe_res["is_microstructurally_stable"] is True

    # High TCP risk alloy (excessive Cr + Mo + W)
    unstable_matrix = {"NI": 0.40, "CR": 0.30, "MO": 0.15, "W": 0.15}
    unstable_res = calculate_tcp_phase_risk_index(unstable_matrix)
    assert unstable_res["is_microstructurally_stable"] is False
    assert "High" in unstable_res["risk_category"]


def test_larson_miller_creep_estimation():
    """Verify Larson-Miller creep life estimation."""
    res_high_stress = estimate_larson_miller_creep_life(stress_MPa=400.0, temperature_celsius=750.0)
    res_low_stress = estimate_larson_miller_creep_life(stress_MPa=150.0, temperature_celsius=750.0)

    assert res_low_stress["estimated_creep_life_hours"] > res_high_stress["estimated_creep_life_hours"]
