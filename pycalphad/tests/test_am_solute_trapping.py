"""
Unit tests for the Aziz Solute Trapping and Additive Manufacturing Rapid Solidification models.
"""

import numpy as np
import pytest
from pycalphad.amkit.microsegregation import (
    calculate_trapped_partition_coefficient,
    simulate_microsegregation,
    simulate_am_rapid_solidification,
)


def test_aziz_solute_trapping_limits():
    """Verify that trapped partition coefficient approaches k_0 as v -> 0 and 1.0 as v -> infty."""
    k_eq = 0.15
    v_diff = 5.0

    # Low velocity limit (equilibrium)
    k_low_v = calculate_trapped_partition_coefficient(k_eq, interface_velocity=1e-6, diffusive_velocity=v_diff)
    assert np.isclose(k_low_v, k_eq, atol=1e-3)

    # Moderate velocity (AM regime: 0.1 - 2.0 m/s)
    k_am = calculate_trapped_partition_coefficient(k_eq, interface_velocity=1.0, diffusive_velocity=v_diff)
    assert k_eq < k_am < 1.0

    # High velocity limit (complete partitionless solute trapping)
    k_high_v = calculate_trapped_partition_coefficient(k_eq, interface_velocity=1e4, diffusive_velocity=v_diff)
    assert np.isclose(k_high_v, 1.0, atol=1e-3)


def test_ohnaka_back_diffusion_microsegregation():
    """Test microsegregation simulation with Ohnaka back-diffusion."""
    res = simulate_microsegregation(
        solute_concentration_initial=0.05,
        partition_coefficient=0.20,
        diffusivity_solid=1e-13,
        solidification_time=0.01,
        dendrite_arm_spacing=2e-6,
        interface_velocity=0.5,
        model="ohnaka"
    )

    f_s = res.grid["f_s"]
    c_l = res.field_data["C_liquid"]
    c_s = res.field_data["C_solid"]

    assert len(f_s) == len(c_l) == len(c_s)
    assert c_l[0] > 0.0
    assert c_l[-1] > c_l[0]  # Solute enriches liquid ahead of advancing interface
    assert res.scalar_outputs["k_trapped"] > 0.20  # Trapping increased k


def test_simulate_am_rapid_solidification():
    """Verify rapid solidification under AM processing parameters."""
    res = simulate_am_rapid_solidification(
        liquidus_temp=1600.0,
        solidus_temp_equilibrium=1400.0,
        solute_concentration=0.04,
        partition_coefficient=0.30,
        cooling_rate=1e6,         # 10^6 K/s (LPBF regime)
        thermal_gradient=5e6,     # 5x10^6 K/m
    )

    assert "interface_velocity_m_s" in res
    assert "k_trapped" in res
    assert res["k_trapped"] >= 0.30
    assert res["interface_velocity_m_s"] > 0.0
    assert len(res["temperatures"]) == len(res["fraction_solid"])
    assert "cracking_assessment" in res
    assert "risk_category" in res["cracking_assessment"]
