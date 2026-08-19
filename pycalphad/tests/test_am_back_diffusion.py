import pytest
import numpy as np
from pycalphad.am.back_diffusion import (
    calculate_clyne_kurz_alpha,
    effective_partition_coefficient,
    simulate_back_diffusion_solidification,
)


def test_calculate_clyne_kurz_alpha_limits():
    # Very high cooling rate -> small alpha (Scheil limit)
    alpha_fast, alpha_prime_fast = calculate_clyne_kurz_alpha(
        D_s=1e-14, cooling_rate=1e6, dendrite_arm_spacing_m=10e-6, freezing_range_K=100.0
    )
    assert alpha_fast < 0.01
    assert alpha_prime_fast < 0.01

    # Very slow cooling rate -> large alpha (Lever rule equilibrium limit)
    alpha_slow, alpha_prime_slow = calculate_clyne_kurz_alpha(
        D_s=1e-11, cooling_rate=1.0, dendrite_arm_spacing_m=10e-6, freezing_range_K=100.0
    )
    assert alpha_slow > 10.0
    assert pytest.approx(alpha_prime_slow, abs=1e-2) == 0.5


def test_effective_partition_coefficient():
    k_eq = 0.3
    # At alpha_prime = 0 (Scheil), k_eff == k_eq
    assert effective_partition_coefficient(k_eq, alpha_prime=0.0) == k_eq

    # At higher alpha_prime, k_eff approaches 1.0 (equilibrium homogenization)
    k_eff_mid = effective_partition_coefficient(k_eq, alpha_prime=0.2)
    assert k_eq < k_eff_mid <= 1.0


def test_simulate_back_diffusion_solidification():
    res = simulate_back_diffusion_solidification(
        liquidus_K=1600.0,
        equilibrium_solidus_K=1500.0,
        k_eq=0.4,
        cooling_rate=1e4,
    )
    assert len(res.temperatures_K) == len(res.fraction_solid)
    assert res.liquidus_K == 1600.0
    assert res.kou_cracking_index >= 0.0
    assert len(res.solute_profile) == len(res.fraction_solid)
