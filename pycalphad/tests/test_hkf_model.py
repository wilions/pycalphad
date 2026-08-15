"""
Unit tests for the Helgeson-Kirkham-Flowers (HKF) Aqueous model.
"""

import numpy as np
import pytest
from pycalphad.models.model_hkf import (
    calculate_hkf_standard_gibbs_energy,
    water_dielectric_constant,
)


def test_water_dielectric_constant():
    """Verify dielectric constant of water decreases with increasing temperature."""
    eps_298, y_298 = water_dielectric_constant(298.15)
    eps_400, y_400 = water_dielectric_constant(400.0)

    assert np.isclose(eps_298, 78.4, atol=0.5)
    assert eps_400 < eps_298


def test_hkf_standard_gibbs_energy():
    """Verify standard partial molar Gibbs energy calculation for aqueous ions."""
    res_na = calculate_hkf_standard_gibbs_energy("NA+", temperature_K=298.15)
    res_cl = calculate_hkf_standard_gibbs_energy("CL-", temperature_K=298.15)
    res_fe = calculate_hkf_standard_gibbs_energy("FE2+", temperature_K=350.0)

    assert np.isclose(res_na["DeltaG_0_J_mol"], -261.9e3, atol=1e3)
    assert np.isclose(res_cl["DeltaG_0_J_mol"], -131.2e3, atol=1e3)
    assert res_fe["DeltaG_0_J_mol"] < 0.0
    assert "Cp_0_J_mol_K" in res_fe
