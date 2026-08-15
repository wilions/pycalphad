"""
Unit tests for the Elasticity Tensor and Polycrystalline VRH Moduli models.
"""

import numpy as np
import pytest
from pycalphad.models.model_elasticity import (
    calculate_voigt_reuss_hill_cubic,
    calculate_voigt_reuss_hill_hexagonal,
    evaluate_alloy_elastic_properties,
)


def test_cubic_vrh_elastic_moduli_nickel():
    """Verify cubic VRH polycrystalline moduli for pure Nickel."""
    # Ni single crystal: C11 = 247 GPa, C12 = 147 GPa, C44 = 125 GPa
    res = calculate_voigt_reuss_hill_cubic(c11=247.0, c12=147.0, c44=125.0)

    assert "bulk_modulus_K_GPa" in res
    assert "shear_modulus_G_GPa" in res
    assert "youngs_modulus_E_GPa" in res
    assert "poissons_ratio_nu" in res

    # Check bounds: B = (247 + 2*147)/3 = 180.33 GPa
    assert np.isclose(res["bulk_modulus_K_GPa"], 180.333, atol=0.1)
    assert 70.0 < res["shear_modulus_G_GPa"] < 100.0
    assert 180.0 < res["youngs_modulus_E_GPa"] < 250.0
    assert 0.28 < res["poissons_ratio_nu"] < 0.35
    assert res["ductility_predicted"] == "Ductile"


def test_hexagonal_vrh_elastic_moduli_titanium():
    """Verify hexagonal VRH polycrystalline moduli for pure alpha-Ti."""
    # Ti single crystal: C11 = 162 GPa, C12 = 92 GPa, C13 = 69 GPa, C33 = 181 GPa, C44 = 47 GPa
    res = calculate_voigt_reuss_hill_hexagonal(
        c11=162.0, c12=92.0, c13=69.0, c33=181.0, c44=47.0
    )

    assert 100.0 < res["bulk_modulus_K_GPa"] < 120.0
    assert 35.0 < res["shear_modulus_G_GPa"] < 55.0
    assert 100.0 < res["youngs_modulus_E_GPa"] < 130.0
    assert 0.28 < res["poissons_ratio_nu"] < 0.36


def test_evaluate_alloy_elastic_properties_multicomponent():
    """Verify alloy elasticity evaluation across temperature."""
    res_298 = evaluate_alloy_elastic_properties(
        bulk_mole_fractions={"NI": 0.60, "CR": 0.20, "FE": 0.15, "AL": 0.05},
        temperature=298.15,
        crystal_structure="cubic"
    )
    res_800 = evaluate_alloy_elastic_properties(
        bulk_mole_fractions={"NI": 0.60, "CR": 0.20, "FE": 0.15, "AL": 0.05},
        temperature=800.0,
        crystal_structure="cubic"
    )

    e_298 = res_298["youngs_modulus_E_GPa"]
    e_800 = res_800["youngs_modulus_E_GPa"]

    assert e_298 > 150.0
    assert e_298 > e_800  # Elastic modulus softens with temperature
