"""
Unit tests for Liquid Surface Tension (Butler Equation) and Dynamic Viscosity models.
"""

import numpy as np
import pytest
from pycalphad.models.model_transport import (
    get_pure_liquid_surface_tension,
    solve_butler_surface_tension,
    calculate_liquid_dynamic_viscosity,
)


def test_pure_element_surface_tensions():
    """Verify pure liquid surface tension values and negative temperature coefficient."""
    sigma_ni_1750, _ = get_pure_liquid_surface_tension("NI", temperature=1750.0)
    sigma_ni_1900, _ = get_pure_liquid_surface_tension("NI", temperature=1900.0)

    assert sigma_ni_1750 > 1.0
    assert sigma_ni_1750 > sigma_ni_1900  # Surface tension decreases with increasing temperature


def test_butler_surface_tension_binary():
    """Verify Butler equation solution for binary Ni-Al liquid alloy."""
    res = solve_butler_surface_tension(
        bulk_mole_fractions={"NI": 0.80, "AL": 0.20},
        temperature=1800.0,
        excess_gibbs_interactions={("AL", "NI"): -50000.0}
    )

    assert "surface_tension" in res
    assert "surface_fractions" in res
    sigma = res["surface_tension"]
    assert 0.5 < sigma < 2.5
    # Lower surface energy element (Al) typically segregates to the surface
    x_s_al = res["surface_fractions"]["AL"]
    assert x_s_al >= 0.20  # Surface enrichment of lower-surface-tension constituent


def test_liquid_dynamic_viscosity():
    """Verify Eyring-Kaptay dynamic viscosity calculation."""
    res_1750 = calculate_liquid_dynamic_viscosity(
        bulk_mole_fractions={"NI": 0.70, "CR": 0.20, "AL": 0.10},
        temperature=1750.0,
        molar_volume=7.5e-6
    )
    res_1950 = calculate_liquid_dynamic_viscosity(
        bulk_mole_fractions={"NI": 0.70, "CR": 0.20, "AL": 0.10},
        temperature=1950.0,
        molar_volume=7.7e-6
    )

    eta_1750 = res_1750["dynamic_viscosity_mPa_s"]
    eta_1950 = res_1950["dynamic_viscosity_mPa_s"]

    assert 1.0 < eta_1750 < 20.0  # Typical liquid metal viscosity is ~2-10 mPa*s
    assert eta_1750 > eta_1950     # Viscosity decreases with temperature
