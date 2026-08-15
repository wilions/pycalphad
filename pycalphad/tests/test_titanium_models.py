"""
Unit tests for Titanium Phase Boundaries, Beta-Transus, and Martensite models.
"""

import numpy as np
import pytest
from pycalphad.titanium import (
    calculate_molybdenum_and_aluminum_equivalence,
    calculate_beta_transus_temperature,
    calculate_titanium_martensite_start,
)


def test_ti64_beta_transus_and_equivalences():
    """Verify Ti-6Al-4V beta-transus temperature and Mo/Al equivalences."""
    # Ti-6Al-4V (Grade 5): 6.0% Al, 4.0% V, 0.18% O, 0.02% C, 0.01% N, 0.15% Fe
    ti64 = {"AL": 6.0, "V": 4.0, "O": 0.18, "C": 0.02, "N": 0.01, "FE": 0.15}
    eq_res = calculate_molybdenum_and_aluminum_equivalence(ti64)
    transus_res = calculate_beta_transus_temperature(ti64)

    assert "Alpha + Beta" in eq_res["alloy_classification"]
    assert 970.0 < transus_res["T_beta_celsius"] < 1030.0  # Ti-64 transus is ~995-1005 °C


def test_titanium_martensite_modes():
    """Verify hexagonal vs orthorhombic martensite in titanium alloys."""
    # Ti-64 forms hexagonal alpha'
    ti64 = {"AL": 6.0, "V": 4.0, "FE": 0.15}
    m_ti64 = calculate_titanium_martensite_start(ti64)
    assert m_ti64["is_martensitic"] is True
    assert "Hexagonal" in m_ti64["martensite_type"]

    # Beta-rich alloy (Ti-10V-2Fe-3Al) retains beta or forms alpha''
    ti1023 = {"AL": 3.0, "V": 10.0, "FE": 2.0}
    m_ti1023 = calculate_titanium_martensite_start(ti1023)
    assert "Orthorhombic" in m_ti1023["martensite_type"] or "Retained" in m_ti1023["martensite_type"]
