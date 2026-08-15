"""
Unit tests for Noble Metals, CIELAB Colors, and Age-Hardening models.
"""

import numpy as np
import pytest
from pycalphad.noble_metals import (
    calculate_gold_alloy_color_and_karat,
    calculate_precious_alloy_age_hardening,
)


def test_gold_alloy_color_and_karat():
    """Verify 18K yellow, rose, and white gold color coordinates and classification."""
    # 18K Yellow Gold: 75% Au, 15% Ag, 10% Cu
    yellow_18k = calculate_gold_alloy_color_and_karat({"AU": 75.0, "AG": 15.0, "CU": 10.0})
    assert np.isclose(yellow_18k["karat_rating"], 18.0)
    assert yellow_18k["alloy_color_classification"] == "Classic Yellow Gold"

    # 18K Rose Gold: 75% Au, 5% Ag, 20% Cu
    rose_18k = calculate_gold_alloy_color_and_karat({"AU": 75.0, "AG": 5.0, "CU": 20.0})
    assert rose_18k["CIE_a_star_red_green"] > yellow_18k["CIE_a_star_red_green"]  # higher redness (+a*)
    assert "Rose" in rose_18k["alloy_color_classification"]

    # 18K White Gold: 75% Au, 15% Pd, 10% Ag
    white_18k = calculate_gold_alloy_color_and_karat({"AU": 75.0, "PD": 15.0, "AG": 10.0})
    assert white_18k["alloy_color_classification"] == "White Gold"


def test_precious_alloy_age_hardening():
    """Verify age-hardening hardness evolution in Au-Cu-Ag and Pt alloys."""
    res_aged = calculate_precious_alloy_age_hardening(
        alloy_type="18K_AU_CU",
        aging_temperature_celsius=300.0,
        aging_time_hours=2.0
    )
    assert res_aged["aged_hardness_HV"] > res_aged["annealed_hardness_HV"]
    assert res_aged["hardness_increase_HV"] > 50.0
