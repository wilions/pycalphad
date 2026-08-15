"""
Unit tests for the Additive Manufacturing Solidification Cracking and Hot Tearing suite.
"""

import numpy as np
import pytest
from pycalphad.am.cracking import (
    calculate_kou_index,
    calculate_clyne_davis_index,
    calculate_rdg_index,
    calculate_freezing_range,
    calculate_terminal_freezing_range,
    classify_cracking_risk,
    evaluate_cracking_susceptibility,
    SolidificationCrackingAssessment,
)


def test_kou_and_csc_cracking_indices():
    """Test quantitative calculation of Kou and Clyne-Davis cracking indices."""
    f_s = np.linspace(0.0, 1.0, 101)
    # Synthetic cooling curve where temperature drops precipitously near f_s -> 1.0 (vulnerable region)
    temperatures = 1000.0 - 200.0 * f_s - 300.0 * (f_s ** 4)

    kou = calculate_kou_index(temperatures, f_s)
    csc = calculate_clyne_davis_index(temperatures, f_s)
    rdg = calculate_rdg_index(temperatures, f_s)
    fr = calculate_freezing_range(temperatures, f_s)
    tfr = calculate_terminal_freezing_range(temperatures, f_s)

    assert kou > 0.0
    assert csc > 0.0
    assert rdg > 0.0
    assert fr > 0.0
    assert tfr > 0.0


def test_cracking_risk_classification():
    """Verify cracking risk categories for benign vs high-cracking alloys."""
    assert classify_cracking_risk(50.0, 0.2) == "Low"
    assert classify_cracking_risk(250.0, 0.8) == "Moderate"
    assert classify_cracking_risk(500.0, 1.8) == "High"
    assert classify_cracking_risk(900.0, 3.5) == "Severe"


def test_evaluate_cracking_susceptibility_assessment():
    """Verify that evaluate_cracking_susceptibility produces a full dataclass assessment."""
    f_s = np.linspace(0.0, 1.0, 50)
    temps = 1800.0 - 500.0 * np.sqrt(f_s)

    assessment = evaluate_cracking_susceptibility(temps, f_s)
    assert isinstance(assessment, SolidificationCrackingAssessment)
    summary = assessment.summary()
    assert "kou_index" in summary
    assert "clyne_davis_index" in summary
    assert "risk_category" in summary
    assert summary["risk_category"] in ["Low", "Moderate", "High", "Severe"]
