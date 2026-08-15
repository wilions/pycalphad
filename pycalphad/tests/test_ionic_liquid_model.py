"""
Unit tests for Two-Sublattice Ionic Liquid model and slag optical basicity.
"""

import numpy as np
import pytest
from pycalphad import Database, variables as v
from pycalphad.models.model_ionic_liquid import (
    ModelIonicLiquid2SL,
    calculate_slag_basicity_index,
)

TEST_IONIC_TDB = """
 ELEMENT CA   FCC_A1                    4.0080E+01  0.0000E+00  0.0000E+00 !
 ELEMENT SI   DIAMOND                   2.8085E+01  0.0000E+00  0.0000E+00 !
 ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  0.0000E+00  0.0000E+00 !
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00 !
 SPECIES CA+2 CA1/+2 !
 SPECIES SIO4-4 SI1O4/-4 !
 SPECIES O-2  O1/-2 !
 SPECIES SIO2 SIO2 !
 SPECIES VA   VA1 !

 TYPE_DEFINITION % SEQ * !
 PHASE IONIC_LIQUID % 2  1.0  1.0 !
 CONSTITUENT IONIC_LIQUID :CA+2:SIO4-4,O-2,SIO2,VA: !

 PARAMETER G(IONIC_LIQUID,CA+2:O-2;0) 298.15 -635000+100.0*T; 6000.0 N !
"""


def test_ionic_liquid_model_instantiation():
    """Verify Two-Sublattice Ionic Liquid model initialization."""
    db = Database(TEST_IONIC_TDB)
    m = ModelIonicLiquid2SL(db, ["CA", "SI", "O", "VA"], "IONIC_LIQUID")
    charges = m.calculate_site_fractions_and_charges()

    assert "Q_charge" in charges
    assert "P_sites" in charges
    assert len(charges["cations"]) >= 1
    assert len(charges["anions_neutrals"]) >= 1

    g_id = m.ideal_mixing_energy_ionic()
    assert g_id != 0


def test_slag_optical_basicity_calculation():
    """Verify optical basicity and V-ratio calculation for basic vs acidic slags."""
    # Basic slag: High CaO
    basic_slag = calculate_slag_basicity_index({"CAO": 0.50, "SIO2": 0.35, "AL2O3": 0.10, "MGO": 0.05})
    assert basic_slag["optical_basicity"] > 0.65
    assert basic_slag["slag_character"] == "Basic"

    # Acidic slag: High SiO2
    acidic_slag = calculate_slag_basicity_index({"CAO": 0.20, "SIO2": 0.65, "AL2O3": 0.15})
    assert acidic_slag["optical_basicity"] < 0.60
    assert acidic_slag["slag_character"] == "Acidic"
