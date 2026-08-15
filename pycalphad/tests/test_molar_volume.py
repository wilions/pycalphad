"""
Unit tests for Molar Volume and Thermal Expansion Model.
"""

import pytest
from pycalphad import Database, variables as v
from pycalphad.models.model_molar_volume import MolarVolumeModel, compute_molar_volume_for_phase

TEST_VOL_TDB = """
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01 !
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00 !
 SPECIES AL   AL1 !
 SPECIES VA   VA1 !

 FUNCTION GHSERAL    2.98150E+02 -7976.15+137.093038*T-24.3671976*T*LN(T)
  -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 6000.0 N !

 TYPE_DEFINITION % SEQ * !
 PHASE FCC_A1 % 1  1.0 !
 CONSTITUENT FCC_A1 :AL: !

 PARAMETER G(FCC_A1,AL;0) 298.15 +GHSERAL#; 6000.0 N !
 PARAMETER V0(FCC_A1,AL;0) 298.15 1.0e-5; 6000.0 N !
 PARAMETER VA(FCC_A1,AL;0) 298.15 2.3e-5*T; 6000.0 N !
"""


def test_molar_volume_model_instantiation():
    """Verify that MolarVolumeModel builds V0, VA, VM, and CTE."""
    db = Database(TEST_VOL_TDB)
    m = MolarVolumeModel(db, ["AL", "VA"], "FCC_A1")
    assert m.V0 != 0
    assert m.VA != 0
    assert m.VM != 0
    assert m.CTE != 0


def test_compute_molar_volume_for_phase():
    """Verify compute_molar_volume_for_phase function."""
    db = Database(TEST_VOL_TDB)
    vol_298 = compute_molar_volume_for_phase(db, ["AL", "VA"], "FCC_A1", temperature=298.15)
    vol_500 = compute_molar_volume_for_phase(db, ["AL", "VA"], "FCC_A1", temperature=500.0)

    assert vol_298 > 0
    assert vol_500 > vol_298  # Thermal expansion causes volume increase
