"""
Unit tests for CALPHAD-IR JSON serialization, deserialization, and schema validation.
"""

import json
from io import StringIO
import pytest
from pycalphad import Database, variables as v
from pycalphad.io.calphad_ir import (
    CALPHAD_IR_SCHEMA_V1,
    CALPHAD_IR_SCHEMA_VERSION,
    database_to_calphad_ir,
    calphad_ir_to_database,
)
from pycalphad.variables import Species

TEST_TDB = """
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01 !
 ELEMENT NI   FCC_A1                    5.8690E+01  4.7865E+03  2.9790E+01 !
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00 !
 SPECIES AL   AL1 !
 SPECIES NI   NI1 !
 SPECIES VA   VA1 !

 FUNCTION GHSERAL    2.98150E+02 -7976.15+137.093038*T-24.3671976*T*LN(T)
  -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 7.00000E+02  Y
  -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
  -5.764227E-06*T**3+74092*T**(-1); 9.33470E+02  Y
  -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);
  2.90000E+03  N !

 FUNCTION GHSERNI    2.98150E+02 -5179.066+117.108792*T-22.096*T*LN(T)
  -.0048407*T**2; 1.72800E+03  Y
  -27840.711+279.135435*T-43.1*T*LN(T)+1.12754E+31*T**(-9);
  3.00000E+03  N !

 TYPE_DEFINITION % SEQ * !
 PHASE FCC_A1 % 1  1.0 !
 CONSTITUENT FCC_A1 :AL,NI: !

 PARAMETER G(FCC_A1,AL;0) 298.15 +GHSERAL#; 2900.0 N !
 PARAMETER G(FCC_A1,NI;0) 298.15 +GHSERNI#; 3000.0 N !
 PARAMETER L(FCC_A1,AL,NI;0) 298.15 -145000+35.0*T; 6000.0 N !
 PARAMETER V0(FCC_A1,AL;0) 298.15 1.0e-5; 6000.0 N !
 PARAMETER VA(FCC_A1,AL;0) 298.15 2.3e-5*T; 6000.0 N !
 PARAMETER C11(FCC_A1,AL;0) 298.15 1.08e11 - 1.2e7*T; 6000.0 N !
 PARAMETER C12(FCC_A1,AL;0) 298.15 0.62e11 - 0.8e7*T; 6000.0 N !
 PARAMETER C44(FCC_A1,AL;0) 298.15 0.28e11 - 0.4e7*T; 6000.0 N !
"""


def test_tdb_property_parameter_parsing():
    """Verify that extended property parameters (V0, VA, C11, C12, C44) parse without error."""
    db = Database(TEST_TDB)
    assert "AL" in db.elements
    assert "NI" in db.elements
    assert "FCC_A1" in db.phases
    params = db._parameters.all()
    param_types = {p["parameter_type"] for p in params}
    assert "G" in param_types
    assert "L" in param_types
    assert "V0" in param_types
    assert "VA" in param_types
    assert "C11" in param_types
    assert "C12" in param_types
    assert "C44" in param_types


def test_database_to_calphad_ir_structure():
    """Verify that database_to_calphad_ir generates a valid CALPHAD-IR dictionary."""
    db = Database(TEST_TDB)
    ir = database_to_calphad_ir(db, database_name="Test-Al-Ni")

    assert ir["calphad_ir_version"] == CALPHAD_IR_SCHEMA_VERSION
    assert ir["metadata"]["database_name"] == "Test-Al-Ni"
    assert len(ir["elements"]) >= 2
    assert "FCC_A1" in ir["phases"]
    assert len(ir["parameters"]) >= 7


def test_calphad_ir_roundtrip():
    """Verify that converting Database -> CALPHAD-IR -> Database preserves all essential elements and parameters."""
    db_orig = Database(TEST_TDB)
    ir_dict = db_orig.to_calphad_ir()
    
    # Verify JSON serializability
    json_str = json.dumps(ir_dict, indent=2)
    assert len(json_str) > 0

    # Reconstruct from JSON
    db_reconstructed = Database.from_calphad_ir(json.loads(json_str))

    assert db_reconstructed.elements == db_orig.elements
    assert set(db_reconstructed.phases.keys()) == set(db_orig.phases.keys())
    assert len(db_reconstructed._parameters.all()) == len(db_orig._parameters.all())

    orig_param_types = sorted(p["parameter_type"] for p in db_orig._parameters.all())
    recon_param_types = sorted(p["parameter_type"] for p in db_reconstructed._parameters.all())
    assert orig_param_types == recon_param_types


def test_database_json_io_format_registration():
    """Verify Database.to_string(fmt='json') and Database.from_string(..., fmt='json')."""
    db_orig = Database(TEST_TDB)
    json_out = db_orig.to_string(fmt="json")
    assert "calphad_ir_version" in json_out
    
    db_from_json = Database.from_string(json_out, fmt="json")
    assert "FCC_A1" in db_from_json.phases
    assert "AL" in db_from_json.elements
