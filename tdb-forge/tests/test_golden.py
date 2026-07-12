import os
import pytest
import numpy as np
from pycalphad import Database, calculate
from pipeline.compile import compile_file

GOLDEN_TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(GOLDEN_TEST_DIR, '..')

def test_golden_alzn_mey():
    # Paths
    json_path = os.path.join(PROJECT_ROOT, 'extracted/alzn_mey.json')
    compiled_tdb_path = os.path.join(PROJECT_ROOT, 'tdbs/alzn_mey_compiled.tdb')
    ref_tdb_path = os.path.abspath(os.path.join(PROJECT_ROOT, '../pycalphad/tests/databases/alzn_mey.tdb'))
    
    # 1. Compile JSON to TDB
    compile_file(json_path, compiled_tdb_path)
    
    # 2. Load compiled and reference databases
    db_compiled = Database(compiled_tdb_path)
    db_ref = Database(ref_tdb_path)
    
    # 3. Compare elements and phases
    # Note: compiled database may contain extra unary elements from SGTE_pure_elements.tdb (like FE, NI, ZR, etc.)
    # We check that the elements of interest are present in both
    for el in ['AL', 'ZN', 'VA']:
        assert el in db_compiled.elements
        assert el in db_ref.elements
        
    for phase in ['LIQUID', 'FCC_A1', 'HCP_A3']:
        assert phase in db_compiled.phases
        assert phase in db_ref.phases
        
    # 4. Compare thermodynamics by calculating properties on a grid
    for phase in ['LIQUID', 'FCC_A1', 'HCP_A3']:
        for T in [300.0, 700.0, 1000.0]:
            # Compiled calculation
            res_compiled = calculate(db_compiled, ['AL', 'ZN'], phase, T=T)
            gm_compiled = res_compiled.GM.values
            
            # Reference calculation
            res_ref = calculate(db_ref, ['AL', 'ZN'], phase, T=T)
            gm_ref = res_ref.GM.values
            
            # Assert they are numerically identical within tolerance of unary discrepancies
            np.testing.assert_allclose(gm_compiled, gm_ref, rtol=1e-3, atol=10.0, err_msg=f"Thermodynamic discrepancy in {phase} at T={T}")
            
    print("Sabine an Mey Al-Zn Golden Test passed successfully!")

def test_golden_alni_dupin():
    # Paths
    json_path = os.path.join(PROJECT_ROOT, 'extracted/alni_dupin_2001.json')
    compiled_tdb_path = os.path.join(PROJECT_ROOT, 'tdbs/alni_dupin_2001_compiled.tdb')
    ref_tdb_path = os.path.abspath(os.path.join(PROJECT_ROOT, '../examples/databases/NI_AL_DUPIN_2001.TDB'))
    
    # 1. Compile JSON to TDB
    compile_file(json_path, compiled_tdb_path)
    
    # 2. Load compiled and reference databases
    db_compiled = Database(compiled_tdb_path)
    db_ref = Database(ref_tdb_path)
    
    # 3. Compare elements and phases
    for el in ['AL', 'NI', 'VA']:
        assert el in db_compiled.elements
        assert el in db_ref.elements
        
    target_phases = ['LIQUID', 'AL3NI1', 'AL3NI2', 'AL3NI5', 'BCC_A2', 'BCC_B2', 'FCC_A1', 'FCC_L12']
    for phase in target_phases:
        assert phase in db_compiled.phases
        assert phase in db_ref.phases
        
    # 4. Compare thermodynamics
    for phase in target_phases:
        for T in [300.0, 700.0, 1000.0]:
            # Compiled calculation
            res_compiled = calculate(db_compiled, ['AL', 'NI', 'VA'], phase, T=T)
            gm_compiled = res_compiled.GM.values
            
            # Reference calculation
            res_ref = calculate(db_ref, ['AL', 'NI', 'VA'], phase, T=T)
            gm_ref = res_ref.GM.values
            
            # Assert they are numerically identical within tolerance of unary discrepancies
            np.testing.assert_allclose(gm_compiled, gm_ref, rtol=1e-3, atol=10.0, err_msg=f"Thermodynamic discrepancy in {phase} at T={T}")
            
    print("Dupin Ni-Al Golden Test passed successfully!")

