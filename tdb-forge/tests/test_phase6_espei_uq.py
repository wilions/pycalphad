"""
Unit tests for Phase 6: Autonomous ESPEI Pipeline, Uncertainty Quantification, and MPEA Validator.
"""

import os
import json
import pytest
import numpy as np
from pipeline.espei_runner import ESPEIPipelineRunner
from pipeline.uncertainty import PropagateParameterUncertainty
from pipeline.mpea_validator import MPEADatabaseValidator

TEST_DIR = os.path.dirname(__file__)
TDB_DIR = os.path.abspath(os.path.join(TEST_DIR, '..', 'tdbs'))

def test_espei_pipeline_runner(tmp_path):
    spec_file = tmp_path / "assessment_spec.json"
    spec_content = {
        "system": "Al-Sc",
        "phase_models_file": "phase_models.json"
    }
    spec_file.write_text(json.dumps(spec_content), encoding='utf-8')
    
    runner = ESPEIPipelineRunner(str(spec_file), str(tmp_path / "output"))
    
    # Generate config
    config_path = runner.generate_espei_config(str(tmp_path / "datasets"), "Al-Sc")
    assert os.path.exists(config_path)

def test_uncertainty_quantification(tmp_path):
    tdb_path = os.path.join(TDB_DIR, "AM_Al_thermo.tdb")
    if not os.path.exists(tdb_path):
        pytest.skip(f"TDB file {tdb_path} not found")

    params_mean = {"L_AL_SC_0": -120000.0, "L_AL_SC_1": 15000.0}
    uq = PropagateParameterUncertainty(tdb_path, params_mean)
    
    samples = uq.sample_parameters_latin_hypercube(num_samples=10)
    assert len(samples) == 10
    assert "L_AL_SC_0" in samples[0]

def test_mpea_database_validator():
    tdb_path = os.path.join(TDB_DIR, "TiZrHfNb_RHEA.tdb")
    if not os.path.exists(tdb_path):
        pytest.skip(f"TDB file {tdb_path} not found")

    validator = MPEADatabaseValidator(tdb_path)
    res = validator.validate_full_mpea()
    assert res["all_passed"] is True
    assert "TI" in res["elements"]
