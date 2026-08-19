import os
import tempfile
import pytest
import numpy as np
from pycalphad import Database, calculate
from pipeline.model_architect import ModelArchitect
from pipeline.parameter_fitter import RedlichKisterParameterFitter


def test_model_architect_phase_generation():
    architect = ModelArchitect()
    res = architect.generate_system_phase_models(
        elements=["Al", "Ni"],
        include_phases=["LIQUID", "FCC_A1", "BCC_A2", "GAMMA_PRIME"],
    )

    assert "components" in res
    assert "AL" in res["components"] and "NI" in res["components"] and "VA" in res["components"]
    assert "LIQUID" in res["phases"]
    assert "GAMMA_PRIME" in res["phases"]
    assert res["phases"]["GAMMA_PRIME"]["sublattice_model"]["sublattice_site_ratios"] == [3.0, 1.0, 1.0]


def test_model_architect_base_tdb():
    architect = ModelArchitect()
    with tempfile.TemporaryDirectory() as tmpdir:
        tdb_path = os.path.join(tmpdir, "base_alni.tdb")
        db = architect.generate_unassessed_base_tdb(
            elements=["Al", "Ni"],
            phases=["LIQUID", "FCC_A1"],
            output_tdb_path=tdb_path,
        )

        assert os.path.exists(tdb_path)
        assert "AL" in db.elements and "NI" in db.elements
        assert "LIQUID" in db.phases and "FCC_A1" in db.phases


def test_redlich_kister_parameter_fitting():
    fitter = RedlichKisterParameterFitter()

    # Generate synthetic subregular mixing data:
    # L0 = -40000 J/mol, L1 = 10000 J/mol
    x_vals = np.linspace(0.1, 0.9, 9)
    true_l0 = -40000.0
    true_l1 = 10000.0
    hm_data = x_vals * (1.0 - x_vals) * (true_l0 + true_l1 * (x_vals - (1.0 - x_vals)))

    fit_res = fitter.fit_binary_mixing_enthalpy(
        element_a="AL",
        element_b="NI",
        phase_name="LIQUID",
        mole_fraction_a=x_vals,
        hm_mix_j_mol=hm_data,
        max_degree=3,
    )

    assert fit_res.order == 1  # Should select order 1 via BIC
    assert fit_res.r2_score > 0.999
    assert pytest.approx(fit_res.l_coefficients[0][0], rel=1e-2) == true_l0
    assert pytest.approx(fit_res.l_coefficients[1][0], rel=1e-2) == true_l1

    # Apply fit to unassessed base database
    architect = ModelArchitect()
    with tempfile.TemporaryDirectory() as tmpdir:
        tdb_path = os.path.join(tmpdir, "fitted_liquid.tdb")
        db = architect.generate_unassessed_base_tdb(["AL", "NI"], ["LIQUID"], tdb_path)
        fitter.apply_fit_to_database(db, fit_res)

        # Run forward calculation with PyCALPHAD
        res = calculate(db, ["AL", "NI"], "LIQUID", T=1500.0)
        assert res.GM.values is not None
        assert np.all(np.isfinite(res.GM.values))
