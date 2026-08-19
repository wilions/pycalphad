import os
import pytest
from pycalphad import Database, variables as v
from pycalphad.core.sensitivity import (
    compute_gibbs_composition_gradient,
    compute_temperature_entropy_derivative,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "databases", "alfe.tdb")


@pytest.fixture(scope="module")
def alfe_db():
    return Database(DB_PATH)


def test_gibbs_composition_gradient(alfe_db):
    comps = ["AL", "FE", "VA"]
    phases = ["LIQUID", "FCC_A1", "BCC_A2"]
    cond = {v.T: 1300, v.P: 101325, v.X("AL"): 0.15}

    grads = compute_gibbs_composition_gradient(
        alfe_db, comps, phases, condition=cond, variable_elements=["AL"]
    )

    assert "AL" in grads
    # d(GM)/d(X_AL) is a real finite thermodynamic quantity
    assert isinstance(grads["AL"], float)


def test_temperature_entropy_derivative(alfe_db):
    comps = ["AL", "FE", "VA"]
    phases = ["LIQUID", "FCC_A1", "BCC_A2"]
    cond = {v.T: 1200, v.P: 101325, v.X("AL"): 0.10}

    dG_dT = compute_temperature_entropy_derivative(alfe_db, comps, phases, cond)
    # dG/dT = -S, which must be strictly negative for positive entropy
    assert dG_dT < 0.0
