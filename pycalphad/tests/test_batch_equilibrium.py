import os
import pytest
import numpy as np
from pycalphad import Database, variables as v, equilibrium
from pycalphad.core.batch_equilibrium import batch_equilibrium, evaluate_composition_grid

DB_PATH = os.path.join(os.path.dirname(__file__), "databases", "alfe.tdb")


@pytest.fixture(scope="module")
def alfe_db():
    return Database(DB_PATH)


def test_batch_equilibrium_consistency(alfe_db):
    comps = ["AL", "FE", "VA"]
    phases = ["LIQUID", "FCC_A1", "BCC_A2"]
    conds_list = [
        {v.T: 1200, v.P: 101325, v.X("AL"): 0.1},
        {v.T: 1300, v.P: 101325, v.X("AL"): 0.2},
        {v.T: 1400, v.P: 101325, v.X("AL"): 0.3},
    ]

    # Run batched
    batch_res = batch_equilibrium(alfe_db, comps, phases, conds_list, max_workers=2)
    assert len(batch_res) == 3

    # Compare with individual sequential calls
    for cond, b_ds in zip(conds_list, batch_res):
        s_ds = equilibrium(alfe_db, comps, phases, cond)
        np.testing.assert_allclose(b_ds.GM.values, s_ds.GM.values, rtol=1e-4, atol=1e-4)
        assert b_ds.attrs.get("converged", False) is True


def test_evaluate_composition_grid(alfe_db):
    comps = ["AL", "FE", "VA"]
    phases = ["LIQUID", "FCC_A1", "BCC_A2"]
    temps = [1100.0, 1300.0]
    compositions = [{"AL": 0.15, "FE": 0.85}, {"AL": 0.25, "FE": 0.75}]

    grid_res = evaluate_composition_grid(
        alfe_db, comps, phases, temperatures=temps, compositions=compositions, max_workers=2
    )

    assert len(grid_res) == 4
    for entry in grid_res:
        assert "stable_phases" in entry
        assert len(entry["stable_phases"]) > 0
        assert "phase_fractions" in entry
        total_frac = sum(entry["phase_fractions"].values())
        assert pytest.approx(total_frac, abs=1e-2) == 1.0
