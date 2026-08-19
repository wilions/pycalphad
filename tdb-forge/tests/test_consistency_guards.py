import os
import tempfile
import json
import pytest
from pycalphad import Database
from pipeline.consistency_guards import ThermodynamicConsistencyGuard, ConsistencyReport
from pipeline.literature_ingest import LiteratureIngestionBridge
from pipeline.model_architect import ModelArchitect


def test_consistency_guard_scan():
    guard = ThermodynamicConsistencyGuard()
    architect = ModelArchitect()

    with tempfile.TemporaryDirectory() as tmpdir:
        tdb_path = os.path.join(tmpdir, "test_alni.tdb")
        db = architect.generate_unassessed_base_tdb(["AL", "NI"], ["LIQUID", "FCC_A1"], tdb_path)

        is_consistent, s_ok, cp_ok, anomalies = guard.scan_phase_stability_and_derivatives(
            db, ["AL", "NI"], ["LIQUID", "FCC_A1"], t_min=500.0, t_max=3000.0, t_steps=5
        )

        assert s_ok is True
        assert is_consistent is True


def test_consistency_guard_holdout_benchmark():
    guard = ThermodynamicConsistencyGuard()
    bridge = LiteratureIngestionBridge(output_dir=tempfile.gettempdir())

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create holdout JSON
        holdout_path = bridge.ingest_matweb_holdout_datasheet(
            alloy_name="Inconel 625",
            composition_wt={"Ni": 0.60, "Cr": 0.22, "Mo": 0.09, "Nb": 0.04, "Fe": 0.05},
            liquidus_c=1350.0,
            solidus_c=1290.0,
            source_key="TEST-INCONEL-625",
        )

        architect = ModelArchitect()
        tdb_path = os.path.join(tmpdir, "inconel.tdb")
        db = architect.generate_unassessed_base_tdb(["NI", "CR"], ["LIQUID"], tdb_path)

        report = guard.generate_full_consistency_report(
            tdb_path=tdb_path,
            holdout_dir=os.path.dirname(holdout_path),
        )

        assert isinstance(report, ConsistencyReport)
        assert report.is_thermodynamically_consistent is True
        assert len(report.holdout_benchmarks) >= 1
        assert report.holdout_benchmarks[0].alloy_name == "Inconel 625"
