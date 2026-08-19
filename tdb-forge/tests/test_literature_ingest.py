import os
import tempfile
import json
import pytest
from pipeline.literature_ingest import (
    PhaseNomenclatureResolver,
    UnitAndCompositionNormalizer,
    LiteratureIngestionBridge,
)


def test_phase_nomenclature_resolver():
    assert PhaseNomenclatureResolver.resolve("liquid") == "LIQUID"
    assert PhaseNomenclatureResolver.resolve("gamma_prime") == "GAMMA_PRIME"
    assert PhaseNomenclatureResolver.resolve("L12") == "GAMMA_PRIME"
    assert PhaseNomenclatureResolver.resolve("FCC_A1") == "FCC_A1"
    assert PhaseNomenclatureResolver.resolve("austenite") == "FCC_A1"
    assert PhaseNomenclatureResolver.resolve("b2") == "BCC_B2"
    assert PhaseNomenclatureResolver.resolve("c14") == "LAVES_C14"


def test_unit_and_composition_normalizer():
    # Temperature conversions
    assert UnitAndCompositionNormalizer.to_kelvin(100.0, "C") == 373.15
    assert UnitAndCompositionNormalizer.to_kelvin(212.0, "F") == 373.15

    # Energy conversions
    assert UnitAndCompositionNormalizer.to_joules_per_mole(5.0, "kJ/mol") == 5000.0
    assert pytest.approx(UnitAndCompositionNormalizer.to_joules_per_mole(1.0, "kcal/mol"), abs=0.1) == 4184.0

    # Weight to mole fractions (Ti-6Al-4V nominal: 90% Ti, 6% Al, 4% V)
    wt = {"Ti": 0.90, "Al": 0.06, "V": 0.04}
    at = UnitAndCompositionNormalizer.weight_to_mole_fractions(wt)
    assert pytest.approx(sum(at.values()), abs=1e-3) == 1.0
    assert "TI" in at and "AL" in at and "V" in at
    assert at["TI"] > 0.80


def test_literature_ingestion_bridge():
    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = LiteratureIngestionBridge(output_dir=tmpdir)

        # 1. Test ZPF Dataset Creation
        zpf_dataset = bridge.build_zpf_dataset(
            components=["AL", "NI"],
            phases=["LIQUID", "FCC_A1"],
            temperatures_k=[1200.0, 1300.0],
            tie_line_values=[
                [["LIQUID", ["AL"], [0.15]], ["FCC_A1", ["AL"], [0.08]]],
                [["LIQUID", ["AL"], [0.22]], ["FCC_A1", ["AL"], [0.11]]],
            ],
            reference="Dupin et al. (2001)",
            doi="10.1007/s11669-001-0012-3",
        )

        zpf_path = os.path.join(bridge.fitting_dir, "zpf_al_ni.json")
        zpf_dataset.save(zpf_path)
        assert os.path.exists(zpf_path)

        with open(zpf_path, "r") as f:
            data = json.load(f)
        assert data["output"] == "ZPF"
        assert len(data["values"]) == 2
        assert data["provenance"]["tier"] == "tier1_primary_fitting"

        # 2. Test MatWeb Holdout Ingestion
        holdout_path = bridge.ingest_matweb_holdout_datasheet(
            alloy_name="Inconel 718",
            composition_wt={"Ni": 0.525, "Cr": 0.19, "Fe": 0.185, "Nb": 0.05, "Mo": 0.03, "Ti": 0.01, "Al": 0.01},
            liquidus_c=1336.0,
            solidus_c=1260.0,
            source_key="MATWEB-INCONEL-718",
        )
        assert os.path.exists(holdout_path)
        with open(holdout_path, "r") as f:
            hw_data = json.load(f)
        assert hw_data["liquidus_C"] == 1336.0
        assert hw_data["provenance"]["tier"] == "tier3_holdout_validation"
