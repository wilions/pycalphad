"""
Golden Benchmark & ICME Modernization Test Suite.

Tests thermodynamic consistency verification, multi-pass AM thermal history simulation,
and multi-phase KWN co-precipitation kinetics.
"""

import os
import numpy as np
import pytest
from pycalphad import Database, variables as v
from pycalphad.io.consistency import (
    verify_thermodynamic_consistency,
    verify_sublattice_integrity,
    CheckSeverity,
)
from pycalphad.am.thermal_history import (
    MultiPassThermalAccumulator,
    ScanTrack,
)
from pycalphad.precipitation.coprecipitation import (
    MultiPhasePrecipitationKinetics,
    PrecipitatePhaseSpec,
)


def test_thermodynamic_consistency_alzn():
    """Verify consistency checking on the standard Al-Zn database."""
    dbf_path = os.path.join(os.path.dirname(__file__), "databases", "alzn_mey.tdb")
    dbf = Database(dbf_path)
    
    report = dbf.verify_consistency(check_cp=True, check_sublattices=True)
    assert report.is_valid is True
    assert report.num_errors == 0
    summary = report.summary()
    assert "VALID" in summary


def test_thermodynamic_consistency_mismatch_detection():
    """Verify that sublattice dimension mismatches are accurately flagged."""
    dbf = Database()
    dbf.elements.add("AL")
    dbf.elements.add("NI")
    dbf.species.add(v.Species("AL"))
    dbf.species.add(v.Species("NI"))
    
    # Intentionally create a phase with mismatched sublattices vs constituents
    dbf.add_phase("BROKEN_PHASE", {}, [1.0, 1.0])
    # Give it only 1 sublattice constituent set instead of 2
    dbf.phases["BROKEN_PHASE"].constituents = (frozenset([v.Species("AL")]),)
    
    report = dbf.verify_consistency()
    assert report.is_valid is False
    assert report.num_errors >= 1
    assert any("Sublattice Dimension Mismatch" in issue.check_name for issue in report.issues)


def test_multi_pass_am_thermal_accumulator():
    """Test multi-pass LPBF thermal history and solidification parameter calculation."""
    acc = MultiPassThermalAccumulator(
        k_th=25.0,
        rho=7800.0,
        cp=500.0,
        t_preheat=350.0,
    )
    
    acc.generate_serpentine_pattern(
        num_tracks=3,
        track_length=2.0e-3,     # 2 mm
        hatch_spacing=100.0e-6,  # 100 um
        layer_thickness=40.0e-6,
        num_layers=1,
        power=250.0,
        velocity=0.9,            # 900 mm/s
        absorptivity=0.35,
    )
    
    assert len(acc.tracks) == 3
    
    # Evaluate at a probe point in the center of the middle track
    probe = (1.0e-3, 100.0e-6, 0.0)
    result = acc.evaluate_point_history(
        probe_pos=probe,
        num_samples=1000,
        liquidus_T=1600.0,
        solidus_T=1500.0,
    )
    
    assert result.temperatures is not None
    assert len(result.temperatures) == 1000
    assert result.summary_metrics["max_temperature_K"] > 1500.0
    assert result.summary_metrics["max_cooling_rate_K_s"] > 1.0e4
    assert result.num_cycles >= 1


def test_multi_phase_coprecipitation():
    """Test multi-phase co-precipitation kinetics with mean-field solute conservation."""
    sim = MultiPhasePrecipitationKinetics(
        matrix_name="FCC_A1",
        matrix_molar_volume=1.0e-5,
        initial_solute_fractions={"ZN": 0.05, "MG": 0.03},
        shear_modulus=27.0e9,
        burgers_vector=0.286e-9,
    )
    
    # Phase 1: Eta-prime (AlZnMg)
    sim.add_precipitate_phase(
        PrecipitatePhaseSpec(
            name="ETA_PRIME",
            interfacial_energy=0.08,
            molar_volume=1.0e-5,
            solute_stoichiometry={"ZN": 0.5, "MG": 0.5},
            elastic_misfit_energy=50.0,
            initial_radius=1.0e-9,
            initial_density=1.0e20,
        )
    )
    
    # Phase 2: T-phase
    sim.add_precipitate_phase(
        PrecipitatePhaseSpec(
            name="T_PHASE",
            interfacial_energy=0.15,
            molar_volume=1.0e-5,
            solute_stoichiometry={"ZN": 0.4, "MG": 0.6},
            elastic_misfit_energy=120.0,
            initial_radius=1.2e-9,
            initial_density=1.0e19,
        )
    )
    
    res = sim.simulate(
        t_total=3600.0,  # 1 hour
        num_steps=100,
        temperature=430.0, # 430 K aging
    )
    
    assert len(res.times) == 100
    # Precipitate phases should grow
    assert res.phase_mean_radii["ETA_PRIME"][-1] >= res.phase_mean_radii["ETA_PRIME"][0]
    # Total strengthening should be positive
    assert res.total_strengthening_MPa[-1] >= 0.0
    # Mean-field solute conservation: matrix solute should be less than or equal to initial
    assert res.matrix_solute_fractions["ZN"][-1] <= 0.05
