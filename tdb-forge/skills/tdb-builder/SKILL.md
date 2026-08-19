---
name: tdb-builder
description: Autonomous thermodynamic database assessment, literature extraction, and validation engine for PyCALPHAD.
---

# TDB Builder: Autonomous Thermodynamic Database Assessment

The `tdb-builder` skill orchestrates end-to-end CALPHAD thermodynamic database construction, integrating literature mining, Compound Energy Formalism (CEF) model generation, Redlich-Kister excess fitting, ESPEI Bayesian optimization, and consistency validation.

## Workflow Overview

```
Literature & PDFs ──► Zotero / MatWeb Extraction ──► ESPEI JSON Datasets ──► Model Architect ──► Parameter Regression / ESPEI MCMC ──► Consistency Guards ──► Validated .tdb
```

## Core Modules & Python Entry Points

### 1. Ingest Literature & Generate ESPEI Datasets
```python
from pipeline.literature_ingest import LiteratureIngestionBridge, UnitAndCompositionNormalizer

bridge = LiteratureIngestionBridge(output_dir="./datasets")

# Build ZPF Tie-Line Dataset
dataset = bridge.build_zpf_dataset(
    components=["AL", "NI"],
    phases=["LIQUID", "FCC_A1"],
    temperatures_k=[1200.0, 1300.0],
    tie_line_values=[
        [["LIQUID", ["AL"], [0.15]], ["FCC_A1", ["AL"], [0.08]]],
        [["LIQUID", ["AL"], [0.22]], ["FCC_A1", ["AL"], [0.11]]],
    ],
    reference="Dupin et al. (2001)",
    doi="10.1007/s11669-001-0012-3"
)
dataset.save("./datasets/fitting_datasets/al_ni_zpf.json")

# Ingest MatWeb Engineering Alloy Holdout
bridge.ingest_matweb_holdout_datasheet(
    alloy_name="Inconel 718",
    composition_wt={"Ni": 0.525, "Cr": 0.19, "Fe": 0.185, "Nb": 0.05, "Mo": 0.03, "Ti": 0.01, "Al": 0.01},
    liquidus_c=1336.0,
    solidus_c=1260.0,
    source_key="MATWEB-INCONEL-718"
)
```

### 2. Generate Sublattice Phase Models & Base TDB
```python
from pipeline.model_architect import ModelArchitect

architect = ModelArchitect()
phase_models = architect.generate_system_phase_models(
    elements=["Al", "Ni", "Ti"],
    include_phases=["LIQUID", "FCC_A1", "BCC_A2", "GAMMA_PRIME"]
)
base_db = architect.generate_unassessed_base_tdb(
    elements=["Al", "Ni", "Ti"],
    phases=["LIQUID", "FCC_A1", "BCC_A2", "GAMMA_PRIME"],
    output_tdb_path="./tdbs/base_al_ni_ti.tdb"
)
```

### 3. Fit Excess Parameters & Run ESPEI
```python
from pipeline.parameter_fitter import RedlichKisterParameterFitter

fitter = RedlichKisterParameterFitter()
fit_res = fitter.fit_binary_mixing_enthalpy(
    element_a="AL",
    element_b="NI",
    phase_name="LIQUID",
    mole_fraction_a=[0.1, 0.3, 0.5, 0.7, 0.9],
    hm_mix_j_mol=[-8500, -21000, -28000, -22000, -9000],
    max_degree=3
)
fitter.apply_fit_to_database(base_db, fit_res)
```

### 4. Run Thermodynamic Consistency Verification & Benchmarks
```python
from pipeline.consistency_guards import ThermodynamicConsistencyGuard

guard = ThermodynamicConsistencyGuard()
report = guard.generate_full_consistency_report(
    tdb_path="./tdbs/assessed_al_ni.tdb",
    holdout_dir="./datasets/holdout_validation"
)

print(f"Consistent: {report.is_thermodynamically_consistent}")
print(f"Validation RMSE: {report.overall_rmse_k} K")
```
