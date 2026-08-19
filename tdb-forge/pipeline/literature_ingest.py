"""
Literature & Experimental Data Ingestion and Harmonization Engine for tdb-forge.
Converts extracted literature tables, Zotero metallurgy records, and MatWeb datasheets
into canonical, schema-validated ESPEI JSON dataset files.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
from dataclasses import dataclass, field


# Canonical Elemental Molar Masses (g/mol)
ELEMENT_MOLAR_MASSES: Dict[str, float] = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "MG": 24.305, "AL": 26.982, "SI": 28.085, "TI": 47.867, "V": 50.942,
    "CR": 51.996, "MN": 54.938, "FE": 55.845, "CO": 58.933, "NI": 58.693,
    "CU": 63.546, "ZN": 65.38, "Y": 88.906, "ZR": 91.224, "NB": 92.906,
    "MO": 95.95, "RU": 101.07, "RH": 102.91, "PD": 106.42, "AG": 107.87,
    "SN": 118.71, "HF": 178.49, "TA": 180.95, "W": 183.84, "RE": 186.21,
    "OS": 190.23, "IR": 192.22, "PT": 195.08, "AU": 196.97, "PB": 207.2,
}

# Standard Phase Nomenclature Mapping Dictionary
PHASE_ALIAS_MAP: Dict[str, str] = {
    "liquid": "LIQUID",
    "liq": "LIQUID",
    "l": "LIQUID",
    "fcc": "FCC_A1",
    "fcc_a1": "FCC_A1",
    "austenite": "FCC_A1",
    "gamma": "FCC_A1",
    "bcc": "BCC_A2",
    "bcc_a2": "BCC_A2",
    "ferrite": "BCC_A2",
    "alpha": "BCC_A2",
    "hcp": "HCP_A3",
    "hcp_a3": "HCP_A3",
    "gamma_prime": "GAMMA_PRIME",
    "g_prime": "GAMMA_PRIME",
    "l12": "GAMMA_PRIME",
    "fcc_l12": "GAMMA_PRIME",
    "ni3al": "GAMMA_PRIME",
    "b2": "BCC_B2",
    "bcc_b2": "BCC_B2",
    "nial": "BCC_B2",
    "sigma": "SIGMA",
    "c14": "LAVES_C14",
    "c15": "LAVES_C15",
    "laves": "LAVES_C14",
    "diamond": "DIAMOND_A4",
    "graphite": "GRAPHITE",
}


class PhaseNomenclatureResolver:
    """
    Standardizes author phase names and crystal structures into canonical CALPHAD phase names.
    """

    @staticmethod
    def resolve(phase_str: str) -> str:
        clean = str(phase_str).strip().lower().replace("-", "_").replace(" ", "_")
        return PHASE_ALIAS_MAP.get(clean, phase_str.upper())


class UnitAndCompositionNormalizer:
    """
    Normalizes units (temperatures to Kelvin, energies to J/mol) and converts
    weight fractions (wt%) to atomic mole fractions (at%).
    """

    @staticmethod
    def to_kelvin(temperature: float, unit: str = "C") -> float:
        u = unit.strip().upper()
        if u in ("C", "CELSIUS", "DEG_C"):
            return float(temperature + 273.15)
        elif u in ("F", "FAHRENHEIT"):
            return float((temperature - 32.0) * (5.0 / 9.0) + 273.15)
        return float(temperature)

    @staticmethod
    def to_joules_per_mole(energy: float, unit: str = "J/mol") -> float:
        u = unit.strip().lower()
        if u in ("kj/mol", "kj"):
            return float(energy * 1000.0)
        elif u in ("cal/mol", "cal"):
            return float(energy * 4.184)
        elif u in ("kcal/mol", "kcal"):
            return float(energy * 4184.0)
        return float(energy)

    @staticmethod
    def weight_to_mole_fractions(wt_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Converts weight percentage/fraction dict to normalized atomic mole fraction dict.
        """
        total_wt = sum(wt_dict.values())
        if total_wt <= 0:
            raise ValueError("Total weight must be positive.")

        moles = {}
        for elem, wt in wt_dict.items():
            e_u = elem.upper()
            mass = ELEMENT_MOLAR_MASSES.get(e_u, 55.845)
            moles[e_u] = (wt / total_wt) / mass

        total_moles = sum(moles.values())
        return {e: round(m / total_moles, 6) for e, m in moles.items()}


@dataclass
class ESPEIDataset:
    components: List[str]
    phases: List[str]
    solver: Dict[str, Any]
    conditions: Dict[str, Any]
    output: str
    values: List[Any]
    weight: float = 1.0
    reference: str = "Extracted Literature Source"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.upper() for c in self.components],
            "phases": [p.upper() for p in self.phases],
            "solver": self.solver,
            "conditions": self.conditions,
            "output": self.output,
            "values": self.values,
            "weight": float(self.weight),
            "reference": self.reference,
            "provenance": self.provenance,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class LiteratureIngestionBridge:
    """
    Harmonizes extracted records from OKF Zotero & MatWeb into stratified ESPEI datasets.
    """

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.fitting_dir = os.path.join(self.output_dir, "fitting_datasets")
        self.holdout_dir = os.path.join(self.output_dir, "holdout_validation")
        os.makedirs(self.fitting_dir, exist_ok=True)
        os.makedirs(self.holdout_dir, exist_ok=True)

    def build_zpf_dataset(
        self,
        components: List[str],
        phases: List[str],
        temperatures_k: List[float],
        tie_line_values: List[List[List[Any]]],
        sublattice_site_ratios: Optional[List[float]] = None,
        sublattice_occupancies: Optional[List[List[str]]] = None,
        weight: float = 1.0,
        reference: str = "Literature ZPF",
        doi: Optional[str] = None,
    ) -> ESPEIDataset:
        """
        Builds an ESPEI Zero Phase Fraction (ZPF) tie-line dataset.
        """
        canon_comps = sorted(list(set([c.upper() for c in components] + ["VA"])))
        canon_phases = [PhaseNomenclatureResolver.resolve(p) for p in phases]

        site_ratios = sublattice_site_ratios or [1.0]
        occupancies = sublattice_occupancies or [[c for c in canon_comps if c != "VA"]]

        dataset = ESPEIDataset(
            components=canon_comps,
            phases=canon_phases,
            solver={
                "mode": "equilibrium",
                "sublattice_site_ratios": site_ratios,
                "sublattice_occupancies": occupancies,
            },
            conditions={
                "P": 101325,
                "T": [round(float(t), 2) for t in temperatures_k],
            },
            output="ZPF",
            values=tie_line_values,
            weight=weight,
            reference=reference,
            provenance={"doi": doi, "tier": "tier1_primary_fitting"},
        )
        return dataset

    def build_mixing_enthalpy_dataset(
        self,
        components: List[str],
        phase: str,
        temperature_k: float,
        compositions_x: List[Dict[str, float]],
        hm_values_j_mol: List[float],
        weight: float = 1.0,
        reference: str = "Calorimetry HM_MIX",
        doi: Optional[str] = None,
    ) -> ESPEIDataset:
        """
        Builds an ESPEI mixing enthalpy (HM_MIX) thermochemical dataset.
        """
        canon_comps = sorted(list(set([c.upper() for c in components] + ["VA"])))
        canon_phase = PhaseNomenclatureResolver.resolve(phase)
        non_va = [c for c in canon_comps if c != "VA"]
        dep_elem = non_va[-1]
        indep_elems = non_va[:-1]

        # Format values array for ESPEI
        # conditions dict with X_(elem) arrays
        cond_dict: Dict[str, Any] = {"P": 101325, "T": float(temperature_k)}
        for elem in indep_elems:
            cond_dict[f"X_{elem}"] = [round(c.get(elem, 0.0), 5) for c in compositions_x]

        dataset = ESPEIDataset(
            components=canon_comps,
            phases=[canon_phase],
            solver={
                "sublattice_site_ratios": [1.0],
                "sublattice_occupancies": [[c for c in canon_comps if c != "VA"]],
            },
            conditions=cond_dict,
            output="HM_MIX",
            values=[[round(float(v), 2) for v in hm_values_j_mol]],
            weight=weight,
            reference=reference,
            provenance={"doi": doi, "tier": "tier1_primary_fitting"},
        )
        return dataset

    def ingest_matweb_holdout_datasheet(
        self,
        alloy_name: str,
        composition_wt: Dict[str, float],
        liquidus_c: float,
        solidus_c: float,
        cp_j_kg_k: Optional[float] = None,
        source_key: Optional[str] = None,
    ) -> str:
        """
        Stores MatWeb engineering alloy measurements in the holdout validation directory.
        """
        at_fracs = UnitAndCompositionNormalizer.weight_to_mole_fractions(composition_wt)
        t_liq_k = UnitAndCompositionNormalizer.to_kelvin(liquidus_c, "C")
        t_sol_k = UnitAndCompositionNormalizer.to_kelvin(solidus_c, "C")

        record = {
            "alloy_name": alloy_name,
            "composition_weight_percent": composition_wt,
            "composition_atomic_fractions": at_fracs,
            "liquidus_C": liquidus_c,
            "liquidus_K": round(t_liq_k, 2),
            "solidus_C": solidus_c,
            "solidus_K": round(t_sol_k, 2),
            "specific_heat_J_kg_K": cp_j_kg_k,
            "provenance": {
                "source": "okf-metweb-metal-mcp",
                "source_key": source_key,
                "tier": "tier3_holdout_validation",
            }
        }

        filename = f"matweb_{alloy_name.lower().replace(' ', '_')}.json"
        out_path = os.path.join(self.holdout_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return out_path
