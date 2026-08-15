"""
Two-Sublattice Ionic Liquid and Molten Slag Thermodynamic Model.

Implements the Hillert-Staffansson-Sundman Two-Sublattice Ionic Liquid Model
(C_i^{+v_i})_P (A_j^{-v_j}, Va^{-Q}, B_k^0)_Q
for molten oxide slags, fluxes, and molten salts.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from symengine import Add, Expr, Mul, S, Symbol, log
from tinydb import where

from pycalphad.model import Model
from pycalphad.variables import Species
import pycalphad.variables as v

R_GAS = 8.314462618


class ModelIonicLiquid2SL(Model):
    """
    Thermodynamic model for Two-Sublattice Ionic Liquids & Slags:
    Sublattice 1: Cations (C_i^{+v_i})
    Sublattice 2: Anions (A_j^{-v_j}), Neutral Species (B_k^0), Vacancies (Va^{-Q})
    """

    def __init__(self, dbe, comps, phase_name, parameters=None):
        super().__init__(dbe, comps, phase_name, parameters=parameters)

    def calculate_site_fractions_and_charges(self) -> Dict[str, Any]:
        """Compute average charge Q and site multiplier ratios P and Q."""
        cations = self.constituents[0] if len(self.constituents) > 0 else set()
        anions_neutrals = self.constituents[1] if len(self.constituents) > 1 else set()

        # Average cation valence Q = sum_i (v_i * y_{C_i})
        q_expr = S.Zero
        for c in cations:
            charge = abs(getattr(c, "charge", 1.0))
            if charge == 0:
                charge = 1.0
            sf = v.SiteFraction(self.phase_name, 0, c)
            q_expr += charge * sf

        # Average anion valence P_anion = sum_j (v_j * y_{A_j})
        p_expr = S.Zero
        for a in anions_neutrals:
            charge = abs(getattr(a, "charge", 0.0))
            sf = v.SiteFraction(self.phase_name, 1, a)
            if a.name == "VA":
                # Vacancy charge term = Q * y_{Va}
                p_expr += q_expr * sf
            else:
                p_expr += charge * sf

        return {
            "Q_charge": q_expr,
            "P_sites": p_expr,
            "cations": list(cations),
            "anions_neutrals": list(anions_neutrals),
        }

    def ideal_mixing_energy_ionic(self) -> Expr:
        """
        Ideal entropy of mixing for Two-Sublattice Ionic Liquid:
        G_id = R * T * [ P * sum_i(y_{C_i} ln y_{C_i}) + Q * sum_j(y_{A_j} ln y_{A_j}) ]
        """
        charges = self.calculate_site_fractions_and_charges()
        p_sites = charges["P_sites"]
        q_sites = charges["Q_charge"]

        cat_term = S.Zero
        for c in charges["cations"]:
            y = v.SiteFraction(self.phase_name, 0, c)
            cat_term += y * log(y)

        an_term = S.Zero
        for a in charges["anions_neutrals"]:
            y = v.SiteFraction(self.phase_name, 1, a)
            an_term += y * log(y)

        return R_GAS * v.T * (p_sites * cat_term + q_sites * an_term)


def calculate_slag_basicity_index(
    oxide_mole_fractions: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate optical basicity and V-ratio for molten oxide slags.

    Optical basicity Lambda = sum_i(x_i * n_i * Lambda_i) / sum_i(x_i * n_i)
    """
    # Optical basicity values for common metallurgical oxides
    OPTICAL_BASICITY = {
        "CAO": 1.00,
        "MGO": 0.78,
        "MNO": 0.95,
        "FEO": 1.00,
        "NA2O": 1.15,
        "K2O": 1.40,
        "AL2O3": 0.60,
        "SIO2": 0.48,
        "TIO2": 0.62,
        "P2O5": 0.40,
        "FE2O3": 0.75,
    }
    OXYGEN_NUMBERS = {
        "CAO": 1, "MGO": 1, "MNO": 1, "FEO": 1, "NA2O": 1, "K2O": 1,
        "AL2O3": 3, "SIO2": 2, "TIO2": 2, "P2O5": 5, "FE2O3": 3
    }

    total_eq_o = 0.0
    weighted_basicity = 0.0
    for ox, x_i in oxide_mole_fractions.items():
        ox_up = ox.upper().replace("-", "").replace("_", "")
        lam = OPTICAL_BASICITY.get(ox_up, 0.70)
        n_o = OXYGEN_NUMBERS.get(ox_up, 1)
        equiv_o = x_i * n_o
        total_eq_o += equiv_o
        weighted_basicity += equiv_o * lam

    lambda_opt = weighted_basicity / total_eq_o if total_eq_o > 0 else 0.5

    # V-ratio = (%CaO + %MgO) / (%SiO2 + %Al2O3)
    c_cao = oxide_mole_fractions.get("CAO", 0.0)
    c_mgo = oxide_mole_fractions.get("MGO", 0.0)
    c_sio2 = oxide_mole_fractions.get("SIO2", 0.0)
    c_al2o3 = oxide_mole_fractions.get("AL2O3", 0.0)

    denom = c_sio2 + c_al2o3
    v_ratio = (c_cao + c_mgo) / denom if denom > 0 else 10.0

    return {
        "optical_basicity": float(lambda_opt),
        "v_ratio_basicity": float(v_ratio),
        "slag_character": "Basic" if lambda_opt >= 0.65 else ("Acidic" if lambda_opt < 0.58 else "Neutral"),
    }
