"""
Elasticity Tensor and Polycrystalline Mechanical Property Framework.

Implements:
1. Single-crystal elastic stiffness tensor coefficients (C_11, C_12, C_44, C_13, C_33) for cubic and hexagonal phases.
2. Voigt-Reuss-Hill (VRH) polycrystalline averaging for:
   - Bulk Modulus (K_VRH)
   - Shear Modulus (G_VRH)
   - Young's Modulus (E_VRH)
   - Poisson's Ratio (nu_VRH)
   - Pugh Ductility Ratio (K / G)
   - Cauchy Pressure (C_12 - C_44)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from symengine import S, Expr, Symbol
from tinydb import where

from pycalphad.model import Model, _extend_ordered_if_subset_of_disorder
from pycalphad.variables import Species
import pycalphad.variables as v


# Pure element single-crystal reference elastic constants at 298.15 K (in GPa)
DEFAULT_ELASTIC_CONSTANTS = {
    # Cubic FCC/BCC: (C11, C12, C44) in GPa, dC11/dT, dC12/dT, dC44/dT in GPa/K
    "AL": {"crystal": "cubic", "C11": 108.0, "C12": 62.0, "C44": 28.0, "dC11_dT": -0.040, "dC12_dT": -0.015, "dC44_dT": -0.012},
    "NI": {"crystal": "cubic", "C11": 247.0, "C12": 147.0, "C44": 125.0, "dC11_dT": -0.052, "dC12_dT": -0.024, "dC44_dT": -0.030},
    "FE": {"crystal": "cubic", "C11": 230.0, "C12": 135.0, "C44": 117.0, "dC11_dT": -0.045, "dC12_dT": -0.020, "dC44_dT": -0.025},
    "CR": {"crystal": "cubic", "C11": 350.0, "C12": 68.0, "C44": 101.0, "dC11_dT": -0.035, "dC12_dT": -0.010, "dC44_dT": -0.018},
    "CO": {"crystal": "hcp", "C11": 307.0, "C12": 165.0, "C13": 103.0, "C33": 358.0, "C44": 75.0, "dC11_dT": -0.040, "dC33_dT": -0.045, "dC44_dT": -0.020},
    "TI": {"crystal": "hcp", "C11": 162.0, "C12": 92.0, "C13": 69.0, "C33": 181.0, "C44": 47.0, "dC11_dT": -0.030, "dC33_dT": -0.035, "dC44_dT": -0.015},
    "CU": {"crystal": "cubic", "C11": 168.0, "C12": 121.0, "C44": 75.0, "dC11_dT": -0.042, "dC12_dT": -0.018, "dC44_dT": -0.022},
    "MO": {"crystal": "cubic", "C11": 463.0, "C12": 161.0, "C44": 109.0, "dC11_dT": -0.038, "dC12_dT": -0.012, "dC44_dT": -0.015},
    "W": {"crystal": "cubic", "C11": 523.0, "C12": 204.0, "C44": 161.0, "dC11_dT": -0.030, "dC12_dT": -0.010, "dC44_dT": -0.012},
    "NB": {"crystal": "cubic", "C11": 246.0, "C12": 134.0, "C44": 29.0, "dC11_dT": -0.032, "dC12_dT": -0.014, "dC44_dT": -0.008},
}


def calculate_voigt_reuss_hill_cubic(
    c11: float,
    c12: float,
    c44: float
) -> Dict[str, float]:
    """
    Calculate Voigt-Reuss-Hill polycrystalline elastic moduli for cubic crystals.

    Formulas:
      Bulk Modulus:
        B_V = B_R = (C11 + 2*C12) / 3
      Shear Modulus:
        G_V = (C11 - C12 + 3*C44) / 5
        G_R = 5*(C11 - C12)*C44 / (4*C44 + 3*(C11 - C12))
      Hill Averages:
        B_VRH = B_V
        G_VRH = 0.5 * (G_V + G_R)
        E_VRH = (9 * B_VRH * G_VRH) / (3 * B_VRH + G_VRH)
        nu_VRH = (3 * B_VRH - 2 * G_VRH) / (2 * (3 * B_VRH + G_VRH))
    """
    c11 = float(c11)
    c12 = float(c12)
    c44 = float(c44)

    # 1. Bulk modulus
    b_v = (c11 + 2.0 * c12) / 3.0
    b_r = b_v
    b_vrh = b_v

    # 2. Shear modulus
    g_v = (c11 - c12 + 3.0 * c44) / 5.0
    denom = 4.0 * c44 + 3.0 * (c11 - c12)
    g_r = (5.0 * (c11 - c12) * c44) / denom if abs(denom) > 1e-6 else g_v
    g_vrh = 0.5 * (g_v + g_r)

    # 3. Young's modulus and Poisson's ratio
    denom_e = 3.0 * b_vrh + g_vrh
    e_vrh = (9.0 * b_vrh * g_vrh) / denom_e if abs(denom_e) > 1e-6 else 0.0
    denom_nu = 2.0 * (3.0 * b_vrh + g_vrh)
    nu_vrh = (3.0 * b_vrh - 2.0 * g_vrh) / denom_nu if abs(denom_nu) > 1e-6 else 0.3

    # 4. Mechanical indicators
    pugh_ratio = b_vrh / g_vrh if g_vrh > 1e-6 else float("inf")
    cauchy_pressure = c12 - c44  # Positive indicates ductile metallic bonding
    anisotropy_zenor = (2.0 * c44) / (c11 - c12) if abs(c11 - c12) > 1e-6 else 1.0

    return {
        "C11_GPa": c11,
        "C12_GPa": c12,
        "C44_GPa": c44,
        "bulk_modulus_K_GPa": float(b_vrh),
        "shear_modulus_G_GPa": float(g_vrh),
        "youngs_modulus_E_GPa": float(e_vrh),
        "poissons_ratio_nu": float(nu_vrh),
        "pugh_ductility_ratio": float(pugh_ratio),
        "cauchy_pressure_GPa": float(cauchy_pressure),
        "zenor_anisotropy_ratio": float(anisotropy_zenor),
        "ductility_predicted": "Ductile" if (pugh_ratio > 1.75 or cauchy_pressure > 0) else "Brittle",
    }


def calculate_voigt_reuss_hill_hexagonal(
    c11: float,
    c12: float,
    c13: float,
    c33: float,
    c44: float
) -> Dict[str, float]:
    """
    Calculate Voigt-Reuss-Hill polycrystalline elastic moduli for hexagonal crystals.
    """
    c11, c12, c13, c33, c44 = float(c11), float(c12), float(c13), float(c33), float(c44)
    c66 = 0.5 * (c11 - c12)

    # Voigt bounds
    m = c11 + c12 + 2.0 * c33 - 4.0 * c13
    c_sq = (c11 + c12) * c33 - 2.0 * (c13 ** 2)
    b_v = (2.0 * (c11 + c12) + c33 + 4.0 * c13) / 9.0
    g_v = (m + 12.0 * c44 + 12.0 * c66) / 30.0

    # Reuss bounds
    b_r = c_sq / m if abs(m) > 1e-6 else b_v
    denom_gr = 2.0 * (c11 + c12) * c44 * c66 + c_sq * (c44 + c66)
    g_r = (5.0 / 2.0) * (c_sq * c44 * c66) / denom_gr if abs(denom_gr) > 1e-6 else g_v

    b_vrh = 0.5 * (b_v + b_r)
    g_vrh = 0.5 * (g_v + g_r)

    denom_e = 3.0 * b_vrh + g_vrh
    e_vrh = (9.0 * b_vrh * g_vrh) / denom_e if abs(denom_e) > 1e-6 else 0.0
    denom_nu = 2.0 * (3.0 * b_vrh + g_vrh)
    nu_vrh = (3.0 * b_vrh - 2.0 * g_vrh) / denom_nu if abs(denom_nu) > 1e-6 else 0.3

    pugh_ratio = b_vrh / g_vrh if g_vrh > 1e-6 else float("inf")

    return {
        "C11_GPa": c11,
        "C12_GPa": c12,
        "C13_GPa": c13,
        "C33_GPa": c33,
        "C44_GPa": c44,
        "C66_GPa": c66,
        "bulk_modulus_K_GPa": float(b_vrh),
        "shear_modulus_G_GPa": float(g_vrh),
        "youngs_modulus_E_GPa": float(e_vrh),
        "poissons_ratio_nu": float(nu_vrh),
        "pugh_ductility_ratio": float(pugh_ratio),
        "ductility_predicted": "Ductile" if pugh_ratio > 1.75 else "Brittle",
    }


class ElasticityModel(Model):
    """
    Elasticity Model evaluating single-crystal stiffness tensor (C11, C12, C44, C13, C33)
    and polycrystalline Voigt-Reuss-Hill elastic moduli.
    """

    def __init__(self, dbe, comps, phase_name, parameters=None):
        super().__init__(dbe, comps, phase_name, parameters=parameters)
        self.build_elasticity(dbe)

    def build_elasticity(self, dbe):
        """Extract symbolic C_ij expressions from Database parameters."""
        phase = dbe.phases[self.phase_name]
        if phase.model_hints.get('ordered_phase', False):
            phase = _extend_ordered_if_subset_of_disorder(dbe, self.components, phase)
        param_search = dbe.search

        def query_cij(param_type):
            q = (
                (where('phase_name') == phase.name) &
                (where('parameter_type') == param_type) &
                (where('constituent_array').test(self._array_validity))
            )
            return self.symbol_replace(self.redlich_kister_sum(phase, param_search, q), self._symbols)

        self.C11 = query_cij('C11')
        self.C12 = query_cij('C12')
        self.C44 = query_cij('C44')
        self.C13 = query_cij('C13')
        self.C33 = query_cij('C33')


def evaluate_alloy_elastic_properties(
    bulk_mole_fractions: Dict[str, float],
    temperature: float = 298.15,
    crystal_structure: str = "cubic"
) -> Dict[str, Any]:
    """
    Evaluate temperature-dependent polycrystalline elastic properties for a multicomponent alloy.
    """
    elements = [k.upper() for k, v in bulk_mole_fractions.items() if v > 1e-6]
    fractions = np.array([bulk_mole_fractions[el] for el in elements], dtype=float)
    fractions = fractions / np.sum(fractions)

    t = float(temperature)

    # Linear / Rule of mixtures single-crystal constants
    c11_tot = 0.0
    c12_tot = 0.0
    c44_tot = 0.0
    c13_tot = 0.0
    c33_tot = 0.0

    for el, x_i in zip(elements, fractions):
        data = DEFAULT_ELASTIC_CONSTANTS.get(
            el,
            {"crystal": "cubic", "C11": 200.0, "C12": 100.0, "C44": 80.0, "dC11_dT": -0.04, "dC12_dT": -0.02, "dC44_dT": -0.02}
        )
        dT = t - 298.15
        c11_i = max(data["C11"] + data.get("dC11_dT", 0.0) * dT, 10.0)
        c12_i = max(data["C12"] + data.get("dC12_dT", 0.0) * dT, 5.0)
        c44_i = max(data["C44"] + data.get("dC44_dT", 0.0) * dT, 5.0)
        c13_i = max(data.get("C13", c12_i) + data.get("dC13_dT", 0.0) * dT, 5.0)
        c33_i = max(data.get("C33", c11_i) + data.get("dC33_dT", 0.0) * dT, 10.0)

        c11_tot += x_i * c11_i
        c12_tot += x_i * c12_i
        c44_tot += x_i * c44_i
        c13_tot += x_i * c13_i
        c33_tot += x_i * c33_i

    if crystal_structure.lower() in ["hcp", "hexagonal"]:
        res = calculate_voigt_reuss_hill_hexagonal(c11_tot, c12_tot, c13_tot, c33_tot, c44_tot)
    else:
        res = calculate_voigt_reuss_hill_cubic(c11_tot, c12_tot, c44_tot)

    res["temperature_K"] = t
    res["crystal_structure"] = crystal_structure
    return res
