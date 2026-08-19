"""
Deterministic Redlich-Kister Polynomial Parameter Fitter with AIC/BIC Model Selection.
Extracts excess Gibbs energy coefficients from thermochemical mixing datasets for tdb-forge.
"""

from typing import Dict, List, Any, Optional, Tuple, Sequence
import math
import numpy as np
from pydantic import BaseModel, Field
from symengine import Symbol
import pycalphad.variables as v
from pycalphad import Database


class RedlichKisterFitResult(BaseModel):
    phase: str
    element_a: str
    element_b: str
    order: int
    l_coefficients: List[Tuple[float, float]] = Field(
        ..., description="List of (a_v, b_v) where L_v(T) = a_v + b_v * T"
    )
    r2_score: float
    rmse: float
    aic: float
    bic: float
    selected_degrees_of_freedom: int


class RedlichKisterParameterFitter:
    """
    Fits binary Redlich-Kister excess interaction parameters with automated model complexity selection.
    """

    def fit_binary_mixing_enthalpy(
        self,
        element_a: str,
        element_b: str,
        phase_name: str,
        mole_fraction_a: Sequence[float],
        hm_mix_j_mol: Sequence[float],
        temperatures_k: Optional[Sequence[float]] = None,
        max_degree: int = 3,
    ) -> RedlichKisterFitResult:
        """
        Fits enthalpy interaction coefficients:
            HM_MIX(x) = x_A * x_B * sum_{v=0}^M a_v * (x_A - x_B)^v
        """
        x_a = np.array(mole_fraction_a, dtype=float)
        x_b = 1.0 - x_a
        y = np.array(hm_mix_j_mol, dtype=float)

        n_samples = len(x_a)
        if n_samples < 2:
            raise ValueError("Need at least 2 data points for parameter fitting.")

        # Baseline factor x_A * x_B
        base_factor = np.maximum(x_a * x_b, 1e-6)
        reduced_y = y / base_factor
        diff_x = x_a - x_b

        best_order = 0
        best_bic = float("inf")
        best_coeffs: np.ndarray = np.array([0.0])
        best_rmse = float("inf")
        best_r2 = 0.0

        # Sweep candidate polynomial orders M in [0, max_degree]
        for m in range(0, min(max_degree + 1, n_samples)):
            # Construct Vandermonde matrix for (x_A - x_B)^v
            X_mat = np.column_stack([diff_x ** v for v in range(m + 1)])

            # Solve regularized least squares (a_v coefficients)
            # Regularization alpha = 1e-4
            reg_eye = 1e-4 * np.eye(m + 1)
            coeffs, residuals, _, _ = np.linalg.lstsq(X_mat.T @ X_mat + reg_eye, X_mat.T @ reduced_y, rcond=None)

            # Predict and evaluate residuals on original HM_MIX scale
            y_pred = base_factor * (X_mat @ coeffs)
            mse = float(np.mean((y - y_pred) ** 2))
            rmse = math.sqrt(max(mse, 1e-8))

            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1.0 - (np.sum((y - y_pred) ** 2) / max(ss_tot, 1e-6)))

            # Number of parameters k = m + 1
            k_params = m + 1
            aic = float(n_samples * math.log(max(mse, 1e-8)) + 2.0 * k_params)
            bic = float(n_samples * math.log(max(mse, 1e-8)) + k_params * math.log(n_samples))

            # Select most parsimonious model via BIC
            if bic < best_bic:
                best_bic = bic
                best_aic = aic
                best_order = m
                best_coeffs = coeffs
                best_rmse = rmse
                best_r2 = max(r2, 0.0)

        # Structure fitted parameters: (a_v, b_v) where b_v = 0 if only HM_MIX provided
        fitted_l_list: List[Tuple[float, float]] = [
            (float(round(best_coeffs[v], 2)), 0.0) for v in range(len(best_coeffs))
        ]

        return RedlichKisterFitResult(
            phase=phase_name.upper(),
            element_a=element_a.upper(),
            element_b=element_b.upper(),
            order=best_order,
            l_coefficients=fitted_l_list,
            r2_score=round(best_r2, 4),
            rmse=round(best_rmse, 2),
            aic=round(best_aic, 2),
            bic=round(best_bic, 2),
            selected_degrees_of_freedom=best_order + 1,
        )

    def apply_fit_to_database(
        self,
        db: Database,
        fit_result: RedlichKisterFitResult,
    ) -> Database:
        """
        Inserts fitted Redlich-Kister excess parameters into a PyCalphad Database object.
        """
        el_a = fit_result.element_a
        el_b = fit_result.element_b
        phase = fit_result.phase

        for v_order, (a_val, b_val) in enumerate(fit_result.l_coefficients):
            # L_v(T) = a_val + b_val * T
            if abs(b_val) > 1e-6:
                expr = a_val + b_val * v.T
            else:
                expr = float(a_val)

            # PyCalphad add_parameter API
            db.add_parameter(
                param_type="L",
                phase_name=phase,
                constituent_array=[(el_a, el_b)],
                param_order=v_order,
                param=expr,
            )

        return db
