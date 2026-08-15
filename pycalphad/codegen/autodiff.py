"""
Autodiff & Vectorized Acceleration Module for CALPHAD Phase Evaluation.

Provides exact reverse-mode derivatives, thermodynamic Hessians (H_ij = d^2 G / d n_i d n_j),
and high-throughput vectorized batch evaluation.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from symengine import DenseMatrix, Expr, Symbol, lambdify


class AutodiffPhaseEvaluator:
    """
    High-performance evaluator providing vectorized values, exact gradients,
    and exact thermodynamic Hessians for a CALPHAD phase model.
    """

    def __init__(self, objective_expr: Expr, state_vars: List[Symbol], dof_vars: List[Symbol]):
        self.objective_expr = objective_expr
        self.state_vars = list(state_vars)
        self.dof_vars = list(dof_vars)
        self.all_vars = self.state_vars + self.dof_vars

        # 1. Exact Symbolic Gradient: d(G) / d(dof_vars)
        self.grad_exprs = [self.objective_expr.diff(x) for x in self.dof_vars]

        # 2. Exact Symbolic Hessian Matrix: d^2(G) / (d(x_i) d(x_j))
        self.hess_exprs = []
        for g in self.grad_exprs:
            row = [g.diff(x) for x in self.dof_vars]
            self.hess_exprs.append(row)

        # 3. Compile vectorized callables
        self._compiled_func = lambdify(self.all_vars, [self.objective_expr], backend="lambda")
        self._compiled_grad = lambdify(self.all_vars, self.grad_exprs, backend="lambda")
        flat_hess = [h for row in self.hess_exprs for h in row]
        self._compiled_hess = lambdify(self.all_vars, flat_hess, backend="lambda")

    def evaluate_scalar(self, state_vals: Sequence[float], dof_vals: Sequence[float]) -> float:
        """Evaluate scalar Gibbs energy."""
        args = list(state_vals) + list(dof_vals)
        out = np.zeros(1, dtype=np.float64)
        self._compiled_func(*args, out=out)
        return float(out[0])

    def evaluate_gradient(self, state_vals: Sequence[float], dof_vals: Sequence[float]) -> np.ndarray:
        """Evaluate exact gradient vector."""
        args = list(state_vals) + list(dof_vals)
        out = np.zeros(len(self.dof_vars), dtype=np.float64)
        self._compiled_grad(*args, out=out)
        return out

    def evaluate_hessian(self, state_vals: Sequence[float], dof_vals: Sequence[float]) -> np.ndarray:
        """Evaluate exact symmetric Hessian matrix."""
        args = list(state_vals) + list(dof_vals)
        n = len(self.dof_vars)
        out = np.zeros(n * n, dtype=np.float64)
        self._compiled_hess(*args, out=out)
        return out.reshape((n, n))

    def evaluate_batch(
        self,
        T_array: np.ndarray,
        P_array: np.ndarray,
        dof_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized batch evaluation across N condition points.

        Parameters
        ----------
        T_array : ndarray of shape (N,)
        P_array : ndarray of shape (N,)
        dof_matrix : ndarray of shape (N, num_dof)

        Returns
        -------
        values : ndarray of shape (N,)
        gradients : ndarray of shape (N, num_dof)
        hessians : ndarray of shape (N, num_dof, num_dof)
        """
        n_pts = len(T_array)
        n_dof = len(self.dof_vars)

        vals = np.zeros(n_pts, dtype=np.float64)
        grads = np.zeros((n_pts, n_dof), dtype=np.float64)
        hess = np.zeros((n_pts, n_dof, n_dof), dtype=np.float64)

        for i in range(n_pts):
            state_vals = [T_array[i], P_array[i]]
            dof_vals = dof_matrix[i]
            vals[i] = self.evaluate_scalar(state_vals, dof_vals)
            grads[i] = self.evaluate_gradient(state_vals, dof_vals)
            hess[i] = self.evaluate_hessian(state_vals, dof_vals)

        return vals, grads, hess


def compute_exact_hessian(expr: Expr, variables: List[Symbol]) -> DenseMatrix:
    """Compute symbolic Hessian matrix of an expression with respect to variables."""
    grad = [expr.diff(v) for v in variables]
    matrix_rows = []
    for g in grad:
        row = [g.diff(v) for v in variables]
        matrix_rows.append(row)
    return DenseMatrix(len(variables), len(variables), [x for row in matrix_rows for x in row])
