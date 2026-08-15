"""
Unit tests for the Autodiff Acceleration and Thermodynamic Hessian Evaluation module.
"""

import numpy as np
import pytest
from symengine import Symbol, sympify
from pycalphad.codegen.autodiff import AutodiffPhaseEvaluator, compute_exact_hessian


def test_autodiff_phase_evaluator_scalar_and_grad():
    """Verify scalar evaluation and exact gradient evaluation against analytical derivatives."""
    T = Symbol("T")
    P = Symbol("P")
    y0 = Symbol("Y0")
    y1 = Symbol("Y1")

    # Synthetic binary Gibbs energy model: G = y0*G0 + y1*G1 + R*T*(y0*ln(y0) + y1*ln(y1)) + y0*y1*L
    R = 8.314
    G0 = -10000.0 + 10.0 * T
    G1 = -20000.0 + 15.0 * T
    L = -35000.0
    G_expr = y0 * G0 + y1 * G1 + y0 * y1 * L

    evaluator = AutodiffPhaseEvaluator(
        objective_expr=G_expr,
        state_vars=[T, P],
        dof_vars=[y0, y1]
    )

    t_val = 1000.0
    p_val = 101325.0
    y_vals = [0.6, 0.4]

    g_val = evaluator.evaluate_scalar([t_val, p_val], y_vals)
    expected_g = 0.6 * (-10000.0 + 10000.0) + 0.4 * (-20000.0 + 15000.0) + 0.6 * 0.4 * (-35000.0)
    assert np.isclose(g_val, expected_g)

    grad = evaluator.evaluate_gradient([t_val, p_val], y_vals)
    # dG/dy0 = G0 + y1*L = 0 + 0.4 * (-35000) = -14000
    # dG/dy1 = G1 + y0*L = -5000 + 0.6 * (-35000) = -26000
    assert np.isclose(grad[0], -14000.0)
    assert np.isclose(grad[1], -26000.0)

    hess = evaluator.evaluate_hessian([t_val, p_val], y_vals)
    # d^2G / dy0^2 = 0, d^2G / dy0 dy1 = L = -35000, d^2G / dy1^2 = 0
    assert np.isclose(hess[0, 0], 0.0)
    assert np.isclose(hess[0, 1], -35000.0)
    assert np.isclose(hess[1, 0], -35000.0)
    assert np.isclose(hess[1, 1], 0.0)


def test_autodiff_batch_evaluation():
    """Verify vectorized batch evaluation across temperature and composition grids."""
    T = Symbol("T")
    P = Symbol("P")
    y0 = Symbol("Y0")

    G_expr = y0 * (-5000.0 + 8.0 * T) + (y0 ** 2) * -15000.0

    evaluator = AutodiffPhaseEvaluator(
        objective_expr=G_expr,
        state_vars=[T, P],
        dof_vars=[y0]
    )

    t_arr = np.linspace(800.0, 1200.0, 10)
    p_arr = np.full(10, 101325.0)
    dof_mat = np.linspace(0.1, 0.9, 10).reshape(-1, 1)

    vals, grads, hess = evaluator.evaluate_batch(t_arr, p_arr, dof_mat)

    assert len(vals) == 10
    assert grads.shape == (10, 1)
    assert hess.shape == (10, 1, 1)
    assert np.all(np.isfinite(vals))
    assert np.all(np.isfinite(grads))
    assert np.all(np.isfinite(hess))
