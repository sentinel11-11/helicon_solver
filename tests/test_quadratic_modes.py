import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from quadratic_modes import solve_quadratic, polynomial_residual


def match(values, expected):
    values = values[np.isfinite(values)]
    assert len(values) == len(expected)
    i, j = linear_sum_assignment(abs(values[:, None]-np.asarray(expected)[None, :]))
    return values[i], np.asarray(expected)[j]


def test_known_roots_survive_extreme_row_and_column_scales():
    a0 = np.diag([-6., -20., -42.]).astype(complex)
    a1 = np.eye(3, dtype=complex)
    a2 = np.eye(3, dtype=complex)
    rows, cols = np.diag([1e-12, 1., 1e12]), np.diag([1e8, 1., 1e-8])
    matrices = [rows@a@cols for a in (a0, a1, a2)]
    values, fields, report = solve_quadratic(*matrices)
    actual, expected = match(values, [-3, 2, -5, 4, -7, 6])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    for i, value in enumerate(values):
        assert polynomial_residual(*matrices, value, fields[:, i]) < 1e-10
    assert report["k_scale_rad_m"] > 0


def test_singular_leading_coefficient_keeps_infinite_root():
    matrices = (np.diag([-6., -2.]), np.eye(2), np.diag([1., 0.]))
    values, fields, _ = solve_quadratic(*matrices)
    actual, expected = match(values, [-3, 2, 2])
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    assert np.count_nonzero(~np.isfinite(values)) == 1


def test_common_scalar_and_equation_order_do_not_change_mixed_system_roots():
    rng = np.random.default_rng(42)
    matrices = [rng.standard_normal((4, 4))+1j*rng.standard_normal((4, 4)) for _ in range(3)]
    original, _, _ = solve_quadratic(*matrices)
    order = [3, 1, 0, 2]
    transformed = [1e8*a[order] for a in matrices]
    values, fields, _ = solve_quadratic(*transformed)
    actual, expected = match(values, original)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    for i, value in enumerate(values):
        assert polynomial_residual(*transformed, value, fields[:, i]) < 1e-10


def test_unscaled_debug_option_agrees_on_well_conditioned_system():
    matrices = (np.diag([-6., -20.]), np.eye(2), np.eye(2))
    scaled, _, _ = solve_quadratic(*matrices)
    raw, _, _ = solve_quadratic(*matrices, scaling="none")
    actual, expected = match(scaled, raw)
    np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_zero_and_incompatible_coefficients_fail_clearly():
    with pytest.raises(ValueError):
        solve_quadratic(np.eye(2), np.eye(3), np.eye(2))
    with pytest.raises(ValueError):
        solve_quadratic(np.eye(2), np.eye(2), np.eye(2), scaling="invalid")
