import copy

import numpy as np
import pytest
from scipy import sparse
from scipy.constants import epsilon_0, mu_0

from boundary import PML
from operator_benchmark import run_case, vacuum_config
from preflight import validate_project_config
from solver import HeliconSolver


def test_cylindrical_jacobian_and_off_diagonal_transform():
    nr, nz, dr, dz = 24, 40, .002, .003
    pml = PML(nr, nz, 4, 5, 6., dr, dz)
    r = (np.arange(nr)+.5)*dr
    factors, mu_inverse = pml.tensor_factors(r)
    i, j = nr-2, nz-3
    width = 4*dr
    stretched_r = r[i]-1j*6*(r[i]-20*dr)**3/(3*width**2)
    s = np.array([pml.sr[i], stretched_r/r[i], pml.sz[j]])
    epsilon = np.array([[2, 1j*.4, .3], [-1j*.4, 3, .2j], [.3, -.2j, 4]], complex)
    jacobian = np.diag(1/s)
    expected = jacobian@epsilon@jacobian.T/np.linalg.det(jacobian)
    result = np.array([[factors[a][b][i, j]*epsilon[a, b] for b in range(3)] for a in range(3)])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose([v[i, j] for v in mu_inverse], s*s/np.prod(s))
    assert factors[0][1][i, j] != factors[0][0][i, j]
    for line in factors:
        for value in line:
            assert value[0, 10] == pytest.approx(1)


def test_zero_stretch_reproduces_original_operator_and_magnetic_intensity():
    cfg = vacuum_config(16, 40, m=1)
    cfg["grid"].update(pml_r=0, pml_z=0)
    cfg["boundary"]["pml_formulation"] = "legacy_stretched"
    old = HeliconSolver(cfg)
    cfg = copy.deepcopy(cfg)
    cfg["boundary"]["pml_formulation"] = "tensor"
    new = HeliconSolver(cfg)
    difference = new.A_cpu-old.A_cpu
    assert not difference.nnz or np.max(abs(difference.data)) < 1e-12
    new.solve()
    for b, h in zip(new.compute_magnetic_field(), new.compute_magnetic_intensity()):
        np.testing.assert_allclose(h, b/mu_0)


def test_passive_conductivity_is_nonnegative_and_vanishes_in_core():
    pml = PML(24, 40, 3, 5, 6., .001, .002)
    loss = pml.passive_electric_loss(2*np.pi*10e6, 3.)
    assert np.all(loss >= 0)
    np.testing.assert_array_equal(loss[:21, 5:-5], 0)
    assert loss[-1, -1] > loss[-1, 10] > 0
    with pytest.raises(ValueError):
        pml.passive_electric_loss(1., -1.)


def test_passive_cold_plasma_operator_and_energy():
    cfg = vacuum_config(16, 64, m=1, formulation="passive_absorber")
    cfg["plasma"].update(n_e_peak=1e18, nu_en=1e7, profile="parabolic")
    cfg["wave"]["frequency"] = 10e6
    cfg["boundary"]["absorber_conductivity_S_m"] = 3.
    cfg["solver"].update(tol=1e-6, require_passive_boundary=True)
    s = HeliconSolver(cfg)
    s.solve()
    report = s.energy_budget()
    s.check_boundary_passivity(report)
    assert report["boundary_passivity_passed"]
    work = report["full_grid_operator_work"]
    assert work["relative_balance_defect"] < 1e-6
    assert work["combined_absorbing_layer_work_W"] > 0
    assert work["source_power_W"] > 0
    weights = sparse.diags(np.tile(np.repeat(s.r, s.Nz), 3))
    stiffness = weights@(s.C_dual@s.Mu_inverse@s.C)
    skew = stiffness-stiffness.conj().T
    assert not skew.nnz or np.max(abs(skew.data)) < 1e-7
    bad = dict(report, boundary_passivity_passed=False)
    with pytest.raises(RuntimeError, match="passivity"):
        s.check_boundary_passivity(bad)


def test_vacuum_tm11_matches_analytic_k_and_tensor_pml_reduces_returning_wave():
    old = run_case(24, 240, m=1, formulation="legacy_stretched")
    new = run_case(24, 240, m=1, formulation="tensor")
    assert new["relative_kz_error"] < .025
    assert new["backward_to_forward_amplitude"] < .02
    assert new["backward_to_forward_amplitude"] < .1*old["backward_to_forward_amplitude"]
    assert new["positive_axial_power_downstream"]


def test_coupled_axis_converges_to_independent_vacuum_reference():
    coarse = run_case(24, 240, m=1, formulation="tensor", axis_regularization="coupled_m1")
    fine = run_case(48, 480, m=1, formulation="tensor", axis_regularization="coupled_m1")
    assert fine["relative_kz_error"] < .011
    assert fine["relative_kz_error"] < .7*coarse["relative_kz_error"]
    assert fine["radial_Ez_shape_relative_error"] < .7*coarse["radial_Ez_shape_relative_error"]
    assert fine["backward_to_forward_amplitude"] < .01


@pytest.mark.parametrize("m", [-1, 1])
def test_coupled_axis_preserves_power_identity_for_both_helicities(m):
    cfg = vacuum_config(16, 64, m=m, formulation="passive_absorber")
    cfg["plasma"].update(n_e_peak=1e18, nu_en=1e7, profile="parabolic")
    cfg["wave"]["frequency"] = 10e6
    cfg["boundary"]["absorber_conductivity_S_m"] = 3.
    cfg["solver"].update(tol=1e-6, axis_regularization="coupled_m1")
    solver = HeliconSolver(cfg)
    solver.solve()
    report = solver.energy_budget()
    assert report["boundary_passivity_passed"]
    assert report["full_grid_operator_work"]["relative_balance_defect"] < 1e-6
    assert solver.A_cpu.shape[0] == 3*solver.Nr*solver.Nz-solver.Nz
    assert solver.residual_equation_space == "axis_constrained_Galerkin"
