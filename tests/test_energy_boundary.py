import copy
from pathlib import Path

import numpy as np
import pytest

from axis_audit import audit_axis
from energy import _surface_box
from main import load_config, save_fields
from preflight import validate_project_config
from run_boundary_validation import padded_config
from solver import HeliconSolver


def small_config():
    cfg = load_config(Path(__file__).resolve().parents[1]/"config_scan_1MHz_200x800.yaml")
    cfg["grid"].update(Nr=16, Nz=64, pml_r=2, pml_z=4)
    cfg["plasma"]["magnetic_field"] = {"model": "uniform"}
    cfg["wave"].update(frequency=10e6, frequencies=[10e6], m_modes=[1], m_mode=1)
    return cfg


def test_buffer_preserves_physical_equilibrium_source_and_steps():
    cfg = small_config()
    # Include axial variation: continuation must freeze the edge, not stretch it.
    cfg["plasma"]["density_axial"] = {"model": "points", "points": [[0, 0.6], [.608, 1.0]]}
    cfg["plasma"]["magnetic_field"] = {"model": "axial_points", "points": [[0, .02], [.608, .04]]}
    before = copy.deepcopy(cfg)
    padded = padded_config(cfg, after_m=.152, before_m=.076)
    assert cfg == before
    s0 = HeliconSolver(cfg, build_operator=False)
    s1 = HeliconSolver(padded, build_operator=False)
    assert s1.L == s0.L == .608
    assert s1.Nz == 88
    assert s1.dz == pytest.approx(s0.dz, rel=1e-14)
    core = (s1.z >= 0) & (s1.z <= s1.L)
    np.testing.assert_allclose(s1.z[core], s0.z, atol=1e-15)
    for field in ("n_e", "nu_e", "nu_i", "B_r", "B_z"):
        np.testing.assert_allclose(getattr(s1.background, field)[:, core], getattr(s0.background, field), rtol=1e-13)
    j0 = s0._build_source().reshape(3, s0.Nr, s0.Nz)
    j1 = s1._build_source().reshape(3, s1.Nr, s1.Nz)
    np.testing.assert_allclose(j1[:, :, core], j0, rtol=1e-12, atol=1e-14)
    np.testing.assert_equal(j1[:, :, ~core], 0)
    assert not np.any(s1.physical_domain_mask()[:, ~core])
    assert np.sum(s1.non_pml_domain_mask()) > np.sum(s1.physical_domain_mask())


def test_global_work_closes_for_closed_unstretched_system(tmp_path):
    cfg = small_config()
    cfg["grid"].update(pml_r=0, pml_z=0)
    cfg["solver"]["artificial_loss"] = .01
    s = HeliconSolver(cfg)
    s.solve()
    budget = s.energy_budget()
    work = budget["full_grid_operator_work"]
    assert work["relative_balance_defect"] < 1e-6
    assert abs(work["curl_stretch_exchange_W"]) < 1e-6*abs(work["source_power_W"])
    assert work["constitutive_quadratic_including_pml_W"] > 0
    assert work["artificial_absorbed_W"] > 0
    arrays = save_fields(s, tmp_path)
    assert arrays["physical_mask"].dtype == bool
    assert (tmp_path/"fields.npz").is_file()


def test_integrated_absorption_excludes_pml_and_buffers():
    cfg = padded_config(small_config(), after_m=.152)
    s = HeliconSolver(cfg, build_operator=False)
    s.compute_absorbed_power = lambda: np.ones((s.Nr, s.Nz))
    weights = 2*np.pi*s.r[:, None]*s.dr*s.dz
    expected = float(np.sum(s.physical_domain_mask()*weights))
    assert s.integrated_absorbed_power() == pytest.approx(expected)
    assert s.integrated_absorbed_power(physical_only=False) > expected


def test_signed_surface_flux_and_small_source_flag():
    s = HeliconSolver(small_config(), build_operator=False)
    mask = s.physical_domain_mask()
    zero = np.zeros_like(mask, dtype=float)
    positive = np.ones_like(zero)*2
    closed = _surface_box(s, mask, zero, positive, zero, zero)
    assert closed["net_outward_W"] == pytest.approx(0)
    expected = 2*np.pi*(s.r_extent-s.cfg["grid"]["pml_r"]*s.dr)**2
    assert closed["left_face_plus_z_W"] == pytest.approx(expected)
    incoming = _surface_box(s, mask, zero, -s.Z, zero, zero)
    assert incoming["net_boundary_inflow"]
    assert incoming["source_small_relative_to_power_terms"]
    assert np.isfinite(incoming["relative_balance_defect"])
    damped = _surface_box(s, mask, zero, zero, zero, positive, artificial=positive)
    assert damped["balance_defect_W"] == pytest.approx(0)
    assert damped["absorbed_power_W"] == 0
    assert damped["artificial_absorbed_power_W"] > 0


def test_axis_audit_exposes_pointwise_closure_without_losing_adjoint_pair():
    report = audit_axis()
    for row in report["tests"]:
        assert row["forward_near_axis_max_m_inv"] < 1e-10
        assert row["dual_first_cell_error_times_dr"] == pytest.approx(1)
        assert row["weighted_adjoint_defect_max"] < 1e-10
    assert not report["axis_discretization_validated"]


@pytest.mark.parametrize("domain", [
    {"axial_buffer_after_m": -1}, {"axial_buffer_before_m": float("nan")},
    {"axial_buffer_medium": "unimplemented"},
])
def test_invalid_buffer_rejected(domain):
    cfg = small_config()
    cfg["domain"] = domain
    with pytest.raises(ValueError):
        validate_project_config(cfg)
    with pytest.raises(ValueError):
        HeliconSolver(cfg, build_operator=False)
