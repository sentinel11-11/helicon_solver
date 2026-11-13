from __future__ import annotations

from pathlib import Path
import copy
import numpy as np
import yaml

from plasma import PlasmaParams, StixTensor
from boundary import PML
from background import Background
from double_layer import DoubleLayerDiagnostics
from solver import HeliconSolver
from probe import ThreeAxisProbe


def base():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "config_test.yaml").read_text())


def test_stix_passive_and_gyro():
    p = PlasmaParams(.03, 1e18, 3.0, 1e7, 6.64e-27, 1, 0)
    eps = StixTensor(p, 2 * np.pi * 9.8e6).get_tensor()
    assert abs(eps[0, 1]) > 1
    antihermitian = (eps - eps.conj().T) / (2j)
    assert np.all(np.linalg.eigvalsh(-antihermitian) >= -1e-7)
    tilted = StixTensor(p, 2 * np.pi * 9.8e6).get_tensor((0.6, 0, .8))
    assert abs(tilted[0, 2]) > 1
    assert np.allclose(np.sort_complex(np.linalg.eigvals(eps)), np.sort_complex(np.linalg.eigvals(tilted)), rtol=1e-8)


def test_pml_excludes_axis():
    pml = PML(20, 60, 3, 6, 3, .01, .01)
    assert pml.sr[0] == 1
    assert pml.sr[-1] != 1
    assert pml.sz[0] != 1 and pml.sz[-1] != 1


def test_axial_background():
    cfg = base()
    cfg["plasma"]["density_axial"] = {"model": "points", "points": [[0, 1], [.5, .5]]}
    cfg["plasma"]["magnetic_field"] = {"model": "axial_points", "points": [[0, .03], [.5, .01]], "near_axis_divergence_free": True}
    bg = Background(cfg, np.linspace(.001, .035, 15), np.linspace(.01, .49, 30))
    assert bg.n_e[0, 0] > bg.n_e[0, -1]
    assert np.any(bg.B_r != 0)
    assert np.max(abs(bg.axisymmetric_divergence_b())) < 1e-7


def test_prescribed_dl():
    cfg = base()
    cfg["double_layer"]["enabled"] = True
    z = np.linspace(0, .5, 101)
    dl = DoubleLayerDiagnostics.from_config(cfg, z, cfg["plasma"])
    assert dl is not None
    assert np.max(dl.electric_field_V_m) > 0
    assert dl.ion_velocity_m_s[-1] > dl.ion_velocity_m_s[0]
    assert dl.status == "prescribed_equilibrium_only"


def test_full_small_fdfd():
    cfg = base()
    cfg["grid"].update(Nr=12, Nz=35, dispersion_Nr=20, dispersion_pml_r=3, pml_r=2, pml_z=3)
    solver = HeliconSolver(cfg)
    assert solver.pml.sr[0] == 1
    solver.solve()
    assert solver.converged and solver.relative_residual < 1e-6
    power = solver.compute_absorbed_power()
    assert np.all(np.isfinite(power))
    assert np.min(power) >= -1e-7 * max(1.0, np.max(abs(power)))
    assert np.all(np.isfinite(solver.compute_magnetic_field()))
    modes = solver.compute_dispersion_modes()
    assert len(modes) > 0


def test_probe_matrix():
    probe = ThreeAxisProbe({"turns": [10, 10, 10], "area_m2": [2e-6] * 3,
                            "series_resistance_ohm": [2] * 3,
                            "series_inductance_H": [2e-7] * 3})
    b = np.array((1+2j, .5-.3j, -.3+.2j)) * 1e-6
    measured = probe.measured_voltage(b, 10e6)
    restored = probe.reconstruct_field(measured, 10e6)
    assert np.allclose(restored, b)


def test_lossless_tensor_and_axis_operator():
    from solver import _axis_derivative
    p = PlasmaParams(.03, 1e18, 3.0, 0, 6.64e-27, 1, 0)
    eps = StixTensor(p, 2 * np.pi * 9.8e6).get_tensor()
    assert np.allclose(eps, eps.conj().T)
    n = 11
    dr = 0.002
    r = (np.arange(n) + .5) * dr
    der_even = _axis_derivative(n, dr, 1)
    der_odd = _axis_derivative(n, dr, -1)
    assert abs((der_even @ (r**2))[0] - 2 * r[0]) < 1e-12
    assert abs((der_odd @ r)[0] - 1.0) < 1e-12


def test_dielectric_and_antenna_outside_plasma():
    cfg = base()
    cfg["plasma"]["radius"] = .0175
    cfg["grid"].update(Nr=14, Nz=35, radial_extent=.055, pml_r=2, pml_z=3)
    cfg["antenna"]["radius_m"] = .039
    cfg["materials"] = {"layers": [{"inner_radius_m": .021,
                                       "outer_radius_m": .025,
                                       "epsilon_r": 3.8,
                                       "loss_tangent": .002}]}
    model = HeliconSolver(cfg)
    assert np.any(model.background.material_mask)
    assert np.all(model.background.n_e[model.background.material_mask] == 0)
    assert np.any(model.dielectric_physical[0][0].real > 3)
    model.solve()
    assert np.all(np.isfinite(model.compute_material_absorption()))
    assert np.isfinite(model.magnetic_divergence_quality())


def test_probe_finite_area_and_phase():
    cfg = base()
    probe = ThreeAxisProbe({"turns": [10, 10, 10], "area_m2": [2e-6] * 3,
                             "delay_s": [0, 1e-9, 0]})
    class Synthetic:
        r = np.linspace(.001, .05, 50)
        z = np.linspace(.001, .5, 100)
        cfg = {"wave": {"m_mode": 1}}
        def compute_magnetic_field(self):
            shape = (len(self.r), len(self.z))
            return np.ones(shape, dtype=complex), 1j * np.ones(shape), np.zeros(shape, dtype=complex)
    avg = probe.area_averaged_field(Synthetic(), .02, .25)
    assert np.allclose(avg, (1, 1j, 0), atol=1e-9)
    voltage = probe.measured_voltage(avg, 10e6)
    assert np.allclose(probe.reconstruct_field(voltage, 10e6), avg)


def test_double_layer_refined_independent_of_rf_mesh():
    cfg = base()
    cfg["double_layer"].update(enabled=True, width_m=.0005, profile_points=501)
    z = np.linspace(0, .5, 50)
    dl = DoubleLayerDiagnostics.from_config(cfg, z, cfg["plasma"])
    assert len(dl.z) > len(z)
    assert np.max(dl.electric_field_V_m) > 1e4
    assert np.max(abs(dl.poisson_consistency_C_m3)) > 0


def test_measured_map_inputs(tmp_path):
    cfg = base()
    path = tmp_path / "density.csv"
    rows = ["r_m,z_m,n_e_m3"]
    for r in (0.0, .02, .04):
        for z in (0.0, .25, .5):
            rows.append(f"{r},{z},{1e18*(1-z):.9e}")
    path.write_text("\n".join(rows), encoding="utf-8")
    cfg["plasma"]["density_map_csv"] = str(path)
    cfg["plasma"]["electron_collision_axial"] = {"points": [[0, 1e7], [.5, 2e7]]}
    bg = Background(cfg, np.linspace(.001, .035, 9), np.linspace(.01, .49, 15))
    assert bg.n_e[0, 0] > bg.n_e[0, -1]
    assert bg.n_e[-1, 0] == 0
    assert bg.nu_e[0, -1] > bg.nu_e[0, 0]


def test_double_layer_disallows_fake_selfconsistent_claim():
    import pytest
    cfg = base()
    cfg["double_layer"]["enabled"] = True
    cfg["double_layer"]["mode"] = "self_consistent"
    with pytest.raises(ValueError, match="not implemented"):
        DoubleLayerDiagnostics.from_config(cfg, np.linspace(0, .5, 100), cfg["plasma"])
