from pathlib import Path

import numpy as np
import yaml

from discretization import derivative_pair, forward_difference, radial_grid_oscillation
from preflight import validate_project_config
from solver import HeliconSolver, select_dispersion_mode


def test_checkerboard_not_in_derivative_nullspace():
    n, h = 40, 0.01
    checkerboard = (-1.0) ** np.arange(n)
    forward = forward_difference(n, h)
    assert np.linalg.norm(forward @ checkerboard) > 1e3
    assert np.linalg.norm(forward @ np.ones(n)) < 1e-12


def test_derivative_pair_is_adjoint_on_unstretched_grid():
    plus, minus = derivative_pair(20, .01, np.ones(20))
    assert np.allclose((minus + plus.T).toarray(), 0)
    gram = (-minus @ plus).toarray()
    assert np.allclose(gram, gram.T.conj())
    assert np.linalg.eigvalsh(gram).min() > -1e-10


def test_checkerboard_roughness_indicator():
    n = 60
    smooth = np.exp(-((np.arange(n) - 25) / 12.0) ** 2)
    alternating = smooth * ((-1.0) ** np.arange(n))
    good = radial_grid_oscillation((smooth, smooth * 1j), 5)
    bad = radial_grid_oscillation((alternating, alternating * 1j), 5)
    assert good < 0.05
    assert bad > 0.5


def test_m_sign_and_release_configuration():
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["wave"]["m_modes"] == [-1, 0, 1, 2]
    assert validate_project_config(cfg).run_count == 20
    cfg["grid"].update(Nr=12, Nz=24, pml_r=2, pml_z=3)
    cfg["wave"]["frequency"] = 5e6
    cfg["wave"]["m_mode"] = -1
    minus = HeliconSolver(cfg)
    cfg["wave"]["m_mode"] = 1
    plus = HeliconSolver(cfg)
    assert (minus.C != plus.C).nnz > 0
    assert minus.A_cpu.shape == plus.A_cpu.shape == (3 * 12 * 24, 3 * 12 * 24)


def test_selection_prefers_mesh_screened_mode():
    def make(k, resolved):
        return {"kz": complex(k, -.3), "radial_grid_resolved": resolved,
                "radial_grid_oscillation": .1 if resolved else .8}
    modes = [make(10, False), make(12, True)]
    assert select_dispersion_mode(modes, 10)["radial_grid_resolved"]
    assert not select_dispersion_mode(modes[:1], 10)["radial_grid_resolved"]
