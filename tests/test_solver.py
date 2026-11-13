import copy
import unittest
from pathlib import Path

import numpy as np
import yaml

from boundary import PML
from plasma import PlasmaParams, StixTensor, density_profile, theoretical_kz
from solver import HeliconSolver, select_dispersion_mode

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


class TestPlasmaModel(unittest.TestCase):
    def test_stix_cold_limit_and_tensor_convention(self):
        p = PlasmaParams(B0=0.03, n_e=1e17, T_e=3.0, nu_en=0.0,
                         ion_mass=6.64e-27, ion_charge=1)
        omega = 2 * np.pi * 12e6
        stix = StixTensor(p, omega)
        expected_s = 1.0 - stix.omega_pe**2 / (omega**2 - stix.omega_ce**2) \
                         - stix.omega_pi**2 / (omega**2 - stix.omega_ci**2)
        self.assertAlmostEqual(stix.S.real, expected_s, places=9)
        tensor = stix.get_tensor()
        self.assertAlmostEqual(tensor[0, 1].imag, stix.D.real, places=9)
        self.assertAlmostEqual(tensor[1, 0].imag, -stix.D.real, places=9)

    def test_ion_plasma_frequency_uses_quasineutral_ion_density(self):
        p = PlasmaParams(B0=0.03, n_e=1e17, T_e=3.0, nu_en=0.0,
                         ion_mass=6.64e-27, ion_charge=2)
        stix = StixTensor(p, 2 * np.pi * 12e6)
        expected = np.sqrt(p.n_e * p.ion_charge * 1.602176634e-19**2 /
                           (8.8541878128e-12 * p.ion_mass))
        self.assertAlmostEqual(stix.omega_pi, expected, delta=1.0)

    def test_density_is_vacuum_outside_plasma(self):
        r = np.array([0.0, 0.5, 1.0, 1.2])
        n = density_profile(r, 1.0, 10.0, "flat")
        np.testing.assert_allclose(n, [10.0, 10.0, 0.0, 0.0])

    def test_pml_sign_for_exp_plus_iwt_outgoing_wave(self):
        pml = PML(32, 64, 4, 8, 1.0, 1e-3, 1e-3)
        self.assertTrue(np.all(np.imag(pml.scale_r()[:4]) == 0))
        self.assertTrue(np.all(np.imag(pml.scale_r()[-4:]) < 0))
        self.assertTrue(np.all(np.imag(pml.scale_z()[[0, 1, -2, -1]]) < 0))


class TestSolver(unittest.TestCase):
    def small_config(self):
        cfg = copy.deepcopy(load_config())
        cfg["grid"].update({"Nr": 24, "Nz": 80, "pml_r": 3, "pml_z": 8,
                            "dispersion_Nr": 40, "dispersion_pml_r": 4})
        cfg["solver"].update({"method": "direct", "direct_max_unknowns": 10000,
                              "tol": 1e-7, "artificial_loss": 0.01})
        cfg["wave"]["frequency"] = 9.8e6
        cfg["wave"]["m_mode"] = 1
        return cfg

    def test_radial_pml_must_be_outside_plasma(self):
        cfg = self.small_config()
        cfg["grid"]["radial_extent"] = 0.026
        cfg["grid"]["pml_r"] = 3
        with self.assertRaisesRegex(ValueError, "PML would overlap the plasma"):
            HeliconSolver(cfg)

    def test_source_has_finite_radial_width(self):
        solver = HeliconSolver(self.small_config())
        source = solver._build_source()
        n = solver.Nr * solver.Nz
        jphi = source[n:2*n].reshape(solver.Nr, solver.Nz)
        active_radial_cells = np.count_nonzero(np.max(np.abs(jphi), axis=1) > 0)
        self.assertGreater(active_radial_cells, 1)

    def test_driven_solve_residual_and_phase_estimate(self):
        solver = HeliconSolver(self.small_config())
        solver.solve()
        self.assertLess(solver.relative_residual, 1e-7)
        self.assertTrue(np.all(np.isfinite(solver.E_mag)))
        kz, alpha = solver.compute_wave_structure()
        if solver.kz_fit_reliable:
            self.assertGreater(kz, 0.0)
            self.assertTrue(np.all(np.isfinite(alpha)))
        else:
            self.assertTrue(np.isnan(kz))
            self.assertTrue(np.all(np.isnan(alpha)))
            self.assertTrue(solver.kz_fit_reason)
        self.assertTrue(np.isfinite(solver.kz_fit_quality))

    def test_radial_eigenproblem_returns_forward_modes(self):
        solver = HeliconSolver(self.small_config())
        modes = solver.compute_dispersion_modes()
        self.assertTrue(modes)
        self.assertTrue(all(np.isfinite(m["kz"].real) and m["kz"].real > 0 for m in modes))
        self.assertTrue(all(m["kz"].imag <= 1e-6 for m in modes))
        ref = abs(theoretical_kz(solver.omega, solver.params.n_e, solver.params.B0))
        selected = select_dispersion_mode(modes, ref)
        self.assertIn("Er", selected)
        self.assertIn("r", selected)
        continued = select_dispersion_mode(
            modes, ref, previous_kz=selected["kz"], frequency_ratio=1.0,
            previous_mode=selected,
        )
        self.assertAlmostEqual(abs(continued["kz"] - selected["kz"]), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
