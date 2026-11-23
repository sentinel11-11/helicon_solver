import unittest

import numpy as np

from wave_analysis import analyze_axial_wave


class SyntheticSolver:
    def __init__(self, k=13.0, alpha=1.8, reflection=0.0, noise=0.0, stationary=False):
        self._solved = True
        self.Nr, self.Nz = 100, 800
        self.L, self.r0 = 1.0, 0.025
        self.dr, self.dz = 0.035 / self.Nr, self.L / self.Nz
        self.r = (np.arange(self.Nr) + 0.5) * self.dr
        self.z = (np.arange(self.Nz) + 0.5) * self.dz
        self.cfg = {
            "grid": {"pml_z": 15},
            "antenna": {"z_position": 0.15, "length": 0.10},
            "solver": {"kz_r_min_fraction": 0.45, "kz_r_max_fraction": 0.9},
        }
        rng = np.random.default_rng(9)
        x = self.z
        wave = (
            np.ones_like(x, dtype=complex)
            if stationary else np.exp((-alpha - 1j * k) * (x - 0.3))
            + reflection * np.exp((alpha + 1j * k) * (x - 0.9))
        )
        self.fields = []
        for amplitude in (1.0, 0.7, 0.3):
            shape = np.exp(-((self.r - 0.017) / 0.014) ** 2)
            field = amplitude * shape[:, None] * wave[None, :]
            field = field + noise * (
                rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
            )
            self.fields.append(field)

    def compute_magnetic_field(self):
        return self.fields


class WaveAnalysisTests(unittest.TestCase):
    def test_clean_wave(self):
        result = analyze_axial_wave(SyntheticSolver())
        self.assertFalse(result.reason)
        self.assertAlmostEqual(result.kz, 13.0, delta=0.05)
        self.assertAlmostEqual(result.alpha, 1.8, delta=0.05)

    def test_wave_with_noise(self):
        result = analyze_axial_wave(SyntheticSolver(noise=0.004))
        self.assertFalse(result.reason)
        self.assertAlmostEqual(result.kz, 13.0, delta=0.15)

    def test_large_field_outside_observation_region_does_not_hide_clean_wave(self):
        solver = SyntheticSolver()
        for field in solver.fields:
            field[solver.r > solver.r0] *= 1e12
        result = analyze_axial_wave(solver)
        self.assertFalse(result.reason)
        self.assertAlmostEqual(result.kz, 13.0, delta=0.05)

    def test_reflection_is_rejected(self):
        result = analyze_axial_wave(SyntheticSolver(reflection=0.9))
        self.assertTrue(np.isnan(result.kz))
        self.assertTrue(result.reason)

    def test_stationary_field_is_rejected(self):
        result = analyze_axial_wave(SyntheticSolver(stationary=True))
        self.assertTrue(np.isnan(result.kz))
        self.assertIn("insufficient phase cycles", result.reason)


if __name__ == "__main__":
    unittest.main()
