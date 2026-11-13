from __future__ import annotations

import unittest

import numpy as np

from probe_optimization import FieldSnapshot, _evaluate


class ProbeOptimizationTests(unittest.TestCase):
    def test_probe_metrics_are_finite(self):
        r = np.linspace(0.001, 0.03, 16)
        z = np.linspace(0.0, 0.6, 24)
        R, Z = np.meshgrid(r, z, indexing="ij")
        decay = np.exp(-Z / 0.5)
        case = FieldSnapshot(
            9.8e6,
            1,
            r,
            z,
            1e-8 * decay * (1.0 + R),
            2e-8 * decay * (1.0 + R),
            3e-8 * decay * (1.0 + R),
        )
        base = {
            "turns": [10, 10, 10],
            "area_m2": [2e-6, 2e-6, 2e-6],
            "series_resistance_ohm": [2, 2, 2],
            "series_inductance_H": [2e-7, 2e-7, 2e-7],
            "load_resistance_ohm": [50, 50, 50],
            "delay_s": [0, 0, 0],
            "cross_talk": np.eye(3).tolist(),
        }
        metrics = _evaluate(
            np.array([12.0, 10.0, 8.0, 2e-6, 3e-6, 4e-6]),
            base,
            [case],
            [(0.005, 0.2), (0.01, 0.3)],
            {"noise_voltage_V": 1e-6, "rings": 2, "angles": 8},
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreater(metrics["min_singular"], 0.0)


if __name__ == "__main__":
    unittest.main()
