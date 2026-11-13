from __future__ import annotations

import unittest

import numpy as np
from scipy.constants import mu_0

from magnetic_system import _loop_field, _solenoid_field, build_magnetic_field, build_magnetic_field_3d
from solver import _dielectric_components


class MagneticSystemTests(unittest.TestCase):
    def test_loop_axis_field_matches_analytic_expression(self):
        radius = 0.04
        current = 2.0
        z = np.array([0.0, 0.03, 0.1])
        br, bz = _loop_field(np.zeros_like(z), z, radius, current)
        expected = mu_0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
        np.testing.assert_allclose(br, 0.0, atol=1e-14)
        np.testing.assert_allclose(bz, expected, rtol=1e-10, atol=1e-14)

    def test_loop_field_matches_direct_biot_savart_integration(self):
        radius = 0.04
        current = 2.0
        rho = 0.012
        z = 0.031
        br, bz = _loop_field(np.array([rho]), np.array([z]), radius, current)
        phi = np.linspace(0.0, 2.0 * np.pi, 400000, endpoint=False)
        dphi = 2.0 * np.pi / phi.size
        source = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)))
        dl = np.column_stack((-radius * np.sin(phi), radius * np.cos(phi), np.zeros_like(phi))) * dphi
        observation = np.array((rho, 0.0, z))
        displacement = observation[None, :] - source
        cross = np.cross(dl, displacement)
        distance = np.linalg.norm(displacement, axis=1)
        field = mu_0 * current / (4.0 * np.pi) * np.sum(cross / distance[:, None] ** 3, axis=0)
        np.testing.assert_allclose(br[0], field[0], rtol=2e-5, atol=2e-10)
        np.testing.assert_allclose(bz[0], field[2], rtol=2e-5, atol=2e-10)

    def test_loop_symmetry(self):
        rho = np.array([0.005, 0.012, 0.02])
        z = np.array([0.02, 0.05, 0.1])
        br_plus, bz_plus = _loop_field(rho, z, 0.04, 1.0)
        br_minus, bz_minus = _loop_field(rho, -z, 0.04, 1.0)
        np.testing.assert_allclose(br_plus, -br_minus, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(bz_plus, bz_minus, rtol=1e-12, atol=1e-14)

    def test_two_section_solenoid_is_sum_of_finite_loops(self):
        spec = {
            "solenoid": {
                "radius_m": 0.04,
                "sections": [
                    {"z_start_m": 0.0, "z_end_m": 0.3, "turns": 2.0, "current_A": 3.0, "axial_samples": 2},
                    {"z_start_m": 0.3, "z_end_m": 0.6, "turns": 2.0, "current_A": 3.0, "axial_samples": 2},
                ],
            }
        }
        z = np.array([0.2, 0.5])
        rho = np.zeros_like(z)
        br, bz = _solenoid_field(rho, z, spec, 0.6)
        expected = np.zeros_like(z)
        for start, end in ((0.0, 0.3), (0.3, 0.6)):
            for position in np.linspace(start, end, 2):
                _, contribution = _loop_field(rho, z - position, 0.04, 3.0 * 2.0 / 2.0)
                expected += contribution
        np.testing.assert_allclose(br, 0.0, atol=1e-14)
        np.testing.assert_allclose(bz, expected, rtol=1e-12, atol=1e-14)

    def test_axisymmetric_and_full_3d_representations_are_explicit(self):
        spec = {
            "model": "plmm_assembly",
            "solenoid": {
                "radius_m": 0.04,
                "sections": [
                    {"z_start_m": 0.0, "z_end_m": 0.3, "turns": 1.0, "current_A": 1.0, "axial_samples": 8},
                    {"z_start_m": 0.3, "z_end_m": 0.6, "turns": 1.0, "current_A": 1.0, "axial_samples": 8},
                ],
            },
            "magnets": {
                "count": 8,
                "length_m": 0.608,
                "cross_section_m": 0.018,
                "center_radius_m": 0.045,
                "remanence_T": 0.1,
                "segments": 4,
                "axisymmetric_average_samples": 8,
            },
            "normalization": {"enabled": False},
        }
        r = (np.arange(8) + 0.5) * 0.03 / 8
        z = (np.arange(10) + 0.5) * 0.6 / 10
        R, Z = np.meshgrid(r, z, indexing="ij")
        br, bz, metadata = build_magnetic_field(r, z, R, Z, spec, 0.03)
        self.assertEqual(metadata["representation"], "axisymmetric_average")
        x = np.array([0.0, 0.01])
        y = np.array([0.0, 0.01])
        zz = np.array([0.0, 0.3])
        X, Y, ZZ = np.meshgrid(x, y, zz, indexing="ij")
        bx3, by3, bz3, metadata3 = build_magnetic_field_3d(X, Y, ZZ, spec, 0.03)
        self.assertEqual(metadata3["representation"], "full_3d")
        self.assertTrue(np.all(np.isfinite(br)))
        self.assertTrue(np.all(np.isfinite(bz)))
        self.assertTrue(np.all(np.isfinite(bx3)))
        self.assertTrue(np.all(np.isfinite(by3)))
        self.assertTrue(np.all(np.isfinite(bz3)))

    def test_local_stix_tensor_rotation_preserves_principal_values(self):
        S = np.array([1.2 + 0.01j])
        D = np.array([0.35 + 0.02j])
        P = np.array([0.8 + 0.03j])
        axial = _dielectric_components(S, D, P, np.array([0.0]), np.array([0.03]))
        tilted = _dielectric_components(S, D, P, np.array([0.018]), np.array([0.024]))
        axial_matrix = np.array([[axial[i][j][0] for j in range(3)] for i in range(3)])
        tilted_matrix = np.array([[tilted[i][j][0] for j in range(3)] for i in range(3)])
        np.testing.assert_allclose(
            np.sort_complex(np.linalg.eigvals(axial_matrix)),
            np.sort_complex(np.linalg.eigvals(tilted_matrix)),
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
