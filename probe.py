from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class MagneticProbe:
    def __init__(self, n_turns, area, resistance=50.0, capacitance=5e-12):
        self.N = float(n_turns)
        self.A = float(area)
        self.R = float(resistance)
        self.C = float(capacitance)

    def transfer_function(self, omega):
        return -1j * omega * self.N * self.A / (1 + 1j * omega * self.R * self.C)

    def measured_voltage(self, B_field, omega):
        return self.transfer_function(omega) * B_field


class ThreeAxisProbe:
    AXES = ("r", "phi", "z")

    def __init__(self, config):
        c = config
        self.turns = np.asarray(c["turns"], dtype=float)
        self.area = np.asarray(c["area_m2"], dtype=float)
        self.series_R = np.asarray(c.get("series_resistance_ohm", [0.0] * 3), dtype=float)
        self.series_L = np.asarray(c.get("series_inductance_H", [0.0] * 3), dtype=float)
        self.load_R = np.asarray(c.get("load_resistance_ohm", [50.0] * 3), dtype=float)
        self.delay = np.asarray(c.get("delay_s", [0.0] * 3), dtype=float)
        self.coupling = np.asarray(c.get("cross_talk", np.eye(3)), dtype=float)
        self.loop_radius = np.sqrt(self.area / np.pi)
        self.calibration_csv = c.get("calibration_csv")
        self._calibration = None
        if self.calibration_csv:
            data = np.genfromtxt(self.calibration_csv, names=True, delimiter=",", encoding="utf-8")
            self._calibration = np.atleast_1d(data)
        for vector in (self.turns, self.area, self.series_R, self.series_L, self.load_R, self.delay):
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError("Three-axis probe parameters require finite length-three vectors")
        if self.coupling.shape != (3, 3):
            raise ValueError("Cross-talk matrix must have shape 3 x 3")
        if np.any(self.area <= 0) or np.any(self.turns <= 0) or np.any(self.load_R <= 0):
            raise ValueError("Probe turns, areas and load resistances must be positive")

    def transfer_matrix(self, frequency_hz):
        omega = 2 * np.pi * frequency_hz
        if self._calibration is not None:
            f = self._calibration["frequency_Hz"]
            if frequency_hz < np.min(f) or frequency_hz > np.max(f):
                raise ValueError("Frequency outside calibrated probe band")
            entries = np.zeros((3, 3), dtype=complex)
            for i, a in enumerate(self.AXES):
                for j, b in enumerate(self.AXES):
                    re = self._calibration[f"H_{a}_{b}_re"]
                    im = self._calibration[f"H_{a}_{b}_im"]
                    entries[i, j] = np.interp(frequency_hz, f, re) + 1j * np.interp(frequency_hz, f, im)
            return entries
        z_series = self.series_R + 1j * omega * self.series_L
        divider = self.load_R / (self.load_R + z_series)
        h = -1j * omega * self.turns * self.area * divider * np.exp(-1j * omega * self.delay)
        return self.coupling @ np.diag(h)

    def measured_voltage(self, B_averaged, frequency_hz):
        return self.transfer_matrix(frequency_hz) @ np.asarray(B_averaged, dtype=complex)

    def reconstruct_field(self, voltage, frequency_hz, condition_limit=1e6):
        h = self.transfer_matrix(frequency_hz)
        condition = np.linalg.cond(h)
        if not np.isfinite(condition) or condition > condition_limit:
            raise ValueError("Probe calibration matrix is poorly conditioned")
        return np.linalg.solve(h, np.asarray(voltage, dtype=complex))

    def area_averaged_field(self, solver, r0, z0, rings=5, angles=24, phi0=0.0):
        br, bp, bz = solver.compute_magnetic_field()
        coords = (solver.r, solver.z)
        interpolators = [RegularGridInterpolator(coords, a, bounds_error=True) for a in (br, bp, bz)]
        base = np.array((r0 * np.cos(phi0), r0 * np.sin(phi0), z0))
        normals = np.array(((np.cos(phi0), np.sin(phi0), 0.0),
                            (-np.sin(phi0), np.cos(phi0), 0.0),
                            (0.0, 0.0, 1.0)))
        tangent = ((np.array((0.0, 0.0, 1.0)), normals[1]),
                   (np.array((0.0, 0.0, 1.0)), -normals[0]),
                   (normals[0], normals[1]))
        result = np.zeros(3, dtype=complex)
        m = int(solver.cfg["wave"].get("m_mode", 0))
        theta = np.arange(angles) * 2 * np.pi / angles
        radial = np.sqrt((np.arange(rings) + 0.5) / rings)
        for i in range(3):
            u, v = tangent[i]
            center = base
            points = (center[None, None, :] + self.loop_radius[i] * radial[:, None, None]
                      * (np.cos(theta)[None, :, None] * u + np.sin(theta)[None, :, None] * v))
            xyz = points.reshape(-1, 3)
            rp = np.hypot(xyz[:, 0], xyz[:, 1])
            ph = np.arctan2(xyz[:, 1], xyz[:, 0])
            samples = np.column_stack((rp, xyz[:, 2]))
            components = np.asarray(np.column_stack([interp(samples) for interp in interpolators]), dtype=complex)
            components *= np.exp(1j * m * ph)[:, None]
            cart = np.column_stack((components[:, 0] * np.cos(ph) - components[:, 1] * np.sin(ph),
                                    components[:, 0] * np.sin(ph) + components[:, 1] * np.cos(ph),
                                    components[:, 2]))
            result[i] = np.mean(cart @ normals[i])
        return result


def simulate_probe_scan(solver, probe, z_positions, r_positions):
    if not getattr(solver, "_solved", False):
        raise RuntimeError("Solve the RF field before generating probe readings")
    br, bp, bz = solver.compute_magnetic_field()
    use_3d = isinstance(probe, ThreeAxisProbe)
    results = []
    for z0 in z_positions:
        j = int(np.argmin(np.abs(solver.z - z0)))
        for r0 in r_positions:
            i = int(np.argmin(np.abs(solver.r - r0)))
            if use_3d:
                values = probe.area_averaged_field(solver, float(r0), float(z0))
                volts = probe.measured_voltage(values, solver.omega / (2 * np.pi))
            else:
                values = np.array((br[i, j], bp[i, j], bz[i, j]))
                volts = np.array([probe.measured_voltage(x, solver.omega) for x in values])
            results.append({
                "z": float(z0), "r": float(r0),
                "absVr": float(abs(volts[0])), "absVphi": float(abs(volts[1])), "absVz": float(abs(volts[2])),
                "phaseBr": float(np.angle(values[0])), "phaseBphi": float(np.angle(values[1])),
                "phaseBz": float(np.angle(values[2])),
                "phaseVr": float(np.angle(volts[0])),
                "phaseVphi": float(np.angle(volts[1])),
                "phaseVz": float(np.angle(volts[2])),
                "Br": values[0], "Bphi": values[1], "Bz": values[2],
            })
    return results
