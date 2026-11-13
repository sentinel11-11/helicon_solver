from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.constants import elementary_charge, epsilon_0


@dataclass
class DoubleLayerDiagnostics:
    z: np.ndarray
    potential_V: np.ndarray
    electric_field_V_m: np.ndarray
    poisson_charge_C_m3: np.ndarray
    ion_velocity_m_s: np.ndarray
    ion_density_m3: np.ndarray
    ion_flux_m2_s: float
    electron_velocity_for_zero_current_m_s: np.ndarray
    potential_drop_V: float
    nominal_width_m: float
    current_balance_A_m2: np.ndarray
    poisson_consistency_C_m3: np.ndarray
    status: str = "prescribed_equilibrium_only"

    @classmethod
    def from_config(cls, config, z, plasma):
        dl = config.get("double_layer", {})
        if not dl.get("enabled", False):
            return None
        if dl.get("mode", "prescribed") != "prescribed":
            raise ValueError("Self-consistent double-layer formation requires a kinetic/electrostatic solver and is not implemented")
        z = np.asarray(z, dtype=float)
        center = float(dl["center_m"])
        width = float(dl["width_m"])
        drop = float(dl["potential_drop_V"])
        phi0 = float(dl.get("upstream_potential_V", 0.0))
        if width <= 0 or drop < 0 or not z[0] < center < z[-1]:
            raise ValueError("Invalid prescribed double-layer geometry")
        z_refined = np.linspace(max(z[0], center - 6 * width),
                                min(z[-1], center + 6 * width),
                                max(201, int(dl.get("profile_points", 401))))
        z = np.unique(np.concatenate((z, z_refined)))
        potential = phi0 - 0.5 * drop * (1.0 + np.tanh((z - center) / width))
        field = -np.gradient(potential, z, edge_order=2)
        rho = -epsilon_0 * np.gradient(np.gradient(potential, z, edge_order=2), z, edge_order=2)
        mass = float(plasma["ion_mass"])
        charge_state = int(plasma["ion_charge"])
        speed0 = float(dl.get("upstream_ion_velocity_m_s", 0.0))
        if speed0 <= 0:
            raise ValueError("upstream_ion_velocity_m_s must be positive for steady ion flux")
        speed = np.sqrt(speed0**2 + 2 * charge_state * elementary_charge * (phi0 - potential) / mass)
        density0 = float(dl.get("upstream_ion_density_m3", float(plasma["n_e_peak"]) / charge_state))
        flux = density0 * speed0
        ni = flux / speed
        ne = np.asarray(dl.get("electron_density_m3", np.full_like(z, density0 * charge_state)), dtype=float)
        if ne.ndim == 0:
            ne = np.full_like(z, float(ne))
        if ne.shape != z.shape or np.any(ne <= 0):
            raise ValueError("electron_density_m3 must be a positive scalar or axial array")
        ve = charge_state * flux / ne
        species_charge = elementary_charge * (charge_state * ni - ne)
        return cls(z, potential, field, rho, speed, ni, flux, ve, drop, width,
                   np.zeros_like(z), species_charge - rho)

    def export(self, target):
        from pathlib import Path
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.column_stack((self.z, self.potential_V, self.electric_field_V_m,
                                  self.poisson_charge_C_m3, self.ion_velocity_m_s,
                                  self.ion_density_m3, self.electron_velocity_for_zero_current_m_s,
                                  self.current_balance_A_m2, self.poisson_consistency_C_m3))
        np.savetxt(path, matrix, delimiter=",", comments="", header="z_m,potential_V,E_z_V_m,rho_poisson_C_m3,ion_velocity_m_s,ion_density_m3,electron_velocity_balanced_m_s,current_A_m2,poisson_residual_C_m3")
        return str(path)
