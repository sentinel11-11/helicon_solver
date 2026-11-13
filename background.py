from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from plasma import density_profile
from magnetic_system import build_magnetic_field


class Background:
    def __init__(self, config, r, z):
        self.config = config
        self.r = np.asarray(r, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")
        p = config["plasma"]
        self.radius = float(p["radius"])
        self.n_e = density_profile(self.R, self.radius, float(p["n_e_peak"]), p.get("profile", "parabolic"))
        self.n_e *= self._axial_factor(p.get("density_axial"))
        if p.get("density_map_csv"):
            data = np.genfromtxt(p["density_map_csv"], names=True, delimiter=",", encoding="utf-8")
            points = np.column_stack((data["r_m"], data["z_m"]))
            interpolator = LinearNDInterpolator(points, data["n_e_m3"], fill_value=np.nan)
            values = np.asarray(interpolator(self.R.ravel(), self.Z.ravel()))
            if np.any(~np.isfinite(values)):
                if not p.get("density_map_allow_nearest_extrapolation", False):
                    raise ValueError("Electron density map does not cover the computational domain")
                mask = ~np.isfinite(values)
                values[mask] = NearestNDInterpolator(points, data["n_e_m3"])(
                    self.R.ravel()[mask], self.Z.ravel()[mask])
            self.n_e = np.where(self.R < self.radius, values.reshape(self.R.shape), 0.0)
        self.nu_e = np.full_like(self.R, float(p.get("nu_en", 0.0)))
        self.nu_i = np.full_like(self.R, float(p.get("nu_in", 0.0)))
        if p.get("electron_collision_axial"):
            spec = p["electron_collision_axial"]
            points = np.asarray(spec["points"], dtype=float)
            self._validate_points(points)
            self.nu_e = np.interp(self.Z, points[:, 0], points[:, 1])
        self.B_r = np.zeros_like(self.R)
        self.B_z = np.full_like(self.R, float(p["B0"]))
        self.magnetic_metadata = {"model": "uniform", "uniform_B0_T": float(p["B0"])}
        self._set_field(p.get("magnetic_field"))
        self.magnetic_divergence = self.divergence_report()
        self.magnetic_metadata["divergence"] = self.magnetic_divergence
        self.relative_permittivity = np.ones_like(self.R, dtype=complex)
        self.material_mask = np.zeros_like(self.R, dtype=bool)
        for layer in config.get("materials", {}).get("layers", []):
            r_min = float(layer["inner_radius_m"])
            r_max = float(layer["outer_radius_m"])
            if r_min < self.radius or r_max <= r_min or r_max > float(config["grid"]["radial_extent"]):
                raise ValueError("Dielectric material annulus must lie outside the plasma and within the domain")
            permittivity = float(layer["epsilon_r"])
            tangent = float(layer.get("loss_tangent", 0.0))
            if permittivity < 1 or tangent < 0:
                raise ValueError("Invalid dielectric layer parameters")
            mask = (self.R >= r_min) & (self.R <= r_max)
            if np.any(self.material_mask & mask):
                raise ValueError("Overlapping material layers")
            self.relative_permittivity[mask] = permittivity * (1.0 - 1j * tangent)
            self.material_mask |= mask
        if np.any(self.n_e < 0) or np.any(self.nu_e < 0) or np.any(self.nu_i < 0):
            raise ValueError("Density and collision frequencies must be nonnegative")
        if not np.all(np.isfinite(self.B_z)) or not np.all(np.isfinite(self.B_r)):
            raise ValueError("Background magnetic field contains non-finite values")

    def _axial_factor(self, spec):
        if not spec or spec.get("model", "uniform") == "uniform":
            return np.ones_like(self.Z)
        if spec["model"] == "points":
            points = np.asarray(spec["points"], dtype=float)
            self._validate_points(points)
            return np.interp(self.Z, points[:, 0], points[:, 1])
        if spec["model"] == "file":
            a = np.genfromtxt(spec["path"], delimiter=",", names=True)
            points = np.column_stack((a["z_m"], a["factor"]))
            self._validate_points(points)
            return np.interp(self.Z, points[:, 0], points[:, 1])
        raise ValueError("Unknown density axial model")

    @staticmethod
    def _validate_points(points):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or np.any(np.diff(points[:, 0]) <= 0) or not np.all(np.isfinite(points)):
            raise ValueError("Profiles require ordered, finite [position, value] points")
        if np.any(points[:, 1] < 0):
            raise ValueError("Profile factors must be nonnegative")

    def _set_field(self, spec):
        if not spec or spec.get("model", "uniform") == "uniform":
            return
        if spec["model"] == "plmm_assembly":
            representation = str(spec.get("representation", "axisymmetric_average"))
            if representation == "full_3d":
                raise ValueError(
                    "The 2-D cylindrical FDFD solver requires magnetic_field.representation=axisymmetric_average; "
                    "use magnetic_3d.py for the full 3-D cusp field"
                )
            if representation != "axisymmetric_average":
                raise ValueError("Unknown magnetic-field representation")
            self.B_r, self.B_z, self.magnetic_metadata = build_magnetic_field(
                self.r, self.z, self.R, self.Z, spec, float(self.config["plasma"]["B0"])
            )
            return
        if spec["model"] == "axial_points":
            points = np.asarray(spec["points"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or np.any(np.diff(points[:, 0]) <= 0):
                raise ValueError("Field profile requires ordered [z_m, B_z_T] points")
            self.B_z = np.interp(self.Z, points[:, 0], points[:, 1])
            if spec.get("near_axis_divergence_free", False):
                slope = np.gradient(self.B_z, self.z, axis=1, edge_order=2)
                self.B_r = -0.5 * self.R * slope
            return
        if spec["model"] == "map_csv":
            path = Path(spec["path"])
            field = np.genfromtxt(path, names=True, delimiter=",", encoding="utf8")
            points = np.column_stack((field["r_m"], field["z_m"]))
            target = np.column_stack((self.R.ravel(), self.Z.ravel()))
            for name, attribute in (("B_r_T", "B_r"), ("B_z_T", "B_z")):
                interp = LinearNDInterpolator(points, field[name], fill_value=np.nan)
                values = np.asarray(interp(target))
                missing = ~np.isfinite(values)
                if np.any(missing):
                    if not spec.get("allow_nearest_extrapolation", False):
                        raise ValueError(f"Magnetic map does not cover domain: {name}")
                    values[missing] = NearestNDInterpolator(points, field[name])(target[missing])
                setattr(self, attribute, values.reshape(self.R.shape))
            return
        raise ValueError("Unknown magnetic-field model")

    def axisymmetric_divergence_b(self):
        rb = self.R * self.B_r
        div = np.gradient(rb, self.r, axis=0, edge_order=2) / self.R
        div += np.gradient(self.B_z, self.z, axis=1, edge_order=2)
        return div

    def divergence_report(self):
        divergence = self.axisymmetric_divergence_b()
        dr = float(np.mean(np.diff(self.r))) if len(self.r) > 1 else 1.0
        dz = float(np.mean(np.diff(self.z))) if len(self.z) > 1 else 1.0
        field_scale = max(
            float(np.max(np.abs(self.B_r))) / max(dr, 1e-30),
            float(np.max(np.abs(self.B_z))) / max(dz, 1e-30),
            1e-30,
        )
        maximum = float(np.max(np.abs(divergence)))
        rms = float(np.sqrt(np.mean(np.abs(divergence) ** 2)))
        return {
            "max_abs_T_m": maximum,
            "rms_T_m": rms,
            "relative_max": maximum / field_scale,
            "relative_rms": rms / field_scale,
            "grid_dr_m": dr,
            "grid_dz_m": dz,
        }
