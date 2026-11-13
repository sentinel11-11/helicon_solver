from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreflightReport:
    unknowns: int
    grid: tuple[int, int]
    method: str
    frequency_count: int
    mode_count: int
    warnings: tuple[str, ...]

    @property
    def run_count(self) -> int:
        return self.frequency_count * self.mode_count


def validate_project_config(config: dict[str, Any]) -> PreflightReport:
    required = ("plasma", "wave", "grid", "solver", "antenna", "output")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required sections: {', '.join(missing)}")

    p = config["plasma"]
    g = config["grid"]
    w = config["wave"]
    s = config["solver"]
    ant = config["antenna"]
    nr, nz = int(g["Nr"]), int(g["Nz"])
    if nr < 8 or nz < 16:
        raise ValueError("grid.Nr must be >= 8 and grid.Nz must be >= 16")
    a = float(p["radius"])
    length = float(p["length"])
    extent = float(g["radial_extent"])
    if not (0 < a < extent and length > 0):
        raise ValueError("Expected 0 < plasma.radius < grid.radial_extent and plasma.length > 0")
    radial_pml = int(g["pml_r"])
    axial_pml = int(g["pml_z"])
    if not (0 <= radial_pml < nr // 2 and 0 <= axial_pml < nz // 2):
        raise ValueError("PML cell counts exceed the allowed grid dimensions")
    if extent * (1.0 - radial_pml / nr) <= a:
        raise ValueError("The radial PML overlaps the plasma radius")
    frequencies = w.get("frequencies") or ([w["frequency"]] if "frequency" in w else [])
    modes = w.get("m_modes", [w.get("m_mode", 0)])
    if not frequencies or any(not np.isfinite(float(f)) or float(f) <= 0 for f in frequencies):
        raise ValueError("wave.frequencies must contain strictly positive frequencies")
    if not modes or any(int(mode) != float(mode) for mode in modes):
        raise ValueError("wave.m_modes must contain integer azimuthal indices")
    method = str(s.get("method", "auto")).lower()
    if method not in ("direct", "auto", "gmres", "bicgstab"):
        raise ValueError(f"Unsupported solver.method: {method}")
    if not np.isfinite(float(s.get("tol", 1e-6))) or float(s.get("tol", 1e-6)) <= 0:
        raise ValueError("solver.tol must be finite and positive")
    if float(p.get("n_e_peak", 0.0)) <= 0 or float(p.get("ion_mass", 0.0)) <= 0:
        raise ValueError("Electron density and ion mass must be positive")
    if float(ant["length"]) <= 0 or float(ant["z_position"]) < 0:
        raise ValueError("Invalid antenna placement")
    if float(ant["z_position"]) + float(ant["length"]) >= length:
        raise ValueError("The antenna must end before the axial domain boundary")
    radius = float(ant.get("radius_m", float(ant.get("radius_factor", 0.8)) * a))
    if not 0 < radius < extent:
        raise ValueError("Antenna radius must be inside the computational domain")
    given_width = float(ant.get("radial_width_m", 0.0))
    if given_width < 0:
        raise ValueError("Antenna radial width must not be negative")
    width = max(given_width, 0.5 * extent / nr)
    if radius + 3 * width >= extent * (1.0 - radial_pml / nr):
        raise ValueError("Antenna radial source overlaps the PML")

    unknowns = 3 * nr * nz
    warnings = []
    if method == "auto" and unknowns > int(s.get("direct_max_unknowns", 25000)):
        warnings.append("The auto solver selects iterative GMRES; ILU may fail on this FDFD matrix")
    if method == "direct" and unknowns >= 200_000:
        warnings.append("Large sparse LU factorization may use much more RAM than the stored matrix")
    if bool(s.get("use_gpu", False)) and method == "direct":
        warnings.append("solver.method=direct uses CPU sparse LU; set solver.method=gmres to use CUDA")
    if config.get("double_layer", {}).get("enabled", False):
        warnings.append("Double-layer profile is prescribed, not self-consistent or RF-coupled")
    if not config.get("plasma", {}).get("density_map_csv"):
        warnings.append("The density profile is illustrative until replaced by measured PLM-M data")
    if g.get("dispersion_Nr", nr) < nr:
        warnings.append("The eigenmode radial grid is coarser than the driven-field radial grid")
    magnetic = p.get("magnetic_field", {})
    if magnetic.get("model") == "plmm_assembly":
        magnets = magnetic.get("magnets", {})
        sections = magnetic.get("solenoid", {}).get("sections", [])
        representation = str(magnetic.get("representation", "axisymmetric_average"))
        if representation == "full_3d":
            raise ValueError("The 2-D FDFD project cannot use representation=full_3d; run magnetic_3d.py separately")
        if representation != "axisymmetric_average":
            raise ValueError("Unknown PLM-M magnetic-field representation")
        if int(magnets.get("count", 8)) != 8:
            raise ValueError("plasma.magnetic_field.magnets.count must be 8 for PLM-M")
        if not np.isclose(float(magnets.get("length_m", 0.608)), 0.608, rtol=0.0, atol=1e-12):
            raise ValueError("PLM-M magnet length must be 0.608 m")
        if not np.isclose(float(magnets.get("cross_section_m", 0.018)), 0.018, rtol=0.0, atol=1e-12):
            raise ValueError("PLM-M magnet cross-section side must be 0.018 m")
        if len(sections) != 2:
            raise ValueError("PLM-M solenoid must be configured as two sections")
        warnings.append("The PLM-M field model uses the supplied 8 x (0.608 m, 0.018 m square) magnet dimensions")
        warnings.append("The 2-D FDFD run uses the axisymmetric average of the discrete 3-D cusp field")
        if magnetic.get("geometry_status", "provisional") != "measured":
            warnings.append("Magnet coordinates, polarities and solenoid calibration are provisional")
        if magnetic.get("normalization", {}).get("enabled", True):
            warnings.append("The assembly field is normalized to plasma.B0 because coil current and remanence calibration are not supplied")
        nozzle = config.get("geometry", {}).get("nozzle", {})
        if nozzle.get("enabled", False):
            if float(nozzle.get("z_end_m", length)) > length + 1e-12:
                raise ValueError("geometry.nozzle.z_end_m must lie inside plasma.length")
            if float(nozzle.get("z_start_m", 0.0)) < 0:
                raise ValueError("geometry.nozzle.z_start_m must be nonnegative")
    return PreflightReport(
        unknowns=unknowns, grid=(nr, nz), method=method,
        frequency_count=len(frequencies), mode_count=len(modes), warnings=tuple(warnings),
    )


def print_preflight(report: PreflightReport) -> None:
    nr, nz = report.grid
    print(f"[preflight] grid={nr} x {nz}; complex unknowns={report.unknowns:,}; method={report.method}")
    print(f"[preflight] planned cases={report.run_count}")
    for warning in report.warnings:
        print(f"[preflight] NOTE: {warning}")
