from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from diagnostics import Diagnostics
from probe import MagneticProbe, ThreeAxisProbe, simulate_probe_scan
from solver import HeliconSolver
from preflight import validate_project_config, print_preflight


def load_config(path):
    location = Path(path).resolve()
    config = yaml.safe_load(location.read_text(encoding="utf-8-sig"))
    for section, field in (("plasma", "density_axial"), ("plasma", "magnetic_field")):
        entry = config.get(section, {}).get(field, {})
        for key in ("path",):
            if key in entry:
                entry[key] = str((location.parent / entry[key]).resolve())
    if config.get("plasma", {}).get("density_map_csv"):
        config["plasma"]["density_map_csv"] = str((location.parent / config["plasma"]["density_map_csv"]).resolve())
    probes = config.get("probe_3axis", {})
    if probes.get("calibration_csv"):
        probes["calibration_csv"] = str((location.parent / probes["calibration_csv"]).resolve())
    return config


def save_fields(solver, target):
    path = Path(target)
    path.mkdir(parents=True, exist_ok=True)
    br, bp, bz = solver.compute_magnetic_field()
    arrays = dict(r=solver.r, z=solver.z, E_r=solver.Er, E_phi=solver.Ephi,
                  E_z=solver.Ez, B_r=br, B_phi=bp, B_z=bz,
                  ne=solver.background.n_e, B0_r=solver.background.B_r,
                  B0_z=solver.background.B_z, absorbed_power=solver.compute_absorbed_power(),
                  dielectric_loss=solver.compute_material_absorption())
    arrays["physical_mask"] = solver.physical_domain_mask()
    arrays["non_pml_mask"] = solver.non_pml_domain_mask()
    hr, hp, hz = solver.compute_magnetic_intensity()
    arrays.update(H_r=hr, H_phi=hp, H_z=hz, absorber_epsilon_loss=solver.absorber_epsilon_loss)
    metadata = {
        "frequency_Hz": float(solver.cfg["wave"]["frequency"]),
        "m": int(solver.cfg["wave"].get("m_mode", 0)),
        "grid": [solver.Nr, solver.Nz],
        "boundary": solver.cfg.get("boundary", {}),
        "axis_regularization": solver.cfg.get("solver", {}).get("axis_regularization", "legacy"),
        "source_fingerprint": getattr(solver, "run_fingerprint", None),
    }
    (path/"fields_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf8")
    if solver.cfg.get("output", {}).get("save_fields_npz", True):
        np.savez_compressed(path / "fields.npz", **arrays)
    if solver.cfg.get("output", {}).get("save_hdf5", False):
        import h5py
        with h5py.File(path / "fields.h5", "w") as h5:
            for name, values in arrays.items():
                h5.create_dataset(name, data=values, compression="gzip")
    return arrays


def main(config_path="config.yaml", *, dry_run=False):
    config = load_config(config_path)
    report = validate_project_config(config)
    print_preflight(report)
    if dry_run:
        return {"grid": list(report.grid), "complex_unknowns": report.unknowns, "method": report.method}
    output = Path(config["output"]["results_dir"])
    if not output.is_absolute():
        output = Path(config_path).resolve().parent / output
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    solver = HeliconSolver(config)
    if solver.double_layer is not None:
        solver.double_layer.export(output / "prescribed_double_layer.csv")
    solver.solve()
    power_budget = solver.energy_budget()
    (output / "energy_budget.json").write_text(json.dumps(power_budget, indent=2, allow_nan=False), encoding="utf8")
    solver.check_boundary_passivity(power_budget)
    diagnostics = Diagnostics(solver, str(output), config["output"].get("dpi", 160))
    diagnostics.plot_field_map()
    diagnostics.plot_radial_profiles()
    diagnostics.plot_magnetic_field_map()
    diagnostics.plot_magnetic_profiles()
    diagnostics.plot_wave_structure()
    diagnostics.plot_absorbed_power()
    diagnostics.plot_poynting()
    diagnostics.plot_power_budget()
    diagnostics.plot_dispersion_map()
    probe_config = config["probe"]
    if config.get("probe_3axis", {}).get("enabled", False):
        probe = ThreeAxisProbe(config["probe_3axis"])
    else:
        probe = MagneticProbe(probe_config["n_turns"], probe_config["area"],
                              probe_config["resistance"], probe_config["capacitance"])
    scan = simulate_probe_scan(solver, probe,
                               config["scan"]["z_positions"], config["scan"]["r_positions"])
    diagnostics.plot_probe_scan(scan)
    with (output / "probe_scan.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ("z", "r", "absVr", "absVphi", "absVz", "phaseVr", "phaseVphi", "phaseVz",
                "phaseBr", "phaseBphi", "phaseBz")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in scan)
    diagnostics.compare_with_theory()
    save_fields(solver, output)
    metadata = {
        "model": "cold_plasma_2p5d_local_gyrotropic_maxwell",
        "frequency_Hz": float(config["wave"]["frequency"]),
        "m": int(config["wave"].get("m_mode", 0)),
        "field_phase_convention": "exp(i*m*phi - i*kz*z + i*omega*t)",
        "relative_residual": solver.relative_residual,
        "residual_equation_space": solver.residual_equation_space,
        "grid": [solver.Nr, solver.Nz],
        "physical_absorbed_power_W": power_budget["physical"]["absorbed_power_W"],
        "power_budget": power_budget,
        "magnetic_divergence_grid_indicator": solver.magnetic_divergence_quality(),
        "magnetic_field_model": solver.background.magnetic_metadata,
        "dispersion_reference_n_e_m3": getattr(solver, "dispersion_reference_n_e", None),
        "dispersion_reference_B0_T": getattr(solver, "dispersion_reference_B0_T", None),
        "double_layer_status": "prescribed_uncoupled" if solver.double_layer else "disabled",
        "double_layer_self_consistent": False,
        "elapsed_s": round(perf_counter() - started, 3),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(f"[done] {output} | residual={solver.relative_residual:.3e}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axisymmetric helicon RF model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without solving")
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
