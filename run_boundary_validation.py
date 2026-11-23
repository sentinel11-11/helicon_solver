"""Controlled boundary-distance experiment, independent of mode selection.

The physical device, source, dr, dz and PML physical thickness stay fixed.
Extra axial buffer cells increase Nz; they do not increase plasma.length.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from axis_audit import audit_axis
from diagnostics import Diagnostics
from domain import axial_domain
from main import load_config, save_fields
from preflight import validate_project_config, print_preflight
from solver import HeliconSolver


def padded_config(config, after_m=0.152, before_m=0.0):
    result = copy.deepcopy(config)
    start, end = axial_domain(config)
    dz = (end-start)/int(config["grid"]["Nz"])
    if not np.all(np.isfinite([after_m, before_m])) or min(after_m, before_m) < 0:
        raise ValueError("Buffer distances must be finite and nonnegative")
    before_cells, after_cells = int(round(before_m/dz)), int(round(after_m/dz))
    if before_cells + after_cells == 0:
        raise ValueError("At least one additional buffer cell is required")
    result.setdefault("domain", {}).update(
        axial_buffer_before_m=-start+before_cells*dz,
        axial_buffer_after_m=end-float(config["plasma"]["length"])+after_cells*dz,
        axial_buffer_medium="edge",
    )
    result["grid"]["Nz"] += before_cells+after_cells
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def main(config_path, frequency_mhz=10.0, after_m=0.152, before_m=0.0,
         output=None, prepare_only=False):
    config = load_config(config_path)
    config["wave"].update(frequency=float(frequency_mhz)*1e6,
                          frequencies=[float(frequency_mhz)*1e6], m_modes=[1], m_mode=1)
    root = Path(output) if output else Path(config_path).resolve().parent/"results_boundary_validation"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root/"axis_audit.json", audit_axis())
    variants = {"baseline": config, "axial_buffer": padded_config(config, after_m, before_m)}
    for name, variant in variants.items():
        variant["output"]["results_dir"] = str(root/name)
        print_preflight(validate_project_config(variant))
        (root/f"config_{name}.yaml").write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf8")
    if prepare_only:
        return variants
    reports, solvers = {}, {}
    for name, variant in variants.items():
        target = root/name
        target.mkdir(exist_ok=True)
        print(f"[boundary validation] {name}: {variant['grid']['Nr']} x {variant['grid']['Nz']}", flush=True)
        try:
            solver = HeliconSolver(variant)
            solver.solve()
            report = solver.energy_budget()
            _write_json(target/"energy_budget.json", report)
            save_fields(solver, target)
            diag = Diagnostics(solver, str(target), variant["output"].get("dpi", 160))
            diag.plot_absorbed_power()
            diag.plot_poynting()
            diag.plot_power_budget()
            with (target/"axial_power.csv").open("w", newline="", encoding="utf8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["z_m", "axial_power_W"])
                physical = solver.physical_domain_mask()
                flux = solver.compute_poynting_z()
                axial_power = 2*np.pi*solver.dr*np.sum(np.where(physical, flux, 0)*solver.r[:, None], axis=0)
                for j in np.flatnonzero(np.any(solver.physical_domain_mask(), axis=0)):
                    writer.writerow([solver.z[j], axial_power[j]])
            reports[name] = {"status": "linear_solve_passed", "residual": solver.relative_residual,
                             "power": report}
            # Retain only fields in a common, fixed observation interval.
            window = (solver.z >= 0.27) & (solver.z <= min(0.55, solver.L-0.03))
            radial = np.any(solver.non_pml_domain_mask(), axis=1)
            values = np.stack((solver.Er[radial][:, window], solver.Ephi[radial][:, window],
                               solver.Ez[radial][:, window]))
            values *= np.sqrt(solver.r[radial])[None, :, None]
            solvers[name] = (solver.z[window].copy(), values)
            del solver, diag
        except Exception as exc:
            reports[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            _write_json(target/"failure.json", reports[name])
        _write_json(root/"comparison.json", reports)
    if len(solvers) == 2:
        z0, field0 = solvers["baseline"]
        z1, field1 = solvers["axial_buffer"]
        if z0.shape == z1.shape and np.allclose(z0, z1, atol=1e-12, rtol=0):
            result = float(np.linalg.norm(field1-field0)/max(np.linalg.norm(field0), 1e-300))
            reports["comparison"] = {"fixed_window_m": [float(z0[0]), float(z0[-1])],
                "relative_E_change": result, "normalization": "baseline r-weighted field L2, radial PML excluded",
                "physical_model_validated": False}
    _write_json(root/"comparison.json", reports)
    print(f"[boundary validation] saved {root/'comparison.json'}", flush=True)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_scan_1MHz_200x800.yaml")
    parser.add_argument("--frequency", type=float, default=10.0, help="MHz")
    parser.add_argument("--buffer-after", type=float, default=0.152, help="additional m, rounded to whole cells")
    parser.add_argument("--buffer-before", type=float, default=0.0, help="additional m, rounded to whole cells")
    parser.add_argument("--output")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    reports = main(args.config, args.frequency, args.buffer_after, args.buffer_before, args.output, args.prepare_only)
    if not args.prepare_only and any(value.get("status") == "failed" for value in reports.values()):
        raise SystemExit(2)
