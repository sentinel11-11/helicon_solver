"""Recompute local radial modes without repeating the expensive driven solve."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from diagnostics import Diagnostics
from main import load_config
from plasma import configured_theoretical_kz
from run_all import _fingerprint, _save_mode
from solver import HeliconSolver, select_dispersion_mode, radial_mode_overlap


def _physical_overlap(a, b, end):
    def trim(mode):
        mask = mode["r"] < end
        return {key: np.asarray(mode[key])[mask] for key in ("r", "Er", "Ephi", "Ez")}
    return radial_mode_overlap(trim(a), trim(b))


def main(config_path, radial_grids=None, output=None, compare_unscaled=False):
    config = load_config(config_path)
    base_nr = int(config["grid"].get("dispersion_Nr", config["grid"]["Nr"]))
    base_pml = int(config["grid"].get("dispersion_pml_r", config["grid"]["pml_r"]))
    grids = [base_nr] if radial_grids is None else sorted(set(radial_grids))
    root = Path(output) if output else Path(config_path).resolve().parent/"results_radial_recheck"
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    methods = ("none", "equilibrated") if compare_unscaled else ("equilibrated",)
    for method in methods:
        previous = None
        for nr in grids:
            cfg = copy.deepcopy(config)
            pml = int(round(base_pml*nr/base_nr))
            cfg["grid"].update(dispersion_Nr=nr, dispersion_pml_r=pml)
            cfg["solver"]["radial_pencil_scaling"] = method
            target = root/method/f"Nr{nr}"
            target.mkdir(parents=True, exist_ok=True)
            solver = HeliconSolver(cfg, build_operator=False)
            solver.run_fingerprint = _fingerprint(cfg)
            modes = solver.compute_dispersion_modes()
            with (target/"radial_spectrum.csv").open("w", newline="", encoding="utf8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["kz_real_rad_m", "kz_imag_rad_m", "qep_residual", "roughness", "mesh_screened"])
                writer.writerows([mode["kz"].real, mode["kz"].imag, mode["qep_relative_residual"],
                                  mode["radial_grid_oscillation"], mode["radial_grid_resolved"]] for mode in modes)
            screened = [mode for mode in modes if mode["radial_grid_resolved"]]
            record = {"radial_Nr": nr, "radial_pml_cells": pml,
                      "scaling": method, "pencil": solver.dispersion_diagnostics,
                      "source_fingerprint": solver.run_fingerprint,
                      "field_solve_repeated": False, "physical_mode_validated": False,
                      "frequency_Hz": float(cfg["wave"]["frequency"]),
                      "m": int(cfg["wave"].get("m_mode", 0)),
                      "driven_grid": [solver.Nr, solver.Nz],
                      "local_z_m": solver.dispersion_z_m,
                      "background_radial_samples": solver.Nr,
                      "boundary": cfg.get("boundary", {}),
                      "runtime": {"system": platform.system(), "numpy": np.__version__, "scipy": scipy.__version__}}
            if not screened:
                record["status"] = "no_mesh_screened_mode"
                reports.append(record)
                (root/"radial_recheck.json").write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf8")
                print(f"[radial recheck] {method}, Nr={nr}: no mode passed mesh screen", flush=True)
                previous = None
                continue
            theory = abs(configured_theoretical_kz(solver.omega, solver.dispersion_reference_n_e,
                        solver.dispersion_reference_B0_T, cfg["plasma"], cfg["solver"],
                        m_mode=int(cfg["wave"].get("m_mode", 0))))
            overlap = None
            if previous is None:
                selected = select_dispersion_mode(screened, theory, require_resolved=True)
                record["selection"] = "heuristic initial seed; theory is a branch hint only"
            else:
                physical_end = solver.r_extent*(1-pml/nr)
                selected = max(screened, key=lambda mode: _physical_overlap(previous, mode, physical_end))
                overlap = _physical_overlap(previous, selected, physical_end)
                record["selection"] = "maximum r-weighted overlap outside radial absorber"
                record["relative_kz_change"] = float(abs(selected["kz"]-previous["kz"])/abs(previous["kz"]))
            record.update(status="numerically_screened_local_mode",
                          kz_real_rad_m=selected["kz"].real, kz_imag_rad_m=selected["kz"].imag,
                          qep_residual=selected["qep_relative_residual"],
                          roughness=selected["radial_grid_oscillation"],
                          physical_overlap_previous_grid=overlap)
            _save_mode(target/"selected_mode.npz", selected)
            Diagnostics(solver, str(target), cfg["output"].get("dpi", 160)).plot_eigenmode_profile(selected)
            reports.append(record)
            (root/"radial_recheck.json").write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf8")
            print(f"[radial recheck] {method}, Nr={nr}: kz={selected['kz']/100} 1/cm; "
                  f"QEP={selected['qep_relative_residual']:.3e}", flush=True)
            previous = selected
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_passive_10MHz_200x800.yaml")
    parser.add_argument("--nr", type=int, nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--compare-unscaled", action="store_true")
    args = parser.parse_args()
    reports = main(args.config, args.nr, args.output, args.compare_unscaled)
    if any(r["status"] != "numerically_screened_local_mode" for r in reports):
        raise SystemExit(2)
