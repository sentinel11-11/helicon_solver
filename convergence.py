from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from main import load_config
from plasma import configured_theoretical_kz
from preflight import validate_project_config
from solver import HeliconSolver, select_dispersion_mode


FIELDS = (
    "Nr", "Nz", "dispersion_Nr", "pml_r", "pml_z", "frequency_MHz", "m_mode",
    "kz_real_rad_m", "kz_imag_rad_m", "attenuation_m_inv", "kz_change_percent",
    "relative_linear_residual", "field_fit_accepted", "field_fit_kz_rad_m", "elapsed_s",
    "radial_grid_oscillation", "radial_grid_resolved", "convergence_accepted",
)


def _mesh(text: str) -> tuple[int, int]:
    try:
        nr, nz = (int(v) for v in text.lower().split("x"))
    except (ValueError, TypeError) as exc:
        raise ValueError("Mesh must have the form Nr x Nz, e.g. 50x400") from exc
    if nr < 8 or nz < 16:
        raise ValueError("Mesh is too small")
    return nr, nz


def main(config_path: str = "config.yaml", *, meshes: list[str],
         frequency_mhz: float | None = None, mode: int | None = None,
         eigen_only: bool = False, output: str = "convergence_scan.csv") -> list[dict]:
    original = load_config(config_path)
    baseline = validate_project_config(original)
    f = float(frequency_mhz * 1e6 if frequency_mhz is not None else original["wave"]["frequency"])
    m = int(mode if mode is not None else original["wave"].get("m_mode", 1))
    sequence = [_mesh(text) for text in meshes]
    if len(set(sequence)) != len(sequence):
        raise ValueError("Convergence meshes must be distinct")
    previous = None
    rows = []
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    for nr, nz in sequence:
        cfg = yaml.safe_load(yaml.safe_dump(original))
        old = original["grid"]
        cfg["grid"]["Nr"], cfg["grid"]["Nz"] = nr, nz
        cfg["grid"]["pml_r"] = max(1, round(int(old["pml_r"]) * nr / baseline.grid[0]))
        cfg["grid"]["pml_z"] = max(1, round(int(old["pml_z"]) * nz / baseline.grid[1]))
        cfg["grid"]["dispersion_Nr"] = max(16, round(int(old.get("dispersion_Nr", baseline.grid[0])) * nr / baseline.grid[0]))
        cfg["grid"]["dispersion_pml_r"] = max(1, round(int(old.get("dispersion_pml_r", old["pml_r"])) * nr / baseline.grid[0]))
        cfg["wave"]["frequency"] = f
        cfg["wave"]["m_mode"] = m
        validate_project_config(cfg)
        print(f"[convergence] grid={nr}x{nz}, m={m}, f={f/1e6:g} MHz", flush=True)
        started = perf_counter()
        solver = HeliconSolver(cfg, build_operator=not eigen_only)
        residual = None
        fit_ok = False
        field_k = None
        if not eigen_only:
            solver.solve()
            residual = solver.relative_residual
            field_k, _ = solver.compute_wave_structure()
            fit_ok = bool(solver.kz_fit_reliable and np.isfinite(field_k))
        modes = solver.compute_dispersion_modes()
        theory = abs(configured_theoretical_kz(
            solver.omega, solver.params.n_e, solver.params.B0,
            solver.cfg["plasma"], solver.cfg.get("solver", {})
        ))
        if previous is None:
            selected = select_dispersion_mode(modes, theory)
        else:
            selected = select_dispersion_mode(modes, theory, previous_kz=previous["kz"],
                                              frequency_ratio=1.0, previous_mode=previous)
        delta = None if previous is None else 100 * abs(selected["kz"].real - previous["kz"].real) / max(abs(previous["kz"].real), 1e-30)
        convergence_accepted = bool(
            selected.get("radial_grid_resolved", False)
            and (delta is None or delta <= float(cfg.get("solver", {}).get("convergence_kz_change_percent", 5.0)))
        )
        row = {
            "Nr": nr, "Nz": nz, "dispersion_Nr": cfg["grid"]["dispersion_Nr"],
            "pml_r": cfg["grid"]["pml_r"], "pml_z": cfg["grid"]["pml_z"],
            "frequency_MHz": f / 1e6, "m_mode": m,
            "kz_real_rad_m": float(selected["kz"].real),
            "kz_imag_rad_m": float(selected["kz"].imag),
            "attenuation_m_inv": float(selected["attenuation_m_inv"]),
            "kz_change_percent": delta,
            "relative_linear_residual": residual,
            "field_fit_accepted": fit_ok,
            "field_fit_kz_rad_m": float(field_k) if fit_ok else None,
            "elapsed_s": round(perf_counter() - started, 3),
            "radial_grid_oscillation": float(selected.get("radial_grid_oscillation", np.nan)),
            "radial_grid_resolved": bool(selected.get("radial_grid_resolved", False)),
            "convergence_accepted": convergence_accepted,
        }
        rows.append(row)
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        previous = selected
        print(f"[convergence] kz={selected['kz'].real / 100:.5f} cm^-1, delta={delta}, elapsed={row['elapsed_s']} s", flush=True)
    print(f"[saved] {target}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDFD radial/axial mesh convergence study")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--grids", nargs="+", required=True, help="meshes, e.g. 40x320 60x480 100x800")
    parser.add_argument("--frequency", type=float, help="frequency in MHz; defaults to wave.frequency")
    parser.add_argument("--mode", type=int, help="azimuthal harmonic")
    parser.add_argument("--eigen-only", action="store_true", help="skip driven-field LU solve")
    parser.add_argument("--output", default="convergence_scan.csv")
    args = parser.parse_args()
    main(args.config, meshes=args.grids, frequency_mhz=args.frequency,
         mode=args.mode, eigen_only=args.eigen_only, output=args.output)
