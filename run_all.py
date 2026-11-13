from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from diagnostics import Diagnostics
from main import load_config, save_fields
from plasma import theoretical_kz
from preflight import print_preflight, validate_project_config
from solver import HeliconSolver, select_dispersion_mode


COLUMNS = (
    "frequency_Hz", "frequency_MHz", "m_mode",
    "kz_radial_eigenmode_real_rad_m", "kz_radial_eigenmode_imag_rad_m",
    "kz_radial_eigenmode_cm_inv", "attenuation_eigenmode_m_inv",
    "kz_driven_field_phase_rad_m", "kz_driven_field_phase_cm_inv",
    "kz_whistler_asymptote_rad_m", "kz_whistler_asymptote_cm_inv",
    "eigenmode_relative_delta_percent", "phase_fit_quality_R2",
    "phase_cycles_over_window", "phase_candidate_spread_rad_m",
    "dispersion_radial_grid_Nr", "eigenmode_damping_ratio",
    "relative_linear_residual", "phase_fit_reliable", "wave_fit_purity",
    "wave_fit_reflection_ratio", "wave_fit_reason", "radial_eigenmode_local_z_m",
    "elapsed_s", "radial_grid_oscillation", "radial_grid_resolved", "status",
)

SOURCE_FILES = (
    "solver.py", "discretization.py", "plasma.py", "boundary.py", "background.py", "magnetic_system.py", "magnetic_3d.py", "double_layer.py",
    "probe.py", "wave_analysis.py", "diagnostics.py", "main.py", "run_all.py",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") if row.get(key) is not None else "" for key in COLUMNS})
    os.replace(temp, path)


def _finite(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _fingerprint(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode("utf-8"))
    source_root = Path(__file__).resolve().parent
    for filename in SOURCE_FILES:
        source = source_root / filename
        digest.update(filename.encode("utf-8"))
        digest.update(source.read_bytes())
    for section in (config.get("plasma", {}), config.get("probe_3axis", {})):
        for key in ("density_map_csv", "calibration_csv"):
            if section.get(key):
                digest.update(Path(section[key]).read_bytes())
        for key in ("density_axial", "magnetic_field"):
            if isinstance(section.get(key), dict) and section[key].get("path"):
                digest.update(Path(section[key]["path"]).read_bytes())
    return digest.hexdigest()


def _tag(frequency: float, mode: int) -> str:
    label = f"{frequency / 1e6:.6f}".rstrip("0").rstrip(".")
    if "." not in label:
        label += ".0"
    return f"f{label}MHz_m{mode}"


def _save_mode(path: Path, mode: dict[str, Any]) -> None:
    np.savez_compressed(
        path, kz=np.array(mode["kz"], dtype=np.complex128),
        r=mode["r"], Er=mode["Er"], Ephi=mode["Ephi"], Ez=mode["Ez"],
        attenuation_m_inv=np.array(mode["attenuation_m_inv"], dtype=float),
        damping_ratio=np.array(mode["damping_ratio"], dtype=float),
        radial_grid_oscillation=np.array(mode.get("radial_grid_oscillation", float("nan")), dtype=float),
        radial_grid_resolved=np.array(mode.get("radial_grid_resolved", False), dtype=bool),
    )


def _load_mode(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "kz": complex(data["kz"].item()),
            "r": data["r"].copy(),
            "Er": data["Er"].copy(), "Ephi": data["Ephi"].copy(),
            "Ez": data["Ez"].copy(),
            "attenuation_m_inv": float(data["attenuation_m_inv"]),
            "damping_ratio": float(data["damping_ratio"]),
            "radial_grid_oscillation": float(data["radial_grid_oscillation"]) if "radial_grid_oscillation" in data else float("nan"),
            "radial_grid_resolved": bool(data["radial_grid_resolved"]) if "radial_grid_resolved" in data else False,
        }


def _plot_summary(rows: list[dict[str, Any]], modes: list[int], path: Path, dpi: int) -> None:
    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in modes:
        sequence = sorted((r for r in valid if r["m_mode"] == mode), key=lambda r: r["frequency_Hz"])
        if not sequence:
            continue
        freq = np.asarray([r["frequency_MHz"] for r in sequence], dtype=float)
        eigen = np.asarray([
            float(r["kz_radial_eigenmode_cm_inv"]) if r.get("radial_grid_resolved", False) else np.nan
            for r in sequence
        ], dtype=float)
        ax.plot(freq, eigen, "o-", label=f"radial eigenmode (mesh-screened), m={mode}")
        unverified = [r for r in sequence if not r.get("radial_grid_resolved", False)]
        if unverified:
            ax.scatter(
                [r["frequency_MHz"] for r in unverified],
                [r["kz_radial_eigenmode_cm_inv"] for r in unverified],
                marker="x", s=50, label=f"unverified radial mode, m={mode}",
            )
        ax.plot(freq, [r["kz_whistler_asymptote_cm_inv"] for r in sequence],
                "--", alpha=0.6, label=f"whistler asymptote, m={mode}")
        accepted = [r for r in sequence if r["phase_fit_reliable"]]
        if accepted:
            ax.scatter([r["frequency_MHz"] for r in accepted],
                       [r["kz_driven_field_phase_cm_inv"] for r in accepted],
                       marker="+", s=55, label=f"accepted field fit, m={mode}")
    ax.set(xlabel="f, MHz", ylabel=r"Re($k_z$), cm$^{-1}$", title="Helicon FDFD dispersion (mesh quality indicated)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def _result_row(solver: HeliconSolver, selected: dict[str, Any], f: float, m: int, elapsed: float) -> dict[str, Any]:
    n_ref = float(getattr(solver, "dispersion_reference_n_e", solver.params.n_e))
    b_ref = float(getattr(solver, "dispersion_reference_B0_T", solver.params.B0))
    k_theory = float(abs(theoretical_kz(solver.omega, n_ref, b_ref, m_mode=m)))
    field_k, _ = solver.compute_wave_structure()
    valid = bool(getattr(solver, "kz_fit_reliable", False) and np.isfinite(field_k))
    return {
        "frequency_Hz": f,
        "frequency_MHz": f / 1e6,
        "m_mode": m,
        "kz_radial_eigenmode_real_rad_m": float(selected["kz"].real),
        "kz_radial_eigenmode_imag_rad_m": float(selected["kz"].imag),
        "kz_radial_eigenmode_cm_inv": float(selected["kz"].real / 100),
        "attenuation_eigenmode_m_inv": float(selected["attenuation_m_inv"]),
        "kz_driven_field_phase_rad_m": float(abs(field_k)) if valid else None,
        "kz_driven_field_phase_cm_inv": float(abs(field_k) / 100) if valid else None,
        "kz_whistler_asymptote_rad_m": k_theory,
        "kz_whistler_asymptote_cm_inv": k_theory / 100,
        "eigenmode_relative_delta_percent": 100 * abs(abs(selected["kz"].real) - k_theory) / max(k_theory, 1e-30),
        "phase_fit_quality_R2": _finite(getattr(solver, "kz_fit_quality", 0)),
        "phase_cycles_over_window": _finite(getattr(solver, "kz_phase_cycles", float("nan"))),
        "phase_candidate_spread_rad_m": _finite(getattr(solver, "kz_candidate_spread", float("nan"))),
        "dispersion_radial_grid_Nr": int(solver.dispersion_grid_Nr),
        "eigenmode_damping_ratio": _finite(selected.get("damping_ratio", float("nan"))),
        "relative_linear_residual": _finite(solver.relative_residual),
        "phase_fit_reliable": valid,
        "wave_fit_purity": _finite(getattr(solver, "kz_fit_purity", float("nan"))),
        "wave_fit_reflection_ratio": _finite(getattr(solver, "kz_reflection_ratio", float("nan"))),
        "wave_fit_reason": getattr(solver, "kz_fit_reason", ""),
        "radial_eigenmode_local_z_m": float(solver.dispersion_z_m),
        "elapsed_s": round(elapsed, 3),
        "radial_grid_oscillation": _finite(selected.get("radial_grid_oscillation", float("nan"))),
        "radial_grid_resolved": bool(selected.get("radial_grid_resolved", False)),
        "status": "ok",
    }


def main(
    config_path: str = "config.yaml",
    *,
    resume: bool = False,
    modes: list[int] | None = None,
    frequencies_mhz: list[float] | None = None,
    dry_run: bool = False,
    save_scan_fields: bool | None = None,
    plots: bool = True,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    path = Path(config_path).resolve()
    base = load_config(str(path))
    report = validate_project_config(base)
    print_preflight(report)

    frequencies = sorted(set(float(f) for f in base["wave"]["frequencies"]))
    azimuthal_modes = sorted(set(int(m) for m in base["wave"]["m_modes"]))
    if modes is not None:
        invalid = set(modes).difference(azimuthal_modes)
        if invalid:
            raise ValueError(f"Modes not present in the configuration: {sorted(invalid)}")
        azimuthal_modes = [m for m in azimuthal_modes if m in modes]
    if frequencies_mhz is not None:
        targets = [float(f) * 1e6 for f in frequencies_mhz]
        for target in targets:
            if not any(np.isclose(f, target, rtol=0, atol=0.5) for f in frequencies):
                raise ValueError(f"Frequency {target / 1e6} MHz not in configuration")
        frequencies = [f for f in frequencies if any(np.isclose(f, target, rtol=0, atol=0.5) for target in targets)]
    if not frequencies or not azimuthal_modes:
        raise ValueError("No cases remain after filtering")

    runs = [(f, m) for f in frequencies for m in azimuthal_modes]
    print("=" * 68)
    print("HELICON FDFD SCAN: vector fields and radial eigenmodes")
    print("=" * 68)
    print("Frequencies, MHz:", [f / 1e6 for f in frequencies])
    print("Azimuthal modes:", azimuthal_modes)
    print("Selected cases:", len(runs))
    if dry_run:
        for f, m in runs:
            print(f"[planned] {_tag(f, m)}")
        return []

    output = Path(base["output"]["results_dir"])
    if not output.is_absolute():
        output = path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(base)
    save_fields_enabled = bool(base["output"].get("save_scan_fields", True)) if save_scan_fields is None else save_scan_fields
    rows: list[dict[str, Any]] = []
    previous: dict[int, tuple[dict[str, Any], float] | None] = {mode: None for mode in azimuthal_modes}

    for count, (frequency, mode) in enumerate(runs, start=1):
        tag = _tag(frequency, mode)
        case_dir = output / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        row_path = case_dir / "result.json"
        mode_path = case_dir / "selected_mode.npz"
        print(f"\n>>> [{count}/{len(runs)}] {tag}", flush=True)
        has_required_fields = not save_fields_enabled or (case_dir / "fields.npz").is_file()
        if resume and row_path.is_file() and mode_path.is_file() and has_required_fields:
            try:
                record = json.loads(row_path.read_text(encoding="utf-8"))
                if record.get("fingerprint") == fingerprint and record["row"]["status"] == "ok":
                    selected = _load_mode(mode_path)
                    previous[mode] = (selected, frequency)
                    rows.append(record["row"])
                    _atomic_csv(output / "dispersion_scan.csv", rows)
                    print("[resume] valid completed case reused", flush=True)
                    continue
            except (KeyError, ValueError, OSError, TypeError) as exc:
                print(f"[resume] stored case not reusable: {exc}", flush=True)

        started = perf_counter()
        try:
            config = yaml.safe_load(yaml.safe_dump(base))
            config["wave"]["frequency"] = frequency
            config["wave"]["m_mode"] = mode
            if save_fields_enabled:
                config["output"]["save_fields_npz"] = True
            (case_dir / "case_config.yaml").write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            solver = HeliconSolver(config)
            solver.solve()
            if save_fields_enabled:
                save_fields(solver, case_dir)
            solver.compute_wave_structure()
            candidate_modes = solver.compute_dispersion_modes()
            k_theory = theoretical_kz(
                solver.omega,
                solver.dispersion_reference_n_e,
                solver.dispersion_reference_B0_T,
                m_mode=mode,
            )
            prior = previous[mode]
            if prior is None:
                selected = select_dispersion_mode(candidate_modes, abs(k_theory))
            else:
                old_mode, previous_frequency = prior
                selected = select_dispersion_mode(
                    candidate_modes, abs(k_theory), previous_kz=old_mode["kz"],
                    frequency_ratio=frequency / previous_frequency, previous_mode=old_mode,
                )
            _save_mode(mode_path, selected)
            if not selected.get("radial_grid_resolved", False):
                print(
                    f"[mesh warning] m={mode}: selected radial eigenmode fails the "
                    f"grid-oscillation screen (indicator={selected.get('radial_grid_oscillation', float('nan')):.3f}); "
                    "the eigenvalue is provisional and must not be used for physical conclusions",
                    flush=True,
                )
            if solver.double_layer is not None:
                solver.double_layer.export(case_dir / "prescribed_double_layer.csv")
            if plots:
                diag = Diagnostics(solver, str(case_dir), base["output"].get("dpi", 160))
                diag.plot_field_map()
                diag.plot_radial_profiles()
                diag.plot_magnetic_field_map()
                diag.plot_magnetic_profiles()
                diag.plot_wave_structure()
                diag.plot_eigenmode_profile(selected)
                diag.compare_with_theory(selected)
                if base["output"].get("scan_extra_plots", True):
                    diag.plot_absorbed_power()
                    diag.plot_poynting()
            elapsed = perf_counter() - started
            row = _result_row(solver, selected, frequency, mode, elapsed)
            previous[mode] = (selected, frequency)
            _atomic_json(row_path, {
                "fingerprint": fingerprint,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_config": path.name,
                "wave_convention": "exp(+i*omega*t - i*kz*z + i*m*phi)",
                "double_layer": "prescribed_uncoupled" if solver.double_layer else "off",
                "row": row,
            })
            print(f"[case done] {tag}; {elapsed:.1f} s; residual={solver.relative_residual:.3e}", flush=True)
        except Exception as exc:
            elapsed = perf_counter() - started
            row = {"frequency_Hz": frequency, "frequency_MHz": frequency / 1e6,
                   "m_mode": mode, "elapsed_s": round(elapsed, 3),
                   "status": f"error: {type(exc).__name__}: {exc}"}
            _atomic_json(case_dir / "failure.json", {
                "created_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
                "error_type": type(exc).__name__, "message": str(exc),
            })
            print(f"[ERROR] {tag}: {type(exc).__name__}: {exc}", flush=True)
        rows.append(row)
        _atomic_csv(output / "dispersion_scan.csv", rows)
        if fail_fast and row["status"] != "ok":
            break

    if plots:
        _plot_summary(rows, azimuthal_modes, output / "dispersion_scan.png", int(base["output"].get("dpi", 160)))
    success = sum(row["status"] == "ok" for row in rows)
    _atomic_json(output / "scan_manifest.json", {
        "source_config": path.name,
        "fingerprint": fingerprint,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(rows),
        "successful_cases": success,
        "failed_cases": len(rows) - success,
        "field_archives_enabled": save_fields_enabled,
        "grid": list(report.grid),
        "wave_convention": "exp(+i*omega*t - i*kz*z + i*m*phi)",
        "physical_validation": "not established; convergence and experimental benchmarks required",
    })
    print(f"\n[finished] {success}/{len(rows)} successful; results: {output}")
    print("[NOTE] Radial eigenmodes are local to the reference z slice; the whistler curve is an asymptote.")
    print("[NOTE] Only accepted driven-field fits are included in the numeric dispersion comparison.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frequency and azimuthal-mode helicon FDFD scan")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration")
    parser.add_argument("--resume", action="store_true", help="reuse matching completed cases")
    parser.add_argument("--modes", nargs="+", type=int, help="subset of configured azimuthal modes")
    parser.add_argument("--frequencies", nargs="+", type=float, metavar="MHZ", help="subset of frequencies, MHz")
    parser.add_argument("--dry-run", action="store_true", help="validate and list cases without solving")
    parser.add_argument("--no-fields", action="store_true", help="do not write per-case complex fields")
    parser.add_argument("--no-plots", action="store_true", help="omit figure generation")
    parser.add_argument("--fail-fast", action="store_true", help="stop after a failed case")
    arguments = parser.parse_args()
    summary = main(arguments.config, resume=arguments.resume, modes=arguments.modes,
                   frequencies_mhz=arguments.frequencies, dry_run=arguments.dry_run,
                   save_scan_fields=not arguments.no_fields, plots=not arguments.no_plots,
                   fail_fast=arguments.fail_fast)
    if any(row.get("status") != "ok" for row in summary):
        raise SystemExit(1)
