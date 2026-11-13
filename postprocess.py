from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wave_analysis import analyze_axial_wave


class StoredField:
    def __init__(self, config: dict[str, Any], archive: Path):
        self.cfg = config
        self._solved = True
        with np.load(archive, allow_pickle=False) as data:
            self.r = data["r"].copy()
            self.z = data["z"].copy()
            self._magnetic = tuple(data[name].copy() for name in ("B_r", "B_phi", "B_z"))
        self.Nr, self.Nz = len(self.r), len(self.z)
        self.r0 = float(config["plasma"]["radius"])
        self.L = float(config["plasma"]["length"])
        self.dr = float(self.r[1] - self.r[0])
        self.dz = float(self.z[1] - self.z[0])

    def compute_magnetic_field(self):
        return self._magnetic


def _finite(value: Any):
    number = float(value)
    return number if np.isfinite(number) else None


def analyze_saved_case(case_dir: str | Path, settings: dict[str, float] | None = None) -> dict[str, Any]:
    root = Path(case_dir).resolve()
    config_path = root / "case_config.yaml"
    field_path = root / "fields.npz"
    if not config_path.is_file() or not field_path.is_file():
        raise FileNotFoundError(f"Missing case_config.yaml or fields.npz in {root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if settings:
        config.setdefault("solver", {}).update(settings)
    model = StoredField(config, field_path)
    result = analyze_axial_wave(model)
    accepted = not bool(result.reason)
    report = {
        "case": root.name,
        "frequency_Hz": float(config["wave"]["frequency"]),
        "m_mode": int(config["wave"]["m_mode"]),
        "accepted": accepted,
        "kz_rad_m": _finite(result.kz),
        "kz_cm_inv": _finite(result.kz / 100),
        "attenuation_m_inv": _finite(result.alpha),
        "purity": _finite(result.purity),
        "phase_fit_R2": _finite(result.quality),
        "phase_cycles": _finite(result.cycles),
        "candidate_spread_rad_m": _finite(result.spread),
        "reflection_ratio_indicator": _finite(result.reflection),
        "reference_component": result.component,
        "reference_radial_coordinate_m": float(model.r[result.radial_index]),
        "rejection_reasons": result.reason,
        "method": "multi_component_single_exponential_with_reflection_diagnostic",
    }
    (root / "wavefit_postprocessed.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report


def main(directory: str, *, all_cases: bool = False, settings: dict[str, float] | None = None) -> list[dict[str, Any]]:
    root = Path(directory).resolve()
    cases = sorted(x.parent for x in root.glob("*/fields.npz")) if all_cases else [root]
    if not cases:
        raise ValueError(f"No archived field cases found in {root}")
    reports = []
    for case in cases:
        try:
            report = analyze_saved_case(case, settings=settings)
            reports.append(report)
            label = "accepted" if report["accepted"] else "rejected"
            print(f"[{label}] {case.name}: kz={report['kz_cm_inv']} cm^-1; {report['rejection_reasons']}")
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            print(f"[error] {case.name}: {type(exc).__name__}: {exc}")
    if all_cases and reports:
        destination = root / "wavefit_postprocessed.csv"
        with destination.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=reports[0].keys())
            writer.writeheader()
            writer.writerows(reports)
        print(f"[saved] {destination}")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-analyze saved complex magnetic fields without another FDFD solve")
    parser.add_argument("directory", help="case directory or scan results directory with --all")
    parser.add_argument("--all", action="store_true", help="process every case containing fields.npz")
    parser.add_argument("--min-cycles", type=float, help="minimum accepted axial phase cycles")
    parser.add_argument("--min-purity", type=float, help="minimum accepted single-mode energy fraction")
    args = parser.parse_args()
    options = {}
    if args.min_cycles is not None:
        options["kz_fit_min_cycles"] = args.min_cycles
    if args.min_purity is not None:
        options["kz_fit_min_purity"] = args.min_purity
    results = main(args.directory, all_cases=args.all, settings=options)
    if not results:
        raise SystemExit(1)
