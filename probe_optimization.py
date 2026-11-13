from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution

from probe import ThreeAxisProbe


@dataclass
class FieldSnapshot:
    frequency_hz: float
    mode: int
    r: np.ndarray
    z: np.ndarray
    br: np.ndarray
    bphi: np.ndarray
    bz: np.ndarray

    def __post_init__(self) -> None:
        self._interpolators = [RegularGridInterpolator((self.r, self.z), field, bounds_error=True)
                               for field in (self.br, self.bphi, self.bz)]
        self.cfg = {"wave": {"m_mode": self.mode}}

    def compute_magnetic_field(self):
        return self.br, self.bphi, self.bz

    def center_field(self, r0: float, z0: float) -> np.ndarray:
        point = np.array([[r0, z0]], dtype=float)
        return np.array([float(interp(point)[0]) for interp in self._interpolators], dtype=complex)


def _load_cases(root: Path, frequencies: list[float] | None, modes: list[int] | None) -> list[FieldSnapshot]:
    cases = []
    for fields_path in sorted(root.rglob("fields.npz")):
        case_config_path = fields_path.parent / "case_config.yaml"
        if not case_config_path.exists():
            continue
        config = yaml.safe_load(case_config_path.read_text(encoding="utf-8-sig"))
        frequency = float(config["wave"]["frequency"])
        mode = int(config["wave"].get("m_mode", 0))
        if frequencies and not any(abs(frequency - f) <= 1.0 for f in frequencies):
            continue
        if modes and mode not in modes:
            continue
        with np.load(fields_path, allow_pickle=False) as data:
            cases.append(FieldSnapshot(frequency, mode, data["r"].copy(), data["z"].copy(),
                                       data["B_r"].copy(), data["B_phi"].copy(), data["B_z"].copy()))
    if not cases:
        raise FileNotFoundError("No matching fields.npz files were found")
    return cases


def _probe_config(base: dict[str, Any], turns: np.ndarray, areas: np.ndarray) -> dict[str, Any]:
    config = dict(base)
    config["turns"] = turns.tolist()
    config["area_m2"] = areas.tolist()
    return config


def _evaluate(design: np.ndarray, base_probe: dict[str, Any], cases: list[FieldSnapshot], positions: list[tuple[float, float]], settings: dict[str, Any]) -> dict[str, float]:
    turns = np.asarray(design[:3], dtype=float)
    areas = np.asarray(design[3:], dtype=float)
    probe = ThreeAxisProbe(_probe_config(base_probe, turns, areas))
    noise_v = float(settings.get("noise_voltage_V", 1e-6))
    rings = int(settings.get("rings", 3))
    angles = int(settings.get("angles", 16))
    min_singular = []
    condition = []
    cross_talk = []
    spatial_error = []
    signal_to_noise = []
    for case in cases:
        matrix = probe.transfer_matrix(case.frequency_hz)
        singular = np.linalg.svd(matrix, compute_uv=False)
        min_singular.append(float(np.min(singular)))
        condition.append(float(np.linalg.cond(matrix)))
        diagonal = np.diag(np.diag(matrix))
        cross_talk.append(float(np.linalg.norm(matrix - diagonal) / max(np.linalg.norm(diagonal), 1e-30)))
        for r0, z0 in positions:
            averaged = probe.area_averaged_field(case, r0, z0, rings=rings, angles=angles)
            center = case.center_field(r0, z0)
            spatial_error.append(float(np.linalg.norm(averaged - center) / max(np.linalg.norm(center), 1e-30)))
            voltage = matrix @ averaged
            signal_to_noise.append(float(np.linalg.norm(voltage) / max(noise_v, 1e-30)))
    min_s = max(min(min_singular), 1e-30)
    cond = max(np.median(condition), 1.0)
    xtalk = float(np.median(cross_talk))
    spatial = float(np.median(spatial_error))
    snr = max(float(np.median(signal_to_noise)), 1e-30)
    score = np.log10(snr) + np.log10(min_s) - 0.5 * np.log10(cond) - xtalk - spatial
    return {"score": float(score), "min_singular": float(min_s), "condition_median": cond,
            "cross_talk_median": xtalk, "spatial_error_median": spatial, "snr_median": snr}


def optimize(config_path: str = "probe_optimization.yaml") -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    cases = _load_cases(Path(config["fields_root"]), config.get("frequencies_Hz"), config.get("modes"))
    positions = [(float(item["r_m"]), float(item["z_m"])) for item in config["positions"]]
    probe_config = config["probe"]
    bounds_cfg = config["bounds"]
    bounds = [tuple(map(float, bounds_cfg["turns"])), tuple(map(float, bounds_cfg["turns"])),
              tuple(map(float, bounds_cfg["turns"])), tuple(map(float, bounds_cfg["area_m2"])),
              tuple(map(float, bounds_cfg["area_m2"])), tuple(map(float, bounds_cfg["area_m2"]))]
    settings = config.get("evaluation", {})
    result = differential_evolution(
        lambda x: -_evaluate(x, probe_config, cases, positions, settings)["score"],
        bounds=bounds,
        maxiter=int(config.get("optimizer", {}).get("maxiter", 12)),
        popsize=int(config.get("optimizer", {}).get("popsize", 6)),
        seed=int(config.get("optimizer", {}).get("seed", 1234)),
        polish=bool(config.get("optimizer", {}).get("polish", True)),
        workers=1,
    )
    metrics = _evaluate(result.x, probe_config, cases, positions, settings)
    output_dir = Path(config.get("output_dir", "probe_optimization_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best = dict(probe_config)
    best["turns"] = result.x[:3].tolist()
    best["area_m2"] = result.x[3:].tolist()
    (output_dir / "best_probe.yaml").write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    report = {"cases": len(cases), "positions": len(positions), "success": bool(result.success),
              "message": str(result.message), "iterations": int(result.nit), "design": best, "metrics": metrics}
    (output_dir / "optimization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative optimization of a three-axis RF magnetic probe")
    parser.add_argument("--config", default="probe_optimization.yaml")
    args = parser.parse_args()
    optimize(args.config)
