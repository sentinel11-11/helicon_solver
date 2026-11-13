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


def _rotation_matrix(angles_deg: np.ndarray) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad(angles_deg)
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    return rz @ ry @ rx


def _evaluate(design: np.ndarray, base_probe: dict[str, Any], cases: list[FieldSnapshot],
              positions: list[tuple[float, float]], settings: dict[str, Any], robust: dict[str, Any] | None = None) -> dict[str, float]:
    turns = np.asarray(design[:3], dtype=float)
    areas = np.asarray(design[3:], dtype=float)
    orientation = np.asarray(design[6:9], dtype=float) if len(design) >= 9 else np.zeros(3)
    rotation = _rotation_matrix(orientation)
    probe = ThreeAxisProbe(_probe_config(base_probe, turns, areas))
    robust = robust or {}
    field_scales = [float(value) for value in robust.get("field_scales", [1.0])]
    noise_scales = [float(value) for value in robust.get("noise_scales", [1.0])]
    if any(value <= 0 for value in field_scales + noise_scales):
        raise ValueError("Robust field and noise scales must be positive")
    noise_v = float(settings.get("noise_voltage_V", 1e-6))
    rings = int(settings.get("rings", 3))
    angles = int(settings.get("angles", 16))
    min_singular = []
    condition = []
    cross_talk = []
    spatial_error = []
    scenario_scores = []
    for case in cases:
        matrix = probe.transfer_matrix(case.frequency_hz) @ rotation
        singular = np.linalg.svd(matrix, compute_uv=False)
        min_singular.append(float(np.min(singular)))
        condition.append(float(np.linalg.cond(matrix)))
        diagonal = np.diag(np.diag(matrix))
        cross_talk.append(float(np.linalg.norm(matrix - diagonal) / max(np.linalg.norm(diagonal), 1e-30)))
        for r0, z0 in positions:
            averaged = rotation @ probe.area_averaged_field(case, r0, z0, rings=rings, angles=angles)
            center = rotation @ case.center_field(r0, z0)
            spatial_error.append(float(np.linalg.norm(averaged - center) / max(np.linalg.norm(center), 1e-30)))
            for field_scale in field_scales:
                for noise_scale in noise_scales:
                    voltage = matrix @ (field_scale * averaged)
                    scenario_scores.append(float(np.linalg.norm(voltage) / max(noise_v * noise_scale, 1e-30)))
    min_s = max(min(min_singular), 1e-30)
    cond = max(np.median(condition), 1.0)
    xtalk = float(np.median(cross_talk))
    spatial = float(np.median(spatial_error))
    snr = max(float(np.min(scenario_scores)), 1e-30)
    score = np.log10(snr) + np.log10(min_s) - 0.5 * np.log10(cond) - xtalk - spatial
    return {"score": float(score), "min_singular": float(min_s), "condition_median": cond,
            "cross_talk_median": xtalk, "spatial_error_median": spatial, "snr_worst_case": snr,
            "orientation_yaw_deg": float(orientation[0]), "orientation_pitch_deg": float(orientation[1]),
            "orientation_roll_deg": float(orientation[2])}


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
    robust = config.get("robust", {})
    optimize_orientation = bool(config.get("optimize_orientation", True))
    if optimize_orientation:
        orientation_bounds = tuple(map(float, config.get("orientation_bounds_deg", [-45.0, 45.0])))
        bounds.extend((orientation_bounds, orientation_bounds, orientation_bounds))
    result = differential_evolution(
        lambda x: -_evaluate(x, probe_config, cases, positions, settings, robust)["score"],
        bounds=bounds,
        maxiter=int(config.get("optimizer", {}).get("maxiter", 12)),
        popsize=int(config.get("optimizer", {}).get("popsize", 6)),
        seed=int(config.get("optimizer", {}).get("seed", 1234)),
        polish=bool(config.get("optimizer", {}).get("polish", True)),
        workers=1,
    )
    metrics = _evaluate(result.x, probe_config, cases, positions, settings, robust)
    output_dir = Path(config.get("output_dir", "probe_optimization_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best = dict(probe_config)
    best["turns"] = result.x[:3].tolist()
    best["area_m2"] = result.x[3:].tolist()
    if optimize_orientation:
        best["orientation_yaw_pitch_roll_deg"] = result.x[6:9].tolist()
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
