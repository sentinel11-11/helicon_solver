from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares

from magnetic_system import build_magnetic_field_3d


def _path_parts(path: str) -> list[str | int]:
    parts: list[str | int] = []
    for item in path.split("."):
        parts.append(int(item) if item.isdigit() else item)
    return parts


def _get_path(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in _path_parts(path):
        value = value[part]
    return value


def _set_path(payload: dict[str, Any], path: str, value: float) -> None:
    parts = _path_parts(path)
    target: Any = payload
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = float(value)


def _load_measurements(path: Path) -> dict[str, np.ndarray]:
    data = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=",", encoding="utf-8-sig"))
    names = set(data.dtype.names or ())
    if "z_m" not in names or "B_z_T" not in names:
        raise ValueError("Measurements require z_m and B_z_T columns")
    radius = data["r_m"] if "r_m" in names else np.zeros(len(data), dtype=float)
    radial = data["B_r_T"] if "B_r_T" in names else None
    result = {"r_m": np.asarray(radius, dtype=float), "z_m": np.asarray(data["z_m"], dtype=float),
              "B_z_T": np.asarray(data["B_z_T"], dtype=float)}
    if radial is not None:
        result["B_r_T"] = np.asarray(radial, dtype=float)
    if "sigma_B_T" in names:
        result["sigma_B_T"] = np.asarray(data["sigma_B_T"], dtype=float)
    else:
        result["sigma_B_T"] = np.full(len(data), 1e-4, dtype=float)
    if np.any(result["sigma_B_T"] <= 0) or not all(np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError("Magnetic measurements must be finite and sigma_B_T must be positive")
    return result


def _predict(config: dict[str, Any], measurements: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    field_spec = config["plasma"]["magnetic_field"]
    x = measurements["r_m"]
    y = np.zeros_like(x)
    z = measurements["z_m"]
    bx, _, bz, _ = build_magnetic_field_3d(x, y, z, field_spec, float(config["plasma"]["B0"]))
    return bz, bx if "B_r_T" in measurements else None


def calibrate(config_path: str, measurements_path: str, output_config: str, report_path: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    measurements = _load_measurements(Path(measurements_path))
    parameters = config.get("field_calibration", {}).get("parameters", [])
    if not parameters:
        raise ValueError("field_calibration.parameters is empty")
    initial = np.asarray([float(item.get("initial", _get_path(config, item["path"]))) for item in parameters])
    lower = np.asarray([float(item["lower"]) for item in parameters])
    upper = np.asarray([float(item["upper"]) for item in parameters])
    if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Invalid field-calibration parameter bounds")

    def residual(values: np.ndarray) -> np.ndarray:
        trial = copy.deepcopy(config)
        for item, value in zip(parameters, values):
            _set_path(trial, item["path"], float(value))
        predicted_z, predicted_r = _predict(trial, measurements)
        residuals = [(predicted_z - measurements["B_z_T"]) / measurements["sigma_B_T"]]
        if predicted_r is not None:
            residuals.append((predicted_r - measurements["B_r_T"]) / measurements["sigma_B_T"])
        return np.concatenate(residuals)

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=int(config.get("field_calibration", {}).get("max_nfev", 200)),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    calibrated = copy.deepcopy(config)
    fitted = {}
    for item, value in zip(parameters, result.x):
        _set_path(calibrated, item["path"], float(value))
        fitted[item["path"]] = float(value)
    output = Path(output_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(calibrated, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "rmse_normalized": float(np.sqrt(np.mean(result.fun**2))),
        "parameters": fitted,
        "output_config": str(output),
    }
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit PLM-M magnetic-model parameters to measured field data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--output-config", default="config_calibrated.yaml")
    parser.add_argument("--report", default="field_calibration_report.json")
    args = parser.parse_args()
    calibrate(args.config, args.measurements, args.output_config, args.report)
