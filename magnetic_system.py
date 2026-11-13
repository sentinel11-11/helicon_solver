from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.constants import mu_0
from scipy.special import ellipe, ellipk


_FIELD_CACHE: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}


def _cache_key(r: np.ndarray, z: np.ndarray, spec: dict[str, Any], target_b0: float) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=float).encode("utf-8")
    digest = hashlib.sha256(payload)
    digest.update(np.asarray(r, dtype=np.float64).tobytes())
    digest.update(np.asarray(z, dtype=np.float64).tobytes())
    digest.update(np.asarray([target_b0], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _section_geometry(section: dict[str, Any], default_start: float, default_end: float) -> tuple[float, float]:
    if "z_start_m" in section or "z_end_m" in section:
        start = float(section.get("z_start_m", default_start))
        end = float(section.get("z_end_m", default_end))
    else:
        length = float(section.get("length_m", default_end - default_start))
        center = float(section.get("z_center_m", 0.5 * (default_start + default_end)))
        start, end = center - 0.5 * length, center + 0.5 * length
    if not np.isfinite(start + end) or end <= start:
        raise ValueError("Each solenoid section must have a positive axial length")
    return start, end


def _loop_field(rho: np.ndarray, z: np.ndarray, radius: float, current: float) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    a2 = radius * radius
    alpha2 = (radius - rho) ** 2 + z * z
    beta2 = (radius + rho) ** 2 + z * z
    beta = np.sqrt(np.maximum(beta2, np.finfo(float).tiny))
    k2 = np.clip(4.0 * radius * rho / np.maximum(beta2, np.finfo(float).tiny), 0.0, 1.0 - 1e-13)
    K = ellipk(k2)
    E = ellipe(k2)
    coefficient = mu_0 * current / (2.0 * np.pi * beta)
    alpha2_safe = np.maximum(alpha2, np.finfo(float).tiny)
    factor = (a2 + rho * rho + z * z) / alpha2_safe
    bz = coefficient * (K + (a2 - rho * rho - z * z) / alpha2_safe * E)
    safe_rho = np.maximum(rho, 1e-12)
    br = coefficient * z / safe_rho * (-K + factor * E)
    axis = rho < 1e-10
    if np.any(axis):
        bz[axis] = mu_0 * current * a2 / (2.0 * (a2 + z[axis] * z[axis]) ** 1.5)
        br[axis] = 0.0
    return br, bz


def _solenoid_field(R: np.ndarray, Z: np.ndarray, spec: dict[str, Any], length: float) -> tuple[np.ndarray, np.ndarray]:
    solenoid = spec.get("solenoid", {})
    sections = solenoid.get("sections", [])
    if len(sections) != 2:
        raise ValueError("plmm_assembly.solenoid.sections must contain exactly two sections")
    radius = float(solenoid.get("radius_m", 0.04))
    if radius <= 0:
        raise ValueError("The solenoid radius must be positive")
    br = np.zeros_like(R, dtype=float)
    bz = np.zeros_like(R, dtype=float)
    default_edges = np.linspace(0.0, length, 3)
    for index, section in enumerate(sections):
        start, end = _section_geometry(section, float(default_edges[index]), float(default_edges[index + 1]))
        turns = float(section.get("turns", 1.0))
        current = float(section.get("current_A", 1.0)) * float(section.get("current_sign", 1.0))
        samples = int(section.get("axial_samples", max(8, min(128, round(abs(turns) / 4.0)))))
        if turns <= 0 or samples < 2:
            raise ValueError("Solenoid turns must be positive and axial_samples must be at least 2")
        positions = np.linspace(start, end, samples)
        weight = turns / samples
        for z_loop in positions:
            loop_br, loop_bz = _loop_field(R, Z - z_loop, radius, current * weight)
            br += loop_br
            bz += loop_bz
    return br, bz


def _magnet_centers(magnets: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "centers_m" in magnets:
        centers = np.asarray(magnets["centers_m"], dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("magnets.centers_m must be an N x 3 array")
        angles = np.arctan2(centers[:, 1], centers[:, 0])
        return centers, angles
    count = int(magnets.get("count", 8))
    radius = float(magnets.get("center_radius_m", 0.045))
    if count != 8:
        raise ValueError("The PLM-M magnetic assembly must contain exactly 8 magnets")
    if radius <= 0:
        raise ValueError("magnets.center_radius_m must be positive")
    angles = np.deg2rad(np.asarray(magnets.get("azimuths_deg", np.arange(count) * 360.0 / count), dtype=float))
    if angles.size != count:
        raise ValueError("magnets.azimuths_deg must contain one angle for each magnet")
    centers = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)))
    return centers, angles


def _magnet_sources(magnets: dict[str, Any], domain_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, angles = _magnet_centers(magnets)
    count = len(centers)
    length = float(magnets.get("length_m", 0.608))
    side = float(magnets.get("cross_section_m", 0.018))
    remanence = float(magnets.get("remanence_T", 1.2))
    segments = int(magnets.get("segments", 16))
    phi_samples = int(magnets.get("axisymmetric_average_samples", 8))
    if count != 8 or not np.isclose(length, 0.608, rtol=0.0, atol=1e-12):
        raise ValueError("The PLM-M magnet geometry requires 8 magnets of length 0.608 m")
    if side <= 0 or remanence < 0 or segments < 2 or phi_samples < 1:
        raise ValueError("Invalid permanent-magnet geometry or sampling")
    polarities = np.asarray(magnets.get("polarities", np.ones(count)), dtype=float)
    if polarities.size != count or np.any(~np.isfinite(polarities)):
        raise ValueError("magnets.polarities must contain one finite value for each magnet")
    if np.any(np.abs(polarities) == 0):
        raise ValueError("Each magnet polarity must be nonzero")
    direction = str(magnets.get("magnetization_direction", "radial")).lower()
    if direction not in ("radial", "tangential", "axial"):
        raise ValueError("magnetization_direction must be radial, tangential, or axial")

    moment_scale = (remanence / mu_0) * side * side * length / segments
    z_center = float(magnets.get("z_center_m", 0.5 * domain_length))
    source_z = z_center + np.linspace(-0.5 * length, 0.5 * length, segments)
    source_positions = []
    source_moments = []
    for center, angle, polarity in zip(centers, angles, polarities):
        if direction == "radial":
            unit = np.array((np.cos(angle), np.sin(angle), 0.0))
        elif direction == "tangential":
            unit = np.array((-np.sin(angle), np.cos(angle), 0.0))
        else:
            unit = np.array((0.0, 0.0, 1.0))
        moment = moment_scale * polarity * unit
        for z_source in source_z:
            source_positions.append((center[0], center[1], z_source))
            source_moments.append(moment)
    return np.asarray(source_positions, dtype=float), np.asarray(source_moments, dtype=float), angles


def _magnet_field_cartesian(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            magnets: dict[str, Any], domain_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_positions, source_moments, _ = _magnet_sources(magnets, domain_length)
    bx = np.zeros_like(x, dtype=float)
    by = np.zeros_like(y, dtype=float)
    bz = np.zeros_like(z, dtype=float)
    factor = mu_0 / (4.0 * np.pi)
    for position, moment in zip(source_positions, source_moments):
        dx = x - position[0]
        dy = y - position[1]
        dz = z - position[2]
        distance2 = dx * dx + dy * dy + dz * dz
        distance = np.sqrt(np.maximum(distance2, 1e-24))
        dot = moment[0] * dx + moment[1] * dy + moment[2] * dz
        inv_r3 = 1.0 / (distance2 * distance)
        inv_r5 = inv_r3 / distance2
        bx += factor * (3.0 * dot * dx * inv_r5 - moment[0] * inv_r3)
        by += factor * (3.0 * dot * dy * inv_r5 - moment[1] * inv_r3)
        bz += factor * (3.0 * dot * dz * inv_r5 - moment[2] * inv_r3)
    return bx, by, bz


def _magnet_field_axisymmetric(R: np.ndarray, Z: np.ndarray, magnets: dict[str, Any], domain_length: float) -> tuple[np.ndarray, np.ndarray]:
    br_avg = np.zeros_like(R, dtype=float)
    bz_avg = np.zeros_like(R, dtype=float)
    phi_samples = int(magnets.get("axisymmetric_average_samples", 8))
    if phi_samples < 1:
        raise ValueError("magnets.axisymmetric_average_samples must be positive")
    phi_values = np.arange(phi_samples, dtype=float) * 2.0 * np.pi / phi_samples
    for phi in phi_values:
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        bx, by, bz = _magnet_field_cartesian(x, y, Z, magnets, domain_length)
        br_avg += (bx * np.cos(phi) + by * np.sin(phi)) / phi_samples
        bz_avg += bz / phi_samples
    return br_avg, bz_avg


def build_magnetic_field(r: np.ndarray, z: np.ndarray, R: np.ndarray, Z: np.ndarray,
                         spec: dict[str, Any], target_b0: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    key = _cache_key(r, z, spec, target_b0)
    cached = _FIELD_CACHE.get(key)
    if cached is not None:
        return cached[0].copy(), cached[1].copy(), dict(cached[2])
    length = float(z[-1] - z[0] + (z[1] - z[0] if len(z) > 1 else 0.0))
    solenoid_br, solenoid_bz = _solenoid_field(R, Z, spec, length)
    magnet_br, magnet_bz = _magnet_field_axisymmetric(R, Z, spec.get("magnets", {}), length)
    normalization = spec.get("normalization", {})
    normalized = bool(normalization.get("enabled", True))
    scale = 1.0
    normalization_mode = str(normalization.get("mode", "total"))
    magnet_scale = float(spec.get("magnets", {}).get("field_scale", 1.0))
    if normalized:
        center = float(normalization.get("center_m", 0.5 * length))
        window = float(normalization.get("window_m", 0.25 * length))
        z_mask = (Z >= center - window) & (Z <= center + window)
        reference_field = solenoid_bz if normalization_mode == "solenoid_only" else solenoid_bz + magnet_bz
        values = reference_field[0, z_mask[0]]
        reference_value = float(np.mean(values)) if values.size else float(np.mean(reference_field))
        if abs(reference_value) < 1e-12:
            raise ValueError("Magnetic assembly normalization reference field is zero")
        scale = float(normalization.get("target_B0_T", target_b0)) / reference_value
    br = solenoid_br * scale + magnet_scale * magnet_br
    bz = solenoid_bz * scale + magnet_scale * magnet_bz
    uniform_bias = float(spec.get("uniform_bias_T", 0.0))
    bz += uniform_bias
    metadata = {
        "model": "plmm_assembly",
        "representation": "axisymmetric_average",
        "magnet_count": 8,
        "magnet_length_m": 0.608,
        "magnet_cross_section_m": 0.018,
        "solenoid_sections": 2,
        "normalization_enabled": normalized,
        "normalization_mode": normalization_mode,
        "normalization_scale": scale,
        "magnet_field_scale": magnet_scale,
        "uniform_bias_T": uniform_bias,
        "axisymmetric_average_samples": int(spec.get("magnets", {}).get("axisymmetric_average_samples", 8)),
    }
    _FIELD_CACHE[key] = (br.copy(), bz.copy(), dict(metadata))
    return br, bz, metadata


def build_magnetic_field_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             spec: dict[str, Any], target_b0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float))
    domain_length = float(np.max(z) - np.min(z))
    if domain_length <= 0:
        raise ValueError("The 3-D magnetic-field grid must span a positive axial length")
    rho = np.hypot(x, y)
    solenoid_br, solenoid_bz = _solenoid_field(rho, z, spec, domain_length)
    solenoid_bx = np.divide(solenoid_br * x, rho, out=np.zeros_like(rho), where=rho > 0)
    solenoid_by = np.divide(solenoid_br * y, rho, out=np.zeros_like(rho), where=rho > 0)
    magnet_bx, magnet_by, magnet_bz = _magnet_field_cartesian(
        x, y, z, spec.get("magnets", {}), domain_length
    )
    normalization = spec.get("normalization", {})
    normalized = bool(normalization.get("enabled", True))
    scale = 1.0
    normalization_mode = str(normalization.get("mode", "total"))
    magnet_scale = float(spec.get("magnets", {}).get("field_scale", 1.0))
    if normalized:
        center = float(normalization.get("center_m", 0.5 * domain_length))
        window = float(normalization.get("window_m", 0.25 * domain_length))
        z_ref = np.linspace(center - window, center + window, 101)
        axis = np.zeros_like(z_ref)
        ref_sol_br, ref_sol_bz = _solenoid_field(axis, z_ref, spec, domain_length)
        if normalization_mode == "solenoid_only":
            reference_field = ref_sol_bz
        else:
            _, _, ref_mag_bz = _magnet_field_cartesian(axis, axis, z_ref, spec.get("magnets", {}), domain_length)
            reference_field = ref_sol_bz + ref_mag_bz
        reference_value = float(np.mean(reference_field))
        if abs(reference_value) < 1e-12:
            raise ValueError("Magnetic assembly normalization reference field is zero")
        scale = float(normalization.get("target_B0_T", target_b0)) / reference_value
    uniform_bias = float(spec.get("uniform_bias_T", 0.0))
    bx = scale * solenoid_bx + magnet_scale * magnet_bx
    by = scale * solenoid_by + magnet_scale * magnet_by
    bz = scale * solenoid_bz + magnet_scale * magnet_bz + uniform_bias
    metadata = {
        "model": "plmm_assembly",
        "representation": "full_3d",
        "magnet_count": 8,
        "magnet_length_m": 0.608,
        "magnet_cross_section_m": 0.018,
        "solenoid_sections": 2,
        "normalization_enabled": normalized,
        "normalization_mode": normalization_mode,
        "normalization_scale": scale,
        "magnet_field_scale": magnet_scale,
        "uniform_bias_T": uniform_bias,
    }
    return bx, by, bz, metadata


def cartesian_divergence(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    bz = np.asarray(bz, dtype=float)
    if bx.ndim != 3 or by.shape != bx.shape or bz.shape != bx.shape:
        raise ValueError("Cartesian field arrays must have matching three-dimensional shapes")
    if bx.shape != (len(x), len(y), len(z)):
        raise ValueError("Cartesian field shape does not match x, y, z coordinates")
    dx = np.gradient(bx, x, axis=0, edge_order=2)
    dy = np.gradient(by, y, axis=1, edge_order=2)
    dz = np.gradient(bz, z, axis=2, edge_order=2)
    return dx + dy + dz
