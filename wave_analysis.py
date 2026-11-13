from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class AxialWaveResult:
    kz: float
    alpha: float
    quality: float
    purity: float
    cycles: float
    spread: float
    reflection: float
    component: str
    radial_index: int
    z: np.ndarray
    phase: np.ndarray
    phase_slope: float
    reason: str


def analyze_axial_wave(solver):
    if not getattr(solver, "_solved", False):
        raise RuntimeError("The field must be solved before wave analysis")

    cfg = solver.cfg
    grid = cfg["grid"]
    settings = cfg.get("solver", {})
    source_end = float(cfg["antenna"]["z_position"]) + float(cfg["antenna"]["length"])
    margin = max(4 * solver.dz, min(0.02, 0.025 * solver.L))
    z_end = solver.L - int(grid["pml_z"]) * solver.dz - margin
    mask_z = (solver.z >= source_end + margin) & (solver.z <= z_end)
    z = solver.z[mask_z]
    if z.size < 12:
        raise ValueError("The downstream region is too short for wave analysis")

    low = float(settings.get("kz_r_min_fraction", 0.45)) * solver.r0
    high = float(settings.get("kz_r_max_fraction", 0.90)) * solver.r0
    if not 0 <= low < high <= solver.r0:
        raise ValueError("Invalid radial wave-analysis interval")
    available = np.flatnonzero((solver.r >= low) & (solver.r <= high))
    if available.size == 0:
        raise ValueError("No radial samples in the wave-analysis interval")
    r_indices = np.unique(available[np.linspace(0, available.size - 1, min(10, available.size)).astype(int)])

    magnetic = solver.compute_magnetic_field()
    names = ("Br", "Bphi", "Bz")
    entries = []
    all_peak = max(float(np.max(np.abs(b[:, mask_z]))) for b in magnetic)
    for name, field in zip(names, magnetic):
        comp_peak = float(np.max(np.abs(field[:, mask_z])))
        for ir in r_indices:
            trace = np.asarray(field[ir, mask_z], dtype=np.complex128)
            peak = float(np.max(np.abs(trace)))
            if not np.all(np.isfinite(trace)):
                continue
            if peak < max(0.15 * comp_peak, 0.025 * all_peak, 1e-30):
                continue
            rms = float(np.sqrt(np.mean(np.abs(trace) ** 2)))
            if rms > 0:
                entries.append((name, int(ir), trace, rms))
    if not entries:
        raise RuntimeError("No usable magnetic-field traces in the downstream region")

    traces = np.asarray([item[2] / item[3] for item in entries])
    x = z - 0.5 * (z[0] + z[-1])
    baseline = float(z[-1] - z[0])
    energy = float(np.sum(np.abs(traces) ** 2))
    kmax = float(settings.get("kz_fit_max_rad_m", 80.0))
    alphamax = float(settings.get("kz_fit_max_alpha_m_inv", 15.0))
    if not 0 < kmax < np.pi / solver.dz:
        raise ValueError("kz_fit_max_rad_m must be below the spatial Nyquist limit")
    if alphamax <= 0:
        raise ValueError("kz_fit_max_alpha_m_inv must be positive")

    def error(k, alpha):
        basis = np.exp((-alpha - 1j * k) * x)
        norm = float(np.vdot(basis, basis).real)
        projection = traces @ basis.conj()
        captured = float(np.vdot(projection, projection).real) / (norm * energy)
        return float(np.clip(1.0 - captured, 0.0, 1.0))

    k_grid = np.linspace(0.1, kmax, 160)
    starts = []
    for alpha in (0.0, min(1.0 / baseline, alphamax), min(3.0 / baseline, alphamax)):
        values = np.asarray([error(k, alpha) for k in k_grid])
        for idx in np.argsort(values)[:3]:
            starts.append((float(values[idx]), float(k_grid[idx]), alpha))
    starts.sort()

    best = None
    for _, k0, a0 in starts[:5]:
        opt = minimize(
            lambda u: error(float(u[0]), float(u[1])),
            (k0, a0), method="L-BFGS-B",
            bounds=((0.1, kmax), (0.0, alphamax)),
            options={"maxiter": 90, "ftol": 1e-10},
        )
        if best is None or float(opt.fun) < best[0]:
            best = (float(opt.fun), float(opt.x[0]), float(opt.x[1]))
    residual, k, alpha = best
    purity = 1.0 - residual

    estimates = []
    for entry in entries:
        _, _, signal, _ = entry
        amplitude = np.abs(signal)
        good = amplitude >= 0.20 * float(np.max(amplitude))
        if good.sum() < max(24, int(0.50 * z.size)):
            continue
        phase = np.unwrap(np.angle(signal))
        zz, pp = z[good], phase[good]
        weights = (amplitude[good] / np.max(amplitude)) ** 2
        mean_z = float(np.average(zz, weights=weights))
        mean_p = float(np.average(pp, weights=weights))
        denom = float(np.sum(weights * (zz - mean_z) ** 2))
        if denom < 1e-20:
            continue
        slope = float(np.sum(weights * (zz - mean_z) * (pp - mean_p)) / denom)
        predicted = mean_p + slope * (zz - mean_z)
        ss_tot = float(np.sum(weights * (pp - mean_p) ** 2))
        r2 = 1.0 - float(np.sum(weights * (pp - predicted) ** 2)) / max(ss_tot, 1e-30)
        if -slope > 0 and r2 >= 0.65:
            estimates.append((-slope, float(r2), entry, phase))

    if estimates:
        estimates.sort(key=lambda item: (item[1], -abs(item[0] - k)), reverse=True)
        ref_k, ref_r2, ref, ref_phase = estimates[0]
        k_values = np.asarray([item[0] for item in estimates])
        spread = float(1.4826 * np.median(np.abs(k_values - np.median(k_values))))
    else:
        ref = max(entries, key=lambda item: item[3])
        ref_phase = np.unwrap(np.angle(ref[2]))
        ref_r2, spread = 0.0, float("inf")

    bf = np.exp((-alpha - 1j * k) * x)
    bb = np.exp((alpha + 1j * k) * x)
    basis = np.column_stack((bf, bb))
    coefs, _, _, _ = np.linalg.lstsq(basis, traces.T, rcond=None)
    recon = basis @ coefs
    reflected_error = float(np.sum(np.abs(traces.T - recon) ** 2) / energy)
    forward_amp = float(np.linalg.norm(coefs[0]))
    reflected_amp = float(np.linalg.norm(coefs[1]))
    reflection = reflected_amp / max(forward_amp, 1e-30)
    reflection_evidence = (residual - reflected_error > 0.08 and reflection > 0.35)

    cycles = k * baseline / (2 * np.pi)
    issues = []
    if z.size < 24:
        issues.append("too few axial samples")
    if cycles < float(settings.get("kz_fit_min_cycles", 1.0)):
        issues.append("insufficient phase cycles")
    if purity < float(settings.get("kz_fit_min_purity", 0.85)):
        issues.append("single-mode fit is poor")
    if len(estimates) < 3:
        issues.append("too few coherent traces")
    if spread > 0.25 * k:
        issues.append("radial/component phase slopes disagree")
    if estimates and abs(np.median([item[0] for item in estimates]) - k) > 0.25 * k:
        issues.append("phase slopes and complex fit disagree")
    if reflection_evidence:
        issues.append("possible reflection or counter-propagating mode")

    reason = "; ".join(issues)
    return AxialWaveResult(
        kz=float("nan") if issues else k,
        alpha=float("nan") if issues else alpha,
        quality=0.0 if issues else ref_r2,
        purity=purity,
        cycles=cycles,
        spread=spread,
        reflection=reflection,
        component=ref[0],
        radial_index=ref[1],
        z=z.copy(),
        phase=ref_phase,
        phase_slope=-k,
        reason=reason,
    )
