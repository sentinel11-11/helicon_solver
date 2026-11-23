"""Independent vacuum TM-waveguide test of the production Maxwell operator.

PEC circular guide: kz**2 = (omega/c)**2 - (j_m1/R)**2.  No radial plasma
dispersion solver or whistler reference is used. A localized impressed Jz
excites the guide, and both axial directions are fitted downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import c
from scipy.optimize import minimize_scalar
from scipy.special import jn_zeros, jv

from solver import HeliconSolver


def vacuum_config(nr, nz, m=1, formulation="tensor"):
    return {
        "plasma": {"radius": .025, "length": .5, "B0": .03, "n_e_peak": 0.,
                   "T_e": 0., "nu_en": 0., "nu_in": 0., "ion_mass": 6.64e-27,
                   "ion_charge": 1, "profile": "flat"},
        "wave": {"frequency": 6e9, "m_mode": m},
        "grid": {"Nr": nr, "Nz": nz, "radial_extent": .035, "pml_r": 0,
                 "pml_z": round(.08/(.5/nz)), "pml_sigma_max": 6.},
        "boundary": {"pec_pml_termination": True, "pml_formulation": formulation},
        "solver": {"method": "direct", "tol": 1e-9, "artificial_loss": 0., "use_gpu": False},
        "antenna": {"z_position": .19, "length": .02, "radius_m": .02,
                    "radial_width_m": .001, "turns": 1., "current_density": 1.},
        "output": {},
    }


def analyze_guide_field(solver, m):
    k0 = solver.omega/c
    kr = float(jn_zeros(abs(m), 1)[0]/solver.r_extent)
    reference = float(np.sqrt(k0*k0-kr*kr))
    rz = (solver.r >= .3*solver.r_extent) & (solver.r <= .8*solver.r_extent)
    zz = (solver.z >= .30) & (solver.z <= .40)
    z = solver.z[zz]
    x = z-z.mean()
    traces = solver.Ez[rz][:, zz].T
    scale = max(float(np.linalg.norm(traces)**2), 1e-300)

    def fit(k):
        basis = np.column_stack((np.exp(-1j*k*x), np.exp(1j*k*x)))
        coeff = np.linalg.lstsq(basis, traces, rcond=None)[0]
        residual = float(np.linalg.norm(basis@coeff-traces)**2/scale)
        return residual, coeff

    solution = minimize_scalar(lambda k: fit(k)[0], bounds=(.7*reference, 1.3*reference),
                               method="bounded", options={"xatol": 1e-8})
    error, coeff = fit(solution.x)
    ratio = float(np.linalg.norm(coeff[1])/max(np.linalg.norm(coeff[0]), 1e-300))
    plane = int(np.argmin(abs(solver.z-.35)))
    radial_reference = jv(abs(m), kr*solver.r)
    actual = solver.Ez[:, plane]
    weight = solver.r
    amplitude = np.vdot(radial_reference*weight, actual)/np.vdot(radial_reference*weight, radial_reference)
    radial_error = float(np.sqrt(np.sum(weight*abs(actual-amplitude*radial_reference)**2)
                                 /max(np.sum(weight*abs(actual)**2), 1e-300)))
    axis_helicity_ratio = None
    if abs(m) == 1:
        er, ep = solver.Er[0, plane], solver.Ephi[0, plane]
        axis_helicity_ratio = float(abs(er+1j*m*ep)/max(np.hypot(abs(er), abs(ep)), 1e-300))
    return {
        "analytic_kz_rad_m": reference, "fitted_kz_rad_m": float(solution.x),
        "relative_kz_error": float(abs(solution.x-reference)/reference),
        "backward_to_forward_amplitude": ratio,
        "two_direction_relative_fit_error": float(np.sqrt(error)),
        "positive_axial_power_downstream": bool(solver.physical_poynting_power(.35) > 0),
        "reference_wall_radius_m": solver.r_extent,
        "discrete_pec_row_radius_m": float(solver.r[-1]),
        "axis_discretization_validated": False,
        "radial_Ez_shape_relative_error": radial_error,
        "first_cell_transverse_helicity_ratio": axis_helicity_ratio,
    }


def run_case(nr, nz, m=1, formulation="tensor", axis_regularization="legacy"):
    config = vacuum_config(nr, nz, m, formulation)
    config["solver"]["axis_regularization"] = axis_regularization
    solver = HeliconSolver(config)
    kr = jn_zeros(abs(m), 1)[0]/solver.r_extent
    radial = jv(abs(m), kr*solver.r)
    axial = np.exp(-0.5*((solver.z-.20)/.004)**2)
    current = np.zeros((3, nr, nz), complex)
    current[2] = radial[:, None]*axial[None, :]
    solver._build_source = lambda: current.ravel().copy()
    solver.solve()
    report = analyze_guide_field(solver, m)
    report.update(grid=[nr, nz], m=m, frequency_Hz=6e9, pml_formulation=formulation,
                  linear_residual=solver.relative_residual, power_budget=solver.energy_budget(),
                  axis_regularization=axis_regularization)
    return report


def main(output, grids=((24, 240), (48, 480), (72, 720)), modes=(0, 1), formulations=("tensor",),
         axis_regularization="coupled_m1"):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for formulation in formulations:
        for m in modes:
            for nr, nz in grids:
                print(f"[vacuum benchmark] {formulation}, m={m}, {nr}x{nz}", flush=True)
                report = run_case(nr, nz, m, formulation, axis_regularization)
                reports.append(report)
                path.write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf8")
                print(f"[vacuum benchmark] kz error={report['relative_kz_error']:.3e}, "
                      f"back/forward={report['backward_to_forward_amplitude']:.3e}", flush=True)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results_operator_validation/vacuum_guide.json")
    parser.add_argument("--quick", action="store_true", help="24x240 and 48x480 only")
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--legacy-axis", action="store_true", help="disable coupled m=+/-1 axis closure")
    args = parser.parse_args()
    main(args.output, grids=((24, 240), (48, 480)) if args.quick else ((24, 240), (48, 480), (72, 720)),
         formulations=("legacy_stretched", "tensor") if args.compare_legacy else ("tensor",),
         axis_regularization="legacy" if args.legacy_axis else "coupled_m1")
