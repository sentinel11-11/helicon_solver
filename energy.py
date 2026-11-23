"""Power accounting for exp(+i*omega*t), with explicit control volumes.

Surface fluxes interpolate collocated S to cell faces. They are a continuum
diagnostic, not an exact finite-volume identity for the curl-pair stencil.
The separate full-grid operator-work identity tests algebraic consistency.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import epsilon_0, mu_0


def _surface_box(solver, mask, radial_flux, axial_flux, q, source, artificial=None):
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size < 2 or cols.size < 2:
        raise ValueError("Power control volume must contain at least 2 x 2 cells")
    i, j0, j1 = int(rows[-1]), int(cols[0]), int(cols[-1])
    if rows[0] != 0 or not np.all(mask[:i+1, j0:j1+1]) or np.sum(mask) != (i+1)*(j1-j0+1):
        raise ValueError("Power control volume must be a cylindrical rectangular box including the axis")

    def axial_face(j, neighbor):
        values = axial_flux[:i+1, j]
        if 0 <= neighbor < solver.Nz:
            values = 0.5 * (values + axial_flux[:i+1, neighbor])
        return float(2*np.pi*solver.dr*np.sum(solver.r[:i+1]*values))

    left_plus_z = axial_face(j0, j0-1)
    right_plus_z = axial_face(j1, j1+1)
    radial_face = radial_flux[i, j0:j1+1]
    if i+1 < solver.Nr:
        radial_face = 0.5*(radial_face + radial_flux[i+1, j0:j1+1])
    radius = float(solver.r[i] + 0.5*solver.dr)
    radial_out = float(2*np.pi*radius*solver.dz*np.sum(radial_face))
    volume = 2*np.pi*solver.r[:, None]*solver.dr*solver.dz
    absorbed = float(np.sum(np.where(mask, q, 0)*volume))
    supplied = float(np.sum(np.where(mask, source, 0)*volume))
    artificial_power = 0.0 if artificial is None else float(np.sum(np.where(mask, artificial, 0)*volume))
    outward = right_plus_z - left_plus_z + radial_out
    defect = supplied - absorbed - artificial_power - outward
    scale = max(abs(supplied), abs(absorbed), abs(artificial_power), abs(left_plus_z), abs(right_plus_z), abs(radial_out), 1e-300)
    return {
        "r_outer_m": radius,
        "z_left_m": float(solver.z[j0]-0.5*solver.dz),
        "z_right_m": float(solver.z[j1]+0.5*solver.dz),
        "source_power_W": supplied,
        "absorbed_power_W": absorbed,
        "artificial_absorbed_power_W": artificial_power,
        "left_face_plus_z_W": left_plus_z,
        "right_face_plus_z_W": right_plus_z,
        "radial_outward_W": radial_out,
        "net_outward_W": outward,
        "balance_defect_W": defect,
        "relative_balance_defect": float(abs(defect)/scale),
        "net_boundary_inflow": bool(outward < -0.01*scale),
        "source_small_relative_to_power_terms": bool(abs(supplied) < 1e-4*scale),
    }


def energy_budget(solver):
    if not getattr(solver, "_solved", False):
        raise RuntimeError("Solve the field before computing the power budget")
    fields = np.stack((solver.Er, solver.Ephi, solver.Ez))
    if not hasattr(solver, "source_current"):
        raise RuntimeError("Source current is required for the power budget")
    current = np.asarray(solver.source_current).reshape(3, solver.Nr, solver.Nz)
    source = -0.5*np.real(np.sum(fields*np.conj(current), axis=0))
    q = solver.compute_absorbed_power() + solver.compute_material_absorption()
    artificial = 0.5*solver.omega*epsilon_0*float(solver.cfg["solver"].get("artificial_loss", 0.0))*np.sum(abs(fields)**2, axis=0)
    radial, axial = solver.compute_poynting()
    physical = _surface_box(solver, solver.physical_domain_mask(), radial, axial, q, source, artificial)
    non_pml = _surface_box(solver, solver.non_pml_domain_mask(), radial, axial, q, source, artificial)

    # Original, unmodified curl-curl operator, before PEC row replacement.
    x = fields.reshape(-1)
    work = np.vdot(x, np.tile(np.repeat(2*np.pi*solver.r*solver.dr*solver.dz, solver.Nz), 3)
                   * (solver.C_dual @ (solver.C @ x)))
    operator_exchange = float(np.imag(work)/(2*solver.omega*mu_0))
    volume = 2*np.pi*solver.r[:, None]*solver.dr*solver.dz
    total_absorbed = float(np.sum((q+artificial)*volume))
    total_source = float(np.sum(source*volume))
    global_defect = total_source-total_absorbed-operator_exchange
    global_scale = max(abs(total_source), abs(total_absorbed), abs(operator_exchange), 1e-300)
    warnings = []
    if non_pml["net_boundary_inflow"]:
        warnings.append("Net power enters the non-PML box through its boundary; audit termination/PML and flux reconstruction")
    if physical["relative_balance_defect"] > 0.1:
        warnings.append("Interpolated physical-domain surface balance defect exceeds 10%; this is not an exact discrete conservation test")
    if operator_exchange < -0.01*global_scale:
        warnings.append("Full-grid curl/stretch work is negative; inspect the combined operator balance and non-PML boundary flux, not this term alone")
    report = {
        "phase_convention": "exp(+i*omega*t + i*m*phi)",
        "azimuthal_normalization": "2*pi for one complex Fourier harmonic with peak phasor amplitude",
        "surface_method": "collocated Poynting flux, arithmetic interpolation to cell faces",
        "physical": physical,
        "non_pml": non_pml,
        "full_grid_operator_work": {
            "source_power_W": total_source,
            "constitutive_quadratic_including_pml_W": float(np.sum(q*volume)),
            "artificial_absorbed_W": float(np.sum(artificial*volume)),
            "curl_stretch_exchange_W": operator_exchange,
            "balance_defect_W": global_defect,
            "relative_balance_defect": float(abs(global_defect)/global_scale),
            "includes_pml": True,
            "is_physical_absorption": False,
        },
        "negative_plasma_absorption_fraction": solver.negative_absorption_fraction,
        "warnings": warnings,
        "physical_model_validated": False,
    }
    solver.power_budget = report
    return report
