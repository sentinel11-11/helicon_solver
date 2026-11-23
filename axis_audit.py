"""Manufactured-field audit; no claim of a complete axis discretization proof."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from discretization import cylindrical_curl_pair


def audit_axis(nr=40, nz=16, radius_m=0.035, length_m=0.608):
    dr, dz = radius_m/nr, length_m/nz
    r = (np.arange(nr)+0.5)*dr
    rows = []
    for m in (0, 1, -1):
        cp, cm = cylindrical_curl_pair(r, dr, dz, np.ones(nr), np.ones(nz), m)
        field = np.zeros((3, nr, nz), complex)
        if m == 0:
            field[2] = 1.0  # constant Cartesian axial field
        else:
            field[0] = 1.0
            field[1] = 1j*m  # smooth constant transverse Cartesian field
        plus = (cp @ field.ravel()).reshape(field.shape)
        dual = (cm @ field.ravel()).reshape(field.shape)
        # Avoid the artificial closure at the two axial and outer radial ends.
        region = (slice(None), slice(0, 3), slice(2, nz-2))
        weights = sparse.diags(np.tile(np.repeat(r, nz), 3))
        defect = weights@cm-cp.conj().T@weights
        rows.append({
            "m": m,
            "analytic_curl": "zero",
            "forward_near_axis_max_m_inv": float(np.max(abs(plus[region]))),
            "dual_near_axis_max_m_inv": float(np.max(abs(dual[region]))),
            "dual_first_cell_error_times_dr": float(np.max(abs(dual[:, 0, 2:-2]))*dr),
            "weighted_adjoint_defect_max": float(np.max(abs(defect.data))) if defect.nnz else 0.0,
        })
    return {
        "grid": [nr, nz], "dr_m": dr,
        "tests": rows,
        "interpretation": (
            "The unstretched pair is weighted-adjoint. The dual curl does not reproduce the "
            "pointwise curl of constant smooth fields at the first radial cell under the current "
            "collocated interpretation. This legacy pointwise audit alone does not validate or "
            "invalidate the new coupled Galerkin closure. Run operator_benchmark.py for independent "
            "waveguide convergence tests; changing one boundary row would destroy the adjoint identity."
        ),
        "axis_discretization_validated": False,
    }
