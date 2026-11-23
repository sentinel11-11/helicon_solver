"""Balanced companion solve; residuals are evaluated in the original polynomial.

The diagonal transformations are invertible and the wave-number scale comes
only from coefficient norms. No theoretical dispersion value enters the solve.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import eig


def solve_quadratic(a0, a1, a2, *, scaling="equilibrated"):
    coefficients = [a.toarray() if sparse.issparse(a) else np.asarray(a, dtype=complex).copy()
                    for a in (a0, a1, a2)]
    n = coefficients[0].shape[0]
    if any(a.shape != (n, n) or not np.all(np.isfinite(a)) for a in coefficients) or n == 0:
        raise ValueError("QEP coefficients must be finite nonempty square matrices of equal shape")
    if scaling not in ("none", "equilibrated"):
        raise ValueError("radial_pencil_scaling must be none or equilibrated")
    gamma = 1.0
    column_scale = np.ones(n)
    if scaling == "equilibrated":
        norms = [float(np.linalg.norm(a, "fro")) for a in coefficients]
        if norms[0] > 0 and norms[2] > 0:
            gamma = float(np.sqrt(norms[0]/norms[2]))
        coefficients = [coefficients[0], gamma*coefficients[1], gamma**2*coefficients[2]]
        row_max = np.maximum.reduce([np.max(abs(a), axis=1) for a in coefficients])
        row_scale = np.divide(1., row_max, out=np.ones(n), where=row_max > 0)
        coefficients = [row_scale[:, None]*a for a in coefficients]
        col_max = np.maximum.reduce([np.max(abs(a), axis=0) for a in coefficients])
        column_scale = np.divide(1., col_max, out=np.ones(n), where=col_max > 0)
        coefficients = [a*column_scale[None, :] for a in coefficients]
    zero = np.zeros((n, n), complex)
    identity = np.eye(n, dtype=complex)
    left = np.block([[zero, identity], [-coefficients[0], -coefficients[1]]])
    right = np.block([[identity, zero], [zero, coefficients[2]]])
    values, vectors = eig(left, right, right=True, check_finite=False)
    finite = np.isfinite(values)
    values[finite] *= gamma
    # Only the field block is needed. Recover its ORIGINAL coordinates.
    fields = column_scale[:, None]*vectors[:n]
    return values, fields, {
        "scaling": scaling, "k_scale_rad_m": gamma, "polynomial_dimension": n,
        "finite_eigenvalues": int(np.count_nonzero(finite)),
        "column_scale_min": float(column_scale.min()),
        "column_scale_max": float(column_scale.max()),
    }


def polynomial_residual(a0, a1, a2, value, field):
    terms = (a0@field, value*(a1@field), value**2*(a2@field))
    return float(np.linalg.norm(sum(terms))/max(sum(np.linalg.norm(t) for t in terms), 1e-300))
