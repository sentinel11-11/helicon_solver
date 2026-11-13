from __future__ import annotations

import numpy as np
from scipy import sparse


def forward_difference(n: int, h: float) -> sparse.csr_matrix:
    if n < 3 or h <= 0:
        raise ValueError("Invalid difference grid")
    rows = np.repeat(np.arange(n - 1), 2)
    cols = np.column_stack((np.arange(n - 1), np.arange(1, n))).ravel()
    vals = np.tile(np.array([-1.0 / h, 1.0 / h]), n - 1)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def derivative_pair(n: int, h: float, stretch: np.ndarray):
    stretch = np.asarray(stretch, dtype=np.complex128)
    if stretch.shape != (n,) or not np.all(np.isfinite(stretch)) or np.any(stretch == 0):
        raise ValueError("Invalid complex coordinate stretch")
    base = forward_difference(n, h)
    scale = sparse.diags(1.0 / stretch, format="csr")
    return (scale @ base).tocsr(), (-base.T @ scale).tocsr()


def cylindrical_curl(dr, dz, r: np.ndarray, m: int) -> sparse.csr_matrix:
    nr = len(r)
    n = dr.shape[0]
    if n % nr or dz.shape != (n, n):
        raise ValueError("Inconsistent cylindrical grid operators")
    nz = n // nr
    radius = np.repeat(np.asarray(r, dtype=float), nz)
    R = sparse.diags(radius, format="csr")
    Rinv = sparse.diags(1.0 / radius, format="csr")
    M = (1j * int(m)) * Rinv
    zero = sparse.csr_matrix((n, n), dtype=complex)
    return sparse.bmat(
        [[zero, -dz, M], [dz, zero, -dr], [-M, Rinv @ dr @ R, zero]],
        format="csr",
    )


def cylindrical_curl_pair(r, dr, dz, sr, sz, m):
    nr = len(r)
    nz = len(sz)
    drp0, drm0 = derivative_pair(nr, dr, sr)
    dzp0, dzm0 = derivative_pair(nz, dz, sz)
    ir = sparse.eye(nr, format="csr")
    iz = sparse.eye(nz, format="csr")
    cp = cylindrical_curl(
        sparse.kron(drp0, iz, format="csr"),
        sparse.kron(ir, dzp0, format="csr"), r, m,
    )
    cm = cylindrical_curl(
        sparse.kron(drm0, iz, format="csr"),
        sparse.kron(ir, dzm0, format="csr"), r, m,
    )
    return cp, cm


def radial_curl_polynomial(r, dr, stretch, m):
    nr = len(r)
    drp, drm = derivative_pair(nr, dr, stretch)
    zero = sparse.csr_matrix((nr, nr), dtype=complex)
    eye = sparse.eye(nr, format="csr", dtype=complex)
    rdiag = sparse.diags(r, format="csr")
    rinv = sparse.diags(1.0 / r, format="csr")
    az = (1j * int(m)) * rinv
    cp0 = sparse.bmat(
        [[zero, zero, az], [zero, zero, -drp], [-az, rinv @ drp @ rdiag, zero]], format="csr"
    )
    cm0 = sparse.bmat(
        [[zero, zero, az], [zero, zero, -drm], [-az, rinv @ drm @ rdiag, zero]], format="csr"
    )
    c1 = sparse.bmat(
        [[zero, 1j * eye, zero], [-1j * eye, zero, zero], [zero, zero, zero]], format="csr"
    )
    return cp0, cm0, c1


def radial_grid_oscillation(fields, pml_cells=0):
    arrays = [np.asarray(a, dtype=np.complex128) for a in fields]
    end = len(arrays[0]) - int(pml_cells)
    if end < 6:
        raise ValueError("Insufficient physical radial cells")
    numerator = 0.0
    denominator = 0.0
    for arr in arrays:
        y = arr[:end]
        delta2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
        numerator += float(np.vdot(delta2, delta2).real)
        denominator += float(np.vdot(y[1:-1], y[1:-1]).real)
    return float(np.sqrt(numerator / max(denominator, 1e-300)) / 4.0)
