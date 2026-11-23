"""Physical geometry and optional, explicitly artificial axial continuation."""
from __future__ import annotations

import numpy as np


def axial_domain(config):
    length = float(config["plasma"]["length"])
    spec = config.get("domain", {})
    before = float(spec.get("axial_buffer_before_m", 0.0))
    after = float(spec.get("axial_buffer_after_m", 0.0))
    if not np.all(np.isfinite([length, before, after])) or length <= 0 or min(before, after) < 0:
        raise ValueError("Physical length must be positive and axial buffers finite and nonnegative")
    medium = str(spec.get("axial_buffer_medium", "edge"))
    if medium != "edge":
        raise ValueError("domain.axial_buffer_medium currently supports only explicit edge continuation")
    return -before, length + after


def non_pml_mask(config, r, z, dr, dz, extra_r=0, extra_z=0):
    nr = int(config["grid"].get("pml_r", 0)) + int(extra_r)
    nz = int(config["grid"].get("pml_z", 0)) + int(extra_z)
    if min(nr, nz) < 0:
        raise ValueError("Mask margins must be nonnegative")
    start, end = axial_domain(config)
    radial_end = float(config["grid"]["radial_extent"]) - nr * dr
    return ((r[:, None] < radial_end) & (z[None, :] > start + nz * dz)
            & (z[None, :] < end - nz * dz))


def physical_mask(config, r, z, dr, dz, extra_r=0, extra_z=0):
    mask = non_pml_mask(config, r, z, dr, dz, extra_r, extra_z)
    return mask & (z[None, :] >= 0.0) & (z[None, :] <= float(config["plasma"]["length"]))
