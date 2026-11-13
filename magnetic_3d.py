from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from magnetic_system import build_magnetic_field_3d, cartesian_divergence
from main import load_config


def main(config_path: str = "config.yaml", output_path: str | None = None) -> Path:
    config = load_config(config_path)
    grid = config["grid"]
    geometry = config.get("geometry", {})
    domain = config["plasma"]
    nx = int(grid.get("Nxy_3d", 48))
    nz = int(grid.get("Nz_3d", min(int(grid.get("Nz", 128)), 256)))
    if nx < 8 or nz < 8:
        raise ValueError("The 3-D magnetic grid must contain at least 8 points per axis")
    extent = float(grid["radial_extent"])
    length = float(domain["length"])
    cartesian_extent = extent / np.sqrt(2.0)
    x = np.linspace(-cartesian_extent, cartesian_extent, nx)
    y = np.linspace(-cartesian_extent, cartesian_extent, nx)
    z = np.linspace(0.0, length, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    bx, by, bz, metadata = build_magnetic_field_3d(
        X, Y, Z, config["plasma"]["magnetic_field"], float(domain["B0"])
    )
    divergence = cartesian_divergence(x, y, z, bx, by, bz)
    if output_path is None:
        output_dir = Path(config["output"]["results_dir"])
        if not output_dir.is_absolute():
            output_dir = Path(config_path).resolve().parent / output_dir
        output_path = str(output_dir / "magnetic_field_3d.npz")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        x=x,
        y=y,
        z=z,
        Bx=bx,
        By=by,
        Bz=bz,
        divB=divergence,
    )
    print(f"[3d magnetic] saved: {target}")
    print(f"[3d magnetic] max|div B|={np.max(np.abs(divergence)):.3e} T/m")
    plasma_mask = np.hypot(X, Y) <= float(domain["radius"])
    print(f"[3d magnetic] plasma max|div B|={np.max(np.abs(divergence[plasma_mask])):.3e} T/m")
    print(f"[3d magnetic] representation={metadata['representation']}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 3-D PLM-M magnetic-field calculation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    main(args.config, args.output)
