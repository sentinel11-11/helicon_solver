from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.constants import elementary_charge, electron_mass, epsilon_0, c

E_CHARGE = elementary_charge
M_E = electron_mass
M_I = 1.67262192369e-27
EPS0 = epsilon_0
C_LIGHT = c


@dataclass(frozen=True)
class PlasmaParams:
    B0: float
    n_e: float
    T_e: float
    nu_en: float
    ion_mass: float = M_I
    ion_charge: int = 1
    nu_in: float = 0.0


class StixTensor:
    def __init__(self, params: PlasmaParams, omega: float):
        if omega <= 0 or not np.isfinite(omega):
            raise ValueError("Angular frequency must be positive")
        if params.n_e < 0 or params.ion_mass <= 0 or params.ion_charge <= 0:
            raise ValueError("Invalid plasma species parameters")
        if params.B0 < 0 or params.nu_en < 0 or params.nu_in < 0:
            raise ValueError("Magnetic magnitude and collision rates must be nonnegative")
        self.p = params
        self.omega = float(omega)
        self._compute()

    def _compute(self):
        p, w = self.p, self.omega
        Z = int(p.ion_charge)
        self.omega_pe = np.sqrt(p.n_e * E_CHARGE**2 / (EPS0 * M_E))
        self.omega_pi = np.sqrt(p.n_e * Z * E_CHARGE**2 / (EPS0 * p.ion_mass))
        self.omega_ce = E_CHARGE * p.B0 / M_E
        self.omega_ci = Z * E_CHARGE * p.B0 / p.ion_mass
        wp2 = (self.omega_pe**2, self.omega_pi**2)
        cycl = (-self.omega_ce, self.omega_ci)
        nu = (p.nu_en, p.nu_in)
        R = 1.0 + 0j
        L = 1.0 + 0j
        P = 1.0 + 0j
        for strength, signed_cycl, collision in zip(wp2, cycl, nu):
            if strength == 0:
                continue
            wr = w - 1j * collision
            right = w * (wr + signed_cycl)
            left = w * (wr - signed_cycl)
            axial = w * wr
            if min(abs(right), abs(left), abs(axial)) <= np.finfo(float).tiny:
                raise ValueError("Cyclotron resonance requires nonzero collisional damping")
            R -= strength / right
            L -= strength / left
            P -= strength / axial
        self.R, self.L, self.P = R, L, P
        self.S, self.D = 0.5 * (R + L), 0.5 * (R - L)

    def get_tensor(self, direction=(0, 0, 1)):
        b = np.asarray(direction, dtype=float)
        mag = np.linalg.norm(b)
        if mag == 0:
            b = np.array((0.0, 0.0, 1.0))
        else:
            b /= mag
        br, bp, bz = b
        cross = np.array([[0, -bz, bp], [bz, 0, -br], [-bp, br, 0]])
        return self.S * np.eye(3, dtype=complex) + (self.P - self.S) * np.outer(b, b) - 1j * self.D * cross


def density_profile(r, r0, n_peak, kind="parabolic"):
    r = np.asarray(r, dtype=float)
    if r0 <= 0 or n_peak < 0:
        raise ValueError("Invalid plasma radius or peak density")
    if kind == "parabolic":
        shape = np.maximum(0.0, 1.0 - (r / r0) ** 2)
    elif kind == "flat":
        shape = np.ones_like(r)
    elif kind == "gaussian":
        shape = np.exp(-(r / (0.5 * r0)) ** 2)
    else:
        raise ValueError(f"Unknown density profile: {kind}")
    return np.where(r < r0, n_peak * shape, 0.0)


def theoretical_kz(omega, n_e, B0, m_mode=1, k_perp=0.0):
    del m_mode
    if omega <= 0 or n_e <= 0 or B0 <= 0 or k_perp < 0:
        raise ValueError("Invalid helicon asymptote parameters")
    wpe2 = n_e * E_CHARGE**2 / (EPS0 * M_E)
    wce = E_CHARGE * B0 / M_E
    factor = omega * wpe2 / (C_LIGHT**2 * wce)
    if k_perp == 0:
        return complex(np.sqrt(factor), 0.0)
    kz2 = 0.5 * (-k_perp**2 + np.sqrt(k_perp**4 + 4 * factor**2))
    return complex(np.sqrt(max(kz2, 0.0)), 0.0)


def parallel_kz_stix(omega, n_e, B0, ion_mass=M_I, ion_charge=1,
                     nu_en=0.0, nu_in=0.0, branch="R"):
    """Return the complex parallel cold-plasma wavenumber from the full Stix tensor.

    The result includes the ion contribution and collisions. ``branch`` selects
    the parallel R or L circularly polarized branch. The sign is chosen for a
    forward wave with nonnegative real part and nonpositive attenuation under
    exp(+i*omega*t - i*kz*z).
    """
    branch = str(branch).upper()
    if branch not in ("R", "L"):
        raise ValueError("branch must be R or L")
    params = PlasmaParams(float(B0), float(n_e), 0.0, float(nu_en),
                          float(ion_mass), int(ion_charge), float(nu_in))
    tensor = StixTensor(params, float(omega))
    epsilon = tensor.R if branch == "R" else tensor.L
    if not np.isfinite(epsilon.real) or not np.isfinite(epsilon.imag):
        raise ValueError("Parallel Stix branch is non-finite")
    root = (float(omega) / C_LIGHT) * np.sqrt(epsilon + 0j)
    candidates = (root, -root)
    valid = [value for value in candidates if value.real >= 0 and value.imag <= 0]
    if valid:
        return complex(min(valid, key=lambda value: abs(value.imag)))
    return complex(max(candidates, key=lambda value: value.real))


def configured_theoretical_kz(omega, n_e, B0, plasma_config=None, solver_config=None, m_mode=1):
    plasma_config = plasma_config or {}
    solver_config = solver_config or {}
    model = str(solver_config.get("theory_model", "whistler")).lower()
    if model in ("full_stix", "stix", "parallel_stix"):
        return parallel_kz_stix(
            omega,
            n_e,
            B0,
            ion_mass=float(plasma_config.get("ion_mass", M_I)),
            ion_charge=int(plasma_config.get("ion_charge", 1)),
            nu_en=float(plasma_config.get("nu_en", 0.0)),
            nu_in=float(plasma_config.get("nu_in", 0.0)),
            branch=str(solver_config.get("parallel_branch", "R")),
        )
    return theoretical_kz(omega, n_e, B0, m_mode=m_mode)
