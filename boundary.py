import numpy as np
from scipy.constants import epsilon_0


class PML:
    def __init__(self, Nr, Nz, pml_r, pml_z, sigma_max, dr, dz):
        if not (0 <= pml_r < Nr // 2 and 0 <= pml_z < Nz // 2):
            raise ValueError("Invalid PML thickness")
        if sigma_max < 0:
            raise ValueError("PML strength must be nonnegative")
        self.Nr, self.Nz = int(Nr), int(Nz)
        self.dr, self.dz = float(dr), float(dz)
        self.sigma_max = float(sigma_max)
        self.pml_r = int(pml_r)
        self.sr = self._stretch(self.Nr, int(pml_r), lower=False)
        self.sz = self._stretch(self.Nz, int(pml_z), lower=True)

    def _stretch(self, n, thickness, lower):
        sigma = np.zeros(n, dtype=float)
        if thickness:
            depth = ((np.arange(thickness, 0, -1) - 0.5) / thickness) ** 2
            if lower:
                sigma[:thickness] = self.sigma_max * depth
            sigma[-thickness:] = self.sigma_max * depth[::-1]
        return 1.0 - 1j * sigma

    def scale_r(self):
        return self.sr

    def scale_z(self):
        return self.sz

    def passive_electric_loss(self, omega, conductivity):
        """Positive isotropic conductivity ramp; an absorber, not a PML."""
        if conductivity is None or not np.isfinite(float(conductivity)) or float(conductivity) <= 0:
            raise ValueError("Passive absorber requires positive boundary.absorber_conductivity_S_m")
        if self.sigma_max == 0:
            return np.zeros((self.Nr, self.Nz))
        profile = (-self.sr.imag[:, None]-self.sz.imag[None, :])/self.sigma_max
        return float(conductivity)*profile/(float(omega)*epsilon_0)

    def tensor_factors(self, r):
        """Cylindrical transformation-optics factors, for exp(+i*omega*t).

        The local stretching is (s_r, r_tilde/r, s_z), not (s_r, 1, s_z).
        epsilon'_ij = det(S)*epsilon_ij/(s_i*s_j), with no conjugation.
        For physical mu_r=1, inverse(mu')_i = s_i**2/det(S).
        """
        r = np.asarray(r, dtype=float)
        if r.shape != (self.Nr,) or np.any(r <= 0):
            raise ValueError("Tensor PML requires positive cell-center radii")
        s_phi = np.ones(self.Nr, dtype=complex)
        if self.pml_r:
            width = self.pml_r*self.dr
            start = (self.Nr-self.pml_r)*self.dr
            depth = np.maximum(r-start, 0.0)
            s_phi -= 1j*self.sigma_max*depth**3/(3*width**2*r)
        shape = (self.Nr, self.Nz)
        stretches = (np.broadcast_to(self.sr[:, None], shape),
                     np.broadcast_to(s_phi[:, None], shape),
                     np.broadcast_to(self.sz[None, :], shape))
        det = stretches[0]*stretches[1]*stretches[2]
        epsilon_factors = tuple(tuple(det/(a*b) for b in stretches) for a in stretches)
        mu_inverse = tuple(a*a/det for a in stretches)
        return epsilon_factors, mu_inverse
