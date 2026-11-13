import numpy as np


class PML:
    def __init__(self, Nr, Nz, pml_r, pml_z, sigma_max, dr, dz):
        if not (0 <= pml_r < Nr // 2 and 0 <= pml_z < Nz // 2):
            raise ValueError("Invalid PML thickness")
        if sigma_max < 0:
            raise ValueError("PML strength must be nonnegative")
        self.Nr, self.Nz = int(Nr), int(Nz)
        self.dr, self.dz = float(dr), float(dz)
        self.sigma_max = float(sigma_max)
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
