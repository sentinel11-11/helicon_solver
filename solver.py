from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import (
    bicgstab as scipy_bicgstab,
    gmres as scipy_gmres,
    spsolve,
    splu,
    spilu,
    LinearOperator,
)
from scipy.constants import mu_0, c, epsilon_0, electron_mass, elementary_charge
try:
    from threadpoolctl import threadpool_limits
except ImportError:  
    threadpool_limits = None

from plasma import StixTensor, PlasmaParams, density_profile, configured_theoretical_kz
from boundary import PML
from background import Background
from double_layer import DoubleLayerDiagnostics
from discretization import cylindrical_curl_pair, radial_curl_polynomial, radial_grid_oscillation, regular_axis_projection
from domain import axial_domain, non_pml_mask, physical_mask
from quadratic_modes import solve_quadratic, polynomial_residual


def _try_cupy():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return None


def _direct_refined(A, b, tol, max_steps=6):
    """Equilibrated LU with corrections from the ORIGINAL system residual.

    Physics and acceptance thresholds are unchanged. Extended arithmetic is
    used only to accumulate the correction residual; the caller still checks
    the original double-precision residual used by previous releases.
    """
    row_max = np.asarray(abs(A).max(axis=1).toarray()).ravel()
    row_scale = np.divide(1.0, row_max, out=np.ones_like(row_max), where=row_max > 0)
    Ar = sparse.diags(row_scale) @ A
    col_max = np.asarray(abs(Ar).max(axis=0).toarray()).ravel()
    col_scale = np.divide(1.0, col_max, out=np.ones_like(col_max), where=col_max > 0)
    equilibrated = (Ar @ sparse.diags(col_scale)).tocsc()
    del Ar
    lu = splu(equilibrated)
    del equilibrated
    x = col_scale * lu.solve(row_scale * b)
    # Some platforms alias longdouble to double; record its actual precision.
    A_ext = A.astype(np.clongdouble)
    b_ext = b.astype(np.clongdouble)
    denom = max(float(np.linalg.norm(b)), np.finfo(float).tiny)
    best_x = x.copy()
    best_residual = float('inf')
    stale = 0
    for step in range(int(max_steps) + 1):
        measured = float(np.linalg.norm(A @ x - b) / denom)
        correction_rhs = b_ext - A_ext @ x.astype(np.clongdouble)
        extended = float(np.linalg.norm(correction_rhs) / denom)
        print(f'[direct-refine] step={step}, original_residual={measured:.3e}, '
              f'extended_residual={extended:.3e}, '
              f'residual_mantissa_bits={np.finfo(np.longdouble).nmant}', flush=True)
        if np.isfinite(measured) and measured < best_residual:
            best_residual = measured
            best_x = x.copy()
            stale = 0
        else:
            stale += 1
        if measured <= tol or step == max_steps or stale >= 2:
            break
        correction = col_scale * lu.solve(row_scale * np.asarray(correction_rhs, dtype=np.complex128))
        if not np.all(np.isfinite(correction)):
            break
        x = x + correction
    return best_x


def _first_derivative(n, h):
    """Second-order central derivative internally, one-sided at end cells."""
    if n < 3:
        raise ValueError("Each grid dimension must contain at least 3 cells")
    mat = sparse.lil_matrix((n, n), dtype=float)
    mat[0, 0], mat[0, 1] = -1.0 / h, 1.0 / h
    for i in range(1, n - 1):
        mat[i, i - 1], mat[i, i + 1] = -0.5 / h, 0.5 / h
    mat[-1, -2], mat[-1, -1] = -1.0 / h, 1.0 / h
    return mat.tocsr()


def _axis_derivative(n, h, parity):
    matrix = _first_derivative(n, h).tolil()
    matrix[0, :] = 0.0
    matrix[0, 0] = -parity / (2.0 * h)
    matrix[0, 1] = 1.0 / (2.0 * h)
    return matrix.tocsr()


def _stix_arrays(density, field, omega, plasma, nu_e=None, nu_i=None):
    w = float(omega)
    wp2 = density * elementary_charge**2 / (epsilon_0 * electron_mass)
    z = int(plasma["ion_charge"])
    wi2 = density * z * elementary_charge**2 / (epsilon_0 * float(plasma["ion_mass"]))
    cycl = ((-elementary_charge / electron_mass) * field,
            (z * elementary_charge / float(plasma["ion_mass"])) * field)
    S = np.ones_like(density, dtype=complex)
    D = np.zeros_like(density, dtype=complex)
    P = np.ones_like(density, dtype=complex)
    R = np.ones_like(density, dtype=complex)
    L = np.ones_like(density, dtype=complex)
    if nu_e is None:
        nu_e = float(plasma.get("nu_en", 0.0))
    if nu_i is None:
        nu_i = float(plasma.get("nu_in", 0.0))
    for strength, omega_c, nu in ((wp2, cycl[0], nu_e), (wi2, cycl[1], nu_i)):
        wstar = w - 1j * nu
        den_r = w * (wstar + omega_c)
        den_l = w * (wstar - omega_c)
        if np.any((np.abs(den_r) < 1e-25) & (strength > 0)) or np.any((np.abs(den_l) < 1e-25) & (strength > 0)):
            raise ValueError("Unregularized cyclotron resonance in background plasma")
        R -= strength / den_r
        L -= strength / den_l
        P -= strength / (w * wstar)
    S = 0.5 * (R + L)
    D = 0.5 * (R - L)
    return S, D, P


def _dielectric_components(S, D, P, Br, Bz):
    mag = np.hypot(Br, Bz)
    br = np.divide(Br, mag, out=np.zeros_like(mag), where=mag > 0)
    bz = np.divide(Bz, mag, out=np.ones_like(mag), where=mag > 0)
    delta = P - S
    return ((S + delta * br**2, 1j * D * bz, delta * br * bz),
            (-1j * D * bz, S, 1j * D * br),
            (delta * br * bz, -1j * D * br, S + delta * bz**2))


class HeliconSolver:
    def __init__(self, config, *, build_operator=True):
        self.cfg = config
        self.cp = _try_cupy() if config.get("solver", {}).get("use_gpu", False) else None
        self.use_gpu = self.cp is not None
        if self.use_gpu:
            props = self.cp.cuda.runtime.getDeviceProperties(0)
            print(f"[GPU] {props['name'].decode(errors='replace')}")
        else:
            print("[CPU] using SciPy sparse solver")
        self._setup_grid()
        self._setup_plasma()
        self._setup_boundary()
        self._operator_ready = False
        if build_operator:
            self._setup_operator()

    def _setup_grid(self):
        g = self.cfg["grid"]
        p = self.cfg["plasma"]
        self.Nr, self.Nz = int(g["Nr"]), int(g["Nz"])
        self.r0, self.L = float(p["radius"]), float(p["length"])
        self.r_extent = float(g.get("radial_extent", 1.35 * self.r0))
        if self.Nr < 8 or self.Nz < 16:
            raise ValueError("Grid too small: use Nr >= 8 and Nz >= 16")
        if not np.isfinite(self.r_extent) or self.r_extent <= self.r0:
            raise ValueError("grid.radial_extent must be larger than plasma.radius")
        pml_r_cells = int(g["pml_r"])
        pml_start = self.r_extent * (1.0 - pml_r_cells / self.Nr)
        if pml_start <= self.r0:
            raise ValueError(
                "The radial PML would overlap the plasma. Increase grid.radial_extent, "
                "reduce grid.pml_r, or refine grid.Nr."
            )
        self.dr = self.r_extent / self.Nr
        self.z_min, self.z_max = axial_domain(self.cfg)
        self.dz = (self.z_max - self.z_min) / self.Nz
        self.r = (np.arange(self.Nr) + 0.5) * self.dr
        self.z = self.z_min + (np.arange(self.Nz) + 0.5) * self.dz
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")

    def _setup_plasma(self):
        p = self.cfg["plasma"]
        self.params = PlasmaParams(
            B0=float(p["B0"]), n_e=float(p["n_e_peak"]),
            T_e=float(p.get("T_e", 0.0)), nu_en=float(p.get("nu_en", 0.0)),
            ion_mass=float(p.get("ion_mass", 1.67262192369e-27)),
            ion_charge=int(p.get("ion_charge", 1)), nu_in=float(p.get("nu_in", 0.0)),
        )
        self.omega = 2.0 * np.pi * float(self.cfg["wave"]["frequency"])
        self.k0 = self.omega / c
        self.stix = StixTensor(self.params, self.omega)
        self.background = Background(self.cfg, self.r, self.z)
        div_report = self.background.magnetic_divergence
        print(
            f"[magnetic] div(B) relative max={div_report['relative_max']:.3e}, "
            f"rms={div_report['relative_rms']:.3e}"
        )
        div_check = self.cfg["plasma"].get("magnetic_field", {}).get("divergence_check", {})
        if div_check.get("enabled", True):
            threshold = float(div_check.get("relative_warn", 0.05))
            if div_report["relative_max"] > threshold:
                print(
                    f"[magnetic warning] div(B) exceeds relative threshold {threshold:.3e}; "
                    "refine the cylindrical grid before interpreting field gradients"
                )
        self.n_e_r = np.mean(self.background.n_e, axis=1)
        self.S2, self.D2, self.P2 = _stix_arrays(
            self.background.n_e, np.hypot(self.background.B_r, self.background.B_z),
            self.omega, p, self.background.nu_e, self.background.nu_i,
        )
        self.S_r = np.mean(self.S2, axis=1)
        self.D_r = np.mean(self.D2, axis=1)
        self.P_r = np.mean(self.P2, axis=1)
        self.double_layer = DoubleLayerDiagnostics.from_config(self.cfg, self.z, p)
        self.equilibrium_coupled_to_wave = False
        if self.double_layer is not None:
            print("[equilibrium] prescribed potential diagnostic; no kinetic CFDL feedback to RF response")

    def _setup_boundary(self):
        g = self.cfg["grid"]
        self.pml = PML(
            self.Nr, self.Nz, int(g["pml_r"]), int(g["pml_z"]),
            float(g["pml_sigma_max"]), self.dr, self.dz,
        )
        self.sr = self.pml.scale_r()
        self.sz = self.pml.scale_z()

    def _setup_operator(self):
        Nr, Nz = self.Nr, self.Nz
        g = self.cfg["grid"]
        m = int(self.cfg["wave"].get("m_mode", 0))

        self.pml_formulation = str(self.cfg.get("boundary", {}).get("pml_formulation", "legacy_stretched"))
        if self.pml_formulation not in ("legacy_stretched", "tensor", "passive_absorber"):
            raise ValueError("Unknown boundary.pml_formulation")
        tensor_pml = self.pml_formulation == "tensor"
        real_curl = self.pml_formulation != "legacy_stretched"
        self.C, self.C_dual = cylindrical_curl_pair(
            self.r, self.dr, self.dz,
            np.ones(self.Nr) if real_curl else self.sr,
            np.ones(self.Nz) if real_curl else self.sz, m,
        )
        if tensor_pml:
            epsilon_factors, mu_inverse = self.pml.tensor_factors(self.r)
        else:
            epsilon_factors = tuple(tuple(np.ones((Nr, Nz)) for _ in range(3)) for _ in range(3))
            mu_inverse = tuple(np.ones((Nr, Nz), complex) for _ in range(3))
        self.mu_inverse = mu_inverse
        self.Mu_inverse = sparse.diags(np.concatenate([a.ravel() for a in mu_inverse]), format="csr")
        self.absorber_epsilon_loss = np.zeros((Nr, Nz))
        if self.pml_formulation == "passive_absorber":
            self.absorber_epsilon_loss = self.pml.passive_electric_loss(
                self.omega, self.cfg.get("boundary", {}).get("absorber_conductivity_S_m"))
        physical = _dielectric_components(
            self.S2, self.D2, self.P2,
            self.background.B_r, self.background.B_z,
        )
        replacement = self.background.material_mask
        if np.any(replacement):
            physical = tuple(tuple(np.where(replacement,
                self.background.relative_permittivity if i == j else 0.0, physical[i][j])
                for j in range(3)) for i in range(3))
        self.dielectric_physical = physical
        loss = float(self.cfg.get("solver", {}).get("artificial_loss", 0.0))
        if loss < 0:
            raise ValueError("artificial_loss must be nonnegative")
        blocks = []
        for row in range(3):
            line = []
            for col in range(3):
                values = physical[row][col].ravel().astype(complex).copy()
                if row == col:
                    values -= 1j * (loss+self.absorber_epsilon_loss.ravel())
                values *= epsilon_factors[row][col].ravel()
                line.append(sparse.diags(values, format="csr"))
            blocks.append(line)
        self.Eps = sparse.bmat(blocks, format="csr")

        self.A_cpu = (self.C_dual @ self.Mu_inverse @ self.C - (self.k0**2) * self.Eps).tocsr()
        self._apply_boundary_termination()
        axis_method = str(self.cfg.get("solver", {}).get("axis_regularization", "legacy"))
        if axis_method not in ("legacy", "coupled_m1"):
            raise ValueError("solver.axis_regularization must be legacy or coupled_m1")
        self.axis_projection = regular_axis_projection(Nr, Nz, m) if axis_method == "coupled_m1" else None
        if self.axis_projection is not None:
            q = self.axis_projection
            self.A_cpu = (q.conj().T@self.A_cpu@q).tocsr()
            print(f"[axis] coupled m={m} Galerkin closure; independent unknowns={self.A_cpu.shape[0]}", flush=True)
        if self.use_gpu:
            import cupyx.scipy.sparse as cpsp
            self.A = cpsp.csr_matrix(self.A_cpu)
        else:
            self.A = self.A_cpu
        self._operator_ready = True

    def _apply_boundary_termination(self):
        boundary = self.cfg.get("boundary", {})
        if not boundary.get("pec_pml_termination", True):
            self.pec_rows = np.empty(0, dtype=int)
            return
        N = self.Nr * self.Nz
        ids = np.arange(N).reshape(self.Nr, self.Nz)
        rows = [N + ids[-1, :], 2 * N + ids[-1, :],
                ids[:, 0], N + ids[:, 0], ids[:, -1], N + ids[:, -1]]
        self.pec_rows = np.unique(np.concatenate(rows))
        a = self.A_cpu.tolil()
        for index in self.pec_rows:
            a.rows[int(index)] = [int(index)]
            a.data[int(index)] = [1.0 + 0j]
        self.A_cpu = a.tocsr()

    def _build_source(self):
        """Build the finite-width equivalent current source."""
        ant = self.cfg["antenna"]
        z0 = float(ant["z_position"])
        antenna_type = str(ant.get("type", "helical")).strip().lower()
        length = float(ant.get("active_length_m", ant.get("length", 0.0)))
        z1 = z0 + length
        if length <= 0 or z0 < 0 or z1 > self.L:
            raise ValueError("Antenna must have positive length and lie inside the z domain")

        r_ant = float(ant.get("radius_m", float(ant.get("radius_factor", 0.8)) * self.r0))
        if not 0.0 < r_ant < self.r_extent:
            raise ValueError("Antenna radius lies outside the cylindrical computational domain")
        if "radial_width_m" in ant:
            width = max(float(ant["radial_width_m"]), 0.5 * self.dr)
        else:
            width = max(float(ant.get("radial_width_cells", 2.0)), 0.5) * self.dr
        J0 = float(ant.get("current_density", 1.0))
        turns = float(ant.get("turns", 1.0))
        if turns <= 0:
            raise ValueError("antenna.turns must be positive")
        handedness = str(ant.get("handedness", "right")).strip().lower()
        if handedness in ("right", "right_handed", "правое", "правая", "+1", "1"):
            handedness_sign = 1.0
        elif handedness in ("left", "left_handed", "левое", "левая", "-1"):
            handedness_sign = -1.0
        else:
            raise ValueError("antenna.handedness must be right or left")
        signed_turns = handedness_sign * turns
        m = int(self.cfg["wave"].get("m_mode", 0))

        two_arm = antenna_type in {"right_half_helix_two_arm", "two_arm_half_helix"}
        if two_arm:
            separation = float(ant.get("arm_azimuth_separation_rad", np.pi))
            phase_offset = float(ant.get("arm_phase_offset_rad", np.pi))
            arm_normalization = float(ant.get("arm_normalization", 0.5))
            arm_imbalance = float(ant.get("arm_imbalance", 0.05))
            if not np.isfinite(separation + phase_offset + arm_normalization + arm_imbalance) or arm_normalization <= 0 or arm_imbalance < 0:
                raise ValueError("Invalid two-arm antenna phase or normalization")
            arm_factor = arm_normalization * (
                1.0 + (1.0 + arm_imbalance)
                * np.exp(1j * (m * separation + phase_offset))
            ) / (2.0 + arm_imbalance)
        else:
            arm_factor = 1.0 + 0j

        axial = np.zeros(self.Nz, dtype=float)
        mask_z = (self.z >= z0) & (self.z <= z1)
        u = (self.z[mask_z] - z0) / length
        axial[mask_z] = np.sin(np.pi * u) ** 2
        k_turn = 2.0 * np.pi * signed_turns / length
        k_phase = m * k_turn
        phase = np.exp(-1j * k_phase * (self.z - z0))

        radial = np.exp(-0.5 * ((self.r - r_ant) / width) ** 2)
       
        pml_r = int(self.cfg["grid"]["pml_r"])
        pml_start = self.r_extent - pml_r * self.dr
        if r_ant + 3.0 * width >= pml_start:
            raise ValueError(
                "Antenna radial profile overlaps the radial PML; reduce radius_factor "
                "or radial_width_cells, or increase the domain radius."
            )
        radial[self.r >= pml_start] = 0.0

       
        tangent_phi = r_ant * k_turn
        norm = np.sqrt(1.0 + tangent_phi**2)
        Jz = J0 * arm_factor * radial[:, None] * axial[None, :] * phase[None, :] / norm
        Jphi = Jz * tangent_phi
        Jr = np.zeros_like(Jz, dtype=complex)

        feed = ant.get("feed_leads", {})
        if feed is True:
            feed = {}
        if feed:
            if not isinstance(feed, dict):
                raise ValueError("antenna.feed_leads must be a mapping or false")
            if bool(feed.get("enabled", True)):
                feed_length = float(feed.get("length_m", 0.020915))
                feed_fraction = float(feed.get("current_fraction", 1.0))
                if feed_length <= 0 or feed_fraction < 0:
                    raise ValueError("Invalid antenna feed-lead parameters")
                feed_start = z0 - feed_length
                feed_mask = (self.z >= feed_start) & (self.z <= z0)
                if feed_start < 0:
                    raise ValueError("Antenna feed leads extend outside the axial domain")
                feed_u = (self.z[feed_mask] - feed_start) / feed_length
                feed_axial = np.sin(np.pi * feed_u) ** 2
                feed_phase = np.exp(-1j * m * k_turn * (self.z[feed_mask] - z0))
                Jz[:, feed_mask] += (
                    J0 * arm_factor * feed_fraction * radial[:, None]
                    * feed_axial[None, :] * feed_phase[None, :]
                )
        if bool(ant.get("enforce_current_continuity", False)):
            dJz_dz = np.gradient(Jz, self.z, axis=1, edge_order=2)
            rhs = -1j * m * Jphi - self.r[:, None] * dJz_dz
            Jr[1:, :] = np.cumsum(
                0.5 * (rhs[:-1, :] + rhs[1:, :]) * np.diff(self.r)[:, None], axis=0
            ) / self.r[1:, None]
            pml_r = int(self.cfg["grid"]["pml_r"])
            Jr[max(self.Nr - pml_r, 0):, :] = 0.0
        J = np.concatenate([Jr.ravel(), Jphi.ravel(), Jz.ravel()]).astype(complex)
        if not np.any(np.abs(J) > 0):
            raise ValueError("Antenna source is zero on the current grid; check z_position/length")
        return J

    def source_divergence_quality(self, source):
        """Return a normalized divergence diagnostic for the equivalent RF source."""
        total = self.Nr * self.Nz
        jr = source[:total].reshape(self.Nr, self.Nz)
        jphi = source[total:2 * total].reshape(self.Nr, self.Nz)
        jz = source[2 * total:].reshape(self.Nr, self.Nz)
        m = int(self.cfg["wave"].get("m_mode", 0))
        divergence = (
            np.gradient(self.r[:, None] * jr, self.r, axis=0, edge_order=2) / self.r[:, None]
            + 1j * m * jphi / self.r[:, None]
            + np.gradient(jz, self.z, axis=1, edge_order=2)
        )
        mask = self.physical_domain_mask()
        numerator = np.linalg.norm(divergence[mask]) * max(self.dr, self.dz)
        denominator = np.linalg.norm(source) * max(self.dr, self.dz, 1e-30)
        return float(numerator / max(denominator, 1e-30))

    def solve(self):
        if not self._operator_ready:
            self._setup_operator()
        self._wave_structure_cached = False
        J = self._build_source()
        self.source_current = J
        self.source_divergence = self.source_divergence_quality(J)
        print(f"[source] normalized div(J)={self.source_divergence:.3e}")
        b = -1j * self.omega * mu_0 * J
        if self.pec_rows.size:
            b[self.pec_rows] = 0.0
        if self.axis_projection is not None:
            b = self.axis_projection.conj().T@b
        tol = float(self.cfg["solver"].get("tol", 1e-6))
        maxit = int(self.cfg["solver"].get("max_iter", 2000))
        method = self.cfg["solver"].get("method", "auto").lower()

        direct_limit = int(self.cfg["solver"].get("direct_max_unknowns", 25000))
        use_direct = (method == "direct") or (
            method == "auto" and self.A_cpu.shape[0] <= direct_limit
        )
        if use_direct:
            print(f"[direct] sparse LU solve on CPU ({self.A_cpu.shape[0]} unknowns)")
            x = _direct_refined(self.A_cpu, b, tol,
                max_steps=int(self.cfg['solver'].get('direct_refine_steps', 6)))
            info = 0
        elif self.use_gpu:
            import cupyx.scipy.sparse.linalg as cpsplinalg
            b_gpu = self.cp.asarray(b)
            xg, info = cpsplinalg.gmres(
                self.A, b_gpu, rtol=tol, atol=0.0, maxiter=maxit, restart=100
            )
            x = self.cp.asnumpy(xg)
        else:
     
            print("[CPU] building ILU preconditioner ...")
            drop_tol = float(self.cfg["solver"].get("ilu_drop_tol", 1e-3))
            fill_factor = float(self.cfg["solver"].get("ilu_fill_factor", 8.0))
            try:
                ilu = spilu(self.A_cpu.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
                M = LinearOperator(self.A_cpu.shape, matvec=ilu.solve, dtype=np.complex128)
            except Exception as exc:
                print(f"[WARN] ILU preconditioner failed ({exc}); proceeding without it")
                M = None

            if method == "bicgstab":
                x, info = scipy_bicgstab(self.A_cpu, b, rtol=tol, atol=0.0,
                                          maxiter=maxit, M=M)
            else:  
                x, info = scipy_gmres(self.A_cpu, b, rtol=tol, atol=0.0,
                                      maxiter=maxit, restart=100, M=M)

        denom = np.linalg.norm(b)
        residual = np.linalg.norm(self.A_cpu @ x - b) / max(denom, np.finfo(float).tiny)
        self.solver_info = int(info)
        self.relative_residual = float(residual)
        self.residual_equation_space = "axis_constrained_Galerkin" if self.axis_projection is not None else "full_collocated"
        self.converged = (info == 0 and np.isfinite(residual) and residual <= max(10 * tol, 1e-7))
        print(f"[solver] info={info}, relative residual={residual:.3e}")
        if not self.converged:
            raise RuntimeError(
                f"FDFD solve did not converge (info={info}, relative residual={residual:.3e}). "
                "Do not interpret the fields or k_z as a physical result; try finer preconditioning, "
                "a smaller test grid, or a better-conditioned formulation."
            )

        N = self.Nr * self.Nz
        if self.axis_projection is not None:
            x = self.axis_projection@x
        self.Er = x[0:N].reshape((self.Nr, self.Nz))
        self.Ephi = x[N:2 * N].reshape((self.Nr, self.Nz))
        self.Ez = x[2 * N:3 * N].reshape((self.Nr, self.Nz))
        self.E_mag = np.sqrt(np.abs(self.Er)**2 + np.abs(self.Ephi)**2 + np.abs(self.Ez)**2)
        self.E_field = self.E_mag
        self._solved = True
        return self.Er, self.Ephi, self.Ez

    def compute_wave_structure(self):
        if getattr(self, "_wave_structure_cached", False):
            return self._kz_cached, self._alpha_cached

        from wave_analysis import analyze_axial_wave

        result = analyze_axial_wave(self)
        self.kz_z = result.z
        self.kz_phase = result.phase
        self.kz_phase_slope = result.phase_slope
        self.kz_phase_fit = result.phase_slope * (result.z - np.mean(result.z)) + np.mean(result.phase)
        self.kz_component = result.component
        self.kz_radial_index = result.radial_index
        self.kz_fit_quality = result.quality
        self.kz_fit_purity = result.purity
        self.kz_phase_cycles = result.cycles
        self.kz_candidate_spread = result.spread
        self.kz_reflection_ratio = result.reflection
        self.kz_fit_reliable = not bool(result.reason)
        self.kz_fit_reason = result.reason
        self.kz_alpha = np.full(result.z.size, result.alpha)
        self._kz_cached = result.kz
        self._alpha_cached = self.kz_alpha
        self._wave_structure_cached = True

        if self.kz_fit_reliable:
            print(
                f"[field-fit] k_z={result.kz/100:.5f} 1/cm, "
                f"alpha={result.alpha/100:.5f} 1/cm, "
                f"phase R^2={result.quality:.3f}, purity={result.purity:.3f}, "
                f"cycles={result.cycles:.2f}"
            )
        else:
            print(
                f"[field-fit] rejected: {result.reason}; "
                f"purity={result.purity:.3f}, cycles={result.cycles:.2f}, "
                f"spread={result.spread/100:.4f} 1/cm"
            )
        return result.kz, self.kz_alpha

    def compute_dispersion_modes(self):
        """Solve the radial quadratic eigenvalue problem for complex k_z.

        The radial eigenproblem may use grid.dispersion_Nr independently of the
        2-D driven-field grid.  This is a convergence control: it improves the
        radial eigenvalue without increasing the size of the full r-z solve.
        Returned modes include their own radial coordinates.

        Uses exp(i*m*phi - i*k_z*z + i*omega*t).
        For an axially varying background this is a local radial slice problem.
        """
        grid = self.cfg["grid"]
        p = self.cfg["plasma"]
        Nr = int(grid.get("dispersion_Nr", self.Nr))
        if Nr < 16:
            raise ValueError("grid.dispersion_Nr must be at least 16")
        dr = self.r_extent / Nr
        r = (np.arange(Nr) + 0.5) * dr
        pml_r = int(grid.get("dispersion_pml_r", max(4, round(grid["pml_r"] * Nr / self.Nr))))
        pml = PML(Nr, self.Nz, pml_r, int(grid["pml_z"]),
                  float(grid["pml_sigma_max"]), dr, self.dz)
        sr = pml.scale_r()
        m = int(self.cfg["wave"].get("m_mode", 0))

        formulation = str(self.cfg.get("boundary", {}).get("pml_formulation", "legacy_stretched"))
        tensor_pml = formulation == "tensor"
        C0_plus, C0_minus, C1 = radial_curl_polynomial(r, dr, sr if formulation == "legacy_stretched" else np.ones(Nr), m)
        radial_pml = PML(Nr, 3, pml_r, 0, float(grid["pml_sigma_max"]), dr, self.dz)
        if tensor_pml:
            eps_factors, mu_inverse = radial_pml.tensor_factors(r)
            eps_factors = tuple(tuple(value[:, 0] for value in line) for line in eps_factors)
            mu_inverse = tuple(value[:, 0] for value in mu_inverse)
        else:
            eps_factors = tuple(tuple(np.ones(Nr) for _ in range(3)) for _ in range(3))
            mu_inverse = tuple(np.ones(Nr, complex) for _ in range(3))
        Mu_inverse = sparse.diags(np.concatenate(mu_inverse), format="csr")
        z_ref = float(self.cfg.get("solver", {}).get("dispersion_z_m", self.L / 2))
        jr = int(np.argmin(np.abs(self.z - z_ref)))
        n_profile = np.interp(r, self.r, self.background.n_e[:, jr])
        br_profile = np.interp(r, self.r, self.background.B_r[:, jr])
        bz_profile = np.interp(r, self.r, self.background.B_z[:, jr])
        nue_profile = np.interp(r, self.r, self.background.nu_e[:, jr])
        nui_profile = np.interp(r, self.r, self.background.nu_i[:, jr])
        S, D, P = _stix_arrays(n_profile, np.hypot(br_profile, bz_profile), self.omega, p,
                               nue_profile, nui_profile)
        physical = _dielectric_components(S, D, P, br_profile, bz_profile)
        if np.any(self.background.material_mask):
            local_er = np.interp(r, self.r, self.background.relative_permittivity[:, jr].real)
            local_loss = np.interp(r, self.r, self.background.relative_permittivity[:, jr].imag)
            material = (np.interp(r, self.r, self.background.material_mask[:, jr].astype(float)) > 0.5)
            material_epsilon = local_er + 1j * local_loss
            physical = tuple(tuple(np.where(material, material_epsilon if i == j else 0.0, physical[i][j])
                            for j in range(3)) for i in range(3))
        loss = float(self.cfg["solver"].get("artificial_loss", 0.0))
        radial_loss = np.zeros(Nr)
        if formulation == "passive_absorber":
            radial_loss = radial_pml.passive_electric_loss(
                self.omega, self.cfg.get("boundary", {}).get("absorber_conductivity_S_m"))[:, 0]
        Eps = sparse.bmat([
            [sparse.diags((physical[i][j] - (1j * (loss+radial_loss) if i == j else 0))*eps_factors[i][j], format="csr")
             for j in range(3)] for i in range(3)
        ], format="csr")

        A0 = C0_minus @ Mu_inverse @ C0_plus - self.k0**2 * Eps
        A1 = C0_minus @ Mu_inverse @ C1 + C1 @ Mu_inverse @ C0_plus
        A2 = C1 @ Mu_inverse @ C1
        if self.cfg.get("boundary", {}).get("pec_pml_termination", True):
            rows = [Nr + Nr - 1, 2 * Nr + Nr - 1]
            matrices = [A0.tolil(), A1.tolil(), A2.tolil()]
            for index in rows:
                for matrix in matrices:
                    matrix.rows[index] = []
                    matrix.data[index] = []
                matrices[0].rows[index] = [index]
                matrices[0].data[index] = [1.0 + 0j]
            A0, A1, A2 = (matrix.tocsr() for matrix in matrices)
        axis_q = regular_axis_projection(Nr, 1, m) if self.cfg.get("solver", {}).get("axis_regularization") == "coupled_m1" else None
        if axis_q is not None:
            adj = axis_q.conj().T
            A0, A1, A2 = ((adj@a@axis_q).tocsr() for a in (A0, A1, A2))
        scaling = str(self.cfg.get("solver", {}).get("radial_pencil_scaling", "equilibrated"))
        if threadpool_limits is None:
            vals, vecs, pencil_report = solve_quadratic(A0, A1, A2, scaling=scaling)
        else:
          
            with threadpool_limits(limits=1, user_api="blas"):
                vals, vecs, pencil_report = solve_quadratic(A0, A1, A2, scaling=scaling)
        self.dispersion_diagnostics = pencil_report
        print(f"[radial-pencil] scaling={scaling}; k_scale={pencil_report['k_scale_rad_m']:.3e} 1/m; "
              f"finite roots={pencil_report['finite_eigenvalues']}", flush=True)

        active = n_profile > max(float(np.max(n_profile)) * 1e-8, 0.0)
        if np.any(active):
            weights = n_profile[active]
            n_reference = float(np.average(n_profile[active], weights=weights))
            b_reference = float(np.average(np.hypot(br_profile[active], bz_profile[active]), weights=weights))
        else:
            n_reference = float(self.params.n_e)
            b_reference = float(self.params.B0)
        if b_reference <= 0 or not np.isfinite(b_reference):
            b_reference = float(self.params.B0)
        self.dispersion_reference_n_e = n_reference
        self.dispersion_reference_B0_T = b_reference
        k_ref = abs(configured_theoretical_kz(
            self.omega, n_reference, b_reference, p, self.cfg.get("solver", {}), m_mode=m
        ))
        k_min = max(1.0, 0.02 * k_ref)
        k_max_multiplier = float(self.cfg.get("solver", {}).get("kz_r_max_multiplier", 20.0))
        if not np.isfinite(k_max_multiplier) or k_max_multiplier <= 1.0:
            raise ValueError("solver.kz_r_max_multiplier must be greater than 1")
        k_max = max(k_max_multiplier * k_ref, 100.0)
        selected = []
        for idx, kval in enumerate(vals):
            if not (np.isfinite(kval.real) and np.isfinite(kval.imag)):
                continue
            if kval.real <= k_min or kval.real >= k_max or kval.imag > 1e-6:
                continue
            damping_ratio = abs(kval.imag) / max(abs(kval.real), 1.0)
          
            if damping_ratio > 0.5:
                continue
            if any(abs(kval - existing["kz"]) < max(1e-2, 1e-3 * abs(kval)) for existing in selected):
                continue

            e = np.asarray(vecs[:, idx], dtype=complex)
            qep_residual = polynomial_residual(A0, A1, A2, kval, e)
            if not np.isfinite(qep_residual) or qep_residual > float(self.cfg.get("solver", {}).get("eigenmode_max_qep_residual", 1e-7)):
                continue
            if axis_q is not None:
                e = axis_q@e
            oscillation = radial_grid_oscillation((e[:Nr], e[Nr:2 * Nr], e[2 * Nr:]), pml_r)
            resolved = oscillation <= float(self.cfg.get("solver", {}).get("max_radial_grid_oscillation", 0.22))
            norm = float(np.max(np.abs(e)))
            if not np.isfinite(norm) or norm == 0:
                continue
            e = e / norm
            selected.append({
                "kz": complex(kval),
                "attenuation_m_inv": float(max(-kval.imag, 0.0)),
                "Er": e[:Nr].copy(), "Ephi": e[Nr:2 * Nr].copy(), "Ez": e[2 * Nr:].copy(),
                "r": r.copy(), "radial_grid_N": Nr,
                "damping_ratio": float(damping_ratio),
                "radial_grid_oscillation": oscillation,
                "radial_grid_resolved": bool(resolved),
                "qep_relative_residual": qep_residual,
                "pml_formulation": formulation,
                "axis_regularization": "coupled_m1" if axis_q is not None else "legacy",
                "qep_residual_equation_space": "axis_constrained_Galerkin" if axis_q is not None else "full_collocated",
            })
        selected.sort(key=lambda mode: (mode["kz"].real, mode["kz"].imag))
        if not selected:
            raise RuntimeError(
                "No forward radial eigenmodes passed the spectral bounds and QEP residual check. "
                "Examine the mode family, radial mesh and boundary conditions."
            )
        self.dispersion_modes = selected
        self.dispersion_diagnostics.update(
            accepted_candidates=len(selected),
            mesh_screened_candidates=sum(mode["radial_grid_resolved"] for mode in selected),
            min_qep_residual=min(mode["qep_relative_residual"] for mode in selected),
            max_qep_residual=max(mode["qep_relative_residual"] for mode in selected),
        )
        self.dispersion_z_m = float(self.z[jr])
        self.dispersion_grid_Nr = Nr
        print(f"[radial-spectrum] candidates={len(selected)}; "
              f"mesh-screened={self.dispersion_diagnostics['mesh_screened_candidates']}; "
              f"QEP residual range={self.dispersion_diagnostics['min_qep_residual']:.3e}.."
              f"{self.dispersion_diagnostics['max_qep_residual']:.3e}", flush=True)
        return selected

    def compute_absorbed_power(self):
        if not getattr(self, "_solved", False):
            raise RuntimeError("Solve the RF field before computing dissipation")
        E = np.stack((self.Er, self.Ephi, self.Ez), axis=0)
        e_eps = np.zeros_like(E)
        for i in range(3):
            for j in range(3):
                e_eps[i] += self.dielectric_physical[i][j] * E[j]
        quadratic = np.sum(np.conj(E) * e_eps, axis=0)
        power = -0.5 * self.omega * epsilon_0 * np.imag(quadratic)
        active = self.background.n_e > 0
        self.negative_absorption_fraction = float(np.mean(power[active] < -1e-8 * max(np.max(abs(power)), 1e-30))) if np.any(active) else 0.0
        self.material_absorption = np.where(self.background.material_mask, power, 0.0)
        return np.where(active, power, 0.0)

    def compute_material_absorption(self):
        self.compute_absorbed_power()
        return self.material_absorption

    def integrated_absorbed_power(self, physical_only=True):
        density = self.compute_absorbed_power()
        if physical_only:
            density = np.where(self.physical_domain_mask(), density, 0.0)
        return float(2 * np.pi * np.sum(density * self.r[:, None]) * self.dr * self.dz)

    def compute_magnetic_field(self):
        """Return (Br, Bphi, Bz) from Faraday's law for exp(+i*omega*t)."""
        if not getattr(self, "_solved", False):
            raise RuntimeError("Call solve() before deriving the magnetic field")
        vec = np.concatenate([self.Er.ravel(), self.Ephi.ravel(), self.Ez.ravel()])
        curl_e = self.C @ vec
        N = self.Nr * self.Nz
        factor = 1j / self.omega 
        B_r = factor * curl_e[:N].reshape(self.Nr, self.Nz)
        B_phi = factor * curl_e[N:2*N].reshape(self.Nr, self.Nz)
        B_z = factor * curl_e[2*N:].reshape(self.Nr, self.Nz)
        return B_r, B_phi, B_z

    def compute_poynting(self):
        H_r, H_phi, H_z = self.compute_magnetic_intensity()
        radial = 0.5 * np.real(self.Ephi * np.conj(H_z) - self.Ez * np.conj(H_phi))
        axial = 0.5 * np.real(self.Er * np.conj(H_phi) - self.Ephi * np.conj(H_r))
        return radial, axial

    def compute_magnetic_intensity(self):
        magnetic = self.compute_magnetic_field()
        return tuple(b*metric/mu_0 for b, metric in zip(magnetic, self.mu_inverse))

    def physical_domain_mask(self, extra_radial_cells=0, extra_axial_cells=0):
        return physical_mask(self.cfg, self.r, self.z, self.dr, self.dz,
                             extra_radial_cells, extra_axial_cells)

    def non_pml_domain_mask(self):
        return non_pml_mask(self.cfg, self.r, self.z, self.dr, self.dz)

    def energy_budget(self):
        from energy import energy_budget
        return energy_budget(self)

    def check_boundary_passivity(self, report):
        if self.cfg.get("solver", {}).get("require_passive_boundary", False) and not report["boundary_passivity_passed"]:
            raise RuntimeError("Absorbing boundary failed the net operator-work passivity check; "
                               "fields and dispersion must not be accepted as a physical result")

    def compute_poynting_z(self, physical_only=False):
        axial = self.compute_poynting()[1]
        if physical_only:
            axial = np.where(self.physical_domain_mask(), axial, np.nan)
        return axial

    def physical_poynting_power(self, z_m, r_max_m=None):
        """Integrate the axial Poynting flux over a physical cross-section."""
        z_index = int(np.argmin(np.abs(self.z - float(z_m))))
        mask = self.physical_domain_mask()[:, z_index]
        if r_max_m is not None:
            mask &= self.r <= float(r_max_m)
        values = self.compute_poynting_z(physical_only=False)[:, z_index]
        values = np.where(mask, values, 0.0)
        return float(2.0 * np.pi * np.sum(values * self.r) * self.dr)

    def magnetic_divergence_quality(self):
        br, bp, bz = self.compute_magnetic_field()
        term_r = np.gradient(self.r[:, None] * br, self.r, axis=0, edge_order=2) / self.r[:, None]
        term_phi = 1j * int(self.cfg["wave"].get("m_mode", 0)) * bp / self.r[:, None]
        term_z = np.gradient(bz, self.z, axis=1, edge_order=2)
        npr = int(self.cfg["grid"]["pml_r"])
        npz = int(self.cfg["grid"]["pml_z"])
        shape = (slice(2, self.Nr - npr - 2), slice(npz + 2, self.Nz - npz - 2))
        div = (term_r + term_phi + term_z)[shape]
        field = np.sqrt(np.abs(br[shape])**2 + np.abs(bp[shape])**2 + np.abs(bz[shape])**2)
        magnitude = np.linalg.norm(div) * self.dr / max(np.linalg.norm(field), 1e-300)
        return float(magnitude)


def select_dispersion_mode(
    modes, k_reference, previous_kz=None, frequency_ratio=1.0, previous_mode=None,
    require_resolved=False,
):
    """Select a radial eigenmode and continue a branch across frequency.

    On the first frequency, the parallel-whistler estimate is only a rough
    branch hint. On later frequencies, radial eigenfunction overlap is used
    together with a weak sqrt(f) predictor; this is more reliable than selecting
    the nearest eigenvalue independently at each frequency. The result is
    still a branch-tracking heuristic, so inspect convergence and field shapes.
    """
    if not modes:
        raise ValueError("modes must be a non-empty list")
    resolved = [mode for mode in modes if mode.get("radial_grid_resolved", False)]
    if require_resolved and not resolved:
        raise RuntimeError("No radial eigenmode passed the configured grid-resolution criterion")
    candidates = resolved if resolved else modes
    target = complex(k_reference)
    if previous_kz is not None:
        target = complex(previous_kz) * np.sqrt(max(float(frequency_ratio), 1e-12))
    scale = max(abs(target), 1.0)

    def vector_on_grid(mode, target_r):
        r0 = np.asarray(mode.get("r", target_r), dtype=float)
        vals = []
        for key in ("Er", "Ephi", "Ez"):
            y = np.asarray(mode[key], dtype=complex)
            if y.size != target_r.size or not np.allclose(r0, target_r):
                y = np.interp(target_r, r0, y.real) + 1j * np.interp(target_r, r0, y.imag)
            vals.append(y)
        return np.concatenate(vals)

    def overlap(a, b):
        r_grid = np.asarray(b.get("r", a.get("r")), dtype=float)
        va = vector_on_grid(a, r_grid)
        vb = vector_on_grid(b, r_grid)
      
        weights = np.sqrt(np.maximum(r_grid, np.finfo(float).tiny))
        w = np.tile(weights, 3)
        va, vb = va * w, vb * w
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(abs(np.vdot(va, vb)) / denom) if denom > 0 else 0.0

    if previous_mode is not None:
        def branch_score(mode):
            kz = mode["kz"]
            shape_penalty = 1.0 - overlap(previous_mode, mode)
            distance = min(abs(kz - target) / scale, 2.0)
            damping = max(-kz.imag, 0.0) / max(abs(kz.real), 1.0)
            grid_penalty = 2.0 * max(float(mode.get("radial_grid_oscillation", 0.0)) - 0.15, 0.0)
            return 0.70 * shape_penalty + 0.25 * distance + 0.05 * damping + grid_penalty
        chosen = min(candidates, key=branch_score)
        result = dict(chosen)
        result["continuation_overlap"] = radial_mode_overlap(previous_mode, chosen)
        result["continuation_distance"] = float(abs(chosen["kz"] - target) / scale)
        return result

    def initial_score(mode):
        kz = mode["kz"]
        distance_real = abs(kz.real - target.real) / scale
        distance_imag = 0.15 * abs(kz.imag - target.imag) / scale
        damping = max(-kz.imag, 0.0) / max(abs(kz.real), 1.0)
        grid_penalty = 2.0 * max(float(mode.get("radial_grid_oscillation", 0.0)) - 0.15, 0.0)
        return distance_real + distance_imag + 0.75 * damping + grid_penalty

    chosen = min(candidates, key=initial_score)
    result = dict(chosen)
    result["continuation_overlap"] = 1.0
    result["continuation_distance"] = float(abs(chosen["kz"] - target) / scale)
    return result


def radial_mode_overlap(mode_a, mode_b) -> float:
    """Return the weighted complex overlap of two radial eigenfunctions.

    The profiles may come from different radial grids. Interpolation onto the
    second mode's grid makes the metric directly usable for mesh convergence
    studies and frequency continuation.
    """
    ra = np.asarray(mode_a.get("r"), dtype=float)
    rb = np.asarray(mode_b.get("r"), dtype=float)
    if ra.ndim != 1 or rb.ndim != 1 or ra.size < 2 or rb.size < 2:
        return 0.0
    vectors = []
    for key in ("Er", "Ephi", "Ez"):
        values = np.asarray(mode_a.get(key), dtype=complex)
        if values.size != ra.size:
            return 0.0
        vectors.append(np.interp(rb, ra, values.real) + 1j * np.interp(rb, ra, values.imag))
    left = np.concatenate(vectors)
    right = np.concatenate([np.asarray(mode_b.get(key), dtype=complex) for key in ("Er", "Ephi", "Ez")])
    weights = np.tile(np.sqrt(np.maximum(rb, np.finfo(float).tiny)), 3)
    left *= weights
    right *= weights
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(abs(np.vdot(left, right)) / denominator) if denominator > 0 else 0.0

