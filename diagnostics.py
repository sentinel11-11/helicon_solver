from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec


class Diagnostics:
    def __init__(self, solver, results_dir="results", dpi=200):
        self.solver = solver
        self.results_dir = results_dir
        self.dpi = int(dpi)
        os.makedirs(results_dir, exist_ok=True)

    def _save(self, fig, name):
        path = os.path.join(self.results_dir, name)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {path}")

    def plot_field_map(self):
        s = self.solver
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        fields = [np.abs(s.Er), np.abs(s.Ephi), np.abs(s.Ez)]
        titles = [r"$|E_r(r,z)|$", r"$|E_\phi(r,z)|$", r"$|E_z(r,z)|$"]
        for ax, E, title in zip(axes, fields, titles):
            vmax = float(np.max(E))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            vmin = max(vmax * 1e-5, np.finfo(float).tiny)
            im = ax.pcolormesh(s.z * 100, s.r * 100, E, shading="auto",
                               cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))
            ax.axvspan(float(s.cfg["antenna"]["z_position"]) * 100,
                       (float(s.cfg["antenna"]["z_position"]) + float(s.cfg["antenna"]["length"])) * 100,
                       color="cyan", alpha=0.12, label="antenna region")
            ax.axhline(s.r0 * 100, color="deepskyblue", linestyle="--", linewidth=1.1)
            ax.set_ylabel("r, cm")
            ax.set_title(title)
            ax.grid(False)
            fig.colorbar(im, ax=ax, label="V/m (magnitude)")
        axes[-1].set_xlabel("z, cm")
        self._save(fig, "field_map_vector.png")

    def plot_radial_profiles(self, z_positions=None):
        s = self.solver
        if z_positions is None:
            z_positions = s.cfg["scan"]["z_positions"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for z0 in z_positions:
            if not (0 <= float(z0) <= s.L):
                continue
            j = int(np.argmin(np.abs(s.z - float(z0))))
            ax.plot(s.r * 100, np.abs(s.Ez[:, j]), label=f"z = {s.z[j]*100:.1f} cm")
        ax.set_xlabel("r, cm")
        ax.set_ylabel(r"$|E_z|$, V/m")
        ax.set_title("Radial profiles of the axial electric field")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "radial_profiles.png")

    def plot_wave_structure(self):
        s = self.solver
        k_local, alpha_local = s.compute_wave_structure()
        z_slice = s.kz_z
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        phase = getattr(s, "kz_phase", None)
        if phase is not None and len(phase) == len(z_slice):
            axes[0].plot(z_slice * 100, phase, "k.", ms=3, label="unwrapped phase")
            if getattr(s, "kz_fit_reliable", False) and np.isfinite(k_local):
                fit = s.kz_phase_slope * (z_slice - np.mean(z_slice)) + np.mean(phase)
                axes[0].plot(z_slice * 100, fit, "r-", lw=1.5,
                             label=f"phase fit: |kz|={abs(k_local)/100:.4f} 1/cm")
            axes[0].set_ylabel("phase, rad")
            axes[0].set_title(
                f"Downstream phase fit ({getattr(s, 'kz_component', 'unknown')}; "
                f"quality={getattr(s, 'kz_fit_quality', 0):.2f}; "
                f"cycles={getattr(s, 'kz_phase_cycles', 0):.2f})"
            )
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, f"kz = {k_local/100:.4f} 1/cm",
                         transform=axes[0].transAxes, ha="center", va="center")
        axes[0].grid(True, alpha=0.3)

        if np.ndim(alpha_local) == 0:
            alpha_plot = np.full_like(z_slice, float(alpha_local))
        else:
            alpha_plot = np.asarray(alpha_local)
            if alpha_plot.size != z_slice.size:
                alpha_plot = np.resize(alpha_plot, z_slice.size)
        axes[1].plot(z_slice * 100, alpha_plot / 100, "b-", lw=1.5)
        axes[1].set_xlabel("z, cm")
        axes[1].set_ylabel(r"$\alpha$, 1/cm")
        axes[1].set_title("Estimated amplitude attenuation coefficient")
        axes[1].grid(True, alpha=0.3)
        self._save(fig, "wave_structure.png")

    def plot_magnetic_field_map(self):
        s = self.solver
        fields = s.compute_magnetic_field()
        titles = (r"$B_r$", r"$B_{\phi}$", r"$B_z$")
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, sharey=True)
        for row, (field, title) in enumerate(zip(fields, titles)):
            magnitude = np.abs(field)
            peak = max(float(magnitude.max()), np.finfo(float).tiny)
            image = axes[row, 0].pcolormesh(
                s.z * 100, s.r * 100, magnitude, shading="auto", cmap="inferno",
                norm=LogNorm(vmin=max(peak * 1e-5, np.finfo(float).tiny), vmax=peak),
            )
            fig.colorbar(image, ax=axes[row, 0], label="|B|, T")
            phase = np.ma.masked_where(magnitude < peak * 1e-4, np.angle(field))
            image = axes[row, 1].pcolormesh(
                s.z * 100, s.r * 100, phase, shading="auto", cmap="twilight",
                vmin=-np.pi, vmax=np.pi,
            )
            fig.colorbar(image, ax=axes[row, 1], label="phase, rad")
            axes[row, 0].set_title(f"{title}: amplitude")
            axes[row, 1].set_title(f"{title}: phase")
            axes[row, 0].set_ylabel("r, cm")
            for ax in axes[row]:
                ax.axhline(s.r0 * 100, ls="--", lw=0.8, color="deepskyblue")
        for ax in axes[-1]:
            ax.set_xlabel("z, cm")
        fig.tight_layout()
        self._save(fig, "magnetic_field_map.png")

    def plot_magnetic_profiles(self, z_positions=None):
        s = self.solver
        if z_positions is None:
            z_positions = s.cfg["scan"]["z_positions"]
        fields = s.compute_magnetic_field()
        titles = (r"$|B_r|$", r"$|B_{\phi}|$", r"$|B_z|$")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
        for z in z_positions:
            if not 0 <= float(z) <= s.L:
                continue
            iz = int(np.argmin(np.abs(s.z - float(z))))
            for ax, field in zip(axes, fields):
                ax.plot(s.r * 100, np.abs(field[:, iz]), label=f"z={s.z[iz]*100:.1f} cm")
        for ax, title in zip(axes, titles):
            ax.set(title=title, xlabel="r, cm", ylabel="|B|, T")
            ax.axvline(s.r0 * 100, ls="--", lw=0.8, color="gray")
            ax.grid(True, alpha=0.3)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, "magnetic_radial_profiles.png")

    def plot_eigenmode_profile(self, mode):
        """Plot the radial electric-field eigenvector for the selected k_z branch."""
        s = self.solver
        fig, ax = plt.subplots(figsize=(8, 5))
        r_mode = np.asarray(mode.get("r", s.r))
        common_peak = max(max(float(np.max(np.abs(mode[key]))) for key in ("Er", "Ephi", "Ez")),
                          np.finfo(float).tiny)
        for name, label in (("Er", r"$|E_r|$"), ("Ephi", r"$|E_\phi|$"), ("Ez", r"$|E_z|$")):
            vals = np.abs(mode[name])
            ax.plot(r_mode * 100, vals / common_peak, label=label)
        kz = mode["kz"]
        ax.set_xlabel("r, cm")
        ax.set_ylabel("relative amplitude (shared normalization)")
        ax.set_title(f"Radial eigenmode profile: kz={kz.real/100:.4f} cm$^{{-1}}$, "
                     f"Im(kz)={kz.imag/100:.4f} cm$^{{-1}}$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._save(fig, "eigenmode_radial_profile.png")

    def plot_absorbed_power(self):
        s = self.solver
        P = s.compute_absorbed_power()
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(s.z * 100, s.r * 100, P, shading="auto", cmap="hot")
        ax.set_xlabel("z, cm")
        ax.set_ylabel("r, cm")
        ax.set_title("Estimated absorbed power density")
        fig.colorbar(im, ax=ax, label=r"W/m$^3$")
        self._save(fig, "absorbed_power.png")

    def plot_poynting(self):
        s = self.solver
        Sz = s.compute_poynting_z()
        fig, ax = plt.subplots(figsize=(10, 6))
        lim = float(np.max(np.abs(Sz)))
        im = ax.pcolormesh(s.z * 100, s.r * 100, Sz, shading="auto", cmap="RdBu_r",
                           vmin=-lim if lim else None, vmax=lim if lim else None)
        ax.set_xlabel("z, cm")
        ax.set_ylabel("r, cm")
        ax.set_title("Time-averaged axial Poynting flux")
        fig.colorbar(im, ax=ax, label=r"W/m$^2$")
        self._save(fig, "poynting.png")

    def plot_dispersion_map(self):
        s = self.solver
        freqs = s.cfg["wave"]["frequencies"]
        n_e = s.params.n_e
        B0 = s.params.B0
        if getattr(s.background, "magnetic_metadata", {}).get("model") == "plmm_assembly":
            z_ref = float(s.cfg.get("solver", {}).get("dispersion_z_m", s.L / 2))
            iz = int(np.argmin(np.abs(s.z - z_ref)))
            profile = s.background.n_e[:, iz]
            field = np.hypot(s.background.B_r[:, iz], s.background.B_z[:, iz])
            active = profile > max(float(np.max(profile)) * 1e-8, 0.0)
            if np.any(active):
                n_e = float(np.average(profile[active], weights=profile[active]))
                B0 = float(np.average(field[active], weights=profile[active]))
        from plasma import theoretical_kz
        k_vals = [theoretical_kz(2 * np.pi * f, n_e, B0) for f in freqs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.asarray(freqs) / 1e6, np.asarray([abs(k) for k in k_vals]) / 100,
                "o-", lw=2, label="quasi-parallel whistler estimate")
        ax.set_xlabel("f, MHz")
        ax.set_ylabel(r"$k_z$, cm$^{-1}$")
        ax.set_title("Asymptotic whistler dispersion (not full H/TG eigenmodes)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._save(fig, "dispersion_map.png")

    def plot_probe_scan(self, probe_results):
        zs = np.array([r["z"] for r in probe_results])
        rs = np.array([r["r"] for r in probe_results])
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for col, (suffix, label) in enumerate((("r", r"$B_r$"), ("phi", r"$B_\phi$"), ("z", r"$B_z$"))):
            amp = np.array([r[f"absV{suffix}"] for r in probe_results])
            phase = np.array([r[f"phaseB{suffix}"] for r in probe_results])
            sca = axes[0, col].scatter(zs * 100, rs * 100, c=amp, cmap="viridis", s=40)
            axes[0, col].set_title(f"{label}: induced |V|")
            axes[0, col].set_xlabel("z, cm")
            axes[0, col].set_ylabel("r, cm")
            fig.colorbar(sca, ax=axes[0, col])
            scp = axes[1, col].scatter(zs * 100, rs * 100, c=phase, cmap="twilight", s=40)
            axes[1, col].set_title(f"{label}: magnetic phase")
            axes[1, col].set_xlabel("z, cm")
            axes[1, col].set_ylabel("r, cm")
            fig.colorbar(scp, ax=axes[1, col])
        self._save(fig, "probe_scan.png")

    def compare_with_theory(self, k_eigenmode=None):
        s = self.solver
        from plasma import theoretical_kz
        from solver import select_dispersion_mode
        n_reference = float(getattr(s, "dispersion_reference_n_e", s.params.n_e))
        b_reference = float(getattr(s, "dispersion_reference_B0_T", s.params.B0))
        k_theory = theoretical_kz(s.omega, n_reference, b_reference,
                                  m_mode=s.cfg["wave"].get("m_mode", 0))
        k_sim, _ = s.compute_wave_structure()
        quality = float(getattr(s, "kz_fit_quality", 0.0))
        if k_eigenmode is None:
            modes = s.compute_dispersion_modes()
            k_eigenmode = select_dispersion_mode(modes, abs(k_theory))
        eigen_resolved = bool(k_eigenmode.get("radial_grid_resolved", False)) if isinstance(k_eigenmode, dict) else False
        eigen_roughness = float(k_eigenmode.get("radial_grid_oscillation", float("nan"))) if isinstance(k_eigenmode, dict) else float("nan")
        k_eigenmode = k_eigenmode["kz"] if isinstance(k_eigenmode, dict) else complex(k_eigenmode)

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["Whistler\nasymptote", "Radial eigenmode\n(mesh-screened)" if eigen_resolved else "Radial eigenmode\n(UNVERIFIED)"]
        values = [abs(k_theory) / 100, abs(k_eigenmode) / 100]
        colors = ["steelblue", "seagreen" if eigen_resolved else "gray"]
        if np.isfinite(k_sim):
            labels.append("Driven-field\nphase fit")
            values.append(abs(k_sim) / 100)
            colors.append("coral")
        ax.bar(labels, values, color=colors)
        ax.set_ylabel(r"$|k_z|$, cm$^{-1}$")
        ax.set_title("Asymptotic theory, radial eigenmode and source-driven field")
        ax.grid(True, alpha=0.3, axis="y")
        self._save(fig, "theory_vs_sim.png")

        err_eigen = abs(abs(k_eigenmode.real) - abs(k_theory)) / max(abs(k_theory), 1e-30) * 100
        err_field = (
            abs(abs(k_sim) - abs(k_theory)) / max(abs(k_theory), 1e-30) * 100
            if np.isfinite(k_sim) else float("nan")
        )
        print(f"[whistler asymptote] k_z = {abs(k_theory)/100:.5f} 1/cm")
        print(f"[radial eigenmode]   k_z = {k_eigenmode.real/100:.5f} 1/cm, "
              f"Im(k_z) = {k_eigenmode.imag/100:.5f} 1/cm")
        if not eigen_resolved:
            print(f"[mesh warning] Unverified radial structure (roughness={eigen_roughness:.3f}); "
                  "do not treat the selected eigenvalue as a physical result")
        if np.isfinite(k_sim):
            print(f"[driven-field fit]   k_z = {abs(k_sim)/100:.5f} 1/cm, phase-fit R^2={quality:.3f}")
        else:
            print(f"[driven-field fit]   rejected: {getattr(s, 'kz_fit_reason', 'insufficient evidence')}")
        print(f"[eigenmode delta vs asymptote] {err_eigen:.2f}% (comparison indicator, not a guaranteed error bound)")
        if np.isfinite(err_field):
            print(f"[field-fit delta vs asymptote] {err_field:.2f}% (source-driven field may contain several modes)")
        f_lh = np.sqrt(s.stix.omega_ci * s.stix.omega_ce) / (2 * np.pi)
        if abs(s.omega / (2 * np.pi) - f_lh) / f_lh < 0.2:
            print(f"[physics warning] f={s.omega/(2*np.pi)/1e6:.3f} MHz is near f_LH≈{f_lh/1e6:.3f} MHz; "
                  "the electron-only sqrt(f) asymptote is not quantitatively reliable here.")
        print("[NOTE] The radial eigenmode solves this code's discretized cold-plasma model with radial PML; it is not a full Chen/Arnush H-TG benchmark until boundary/model convergence is checked.")
        return k_theory, k_sim, err_field
