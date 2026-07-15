"""Effective (single-mode) projection of the thermo-kinetic operator (Eqs. 14-17).

Projecting the growth eigenproblem onto the thermo mode alone neglects the
kinetic rotation of the eigenmodes, but reduces the problem to a scalar,
pseudo-binary dispersion relation -- cheap enough for high-throughput
screening of an entire composition space. The projection uses the eigenvector
of the *untransformed* Hessian (:func:`early5d.hessian.raw_thermo_mode`), not
the gauge-invariant :func:`early5d.hessian.thermo_mode` (Eq. 8) used
elsewhere -- see that function's docstring for why. ``effective_ctld`` is a
provable lower bound on the full eigenproblem-based CTLD
(:func:`early5d.kinetics.ctld`): ``tau_eff <= tau``.
"""

import numpy as np

from early5d import hessian
from early5d.linalg import effective_projection


def effective_mobility(M, thermo_mode):
    """``M_eff = v0^T M v0`` (Eq. 15): the mobility matrix's Rayleigh quotient
    along the thermo mode."""
    return effective_projection(thermo_mode, M)


def effective_gradient_energy(M, kappa, thermo_mode, M_eff=None):
    """``kappa_eff = v0^T M kappa v0 / M_eff`` (Eq. 16)."""
    if M_eff is None:
        M_eff = effective_mobility(M, thermo_mode)
    return float(thermo_mode @ M @ kappa @ thermo_mode) / M_eff


def effective_growth_rate(M_eff, xi0, kappa_eff, wavenumbers):
    """``R(k) = -M_eff k^2 (xi0 + kappa_eff k^2)`` (Eq. 14)."""
    k2 = np.asarray(wavenumbers) ** 2
    return -M_eff * k2 * (xi0 + kappa_eff * k2)


def effective_ctld(xi0, M_eff, kappa_eff):
    """``tau_eff = 4 kappa_eff / (M_eff xi0^2)`` (Eq. 17)."""
    return 4.0 * kappa_eff / (M_eff * xi0 ** 2)


def effective_wavenumbers(xi0, kappa_eff, n_k=1000):
    """Critical and fastest-growing wavenumbers of the effective dispersion
    relation, and a sampling grid between 0 and the critical wavenumber."""
    k_critical = np.sqrt(np.abs(xi0) / kappa_eff)
    k_fastest = np.sqrt(np.abs(xi0) / (2 * kappa_eff))
    wavenumbers = np.linspace(0.01 * k_critical, 0.99 * k_critical, n_k)
    return wavenumbers, k_critical, k_fastest


def solve_effective_dispersion(Hg, M, kappa, n_k=1000):
    """Single-mode projection of the growth eigenproblem (Eqs. 14-17).

    Projects onto the eigenvector of the untransformed Hessian
    (:func:`early5d.hessian.raw_thermo_mode`), which is what makes the
    algebraic reduction ``Hg v0 = xi0 v0`` exact. Returns a dict with
    ``M_eff``, ``kappa_eff``, ``tau_eff``, the effective dispersion relation
    (``wavenumbers``, ``growth_rates``), and ``k_critical``/``k_fastest``.
    """
    thermo_mode, xi0 = hessian.raw_thermo_mode(Hg)

    M_eff = effective_mobility(M, thermo_mode)
    kappa_eff = effective_gradient_energy(M, kappa, thermo_mode, M_eff)
    wavenumbers, k_critical, k_fastest = effective_wavenumbers(xi0, kappa_eff, n_k)
    growth_rates = effective_growth_rate(M_eff, xi0, kappa_eff, wavenumbers)

    return {
        "M_eff": M_eff, "kappa_eff": kappa_eff,
        "wavenumbers": wavenumbers, "growth_rates": growth_rates,
        "k_critical": k_critical, "k_fastest": k_fastest,
        "tau_eff": effective_ctld(xi0, M_eff, kappa_eff),
        "thermo_mode": thermo_mode, "xi0": xi0,
    }
