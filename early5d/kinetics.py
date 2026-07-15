"""Thermo-kinetic growth operator, dispersion relation, and CTLD (Eqs. 7, 9, 12-13).

    K(k) = M (Hg + kappa k^2)          thermo-kinetic operator, Eq. (9)
    R(k) = -k^2 mu0(k)                 growth rate at the most unstable eigenvalue, Eq. (12)
    tau  = 1 / max_k{R(k)}             characteristic time of linear decomposition, Eq. (13)
"""

import numpy as np

from early5d.hessian import wavenumber_cutoff


def thermo_kinetic_operator(Hg, M, kappa, wavenumber):
    """``K(k) = M (Hg + kappa k^2)`` (Eq. 9)."""
    return M @ (Hg + kappa * wavenumber ** 2)


def solve_dispersion_relation(Hg, M, kappa, n_k=1000):
    """Solve the thermo-kinetic eigenproblem across the unstable wavenumber
    spectrum (Eqs. 7, 9, 12-13).

    Returns a dict with the full dispersion relation (``wavenumbers``,
    ``growth_rates``) and the fastest-growing mode (``k_max``, ``R_max``,
    ``mu0_max``, ``growth_mode``, ``tau``). If the composition is
    thermodynamically stable (no unstable wavenumber), the arrays are
    ``None`` and ``tau`` is ``inf``.
    """
    n = Hg.shape[0]
    k_cut = wavenumber_cutoff(Hg, kappa)

    stable_result = {
        "wavenumbers": None, "growth_rates": None,
        "k_max": np.nan, "R_max": np.nan, "mu0_max": np.nan,
        "growth_mode": np.full(n, np.nan), "tau": np.inf,
    }
    if k_cut == 0.0:
        return stable_result

    wavenumbers = np.linspace(0.001 * k_cut, 0.999 * k_cut, n_k)
    growth_rates = np.empty(n_k)

    R_max, k_max, mu0_max, mode_max = -np.inf, np.nan, np.nan, np.full(n, np.nan)
    for i, k in enumerate(wavenumbers):
        eigenvalues, eigenvectors = np.linalg.eig(thermo_kinetic_operator(Hg, M, kappa, k))
        order = np.argsort(eigenvalues.real)
        mu0 = eigenvalues[order[0]].real
        growth_rates[i] = -(k ** 2) * mu0
        if growth_rates[i] > R_max:
            R_max, k_max, mu0_max = growth_rates[i], k, mu0
            mode_max = eigenvectors[:, order[0]].real

    if R_max <= 0:
        return stable_result

    return {
        "wavenumbers": wavenumbers, "growth_rates": growth_rates,
        "k_max": k_max, "R_max": R_max, "mu0_max": mu0_max,
        "growth_mode": mode_max, "tau": 1.0 / R_max,
    }


def ctld(Hg, M, kappa, n_k=1000):
    """Characteristic time of linear decomposition, Eq. (13) (shortcut for
    ``solve_dispersion_relation(...)['tau']``)."""
    return solve_dispersion_relation(Hg, M, kappa, n_k)["tau"]
