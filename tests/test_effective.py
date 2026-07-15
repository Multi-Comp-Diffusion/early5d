import numpy as np
import pytest

from early5d import effective, kinetics


def test_effective_matches_full_ctld_for_isotropic_mobility():
    """With isotropic mobility, M and Hg commute, so the growth mode equals the
    thermo mode exactly (no kinetic rotation) and tau_eff == tau (Eq. 17 note)."""
    Hg = np.diag([-2.0, 5.0])
    M = 3.0 * np.eye(2)
    kappa = 1e-2 * np.eye(2)

    full = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=3000)
    eff = effective.solve_effective_dispersion(Hg, M, kappa, n_k=3000)

    assert eff["tau_eff"] == pytest.approx(full["tau"], rel=1e-3)


def test_effective_ctld_is_lower_bound_for_anisotropic_mobility():
    """tau_eff <= tau in general (the paper's central inequality): the single-mode
    projection neglects kinetic rotation and cannot predict a *slower* decomposition
    than the true (fastest-growing) eigenmode."""
    Hg = np.array([[-2.0, 0.6], [0.6, 1.0]])
    M = np.array([[1.0, 0.5], [0.5, 3.0]])
    kappa = 0.01 * np.eye(2)

    full = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=3000)
    eff = effective.solve_effective_dispersion(Hg, M, kappa, n_k=3000)

    assert eff["tau_eff"] <= full["tau"] + 1e-9


def test_effective_growth_rate_matches_definition():
    M_eff, xi0, kappa_eff = 2.0, -3.0, 0.1
    k = np.linspace(0, 1, 10)
    R = effective.effective_growth_rate(M_eff, xi0, kappa_eff, k)
    assert np.allclose(R, -M_eff * k ** 2 * (xi0 + kappa_eff * k ** 2))


def test_effective_ctld_matches_definition():
    M_eff, xi0, kappa_eff = 2.0, -3.0, 0.1
    tau_eff = effective.effective_ctld(xi0, M_eff, kappa_eff)
    assert tau_eff == pytest.approx(4 * kappa_eff / (M_eff * xi0 ** 2))
