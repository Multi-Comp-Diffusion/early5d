import numpy as np
import pytest

from early5d import kinetics


def test_dispersion_relation_binary_analytic_reduction():
    """For a single independent component, R(k) = -M k^2 (H + kappa k^2) exactly (Eq. 7)."""
    H, M, kappa_value = -4.0, 2.0, 0.1
    Hg = np.array([[H]])
    M_matrix = np.array([[M]])
    kappa = np.array([[kappa_value]])

    result = kinetics.solve_dispersion_relation(Hg, M_matrix, kappa, n_k=500)
    k = result["wavenumbers"]
    expected = -M * k ** 2 * (H + kappa_value * k ** 2)

    assert np.allclose(result["growth_rates"], expected, rtol=1e-10)
    assert result["tau"] == pytest.approx(1.0 / np.max(expected), rel=1e-3)


def test_stable_composition_returns_infinite_ctld():
    Hg = np.diag([1.0, 2.0])
    M = np.eye(2)
    kappa = np.eye(2)
    result = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=100)
    assert result["wavenumbers"] is None
    assert result["tau"] == np.inf


def test_ctld_is_positive_for_unstable_composition():
    Hg = np.diag([-3.0, 1.0])
    M = np.array([[2.0, 0.2], [0.2, 1.0]])
    kappa = 0.05 * np.eye(2)
    tau = kinetics.ctld(Hg, M, kappa, n_k=500)
    assert tau > 0
    assert np.isfinite(tau)


def test_thermo_kinetic_operator_matches_definition():
    Hg = np.array([[-1.0, 0.2], [0.2, 0.5]])
    M = np.array([[1.0, 0.1], [0.1, 2.0]])
    kappa = 0.01 * np.eye(2)
    k = 1.5e8
    K = kinetics.thermo_kinetic_operator(Hg, M, kappa, k)
    assert np.allclose(K, M @ (Hg + kappa * k ** 2))
