import numpy as np
import pytest

from early5d import linalg, mobility


def test_mobility_matrix_is_symmetric_positive_semidefinite():
    mobilities = np.array([1.0, 2.0, 0.5, 3.0, 1.5])  # [Fe, Mn, Ni, Co, Cu]
    composition = np.array([0.4, 0.1, 0.2, 0.2, 0.1])
    M = mobility.mobility_matrix(mobilities, composition)
    assert linalg.is_symmetric(M)
    assert linalg.is_positive_semidefinite(M)


def test_mobility_matrix_map_matches_single_point():
    mobilities = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    composition = np.array([0.4, 0.1, 0.2, 0.2, 0.1])
    single = mobility.mobility_matrix(mobilities, composition)
    batched = mobility.mobility_matrix_map(mobilities[None, :], composition[None, :])
    assert np.allclose(single, batched[0])


def test_dominant_mode_has_largest_eigenvalue():
    M = np.array([[2.0, 0.3], [0.3, 5.0]])
    mode, value = mobility.dominant_mode(M)
    eigenvalues = np.linalg.eigvalsh(M)
    assert value == pytest.approx(eigenvalues.max())
    assert np.allclose(M @ mode, value * mode)
