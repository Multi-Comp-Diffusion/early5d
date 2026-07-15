import numpy as np
import pytest

from early5d import misalignment


def test_angle_between_identical_vectors_is_zero():
    a = np.array([1.0, 2.0, -1.0, 0.5])
    assert misalignment.angle_between(a, a) == pytest.approx(0.0, abs=1e-8)


def test_angle_between_opposite_vectors_is_zero_when_acute():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    assert misalignment.angle_between(a, -a, acute=True) == pytest.approx(0.0, abs=1e-8)


def test_angle_between_opposite_vectors_is_180_when_not_acute():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    assert misalignment.angle_between(a, -a, acute=False) == pytest.approx(180.0)


def test_angle_between_orthogonal_vectors_is_90():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert misalignment.angle_between(a, b) == pytest.approx(90.0)


def test_angle_between_is_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=4), rng.normal(size=4)
    assert misalignment.angle_between(a, b) == pytest.approx(misalignment.angle_between(b, a))


def test_acute_angle_bounds():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.normal(size=4), rng.normal(size=4)
        angle = misalignment.angle_between(a, b, acute=True)
        assert 0.0 <= angle <= 90.0 + 1e-8


def test_identity_metric_matches_euclidean():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=3), rng.normal(size=3)
    euclidean = misalignment.angle_between(a, b)
    weighted = misalignment.angle_between(a, b, metric=np.eye(3))
    assert euclidean == pytest.approx(weighted)


def test_gram_schmidt_plane_is_orthonormal():
    a = np.array([1.0, 0.3, -0.2, 0.1])
    b = np.array([0.1, 1.0, 0.2, -0.3])
    e1, e2 = misalignment.gram_schmidt_plane(a, b)
    assert np.linalg.norm(e1) == pytest.approx(1.0)
    assert np.linalg.norm(e2) == pytest.approx(1.0)
    assert e1 @ e2 == pytest.approx(0.0, abs=1e-8)
