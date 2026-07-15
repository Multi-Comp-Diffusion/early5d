"""Small shared linear-algebra helpers used throughout the package."""

import numpy as np


def effective_projection(vector, matrix):
    """``v^T A v`` -- the Rayleigh-quotient numerator used for effective
    mobility/gradient-energy projections (Eqs. 15-16)."""
    return float(vector @ matrix @ vector)


def is_symmetric(matrix, rtol=1e-10, atol=1e-12):
    """``matrix == matrix.T`` within ``np.allclose``'s ``rtol``/``atol``."""
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def is_positive_semidefinite(matrix, tol=1e-12):
    """True if ``matrix`` is symmetric and has no eigenvalue below ``-tol``."""
    if not is_symmetric(matrix):
        return False
    return bool(np.linalg.eigvalsh(matrix).min() >= -tol)


def to_full_mode(vector):
    """Prepend the implicit reference-species component ``-sum(vector)`` to an
    independent-component eigenvector, so all species appear explicitly. Valid
    for eigenvectors (which represent perturbations, unlike compositions):
    since every composition satisfies ``sum(X) = 1``, a perturbation must
    satisfy ``sum(delta X) = 0``."""
    vector = np.asarray(vector, dtype=float)
    return np.concatenate([[-vector.sum()], vector])


def align_mode(vector):
    """Canonical sign convention: flip so the larger-magnitude excursion is positive."""
    vector = np.asarray(vector, dtype=float)
    return -vector if abs(vector.min()) > abs(vector.max()) else vector


def normalized_full_mode(vector):
    """Full-species, canonically-oriented, unit-normalized mode, for
    side-by-side bar-shard comparison across models (Fig. 4)."""
    full = align_mode(to_full_mode(vector))
    return full / np.linalg.norm(full)
