"""Onsager mobility matrix from CALPHAD atomic mobilities (Eq. 4).

M_ij = sum_k (delta_jk - X_j)(delta_ik - X_i) X_k Mob_k

where k runs over all species (including the reference) and i, j run over the
independent (non-reference) components.
"""

import numpy as np


def mobility_matrix(mobilities, composition):
    """Onsager mobility matrix at a single composition (Eq. 4).

    ``mobilities`` and ``composition`` are both full-species arrays ordered
    ``[reference, *elements]`` (see :func:`early5d.calphad.full_species` /
    :func:`early5d.calphad.full_composition`). Returns the ``(n-1, n-1)``
    matrix for the ``n-1`` independent components.
    """
    mobilities = np.asarray(mobilities, dtype=float)
    composition = np.asarray(composition, dtype=float)
    n_species = len(composition)
    n_independent = n_species - 1

    kronecker = np.eye(n_independent, n_species, k=1)   # kronecker[i, k] = delta(k, i+1)
    x_independent = composition[1:]
    delta_minus_x = kronecker - x_independent[:, None]  # delta_ik - X_i, shape (n_independent, n_species)
    weighted_mobility = composition * mobilities        # X_k * Mob_k

    return np.einsum("ik,jk,k->ij", delta_minus_x, delta_minus_x, weighted_mobility)


def mobility_matrix_map(mobilities, compositions):
    """Vectorized :func:`mobility_matrix` over many grid points.

    ``mobilities`` and ``compositions`` have shape ``(n_points, n_species)``;
    returns an array of shape ``(n_points, n-1, n-1)``.
    """
    mobilities = np.asarray(mobilities, dtype=float)
    compositions = np.asarray(compositions, dtype=float)
    n_species = compositions.shape[1]
    n_independent = n_species - 1

    kronecker = np.eye(n_independent, n_species, k=1)
    x_independent = compositions[:, 1:]
    delta_minus_x = kronecker[None, :, :] - x_independent[:, :, None]  # (P, n_independent, n_species)
    weighted_mobility = compositions * mobilities                       # (P, n_species)

    return np.einsum("pik,pjk,pk->pij", delta_minus_x, delta_minus_x, weighted_mobility)


def dominant_mode(M):
    """Eigenvector of the mobility matrix with the largest eigenvalue -- the
    direction of fastest atomic transport (the "mobility mode" of Fig. 4)."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvectors[:, -1], eigenvalues[-1]


def eigvalsh_descending(M):
    """The mobility matrix's full eigenvalue spectrum, descending (largest
    first, matching :func:`dominant_mode`'s own eigenvalue as element 0)."""
    return np.linalg.eigvalsh(M)[::-1]


def eigenvalue_gap(M):
    """Relative gap ``(lambda_1 - lambda_2) / lambda_1`` between the mobility
    matrix's two largest eigenvalues -- how well-separated (and hence how
    robustly defined) :func:`dominant_mode` is from the next-fastest
    transport direction. A small gap means the "dominant" mode is close to
    degenerate with another direction and is thus more sensitive to small
    perturbations of the composition or mobility data."""
    lambda_1, lambda_2 = eigvalsh_descending(M)[:2]
    return float((lambda_1 - lambda_2) / lambda_1)
