"""Gibbs free-energy Hessian and its gauge-invariant (Cartesian) representation.

Implements Eqs. (5), (8), and (10)-(11) of the manuscript: a finite-difference
Hessian of the chemical potential on the CALPHAD composition grid, the
orthogonalization that removes the gauge (basis) ambiguity of the oblique
Gibbs composition space, and the wavenumber cutoff of the unstable spectrum.
"""

import numpy as np
import scipy.linalg as la

from early5d import calphad

# Orthogonalization matrix O (Eq. 8) mapping the oblique 4-component Gibbs
# simplex basis (for the 5-component Fe-Mn-Ni-Co-Cu system studied here) to a
# Cartesian composition space, following de Fontaine's construction (refs.
# 28, 29, 48 of the manuscript). Valid for exactly 4 independent components;
# a differently-sized system needs its own orthogonalization matrix.
ORTHOGONALIZATION_MATRIX = np.array([
    [-1, -1 / np.sqrt(3), -1 / np.sqrt(6), -1 / np.sqrt(10)],
    [1, -1 / np.sqrt(3), -1 / np.sqrt(6), -1 / np.sqrt(10)],
    [0, 2 / np.sqrt(3), -1 / np.sqrt(6), -1 / np.sqrt(10)],
    [0, 0, np.sqrt(3 / 2), -1 / np.sqrt(10)],
])


def hessian_at(df, composition, grid_spacing, elements=None, index=None, decimals=4):
    """Finite-difference Hessian ``Hg`` of the Gibbs free energy at ``composition`` (Eq. 5).

    Central differences of the chemical-potential vector are used along each
    independent-component axis; one-sided differences are used at the edge of
    the composition simplex where one neighbor would fall outside it.
    """
    elements = elements if elements is not None else calphad.composition_columns(df)
    if index is None:
        index, decimals = calphad.build_index(df, elements, decimals)
    n = len(elements)
    X = np.asarray(composition, dtype=float)
    mu0 = calphad.chemical_potential(df, X, elements, index, decimals)

    H = np.empty((n, n))
    for j in range(n):
        step = np.zeros(n)
        step[j] = grid_spacing
        X_plus, X_minus = X + step, X - step
        can_plus = X_plus.sum() <= 1.0 + 1e-9 and np.all(X_plus >= -1e-9)
        can_minus = np.all(X_minus >= -1e-9)

        if can_plus and can_minus:
            mu_plus = calphad.chemical_potential(df, X_plus, elements, index, decimals)
            mu_minus = calphad.chemical_potential(df, X_minus, elements, index, decimals)
            H[j, :] = (mu_plus - mu_minus) / (2 * grid_spacing)
        elif can_plus:
            mu_plus = calphad.chemical_potential(df, X_plus, elements, index, decimals)
            H[j, :] = (mu_plus - mu0) / grid_spacing
        elif can_minus:
            mu_minus = calphad.chemical_potential(df, X_minus, elements, index, decimals)
            H[j, :] = (mu0 - mu_minus) / grid_spacing
        else:
            raise ValueError(f"composition {X} has no valid neighbor along axis {j}")

    return 0.5 * (H + H.T)


def orthogonalize(matrix, O=ORTHOGONALIZATION_MATRIX):
    """Gauge-invariant transform ``O^T matrix O`` (Eq. 8).

    ``O``'s default is only valid for a 4-independent-component (5-species)
    system; passing a differently-sized ``matrix`` without also supplying the
    matching ``O`` for that system raises ``ValueError`` rather than silently
    orthogonalizing against a mismatched (and physically meaningless) basis.
    """
    n = matrix.shape[0]
    if O.shape[0] != n:
        raise ValueError(
            f"orthogonalization matrix O has dimension {O.shape[0]}, but matrix has "
            f"dimension {n}; the default O is only valid for a 4-independent-component "
            f"system -- pass your own O for a differently-sized one"
        )
    return O.T @ matrix @ O


def eigen(matrix):
    """Ascending eigenvalues/eigenvectors of a symmetric matrix."""
    return np.linalg.eigh(matrix)


def thermo_mode(Hg, O=ORTHOGONALIZATION_MATRIX):
    """Thermodynamically unstable mode ``v0`` (Eq. 8): the eigenvector of the
    gauge-invariant Hessian belonging to its most negative eigenvalue,
    transformed back to the original (oblique) composition-space basis."""
    eigenvalues, eigenvectors = eigen(orthogonalize(Hg, O))
    v0_hat = eigenvectors[:, 0]  # ascending order -> most negative first
    n = Hg.shape[0]
    return O[:n, :n] @ v0_hat, eigenvalues[0]


def raw_thermo_mode(Hg):
    """Smallest eigenvalue/eigenvector of the *raw* (non-orthogonalized) Hessian.

    This is the vector actually used by :mod:`early5d.effective`'s single-mode
    projection (Eqs. 14-17), distinct from the gauge-invariant :func:`thermo_mode`
    (Eq. 8) used for direction comparisons elsewhere (e.g. the bar-shard and
    misalignment-angle analyses). The distinction matters because the
    single-mode projection needs ``Hg v0 = xi0 v0`` to hold *exactly* so that
    ``Hg`` cancels out of the projection algebraically -- true for this raw
    eigenvector by construction, but not for :func:`thermo_mode`'s ``O``-transferred
    vector. The trade-off: unlike the full eigenproblem-based CTLD, the
    resulting effective quantities depend weakly (single-digit percent, for
    the compositions studied here) on the arbitrary choice of dependent
    (reference) species -- immaterial in practice since a single CALPHAD
    export uses one fixed reference species throughout.
    """
    eigenvalues, eigenvectors = eigen(Hg)
    return eigenvectors[:, 0], eigenvalues[0]


def is_spinodal(Hg, O=ORTHOGONALIZATION_MATRIX):
    """True if the gauge-invariant Hessian has at least one negative eigenvalue."""
    eigenvalues = np.linalg.eigvalsh(orthogonalize(Hg, O))
    return bool(eigenvalues[0] < 0)


def wavenumber_cutoff(Hg, kappa):
    """Upper bound ``k_cut`` of the unstable wavenumber spectrum (Eqs. 10-11):
    ``k_cut = sqrt(-gamma0)``, with ``gamma0`` the smallest eigenvalue of the
    generalized eigenproblem ``Hg x = gamma kappa x``. Returns 0.0 if the
    composition is thermodynamically stable."""
    if np.all(kappa == 0):
        eigenvalues = np.linalg.eigvalsh(Hg)
    else:
        eigenvalues = la.eigvalsh(Hg, kappa)
    gamma0 = eigenvalues[0]
    return float(np.sqrt(-gamma0)) if gamma0 < 0 else 0.0
