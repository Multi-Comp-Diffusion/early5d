"""Sensitivity of the effective (single-mode) quantities to the choice of
reference (dependent) species (Sec. 4 discussion).

``M_eff``, ``kappa_eff``, and ``tau_eff`` (Eqs. 15-17) project onto ``v0``,
the eigenvector of the *untransformed* Hessian
(:func:`early5d.hessian.raw_thermo_mode`) -- which, unlike the gauge-invariant
:func:`early5d.hessian.thermo_mode` (Eq. 8), depends on which species is
treated as the implicit/dependent one. This module recomputes those
quantities under every alternate reference-species choice, to quantify that
dependence.

Reference-species swap: switching the dependent species from ``r`` to ``s``
in the independent-component list ``E = [e_1, ..., e_{N-1}]`` gives new
independent components ``E' = [r, *E \\ {s}]``, with

    mu'_i = mu_i - mu_j   (i != s, r; j = index of s in E)
    mu'_r = -mu_j

This is linear in ``mu`` (``mu' = A @ mu``, for an integer matrix ``A``
depending only on which species becomes the new reference). Since
``Hg = d(mu)/dX`` and the composition variables transform via the same
substitution (``X_r = 1 - sum(X_E)`` <-> ``X_s = 1 - sum(X_E')``), the
Hessian transforms as the congruence ``Hg' = A @ Hg @ A.T``.

``kappa`` is *not* transformed this way -- it is redefined as the same
isotropic ``kappa_scalar * I`` directly in the new coordinates, matching the
manuscript's actual convention (an isotropic tensor in one oblique frame is
not isotropic when consistently transformed into another, which is also why
even the full eigenproblem CTLD is not perfectly gauge-invariant here).

``M`` needs no such derivation: :func:`early5d.mobility.mobility_matrix`
already takes the full-species mobilities/composition and only needs the new
reference species moved to index 0 -- exact, no transform required.
"""

import numpy as np

from early5d import calphad, effective, hessian, kinetics, mobility


def reference_swap_matrix(old_reference, old_elements, new_reference):
    """``A`` (n x n) with ``mu' = A @ mu``; returns ``(A, new_elements)``,
    where ``new_elements = [old_reference, *old_elements minus new_reference]``
    is the resulting independent-component order."""
    j = old_elements.index(new_reference)
    new_elements = [old_reference] + [e for e in old_elements if e != new_reference]
    n = len(old_elements)
    A = np.zeros((n, n))
    for i, species in enumerate(new_elements):
        if species == old_reference:
            A[i, j] = -1.0
        else:
            k = old_elements.index(species)
            A[i, k] = 1.0
            A[i, j] += -1.0
    return A, new_elements


def swapped_hessian(Hg, old_reference, old_elements, new_reference):
    """``Hg`` re-expressed under ``new_reference`` instead of ``old_reference``."""
    A, new_elements = reference_swap_matrix(old_reference, old_elements, new_reference)
    return A @ Hg @ A.T, new_elements


def swapped_mobility_matrix(df, composition, elements, full_species, index, decimals, new_reference):
    """``M`` re-expressed under ``new_reference``, by exact re-indexing (no transform needed)."""
    mobilities_full = calphad.atomic_mobilities(df, composition, elements, index, decimals)
    composition_full = calphad.full_composition(composition)
    s = full_species.index(new_reference)
    order = [s] + [i for i in range(len(full_species)) if i != s]
    return mobility.mobility_matrix(mobilities_full[order], composition_full[order])


def reference_sensitivity(df, composition, kappa_scalar, elements, full_species,
                           grid_spacing, index=None, decimals=4, n_k=2000):
    """Effective (Eqs. 15-17) and full eigenproblem (Eq. 13) dispersion
    quantities, recomputed under every possible choice of reference species.

    ``full_species`` is ``[old_reference, *elements]`` (see
    :func:`early5d.calphad.full_species`); ``kappa_scalar`` is redefined as
    the isotropic ``kappa_scalar * I`` fresh in every candidate frame,
    matching the manuscript's convention (see the module docstring).

    Returns a dict keyed by species name (the original reference and every
    element), each entry ``{"M_eff", "kappa_eff", "tau_eff", "tau"}``. The
    entry for ``full_species[0]`` (the original reference) is the quantity
    actually reported in the paper; the others quantify its gauge-dependence.
    """
    elements = list(elements)
    old_reference = full_species[0]
    if index is None:
        index, decimals = calphad.build_index(df, elements, decimals)

    Hg = hessian.hessian_at(df, composition, grid_spacing, elements, index, decimals)
    mobilities = calphad.atomic_mobilities(df, composition, elements, index, decimals)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))
    kappa = kappa_scalar * np.eye(len(elements))

    def summarize(Hg_frame, M_frame, kappa_frame):
        eff = effective.solve_effective_dispersion(Hg_frame, M_frame, kappa_frame, n_k=n_k)
        tau = kinetics.ctld(Hg_frame, M_frame, kappa_frame, n_k=n_k)
        return {"M_eff": eff["M_eff"], "kappa_eff": eff["kappa_eff"], "tau_eff": eff["tau_eff"], "tau": tau}

    results = {old_reference: summarize(Hg, M, kappa)}
    for new_ref in elements:
        Hg_s, _ = swapped_hessian(Hg, old_reference, elements, new_ref)
        M_s = swapped_mobility_matrix(df, composition, elements, full_species, index, decimals, new_ref)
        kappa_s = kappa_scalar * np.eye(len(elements))
        results[new_ref] = summarize(Hg_s, M_s, kappa_s)

    return results
