"""Composition-simplex scans producing ternary cross-sections for every pair
of species held at a fixed fraction (Figs. 1/2, generalized).

The manuscript's Fig. 1b/2b panel (Fe = Co = 20 at %, Mn/Ni/Cu varying) is one
of ``C(5, 2) = 10`` such cross-sections obtainable from a 5-species system by
holding a different pair of species fixed. :func:`scan_composition_grid`
computes a chosen early-stage parameter once over the whole grid (restricted
to the spinodal region); :func:`pairwise_ternary_slices` then extracts the
ternary cross-section for every (or a chosen subset of) pair of species held
at a common fraction (default 20 at % each, matching the manuscript's
convention), skipping pairs with too few grid points to plot.
"""

import itertools

import numpy as np

from early5d import calphad, effective, hessian, kinetics, mobility

FIELDS = ("tau", "tau_eff", "xi0", "k_max")


def scan_composition_grid(df, kappa, elements=None, grid_spacing=0.05, index=None,
                           decimals=4, field="tau", n_k=30):
    """Evaluate ``field`` at every grid point inside the spinodal region.

    ``field``: ``"tau"`` (full eigenproblem CTLD), ``"tau_eff"`` (cheap
    single-mode projection), ``"xi0"`` (thermodynamic driving force), or
    ``"k_max"`` (eigenproblem-based fastest-growing wavenumber). Points at the
    grid's edge (missing a finite-difference neighbor) or outside the
    spinodal region are skipped.

    Returns ``(full_compositions, values)``: ``full_compositions`` has shape
    ``(P, n_species)`` in :func:`early5d.calphad.full_species` order
    (``[reference, *elements]``), ``values`` has shape ``(P,)``.
    """
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}, got {field!r}")

    elements = elements if elements is not None else calphad.composition_columns(df)
    if index is None:
        index, decimals = calphad.build_index(df, elements, decimals)
    compositions = df[[f"x({e})" for e in elements]].to_numpy(dtype=float)

    valid_compositions, values = [], []
    for composition in compositions:
        try:
            Hg = hessian.hessian_at(df, composition, grid_spacing, elements, index, decimals)
        except (KeyError, ValueError):
            continue  # composition sits at the extracted grid's edge; skip

        if not hessian.is_spinodal(Hg):
            continue

        if field == "xi0":
            value = hessian.thermo_mode(Hg)[1]
        else:
            mobilities = calphad.atomic_mobilities(df, composition, elements, index, decimals)
            M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))
            if field == "tau":
                value = kinetics.ctld(Hg, M, kappa, n_k=n_k)
            elif field == "tau_eff":
                value = effective.solve_effective_dispersion(Hg, M, kappa, n_k=n_k)["tau_eff"]
            else:  # k_max
                value = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=n_k)["k_max"]

        if not np.isfinite(value):
            # numerically degenerate point: is_spinodal (checked from Hg alone) doesn't
            # guarantee R_max > 0 once the composition-dependent mobility M is folded in,
            # so kinetics.ctld/solve_dispersion_relation can still fall back to its
            # "stable" (inf tau / nan k_max) branch here; not a usable field value
            continue

        valid_compositions.append(composition)
        values.append(value)

    if not valid_compositions:
        return np.empty((0, len(elements) + 1)), np.empty(0)

    full_compositions = np.array([calphad.full_composition(c) for c in valid_compositions])
    return full_compositions, np.array(values, dtype=float)


def pairwise_ternary_slices(full_compositions, values, species, fixed_fraction=0.20,
                             pairs=None, min_points=3, tol=1e-6):
    """Ternary cross-section of ``values`` for every pair of species held at
    ``fixed_fraction`` (default: all ``C(len(species), 2)`` pairs, matching
    the manuscript's 20:20 at % convention).

    ``species``: the full species order matching ``full_compositions``'s
    columns (:func:`early5d.calphad.full_species`). ``pairs`` restricts to a
    given subset of 2-tuples of species names (default: every pair).

    Returns a dict keyed by ``(species_a, species_b)`` (species-list order per
    pair), each entry ``{"ternary_elements": (e1, e2, e3), "ternary_compositions":
    (P, 3), "values": (P,)}`` for the three remaining (varying) species, in
    their relative order within ``species`` -- pairs with fewer than
    ``min_points`` grid points on that slice are omitted entirely.
    """
    species = list(species)
    pairs = list(pairs) if pairs is not None else list(itertools.combinations(species, 2))

    results = {}
    for pair in pairs:
        i, j = species.index(pair[0]), species.index(pair[1])
        on_slice = (np.isclose(full_compositions[:, i], fixed_fraction, atol=tol) &
                    np.isclose(full_compositions[:, j], fixed_fraction, atol=tol))
        if on_slice.sum() < min_points:
            continue

        remaining = [k for k in range(len(species)) if k not in (i, j)]
        ternary_elements = tuple(species[k] for k in remaining)
        ternary_compositions = full_compositions[np.ix_(on_slice, remaining)]
        ternary_compositions = ternary_compositions / ternary_compositions.sum(axis=1, keepdims=True)

        results[pair] = {
            "ternary_elements": ternary_elements,
            "ternary_compositions": ternary_compositions,
            "values": values[on_slice],
        }
    return results
