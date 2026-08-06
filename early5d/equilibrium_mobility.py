"""Dominant mobility eigenmode at equilibrium phase compositions, compared
against the nominal (parent) alloy's own mobility mode (Fig. 4, extended).

The Onsager mobility matrix (Eq. 4) is defined at any single composition
without needing finite-difference neighbors (unlike the Hessian), so it can
be evaluated directly at equilibrium phase compositions from an external
convex-hull/tie-line calculation (see the README's Scope section) -- letting
the fastest-diffusing direction be compared across the parent alloy and each
of its fully-decomposed equilibrium end-members, not just at the nominal
composition as in :mod:`early5d.workflow`.
"""

from early5d import calphad, linalg, mobility


def _mobility_matrix_at(df, composition, elements, index, decimals):
    """Onsager mobility matrix (Eq. 4) at a single composition -- no finite-difference
    neighbors needed, unlike the Hessian (see the module docstring)."""
    mobilities = calphad.atomic_mobilities(df, composition, elements, index, decimals)
    return mobility.mobility_matrix(mobilities, calphad.full_composition(composition))


def dominant_mobility_mode(df, composition, elements=None, index=None, decimals=4):
    """Dominant mobility eigenmode at a single (reduced-composition) point,
    in full-species, canonically-oriented, unit-normalized form -- ready for
    :func:`early5d.plotting.plot_eigenmode_bars`.

    Unlike :func:`early5d.workflow.analyze_composition`, this only needs the
    mobility matrix at ``composition`` (no finite-difference stencil), so it
    works for any composition present in ``df`` as an exact row -- including
    equilibrium phase compositions, which typically aren't part of the fine
    local stencil bundled around one specific worked alloy.
    """
    elements = elements if elements is not None else calphad.composition_columns(df)
    if index is None:
        index, decimals = calphad.build_index(df, elements, decimals)
    M = _mobility_matrix_at(df, composition, elements, index, decimals)
    mode, _ = mobility.dominant_mode(M)
    return linalg.normalized_full_mode(mode)


def compare_mobility_modes(df, compositions, elements=None, index=None, decimals=4):
    """Dominant mobility mode at every named (reduced-composition) point in
    ``compositions`` (e.g. ``{"nominal": ..., "phase A": ..., "phase B": ...}``).

    Returns ``{"labels": [...], "modes": [...], "eigenvalues": [...],
    "eigenvalue_gaps": [...]}``, ready for
    :func:`early5d.plotting.plot_eigenmode_bars` (``labels`` preserves
    ``compositions``' insertion order). ``eigenvalues`` is each composition's
    full mobility-matrix eigenvalue spectrum, descending
    (``eigenvalues[k][0]`` is the dominant one, i.e. ``modes[k]``'s own
    eigenvalue); ``eigenvalue_gaps`` is each composition's
    :func:`early5d.mobility.eigenvalue_gap` -- how well-separated (and hence
    how robustly defined) that composition's dominant mode is from the
    next-fastest transport direction.
    """
    elements = elements if elements is not None else calphad.composition_columns(df)
    if index is None:
        index, decimals = calphad.build_index(df, elements, decimals)
    labels = list(compositions)
    modes, eigenvalues, eigenvalue_gaps = [], [], []
    for label in labels:
        M = _mobility_matrix_at(df, compositions[label], elements, index, decimals)
        mode, _ = mobility.dominant_mode(M)
        modes.append(linalg.normalized_full_mode(mode))
        eigenvalues.append(mobility.eigvalsh_descending(M))
        eigenvalue_gaps.append(mobility.eigenvalue_gap(M))
    return {"labels": labels, "modes": modes, "eigenvalues": eigenvalues, "eigenvalue_gaps": eigenvalue_gaps}
