"""Full composition-space sweep of every early-stage parameter, saved as data
rather than just plotted (Figs. 1/2, Sec. 4).

:func:`sweep_composition_grid` evaluates, at every grid point inside the
spinodal region: the thermodynamic driving force (``xi0``), the effective
(single-mode) quantities (``M_eff``, ``kappa_eff``, ``tau_eff``, Eqs. 15-17),
the full eigenproblem quantities (``tau``, ``k_max``, Eqs. 12-13) -- all under
the CALPHAD export's own reference species -- *and* how ``M_eff``,
``kappa_eff``, ``tau_eff``, and ``tau`` shift under every alternate choice of
reference species (:mod:`early5d.gauge_sensitivity`), quantifying the
gauge-dependence discussed in Sec. 4 across the whole composition space
rather than just the two worked alloys.

:func:`save_sweep`/:func:`load_sweep` round-trip the result through a
compressed ``.npz`` archive.
"""

import numpy as np

from early5d import calphad, effective, gauge_sensitivity, hessian, kinetics, mobility


def sweep_composition_grid(df, kappa_scalar, elements, full_species, grid_spacing,
                            index=None, decimals=4, n_k=30):
    """Evaluate every early-stage parameter, and its reference-species
    sensitivity, at every valid grid point inside the spinodal region.

    ``full_species`` is ``[reference, *elements]`` (see
    :func:`early5d.calphad.full_species`); ``kappa_scalar`` is used as the
    isotropic ``kappa_scalar * I`` gradient-energy tensor, redefined fresh in
    each candidate reference frame (see :mod:`early5d.gauge_sensitivity`).

    Returns a dict of arrays, all with a leading dimension ``P`` (the number
    of valid points) except ``reference_species`` (the fixed, ``P``-independent
    order of the ``*_by_reference`` columns) and the scalar/list metadata
    (``elements``, ``grid_spacing``, ``kappa_scalar``):

    - ``compositions``: ``(P, n_species)``, full-species mole fractions
    - ``xi0``, ``M_eff``, ``kappa_eff``, ``tau_eff``, ``tau``, ``k_max``: ``(P,)``,
      under the paper's own (``full_species[0]``) reference species
    - ``M_eff_by_reference``, ``kappa_eff_by_reference``, ``tau_eff_by_reference``,
      ``tau_by_reference``: ``(P, n_species)``, recomputed with each of
      ``reference_species`` in turn (column order matches ``reference_species``)
    """
    elements = list(elements)
    reference_species = list(full_species)
    if index is None:
        index, decimals = calphad.build_index(df, elements, decimals)
    compositions = df[[f"x({e})" for e in elements]].to_numpy(dtype=float)
    kappa = kappa_scalar * np.eye(len(elements))

    rows = []
    for composition in compositions:
        try:
            Hg = hessian.hessian_at(df, composition, grid_spacing, elements, index, decimals)
        except (KeyError, ValueError):
            continue  # composition sits at the extracted grid's edge; skip

        if not hessian.is_spinodal(Hg):
            continue

        mobilities = calphad.atomic_mobilities(df, composition, elements, index, decimals)
        M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

        try:
            # near a pure-single-element corner (an independent component's mole
            # fraction essentially zero), M can collapse to (numerically) singular,
            # dividing by zero in kappa_eff/tau_eff's formula -- not a usable point
            eff = effective.solve_effective_dispersion(Hg, M, kappa, n_k=n_k)
        except ZeroDivisionError:
            continue
        dispersion = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=n_k)
        if dispersion["wavenumbers"] is None or not np.isfinite(dispersion["tau"]):
            continue  # numerically degenerate point, see ternary_workflow.scan_composition_grid

        try:
            sensitivity = gauge_sensitivity.reference_sensitivity(
                df, composition, kappa_scalar, elements, full_species, grid_spacing,
                index=index, decimals=decimals, n_k=n_k,
            )
        except ZeroDivisionError:
            continue  # same degeneracy as above, but for one of the *swapped* reference frames

        rows.append({
            "composition": calphad.full_composition(composition),
            "xi0": eff["xi0"], "M_eff": eff["M_eff"], "kappa_eff": eff["kappa_eff"],
            "tau_eff": eff["tau_eff"], "tau": dispersion["tau"], "k_max": dispersion["k_max"],
            "M_eff_by_reference": [sensitivity[s]["M_eff"] for s in reference_species],
            "kappa_eff_by_reference": [sensitivity[s]["kappa_eff"] for s in reference_species],
            "tau_eff_by_reference": [sensitivity[s]["tau_eff"] for s in reference_species],
            "tau_by_reference": [sensitivity[s]["tau"] for s in reference_species],
        })

    n_species = len(reference_species)
    if not rows:
        scalar, vector = np.empty(0), np.empty((0, n_species))
        result = {
            "compositions": np.empty((0, n_species)), "xi0": scalar, "M_eff": scalar,
            "kappa_eff": scalar, "tau_eff": scalar, "tau": scalar, "k_max": scalar,
            "M_eff_by_reference": vector, "kappa_eff_by_reference": vector,
            "tau_eff_by_reference": vector, "tau_by_reference": vector,
        }
    else:
        result = {
            "compositions": np.array([r["composition"] for r in rows]),
            "xi0": np.array([r["xi0"] for r in rows]),
            "M_eff": np.array([r["M_eff"] for r in rows]),
            "kappa_eff": np.array([r["kappa_eff"] for r in rows]),
            "tau_eff": np.array([r["tau_eff"] for r in rows]),
            "tau": np.array([r["tau"] for r in rows]),
            "k_max": np.array([r["k_max"] for r in rows]),
            "M_eff_by_reference": np.array([r["M_eff_by_reference"] for r in rows]),
            "kappa_eff_by_reference": np.array([r["kappa_eff_by_reference"] for r in rows]),
            "tau_eff_by_reference": np.array([r["tau_eff_by_reference"] for r in rows]),
            "tau_by_reference": np.array([r["tau_by_reference"] for r in rows]),
        }

    result["reference_species"] = np.array(reference_species)
    result["elements"] = np.array(elements)
    result["grid_spacing"] = np.array(grid_spacing)
    result["kappa_scalar"] = np.array(kappa_scalar)
    return result


def save_sweep(path, sweep):
    """Save a :func:`sweep_composition_grid` result to a compressed ``.npz`` archive."""
    np.savez_compressed(path, **sweep)


def load_sweep(path):
    """Load a sweep saved by :func:`save_sweep`.

    ``reference_species``/``elements`` come back as plain lists of ``str``
    and ``grid_spacing``/``kappa_scalar`` as plain floats (``.npz`` stores
    everything as arrays, including 0-d ones for scalars).
    """
    with np.load(path) as data:
        result = {key: data[key] for key in data.files}
    result["reference_species"] = [str(s) for s in result["reference_species"]]
    result["elements"] = [str(e) for e in result["elements"]]
    result["grid_spacing"] = float(result["grid_spacing"])
    result["kappa_scalar"] = float(result["kappa_scalar"])
    return result
