"""End-to-end per-composition analysis, bundling Secs. 2.1-2.4 into one call.

Wires together the pipeline that :mod:`examples`' 02-05 scripts each repeat by
hand (Hessian and spinodal check, mobility matrix, thermo-kinetic dispersion
relation and CTLD, single-mode effective projection, eigenmode bar-shard data,
and -- given an equilibrium tie-line -- misalignment angles) for a single
composition, so exploring one alloy doesn't require re-deriving that plumbing.
"""

import numpy as np

from early5d import calphad, effective, hessian, kinetics, linalg, misalignment, mobility


def analyze_composition(df, composition, kappa, elements=None, grid_spacing=0.01,
                         index=None, decimals=4, equilibrium_mode=None, n_k=2000):
    """Run every composition-level analysis (Figs. 3-5) for one point.

    ``equilibrium_mode`` is the full-species tie-line direction (``[Fe, Mn,
    Ni, Co, Cu]`` for this system, e.g. ``PHASE_A - PHASE_B``); if given, the
    result includes the equilibrium bar and misalignment angles, otherwise
    those entries are left ``None``. If the composition is thermodynamically
    stable (outside the spinodal region), there is no unstable mode to
    project or compare, so ``dispersion``, ``effective``, ``mobility_mode``,
    ``bars``, and ``misalignment`` are all ``None``.

    Returns a dict: ``composition``, ``elements``, ``Hg``, ``M``,
    ``is_spinodal``, ``thermo_mode``, ``xi0``, ``dispersion`` (from
    :func:`early5d.kinetics.solve_dispersion_relation`), ``effective`` (from
    :func:`early5d.effective.solve_effective_dispersion`), ``mobility_mode``,
    ``bars`` (``{"modes": [...], "labels": [...]}`` for
    :func:`early5d.plotting.plot_eigenmode_bars`), and ``misalignment``
    (``{"vectors", "labels", "metric", "angles"}`` for
    :func:`early5d.plotting.plot_misalignment`).
    """
    elements = elements if elements is not None else calphad.composition_columns(df)
    if index is None:
        index, decimals = calphad.build_index(df, elements, decimals)

    Hg = hessian.hessian_at(df, composition, grid_spacing, elements, index, decimals)
    mobilities = calphad.atomic_mobilities(df, composition, elements, index, decimals)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))
    thermo_v0, xi0 = hessian.thermo_mode(Hg)

    result = {
        "composition": np.asarray(composition, dtype=float),
        "elements": elements,
        "Hg": Hg,
        "M": M,
        "is_spinodal": hessian.is_spinodal(Hg),
        "thermo_mode": thermo_v0,
        "xi0": xi0,
        "dispersion": None,
        "effective": None,
        "mobility_mode": None,
        "bars": None,
        "misalignment": None,
    }
    if not result["is_spinodal"]:
        return result

    dispersion = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=n_k)
    eff = effective.solve_effective_dispersion(Hg, M, kappa, n_k=n_k)
    mobility_mode, _ = mobility.dominant_mode(M)

    result["dispersion"] = dispersion
    result["effective"] = eff
    result["mobility_mode"] = mobility_mode

    bar_labels = ["thermo", "mobility", "growth"]
    bar_modes = [
        linalg.normalized_full_mode(thermo_v0),
        linalg.normalized_full_mode(mobility_mode),
        linalg.normalized_full_mode(dispersion["growth_mode"]),
    ]

    if equilibrium_mode is not None:
        equilibrium_mode = np.asarray(equilibrium_mode, dtype=float)
        equilibrium_reduced = equilibrium_mode[1:]  # drop the implicit reference component
        bar_labels.append("equilibrium")
        bar_modes.append(linalg.align_mode(equilibrium_mode) / np.linalg.norm(equilibrium_mode))

        M_inv = misalignment.inverse_mobility_metric(M)
        vectors = [thermo_v0, dispersion["growth_mode"], equilibrium_reduced]
        vector_labels = ["thermo", "growth", "equilibrium"]
        angles = {}
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                key = f"{vector_labels[i]}-{vector_labels[j]}"
                angles[key] = {
                    "euclidean": misalignment.angle_between(vectors[i], vectors[j]),
                    "weighted": misalignment.angle_between(vectors[i], vectors[j], metric=M_inv),
                }
        result["misalignment"] = {
            "vectors": vectors, "labels": vector_labels, "metric": M_inv, "angles": angles,
        }

    result["bars"] = {"modes": bar_modes, "labels": bar_labels}
    return result
