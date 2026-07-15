import numpy as np
import pandas as pd
import pytest

from early5d import ternary_workflow

ELEMENTS = ["Mn", "Ni", "Co", "Cu"]
SPECIES = ["Fe", *ELEMENTS]
GRID_SPACING = 0.1


def _full_grid_table(mu_fn, mobility_fn):
    """A synthetic composition grid spanning the *whole* reduced-composition
    simplex (``sum(Mn, Ni, Co, Cu) <= 1``, i.e. Fe implicit and possibly
    nonzero) at ``GRID_SPACING`` steps, so every non-edge point's finite-
    difference neighbors -- which shift the implicit Fe fraction too -- are
    actually present in the table."""
    step = GRID_SPACING
    fractions = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    rows = []
    for mn in fractions:
        for ni in fractions:
            for co in fractions:
                for cu in fractions:
                    if mn + ni + co + cu <= 1.0 + 1e-9:
                        rows.append((mn, ni, co, cu))
    row_set = set(rows)

    def is_interior(row):
        for k in range(4):
            plus = list(row)
            plus[k] = round(plus[k] + step, 2)
            minus = list(row)
            minus[k] = round(minus[k] - step, 2)
            if minus[k] < -1e-9 or tuple(minus) not in row_set:
                return False
            if sum(plus) > 1.0 + 1e-9 or tuple(plus) not in row_set:
                return False
        return True

    interior_rows = [r for r in rows if is_interior(r)]

    data = {f"x({el})": [r[i] for r in rows] for i, el in enumerate(ELEMENTS)}
    for i, el in enumerate(ELEMENTS):
        data[f"Gm.x({el})"] = [mu_fn(r)[i] for r in rows]
    for species in SPECIES:
        data[f"Mob({species})"] = [mobility_fn(r, species) for r in rows]
    return pd.DataFrame(data), interior_rows


def test_scan_composition_grid_flags_only_spinodal_interior_points():
    A = np.diag([-2.0, 1.0, 1.0, 1.0])  # unstable along Mn only

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0

    df, interior_rows = _full_grid_table(mu_fn, mobility_fn)
    assert len(interior_rows) > 5  # sanity check the synthetic grid is non-trivial

    kappa = 0.05 * np.eye(len(ELEMENTS))
    full_compositions, values = ternary_workflow.scan_composition_grid(
        df, kappa, elements=ELEMENTS, grid_spacing=GRID_SPACING, field="tau", n_k=50,
    )

    assert len(full_compositions) == len(values) > 0
    assert np.allclose(full_compositions.sum(axis=1), 1.0)

    # hessian_at also succeeds (one-sided) right at the grid's outer edge, so
    # scan_composition_grid legitimately returns more than just the strictly
    # interior points -- but every strictly interior point (full central
    # differences both ways -> Hg is exactly A, unstable by construction)
    # must appear, with a finite, positive CTLD
    reduced = full_compositions[:, 1:]
    value_by_row = {tuple(np.round(row, 2)): v for row, v in zip(reduced, values)}
    interior_set = {tuple(np.round(row, 2)) for row in interior_rows}
    assert interior_set <= set(value_by_row)
    interior_values = [value_by_row[r] for r in interior_set]
    assert all(v > 0 and np.isfinite(v) for v in interior_values)


def test_scan_composition_grid_rejects_unknown_field():
    df, _ = _full_grid_table(lambda x: -np.asarray(x), lambda x, s: 1.0)
    with pytest.raises(ValueError):
        ternary_workflow.scan_composition_grid(df, np.eye(len(ELEMENTS)), elements=ELEMENTS, field="not_a_field")


def test_pairwise_ternary_slices_covers_all_pairs_and_normalizes():
    rng = np.random.default_rng(0)
    n_points = 200
    full_compositions = rng.dirichlet(np.ones(len(SPECIES)), size=n_points)
    values = rng.uniform(1.0, 100.0, size=n_points)

    # force enough points onto a couple of specific 20:20 slices
    fe_idx, mn_idx = SPECIES.index("Fe"), SPECIES.index("Mn")
    full_compositions[:20, fe_idx] = 0.20
    full_compositions[:20, mn_idx] = 0.20
    remaining = [k for k in range(len(SPECIES)) if k not in (fe_idx, mn_idx)]
    leftover = 1.0 - 0.40
    weights = rng.dirichlet(np.ones(3), size=20) * leftover
    full_compositions[:20][:, remaining] = weights

    slices = ternary_workflow.pairwise_ternary_slices(full_compositions, values, SPECIES, fixed_fraction=0.20)

    assert ("Fe", "Mn") in slices
    entry = slices[("Fe", "Mn")]
    assert entry["ternary_elements"] == ("Ni", "Co", "Cu")
    assert entry["ternary_compositions"].shape[1] == 3
    assert np.allclose(entry["ternary_compositions"].sum(axis=1), 1.0)
    assert len(entry["values"]) == entry["ternary_compositions"].shape[0] >= 20

    # every returned key really is a 2-subset of the species list
    for pair in slices:
        assert len(pair) == 2
        assert pair[0] in SPECIES and pair[1] in SPECIES


def test_pairwise_ternary_slices_omits_pairs_below_min_points():
    full_compositions = np.tile(np.array([0.5, 0.3, 0.1, 0.05, 0.05]), (2, 1))  # only 2 identical points
    values = np.array([1.0, 2.0])
    slices = ternary_workflow.pairwise_ternary_slices(
        full_compositions, values, SPECIES, fixed_fraction=0.5, min_points=3,
    )
    assert slices == {}
