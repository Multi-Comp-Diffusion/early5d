import numpy as np
import pandas as pd
import pytest

from early5d import gauge_sensitivity

ELEMENTS = ["Mn", "Ni", "Co", "Cu"]
FULL_SPECIES = ["Fe", *ELEMENTS]


def _stencil_table(center, grid_spacing, mu_fn, mobility_fn):
    center = np.asarray(center, dtype=float)
    rows = [center]
    for j in range(len(center)):
        step = np.zeros(len(center))
        step[j] = grid_spacing
        rows.append(center + step)
        rows.append(center - step)

    data = {f"x({el})": [row[i] for row in rows] for i, el in enumerate(ELEMENTS)}
    for i, el in enumerate(ELEMENTS):
        data[f"Gm.x({el})"] = [mu_fn(row)[i] for row in rows]
    for species in FULL_SPECIES:
        data[f"Mob({species})"] = [mobility_fn(row, species) for row in rows]
    return pd.DataFrame(data)


def test_reference_swap_matrix_matches_hand_derivation():
    # swapping Fe <-> Mn: mu'_Fe = -mu_Mn, mu'_Ni = mu_Ni - mu_Mn, etc.
    A, new_elements = gauge_sensitivity.reference_swap_matrix("Fe", ELEMENTS, "Mn")
    assert new_elements == ["Fe", "Ni", "Co", "Cu"]
    expected = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
    ])
    assert np.allclose(A, expected)


def test_swapped_hessian_matches_analytic_ground_truth():
    """For G(X) = 0.5 X^T C X over the full 5-species mole-fraction vector,
    mu_i^(r) = g_i - g_r (g = C X) has a closed form for *any* reference r --
    an independent ground truth (no finite differences) to check the
    congruence transform against."""
    rng = np.random.default_rng(0)
    C = rng.standard_normal((5, 5))
    C = 0.5 * (C + C.T)

    def analytic_Hg(reference):
        others = [s for s in range(5) if s != reference]
        n = len(others)
        Hg = np.empty((n, n))
        for jj, j in enumerate(others):
            for ii, i in enumerate(others):
                Hg[jj, ii] = (C[i, j] - C[i, reference]) - (C[reference, j] - C[reference, reference])
        return Hg, others

    Hg0, elements0 = analytic_Hg(0)
    for new_ref in [1, 2, 3, 4]:
        A, new_elements = gauge_sensitivity.reference_swap_matrix(0, elements0, new_ref)
        Hg_transformed = A @ Hg0 @ A.T

        Hg_truth, elements_truth = analytic_Hg(new_ref)
        perm = [elements_truth.index(e) for e in new_elements]
        Hg_truth_reordered = Hg_truth[np.ix_(perm, perm)]

        assert np.allclose(Hg_transformed, Hg_truth_reordered, atol=1e-10)


def test_swapped_hessian_preserves_inertia():
    """A congruence transform must preserve the number of negative eigenvalues
    (Sylvester's law) -- a necessary invariant regardless of the specific
    composition or CALPHAD data used."""
    Hg = np.diag([-2.0, 1.0, 3.0, 0.5])
    for new_ref in ELEMENTS:
        Hg_swapped, _ = gauge_sensitivity.swapped_hessian(Hg, "Fe", ELEMENTS, new_ref)
        n_neg = int(np.sum(np.linalg.eigvalsh(Hg) < 0))
        n_neg_swapped = int(np.sum(np.linalg.eigvalsh(Hg_swapped) < 0))
        assert n_neg == n_neg_swapped


def test_reference_sensitivity_lower_bound_holds_for_every_reference():
    center = np.array([0.10, 0.20, 0.20, 0.30])
    grid_spacing = 0.01
    A = np.array([[-2.0, 0.1, 0.0, 0.0],
                  [0.1, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.5, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0 + 0.1 * FULL_SPECIES.index(species)  # non-uniform, so M isn't trivially isotropic

    df = _stencil_table(center, grid_spacing, mu_fn, mobility_fn)
    kappa_scalar = 0.05

    results = gauge_sensitivity.reference_sensitivity(
        df, center, kappa_scalar, ELEMENTS, FULL_SPECIES, grid_spacing,
    )

    assert set(results) == set(FULL_SPECIES)
    for species, values in results.items():
        assert values["tau_eff"] <= values["tau"] + 1e-9


def test_reference_sensitivity_kappa_eff_is_gauge_invariant():
    """kappa_eff = v0^T M kappa v0 / M_eff = kappa_scalar exactly whenever
    kappa = kappa_scalar * I, for *any* v0 -- an algebraic identity, so it
    must hold under every reference-species choice regardless of the data."""
    center = np.array([0.10, 0.20, 0.20, 0.30])
    grid_spacing = 0.01
    A = np.array([[-2.0, 0.1, 0.0, 0.0],
                  [0.1, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.5, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0 + 0.1 * FULL_SPECIES.index(species)

    df = _stencil_table(center, grid_spacing, mu_fn, mobility_fn)
    kappa_scalar = 0.05

    results = gauge_sensitivity.reference_sensitivity(
        df, center, kappa_scalar, ELEMENTS, FULL_SPECIES, grid_spacing,
    )
    for species, values in results.items():
        assert values["kappa_eff"] == pytest.approx(kappa_scalar, rel=1e-9)


def test_reference_sensitivity_includes_bounded_mobility_ratio():
    """mobility_ratio = M_eff / lambda_1(M) in [0, 1] for every reference choice --
    the frame-specific thermo mode can capture at most all of M's largest eigenvalue."""
    center = np.array([0.10, 0.20, 0.20, 0.30])
    grid_spacing = 0.01
    A = np.array([[-2.0, 0.1, 0.0, 0.0],
                  [0.1, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.5, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0 + 0.1 * FULL_SPECIES.index(species)

    df = _stencil_table(center, grid_spacing, mu_fn, mobility_fn)
    kappa_scalar = 0.05

    results = gauge_sensitivity.reference_sensitivity(
        df, center, kappa_scalar, ELEMENTS, FULL_SPECIES, grid_spacing,
    )
    for species, values in results.items():
        assert "mobility_ratio" in values
        assert 0.0 <= values["mobility_ratio"] <= 1.0 + 1e-9
