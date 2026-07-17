import numpy as np
import pandas as pd
import pytest

from early5d import equilibrium_mobility, mobility

ELEMENTS = ["Mn", "Ni", "Co", "Cu"]
FULL_SPECIES = ["Fe", *ELEMENTS]


def _table(compositions, mobility_fn):
    """A minimal synthetic CALPHAD table with one row per given (reduced)
    composition -- no finite-difference neighbors needed, since the mobility
    matrix (unlike the Hessian) only needs the exact composition itself."""
    data = {f"x({el})": [c[i] for c in compositions] for i, el in enumerate(ELEMENTS)}
    for i, el in enumerate(ELEMENTS):
        data[f"Gm.x({el})"] = [0.0 for _ in compositions]  # unused by this module
    for species in FULL_SPECIES:
        data[f"Mob({species})"] = [mobility_fn(c, species) for c in compositions]
    return pd.DataFrame(data)


def test_dominant_mobility_mode_matches_manual_computation():
    composition = np.array([0.10, 0.20, 0.20, 0.30])

    def mobility_fn(c, species):
        return {"Fe": 1.0, "Mn": 5.0, "Ni": 0.5, "Co": 0.3, "Cu": 2.0}[species]

    df = _table([composition], mobility_fn)
    index = {tuple(np.round(composition, 4)): 0}

    mode = equilibrium_mobility.dominant_mobility_mode(df, composition, ELEMENTS, index, decimals=4)

    # cross-check against the existing, already-tested building blocks directly
    from early5d import calphad, linalg
    mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals=4)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))
    expected_mode, _ = mobility.dominant_mode(M)
    expected = linalg.normalized_full_mode(expected_mode)

    assert np.allclose(mode, expected)
    assert mode.shape == (len(FULL_SPECIES),)
    assert np.isclose(np.linalg.norm(mode), 1.0)


def test_compare_mobility_modes_preserves_label_order_and_shapes():
    compositions_list = [
        np.array([0.10, 0.20, 0.20, 0.30]),
        np.array([0.12, 0.25, 0.26, 0.11]),
        np.array([0.03, 0.05, 0.01, 0.91]),
    ]

    def mobility_fn(c, species):
        return {"Fe": 1.0, "Mn": 5.0, "Ni": 0.5, "Co": 0.3, "Cu": 2.0}[species]

    df = _table(compositions_list, mobility_fn)
    index = {tuple(np.round(c, 4)): i for i, c in enumerate(compositions_list)}

    named = {"nominal": compositions_list[0], "phase A": compositions_list[1], "phase B": compositions_list[2]}
    result = equilibrium_mobility.compare_mobility_modes(df, named, ELEMENTS, index, decimals=4)

    assert result["labels"] == ["nominal", "phase A", "phase B"]
    assert len(result["modes"]) == 3
    assert len(result["eigenvalues"]) == 3
    assert len(result["eigenvalue_gaps"]) == 3
    for mode in result["modes"]:
        assert mode.shape == (len(FULL_SPECIES),)
        assert np.isclose(np.linalg.norm(mode), 1.0)
    for eigenvalues in result["eigenvalues"]:
        assert eigenvalues.shape == (len(ELEMENTS),)
        assert np.all(np.diff(eigenvalues) <= 1e-12)  # descending
        assert np.all(eigenvalues >= -1e-12)  # M is positive semi-definite
    for gap in result["eigenvalue_gaps"]:
        assert 0.0 <= gap <= 1.0  # (lambda_1 - lambda_2)/lambda_1, both eigenvalues of an SPD matrix


def test_eigvalsh_descending_matches_eigvalsh_reversed():
    composition = np.array([0.10, 0.20, 0.20, 0.30])

    def mobility_fn(c, species):
        return {"Fe": 1.0, "Mn": 5.0, "Ni": 0.5, "Co": 0.3, "Cu": 2.0}[species]

    df = _table([composition], mobility_fn)
    index = {tuple(np.round(composition, 4)): 0}

    from early5d import calphad
    mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals=4)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

    assert np.allclose(mobility.eigvalsh_descending(M), np.linalg.eigvalsh(M)[::-1])
    assert mobility.eigvalsh_descending(M)[0] == pytest.approx(mobility.dominant_mode(M)[1])


def test_eigenvalue_gap_matches_manual_computation_and_zero_for_degenerate_top_pair():
    composition = np.array([0.10, 0.20, 0.20, 0.30])

    def mobility_fn(c, species):
        return {"Fe": 1.0, "Mn": 5.0, "Ni": 0.5, "Co": 0.3, "Cu": 2.0}[species]

    df = _table([composition], mobility_fn)
    index = {tuple(np.round(composition, 4)): 0}

    from early5d import calphad
    mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals=4)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))
    eigenvalues = np.linalg.eigvalsh(M)
    expected_gap = (eigenvalues[-1] - eigenvalues[-2]) / eigenvalues[-1]

    assert mobility.eigenvalue_gap(M) == pytest.approx(expected_gap)

    # degenerate top pair (M = identity) -> lambda_1 == lambda_2 -> zero gap
    assert mobility.eigenvalue_gap(np.eye(4)) == pytest.approx(0.0)


def test_dominant_mobility_mode_raises_for_missing_composition():
    composition = np.array([0.10, 0.20, 0.20, 0.30])

    def mobility_fn(c, species):
        return 1.0

    df = _table([composition], mobility_fn)
    index = {tuple(np.round(composition, 4)): 0}

    with pytest.raises(KeyError):
        equilibrium_mobility.dominant_mobility_mode(df, np.array([0.5, 0.0, 0.0, 0.0]), ELEMENTS, index, decimals=4)
