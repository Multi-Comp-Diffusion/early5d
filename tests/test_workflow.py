import numpy as np
import pandas as pd
import pytest

from early5d import workflow

ELEMENTS = ["Mn", "Ni", "Co", "Cu"]


def _stencil_table(center, grid_spacing, mu_fn, mobility_fn):
    """A minimal synthetic CALPHAD table: a composition and its 8 (+/- one
    grid step per axis) neighbors, enough for :func:`hessian.hessian_at` and
    :func:`calphad.atomic_mobilities`."""
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
    for species in ["Fe", *ELEMENTS]:
        data[f"Mob({species})"] = [mobility_fn(row, species) for row in rows]
    return pd.DataFrame(data)


def test_analyze_composition_unstable_matches_manual_pipeline():
    """A quadratic (in the independent components) chemical potential gives an
    exactly linear mu -> constant Hessian, so the workflow's Hg/M/dispersion
    should reproduce a hand-computed reference exactly."""
    center = np.array([0.10, 0.20, 0.20, 0.30])
    grid_spacing = 0.01
    A = np.array([[-2.0, 0.1, 0.0, 0.0],
                  [0.1, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.5, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0  # uniform mobility -> M is a fixed, composition-independent matrix

    df = _stencil_table(center, grid_spacing, mu_fn, mobility_fn)
    kappa = 0.05 * np.eye(len(ELEMENTS))

    result = workflow.analyze_composition(df, center, kappa, elements=ELEMENTS, grid_spacing=grid_spacing)

    assert np.allclose(result["Hg"], A, atol=1e-8)
    assert result["is_spinodal"] is True
    assert result["dispersion"] is not None
    assert result["dispersion"]["tau"] > 0
    assert result["effective"]["tau_eff"] <= result["dispersion"]["tau"] + 1e-9
    assert result["misalignment"] is None
    assert result["bars"]["labels"] == ["thermo", "mobility", "growth"]


def test_analyze_composition_stable_skips_downstream_analyses():
    center = np.array([0.10, 0.20, 0.20, 0.30])
    grid_spacing = 0.01
    A = np.diag([1.0, 2.0, 3.0, 4.0])  # positive-definite -> thermodynamically stable

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0

    df = _stencil_table(center, grid_spacing, mu_fn, mobility_fn)
    kappa = 0.05 * np.eye(len(ELEMENTS))

    result = workflow.analyze_composition(df, center, kappa, elements=ELEMENTS, grid_spacing=grid_spacing)

    assert result["is_spinodal"] is False
    for key in ("dispersion", "effective", "mobility_mode", "bars", "misalignment"):
        assert result[key] is None


def test_analyze_composition_with_equilibrium_mode_adds_misalignment():
    center = np.array([0.10, 0.20, 0.20, 0.30])
    grid_spacing = 0.01
    A = np.array([[-2.0, 0.1, 0.0, 0.0],
                  [0.1, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.5, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0

    df = _stencil_table(center, grid_spacing, mu_fn, mobility_fn)
    kappa = 0.05 * np.eye(len(ELEMENTS))
    equilibrium_mode = np.array([0.26, 0.09, 0.20, 0.25, -0.19])  # arbitrary full-species tie-line

    result = workflow.analyze_composition(
        df, center, kappa, elements=ELEMENTS, grid_spacing=grid_spacing, equilibrium_mode=equilibrium_mode,
    )

    assert result["misalignment"] is not None
    assert set(result["misalignment"]["angles"]) == {"thermo-growth", "thermo-equilibrium", "growth-equilibrium"}
    for angles in result["misalignment"]["angles"].values():
        assert 0.0 <= angles["euclidean"] <= 90.0
        assert 0.0 <= angles["weighted"] <= 90.0
    assert result["bars"]["labels"] == ["thermo", "mobility", "growth", "equilibrium"]
    assert len(result["bars"]["modes"]) == 4


def test_analyze_composition_raises_for_missing_neighbor():
    """A composition whose finite-difference neighbor isn't in the table
    should surface as a clear error, not silently wrong output."""
    center = np.array([0.10, 0.20, 0.20, 0.30])
    grid_spacing = 0.01

    def mu_fn(x):
        return -np.asarray(x)

    def mobility_fn(x, species):
        return 1.0

    full_table = _stencil_table(center, grid_spacing, mu_fn, mobility_fn)
    incomplete = full_table.iloc[:-1]  # drop one neighbor row
    kappa = 0.05 * np.eye(len(ELEMENTS))

    with pytest.raises((KeyError, ValueError)):
        workflow.analyze_composition(incomplete, center, kappa, elements=ELEMENTS, grid_spacing=grid_spacing)
