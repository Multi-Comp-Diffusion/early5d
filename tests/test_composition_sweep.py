import os

import numpy as np
import pandas as pd
import pytest

from early5d import composition_sweep

ELEMENTS = ["Mn", "Ni", "Co", "Cu"]
FULL_SPECIES = ["Fe", *ELEMENTS]
GRID_SPACING = 0.25


def _full_grid_table(mu_fn, mobility_fn):
    """A small synthetic composition grid spanning the whole reduced-composition
    simplex (Fe implicit, possibly nonzero) at a coarse spacing, kept small
    since sweep_composition_grid pays a ~6x cost per point (one direct
    computation plus one per candidate reference species)."""
    step = GRID_SPACING
    fractions = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    rows = []
    for mn in fractions:
        for ni in fractions:
            for co in fractions:
                for cu in fractions:
                    if mn + ni + co + cu <= 1.0 + 1e-9:
                        rows.append((mn, ni, co, cu))

    data = {f"x({el})": [r[i] for r in rows] for i, el in enumerate(ELEMENTS)}
    for i, el in enumerate(ELEMENTS):
        data[f"Gm.x({el})"] = [mu_fn(r)[i] for r in rows]
    for species in FULL_SPECIES:
        data[f"Mob({species})"] = [mobility_fn(r, species) for r in rows]
    return pd.DataFrame(data)


def _synthetic_sweep():
    A = np.diag([-2.0, 1.0, 1.5, 1.0])  # constant Hessian, unstable along Mn

    def mu_fn(x):
        return A @ np.asarray(x)

    def mobility_fn(x, species):
        return 1.0 + 0.1 * FULL_SPECIES.index(species)

    df = _full_grid_table(mu_fn, mobility_fn)
    return composition_sweep.sweep_composition_grid(
        df, kappa_scalar=0.05, elements=ELEMENTS, full_species=FULL_SPECIES,
        grid_spacing=GRID_SPACING, n_k=50,
    )


def test_sweep_shapes_and_metadata_are_consistent():
    sweep = _synthetic_sweep()
    n_points = len(sweep["tau"])
    assert n_points > 0

    assert sweep["compositions"].shape == (n_points, len(FULL_SPECIES))
    for field in ("xi0", "M_eff", "kappa_eff", "tau_eff", "tau", "k_max"):
        assert sweep[field].shape == (n_points,)
    for field in ("M_eff_by_reference", "kappa_eff_by_reference", "tau_eff_by_reference", "tau_by_reference"):
        assert sweep[field].shape == (n_points, len(FULL_SPECIES))

    assert list(sweep["reference_species"]) == FULL_SPECIES
    assert list(sweep["elements"]) == ELEMENTS
    assert sweep["grid_spacing"] == pytest.approx(GRID_SPACING)
    assert sweep["kappa_scalar"] == pytest.approx(0.05)
    assert np.allclose(sweep["compositions"].sum(axis=1), 1.0)


def test_sweep_default_reference_column_matches_direct_quantities():
    """The by-reference column for full_species[0] (Fe) should reproduce the
    same M_eff/kappa_eff/tau_eff/tau reported directly for that composition."""
    sweep = _synthetic_sweep()
    fe_index = list(sweep["reference_species"]).index("Fe")
    assert np.allclose(sweep["M_eff_by_reference"][:, fe_index], sweep["M_eff"])
    assert np.allclose(sweep["kappa_eff_by_reference"][:, fe_index], sweep["kappa_eff"])
    assert np.allclose(sweep["tau_eff_by_reference"][:, fe_index], sweep["tau_eff"])
    assert np.allclose(sweep["tau_by_reference"][:, fe_index], sweep["tau"])


def test_sweep_lower_bound_and_kappa_eff_invariance_hold_everywhere():
    sweep = _synthetic_sweep()
    assert np.all(sweep["tau_eff_by_reference"] <= sweep["tau_by_reference"] + 1e-9)
    assert np.allclose(sweep["kappa_eff_by_reference"], sweep["kappa_scalar"])


def test_save_and_load_sweep_round_trips(tmp_path):
    sweep = _synthetic_sweep()
    path = os.path.join(tmp_path, "sweep.npz")
    composition_sweep.save_sweep(path, sweep)
    loaded = composition_sweep.load_sweep(path)

    assert loaded["reference_species"] == list(sweep["reference_species"])
    assert loaded["elements"] == list(sweep["elements"])
    assert loaded["grid_spacing"] == pytest.approx(sweep["grid_spacing"])
    assert loaded["kappa_scalar"] == pytest.approx(sweep["kappa_scalar"])
    for field in ("compositions", "xi0", "tau", "tau_eff", "tau_by_reference"):
        assert np.allclose(loaded[field], sweep[field])
