import os

import numpy as np
import pytest

from early5d import calphad, hessian

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "data", "tchea4_mobfe5_873k_alloy_stencils.dat"
)
ELEMENTS = ["Mn", "Ni", "Co", "Cu"]
ALLOY_1 = np.array([0.10, 0.20, 0.20, 0.30])
ALLOY_2 = np.array([0.20, 0.20, 0.15, 0.30])


@pytest.fixture(scope="module")
def stencil_table():
    df = calphad.load_calphad_table(DATA_PATH)
    index, decimals = calphad.build_index(df, ELEMENTS)
    return df, index, decimals


@pytest.mark.parametrize("composition", [ALLOY_1, ALLOY_2])
def test_hessian_is_symmetric(stencil_table, composition):
    df, index, decimals = stencil_table
    Hg = hessian.hessian_at(df, composition, 0.01, ELEMENTS, index, decimals)
    assert np.allclose(Hg, Hg.T)


@pytest.mark.parametrize("composition", [ALLOY_1, ALLOY_2])
def test_both_worked_alloys_are_spinodal(stencil_table, composition):
    df, index, decimals = stencil_table
    Hg = hessian.hessian_at(df, composition, 0.01, ELEMENTS, index, decimals)
    assert hessian.is_spinodal(Hg)


def test_orthogonalization_preserves_symmetry(stencil_table):
    df, index, decimals = stencil_table
    Hg = hessian.hessian_at(df, ALLOY_1, 0.01, ELEMENTS, index, decimals)
    Hg_hat = hessian.orthogonalize(Hg)
    assert np.allclose(Hg_hat, Hg_hat.T)


def test_thermo_mode_matches_reported_driving_force(stencil_table):
    # Paper (Fig. 4a caption): xi0 = -14.3 kJ/mol for Fe20Mn10Ni20Co20Cu30 at 873 K.
    df, index, decimals = stencil_table
    Hg = hessian.hessian_at(df, ALLOY_1, 0.01, ELEMENTS, index, decimals)
    _, xi0 = hessian.thermo_mode(Hg)
    assert xi0 / 1000 == pytest.approx(-14.3, abs=0.1)


def test_orthogonalize_rejects_mismatched_dimension():
    # default O is only valid for a 4-independent-component system; a
    # differently-sized matrix must not be silently truncated against it
    Hg_3x3 = np.eye(3)
    with pytest.raises(ValueError):
        hessian.orthogonalize(Hg_3x3)


def test_wavenumber_cutoff_zero_for_stable_hessian():
    Hg = np.diag([1.0, 2.0])
    kappa = np.eye(2)
    assert hessian.wavenumber_cutoff(Hg, kappa) == 0.0


def test_wavenumber_cutoff_matches_binary_formula():
    # 1-component (binary) case: k_cut = sqrt(-h / kappa) (Eq. 11).
    h, kappa_value = -4.0, 0.5
    Hg = np.array([[h]])
    kappa = np.array([[kappa_value]])
    expected = np.sqrt(-h / kappa_value)
    assert hessian.wavenumber_cutoff(Hg, kappa) == pytest.approx(expected)
