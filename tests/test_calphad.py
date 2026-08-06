import os

import numpy as np
import pytest

from early5d import calphad

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "tchea4_mobfe5_873k_alloy_stencils.dat"
)
ELEMENTS = ["Mn", "Ni", "Co", "Cu"]
ALLOY_1 = np.array([0.10, 0.20, 0.20, 0.30])


@pytest.fixture(scope="module")
def stencil_table():
    return calphad.load_calphad_table(DATA_PATH)


def test_composition_columns_matches_bundled_elements(stencil_table):
    assert calphad.composition_columns(stencil_table) == ELEMENTS


def test_reference_species_is_iron(stencil_table):
    assert calphad.reference_species(stencil_table, ELEMENTS) == "Fe"


def test_reference_species_raises_if_not_exactly_one():
    # every element (incl. the true reference) claimed as "independent" -> no Mob(...)
    # column is left over for a reference species
    df = calphad.load_calphad_table(DATA_PATH)
    with pytest.raises(ValueError):
        calphad.reference_species(df, elements=[*ELEMENTS, "Fe"])


def test_full_species_prepends_reference(stencil_table):
    assert calphad.full_species(stencil_table, ELEMENTS) == ["Fe", *ELEMENTS]


def test_row_at_raises_for_missing_composition(stencil_table):
    with pytest.raises(KeyError):
        calphad.row_at(stencil_table, [0.99, 0.0, 0.0, 0.0], ELEMENTS)


def test_chemical_potential_and_atomic_mobilities_shapes(stencil_table):
    index, decimals = calphad.build_index(stencil_table, ELEMENTS)
    mu = calphad.chemical_potential(stencil_table, ALLOY_1, ELEMENTS, index, decimals)
    mobilities = calphad.atomic_mobilities(stencil_table, ALLOY_1, ELEMENTS, index, decimals)
    assert mu.shape == (len(ELEMENTS),)
    assert mobilities.shape == (len(ELEMENTS) + 1,)  # reference + independent components
    assert np.all(np.isfinite(mu))
    assert np.all(mobilities > 0)  # atomic mobilities are physically positive


def test_build_index_rounds_to_requested_decimals(stencil_table):
    index, decimals = calphad.build_index(stencil_table, ELEMENTS, decimals=2)
    assert decimals == 2
    assert all(len(key) == len(ELEMENTS) for key in index)
    # a composition that only matches at 2 decimals (not the file's exact float) still resolves
    row = calphad.row_at(stencil_table, ALLOY_1 + 1e-9, ELEMENTS, index, decimals)
    assert row is not None


def test_full_composition_prepends_implicit_reference():
    full = calphad.full_composition(ALLOY_1)
    assert full[0] == pytest.approx(1.0 - ALLOY_1.sum())
    assert np.allclose(full[1:], ALLOY_1)
    assert full.sum() == pytest.approx(1.0)
