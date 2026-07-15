"""Loading and indexing of Thermo-Calc CALPHAD property-diagram exports.

Expected columns (see README for the full format): ``Gm.x(<element>)`` for
the chemical potential of each independent (non-reference) component,
``Mob(<species>)`` for the atomic mobility of every species including the
reference, and ``x(<element>)`` for the composition of every independent
component. The reference species (e.g. Fe in the Fe-Mn-Ni-Co-Cu system) has
no ``x(...)`` column; its mole fraction follows from ``1 - sum(x_i)``.
"""

import numpy as np
import pandas as pd


def load_calphad_table(path):
    """Load a tab-separated Thermo-Calc property-diagram export into a DataFrame."""
    return pd.read_table(path, sep="\t")


def composition_columns(df):
    """Independent-component symbols, e.g. ``['Mn', 'Ni', 'Co', 'Cu']``, inferred
    from the ``x(...)`` columns, in file order."""
    return [c[2:-1] for c in df.columns if c.startswith("x(") and c.endswith(")")]


def reference_species(df, elements=None):
    """The one species with a ``Mob(...)`` column but no ``x(...)`` column
    (e.g. Fe), whose mole fraction is implicit (``1 - sum(x_i)``)."""
    elements = elements if elements is not None else composition_columns(df)
    mob_species = [c[4:-1] for c in df.columns if c.startswith("Mob(") and c.endswith(")")]
    reference = [s for s in mob_species if s not in elements]
    if len(reference) != 1:
        raise ValueError(f"expected exactly one reference species, found {reference}")
    return reference[0]


def full_species(df, elements=None):
    """Species order used throughout this package: ``[reference, *elements]``."""
    elements = elements if elements is not None else composition_columns(df)
    return [reference_species(df, elements), *elements]


def build_index(df, elements=None, decimals=4):
    """Map rounded composition tuples to row indices for O(1) grid-neighbor lookups.

    The CALPHAD exports used here are regular composition grids (e.g. 1 at % or
    5 at % steps); rounding avoids floating-point mismatches between a
    requested composition and the grid values actually stored in the table.
    """
    elements = elements if elements is not None else composition_columns(df)
    coords = df[[f"x({e})" for e in elements]].to_numpy(dtype=float)
    return {tuple(np.round(row, decimals)): i for i, row in enumerate(coords)}, decimals


def row_at(df, composition, elements=None, index=None, decimals=4):
    """Row (as a Series) matching an independent-component composition vector."""
    elements = elements if elements is not None else composition_columns(df)
    if index is None:
        index, decimals = build_index(df, elements, decimals)
    key = tuple(np.round(np.asarray(composition, dtype=float), decimals))
    row_index = index.get(key)
    if row_index is None:
        raise KeyError(f"composition {key} not found in table (elements={elements})")
    return df.iloc[row_index]


def chemical_potential(df, composition, elements=None, index=None, decimals=4):
    """Chemical-potential vector ``mu`` (Eq. 2) at a composition, one entry per
    independent component, in the order given by ``elements``."""
    elements = elements if elements is not None else composition_columns(df)
    row = row_at(df, composition, elements, index, decimals)
    return row[[f"Gm.x({e})" for e in elements]].to_numpy(dtype=float)


def atomic_mobilities(df, composition, elements=None, index=None, decimals=4):
    """Atomic mobility vector, species order ``[reference, *elements]``."""
    elements = elements if elements is not None else composition_columns(df)
    row = row_at(df, composition, elements, index, decimals)
    species = full_species(df, elements)
    return row[[f"Mob({s})" for s in species]].to_numpy(dtype=float)


def full_composition(composition):
    """Prepend the implicit reference-species fraction: ``[1 - sum(x), *x]``."""
    composition = np.asarray(composition, dtype=float)
    return np.concatenate([[1.0 - composition.sum()], composition])
