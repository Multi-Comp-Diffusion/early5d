"""Shared constants for the example scripts (not part of the installed package).

The two worked alloys and gradient-energy coefficient match the manuscript's
Fe-Mn-Ni-Co-Cu case study at 873 K.
"""

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOY_STENCILS = os.path.join(DATA_DIR, "tchea4_mobfe5_873k_alloy_stencils.dat")
COARSE_GRID = os.path.join(DATA_DIR, "tchea4_mobfe5_873k_coarse5pct.dat")
EQUILIBRIUM_PHASES = os.path.join(DATA_DIR, "tchea4_mobfe5_873k_equilibrium_phases.dat")

ELEMENTS = ["Mn", "Ni", "Co", "Cu"]  # independent components; Fe is the implicit reference
GRID_SPACING = 0.01
TEMPERATURE_K = 873
KAPPA_SCALAR = 5e-16  # J m^2 / mol, isotropic in the (Mn, Ni, Co, Cu) reduced coordinates
KAPPA_ISOTROPIC = KAPPA_SCALAR * np.eye(len(ELEMENTS))

ALLOY_1 = {
    "label": "Fe20Mn10Ni20Co20Cu30",
    "composition": np.array([0.10, 0.20, 0.20, 0.30]),  # (Mn, Ni, Co, Cu)
}
ALLOY_2 = {
    "label": "Fe15Mn20Ni20Co15Cu30",
    "composition": np.array([0.20, 0.20, 0.15, 0.30]),
}

# Equilibrium tie-lines at 873 K under the unmodified (chemical-only, no
# elastic contribution) Gibbs free energy, Table 2 of Marschall, Murugan &
# Darvishi Kamachali, Materialia 44 (2025) 102546 -- "A2" there is
# Fe20Mn10Ni20Co20Cu30 (ALLOY_1), "A5" is Fe15Mn20Ni20Co15Cu30 (ALLOY_2).
# Full 5-component vectors: [Fe, Mn, Ni, Co, Cu].
PHASE_A = np.array([0.26, 0.12, 0.25, 0.26, 0.11])  # A2's FeCo-rich phase
PHASE_B = np.array([0.00, 0.03, 0.05, 0.01, 0.92])  # A2's Cu-rich phase
EQUILIBRIUM_MODE = PHASE_A - PHASE_B  # tie-line direction, oblique Gibbs-space (5 components)

# PHASE_B's published mole fractions sum to 1.01, not 1.0 -- a rounding
# artifact of the paper's 2-decimal display precision that doesn't matter for
# EQUILIBRIUM_MODE above (only ever used as an approximate tie-line
# direction) but does matter for an *exact* CALPHAD row lookup (needed to
# evaluate the mobility matrix there, examples/12): this is PHASE_B's
# nearest valid grid point (Cu 0.92 -> 0.91; Fe unaffected, still 0.00).
PHASE_B_GRID = np.array([0.00, 0.03, 0.05, 0.01, 0.91])

PHASE_A2 = np.array([0.19, 0.22, 0.23, 0.18, 0.18])  # A5's FeCo-rich phase (sums to 1.00 exactly)
PHASE_B2 = np.array([0.00, 0.08, 0.06, 0.01, 0.85])  # A5's Cu-rich phase (sums to 1.00 exactly)
