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

# Equilibrium tie-line for Fe20Mn10Ni20Co20Cu30 at 873 K, already public via the
# phase5d repository's README (computed there with a convex-hull toolkit that
# is not yet public on its own). Full 5-component vectors: [Fe, Mn, Ni, Co, Cu].
PHASE_A = np.array([0.26, 0.12, 0.25, 0.26, 0.11])  # Fe/Co/Ni-rich phase
PHASE_B = np.array([0.00, 0.03, 0.05, 0.01, 0.92])  # Cu-rich phase
EQUILIBRIUM_MODE = PHASE_A - PHASE_B  # tie-line direction, oblique Gibbs-space (5 components)
