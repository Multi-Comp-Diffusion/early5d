"""Gibbs-energy Hessian, gauge-invariant orthogonalization, and the spinodal
sign test (Secs. 2.1/2.2 of the manuscript).

For each of the two worked alloys, builds the finite-difference Hessian from
the bundled CALPHAD stencil, orthogonalizes it to the gauge-invariant
Cartesian basis (Eq. 8), and reports the thermodynamic driving force (the
Hessian's most negative eigenvalue) and the thermo mode.
"""

import numpy as np

from early5d import calphad, hessian

from _common import ALLOY_1, ALLOY_2, ALLOY_STENCILS, ELEMENTS, GRID_SPACING


def analyse(alloy, df, index, decimals):
    Hg = hessian.hessian_at(df, alloy["composition"], GRID_SPACING, ELEMENTS, index, decimals)
    spinodal = hessian.is_spinodal(Hg)
    v0, xi0 = hessian.thermo_mode(Hg)

    print(f"\n{alloy['label']}")
    print(f"  Hessian symmetric: {np.allclose(Hg, Hg.T)}")
    print(f"  inside spinodal:   {spinodal}")
    print(f"  xi0 (thermo driving force): {xi0 / 1000:.1f} kJ/mol")
    print(f"  thermo mode (Mn, Ni, Co, Cu direction): {np.round(v0, 3)}")


def main():
    df = calphad.load_calphad_table(ALLOY_STENCILS)
    index, decimals = calphad.build_index(df, ELEMENTS)

    for alloy in (ALLOY_1, ALLOY_2):
        analyse(alloy, df, index, decimals)


if __name__ == "__main__":
    main()
