"""Dominant mobility eigenmode comparison: nominal (parent) alloy vs. its two
fully-decomposed equilibrium phases, for both worked alloys (Fig. 4, extended).

Example 03 compares the thermo/mobility/growth/equilibrium modes, all
evaluated *at the nominal composition* -- the mobility mode there describes
the fastest-diffusing direction of the parent alloy before decomposition.
This example instead evaluates the dominant mobility eigenmode separately at
each of the two coexisting equilibrium phase compositions, to see
whether/how the fastest-diffusing direction shifts once the alloy has fully
separated -- for both worked alloys, one figure each.

Equilibrium phases at 873 K (unmodified, chemical-only Gibbs free energy)
are Table 2 of Marschall, Murugan & Darvishi Kamachali, Materialia 44 (2025)
102546: "A2" there is Fe20Mn10Ni20Co20Cu30 (ALLOY_1, PHASE_A/PHASE_B_GRID),
"A5" is Fe15Mn20Ni20Co15Cu30 (ALLOY_2, PHASE_A2/PHASE_B2) -- see _common.py.

The mobility matrix needs no finite-difference neighbors (unlike the
Hessian), so this only requires the exact CALPHAD row at each composition --
bundled here as a small, separate 4-row data extract (data/
tchea4_mobfe5_873k_equilibrium_phases.dat) alongside the alloy stencils.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from early5d import calphad, equilibrium_mobility, plotting

from _common import (
    ALLOY_1, ALLOY_2, ALLOY_STENCILS, ELEMENTS, EQUILIBRIUM_PHASES, OUTPUT_DIR,
    PHASE_A, PHASE_A2, PHASE_B2, PHASE_B_GRID,
)

FULL_ELEMENTS = ["Fe", *ELEMENTS]
# matplotlib's default mathtext doesn't support \bm (only real LaTeX + the bm
# package does); \mathbf is the closest built-in equivalent for this figure
PLOT_LABELS = [r"$\mathbf{X}_0$", r"$\mathbf{X}_\mathrm{FeCo}$", r"$\mathbf{X}_\mathrm{Cu}$"]


def analyse(alloy, phase_a, phase_b, df, index, decimals):
    compositions = {
        "nominal": alloy["composition"],
        "phase A (FeCo-rich)": phase_a[1:],  # drop the implicit Fe entry -> reduced (Mn, Ni, Co, Cu) form
        "phase B (Cu-rich)": phase_b[1:],
    }
    result = equilibrium_mobility.compare_mobility_modes(df, compositions, ELEMENTS, index, decimals)

    print(f"\n{alloy['label']}")
    for label, mode, eigenvalues, gap in zip(
        result["labels"], result["modes"], result["eigenvalues"], result["eigenvalue_gaps"]
    ):
        print(f"  {label}: {dict(zip(FULL_ELEMENTS, mode.round(3)))}")
        print(f"    eigenvalues (mol m^2/(J s), descending) = {[f'{ev:.3e}' for ev in eigenvalues]}  "
              f"gap (lambda_1-lambda_2)/lambda_1 = {gap:.1%}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plotting.plot_eigenmode_bars(
        ax, modes=result["modes"], labels=PLOT_LABELS, elements=FULL_ELEMENTS,
        ylabel="normalized mobility mode", eigenvalues=result["eigenvalues"],
    )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"equilibrium_mobility_modes_{alloy['label']}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  figure saved to {out_path}")


def main():
    df = pd.concat([calphad.load_calphad_table(ALLOY_STENCILS), calphad.load_calphad_table(EQUILIBRIUM_PHASES)],
                    ignore_index=True)
    index, decimals = calphad.build_index(df, ELEMENTS)

    analyse(ALLOY_1, PHASE_A, PHASE_B_GRID, df, index, decimals)
    analyse(ALLOY_2, PHASE_A2, PHASE_B2, df, index, decimals)


if __name__ == "__main__":
    main()
