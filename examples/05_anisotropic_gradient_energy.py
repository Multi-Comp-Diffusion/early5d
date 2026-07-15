"""Sensitivity of the dispersion relation to an anisotropic gradient-energy
coefficient (Fig. 7).

Varies the Cu diagonal entry of the gradient-energy tensor kappa while
holding the other three components fixed at the isotropic reference value,
and re-solves the thermo-kinetic eigenproblem for both worked alloys. Cu
dominates the interfacial-energy contribution to the early-stage dynamics in
this alloy system (matching the paper's finding), shifting the fastest-growing
wavenumber and CTLD noticeably; Mn and Ni have negligible effect (not swept
here for brevity).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from early5d import calphad, hessian, kinetics, mobility, plotting

from _common import ALLOY_1, ALLOY_2, ALLOY_STENCILS, ELEMENTS, GRID_SPACING, KAPPA_SCALAR, OUTPUT_DIR

KAPPA_CU_VALUES = [5e-15, 1e-15, 5e-16, 1e-16, 5e-17, 1e-17]
CU_INDEX = ELEMENTS.index("Cu")


def kappa_with_cu(kappa_cu):
    kappa = KAPPA_SCALAR * np.eye(len(ELEMENTS))
    kappa[CU_INDEX, CU_INDEX] = kappa_cu
    return kappa


def analyse(alloy, df, index, decimals):
    composition = alloy["composition"]
    Hg = hessian.hessian_at(df, composition, GRID_SPACING, ELEMENTS, index, decimals)
    mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    print(f"\n{alloy['label']}")
    for kappa_cu in KAPPA_CU_VALUES:
        kappa = kappa_with_cu(kappa_cu)
        result = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=2000)
        label = rf"$\kappa_\mathrm{{Cu}} = {plotting.latex_scientific(kappa_cu)}$"
        print(f"  kappa_Cu={kappa_cu:.0e}: k_max={result['k_max'] * 1e-9:.2f} nm^-1, tau={result['tau']:.1f} s")
        plotting.plot_dispersion(ax, result["wavenumbers"], result["growth_rates"], label=label)

    ax.legend(frameon=False, fontsize=10)
    # pinned margins (see 02_dispersion_relation.py) so both alloys' panels share
    # the same plot-box geometry regardless of each one's tick-label width
    fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.16)
    out_path = os.path.join(OUTPUT_DIR, f"anisotropic_gradient_energy_{alloy['label']}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  figure saved to {out_path}")


def main():
    df = calphad.load_calphad_table(ALLOY_STENCILS)
    index, decimals = calphad.build_index(df, ELEMENTS)

    for alloy in (ALLOY_1, ALLOY_2):
        analyse(alloy, df, index, decimals)


if __name__ == "__main__":
    main()
