"""Bar-shard comparison of thermo / mobility / growth / equilibrium modes (Fig. 4).

Uses the gauge-invariant thermo mode (Eq. 8) -- its driving force matches the
paper's reported xi0 = -14.3 kJ/mol for Fe20Mn10Ni20Co20Cu30 -- together with
the mobility matrix's dominant eigenvector and the fastest-growing
thermo-kinetic eigenmode. The equilibrium mode is the tie-line direction
between the two coexisting phases, taken from phase5d's already-published
example composition for this alloy (equilibrium-phase computation itself is
out of scope here -- see the README's Scope section).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from early5d import calphad, hessian, kinetics, linalg, mobility, plotting

from _common import ALLOY_1, ALLOY_STENCILS, ELEMENTS, EQUILIBRIUM_MODE, GRID_SPACING, KAPPA_ISOTROPIC, OUTPUT_DIR

FULL_ELEMENTS = ["Fe", *ELEMENTS]


def main():
    df = calphad.load_calphad_table(ALLOY_STENCILS)
    index, decimals = calphad.build_index(df, ELEMENTS)
    composition = ALLOY_1["composition"]

    Hg = hessian.hessian_at(df, composition, GRID_SPACING, ELEMENTS, index, decimals)
    mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

    thermo_v0, xi0 = hessian.thermo_mode(Hg)
    mobility_mode, _ = mobility.dominant_mode(M)
    dispersion = kinetics.solve_dispersion_relation(Hg, M, KAPPA_ISOTROPIC, n_k=2000)

    print(f"{ALLOY_1['label']}: xi0 = {xi0 / 1000:.1f} kJ/mol (paper: -14.3), "
          f"mu0 = {dispersion['mu0_max']:.2e} m^2/s (paper: -7.2e-21)")

    thermo_bar = linalg.normalized_full_mode(thermo_v0)
    mobility_bar = linalg.normalized_full_mode(mobility_mode)
    growth_bar = linalg.normalized_full_mode(dispersion["growth_mode"])
    equilibrium_bar = linalg.align_mode(EQUILIBRIUM_MODE) / np.linalg.norm(EQUILIBRIUM_MODE)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plotting.plot_eigenmode_bars(
        ax,
        modes=[thermo_bar, mobility_bar, growth_bar, equilibrium_bar],
        labels=["thermo", "mobility", "growth", "equilibrium"],
        elements=FULL_ELEMENTS,
    )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"eigenmode_bars_{ALLOY_1['label']}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure saved to {out_path}")


if __name__ == "__main__":
    main()
