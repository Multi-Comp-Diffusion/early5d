"""Growth-rate dispersion relation: eigenproblem-based vs. effective (Fig. 3).

For each worked alloy: solves the full thermo-kinetic eigenproblem (Eq. 9) to
get the CTLD, and compares it against the cheap single-mode projection
(Eqs. 14-17), reproducing the paper's published values exactly (within the
resolution of the bundled 1 at % local stencil): tau = 15.0 s / tau_eff = 8.8 s
for Fe20Mn10Ni20Co20Cu30, tau = 50.3 s / tau_eff = 15.3 s for
Fe15Mn20Ni20Co15Cu30 (both at 873 K). Both quantities use the same isotropic
(kappa_scalar * I) gradient-energy tensor in the reduced (Mn, Ni, Co, Cu)
coordinates; see :mod:`early5d.effective` for why the single-mode projection
uses the untransformed Hessian's own eigenvector rather than the
gauge-invariant :func:`early5d.hessian.thermo_mode`.
"""

import os

import matplotlib.pyplot as plt

from early5d import calphad, effective, hessian, kinetics, mobility, plotting

from _common import ALLOY_1, ALLOY_2, ALLOY_STENCILS, ELEMENTS, GRID_SPACING, KAPPA_ISOTROPIC, OUTPUT_DIR


def analyse(alloy, df, index, decimals):
    composition = alloy["composition"]
    Hg = hessian.hessian_at(df, composition, GRID_SPACING, ELEMENTS, index, decimals)
    mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

    eigenmode = kinetics.solve_dispersion_relation(Hg, M, KAPPA_ISOTROPIC, n_k=2000)
    eff = effective.solve_effective_dispersion(Hg, M, KAPPA_ISOTROPIC, n_k=2000)

    print(f"\n{alloy['label']}")
    print(f"  eigenproblem: k_max = {eigenmode['k_max'] * 1e-9:.2f} nm^-1   tau = {eigenmode['tau']:.1f} s")
    print(f"  effective:    k_max = {eff['k_fastest'] * 1e-9:.2f} nm^-1   tau_eff = {eff['tau_eff']:.1f} s")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plotting.plot_dispersion(ax, eigenmode["wavenumbers"], eigenmode["growth_rates"], label="eigenmode")
    plotting.plot_dispersion(ax, eff["wavenumbers"], eff["growth_rates"], label="effective")
    # pinned margins (rather than tight_layout, which would size the axes box to each
    # alloy's tick-label width individually) so both alloys' panels share the same
    # plot-box geometry and are visually comparable side by side
    fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.16)
    out_path = os.path.join(OUTPUT_DIR, f"dispersion_relation_{alloy['label']}.png")
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
