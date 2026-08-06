"""Misalignment angles between thermo, growth, and equilibrium modes (Fig. 5).

Computes the Euclidean and inverse-mobility-weighted (M^-1-metric) angles
(Eq. 19) between the thermodynamic instability direction, the fastest-growing
thermo-kinetic mode, and the equilibrium tie-line, for Fe20Mn10Ni20Co20Cu30.

This is illustrative rather than an exact reproduction of Fig. 5: the modes
here are compared directly in the reduced (independent-component) composition
space, whereas the paper additionally re-expresses them in an orthogonalized
Cartesian basis before measuring angles. The qualitative result is the same
(the equilibrium mode is markedly more misaligned from the thermo/growth
modes than they are from each other), but exact angle values will differ
slightly from the paper's.

The equilibrium tie-line composition is an external input -- see
``analysis/_common.py`` and the README's Scope section.
"""

import os

import matplotlib.pyplot as plt

from early5d import calphad, hessian, kinetics, misalignment, mobility, plotting

from _common import ALLOY_1, ALLOY_STENCILS, ELEMENTS, EQUILIBRIUM_MODE, GRID_SPACING, KAPPA_ISOTROPIC, OUTPUT_DIR


def main():
    df = calphad.load_calphad_table(ALLOY_STENCILS)
    index, decimals = calphad.build_index(df, ELEMENTS)
    composition = ALLOY_1["composition"]

    Hg = hessian.hessian_at(df, composition, GRID_SPACING, ELEMENTS, index, decimals)
    mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals)
    M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

    thermo_mode, _ = hessian.thermo_mode(Hg)
    dispersion = kinetics.solve_dispersion_relation(Hg, M, KAPPA_ISOTROPIC, n_k=2000)
    growth_mode = dispersion["growth_mode"]
    equilibrium_mode = EQUILIBRIUM_MODE[1:]  # drop the implicit reference (Fe) component

    M_inv = misalignment.inverse_mobility_metric(M)
    vectors = [thermo_mode, growth_mode, equilibrium_mode]
    pair_labels = [("thermo", "growth"), ("thermo", "equilibrium"), ("growth", "equilibrium")]

    print(f"{ALLOY_1['label']} misalignment angles:")
    for (a, b) in pair_labels:
        va = vectors[["thermo", "growth", "equilibrium"].index(a)]
        vb = vectors[["thermo", "growth", "equilibrium"].index(b)]
        euclidean = misalignment.angle_between(va, vb)
        weighted = misalignment.angle_between(va, vb, metric=M_inv)
        print(f"  {a}-{b}: {euclidean:.1f} deg (Euclidean), {weighted:.1f} deg (M^-1-metric)")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plotting.plot_misalignment(ax, vectors, ["thermo (T)", "growth (K)", "equilibrium (E)"], metric=M_inv)
    out_path = os.path.join(OUTPUT_DIR, f"misalignment_angles_{ALLOY_1['label']}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {out_path}")


if __name__ == "__main__":
    main()
