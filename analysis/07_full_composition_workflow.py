"""Full per-composition workflow: every Sec. 2.1-2.4 analysis for one alloy,
in a single run (Figs. 3, 4, 5, 7).

Consolidates what analysis/02-05 do separately -- dispersion relation,
eigenmode bar-shards, misalignment angles, anisotropic Cu gradient-energy
sweep -- behind one call to :func:`early5d.workflow.analyze_composition`, so
exploring a single composition doesn't require re-deriving the Hessian/
mobility-matrix plumbing by hand.

Defaults to the manuscript's Fe20Mn10Ni20Co20Cu30 worked example (ALLOY_1):

    python 07_full_composition_workflow.py

Point it at a different composition on the same fine (1 at %) grid used by
the two bundled worked alloys -- only Fe15Mn20Ni20Co15Cu30 (ALLOY_2) and its
immediate finite-difference neighborhood actually have data, since the
bundled stencil file only covers those two points' local neighborhoods (see
the README's Data format section):

    python 07_full_composition_workflow.py --composition 0.20 0.20 0.15 0.30 --dataset stencils

To analyze an arbitrary composition anywhere in the Fe-Mn-Ni-Co-Cu simplex,
switch to the bundled 5 at % whole-simplex grid (coarser Hessian resolution,
but no restriction on which composition):

    python 07_full_composition_workflow.py --composition 0.15 0.25 0.10 0.25 --dataset coarse

Misalignment angles (Fig. 5) are only computed if an equilibrium tie-line is
given: --use-published-tie-line reuses the manuscript's own public
Fe20Mn10Ni20Co20Cu30 tie-line (only physically meaningful for that
composition), or pass your own via --phase-a/--phase-b (full 5-component
mole fractions of the two coexisting phases).

The bundled data is a single fixed temperature (873 K, see the README's Data
format section) -- --temperature only relabels the printed/plotted output; to
actually analyze a different temperature, point --data at your own Thermo-Calc
export at that temperature.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from early5d import calphad, kinetics, labels, plotting, workflow

from _common import (
    ALLOY_1, ALLOY_STENCILS, COARSE_GRID, ELEMENTS, GRID_SPACING, KAPPA_SCALAR,
    OUTPUT_DIR, PHASE_A, PHASE_B, TEMPERATURE_K,
)

FULL_ELEMENTS = ["Fe", *ELEMENTS]
KAPPA_CU_SWEEP = [5e-15, 1e-15, 5e-16, 1e-16, 5e-17, 1e-17]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--composition", type=float, nargs=4, metavar=("Mn", "Ni", "Co", "Cu"),
                         default=list(ALLOY_1["composition"]), help="mole fractions of the 4 independent components")
    parser.add_argument("--label", default=None, help="output-file label (default: derived from composition)")
    parser.add_argument("--dataset", choices=["stencils", "coarse"], default="stencils",
                         help="'stencils': fine 1 at%% grid, only around the two worked alloys; "
                              "'coarse': 5 at%% grid spanning the whole simplex")
    parser.add_argument("--data", default=None, help="path to your own CALPHAD export, overriding --dataset")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K,
                         help="label only -- see the module docstring")
    parser.add_argument("--kappa", type=float, default=KAPPA_SCALAR,
                         help="isotropic gradient-energy coefficient (J m^2/mol)")
    parser.add_argument("--aniso-sweep", dest="aniso_sweep", action="store_true", default=True,
                         help="also sweep the Cu gradient-energy coefficient (Fig. 7, on by default)")
    parser.add_argument("--no-aniso-sweep", dest="aniso_sweep", action="store_false")
    parser.add_argument("--use-published-tie-line", action="store_true",
                         help="use the manuscript's public tie-line (only meaningful for the default composition)")
    parser.add_argument("--phase-a", type=float, nargs=5, metavar=("Fe", "Mn", "Ni", "Co", "Cu"), default=None)
    parser.add_argument("--phase-b", type=float, nargs=5, metavar=("Fe", "Mn", "Ni", "Co", "Cu"), default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    composition = np.array(args.composition)
    label = args.label or labels.composition_label(composition, ELEMENTS, reference="Fe", unicode=False)

    if args.data is not None:
        data_path, grid_spacing = args.data, GRID_SPACING if args.dataset == "stencils" else 0.05
    elif args.dataset == "stencils":
        data_path, grid_spacing = ALLOY_STENCILS, GRID_SPACING
    else:
        data_path, grid_spacing = COARSE_GRID, 0.05
    kappa = args.kappa * np.eye(len(ELEMENTS))

    equilibrium_mode = None
    if args.use_published_tie_line:
        equilibrium_mode = PHASE_A - PHASE_B
    elif args.phase_a is not None and args.phase_b is not None:
        equilibrium_mode = np.array(args.phase_a) - np.array(args.phase_b)

    df = calphad.load_calphad_table(data_path)
    index, decimals = calphad.build_index(df, ELEMENTS, decimals=4)

    print(f"{label} at {args.temperature:.0f} K")
    result = workflow.analyze_composition(
        df, composition, kappa, elements=ELEMENTS, grid_spacing=grid_spacing,
        index=index, decimals=decimals, equilibrium_mode=equilibrium_mode, n_k=2000,
    )

    print(f"  xi0 = {result['xi0'] / 1000:.1f} kJ/mol, inside spinodal region: {result['is_spinodal']}")
    if not result["is_spinodal"]:
        print("  composition is thermodynamically stable -- no dispersion relation/CTLD to report or plot")
        return

    dispersion, eff = result["dispersion"], result["effective"]
    print(f"  eigenproblem: k_max = {dispersion['k_max'] * 1e-9:.2f} nm^-1   tau = {dispersion['tau']:.1f} s")
    print(f"  effective:    k_max = {eff['k_fastest'] * 1e-9:.2f} nm^-1   tau_eff = {eff['tau_eff']:.1f} s")

    # dispersion relation (Fig. 3)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plotting.plot_dispersion(ax, dispersion["wavenumbers"], dispersion["growth_rates"], label="eigenmode")
    plotting.plot_dispersion(ax, eff["wavenumbers"], eff["growth_rates"], label="effective")
    fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.16)
    out_path = os.path.join(OUTPUT_DIR, f"workflow_dispersion_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  dispersion-relation figure saved to {out_path}")

    # eigenmode bar-shards (Fig. 4)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plotting.plot_eigenmode_bars(ax, modes=result["bars"]["modes"], labels=result["bars"]["labels"], elements=FULL_ELEMENTS)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"workflow_bars_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  eigenmode-bars figure saved to {out_path}")

    # misalignment angles (Fig. 5), only if an equilibrium tie-line was given
    if result["misalignment"] is not None:
        m = result["misalignment"]
        for pair, angles in m["angles"].items():
            print(f"  {pair}: {angles['euclidean']:.1f} deg (Euclidean), {angles['weighted']:.1f} deg (M^-1-metric)")
        plot_labels = [f"{l} ({l[0].upper()})" for l in m["labels"]]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        plotting.plot_misalignment(ax, m["vectors"], plot_labels, metric=m["metric"])
        out_path = os.path.join(OUTPUT_DIR, f"workflow_misalignment_{label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  misalignment-angles figure saved to {out_path}")

    # anisotropic Cu gradient-energy sweep (Fig. 7)
    if args.aniso_sweep:
        cu_index = ELEMENTS.index("Cu")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        for kappa_cu in KAPPA_CU_SWEEP:
            kappa_sweep = args.kappa * np.eye(len(ELEMENTS))
            kappa_sweep[cu_index, cu_index] = kappa_cu
            sweep = kinetics.solve_dispersion_relation(result["Hg"], result["M"], kappa_sweep, n_k=2000)
            sweep_label = rf"$\kappa_\mathrm{{Cu}} = {plotting.latex_scientific(kappa_cu)}$"
            plotting.plot_dispersion(ax, sweep["wavenumbers"], sweep["growth_rates"], label=sweep_label)
        ax.legend(frameon=False, fontsize=10)
        fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.16)
        out_path = os.path.join(OUTPUT_DIR, f"workflow_aniso_sweep_{label}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  anisotropic-sweep figure saved to {out_path}")


if __name__ == "__main__":
    main()
