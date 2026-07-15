"""Pairwise 20:20 ternary composition maps for a chosen early-stage parameter
(Figs. 1/2, generalized to every species pair).

The manuscript's Fig. 1b/2b panel (Fe = Co = 20 at %, Mn/Ni/Cu varying) is one
of ``C(5, 2) = 10`` possible ternary cross-sections obtainable by holding a
different pair of the five species (Fe, Mn, Ni, Co, Cu) at 20 at % each.
Scans the bundled 5 at % coarse grid once for the chosen field, then produces
one ternary map (plus one shared colorbar) per pair that has enough grid
points on that slice, skipping the rest -- see the README's Data format
section for pointing this at your own full-resolution (1 at %) grid.

    python 08_pairwise_ternary_maps.py                    # CTLD (tau), the default
    python 08_pairwise_ternary_maps.py --field tau_eff     # effective-CTLD
    python 08_pairwise_ternary_maps.py --field xi0         # thermodynamic driving force
    python 08_pairwise_ternary_maps.py --field k_max       # fastest-growing wavenumber

Each panel's remaining (varying) three species are placed top/right/left in
their relative order within ``[Fe, Mn, Ni, Co, Cu]`` (skipping the two fixed
ones) -- a fixed, consistent convention across all ten panels, rather than
the per-panel hand-tuned corner arrangement used for the single Fe/Co panel
in examples/06_ctld_composition_maps.py.

Requires the optional python-ternary dependency: pip install -e ".[ternary]"
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from early5d import calphad, plotting, stats, ternary_workflow

from _common import COARSE_GRID, ELEMENTS, KAPPA_SCALAR, OUTPUT_DIR

COARSE_GRID_SPACING = 0.05
N_K_MAP = 30  # coarse wavenumber sampling (cf. paper's n_k=100 on the 1 at % grid)

FIELD_LABELS = {
    "tau": r"CTLD $\tau$ (s)",
    "tau_eff": r"effective-CTLD $\tau_\mathrm{eff}$ (s)",
    "xi0": r"driving force $\xi_0$ (J/mol)",
    "k_max": r"wavenumber $k_\mathrm{max}$ (m$^{-1}$)",
}
LOG_FIELDS = {"tau", "tau_eff"}  # plotted on a log10 color scale; xi0/k_max are linear


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", choices=sorted(ternary_workflow.FIELDS), default="tau")
    parser.add_argument("--kappa", type=float, default=KAPPA_SCALAR, help="isotropic gradient-energy coefficient (J m^2/mol)")
    parser.add_argument("--fixed-fraction", type=float, default=0.20, help="mole fraction each of the pair is held at")
    return parser.parse_args()


def main():
    args = parse_args()
    kappa = args.kappa * np.eye(len(ELEMENTS))

    df = calphad.load_calphad_table(COARSE_GRID)
    index, decimals = calphad.build_index(df, ELEMENTS, decimals=4)
    species = calphad.full_species(df, ELEMENTS)

    t0 = time.time()
    full_compositions, values = ternary_workflow.scan_composition_grid(
        df, kappa, elements=ELEMENTS, grid_spacing=COARSE_GRID_SPACING,
        index=index, decimals=decimals, field=args.field, n_k=N_K_MAP,
    )
    print(f"{args.field}: evaluated {len(df)} grid points ({time.time() - t0:.1f} s); "
          f"{len(values)} inside the spinodal region.")
    if args.field in LOG_FIELDS:
        print(stats.log_scale_summary(values))
    else:
        print({"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())})

    slices = ternary_workflow.pairwise_ternary_slices(full_compositions, values, species, fixed_fraction=args.fixed_fraction)
    n_pairs = len(species) * (len(species) - 1) // 2
    print(f"\n{len(slices)} / {n_pairs} species pairs have enough grid points at "
          f"{args.fixed_fraction * 100:.0f}:{args.fixed_fraction * 100:.0f} at % to plot.")

    if not slices:
        return

    label = FIELD_LABELS[args.field]
    for (species_a, species_b), slice_data in slices.items():
        e1, e2, e3 = slice_data["ternary_elements"]
        plot_values = np.log10(slice_data["values"]) if args.field in LOG_FIELDS else slice_data["values"]

        fig, tax = plotting.ternary_axes(scale=100)
        plotting.ternary_scatter(tax, slice_data["ternary_compositions"], plot_values, scale=100)
        plotting.ternary_corner_labels(tax, right=e2, top=e1, left=e3)
        plotting.ternary_annotation(tax, scale=100, constant_elements={species_a: args.fixed_fraction, species_b: args.fixed_fraction})
        out_path = os.path.join(OUTPUT_DIR, f"pairwise_ternary_{args.field}_{species_a}{species_b}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {species_a}-{species_b} ({len(slice_data['values'])} points): {out_path}")

    # one shared colorbar, scaled to the full (unsliced) grid so every panel is comparable
    plot_values_all = np.log10(values) if args.field in LOG_FIELDS else values
    fig_cbar, _ = plotting.ternary_colorbar(plot_values_all, label=label, log_tag=args.field in LOG_FIELDS)
    cbar_path = os.path.join(OUTPUT_DIR, f"pairwise_ternary_{args.field}_colorbar.png")
    fig_cbar.savefig(cbar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cbar)
    print(f"\nshared colorbar saved to {cbar_path}")


if __name__ == "__main__":
    main()
