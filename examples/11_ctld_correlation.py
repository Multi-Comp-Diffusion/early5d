"""CTLD vs. effective-CTLD correlation plot, for every reference-species
choice (Sec. 4 discussion).

Reproduces the manuscript's original tau-vs-effective-tau scatter (log-log,
with a 1:1 reference line) across the whole composition space, generalized to
show all five possible reference-species choices at once rather than just
the paper's own (Fe) convention -- points consistently below the 1:1 line
show tau_eff underestimating tau; how far below (and how that changes across
reference-species choices) is exactly the gauge-dependence quantified in
examples/09 and 10.

Reuses examples/10's sweep if it already produced
examples/output/composition_sweep.npz (fast); otherwise runs it fresh here
(slow, several minutes -- see examples/10_composition_sweep_data.py's
docstring).
"""

import os

import matplotlib.pyplot as plt

from early5d import calphad, composition_sweep, plotting

from _common import COARSE_GRID, ELEMENTS, KAPPA_SCALAR, OUTPUT_DIR

COARSE_GRID_SPACING = 0.05
N_K_MAP = 30
SWEEP_PATH = os.path.join(OUTPUT_DIR, "composition_sweep.npz")
COLORS = {"Fe": "tab:blue", "Mn": "tab:orange", "Ni": "tab:green", "Co": "tab:red", "Cu": "tab:purple"}


def get_sweep():
    if os.path.exists(SWEEP_PATH):
        print(f"reusing existing sweep at {SWEEP_PATH}")
        return composition_sweep.load_sweep(SWEEP_PATH)

    print(f"no existing sweep at {SWEEP_PATH}; computing fresh (several minutes)")
    df = calphad.load_calphad_table(COARSE_GRID)
    index, decimals = calphad.build_index(df, ELEMENTS, decimals=4)
    full_species = calphad.full_species(df, ELEMENTS)
    sweep = composition_sweep.sweep_composition_grid(
        df, KAPPA_SCALAR, ELEMENTS, full_species, COARSE_GRID_SPACING,
        index=index, decimals=decimals, n_k=N_K_MAP,
    )
    composition_sweep.save_sweep(SWEEP_PATH, sweep)
    return sweep


def main():
    sweep = get_sweep()
    reference_species = sweep["reference_species"]
    print(f"{len(sweep['tau'])} points; reference_species = {reference_species}")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for i, species in enumerate(reference_species):
        plotting.plot_ctld_correlation(
            ax, sweep["tau_eff_by_reference"][:, i], sweep["tau_by_reference"][:, i],
            label=species, color=COLORS.get(species), marker_size=6, alpha=0.25,
        )
    plotting.plot_ctld_reference_line(ax)
    ax.legend(frameon=False, fontsize=plotting.LEGEND_FONTSIZE, markerscale=3, loc="upper left")
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "ctld_correlation_by_reference.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure saved to {out_path}")


if __name__ == "__main__":
    main()
