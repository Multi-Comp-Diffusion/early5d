"""CTLD and effective-CTLD mapped across the composition simplex (Figs. 1/2).

Uses the bundled 5 at % coarse subsample of the full CALPHAD grid (real data,
just coarser than the paper's 1 at % maps -- see the README's Data format
section for how to point this at your own full-resolution grid). For every
composition inside the spinodal region, computes the effective-CTLD (cheap,
closed-form) and the full eigenproblem-based CTLD, then reports log-scale
summary statistics and plots one ternary cross-section (Fe and Co both held
at 20 at %, matching the paper's Fig. 1b/2b panel).

Requires the optional ``python-ternary`` dependency: ``pip install -e ".[ternary]"``.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from early5d import calphad, effective, hessian, kinetics, mobility, plotting, stats

from _common import COARSE_GRID, ELEMENTS, KAPPA_ISOTROPIC, OUTPUT_DIR

COARSE_GRID_SPACING = 0.05
N_K_MAP = 30  # coarse wavenumber sampling for the full eigenproblem CTLD (cf. paper's n_k=100 on the 1 at % grid)


def main():
    df = calphad.load_calphad_table(COARSE_GRID)
    index, decimals = calphad.build_index(df, ELEMENTS, decimals=4)
    compositions = df[[f"x({e})" for e in ELEMENTS]].to_numpy(dtype=float)

    tau_map, tau_eff_map, valid_compositions = [], [], []
    t0 = time.time()
    for composition in compositions:
        try:
            Hg = hessian.hessian_at(df, composition, COARSE_GRID_SPACING, ELEMENTS, index, decimals)
        except (KeyError, ValueError):
            continue  # composition sits at the extracted grid's edge; skip

        if not hessian.is_spinodal(Hg):
            continue

        mobilities = calphad.atomic_mobilities(df, composition, ELEMENTS, index, decimals)
        M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

        tau_eff = effective.solve_effective_dispersion(Hg, M, KAPPA_ISOTROPIC)["tau_eff"]
        tau = kinetics.ctld(Hg, M, KAPPA_ISOTROPIC, n_k=N_K_MAP)

        tau_map.append(tau)
        tau_eff_map.append(tau_eff)
        valid_compositions.append(composition)

    valid_compositions = np.array(valid_compositions)
    tau_map, tau_eff_map = np.array(tau_map), np.array(tau_eff_map)
    print(f"Evaluated {len(compositions)} grid points ({time.time() - t0:.1f} s); "
          f"{len(tau_map)} inside the spinodal region.")

    print("\nCTLD (tau):", stats.log_scale_summary(tau_map))
    print("\neffective-CTLD (tau_eff):", stats.log_scale_summary(tau_eff_map))

    # Ternary cross-section: Fe = Co = 20 at % (Mn/Ni/Cu vary), matching Fig. 1b/2b
    fe = 1.0 - valid_compositions.sum(axis=1)
    co = valid_compositions[:, ELEMENTS.index("Co")]
    on_slice = np.isclose(fe, 0.20, atol=1e-6) & np.isclose(co, 0.20, atol=1e-6)

    if on_slice.sum() < 3:
        print("\nToo few points on the Fe=Co=20% slice in the coarse grid to plot; skipping ternary figure.")
        return

    slice_compositions = valid_compositions[on_slice]
    ternary_fractions = slice_compositions[:, [ELEMENTS.index(e) for e in ("Ni", "Mn", "Cu")]]
    ternary_fractions = ternary_fractions / ternary_fractions.sum(axis=1, keepdims=True)

    fig, tax = plotting.ternary_axes(scale=100)
    plotting.ternary_scatter(tax, ternary_fractions, np.log10(tau_map[on_slice]), scale=100)
    plotting.ternary_corner_labels(tax, right="Ni", top="Mn", left="Cu")
    plotting.ternary_annotation(tax, scale=100, constant_elements={"Fe": 0.20, "Co": 0.20})
    out_path = f"{OUTPUT_DIR}/ctld_ternary_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nternary figure saved to {out_path}")

    # colorbar saved as its own figure, matching the manuscript's convention
    fig_cbar, _ = plotting.ternary_colorbar(np.log10(tau_map[on_slice]), label=r"CTLD $\tau$ (s)", log_tag=True)
    cbar_path = f"{OUTPUT_DIR}/ctld_ternary_map_colorbar.png"
    fig_cbar.savefig(cbar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cbar)
    print(f"colorbar figure saved to {cbar_path}")


if __name__ == "__main__":
    main()
