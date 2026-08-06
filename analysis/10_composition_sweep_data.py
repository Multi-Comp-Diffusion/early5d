"""Full composition-space sweep of every early-stage parameter, saved as a
data file rather than just plotted (Figs. 1/2, Sec. 4).

For every composition on the bundled 5 at % coarse grid inside the spinodal
region, computes xi0, M_eff, kappa_eff, tau_eff, tau, k_max under the paper's
own (Fe) reference species, *and* how M_eff/kappa_eff/tau_eff/tau shift under
every alternate reference-species choice (early5d.gauge_sensitivity) --
quantifying the gauge-dependence discussed in Sec. 4 across the whole
composition space, not just the two worked alloys. Saves the result as a
compressed .npz archive (a zip file of named numpy arrays) instead of (or
alongside) a figure, for downstream analysis/plotting outside this repo.

This pays a real cost: a reference-sensitivity sweep is ~6x a single-field
scan (one direct computation, plus one recomputation per candidate reference
species), on top of already being the more expensive multi-field version of
what analysis/06 and 08 do -- expect several minutes on the bundled coarse
grid.

    python 10_composition_sweep_data.py

Loading the result back:

    from early5d import composition_sweep
    sweep = composition_sweep.load_sweep("analysis/output/composition_sweep.npz")
    sweep["tau_eff"]                  # (P,) at the paper's own reference species
    sweep["tau_eff_by_reference"]     # (P, 5): same, recomputed per reference_species entry
    sweep["reference_species"]       # ["Fe", "Mn", "Ni", "Co", "Cu"]
"""

import time

import numpy as np

from early5d import calphad, composition_sweep, stats

from _common import COARSE_GRID, ELEMENTS, KAPPA_SCALAR, OUTPUT_DIR

COARSE_GRID_SPACING = 0.05
N_K_MAP = 30  # coarse wavenumber sampling (cf. paper's n_k=100 on the 1 at % grid)


def main():
    df = calphad.load_calphad_table(COARSE_GRID)
    index, decimals = calphad.build_index(df, ELEMENTS, decimals=4)
    full_species = calphad.full_species(df, ELEMENTS)

    t0 = time.time()
    sweep = composition_sweep.sweep_composition_grid(
        df, KAPPA_SCALAR, ELEMENTS, full_species, COARSE_GRID_SPACING,
        index=index, decimals=decimals, n_k=N_K_MAP,
    )
    n_points = len(sweep["tau"])
    print(f"Evaluated {len(df)} grid points ({time.time() - t0:.1f} s); {n_points} usable (spinodal, non-degenerate).")

    print("\nCTLD (tau), paper's own reference:", stats.log_scale_summary(sweep["tau"]))
    print("\neffective-CTLD (tau_eff), paper's own reference:", stats.log_scale_summary(sweep["tau_eff"]))

    print(f"\nreference_species order: {list(sweep['reference_species'])}")
    spread = 100 * (sweep["tau_eff_by_reference"].max(axis=1) - sweep["tau_eff_by_reference"].min(axis=1)) / sweep["tau_eff"]
    print(f"tau_eff spread across all 5 reference choices, relative to the paper's own value: "
          f"median {np.median(spread):.0f}%, q75 {np.percentile(spread, 75):.0f}%, max {spread.max():.0f}%")
    assert np.all(sweep["tau_eff_by_reference"] <= sweep["tau_by_reference"] + 1e-6)
    print("tau_eff <= tau holds at every point, for every reference-species choice.")

    out_path = f"{OUTPUT_DIR}/composition_sweep.npz"
    composition_sweep.save_sweep(out_path, sweep)
    print(f"\nsweep data saved to {out_path}")


if __name__ == "__main__":
    main()
