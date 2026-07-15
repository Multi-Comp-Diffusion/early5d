"""Sensitivity of tau_eff (and M_eff, kappa_eff, tau) to the choice of
reference (dependent) species (Sec. 4 discussion).

The effective quantities (Eqs. 15-17) project onto v0, the eigenvector of the
*untransformed* Hessian -- unlike the gauge-invariant thermo mode (Eq. 8)
used elsewhere in this package, v0 depends on which species is treated as
implicit/dependent (Fe, in the paper's own convention). This recomputes
tau_eff for both worked alloys under every alternate choice of reference
species and plots the spread, quantifying that dependence directly rather
than just asserting it.

Two things are worth checking alongside the bar chart (both printed below):
kappa_eff stays *exactly* fixed regardless of reference species (an algebraic
identity whenever kappa = kappa_scalar * I, true for any v0), and the
provable bound tau_eff <= tau holds for every reference choice -- the
property that actually matters for using tau_eff as a safe, cheap
high-throughput screening estimate is unaffected by this gauge-dependence.
"""

import os

import matplotlib.pyplot as plt

from early5d import calphad, gauge_sensitivity, plotting

from _common import ALLOY_1, ALLOY_2, ALLOY_STENCILS, ELEMENTS, GRID_SPACING, KAPPA_SCALAR, OUTPUT_DIR

FULL_SPECIES = ["Fe", *ELEMENTS]


def analyse(alloy, df, index, decimals):
    composition = alloy["composition"]
    results = gauge_sensitivity.reference_sensitivity(
        df, composition, KAPPA_SCALAR, ELEMENTS, FULL_SPECIES, GRID_SPACING,
        index=index, decimals=decimals, n_k=2000,
    )

    print(f"\n{alloy['label']}")
    for species in FULL_SPECIES:
        r = results[species]
        used = " (paper's convention)" if species == "Fe" else ""
        print(f"  reference={species}: M_eff={r['M_eff']:.3e}  kappa_eff={r['kappa_eff']:.3e}  "
              f"tau_eff={r['tau_eff']:.2f} s  tau={r['tau']:.2f} s{used}")
        assert r["tau_eff"] <= r["tau"] + 1e-9, "tau_eff <= tau bound violated"

    kappa_eff_values = [results[s]["kappa_eff"] for s in FULL_SPECIES]
    print(f"  kappa_eff range across all 5 references: "
          f"{min(kappa_eff_values):.6e} to {max(kappa_eff_values):.6e} (exactly fixed, as expected)")

    tau_eff_values = [results[s]["tau_eff"] for s in FULL_SPECIES]
    tau_fe = results["Fe"]["tau"]  # the paper's own reported (full eigenproblem) CTLD
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plotting.plot_reference_sensitivity(
        ax, FULL_SPECIES, tau_eff_values, highlight_species="Fe",
        ylabel=r"effective-CTLD $\tau_\mathrm{eff}$ (s)",
        reference_line=tau_fe, reference_label=r"CTLD $\tau$",
    )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"reference_sensitivity_{alloy['label']}.png")
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
