# early5d

[![Tests](https://github.com/nmmatters/early5d/actions/workflows/tests.yml/badge.svg)](https://github.com/nmmatters/early5d/actions/workflows/tests.yml)

**Phase stability analysis in complex multi-component alloys**

`early5d` computes and analyzes the thermo-kinetic eigenproblem that governs the earliest
stage of spinodal decomposition in a multi-component alloy such as the High Entropy Alloys (HEAs). 
Early-stage analysis of the linearized multi-component Cahn–Hilliard are performed. Starting directly
from CALPHAD thermodynamic and mobility data (Gibbs energy derivatives + 
atomic mobilities on a composition grid), it builds the Gibbs-energy Hessian 
and the Onsager mobility matrix, solves the thermo-kinetic growth operator, 
and reports the dispersion relation, the characteristic time of linear 
decomposition (CTLD), and the misalignment between thermodynamic 
driving forces and kinetic transport pathways.

Please see details in and cite:

N. Marschall, R. Darvishi Kamachali, *Why High-Entropy Alloys May Decompose Slower: Misalignment of Early-Pathways and Equilibrium Stages* (2026, in preparation).

This development sits alongside [`phase5d`](https://github.com/nmmatters/phase5d) (5D
phase-diagram visualization), which handles the equilibrium/phase-stability
side of the same CALPHAD datasets.

---

## Concept

For an *N*-component alloy with composition `X = (X1, ..., X(N-1))` (the
*N*-th component follows from `sum(Xi) = 1`), decomposition is governed by
the linearized multi-component Cahn–Hilliard equation. A small perturbation
`δX` evolves as a plane wave `δX = A exp(-i k·r + R t)`, giving the
dispersion relation

```
R(k) = -k² M (Hg + κ k²)
```

where `Hg` is the Hessian of the Gibbs free energy density, `M` is the
Onsager mobility matrix, and `κ` is the gradient-energy tensor. Decomposition
requires a negative eigenvalue of the **thermo-kinetic operator**

```
K(k) = M (Hg + κ k²)
```

The eigenvector of `Hg` with the most negative eigenvalue defines the
**thermo mode** — the direction in composition space along which
thermodynamic driving forces are strongest. The eigenvector of `K(kmax)`
with the most negative eigenvalue defines the **growth mode** — the actual
fastest-growing composition fluctuation, which is in general *rotated* away
from the thermo mode by the anisotropic mobility matrix.

The inverse of the maximum growth rate defines the **characteristic time of
linear decomposition (CTLD)**:

```
τ = 1 / max_k{ -k² μ0(k) }
```

Projecting the eigenproblem onto the thermo mode instead (neglecting kinetic
rotation) gives a cheap, scalar **effective-CTLD** `τeff` — useful for
high-throughput screening of large composition spaces, and a provable lower
bound: `τeff ≤ τ`.

Comparing the thermo mode, the growth mode, and the **equilibrium mode**
(the tie-line direction between coexisting equilibrium phases) via a
mobility-weighted misalignment angle reveals a hidden kinetic barrier: alloys
whose kinetically-accessible growth direction is misaligned with the
equilibrium tie-line decompose more slowly than either the thermodynamics or
the intrinsic atomic mobilities alone would suggest.

---

## Installation

```bash
git clone https://github.com/nmmatters/early5d.git
cd early5d
pip install -e .
```

Ternary cross-section plotting (`early5d.plotting.ternary_axes`/`ternary_scatter`)
additionally requires [`python-ternary`](https://github.com/marcharper/python-ternary):

```bash
pip install -e ".[ternary]"
```

---

## Quick start

```python
import numpy as np
from early5d import calphad, hessian, mobility, kinetics, effective

# 1. Load a CALPHAD export (Thermo-Calc TCHEA4/MOBFE5 style, see Data format)
df = calphad.load_calphad_table("data/tchea4_mobfe5_873k_alloy_stencils.dat")
elements = ["Mn", "Ni", "Co", "Cu"]  # independent components; Fe is the implicit reference
index, decimals = calphad.build_index(df, elements)

composition = np.array([0.10, 0.20, 0.20, 0.30])  # (Mn, Ni, Co, Cu); Fe implicit
grid_spacing = 0.01

# 2. Gibbs-energy Hessian at this composition (finite differences on the grid)
Hg = hessian.hessian_at(df, composition, grid_spacing, elements, index, decimals)

# 3. Onsager mobility matrix from CALPHAD atomic mobilities
mobilities = calphad.atomic_mobilities(df, composition, elements, index, decimals)
M = mobility.mobility_matrix(mobilities, calphad.full_composition(composition))

# 4. Thermo-kinetic dispersion relation and CTLD
kappa = 5e-16 * np.eye(4)  # J m^2 / mol, isotropic in the reduced (Mn, Ni, Co, Cu) coordinates
result = kinetics.solve_dispersion_relation(Hg, M, kappa, n_k=1000)
print(f"CTLD: {result['tau']:.2f} s")

# 5. Cheap effective (single-mode) estimate -- see early5d/effective.py for why
#    this uses the untransformed Hessian's own eigenvector
eff = effective.solve_effective_dispersion(Hg, M, kappa, n_k=1000)
print(f"effective-CTLD: {eff['tau_eff']:.2f} s   (<= CTLD: {eff['tau_eff'] <= result['tau']})")
```

See [`analysis/`](analysis/) for full, runnable scripts that reproduce the
key figures of the paper.

---

## Data format

`early5d` expects a Thermo-Calc property-diagram export with (at least) these
tab-separated columns, one row per composition grid point:

| Column | Meaning |
|---|---|
| `Gm.x(Mn)`, `Gm.x(Ni)`, `Gm.x(Co)`, `Gm.x(Cu)` | Chemical potential (composition derivative of `Gm`) relative to a reference species |
| `Mob(Fe)`, `Mob(Mn)`, `Mob(Ni)`, `Mob(Co)`, `Mob(Cu)` | Atomic mobility per species |
| `x(Mn)`, `x(Ni)`, `x(Co)`, `x(Cu)` | Composition (the reference species, e.g. Fe, follows from `1 - sum`) |

This is the same export format used by [`phase5d`](https://github.com/nmmatters/phase5d)
(see its `data/tchea4_mobfe5_873k_n100.dat`); `early5d.calphad.load_calphad_table`
reuses it directly. Most of the pipeline (`calphad`, `hessian.hessian_at`,
`mobility`, `kinetics`, `effective`) is dimension-agnostic and works for any
alloy system, not just five components — the composition-vector length is
inferred from the columns present. The one exception is the gauge-invariant
orthogonalization (`hessian.thermo_mode`, `hessian.is_spinodal`,
`hessian.orthogonalize`): their default `ORTHOGONALIZATION_MATRIX` is only
valid for a 4-independent-component (5-species) system; calling them on a
differently-sized Hessian raises `ValueError` rather than silently
orthogonalizing against a mismatched basis, and a different-sized system
needs its own orthogonalization matrix derived the same way (see
`early5d/hessian.py`'s module docstring).

### Bundled analysis data

The full 1 at % composition grid used in the paper (~4.6M rows, ~750 MB) is
**not** committed to this repository (same policy as `phase5d`: raw CALPHAD
exports are large, regenerable from Thermo-Calc, and not suited to git).
Instead, [`data/`](data/) ships three small, real extracts from that grid for
the Fe–Mn–Ni–Co–Cu system at 873 K (and doubles as the place to drop your own
full-resolution export — see [`data/README.md`](data/README.md)):

| File | Content | Used by |
|---|---|---|
| `tchea4_mobfe5_873k_alloy_stencils.dat` | The two worked alloys (Fe20Mn10Ni20Co20Cu30, Fe15Mn20Ni20Co15Cu30) plus their 8 axis-neighbor grid points each (the finite-difference stencil needed for an exact local Hessian) | `02`–`05` |
| `tchea4_mobfe5_873k_coarse5pct.dat` | A 5 at % subsample of the full grid (~9700 points, still real CALPHAD data, just coarser than the paper's 1 at % maps) | `06` |
| `tchea4_mobfe5_873k_equilibrium_phases.dat` | The four equilibrium tie-line endpoint compositions for both worked alloys (`PHASE_A`/`PHASE_B_GRID`, `PHASE_A2`/`PHASE_B2`; Table 2 of Marschall, Murugan & Darvishi Kamachali, *Materialia* 44 (2025) 102546), single exact rows — no finite-difference neighbors needed for the mobility matrix | `12` |

To reproduce the paper's exact 1 at % composition maps, point
`calphad.load_calphad_table` at your own full-resolution Thermo-Calc export
(TCHEA4 + MOBFE5 databases, or your own alloy system).

---

## Analysis scripts

See the [`analysis/`](analysis/) directory:

| File | Reproduces |
|------|------------|
| [`01_hessian_and_spinodal.py`](analysis/01_hessian_and_spinodal.py) | Gibbs-energy Hessian, gauge-invariant orthogonalization, spinodal sign test |
| [`02_dispersion_relation.py`](analysis/02_dispersion_relation.py) | Fig. 3 — eigenproblem-based vs. effective growth-rate dispersion and CTLD |
| [`03_eigenmode_bar_shards.py`](analysis/03_eigenmode_bar_shards.py) | Fig. 4 — thermo / mobility / growth / equilibrium mode comparison |
| [`04_misalignment_angles.py`](analysis/04_misalignment_angles.py) | Fig. 5 — mobility-weighted misalignment angles between thermo, growth, and equilibrium modes |
| [`05_anisotropic_gradient_energy.py`](analysis/05_anisotropic_gradient_energy.py) | Fig. 7 — sensitivity of the dispersion relation to an anisotropic gradient-energy coefficient |
| [`06_ctld_composition_maps.py`](analysis/06_ctld_composition_maps.py) | Figs. 1/2 — CTLD and effective-CTLD mapped across the composition simplex, plotted as ternary cross-sections |
| [`07_full_composition_workflow.py`](analysis/07_full_composition_workflow.py) | Figs. 3, 4, 5, 7 for one composition in a single run, via [`early5d.workflow.analyze_composition`](early5d/workflow.py) |
| [`08_pairwise_ternary_maps.py`](analysis/08_pairwise_ternary_maps.py) | Figs. 1/2, generalized — a ternary map of a chosen early-stage parameter for every 20:20 at % species pair, via [`early5d.ternary_workflow`](early5d/ternary_workflow.py) |
| [`09_reference_gauge_sensitivity.py`](analysis/09_reference_gauge_sensitivity.py) | Sec. 4 discussion — sensitivity of `M_eff`/`kappa_eff`/`tau_eff` (and `mobility_ratio = M_eff / lambda_1(M)`) to the choice of reference species, via [`early5d.gauge_sensitivity`](early5d/gauge_sensitivity.py) |
| [`10_composition_sweep_data.py`](analysis/10_composition_sweep_data.py) | Every early-stage parameter (and its reference-species sensitivity) over the whole composition grid, saved as a `.npz` data file via [`early5d.composition_sweep`](early5d/composition_sweep.py) |
| [`11_ctld_correlation.py`](analysis/11_ctld_correlation.py) | CTLD-vs-effective-CTLD correlation scatter (log-log, 1:1 line), for every reference-species choice |
| [`12_equilibrium_mobility_modes.py`](analysis/12_equilibrium_mobility_modes.py) | Fig. 4, extended — dominant mobility eigenmode compared across each worked alloy's nominal and equilibrium phase compositions, via [`early5d.equilibrium_mobility`](early5d/equilibrium_mobility.py) |

Run from the repository root:

```bash
python analysis/01_hessian_and_spinodal.py
python analysis/02_dispersion_relation.py
python analysis/06_ctld_composition_maps.py   # requires: pip install -e ".[ternary]"
```

`07_full_composition_workflow.py` runs everything relevant for one chosen
composition (dispersion relation, eigenmode bars, misalignment angles,
anisotropic sweep) instead of hardcoding one of the two worked alloys per
script:

```bash
python analysis/07_full_composition_workflow.py                                                   # default: Fe20Mn10Ni20Co20Cu30
python analysis/07_full_composition_workflow.py --composition 0.20 0.20 0.15 0.30                  # the other worked alloy
python analysis/07_full_composition_workflow.py --composition 0.15 0.25 0.10 0.25 --dataset coarse # any composition, coarser grid
python analysis/07_full_composition_workflow.py --use-published-tie-line                           # also compute misalignment angles (Fig. 5)
```

See the script's docstring (`--help`) for the full set of options, including
pointing `--data` at your own CALPHAD export (e.g. for a different
temperature than the bundled 873 K).

`08_pairwise_ternary_maps.py` generalizes `06_ctld_composition_maps.py`'s
single Fe = Co = 20 at % panel: for a 5-species system there are
`C(5, 2) = 10` such 20:20 at % pairs, and this script scans the grid once for
a chosen early-stage parameter and produces a ternary map for every pair that
has enough grid points (plus one shared colorbar):

```bash
python analysis/08_pairwise_ternary_maps.py                  # CTLD (tau), the default
python analysis/08_pairwise_ternary_maps.py --field tau_eff   # effective-CTLD
python analysis/08_pairwise_ternary_maps.py --field xi0       # thermodynamic driving force
python analysis/08_pairwise_ternary_maps.py --field k_max     # fastest-growing wavenumber
```

`09_reference_gauge_sensitivity.py` quantifies how much `M_eff`/`kappa_eff`/
`tau_eff` (Eqs. 15-17) depend on which species is treated as the implicit
reference: they project onto the eigenvector of the *untransformed* Hessian,
which (unlike the gauge-invariant thermo mode, Eq. 8) is basis-dependent.
Recomputes the effective quantities under every alternate reference-species
choice for both worked alloys and plots the spread, alongside two sanity
checks printed to console: `kappa_eff` is exactly unchanged regardless of
reference (an algebraic identity whenever `kappa = kappa_scalar * I`), and
the `tau_eff <= tau` bound — the property that actually matters for using
`tau_eff` as a safe screening estimate — holds for every reference choice
tested. Also prints `mobility_ratio = M_eff / lambda_1(M)`
(`early5d.effective.effective_mobility_ratio`) for every reference choice —
how much of the mobility matrix's largest eigenvalue the reference-frame's
thermo mode actually captures, a major additional (non-gauge) reason
`tau_eff` underestimates `tau`:

```bash
python analysis/09_reference_gauge_sensitivity.py
```

`10_composition_sweep_data.py` runs the same reference-species sensitivity
check as `09`, but over the *whole* bundled composition grid instead of just
the two worked alloys, and saves every field (`xi0`, `M_eff`, `kappa_eff`,
`tau_eff`, `tau`, `k_max`, plus each of the four latter recomputed under
every alternate reference species) to a compressed `.npz` data file rather
than (only) a figure -- this one pays a real cost (~6x a single-field scan,
expect several minutes on the bundled coarse grid):

```bash
python analysis/10_composition_sweep_data.py
```

```python
from early5d import composition_sweep
sweep = composition_sweep.load_sweep("analysis/output/composition_sweep.npz")
sweep["tau_eff"]                # (P,) at the paper's own reference species
sweep["tau_eff_by_reference"]   # (P, 5): same, recomputed per reference_species entry
sweep["reference_species"]      # ["Fe", "Mn", "Ni", "Co", "Cu"]
```

`11_ctld_correlation.py` reproduces the manuscript's original CTLD-vs-
effective-CTLD scatter (log-log, with a 1:1 reference line) across the whole
composition space, generalized to show all five reference-species choices at
once instead of just the paper's own (Fe) convention. Reuses `10`'s saved
sweep if present (fast); otherwise runs it fresh (slow, see `10`'s note):

```bash
python analysis/11_ctld_correlation.py
```

`12_equilibrium_mobility_modes.py` evaluates the dominant mobility eigenmode
(the fastest-diffusing direction) not just at the nominal (parent) alloy
composition, as script `03` does, but separately at each of its two
coexisting equilibrium phase compositions, for both worked alloys — the
mobility matrix needs no finite-difference stencil, just the exact
composition, so this works directly on the small bundled
`equilibrium_phases` extract. Also reports each composition's mobility-matrix
eigenvalue gap (`early5d.mobility.eigenvalue_gap`) — how well-separated the
dominant mode is from the next-fastest transport direction:

```bash
python analysis/12_equilibrium_mobility_modes.py
```

---

## Tests

[`tests/`](tests/) is unrelated to the alloy system above — it unit-tests
`early5d`'s own correctness (Hessian symmetry, gauge-invariance, the
`tau_eff <= tau` bound, misalignment-angle bounds, etc.) against small
synthetic data, and runs automatically in CI on every push:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Scope

`early5d` implements the **linearized, chemical** early-stage theory only
(Sections 2.1–2.4 of the paper). It intentionally does **not** include:

- Elastic/coherency-strain contributions to the Hessian (a separate paper,
  N. Marschall, J. Murugan, R. Darvishi Kamachali, *Materialia* 44 (2025) 102546).
- A full nonlinear Cahn–Hilliard phase-field solver, or comparison to
  TEM/APT microstructures (separate validation work) — see
  [`phasefield5d`](https://github.com/nmmatters/phasefield5d).
- Convex-hull / equilibrium-phase-diagram computation. Equilibrium tie-line
  compositions are treated as an external input (see the bundled scripts'
  hardcoded tie-line, sourced from `phase5d`'s published example composition);
  computing them from scratch is the job of a separate (currently
  unpublished) convex-hull toolkit.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use **early5d** in your research and publish the results, please cite
the manuscript this repository accompanies. A journal article is in
preparation; this section will be updated with the full reference and DOI
when it is available. Until then, please cite the repository directly — see
[`CITATION.cff`](CITATION.cff).

---

## Built with Claude Code

This library was developed with the assistance of
[Claude Code](https://claude.ai/claude-code) — Anthropic's agentic coding
tool — which was used to extract, clean, and restructure the analysis code
from the authors' internal research repository into this standalone,
publication-ready package.
