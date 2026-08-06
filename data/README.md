# data/

This directory ships three small, real, git-committed CALPHAD data extracts
for the Fe–Mn–Ni–Co–Cu system at 873 K, used by the [`analysis/`](../analysis/)
scripts (see the [main README](../README.md#bundled-analysis-data) for what
each file contains and which scripts use it):

- `tchea4_mobfe5_873k_alloy_stencils.dat`
- `tchea4_mobfe5_873k_coarse5pct.dat`
- `tchea4_mobfe5_873k_equilibrium_phases.dat`

It's also where you can drop your **own** full-resolution CALPHAD export to
reproduce the paper's exact 1 at % composition maps. Raw Thermo-Calc exports
are large (hundreds of MB at 1 at % resolution) and not suited to version
control, so `.gitignore` excludes every file in this directory except the
three named above — anything else you place here (e.g. your own
`tchea4_mobfe5_873k_n100.dat`) stays local and is never committed.

Point `early5d.calphad.load_calphad_table` at your file (see the column
format described in the [main README](../README.md#data-format)) to use it.
