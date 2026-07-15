# data/

This directory is for your own full-resolution CALPHAD exports and is
git-ignored (see the repository root `.gitignore`) — raw Thermo-Calc exports
are large (hundreds of MB at 1 at % resolution) and not suited to version
control.

Drop a Thermo-Calc property-diagram export here (see the column format
described in the [main README](../README.md#data-format)) and point
`early5d.calphad.load_calphad_table` at it to reproduce the paper's full
composition-space maps (Figs. 1/2).

For runnable examples with small, git-committed, real CALPHAD data, see
[`examples/data/`](../examples/data/) instead.
