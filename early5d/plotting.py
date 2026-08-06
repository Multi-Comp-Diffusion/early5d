"""Plotting helpers for dispersion relations, eigenmode bar-shards, misalignment
geometry, and ternary composition-map cross-sections (Figs. 1-5, 7).

Styling matches the manuscript's figures: no more than ~5 ticks per axis,
scientific notation rendered as ``$10^{n}$`` rather than ``1e-n``, inward
ticks with heavy spines, and (for the ternary maps) the ``gist_rainbow_r``
colormap with the triangular boundary as the only frame -- no rectangular
matplotlib axes box.

Ternary plotting requires the optional ``python-ternary`` dependency
(``pip install "early5d[ternary]"``); the rest of this module only needs
matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter

from early5d.misalignment import angle_between, gram_schmidt_plane, project_to_plane

FONTSIZE = 14
LEGEND_FONTSIZE = 11

# set once, at import time, rather than inside _style_axes: rcParams only
# affects text created *after* the update, so relying on a call inside
# _style_axes (itself called after set_xlabel/set_ylabel) made a figure's
# label size depend on whether it was the first one created in the process
plt.rcParams.update({"font.size": FONTSIZE})


def latex_scientific(value):
    """Format a number in LaTeX scientific notation, e.g. ``5e-15`` -> ``r"5\\times10^{-15}"``,
    ``1e-15`` -> ``r"1\\times10^{-15}"`` -- the mantissa is always shown (even
    when 1) so a column of labels stays the same length/alignment."""
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    return rf"{mantissa:.0f}\times10^{{{exponent}}}"


def _style_axes(ax, n_ticks=5, x_ticks=True, y_ticks=True, box=True):
    """Shared classical-plot styling: heavy inward ticks, <= n_ticks per axis,
    and scientific notation as ``$10^{n}$`` rather than ``1e-n``.

    ``x_ticks=False`` skips the x-axis tick locator/formatter -- needed for
    categorical axes (e.g. element names) where a numeric locator would
    clobber ``set_xticks``/``set_xticklabels``.

    ``box=False`` drops the top/right spines and their ticks, leaving just
    the bottom/left axes -- used for the misalignment plot, where the dashed
    unit circle already frames the diagram and a full rectangle is redundant.
    """
    spines = ax.spines.values() if box else (ax.spines["bottom"], ax.spines["left"])
    for spine in spines:
        spine.set_linewidth(2.0)
    if not box:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4, width=1, which="major", top=box, right=box, labelsize=FONTSIZE)
    ax.tick_params(direction="in", length=3, width=0.8, which="minor", top=box, right=box)
    # set directly on the already-created label Text objects rather than relying
    # on rcParams (which only affects text created *after* the update) -- makes
    # the size independent of call order/how many figures came before this one
    ax.xaxis.label.set_fontsize(FONTSIZE)
    ax.yaxis.label.set_fontsize(FONTSIZE)

    if x_ticks:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(formatter)
        if ax.get_xscale() == "linear":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    if y_ticks:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        if ax.get_yscale() == "linear":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    return ax


def plot_dispersion(ax, wavenumbers, growth_rates, label=None, scale_k=1e-9, linewidth=1.8, **kwargs):
    """Growth-rate dispersion relation ``R(k)`` (Fig. 3 style)."""
    ax.plot(np.asarray(wavenumbers) * scale_k, growth_rates, label=label, linewidth=linewidth, **kwargs)
    ax.set_xlabel(r"wavenumber $k$ (nm$^{-1}$)")
    ax.set_ylabel(r"growth rate $R(k)$ (s$^{-1}$)")
    _style_axes(ax)
    if label:
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    return ax


def _style_log_axes(ax, n_ticks=5):
    """Log-log scatter styling: heavy inward ticks and scientific notation as
    ``$10^{n}$`` -- like :func:`_style_axes`, but using a ``LogLocator``/
    ``FuncFormatter`` pair suited to log axes instead of ``_style_axes``'s
    ``ScalarFormatter`` (meant for linear axes with occasional large values).
    """
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(direction="in", length=4, width=1, which="major", top=True, right=True, labelsize=FONTSIZE)
    ax.tick_params(direction="in", length=3, width=0.8, which="minor", top=True, right=True)
    ax.xaxis.label.set_fontsize(FONTSIZE)
    ax.yaxis.label.set_fontsize(FONTSIZE)

    formatter = FuncFormatter(lambda x, pos: rf"$10^{{{int(round(np.log10(x)))}}}$" if x > 0 else "")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=n_ticks))
        axis.set_major_formatter(formatter)
        axis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=n_ticks * 2))
        axis.set_minor_formatter(NullFormatter())
    return ax


def plot_ctld_correlation(ax, tau_eff, tau, label=None, color=None, marker_size=6, alpha=0.3):
    """Scatter ``tau_eff`` (x) against ``tau`` (y) on log-log axes -- the
    CTLD-vs-effective-CTLD correlation plot from the manuscript's original
    analysis, generalized here to accept one series per reference-species
    choice (see :func:`plot_ctld_reference_line` for the accompanying 1:1
    line, and ``analysis/11_ctld_correlation.py`` for a full example).
    """
    tau_eff, tau = np.asarray(tau_eff, dtype=float), np.asarray(tau, dtype=float)
    finite = np.isfinite(tau_eff) & np.isfinite(tau) & (tau_eff > 0) & (tau > 0)
    ax.scatter(tau_eff[finite], tau[finite], s=marker_size, alpha=alpha, label=label, color=color, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"effective-CTLD $\tau_\mathrm{eff}$ (s)")
    ax.set_ylabel(r"CTLD $\tau$ (s)")
    _style_log_axes(ax)
    return ax


def plot_ctld_reference_line(ax, lo=None, hi=None, label=r"$\tau = \tau_\mathrm{eff}$"):
    """1:1 reference line for :func:`plot_ctld_correlation`; defaults to
    spanning the axes' current limits if ``lo``/``hi`` aren't given."""
    if lo is None or hi is None:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        lo = min(xlim[0], ylim[0]) if lo is None else lo
        hi = max(xlim[1], ylim[1]) if hi is None else hi
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label=label)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    return ax


def plot_eigenmode_bars(ax, modes, labels, elements, colors=None,
                         ylabel="normalized mode amplitude", eigenvalues=None):
    """Bar-shard comparison of normalized mode amplitudes across alloy components (Fig. 4).

    ``eigenvalues`` (optional, one array per mode, descending, as returned by
    :func:`early5d.equilibrium_mobility.compare_mobility_modes`) adds a small
    inset axes in the upper-right corner plotting each mode's dominant
    eigenvalue (``eigenvalues[i][0]``) as a marker, connected by a dashed
    line. It is a genuinely separate nested axes with its own (log-scaled)
    y-axis rather than an overlay on the bar axis, since the eigenvalue's
    units/scale (mol m^2 J^-1 s^-1, spanning decades -- the raw mobility
    matrix's own units, not the m^2/s of an M*Hg-type diffusivity) have
    nothing in common with the dimensionless bar heights, and it has no
    natural x-position among the per-*element* bar groups (it is one scalar
    per *mode*, not per element).
    """
    x = np.arange(len(elements))
    n_modes = len(modes)
    width = 0.8 / n_modes
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [cycle[i % len(cycle)] for i in range(n_modes)]
    for i, (mode, label) in enumerate(zip(modes, labels)):
        offset = (i - (n_modes - 1) / 2) * width
        ax.bar(x + offset, mode, width=width, label=label, color=colors[i])
    ax.axhline(0, lw=1, color="k")
    ax.set_ylabel(ylabel)
    _style_axes(ax, x_ticks=False)  # categorical x-axis: keep element names, not a numeric locator
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend(frameon=False, loc="upper left", fontsize=LEGEND_FONTSIZE)

    if eigenvalues is not None:
        _eigenvalue_inset(ax, eigenvalues, labels, colors)
    return ax


def _eigenvalue_inset(ax, eigenvalues, labels, colors):
    """Small nested axes (upper-right corner of ``ax``) plotting each mode's
    dominant eigenvalue (``eigenvalues[i][0]``, descending order) as a
    marker connected by a dashed line, on its own log-scaled y-axis."""
    lambda_1 = [np.asarray(ev)[0] for ev in eigenvalues]
    xpos = np.arange(len(lambda_1))

    inset = ax.inset_axes([0.55, 0.60, 0.42, 0.36])  # same upper-right anchor, ~20% larger
    inset.plot(xpos, lambda_1, "--", color="gray", linewidth=1.2, zorder=1)
    inset.scatter(xpos, lambda_1, color=colors[:len(lambda_1)], edgecolors="k", linewidth=0.8, zorder=2, s=25)
    inset.set_yscale("log")
    inset.set_xlim(-0.5, len(lambda_1) - 0.5)
    inset.set_xticks(xpos)
    inset.set_xticklabels(labels, fontsize=8)
    inset.set_ylabel(r"$m_\mathrm{max}$ (mol m$^2$/(J s))", fontsize=9, labelpad=2)
    inset.tick_params(labelsize=8, direction="in", length=3, which="both")
    for spine in inset.spines.values():
        spine.set_linewidth(1.2)
    return inset


def plot_reference_sensitivity(ax, species, values, highlight_species=None, ylabel=None,
                                reference_line=None, reference_label=None):
    """Bar chart of a quantity (e.g. ``tau_eff``) recomputed under every
    candidate reference-species choice (Sec. 4 discussion's gauge-dependence
    check), with the species actually used in the paper highlighted.

    ``reference_line`` optionally draws a dashed horizontal line (e.g. the
    paper's own reported full-eigenproblem CTLD, tau) so the bars can be
    compared against it directly.
    """
    x = np.arange(len(species))
    colors = ["tab:orange" if s == highlight_species else "tab:blue" for s in species]
    ax.bar(x, values, color=colors)
    if reference_line is not None:
        ax.axhline(reference_line, color="k", linestyle="--", linewidth=1.5, label=reference_label)
        if reference_label:
            ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    ax.set_ylabel(ylabel)
    _style_axes(ax, x_ticks=False)  # categorical x-axis: keep species names, not a numeric locator
    ax.set_xticks(x)
    ax.set_xticklabels(species)
    return ax


def plot_misalignment(ax, vectors, labels, metric=None):
    """2D projection of thermo/growth/equilibrium modes with pairwise
    misalignment angles annotated (Fig. 5). Returns the computed angles.

    Vectors are rescaled to unit length *under the given metric* before
    projecting, matching the paper's convention ("all vectors are
    represented in unit length") -- otherwise the arbitrary absolute scale
    of a physical metric like ``M^-1`` (which can span decades) would
    dominate the plotted coordinates instead of the angles between vectors.
    """
    G = np.eye(len(vectors[0])) if metric is None else np.asarray(metric, dtype=float)
    unit_vectors = [v / np.sqrt(v @ G @ v) for v in vectors]

    # Eigenvector/tie-line signs are otherwise arbitrary; orient every mode
    # onto the same side as the first (thermo) mode so the plotted arrows
    # match the reported *acute* angles instead of their anti-parallel twins.
    reference = unit_vectors[0]
    unit_vectors = [v if (v @ G @ reference) >= 0 else -v for v in unit_vectors]

    e1, e2 = gram_schmidt_plane(unit_vectors[0], unit_vectors[-1], metric)
    coords = [project_to_plane(v, e1, e2, metric) for v in unit_vectors]

    # all worked-example vectors land in the upper-right quadrant (thermo mode
    # defines +x by construction, and growth/equilibrium deviate only slightly);
    # only that quarter of the unit circle is drawn/shown
    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="gray", lw=1)
    for (x, y), label in zip(coords, labels):
        x, y = float(x), float(y)
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
        # radial offset + sign-based alignment (instead of leading spaces) so
        # labels clear the arrowhead without being clipped/wrapped at the axes edge
        text_x, text_y = x * 1.08, y * 1.08
        ax.text(text_x, text_y, label, va="center", ha="left" if text_x >= 0 else "right")
    margin = 1.35
    ax.set_xlim(-0.05, margin)
    ax.set_ylim(-0.05, margin)
    ax.set_aspect("equal")
    ax.set_xlabel("thermo-unstable mode")
    ax.set_ylabel("⊥ deviation")
    _style_axes(ax, box=False)

    angles = {}
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            angles[f"{labels[i]}-{labels[j]}"] = angle_between(vectors[i], vectors[j], metric)
    return angles


def ternary_axes(scale=100, figsize=(6, 5.5), dpi=150, fontsize=16):
    """A fresh matplotlib figure + python-ternary axes pair (Figs. 1/2 style):
    triangular boundary only (no rectangular matplotlib axes box), and at
    most 3 ticks per edge (``multiple=scale/2``)."""
    import ternary  # optional dependency, see module docstring

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    tax.boundary(linewidth=2)

    tick_multiple = scale / 2
    for axis in ("l", "r", "b"):
        tax.ticks(axis=axis, multiple=tick_multiple, linewidth=1, offset=0.035, fontsize=fontsize)

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")  # drop the rectangular matplotlib frame; keep only the triangle
    return fig, tax


def ternary_corner_labels(tax, right, top, left, fontsize=16):
    """Bold corner labels in the manuscript's exact placement/offset convention."""
    tax.right_corner_label("\n      " + right, fontsize=fontsize, weight="bold", offset=-0.1)
    tax.top_corner_label(top, fontsize=fontsize, weight="bold", offset=0.3)
    tax.left_corner_label("\n" + left + "      ", fontsize=fontsize, weight="bold", offset=0.15)


def ternary_annotation(tax, scale, constant_elements, fontsize=16, rotation_angle=-55):
    """Rotated in-triangle annotation stating the two held-constant components
    (e.g. ``{"Fe": 0.20, "Co": 0.20}``), matching the manuscript's convention
    -- used instead of a plot title, which tends to collide with the top
    corner's tick label."""
    (elements, fractions) = zip(*constant_elements.items())
    tax.annotate(
        f"   {elements[0]}: {fractions[0] * 100:.0f} at.% \n   {elements[1]}: {fractions[1] * 100:.0f} at.%",
        (scale / 2 + scale / 15, scale / 3 + scale / 15),
        rotation=rotation_angle, weight="bold", fontsize=fontsize - 2,
    )


def ternary_scatter(tax, ternary_compositions, values, scale=100, cmap="gist_rainbow_r", marker_size=16):
    """Scatter a scalar field (e.g. CTLD) onto a ternary cross-section.

    ``ternary_compositions``: ``(P, 3)`` fractions of the three components
    spanning the slice (summing to a constant); ``values``: the scalar field
    at each point. Does not draw a colorbar -- see :func:`ternary_colorbar`,
    which (matching the manuscript's figures) saves it as a separate figure
    rather than attaching it to the ternary axes.
    """
    points = (np.asarray(ternary_compositions, dtype=float) * scale).tolist()
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    tax.scatter(points, c=values, cmap=cmap, vmin=vmin, vmax=vmax, marker="h", s=marker_size)
    return tax


def ternary_colorbar(values, label, cmap="gist_rainbow_r", log_tag=False,
                     fontsize=16, orientation="horizontal", dpi=150, size=None):
    """A standalone colorbar figure for a :func:`ternary_scatter` field,
    matching the manuscript's convention of saving/placing the colorbar
    separately from the ternary plot itself. ``log_tag=True`` renders ticks
    as ``$10^{n}$`` (for a field already in log10 space) instead of raw values.

    ``size`` defaults to ``(4, 0.4)`` for ``log_tag=True`` (short "$10^n$"
    labels); a linear field's plain/scientific-notation labels need more
    horizontal room, so the default widens to ``(5.5, 0.5)`` when
    ``log_tag=False``.
    """
    if size is None:
        size = (4, 0.4) if log_tag else (5.5, 0.5)
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)

    fig_cbar, ax_cbar = plt.subplots(figsize=size, dpi=dpi)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation=orientation)
    cbar.set_label(label, fontsize=fontsize)
    ax_cbar.tick_params(labelsize=fontsize)
    # integer=True when log_tag: tick *values* must land on whole decades, or
    # the "$10^{n}$" formatter (which rounds to the nearest integer) can print
    # the same label twice for two ticks that round to the same exponent.
    # nbins is lower for the linear case: its labels are wider (a full
    # mantissa + exponent, vs. a bare small integer for log_tag), so 5 of them
    # crowd/overlap in the same horizontal space.
    cbar.locator = MaxNLocator(nbins=5 if log_tag else 4, integer=log_tag)
    cbar.update_ticks()
    if log_tag:
        cbar.formatter = FuncFormatter(lambda x, pos: rf"$10^{{{int(round(x))}}}$")
    else:
        # plain numbers get too wide for this figure's narrow aspect ratio and
        # overlap; fall back to the same "$n \times 10^{m}$" scientific
        # notation used elsewhere once a tick magnitude warrants it
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        cbar.formatter = formatter
    cbar.update_ticks()
    return fig_cbar, cbar
