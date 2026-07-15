"""Composition label formatting, e.g. ``Fe20Mn10Ni20Co20Cu30``."""

import numpy as np


def composition_label(composition, elements, reference, unicode=True):
    """Format a composition vector (independent components + implicit
    reference species) as an at.% alloy label."""
    composition = np.asarray(composition, dtype=float)
    x_reference = 1.0 - composition.sum()
    fractions = np.round(np.concatenate([[x_reference], composition]) * 100).astype(int)
    species = [reference, *elements]

    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    parts = []
    for element, fraction in zip(species, fractions):
        fraction_str = str(fraction).translate(subscripts) if unicode else str(fraction)
        parts.append(f"{element}{fraction_str}")
    return "".join(parts)
