"""Misalignment angles between thermo, growth, and equilibrium modes (Eq. 19).

The mobility-weighted (inverse-mobility metric) angle between two arbitrary
composition-space vectors quantifies how strongly thermodynamic driving
forces, kinetic transport, and equilibrium tie-lines diverge in direction.
"""

import numpy as np


def angle_between(a, b, metric=None, degrees=True, acute=True):
    """Angle between vectors ``a`` and ``b``, optionally under a metric matrix
    ``G`` (Euclidean if ``metric`` is ``None``); Eq. (19) with ``G = M^-1``
    for the mobility-weighted angle.

    ``acute`` folds the angle into ``[0, 90]`` degrees (the sign of a mode's
    direction is arbitrary), matching the manuscript's convention.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if metric is None:
        numerator = a @ b
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
    else:
        G = np.asarray(metric, dtype=float)
        numerator = a @ G @ b
        denominator = np.sqrt((a @ G @ a) * (b @ G @ b))
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    if acute:
        cosine = abs(cosine)
    angle = np.arccos(cosine)
    return float(np.degrees(angle)) if degrees else float(angle)


def inverse_mobility_metric(M):
    """``M^-1``, the metric used for the mobility-weighted misalignment angle (Eq. 19)."""
    return np.linalg.inv(M)


def gram_schmidt_plane(a, b, metric=None):
    """G-orthonormal 2D basis ``(e1, e2)`` spanning ``{a, b}``, used to project
    higher-dimensional modes into a plane for visualization (Fig. 5)."""
    G = np.eye(len(a)) if metric is None else np.asarray(metric, dtype=float)

    def inner(u, v):
        return float(u @ G @ v)

    e1 = a / np.sqrt(inner(a, a))
    b_perp = b - inner(b, e1) * e1
    e2 = b_perp / np.sqrt(inner(b_perp, b_perp))
    return e1, e2


def project_to_plane(vector, e1, e2, metric=None):
    """2D coordinates of ``vector`` in the ``(e1, e2)`` plane, under the same metric."""
    G = np.eye(len(vector)) if metric is None else np.asarray(metric, dtype=float)
    return np.array([float(vector @ G @ e1), float(vector @ G @ e2)])
