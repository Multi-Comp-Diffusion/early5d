"""Log-scale summary statistics for CTLD-type composition maps (Figs. 1/2)."""

import numpy as np


def log_scale_summary(values):
    """Median, geometric mean, quartiles, extrema, and log10-scale standard
    deviation of a positive-valued array, matching the statistics reported
    for the CTLD/effective-CTLD histograms (Figs. 1a/2a)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    log_values = np.log10(values)
    return {
        "median": float(np.median(values)),
        "geometric_mean": float(10 ** np.mean(log_values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
        "log10_std": float(np.std(log_values)),
    }
