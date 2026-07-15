"""early5d: early-stage analysis of the linearized multi-component Cahn-Hilliard equation.

Companion code for N. Marschall & R. Darvishi Kamachali, "Why High-Entropy
Alloys May Decompose Slower: Misalignment of Early-Pathways and Equilibrium
Stages" (2026, in preparation).
"""

from early5d import (
    calphad, composition_sweep, effective, gauge_sensitivity, hessian, kinetics,
    labels, linalg, misalignment, mobility, plotting, stats, ternary_workflow, workflow,
)

__version__ = "0.1.0"

__all__ = [
    "calphad", "composition_sweep", "effective", "gauge_sensitivity", "hessian",
    "kinetics", "labels", "linalg", "misalignment", "mobility", "plotting", "stats",
    "ternary_workflow", "workflow", "cite",
]


def cite():
    """Print the citation for this software."""
    print(
        "N. Marschall, R. Darvishi Kamachali, \"Why High-Entropy Alloys May "
        "Decompose Slower: Misalignment of Early-Pathways and Equilibrium "
        "Stages\" (2026, in preparation).\n"
        "Software: https://github.com/nmmatters/early5d"
    )
