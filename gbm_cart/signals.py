"""
signals.py — Candidate input signals for the GBM CAR-T circuit and how they
are sampled to represent "tumor" vs "healthy" microenvironments.

Signals modeled (all on a 0-1 normalized scale):
    hypoxia         - elevated in tumor (hypoxic core), low in healthy tissue
    metabolite      - elevated in tumor (altered metabolism), low in healthy tissue
    cytokine        - elevated in tumor (inflammatory microenvironment), low in healthy
    healthy_marker  - a *normal-parenchyma identity marker*. Unlike the three
                       tumor-associated signals above, this one runs the other
                       way: HIGH in healthy tissue, LOW/disrupted in tumor.

Why `healthy_marker` exists at all
-----------------------------------
Early AND_NOT circuit designs tried to build a "tumor AND NOT healthy" gate by
inverting one of the *tumor-associated* signals (e.g. requiring metabolite to
be LOW). That's backwards: metabolite is elevated in tumor tissue, so
inverting it actively excludes the tumor cells you want to target. A correct
AND_NOT circuit needs a dedicated signal that is independently anti-correlated
with tumor state — a marker that is present in healthy tissue and lost/
disrupted in tumor — and inverts *that* instead. `healthy_marker` plays that
role.

Why the sampled tumor/healthy ranges overlap
----------------------------------------------
An earlier version of SIGNAL_RANGES left an explicit numerical gap between
the top of the healthy range and the bottom of the tumor range for each
signal (e.g. healthy up to 0.45, tumor starting at 0.55). That gap is not a
biological claim — real tumor and healthy microenvironments overlap in
marker expression — and having no overlap silently made every discrimination
task artificially easy. The ranges below intentionally overlap, which makes
circuit comparisons a meaningful test of a topology's ability to separate two
overlapping distributions rather than two disjoint ones.
"""

import numpy as np
import pandas as pd

# Signal ranges: (low, high) uniform sampling bounds per regime, per signal.
# tumor/healthy ranges overlap on purpose (see module docstring).
SIGNAL_RANGES = {
    "hypoxia":       {"tumor": (0.35, 1.00), "healthy": (0.00, 0.65)},
    "metabolite":    {"tumor": (0.30, 1.00), "healthy": (0.00, 0.60)},
    "cytokine":      {"tumor": (0.25, 1.00), "healthy": (0.00, 0.55)},
    "healthy_marker": {"tumor": (0.00, 0.65), "healthy": (0.35, 1.00)},  # inverted vs. the above three
}

# Boolean-stage default activation cutoff: a signal counts as "ON" above this.
# Only used where a circuit doesn't specify its own per-signal stringency.
ACTIVATION_THRESHOLD = 0.5

# Stringency shifts the per-signal activation threshold used during full
# topology screening. 'low' = easier to trigger (lower bar), 'high' = harder
# to trigger (stricter bar). Baseline single-antigen circuits use the default.
STRINGENCY_THRESHOLDS = {
    "low": 0.40,
    "medium": 0.50,
    "high": 0.60,
    "na": 0.50,
}


def sample_regime(regime, n_samples, signal_ranges=None, seed=None):
    """Draw n_samples of each signal for a given regime ('tumor' or 'healthy').

    signal_ranges defaults to the module-level SIGNAL_RANGES, but the default
    is resolved *inside* the function body (looked up at call-time) rather
    than bound as a mutable default argument. Python binds default argument
    values once at def-time, so `signal_ranges=SIGNAL_RANGES` in the function
    signature would silently keep pointing at whatever SIGNAL_RANGES was when
    the module was first imported, even if the caller later reassigns the
    module-level dict. Passing `signal_ranges=None` and resolving it here
    avoids that trap.
    """
    if signal_ranges is None:
        signal_ranges = SIGNAL_RANGES
    rng = np.random.default_rng(seed)
    samples = {}
    for sig, ranges in signal_ranges.items():
        low, high = ranges[regime]
        samples[sig] = rng.uniform(low, high, n_samples)
    return pd.DataFrame(samples)


def representative_reading(regime, signal_ranges=None):
    """Return a single representative signal reading for a regime, using the
    midpoint of each signal's range. Used for illustrative deterministic ODE
    time courses (not for population-level statistics — use sample_regime
    for that).
    """
    if signal_ranges is None:
        signal_ranges = SIGNAL_RANGES
    return {
        signal: float(np.mean(ranges[regime]))
        for signal, ranges in signal_ranges.items()
    }
