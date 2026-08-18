"""
circuits.py — Circuit topology enumeration and evaluation logic (Boolean and
ODE / Hill-kinetics stages).

This module contains only the final, corrected evaluation logic. Earlier
notebook iterations had a version of AND_NOT that inverted one of the
*tumor-associated* signals (e.g. requiring metabolite to be low), which is
biologically backwards since that signal is elevated in tumor tissue — it
made the circuit exclude the very cells it should target. The fix (used
throughout this module) is that AND_NOT always inverts the dedicated
`healthy_marker` signal, never one of the tumor-positive inputs. See
signals.py for why that signal exists.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .signals import ACTIVATION_THRESHOLD, STRINGENCY_THRESHOLDS

# Candidate input signals (from the GBM neuroenvironment) used during the
# full combinatorial topology screen.
SIGNALS = ["hypoxia", "metabolite", "cytokine"]

# Gate architectures screened. AND_NOT = AND-gate over tumor-associated
# inputs, with the healthy-tissue identity marker required to be OFF.
GATE_TYPES = ["AND", "OR", "AND_NOT"]

STRINGENCY_LEVELS = ["low", "medium", "high"]


def enumerate_topologies(signals=SIGNALS, gate_types=GATE_TYPES,
                          stringency_levels=STRINGENCY_LEVELS, max_inputs=3):
    """Enumerate the full library of candidate circuits: single-antigen
    baselines plus every combination of 2-3 tumor-associated input signals
    under each gate type and stringency level.
    """
    topologies = []
    circuit_id = 0

    # Single-antigen baseline circuits (control group)
    for sig in signals:
        topologies.append({
            "circuit_id": circuit_id,
            "gate_type": "SINGLE",
            "inputs": (sig,),
            "stringency": "na",
            "is_baseline": True,
        })
        circuit_id += 1

    # Multi-input gated circuits
    for n_inputs in range(2, max_inputs + 1):
        for input_combo in combinations(signals, n_inputs):
            for gate in gate_types:
                if gate == "AND_NOT" and n_inputs < 2:
                    continue
                for stringency in stringency_levels:
                    topologies.append({
                        "circuit_id": circuit_id,
                        "gate_type": gate,
                        "inputs": input_combo,
                        "stringency": stringency,
                        "is_baseline": False,
                    })
                    circuit_id += 1

    return topologies


def evaluate_circuit(circuit, sample_row, threshold=None):
    """Evaluate a single circuit's Boolean output for one sample row of
    signal values.

    threshold: explicit per-signal activation cutoff. If None, the cutoff is
    looked up from STRINGENCY_THRESHOLDS using the circuit's own stringency
    level (final behavior — earlier notebook versions used one global
    ACTIVATION_THRESHOLD for every circuit regardless of stringency).

    AND_NOT circuits require every tumor-associated input to be ON *and* the
    healthy_marker signal to be OFF (i.e. low/disrupted, as it is in tumor
    tissue) — see module docstring for why this must be a dedicated marker
    signal rather than an inverted tumor-positive input.
    """
    if threshold is None:
        threshold = STRINGENCY_THRESHOLDS[circuit["stringency"]]

    if circuit["gate_type"] == "AND_NOT":
        tumor_inputs_on = [sample_row[sig] > threshold for sig in circuit["inputs"]]
        healthy_marker_off = sample_row["healthy_marker"] <= threshold
        return all(tumor_inputs_on) and healthy_marker_off

    inputs_on = [sample_row[sig] > threshold for sig in circuit["inputs"]]
    if circuit["gate_type"] == "SINGLE":
        return inputs_on[0]
    elif circuit["gate_type"] == "AND":
        return all(inputs_on)
    elif circuit["gate_type"] == "OR":
        return any(inputs_on)
    else:
        raise ValueError(f"Unknown gate_type: {circuit['gate_type']}")


def score_circuit(circuit, tumor_samples, healthy_samples):
    """Compute activation rate in tumor vs healthy regimes and the
    tumor:healthy specificity ratio for one circuit, using each circuit's own
    stringency-derived threshold.
    """
    tumor_activation = tumor_samples.apply(
        lambda row: evaluate_circuit(circuit, row), axis=1).mean()
    healthy_activation = healthy_samples.apply(
        lambda row: evaluate_circuit(circuit, row), axis=1).mean()

    # Avoid div-by-zero; treat 0 healthy activation as very high (but capped) specificity.
    epsilon = 1e-3
    specificity_ratio = tumor_activation / max(healthy_activation, epsilon)

    return {
        "circuit_id": circuit["circuit_id"],
        "gate_type": circuit["gate_type"],
        "inputs": circuit["inputs"],
        "stringency": circuit["stringency"],
        "is_baseline": circuit["is_baseline"],
        "tumor_activation": tumor_activation,
        "healthy_activation": healthy_activation,
        "specificity_ratio": specificity_ratio,
    }


def evaluate_circuit_at_threshold(circuit, sample_row, threshold):
    """Same evaluation logic as evaluate_circuit, but with an explicit shared
    threshold instead of stringency-derived per-circuit thresholds. Used for
    sweeping a single circuit's dose-response / tradeoff curve across a range
    of thresholds.
    """
    return evaluate_circuit(circuit, sample_row, threshold=threshold)


# ---------------------------------------------------------------------------
# Vectorized Boolean gates (DataFrame-wide, no per-row Python loop)
# ---------------------------------------------------------------------------
# These operate on a whole regime's sampled DataFrame at once and back the
# fixed, named "shortlisted" circuits used for the final dose-response /
# ranking figures (see simulation.CIRCUITS_BOOL).

def or_gate(signals_df, inputs, threshold):
    active = pd.Series(False, index=signals_df.index)
    for sig in inputs:
        active |= signals_df[sig] > threshold
    return active


def and_gate(signals_df, inputs, threshold):
    active = pd.Series(True, index=signals_df.index)
    for sig in inputs:
        active &= signals_df[sig] > threshold
    return active


def and_not_gate(signals_df, and_inputs, not_input, threshold):
    """AND over `and_inputs`, with `not_input` (the healthy-tissue marker)
    required to be LOW for activation."""
    active = and_gate(signals_df, and_inputs, threshold)
    active &= signals_df[not_input] < threshold
    return active


# Named, final shortlisted circuits used throughout the ODE stage and the
# summary figures.
CIRCUITS_BOOL = {
    "SINGLE (hypoxia)": lambda s, t: s["hypoxia"] > t,
    "OR (hypoxia, metabolite)": lambda s, t: or_gate(s, ["hypoxia", "metabolite"], t),
    "AND (3-input)": lambda s, t: and_gate(s, ["hypoxia", "metabolite", "cytokine"], t),
    "AND_NOT (hyp, met, healthy-marker)": lambda s, t: and_not_gate(
        s, ["hypoxia", "metabolite"], "healthy_marker", t
    ),
}


# ---------------------------------------------------------------------------
# ODE / Hill-kinetics stage
# ---------------------------------------------------------------------------

def hill_activation(signal, K, n):
    """Hill function: fraction of promoter activation given signal level.
    K = half-activation threshold, n = Hill coefficient (cooperativity).
    """
    return signal ** n / (K ** n + signal ** n)


def multi_input_ode(t, y, signal_fns, gate_type, K, n, beta, gamma,
                     healthy_marker_fn=None, K_hm=0.5, n_hm=2):
    """Right-hand side of the CAR-T activation-protein ODE for a multi-input
    gated circuit.

    y[0] = CAR-T activation protein level.
    signal_fns: list of functions signal_fn(t) for each tumor-associated input.
    gate_type: 'AND', 'OR', 'AND_NOT'.
      - AND: multiply Hill activations (all inputs must be active) -> product
        approximates AND logic.
      - OR: use 1 - product(1 - hill_i) (probabilistic/noisy-OR combination).
      - AND_NOT: AND of tumor signals * (1 - Hill(healthy_marker)) — marker
        presence inhibits activation, mirroring the Boolean-stage fix.
    """
    protein = y[0]
    hills = [hill_activation(fn(t), K, n) for fn in signal_fns]

    if gate_type == "AND":
        gate_activation = np.prod(hills)
    elif gate_type == "OR":
        gate_activation = 1 - np.prod([1 - h for h in hills])
    elif gate_type == "AND_NOT":
        tumor_gate = np.prod(hills)
        marker_hill = hill_activation(healthy_marker_fn(t), K_hm, n_hm)
        gate_activation = tumor_gate * (1 - marker_hill)
    else:
        raise ValueError(gate_type)

    production = beta * gate_activation
    degradation = gamma * protein
    return [production - degradation]


def steady_state_for_sample(signal_vals, gate_type, healthy_marker_val=None,
                             K=0.5, n=2, beta=1.0, gamma=0.5):
    """Compute steady-state activation for a single sample of signal values
    directly (no time integration needed at steady state, since
    d(protein)/dt = 0 implies protein_ss = beta * gate_activation / gamma).

    Note: in the tumor:healthy *ratio* (specificity_ratio = tumor_ss /
    healthy_ss) both beta and gamma cancel algebraically, since they multiply
    both the numerator and denominator identically. They still matter for the
    absolute activation level and are kept as explicit parameters for the ODE
    time-course simulations and for sensitivity analysis, but the ratio-based
    metrics used for architecture comparison are insensitive to their exact
    values.
    """
    hills = [hill_activation(v, K, n) for v in signal_vals]
    if gate_type == "AND":
        gate_activation = np.prod(hills)
    elif gate_type == "OR":
        gate_activation = 1 - np.prod([1 - h for h in hills])
    elif gate_type == "AND_NOT":
        tumor_gate = np.prod(hills)
        marker_hill = hill_activation(healthy_marker_val, K, n)
        gate_activation = tumor_gate * (1 - marker_hill)
    else:
        raise ValueError(gate_type)
    return beta * gate_activation / gamma


# ---------------------------------------------------------------------------
# Vectorized Hill-combination functions (population-scale, array inputs)
# ---------------------------------------------------------------------------
# Equivalent math to steady_state_for_sample, but operating on whole arrays
# of sampled microenvironments at once — used for the large-n population
# robustness sweeps and heatmaps where a per-row Python loop would be slow.

def combo_or(signals_df, K, n, inputs):
    product = np.ones_like(signals_df[inputs[0]], dtype=float)
    for sig in inputs:
        product *= (1 - hill_activation(signals_df[sig], K, n))
    return 1 - product


def combo_and(signals_df, K, n, inputs):
    output = np.ones_like(signals_df[inputs[0]], dtype=float)
    for sig in inputs:
        output *= hill_activation(signals_df[sig], K, n)
    return output


def combo_and_not(signals_df, K, n, and_inputs, not_input):
    return combo_and(signals_df, K, n, and_inputs) * (
        1 - hill_activation(signals_df[not_input], K, n)
    )


# Named, final shortlisted circuits for the vectorized Hill-combination form,
# mirroring CIRCUITS_BOOL above but for the continuous ODE/steady-state stage.
GATE_COMBOS = {
    "OR (hypoxia, metabolite)": lambda sig, K, n: combo_or(sig, K, n, ["hypoxia", "metabolite"]),
    "AND (3-input)": lambda sig, K, n: combo_and(sig, K, n, ["hypoxia", "metabolite", "cytokine"]),
    "AND_NOT (hyp, met, healthy-marker)": lambda sig, K, n: combo_and_not(
        sig, K, n, ["hypoxia", "metabolite"], "healthy_marker"
    ),
}
