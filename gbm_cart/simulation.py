"""
simulation.py — Sweep runners, checkpointing utilities, dose-response
tradeoff curves, and ODE time-course simulation wrappers.
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .circuits import evaluate_circuit_at_threshold, multi_input_ode
from .signals import sample_regime

DEFAULT_CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(data, name, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Save results dict/list to disk as pickle, timestamped + a 'latest' copy."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_ts = os.path.join(checkpoint_dir, f"{name}_{ts}.pkl")
    path_latest = os.path.join(checkpoint_dir, f"{name}_latest.pkl")
    with open(path_ts, "wb") as f:
        pickle.dump(data, f)
    with open(path_latest, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path_latest}")
    return path_latest


def load_checkpoint(name, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Load latest checkpoint if it exists, else return None."""
    path_latest = os.path.join(checkpoint_dir, f"{name}_latest.pkl")
    if os.path.exists(path_latest):
        with open(path_latest, "rb") as f:
            data = pickle.load(f)
        print(f"Resumed from checkpoint: {path_latest}")
        return data
    print(f"No checkpoint found for '{name}' — starting fresh.")
    return None


def run_chunked_sweep(param_grid, sim_fn, name, chunk_size=50, checkpoint_every=1,
                       checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Run sim_fn over param_grid in chunks, checkpointing progress so an
    interrupted run doesn't lose completed work.

    param_grid: list of dicts, each a parameter combination to simulate.
    sim_fn: function(params) -> result dict.
    name: string used for checkpoint filenames.
    """
    state = load_checkpoint(name, checkpoint_dir)
    if state is None:
        state = {"completed_idx": 0, "results": []}

    start_idx = state["completed_idx"]
    total = len(param_grid)
    print(f"Starting at index {start_idx}/{total}")

    for chunk_start in range(start_idx, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        for i in range(chunk_start, chunk_end):
            result = sim_fn(param_grid[i])
            state["results"].append(result)
            state["completed_idx"] = i + 1

        if (chunk_end // chunk_size) % checkpoint_every == 0:
            save_checkpoint(state, name, checkpoint_dir)
            print(f"Progress: {chunk_end}/{total}")

    save_checkpoint(state, name, checkpoint_dir)
    print("Sweep complete.")
    return state["results"]


# ---------------------------------------------------------------------------
# Boolean-stage dose-response tradeoff curves (per named circuit dict)
# ---------------------------------------------------------------------------

def tradeoff_curve(circuit, tumor_samples, healthy_samples,
                    thresholds=np.linspace(0.05, 0.95, 37)):
    """Sweep a shared activation threshold and record tumor/healthy
    activation rates for one circuit, producing its dose-response curve.
    """
    rows = []
    for t in thresholds:
        tumor_act = tumor_samples.apply(
            lambda r: evaluate_circuit_at_threshold(circuit, r, t), axis=1).mean()
        healthy_act = healthy_samples.apply(
            lambda r: evaluate_circuit_at_threshold(circuit, r, t), axis=1).mean()
        rows.append({"threshold": t, "tumor_activation": tumor_act, "healthy_activation": healthy_act})
    return pd.DataFrame(rows)


def summarize_tradeoff_curves(curves, sensitivity_target=0.80):
    """Given {architecture_name: tradeoff_curve_df}, find each architecture's
    best specificity ratio among thresholds meeting the sensitivity target.
    """
    summary = []
    for key, curve in curves.items():
        eligible = curve[curve["tumor_activation"] >= sensitivity_target].copy()
        if len(eligible) == 0:
            summary.append({
                "architecture": key,
                "max_sensitivity_achieved": curve["tumor_activation"].max(),
                "meets_target": False,
                "best_specificity": None,
            })
        else:
            eligible["spec_ratio"] = eligible["tumor_activation"] / eligible["healthy_activation"].clip(lower=1e-3)
            best = eligible.loc[eligible["spec_ratio"].idxmax()]
            summary.append({
                "architecture": key,
                "max_sensitivity_achieved": curve["tumor_activation"].max(),
                "meets_target": True,
                "best_specificity": best["spec_ratio"],
                "threshold_used": best["threshold"],
            })
    return pd.DataFrame(summary).sort_values("best_specificity", ascending=False)


# ---------------------------------------------------------------------------
# Vectorized threshold sweep (for the named CIRCUITS_BOOL shortlist)
# ---------------------------------------------------------------------------

def threshold_sweep(gate_fn, seed, n_samples=5000, n_thresholds=37):
    """Sweep a shared activation threshold for one vectorized gate function
    (see circuits.CIRCUITS_BOOL), returning threshold/tumor-rate/healthy-rate
    arrays for the whole regime at once.
    """
    tumor = sample_regime("tumor", n_samples, seed=seed)
    healthy = sample_regime("healthy", n_samples, seed=seed + 1)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tumor_p = np.array([gate_fn(tumor, t).mean() for t in thresholds])
    healthy_p = np.array([gate_fn(healthy, t).mean() for t in thresholds])
    return thresholds, tumor_p, healthy_p


def best_specificity(thresholds, tumor_p, healthy_p, min_sensitivity=0.80):
    """Among thresholds meeting min_sensitivity, pick the one minimizing
    healthy activation, and return (ratio, threshold, max_sensitivity_seen).
    Returns (None, None, max_sensitivity_seen) if the target is never met.
    """
    mask = tumor_p >= min_sensitivity
    if not mask.any():
        return None, None, tumor_p.max()

    valid_indices = np.where(mask)[0]
    best_index = valid_indices[np.argmin(healthy_p[valid_indices])]
    ratio = tumor_p[best_index] / max(healthy_p[best_index], 1e-4)
    return ratio, thresholds[best_index], tumor_p.max()


# ---------------------------------------------------------------------------
# ODE time-course simulation
# ---------------------------------------------------------------------------

def simulate_multi_circuit(signal_fns, gate_type, healthy_marker_fn=None,
                            K=0.5, n=2, beta=1.0, gamma=0.5, t_span=(0, 20), n_points=200):
    """Integrate the CAR-T activation-protein ODE for a multi-input gated
    circuit over time, given time-varying (or constant) input signal functions.
    """
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(
        multi_input_ode, t_span, y0=[0],
        args=(signal_fns, gate_type, K, n, beta, gamma, healthy_marker_fn),
        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9,
    )
    return sol.t, sol.y[0]


def population_steady_state(combo_fn, regime, beta, gamma, n_samples=1000,
                             K=0.5, n=2, seed=42):
    """Vectorized population-level steady-state activation for many sampled
    microenvironments at once (see circuits.combo_or/combo_and/combo_and_not).
    """
    signals = sample_regime(regime, n_samples, seed=seed)
    combo_value = combo_fn(signals, K, n)
    return beta * combo_value / gamma
