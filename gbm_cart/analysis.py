"""
analysis.py — Sensitivity analysis and robustness testing for shortlisted
circuits: seed stability, boundary-artifact detection, parameter sensitivity
sweeps, and measurement-noise robustness.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .circuits import evaluate_circuit_at_threshold, multi_input_ode, steady_state_for_sample
from .signals import SIGNAL_RANGES, sample_regime


def check_stability(circuit, threshold, n_samples=5000, n_seeds=5, signal_ranges=None):
    """Re-run a circuit's tumor:healthy specificity ratio across several
    independent random seeds. A result that only looks good for one lucky
    seed (rather than being stable across seeds) is not a trustworthy
    discriminator.
    """
    ratios = []
    for seed in range(n_seeds):
        t_samp = sample_regime("tumor", n_samples, signal_ranges=signal_ranges, seed=seed * 2)
        h_samp = sample_regime("healthy", n_samples, signal_ranges=signal_ranges, seed=seed * 2 + 1)
        tumor_act = t_samp.apply(lambda r: evaluate_circuit_at_threshold(circuit, r, threshold), axis=1).mean()
        healthy_act = h_samp.apply(lambda r: evaluate_circuit_at_threshold(circuit, r, threshold), axis=1).mean()
        ratio = tumor_act / max(healthy_act, 1e-4)
        ratios.append({"seed": seed, "tumor_activation": tumor_act, "healthy_activation": healthy_act, "ratio": ratio})
    return pd.DataFrame(ratios)


def is_boundary_artifact(circuit, threshold, signal_ranges=None):
    """Flag results where the "optimal" threshold sits at or above the
    healthy range's max for every input signal. Such a threshold isn't
    exploiting any real tumor/healthy separation — it's just past the
    healthy distribution's ceiling for every signal involved, which is a
    boundary artifact of the sampling range rather than genuine
    discrimination.
    """
    if signal_ranges is None:
        signal_ranges = SIGNAL_RANGES
    healthy_maxes = [signal_ranges[sig]["healthy"][1] for sig in circuit["inputs"]]
    return threshold >= max(healthy_maxes)


def parameter_sensitivity_sweep(compute_ratio_fn, gate_types, perturbations, base_params):
    """For each gate type and each parameter in `perturbations`
    ({param_name: [values...]}), hold all other parameters at base_params and
    recompute the specificity ratio via compute_ratio_fn(gate_type, **params).
    Returns {gate_type: {param_name: [ratios...]}}.
    """
    results = {}
    for gate in gate_types:
        results[gate] = {}
        for param, values in perturbations.items():
            ratios = []
            for v in values:
                p = base_params.copy()
                p[param] = v
                ratios.append(compute_ratio_fn(gate, **p))
            results[gate][param] = ratios
            print(f"{gate} | {param}: {[f'{r:.2f}' for r in ratios]}")
    return results


def compute_ratio_for_params(gate_type, K=0.5, n=2, beta=1.0, gamma=0.5,
                              n_samples=500, seed=42, signal_ranges=None):
    """Compute the tumor:healthy specificity ratio at steady state for a
    given (K, n, beta, gamma) parameter set, sampling fresh microenvironments
    at the given seed. Used both for baseline-seed-stability checks and for
    the K x n sensitivity heatmap.
    """
    tumor_s = sample_regime("tumor", n_samples, signal_ranges=signal_ranges, seed=seed)
    healthy_s = sample_regime("healthy", n_samples, signal_ranges=signal_ranges, seed=seed + 1)

    tumor_ss, healthy_ss = [], []
    for i in range(n_samples):
        t_row, h_row = tumor_s.iloc[i], healthy_s.iloc[i]
        hm_t = t_row["healthy_marker"] if gate_type == "AND_NOT" else None
        hm_h = h_row["healthy_marker"] if gate_type == "AND_NOT" else None
        tumor_ss.append(steady_state_for_sample([t_row["hypoxia"], t_row["metabolite"]], gate_type, hm_t, K, n, beta, gamma))
        healthy_ss.append(steady_state_for_sample([h_row["hypoxia"], h_row["metabolite"]], gate_type, hm_h, K, n, beta, gamma))

    return float(np.mean(tumor_ss) / max(np.mean(healthy_ss), 1e-3))


def sensitivity_heatmap(gate_type, K_range, n_range, seed=42):
    """Compute a K x n grid of specificity ratios for one gate type, for a
    2D heatmap of how robust the discrimination is to Hill-function
    parameter choices.
    """
    heatmap_data = np.zeros((len(n_range), len(K_range)))
    for i, n_val in enumerate(n_range):
        for j, K_val in enumerate(K_range):
            heatmap_data[i, j] = compute_ratio_for_params(gate_type, K=K_val, n=n_val, seed=seed)
    return heatmap_data


def robustness_sweep(gate_types, n_samples=1000, K=0.5, n=2, beta=1.0, gamma=0.5, seed_tumor=10, seed_healthy=11):
    """Population-level steady-state robustness: for each gate type, compute
    steady-state activation across many sampled microenvironments per regime,
    returning per-gate distributions and summary statistics.
    """
    tumor_s = sample_regime("tumor", n_samples, seed=seed_tumor)
    healthy_s = sample_regime("healthy", n_samples, seed=seed_healthy)

    results = {}
    for gate in gate_types:
        tumor_ss, healthy_ss = [], []
        for i in range(n_samples):
            t_row, h_row = tumor_s.iloc[i], healthy_s.iloc[i]
            hm_t = t_row["healthy_marker"] if gate == "AND_NOT" else None
            hm_h = h_row["healthy_marker"] if gate == "AND_NOT" else None
            tumor_ss.append(steady_state_for_sample([t_row["hypoxia"], t_row["metabolite"]], gate, hm_t, K, n, beta, gamma))
            healthy_ss.append(steady_state_for_sample([h_row["hypoxia"], h_row["metabolite"]], gate, hm_h, K, n, beta, gamma))

        tumor_ss = np.array(tumor_ss)
        healthy_ss = np.array(healthy_ss)
        ratio = tumor_ss.mean() / max(healthy_ss.mean(), 1e-3)
        results[gate] = {
            "tumor_mean": tumor_ss.mean(), "tumor_std": tumor_ss.std(),
            "healthy_mean": healthy_ss.mean(), "healthy_std": healthy_ss.std(),
            "ratio": ratio, "tumor_ss": tumor_ss, "healthy_ss": healthy_ss,
        }
        print(f"{gate}: tumor={tumor_ss.mean():.3f}±{tumor_ss.std():.3f}, "
              f"healthy={healthy_ss.mean():.3f}±{healthy_ss.std():.3f}, ratio={ratio:.2f}")
    return results


def steady_state_with_noise(signal_vals, gate_type, healthy_marker_val=None, noise_std=0.1,
                             K=0.5, n=2, beta=1.0, gamma=0.5, n_trials=500, seed=42):
    """Fix a representative signal reading for a regime, then add
    measurement noise to it n_trials times. Tests whether a circuit still
    discriminates reliably when the *same* underlying biological state is
    measured imperfectly (as opposed to biological variability across
    different microenvironments, which robustness_sweep tests).
    """
    rng = np.random.default_rng(seed)
    outputs = []
    for _ in range(n_trials):
        noisy_signals = [np.clip(v + rng.normal(0, noise_std), 0, 1) for v in signal_vals]
        noisy_marker = None
        if healthy_marker_val is not None:
            noisy_marker = np.clip(healthy_marker_val + rng.normal(0, noise_std), 0, 1)
        outputs.append(steady_state_for_sample(noisy_signals, gate_type, noisy_marker, K, n, beta, gamma))
    return np.array(outputs)


def noise_overlap_risk(tumor_out, healthy_out):
    """Given noise-trial output arrays for tumor and healthy readings, check
    whether measurement noise ever causes misclassification risk: does the
    healthy 90th percentile exceed the tumor 10th percentile?
    """
    tumor_p10 = np.percentile(tumor_out, 10)
    healthy_p90 = np.percentile(healthy_out, 90)
    return healthy_p90 > tumor_p10, tumor_p10, healthy_p90


def solver_tolerance_check(signal_fns, gate_type, tolerance_sets, t_end=20,
                            K=0.5, n=2, beta=1.0, gamma=0.5, healthy_marker_fn=None):
    """Sanity check that the ODE steady-state result is not an artifact of
    loose solver tolerances: re-solve at several (rtol, atol) settings and
    compare the resulting steady-state value at t_end.
    """
    rows = []
    for tol in tolerance_sets:
        sol = solve_ivp(
            multi_input_ode, (0, t_end), y0=[0],
            args=(signal_fns, gate_type, K, n, beta, gamma, healthy_marker_fn),
            t_eval=[t_end], method="RK45", **tol,
        )
        rows.append({"rtol": tol["rtol"], "atol": tol["atol"], "steady_state": sol.y[0][0]})
        print(f"rtol={tol['rtol']:.0e}, atol={tol['atol']:.0e} -> steady-state = {sol.y[0][0]:.6f}")
    return pd.DataFrame(rows)
