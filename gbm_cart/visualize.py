"""
visualize.py — All plotting functions for the Boolean screening stage, the
ODE / robustness stage, and sensitivity analysis. Each function optionally
saves the figure to `results_dir` if provided.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, results_dir, filename):
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches="tight")


def plot_boolean_screen_summary(curves, key_archs, df_summary, results_dir=None,
                                 filename="boolean_screen_summary.png"):
    """Two-panel figure: (left) dose-response tradeoff curves for a chosen
    set of key architectures, (right) ranked bar chart of specificity ratio
    at the matched-sensitivity target, restricted to non-artifact results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key in key_archs:
        c = curves[key]
        axes[0].plot(c["threshold"], c["tumor_activation"], label=f"{key} (tumor)", linewidth=2)
        axes[0].plot(c["threshold"], c["healthy_activation"], label=f"{key} (healthy)", linestyle="--", alpha=0.6)
    axes[0].axhline(0.80, color="red", linestyle=":", label="80% sensitivity target")
    axes[0].set_xlabel("Activation threshold")
    axes[0].set_ylabel("Activation rate")
    axes[0].set_title("Dose-response: tumor vs healthy activation")
    axes[0].legend(fontsize=7, loc="upper right")

    plot_df = df_summary[df_summary["meets_target"] & ~df_summary.get("boundary_artifact", False)]
    plot_df = plot_df.sort_values("best_specificity")
    axes[1].barh(plot_df["architecture"], plot_df["best_specificity"], color="steelblue")
    axes[1].set_xlabel("Specificity ratio @ target sensitivity")
    axes[1].set_title("Comparative ranking (validated, artifact-free)")

    plt.tight_layout()
    _save(fig, results_dir, filename)
    return fig


def plot_dose_response_and_ranking(sweep_results, ranking, results_dir=None,
                                    filename="fig1_dose_response_and_ranking.png"):
    """Final shortlist figure: dose-response curves for the named circuits in
    circuits.CIRCUITS_BOOL, plus a ranked bar chart of specificity ratio at
    >=80% tumor sensitivity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(sweep_results)))
    for (name, (thresholds, tumor_p, healthy_p)), color in zip(sweep_results.items(), colors):
        axes[0].plot(thresholds, tumor_p, color=color, linestyle="-", label=f"{name} — tumor")
        axes[0].plot(thresholds, healthy_p, color=color, linestyle="--", alpha=0.7, label=f"{name} — healthy")

    axes[0].axhline(0.80, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Boolean activation threshold")
    axes[0].set_ylabel("Activation probability")
    axes[0].set_title("Dose-response tradeoff curves\n(tumor vs. healthy)")
    axes[0].legend(fontsize=7, loc="center right")
    axes[0].set_ylim(-0.02, 1.02)

    ranking_sorted = sorted(ranking, key=lambda x: (x[1] is None, -(x[1] or 0)))
    names = [item[0] for item in ranking_sorted]
    ratios = [item[1] if item[1] is not None else 0 for item in ranking_sorted]
    bar_colors = ["#2ca02c" if "AND_NOT" in name else "#1f77b4" if "AND" in name else "#d62728" for name in names]

    bars = axes[1].barh(names, ratios, color=bar_colors)
    for bar, (name, ratio, max_sensitivity) in zip(bars, ranking_sorted):
        label = f"{ratio:.1f}" if ratio is not None else f"never \u226580% (max {max_sensitivity*100:.1f}%)"
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)

    axes[1].set_xlabel("Specificity ratio @ \u226580% tumor sensitivity")
    axes[1].set_title("Comparative ranking (Boolean stage)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    _save(fig, results_dir, filename)
    return fig


def plot_ode_dynamics(gate_combos, representative_reading_fn, t_span=(0, 15),
                       K=0.5, n=4, beta=2.0, gamma=1.0, results_dir=None,
                       filename="fig2_ode_shortlist_dynamics.png"):
    """Time-course ODE dynamics for each shortlisted circuit, comparing a
    representative tumor reading vs. a representative healthy reading.
    """
    from scipy.integrate import solve_ivp

    t_eval = np.linspace(*t_span, 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (name, combo_fn) in zip(axes, gate_combos.items()):
        for regime, color in [("tumor", "crimson"), ("healthy", "steelblue")]:
            signals = representative_reading_fn(regime)
            signals_arr = {k: np.array([v]) for k, v in signals.items()}
            combo_value = combo_fn(signals_arr, K, n)[0]

            def rhs(t, P, c=combo_value):
                return beta * c - gamma * P

            solution = solve_ivp(rhs, t_span, y0=[0.0], t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)
            steady_state = beta * combo_value / gamma
            ax.plot(solution.t, solution.y[0], color=color, label=f"{regime} (ss={steady_state:.2f})")

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Time (a.u.)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Circuit output P(t)")
    fig.suptitle("ODE-stage dynamics: shortlisted circuits")
    plt.tight_layout()
    _save(fig, results_dir, filename)
    return fig


def plot_robustness_histograms(gate_combos, population_steady_state_fn, results_dir=None,
                                filename="fig3_robustness_histograms.png"):
    """Population-level robustness figure: tumor vs. healthy steady-state
    output distributions across many sampled microenvironments, per
    shortlisted circuit. Returns (figure, pop_summary dict).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    pop_summary = {}

    for ax, (name, combo_fn) in zip(axes, gate_combos.items()):
        tumor_ss = population_steady_state_fn(combo_fn, "tumor")
        healthy_ss = population_steady_state_fn(combo_fn, "healthy")
        pop_summary[name] = (tumor_ss.mean(), tumor_ss.std(), healthy_ss.mean(), healthy_ss.std())

        bins = np.linspace(0, max(tumor_ss.max(), healthy_ss.max()), 40)
        ax.hist(healthy_ss, bins=bins, alpha=0.6, color="steelblue", label=f"healthy (\u03bc={healthy_ss.mean():.2f})")
        ax.hist(tumor_ss, bins=bins, alpha=0.6, color="crimson", label=f"tumor (\u03bc={tumor_ss.mean():.2f})")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Steady-state circuit output")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count (of sampled microenvironments)")
    fig.suptitle("Population-level robustness: tumor vs. healthy steady-state distributions")
    plt.tight_layout()
    _save(fig, results_dir, filename)
    return fig, pop_summary


def print_population_summary(pop_summary):
    """Print the tumor/healthy mean±sd and specificity ratio table matching
    the population robustness figure.
    """
    print(f"{'Architecture':40s}{'tumor mean\u00b1sd':>16s}{'healthy mean\u00b1sd':>18s}{'ratio':>8s}")
    for name, (tumor_mean, tumor_std, healthy_mean, healthy_std) in pop_summary.items():
        ratio = tumor_mean / max(healthy_mean, 1e-4)
        print(f"{name:40s}{tumor_mean:6.2f}\u00b1{tumor_std:<8.2f}{healthy_mean:6.2f}\u00b1{healthy_std:<10.2f}{ratio:8.2f}")


def plot_sensitivity_lines(sensitivity_results, perturbations, results_dir=None,
                            filename="ode_sensitivity_analysis.png"):
    """Line plots of specificity ratio vs. each perturbed parameter, one
    panel per gate type.
    """
    gate_types = list(sensitivity_results.keys())
    fig, axes = plt.subplots(1, len(gate_types), figsize=(6.5 * len(gate_types), 5), sharey=True)
    if len(gate_types) == 1:
        axes = [axes]

    for ax, gate in zip(axes, gate_types):
        for param, values in perturbations.items():
            ratios = sensitivity_results[gate][param]
            ax.plot(range(len(values)), ratios, marker="o", label=param)
        ax.set_title(f"{gate}: parameter sensitivity")
        ax.set_xlabel("Perturbation index (low \u2192 high)")
        ax.legend()
    axes[0].set_ylabel("Specificity ratio")
    plt.tight_layout()
    _save(fig, results_dir, filename)
    return fig


def plot_sensitivity_heatmaps(heatmaps, K_range, n_range, results_dir=None,
                               filename="ode_sensitivity_heatmap.png"):
    """K x n specificity-ratio heatmaps, one panel per gate type.
    heatmaps: {gate_type: 2D array of shape (len(n_range), len(K_range))}.
    """
    gate_types = list(heatmaps.keys())
    fig, axes = plt.subplots(1, len(gate_types), figsize=(6.5 * len(gate_types), 5))
    if len(gate_types) == 1:
        axes = [axes]

    for ax, gate in zip(axes, gate_types):
        im = ax.imshow(heatmaps[gate], aspect="auto", origin="lower", cmap="viridis",
                        extent=[K_range[0], K_range[-1], n_range[0], n_range[-1]])
        ax.set_xlabel("K (half-activation threshold)")
        ax.set_ylabel("n (Hill coefficient)")
        ax.set_title(f"{gate}: specificity ratio landscape")
        plt.colorbar(im, ax=ax, label="Specificity ratio")

    plt.tight_layout()
    _save(fig, results_dir, filename)
    return fig
