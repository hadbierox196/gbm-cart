# GBM CAR-T Circuit Design — Computational Pipeline

A purely computational pipeline for screening synthetic-biology "circuit"
designs (Boolean logic gates over tumor-microenvironment signals) intended to
drive CAR-T activation specifically in glioblastoma (GBM) tissue while
sparing healthy tissue. No wet-lab data is generated here — this is
simulation only, built to run comfortably on the Colab free tier or on a
laptop CPU.

## Approach

The pipeline is hybrid, in two stages:

1. **Boolean screening.** Every combination of candidate microenvironment
   signals (hypoxia, an altered-metabolism metabolite, an inflammatory
   cytokine) is wired into `SINGLE` / `AND` / `OR` / `AND_NOT` logic gates and
   screened cheaply across simulated tumor and healthy microenvironments, to
   narrow a large combinatorial design space down to a handful of promising
   architectures.
2. **ODE mass-action / Hill kinetics.** The shortlisted architectures are
   re-simulated as continuous dynamical systems (`scipy.integrate.solve_ivp`)
   to check dose-response behavior, parameter sensitivity, and robustness to
   biological and measurement noise — things a Boolean approximation can't
   capture.

### A design note worth knowing before you read the code

An `AND_NOT` gate is meant to mean "activate on tumor signals, but *not* in
healthy tissue." An early version of this pipeline implemented that by
inverting one of the tumor-associated inputs itself (e.g. requiring the
metabolite signal to be *low*) — which is backwards, since that signal is
*elevated* in tumor tissue, so the circuit was partly excluding the tumor
cells it was supposed to target. The fix, used throughout this codebase, is
a dedicated `healthy_marker` signal: a normal-tissue identity marker that is
independently high in healthy tissue and low/disrupted in tumor. `AND_NOT`
inverts *that* signal instead. See the docstrings in `gbm_cart/signals.py`
and `gbm_cart/circuits.py` for the full rationale.

## Repository layout

```
gbm_cart_circuits/
├── gbm_cart/                 # the pipeline as an importable package
│   ├── signals.py            # signal range definitions + regime sampling
│   ├── circuits.py           # topology enumeration + Boolean/ODE circuit evaluation (final logic only)
│   ├── simulation.py         # sweep runners, checkpointing, dose-response, ODE time courses
│   ├── analysis.py           # sensitivity analysis + robustness testing
│   └── visualize.py          # all plotting functions
├── notebooks/
│   └── 01_pipeline_walkthrough.ipynb   # narrative notebook: imports the package, runs the full pipeline in order
├── results/                  # generated figures/checkpoints (not committed — see results/README.md)
├── requirements.txt
└── .gitignore
```

## Setup

### Local

```bash
git clone <this-repo-url>
cd gbm_cart_circuits
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
jupyter lab notebooks/01_pipeline_walkthrough.ipynb
```

### Colab

1. Upload the repo folder to Google Drive (or `git clone` it into a Colab
   cell via `!git clone <url>`).
2. Open `notebooks/01_pipeline_walkthrough.ipynb` in Colab.
3. Run the first cell (`pip install -r ../requirements.txt` — add a `!` prefix
   if running as a Colab shell command) — the free tier is sufficient; no GPU
   needed.
4. Run all cells top to bottom.

The notebook adds the repo root to `sys.path` so `import gbm_cart` works
whether you're running locally or from Colab, as long as the notebook stays
inside `notebooks/` relative to the package.

## Reproducing results

Run `notebooks/01_pipeline_walkthrough.ipynb` top to bottom (or via
`jupyter nbconvert --to notebook --execute --inplace notebooks/01_pipeline_walkthrough.ipynb`).
All randomness is seeded, so results and figures are deterministic. See
`results/README.md` for exactly which files get produced.

The notebook walks through, in order:

1. Signal model and sampling
2. Full combinatorial Boolean topology screen (all gate types × input
   combinations × stringency levels)
3. Dose-response tradeoff curves, boundary-artifact filtering, and seed
   stability checks
4. Final shortlist: `OR (hypoxia, metabolite)`, `AND (3-input)`, and the
   corrected `AND_NOT (hyp, met, healthy-marker)`, plus the single-antigen
   baseline
5. ODE mass-action kinetics for the shortlisted circuits
6. Population-level robustness (steady-state output distributions across
   1000 sampled microenvironments per regime)
7. Parameter sensitivity (K, n, gamma perturbations; K×n heatmap)
8. Measurement-noise robustness
9. Solver-tolerance sanity check

## Status / caveats

- Signal ranges in `gbm_cart/signals.py` are placeholders for pipeline
  development, not literature-sourced values — replace them with
  literature-backed ranges before drawing biological conclusions.
- This is a screening and dynamics-sanity tool, not a validated predictive
  model of in vivo CAR-T behavior.
