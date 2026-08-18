# results/

This folder is where the pipeline writes generated figures (`.png`), summary
tables (`.csv`), and checkpoints (`checkpoints/*.pkl`). It's intentionally
kept empty in the repo (see `.gitignore`) — raw outputs are regenerated
locally rather than committed, since they're fully deterministic given the
fixed seeds used in `notebooks/01_pipeline_walkthrough.ipynb`.

## Regenerating

From the repo root:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_pipeline_walkthrough.ipynb
```

This reproduces, in order:

- `boolean_screen_summary.png` — Boolean-stage dose-response + ranking (full topology screen)
- `fig1_dose_response_and_ranking.png` — final shortlist dose-response + ranking
- `fig2_ode_shortlist_dynamics.png` — ODE time-course dynamics for shortlisted circuits
- `fig3_robustness_histograms.png` — population-level robustness distributions
- `ode_sensitivity_analysis.png` — parameter sensitivity line plots
- `ode_sensitivity_heatmap.png` — K × n specificity-ratio heatmaps
- `checkpoints/*.pkl` — intermediate results (Boolean screen, tradeoff curves, robustness, sensitivity)

Runtime is a few minutes on a laptop CPU (no GPU required).
