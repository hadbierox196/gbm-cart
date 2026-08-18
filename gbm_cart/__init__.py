"""
gbm_cart — Computational pipeline for GBM CAR-T circuit design.

Boolean logic for full combinatorial topology screening (cheap, fast),
followed by ODE mass-action / Hill kinetics (scipy solve_ivp) for the
shortlisted circuits (dose-response, sensitivity, robustness).

Modules:
    signals    - signal range definitions + regime sampling
    circuits   - topology enumeration + Boolean/ODE circuit evaluation
    simulation - sweep runners, checkpointing, dose-response, ODE time courses
    analysis   - sensitivity analysis and robustness testing
    visualize  - all plotting functions
"""

__version__ = "0.1.0"
