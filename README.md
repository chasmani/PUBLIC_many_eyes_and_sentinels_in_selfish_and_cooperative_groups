# Many-Eyes and Sentinels in Selfish and Cooperative Groups

Code for Pilgrim et al., "Many-Eyes and Sentinels in Selfish and Cooperative Groups."

## Overview

This paper presents a minimal model of collective vigilance showing that many-eyes and sentinel strategies are alternate solutions to the same adaptive problem. Which strategy is preferred depends on the curvature of the individual cost function: convex costs (e.g. open habitats) favour many-eyes; concave costs (e.g. environments with vantage points) favour sentinels. The same dichotomy holds in both selfish and cooperative groups.

## Model

Individual fitness is given by

    f_i = b(S) - c(v_i)

where S = sum of individual vigilances, b(S) = r(1 - exp(-S)) is the shared benefit of predator detection, and c(v) = (exp(alpha * v) - 1) / alpha is the individual cost of vigilance. The parameter alpha controls cost curvature: alpha > 0 gives convex costs (many-eyes), alpha < 0 gives concave costs (sentinels). The turn-taking extensions add a dynamic energetic state, with the cost of vigilance scaled by an individual's energy reserves.

## Code Structure

The core parameter-space model is implemented in `sims.py`, which contains:

- `get_b(r, S)`: shared benefit of collective vigilance.
- `get_c(alpha, v)`: individual cost of vigilance.
- `get_f_i(alpha, r, vv, i)`: individual fitness.
- `best_response_selfish(r, vv, alpha, i)`: selfish best response, maximising individual fitness.
- `best_response_coop(r, vv, alpha, i)`: cooperative best response, maximising group fitness.
- `simulate(r, alpha, N, v_max, group_type)`: runs best-response dynamics to convergence for a given parameter combination, returning the number of vigilant individuals and mean fitness.
- `run_sims(N, resolution)`: sweeps the (r, alpha) parameter space and writes results to CSV.

Each figure has a corresponding self-contained plotting script:

- Figures 1, 2, and 4 are analytical and can be generated directly.
- Figure 3 and the SI group-size figure plot results from the parameter sweep and require running `sims.py` first.
- Figure 5, the SI energetic-ordering figure, and the SI robustness analysis run their own energy-dependent turn-taking simulations and are fully self-contained (they do not require `sims.py`).

## Reproducing Figures

First, run the parameter-space simulations (this may take some time at high resolution):

```bash
python sims.py
```

This produces `sim_results_N_4.csv`, `sim_results_N_16.csv`, and `sim_results_N_64.csv`.

Then generate each figure:

```bash
python plot_fig_1_dynamical_system_with_cost_curves.py   # Figure 1: Best-response dynamics
python plot_fig_2_optimals_tight_with_cost_curves.py     # Figure 2: Optimal vigilance strategies
python plot_fig_3_sims.py                                # Figure 3: Parameter space (requires sims.py)
python plot_fig_4_extended.py                            # Figure 4: Model extensions
python plot_fig_5_turntaking.py                          # Figure 5: Turn-taking with energy dynamics
python plot_fig_si_sims_varying_N.py                     # SI Figure: Parameter space varying N (requires sims.py)
python plot_fig_si_turntaking_energetic_ordering.py      # SI Figure: Turn-taking via energetic ordering
```

Figures are saved to the `images/` folder.

To reproduce the SI turn-taking robustness table:

```bash
python analyse_turntaking_robustness.py                  # SI Table: robustness of turn-taking
```

This writes `turntaking_robustness.csv` and a formatted LaTeX table `turntaking_robustness.tex`.

## Requirements

Python 3.8+ with NumPy, SciPy, matplotlib, pandas, and seaborn.

## License

MIT License.
