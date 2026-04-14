# Many-Eyes and Sentinels in Selfish and Cooperative Groups

Code for Pilgrim et al., "Many-Eyes and Sentinels in Selfish and Cooperative Groups."

## Overview

This paper presents a minimal model of collective vigilance showing that many-eyes and sentinel strategies are alternate solutions to the same adaptive problem. Which strategy is preferred depends on the curvature of the individual cost function: convex costs (e.g. open habitats) favour many-eyes; concave costs (e.g. environments with vantage points) favour sentinels. The same dichotomy holds in both selfish and cooperative groups.

## Reproducing Figures

Figures 3 and the SI figures rely on simulation results. To generate these, first run:

```bash
python sims.py
```

This produces CSV files (`sim_results_N_4.csv`, `sim_results_N_16.csv`, `sim_results_N_64.csv`) that are read by the plotting scripts below.

Each figure in the paper has a corresponding plotting script:

```bash
python plot_fig_1_dynamical_system_with_cost_curves.py   # Figure 1: Best-response dynamics
python plot_fig_2_optimals_tight_with_cost_curves.py     # Figure 2: Optimal vigilance strategies
python plot_fig_3_sims.py                                # Figure 3: Parameter space (requires sims.py)
python plot_fig_4_extended_tight.py                      # Figure 4: Model extensions
python plot_fig_5_turntaking.py                          # Figure 5: Turn-taking with energy dynamics
python plot_fig_si_sims_varying_N.py                     # SI Figure: Parameter space for varying N (requires sims.py)
```

Figures are saved to the `images/` folder.
