"""
Robustness of turn-taking in the energy-dependent vigilance model.

Runs the selfish energy-dependent simulation (the same model as Figure 5) for
one seed per parameter combination and reports, for each run:

  - mean proportion of time vigilant (per individual, averaged across the group)
  - SD of that proportion across individuals (fairness / "stuck sentinel" check)
  - number of individuals vigilant per period ("sentinels per period")

Parameters are swept one-at-a-time around a baseline matched to Figure 5, for
both the concave (sentinel turn-taking) and convex (many-eyes) regimes. The
contrast is the point: concave gives ~1/N time each and ~1 vigilant per period;
convex gives ~all the time each and ~N vigilant per period.

Outputs:
  - turntaking_robustness.csv   (numeric results)
  - turntaking_robustness.tex   (LaTeX table for the SI)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


# ── Run settings ─────────────────────────────────────────────────────

N_SEEDS = 1
N_ROUNDS = 10000
BURNIN = 50
VIGILANCE_THRESHOLD = 1e-3


# ── Model (energy-dependent cost, selfish best response) ─────────────

def cost(v, alpha, e, eps=0.1, cost_scale=1.0):
    """Opportunity cost of vigilance, with magnitude set by energetic state."""
    if abs(alpha) < 1e-10:
        return cost_scale / (eps + e) * v
    return cost_scale / (eps + e) * (np.exp(alpha * v) - 1) / alpha


def benefit(S, r):
    return r * (1 - np.exp(-S))


def best_response_selfish(e_i, S_others, alpha, r, v_max, cost_scale):
    res = minimize_scalar(
        lambda v: -(benefit(S_others + v, r)
                    - cost(v, alpha, e_i, cost_scale=cost_scale)),
        bounds=(0, v_max), method="bounded",
    )
    return res.x


def simulate(N, alpha, r, gamma, m, e_max, e_init, v_max,
             cost_scale, n_rounds, seed):
    """Run the energy-dependent selfish dynamics, returning the vigilance history."""
    rng = np.random.default_rng(seed)
    energies = np.full(N, e_init) + rng.uniform(-0.5, 0.5, N)
    vigilances = rng.uniform(0.1, 1.0, N)
    hist_v = np.zeros((n_rounds, N))

    for t in range(n_rounds):
        for i in rng.permutation(N):
            S_others = vigilances.sum() - vigilances[i]
            vigilances[i] = best_response_selfish(
                energies[i], S_others, alpha, r, v_max, cost_scale)
        hist_v[t] = vigilances.copy()

        for i in range(N):
            foraging = np.exp(-vigilances[i])
            gain = gamma * foraging * rng.exponential(1.0)
            energies[i] = np.clip(energies[i] + gain - m, 0, e_max)

    return hist_v


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(hist_v, burnin, threshold=VIGILANCE_THRESHOLD):
    post = hist_v[burnin:]
    is_vigilant = post > threshold

    proportion_vigilant = is_vigilant.mean(axis=0)          # per individual
    n_vigilant_per_period = is_vigilant.sum(axis=1)          # per period

    return {
        "mean_prop_vigilant": proportion_vigilant.mean(),
        "sd_prop_across_individuals": proportion_vigilant.std(),
        "n_per_period": n_vigilant_per_period.mean(),
    }


def run_config(params, alpha, n_seeds, n_rounds, burnin):
    """Average metrics over seeds for one parameter combination."""
    rows = []
    for seed in range(n_seeds):
        hist_v = simulate(alpha=alpha, n_rounds=n_rounds, seed=seed, **params)
        rows.append(compute_metrics(hist_v, burnin))
    return pd.DataFrame(rows).mean().to_dict()


# ── Sweep definition ─────────────────────────────────────────────────

BASELINE = dict(N=4, r=5.0, gamma=1.0, m=0.3, e_max=10.0, e_init=5.0, v_max=2.0)

VARIATIONS = {
    "r": [0.5, 50.0],
    "N": [2, 4, 8, 16],
    "m": [0.1, 0.2, 0.3, 0.5],
    "e_max": [5.0, 10.0, 20.0],
    "cost_scale": [0.1, 1.0, 10.0],
}

# Baseline cost scaling differs by regime, matching Figure 5 / Methods.
REGIMES = {
    "sentinel (concave)": dict(alpha=-1.0, cost_scale=10.0),
    "many-eyes (convex)": dict(alpha=1.0, cost_scale=10.0),
}


def build_configs(baseline):
    """Baseline plus one-at-a-time variations of each parameter."""
    configs = [("baseline", dict(baseline))]
    for param, values in VARIATIONS.items():
        for value in values:
            if value == baseline[param]:
                continue
            params = dict(baseline)
            params[param] = value
            configs.append((param, params))
    return configs


def run_sweep(n_seeds, n_rounds, burnin):
    records = []
    for regime_name, regime in REGIMES.items():
        baseline = dict(BASELINE, cost_scale=regime["cost_scale"])
        for varied, params in build_configs(baseline):
            metrics = run_config(params, regime["alpha"],
                                 n_seeds, n_rounds, burnin)
            records.append({
                "regime": regime_name,
                "varied": varied,
                **{k: params[k] for k in ("N", "r", "m", "e_max", "cost_scale")},
                **metrics,
            })
            print(f"  {regime_name:20s} {varied:10s} "
                  f"N={params['N']:>2} r={params['r']:>5} m={params['m']} "
                  f"e_max={params['e_max']:>4} C={params['cost_scale']:>4} | "
                  f"prop={metrics['mean_prop_vigilant']:.3f} "
                  f"sd={metrics['sd_prop_across_individuals']:.3f} "
                  f"n/period={metrics['n_per_period']:.2f}")
    return pd.DataFrame.from_records(records)


# ── LaTeX output ─────────────────────────────────────────────────────

REGIME_LABEL = {
    "sentinel (concave)": r"concave, $\alpha=-1$",
    "many-eyes (convex)": r"convex, $\alpha=1$",
}

VARIED_LABEL = {
    "baseline": "baseline",
    "r": "$r$",
    "N": "$N$",
    "m": "$m$",
    "e_max": "$e_{max}$",
    "cost_scale": "$C$",
}


def write_latex_table(df, filename):
    lines = [
        r"\begin{tabular}{ll|rrrrr|rrr}",
        r"\toprule",
        r"Cost Scaling & Varied & $N$ & $r$ & $m$ & $e_{max}$ & $C$ & "
        r"Mean Prop. Vigilant & SD & Mean \# vigilant/period \\",
        r"\midrule",
    ]
    prev_regime = None
    for _, row in df.iterrows():
        if prev_regime is not None and row["regime"] != prev_regime:
            lines.append(r"\midrule")
        prev_regime = row["regime"]
        lines.append(
            f"{REGIME_LABEL[row['regime']]} & {VARIED_LABEL[row['varied']]} & "
            f"{int(row['N'])} & {row['r']:g} & {row['m']:g} & "
            f"{row['e_max']:g} & {row['cost_scale']:g} & "
            f"{row['mean_prop_vigilant']:.3f} & "
            f"{row['sd_prop_across_individuals']:.3f} & "
            f"{row['n_per_period']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(filename, "w") as fp:
        fp.write("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"Running sweep: n_seeds={N_SEEDS}, n_rounds={N_ROUNDS}, burnin={BURNIN}")
    df = run_sweep(N_SEEDS, N_ROUNDS, BURNIN)
    df.to_csv("turntaking_robustness.csv", index=False)
    write_latex_table(df, "turntaking_robustness.tex")
    print("\nSaved turntaking_robustness.csv and turntaking_robustness.tex")


if __name__ == "__main__":
    main()
