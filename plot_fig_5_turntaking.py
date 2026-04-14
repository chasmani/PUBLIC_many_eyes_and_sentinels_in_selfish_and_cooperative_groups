"""
Turn-taking figure v2: 4-panel layout.

Top row:    cost curves c(v, e) for two energy levels
  Top-left:  concave (alpha=-1)
  Top-right: convex  (alpha=+1)

Bottom row: vigilance time series for N=4 group
  Bottom-left:  concave -> turn-taking (one sentinel, others at zero)
  Bottom-right: convex  -> many-eyes (all at moderate vigilance)

Each bottom panel has N=4 subpanels (one per individual), boxed axes.
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Model ────────────────────────────────────────────────────────────

def cost(v, alpha, e, eps=0.1, cost_scale=1.0):
    if abs(alpha) < 1e-10:
        return cost_scale / (eps + e) * v
    return cost_scale / (eps + e) * (np.exp(alpha * v) - 1) / alpha


def benefit(S, r):
    return r * (1 - np.exp(-S))


def best_response(e_i, S_others, alpha, r, v_max, cost_scale):
    res = minimize_scalar(
        lambda v: -(benefit(S_others + v, r)
                     - cost(v, alpha, e_i, cost_scale=cost_scale)),
        bounds=(0, v_max), method="bounded",
    )
    return res.x

def get_group_fitness(vigilances, energies, alpha, r, cost_scale):
    S = vigilances.sum()
    b = benefit(S, r)
    total = 0.0
    N = len(vigilances)
    for i in range(N):
        total += b - cost(vigilances[i], alpha, energies[i], cost_scale=cost_scale)
    return total

def cooperative_best_response(i, vigilances, energies, alpha, r, v_max, cost_scale):
    """Vigilance for i that maximises mean group fitness."""
    N = len(vigilances)

    def neg_group_fitness(v_i):
        vigilances_temp = np.copy(vigilances)
        vigilances_temp[i] = v_i
        return -get_group_fitness(vigilances_temp, energies, alpha, r, cost_scale)
    
    res = minimize_scalar(
        neg_group_fitness,
        bounds=(0, v_max), method="bounded",
    )
    return res.x

    
def simulate(N, alpha, r, gamma, m, e_max, e_init, v_max,
             cost_scale, n_rounds, seed, group_type='selfish'):
    rng = np.random.default_rng(seed)
    energies = np.full(N, e_init) + rng.uniform(-0.5, 0.5, N)
    vigilances = rng.uniform(0.1, 1.0, N)
    hist_v = np.zeros((n_rounds, N))

    for t in range(n_rounds):
        for i in rng.permutation(N):
            if group_type == 'cooperative':
                vigilances[i] = cooperative_best_response(
                    i, vigilances, energies, alpha, r, v_max, cost_scale)
            elif group_type == 'selfish':
                S_others = vigilances.sum() - vigilances[i]
                vigilances[i] = best_response(
                    energies[i], S_others, alpha, r, v_max, cost_scale)
        hist_v[t] = vigilances.copy()

        for i in range(N):
            foraging = np.exp(-vigilances[i])
            gain = gamma * foraging * rng.exponential(1.0)
            energies[i] = np.clip(energies[i] + gain - m, 0, e_max)

    return hist_v


# ── Simulate with N=4 ───────────────────────────────────────────────


def sim_and_plot(group_type):

    shared = dict(N=4, gamma=1.0, m=0.3, e_max=10.0,
                  e_init=5.0, v_max=2, n_rounds=300, seed=42)

    if group_type == 'cooperative':
        r = .5
        shared['r'] = r
        shared['group_type'] = 'cooperative'
    elif group_type == 'selfish':
        r = 5.0
        shared['r'] = r
        shared['group_type'] = 'selfish'

    cost_scale_convex = 1.0
    cost_scale_concave = 10.0

    hv_convex = simulate(alpha=1.0, cost_scale=cost_scale_convex, **shared)
    hv_concave = simulate(alpha=-1.0, cost_scale=cost_scale_concave, **shared)

    burnin = 50


    # ── Diagnostics ─────────────────────────────────────────────────────

    v_cx = hv_convex[burnin:]
    print(f"Convex:  mean v = {v_cx.mean():.3f}, "
        f"range = [{v_cx.min():.3f}, {v_cx.max():.3f}]")

    v_cc = hv_concave[burnin:]
    sentinel_idx = np.argmax(hv_concave[burnin:], axis=1)
    switches = np.sum(sentinel_idx[1:] != sentinel_idx[:-1])
    print(f"Concave: sentinel v = {v_cc.max(axis=1).mean():.3f}, "
        f"switches = {switches}")


    # ── Figure ──────────────────────────────────────────────────────────

    N = shared["N"]
    v_max = shared["v_max"]

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                        height_ratios=[1, 2],
                        # Add some guttering
                        hspace=0.4, wspace=0.2)


    # Top-left: concave cost curves
    ax_cc = fig.add_subplot(gs[0, 0])
    v_arr = np.linspace(0, v_max, 200)
    e_low, e_high = 2, 5
    ax_cc.plot(v_arr, cost(v_arr, alpha=-1.0, e=e_low, cost_scale=cost_scale_concave),
            'k-', label=f'low energy ($e_i=${e_low})')
    ax_cc.plot(v_arr, cost(v_arr, alpha=-1.0, e=e_high, cost_scale=cost_scale_concave),
            'k--', label=f'high energy ($e_i=${e_high})')
    ax_cc.set_xlabel('Individual vigilance, $v_i$')
    ax_cc.set_ylabel('Cost, $c(v_i)$')
    ax_cc.legend()

    # Top-right: convex cost curves
    ax_cx = fig.add_subplot(gs[0, 1])
    ax_cx.plot(v_arr, cost(v_arr, alpha=1.0, e=e_low, cost_scale=cost_scale_convex),
            'k-', label=f'low energy ($e_i=${e_low})')
    ax_cx.plot(v_arr, cost(v_arr, alpha=1.0, e=e_high, cost_scale=cost_scale_convex),
            'k--', label=f'high energy ($e_i=${e_high})')
    ax_cx.set_xlabel('Individual vigilance, $v_i$')
    ax_cx.set_ylabel('Cost, $c(v_i)$')

    # Set ylims to 0 and 5 for both plots
    ax_cc.set_ylim(0, 5)
    ax_cx.set_ylim(0, 5)

    ax_cx.legend()

    # Bottom-left: concave -> turn-taking, N=4 subpanels
    ax_bl = fig.add_subplot(gs[1, 0])
    # Remove all spines and ticks from the main bottom-left panel
    ax_bl.spines['top'].set_visible(False)
    ax_bl.spines['right'].set_visible(False)
    ax_bl.spines['bottom'].set_visible(False)
    ax_bl.spines['left'].set_visible(False)
    ax_bl.set_xticks([])
    ax_bl.set_yticks([])

    gs_bl = gs[1, 0].subgridspec(N, 1, hspace=0.1)
    t = np.arange(burnin, shared["n_rounds"])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


    for i in range(N):
        ax = fig.add_subplot(gs_bl[i])
        ax.plot(t, hv_concave[burnin:, i], lw=1.5, color=colors[i])
        ax.set_ylim(-0.2, v_max + 0.2)
        
        ax.set_xlim(t[0], t[-1])
        for spine in ax.spines.values():
            spine.set_visible(True)
        if i < N - 1:
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xlabel('Round')
        ax.set_yticks([0, v_max])

        ax.set_ylabel(f'$v_{i+1}$', rotation=0, labelpad=10)
        # Plot hline at x=0 
        ax.axhline(0, color='gray', lw=0.5)
        ax.axhline(2, color='gray', lw=0.5)

    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.spines['top'].set_visible(False)
    ax_br.spines['right'].set_visible(False)
    ax_br.spines['bottom'].set_visible(False)
    ax_br.spines['left'].set_visible(False)     
    ax_br.set_xticks([])
    ax_br.set_yticks([])

    # Bottom-right: convex -> many-eyes, N=4 subpanels
    gs_br = gs[1, 1].subgridspec(N, 1, hspace=0.1)

    for i in range(N):
        ax = fig.add_subplot(gs_br[i])
        ax.plot(t, hv_convex[burnin:, i], lw=1.5, color=colors[i])
        ax.set_ylim(-0.2, v_max + 0.2)
        ax.set_xlim(t[0], t[-1])
        for spine in ax.spines.values():
            spine.set_visible(True)
        if i < N - 1:
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xlabel('Round')
        ax.set_yticks([0, v_max])
        ax.set_ylabel(f'$v_{i+1}$', rotation=0, labelpad=10)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axhline(2, color='gray', lw=0.5)


    # Extract axes from gridspec
    ax_cc.annotate("a", xy=(0.02, 1.05), xycoords='axes fraction',
                fontsize=20, fontweight='bold')
    ax_cx.annotate("b", xy=(0.02, 1.05), xycoords='axes fraction',
                fontsize=20, fontweight='bold')
    ax_bl.annotate("c", xy=(0.02, 1.05), xycoords='axes fraction',
                    fontsize=20, fontweight='bold')
    ax_br.annotate("d", xy=(0.02, 1.05), xycoords='axes fraction',
                    fontsize=20, fontweight='bold')


    # Remove top and right spines from top two axes
    for ax in (ax_cc, ax_cx):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # remove legend box
    ax_cc.legend(frameon=False)
    ax_cx.legend(frameon=False)

    plt.tight_layout()

    plt.savefig('images/fig_turn_taking_{}.png'.format(group_type), dpi=600, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    sim_and_plot('cooperative')
    sim_and_plot('selfish')