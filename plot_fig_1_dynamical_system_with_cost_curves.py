import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

COLOR_BLACK = '#0f0f0f'
LINEWIDTH = 3
FONTSIZE=12
OPTIMAL_COLOR = "#F62459"
OPTIMAL_EDGE_COLOR = "white"
OPTIMAL_MARKER_EDGE_WIDTH = 2


def get_c(alpha, v):
    return (np.exp(alpha * v) - 1) / alpha if alpha != 0 else v

# Best response dynamics for c(v) = (e^{α v} - 1)/α
def best_response_dynamics(v1, v2, alpha, r):
    # Gradient ascent dynamics: v_i' = ∂f_i/∂v_i
    S = v1 + v2
    df1 = r * np.exp(-S) - np.exp(alpha * v1)
    df2 = r * np.exp(-S) - np.exp(alpha * v2)
    return df1, df2

# Grid setup
v_max = 2
v_vals = np.linspace(0, v_max, 200)
V1, V2 = np.meshgrid(v_vals, v_vals)

# Parameters
r = 2 # So that log r = 1, simplifies algebra
alphas = [-1, 0, 1]
titles = [r'$\alpha = -0.5$', r'$\alpha = 0$', r'$\alpha = 0.5$']

# Function to compute the symmetric equilibrium v*
def compute_symmetric_equilibrium(alpha, r, N=2):
    if alpha + N == 0:
        return None  # Avoid division by zero
    return np.log(r) / (alpha + N)

def compute_sentinel_equilibrium(alpha, r, v_max):
    """
    Compute the sentinel equilibrium v* for a given alpha and r.
    This is the point where the best response dynamics converge.
    """
    v_star = np.log(r)/(alpha + 1) 
    if v_star > v_max:
        return v_max
    return v_star

# Create figure with GridSpec for custom layout
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, figure=fig, 
                      height_ratios=[1, 2],
                      # Add some guttering
                      hspace=0.3, wspace=0.3)

# Create axes
cost_axes = []
phase_axes = []

# Create first column (reference axes)
cost_axes.append(fig.add_subplot(gs[0, 0]))
phase_axes.append(fig.add_subplot(gs[1, 0], sharex=cost_axes[0]))

# Create remaining columns with shared axes
for i in range(1, 3):
    # Share y-axis with first cost plot, share x-axis with corresponding phase plot
    cost_ax = fig.add_subplot(gs[0, i])
    phase_ax = fig.add_subplot(gs[1, i], sharex=cost_ax)
    
    cost_axes.append(cost_ax)
    phase_axes.append(phase_ax)

def plot_cost_curves(cost_axes, v_max):

    # Plot cost curves
    v_curve = np.linspace(0, v_max, 100)
    for i, alpha in enumerate(alphas):
        ax_cost = cost_axes[i]
        
        # Plot cost curve c(v) = (e^{α v} - 1)/α
        c_vals = [get_c(alpha, v) for v in v_curve]

        # Add cost function equation as text
        if alpha == -1:
            cost_eq = r'$c(v_i) = 1 - e^{-v_i}$'
        elif alpha == 0:
            cost_eq = r'$c(v_i) = v_i$'
        elif alpha == 1:
            cost_eq = r'$c(v_i) =e^{v_i} - 1$'

        

        ax_cost.plot(v_curve, c_vals, color=COLOR_BLACK, linewidth=LINEWIDTH, label=cost_eq)
        
        # Set title and labels
        if i == 0:
            ax_cost.set_ylabel('$c(v_i)$', fontsize=FONTSIZE)
        ax_cost.set_xlabel('$v_i$', fontsize=FONTSIZE)
        
        # Share x-axis with phase diagram below
        ax_cost.set_xlim(0, v_max)
        
        

        ax_cost.spines['top'].set_visible(False)
        ax_cost.spines['right'].set_visible(False)
        ax_cost.spines['left'].set_visible(False)
        ax_cost.spines['bottom'].set_visible(False)

        # Draw new axes lines
        ax_cost.axhline(0, color='black', lw=0.5)
        ax_cost.axvline(0, color='black', lw=0.5)

        # Legend
        # Add text for the line equation
        ax_cost.text(0.1, 0.8, cost_eq, transform=ax_cost.transAxes, fontsize=FONTSIZE, color=COLOR_BLACK,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0))

plot_cost_curves(cost_axes, v_max)

# Plot phase diagrams
for i, alpha in enumerate(alphas):
    U = np.zeros_like(V1)
    W = np.zeros_like(V2)

    for j in range(V1.shape[0]):
        for k in range(V1.shape[1]):
            u, w = best_response_dynamics(V1[j, k], V2[j, k], alpha, r)
            U[j, k] = u
            W[j, k] = w

    # Compute symmetric equilibrium
    v_star = compute_symmetric_equilibrium(alpha, r)

    ax = phase_axes[i]
    if alpha == 0:
        density = 0.45
    else:
        density =0.9

    ax.streamplot(V1, V2, U, W, color='grey', density=density, arrowsize=1)
    ax.set_xlabel("$v_1$", fontsize=FONTSIZE)
    if i == 0:
        ax.set_ylabel("$v_2$", fontsize=FONTSIZE)
    ax.set_xlim(0, v_max)
    ax.set_ylim(0, v_max)

    if alpha < 0:
        # Unstable equilibrium
        v_star = compute_symmetric_equilibrium(alpha, r)
        if v_star is not None and 0 <= v_star <= v_max:
            unstable = ax.scatter(v_star, v_star, color="white", edgecolors="black", s=100, linewidth=3, zorder=100)
        
        v_sentinel = compute_sentinel_equilibrium(alpha, r, v_max)
        if v_sentinel is not None and 0 <= v_sentinel <= v_max:
            # Generate -x,y line based on v_sentinel
            ax.scatter(0, v_sentinel, color=OPTIMAL_COLOR, edgecolor=OPTIMAL_EDGE_COLOR, linewidth=OPTIMAL_MARKER_EDGE_WIDTH, s=100, zorder=100)
            ax.scatter(v_sentinel, 0, color=OPTIMAL_COLOR, edgecolor=OPTIMAL_EDGE_COLOR, linewidth=OPTIMAL_MARKER_EDGE_WIDTH, s=100, zorder=100)

    if alpha == 0:
        v_star = compute_symmetric_equilibrium(alpha, r)
        if v_star is not None and 0 <= v_star <= v_max:
            # Generate -x,y line based on v_star
            x = np.linspace(0, 2*v_star, 100)
            y = 2*v_star - x
            ridge, = ax.plot(x, y, color="black", linewidth=2, zorder=100, linestyle='--')

    if alpha > 0:
        v_star = compute_symmetric_equilibrium(alpha, r)
        if v_star is not None and 0 <= v_star <= v_max:
            stable = ax.scatter(v_star, v_star, color=OPTIMAL_COLOR, edgecolor=OPTIMAL_EDGE_COLOR, s=100, zorder=100)

    ax.set_xlim(-0.1, v_max + .1)
    ax.set_ylim(-0.1, v_max + .1)

    # Hide axes lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Draw new axes lines
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.axhline(v_max, color='black', lw=0.5)
    ax.axvline(v_max, color='black', lw=0.5)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

# Create legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", label='Stable', markerfacecolor=OPTIMAL_COLOR, markeredgecolor=OPTIMAL_EDGE_COLOR, markersize=10, markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', label='Unstable', markerfacecolor='white', markeredgecolor='black', markersize=10, markeredgewidth=3),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Ridge')
]

phase_axes[1].legend(handles=legend_elements,
                    loc='upper center', 
                    bbox_to_anchor=(0.5, -0.15), 
                    ncol=3, 
                    frameon=False, 
                    fontsize=FONTSIZE)

# Annotate with letters
for i, ax in enumerate(cost_axes):
        ax.annotate(chr(97+i), xy=(0.06, 1.05), xycoords='axes fraction', fontsize=20, fontweight='bold')
for i, ax in enumerate(phase_axes):
        ax.annotate(chr(100+i), xy=(0.06, 1.05), xycoords='axes fraction', fontsize=20, fontweight='bold')

plt.savefig("images/dynamical_system_plot.png", dpi=600, bbox_inches='tight')
plt.show()