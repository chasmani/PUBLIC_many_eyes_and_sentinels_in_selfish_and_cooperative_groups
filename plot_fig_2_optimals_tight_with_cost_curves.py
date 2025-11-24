import math

import numpy as np
import matplotlib.pyplot as plt

OPTIMAL_COLOR = "#F62459"
OPTIMAL_EDGE_COLOR = "white"
OPTIMAL_MARKER_SIZE = 50
OPTIMAL_MARKER_LINEWIDTH = 2

LINEWIDTH = 3
COLOR_MANY_EYES = "#2ecc71"
COLOR_SENTINEL = "#3498db"
COLOR_S_SHAPED = "#404040"
CMAP = "Greys"

COLOR_BLACK = '#0f0f0f'
LINEWIDTH = 3
FONTSIZE=12

def get_c(alpha, v):
    return (np.exp(alpha * v) - 1) / alpha if alpha != 0 else v

def get_b(r, S):
    """
    Calculate the benefit based on the given r and S.
    """
    return r * (1 - np.exp(-S))

def get_f_bar(N, alpha, r, n, v):
     

    S = n*v
    b = get_b(r, S)
    B = N*b

    c = get_c(alpha, v)
    C = n*c

    F = B - C
    
    if F < 0:
        return np.nan
    
    return F/N

def get_v_star(alpha, n):

    return np.log(1-n*alpha)/(-alpha)

def plot_cost_curves(cost_axes, v_max=3):

    alphas = [-1, 0, 1]

    # Plot cost curves
    v_curve = np.linspace(0, v_max, 100)
    for i, alpha in enumerate(alphas):
        ax_cost = cost_axes[i]
        
        # Plot cost curve c(v) = (e^{α v} - 1)/α
        c_vals = [get_c(alpha, v) for v in v_curve]

        # Add cost function equation as text
        if alpha == -1:
            cost_eq = r'$c(v) = 1 - e^{-v}$'
        elif alpha == 0:
            cost_eq = r'$c(v) = v$'
        elif alpha == 1:
            cost_eq = r'$c(v) =e^{v} - 1$'



        ax_cost.plot(v_curve, c_vals, color=COLOR_BLACK, linewidth=LINEWIDTH, label=cost_eq)
        
        # Set title and labels
        if i == 0:
            ax_cost.set_ylabel('Cost of Vigilance, $c(v)$', fontsize=FONTSIZE)
        ax_cost.set_xlabel('Vigilance, $v$', fontsize=FONTSIZE)
        
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
        ax_cost.text(0.1, 0.9, cost_eq, transform=ax_cost.transAxes, fontsize=FONTSIZE, color=COLOR_BLACK,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0))



def plot_optimals():

    N = 16

    n_vigilances = 1000
    vigilances = np.linspace(0,3, n_vigilances)

    fig = plt.figure(figsize=(12, 7))
    gs = plt.GridSpec(3, 3, figure=fig, 
                      height_ratios=[1, 2, 0.05],
                      # Add some guttering
                      hspace=0.6, wspace=0.2)


    cost_axes = []
    for i in range(3):
        cost_axes.append(fig.add_subplot(gs[0, i]))

    plot_cost_curves(cost_axes, v_max=3)
    

    ############# 
    # Sentinels
    #############
    
    #############
    # HEATMAPS
    #############

    ns = np.arange(0, N+1)

    utilities_indifference = np.zeros((len(ns), len(vigilances)))
    utilities_many_eyes = np.zeros((len(ns), len(vigilances)))
    utilities_sentinel = np.zeros((len(ns), len(vigilances)))

    r = 1

    alpha_indifference = 0
    alpha_many_eyes = 1
    alpha_sentinel = -1

    for i, n in enumerate(ns):
        for j, v in enumerate(vigilances):
            utilities_indifference[i, j] = get_f_bar(N=N, alpha=alpha_indifference, r=r, n=n, v=v)
            utilities_many_eyes[i, j] = get_f_bar(N=N, alpha=alpha_many_eyes, r=r, n=n, v=v)
            utilities_sentinel[i, j] = get_f_bar(N=N, alpha=alpha_sentinel, r=r, n=n, v=v)
            
    cmin = min(np.nanmin(utilities_indifference), np.nanmin(utilities_many_eyes), np.nanmin(utilities_sentinel))
    cmax = max(np.nanmax(utilities_indifference), np.nanmax(utilities_many_eyes), np.nanmax(utilities_sentinel))

    from matplotlib.colors import CenteredNorm, PowerNorm, SymLogNorm, LogNorm
    # Heatmaps
    ax4 = fig.add_subplot(gs[1, 0], sharex=cost_axes[0])

    norm = PowerNorm(gamma=2, vmin=cmin, vmax=cmax)

    im_sentinel = ax4.imshow(utilities_sentinel, extent=[min(vigilances), max(vigilances), 0, N+1], cmap=CMAP, aspect='auto', origin='lower',
                              norm=norm)
    
    plt.xlabel('Watcher Vigilance, v', fontsize=FONTSIZE)
    plt.ylabel('Watchers, n', fontsize=FONTSIZE)

    n_max_sentinel, v_max_sentinel_index = np.unravel_index(np.nanargmax(utilities_sentinel, axis=None), utilities_sentinel.shape)
    v_max_sentinel = v_max_sentinel_index/n_vigilances * (max(vigilances) - min(vigilances)) + min(vigilances)
    
    # Add black circle at minimum for sentinel (ax4)
    ax4.scatter(x=v_max_sentinel, y=n_max_sentinel+0.5, color=OPTIMAL_COLOR, s=OPTIMAL_MARKER_SIZE, edgecolor=OPTIMAL_EDGE_COLOR, linewidth=OPTIMAL_MARKER_LINEWIDTH)

    ax5 = fig.add_subplot(gs[1, 1], sharey=ax4, sharex=cost_axes[1])

    im_indifference = ax5.imshow(utilities_indifference, extent=[min(vigilances), max(vigilances), 0, N+1], cmap=CMAP, aspect='auto', origin='lower',
                                    norm=norm)
    plt.xlabel('Watcher Vigilance, v', fontsize=FONTSIZE)
    
    # Ad black circles at maximum for each n
    for n in range(1,N+1):
        v_max_indifference_index = np.nanargmax(utilities_indifference[n, :])
        v_max_indifference = v_max_indifference_index/n_vigilances * (max(vigilances) - min(vigilances)) + min(vigilances)
        ax5.scatter(x=v_max_indifference, y=n+0.5, color=OPTIMAL_COLOR, s=OPTIMAL_MARKER_SIZE, edgecolor=OPTIMAL_EDGE_COLOR, linewidth=OPTIMAL_MARKER_LINEWIDTH)

    
    ax6 = fig.add_subplot(gs[1, 2], sharey=ax4, sharex=cost_axes[2])

    im_many_eyes = ax6.imshow(utilities_many_eyes, extent=[min(vigilances), max(vigilances), 0, N+1], cmap=CMAP, aspect='auto', origin='lower',
                              norm=norm)

    plt.xlabel('Watcher Vigilance, v', fontsize=FONTSIZE)
    
    # Add black circle at maximum for many eyes (ax6)
    n_max_many_eyes, v_max_many_eyes_index = np.unravel_index(np.nanargmax(utilities_many_eyes, axis=None), utilities_many_eyes.shape)
    v_max_many_eyes = v_max_many_eyes_index/n_vigilances * (max(vigilances) - min(vigilances)) + min(vigilances)
    ax6.scatter(x=v_max_many_eyes, y=n_max_many_eyes+0.5, color=OPTIMAL_COLOR, s=OPTIMAL_MARKER_SIZE, edgecolor=OPTIMAL_EDGE_COLOR, linewidth=OPTIMAL_MARKER_LINEWIDTH)

    for ax in [ax4, ax5, ax6]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            # PLot axis lines
            ax.axhline(0, color='grey', linewidth=0.5)
            ax.axvline(0, color='grey', linewidth=0.5)

    yticks = np.arange(0, N+1, 2)
    ax5.set_yticks(yticks + 0.5)
    ax5.set_yticklabels(yticks)

    cax = fig.add_subplot(gs[2, 1])
    cbar = plt.colorbar(im_sentinel, cax=cax, orientation='horizontal', pad=0.1)
    cbar.set_label(r'Average Fitness, $\bar{f}$', fontsize=FONTSIZE)
    current_ticks = cbar.get_ticks()
    new_ticks = current_ticks[[0,2,3]]  # Remove index 1 (second tick)
    cbar.set_ticks(new_ticks)

    # Annotate with letters
    for i, ax in enumerate([ax4, ax5, ax6]):
        ax.annotate(chr(100+i), xy=(0.03, 1.05), xycoords='axes fraction', fontsize=20, fontweight='bold')
        # Make x-axis slightly bigger
        ax.set_xlim(min(vigilances), max(vigilances) + 0.2)
        # Set equal aspect ration

    for i, ax in enumerate(cost_axes):
        ax.annotate(chr(97+i), xy=(0.03, 1.05), xycoords='axes fraction', fontsize=20, fontweight='bold')

        
    plt.savefig("images/optimals.png", dpi=600)
   
    plt.show()
    

plot_optimals()