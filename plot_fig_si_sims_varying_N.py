import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

COLOR_MANY_EYES = "#2ecc71"
COLOR_SENTINEL_N_1 = "#3498db"
COLOR_SENTINEL_N_2 = "#1d6ca1"
COLOR_SENTINEL_N_3 = "#145A80"
COLOR_SENTINEL_N_4 = "#0C3B5E"
COLOR_NO_VIGILANCE = "#404040"
FONTSIZE = 16

def plot_heatmaps_for_N(N):


    # Load the data
    df = pd.DataFrame(pd.read_csv('sim_results_N_{N}.csv'.format(N=N), delimiter=';',
                                names=["group_type", "N", "v_max", "r", "alpha", "r_max", "alpha_mod_max", "resolution", "n_watchers", "f_mean"]))


    if N == 4:

        # Define custom colors for each classification
        classification_colors = {
            0: COLOR_NO_VIGILANCE, 
            1: COLOR_SENTINEL_N_1,  
            2: COLOR_SENTINEL_N_2,  
            3: COLOR_SENTINEL_N_3,
            4: COLOR_MANY_EYES,
        }

    if N == 64:
        
        # Define custom colors for each classification
        classification_colors = {
            64: COLOR_MANY_EYES,
            0: COLOR_NO_VIGILANCE, 
            1: COLOR_SENTINEL_N_1,  
            2: COLOR_SENTINEL_N_2,  
            3: COLOR_SENTINEL_N_3,
            4: COLOR_SENTINEL_N_4,
        }

    def create_heatmap_data(df_subset):
        """Create pivot table and color matrix for a subset of data"""
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = df_subset.pivot_table(
            values='n_watchers', 
            index='r', 
            columns='alpha',
            aggfunc='first'  # In case there are duplicates
        )
        
        # Map each classification directly to its color
        color_matrix = np.zeros(pivot_df.shape + (4,))  # RGBA matrix
        
        # Create a color matrix directly
        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                cell_value = pivot_df.iloc[i, j]
                if pd.notnull(cell_value):
                    # Convert color to RGBA
                    color_matrix[i, j] = to_rgba(classification_colors[cell_value])
                else:
                    # Transparent for NaN values
                    color_matrix[i, j] = (0, 0, 0, 0)
        
        return pivot_df, color_matrix

    def find_nearest_indices(array, target_values):
        """Find indices in array that are closest to target values"""
        indices = []
        for target in target_values:
            idx = np.argmin(np.abs(array - target))
            indices.append(idx)
        return indices

    # Filter data for each group type
    df_selfish = df[df['group_type'] == 'selfish']
    df_coop = df[df['group_type'] == 'coop']

    print(df_coop)
    print(df_selfish)

    # Create data for both heatmaps
    pivot_selfish, color_matrix_selfish = create_heatmap_data(df_selfish)
    pivot_coop, color_matrix_coop = create_heatmap_data(df_coop)

    # Define desired tick values
    desired_r_values = [0.01, 0.1, 1, 10, 100]
    desired_alpha_values = [-2, -1, 0, 1, 2]

    # Find indices for desired tick positions (using selfish data as reference)
    r_tick_indices = find_nearest_indices(pivot_selfish.index.values, desired_r_values)
    alpha_tick_indices = find_nearest_indices(pivot_selfish.columns.values, desired_alpha_values)
    # Find indices for coop data (in case it has different dimensions)
    r_tick_indices_coop = find_nearest_indices(pivot_coop.index.values, desired_r_values)
    alpha_tick_indices_coop = find_nearest_indices(pivot_coop.columns.values, desired_alpha_values)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Right subplot - Cooperative
    ax1.imshow(color_matrix_coop, origin='lower')
    ax1.set_title('Cooperative', fontsize=24, fontweight='bold')
    ax1.set_ylabel(r'Predation Threat, $r$', fontsize=16)


    # Set custom ticks for coop plot
    ax1.set_xticks(alpha_tick_indices_coop)
    ax1.set_xticklabels(desired_alpha_values)
    ax1.set_yticks(r_tick_indices_coop)
    ax1.set_yticklabels(desired_r_values)
    ax1.set_aspect('equal')

    # Left subplot - Selfish
    ax2.imshow(color_matrix_selfish, origin='lower')
    ax2.set_title('Selfish', fontsize=24, fontweight='bold')

    # Set custom ticks for selfish plot
    ax2.set_xticks(alpha_tick_indices)
    ax2.set_xticklabels(desired_alpha_values)
    ax2.set_yticks(r_tick_indices)
    ax2.set_yticklabels(desired_r_values)
    ax2.set_aspect('equal')

    for ax in (ax1, ax2):
        ax.text(50, -12, r'Vigilance Cost Scaling, $\alpha$', ha='center', va='top', fontsize=16)
            
            # Annotate line type
        ax.text(10, -6, r"Concave $\alpha < 0$", ha='center', va='top', fontsize=FONTSIZE)
        ax.text(50, -6, r"Linear $\alpha = 0$", ha='center', va='top', fontsize=FONTSIZE)
        ax.text(90, -6, r"Convex $\alpha > 0$", ha='center', va='top', fontsize=FONTSIZE)


    ax1.text(2, 100, "a", ha='center', va='bottom', fontsize=24, fontweight='bold')
    ax2.text(2, 100, "b", ha='center', va='bottom', fontsize=24, fontweight='bold')


    if N == 4:
        ax1.text(50, 15, "No Vigilance", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(75, 65, "Many-Eyes", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(25, 45, r"Sentinel $n=1$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(25, 80, r"Sentinel $n=2$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(8, 95, r"$n=3$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")

        ax2.text(50, 25, "No Vigilance", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax2.text(75, 75, "Many-Eyes", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax2.text(25, 60, r"Sentinel $n=1$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax2.text(25, 90, r"Sentinel $n=2$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")

    if N == 64:

        ax1.text(50, 2, "No Vigilance", ha='center', va='center', fontsize=FONTSIZE/2, color="#ffffff")
        ax1.text(75, 55, "Many-Eyes", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(25, 15, r"Sentinel $n=1$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(25, 50, r"Sentinel $n=2$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(25, 80, r"Sentinel $n=3$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax1.text(7, 97, r"$n=4$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")

        ax2.text(50, 25, "No Vigilance", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax2.text(75, 75, "Many-Eyes", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax2.text(25, 60, r"Sentinel $n=1$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")
        ax2.text(25, 90, r"Sentinel $n=2$", ha='center', va='center', fontsize=FONTSIZE, color="#ffffff")



    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Optional: Save the figure
    plt.savefig("images/sims_parameter_space_N_{}.png".format(N), dpi=600, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    plot_heatmaps_for_N(N=4)