import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arc, Rectangle, FancyBboxPatch, Ellipse, RegularPolygon
from matplotlib.transforms import blended_transform_factory


COLOR_LOW_VIGILANCE = "#3EFF8E"
COLOR_MEDIUM_VIGILANCE = "#25AB5D"
COLOR_HIGH_VIGILANCE = "#3498db"
COLOR_BLACK = "#404040"
FONTSIZE=16
MARKER_SIZE = 100

def generate_group_positions(n_edge=10, n_interior=6, edge_radius=1.0, 
                            interior_radius=0.6, min_dist=0.25, 
                            max_attempts=1000, seed=2):
    """
    Generate positions for a group with edge and interior individuals.
    
    Parameters:
    -----------
    n_edge : int
        Number of individuals on the edge
    n_interior : int
        Number of individuals in the interior
    edge_radius : float
        Radius of the edge circle
    interior_radius : float
        Maximum radius for interior positions
    min_dist : float
        Minimum distance between all individuals
    max_attempts : int
        Maximum attempts to place each individual
    
    Returns:
    --------
    positions : list of tuples
        [(x1, y1), (x2, y2), ...] for all individuals
    edge_indices : list
        Indices of edge individuals
    interior_indices : list
        Indices of interior individuals
    """
    
    np.random.seed(seed)

    # Generate edge positions - on a circle with separation constraint
    x_edge = []
    y_edge = []
    
    # Start with evenly spaced angles
    base_angles = np.linspace(0, 2*np.pi, n_edge, endpoint=False)
    
    for i in range(n_edge):
        placed = False
        for attempt in range(max_attempts):
            # Start from base angle with increasing jitter range
            jitter_range = 0.2 + (attempt / max_attempts) * 0.3  # increase jitter if struggling
            angle = base_angles[i] + np.random.uniform(-jitter_range, jitter_range)
            
            x_new = edge_radius * np.cos(angle)
            y_new = edge_radius * np.sin(angle)
            
            # Check if first point
            if len(x_edge) == 0:
                x_edge.append(x_new)
                y_edge.append(y_new)
                placed = True
                break
            
            # Check distance to all existing edge points
            distances = np.sqrt((np.array(x_edge) - x_new)**2 + 
                              (np.array(y_edge) - y_new)**2)
            
            if np.all(distances > min_dist):
                x_edge.append(x_new)
                y_edge.append(y_new)
                placed = True
                break
        
        if not placed:
            print(f"Warning: Could not place edge individual {i} after {max_attempts} attempts")
            print(f"Consider increasing edge_radius or decreasing n_edge or min_dist")
    
    x_edge = np.array(x_edge)
    y_edge = np.array(y_edge)
    
    # Generate interior positions with separation
    x_interior = []
    y_interior = []
    
    for i in range(n_interior):
        placed = False
        for attempt in range(max_attempts):
            # Propose new position
            x_new = np.random.uniform(-interior_radius, interior_radius)
            y_new = np.random.uniform(-interior_radius, interior_radius)
            
            # Check if first point
            if len(x_interior) == 0:
                x_interior.append(x_new)
                y_interior.append(y_new)
                placed = True
                break
            
            # Check distance to all existing interior points
            distances = np.sqrt((np.array(x_interior) - x_new)**2 + 
                              (np.array(y_interior) - y_new)**2)
            
            # Also check distance to edge points
            distances_to_edge = np.sqrt((x_edge - x_new)**2 + (y_edge - y_new)**2)
            
            if np.all(distances > min_dist) and np.all(distances_to_edge > min_dist):
                x_interior.append(x_new)
                y_interior.append(y_new)
                placed = True
                break
        
        if not placed:
            print(f"Warning: Could not place interior individual {i} after {max_attempts} attempts")
    
    # Combine into list of tuples
    positions = [(x_edge[i], y_edge[i]) for i in range(len(x_edge))]
    positions.extend([(x_interior[i], y_interior[i]) for i in range(len(x_interior))])
    
    # Create index lists
    edge_indices = list(range(len(x_edge)))
    interior_indices = list(range(len(x_edge), len(x_edge) + len(x_interior)))
    
    return positions, edge_indices, interior_indices


def get_c_convex(v):

    return 0.3*(np.exp(v/2) - 1)

def get_marginal_c_convex(v):

    return 0.3*(1/2*np.exp(v/2))

def get_c_concave(v):

    return 1 - np.exp(-v)

def get_c_inverse_s(v):

    switch_point = 3

    if v <= switch_point:
        return get_c_concave(v)
    else:
        return get_c_concave(switch_point) + get_c_convex(v - switch_point)



def get_c_s_shaped(v):

    switch_point = 1.5
    if v <= switch_point:
        return get_c_convex(v)
    else: 
        linear_gradient = get_marginal_c_convex(switch_point)

        return get_c_convex(switch_point) + linear_gradient * get_c_concave((v - switch_point))


LEVELS_AND_COLORS = {
    "none": "white",
    "low": COLOR_LOW_VIGILANCE,
    "medium": COLOR_MEDIUM_VIGILANCE,
    "high": COLOR_HIGH_VIGILANCE
}

def add_low(x,y,ax, circle_radius, level="low"):

    alpha = 1


    circle = Circle((x, y), circle_radius, fill=True, alpha = alpha,
                    linewidth=0, facecolor=LEVELS_AND_COLORS[level])
    
    ax.add_artist(circle)
    
    arc = Arc((x, y), circle_radius*2,    circle_radius*2, angle=0, 
                      theta1=0, theta2=360, 
                      linewidth=3, color="#404040")

    ax.add_artist(arc)

def add_mid(x,y,ax, circle_radius):

    alpha = 1
    
    # Filled pentagon
    pentagon = RegularPolygon((x, y), 
                              numVertices=5,
                              radius=circle_radius,
                              fill=True,
                              alpha=alpha,
                              linewidth=0,
                              facecolor=LEVELS_AND_COLORS["medium"])
    
    ax.add_artist(pentagon)
    
    # Outline pentagon
    pentagon = RegularPolygon((x, y), 
                              numVertices=5,
                              radius=circle_radius,
                              fill=False,
                              linewidth=3,
                              edgecolor="#404040")
    
    ax.add_artist(pentagon)


def add_high(x,y,ax, circle_radius):

    alpha = 1

    square = Rectangle((x - circle_radius, y - circle_radius), 
       width=circle_radius*2, 
       height=circle_radius*2,
       fill=True, 
       alpha=alpha,
       linewidth=0, 
       facecolor=LEVELS_AND_COLORS["high"])

    ax.add_artist(square)

    square = Rectangle((x - circle_radius, y - circle_radius), 
       width=circle_radius*2, 
       height=circle_radius*2,
       fill=False, 
       linewidth=3, 
       color="#404040")

    ax.add_artist(square)

def add_zero(x, y, ax, circle_radius):

    alpha = 1

    # Filled triangle
    triangle = RegularPolygon((x, y), 
                              numVertices=3,
                              radius=circle_radius,
                              fill=True,
                              alpha=alpha,
                              linewidth=0,
                              facecolor=LEVELS_AND_COLORS["none"])
    
    ax.add_artist(triangle)
    
    # Outline triangle
    triangle = RegularPolygon((x, y), 
                              numVertices=3,
                              radius=circle_radius,
                              fill=False,
                              linewidth=3,
                              edgecolor="#404040")
    
    ax.add_artist(triangle)


def plot_circles(sim_type="sentinel", ax=None):

    circle_radius = 0.1

    if sim_type == "edge":

        positions, edge_indices, interior_indices = generate_group_positions(
            n_edge=10, n_interior=6, edge_radius=1.0, 
            interior_radius=0.6, min_dist=0.4, max_attempts=1000, seed=3)

        for i, (x,y) in enumerate(positions):
            if i in edge_indices:
                add_mid(x,y,ax, circle_radius)
            else:
                add_low(x,y,ax, circle_radius)

    elif sim_type == "many-eyes":
        positions, edge_indices, interior_indices = generate_group_positions(
            n_edge=8, n_interior=8, edge_radius=1.0, 
            interior_radius=0.6, min_dist=0.4, max_attempts=1000, seed=0)

        for i, (x,y) in enumerate(positions):
            add_low(x,y,ax, circle_radius)


    elif sim_type == "sentinel":
        positions, edge_indices, interior_indices = generate_group_positions(
            n_edge=8, n_interior=8, edge_radius=1.0, 
            interior_radius=0.6, min_dist=0.4, max_attempts=1000, seed=0)

        sentinel_index = 12

        for i, (x,y) in enumerate(positions):
            if i == sentinel_index:
                add_high(x,y,ax, circle_radius)
            else:
                add_zero(x,y,ax, circle_radius)

    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)

    ax.set_aspect('equal')
    ax.set_axis_off()


def plot_cost_curve(sim_type="edge", ax=None):

    COLOR_BLACK = "#404040"
    LINEWIDTH = 3   
    
    if sim_type == "many-eyes":
        v_max = 10
        v_curve = np.linspace(0, v_max, 100)
        cs = [get_c_s_shaped(v) for v in v_curve]
        ax.set_xlim(0, v_max*1.1)  
        ax.plot(v_curve, cs, color=COLOR_BLACK, linewidth = LINEWIDTH)

        v_star_many_eyes = 0.22
        c_star_many_eyes = get_c_s_shaped(v_star_many_eyes)
        gradient_star = c_star_many_eyes/v_star_many_eyes

        many_eyes_line = [gradient_star*v for v in v_curve]
        ax.plot(v_curve, many_eyes_line, color=COLOR_LOW_VIGILANCE, linewidth = LINEWIDTH, linestyle="--")
        ax.scatter(v_star_many_eyes, c_star_many_eyes, color=COLOR_LOW_VIGILANCE, s=MARKER_SIZE, edgecolors=COLOR_BLACK, linewidth=LINEWIDTH, zorder=5)


        v_star_sentinel = 1.5
        c_star_sentinel = get_c_s_shaped(v_star_sentinel)

        gradient_sentinel = c_star_sentinel/v_star_sentinel
        sentinel_line = [gradient_sentinel*v for v in v_curve]

        ax.plot(v_curve, sentinel_line, color="lightgrey", linewidth = LINEWIDTH, linestyle="--")
        ax.scatter(v_star_sentinel, c_star_sentinel, color="lightgray", s=MARKER_SIZE, edgecolors="grey", linewidth=LINEWIDTH, zorder=5, marker="s")

        y_top = get_c_s_shaped(v_max)*1.1
        ax.set_ylim(bottom=-y_top*0.05, top=y_top)

    elif sim_type == "sentinel":
        v_max = 10
        v_curve = np.linspace(0, v_max, 100)
        cs = [get_c_s_shaped(v) for v in v_curve]
        ax.set_xlim(0, v_max*1.1)  
        ax.plot(v_curve, cs, color=COLOR_BLACK, linewidth = LINEWIDTH)

        v_star_many_eyes = 0.51
        c_star_many_eyes = get_c_s_shaped(v_star_many_eyes)
        gradient_star = c_star_many_eyes/v_star_many_eyes

        many_eyes_line = [gradient_star*v for v in v_curve]
        ax.plot(v_curve, many_eyes_line, color="lightgrey", linewidth = LINEWIDTH, linestyle="--")
        ax.scatter(v_star_many_eyes, c_star_many_eyes, color="lightgrey", s=MARKER_SIZE, edgecolors="grey", linewidth=LINEWIDTH, zorder=5)

        v_star_sentinel = v_max
        c_star_sentinel = get_c_s_shaped(v_star_sentinel)

        gradient_sentinel = c_star_sentinel/v_star_sentinel
        sentinel_line = [gradient_sentinel*v for v in v_curve]

        ax.plot(v_curve, sentinel_line, color=COLOR_HIGH_VIGILANCE, linewidth = LINEWIDTH, linestyle="--")
        ax.scatter(v_star_sentinel, c_star_sentinel, color=COLOR_HIGH_VIGILANCE, s=MARKER_SIZE, edgecolors=COLOR_BLACK, linewidth=LINEWIDTH, zorder=5, marker="s")

        y_top = get_c_s_shaped(v_max)*1.1
        ax.set_ylim(bottom=-y_top*0.05, top=y_top)

        
    elif sim_type == "edge":
        v_max = 3

        v_curve = np.linspace(0, v_max, 100)
        cs_1 = [get_c_convex(v)*2 for v in v_curve]
        cs_2 = [get_c_convex(v) for v in v_curve]

        ax.set_xlim(0, v_max*1.1)
        ax.plot(v_curve, cs_1, color=COLOR_BLACK, linewidth = LINEWIDTH)
        ax.plot(v_curve, cs_2, color=COLOR_BLACK, linewidth = LINEWIDTH)

        v_star_1 = 0.75
        c_star_1 = get_c_convex(v_star_1)*2
        marginal_c_star_1 = get_marginal_c_convex(v_star_1)*2
        tangent_curve = [marginal_c_star_1*(v - v_star_1) + c_star_1 for v in v_curve]
        # Remove negative costs, replace with NaNs
        tangent_curve = [tc if tc >= 0 else np.nan for tc in tangent_curve]

        ax.plot(v_curve, tangent_curve, color=COLOR_LOW_VIGILANCE, linewidth = LINEWIDTH, linestyle="dotted")
        ax.scatter(v_star_1, c_star_1, color=COLOR_LOW_VIGILANCE, s=MARKER_SIZE, edgecolors=COLOR_BLACK, linewidth=LINEWIDTH, zorder=5)

        print(marginal_c_star_1)

        v_star_2 = 2 * (np.log(marginal_c_star_1) - np.log(3/20))

        c_star_2 = get_c_convex(v_star_2)
        tangent_curve_2 = [marginal_c_star_1*(v - v_star_2) + c_star_2 for v in v_curve]
        tangent_curve_2 = [tc if tc >= 0 else np.nan for tc in tangent_curve_2]

        

        c_max_2 = get_c_convex(v_max)

        y_top = c_max_2
        ax.set_ylim(bottom=-y_top*0.05, top=y_top)


        ax.plot(v_curve, tangent_curve_2, color=COLOR_MEDIUM_VIGILANCE, linewidth = LINEWIDTH, linestyle="dotted")
        ax.scatter(v_star_2, c_star_2, color=COLOR_MEDIUM_VIGILANCE, s=MARKER_SIZE, edgecolors=COLOR_BLACK, linewidth=LINEWIDTH, zorder=5,  marker="p")


    # Share x-axis with phase diagram below
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

        # Draw new axes lines
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)

   

def plot_group_schematics():

    """
    PLot 3 subplots, one for each of edges, many-eyes and senitnels
    """

    fig, axs = plt.subplots(2, 3, figsize=(20,10))

    plot_cost_curve("many-eyes", axs[0,0])
    plot_cost_curve("sentinel", axs[0,1])
    plot_cost_curve("edge", axs[0,2])

    axs[0,0].set_ylabel("Cost of Vigilance, $c(v)$", fontsize=FONTSIZE)

    for ax in axs[0,:]:
        ax.set_xlabel("Vigilance, $v$", fontsize=FONTSIZE)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

  
    plot_circles("many-eyes", axs[1,0])
    plot_circles("sentinel", axs[1,1])  
    plot_circles("edge", axs[1,2])

    # Annotate with letters
    for i, ax in enumerate([axs[0,0], axs[0,1], axs[0,2]]):
        ax.annotate(chr(97+i), xy=(0.03, .95), xycoords='axes fraction', fontsize=20, fontweight='bold')
        # Make x-axis slightly bigger

    legend_size = 159
    legend_elements = [
        plt.scatter(0, 10, marker = "^", facecolor='white', edgecolor=COLOR_BLACK, linewidth=3, s=legend_size, label='No vigilance'),
        plt.scatter(0, 10, facecolor=COLOR_LOW_VIGILANCE, edgecolor=COLOR_BLACK, linewidth=3, s=legend_size, label='Many-Eyes (low)'),
        plt.scatter(0, 10, marker="p", facecolor=COLOR_MEDIUM_VIGILANCE, edgecolor=COLOR_BLACK, linewidth=3, s=legend_size, label='Many-Eyes (medium)'),
        plt.scatter(0, 10, marker="s", facecolor=COLOR_HIGH_VIGILANCE, edgecolor=COLOR_BLACK, linewidth=3, s=legend_size, label='Sentinel')
    ]
    
    # Add legend centered below the figure
    fig.legend(handles=legend_elements, 
              loc='lower center', 
              ncol=4, 
              fontsize=FONTSIZE,
              frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(bottom=0.08)  # Make room for legend

    plt.suptitle("                  Behavioural Switching                                           Edge Effects", fontsize=FONTSIZE+4)

    plt.savefig("images/extended_schematics.png", dpi=600, bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    plot_group_schematics()