


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar

import seaborn as sns
import scipy
import csv

def append_to_csv(csv_list, output_filename):
    with open(output_filename, 'a', newline='') as fp:
        a = csv.writer(fp, delimiter=';')
        data = [csv_list]
        a.writerows(data)

    
def get_c_convex(v):

    return np.exp(v) - 1


def get_c_concave(v):

    return 1 - np.exp(-v)



def get_marginal_c_convex(v):

    return np.exp(v)

def get_c_convex(v):

    return 0.3*(np.exp(v/2) - 1)

def get_marginal_c_convex(v):

    return 0.3*(1/2*np.exp(v/2))



def get_c_inverse_s(v):

    switch_point = 3

    if v <= switch_point:
        return get_c_concave(v)
    else:
        return get_c_concave(switch_point) + get_c_convex(v - switch_point)


def get_c_s_shaped(v):

    switch_point = 1.5
    if v <= switch_point:
        return get_c_convex(v/10)*0.3
    else: 
        linear_gradient = get_marginal_c_convex(switch_point/10)*0.3

        return get_c_convex(switch_point)*0.3 + linear_gradient * get_c_concave((v - switch_point))



def get_c_convex_edges(v, i):

    n_edge = 8
    if i < n_edge:
        edge = True
    else:
        edge = False

    if edge:
        return 0.9* get_c_convex(v)
    else:
        return get_c_convex(v)


def plot_cost_curve():

    vv = np.linspace(0,10,100)
    cc = [get_c_s_shaped(v) for v in vv]

    plt.plot(vv,cc)
    plt.show()


def get_c(v_i, cost_type, i):

    if cost_type == "inverse S":
        return get_c_inverse_s(v_i)
    elif cost_type == "concave":
        return get_c_concave(v_i)
    elif cost_type == "convex":
        return get_c_convex(v_i)
    elif cost_type == "S shaped":
        return get_c_s_shaped(v_i)
    elif cost_type == "edge":
        return get_c_convex_edges(v_i, i)


def get_b(r, S):
    """
    Calculate the g value based on the given n and v values.
    """
    return r * (1 - np.exp(-S))

def get_f_i(r, vv, i, cost_type):
    """
    Calculate the f value based on the given alpha, r, S, and v values.
    """
    v_i = vv[i]

    c = get_c(v_i, cost_type, i)

    S = np.sum(vv)
    b = get_b(r, S)

    return b - c

def get_F(N, r, vv, i, cost_type):

    F = 0
    for j in range(N):
        F += get_f_i(r, vv, j, cost_type)
    
    return F


def best_response_selfish(r, vv, i, cost_type):

    v_max = 10

    N = len(vv)
    
    def objective(v_i_new):
        # Create a copy of vv with the new v_i value
        vv_temp = vv.copy()
        vv_temp[i] = v_i_new
        # Return negative F because minimize_scalar minimizes
        return -get_f_i(r, vv_temp, i, cost_type)
    
    # Find the optimal v_i
    result = minimize_scalar(objective, bounds=(0, v_max), method='bounded')

    return result.x    

def best_response_coop(r, vv, i, cost_type):

    v_max = 10

    N = len(vv)
    
    def objective(v_i_new):
        # Create a copy of vv with the new v_i value
        vv_temp = vv.copy()
        vv_temp[i] = v_i_new
        # Return negative F because minimize_scalar minimizes
        return -get_F(N, r, vv_temp, i, cost_type)
    
    # Find the optimal v_i
    result = minimize_scalar(objective, bounds=(0, v_max), method='bounded')

    return result.x


def best_response_coop_many_eyes_only(r, vv, i, v_max=10):

    N = len(vv)
    
    def objective(v_i_new):
        # Create a copy of vv with the new v_i value
        vv_temp = vv.copy()
        vv_temp = np.array([v_i_new]*N)
        # Return negative F because minimize_scalar minimizes
        return -get_F(N, r, vv_temp, i, cost_type="S shaped")
    
    # Find the optimal v_i
    result = minimize_scalar(objective, bounds=(0, v_max), method='bounded')

    return result.x

def best_response_coop_sentinel_only(r, vv, i, v_max=10):

    N = len(vv)
    vv_temp = np.zeros(N)

    def objective(v_i_new):
        # Create a copy of vv with the new v_i value
        
        vv_temp[0] = v_i_new
        # Return negative F because minimize_scalar minimizes
        return -get_F(N, r, vv_temp, i, cost_type="S shaped")
    
    # Find the optimal v_i
    result = minimize_scalar(objective, bounds=(0, v_max), method='bounded', options={'xatol': 1e-8})

    return result.x



def simulate(r, N, timesteps=1000, group_type="coop", cost_type="S shaped"):

    vv = np.random.uniform(0, 10, N)

    tolerance = 1e-5

    for _ in range(timesteps):
        last_vv = np.copy(vv)
        # Rnadomyl genearte an order for all players
        # to update their vigilance
        ii = np.random.permutation(N)
        
        # Update each player's vigilance based on the best response
        for i in ii:
            if group_type == "selfish":
                vv[i] = best_response_selfish(r=r, vv=vv, i=i, cost_type=cost_type)
            elif group_type == "coop":
                vv[i] = best_response_coop(r=r, vv=vv, i=i, cost_type=cost_type)


        # Check if the vigilance values have converged
        if np.allclose(vv, last_vv, atol=tolerance):
            break

    # After all timesteps, we have the final vigilance values

    # Count how many are non-zero
    n_watchers = np.sum(vv > tolerance)

    return n_watchers

def simulate_many_eyes_only(r, N):

    vv = np.random.uniform(0, 10, N)

    tolerance = 1e-5

    v = best_response_coop_many_eyes_only(r=r, vv=vv, i=0)
    return v

def simulate_sentinel_only(r, N):

    vv = np.random.uniform(0, 10, N)

    tolerance = 1e-5

    v = best_response_coop_sentinel_only(r=r, vv=vv, i=0)
    return v


def simulate_behavioural_switching_coop():

    r = 0.01

    # cooperative
    n_optimal = simulate(r=r, N=16, timesteps=1000)
    v_many_eyes = simulate_many_eyes_only(r=r, N=16)
    v_sentinel = simulate_sentinel_only(r=r, N=16)

    print("r:", r)
    print("n_optimal:", n_optimal)
    print("v_many_eyes:", v_many_eyes)
    print("v_sentinel:", v_sentinel)

    r = 1

    # cooperative
    n_optimal = simulate(r=r, N=16, timesteps=1000)
    v_many_eyes = simulate_many_eyes_only(r=r, N=16)
    v_sentinel = simulate_sentinel_only(r=r, N=16)

    print("\nr:", r)
    print("n_optimal:", n_optimal)
    print("v_many_eyes:", v_many_eyes)
    print("v_sentinel:", v_sentinel)


def simulate_behavioural_switching_selfish():

    r = 1

    # selfish
    n_optimal = simulate(r=r, N=16, timesteps=1000, group_type="selfish")
    

    print("r:", r)
    print("n_optimal:", n_optimal)
    
    r = 20

    # selfish
    n_optimal = simulate(r=r, N=16, timesteps=1000, group_type="selfish")
    
    print("\nr:", r)
    print("n_optimal:", n_optimal)
    
if __name__ == "__main__":
    
    simulate_behavioural_switching_coop()