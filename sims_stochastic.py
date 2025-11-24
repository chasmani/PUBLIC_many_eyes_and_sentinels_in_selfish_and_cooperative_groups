


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import seaborn as sns
import scipy
import csv

def append_to_csv(csv_list, output_filename):
    with open(output_filename, 'a', newline='') as fp:
        a = csv.writer(fp, delimiter=';')
        data = [csv_list]
        a.writerows(data)

    
def get_h(alpha, v):
    """
    Calculate the h value based on the given alpha and v values.
    """
    if alpha == 0:
        return v
    return (np.exp(alpha*v) - 1)/alpha

def get_b(r, S):
    """
    Calculate the g value based on the given n and v values.
    """
    return r * (1 - np.exp(-S))

def get_f_i(alpha, r, vv, i):
    """
    Calculate the f value based on the given alpha, r, S, and v values.
    """
    v_i = vv[i]

    h = get_h(alpha, v_i)

    S = np.sum(vv)
    b = get_b(r, S)

    return b - h

def get_F(N, alpha, r, vv):

    F = 0
    for i in range(N):
        F += get_f_i(alpha, r, vv, i)
        
    return F

def get_dh_dv_i(alpha, v_i):

    return np.exp(alpha * v_i)

def get_db_dv_i(r, vv, i):
    """
    Calculate the derivative of b with respect to v_i.
    """
    S = np.sum(vv)
    return r * np.exp(-S)

def get_df_dv_i(alpha, r, vv, i):
    """
    Calculate the derivative of f with respect to v_i.
    """
    db_dv_i = get_db_dv_i(r, vv, i)
    dh_dv_i = get_dh_dv_i(alpha, vv[i])
    return db_dv_i -dh_dv_i

def get_dF_dv_i(N, alpha, r, vv, i):

    db_dv_i = get_db_dv_i(r, vv, i)
    dh_dv_i = get_dh_dv_i(alpha, vv[i])
    return db_dv_i - (dh_dv_i / N)




def best_response_coop(r, vv, alpha, i, v_max=10):

    N = len(vv)
    
    # Objective function: negative of group fitness (to minimize)
    def neg_F(v_i):
        vv_temp = np.copy(vv)
        vv_temp[i] = v_i
        return -get_F(N, alpha, r, vv_temp)

    # Use bounded scalar minimization
    res = scipy.optimize.minimize_scalar(
        neg_F,
        bounds=(0, v_max),
        method='bounded'
    )
    
    return res.x

def best_response_selfish(r, vv, alpha, i, v_max=3):

    def neg_f_i(v_i):
        vv_temp = np.copy(vv)
        vv_temp[i] = v_i
        return -get_f_i(alpha, r, vv_temp, i)

    # Use bounded scalar minimization
    res = scipy.optimize.minimize_scalar(
        neg_f_i,
        bounds=(0, v_max),
        method='bounded'
    )
    
    return res.x


def get_stationary_point_v_given_n(r, alpha, n, N=10):
    """
    Find the Nash Equilibrium given n agents.
    """

    v_star = np.log(r) / (alpha + n)
    return v_star

def check_perturbations(r, alpha, n, N=10, v_max=10):
    """
    Check the perturbations around the stationary point.
    """
    n = 2
    v_star = get_stationary_point_v_given_n(r, alpha, n, N=N)
    
    vv = np.zeros(N)
    vv[0] = v_star
    vv[1] = v_star + 0.00001
    vv[0] = best_response_i(r=r, vv=vv, alpha=alpha, i=0, v_max=v_max)
    
    for _ in range(50):
        i = np.random.randint(0, 2)
        vv[i] = best_response_i(r=r, vv=vv, alpha=alpha, i=i)


def simulate(r, alpha, N, v_max, timesteps=1000, group_type="selfish"):

    vv = np.random.uniform(0, v_max, N)

    for _ in range(timesteps):
        last_vv = np.copy(vv)
        # Rnadomyl genearte an order for all players
        # to update their vigilance
        ii = np.random.permutation(N)
        
        # Update each player's vigilance based on the best response
        for i in ii:
            if group_type == "selfish":
                vv[i] = best_response_selfish(r=r, vv=vv, alpha=alpha, i=i, v_max=v_max)
            elif group_type == "coop":
                vv[i] = best_response_coop(r=r, vv=vv, alpha=alpha, i=i, v_max=v_max)

        # Check if the vigilance values have converged
        if np.allclose(vv, last_vv, atol=1e-5):
            break

    # After all timesteps, we have the final vigilance values

    # Count how many are non-zero (considering the tolerance of 1e-5)
    n_watchers = np.sum(vv > 1e-4)

    f_mean = np.mean([get_f_i(alpha, r, vv, i) for i in range(N)])

    return n_watchers, f_mean

def run_sims():

    alpha_mod_max = 2
    resolution = 500
    r_min = 0.01
    r_max = 100
    timesteps = 10000

    alphas = np.linspace(-alpha_mod_max,alpha_mod_max,resolution)
    rs = np.logspace(-2, 2, resolution)

    N = 16
    v_max = 3

    seed = 1
    np.random.seed(seed)

    for i, r in enumerate(rs):
        for j, alpha in enumerate(alphas):
            print(r, alpha)
            # Call your classification function
            n_watchers, f_mean = simulate(r, alpha, N, v_max, timesteps=timesteps, group_type="selfish")

            csv_row = ["selfish", N, v_max, r, alpha, r_max, alpha_mod_max, resolution, n_watchers, f_mean]
            append_to_csv(csv_row, "stochastic_sim_results_large.csv")

            n_watchers, f_mean = simulate(r, alpha, N, v_max, timesteps=timesteps, group_type="coop")

            csv_row = ["coop", N, v_max, r, alpha, r_max, alpha_mod_max, resolution, n_watchers, f_mean]
            append_to_csv(csv_row, "stochastic_sim_results_large.csv")


if __name__ == "__main__":
    run_sims()
    #simulate(r=0.01, alpha=0.5, N=16, v_max=3, timesteps=1000)