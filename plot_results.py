import matplotlib.pyplot as plt
import numpy as np

def plot_ols_results(ols_results: dict) -> None:
    """
    Plot the OLS MSE results.
    """
    # Sort by reg_lambda
    sorted_reg_lmb = sorted(ols_results.keys())
    sorted_mse = [ols_results[rl] for rl in sorted_reg_lmb]

    plt.figure(figsize=(5, 4))
    plt.plot(sorted_reg_lmb, sorted_mse, marker="o", label="OLS MSE")
    plt.title("OLS MSE vs. reg_lambda")
    plt.xlabel("reg_lambda")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show() 

def plot_gd_results(gd_results: dict) -> None:
    """
    Plot a heatmap of the Gradient Descent MSE results.
    """
    # Collect unique reg_lambdas & momenta
    gd_reg_lmbs = sorted(list({k[0] for k in gd_results.keys()}))
    gd_moms = sorted(list({k[1] for k in gd_results.keys()}))

    # Build a 2D array of MSE for heatmap
    mse_matrix = np.zeros((len(gd_reg_lmbs), len(gd_moms)))
    
    for i, r_lmb in enumerate(gd_reg_lmbs):
        for j, mom in enumerate(gd_moms):
            mse_matrix[i, j] = gd_results[(r_lmb, mom)]["final_mse"]

    plt.figure(figsize=(6, 5))
    c = plt.imshow(mse_matrix, 
                   origin='lower',  # so the [0,0] is bottom-left
                   aspect='auto', 
                   extent=(min(gd_moms), max(gd_moms), min(gd_reg_lmbs), max(gd_reg_lmbs)))
    plt.colorbar(c, label="MSE")
    plt.xlabel("Momentum")
    plt.ylabel("reg_lambda")
    plt.title("MSE Heatmap for Gradient Descent")
    plt.tight_layout()
    plt.show()        