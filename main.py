import numpy as np
import matplotlib.pyplot as plt

from syn_data import generate_dataset  # Assuming your updated file is named syn_data.py
from lin_reg import LinearRegression   # Assuming your linear regression code is in lin_reg.py


def main():
    N = 300
    X, y = generate_dataset(N)

    # 2. Define Combinations of Hyperparameters
    #    --------------------------------------
    methods = ["ols", "gradient_descent"]
    reg_lambdas = np.linspace(0.0, 0.8, 10)
    momenta = np.linspace(0.0, 0.8, 10)

    # We'll store results in a dictionary:
    #   results[(method, reg_lambda, momentum)] = {
    #       "final_mse": ...,
    #       "cost_history": ...
    #   }
    # Note: OLS doesn't have an iterative cost history, so we may store an empty list or None.
    results = {}

    # Loop Over Configurations and Fit
    for method in methods:
        for reg_lambda in reg_lambdas:
            for momentum in momenta:
                # Skip momentum for OLS, as it has no meaning
                if method == "ols" and momentum != 0.0:
                    continue

                # Create the model
                model = LinearRegression(
                    reg_lambda=reg_lambda,
                    momentum=momentum,
                    step_size=0.01,   # could tune
                    num_epochs=300    # could tune
                )

                # Fit the model
                if method == "ols":
                    model.fit(X, y, method="ols")
                    cost_history = None  # Not iterative, so no history
                else:
                    cost_history = model.fit_grad_descent(X, y)

                # Evaluate final MSE on training set
                predictions = model.predict(X)
                # Convert predictions to 1D if needed
                if predictions.ndim == 2:
                    predictions = predictions.flatten()

                # Mean Squared Error
                mse = np.mean((y - predictions) ** 2)

                # Store in results dictionary
                key = (method, reg_lambda, momentum)
                results[key] = {
                    "final_mse": mse,
                    "cost_history": cost_history
                }
                print(f"Finished {key}: MSE = {mse:.5f}")

    # --------------------------------------------------
    #  (A) Separate OLS results from GD results
    # --------------------------------------------------
    ols_results = {}
    gd_results = {}
    
    for (method, reg_lmb, mom) in results.keys():
        if method == "ols":
            ols_results[reg_lmb] = results[(method, reg_lmb, mom)]["final_mse"]
        else:  # gradient_descent
            gd_results[(reg_lmb, mom)] = results[(method, reg_lmb, mom)]

    # --------------------------------------------------
    #  (B) Plot OLS MSE vs. reg_lambda
    #      (momentum is always 0 here)
    # --------------------------------------------------
    if ols_results:
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

    # --------------------------------------------------
    #  (C) Heatmap of Gradient Descent MSE
    #      X-axis: momentum, Y-axis: reg_lambda
    # --------------------------------------------------
    # 1. Collect unique reg_lambdas & momenta
    gd_reg_lmbs = sorted(list({k[0] for k in gd_results.keys()}))
    gd_moms = sorted(list({k[1] for k in gd_results.keys()}))

    # 2. Build a 2D array of MSE for heatmap
    mse_matrix = np.zeros((len(gd_reg_lmbs), len(gd_moms)))
    
    for i, r_lmb in enumerate(gd_reg_lmbs):
        for j, mom in enumerate(gd_moms):
            mse_matrix[i, j] = gd_results[(r_lmb, mom)]["final_mse"]

    plt.figure(figsize=(6, 5))
    # imshow or pcolormesh can be used for heatmaps
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

    # --------------------------------------------------
    #  (D) Pick a small subset of GD combos for cost-curve plotting
    #      e.g., the best 3 combos and worst 3 combos by final MSE
    # --------------------------------------------------
    # 1. Extract (reg_lmb, mom, final_mse, cost_history) in a list
    all_gd_data = []
    for (r_lmb, mom), val in gd_results.items():
        final_mse = val["final_mse"]
        cost_hist = val["cost_history"]
        all_gd_data.append((r_lmb, mom, final_mse, cost_hist))
    
    # 2. Sort by final_mse to find best/worst
    all_gd_data.sort(key=lambda x: x[2])  # sort by final_mse ascending
    
    # best 3
    best_3 = all_gd_data[:3]
    # worst 3
    worst_3 = all_gd_data[-3:]
    chosen_runs = best_3 + worst_3

    # 3. Plot them
    plt.figure(figsize=(7, 5))
    for (r_lmb, mom, mse_val, cost_hist) in chosen_runs:
        label = f"λ={r_lmb:.2f}, mom={mom:.2f}, MSE={mse_val:.3f}"
        plt.plot(cost_hist, label=label)

    plt.title("Cost Curves for Selected GD Runs (Best & Worst)")
    plt.xlabel("Epoch")
    plt.ylabel("Cost (MSE portion)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()