import pandas as pd
import numpy as np
from get_data import get_ticker_data, get_labeled_data, get_lag_ret_data, get_calc_stats

def markowitz_frontier(mu, Sigma, num_points=50):
    """
    Generates points on the unconstrained Markowitz frontier
    for returns between min(mu) and max(mu).
    """
    n = len(mu)
    ones = np.ones(n)
    Sigma_inv = np.linalg.inv(Sigma)

    # Compute A, B, C, D
    A = mu.T @ Sigma_inv @ mu
    B = mu.T @ Sigma_inv @ ones
    C = ones.T @ Sigma_inv @ ones
    D = A*C - B**2

    def weights_for_return(r_target):
        term1 = (A * Sigma_inv @ mu - B * Sigma_inv @ ones) / D
        term2 = (C * Sigma_inv @ ones - A * Sigma_inv @ mu) / D
        return term1 + r_target * term2

    # Create the range of returns to evaluate
    r_min, r_max = min(mu)*0.9, max(mu)*1.1
    target_returns = np.linspace(r_min, r_max, num_points)

    frontier = []
    for r_target in target_returns:
        w = weights_for_return(r_target)
        portfolio_return = w @ mu
        portfolio_risk = np.sqrt(w.T @ Sigma @ w)
        frontier.append((portfolio_risk, portfolio_return, w))
    
    return frontier

def main():
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'AAON', 'CFR']
    data = get_ticker_data(tickers)
    data = get_lag_ret_data(data, tickers)
    means_s, stds_s, cov_df, corr_df = get_calc_stats(data)
    
    # Convert to numpy arrays
    mu = means_s.values           
    Sigma = cov_df.values

    frontier = markowitz_frontier(mu, Sigma)
    for idx, (risk, ret, w) in enumerate(frontier):
        print(f"Frontier Pt {idx}: Risk={risk:.4f}, Return={ret:.4f}, Weights={w}")
    
    # plot risk vs return
    import matplotlib.pyplot as plt
    risks = [pt[0] for pt in frontier]
    returns = [pt[1] for pt in frontier]
    plt.plot(risks, returns, marker='o')
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Return")
    plt.title("Unconstrained Markowitz Frontier")
    plt.show()

if __name__ == "__main__":
    main()