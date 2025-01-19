import matplotlib.pyplot as plt
import numpy as np

def mv_simulate_data(num_pnts, num_feat, plot=False):
    """
    Generate synthetic data for a multi-variable linear model,
    plus a random intercept column and random noise. Also plots a 3D scatter.
    
    Args:
        num_pnts (int): Number of data points.
        num_feat (int): Number of features (excluding intercept).
    Returns:
        x (ndarray): Feature matrix of shape (num_pnts, num_feat+1).
        y (ndarray): Labels of shape (num_pnts, 1).
        beta (ndarray): True beta vector used for data generation, shape (num_feat+1, 1).
    """    
    # Features
    x = np.random.uniform(-1, 1, (num_pnts, num_feat)) # n x p
    x = np.hstack((x, np.ones((num_pnts,1)))) # n x (p + 1) # Adding column of ones for intercept

    # Beta
    beta = np.random.uniform(-1, 1, (num_feat + 1, 1)) # (p + 1) x 1
    noise = np.random.normal(0, 0.1, (num_pnts, 1)) # n x 1
    
    # Label
    y = x @ beta + noise
    
    if plot and (num_feat == 2):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], y, color='blue', label="Data Points")
        ax.set_xlabel("Feature 1 (x1)")
        ax.set_ylabel("Feature 2 (x2)")
        ax.set_zlabel("Labels (z)")
        ax.set_title("3D Scatter Plot of Features and Labels")
        plt.show()

    return x, y, beta