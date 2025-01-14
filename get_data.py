import yfinance as yf
from sklearn.model_selection import train_test_split
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

def lagged_returns(num_lags=3):
    tickers = ['AAPL', 'MSFT']
    data = yf.download(tickers, period='1y', interval='1d')

    # Calculate daily Percent Change in Close Price for each ticker
    for ticker in tickers:
        data[('Percent_change', ticker)] = data['Close'][ticker].pct_change() * 100

    # Drop NaN values 
    data.dropna(inplace=True)

    # Calculate Average Daily Returns
    avg_daily_returns = {
        ticker : data['Percent_change'][ticker].mean()
        for ticker in tickers
    }

    # print("Average Daily Returns")
    # for ticker, avg_ret in avg_daily_returns.items():
    #     print(f"{ticker}: {avg_ret:.2f}%")

    # Create labeled data
    
    for ticker in tickers:
        for lag in range(1, num_lags + 1):
            data[(f'Lagged_{ticker}', f'{lag}_d')] = data[('Percent_change', ticker)].shift(lag) 
    
    # Drop NaN values 
    data.dropna(inplace=True)

    lagged_features = [col for col in data.columns if 'Lagged' in col[0]]
    X = data[lagged_features].values
    y = data['Percent_change']['AAPL'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # # Output shapes of the splits
    # print("Shapes:")
    # print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    # print(X_train[:5])
    # print(y_train[:5])
    # print(X_test[:5])
    # print(y_test[:5])


    return X_train, X_test, y_train, y_test

# lagged_returns()

# # Calculate Std of Daily Returns
# std_daily_returns = {
#     ticker : data['Percent_change'][ticker].std()
#     for ticker in tickers
# }

# print("Std of Daily Returns")
# for ticker, std in std_daily_returns.items():
#     print(f"{ticker}: {std:.2f}")

# # Correlation Matrix Between Assets
# corr_matrix = data['Percent_change'].corr()
# print("Correlation Matrix:")
# print(corr_matrix)

# # Covariance Matrix from Daily Returns
# cov_matrix = data['Percent_change'].cov()
# print("Covariance Matrix:")
# print(cov_matrix)