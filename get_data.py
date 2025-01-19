import yfinance as yf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math

def get_ticker_data(tickers=['AAPL', 'MSFT'], period='1y', interval='1d'):
    """
    Download price data from Yahoo Finance for given tickers and timeframe.

    :param tickers: list or tuple of ticker symbols, e.g. ['AAPL', 'MSFT']
    :param period: data period, e.g. '1y'
    :param interval: data interval, e.g. '1d'
    :param kwargs: additional keyword arguments for yf.download
    :return: Pandas DataFrame with downloaded data
    """    
    data = yf.download(tickers, period, interval)
    return data

def get_lag_ret_data(data, tickers, num_lags=3):
    """
    Adds percent-change columns (Percent_change) and lagged values for each ticker.
    
    :param data: Pandas DataFrame (e.g., from yfinance)
    :param tickers: list of ticker symbols
    :param num_lags: number of lagged days to create
    :return: Modified DataFrame with new columns
    """    
    df = data.copy()

    # Calculate daily Percent Change in Close Price for each ticker
    for ticker in tickers:
        df[('Percent_change', ticker)] = df['Close'][ticker].pct_change() * 100

    # Drop NaN values 
    df.dropna(inplace=True)

    # Create lagged  data
    for ticker in tickers:
        for lag in range(1, num_lags + 1):
            df[(f'Lagged_{ticker}', f'{lag}_d')] = df[('Percent_change', ticker)].shift(lag) 
    
    # Drop NaN values 
    df.dropna(inplace=True)   
    return df

# Calculate Averages of column
def calc_avg_ret(data, tickers, col_name='Percent_change'):
    """
    Calculate the average return for each ticker under col_name.

    :param data: Pandas DataFrame
    :param tickers: list of tickers
    :param col_name: top-level column name (e.g. 'Percent_change')
    :return: dict of {ticker: average return}
    """    
    avg_returns = {}
    for ticker in tickers:
        total = 0 
        series = data[col_name][ticker]
        for val in series:
            total += val
        avg_returns[ticker] = total / len()

    print(f"Averages of {col_name}")
    for ticker, avg_ret in avg_returns.items():
        print(f"{ticker}: {avg_ret:.2f}%")
    
    return avg_returns

# Calculate Std of Column
def calc_std(data, tickers, col_name='Percent_change'):
    std_dic = {}

    for ticker in tickers:
        total = 0
        for val in data[col_name][ticker]:
            total += val
        
        mean = total / len(data[col_name][ticker])
        total = 0
        for val in data[col_name][ticker]:
            total += (val - mean) ** 2
        
        var = total / (len(data[col_name][ticker]) - 1)
        std = math.sqrt(var)
        std_dic[ticker] = std

    print(f"Std of {col_name}")
    for ticker, std in std_dic.items():
        print(f"{ticker}: {std:.2f}")

    return std_dic

# Calculate Covariance Matrix Between Assets
def calc_cov(data, col_name='Percent_change'):
    df = data[col_name]
    columns = df.columns
    n_cols = len(columns)
    
    # Initialize covariance matrix
    cov_matrix = [[0.0 for _ in range(n_cols)] for _ in range(n_cols)]

    # Compute means
    means = {}
    for col in columns:
        total = 0
        for val in df[col]:
            total += val
        means[col] = total / len(df[col])

    # Compute sample covariance
    for i in range(n_cols):
        i_col = columns[i]
        n = len(df[i_col])  # same length across columns assumed
        for j in range(i, n_cols):
            j_col = columns[j]
            cov_sum = 0
            for k in range(n):
                cov_sum += (df[i_col][k] - means[i_col]) * (df[j_col][k] - means[j_col])

            cov_val = cov_sum / (n - 1)
            cov_matrix[i][j] = cov_val
            cov_matrix[j][i] = cov_val  # symmetric

    print("Covariance Matrix:")
    print(cov_matrix)
    return cov_matrix

# Calculate Correlation Matrix 
def calc_corr(data, cov_matrix, std_dict, col_name='Percent_change'):
    df = data[col_name]
    columns = df.columns
    n_cols = len(columns)
    corr_matrix = [[0.0 for _ in range(n_cols)] for _ in range(n_cols)]

    for i in range(n_cols):
        for j in range(i, n_cols):
            i_col = columns[i]
            j_col = columns[j]
            denom = (std_dict[i_col] * std_dict[j_col])
            if (denom == 0):
                corr_val = 0
            else:
                corr_val = cov_matrix[i][j] / denom
            corr_matrix[i][j] = corr_val
            corr_matrix[j][i] = corr_val
    
    print("Correlation Matrix:")
    print(corr_matrix)
    return corr_matrix



    # lagged_features = [col for col in data.columns if 'Lagged' in col[0]]
    # X = data[lagged_features].values
    # y = data['Percent_change']['AAPL'].values.reshape(-1, 1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # # Output shapes of the splits
    # print("Shapes:")
    # print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    # return X_train, X_test, y_train, y_test