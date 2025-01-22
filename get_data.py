import yfinance as yf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

def get_ticker_data(tickers=['AAPL', 'MSFT'], period='1y', interval='1d'):
    """
    Download price data from Yahoo Finance for given tickers and timeframe.

    :param tickers: list or tuple of ticker symbols, e.g. ['AAPL', 'MSFT']
    :param period: data period, e.g. '1y'
    :param interval: data interval, e.g. '1d'
    :param kwargs: additional keyword arguments for yf.download
    :return: Pandas DataFrame with downloaded data
    """    
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', auto_adjust=True)
    return data

def get_lag_ret_data(df, tickers, num_lags=3):
    """
    Adds percent-change columns (Percent_change) and lagged values for each ticker.
    
    :param data: Pandas DataFrame (e.g., from yfinance)
    :param tickers: list of ticker symbols
    :param num_lags: number of lagged days to create
    :return: Modified DataFrame with new columns
    """    
    # Calculate daily Percent Change in Close Price for each ticker
    for ticker in tickers:
        df[(ticker, 'Percent_change')] = [0.0 if i == 0 
            else ((df[ticker]['Close'].iloc[i] - df[ticker]['Close'].iloc[i-1]) / df[ticker]['Close'].iloc[i-1] * 100)
            for i in range(len(df))]

    # Drop NaN values 
    df.dropna(inplace=True)

    # Create lagged  data
    for ticker in tickers:
        for lag in range(1, num_lags + 1):
            df[(f'Lagged_{ticker}', f'{lag}_d')] = df[(ticker, 'Percent_change')].shift(lag) 
    
    # Drop NaN values 
    df.dropna(inplace=True)   
    return df

def get_calc_stats(data, col_name='Percent_change'):
    """
    Compute sample means, sample standard deviations, covariance matrix, 
    and correlation matrix for the given data.
    
    :param data: a Pandas DataFrame with shape (n_rows, n_assets)
    :param col_name: name of the top-level column if multi-index
    :return: (means_s, stds_s, cov_df, corr_df) as DataFrames and Series
    """

    df = data.xs(col_name, axis=1, level=1)
    columns = df.columns
    n_cols = len(columns)

    # Calculate means
    means = {}
    for col in columns:
        total = 0.0
        for val in df[col]:
            total += val
        
        means[col] = total / len(df[col])
    
    # Calculate Std
    sum_sqr_diff = {col : 0.0 for col in columns}
    for col in columns:
        for val in df[col]:
            sum_sqr_diff[col] += (val - means[col]) ** 2
    
    stds = {}
    for col in columns:
        stds[col] = math.sqrt(sum_sqr_diff[col] / (len(df[col]) - 1))
    
    # Calculate Covarince
    cov_matrix = [[0.0] * n_cols for _ in range(n_cols)]
    for i in range(n_cols):
        i_col = columns[i]
        for j in range(i, n_cols):
            j_col = columns[j]
            cov_sum = 0.0
            for k in range(len(df[i_col])):
                cov_sum += (df[i_col][k] - means[i_col]) * (df[j_col][k] - means[j_col])
            
            cov_val = cov_sum / (len(df[i_col]) - 1)
            cov_matrix[i][j] = cov_val
            cov_matrix[j][i] = cov_val

    # Calculate Correlation
    corr_matrix = [[0.0] * n_cols for _ in range(n_cols)]
    for i in range(n_cols):
        i_col = columns[i]
        for j in range(i, n_cols):
            j_col = columns[j]   
            denom = stds[i_col] * stds[j_col] 
            if (denom == 0):
                corr_matrix[i][j] = float('nan')
                corr_matrix[j][i] = float('nan')
            else:
                corr_matrix[i][j] = cov_matrix[i][j] / denom
                corr_matrix[j][i] = cov_matrix[i][j] / denom

    
    print("Covariance Matrix")
    print(cov_matrix)
    print("Correlation Matrix")
    print(corr_matrix)

    cov_df = pd.DataFrame(cov_matrix, index=columns, columns=columns)
    corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
    means_s = pd.Series(means)
    stds_s = pd.Series(stds)

    return means_s, stds_s, cov_df, corr_df

def get_labeled_data(data):
    lagged_features = [col for col in data.columns if 'Lagged' in col[0]]
    X = data[lagged_features].values
    y = data['AAPL']['Percent_change'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Output shapes of the splits
    print("Shapes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test    