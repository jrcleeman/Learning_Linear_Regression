import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(n_samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of size n_samples with custom recurrence rules:

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed      : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 101)
        The generated feature matrix, with 101 entries per sample.
    y : ndarray of shape (n_samples,)
        The generated target values.
    """
    if (seed is not None):
        np.random.seed(seed)

    # Initialize the entire feature matrix and label vector
    X = np.zeros((n_samples, 101))
    y = np.zeros(n_samples)

    # Precompute powers of -0.88 for the label calculation
    powers = np.power(-0.88, np.arange(1, 51))

    # Intercept or bias-like term
    X[:, 0] = 1

    # First 50 entries (indices 1..50) ~ N(0, 1)
    X[:, 1:51] = np.random.normal(loc=0, scale=1, size=(n_samples, 50))

    # i = 51..60
    noise_51_60 = np.random.normal(loc=0, scale=np.sqrt(0.1), size=(n_samples, 10))
    # X[:,1] is shape (n_samples,) so reshape it to (n_samples,1) for broadcasting.
    X[:, 51:61] = X[:, 1].reshape(-1, 1) + 0.5 * X[:, 1:11] + noise_51_60

    # i = 61..70
    noise_61_70 = np.random.normal(0, np.sqrt(0.1), (n_samples, 10))
    X[:, 61:71] = X[:, 1:11] - X[:, 11:21] + X[:, 21:31] + noise_61_70

    # i = 71..80
    noise_71_80 = np.random.normal(0, np.sqrt(0.1), (n_samples, 10))
    j = np.arange(1, 11)  # j from 1 to 10
    col_idx = 6 * j     # columns [6, 12, ..., 60]
    X[:, 71:81] = X[:, col_idx] + 3 * X[:, 1:11] + noise_71_80

    # i = 81..90
    X[:, 81:91] = 5 - X[:, 71:81]

    # i = 91..100
    j = np.arange(1, 11)
    source_idx1 = 50 + 4 * j  # e.g., for j=1: column 54, j=2: column 58, etc.
    source_idx2 = 50 + 3 * j  # e.g., for j=1: column 53, j=2: column 56, etc.
    noise_91_100 = np.random.normal(0, np.sqrt(0.1), (n_samples, 10))
    X[:, 91:101] = 0.5 * X[:, source_idx1] + 0.5 * X[:, source_idx2] + noise_91_100

    # Compute the label y[n]
    X_label = X[:, 2:101:2]  # shape (n_samples, 50)
    noise_y = np.random.normal(0, np.sqrt(0.01), n_samples)
    # Multiply elementwise and sum across columns.
    y = np.sum(X_label * powers, axis=1) + noise_y

    return X, y
