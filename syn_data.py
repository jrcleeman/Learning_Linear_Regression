import numpy as np
import matplotlib.pyplot as plt

def mv_simulate_data(num_pnts, num_feat, plot=False):
    """
    Generate synthetic data for a multi-variable linear model,
    including:
        - Random features in the range [-1, 1].
        - A column of ones for the intercept.
        - A randomly generated true beta (including intercept).
        - Gaussian noise added to the labels.
    Optionally plot a 3D scatter if num_feat == 2 and plot=True.

    Parameters
    ----------
    num_pnts : int
        Number of data points to generate.
    num_feat : int
        Number of features (excluding intercept).
    plot : bool, default=False
        If True and num_feat == 2, creates a 3D scatter plot of data.

    Returns
    -------
    x : ndarray of shape (num_pnts, num_feat + 1)
        The generated feature matrix (last column is intercept = 1).
    y : ndarray of shape (num_pnts, 1)
        The target values (noisy linear combination of x).
    beta : ndarray of shape (num_feat + 1, 1)
        The "true" beta coefficients used to generate y
        (includes the intercept).
    """
    # Random feature matrix: uniform in [-1, 1]
    x = np.random.uniform(-1, 1, (num_pnts, num_feat))
    # Append a column of ones for the intercept
    x = np.hstack((x, np.ones((num_pnts, 1))))

    # Generate a random "true" beta (including intercept)
    beta = np.random.uniform(-1, 1, (num_feat + 1, 1))

    # Add Gaussian noise
    noise = np.random.normal(loc=0, scale=0.1, size=(num_pnts, 1))

    # Calculate labels
    y = x @ beta + noise

    # Optional 3D plotting if we have exactly 2 features (plus intercept)
    if plot and (num_feat == 2):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # x[:,0] and x[:,1] are the two features; y is the target
        ax.scatter(x[:, 0], x[:, 1], y, color='blue', label="Data Points")
        ax.set_xlabel("Feature 1 (x1)")
        ax.set_ylabel("Feature 2 (x2)")
        ax.set_zlabel("Labels (z)")
        ax.set_title("3D Scatter Plot of Features and Labels")
        plt.show()

    return x, y, beta


def generate_dataset(n_samples):
    """
    Generate a dataset of size n_samples with custom rules:
    - x[0] is set to 1 (like an intercept).
    - x[1:51] are standard normal random variables.
    - Subsequent indices (51 to 100) are defined by specific
      recurrence relations plus Gaussian noise.
    - The label y (1D array) is computed via dot product
      with powers of -0.88 and some additional noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    X : ndarray of shape (n_samples, 101)
        The generated feature matrix, with 101 entries per sample.
    y : ndarray of shape (n_samples,)
        The generated target values.
    """
    # Initialize the entire feature matrix and label vector
    X = np.zeros((n_samples, 101))
    y = np.zeros(n_samples)

    # Precompute powers of -0.88 for the label calculation
    # We'll use this to compute y below.
    powers = np.power(-0.88, np.arange(1, 51))

    for n in range(n_samples):
        # Each row 'x' has 101 features
        x = np.zeros(101)

        # Intercept or bias-like term
        x[0] = 1

        # First 50 entries (indices 1..50) ~ N(0, 1)
        x[1:51] = np.random.normal(loc=0, scale=1, size=50)

        # i = 51..60
        for i in range(51, 61):
            noise_term = np.random.normal(loc=0, scale=np.sqrt(0.1))
            x[i] = x[1] + 0.5 * x[i - 50] + noise_term

        # i = 61..70
        for i in range(61, 71):
            noise_term = np.random.normal(loc=0, scale=np.sqrt(0.1))
            x[i] = x[i - 60] - x[i - 50] + x[i - 40] + noise_term

        # i = 71..80
        for i in range(71, 81):
            noise_term = np.random.normal(loc=0, scale=np.sqrt(0.1))
            x[i] = x[6 * (i - 70)] + 3 * x[i - 70] + noise_term

        # i = 81..90
        for i in range(81, 91):
            # No noise here, purely deterministic
            x[i] = 5 - x[i - 10]

        # i = 91..100
        for i in range(91, 101):
            noise_term = np.random.normal(loc=0, scale=np.sqrt(0.1))
            # Notice how we're indexing x at positions that shift by 3 and 4
            x[i] = 0.5 * x[50 + (i - 90) * 4] + 0.5 * x[50 + (i - 90) * 3] + noise_term

        # Compute the label y[n]
        #   - Dot product between 'powers' (length 50) and every other element in x[2..100]
        #   - x[2:101:2] picks x2, x4, x6, ... x100
        #   - plus small Gaussian noise
        y_noise = np.random.normal(loc=0, scale=np.sqrt(0.01))
        label_value = np.dot(powers, x[2:101:2]) + y_noise

        # Store row in X and label in y
        X[n] = x
        y[n] = label_value

    return X, y
