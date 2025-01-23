import numpy as np

class LinearRegression:
    """
    A simple Linear Regression model, can be fit using OLS or gradient descent.
    With optional Ridge (L2) regularization and optional momentum term.

    Parameters:
    -----------
    reg_lambda : float, default=0.0
        Regularization strength for ridge penalty (L2).
    momentum : float, default=0.0
        Momentum factor for gradient descent.
    step_size : float, default=0.01
        Learning rate for gradient descent.
    num_epochs : int, default=1000
        Number of iterations for gradient descent.
    """
    def __init__(self, reg_lambda=0.0, momentum=0.0, step_size=0.01, num_epochs=1000):
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.step_size = step_size
        self.num_epochs = num_epochs
        
        # Model parameters (learned weights) will be stored here
        self.beta = None
        # For momentum-based gradient descent, velocity is needed
        self.velocity = None

    def fit(self, x, y, method="ols"):
        """
        Fit the linear regression model using the specified method.

        Parameters:
        -----------
        x : np.ndarray, shape (n_samples, n_features)
            The training input data, with the last column typically being 1s for intercept.
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
            The training target values.
        method : {"ols", "gradient_descent"}, default="ols"
            The training method to use.
        """
        if method == "ols":
            self.fit_ols(x, y)
        elif method == "gradient_descent":
            self.fit_grad_descent(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

    def cost(self, x, y):
        """
        Compute the MSE cost (no regularization term added here by default).

        Parameters:
        -----------
        x : np.ndarray
            The input data.
        y : np.ndarray
            The target values.
        
        Returns:
        --------
        float
            The MSE cost.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        residuals = y - (x @ self.beta)
        mse = (residuals ** 2).mean() / 2  # 1/(2n)*sum(...), factoring out "2" is optional
        return mse

    def fit_ols(self, x, y):
        """
        Solve for beta analytically, including ridge regularization but
        excluding the intercept from penalty.

        Parameters:
        -----------
        x : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_features = x.shape[1]
        
        # Create identity matrix. We'll zero out the intercept row/col.
        # Assuming the last column of x is the intercept, so that is index n_features-1.
        I = np.identity(n_features)
        I[-1, -1] = 0.0  # No penalty on the intercept parameter

        # (X^T X + lambda * I)^(-1) (X^T y)
        self.beta = np.linalg.inv(x.T @ x + self.reg_lambda * I) @ (x.T @ y)

    def grad(self, x, y):
        """
        Compute gradient of MSE + ridge penalty w.r.t. beta,
        EXCLUDING the intercept from the penalty.

        Parameters:
        -----------
        x : np.ndarray
            The input data (n_samples, n_features).
        y : np.ndarray
            The target values (n_samples,) or (n_samples, 1).

        Returns:
        --------
        np.ndarray
            The gradient vector (same shape as beta).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = len(y)
        
        # Gradient of the MSE part
        base_grad = (x.T @ (x @ self.beta - y)) / n_samples
        
        # Ridge penalty gradient = lambda * beta, but skip intercept (last component).
        reg_grad = self.reg_lambda * np.copy(self.beta)
        reg_grad[-1, 0] = 0.0  # do not penalize the intercept

        return base_grad + reg_grad

    def fit_grad_descent(self, x, y):
        """
        Fit the model using gradient descent with optional momentum.

        Parameters:
        -----------
        x : np.ndarray
            The input data (n_samples, n_features).
        y : np.ndarray
            The target values (n_samples,) or (n_samples, 1).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_features = x.shape[1]
        # Initialize beta randomly
        self.beta = np.random.normal(loc=0, scale=1, size=(n_features, 1))

        # Initialize velocity for momentum
        self.velocity = np.zeros_like(self.beta)

        cost_hist = []

        for i in range(self.num_epochs):
            loss = self.cost(x, y)
            cost_hist.append(loss)

            # Compute gradient
            g = self.grad(x, y)

            # Update velocity and beta
            self.velocity = self.momentum * self.velocity + self.step_size * g
            self.beta = self.beta - self.velocity

            # Print progress every 100 epochs
            if i % 100 == 0:
                print(f"Epoch {i}, Loss = {loss:.5f}")

        return cost_hist

    def predict(self, x):
        """
        Predict values using the trained model.

        Parameters:
        -----------
        x : np.ndarray, shape (n_samples, n_features)

        Returns:
        --------
        np.ndarray
            The predicted values (n_samples, 1) by default.
        """
        if self.beta is None:
            raise ValueError("Model not fit yet")

        return x @ self.beta