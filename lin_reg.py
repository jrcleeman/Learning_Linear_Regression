import numpy as np
import logging

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
        Maximum number of epochs (iterations) for gradient descent.
    """
    def __init__(self, reg_lambda: float = 0.0, momentum: float = 0.0, 
                 step_size: float = 0.01, num_epochs: int = 1000) -> None:
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.step_size = step_size
        self.num_epochs = num_epochs
        
        # Model parameters (learned weights) will be stored here
        self.beta = None
        # For momentum-based gradient descent, velocity is needed
        self.velocity = None

    def cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the regularized mean squared error (MSE + ridge penalty, excluding the intercept).

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
        if self.beta is None:
            raise ValueError("Model parameters are not set. Call fit_ols or fit_grad_descent first.")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        residuals = y - (x @ self.beta)
        mse = (residuals ** 2).mean() / 2.0
        reg_term = self.reg_lambda * np.sum(self.beta[:-1]**2) / 2.0
        return mse + reg_term

    def fit_ols(self, x: np.ndarray, y: np.ndarray) -> None:
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
        self.beta = np.linalg.solve(x.T @ x + self.reg_lambda * I, x.T @ y)
        
        logging.debug("OLS fit completed.")

    def grad(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    def fit_grad_descent(self, x: np.ndarray, y: np.ndarray, 
                         early_stopping: bool = True, tol: float = 1e-5, 
                         patience: int = 5, grad_tol: float = 1e-4) -> list[float]:
        """
        Fit the model using gradient descent with optional momentum and optional early stopping.

        Parameters:
        -----------
        x : np.ndarray
            The input data (n_samples, n_features).
        y : np.ndarray
            The target values (n_samples,) or (n_samples, 1).
        early_stopping : bool, default=False
            If True, stop training early if loss does not improve sufficiently.
        tol : float, default=1e-5
            Minimum improvement in loss to reset patience.
        patience : int, default=5
            Number of epochs with no sufficient improvement allowed before stopping.

        Returns:
        --------
        list of float
            The history of cost values for each epoch (or until early stopping).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_features = x.shape[1]
        # Initialize beta randomly
        self.beta = np.random.normal(loc=0, scale=1, size=(n_features, 1))

        # Initialize velocity for momentum
        self.velocity = np.zeros_like(self.beta)

        cost_hist = []

        # Variables for early stopping
        best_loss = float('inf')
        best_beta = np.copy(self.beta)
        wait = 0  # how many epochs since the last sufficient improvement

        for i in range(self.num_epochs):
            loss = self.cost(x, y)
            cost_hist.append(loss)

            # Check if improvement is "significant" (larger than tol)
            if early_stopping:
                if loss < best_loss - tol:
                    best_loss = loss
                    best_beta = np.copy(self.beta)
                    wait = 0  # reset patience
                else:
                    wait += 1
                    if wait >= patience:
                        # Restore best beta and break
                        logging.debug(f"Early stopping at epoch {i}, best loss = {best_loss:.6f}")
                        self.beta = best_beta
                        break

            # Compute gradient
            g = self.grad(x, y)
            
            # Convergence check based on gradient norm
            if (np.linalg.norm(g) < grad_tol):
                logging.debug(f"Convergence reached at epoch {i}: gradient norm {np.linalg.norm(g):.6e} < {grad_tol}")
                break

            # Update velocity and beta
            self.velocity = self.momentum * self.velocity + self.step_size * g
            self.beta = self.beta - self.velocity

            # Print progress every 100 epochs
            if i % 100 == 0:
                logging.debug(f"Epoch {i}, Loss = {loss:.5f}")

        return cost_hist

    def predict(self, x: np.ndarray) -> np.ndarray:
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
        if x.shape[1] != self.beta.shape[0]:
            raise ValueError("Dimension mismatch between input features and learned beta.")        

        return x @ self.beta
