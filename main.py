import logging
from tqdm import tqdm
import numpy as np

from syn_data import generate_dataset  
from lin_reg import LinearRegression   
from plot_results import plot_ols_results, plot_gd_results

# Configure logging: output to both a file and console
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)

np.random.seed(42)

def run_ols_experiments(X: np.ndarray, y: np.ndarray, reg_lambdas: np.ndarray) -> dict:
    """
    Run OLS experiments over a range of regularization parameters.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        reg_lambdas (np.ndarray): Array of regularization parameter values.

    Returns:
        dict: Mapping of reg_lambda to the computed MSE.
    """
    ols_results = {}
    for reg_lambda in tqdm(reg_lambdas, desc="OLS Hyperparameters"):
        logging.debug(f"Running OLS experiment for reg_lambda={reg_lambda:.3f}")
        model = LinearRegression(reg_lambda, momentum=0.0, step_size=0.01, num_epochs=300)
        model.fit_ols(X, y)
        predictions = model.predict(X)
        predictions = predictions.flatten() if predictions.ndim == 2 else predictions
        mse = np.mean((y - predictions) ** 2)
        ols_results[reg_lambda] = mse
        logging.debug(f"Finished OLS for reg_lambda={reg_lambda:.3f} with MSE={mse:.5f}")
    
    return ols_results

def run_gd_experiments(X: np.ndarray, y: np.ndarray, reg_lambdas: np.ndarray, momenta: np.ndarray) -> dict:
    """
    Run Gradient Descent experiments over a range of regularization and momentum parameters.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        reg_lambdas (np.ndarray): Array of regularization parameter values.
        momenta (np.ndarray): Array of momentum parameter values.

    Returns:
        dict: Mapping of reg_lambda and momentum to the computed MSE and cost history.
    """    
    gd_results = {}
    # Gradient descent calculations
    for reg_lambda in tqdm(reg_lambdas, desc="Grad Des Hyperparameters"):
        for momentum in momenta:
            logging.debug(f"Running GD experiment for reg_lambda={reg_lambda:.3f}, momentum={momentum:.3f}")
            model = LinearRegression(reg_lambda, momentum, step_size=0.01, num_epochs=300)
            cost_hist = model.fit_grad_descent(X, y, early_stopping=True)
            predictions = model.predict(X)
            predictions = predictions.flatten() if predictions.ndim == 2 else predictions
            mse = np.mean((y - predictions) ** 2)
            key = (reg_lambda, momentum)
            gd_results[key] = {"final_mse": mse, "cost_history": cost_hist}   
            logging.debug(f"Finished GD for reg_lambda={reg_lambda:.3f}, momentum={momentum:.3f} with MSE={mse:.5f}") 
    
    return gd_results

def main() -> None:
    logging.info("Starting main experiment")
    X, y = generate_dataset(n_samples=300, seed=42)
    logging.info("Dataset generated")

    # Define Combinations of Hyperparameters
    reg_lambdas = np.linspace(0.0, 0.8, 10)
    momenta = np.linspace(0.0, 0.8, 10)

    # OLS
    logging.info("Starting OLS experiments")
    ols_results = run_ols_experiments(X, y, reg_lambdas)    
    plot_ols_results(ols_results)
    logging.info("OLS experiments completed and results plotted")

    # Gradient Descent
    logging.info("Starting Gradient Descent experiments")
    gd_results = run_gd_experiments(X, y, reg_lambdas, momenta)
    plot_gd_results(gd_results)
    logging.info("Gradient Descent experiments completed and results plotted")

if __name__ == "__main__":
    main()