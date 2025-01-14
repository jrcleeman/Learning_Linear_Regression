import numpy as np
import matplotlib.pyplot as plt
from get_data import lagged_returns

class LinearRegression:
    def __init__(self):
        pass

    def cost(self, x, y):
        residuals = y - (x @ self.beta)
        return (residuals ** 2).sum() / (2 * len(y))

    # Manual computation
    def fit_ols(self, x, y):
        self.beta = np.linalg.inv(x.transpose() @ x) @ x.transpose() @ y

    def grad(self, x, y):
        grad = (x.transpose() @ ((x @ self.beta) - y)) / len(y)
        return grad
    
    # Gradient Descent
    def fit_grad_descent(self, x, y, step_size=0.01, num_epochs=1000):
        self.beta = np.random.normal(0,1,(x.shape[1], 1)) # initial guess
        cost_hist = []

        for i in range(num_epochs):
            loss = self.cost(x, y)
            cost_hist.append(loss)
            self.beta = self.beta - (step_size * self.grad(x, y)) # Update

            if (i % 100) == 0:
                print("Epoch " + str(i) + " Loss is: " + str(loss) + "\n")

        return cost_hist
    
    def predict(self, x):
        print(x @ self.beta)
        return x @ self.beta

def plot_training(beta, beta_star, x, y):
    print("Beta found: " + str(beta))
    print("True Beta: " + str(beta_star))

    # plot beta plane
    x1_min, x1_max = np.min(x[:,0]), np.max(x[:,0])
    x2_min, x2_max = np.min(x[:,1]), np.max(x[:,1])
    x1_vals = np.expand_dims(np.linspace(x1_min, x1_max, 50), axis=1)
    x2_vals = np.expand_dims(np.linspace(x2_min, x2_max, 50), axis=1)
    x1, x2 = np.meshgrid(x1_vals, x2_vals)

    # Flatten each mesh
    flat_X1 = x1.ravel()  # shape (2500,)
    flat_X2 = x2.ravel()  # shape (2500,)

    # Create a feature matrix for these points (assuming order: x1, x2, 1)
    ones = np.ones_like(flat_X1)  
    mesh_features = np.column_stack((flat_X1, flat_X2, ones))  # shape (2500, 3)

    # Predict z for each point
    Z_pred_flat = mesh_features @ beta  # shape (2500,1) or (2500,) depending on dims

    # Reshape back to (50, 50)
    Z_pred = Z_pred_flat.reshape(x1.shape)  # or (50, 50) 

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points with the same color
    ax.scatter(x[:, 0], x[:, 1], y, color='blue', label="Data Points")

    ax.plot_surface(x1, x2, Z_pred, alpha=0.7, cmap='viridis')

    # Add labels and title
    ax.set_xlabel("Feature 1 (x1)")
    ax.set_ylabel("Feature 2 (x2)")
    ax.set_zlabel("Labels (z)")
    ax.set_title("3D Scatter Plot of Features and Labels")

    # Show the plot
    plt.show()


def main():
    np.random.seed(42)   
    # num_pnts = 1000 
    # num_feat = 2
    # x, y, beta_star = mv_simulate_data(num_pnts, num_feat)
    # beta, cost_hist = grad_descent(x, y)
    # plot_training(beta, beta_star, x, y)

    X_train, X_test, y_train, y_test = lagged_returns()
    model = LinearRegression()
    model.fit_grad_descent(X_train, y_train)
    test_rmse = model.cost(X_test, y_test)
    print(f'Test RMSE: {test_rmse}')

main()