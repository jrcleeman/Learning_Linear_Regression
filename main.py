import numpy as np
import matplotlib.pyplot as plt

# single variable simulate data
def sv_simulate_data(num_points):
    x_y_pairs = np.array([[0,0] for _ in range(num_points)], dtype=float)
    beta_0 = 0.5
    beta_1 = 0.4

    for i in range(num_points):
        val= np.random.normal(0,1)
        x_y_pairs[i][0] = val
        x_y_pairs[i][1] = (x_y_pairs[i][0] * beta_1) + beta_0 + np.random.normal(0,0.1)

    # plot
    x = x_y_pairs[:,0]
    y = x_y_pairs[:,1]

    # Plot the points
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='red', label='Points')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of X, Y Pairs')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()
    
    return x, y, beta_0, beta_1

# multi variable simulate data
def mv_simulate_data(num_points):
    num_features = 2

    feature_array = np.random.uniform(-5, 5, (num_points, num_features)) # n x p
    feature_array = np.hstack((feature_array, np.ones((num_points,1)))) # n x (p + 1) # Adding column of ones for intercept
    beta_vector = np.random.uniform(1, 3, (num_features + 1, 1)) # (p + 1) x 1
    noise_vector = np.random.normal(0, 1, (num_points, 1)) # n x 1
    
    labels = feature_array @ beta_vector + noise_vector
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points with the same color
    ax.scatter(feature_array[:, 0], feature_array[:, 1], labels, color='blue', label="Data Points")

    # Add labels and title
    ax.set_xlabel("Feature 1 (x1)")
    ax.set_ylabel("Feature 2 (x2)")
    ax.set_zlabel("Labels (z)")
    ax.set_title("3D Scatter Plot of Features and Labels")

    # Show the plot
    plt.show()

    return feature_array, labels, beta_vector

# Multi variable
def ols_mv(feature_arr, label_arr):
    beta = np.linalg.inv(feature_arr.transpose() @ feature_arr) @ feature_arr.transpose() @ label_arr
    return beta

# Single variable
def ols_sv(features, labels, num_points, true_beta_0, true_beta_1):
    sum_x = np.sum(features)
    mean_x = sum_x / num_points
    sum_y = np.sum(labels)
    mean_y = sum_y / num_points    

    # Solving for beta_1
    numerator = 0
    denominator = 0 
    for i in range(num_points):
        numerator += ((features[i] - mean_x) * (labels[i] - mean_y))
        denominator += ((features[i] - mean_x) * (features[i] - mean_x))
    
    beta_1 = numerator/denominator
    beta_0 = mean_y - (beta_1*mean_x)

    x_min, x_max = np.min(features), np.max(features)
    x_points = np.linspace(x_min, x_max, 100)
    y_points = beta_1*x_points + beta_0
    y_points_true = true_beta_1*x_points + true_beta_0

    plt.scatter(features, labels, color="red", label="Data")
    plt.plot(x_points, y_points, color="blue", label="Fitted Line")
    plt.plot(x_points, y_points_true, color="green", label="True Line")
    plt.legend()
    plt.show()

    return beta_1, beta_0

def gd_mv(features_array, labels_array, true_beta):
    def cost(features_array, beta, labels_array):
        pred_vec = features_array @ beta
        delta = labels_array - pred_vec #.squeeze()
        total_cost = sum(delta) / (2 * len(labels_array))
        return total_cost[0]

    def gradient(features_array, beta, labels_array):
        temp = (features_array @ beta) - labels_array
        temp2 = features_array.transpose() @ temp
        grad = temp2 / len(labels_array)
        return grad

    beta = np.random.normal(0,1,(features_array.shape[1], 1)) #initial guess
    total_cost = []
    num_epochs = 1000
    step_size = 0.01

    for i in range(num_epochs):
        loss = cost(features_array, beta, labels_array)
        print("Current Loss is: " + str(loss) + "\n")
        total_cost.append(loss)

        beta = beta - (step_size * gradient(features_array, beta, labels_array))    

    print("Beta found: " + str(beta))
    print("True Beta: " + str(true_beta))

    # plot beta plane
    x_min, x_max = np.min(features_array[:,0]), np.max(features_array[:,0])
    y_min, y_max = np.min(features_array[:,1]), np.max(features_array[:,1])
    x1_vals = np.expand_dims(np.linspace(x_min, x_max, 50), axis=1)
    x2_vals = np.expand_dims(np.linspace(y_min, y_max, 50), axis=1)
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
    ax.scatter(features_array[:, 0], features_array[:, 1], labels_array, color='blue', label="Data Points")

    ax.plot_surface(x1, x2, Z_pred, alpha=0.7, cmap='viridis')

    # Add labels and title
    ax.set_xlabel("Feature 1 (x1)")
    ax.set_ylabel("Feature 2 (x2)")
    ax.set_zlabel("Labels (z)")
    ax.set_title("3D Scatter Plot of Features and Labels")

    # Show the plot
    plt.show()


def gd_sv(features, labels, num_points, true_beta_0, true_beta_1):
    def cost(pred_vec, labels):
        total_cost = 0
        for i in range(len(pred_vec)):
            total_cost += ((labels[i] - pred_vec[i]) ** 2)
        
        return total_cost / (2 * len(pred_vec))
    
    def cost_wrt_beta_0(pred_vec, labels):
        total_cost = 0
        for i in range(len(pred_vec)):
            total_cost += (labels[i] - pred_vec[i])
        
        return -1 * total_cost / len(pred_vec)
    
    def cost_wrt_beta_1(pred_vec, labels, x_vec):
        total_cost = 0
        for i in range(len(pred_vec)):
            total_cost += (x_vec[i] * (labels[i] - pred_vec[i]))
        
        return -1 * total_cost / len(pred_vec) 
      
    step_size = 0.01
    beta_0 = 0.1 # initial guess
    beta_1 = 0.1 # initial guess
    num_epochs = 500

    pred = lambda b_0, b_1, x : b_0 + (b_1 * x)
    cost_hist = []
    for i in range(num_epochs):
        pred_vec = [pred(beta_0, beta_1, x) for x in features]
        cost_hist.append(cost(pred_vec, labels))

        # Gradient descent
        beta_0 = beta_0 - (step_size * cost_wrt_beta_0(pred_vec, labels))
        beta_1 = beta_1 - (step_size * cost_wrt_beta_1(pred_vec, labels, x_vec=features))

    print("True Value of Beta_0: " + str(true_beta_0))        
    print("Final Value of Beta_0: " + str(beta_0))
    print("True Value of Beta_1: " + str(true_beta_1))
    print("Final Value of Beta_1: " + str(beta_1))

    plt.figure(figsize=(8, 5))
    plt.plot([x for x in range(num_epochs)], cost_hist, color='red', label='Points')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()

    x_min, x_max = np.min(features), np.max(features)
    x_points = np.linspace(x_min, x_max, 100)
    y_points = beta_1*x_points + beta_0
    y_points_true = true_beta_1*x_points + true_beta_0

    plt.scatter(features, labels, color="red", label="Data")
    plt.plot(x_points, y_points, color="blue", label="Fitted Line")
    plt.plot(x_points, y_points_true, color="green", label="True Line")
    plt.legend()
    plt.show()  

def main():
    np.random.seed(42)   
    num_points = 1000 
    # features,labels, true_beta_0, true_beta_1 = sv_simulate_data(num_points)
    # gd_sv(features, labels,num_points, true_beta_0, true_beta_1)   
    # ols_sv(features,labels,num_points, true_beta_0, true_beta_1)
 
    features_array, labels_array, true_beta = mv_simulate_data(num_points)
    # ols_mv(feature_array, label_array)
    gd_mv(features_array, labels_array, true_beta)


main()