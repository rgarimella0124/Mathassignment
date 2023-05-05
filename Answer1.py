import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


# Define the cost function
def cost_function(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = 1 / (2 * m) * np.sum((h - y) ** 2)
    return J


# Define the gradient function
def gradient(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    grad = 1 / m * X.T.dot(h - y)
    return grad


# Define the Vanilla Gradient Descent function
def vanilla_gradient_descent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    for i in tqdm(range(num_iters), desc="Vanilla Gradient Descent"):
        grad = gradient(X, y, theta)
        theta = theta - alpha * grad
        J_history[i] = cost_function(X, y, theta)
    return theta, J_history


# Define the Stochastic Gradient Descent function
def stochastic_gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    # print("=====================Running Stochastic Gradient Descent=================")
    for i in tqdm(range(num_iters), desc="Stochastic Gradient Descent"):
        for j in range(m):
            rand_index = np.random.randint(0, m)
            X_j = X[rand_index, :].reshape(1, X.shape[1])
            y_j = y[rand_index].reshape(1, 1)
            grad = gradient(X_j, y_j, theta)
            theta = theta - alpha * grad
        J_history[i] = cost_function(X, y, theta)

    return theta, J_history


# Define the Batch Stochastic Gradient Descent function
def batch_stochastic_gradient_descent(X, y, theta, alpha, num_iters, batch_size):
    m = len(y)
    J_history = np.zeros(num_iters)
    # print(
    #     "====================Running Batch Stochastic Gradient Descent======================="
    # )
    for i in tqdm(range(num_iters), desc="Batch Stochastic Gradient Descent"):
        rand_indices = np.random.randint(0, m, size=batch_size)
        X_b = X[rand_indices, :]
        y_b = y[rand_indices]
        grad = gradient(X_b, y_b, theta)
        theta = theta - alpha * grad
        J_history[i] = cost_function(X, y, theta)

    return theta, J_history


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1)

# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.01
num_iters = 1000
batch_size = 100

# Run Vanilla Gradient Descent and record running time
start_time = time.time()
theta_vgd, J_history_vgd = vanilla_gradient_descent(X, y, theta, alpha, num_iters)
end_time = time.time()
y_pred = X.dot(theta_vgd)
accuracy_vgd = 1 - mean_squared_error(y, y_pred)
print("Theta: ", theta_vgd.ravel()[:5], "...")
print("Cost: ", J_history_vgd[-1])
print("Running time: ", end_time - start_time, "seconds")
print("Accuracy is: ", accuracy_vgd)


# Run Stochastic Gradient Descent and record running time
start_time = time.time()
theta_sgd, J_history_sgd = stochastic_gradient_descent(X, y, theta, alpha, num_iters)
end_time = time.time()
y_pred = X.dot(theta_sgd)
accuracy_sgd = 1 - mean_squared_error(y, y_pred)
print("Theta: ", theta_sgd.ravel()[:5], "...")
print("Cost: ", J_history_sgd[-1])
print("Running time: ", end_time - start_time, "seconds")
print("Accuracy is: ", accuracy_sgd)

# Run Batch Stochastic Gradient Descent and record running time
start_time = time.time()
theta_bsgd, J_history_bsgd = batch_stochastic_gradient_descent(
    X, y, theta, alpha, num_iters, batch_size
)
end_time = time.time()
y_pred = X.dot(theta_bsgd)
accuracy_bsgd = 1 - mean_squared_error(y, y_pred)
print("Theta: ", theta_bsgd.ravel()[:5], "...")
print("Cost: ", J_history_bsgd[-1])
print("Running time: ", end_time - start_time, "seconds")
print("Accuracy is: ", accuracy_bsgd)
