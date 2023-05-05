import numpy as np
import time
from tqdm import tqdm


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def vanilla_gradient_descent(X, y, alpha=0.01, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for i in tqdm(range(num_iterations), desc="Vanilla Gradient Descent"):
        hypothesis = sigmoid(X.dot(theta))
        error = hypothesis - y.reshape(-1, 1)
        gradient = X.T.dot(error) / m
        theta -= alpha * gradient

    return theta


def stochastic_gradient_descent(X, y, alpha=0.01, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for i in tqdm(range(num_iterations), desc="Stochastic Gradient Descent"):
        for j in range(m):
            hypothesis = sigmoid(X[j, :].dot(theta))
            error = hypothesis - y[j]
            gradient = X[j, :].reshape(-1, 1) * error
            theta -= alpha * gradient

    return theta


def batch_stochastic_gradient_descent(
    X, y, batch_size=20, alpha=0.01, num_iterations=1000
):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for i in tqdm(range(num_iterations), desc="Batch Stochastic Gradient Descent"):
        batch_indices = np.random.randint(0, m, batch_size)
        X_batch = X[batch_indices, :]
        y_batch = y[batch_indices].reshape(-1, 1)
        hypothesis = sigmoid(X_batch.dot(theta))
        error = hypothesis - y_batch
        gradient = X_batch.T.dot(error) / batch_size
        theta -= alpha * gradient

    return theta


# Generate random data
m = 50000
n = 150
X = np.random.randn(m, n)
theta_true = np.random.randn(n, 1)
y = sigmoid(X.dot(theta_true) + np.random.randn(m, 1))

# Test the three algorithms
start_time = time.time()
theta_vanilla = vanilla_gradient_descent(X, y)
print(
    f"Vanilla GD: {time.time() - start_time:.4f} seconds, accuracy: {np.mean(np.abs(theta_vanilla - theta_true)):.4f}"
)

start_time = time.time()
theta_stochastic = stochastic_gradient_descent(X, y)
print(
    f"Stochastic GD: {time.time() - start_time:.4f} seconds, accuracy: {np.mean(np.abs(theta_stochastic - theta_true)):.4f}"
)

start_time = time.time()
theta_batch = batch_stochastic_gradient_descent(X, y)
print(
    f"Batch SGD: {time.time() - start_time:.4f} seconds, accuracy: {np.mean(np.abs(theta_batch - theta_true)):.4f}"
)
