import numpy as np
import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sgd_adam(X, y, theta, alpha, num_iters, m):
    b1, b2 = 0.9, 0.999
    epsilon = 1e-8
    mt, vt = np.zeros(theta.shape), np.zeros(theta.shape)
    t = 0

    for _ in tqdm(range(num_iters), desc="Adam"):
        for i in range(m):
            t += 1
            xi = X[i, :].reshape(1, -1)
            yi = y[i, :].reshape(1, -1)
            gradient = (1 / m) * xi.T * (sigmoid(xi @ theta) - yi)

            mt = b1 * mt + (1 - b1) * gradient
            vt = b2 * vt + (1 - b2) * (gradient**2)

            mt_corrected = mt / (1 - b1**t)
            vt_corrected = vt / (1 - b2**t)

            theta -= alpha * mt_corrected / (np.sqrt(vt_corrected) + epsilon)
    return theta


def sgd_rmsprop(X, y, theta, alpha, num_iters, m):
    b = 0.9
    epsilon = 1e-8
    vt = np.zeros(theta.shape)

    for _ in tqdm(range(num_iters), desc="RMSProp"):
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i, :].reshape(1, -1)
            gradient = (1 / m) * xi.T * (sigmoid(xi @ theta) - yi)

            vt = b * vt + (1 - b) * (gradient**2)

            theta -= alpha * gradient / (np.sqrt(vt) + epsilon)
    return theta


def sgd_adagrad(X, y, theta, alpha, num_iters, m):
    epsilon = 1e-8
    vt = np.zeros(theta.shape)

    for _ in tqdm(range(num_iters), desc="Adagrad"):
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i, :].reshape(1, -1)
            gradient = (1 / m) * xi.T * (sigmoid(xi @ theta) - yi)

            vt += gradient**2

            theta -= alpha * gradient / (np.sqrt(vt) + epsilon)
    return theta


def evaluate_time_accuracy(optimizer, X, y, theta, alpha, num_iters, m):
    start_time = time.time()
    theta_optimized = optimizer(X, y, theta, alpha, num_iters, m)
    end_time = time.time()
    time_taken = end_time - start_time

    y_pred = sigmoid(X @ theta_optimized) >= 0.5
    accuracy = accuracy_score(y, y_pred)

    return time_taken, accuracy


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1) >= 0.5  # generate binary labels

# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.01
num_iters = 1000

optimizers = {
    "Adam": sgd_adam,
    "RMSProp": sgd_rmsprop,
    "Adagrad": sgd_adagrad,
}

for name, optimizer in optimizers.items():
    time_taken, accuracy = evaluate_time_accuracy(
        optimizer, X, y, theta, alpha, num_iters, m
    )
    print(f"{name}: Time taken = {time_taken:.4f} seconds, Accuracy = {accuracy:.4f}")


