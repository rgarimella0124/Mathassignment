import numpy as np
import time
from tqdm import tqdm


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


def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


def batch_stochastic_gradient_descent(
    X, y, theta, alpha, num_iters, optimizer="batch_sgd", batch_size=100, **kwargs
):
    assert optimizer in [
        "adam",
        "rmsprop",
        "adagrad",
        "batch_sgd",
    ], "Invalid optimizer. Choose from 'adam', 'rmsprop', 'adagrad', or 'batch_sgd'."

    m = len(y)
    m_t = np.zeros(theta.shape)
    v_t = np.zeros(theta.shape)
    J_history = np.zeros(num_iters)
    start_time = time.time()

    for t in tqdm(range(num_iters), desc=f"With {optimizer.capitalize()} Optimizer"):
        if optimizer == "batch_sgd":
            rand_indices = np.random.randint(0, m, size=batch_size)
            X_b = X[rand_indices, :]
            y_b = y[rand_indices]
            grad = gradient(X_b, y_b, theta)
        else:
            grad = gradient(X, y, theta)

        if optimizer == "adam":
            beta1 = kwargs.get("beta1", 0.9)
            beta2 = kwargs.get("beta2", 0.999)
            epsilon = kwargs.get("epsilon", 1e-8)
            m_t = beta1 * m_t + (1 - beta1) * grad
            v_t = beta2 * v_t + (1 - beta2) * (grad**2)
            m_t_hat = m_t / (1 - beta1 ** (t + 1))
            v_t_hat = v_t / (1 - beta2 ** (t + 1))
            theta = theta - alpha * m_t_hat / (np.sqrt(v_t_hat) + epsilon)

        elif optimizer == "rmsprop":
            beta = kwargs.get("beta", 0.9)
            epsilon = kwargs.get("epsilon", 1e-8)
            v_t = beta * v_t + (1 - beta) * (grad**2)
            theta = theta - alpha * grad / (np.sqrt(v_t) + epsilon)

        elif optimizer == "adagrad":
            epsilon = kwargs.get("epsilon", 1e-8)
            v_t = v_t + grad**2
            theta = theta - alpha * grad / (np.sqrt(v_t) + epsilon)

        elif optimizer == "batch_sgd":
            theta = theta - alpha * grad

        J_history[t] = cost_function(X, y, theta)

    end_time = time.time()
    time_taken = end_time - start_time
    accuracy_bsgd = accuracy(y, (X.dot(theta) >= 0.5).astype(int))
    return theta, J_history, accuracy_bsgd, time_taken


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1) >= 0.5

# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.1
num_iters = 1000


print("Batch Stochastic Gradient Descent")

# # Run Batch Stochastic Gradient Descent with Adam Optimizer and record running time
(
    theta,
    adam_loss,
    adam_accuracy,
    adam_time_taken,
) = batch_stochastic_gradient_descent(X, y, theta, alpha, num_iters, "adam")
print("Theta: ", theta_avgd.ravel()[:5])
print("Cost: ", adam_loss[-1])
print("Accuracy: ", adam_accuracy)
print("Running time: ", adam_time_taken, "seconds")


# # Run Batch Stochastic Gradient Descent with rmsprop Optimizer and record running time
(
    theta,
    adam_loss,
    adam_accuracy,
    adam_time_taken,
) = batch_stochastic_gradient_descent(X, y, theta, alpha, num_iters, "rmsprop")
print("Theta: ", theta_avgd.ravel()[:5])
print("Cost: ", adam_loss)
print("Accuracy: ", adam_accuracy)
print("Running time: ", adam_time_taken, "seconds")


# # Run Batch Stochastic Gradient Descent with adagrad Optimizer and record running time
(
    theta,
    adam_loss,
    adam_accuracy,
    adam_time_taken,
) = batch_stochastic_gradient_descent(X, y, theta, alpha, num_iters, "adagrad")
print("Theta: ", theta_avgd.ravel()[:5])
print("Cost: ", adam_loss)
print("Accuracy: ", adam_accuracy)
print("Running time: ", adam_time_taken, "seconds")
