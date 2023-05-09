import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def cost_function(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = 1 / (2 * m) * np.sum((h - y) ** 2)
    return J


def gradient(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    grad = 1 / m * X.T.dot(h - y)
    return grad


def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


def stochastic_gradient_descent(
    X, y, theta, alpha, num_iters, optimizer="stochastic", **kwargs
):
    assert optimizer in [
        "adam",
        "rmsprop",
        "adagrad",
        "stochastic",
    ], "Invalid optimizer. Choose from 'adam', 'rmsprop', 'adagrad', or 'stochastic'."

    m = len(y)
    m_t = np.zeros(theta.shape)
    v_t = np.zeros(theta.shape)
    J_history = np.zeros(num_iters)
    start_time = time.time()

    for t in tqdm(range(num_iters), desc=f"With {optimizer.capitalize()} Optimizer"):
        if optimizer == "stochastic":
            for j in range(m):
                rand_index = np.random.randint(0, m)
                X_j = X[rand_index, :].reshape(1, X.shape[1])
                y_j = y[rand_index].reshape(1, 1)
                grad = gradient(X_j, y_j, theta)
        else:
            grad = gradient(X, y, theta)

        if optimizer == "adam":
            beta1 = kwargs.get("beta1", 0.9)
            beta2 = kwargs.get("beta2", 0.2)
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

        elif optimizer == "stochastic":
            theta = theta - alpha * grad

        J_history[t] = cost_function(X, y, theta)

    end_time = time.time()
    time_taken = end_time - start_time
    accuracy_sgd = accuracy(y, (X.dot(theta) >= 0.5).astype(int))
    return theta, J_history, accuracy_sgd, time_taken


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1) >= 0.5  # generate binary labels


# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.01
num_iters = 1000


print("Stochastic Gradient Descent")
# # Run Stochastic Gradient Descent with Adam Optimizer and record running time
(
    theta_adam,
    adam_loss,
    adam_accuracy,
    adam_time_taken,
) = stochastic_gradient_descent(X, y, theta, alpha, num_iters, "adam")

print("Theta: ", theta_adam.ravel()[:5])
print("Cost: ", adam_loss[-1])
print("Training accuracy: ", adam_accuracy)
print("Running time: ", adam_time_taken, "seconds")

# Run Stochastic Gradient Descent with RMSProp Optimizer and record running time
(
    theta_rmsvgd,
    rmsprop_loss,
    rms_accuracy,
    rms_time_taken,
) = stochastic_gradient_descent(X, y, theta, alpha, num_iters, "rmsprop")

print("Theta: ", theta_rmsvgd.ravel()[:5])
print("Cost: ", rmsprop_loss[-1])
print("Training accuracy: ", rms_accuracy)
print("Running time: ", rms_time_taken, "seconds")

# # Run Stochastic Gradient Descent with AdaGrad Optimizer and record running time
(
    theta_ada_grad,
    adagrad_loss,
    ada_grad_accuracy,
    ada_grad_time_taken,
) = stochastic_gradient_descent(X, y, theta, alpha, num_iters, "adagrad")
print("Theta: ", theta_ada_grad.ravel()[:5])
print("Cost: ", adagrad_loss[-1])
print("Training accuracy: ", ada_grad_accuracy)
print("Running time: ", ada_grad_time_taken, "seconds")


# Plot the cost function over iterations for each optimizer
plt.figure(figsize=(12, 8))
plt.plot(range(num_iters), adam_loss, label="Adam")
plt.plot(range(num_iters), rmsprop_loss, label="RMSProp")
plt.plot(range(num_iters), adagrad_loss, label="AdaGrad")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost function over iterations for Stochastic Gradient Descent")
plt.legend()
plt.show()
