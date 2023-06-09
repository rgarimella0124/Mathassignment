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


def VanillaGradientDescent(X, y, theta, alpha, num_iters, optimizer="adam", **kwargs):
    assert optimizer in [
        "adam",
        "rmsprop",
        "adagrad",
        "vanilla",
    ], "Invalid optimizer. Choose from 'adam', 'rmsprop', 'adagrad', or 'vanilla'."

    m = len(y)
    m_t = np.zeros(theta.shape)
    v_t = np.zeros(theta.shape)
    J_history = np.zeros(num_iters)
    start_time = time.time()

    for t in tqdm(range(num_iters), desc=f"With {optimizer.capitalize()} Optimizer"):
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

        elif optimizer == "vanilla":
            theta = theta - alpha * grad

        J_history[t] = cost_function(X, y, theta)

    end_time = time.time()
    time_taken = end_time - start_time
    accuracy_vgd = accuracy(y, (X.dot(theta) >= 0.5).astype(int))
    return theta, J_history, accuracy_vgd, time_taken


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1) >= 0.5  # generate binary labels


# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.01
num_iters = 1000


print("Vanilla Gradient Descent")
# # Run Vanilla Gradient Descent with Adam Optimizer and record running time
(
    theta_avgd,
    J_history_avgd,
    adam_accuracy,
    adam_time_taken,
) = VanillaGradientDescent(X, y, theta, alpha, num_iters, "adam")
print("Vanilla Gradient Descent with Adam Optimizer ")
print("Theta: ", theta_avgd.ravel()[:5])
print("Cost: ", J_history_avgd[-1])
print("Training accuracy: ", adam_accuracy)
print("Running time: ", adam_time_taken, "seconds")

# Run Vanilla Gradient Descent with RMSProp Optimizer and record running time
(
    theta_rmsvgd,
    J_history_rmsvgd,
    rms_accuracy,
    rms_time_taken,
) = VanillaGradientDescent(X, y, theta, alpha, num_iters, "rmsprop")
print("Vanilla Gradient Descent with ")
print("Theta: ", theta_rmsvgd.ravel()[:5])
print("Cost: ", J_history_rmsvgd[-1])
print("Training accuracy: ", rms_accuracy)
print("Running time: ", rms_time_taken, "seconds")

# # Run Vanilla Gradient Descent with AdaGrad Optimizer and record running time
(
    theta_ada_grad_vgd,
    J_history_ada_grad,
    ada_grad_accuracy,
    ada_grad_time_taken,
) = VanillaGradientDescent(X, y, theta, alpha, num_iters, "adagrad")
print("Vanilla Gradient Descent with ")
print("Theta: ", theta_ada_grad_vgd.ravel()[:5])
print("Cost: ", J_history_ada_grad[-1])
print("Training accuracy:", ada_grad_accuracy)
print("Running time: ", ada_grad_time_taken, "seconds")


# Plot the cost function over iterations for each optimizer
plt.figure(figsize=(12, 8))
plt.plot(range(num_iters), J_history_avgd, label="Adam")
plt.plot(range(num_iters), J_history_rmsvgd, label="RMSProp")
plt.plot(range(num_iters), J_history_ada_grad, label="AdaGrad")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost function over iterations for Vanilla Gradient Descent")
plt.legend()
plt.show()