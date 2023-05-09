import numpy as np
import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def sigmoid(z):
  return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    total_cost = 0
    
    for i in range(m):
        if y[i] == 1:
            total_cost -= np.log(h[i])
        else:
            total_cost -= np.log(1 - h[i])
    
    cost = total_cost / m
    return cost




def batch_stochastic_gradient_descent(X,
                                      y,
                                      theta,
                                      alpha,
                                      num_iters,
                                      batch_size,
                                      optimization=None):
  m, n = X.shape
  costs = []
  beta1 = 0.9
  beta2 = 0.999
  epsilon = 1e-8

  if optimization == "adam":
    mt = np.zeros((n, 1))
    vt = np.zeros((n, 1))
  elif optimization == "adagrad":
    gt = np.zeros((n, 1))
  elif optimization == "rmsprop":
    et = np.zeros((n, 1))

  for i in tqdm(range(num_iters), desc="Iterations"):
    indices = np.random.permutation(m)
    X = X[indices]
    y = y[indices]
    for j in range(0, m, batch_size):
      X_batch = X[j:j + batch_size]
      y_batch = y[j:j + batch_size]

      grad = (1 /
              batch_size) * (X_batch.T @ (sigmoid(X_batch @ theta) - y_batch))

      if optimization == "adam":
        mt = beta1 * mt + (1 - beta1) * grad
        vt = beta2 * vt + (1 - beta2) * (grad**2)
        mt_hat = mt / (1 - beta1**(i + 1))
        vt_hat = vt / (1 - beta2**(i + 1))
        theta = theta - alpha * mt_hat / (np.sqrt(vt_hat) + epsilon)
      elif optimization == "adagrad":
        gt += grad**2
        theta = theta - alpha * grad / (np.sqrt(gt) + epsilon)
      elif optimization == "rmsprop":
        et = beta2 * et + (1 - beta2) * (grad**2)
        theta = theta - alpha * grad / (np.sqrt(et) + epsilon)
      else:
        theta = theta - alpha * grad

      cost = cost_function(X_batch, y_batch, theta)
      costs.append(cost)

  return theta, costs


def predict(X, theta):
  return (sigmoid(X @ theta) >= 0.5).astype(int)


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1) >= 0.5  # generate binary labels

# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.01
num_iters = 1000
# Set batch_size
batch_size = 24

# Adam optimization
start_time = time.time()
theta_adam, costs_adam = batch_stochastic_gradient_descent(X,
                                                           y,
                                                           theta,
                                                           alpha,
                                                           num_iters,
                                                           batch_size,
                                                           optimization="adam")
time_adam = time.time() - start_time
y_pred_adam = predict(X, theta_adam)
accuracy_adam = accuracy_score(y, y_pred_adam)

# Adagrad optimization
start_time = time.time()
theta_adagrad, costs_adagrad = batch_stochastic_gradient_descent(
  X, y, theta, alpha, num_iters, batch_size, optimization="adagrad")
time_adagrad = time.time() - start_time
y_pred_adagrad = predict(X, theta_adagrad)
accuracy_adagrad = accuracy_score(y, y_pred_adagrad)

# RMSprop optimization
start_time = time.time()
theta_rmsprop, costs_rmsprop = batch_stochastic_gradient_descent(
  X, y, theta, alpha, num_iters, batch_size, optimization="rmsprop")
time_rmsprop = time.time() - start_time
y_pred_rmsprop = predict(X, theta_rmsprop)
accuracy_rmsprop = accuracy_score(y, y_pred_rmsprop)

# Display results
print("Adam optimization:")
print(f"  Time taken: {time_adam:.4f} seconds")
print(f"  Accuracy: {accuracy_adam * 100:.2f}%")

print("\nAdagrad optimization:")
print(f"  Time taken: {time_adagrad:.4f} seconds")
print(f"  Accuracy: {accuracy_adagrad * 100:.2f}%")

print("\nRMSprop optimization:")
print(f"  Time taken: {time_rmsprop:.4f} seconds")
print(f"  Accuracy: {accuracy_rmsprop * 100:.2f}%")


# Function to plot costs
def plot_costs(costs_adam, costs_adagrad, costs_rmsprop):
  plt.figure(figsize=(10, 6))

  # Calculate the number of cost values to average
  num_costs_to_average = len(costs_adam) // num_iters

  # Calculate average costs for smoother plots
  costs_adam_avg = [
    np.mean(costs_adam[i:i + num_costs_to_average])
    for i in range(0, len(costs_adam), num_costs_to_average)
  ]
  costs_adagrad_avg = [
    np.mean(costs_adagrad[i:i + num_costs_to_average])
    for i in range(0, len(costs_adagrad), num_costs_to_average)
  ]
  costs_rmsprop_avg = [
    np.mean(costs_rmsprop[i:i + num_costs_to_average])
    for i in range(0, len(costs_rmsprop), num_costs_to_average)
  ]

  plt.plot(costs_adam_avg, label="Adam", lw=2)
  plt.plot(costs_adagrad_avg, label="Adagrad", lw=2)
  plt.plot(costs_rmsprop_avg, label="RMSprop", lw=2)

  plt.xlabel("Iterations")
  plt.ylabel("Cost")
  plt.title("Cost function values over iterations")
  plt.legend()
  plt.grid()
  plt.show()


# Plot costs for Adam, Adagrad, and RMSprop
plot_costs(costs_adam, costs_adagrad, costs_rmsprop)
