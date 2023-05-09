import numpy as np
import time
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define the Batch Stochastic Gradient Descent function with Adam optimizer
def batch_stochastic_gradient_descent_adam(X, y, theta, alpha, num_iters, batch_size):
    optimizer = Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    start_time = time.time()
    J_history = []
    num_batches = int(np.ceil(len(y) / batch_size))
    total_batches = num_iters * num_batches
    for i in tqdm(range(total_batches)):
        batch_idx = i % num_batches
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(y))
        X_batch = X[batch_start:batch_end, :]
        y_batch = y[batch_start:batch_end]
        history = model.train_on_batch(X_batch, y_batch)
        J_history.append(history[0])
        if (i + 1) % num_batches == 0:
            model_weights = model.get_weights()[0]
            theta = (i / num_batches * theta + model_weights) / ((i / num_batches) + 1)
    end_time = time.time()
    accuracy = model.evaluate(X, y, verbose=0)[1]
    return theta, J_history, accuracy, end_time - start_time


# Define the Batch Stochastic Gradient Descent function with RMS Prop optimizer
def batch_stochastic_gradient_descent_rmsprop(
    X, y, theta, alpha, num_iters, batch_size, decay_rate=0.9, epsilon=1e-8
):
    optimizer = RMSprop(learning_rate=alpha, decay=decay_rate, epsilon=epsilon)
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    start_time = time.time()
    J_history = []
    num_batches = int(np.ceil(len(y) / batch_size))
    for i in range(num_iters):
        for j in range(num_batches):
            batch_start = j * batch_size
            batch_end = min((j + 1) * batch_size, len(y))
            X_batch = X[batch_start:batch_end, :]
            y_batch = y[batch_start:batch_end]
            history = model.train_on_batch(X_batch, y_batch)
            J_history.append(history[0])
    end_time = time.time()
    accuracy = model.evaluate(X, y, verbose=0)[1]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


# Define the Batch Stochastic Gradient Descent function with AdaGrad optimizer
def adagrad_batch_stochastic_gradient_descent(
    X, y, theta, alpha, num_iters, batch_size, epsilon=1e-8
):
    optimizer = Adagrad(learning_rate=alpha, epsilon=epsilon)
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    start_time = time.time()
    J_history = []
    num_batches = int(np.ceil(len(y) / batch_size))
    for i in range(num_iters):
        for j in range(num_batches):
            batch_start = j * batch_size
            batch_end = min((j + 1) * batch_size, len(y))
            X_batch = X[batch_start:batch_end, :]
            y_batch = y[batch_start:batch_end]
            history = model.train_on_batch(X_batch, y_batch)
            J_history.append(history[0])
    end_time = time.time()
    accuracy = model.evaluate(X, y, verbose=0)[1]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.randint(2, size=(m, 1))

# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.1
num_iters = 1000
batch_size = 100


# # Run Batch Stochastic Gradient Descent with Adam Optimizer and record running time
(
    theta_avgd,
    J_history_avgd,
    adam_accuracy,
    adam_time_taken,
) = batch_stochastic_gradient_descent_adam(X, y, theta, alpha, num_iters, batch_size)
print("Theta: ", theta_avgd.ravel()[:5])
print("Cost: ", J_history_avgd[-1])
print("Accuracy: ", adam_accuracy)
print("Running time: ", adam_time_taken, "seconds")


# # Run Batch Stochastic Gradient Descent with RMSProp Optimizer and record running time
# (
#     theta_rmsvgd,
#     J_history_rmsvgd,
#     rms_accuracy,
#     rms_time_taken,
# ) = batch_stochastic_gradient_descent_rmsprop(X, y, theta, alpha, num_iters, batch_size)
# print("Theta: ", theta_rmsvgd.ravel()[:5])
# print("Cost: ", J_history_rmsvgd[-1])
# print("Accuracy: ", rms_accuracy)
# print("Running time: ", rms_time_taken, "seconds")

# # # Run Batch Stochastic Gradient Descent with AdaGrad Optimizer and record running time
# (
#     theta_ada_grad_vgd,
#     J_history_ada_grad_vgd,
#     ada_grad_accuracy,
#     ada_grad_time_taken,
# ) = batch_stochastic_gradient_descent_adagrad(X, y, theta, alpha, num_iters, batch_size)
# print("Vanilla Gradient Descent with ")
# print("Theta: ", theta_ada_grad_vgd.ravel()[:5])
# print("Cost: ", J_history_ada_grad_vgd[-1])
# print("Accuracy: ", rms_accuracy)
# print("Running time: ", ada_grad_time_taken, "seconds")
