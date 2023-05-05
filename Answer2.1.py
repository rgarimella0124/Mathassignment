import numpy as np
import time
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers import Dense


# Optimizing the Vanilla Gradient Descent function
def adam_vanilla_gradient_descent(X, y, theta, alpha, num_iters):
    optimizer = Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    start_time = time.time()
    history = model.fit(X, y, epochs=num_iters, batch_size=64, verbose=0)
    end_time = time.time()
    J_history = history.history["loss"]
    accuracy = history.history["binary_accuracy"]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


def rmsprop_vanilla_gradient_descent(X, y, theta, alpha, num_iters):
    optimizer = RMSprop(learning_rate=alpha, rho=0.9, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    start_time = time.time()
    history = model.fit(X, y, epochs=num_iters, batch_size=64, verbose=0)
    end_time = time.time()
    J_history = history.history["loss"]
    accuracy = history.history["binary_accuracy"]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


def adagrad_vanilla_gradient_descent(X, y, theta, alpha, num_iters):
    optimizer = Adagrad(learning_rate=alpha, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    start_time = time.time()
    history = model.fit(X, y, epochs=num_iters, batch_size=64, verbose=0)
    end_time = time.time()
    J_history = history.history["loss"]
    accuracy = history.history["binary_accuracy"]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.randint(2, size=(m, 1))

# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.01
num_iters = 100

# # Run Vanilla Gradient Descent with Adam Optimizer and record running time
(
    theta_avgd,
    J_history_avgd,
    adam_accuracy,
    adam_time_taken,
) = adam_vanilla_gradient_descent(X, y, theta, alpha, num_iters)
print("Vanilla Gradient Descent with Adam Optimizer ")
print("Theta: ", theta_avgd.ravel()[:5])
print("Cost: ", J_history_avgd[-1])
print("Training accuracy: {:.2f}%".format(adam_accuracy[-1] * 100))
print("Running time: ", adam_time_taken, "seconds")

# Run Vanilla Gradient Descent with RMSProp Optimizer and record running time
(
    theta_rmsvgd,
    J_history_rmsvgd,
    rms_accuracy,
    rms_time_taken,
) = rmsprop_vanilla_gradient_descent(X, y, theta, alpha, num_iters)
print("Vanilla Gradient Descent with ")
print("Theta: ", theta_rmsvgd.ravel()[:5])
print("Cost: ", J_history_rmsvgd[-1])
print("Training accuracy: {:.2f}%".format(rms_accuracy[-1] * 100))
print("Running time: ", rms_time_taken, "seconds")

# # Run Vanilla Gradient Descent with AdaGrad Optimizer and record running time
(
    theta_ada_grad_vgd,
    J_history_ada_grad_vgd,
    ada_grad_accuracy,
    ada_grad_time_taken,
) = rmsprop_vanilla_gradient_descent(X, y, theta, alpha, num_iters)
print("Vanilla Gradient Descent with ")
print("Theta: ", theta_ada_grad_vgd.ravel()[:5])
print("Cost: ", J_history_ada_grad_vgd[-1])
print("Training accuracy: {:.2f}%".format(ada_grad_accuracy[-1] * 100))
print("Running time: ", ada_grad_time_taken, "seconds")
