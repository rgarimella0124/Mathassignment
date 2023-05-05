import numpy as np
import time
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout

def adam_stochastic_gradient_descent(X, y, theta, alpha, num_iters):
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
    for i in range(num_iters):
        rand_index = np.random.randint(0, len(y))
        X_j = X[rand_index, :].reshape(1, X.shape[1])
        y_j = y[rand_index].reshape(1, 1)
        history = model.train_on_batch(X_j, y_j)
        J_history.append(history[0])
    end_time = time.time()
    accuracy = model.evaluate(X, y, verbose=0)[1]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


def rmsprop_stochastic_gradient_descent(X, y, theta, alpha, num_iters):
    optimizer = RMSprop(learning_rate=alpha, rho=0.9, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    start_time = time.time()
    J_history = []
    for i in range(num_iters):
        rand_index = np.random.randint(0, len(y))
        X_j = X[rand_index, :].reshape(1, X.shape[1])
        y_j = y[rand_index].reshape(1, 1)
        history = model.train_on_batch(X_j, y_j)
        J_history.append(history[0])
    end_time = time.time()
    accuracy = model.evaluate(X, y, verbose=0)[1]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


def adagrad_stochastic_gradient_descent(X, y, theta, alpha, num_iters):
    optimizer = Adagrad(learning_rate=alpha, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    start_time = time.time()
    J_history = []
    for i in range(num_iters):
        rand_index = np.random.randint(0, len(y))
        X_j = X[rand_index, :].reshape(1, X.shape[1])
        y_j = y[rand_index].reshape(1, 1)
        history = model.train_on_batch(X_j, y_j)
        J_history.append(history[0])
    end_time = time.time()
    accuracy = model.evaluate(X, y, verbose=0)[1]
    return model.get_weights()[0], J_history, accuracy, end_time - start_time


# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1)

# Set hyperparameters
theta = np.zeros((n, 1))
alpha = 0.01
num_iters = 100

# # Run Stochastic Gradient Descent with Adam Optimizer and record running time
(
    theta_avgd,
    J_history_avgd,
    adam_accuracy,
    adam_time_taken,
) = adam_stochastic_gradient_descent(X, y, theta, alpha, num_iters)
print("Stochastic Gradient Descent with Adam Optimizer ")
print("Theta: ", theta_avgd.ravel()[:5])
print("Cost: ", J_history_avgd[-1])
# print("Training accuracy: {:.2f}%".format(adam_accuracy[-1] * 100))
print("Accuracy: ", adam_accuracy)
print("Running time: ", adam_time_taken, "seconds")


# # # Run Stochastic Gradient Descent with RMSProp Optimizer and record running time
# (
#     theta_avgd,
#     J_history_avgd,
#     adam_accuracy,
#     adam_time_taken,
# ) = rmsprop_stochastic_gradient_descent(X, y, theta, alpha, num_iters)
# print("Stochastic Gradient Descent with RMSProp Optimizer ")
# print("Theta: ", theta_avgd.ravel()[:5])
# print("Cost: ", J_history_avgd[-1])
# print("Training accuracy: {:.2f}%".format(adam_accuracy[-1] * 100))
# print("Running time: ", adam_time_taken, "seconds")


# # # Run Stochastic Gradient Descent with ADAgrad Optimizer and record running time
# (
#     theta_avgd,
#     J_history_avgd,
#     adam_accuracy,
#     adam_time_taken,
# ) = adagrad_stochastic_gradient_descent(X, y, theta, alpha, num_iters)
# print("Stochastic Gradient Descent with ADAGrad Optimizer ")
# print("Theta: ", theta_avgd.ravel()[:5])
# print("Cost: ", J_history_avgd[-1])
# print("Training accuracy: {:.2f}%".format(adam_accuracy[-1] * 100))
# print("Running time: ", adam_time_taken, "seconds")
