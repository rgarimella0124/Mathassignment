import numpy as np
import time
import matplotlib.pyplot as plt

# Generate random data
m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1)

# centering the data
start = time.time()

X_centered = X - np.mean(X, axis=0)

centering_time = time.time() - start
print("X_centered:")
print(X_centered)
print(f"Time taken for centering: {centering_time:.5f} seconds")

# computing the covariance matrix
start = time.time()

cov_matrix = np.cov(X_centered.T)

covariance_time = time.time() - start
print("covariance matrix:")
print(cov_matrix)
print(f"Time taken for covariance matrix computation: {covariance_time:.5f} seconds")

# finding the eigenvalues and eigenvectors
start = time.time()

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

eigenvectors_time = time.time() - start
print("eigenvalues:")
print(eigenvalues)
print("eigenvectors:")
print(eigenvectors)
print(f"Time taken for eigenvectors computation: {eigenvectors_time:.5f} seconds")


# sorting the eigenvectors in decreasing order of eigenvalues
start = time.time()

sorted_indices = eigenvalues.argsort()[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

sorting_time = time.time() - start

print("sorted eigenvectors:")
print(sorted_eigenvectors)
print(f"Time taken for sorting eigenvectors: {sorting_time:.5f} seconds")

# Projecting the centered data onto the sorted eigenvectors to obtain the principal components.
start = time.time()

X_pca = X_centered.dot(sorted_eigenvectors)

projection_time = time.time() - start

print("X_pca:")
print(X_pca)
print(f"Time taken for projection: {projection_time:.5f} seconds")


# Compute the explained variance ratio
explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cumulative_explained_variances = np.cumsum(explained_variances)

# Plot the explained variance ratio
plt.plot(range(1, n + 1), explained_variances, "-o", label="Explained variance")
plt.plot(
    range(1, n + 1),
    cumulative_explained_variances,
    "-s",
    label="Cumulative explained variance",
)
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("Explained variance ratio of the principal components")
plt.legend(loc="best")
plt.show()
