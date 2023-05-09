import numpy as np
from sklearn.utils.extmath import randomized_svd
import time
import matplotlib.pyplot as plt

# Matrix from Problem 1 containing 50000 rows and 150 columns and generating random data matrix and target vector
m = 50000  
n = 150  
X = np.random.rand(m, n) 
y = np.random.rand(m, 1)  

# center the data by subtracting the mean of each column
X_centered = X - np.mean(X, axis=0)  

# print the centered data matrix
print("X_centered:")
print(X_centered)

# compute the randomized singular value decomposition (SVD)
start_time = time.time()
# use only the top 10 singular values/vectors
U, s, Vt = randomized_svd(X_centered, n_components=10)  
end_time = time.time()

# print the left singular vectors, singular values, and right singular vectors
print("U:")
print(U)  
print("s:")
print(s)  
print("Vt:")
print(Vt)  

# print the time taken to compute the SVD
print("Time taken for computing SVD:", end_time - start_time, "seconds")

# project the data onto the new basis defined by the right singular vectors
start_time = time.time()
X_pca = X_centered.dot(Vt.T)  
end_time = time.time()

# print the projected data
print("X_pca:")
print(X_pca)

# print the time taken to project the data
print("Time taken for projection:", end_time - start_time, "seconds")

# assuming X_pca contains the projected data obtained using randomized SVD
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()