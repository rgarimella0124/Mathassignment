import numpy as np
from sklearn.utils.extmath import randomized_svd
import time

m = 50000
n = 150
X = np.random.rand(m, n)
y = np.random.rand(m, 1)

# centering the data
X_centered = X - np.mean(X, axis=0)
print("X_centered:")
print(X_centered)

# computing the randomized singular value decomposition (SVD)
start_time = time.time()
U, s, Vt = randomized_svd(X_centered, n_components=10)
print("U:")
print(U)
print("s:")
print(s)
print("Vt:")
print(Vt)


end_time = time.time()
print("Time taken for computing SVD:", end_time - start_time, "seconds")

# projecting the data onto the new basis
start_time = time.time()
X_pca = X_centered.dot(Vt.T)
print("X_pca:")
print(X_pca)

end_time = time.time()
print("Time taken for projection:", end_time - start_time, "seconds")
