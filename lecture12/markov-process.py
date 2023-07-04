import numpy as np

# number of transitions
n = 10000

# stochastic matrix
A = np.array([
    [0.3, 0.2, 0.7],
    [0.3, 0.5, 0.1],
    [0.4, 0.3, 0.2]
    ])

# initial state vector
x0 = np.array([0, 1, 0])

# calculate xn (final state vector) with simple matrix multiplication
xn = x0

# relatively slow and tedious
for step in range(n):
    xn = A.dot(xn)

print("x0:\n", x0, "\n")
print("x1:\n", A.dot(x0), "\n")
print("final state vector (xn):\n", xn, "\n")

# calculate xn (final state vector) via spectral theorem shortcut
_, eigVecs = np.linalg.eig(A)
diagonalizedMatrix = np.linalg.inv(eigVecs).dot(A).dot(eigVecs)
mainDiag = np.diag(diagonalizedMatrix)
B = np.eye(A.shape[0])

# this is much faster!
for i in range(B.shape[0]):
    B[i, i] = mainDiag[i] ** n

finalStochasticMatrix = eigVecs.dot(B).dot(np.linalg.inv(eigVecs))

print("final stochastic matrix:\n", finalStochasticMatrix, "\n")
print("final state vector (xn) again:\n", finalStochasticMatrix.dot(x0), "\n")

