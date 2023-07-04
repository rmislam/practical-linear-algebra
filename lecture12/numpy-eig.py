import numpy as np

A = np.array([
    [2, 4, 6],
    [8, -1, 4],
    [-13, 4, 0]
    ])

eigVals, eigVecs = np.linalg.eig(A)
print("Eigenvalues of 3 x 3 example matrix:\n", eigVals, "\n")
print("Eigenvectors of 3 x 3 example matrix:\n", eigVecs, "\n")

identityMatrix = np.eye(5)
eigVals, eigVecs = np.linalg.eig(identityMatrix)
print("Eigenvalues of 5 x 5 identity matrix:\n", eigVals, "\n")
print("Eigenvectors of 5 x 5 identity matrix:\n", eigVecs, "\n")
