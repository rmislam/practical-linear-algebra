import numpy as np

# generate a random matrix
A = np.random.randint(low=-99, high=99, size=(4, 3))
print('A:\n', A, '\n')

# compute singular value decomposition
U, S, V = np.linalg.svd(A)

# S is list of singular values, so use np.diag() to create matrix
SMat = np.diag(S)

# round for ease of reading
print('U:\n', np.round(U, 1), '\n')
print('S:\n', np.round(SMat, 1), '\n')
print('V^T:\n', np.round(V.transpose(), 1), '\n')
