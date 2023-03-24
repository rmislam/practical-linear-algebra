import numpy as np

m = 5  # number of rows of A
n = 3  # number of columns of A

# Thin QR
A = np.random.rand(m, n)
Q, R = np.linalg.qr(A)
print('Thin QR:')
print('\nA:')
print(np.round(A, 2))
print('\nQ:')
print(np.round(Q, 2))
print('\nR:')
print(np.round(R, 2))

# Full QR
A_extended = np.hstack((A, np.eye(m)))
Q, R = np.linalg.qr(A_extended)
print('\n\nFull QR:')
print('\nA_extended:')
print(np.round(A_extended, 2))
print('\nQ:')
print(np.round(Q, 2))
print('\nR:')
print(np.round(R, 2))
