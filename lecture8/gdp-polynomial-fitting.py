from numpy import array
from numpy.linalg import lstsq

# Time is measured in years since 1970
# World GDP is measured in trillions of USD

y = array([3.4,   # 1970
           11.3,  # 1980
           22.8,  # 1990
           33.8,  # 2000
           66.6,  # 2010
           84.9]) # 2020

# Vandermonde matrix
A = array([[1, 0, 0, 0],
           [1, 10, 10 ** 2, 10 ** 3],
           [1, 20, 20 ** 2, 20 ** 3],
           [1, 30, 30 ** 2, 30 ** 3],
           [1, 40, 40 ** 2, 40 ** 3],
           [1, 50, 50 ** 2, 50 ** 3]])

x = lstsq(A, y, rcond=None)[0]
print('Least squares estimate of x:', x)
print('Difference between true and estimated y:', y - A.dot(x))

y_2030 = array([1, 60, 60 ** 2, 60 ** 3]).dot(x)  # 2030 - 1970 = 60
print('Predicted World GDP in 2030 in trillions of USD:', y_2030)
