from numpy import array
from numpy.linalg import lstsq

# All values in y and A are in units of degrees Celsius

y = array([20.3,
           25.6,
           27.2,
           21.1,
           17.8,
           22.9])

A = array([[18.7, 19.9, 20.6, 21.4],
           [21.8, 23.0, 26.8, 27.4],
           [24.5, 26.6, 28.0, 29.1],
           [18.2, 18.5, 21.1, 23.4],
           [14.3, 16.9, 17.9, 20.1],
           [22.1, 22.3, 24.4, 25.1]])

x = lstsq(A, y, rcond=None)[0]
y_diff = y - A.dot(x)

print('Least squares estimate of x:', x)
print('Difference between true and estimated y:', y_diff)
