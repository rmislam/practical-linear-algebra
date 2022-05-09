import numpy as np
from sklearn import neighbors

## Playing with numpy
a = 3  # scalar
x = np.array([4, 5, 6])  # vector
y = np.array([7, 8, 9])

x_scaled = a * x   # scalar-vector product
sum_xy = x + y     # sum of two vectors

prod_xy = x * y    # elementwise product of two vectors
x_dot_y = x.dot(y) # dot product of two vectors

norm_x = np.linalg.norm(x)  # norm of vector
norm_y = np.linalg.norm(y)

dist_yx = np.linalg.norm(y - x)  # norm of vector difference (distance between two points)
dist_xy = np.linalg.norm(x - y)

## Movie recommendation example
possible_genres = ["horror", "action", "comedy", "romance", "drama"]

# Ratings of movies by each user (input features)
emma_x = np.array([1, 3, 5, 3, 4, 2])
alex_x = np.array([5, 1, 1, 2, 5, 2])
kate_x = np.array([2, 2, 3, 3, 2, 2])
carl_x = np.array([1, 1, 4, 5, 5, 2])
lily_x = np.array([3, 3, 2, 4, 1, 2])
sean_x = np.array([4, 3, 4, 2, 1, 2])

# Each user's favorite genre (feature to predict)
emma_y = "action"
alex_y = "horror"
kate_y = "comedy"
carl_y = "action"
lily_y = "comedy"
sean_y = ""  # try to predict sean's favorite genre!

# KNN is as simple as this
dist_emma_to_sean = np.linalg.norm(emma_x - sean_x)

# KNN with sklearn
k = 1  # try different values of k
clf = neighbors.KNeighborsClassifier(k)

data_x = np.array([emma_x, alex_x, kate_x, carl_x, lily_x])
data_y = np.array([emma_y, alex_y, kate_y, carl_y, lily_y])

clf.fit(data_x, data_y)

sean_y = clf.predict(np.array([sean_x]))[0]
