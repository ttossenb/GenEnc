import math
import numpy as np
from numpy.random import multivariate_normal

mean = np.array([[0, 0], [100, 0], [0, 100]])
cov = ([[[1, 0], [0, 1]], [[1, 0], [0, 2]], [[3 / 2, -1 / 2], [-1 / 2, 3 / 2]]])

K = 3
D = 2


n = 100000  #how many predictions of each component we will display

#generate randoms
randoms = np.zeros([K, n, D])
for k in range(K):
    randoms[k] = multivariate_normal(mean[k].flatten(), cov[k], n)
randoms = randoms.reshape(K * n, D)
np.random.shuffle(randoms)

np.save("test_points", randoms)
