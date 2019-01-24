import numpy as np


treshold = 0.3 * 1.56

mean = np.load('mean.npy')
cov = np.load('cov.npy')
C = mean.shape[0]
D = mean.shape[1]

uj = True
redmean = mean[0].reshape((1, D, 1))
index_list = np.array([0])

for i in range(1, C):
    R = redmean.shape[0]
    uj = True
    for j in range(R):
        #print("i: ", i)
        #print("j: ", j)
        #print(np.linalg.norm(mean[i].reshape((D,)) - redmean[j].reshape((D,))))
        if np.linalg.norm(mean[i].reshape((D,)) - redmean[j].reshape((D,))) < treshold:
            uj = False
    if uj:
        redmean = np.append(redmean, mean[i].reshape((1, D, 1)), axis=0)
        index_list = np.append(index_list, np.array([i]), axis=0)

#print(mean)
#print(R)
#print(redmean)
print(index_list)
np.save("red_mean", mean[index_list])
np.save("red_cov", cov[index_list])
