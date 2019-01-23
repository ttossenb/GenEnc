import numpy as np
#from numpy.linalg import inv
#from numpy.linalg import det
#import pandas as pd
from keras import backend as K
#import tensorflow as tf
import math
import time
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


A = np.load('test_points.npy', mmap_mode='r')
#A = np.load('latent_points.npy', mmap_mode='r')
N = A.shape[0]
D = A.shape[1]

sigmasq = 2
#sigmasq = 6.65
c = 20
beta = 0.96
v_min = 100
sp_min = 40

x = A[0, :].reshape(D, 1)
C = 1
mean = x[np.newaxis, :]
#mean.shape=(1, D, 1) atm
sp = np.array([1], dtype=np.float64)
v = np.array([1], dtype=np.float64)
p = np.array([1], dtype=np.float64)
cov = c * sigmasq * (np.identity(D, dtype=np.float64)[np.newaxis, :])

g_distance = K.tf.Graph()
g_update = K.tf.Graph()

with g_distance.as_default():
    mean_ind = K.tf.placeholder(dtype=K.tf.float64)
    x_ind = K.tf.placeholder(dtype=K.tf.float64)
    cov_ind = K.tf.placeholder(dtype=K.tf.float64)

    e_d = K.reshape(x_ind, (1, D, 1)) - mean_ind
    eT_d = K.tf.transpose(e_d, perm=[0, 2, 1])
    d2M_d = K.tf.squeeze(eT_d @ K.tf.linalg.inv(cov_ind) @ e_d)

with g_update.as_default():
    d2M_inu = K.tf.placeholder(dtype=K.tf.float64)
    cov_inu = K.tf.placeholder(dtype=K.tf.float64)
    p_inu = K.tf.placeholder(dtype=K.tf.float64)
    v_inu = K.tf.placeholder(dtype=K.tf.float64)
    sp_inu = K.tf.placeholder(dtype=K.tf.float64)
    e_inu = K.tf.placeholder(dtype=K.tf.float64)
    mean_inu = K.tf.placeholder(dtype=K.tf.float64)
    C_ones_inu = K.tf.placeholder(dtype=K.tf.float64)

    pri_u = K.exp(-0.5 * d2M_inu) / (math.pow(2. * math.pi, D / 2.) * K.sqrt(K.tf.linalg.det(cov_inu)))
    post_u = (pri_u * p_inu) / K.sum(pri_u * p_inu)
    v_u = v_inu + C_ones_inu
    sp_u = sp_inu + post_u
    omega_u = K.tf.expand_dims(K.tf.expand_dims((post_u / sp_u), -1), -1)
    deltamean_u = omega_u * e_inu
    mean_u = mean_inu + deltamean_u
    estar_u = K.expand_dims(x, axis=0) - mean_u
    temp1_u = omega_u * (estar_u @ K.tf.transpose(estar_u, perm=[0, 2, 1]))
    temp2_u = deltamean_u @ K.tf.transpose(deltamean_u, perm=[0, 2, 1])
    temp3_u = K.tf.expand_dims(K.tf.expand_dims(C_ones_inu, -1), -1) - omega_u
    cov_u = (temp3_u * cov_inu) + temp1_u - temp2_u
    p_u = sp_u / K.sum(sp_u)

    # x.shape=(C, 1)
    # e.shape=(C, D, 1)
    # d2M.shape=(C,)
    # p.shape=(C,)
    # pri.shape=(C,)
    # post.shape=(C,)
    # v.shape=(C,)
    # sp.shape=(C,)
    # omega.shape=(C,)
    # deltamean.shape=(C, D, 1)
    # mean.shape=(C, D, 1)
    # cov.shape=(C, D, D)

sess_distance = K.tf.Session(graph=g_distance)
sess_update = K.tf.Session(graph=g_update)


n = 1
while n < N:
    #if n==3:
    #    print(cov)
    #print(n)
    if n % 100 == 0:
        print(n, " / ", N)

    x = A[n, :].reshape(D, 1)
    e = np.expand_dims(x, axis=0) - mean

    #start = time.time()
    #end = time.time()

    [d2M, e] = sess_distance.run([d2M_d, e_d], feed_dict={mean_ind:mean, x_ind:x, cov_ind:cov})

    if stats.chisqprob(np.amin(d2M, axis=0), D) > 1 - beta:
        C_ones=np.ones((C,), dtype=np.float64)
        if n==2:
            prit = K.exp(-0.5 * d2M) / (math.pow(2. * math.pi, D / 2.) * K.sqrt(K.tf.linalg.det(cov)))
            postt = (prit * p) / K.sum(prit * p)
            vt=v + C_ones
            spt = sp + postt
            omegat = K.tf.expand_dims(K.tf.expand_dims((postt / spt), -1), -1)
            deltameant = omegat * e
            meant = mean + deltameant
            estart = K.expand_dims(x, axis=0) - meant
            temp1t = omegat * (estart @ K.tf.transpose(estart, perm=[0, 2, 1]))
            temp2t = deltameant @ K.tf.transpose(deltameant, perm=[0, 2, 1])
            temp3t = K.tf.expand_dims(K.tf.expand_dims(C_ones, -1), -1) - omegat
            covt = (temp3t * cov) + temp1t - temp2t
            pt = spt / K.sum(spt)
            print(K.eval(covt))
        [mean, sp, v, p, cov] = sess_update.run([mean_u, sp_u, v_u, p_u, cov_u], feed_dict={d2M_inu:d2M, cov_inu:cov, p_inu:p, v_inu:v, sp_inu:sp, e_inu:e, mean_inu:mean, C_ones_inu:C_ones})
        if n==2:
            print(cov)
    else:
        if n < N / 100:
            C = C + 1

            mean = np.append(mean, x[np.newaxis, :], axis=0)
            sp = np.append(sp, np.array([1]), axis=0)
            v = np.append(v, np.array([1]), axis=0)
            p = np.append(p, np.array([1 / np.sum(sp)]), axis=0)
            cov = np.append(cov, c * sigmasq * (np.identity(D)[np.newaxis, :]), axis=0)

            print("New component created. Number of components:", C)
    if len(np.nonzero(v == v_min)[0]) != 0:
        # print(np.nonzero(v == v_min))
        j_elim = np.nonzero(v == v_min)[0][0]
        print("Eliminating ", j_elim, "-th component.")

        if sp[j_elim] < sp_min:
            v = np.delete(v, j_elim, axis=0)
            e = np.delete(e, j_elim, axis=0)
            d2M = np.delete(d2M, j_elim, axis=0)
            sp = np.delete(sp, j_elim, axis=0)
            p = sp / np.sum(sp)
            # pri = np.exp(-0.5 * d2M) / (np.power(2. * np.pi, D / 2.) * np.sqrt(det(cov)))
            # post = (pri * p) / np.sum(pri * p)
            mean = np.delete(mean, j_elim, axis=0)
            # estar = np.delete(estar, j_elim, axis=0)
            cov = np.delete(cov, j_elim, axis=0)

            C = C - 1
    n = n + 1
    #print(end - start)

print("Total number of components: ", C)

np.save("cov", cov)
np.save("mean", mean)

print(mean)
print(cov)
