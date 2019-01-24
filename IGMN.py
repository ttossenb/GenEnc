import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pandas as pd
from keras import backend as K
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


#A = np.load('test_points.npy', mmap_mode='r')
A = np.load('latent_points.npy', mmap_mode='r')
#np.random.shuffle(A)
N = A.shape[0]
D = A.shape[1]

#sigmasq = 2
sigmasq = 6.65
c = 0.8
beta = 0.05
v_min = 101
sp_min = 8

x = A[0, :].reshape(D, 1)
C = 1
mean = x[np.newaxis, :]
#mean.shape=(1, D, 1) atm
sp = np.array([1])
v = np.array([1])
p = np.array([1])
cov = c * sigmasq * (np.identity(D)[np.newaxis, :])


def create():
    global C
    global mean
    global sp
    global v
    global p
    global cov

    C = C+1
    mean = np.append(mean, x[np.newaxis, :], axis=0)
    sp = np.append(sp, np.array([1]), axis=0)
    v = np.append(v, np.array([1]), axis=0)
    p = np.append(p, np.array([1 / np.sum(sp)]), axis=0)
    cov = np.append(cov, c * sigmasq * (np.identity(D)[np.newaxis, :]), axis=0)

    print("New component created. Number of components:", C)


def update():
    global p
    global pri
    global post
    global v
    global sp
    global omega
    global deltamean
    global mean
    global e
    global cov

    #print(det(cov))
    #print(cov)
    #print(d2M)
    pri = np.exp(-0.5 * d2M) / (np.power(2. * np.pi, D / 2.) * np.sqrt(det(cov)))
    post = (pri * p)/np.sum(pri * p)
    v = v + np.ones(C)
    sp = sp + post
    omega = post / sp
    deltamean = omega.reshape(C, 1, 1) * e
    mean = mean + deltamean
    estar = x - mean
    #print("elso")
    cov = ((np.ones(C)-omega).reshape(C, 1, 1) * cov) + (omega.reshape(C, 1, 1) * (estar@np.transpose(estar, (0, 2, 1)))) - (deltamean @ np.transpose(deltamean, (0, 2, 1)))
    if n==2:
        print(cov)
    #print("cov: ", cov)
    p = sp / np.sum(sp)

    #x.shape=(D, 1)
    #e.shape=(C, D, 1)
    #d2M.shape=(C,)
    #p.shape=(C,)
    #pri.shape=(C,)
    #post.shape=(C,)
    #v.shape=(C,)
    #sp.shape=(C,)
    #omega.shape=(C,)
    #deltamean.shape=(C, D, 1)
    #mean.shape=(C, D, 1)
    #cov.shape=(C, D, D)


def eliminateRedundants():
    global v
    global v_min
    global sp
    global sp_min
    global e
    global d2M
    global sp
    global pri
    global post
    global omega
    global deltamean
    global mean
    global estar
    global cov
    global p
    global C

    if len(np.nonzero(v == v_min)[0]) != 0:
        #print(np.nonzero(v == v_min))
        j_elim = np.nonzero(v == v_min)[0][0]
        print("Eliminating ", j_elim, "-th component.")

        if sp[j_elim] < sp_min:
            v=np.delete(v, j_elim, axis=0)
            e=np.delete(e, j_elim, axis=0)
            d2M=np.delete(d2M, j_elim, axis=0)
            sp = np.delete(sp, j_elim, axis=0)
            p = sp / np.sum(sp)
            #pri = np.exp(-0.5 * d2M) / (np.power(2. * np.pi, D / 2.) * np.sqrt(det(cov)))
            #post = (pri * p) / np.sum(pri * p)
            omega = np.delete(omega, j_elim, axis=0)
            deltamean = np.delete(deltamean, j_elim, axis=0)
            mean = np.delete(mean, j_elim, axis=0)
            #estar = np.delete(estar, j_elim, axis=0)
            cov = np.delete(cov, j_elim, axis=0)
            C = C - 1



n = 1
while n < N:
    #if n==3:
    #    print(cov)
    if n % 100 == 0:
        print(n, " / ", N)
    #print("n: ", n)
    x = A[n, :].reshape(D, 1)
    #print("x: ", x)
    e = x - mean
    #print(mean)
    #print(cov)
    #print("n: ", n)
    #print("e: ", e)
    #print(inv(cov))
    d2M = (np.transpose(e, (0, 2, 1)) @ inv(cov) @ e).reshape(C,)
    #print("d2M: ", d2M)
    #print("prob: ", stats.chisqprob(np.amin(d2M, axis=0), D))
    if stats.chisqprob(np.amin(d2M, axis=0), D) > 1 - beta:
        update()
    else:
        if n < N / 2:
            create()
    eliminateRedundants()
    n = n + 1

print("Total number of components: ", C)

#maxdist=0.05
#for i in range(0, C):
#    for j in range(i, C):
#        if j

#print("Total number of components after eliminating umbrellas: ", C)

np.save("cov", cov)
np.save("mean", mean)

print(mean)
print(cov)
