import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pandas as pd
from scipy import stats
stats.chisqprob=lambda chisq, df: stats.chi2.sf(chisq, df)


A=np.load('LatentPoints.npy', mmap_mode='r')
N=A.shape[0]
D=A.shape[1]

#TODO sigmasq=...
c=0.01
beta=0.1

x=A[0, :].reshape(D, 1)
K=1
mean=x[np.newaxis, :]
#mean.shape=(1, D, 1) atm
sp=np.array([1])
v=np.array([1])
p=np.array([1])
cov=c*sigmasq*(np.identity(D)[np.newaxis, :])

def create():
    global K
    global mean
    global sp
    global v
    global p
    global cov

    K=K+1
    mean=np.append(mean, x[np.newaxis, :], axis=0)
    sp=np.append(sp, np.array([1]), axis=0)
    v=np.append(v, np.array([1]), axis=0)
    p=np.append(p, np.array([1/np.sum(sp)]), axis=0)
    cov=np.append(cov, c*(np.identity(D)[np.newaxis, :]), axis=0)

def update():
    global pri
    global post
    global v
    global sp
    global omega
    global deltamean
    global mean
    global e
    global cov

    pri=np.exp(-0.5*d2M)/(np.power(2.*np.pi, D/2.)*np.sqrt(det(cov)))
    post=(pri*p)/np.sum(pri*p)
    v=v+np.ones(K)
    sp=sp+post
    omega=post/sp
    deltamean=omega.reshape(K, 1, 1)*e
    mean=mean+deltamean
    estar=x-mean
    cov=((np.ones(K)-omega).reshape(K, 1, 1)*cov)+(omega.reshape(K, 1, 1)*(estar@np.transpose(estar, (0, 2, 1))))-(deltamean@np.transpose(deltamean, (0, 2, 1)))

    #x.shape=(D, 1)
    #e.shape=(K, D, 1)
    #d2M.shape=(K,)
    #p.shape=(K,)
    #pri.shape=(K,)
    #post.shape=(K,)
    #v.shape=(K,)
    #sp.shape=(K,)
    #omega.shape=(K,)
    #deltamean.shape=(K, D, 1)
    #mean.shape=(K, D, 1)
    #cov.shape=(K, D, D)

n=1
while n<N :
    x=A[n, :].reshape(D, 1)
    e=x-mean
    d2M=(np.transpose(e, (0, 2, 1))@inv(cov)@e).reshape(K,)
    if stats.chisqprob(np.amin(d2M, axis=0), D)>1-beta :
        update()
    else:
        create()

