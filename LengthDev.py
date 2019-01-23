import numpy as np


A=np.load('LatentPoints.npy', mmap_mode='r')
N=A.shape[0]
D=A.shape[1]

#inic
mu_c=np.zeros(D)
S=0
n=0

while n<N :
    #read a row
    x=A[n]

    #online algorithm
    n=n+1
    mu_p=mu_c
    mu_c=mu_p+((x-mu_p)/n)
    S=S+np.inner(x-mu_c, x-mu_p)

sigmasq=S/N
print(sigmasq)
