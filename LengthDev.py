import numpy as np


#this program calculates the squared (L2) deviation of a point set

#A=np.load('latent_points.npy', mmap_mode='r')
A=np.load('latent_points_network_1m.npy', mmap_mode='r')
#number of points
N=A.shape[0]
#latent dim
D=A.shape[1]

#print(N)
#print(D)

#inic
mu_c=np.zeros(D)
S=0
n=0

while n<N:
    #read a row
    x=A[n]

    #online algorithm
    n=n+1
    mu_p=mu_c
    mu_c=mu_p+((x-mu_p)/n)
    S=S+np.inner(x-mu_c, x-mu_p)

sigmasq=S/N
print(sigmasq)
