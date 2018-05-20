'''
Shibo Yao
time: Feb 19, 2018
MGMT 782 Lecture 5 Clustering 
Kmeans++
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
import scipy
from scipy.cluster import hierarchy



def gen_data(n, rho, mu):
    sigx = 0.2
    sigy = 0.5
    cov = np.array([[sigx*sigx, sigx*sigy*rho], [sigx*sigy*rho, sigy*sigy]])
    data = np.random.multivariate_normal(mu, cov, size = n)

    return data


def kpp_init(X, K):
    C = [X[0]]
    for k in range(1, K):
        D2 = scipy.array([min([scipy.inner(c-x,c-x) for c in C]) for x in X])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = scipy.rand()
        for j, p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C



def kmeansPP(k, data):
    ctrs = kpp_init(data, k)
    centroids = ctrs

    maxi = 1000
    i = 0
    alpha = 0.0001
    while i < maxi:
        i += 1
        
        ctrsprior = centroids
        clusters = [[] for _ in range(k)]

        for pt in data:
            norm_to_center = [norm(pt-ctr) for ctr in centroids]
            minval = min(norm_to_center)
            minind = norm_to_center.index(minval)
            clusters[minind].append(pt)

        centroids = [np.mean(lst, axis = 0) for lst in clusters]
        print(i, centroids)
        print(norm(np.array(centroids) - np.array(ctrsprior)))
        if norm(np.array(centroids) - np.array(ctrsprior)) < alpha:
            break

    return np.array(centroids)


def dendro_plot(data):
    clust = hierarchy.linkage(data, 'average')
    plt.figure()
    hierarchy.dendrogram(clust)
    



if __name__ == "__main__":
    n = 25
    center1 = np.array([1.,1.])
    data1 = gen_data(n, 0.6, center1)

    center2 = np.array([-1,0.5])
    data2 = gen_data(n, -0.8, center2)

    center3 = np.array([0,-1])
    data3 = gen_data(n, 0.2, center3)

    data = np.concatenate((data1, data2, data3))
    result = kmeansPP(3, data)
    plt.figure()
    '''
    plt.scatter(data1[:,0], data1[:,1], c = 'green')
    plt.scatter(data2[:,0], data2[:,1], c = 'red')
    plt.scatter(data3[:,0], data3[:,1], c = 'blue')
    '''
    plt.plot(data[:,0], data[:,1], 'kx')
    plt.scatter(result[:,0], result[:,1], c = 'red')

    dendro_plot(data)
    plt.show()




