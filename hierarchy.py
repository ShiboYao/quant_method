'''
Shibo Yao
time: Feb 19, 2018
MGMT 782 Lecture 5 Clustering 
Hierarchical Clustering 
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

'''
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
'''

def dendro_plot(data):
    clust = hierarchy.linkage(data, 'average')
    plt.figure()
    hierarchy.dendrogram(clust)
    ### Need to write a script for hierarchical clustering from scratch. 



if __name__ == "__main__":
    n = 10

    d1 = 1.5
    center11 = np.array([3,3])
    data11 = gen_data(n, 0.3, center11)
    center12 = np.array([3-d1,3+d1])
    data12 = gen_data(n, 0.3, center12)
    center13 = np.array([3+d1,3+d1])
    data13 = gen_data(n, 0.3, center13)
    data1 = np.concatenate((data11, data12, data13))

    d2 = 1.5
    center21 = np.array([-3,3])
    data21 = gen_data(n, -0.3, center21)
    center22 = np.array([-3-d2,3+d2])
    data22 = gen_data(n, -0.3, center22)
    center23 = np.array([-3-d2,3-d2])
    data23 = gen_data(n, -0.3, center23)
    center24 = np.array([-3+d2,3+d2])
    data24 = gen_data(n, -0.3, center24)
    data2 = np.concatenate((data21, data22, data23, data24))

    d3 = 1.5
    center31 = np.array([0,-3])
    data31 = gen_data(n, 0.2, center31)
    center32 = np.array([0+d2,-3+d2])
    data32 = gen_data(n, 0.2, center32)
    center33 = np.array([0+d2,-3-d2])
    data33 = gen_data(n, 0.2, center33)
    center34 = np.array([0-d2,-3+d2])
    data34 = gen_data(n, 0.2, center34)
    center35 = np.array([0-d2,-3-d2])
    data35 = gen_data(n, 0.2, center35)
    data3 = np.concatenate((data31, data32, data33, data34, data35))

    data = np.concatenate((data1, data2, data3))


    plt.scatter(data1[:,0], data1[:,1], c = 'green')
    plt.scatter(data2[:,0], data2[:,1], c = 'red')
    plt.scatter(data3[:,0], data3[:,1], c = 'blue')
    '''
    result = kmeansPP(3, data)
    plt.figure()

    plt.plot(data[:,0], data[:,1], 'kx')
    plt.scatter(result[:,0], result[:,1], c = 'red')
    '''
    dendro_plot(data)
    plt.show()




