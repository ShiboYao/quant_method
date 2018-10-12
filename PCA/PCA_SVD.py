'''
Shibo Yao
time: Feb 27, 2018
MGMT782 lecture7 PCA, SVD
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


def gen_data(n, rho):
    mean = np.array([0., 0.])
    sigx = 0.2
    sigy = 0.8
    cov = np.array([[sigx*sigx, sigx*sigy*rho], [sigx*sigy*rho, sigy*sigy]])

    data = np.random.multivariate_normal(mean, cov, size = n)

    return data


if __name__ == '__main__':
    n = 1000
    rho = 0.85
    data = gen_data(n, rho)

    u, s, vh = la.svd(data, full_matrices = False)
    princomps = np.dot(u, np.diag(s))
    #print(u.shape)
    print('Singular values:\n',s, '\n')
    print('Right singular vectors:\n', vh, '\n')

    PP = 300
    plt.figure()
    plt.plot(data[:PP,0], data[:PP,1], 'x')

    ax = plt.axes()
    ax.arrow(0, 0, vh[0,0], vh[0,1], head_width = 0.05, head_length = 0.1, fc = 'k', ec = 'k')
    ax.arrow(0, 0, vh[1,0], vh[1,1], head_width = 0.05, head_length = 0.1, fc = 'k', ec = 'k')
    plt.axis('equal')

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(princomps[:,0], bins = 50)
    plt.subplot(1,2,2)
    plt.hist(princomps[:,1], bins = 50)
    plt.show()




