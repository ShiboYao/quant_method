'''
Shibo Yao
Time: Feb. 20, 2018
MGMT782 Lecture 6 Mixed Model
multivariate Gaussian estimation using EM algorithm
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import bivariate_normal
import time
start_time = time.time()




def samp_norm_mix(lam, mu1, mu2, cov1, cov2, n):
    w1 = lam
    w2 = 1 - lam
    model_pts = np.random.multinomial(n, [w1, w2])
    mod1data = np.random.multivariate_normal(mu1, cov1, n)
    mod2data = np.random.multivariate_normal(mu2, cov2, n)
    data = np.append(mod1data, mod2data, axis = 0)

    return data


def multi_gau_pdf(x, mu, cov):
    size = len(x)
    if size == len(mu) and (size, size) == cov.shape:
        det = np.linalg.det(cov)
        if det == 0:
            raise NameError("The covariance matrix can't be singular!")

        norm_const = 1.0/(math.pow((2*np.pi), float(size)/2) * math.pow(det,1.0/2))
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(cov)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    else:
        raise NameError("The dimensions of the input don't match")

    return norm_const * result
    

def mix_modl(x, mu1, mu2, cov1, cov2, lam):
    pdf1 = multi_gau_pdf(x, mu1, cov1)
    pdf2 = multi_gau_pdf(x, mu2, cov2)
    return lam*pdf1 + (1-lam)*pdf2


def minus_ll(x, mu1, mu2, cov1, cov2, lam):
    mix_pdf = mix_modl(x, mu1, mu2, cov1, cov2, lam)
    ll = np.sum(np.log(mix_pdf))
    return -ll


def error_fun(params, old):
    err = 0
    dim = len(params)
    for i in range(dim):
        err += np.sum((params[i] - old[i])**2)

    return err
        

def mu_update(w, x):
    mu = []
    for i in range(len(w)):
        mu_i = [sum(w[i]*x[:,j])/sum(w[i]) for j in range(x.shape[1])]
        mu.append(np.array(mu_i))

    return mu


def cov_update(w, mu, x):
    cov = []
    for j in range(len(w)):
        cov_j = np.zeros([x.shape[1], x.shape[1]])
        for i in range(x.shape[0]):
            ys = np.reshape(x[i] - mu[j], (2,1))
            cov_j += w[j][i] * np.dot(ys, ys.T)
        cov_j /= sum(w[j])
        cov.append(np.array(cov_j))

    return cov


def em_alg(data):
    lam0 = 0.5
    lam1 = 1 - lam0
    mu10 = np.mean(data, axis = 0)
    cov10 = np.cov(data.T)
    mu20 = mu10 * 0.9
    cov20 = cov10 * 0.9
    dim = data.shape[1]

    MAXITR = 5000
    itr = 0
    params = [mu10, mu20, cov10, cov20]
    oldparams = [np.zeros(mu10.shape), np.zeros(mu20.shape), np.zeros(cov10.shape), np.zeros(cov20.shape)]
    err = 999
    while err > 1e-4 and itr < MAXITR:
        itr += 1
        err = error_fun(params, oldparams)

        ## Compute wij from the initial guess. 
        muvec = [params[0], params[1]]
        covvec = [params[2], params[3]]
        wgtvec = [lam0, lam1]
        print("EM iteration:", itr)
        print("Mean vectors:\n", muvec)
        print("Covariance matrices:\n", covvec[0], '\n', covvec[1])
        print("Weights: ", wgtvec)
        print("Error: ", err)

        wi0num = [wgtvec[0]*multi_gau_pdf(xi, muvec[0], covvec[0]) for xi in data]
        wi1num = [wgtvec[1]*multi_gau_pdf(xi, muvec[1], covvec[1]) for xi in data]
    
        denom = [sum([wgtvec[k]*multi_gau_pdf(xi, muvec[k], covvec[k]) for k in range(len(wgtvec))]) for xi in data]

        wi0 = np.array(wi0num)/np.array(denom)
        wi1 = np.array(wi1num)/np.array(denom)
        w = [wi0, wi1]

        ## Compute updated lambda i. 
        lam0 = np.mean(wi0)
        lam1 = np.mean(wi1)

        ## Update mu and cov. 
        muvec = mu_update(w, data)
        covvec = cov_update(w, muvec, data)

        oldparams = params
        params = [muvec[0], muvec[1], covvec[0], covvec[1]]
    print("Success.")

    return params, lam0

'''
def visual_gau(dat, param):
'''
if __name__ == "__main__":
    mu1 = [69.1, 159.1] ## Height and weight. 
    mu2 = [62.7, 130.7]
    cov1 = [[7.2, 10.1], [10.1, 68.1]]
    cov2 = [[5.7, 8.9], [8.9, 54.5]]
    lam = 0.7
    n = 1000 

    data = samp_norm_mix(lam, mu1, mu2, cov1, cov2, n)
    #params = [np.array(mu1), np.array(mu2), np.array(cov1), np.array(cov2)]
    params, lam = em_alg(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    x = data[:,0]
    y = data[:,1]
    
    ## Visualize original data. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    hist, xedges, yedges = np.histogram2d(x, y, bins = 40, range=[[40, 90], [100, 190]])

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    plt.show()
    
    ## Visualize estimated distribution. 
    # The first Multivariate_Gaussian. 
    x1 = np.linspace(40, 90, 1000)
    y1 = np.linspace(100, 190, 1000)

    mu_x1 = params[0][0]
    mu_y1 = params[0][1]
    sig_x1 = params[2][0,0]**0.5
    sig_y1 = params[2][1,1]**0.5
    cov_xy1 = params[2][0,1]

    X1, Y1 = np.meshgrid(x1, y1)
    Z1 = bivariate_normal(X1, Y1, sig_x1, sig_y1, mu_x1, mu_y1, cov_xy1)
    
    x2 = np.linspace(40, 90, 1000)
    y2 = np.linspace(100, 190, 1000)

    mu_x2 = params[1][0]
    mu_y2 = params[1][1]
    sig_x2 = params[3][0,0]**0.5
    sig_y2 = params[3][1,1]**0.5
    cov_xy2 = params[3][0,1]

    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = bivariate_normal(X2, Y2, sig_x2, sig_y2, mu_x2, mu_y2, cov_xy2)

    Z = lam*Z1 + (1-lam)*Z2

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X1, Y1, Z,cmap='viridis',linewidth=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('pdf')
    plt.show()

    '''
    # The second. 
    x2 = np.linspace(40, 90, 1000)
    y2 = np.linspace(100, 190, 1000)

    mu_x2 = params[1][0]
    mu_y2 = params[1][1]
    sig_x2 = params[3][0,0]**0.5
    sig_y2 = params[3][1,1]**0.5
    cov_xy2 = params[3][0,1]

    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = bivariate_normal(X2, Y2, sig_x2, sig_y2, mu_x2, mu_y2, cov_xy2)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X2, Y2, Z2,cmap='viridis',linewidth=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('pdf')
    plt.show()
    '''

