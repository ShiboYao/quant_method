'''
Shibo Yao
Time: Feb. 20, 2018
MGMT782 Lecture 6 Mixed Model
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def samp_norm_mix(lam, mu1, mu2, sig1, sig2, n):
    w1 = lam
    w2 = 1 - lam
    model_pts = np.random.multinomial(n, [w1, w2])
    mod1data = sig1 * np.random.randn(model_pts[0]) + mu1
    mod2data = sig2 * np.random.randn(model_pts[1]) + mu2
    data = np.concatenate((mod1data, mod2data)).flatten()

    return data


def gau_dist(x, mu, sig):
    y = 1./np.sqrt(2.*np.pi*sig**2)*np.exp(-(x-mu)**2/(2.*sig**2))

    return y

    
def mix_modl(x, mu1, mu2, sig1, sig2, lam):

    return lam*gau_dist(x, mu1, sig1) + (1-lam)*gau_dist(x, mu2, sig2)


def log_like_gau_mix(data, mu1, mu2, sig1, sig2, lam):

    return np.sum(np.log(mix_modl(data, mu1, mu2, sig1, sig2, lam)))


def em_alg(data):
    lam0 = 0.5
    lam1 = 1 - lam0
    mu10 = np.mean(data)
    sig10 = np.std(data)
    mu20 = mu10 * 0.9
    sig20 = sig10 * 0.9

    MAXITR = 5000
    itr = 0
    params = np.array([mu10, mu20, sig10, sig20])
    oldparams = np.zeros([len(params)])
    while sum((oldparams - params)**2) > 1e-8 and itr < MAXITR:
        itr += 1
        err = str(np.sum(oldparams - params)**2)

        ## Compute wij from the initial guess. 
        muvec = [params[0], params[1]]
        sigvec = [params[2], params[3]]
        wgtvec = [lam0, lam1]
        print("EM iteration:", itr)
        print(muvec)
        print(sigvec)
        print(wgtvec)
        print("Error: ", err)

        wi0num = [wgtvec[0]*gau_dist(xi, muvec[0], sigvec[0]) for xi in data]
        wi1num = [wgtvec[1]*gau_dist(xi, muvec[1], sigvec[1]) for xi in data]
    
        denom = [sum([wgtvec[k]*gau_dist(xi, muvec[k], sigvec[k]) for k in range(len(muvec))]) for xi in data]

        wi0 = np.array(wi0num)/np.array(denom)
        wi1 = np.array(wi1num)/np.array(denom)

        ## Compute updated lambda i. 
        lam0 = np.mean(wi0)
        lam1 = np.mean(wi1)

        ## Maximize the log likelihood. 
        log_like_min = lambda x: -log_like_gau_mix(data, x[0], x[1], x[2], x[3], lam0)
        res = minimize(log_like_min, [muvec[0], muvec[1], sigvec[0], sigvec[1]])
        oldparams = params.copy()
        params = res.x

    return params, lam0


## HW multivariate case

if __name__ == "__main__":
    mu1 = 69.1
    mu2 = 63.7
    std1 = 2.9
    std2 = 2.7
    lam = 0.3
    n = 2000 

    data = samp_norm_mix(lam, mu1, mu2, std1, std2, n)

    params, lam = em_alg(data)
    #params = [67, 63, 3.2, 2.7]
    #lam = 0.4
    x = np.linspace(50, 80, 2000)
    y1 = [gau_dist(i, params[0], params[2])*250*lam for i in x]
    y2 = [gau_dist(j, params[1], params[3])*250*(1-lam) for j in x]
   

    plt.figure()
    plt.hist(data, 250)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()

