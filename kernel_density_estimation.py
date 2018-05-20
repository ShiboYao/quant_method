'''
MGMT 782 
Shibo Yao
time: Mar. 6, 2018
MLE, Kernel Density Estimation
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import t

## Maximum Liklihood Estimation 
def normal_mle_params(x):
    n = len(x)
    muhat = np.sum(x)/float(n)
    sighat = np.sqrt(np.sum((x - muhat)**2)/float(n))

    return muhat, sighat


def minus_standard_student_ll(x, df):
    n = len(x)
    t1 = np.log(gamma((df + 1)/2.)) * n
    t2 = -0.5 * np.log(np.pi*df) * n
    t3 = -np.log(gamma(df/2.)) * n
    t4 = -(df + 1)/2. * np.sum(np.log(1. + x*x/df))

    return -(t1+t2+t3+t4)


def epker(x):
    if x >= 1 or x <= -1:
        return 0.
    else:
        return 0.75*(1 - x**2)


def kerdenest(x, xi, ker, h):
    n = float(len(xi))
    return sum([ker((xival - x)/h) for xival in xi])/(n*h)


if __name__ == "__main__":
    n = 10000

    sig = 0.55
    mu = 1.13

    normsamp = sig*np.random.randn(n) + mu

    muhat, sighat = normal_mle_params(normsamp)

    #plt.hist(normsamp, bins = 100)
    #plt.show()

    df = 2.3
    studtsamp = t.rvs(df, size = n)
    #plt.hist(studtsamp, bins = 50)
    #plt.show()
    objfun = lambda x: minus_standard_student_ll(studtsamp, x)
    params = minimize(objfun, 1.)

    print(params.x)

    # Kernel Density Estimation
    data1 = np.random.randn(n)
    data2 = np.random.rand(int(n/3))
    data3 = 2. * np.random.randn(n) + 1
    data = np.concatenate([data1, data2, data3])
    print(data)

    h = 0.5
    xgrd = np.linspace(-5, 8, 100)
    ygrd = np.array([kerdenest(xval, data, epker, h) for xval in xgrd])

    plt.figure()
    plt.hist(data, bins = 100, normed = True)
    plt.plot(xgrd, ygrd)
    plt.title("Kernel Density Estimation")
    plt.show()
