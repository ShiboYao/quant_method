'''
Shibo Yao
MGMT782 Lec2 n-dimensional ordinary least square
'''

import numpy as np
import matplotlib.pyplot as plt

def gen_data(n ,rho):


    mean = np.array([0., 0.])

    sigx = 0.2
    sigy = 0.5
    cov = np.array([[sigx*sigx, sigx*sigy*rho], [sigx*sigy*rho, sigy*sigy]])

    data = np.random.multivariate_normal(mean, cov, size = n)

    return data

data = gen_data(250, 0.85)

X = data[:,0]
y = data[:,1]


def beta_h(X, y):

    n = len(y)
    cov = sum(X*y) - sum(X)*sum(y)/n
    var_x = sum(X**2) - (sum(X))**2 / n
    beta = cov/var_x

    return beta

print(beta_h(X, y))

def se_beta(X, y):
    n = len(y)
    beta = beta_h(X, y)
    beta_0 = sum(y)/n - beta*sum(X)/n
    res = beta*X + beta_0 - y
    X_bar = sum(X)/n
    se = (sum(res**2)/(n-2)/sum((X - X_bar)**2))**0.5

    return se

print(se_beta(X, y))

def y_hat(X, y):
    n = len(y)
    beta = beta_h(X, y)
    beta_0 = sum(y)/n - beta*sum(X)/n

    y_ = X * beta + beta_0
    return y_

y_ = y_hat(X, y)

def r_sq(y, y_hat):
    res = y - y_hat
    SSE = sum(res**2)
    y_bar = sum(y)/len(y)
    SST = sum((y - y_bar)**2)
    r_2 = 1 - SSE/SST

    return r_2

print(r_sq(y, y_))


plt.figure()
plt.plot(X, y, 'x')
plt.plot(X, y_)
plt.show()



