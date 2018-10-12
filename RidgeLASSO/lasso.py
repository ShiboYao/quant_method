'''
Shibo Yao
time: Jan 30, 2018
MGMT 782 Lectur 3 LASSO and Ridge Regression
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize


def gen_data(n, rho):
    mean = np.array([0., 0.])

    sigx = 0.2
    sigy = 0.5
    cov = np.array([[sigx*sigx, sigx*sigy*rho], [sigx*sigy*rho, sigy*sigy]])

    data = np.random.multivariate_normal(mean, cov, size = n)

    return data

## 1) ridge regression in a 1D setting

def ridge_beta(x, y, lam):
    lam = 0.03
    mat1 = inv(np.dot(x.T, x))
    mat2 = np.dot(x.T, y)
    beta = np.dot(mat1, mat2)
    beta_ridge = np.dot(mat1 + lam*np.eye(len(mat1)), mat2)

    return beta_ridge

def ols_beta(x, y):
    mat1 = inv(np.dot(x.T, x))
    mat2 = np.dot(x.T, y)
    beta = np.dot(mat1, mat2)
    
    return beta


## 2) LASSO regression (no closed form formula)
## load the diabetes dataset (boston housing dataset)
## apply lasso and ridge on them and compare to OLS with different lambda values

def lasso_objf(A, B, lam):
    obj = np.sum((A + B*x1 - y1)**2) + lam*abs(B)
    return obj

## 3) Implement a simple version of cross-validation to select the optimal lambda value for each model

if __name__ == "__main__":
    n = 300
    data = gen_data(n, 0.7)
    x1 = data[:, 0]
    y1 = data[:, 1]

    data2 = gen_data(n, 0.5)

    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)
    data3 = pd.concat([df1, df2], axis = 1).values

    x2 = data3[:, :3]
    y2 = data3[:, -1]

    lam = 0.005
    beta1 = ols_beta(x2, y2)
    beta2 = ridge_beta(x2, y2, lam)


    obj_opt = lambda x:lasso_objf(x[0], x[1], lam)
    a0 = 0.
    b0 = 1.

    res = minimize(obj_opt, [a0, b0])
    ahat, bhat = res.x
    print(res)
