'''
Shibo Yao
time: Feb 1, 2018
MGMT 782 Lectur 3 LASSO and Ridge Regression HW
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
from sklearn import datasets

## load data
data = datasets.load_diabetes(True)
X = data[0]
y = data[1]
## initialize hyperparameter
lam = 0.05
k = 10
lam_tune = [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.00001, 0]


def ridge_beta(x, y, lam):
    ones = np.ones(len(x))
    x = np.column_stack((ones,x))
    mat1 = inv(np.dot(x.T, x) + lam*np.eye(x.shape[1]))
    mat2 = np.dot(x.T, y)
    beta_ridge = np.dot(mat1, mat2)
    
    return beta_ridge
# ridge_beta(X, y, 0.05)


def ols_beta(x, y):
    ones = np.ones(len(x))
    x = np.column_stack((ones,x))
    mat1 = inv(np.dot(x.T, x))
    mat2 = np.dot(x.T, y)
    beta = np.dot(mat1, mat2)
    
    return beta


def lasso_objf(B, lam, x, y):
    ones = np.ones(len(x))
    x = np.column_stack((ones,x))
    obj = sum((y - np.dot(x, B))**2)/len(y) + sum(lam*abs(B))
    return obj


def mse(x, y, B):
    ones = np.ones(len(x))
    x = np.column_stack((ones,x))
    err = sum((y - np.dot(x, B))**2)/(len(y-2))
    return err

def exclude(x, y, index):
    start, end = index[0], index[-1]
    x1 = x[:start]
    y1 = y[:start]
    x2 = x[end:]
    y2 = y[end:]
    data = [np.append(x1, x2, axis = 0), np.append(y1, y2, axis = 0)]
    return data

def k_split(x, y, k, shuffle = False):
    if shuffle == True:
        print("Shuffle to be added.")
    n = x.shape[0]
    dur = int(n/k)
    data_train = []
    data_test = []
    for i in range(k):
        start = i * dur
        end = (i+1) * dur
        data_test.append([x[start:end], y[start:end]])
        data_train.append(exclude(x, y, list(range(start,end+1))))
        
    return data_train, data_test
# data_train, data_test = k_split(X, y, 4)

def k_cv(x, y, k, model = "ols", lam = 0.1):
    data_train, data_test = k_split(x, y, k)
    err = 0
    for i in range(k):
        y_train = data_train[i][1]
        x_train = data_train[i][0]
        y_test = data_test[i][1]
        x_test = data_test[i][0]
        if model == "ols":
            beta = ols_beta(x_train, y_train)
        elif model == "ridge":
            beta = ridge_beta(x_train, y_train, lam)
        elif model == "lasso":
            obj_opt = lambda B:lasso_objf(B, lam, X, y)
            B = np.zeros(X.shape[1]+1)
            res = minimize(obj_opt, B, method = 'Powell')
            beta = res.x
        else:
            print("Select from ols, ridge and lasso!")
        err += mse(x_test, y_test, beta)
        
    return err/k
# k_cv(X, y, 4, model = "lasso", lam = 0.05)

def grid_search(k, lam_tune, mod):
    ls = np.inf
    ls_lam = 0
    if mod == "ridge":
        print("Ridge grid search:")
    elif mod == "lasso":
        print("LASSO grid search:")
    else :
        print("Select from ridge and lasso!\n")
        
    for lamd in lam_tune:
        cost = k_cv(X, y, k = k, model = mod, lam = lamd)
        if cost < ls:
            ls, ls_lam = cost, lamd
        print(lamd)
        print(cost)
    print("Best parameter value: ",ls_lam)
    print("Corresponding prediction err: ",ls)
    print("\n")




if __name__ == "__main__":

    print("OLS_beta:\n", ols_beta(X, y), '\n')
    print("Ridge_beta:\n", ridge_beta(X, y, lam), '\n')
    print("LASSO_beta:")
    
    obj_opt = lambda B:lasso_objf(B, lam, X, y)
    B = np.zeros(X.shape[1]+1)
    res = minimize(obj_opt, B, method = 'Powell')
    print(res.x, '\n')



    grid_search(k, lam_tune, "lasso")
    grid_search(k, lam_tune, "ridge")
    print("OLS error:", k_cv(X, y, k), '\n')


