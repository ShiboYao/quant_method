'''
Shibo Yao
python3
'''
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_boston



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


def beta_lasso(X, y, lam):
    obj_opt = lambda B:lasso_objf(B, lam, X, y)
    B = np.zeros(X.shape[1]+1)
    res = minimize(obj_opt, B, method = 'Powell')
    
    return(res.x)


def k_cv(x, y, k, lam = 0.1):
    data_train, data_test = k_split(x, y, k)
    err = 0
    for i in range(k):
        y_train = data_train[i][1]
        x_train = data_train[i][0]
        y_test = data_test[i][1]
        x_test = data_test[i][0]

        beta = beta_lasso(x_train, y_train, lam)

        err += mse(x_test, y_test, beta)
        
    return err/k
# k_cv(X, y, 4, model = "lasso", lam = 0.05)


def grid_search(X, y, k, lam_tune):
    ls = np.inf
    ls_lam = 0

    print("LASSO grid search:")

        
    for lamd in lam_tune:
        cost = k_cv(X, y, k = k, lam = lamd)
        if cost < ls:
            ls, ls_lam = cost, lamd
        print(lamd)
        print(cost)
    print("Best parameter value: ",ls_lam)
    print("Corresponding prediction err: ",ls)
    print("\n")



def feature_select(n, X, y, lam_lis):

    for lam in lam_lis:
        count = 0
        beta = beta_lasso(X, y, lam)
        for i in beta:
            if np.abs(i) > 0.001:
                 count += 1
        print(count)
        if count == n:
            print("lam:", lam)
            print(n, beta)
            return True
        elif lam == lam_lis[-1]:
            print("Not found.")
            return False
    


if __name__ == "__main__":
    
    boston = load_boston()
    X = boston['data']
    y = boston['target']

    
    lam = 0.05
    k = 10
    lam_tune = [50, 20, 10, 5, 1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.00001, 0]
    lam_tune2 = [50, 40, 30, 20, 10]

    
    grid_search(X, y, k, lam_tune)

    #beta_lasso(X, y, lam)
    print(feature_select(4, X, y, lam_tune2))
    
    
    
    
    
