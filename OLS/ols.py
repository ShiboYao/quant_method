'''
Shibo Yao
Ordinal Least Square, function call
'''
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.datasets import load_boston
import statsmodels.api as sm
    
    
if __name__ == "__main__":
    
    boston = load_boston()
    X = boston['data']
    y = boston['target']
    y = y.reshape(-1,1)

    var_names = boston['feature_names']


    print(X.shape)
    print(y.shape)
    print(var_names)


    regr = LinearRegression()
    regr.fit(X, y)
    print(regr.coef_)

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
