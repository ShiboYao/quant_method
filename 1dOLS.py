'''
Shibo Yao
MGMT782 Lec2 one-dimensional ordinary least square
'''
import numpy as np
import pandas as pd

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression 

import statsmodels.api as sm
from scipy import stats

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


