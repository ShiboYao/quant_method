'''
Shibo Yao
time: Jan 6, 2018
MGMT 782 Lecture 4 Kernel Regression
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 



def gen_test_data(n):  # test data for kernel regression 
    xgrd = np.linspace(-0.5, 2.5, n)
    rannum = np.random.randn(n)
    ygrd = np.sin(xgrd) + 0.15*rannum
    return xgrd, ygrd


def gau_ker(x): # Gaussian Kernel for kernel regression. 
    res = 1/np.sqrt(1*np.pi)*np.exp(-x**2/2)
    return res


def epan_ker(x): # Epanechnikov Kernel for kernel regression. 
	x[abs(x)>1] = 1
	u1 = max(x)
	u0 = min(x)
	F1 = u1 - u1**3/3
	F0 = u0 - u0**3/3
	integral = F1 - F0
	return (1-x**2)/integral
    

def ker_reg_order_zero(x, xdata, ydata, ker, h):
    w = ker((x-xdata)/h)
    return np.dot(w, ydata)/np.sum(w)


def cross_validation_error(h, xdata, ydata):
    err = []
    for i in range(len(xdata)):
        xdata_train = np.append(xdata[:i], xdata[i+1:])
        ydata_train = np.append(ydata[:i], ydata[i+1:])
        xdata_test = xdata[i]
        ydata_test = ydata[i]

        ymodel = ker_reg_order_zero(xdata_test, xdata_train, ydata_train, epan_ker, h)
        err.append((ymodel - ydata_test)**2)

    total_cross_validation_err = np.mean(err)
    return total_cross_validation_err


if __name__ == "__main__":
    print("Running...")
    n = 300
    seed = 12345
    np.random.seed(seed)
    xdata, ydata = gen_test_data(n)

    h1 = 0.2  # Bandwidth. 
    xgrd = np.linspace(-0.75, 2.75, 100)
    kerreg1 = np.array([ker_reg_order_zero(xvl, xdata, ydata, epan_ker, h1) for xvl in xgrd])

    ## Find out the optimal value for h. 
    objf = lambda h:cross_validation_error(h, xdata, ydata)
    res = minimize(objf, 0.25)
    print("Optimal Bandwidth:")
    print(float(res.x))

    ## cross validation error
    hgrd = np.linspace(0.1, 2.0, 100)
    err = np.array([cross_validation_error(h, xdata, ydata) for h in hgrd])
    plt.plot(hgrd, err)
    plt.show()


        
    plt.figure()
    plt.plot(xdata, ydata, 'kx')
    plt.plot(xgrd, kerreg1, 'r')
    plt.title("Kernel Regression")
    plt.show()




