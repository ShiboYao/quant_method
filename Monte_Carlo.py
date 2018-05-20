'''
Shibo Yao
April 10, 2018
MGMT782 Lecture 10 Monte Carlo simulation
'''

import numpy as np
import matplotlib.pyplot as plt

def std(x): # unbiased standard deviation
    n = len(x)
    var = 1/(n-1)*sum((x - np.mean(x))**2)

    return var**.5


def normal_pdf(x, mu, sig):

    return 1/(2*np.pi)**.5/sig * np.exp(-(x-mu)**2/(2*sig**2))
    
    

if __name__ == "__main__":
    ## example1, order statistics
    n = 1000
    num = 100000
    qlst = []
    stdlst = []
    samplst = []

    for i in range(num):
        samp = np.random.randn(n)
        srtsamp = sorted(samp)
        q97 = srtsamp[6]
        qlst.append(q97)
        stdlst.append(np.std(samp))
        samplst.extend(samp)
        #stdlst.append(std(samp))

    print("MC mean:", np.mean(qlst))
    print("MC STD:", 1/np.sqrt(num))
    print("Emperical Original STD:", np.mean(stdlst))
    print("Emperical Original STD full:", np.std(samplst))

    plt.hist(qlst, bins = 200, density = 1)
    xvls = np.linspace(min(qlst), max(qlst), 100)
    yvls = normal_pdf(xvls, np.mean(qlst), np.std(qlst))
    plt.plot(xvls, yvls)
    plt.show()
    plt.hist(stdlst, bins = 200, density = 1)
    xvls = np.linspace(min(stdlst), max(stdlst), 100)
    yvls = normal_pdf(xvls, 1, np.std(stdlst))
    plt.plot(xvls, yvls)
    plt.show()

