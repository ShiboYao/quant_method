'''
Shibo Yao
data: Mar. 26, 2018
MGMT782 Lecture 9 Linear Programming
'''

import numpy as np
from scipy.optimize import linprog
import time


if __name__ == "__main__":
    m_cons = [10, 100, 1000]
    n_var = [100, 1000, 1e4, 1e5]
    iterations = 10000

    t0 = 0
    t1 = 0
    for m in range(len(m_cons)):
        for n in range(len(n_var)):
            timing = np.zeros([len(m_cons), len(n_var)])
            steps = timing.copy()
            t = 0
            step = 0
            for i in range(iterations):
                A = np.random.randint(-100, 100, m*n)
                A = A.reshape(m, n)
                b = np.random.randint(-50, 10, m)
                c = np.random.randint(-100, 100, n)

                t0 = time.time()
                res = linprog(-c, A_ub = A, b_ub = b)
                t1 = time.time()
                t = t+t1-t0
                # optvars = res.x 
                #maxval = -res.fun
                step += res.nit

            timing[m,n] = t/iterations
            steps[m,n] = step/iterations

    print(timing)
    print(steps)
    
