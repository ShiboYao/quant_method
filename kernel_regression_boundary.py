'''
This script proposes an approach for fixing the boundary issue for kernel regressions.
Shibo Yao
Feb. 11 , 2018
'''


def epan_ker(x): # Take Epanechnikov kernel as an example. 
	x[abs(x)>1] = 1.
	u1 = max(x)
	u0 = min(x)
	F1 = u1 - u1**3/3
	F0 = u0 - u0**3/3
	integral = F1 - F0
	return (1-x**2)/integral


