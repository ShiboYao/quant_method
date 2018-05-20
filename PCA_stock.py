'''
Shibo Yao
time: Feb 27, 2018
MGMT782 PCA on stock data
'''


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import datetime

def get_single_stock_data(start_date,end_date,symbol,source):
    '''
     start_date: datetime.datetime; first date to download stock data
     end_date: datetime.datetime; last date to download stock data
    symbol: string; stock symbol specific to the data source 
    source: string; data source to get stock info from

    Example: 
    start_date = datetime.datetime(2010,1,1)
    end_date = datetime.datetime(2018,1,1)
    symbol = "IBM"   ("F")
    source = "yahoo"   ("google")
    '''
    data = web.DataReader(symbol,source,start_date,end_date)
    return data



if __name__ == '__main__':
    start_date = datetime.datetime(2014, 7, 8)
    end_date = datetime.datetime(2016, 7, 8)
    symbol = 'SPY'
    source = 'yahoo'

    spxprice = get_single_stock_data(start_date, end_date, symbol, source)
    spxprice = spxprice.Close
    spxrtn = spxprice.pct_change(1).dropna()
    spxcumrtn = spxrtn.cumsum()

    df = pd.read_csv('sp100.csv')
    df.Date = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    rtndf = df.pct_change(1)
    rtndf = rtndf['2014-07-08':'2016-07-08']

    rtndf = rtndf.dropna(axis = 1)

    covmat = rtndf.cov()
    evls, evecs = la.eig(covmat)
    per_var_exp = evls.cumsum()/np.sum(evls)

    large_evec = evecs[:,0]
    proj_data = np.dot(rtndf, large_evec)


    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(spxcumrtn.index, spxcumrtn.values)
    plt.title("SP 500 cum Return")
    plt.subplot(1,2,2)
    plt.plot(-proj_data.cumsum())
    plt.title("First Principle Component")
    plt.show()


    plt.figure()
    plt.plot(evls)
    plt.xlabel("Eigen Values of Nasdaq100")
    plt.figure()
    plt.plot(per_var_exp)
    plt.title("Cumulative Sum of Eigenvalues")
    plt.xlabel("")
    plt.ylabel("Percentage Variance Explained")
    plt.show()



