'''
Shibo Yao
April 10, 2018
MGMT782 Lecture 10 Monte Carlo
'''

# option price
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web


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


def normal_pdf(x, mu, sig):

    return 1/(2*np.pi)**.5/sig * np.exp(-(x-mu)**2/(2*sig**2))
## student t distribution better for returns

if __name__ == '__main__':
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime(2018, 4, 10)
    symbol = 'SPY'
    source = 'yahoo'

    spxprice = get_single_stock_data(start_date, end_date, symbol, source)
    spxprice = spxprice.Close
    spxrtn = spxprice.pct_change(1).dropna()

    T = .25 # Thre-month maturity of the option
    S0 = 100
    strike = 90

    # estimate the daily return distribution of spy returns
    # and scale to three month
    ## MLE estimators
    muhat = np.mean(spxrtn)
    sighat = np.std(spxrtn)

    ## MC method
    nsamp = 63 # number of trading days in a quarter
    runs = 1000
    optprc = []

    for i in range(runs):
        samp = sighat*np.random.randn(nsamp) + muhat
        Mr = np.sum(samp)
        finalstockprice = S0*(1 + Mr)
        option_price = max(0, finalstockprice-strike)
        optprc.append(option_price)

    print("MC option price:", np.mean(optprc))

    plt.figure()
    plt.hist(spxrtn, bins = 100, density = True)

    xvls = np.linspace(-.05, .05, 100)
    yvls = normal_pdf(xvls, muhat, sighat)
    plt.plot(xvls, yvls)
    plt.show()



