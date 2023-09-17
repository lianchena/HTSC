import pandas as pd
import numpy as np
import scipy.optimize
import warnings
import pylab
from datetime import datetime
today = datetime.today().strftime("%Y-%m-%d") # format '2023-09-08'
today = today.replace('-', '') # format'20230908'
# iFinDPy 试用权限只支持最近5年的历史行情
import akshare as ak #https://akshare.xyz/data/index.html \n #https://akshare.xyz/tutorial.html#id1

'''datasource'''
# 1.portfolio = fund + bond + currency etc.
# 2.portfolio = equity1 + equity2 + equity3 etc.
# 股票收盘价 data_equity("600129","20230401",today)
def data_equity(code, start, end):
    dataframe_equity = ak.stock_zh_a_hist(symbol=code, period="daily", start_date = start, end_date = end, adjust="")
    dataframe_equity.insert(0,column=code, value=dataframe_equity['收盘'])
    dataframe_equity.insert(0,column='date', value=dataframe_equity['日期'])
    return dataframe_equity[['date',code]]
# 基金各股票 all_stock("501219","20230301","20230907",2023,1)
def all_stock(fund_code,start,end,year,Q):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    all=ak.fund_portfolio_hold_em(symbol=fund_code, date=year)
    all=all.loc[all['季度'] == str(year)+'年'+str(Q)+'季度股票投资明细']
    all.insert(0,column=fund_code, value=all['股票代码'])
    return all[[fund_code]]
# 前十重仓股票收盘价 portfolio_stock("501219","20170301","20210907",2023,2,10)
def portfolio_stock(fund_code,start,end,year,Q,top_stock):
    stock_list = all_stock(fund_code,start,end,year,Q)
    res = data_equity(stock_list[fund_code][0],start,end)[['date']]
    for i in range(0,top_stock):
        dataframe=pd.DataFrame()
        dataframe = data_equity(stock_list[fund_code][i], start, end)
        res = pd.merge(res, dataframe, how='outer', on = 'date') 
    res.rename(columns={'date': 'datetime'}, inplace=True)
    res.insert(0,column='date', value=pd.to_datetime(res['datetime']))
    res.set_index('date', inplace=True)
    res=res.drop('datetime', axis=1)
    return res
# GFC后收盘价走势
portfolio2=portfolio_stock("501219","20170301","20210907",2023,2,10)
def graph(portfolio,datetime):
    portfolio[portfolio.index>= datetime].plot(figsize=(15,10));
    return pylab.show()
graph(portfolio2,"2008-01-01")


'''calculation'''
# returns
def returns(portfolio,frequency=252):
    total_return=portfolio2.iloc[-1, :] / portfolio2.iloc[0, :]
    period=(portfolio2.index[-1] - portfolio2.index[0]).days/365
    mean=total_return**(1 / period) - 1
    std=(portfolio.pct_change()[1:].std())*np.sqrt(frequency)
    res = pd.DataFrame({"mean":mean,"std": std})
    return np.array(mean)
returns(portfolio2)

# calculating the covariance matrix
def covariance(portfolio,frequency=252):
    res = portfolio.pct_change().dropna(how="all").cov()*frequency
    # plotting.plot_covariance(res, plot_correlation=True);
    return res # pylab.show()
covariance(portfolio2)

# optimized weights with minimum variance given target mean and conditions on weights
def optimization_var(returns,sample_cov,target=0.15, L=0, H=1): # Set the target mean return
    n = len(returns) # Total number of assets
    initial_weights = np.ones(n) / n # Initial guess for weights
    def objective_function(weights, cov_matrix): # Define the objective function to minimize (variance of the portfolio)
        return weights.dot(cov_matrix).dot(weights)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.dot(weights, returns) - target},  # target return
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(L, H) for _ in range(n)] # Set bounds for weights (between 0 and 1)
    result = scipy.optimize.minimize(objective_function, initial_weights, args=(sample_cov,), constraints=constraints, bounds=bounds)
    return "Optimal weights:"+str(result.x)+". Minimum portfolio variance:"+str(result.fun) 
optimization_var(returns(portfolio2),covariance(portfolio2))    




