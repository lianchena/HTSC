import iFinDPy
from iFinDPy import *
THS_iFinDLogin('htsc2895','') #试用权限只支持最近5年的历史行情
import akshare as ak #https://akshare.xyz/data/index.html \n #https://akshare.xyz/tutorial.html#id1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
import pylab
import math
from datetime import datetime
today = datetime.today().strftime("%Y-%m-%d") # format '2023-09-08'
today = today.replace('-', '') # format'20230908'



'''datasource'''
# 1.portfolio = fund + bond + currency etc.
# 股票型基金收盘价 data_fund("sh501219", "20230101", "20230630")[['date']]
def data_fund(code, start, end):
    dataframe_fund = ak.stock_zh_index_daily_em(symbol = code, start_date = start, end_date = end)
    dataframe_fund.insert(0,column=code, value=dataframe_fund['close'])
    return dataframe_fund[['date',code]]
# 导入组合各基金收盘价
# portfolio(["sh000300"], "20120101", "20230630")
# portfolio(["sh000300","sh501219","sz166009"], "20120101", "20230630")
def portfolio(code,start,end):
    res = data_fund(code[0],start,end)[['date']]
    for i in range(0,len(code)):
        dataframe = data_fund(code[i], start, end)
        res = pd.merge(res, dataframe, how='outer', on = 'date') 
    res.rename(columns={'date': 'datetime'}, inplace=True)
    res.insert(0,column='date', value=pd.to_datetime(res['datetime']))
    res.set_index('date', inplace=True)
    res=res.drop('datetime', axis=1)
    return res 
# GFC后收盘价走势
def graph(portfolio,datetime):
    portfolio[portfolio.index>= datetime].plot(figsize=(15,10));
    return pylab.show()
portfolio1=portfolio(["sh501219","sz166009"], "20120101", "20230630")
graph(portfolio1,"2008-01-01")

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
graph(portfolio2,"2008-01-01")





'''calculation'''
import pypfopt #https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html \n #https://github.com/robertmartin8/PyPortfolioOpt/tree/master/cookbook
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt.expected_returns import mean_historical_return
mean = expected_returns.mean_historical_return(portfolio2)



# 收益、标准差
def mean_variance(portfolio,frequency=252):
    #mu=portfolio.pct_change().dropna(how="all")
    mu=expected_returns.mean_historical_return(portfolio)
    std=(portfolio.pct_change()[1:].std())*np.sqrt(frequency)
    res = pd.DataFrame({"mu": mu, "std": std})
    return res
mean_variance(portfolio2)
'''
price_returns(portfolio2)
ann_std(portfolio2, freq='Period=year')

total_return=portfolio2.iloc[-1, :] / portfolio2.iloc[0, :]
period=(portfolio2.index[-1] - portfolio2.index[0]).days/365
(total_return**(1 / period) - 1)
np.array(total_return**(1 / period) - 1)
expected_returns.mean_historical_return(portfolio2)
ann_return(portfolio2)'''


# Calculating the covariance matrix
def covariance(portfolio,frequency=252):
    res = portfolio.pct_change().dropna(how="all").cov()*frequency
    # res = risk_models.sample_cov(portfolio2, frequency=252)
    # plotting.plot_covariance(res, plot_correlation=True);
    return res # pylab.show()
covariance(portfolio2)

# optimized weights given target mean and conditions on weights
def optimization(mean,sample_cov,target=0.15, L=0, H=1): # Set the target mean return
    n = len(mean) # Total number of assets
    initial_weights = np.ones(n) / n # Initial guess for weights
    def objective_function(weights, cov_matrix): # Define the objective function to minimize (variance of the portfolio)
        return weights.dot(cov_matrix).dot(weights)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.dot(weights, mean) - target},  # target return
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Sum of weights should be 1
    {'type': 'ineq', 'fun': lambda weights: weights},  # weights >= 0
    {'type': 'ineq', 'fun': lambda weights: 1 - weights})
    bounds = [(L, H) for _ in range(n)] # Set bounds for weights (between 0 and 1)
    # Use the minimize function for constrained optimization
    result = scipy.optimize.minimize(objective_function, initial_weights, args=(sample_cov,), constraints=constraints, bounds=bounds)
    #ef = EfficientFrontier(mean, sample_cov)  #alternative method
    #weights = ef.efficient_return(0.15) #minimises risk for a given target return
    # ef.min_volatility() #optimizes for minimum volatility
    # ef.max_sharpe() #optimizes for maximal Sharpe ratio (a.k.a the tangency portfolio)
    # ef.efficient_risk() #maximises return for a given target risk
    return "Optimal weights:"+str(result.x)+". Minimum portfolio variance:"+str(result.fun) 
    #return weights
optimization(mean,covariance(portfolio2))    


'''2023.9.11 MON'''
# max return
# optimized weights with maximum return given conditions on weights and standard deviation
def optimization_ret(returns,cov_matrix,w1=0.4,w2=0.5,target=0.1): # Set the target mean return
    n = len(returns) # Total number of assets
    initial_weights = np.ones(n) / n # Initial guess for weights
    bounds = [(0, 1) for _ in range(n)] # Set bounds for weights (between 0 and 1)    
    # Define constraints for each asset
    def asset1_constraint(weights):
        return w1 - weights[0]  # Constraint for asset 1: 0 <= Weight <= w1
    def asset2_constraint(weights):
        return w2 - weights[1]  # Constraint for asset 2: 0 <= Weight <= w2
    
    # Define the objective function to maximize (portfolio return)
    def objective_function(weights, returns):
        return -np.sum(returns * weights)  # maximize the negative return
    # Define the constraint function for portfolio standard deviation <= 10%
    def risk_constraint(weights, cov_matrix):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return target - portfolio_volatility  # Constraint: Standard Deviation <= 10%
    # Define constraints as a list of custom functions for each asset
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Full investment constraint
        {'type': 'ineq', 'fun': risk_constraint, 'args': (cov_matrix,)},  # Risk constraint
        {'type': 'ineq', 'fun': asset1_constraint},  # Constraint for asset 1
        {'type': 'ineq', 'fun': asset2_constraint}]  # Constraint for asset 2
    result = minimize(objective_function, initial_weights, args=(returns,), method='SLSQP', constraints=constraints, bounds=bounds)
    SD=np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
    return "Optimal Portfolio Weights:"+str(result.x)+". Optimal Portfolio Return:"+str(-result.fun)+". Optimal Portfolio Standard Deviation:"+str(SD) 
optimization_ret(returns(portfolio),covariance(portfolio))    

'''graph'''
# https://www.analyticsvidhya.com/blog/2023/06/optimizing-portfolios-with-the-mean-variance-method-in-python/
# Plot efficient frontier with Monte Carlo sim
ef = EfficientFrontier(mu, S)
fig, ax = plt.subplots(figsize= (10,10))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu, S)
ef2.max_sharpe()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()
# Plot random portfolios
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
ax.scatter(std_tangent, ret_tangent, c='red', marker='X',s=150, label= 'Max Sharpe')
# Format
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.show()







''' draft'''
# 股票基金 data_fund("sh501219", "20230101", "20230630")
def data_fund(code, start, end):
    dataframe_fund = ak.stock_zh_index_daily_em(symbol = code, start_date = start, end_date = end)
    dataframe_fund['date'] = pd.to_datetime(dataframe_fund['date'])
    dataframe_fund.insert(0,column='fund', value=[code]*len(dataframe_fund))
    dataframe_fund.insert(4,column='ret_stock', value=dataframe_fund['close'].pct_change()[1:])
    return dataframe_fund[['date','fund','close','ret_stock']]
stock = data_fund("sh000300", "20120101", "20230630")
stock["ret_stock"].describe()
# 中证全债及指数 data_index("H11001", "20120101", today)
def data_index(code, start, end):
    dataframe_index = ak.stock_zh_index_hist_csindex(symbol = code, start_date = start, end_date = end)
    dataframe_index.insert(0,column='date', value=pd.to_datetime(dataframe_index['日期']))
    dataframe_index.insert(4,column='ret_bond', value=dataframe_index['收盘'].pct_change()[1:])
    return dataframe_index[['date','指数代码','指数中文简称','收盘', 'ret_bond']]
bond = data_index("H11001", "20230101", "20230630")
portfolio = pd.merge(stock, bond, how='outer', on = 'date')

# 股票占基金净值比例
test1=ak.fund_portfolio_hold_em(symbol="008286", date="2023") 
test1.head()
test1[test1['季度'] == '2023年2季度股票投资明细'] 
ak.stock_report_fund_hold(symbol="基金持仓", date="20200630")
port_stock=ak.stock_report_fund_hold_detail(symbol="008286", date="20230630").sort_values(by=['持股市值'], ascending=False)
port_stock.head()
port_stock.sort_values(by=['持股市值'], ascending=False)
# 沪深300最新成分券权重
test2=ak.index_stock_cons_weight_csindex(symbol = "000300") 
test2.head()

# 股票数据
share=ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20210907', adjust="")
print(share)
share.head()
# 日频
index_hist_cni_df = ak.index_hist_cni(symbol="399005")
print(index_hist_cni_df)
# 股指收盘价
test3=ak.stock_zh_index_daily_em(symbol="sh501219",start_date="20120101",end_date = "20230630")
test3.head()
#沪深300
ak.stock_zh_index_hist_csindex(symbol="000300",start_date="20120101",end_date = "20130630")[['日期','指数代码','指数中文简称','收盘']].head()
# 中证全债
test4=ak.stock_zh_index_hist_csindex(symbol="H11001",start_date="20120101",end_date = "20230630")
test4[['日期','指数代码','指数中文简称','收盘']].head()

ak.fund_open_fund_daily_em
ak.fund_lof_hist_em(symbol="501219", period="daily", start_date="20120101", end_date="20230630", adjust="")
ak.fund_etf_hist_em(symbol="513500", period="daily", start_date="20000101", end_date="20230201", adjust="qfq")







'''output'''
portfolio.to_csv("portfolio.csv",index=True)
portfolio = pd.read_csv("portfolio.csv",parse_dates=True,index_col="date")
portfolio[portfolio.index >= "2023-04-01"].plot(figsize=(15,10));
df.to_excel(r'C:\Users\Ron\Desktop\export_dataframe.xlsx', index=False)






'''test'''
stock=ak.stock_report_fund_hold_detail(symbol="501219", date="20230630").sort_values(by=['持股市值'], ascending=False)
stock['股票代码'].transpose()
stock.transpose()
s = pd.Series([90, 91, 85,99])
s[-1:]
q = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Height': [5.1, 6.2, 5.1, 5.2],
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}
q=pd.DataFrame(q)
q.transpose()
l = {'Name': [2, 3, 4, 5],
        'Height': [5, 6, 5, 5],
        'Qualification': [1, 2, 3, 4]}
l=pd.DataFrame(l)
l
l.drop('Name', axis=1)
dataframe=pd.DataFrame()
pd.merge(l, dataframe, how='outer', on = 'date') 

l['City'] = ['New York', 'San Francisco', 'Los Angeles']
q[['Name']]
q['Name']
q.loc[q['Name'] == 2]
q.insert(len(q)-1,column={['w','gg']}, value=[["1"*len(q),"x"*len(q)]])
q.insert(len(q)-1,column=[['w'],['gg']], value=[{"1"*len(q),"x"*len(q)}])
def port_return_to_value(returns):
    port_return = returns.apply(lambda x: x+1)
    port_value = 100 * np.cumprod(port_return)
    return port_value
q.apply(lambda x: x+1)







'''previous intern'''
import pandas as pd
import numpy as np
import datetime
from WindPy import w
import math
import scipy.optimize

if w.isconnected() == 0:
    w.start()


# 此函数为导入式函数，在输入的时候Week_num表示需要的日期数，若需要4周的收益率，week_num需要等于5，额外+1，date_end为数据结束日，一般为当天
def import_data(fund_code, date_end, month_num=9, type='NAV_adj'):
    origin_data = w.wsd(fund_code, type, "ED-"+str(month_num)+"M", date_end, "")
    data = pd.DataFrame(origin_data.Data).T
    data.columns = fund_code
    data.index = origin_data.Times
    data.index = pd.DatetimeIndex(data.index)  # 把index类型转换成DatetimeIndex
    return data


def price_returns(price_series, window_length=1):
    returns = price_series.pct_change(periods=window_length).dropna()
    return returns


def cvar_cal(returns, weights, alpha=0.05):

    portfolio_return = pd.DataFrame(np.dot(returns, weights), columns=['portfolio_return'], index=returns.index)  # 计算资产组合收益率并形成一个df
    sorted_portfolio_return = portfolio_return.sort_values(by='portfolio_return', ascending=True).reset_index(drop=True) # 对资产收益率进行排序
    tail_num = round(len(sorted_portfolio_return) * alpha)  # 取出小于阈值的收益率个数
    cvar = np.average(sorted_portfolio_return.iloc[0:tail_num, ])
    return cvar


def mean_cvar(returns, target_cvar, weight_constraints, alpha=0.05):
    asset_num = returns.shape[1]  # 定义一共有几个资产
    weights = np.ones([asset_num])/asset_num # 初始值

    def fitness(returns, weights):
        port_return = np.average(np.dot(weights, returns))
        return -port_return

    bounds = [tuple(weight_constraints[i]) for i in range(asset_num)]
    constraints = ({'type': 'eq', 'fun': lambda weights: 1.- sum(weights)},
                   {'type': 'ineq', 'fun': lambda weights: cvar_cal(returns=returns, weights=weights, alpha=alpha) -target_cvar})

    optimized = scipy.optimize.minimize(fitness, weights, returns, method='SLSQP', constraints=constraints, bounds=bounds)
    weights = optimized.x
    cvar = cvar_cal(returns=returns, weights=weights)

    return weights, cvar, optimized


# 把输入的收益率的df转变为可以输出的组合的净值数据，收益率的输入必须为dataframe形式
def port_return_to_value(returns):
    port_return = returns.apply(lambda x: x+1)
    port_value = 100 * np.cumprod(port_return)
    return port_value


def ann_return(input_dataframe):
    df = input_dataframe
    total_return = df.iloc[-1, :] / df.iloc[0, :]
    time_gap = df.index[-1] - df.index[0]
    time_gap = time_gap.days / 365
    re = np.array(total_return**(1 / time_gap) - 1)
    return re


def ann_std(input_dataframe, freq='Period=W'):
    df = input_dataframe
    df_pct = df.pct_change()[1:]
    df_pct_std = df_pct.std()
    if freq == 'Period=W':
        df_pct_std = df_pct_std*math.sqrt(52)
    elif freq == 'Period=M':
        df_pct_std = df_pct_std*math.sqrt(12)
    elif freq == 'Period=Q':
        df_pct_std = df_pct_std*math.sqrt(4)
    else:
        df_pct_std = df_pct_std*math.sqrt(252)
    return np.array(df_pct_std)


def max_drawdown_absolute(input_dataframe):

    r = input_dataframe/input_dataframe.iloc[0, :]
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    return np.array(mdd)


def sharpe_ratio(input_dataframe, rf=0.03, freq='Period=W'):

    numerator = ann_return(input_dataframe=input_dataframe)
    numerator = numerator - rf  # /100表示将百分化的小数化
    denominator = ann_std(input_dataframe=input_dataframe, freq=freq)
    sr = np.array(numerator / denominator)
    return sr
