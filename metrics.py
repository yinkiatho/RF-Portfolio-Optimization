import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import quantstats as qs
import copy
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from datetime import datetime, timedelta
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import CLA, plotting
from pypfopt import hierarchical_portfolio
from pypfopt import plotting

DATA_DIR = (os.getcwd() + "/data/")
PREDICTION_DIR = (os.getcwd() + "/predictions_new/")

sample_index = pd.read_csv(
    DATA_DIR + 'AAPL.csv', index_col='Date', parse_dates=True)
mask = (sample_index.index >= start_date) & (sample_index.index <= end_date)
sample_index = sample_index.loc[mask]
def train_test_split(data, perc):
  data = data.values
  n = int(len(data) * (1 - perc))
  return data[:n], data[n:]


def RF_predict(train, val, final_model):
  train = np.array(train)
  X, y = train[:, :-1], train[:, -1]
  model = final_model
  model.fit(X, y)
  val = np.array(val).reshape(1, -1)
  pred = model.predict(val)
  return pred[0]



def walk_forward_validate(data, perc, final_model):
  predictions = []
  train, test = train_test_split(data, perc)
  history = [x for x in train]

  for i in range(len(test)):
    X_test, y_test = test[i, :-1], test[i, -1]
    pred = RF_predict(history, X_test, final_model)
    predictions.append(pred)

    history.append(test[i])

  return predictions, test[:, -1]


def make_to_monthly(ticker):
    df = pd.read_csv(DATA_DIR + ticker + '.csv',
                     parse_dates=True, index_col='Date')
    #print(df.columns)
    agg_functions = {
        'Open': 'first',   # First value in the month for 'Open'
        'Close': 'last',   # Last value in the month for 'Close'
        'Adj Close': 'last',  # Last value in the month for 'Adj Close'
        'Volume': 'mean',   # Sum of 'Volume' in the month
        'Returns': 'prod',
        'High': 'max',     # Maximum value of 'High' in the month
        'Low': 'min',       # Minimum value of 'Low' in the month
        'Stock Momentum': 'mean',  # Mean of 'Stock Momentum' in the month
        'Short Term Reversal': 'mean',  # Mean of 'Short Term Reversal' in the month
        'Long Term Reversal': 'mean',  # Mean of 'Long Term Reversal' in the month
        'Total Returns': 'mean',  # Mean of 'Total Returns' in the month
        'Market_Beta': 'mean',
        'Turnover Volatility': 'mean',
        'Total Return Volatility': 'mean',
        'SMA_5': 'last',
        'SMA_20': 'last',
        'SMA_50': 'last',
        'SMA_252': 'last',
        'adv20': 'last',
        'VWAP': 'last',
        'log_returns': 'mean',
        'volatility_30': 'last',
        'volatility_60': 'last',
        'annual_volatility': 'last',
        'RSI(2)': 'last',
        'RSI(7)': 'last',
        'RSI(14)': 'last',
        'CCI(30)': 'last', 'CCI(50)': 'last', 'CCI(100)': 'last', 'BBWidth': 'last', 'Williams': 'last'
    }

    monthly = df.resample('M').agg(agg_functions)
    return monthly
  
  
  
def get_top_n_tickers(year, month, n):
    #print("Getting top n tickers for year: " + str(year) + " month: " + str(month))
    results = []
    for ticker in get_current_predictions():
        df = pd.read_csv(PREDICTION_DIR + ticker + '_predictions.csv', index_col='Date', parse_dates=True)
        mask = (df.index.year == year) & (df.index.month == month)
        df = df.loc[mask]
        #print(ticker, df['Next Month Returns Predictions'][0])
        results.append((ticker, df['Next Month Returns Predictions'][0]))
    
    #print(results)
    results.sort(key=lambda x: x[1], reverse=True)
    tickers = [i[0] for i in results[:n]]
    pred_vector = [i[1] for i in results[:n]]
    #print(pred_vector)
    #print(year, month)
    return tickers, pred_vector

def get_close_prices(year, month, d, tickers):
    df = pd.DataFrame()
    days = d * 365
    target_date = datetime(year, month, 1) - timedelta(days=days)
    for ticker in tickers:
        data = pd.read_csv(DATA_DIR + ticker + '.csv',
                           index_col='Date', parse_dates=True)
        mask = (data.index >= target_date) & (data.index <= datetime(year, month, 1))
        data = data.loc[mask]
        df[ticker] = data['Close']
        df.index = data.index

    return df


def generate_predicted_historical_returns(year, month, d, tickers):
    d = d * 365
    target_date = datetime(year, month, 1) - timedelta(days=d)
    df = pd.DataFrame()
    for ticker in tickers:
        prediction = pd.read_csv(PREDICTION_DIR + ticker + '_predictions.csv', index_col='Date', parse_dates=True)
        mask = (prediction.index >= target_date) & (prediction.index <= datetime(year, month, 1))
        prediction = prediction.loc[mask]
        df[ticker] = prediction['Next Month Returns Predictions']
        df.index = prediction.index
    return df
    


def get_top_n_tickers_combined(start_year, start_month, end_year, end_month, n):
    output = []
    pred_vectors = []
    curr_year, curr_month = start_year, start_month
    while not (curr_year > end_year or (curr_year == end_year and curr_month > end_month)):
        tickers, pred_vector = get_top_n_tickers(curr_year, curr_month, n)
        output.append(tickers)
        pred_vectors.append(pred_vector)
        
        if curr_month == 12:
            curr_month = 1
            curr_year += 1
        else:
            curr_month += 1
    return output, pred_vectors


def generate_close_data(tickers, month, year):
    df = pd.DataFrame()
    for ticker in tickers:
        data = pd.read_csv(DATA_DIR + ticker + '.csv',
                           index_col='Date', parse_dates=True)
        data = data.loc[data.index.month == month]
        data = data.loc[data.index.year == year]
        df[ticker] = data['Close']
        df.index = data.index
    return df


def generate_all_close_data(tickers, start_year, start_month, end_year, end_month):
    output = pd.DataFrame()
    curr_year, curr_month = start_year, start_month
    curr_idx = 0
    while not (curr_year > end_year or (curr_year == end_year and curr_month > end_month)):
        data = generate_close_data(tickers[curr_idx], curr_month, curr_year)
        output = pd.concat([output, data], axis=0, join='outer')
        output = output.reset_index(drop=True)
        if curr_month == 12:
            curr_month = 1
            curr_year += 1
        else:
            curr_month += 1
        curr_idx += 1
        
    output.index = sample_index.index
    return output


def plot_efficient_frontier_and_max_sharpe(ef):
    # Optimize portfolio for maximal Sharpe ratio
    # ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(10, 5))
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the max sharpe portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*",
               s=100, c="r",     label="Max Sharpe")
# Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes,
               edgecolors="black", cmap="YlGnBu")
# Output
    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.legend()
    plt.tight_layout()
    plt.show()
