
import os
import pandas as pd
import numpy as np
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
from metrics import *
from BasePortfolio import BasePortfolio
# Get fundamental data for each stock in the ticker and append to the dataframe

class MVP(BasePortfolio):
    
    def __init__(self) -> None:
        super().__init__()
        # Meta Variables
        self.DATA_DIR = (os.getcwd() + "/data/")
        self.PREDICTION_DIR = (os.getcwd() + "/predictions_new/")
        self.start_date = '2014-01-01'
        self.end_date = "2019-11-30"
        self.tickers = self.get_all_symbols()
        self.predicted_tickers = self.get_current_predictions()
        self.sample_index = self.read_sample_index()
        self.best_model = None
        self.results = None
        
        
    def get_all_symbols(self):
        return [v.strip('.csv') for v in os.listdir(self.DATA_DIR)]

    def get_current_predictions(self):
        return [v.strip('_predictions.csv') for v in os.listdir(self.PREDICTION_DIR)]
    
    def read_sample_index(self):
        sample_index = pd.read_csv(self.DATA_DIR + 'AAPL.csv', index_col='Date', parse_dates=True)
        mask = (sample_index.index >= self.start_date) & (sample_index.index <= self.end_date)
        sample_index = sample_index.loc[mask]
        return sample_index
    
    
    def get_top_n_tickers(self, year, month, n):
    #print("Getting top n tickers for year: " + str(year) + " month: " + str(month))
        results = []
        for ticker in self.get_current_predictions():
            df = pd.read_csv(self.PREDICTION_DIR + ticker + '_predictions.csv', index_col='Date', parse_dates=True)
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

    def get_close_prices(self, year, month, d, tickers):
        df = pd.DataFrame()
        days = d * 365
        target_date = datetime(year, month, 1) - timedelta(days=days)
        for ticker in tickers:
            data = pd.read_csv(self.DATA_DIR + ticker + '.csv',
                               index_col='Date', parse_dates=True)
            mask = (data.index >= target_date) & (data.index <= datetime(year, month, 1))
            data = data.loc[mask]
            df[ticker] = data['Close']
            df.index = data.index

        return df


    def generate_predicted_historical_returns(self, year, month, d, tickers):
        d = d * 365
        target_date = datetime(year, month, 1) - timedelta(days=d)
        df = pd.DataFrame()
        for ticker in tickers:
            prediction = pd.read_csv(self.PREDICTION_DIR + ticker + '_predictions.csv', index_col='Date', parse_dates=True)
            mask = (prediction.index >= target_date) & (prediction.index <= datetime(year, month, 1))
            prediction = prediction.loc[mask]
            df[ticker] = prediction['Next Month Returns Predictions']
            df.index = prediction.index
        return df
    


    def get_top_n_tickers_combined(self, start_year, start_month, end_year, end_month, n):
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


    def generate_close_data(self, tickers, month, year):
        df = pd.DataFrame()
        for ticker in tickers:
            data = pd.read_csv(self.DATA_DIR + ticker + '.csv',
                               index_col='Date', parse_dates=True)
            data = data.loc[data.index.month == month]
            data = data.loc[data.index.year == year]
            df[ticker] = data['Close']
            df.index = data.index
        return df


    def generate_all_close_data(self, tickers, start_year, start_month, end_year, end_month):
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
        
        output.index = self.sample_index.index
        return output

    
    
    def generate_mv_models(self, start_month, start_year):
    
        curr_year, curr_month = start_year, start_month
        dfs = {
            "d": [],
            "dfs": []
        }
        curr_max = 0
        curr_model = None
    
        for d in range(1, 4):
            df = {
            'sharpes': [],
            'expected_return': [],
            'annual_volatility': [],
            'num_stocks': [],
            'weights':[]
            }
        #while not (curr_year > end_year or (curr_year == end_year and curr_month > end_month)):
            for i in range(25, 275, 25):
                tickers, pred_vector = self.get_top_n_tickers(curr_year, curr_month, i)
                close_data = self.get_close_prices(curr_year, curr_month, d, tickers)
                predicted_returns = self.generate_predicted_historical_returns(curr_year, curr_month, d, tickers)
                #print(close_data)
                #print(predicted_returns)
                mu = predicted_returns.mean()
                #mu = expected_returns.mean_historical_return(close_data)
                #print(mu)
                S = CovarianceShrinkage(close_data).ledoit_wolf()
                #print(S)\
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                raw_weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
                #print(pd.Series(cleaned_weights).plot.pie(figsize=(10, 10)))
                # ef.save_weights_to_file("weights.csv")  # saves to file
                #print(cleaned_weights)
                #print(raw_weights)
                ef.portfolio_performance(verbose=False)

                ex_return = ef.portfolio_performance()[0]
                df['expected_return'].append(ex_return)
                df['num_stocks'].append(i)
                df['annual_volatility'].append(ef.portfolio_performance()[1])
                df['sharpes'].append(ef.portfolio_performance()[2])
                df['weights'].append(cleaned_weights)

                if ef.portfolio_performance()[2] > curr_max:
                    curr_max = ef.portfolio_performance()[2]
                    curr_model = ef
            #print(ef.portfolio_performance()[0]) # sharpe ratio
            dfs["d"].append(d)
            dfs['dfs'].append(df)
        # Plot Graph
        #plt.plot(df['num_stocks'], df['sharpes'])
        #print(dfs)
        self.best_model = curr_model
        self.results = dfs
        return dfs, curr_model
    
    def print_summary(self):
        print(f"Performance of best portfolio: {self.best_model.portfolio_performance(verbose=True)}")
        print(f"Best Portfolio Weights: {self.best_modelbest_model.clean_weights()}")
        
    def plot_weights(self, weights):
        pd.Series(weights).plot.pie(figsize=(10, 10))
        return
    
    
    def plot_graph_results(self):
        #Plot Graph results
        #print(results)
        # Make plot grid 2x2
        #fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        d = [1, 2, 3]
        for i, result in enumerate(self.results['dfs']):
            #print(i, result)
            plt.plot(result['num_stocks'], result['sharpes'])
            plt.xlabel('Number of Stocks')
            plt.ylabel('Sharpe Ratio')
            #plt.legend(['d = ' + str(d[i])])
            plt.title('MV: Sharpe Ratio vs Number of Stocks')
        plt.legend(['d = 1', 'd = 2', 'd = 3'])
        plt.show()

        for i, result in enumerate(self.results['dfs']):
            #print(i, result)
            plt.plot(result['num_stocks'], result['expected_return'])
            plt.xlabel('Number of Stocks')
            plt.ylabel('Expected Return')
            # plt.legend(['d = ' + str(d[i])])
            plt.title('MV: Expected Return vs Number of Stocks')
        plt.legend(['d = 1', 'd = 2', 'd = 3'])
        plt.show()
        
        
        
    def print_comparison_quantstats(self):
        weights = self.best_model.clean_weights()
        # For each stock, get the weights, get daily historical return from 2014-2018
        # Get SP500 Benchmark
        sp500 = pd.read_csv(DATA_DIR + 'GSPC.csv', index_col='Date', parse_dates=True)
        mask = (sp500.index >= datetime(2014, 1, 1)) & (sp500.index <= datetime(2019, 12, 31))
        sp500 = sp500.loc[mask]
        sp500 = sp500['Close']
        sp500 = sp500.pct_change().dropna()
        #print(sp500)

        returns = pd.DataFrame()
        for ticker, weight in weights.items():
    
            # Get historical data for each stock
            if weight == 0:
                continue
            data = pd.read_csv(DATA_DIR + ticker + '.csv',
                                   index_col='Date', parse_dates=True)
            mask = (data.index >= datetime(2014, 1, 1)) & (data.index <= datetime(2019, 11, 30))
            data = data.loc[mask]
            data = data['Close']
            data = data.pct_change().dropna()
            returns[ticker] = data * weight

        returns['Portfolio'] = returns.sum(axis=1)
        optimized_portfolio = returns['Portfolio']


        #print(qs.reports.full(optimized_portfolio, benchmark=sp500))
        return qs.reports.full(optimized_portfolio, benchmark=sp500)

        
    