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

class HRP(BasePortfolio):
    
    def __init__(self) -> None:
        super().__init__()
        self.best_model = None
        self.results = None
        
        
    def generate_hrp_models(self, start_month, start_year, end_month, end_year, d, n):

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
                'weights': []
            }
        # while not (curr_year > end_year or (curr_year == end_year and curr_month > end_month)):
            for i in range(25, 275, 25):
                tickers, pred_vector = super.get_top_n_tickers(curr_year, curr_month, i)
                close_data = super.get_close_prices(curr_year, curr_month, d, tickers)
                predicted_returns = super.generate_predicted_historical_returns(
                    curr_year, curr_month, d, tickers)
                rets = expected_returns.returns_from_prices(close_data)

                hrp = hierarchical_portfolio.HRPOpt(rets)
                # hrp.add_objective(objective_functions.L2_reg, gamma=0.1)
                raw_weights = hrp.optimize()
                cleaned_weights = hrp.clean_weights()
                hrp.portfolio_performance(verbose=False)

                ex_return = hrp.portfolio_performance()[0]
                df['expected_return'].append(ex_return)
                df['num_stocks'].append(i)
                df['annual_volatility'].append(hrp.portfolio_performance()[1])
                df['sharpes'].append(hrp.portfolio_performance()[2])
                df['weights'].append(cleaned_weights)

                if hrp.portfolio_performance()[2] > curr_max:
                    curr_max = hrp.portfolio_performance()[2]
                    curr_model = hrp
            # print(ef.portfolio_performance()[0]) # sharpe ratio
            dfs["d"].append(d)
            dfs['dfs'].append(df)

        # Plot Graph
        # plt.plot(df['num_stocks'], df['sharpes'])
        # print(dfs)
        self.best_model = curr_model
        self.results = dfs
        return dfs, curr_model
    
    
    def print_summary(self):
        print(f"Performance of best portfolio: {self.best_model.portfolio_performance(verbose=True)}")
        print(f"Best Portfolio Weights: {self.best_modelbest_model.clean_weights()}")
        
    def plot_weights(self):
        pd.Series(self.best_model.clean_weights()).plot.pie(figsize=(10, 10))
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


        print(qs.reports.full(optimized_portfolio, benchmark=sp500))
        return qs.reports.full(optimized_portfolio, benchmark=sp500)
    
        