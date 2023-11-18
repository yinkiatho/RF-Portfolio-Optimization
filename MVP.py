
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
        self.best_model = None
        self.results = None
        self.portfolio_weights = pd.DataFrame(columns=['Date', 'Ticker', 'Weight'])
        self.performances = pd.DataFrame(columns=['Number Historical Years', 'Number of Stocks', 'Portfolio_Return', 'Portfolio_Volatility', 'Portfolio_Sharpe'])
        
    
    
    def rebalance_portfolio(self, stock_tickers, current_date, d):
        # Filter stock data based on input tickers and current date
        #relevant_data = stock_data[stock_tickers]
        #relevant_data = stock_data[(stock_data['Date'] <= current_date)]

        # Calculate expected returns and covariances
        #tickers, pred_vector = get_top_n_tickers(current_date.year, current_date.month, 25)
        output = super().generate_past_close_data(stock_tickers, current_date.month, 
                                                  current_date.year, num_years=d)
        mu = expected_returns.mean_historical_return(output)
        covariance_matrix = CovarianceShrinkage(output).ledoit_wolf()

        # Define portfolio objective and constraints
        objective = EfficientFrontier(mu, covariance_matrix, weight_bounds=(0, 0.1))
        objective.add_objective(objective_functions.L2_reg, gamma=0.1)
        raw_weights = objective.max_sharpe()
        cleaned_weights = objective.clean_weights()
        #print(cleaned_weights)
        # Optimize portfolio
        #optimized_portfolio = objective.optimize(expected_returns, covariance_matrix)

        # Rebalance portfolio based on optimized weights
        for stock_index, stock_ticker in enumerate(stock_tickers):
            # Add optimized weights to portfolio weights DataFrame
            weight = cleaned_weights[stock_ticker]
            new_row = pd.DataFrame({'Date': current_date, 'Ticker': stock_ticker, 'Weight': weight},  index=[0])
            new_result = pd.concat([self.portfolio_weights, new_row], axis=0, ignore_index=True)
            self.portfolio_weights = new_result
            
    
    def generate_mv_models_two(self, n, d, start_date='2014-01-01', end_date='2019-11-30'):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        current_date = start_date
        while current_date <= end_date:
        # Get input vector of stock tickers for the current month
            stock_tickers, pred_vector = super().get_top_n_tickers(current_date.year, current_date.month, n)
            #print(stock_tickers)
            # Rebalance portfolio using the input tickers
            self.rebalance_portfolio(stock_tickers, current_date, d)
            # Increment the date to the next month
            current_date = current_date + pd.DateOffset(months=1)
        print("Portfolio rebalanced successfully!")
        
        # Processing and generating report
        wide_df = self.portfolio_weights.pivot(index='Date', columns='Ticker', values='Weight')
        wide_df = wide_df.reset_index()
        wide_df.fillna(0, inplace=True)
        wide_df = wide_df.set_index('Date')
        
        # Getting returns
        returns_data = super().get_returns_data_total()
        stocks = list(wide_df.columns)
        returns_data = returns_data[stocks]
        
        # Merging the dataframes together
        wide_df.index = pd.to_datetime(wide_df.index)
        returns_data.index = pd.to_datetime(returns_data.index)
        merged_data = pd.concat([wide_df, returns_data], axis=1 ,keys=['Portfolio_Weights', 'Stock_Returns'])
        merged_data.columns = [f'{col[0]}_{col[1]}' for col in merged_data.columns]
        # Reset the index to have 'Date' as a regular column
        merged_data = merged_data.reset_index()
        merged_data = merged_data.rename(columns={'index': 'Date'})
        merged_data = merged_data.set_index('Date')
        for ticker in wide_df.columns:  # Exclude 'Date' column
            merged_data[f'{ticker}_Weighted_Return'] = merged_data['Portfolio_Weights_' + ticker] * merged_data['Stock_Returns_' + ticker]
    
        merged_data['Portfolio_Return'] = merged_data.filter(like='_Weighted_Return').sum(axis=1)   
        portfolio = merged_data[['Portfolio_Return']]
        
        
        # Generating QuantStats
        sp500 = pd.read_csv(DATA_DIR + 'GSPC.csv', index_col='Date', parse_dates=True)
        mask = (sp500.index >= datetime(2013, 12, 1)) & (sp500.index <= datetime(2019, 11, 30))
        sp500 = sp500.loc[mask]
        sp500 = sp500['Close']
        sp500 = sp500.resample('M').first().pct_change().dropna()  
        optimized_portfolio = portfolio['Portfolio_Return']

        portfolio.index = sp500.index
        #print("Optimized Portfolio:")
        #print(optimized_portfolio)
        
        #print("\nS&P 500 Benchmark:")
        #print(sp500)
        qs.extend_pandas()
        qs.reports.html(optimized_portfolio , benchmark=sp500, output='mv_stats.html', periods_per_year=12)
        return qs.stats.sharpe(optimized_portfolio, periods=12), qs.stats.volatility(optimized_portfolio, periods=12), qs.stats.cagr(optimized_portfolio, periods=12)
        
        
    def generate_all_models(self, start_date='2014-01-01', end_date='2019-11-30'):
        
        for d in range(1, 4):
            for n in range(25, 275, 25):
                print("Generating MVP Model for d = " + str(d) + " and n = " + str(n))
                sharpe, volatility, CAGR = self.generate_mv_models_two(n, d, start_date, end_date)
                new_row = pd.DataFrame({'Number Historical Years': d, 'Number of Stocks': n, 'Portfolio_Return': CAGR, 'Portfolio_Volatility': volatility, 'Portfolio_Sharpe': sharpe},  index=[0])
                new_result = pd.concat([self.performances, new_row], axis=0, ignore_index=True)
                self.performances = new_result
                self.portfolio_weights = pd.DataFrame(columns=['Date', 'Ticker', 'Weight'])
                
        return self.performances
        
    
        
    
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
    
    def get_quantstats(self):
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
        
        return optimized_portfolio, sp500

        
    