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


class BasePortfolio():
    def __init__(self) -> None:
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