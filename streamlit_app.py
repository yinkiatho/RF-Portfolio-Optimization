from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import streamlit as st
import pandas_ta as ta
import pandas as pd
# import pandas_profiling
import numpy as np
from math import sqrt
from keras.models import load_model
# from streamlit_pandas_profiling import st_profile_report
from streamlit_extras.metric_cards import style_metric_cards
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from MVP import MVP
from HRP import HRP
from NaivePortfolio import NaivePorfolio


# Parameters
start_month, start_year = None, None
end_month, end_year = None, None
d = None
n = None

st.set_page_config(
    page_title="Random Forest Stock Selection and Portfolio Optimisation",
    page_icon="🧊",
    layout="wide",
)


# Define the Streamlit app
st.title("Random Forest Stock Selection and Portfolio Optimisation")


# Side Bar
st.sidebar.header("Model Configuration")

with st.sidebar:
    # General Analysis 
    
    # Choose Date
    start_date = st.date_input("Choose Prediction Date", min_value=datetime(2014, 1, 1), 
                               max_value=datetime(2019, 11, 30), 
                               value=datetime(2014, 1, 1))
    start_month, start_year = start_date.month, start_date.year
    
    # Choose Number of Stocks
    n = st.slider("Number of Stocks", 25, 275, 25)
    
    # Choose Prediction Window
    d = st.slider("Number of Historical Years", 1, 3, 1)
    
    

# General Writeup about Project
st.header("Project Description", divider=True)
st.link_button("Project Paper", url="https://www.tandfonline.com/doi/epdf/10.1080/1331677X.2021.1875865?needAccess=true")

# URL Link: https://www.tandfonline.com/doi/epdf/10.1080/1331677X.2021.1875865?needAccess=true
st.text("Based on the paper: 'A novel stock selection and portfolio optimization model based on random forest and hierarchical risk parity' by Qian, Y., Zhang, Y., & Zhang, Y. (2021).")
st.write("With the usage of an ensemble machine learning model, this project aims to capture the non-linear relationships between stock features and stock returns. \
        Trained on historical data from 1999 to 2014, the model aims predict the returns of stocks in the next month. \
        Based on a desired number of stocks, the stocks with the highest predicted returns are then selected to be part of the portfolio.\n \
        Comparison is made between portfolios optimised using the Hierarchical Risk Parity (HRP) algorithm, Naive Portfolio Diversification and Mean-Variance Optimisations Techniques, \
        aiming to maximise Sharpe Ratio. \
        These weights are then used to construct a new portfolio as to which its performance is evaluated against the S&P 500 Index.")

st.subheader("Model Architecture")
st.write("Ensemble Machine Learning Model built with Random Forest Regressor and Multi-layer Perceptron regressor, stacked together with a Gradient Boosting Regressor.")
st.write("Optimised using GridSearchCV and Walk Foward Validation")

with st.expander("Model Code"):
    code = """for ticker in tickers:
    
    print(f"Ticker: {ticker}"
    ticker_data = make_to_monthly(ticker)
    ticker_data['Next Month Close'] = ticker_data['Close'].shift(-1)
    ticker_data.dropna(inplace=True)
    ticker_data['Next Month Returns'] = (ticker_data['Next Month Close'] - ticker_data['Close']) / ticker_data['Close']
   
    features = ticker_data[['Open', 'Close', 'Adj Close', 'Volume', 'Returns', 'High', 'Low',
                            'Stock Momentum', 'Short Term Reversal', 'Long Term Reversal',
                            'Total Returns', 'Market_Beta', 'Turnover Volatility',
                            'Total Return Volatility', 'SMA_5', 'SMA_20', 'SMA_50', 'SMA_252',
                            'adv20', 'VWAP', 'log_returns', 'volatility_30', 'volatility_60',
                            'annual_volatility', 'RSI(2)', 'RSI(7)', 'RSI(14)', 'CCI(30)',
                            'CCI(50)', 'CCI(100)', 'BBWidth', 'Williams']]

    target = ticker_data['Next Month Returns']

    train, test = train_test_split(ticker_data, 0.3)

    X_train = train[:, :-1]  # input as columns
    y_train = train[:, -1]  # output as rows

    
    # Building RF model
    random_forest = RandomForestRegressor(
        n_jobs=-1, random_state=123, oob_score=True, warm_start=True)
    
    param_grid = {
        'n_estimators': [ 600, 700, 800, 900, 1000],
        'max_depth': [10, 20,30, 40, 50, None],
        'min_samples_leaf': [1, 2, 4, 75, 100, 125],
        'criterion': ['absolute_error', 'squared_error', 'friedman_mse', 'poisson'],
        'max_features': [None, 'sqrt', 'log2'],
    }
    
    grid_search = GridSearchCV(estimator=random_forest,
                              param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f"Best Params: {grid_search.best_params_}")
    rf_best_params = grid_search.best_params_
   
    # Build MLPRegressor
    mlp = MLPRegressor(max_iter=100000000000000, random_state=123, warm_start=True)
    
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 50, 25), (100, 100, 100), (200, 100, 50, 25), (200, 200, 200)],
        'activation': ['relu', 'tanh', 'identity'],
         'learning_rate': ['constant', 'adaptive'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f"Best Params: {grid_search.best_params_}")
    
    mlp_best_params = grid_search.best_params_
    
    base_models = [
        ('rf', RandomForestRegressor(**rf_best_params)),
        ('nn', MLPRegressor(**mlp_best_params))
    ]
    
    # Grid Search for meta learner
    param_grid = {
        'final_estimator__n_estimators': [250, 225, 275, 200, 300, 175],
        'final_estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        # Example parameter choices
        'final_estimator__max_depth': [6, 5, 8, None]
    }

    grid_search = GridSearchCV(estimator=stack_model, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    meta_learner = grid_search.best_estimator_
    
    
    final_stack_model = StackingRegressor(
        estimators=base_models, final_estimator=meta_learner)
        
    # Walk Forward Validation
    y_pred, y_actual = walk_forward_validate(ticker_data, 0.3, final_stack_model)

    # Add the predictions to the dataframe only for the test data, avoid look ahead bias
    ticker_data['Next Month Returns Predictions'] = np.nan
    ticker_data.iloc[-len(test):,
                     ticker_data.columns.get_loc('Next Month Returns Predictions')] = y_pred
    ticker_data[['Next Month Returns', 'Next Month Returns Predictions']].plot(
        figsize=(15, 5))

    data = ticker_data[['Next Month Returns',
                        'Next Month Returns Predictions']]
    data.dropna(inplace=True)
    data.index = pd.to_datetime(data.index)
    # data.resample('M').prod().plot(figsize=(15, 5))


    final_df = ticker_data[['Next Month Returns',
                            'Next Month Returns Predictions']]
    
    final_df[['Next Month Returns', 'Next Month Returns Predictions']].plot(
        figsize=(15, 5))
    
    final_df['Close'] = ticker_data['Close']
    final_df.dropna(inplace=True)
    final_df.to_csv(f"predictions_new/{ticker}_predictions.csv")
    
    print(f"Mean Absolute Error: {mean_absolute_error(y_actual, y_pred)}")"""
    
    st.code(code, language='python')

    

       
    
    
    

    # Calculate profits or any other relevant metrics here
    # You can add more sections to display additional charts and tables

# Optionally, add a sidebar to customize model parameters or other settings
# st.sidebar.header("Model Configuration")
# model_parameter = st.sidebar.slider("Parameter Name", min_value, max_value, default_value)

# Add any other customization and enhancements you need for your specific application