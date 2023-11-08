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

# URL Link: https://www.tandfonline.com/doi/epdf/10.1080/1331677X.2021.1875865?needAccess=true
st.text("This project is based on the paper: 'A novel stock selection and portfolio optimization model based on random forest and hierarchical risk parity' by Qian, Y., Zhang, Y., & Zhang, Y. (2021).")


    

    
    
    

    # Calculate profits or any other relevant metrics here
    # You can add more sections to display additional charts and tables

# Optionally, add a sidebar to customize model parameters or other settings
# st.sidebar.header("Model Configuration")
# model_parameter = st.sidebar.slider("Parameter Name", min_value, max_value, default_value)

# Add any other customization and enhancements you need for your specific application