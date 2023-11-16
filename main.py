import streamlit.components.v1 as components
from jinja2 import Template
from datetime import datetime
import os
import streamlit as st
import pandas_ta as ta
import pandas as pd
# import pandas_profiling
import quantstats as qs
import numpy as np
import base64
# from streamlit_pandas_profiling import st_profile_report
from streamlit_extras.metric_cards import style_metric_cards
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import matplotlib.pyplot as plt
from MVP import MVP
from HRP import HRP
from NaivePortfolio import NaivePorfolio
import webbrowser



mvp = MVP()
mvp.generate_mv_models_two(50)
    
