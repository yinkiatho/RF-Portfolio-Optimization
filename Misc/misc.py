for ticker in top_100_sp500_tickers.tolist():
    #    if ticker not in tickers or ticker in predicted_tickers:
    #        continue

tickers = ['AAPL']
for ticker in tickers:
    ticker_data = pd.read_csv(
        f"data/{ticker}.csv", index_col='Date', parse_dates=True)
    ticker_data['Next Day Close'] = ticker_data['Close'].shift(-1)
    ticker_data.dropna(inplace=True)
    # print(ticker_data.head())
    features = ticker_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                            'Returns', 'Short Term Reversal', 'Stock Momentum',
                            'Long Term Reversal', 'Market_Beta', 'Turnover Volatility', 'Dividends',
                            'Total Returns', 'Total Return Volatility', 'SMA_5', 'SMA_20', 'SMA_50',
                            'SMA_252', 'adv20', 'VWAP', 'log_returns', 'volatility_30',
                            'volatility_60', 'annual_volatility', 'RSI(2)', 'RSI(7)', 'RSI(14)',
                            'CCI(30)', 'CCI(50)', 'CCI(100)', 'BBWidth', 'Williams']]

    target = ticker_data['Next Day Close']

    train, test = train_test_split(ticker_data, 0.3)

    X_train = train[:, :-1]  # input as columns
    y_train = train[:, -1]  # output as rows

    # Building RF model
    '''
    random_forest = RandomForestRegressor(
        n_jobs=-1, random_state=123, oob_score=True)
    
    param_grid = {
        'n_estimators': [100, 125, 130, 150],
        'max_depth': [15, 10, 20, 25, None],
        'min_samples_leaf': [75, 100, 125],
        'criterion': ['absolute_error', 'squared_error', 'friedman_mse'],
        'max_features': [None, 'sqrt', 'log2'],
    }

    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f"Best Params: {grid_search.best_params_}")
    
    rf = RandomForestRegressor(
        oob_score=True, n_jobs=-1, random_state=123, **grid_search.best_params_)
    '''
    rf = RandomForestRegressor(criterion='friedman_mse', max_depth=None, max_features=None,
                               min_samples_leaf=75, n_estimators=150, n_jobs=-1, random_state=123, oob_score=True)
    # Walk Forward Validation
    y_actual, y_pred = walk_forward_validate(ticker_data, 0.3, rf)

    # Add the predictions to the dataframe only for the test data, avoid look ahead bias
    ticker_data['Next Day Close Predictions'] = np.nan
    ticker_data.iloc[-len(test):,
                     ticker_data.columns.get_loc('Next Day Close Predictions')] = y_pred
    ticker_data[['Next Day Close', 'Next Day Close Predictions']].plot(
        figsize=(15, 5))

    data = ticker_data[['Next Day Close', 'Next Day Close Predictions']]
    data.dropna(inplace=True)
    data.index = pd.to_datetime(data.index)
    # data.resample('M').prod().plot(figsize=(15, 5))

    ticker_data['Returns Predictions'] = (
        ticker_data['Next Day Close Predictions'] - ticker_data['Close']) / ticker_data['Close'] + 1
    ticker_data['Returns'] = (ticker_data['Next Day Close'] -
                              ticker_data['Close']) / ticker_data['Close'] + 1

    final_df = ticker_data['Returns Predictions'].resample('M').prod()
    final_df = pd.DataFrame(final_df)

    df_close = ticker_data['Close'].resample('M').last()
    df_close = pd.DataFrame(df_close)
    final_df['Close'] = df_close['Close']
    final_df.dropna(inplace=True)
    final_df.to_csv(f"predictions/{ticker}_predictions.csv")


for ticker in ['AAPL']:
    ticker_data = pd.read_csv(
        f"data/{ticker}.csv", index_col='Date', parse_dates=True)
    print(ticker_data.head())
    ticker_data.resample('M').last().plot(figsize=(15, 5))
    print(ticker_data.head())
    ticker_data['Next Day Close'] = ticker_data['Close'].shift(-1)
    ticker_data.dropna(inplace=True)
    # print(ticker_data.head())
    features = ticker_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                            'Returns', 'Short Term Reversal', 'Stock Momentum',
                            'Long Term Reversal', 'Market_Beta', 'Turnover Volatility', 'Dividends',
                            'Total Returns', 'Total Return Volatility', 'SMA_5', 'SMA_20', 'SMA_50',
                            'SMA_252', 'adv20', 'VWAP', 'log_returns', 'volatility_30',
                            'volatility_60', 'annual_volatility', 'RSI(2)', 'RSI(7)', 'RSI(14)',
                            'CCI(30)', 'CCI(50)', 'CCI(100)', 'BBWidth', 'Williams']]

    target = ticker_data['Next Day Close']

    train, test = train_test_split(ticker_data, 0.3)

    X_train = train[:, :-1]  # input as columns
    y_train = train[:, -1]  # output as rows

    # Building RF model
    '''
    random_forest = RandomForestRegressor(
        n_jobs=-1, random_state=123, oob_score=True)
    
    param_grid = {
        'n_estimators': [100, 125, 130, 150],
        'max_depth': [15, 10, 20, 25, None],
        'min_samples_leaf': [75, 100, 125],
        'criterion': ['absolute_error', 'squared_error', 'friedman_mse'],
        'max_features': [None, 'sqrt', 'log2'],
    }

    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f"Best Params: {grid_search.best_params_}")
    
    rf = RandomForestRegressor(
        oob_score=True, n_jobs=-1, random_state=123, **grid_search.best_params_)
    '''
    rf = RandomForestRegressor(criterion='friedman_mse', max_depth=None, max_features=None,
                               min_samples_leaf=75, n_estimators=150, n_jobs=-1, random_state=123, oob_score=True)
    # Walk Forward Validation
    y_actual, y_pred = walk_forward_validate(ticker_data, 0.3, rf)

    # Add the predictions to the dataframe only for the test data, avoid look ahead bias
    ticker_data['Next Day Close Predictions'] = np.nan
    ticker_data.iloc[-len(test):,
                     ticker_data.columns.get_loc('Next Day Close Predictions')] = y_pred
    ticker_data[['Next Day Close', 'Next Day Close Predictions']].plot(
        figsize=(15, 5))

    data = ticker_data[['Next Day Close', 'Next Day Close Predictions']]
    data.dropna(inplace=True)
    data.index = pd.to_datetime(data.index)
    # data.resample('M').prod().plot(figsize=(15, 5))

    ticker_data['Returns Predictions'] = (
        ticker_data['Next Day Close Predictions'] - ticker_data['Close']) / ticker_data['Close'] + 1
    ticker_data['Returns'] = (ticker_data['Next Day Close'] -
                              ticker_data['Close']) / ticker_data['Close'] + 1

    final_df = ticker_data['Returns Predictions'].resample('M').prod()
    final_df = pd.DataFrame(final_df)

    df_close = ticker_data['Close'].resample('M').last()
    df_close = pd.DataFrame(df_close)
    final_df['Close'] = df_close['Close']
    final_df.dropna(inplace=True)
    final_df.to_csv(f"predictions/{ticker}_predictions.csv")
