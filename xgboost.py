import yfinance as yf
from finta import TA
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


# Commented out IPython magic to ensure Python compatibility.
# Import libraries

# from IPython.core.debugger import set_trace

plt.style.use(style='seaborn')
# %matplotlib inline

df = pd.read_csv(DATA_DIR + "MSFT" + '.csv',
                 parse_dates=True, index_col='Date')


# Calculate SMA, RSI and OBV
df['SMA200'] = TA.SMA(df, 200)
df['RSI'] = TA.RSI(df)
df['ATR'] = TA.ATR(df)
df['BBWidth'] = TA.BBWIDTH(df)
df['Williams'] = TA.WILLIAMS(df)

# df.fillna(0, inplace=True)
df = df.iloc[200:, :]

df.tail(10)

df = df.iloc[200:, :]
df['target'] = df.Close.shift(-1)
df.dropna(inplace=True)

# Train test split


def train_test_split(data, perc):
  data = data.values
  n = int(len(data) * (1 - perc))
  return data[:n], data[n:]


train, test = train_test_split(df, 0.2)
train

print(len(df))
print(len(train))
print(len(test))
train[0, :-1]

# Seperate the train data into feature and target data
X = train[:, :-1]
y = train[:, -1]
y

# Import XGBoost Regressor
'''
params = {'max_depth': [3, 6, 10],
          'learning_rate': [0.1, 0.2, 0.3],
          'n_estimators': [100, 500, 1000],
          'colsample_bytree': [0.3, 0.7]}

xgbr = XGBRegressor(seed=20)
clf = GridSearchCV(estimator=xgbr,
                   param_grid=params,
                   scoring='neg_mean_squared_error',
                   verbose=1)
clf.fit(X, y)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))


# model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
# model.fit(X,y)

# Import XGBRegressor from xgboost fit X,y data into model.
'''
model = XGBRegressor(objective='reg:squarederror', n_estimators=350,
                     colsample_bytree=0.7, learning_rate=0.05, max_depth=3, gamma=5)
model.fit(X, y)

# Check the test set first item
test[0, :]
# val = np.array(test[0, :-1]).reshape(1, -1)

# Let's see what our model predict # model.predic need 2d array that's why we do reshape
val = np.array(test[0, :-1]).reshape(1, -1)
pred = model.predict(val)
pred

# Predict Method def
# Separete the train set into feature(X) and target (y) and predict one sample at a time.
# model = XGBRegressor(objective='reg:squarederror', n_estimators=350, learning_rate=0.05, colsample_bytree=0.7,max_depth=3,gamma=5)


def xgb_predict(train, val):
  train = np.array(train)
  X, y = train[:, :-1], train[:, -1]
  model = XGBRegressor(objective='reg:squarederror', n_estimators=350,
                       learning_rate=0.05, colsample_bytree=0.7, max_depth=3, gamma=5)
  model.fit(X, y)
  val = np.array(val).reshape(1, -1)
  pred = model.predict(val)
  return pred[0]


xgb_predict(train, test[0, :-1])

# Mean Absolute Percentage Error(MAPE)


def mape(actual, pred):
  actual, pred = np.array(actual), np.array(pred)
  mape = np.mean(np.abs((actual-pred)/actual))*100
  return mape

# Walk forward validation
# Since we are making Next day price prediciton. We will predict first record of test dataset
# After that we add real observation from test dataset and refit the model and then predict the next observation from test set. and so on.


def validate(data, perc):
  predictions = []
  train, test = train_test_split(data, perc)
  history = [x for x in train]

  for i in range(len(test)):
    X_test, y_test = test[i, :-1], test[i, -1]
    pred = xgb_predict(history, X_test)
    predictions.append(pred)

    history.append(test[i])

  error = mean_squared_error(test[:, -1], predictions, squared=False)
  MAPE = mape(test[:, -1], predictions)
  return error, MAPE, test[:, -1], predictions


rmse, MAPE, y, pred = validate(df, 0.2)

print("RMSE: " f'{rmse}')
print("MAPE: " f'{MAPE}')
print(y)
print(pred)

# Add test and pred array.
pred = np.array(pred)
test_pred = np.c_[test, pred]


df_TP = pd.DataFrame(test_pred, columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                         'Returns', 'Short Term Reversal', 'Stock Momentum',
                                         'Long Term Reversal', 'Market_Beta', 'Turnover Volatility', 'Dividends',
                                         'Total Returns', 'Total Return Volatility', 'SMA_5', 'SMA_20', 'SMA_50',
                                         'SMA_252', 'adv20', 'VWAP', 'log_returns', 'volatility_30',
                                         'volatility_60', 'annual_volatility', 'RSI(2)', 'RSI(7)', 'RSI(14)',
                                         'CCI(30)', 'CCI(50)', 'CCI(100)', 'SMA200', 'RSI', 'ATR', 'BBWidth', 'Williams', 'Target', 'Pred'])


df_TP.index = df.index


# Show the close price
plt.figure(figsize=(15, 12))
plt.title('Next Day Close Price of Microsoft and Predicted Price', fontsize=18)
plt.subplot(211)
plt.plot(df_TP['Target'], label="Next day Actual Closing Price", color='cyan')
plt.plot(df_TP['Pred'], label="Predicted Price", color='green', alpha=1)
plt.xlabel('Date', fontsize=18)
plt.legend(loc="upper left")
plt.ylabel('Price in USD $', fontsize=18)


plt.figure(figsize=(15, 9))
plt.title('RSI of ATR', fontsize=18)
plt.subplot(212)
plt.plot(df_TP['RSI'], label="RSI", color='green', alpha=0.3)
plt.plot(df_TP['ATR'], label="ATR", color='red', alpha=0.3)
plt.xlabel('Date', fontsize=18)
plt.legend(loc="upper left")
plt.ylabel('RSI Range', fontsize=18)
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df_TP['RSI'], label="RSI range", color='green', alpha=0.3)
ax2.plot(df_TP['ATR'], label="ATR Range", color='red', alpha=0.3)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('RSI Range')
ax2.set_ylabel('ATR Range')
