import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


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
