import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor



def train_test_split_two(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]


def stacking_predict(train, val, model):
  train = np.array(train)
  X, y = train[:, :-1], train[:, -1]
  model = model
  model.fit(X, y)
  val = np.array(val).reshape(1, -1)
  pred = model.predict(val)
  return pred[0]
