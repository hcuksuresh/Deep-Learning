# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:19:57 2018

@author: sukandulapati
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('HDFCBANK.NS.csv')
#dataset_train = dataset_train[dataset_train.Open.str.contains("null") == False]
dataset_train.info()
dataset_train['Date'] = pd.to_datetime(dataset_train['Date'])
dataset_train = dataset_train.sort_values('Date').reset_index(drop=True)

dataset_train[['Open','High','Low', 'Close']] = dataset_train[['Open','High','Low', 'Close']].apply(pd.to_numeric)

training_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

a = int(0.80 * len(training_set_scaled))
Xtrain = training_set_scaled[0:a:,]
X_test = training_set_scaled[a:,]

training_set_scaled = Xtrain

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Building the RNN
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.summary()
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs =10, batch_size = 32)

# from tensorflow.keras import load_model
# regressor.save('hdfc_model.h5')

#Testing and evaluating the model

training_set_scaled = X_test
X_test = []
y_test = []
for i in range(60, len(training_set_scaled)):
    X_test.append(training_set_scaled[i-60:i, 0])
    y_test.append(training_set_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
real_stock_price = sc.inverse_transform(y_test.reshape(127,1))
# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#y_test = np.reshape(-1,1)

from tensorflow.keras.models import load_model
hdfc_model = load_model('hdfc_model.h5')
#regressor = load_model('hdfc_model.h5')

predicted_stock_price = hdfc_model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real HDFC Bank Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted HDFC Bank Stock Price')
plt.title('HDFC Bank Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('HDFC Stock Price')
plt.legend()
plt.show()

# Evaluating the RNN model since regression for continuous variable
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

import numpy
relative_rmse = rmse/numpy.mean(pd.DataFrame(real_stock_price[0]))
print('RMSE : ' +str(rmse))
print('Relative RMSE: ' +str(relative_rmse))

#################################3

new_data = pd.read_csv('test_hdfc_new.csv')

new_test_set = new_data.iloc[:, 1:2].values
new_test_set = sc.transform(new_test_set)


training_set_scaled = new_test_set
new_test = []
new_y_test = []
for i in range(60, len(training_set_scaled)):
   new_test.append(training_set_scaled[i-60:i, 0])
   new_y_test.append(training_set_scaled[i, 0])

new_X_test, new_y_test = np.array(new_test), np.array(new_y_test)
real_stock_price = sc.inverse_transform(new_y_test.reshape(688,1))
# Reshaping
new_X_test = np.reshape(new_X_test, (new_X_test.shape[0], new_X_test.shape[1], 1))


predicted_stock_price = hdfc_model.predict(new_X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real HDFC Bank Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted HDFC Bank Stock Price')
plt.title('HDFC Bank Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('HDFC Stock Price')
plt.legend()
plt.show()

# Evaluating the RNN model since regression for continuous variable
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

import numpy
relative_rmse = rmse/numpy.mean(pd.DataFrame(real_stock_price[0]))
print('RMSE : ' +str(rmse))
print('Relative RMSE: ' +str(relative_rmse))

#####################################









XOtrain = dataset_train.iloc[0:a]
XOtest = dataset_train.iloc[a:]

real_stock_price = XOtest.iloc[:, [4,7]].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((XOtrain['Close'], XOtest['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(XOtest) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 60 + len(XOtest)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

from keras.models import load_model
regressor = load_model('techm_model.h5')
#regressor = load_model('hdfc_model.h5')

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real AXIS Bank Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted AXIS Bank Stock Price')
plt.title('AXIS Bank Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AXIS Stock Price')
plt.legend()
plt.show()

# Evaluating the RNN model since regression for continuous variable
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

import numpy
relative_rmse = rmse/numpy.mean(pd.DataFrame(real_stock_price[0]))
rmse
relative_rmse