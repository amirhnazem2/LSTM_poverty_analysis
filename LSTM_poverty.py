from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

SET_SIZE_YEARS = 22
# n_train_hours = 365 * 24
n_train_years = 10

n_years = 5
n_features = 2

num_epochs = 100
batch_size = 5

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

from datetime import datetime
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
# dataset = read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
# dataset.drop('No', axis=1, inplace=True)
# # manually specify column names
# dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# dataset.index.name = 'date'
# # mark all NA values with 0
# dataset['pollution'].fillna(0, inplace=True)
# # drop the first 24 hours
# dataset = dataset[24:]
# # trim the dataset
# dataset = dataset[:SET_SIZE_HOURS]
# # summarize first 5 rows
# print(dataset.head(5))

# load dataset
dataset = read_csv('datasets/poverty.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()



# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
# n_years = 3
# n_features = 2
# frame as supervised learning
reframed = series_to_supervised(scaled, n_years, 1)
print(reframed.shape)

# drop columns we don't want to predict
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
# n_train_hours = 365 * 24
train = values[:n_train_years, :]
test = values[n_train_years:, :]
# split into input and outputs
n_obs = n_years * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_years, n_features))
test_X = test_X.reshape((test_X.shape[0], n_years, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction on test data
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_years * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# make a prediction on train data
yhat_train = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], n_years * n_features))
# invert scaling for forecast
inv_yhat_train = concatenate((yhat_train, train_X[:, -(n_features-1):]), axis=1)
inv_yhat_train = scaler.inverse_transform(inv_yhat_train)
inv_yhat_train = inv_yhat_train[:,0]

# invert scaling for whole actual
y_vals = values[:,-n_features]
x_vals = values[:, :n_obs]
y_vals = y_vals.reshape((len(y_vals), 1))
inv_y = concatenate((y_vals, x_vals[:, -(n_features-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# invert scaling for test actual
test_y = test_y.reshape((len(test_y), 1))
inv_y_test = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
inv_y_test = scaler.inverse_transform(inv_y_test)
inv_y_test = inv_y_test[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y_test, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# shift train predictions for plotting
trainPredictPlot = np.empty_like(inv_y)
trainPredictPlot[:] = np.nan
trainPredictPlot[:n_train_years] = inv_yhat_train

# shift test predictions for plotting
testPredictPlot = np.empty_like(inv_y)
testPredictPlot[:] = np.nan
testPredictPlot[n_train_years:] = inv_yhat

# plot predistions
# actual data
plt.plot(inv_y)
# train predictions
plt.plot(trainPredictPlot)
# test predictions
plt.plot(testPredictPlot)
plt.show()

