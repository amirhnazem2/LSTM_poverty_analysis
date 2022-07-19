from math import sqrt
import numpy as np
import tensorflow as tf
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

SET_SIZE_MONTHS = 264
# n_train_hours = 365 * 24
n_train_months = 100

n_rollback_months = 1*12
n_features = 3

num_epochs = 200
batch_size = 50

LSTM_n_hidden = 100
lr = 1e-3

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
dataset = read_csv('datasets/povert_snap_inflation_month.csv', header=0, index_col=0)
values = dataset.values
# smooth out poverty values
x = np.arange(0, SET_SIZE_MONTHS, 1, dtype=int)
y = values[:,0]
z = np.polyfit(x, y, 3*SET_SIZE_MONTHS//12)
p = np.poly1d(z)
values[:,0] = p(x)


# specify columns to plot
groups = [0, 1,2]
i = 1
# plot each column
pyplot.figure()
years = np.array(dataset.index)
x_vals = np.array(range(len(years)))
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(x_vals,values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	plt.xticks(x_vals[x_vals % (12*5) == 0], years[x_vals % (12*5)  == 0])
	i += 1

pyplot.show()



# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, n_rollback_months, 1)
print(reframed.shape)

# drop columns we don't want to predict
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values_reframed = reframed.values
# n_train_hours = 365 * 24
train = values_reframed[:n_train_months, :]
test = values_reframed[n_train_months:, :]
# split into input and outputs
n_obs = n_rollback_months * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_rollback_months, n_features))
test_X = test_X.reshape((test_X.shape[0], n_rollback_months, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(LSTM_n_hidden, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
opt = tf.keras.optimizers.Adam(learning_rate = lr)
model.compile(loss='mse', optimizer=opt,metrics=['accuracy'])
# fit network
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=num_epochs,
    verbose=0,
    mode='min',
    baseline=None,
    restore_best_weights=True
)
history = model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping_monitor], validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction on test data
yhat = model.predict(test_X)


test_X = test_X.reshape((test_X.shape[0], n_rollback_months * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# make a prediction on train data
yhat_train = model.predict(train_X)

train_X = train_X.reshape((train_X.shape[0], n_rollback_months * n_features))
# invert scaling for forecast
inv_yhat_train = concatenate((yhat_train, train_X[:, -(n_features-1):]), axis=1)
inv_yhat_train = scaler.inverse_transform(inv_yhat_train)
inv_yhat_train = inv_yhat_train[:,0]


inv_y = values[:,-n_features]
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
trainPredictPlot[n_rollback_months:n_train_months + n_rollback_months] = inv_yhat_train

# shift test predictions for plotting
testPredictPlot = np.empty_like(inv_y)
testPredictPlot[:] = np.nan
testPredictPlot[n_train_months+n_rollback_months:] = inv_yhat

# plot predistions
# actual data
plt.plot(inv_y)
# train predictions
plt.plot(trainPredictPlot)
# test predictions
plt.plot(testPredictPlot)

plt.xticks(x_vals[x_vals % (12*5) == 0], years[x_vals % (12*5)  == 0])
plt.show()


# predict future poverty based on different SNAP and stable inflation

months_to_predict = 2*n_rollback_months

for i in range(10):
	SNAP_pop =(0.5 + i * 0.1) * values[-1,1]
	test_values = np.reshape([values[-1,0], SNAP_pop ,values[-1,2]] * months_to_predict, (months_to_predict, 3))
	values_combined = np.concatenate((values,test_values),axis = 0)
	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values_combined)

	# frame as supervised learning
	reframed = series_to_supervised(scaled, n_rollback_months, 1)
	# print(reframed.shape)
	# print(reframed.head())

	# split into train and test sets
	values_reframed = reframed.values
	# test = values_reframed[:-years_to_predict-n_rollback_months, :]
	test = values_reframed[-months_to_predict-n_rollback_months:, :]
	test_X = test[:, :n_obs]
	# reshape input to be 3D [samples, timesteps, features]
	test_X = test_X.reshape((test_X.shape[0], n_rollback_months, n_features))
	# print( test_X.shape)

	# predict
	yhat = model.predict(test_X)

	test_X = test_X.reshape((test_X.shape[0], n_rollback_months * n_features))
	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]

	plt.plot(inv_yhat, label='%.2e' % SNAP_pop)
	x_vals_predict = np.array(range(inv_yhat.shape[0]))
	years_predict = np.array(range(years[-n_rollback_months],years[-n_rollback_months]+int(np.ceil(months_to_predict/12))+1) ).astype(int)
	years_predict = np.repeat(years_predict,12 )
	plt.xticks(x_vals_predict[x_vals_predict % (12 ) == 0], years_predict[x_vals_predict % (12 ) == 0])

plt.legend( loc='lower left', borderaxespad=0. , fontsize = 'xx-small')
plt.show()
print("done!")

