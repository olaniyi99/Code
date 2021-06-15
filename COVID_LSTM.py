import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import datetime
from statistics import mean
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras

np.random.seed(7)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
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
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


parse_dates = ['Date']
load_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/hourlyload_data.csv')#half hourly power demand data 
met_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/fullyear_temp_data_london.csv') #meterological hourly data
london_mobility_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/London_mobility.csv') #daily mobility data
Holiday = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/Holidays.csv') #holiday data

load_data['Date'] = pd.to_datetime(load_data['Date'] + ' '  + load_data['Time'], dayfirst=True)
met_data['Date'] = pd.to_datetime(met_data['Date'] + ' '  + met_data['Time'], dayfirst=True)

Holiday['Date'] = pd.to_datetime(Holiday['Date'])

n_train_hours = int(365 * 24 * 0.67)

train_dates = pd.to_datetime(load_data['Date'])
test_dates = train_dates.iloc[n_train_hours+1:]
#test_dates = test_dates.dt.strftime('%d/%m/%Y %H')
#test_dates = test_dates.dt.strftime('%d/%m/%Y')
train_dates = train_dates.iloc[:n_train_hours]


load_data.drop('Time', inplace=True, axis=1)
met_data.drop('Time', inplace=True, axis=1)

dates = load_data.Date.values

load_data = load_data.set_index('Date')
met_data = met_data.set_index('Date')





temp_original = met_data.Temp.values
humidity_original = met_data.Humidity.values
dew_original = met_data.DewPoint.values
mobility_original = london_mobility_data.Index.values
precip_original = met_data.Precipitation.values
wind_dir_original = met_data.WindDirection.values
wind_speed_original = met_data.WindSpeed.values
wind_gust_original = met_data.WindGust.values
pressure_original = met_data.Pressure.values
D = load_data.Demand.values #half hourly power demand values 

dew = np.zeros(len(D))
temp = np.zeros(len(D))
humidity = np.zeros(len(D))
precip = np.zeros(len(D))
wind_dir = np.zeros(len(D))
wind_speed = np.zeros(len(D))
wind_gust = np.zeros(len(D))
pressure = np.zeros(len(D))
mobility = np.zeros(len(D))

#resampling
for i in range(len(D)):
    n_day = int(i/24)
    temp[i] = temp_original[n_day]
    dew[i] = dew_original[n_day] 
    humidity[i] = humidity_original[n_day]
    precip[i] = precip_original[n_day]
    wind_dir[i] = wind_dir_original[n_day]
    wind_speed[i] = wind_speed_original[n_day]
    wind_gust[i] = wind_gust_original[n_day]
    pressure[i] = precip_original[n_day]
    mobility[i] = mobility_original[n_day]

test_data = {
            'Date': dates,
            'Demand': D,
            'Temp': temp,
            'DewPoint': dew,
            'Humidity': humidity,
            'Precipitation': precip,
            'WindDirection': wind_dir,
            'WindSpeed': wind_speed,
            'WindGust': wind_gust,
            'Pressure': pressure,
            'Mobility': mobility
            }

test_data = pd.DataFrame(test_data)
# test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['Day_of_week'] = test_data.Date.dt.dayofweek
#monday = 0, tuesday = 1, wednesday = 2, thursday = 3, friday = 4, saturday = 5, sunday = 6
test_data['Hour_of_day'] = test_data.Date.dt.hour
test_data['Month'] = test_data.Date.dt.month
test_data['Day'] = test_data.Date.dt.day
test_data['Year'] = test_data.Date.dt.year

Holiday['Day'] = Holiday.Date.dt.day
Holiday['Month'] = Holiday.Date.dt.month
Holiday['Year'] = Holiday.Date.dt.year

#weekend information
test_data['is_weekend'] = test_data['Day_of_week'].isin([5, 6])
test_data['is_weekend'] = test_data['is_weekend'].astype(int)

#rush hour/peak demand information
test_data.loc[(test_data['Hour_of_day'] >= 7) & (test_data['Hour_of_day'] <= 10), 'is_rush_hour'] = 1  
test_data.loc[(test_data['Hour_of_day'] >= 16) & (test_data['Hour_of_day'] <= 19), 'is_rush_hour'] = 1
test_data['is_rush_hour'] = test_data['is_rush_hour'].fillna(0)

#working hour information
test_data.loc[(test_data['Hour_of_day'] >= 9) & (test_data['Hour_of_day'] <= 17), 'is_working_hour'] = 1
test_data['is_working_hour'] = test_data['is_working_hour'].fillna(0)

#clap for carers information march
test_data.loc[(test_data['Hour_of_day'] == 20) & (test_data['Month'] == 3 ) & (test_data['Day'] == 26), 'clap_for_carers'] = 1
#clap for carers information april
test_data.loc[(test_data['Hour_of_day'] == 20) & (test_data['Month'] == 4 ) & (test_data['Day_of_week'] == 3), 'clap_for_carers'] = 1
#clap for carers information May
test_data.loc[(test_data['Hour_of_day'] == 20) & (test_data['Month'] == 5 ) & (test_data['Day_of_week'] == 3), 'clap_for_carers'] = 1
test_data['clap_for_carers'] = test_data['clap_for_carers'].fillna(0)

#Holidays
# test_data['Holidays'] = np.where((test_data['Day'] == Holiday['Day']) & (test_data['Month'] == Holiday['Month']) & (test_data['Year'] == Holiday['Year']), 'True', 'False')


test_data = test_data.set_index('Date')
print (test_data.head(2))
test_data.drop('Day', inplace=True, axis=1)
test_data.drop('Month', inplace=True, axis=1)
test_data.drop('Year', inplace=True, axis=1)
print(test_data.head(5))


# print(test_data.is_working_hour.head(50))

values = test_data.values
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

lag = 1

reframed = series_to_supervised(scaled, lag, 1)
print(reframed.head(1))
reframed.drop(reframed.iloc[:, (16*lag)+1:], inplace = True, axis = 1)
#reframed.drop(reframed.columns[[33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]], axis=1, inplace=True)
print(reframed.head(1))

values = reframed.values



train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def baseline_model():
    model = Sequential()
    model.add(LSTM(10, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
    #model.add(Dropout(0.2))
    # model.add(LSTM(20, return_sequences=False)) 
    model.add(Dense(1))
    #model.add(Activation('softmax'))
    #opt = keras.optimizers.Adam(learning_rate=0.05)
    model.compile(loss='mse', optimizer='Adam', metrics=['acc'])
    return model

model = baseline_model()
#model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=50, mode='min')]
# fit network
history = model.fit(train_X, train_y, epochs=500, batch_size=24, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=callbacks)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()




predict_X, predict_y = values[:, :-1], values[:, -1]
# reshape input to be 3D [samples, timesteps, features]
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

yhat = model.predict(predict_X)
test_X = test_X.reshape((predict_X.shape[0], predict_X.shape[2]))
inv_yhat = np.concatenate((yhat, predict_X[:, 1:]), axis=1)
print(inv_yhat.shape)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(predict_y), 1))
inv_y = np.concatenate((predict_y, predict_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

err = inv_y - inv_yhat

errpct = abs(err)/(inv_y)*100
MAPE_fy = mean(errpct)
print(MAPE_fy)


rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


plt.plot(test_dates, inv_y, color = 'black', label = 'Actual Load Demand')
plt.plot(test_dates, inv_yhat, color = 'green', label = 'Predicted Load Demand')
plt.title('Load Demand Prediction')
plt.xlabel('Hours')
plt.ylabel('Power')
plt.legend()
plt.show()

plt.plot(test_dates, errpct)
plt.title('MAPE')
plt.xlabel('Hours')
plt.ylabel('MAPE')
plt.show()



