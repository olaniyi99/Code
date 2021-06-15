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
#from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

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
mobility_original = london_mobility_data.London.values
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
scaled_data = scaler.fit_transform(values)

features = scaled_data
target = scaled_data[:,0]

print(features.shape)
print(target.shape)

timesteps = 24*7
ts_generator = TimeseriesGenerator(features, target, length=24, sampling_rate=1, batch_size=1, stride=1)

train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.15, random_state=123, shuffle = False)


train_generator = TimeseriesGenerator(train_X, train_Y, length = timesteps, sampling_rate=1, batch_size=timesteps)
test_generator = TimeseriesGenerator(test_X, test_Y, length=timesteps, sampling_rate=1, batch_size=timesteps)

units = 19
num_features = 16
num_epoch = 50000
learning_rate = 0.001
model = Sequential()
model.add(LSTM(units, input_shape=(timesteps, num_features)))
model.add(LeakyReLU(alpha = 0.1))
model.add(Dropout(0.1))
model.add(Dense(1))
opt = keras.optimizers.Adam(lr = learning_rate)
callback = [EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True)]
model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    

model.summary()

# fit network
history = model.fit_generator(train_generator, epochs=num_epoch, validation_data=test_generator, shuffle=False, callbacks=callback)
model.evaluate_generator(test_generator, verbose=0)



plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

yhat = model.predict_generator(ts_generator)
print(yhat.shape)

df_pred = pd.concat([pd.DataFrame(yhat), pd.DataFrame(test_X[:,1:][timesteps:])], axis=1)
rev_trans = scaler.inverse_transform(df_pred)

df_final = test_data[yhat.shape[0]*-1:]
df_final["Demand_pred"] = rev_trans[:,0]


#plt.plot(df_final['Demand'], color = 'black', label = 'Actual Load Demand')
#plt.plot(df_final['Demand_pred'], color = 'green', label = 'Predicted Load Demand')
#plt.title('Load Demand Prediction')
#plt.xlabel('Date')
#plt.ylabel('Power')
#plt.legend()
#plt.show()

validation_phase0 = df_final.loc['2020-01-02':'2020-01-08']
final_dates = df_final.index

#plt.bar(final_dates, validation_phase0['Demand'], label = 'Actual Load Demand')
#plt.bar(final_dates, validation_phase0['Demand_pred'], label = 'Predicted Load Demand')
#plt.ylabel('Power')
#plt.xticks(final_dates)
#plt.title('Load Demand Prediction')
#plt.legend(loc='best')
#plt.show()

inv_y = df_final.Demand.values
inv_yhat = df_final.Demand_pred.values
err = inv_y - inv_yhat
errpct = abs(err)/(inv_y)*100

df_final['Error'] = errpct

MAPE_fy = mean(errpct)
print(MAPE_fy)

rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
mae =  mean(err)
print('Test MAPE: %.3f' % MAPE_fy)
print(mae)



plt.plot(final_dates, errpct)
plt.title('MAPE')
plt.xlabel('Hours')
plt.ylabel('MAPE')
plt.show()


df_final.to_csv('/Users/davidadeyemi/Desktop/FYP/Code/finalresults_noCOVID.csv')