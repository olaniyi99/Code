import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
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
from datetime import datetime, timedelta 
from meteostat import Point, Hourly


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

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 31)

# Create Point for Birmingham, London, Manchester
Birmingham = Point(52.4814, -1.8998)
London = Point(51.5085, -0.1257)
Manchester = Point(53.481, -2.2374)


# Get hourly data
data_birmingham = Hourly(Birmingham, start, end)
data_birmingham = data_birmingham.fetch()
data_birmingham = pd.DataFrame(data_birmingham)
data_birmingham = data_birmingham[["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "wpgt", "pres"]]
	
data_london = Hourly(London, start, end)
data_london = data_london.fetch()
data_london = pd.DataFrame(data_london)
data_london = data_london[["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "wpgt", "pres"]]

data_manchester = Hourly(Manchester, start, end)
data_manchester = data_manchester.fetch()
data_manchester = pd.DataFrame(data_manchester)
data_manchester = data_manchester[["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "wpgt", "pres"]]

parse_dates = ['Date']
load_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/hourlyload_data.csv')#half hourly power demand data 
met_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/fullyear_temp_data_london.csv') #meterological hourly data
london_mobility_data = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/London_mobility.csv') #daily mobility data
Holiday = pd.read_csv('/Users/davidadeyemi/Desktop/FYP/Code/Holidays.csv') #holiday data
Covid_data = pd.read_csv('COVID_19_cases_2020.csv') #covid daily cases

load_data['Date'] = pd.to_datetime(load_data['Date'] + ' '  + load_data['Time'], dayfirst=True)
met_data['Date'] = pd.to_datetime(met_data['Date'] + ' '  + met_data['Time'], dayfirst=True)

Holiday['Date'] = pd.to_datetime(Holiday['Date'])
Covid_data['Date'] = pd.to_datetime(Covid_data['Date'])

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


#mobility
mobility_original_london = london_mobility_data.London.values
mobility_original_manchester = london_mobility_data.Manchester.values
mobility_original_birmingham = london_mobility_data.Birmingham.values

#london meteorological data 
temp_original_london = data_london.temp.values
humidity_original_london = data_london.rhum.values
dew_original_london = data_london.dwpt.values
precip_original_london = data_london.prcp.values
wind_dir_original_london = data_london.wdir.values
wind_speed_original_london = data_london.wspd.values
wind_gust_original_london = data_london.wpgt.values
pressure_original_london = data_london.pres.values

#manchester meteorological data 
temp_original_manchester = data_manchester.temp.values
humidity_original_manchester = data_manchester.rhum.values
dew_original_manchester = data_manchester.dwpt.values
precip_original_manchester = data_manchester.prcp.values
wind_dir_original_manchester = data_manchester.wdir.values
wind_speed_original_manchester = data_manchester.wspd.values
wind_gust_original_manchester = data_manchester.wpgt.values
pressure_original_manchester = data_manchester.pres.values

#birmingham meteorological data 
temp_original_birmingham = data_birmingham.temp.values
humidity_original_birmingham = data_birmingham.rhum.values
dew_original_birmingham = data_birmingham.dwpt.values
precip_original_birmingham = data_birmingham.prcp.values
wind_dir_original_birmingham = data_birmingham.wdir.values
wind_speed_original_birmingham = data_birmingham.wspd.values
wind_gust_original_birmingham = data_birmingham.wpgt.values
pressure_original_birmingham = data_birmingham.pres.values

D = load_data.Demand.values #half hourly power demand values 
cases_original = Covid_data.Cases.values


dew_london = np.zeros(len(D))
temp_london = np.zeros(len(D))
humidity_london = np.zeros(len(D))
precip_london = np.zeros(len(D))
wind_dir_london = np.zeros(len(D))
wind_speed_london = np.zeros(len(D))
wind_gust_london = np.zeros(len(D))
pressure_london = np.zeros(len(D))

dew_manchester = np.zeros(len(D))
temp_manchester = np.zeros(len(D))
humidity_manchester = np.zeros(len(D))
precip_manchester = np.zeros(len(D))
wind_dir_manchester = np.zeros(len(D))
wind_speed_manchester = np.zeros(len(D))
wind_gust_manchester = np.zeros(len(D))
pressure_manchester = np.zeros(len(D))

dew_birmingham = np.zeros(len(D))
temp_birmingham = np.zeros(len(D))
humidity_birmingham = np.zeros(len(D))
precip_birmingham = np.zeros(len(D))
wind_dir_birmingham = np.zeros(len(D))
wind_speed_birmingham = np.zeros(len(D))
wind_gust_birmingham = np.zeros(len(D))
pressure_birmingham = np.zeros(len(D))

cases = np.zeros(len(D))

mobility_london = np.zeros(len(D))
mobility_manchester = np.zeros(len(D))
mobility_birmingham = np.zeros(len(D))


#resampling
for i in range(len(D)):
    n_day = int(i/24)
    temp_london[i] = temp_original_london[n_day]
    dew_london[i] = dew_original_london[n_day] 
    humidity_london[i] = humidity_original_london[n_day]
    precip_london[i] = precip_original_london[n_day]
    wind_dir_london[i] = wind_dir_original_london[n_day]
    wind_speed_london[i] = wind_speed_original_london[n_day]
    wind_gust_london[i] = wind_gust_original_london[n_day]
    pressure_london[i] = precip_original_london[n_day]

    temp_manchester[i] = temp_original_manchester[n_day]
    dew_manchester[i] = dew_original_manchester[n_day] 
    humidity_manchester[i] = humidity_original_manchester[n_day]
    precip_manchester[i] = precip_original_manchester[n_day]
    wind_dir_manchester[i] = wind_dir_original_manchester[n_day]
    wind_speed_manchester[i] = wind_speed_original_manchester[n_day]
    wind_gust_manchester[i] = wind_gust_original_manchester[n_day]
    pressure_manchester[i] = precip_original_manchester[n_day]

    temp_birmingham[i] = temp_original_birmingham[n_day]
    dew_birmingham[i] = dew_original_birmingham[n_day] 
    humidity_birmingham[i] = humidity_original_birmingham[n_day]
    precip_birmingham[i] = precip_original_birmingham[n_day]
    wind_dir_birmingham[i] = wind_dir_original_birmingham[n_day]
    wind_speed_birmingham[i] = wind_speed_original_birmingham[n_day]
    wind_gust_birmingham[i] = wind_gust_original_birmingham[n_day]
    pressure_birmingham[i] = precip_original_birmingham[n_day]

    cases[i] = cases_original[n_day]
    mobility_london[i] = mobility_original_london[n_day]
    mobility_birmingham[i] = mobility_original_birmingham[n_day]
    mobility_manchester[i] = mobility_original_manchester[n_day]


test_data = {
            'Date': dates,
            'Demand': D,
            'Temp_london': temp_london,
            'DewPoint_london': dew_london,
            'Humidity_london': humidity_london,
            'Precipitation_london': precip_london,
            'WindDirection_london': wind_dir_london,
            'WindSpeed_london': wind_speed_london,
            'WindGust_london': wind_gust_london,
            'Pressure_london': pressure_london,
			'Temp_manchester': temp_manchester,
            'DewPoint_manchester': dew_manchester,
            'Humidity_manchester': humidity_manchester,
            'Precipitation_manchester': precip_manchester,
            'WindDirection_manchester': wind_dir_manchester,
            'WindSpeed_manchester': wind_speed_manchester,
            'WindGust_manchester': wind_gust_manchester,
            'Pressure_manchester': pressure_manchester,
			'Temp_birmingham': temp_birmingham,
            'DewPoint_birmingham': dew_birmingham,
            'Humidity_birmingham': humidity_birmingham,
            'Precipitation_birmingham': precip_birmingham,
            'WindDirection_birmingham': wind_dir_birmingham,
            'WindSpeed_birmingham': wind_speed_birmingham,
            'WindGust_birmingham': wind_gust_birmingham,
            'Pressure_birmingham': pressure_birmingham,
            'Mobility_London': mobility_london,
            'Mobility_birmingham': mobility_birmingham,
            'Mobility_manchester': mobility_manchester,
            'COVID_Cases': cases
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

test_data = test_data.set_index('Date')

#hollidays
test_data.loc['2020-01-01', 'is_holiday'] = 1
test_data.loc['2020-04-10', 'is_holiday'] = 1
test_data.loc['2020-04-13', 'is_holiday'] = 1
test_data.loc['2020-05-08', 'is_holiday'] = 1
test_data.loc['2020-05-25', 'is_holiday'] = 1
test_data.loc['2020-08-31', 'is_holiday'] = 1
test_data.loc['2020-12-25', 'is_holiday'] = 1
test_data.loc['2020-12-28', 'is_holiday'] = 1

test_data['is_holiday'] = test_data['is_holiday'].fillna(0)



test_data.drop('Day', inplace=True, axis=1)
test_data.drop('Month', inplace=True, axis=1)
test_data.drop('Year', inplace=True, axis=1)

validation_phase0 = test_data.loc['2020-01-02':'2020-01-08']
validation_phase1 = test_data.loc['2020-03-30':'2020-04-05']
validation_phase2 = test_data.loc['2020-07-06':'2020-07-12']
validation_phase3 = test_data.loc['2020-08-17':'2020-08-23']
validation_phase4 = test_data.loc['2020-11-23':'2020-11-29']

#removing validation set from testing data
#start_date = pd.to_datetime('2020-01-02')
#end_date = pd.to_datetime('2020-01-08')
#time_change = timedelta(days=1)

#while start_date <= end_date:
	#test_data.drop(start_date, inplace=True, axis=0)
	#start_date = start_date + time_change

#start_date = pd.to_datetime('2020-03-30')
#end_date = pd.to_datetime('2020-04-05')
#time_change = timedelta(days=1)

#while start_date <= end_date:
	#test_data.drop(start_date, inplace=True, axis=0)
	#start_date = start_date + time_change

#start_date = pd.to_datetime('2020-07-06')
#end_date = pd.to_datetime('2020-07-12')
#time_change = timedelta(days=1)

#while start_date <= end_date:
	#test_data.drop(start_date, inplace=True, axis=0)
	#start_date = start_date + time_change

#start_date = pd.to_datetime('2020-08-17')
#end_date = pd.to_datetime('2020-08-23')
#time_change = timedelta(days=1)

#while start_date <= end_date:
	#test_data.drop(start_date, inplace=True, axis=0)
	#start_date = start_date + time_change

#start_date = pd.to_datetime('2020-11-23')
#end_date = pd.to_datetime('2020-11-29')
#time_change = timedelta(days=1)

#while start_date <= end_date:
	#test_data.drop(start_date, inplace=True, axis=0)
	#start_date = start_date + time_change





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

train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.1, random_state=123, shuffle = False)


train_generator = TimeseriesGenerator(train_X, train_Y, length = timesteps, sampling_rate=1, batch_size=timesteps)
test_generator = TimeseriesGenerator(test_X, test_Y, length=timesteps, sampling_rate=1, batch_size=timesteps)

units = len(test_data.columns) + 20
num_features = len(test_data.columns)
num_epoch = 50000
learning_rate = 0.0001
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
df_pred = pd.concat([pd.DataFrame(yhat), pd.DataFrame(test_X[:,1:][timesteps:])], axis=1)
rev_trans = scaler.inverse_transform(df_pred)

df_final = test_data[yhat.shape[0]*-1:]
df_final["Demand_pred"] = rev_trans[:,0]

#calculating errors 
inv_y = df_final.Demand.values
inv_yhat = df_final.Demand_pred.values
err = inv_y - inv_yhat
errpct = abs(err)/(inv_y)*100
df_final['Error'] = errpct
final_dates = df_final.index
mae =  mean(err)
MAPE_fy = mean(errpct)
#rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test MAPE: %.3f' % MAPE_fy)
#print('Test RMSE: %.3f' % rmse)
print(mae)

df_final.to_csv('/Users/davidadeyemi/Desktop/FYP/Code/finalresults.csv')


#validation
#phase_0 = validation_phase0.values
#phase_0 = phase_0.astype('float32')
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled_data = scaler.fit_transform(phase_0)
#features = scaled_data
#arget = scaled_data[:,0]
#ts_generator = TimeseriesGenerator(features, target, length=1, sampling_rate=1, batch_size=1, stride=1)







#yhat = model.predict_generator(ts_generator)
#print(yhat)
#df_pred = pd.concat([pd.DataFrame(yhat), pd.DataFrame(features[:,1:][0:])], axis=1)
#df_pred = pd.DataFrame(yhat)
#rev_trans = scaler.inverse_transform(df_pred)
#df_final = validation_phase0
#df_final["Demand_pred"] = rev_trans[:,0]
#print (df_final)
#print(validation_phase0)



#plt.plot(df_final['Demand'], color = 'black', label = 'Actual Load Demand')
#plt.plot(df_final['Demand_pred'], color = 'green', label = 'Predicted Load Demand')
#plt.title('Load Demand Prediction')
#plt.xlabel('Date')
#plt.ylabel('Power')
#plt.legend()
#plt.show()





#plt.plot(final_dates, errpct)
#plt.title('MAPE')
#plt.xlabel('Hours')
#plt.ylabel('MAPE')
#plt.show()


